"""内核 18 测试：密度滤波器（三种核）。

测试三级来源（`docs/重构设计文档.md` §1.4 与 `CLAUDE.md` §4）：

  来源 A 解析解/解析性质 —— 行归一化（每行和 = 1）、镜像展开式的解析验证、
         邻居核的「共享一条边」判据、单元素退化、变半径核的逐单元半径
  来源 B 与 legacy 一致 —— 照抄 ``MMA10:1259-1379 / 1382-1505 / 1508-1639``
         整段比对（fast 层必须与 legacy **逐位**且含 CSR 结构）
  来源 C reference vs fast —— ``np.allclose(rtol=1e-12)``；``extend=False``
         时逐位一致（每元只累加一次），``extend=True`` 时差 ≤ 1 ulp
         （bucket 遍历顺序决定三项累加顺序，见 reference 模块 docstring 第二节）

⚠ 三条登记事项（设计文档 §12 之外的「只登记不改」清单，见
``reference/filter.py`` 模块 docstring 第四节）：

1. **``_x_min = max(np.min(...))`` 疑似 ``min`` 笔误**：只影响 bucket 原点、
   不影响权重；真实网格上 max 版与 min 版**逐位一致**。本文件用
   ``test_max_min_bbox_typo_is_inert`` 把「数值无害」固化成可执行检查。
2. **邻居核的 ``is_neighbour`` 索引错位**（``MMA10:1597``）：legacy 用矩阵索引
   ``element_S[i]``，正确语义用 ``list(elements)[i]``。真实网格上 5285/6362 行
   全零 —— 但该核全仓库**零调用者**，属死代码。本文件 C 级对比只在**单元 id ==
   矩阵索引**的小网格上进行（错位不激活），并把正确语义（reference 侧）单独测。
3. **三种核在 MMA10 与 PM_solver2 逐字相同**（本批文本 diff 核实），邻居核仅
   MMA10 有。

运行::

    python -m pytest tests/kernels/test_filter.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp_sparse

from motorfem.kernels.reference import filter as ref_filter

try:
    from motorfem.kernels.fast import filter as fast_filter
except ImportError:  # pragma: no cover
    fast_filter = None

requires_fast = pytest.mark.skipif(fast_filter is None, reason="fast/ 层尚未落地")

PARTS = 4
#: 生产参数（MMA10:142-145）：filter_radius=2e-3, p=2；grad radius=0.4e-3, p=0.5
FILTER_RADIUS = 2e-3
FILTER_P_EXP = 2
GRAD_RADIUS = 0.4e-3
GRAD_P_EXP = 0.5


def make_grid(nx=3, ny=2, scale=1e-3):
    """矩形 ``nx × ny`` 网格剖成三角形。

    单元 id 从 0 连续（== 矩阵索引），区域号 = 1。这是**刻意**的：邻居核的
    legacy ``is_neighbour`` 用矩阵索引访问 ``element_S``，只有单元 id == 矩阵
    索引时才与正确语义一致（登记事项 2 的回避）。
    """
    n_node = (nx + 1) * (ny + 1)
    node = np.zeros((n_node, 4))
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            k = iy * (nx + 1) + ix
            node[k] = [k, ix * scale, iy * scale, 0]
    element_S = []
    eid = 0
    for iy in range(ny):
        for ix in range(nx):
            a = iy * (nx + 1) + ix
            b = a + 1
            c = a + (nx + 1)
            d = c + 1
            element_S.append([eid, a, b, c, 1])
            eid += 1
            element_S.append([eid, b, d, c, 1])
            eid += 1
    return node, element_S


@pytest.fixture(scope="module")
def grid():
    node, element_S = make_grid()
    return node, element_S, set(range(len(element_S)))


# =============================================================================
# 来源 A —— 解析性质
# =============================================================================

def test_rows_sum_to_one(grid):
    """三种核的行和都必须是 1（行归一化）。"""
    node, element_S, elements = grid
    for filt in (
        ref_filter.compute_filter(node, element_S, elements, FILTER_RADIUS),
        ref_filter.compute_filter(
            node, element_S, elements, FILTER_RADIUS, extend=False),
        ref_filter.compute_filter_variable(
            node, element_S, elements, lambda x, y: 1.5e-3, 1.5e-3),
        ref_filter.compute_filter_neighbour(node, element_S, elements, FILTER_RADIUS),
    ):
        rowsums = np.asarray(filt.sum(axis=1)).ravel()
        assert np.allclose(rowsums, 1.0, rtol=1e-12), rowsums


def test_single_element_degenerates_to_identity():
    """单单元退化：滤波后密度不变（行 = [1]）。

    单单元时 ``W[0,0]`` 只有对角项（本尊距离 0），归一化后 = 1。
    """
    node = np.array([[0, 0.0, 0.0, 0], [1, 1e-3, 0.0, 0], [2, 0.0, 1e-3, 0]])
    element_S = [[0, 0, 1, 2, 1]]
    elements = {0}
    W = ref_filter.compute_filter(node, element_S, elements, 2e-3)
    assert W.shape == (1, 1)
    assert W[0, 0] == pytest.approx(1.0, rel=1e-15)


def test_two_elements_hand_computed():
    """两单元手算：仅本尊项（extend=False）。

    用 ``make_grid(1, 1)`` 的两个三角形：
      单元 0 节点 (0,1,4) → 质心 (1e-3/3, 1e-3/3)
      单元 1 节点 (1,5,4) → 质心 (2e-3/3, 2e-3/3)
    距离 d = √2·1e-3/3。取 radius = 1e-3（> d）：两两都在半径内，
    权重 ``w = (r − d)``（p=1）、对角 ``r``，行和 = 1 归一化。
    """
    node, element_S = make_grid(nx=1, ny=1)
    elements = {0, 1}
    r = 1e-3
    d = np.sqrt(2) / 3 * 1e-3
    w = (r - d) ** 1
    W = ref_filter.compute_filter(node, element_S, elements, r, extend=False, p_filter=1)
    Wa = W.toarray()
    # 行 0：对角 r，非对角 w；归一化后 /(r+w)
    assert Wa[0, 0] == pytest.approx(r / (r + w), rel=1e-15)
    assert Wa[0, 1] == pytest.approx(w / (r + w), rel=1e-15)
    assert Wa[1, 0] == pytest.approx(w / (r + w), rel=1e-15)
    assert Wa[1, 1] == pytest.approx(r / (r + w), rel=1e-15)


def test_mirror_expansion_matches_legacy_definition():
    """``mirror`` 展开式与 legacy 的 ``_rot_x/_rot_y/_mirror`` 组合逐位一致。

    独立计算（不复用实现），照抄 ``MMA10:177-190``。
    """
    def rot_x(x, y, angle):
        return x * np.cos(angle) - y * np.sin(angle)

    def rot_y(x, y, angle):
        return x * np.sin(angle) + y * np.cos(angle)

    def legacy_mirror(x, y, angle):
        return rot_x(x, y, -angle), rot_y(x, -y, angle)

    rng = np.random.default_rng(7)
    for angle in (0.0, 2 * np.pi / PARTS, 0.3, 1.1):
        x = rng.uniform(-0.05, 0.05, 20)
        y = rng.uniform(-0.05, 0.05, 20)
        got = ref_filter.mirror(x, y, angle)
        want = legacy_mirror(x, y, angle)
        assert np.array_equal(got[0], want[0]), f"angle={angle} (x)"
        assert np.array_equal(got[1], want[1]), f"angle={angle} (y)"


def test_mirror_is_an_isometry():
    """镜像保距（解析性质）：‖mirror_a(p) − mirror_a(q)‖ == ‖p − q‖。"""
    rng = np.random.default_rng(11)
    p = rng.uniform(-1, 1, 2)
    q = rng.uniform(-1, 1, 2)
    for angle in (0.0, 2 * np.pi / PARTS):
        mp = ref_filter.mirror(p[None, :], q[None, :], angle) if False else None
    # 逐点验证
    for angle in (0.0, 2 * np.pi / PARTS):
        px, py = ref_filter.mirror(np.array([p[0]]), np.array([p[1]]), angle)
        qx, qy = ref_filter.mirror(np.array([q[0]]), np.array([q[1]]), angle)
        d_before = np.hypot(p[0] - q[0], p[1] - q[1])
        d_after = np.hypot(px[0] - qx[0], py[0] - qy[0])
        assert d_before == pytest.approx(d_after, rel=1e-12), f"angle={angle}"


def test_neighbour_weights_are_edge_adjacency(grid):
    """邻居核的权重 = 「共享一条边」的指示函数（解析判据）。

    在 ``3×2`` 网格上独立数共享边单元对（不复用实现）：两三角形共一条边
    当且仅当它们共 2 个节点。归一化前行权重必须是 0 或 1（不是距离函数）。
    """
    node, element_S, elements = grid
    W = ref_filter.compute_filter_neighbour(node, element_S, elements, FILTER_RADIUS)
    Wa = W.toarray()
    n = len(elements)
    for i in range(n):
        for j in range(n):
            nodes_i = set(element_S[i][1:4])
            nodes_j = set(element_S[j][1:4])
            shared = len(nodes_i & nodes_j)
            if shared == 2 and i != j:
                assert Wa[i, j] > 0, f"({i},{j}) 共享边应有邻居权重"
            else:
                assert Wa[i, j] == 0, f"({i},{j}) 不应有邻居权重 shared={shared}"
    # 对角元：自己与自己共享 3 个节点，不是 2 ⇒ 0
    assert np.all(np.diag(Wa) == 0)


def test_variable_radius_uses_per_element_radius():
    """变半径核：每个单元用自己的半径。

    用 ``make_grid(1, 1)`` 的两单元：质心距 d = √2/2·1e-3 ≈ 0.707e-3。
    单元 0 的 radius_func 返回 0.3e-3（< d，无邻居），单元 1 返回 2e-3（> d，
    有邻居）⇒ 行 0 只有对角（孤立），行 1 有邻居。验证 ``r_i`` 逐单元生效。
    """
    node, element_S = make_grid(nx=1, ny=1)
    elements = {0, 1}
    # 单元 0 质心 (0.5e-3, 0)、单元 1 质心 (1.0e-3, 0.5e-3)

    def radius_func(x, y):
        # 质心 x>0.5e-3 的是单元 1
        return 2e-3 if x > 0.5e-3 else 0.3e-3

    W = ref_filter.compute_filter_variable(
        node, element_S, elements, radius_func, max_radius=2e-3)
    Wa = W.toarray()
    # 行 0：对角 r0=0.3e-3，邻居距 d>r0 ⇒ 无邻居 ⇒ 归一化后 = 1
    assert Wa[0, 0] == pytest.approx(1.0, rel=1e-15)
    assert Wa[0, 1] == pytest.approx(0.0, abs=1e-15)
    # 行 1：对角 2e-3，邻居距 d<2e-3 ⇒ 有邻居
    assert 0 < Wa[1, 1] < 1
    assert 0 < Wa[1, 0] < 1


# =============================================================================
# 登记事项的可执行检查
# =============================================================================

def test_max_min_bbox_typo_is_inert():
    """登记事项 1：``_x_min = max(np.min(...))`` 用 ``max`` 疑似笔误。

    max/min 只影响 bucket 网格原点，不影响权重（权重只依赖质心距）。
    本测试用**故意不对称**的网格（质心 x 起点 0.2e-3 而非 0，使三列顶点的
    x 最小值不同 —— 正是 max/min 会产生差异的构造）验证：fast（max 版）
    与 reference（O(N²)，无 bbox）仍一致。依据：bucket 网格精确性只依赖
    「桶尺寸 = radius」，不依赖原点（reference docstring 第二节）。
    """
    from motorfem.kernels.fast import filter as fast_f

    # 不对称网格：左列节点 x = 0.2e-3，使 min(_x1)≠min(_x2)≠min(_x3) 完全可能
    node, element_S = make_grid(nx=3, ny=2, scale=1e-3)
    # 把 x=0 的一列挪到 0.2e-3（保持单元结构，只移动坐标）
    for k in range(node.shape[0]):
        if node[k, 1] == 0.0:
            node[k, 1] = 0.2e-3
    elements = set(range(len(element_S)))

    F = fast_f.compute_filter(node, element_S, elements, 1.5e-3, extend=True)
    R = ref_filter.compute_filter(node, element_S, elements, 1.5e-3, extend=True)
    assert np.allclose(F.toarray(), R.toarray(), rtol=1e-12)
    assert np.allclose(np.asarray(F.sum(axis=1)).ravel(), 1.0, rtol=1e-12)


def test_neighbour_index_offset_is_registered():
    """登记事项 2：legacy 邻居核的 ``is_neighbour`` 用矩阵索引访问
    ``element_S``，真实网格上（单元 id ≠ 矩阵索引）会算错。

    本文件**不**断言 legacy 的行为（它是死代码、且照抄 legacy 是 fast 层的
    职责），只断言 reference 用的是**正确语义**：``list(elements)[i]`` 取单元。
    传 ``list`` 保证矩阵顺序 == 传入顺序（避免 set 迭代顺序干扰断言）。
    """
    node, element_S = make_grid(nx=3, ny=2)
    # 构造单元 id ≠ 矩阵索引的情况：拓扑区只取后 4 个单元（id 8..11）
    elements = list(range(8, 12))
    # reference 正确语义：邻居判定用 element_S[elem[i]]
    W = ref_filter.compute_filter_neighbour(node, element_S, elements, 2e-3)
    Wa = W.toarray()
    # 独立验证：4 个单元（id 8..11）里哪些共享边。
    # 归一化后：邻居列 > 0（= 1/行邻居数），非邻居列 == 0，行和 = 1。
    def shared(i, j):
        return len(set(element_S[i][1:4]) & set(element_S[j][1:4])) == 2

    for i, eid_i in enumerate(elements):
        n_neigh = sum(1 for j, eid_j in enumerate(elements) if i != j and shared(eid_i, eid_j))
        for j, eid_j in enumerate(elements):
            if i != j and shared(eid_i, eid_j):
                assert Wa[i, j] == pytest.approx(1.0 / n_neigh, rel=1e-15), f"({i},{j})"
            else:
                assert Wa[i, j] == pytest.approx(0.0, abs=1e-15), f"({i},{j})"


# =============================================================================
# 来源 B —— 与 legacy 逐位（照抄 MMA10）
# =============================================================================

def _legacy_rot_x(x, y, angle):
    return x * np.cos(angle) - y * np.sin(angle)


def _legacy_rot_y(x, y, angle):
    return x * np.sin(angle) + y * np.cos(angle)


def _legacy_mirror(x, y, angle):
    return _legacy_rot_x(x, y, -angle), _legacy_rot_y(x, -y, angle)


def _legacy_dist_p(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def legacy_compute_filter(node, element_S, elements, radius=2e-3, extend=True,
                          p_filter=2, parts=PARTS):
    """照抄 ``MMA10:1259-1379``（把全局量换成参数）。"""
    N_elem_topo_opti = len(elements)
    _element_S_array = np.array(element_S).astype(int)
    _x1 = node[_element_S_array[list(elements), 1], 1]
    _y1 = node[_element_S_array[list(elements), 1], 2]
    _x2 = node[_element_S_array[list(elements), 2], 1]
    _y2 = node[_element_S_array[list(elements), 2], 2]
    _x3 = node[_element_S_array[list(elements), 3], 1]
    _y3 = node[_element_S_array[list(elements), 3], 2]
    _x_max = max(np.max(_x1), np.max(_x2), np.max(_x3))
    _y_max = max(np.max(_y1), np.max(_y2), np.max(_y3))
    _x_min = max(np.min(_x1), np.min(_x2), np.min(_x3))
    _y_min = max(np.min(_y1), np.min(_y2), np.min(_y3))
    if extend:
        _x_extend = radius * 2
        _y_extend = radius * 2
    else:
        _x_extend = 0
        _y_extend = 0
    _x = (_x1 + _x2 + _x3) / 3
    _y = (_y1 + _y2 + _y3) / 3
    if extend:
        _x_ext1, _y_ext1 = _legacy_mirror(_x, _y, 2 * np.pi / parts)
        _x_ext2, _y_ext2 = _legacy_mirror(_x, _y, 0)
    _dl = radius
    N_x_region = int(np.ceil((_x_max - _x_min + _x_extend * 2) / _dl))
    N_y_region = int(np.ceil((_y_max - _y_min + _y_extend * 2) / _dl))
    _region = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
    _x_index = ((_x - (_x_min - _x_extend)) // _dl).astype(int)
    _y_index = ((_y - (_y_min - _y_extend)) // _dl).astype(int)
    if extend:
        _region_ext1 = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
        _region_ext2 = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
        _x_rot1_index = ((_x_ext1 - (_x_min - _x_extend)) // _dl).astype(int)
        _y_rot1_index = ((_y_ext1 - (_y_min - _y_extend)) // _dl).astype(int)
        _x_rot2_index = ((_x_ext2 - (_x_min - _x_extend)) // _dl).astype(int)
        _y_rot2_index = ((_y_ext2 - (_y_min - _y_extend)) // _dl).astype(int)
    for i_elem in range(N_elem_topo_opti):
        _region[_x_index[i_elem]][_y_index[i_elem]].append(i_elem)
        if extend:
            if 0 <= _x_rot1_index[i_elem] < N_x_region and 0 <= _y_rot1_index[i_elem] < N_y_region:
                _region_ext1[_x_rot1_index[i_elem]][_y_rot1_index[i_elem]].append(i_elem)
            if 0 <= _x_rot2_index[i_elem] < N_x_region and 0 <= _y_rot2_index[i_elem] < N_y_region:
                _region_ext2[_x_rot2_index[i_elem]][_y_rot2_index[i_elem]].append(i_elem)
    _filter = sp_sparse.lil_matrix((N_elem_topo_opti, N_elem_topo_opti))
    for i_x in range(N_x_region):
        for i_y in range(N_y_region):
            for i_elem1 in _region[i_x][i_y]:
                for di_x in [-1, 0, 1]:
                    if di_x + i_x < 0 or di_x + i_x >= N_x_region:
                        continue
                    for di_y in [-1, 0, 1]:
                        if di_y + i_y < 0 or di_y + i_y >= N_y_region:
                            continue
                        for i_elem2 in _region[i_x + di_x][i_y + di_y]:
                            _dist = _legacy_dist_p(
                                _x[i_elem1], _y[i_elem1], _x[i_elem2], _y[i_elem2])
                            if _dist < radius:
                                _filter[i_elem1, i_elem2] += (radius - _dist) ** p_filter
                        if extend:
                            for i_elem2 in _region_ext1[i_x + di_x][i_y + di_y]:
                                _dist_ext1 = _legacy_dist_p(
                                    _x[i_elem1], _y[i_elem1], _x_ext1[i_elem2], _y_ext1[i_elem2])
                                if _dist_ext1 < radius:
                                    _filter[i_elem1, i_elem2] += (radius - _dist_ext1) ** p_filter
                            for i_elem2 in _region_ext2[i_x + di_x][i_y + di_y]:
                                _dist_ext2 = _legacy_dist_p(
                                    _x[i_elem1], _y[i_elem1], _x_ext2[i_elem2], _y_ext2[i_elem2])
                                if _dist_ext2 < radius:
                                    _filter[i_elem1, i_elem2] += (radius - _dist_ext2) ** p_filter
    _filter = _filter.tocsr()
    for i_row in range(_filter.shape[1]):
        _filter[i_row, :] /= _filter[i_row, :].sum()
    return _filter


def legacy_compute_filter_variable(node, element_S, elements, radius_func,
                                   max_radius=2e-3, extend=True, p_filter=2,
                                   parts=PARTS):
    """照抄 ``MMA10:1382-1505``（把全局量换成参数）。"""
    N_elem_topo_opti = len(elements)
    _element_S_array = np.array(element_S).astype(int)
    _x1 = node[_element_S_array[list(elements), 1], 1]
    _y1 = node[_element_S_array[list(elements), 1], 2]
    _x2 = node[_element_S_array[list(elements), 2], 1]
    _y2 = node[_element_S_array[list(elements), 2], 2]
    _x3 = node[_element_S_array[list(elements), 3], 1]
    _y3 = node[_element_S_array[list(elements), 3], 2]
    _x_max = max(np.max(_x1), np.max(_x2), np.max(_x3))
    _y_max = max(np.max(_y1), np.max(_y2), np.max(_y3))
    _x_min = max(np.min(_x1), np.min(_x2), np.min(_x3))
    _y_min = max(np.min(_y1), np.min(_y2), np.min(_y3))
    if extend:
        _x_extend = max_radius * 2
        _y_extend = max_radius * 2
    else:
        _x_extend = 0
        _y_extend = 0
    _x = (_x1 + _x2 + _x3) / 3
    _y = (_y1 + _y2 + _y3) / 3
    if extend:
        _x_ext1, _y_ext1 = _legacy_mirror(_x, _y, 2 * np.pi / parts)
        _x_ext2, _y_ext2 = _legacy_mirror(_x, _y, 0)
    _dl = max_radius
    N_x_region = int(np.ceil((_x_max - _x_min + _x_extend * 2) / _dl))
    N_y_region = int(np.ceil((_y_max - _y_min + _y_extend * 2) / _dl))
    _region = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
    _x_index = ((_x - (_x_min - _x_extend)) // _dl).astype(int)
    _y_index = ((_y - (_y_min - _y_extend)) // _dl).astype(int)
    if extend:
        _region_ext1 = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
        _region_ext2 = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
        _x_rot1_index = ((_x_ext1 - (_x_min - _x_extend)) // _dl).astype(int)
        _y_rot1_index = ((_y_ext1 - (_y_min - _y_extend)) // _dl).astype(int)
        _x_rot2_index = ((_x_ext2 - (_x_min - _x_extend)) // _dl).astype(int)
        _y_rot2_index = ((_y_ext2 - (_y_min - _y_extend)) // _dl).astype(int)
    for i_elem in range(N_elem_topo_opti):
        _region[_x_index[i_elem]][_y_index[i_elem]].append(i_elem)
        if extend:
            if 0 <= _x_rot1_index[i_elem] < N_x_region and 0 <= _y_rot1_index[i_elem] < N_y_region:
                _region_ext1[_x_rot1_index[i_elem]][_y_rot1_index[i_elem]].append(i_elem)
            if 0 <= _x_rot2_index[i_elem] < N_x_region and 0 <= _y_rot2_index[i_elem] < N_y_region:
                _region_ext2[_x_rot2_index[i_elem]][_y_rot2_index[i_elem]].append(i_elem)
    _filter = sp_sparse.lil_matrix((N_elem_topo_opti, N_elem_topo_opti))
    for i_x in range(N_x_region):
        for i_y in range(N_y_region):
            for i_elem1 in _region[i_x][i_y]:
                radius = radius_func(_x[i_elem1], _y[i_elem1])
                for di_x in [-1, 0, 1]:
                    if di_x + i_x < 0 or di_x + i_x >= N_x_region:
                        continue
                    for di_y in [-1, 0, 1]:
                        if di_y + i_y < 0 or di_y + i_y >= N_y_region:
                            continue
                        for i_elem2 in _region[i_x + di_x][i_y + di_y]:
                            _dist = _legacy_dist_p(
                                _x[i_elem1], _y[i_elem1], _x[i_elem2], _y[i_elem2])
                            if _dist < radius:
                                _filter[i_elem1, i_elem2] += (radius - _dist) ** p_filter
                        if extend:
                            for i_elem2 in _region_ext1[i_x + di_x][i_y + di_y]:
                                _dist_ext1 = _legacy_dist_p(
                                    _x[i_elem1], _y[i_elem1], _x_ext1[i_elem2], _y_ext1[i_elem2])
                                if _dist_ext1 < radius:
                                    _filter[i_elem1, i_elem2] += (radius - _dist_ext1) ** p_filter
                            for i_elem2 in _region_ext2[i_x + di_x][i_y + di_y]:
                                _dist_ext2 = _legacy_dist_p(
                                    _x[i_elem1], _y[i_elem1], _x_ext2[i_elem2], _y_ext2[i_elem2])
                                if _dist_ext2 < radius:
                                    _filter[i_elem1, i_elem2] += (radius - _dist_ext2) ** p_filter
    _filter = _filter.tocsr()
    for i_row in range(_filter.shape[1]):
        _filter[i_row, :] /= _filter[i_row, :].sum()
    return _filter


def legacy_compute_filter_neighbour(node, element_S, elements, radius=2e-3,
                                    extend=True, parts=PARTS):
    """照抄 ``MMA10:1508-1639``（把全局量换成参数）。

    ⚠ 含 legacy 的 ``is_neighbour`` 矩阵索引错位（登记事项 2）—— 本函数只用于
    fast 层的 B 级对比，不做正确性判断。
    """
    N_elem_topo_opti = len(elements)
    _element_S_array = np.array(element_S).astype(int)
    _x1 = node[_element_S_array[list(elements), 1], 1]
    _y1 = node[_element_S_array[list(elements), 1], 2]
    _x2 = node[_element_S_array[list(elements), 2], 1]
    _y2 = node[_element_S_array[list(elements), 2], 2]
    _x3 = node[_element_S_array[list(elements), 3], 1]
    _y3 = node[_element_S_array[list(elements), 3], 2]
    _x_max = max(np.max(_x1), np.max(_x2), np.max(_x3))
    _y_max = max(np.max(_y1), np.max(_y2), np.max(_y3))
    _x_min = max(np.min(_x1), np.min(_x2), np.min(_x3))
    _y_min = max(np.min(_y1), np.min(_y2), np.min(_y3))
    if extend:
        _x_extend = radius * 2
        _y_extend = radius * 2
    else:
        _x_extend = 0
        _y_extend = 0
    _x = (_x1 + _x2 + _x3) / 3
    _y = (_y1 + _y2 + _y3) / 3
    if extend:
        _x_ext1, _y_ext1 = _legacy_mirror(_x, _y, 2 * np.pi / parts)
        _x_ext2, _y_ext2 = _legacy_mirror(_x, _y, 0)
    _dl = radius
    N_x_region = int(np.ceil((_x_max - _x_min + _x_extend * 2) / _dl))
    N_y_region = int(np.ceil((_y_max - _y_min + _y_extend * 2) / _dl))
    _region = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
    _x_index = ((_x - (_x_min - _x_extend)) // _dl).astype(int)
    _y_index = ((_y - (_y_min - _y_extend)) // _dl).astype(int)
    if extend:
        _region_ext1 = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
        _region_ext2 = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
        _x_rot1_index = ((_x_ext1 - (_x_min - _x_extend)) // _dl).astype(int)
        _y_rot1_index = ((_y_ext1 - (_y_min - _y_extend)) // _dl).astype(int)
        _x_rot2_index = ((_x_ext2 - (_x_min - _x_extend)) // _dl).astype(int)
        _y_rot2_index = ((_y_ext2 - (_y_min - _y_extend)) // _dl).astype(int)
    for i_elem in range(N_elem_topo_opti):
        _region[_x_index[i_elem]][_y_index[i_elem]].append(i_elem)
        if extend:
            if 0 <= _x_rot1_index[i_elem] < N_x_region and 0 <= _y_rot1_index[i_elem] < N_y_region:
                _region_ext1[_x_rot1_index[i_elem]][_y_rot1_index[i_elem]].append(i_elem)
            if 0 <= _x_rot2_index[i_elem] < N_x_region and 0 <= _y_rot2_index[i_elem] < N_y_region:
                _region_ext2[_x_rot2_index[i_elem]][_y_rot2_index[i_elem]].append(i_elem)
    _filter = sp_sparse.lil_matrix((N_elem_topo_opti, N_elem_topo_opti))
    for i_x in range(N_x_region):
        for i_y in range(N_y_region):
            for i_elem1 in _region[i_x][i_y]:
                for di_x in [-1, 0, 1]:
                    if di_x + i_x < 0 or di_x + i_x >= N_x_region:
                        continue
                    for di_y in [-1, 0, 1]:
                        if di_y + i_y < 0 or di_y + i_y >= N_y_region:
                            continue
                        for i_elem2 in _region[i_x + di_x][i_y + di_y]:
                            if _is_neighbour(element_S, i_elem1, i_elem2):
                                _filter[i_elem1, i_elem2] = 1
                        if extend:
                            for i_elem2 in _region_ext1[i_x + di_x][i_y + di_y]:
                                if _is_neighbour(element_S, i_elem1, i_elem2):
                                    _filter[i_elem1, i_elem2] = 1
                            for i_elem2 in _region_ext2[i_x + di_x][i_y + di_y]:
                                if _is_neighbour(element_S, i_elem1, i_elem2):
                                    _filter[i_elem1, i_elem2] = 1
    _filter = _filter.tocsr()
    for i_row in range(_filter.shape[1]):
        _filter[i_row, :] /= _filter[i_row, :].sum()
    return _filter


def _is_neighbour(element_S, i_elem1, i_elem2):
    """照抄 ``MMA10:1596-1599``（矩阵索引 —— 登记事项 2）。"""
    nodes_1 = set(element_S[i_elem1][1:4])
    nodes_2 = set(element_S[i_elem2][1:4])
    return len(nodes_1.intersection(nodes_2)) == 2


@requires_fast
@pytest.mark.parametrize("radius", [1e-3, 2e-3, 3e-3])
@pytest.mark.parametrize("p_filter", [1, 2, 0.5])
@pytest.mark.parametrize("extend", [True, False])
def test_fast_matches_legacy_compute_filter(grid, radius, p_filter, extend):
    node, element_S, elements = grid
    expected = legacy_compute_filter(
        node, element_S, elements, radius=radius, extend=extend, p_filter=p_filter)
    got = fast_filter.compute_filter(
        node, element_S, elements, radius, extend=extend, p_filter=p_filter)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.data, expected.data)


@requires_fast
def test_fast_matches_legacy_compute_filter_variable(grid):
    node, element_S, elements = grid
    rfunc = lambda x, y: 1.0e-3 + 0.5e-3 * (x / (2e-3))   # 随 x 变
    max_radius = 1.5e-3
    expected = legacy_compute_filter_variable(
        node, element_S, elements, rfunc, max_radius=max_radius, extend=True, p_filter=1)
    got = fast_filter.compute_filter_variable(
        node, element_S, elements, rfunc, max_radius, extend=True, p_filter=1)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.data, expected.data)


@requires_fast
def test_fast_matches_legacy_compute_filter_neighbour(grid):
    node, element_S, elements = grid
    expected = legacy_compute_filter_neighbour(
        node, element_S, elements, radius=2e-3, extend=True)
    got = fast_filter.compute_filter_neighbour(
        node, element_S, elements, 2e-3, extend=True)
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)
    assert np.array_equal(got.data, expected.data)


# =============================================================================
# 来源 C —— reference vs fast
# =============================================================================

@requires_fast
@pytest.mark.parametrize("radius", [1e-3, 2e-3, 3e-3])
@pytest.mark.parametrize("p_filter", [1, 2, 0.5])
@pytest.mark.parametrize("extend", [True, False])
def test_fast_matches_reference_compute_filter(grid, radius, p_filter, extend):
    node, element_S, elements = grid
    expected = ref_filter.compute_filter(
        node, element_S, elements, radius, extend=extend, p_filter=p_filter)
    got = fast_filter.compute_filter(
        node, element_S, elements, radius, extend=extend, p_filter=p_filter)
    # rtol=1e-12（extend=True 时最多差 1 ulp，见模块 docstring 第二节）
    assert np.allclose(got.toarray(), expected.toarray(), rtol=1e-12)
    if not extend:
        assert np.array_equal(got.toarray(), expected.toarray())


@requires_fast
def test_fast_matches_reference_compute_filter_variable(grid):
    node, element_S, elements = grid
    rfunc = lambda x, y: 1.0e-3 + 0.5e-3 * (x / (2e-3))
    max_radius = 1.5e-3
    expected = ref_filter.compute_filter_variable(
        node, element_S, elements, rfunc, max_radius, extend=True, p_filter=1)
    got = fast_filter.compute_filter_variable(
        node, element_S, elements, rfunc, max_radius, extend=True, p_filter=1)
    assert np.allclose(got.toarray(), expected.toarray(), rtol=1e-12)


@requires_fast
def test_fast_matches_reference_compute_filter_neighbour(grid):
    node, element_S, elements = grid
    expected = ref_filter.compute_filter_neighbour(
        node, element_S, elements, 2e-3, extend=True)
    got = fast_filter.compute_filter_neighbour(
        node, element_S, elements, 2e-3, extend=True)
    # 邻居核权重是 0/1（无浮点累加）⇒ 逐位
    assert np.array_equal(got.toarray(), expected.toarray())


# =============================================================================
# 防退化 —— B/C 级测试不能空转
# =============================================================================

def test_bbox_and_radius_actually_limit_the_filter(grid):
    """防退化：若 ``radius`` 大得超过整个域，每个单元都能看到所有单元
    （filter 稠密、行和仍 1）；若小得只看到自己，则退化为恒等。

    两个极端都测，确保「半径确实在起作用」，而不是测试里全连 / 全不连。
    """
    node, element_S, elements = grid
    n = len(elements)
    # 大半径：全连
    big = ref_filter.compute_filter(node, element_S, elements, 10.0, extend=False)
    assert big.nnz == n * n
    # 小半径：只看到自己（对角 r，其余 0 ⇒ 恒等）
    small = ref_filter.compute_filter(node, element_S, elements, 1e-9, extend=False)
    assert np.allclose(small.toarray(), np.eye(n), rtol=1e-12)
