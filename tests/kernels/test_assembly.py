"""内核 8 测试：位置矩阵族装配。

测试三级来源：

  来源 A 解析性质      —— 子任务 1：普通单元 ``T = I``、从属单元 ``T = diag(符号)``、
                          气隙单元的插值行 = 内核 7 那一行的「去对角、取负」、``T`` 的
                          语义（``A_local = T·A_active``）用内核 7 的解析场直接验、
                          「零权重的列仍留在 index 里」、分支优先级
                          子任务 2：``csr((K0_data·(Kconst@γ), …))`` 重装出逐单元求和、
                          ``g0i_mat @ A`` = 单元刚度乘 A、``gswap1⊙gswap2`` = 外积、
                          ``bxcoef_mat @ A`` = 各单元的 Bx 系数、掩码划分、计数器关系、
                          ``Jabobi2`` 与 ``K0`` 位置重复、``lil`` 丢零
  来源 B 与 legacy 一致 —— 照抄 ``MMA10:1693-1773``（子任务 1）与 ``1872-2072``（子任务 2）
                          到本文件，逐位比对（``Bxcoef/Bycoef`` 两项为 1 ULP，见该测试）
  来源 C ref vs fast   —— 见本文件末节（``fast/`` 落地后启用）

**脚手架来源说明**：本文件的 legacy 状态（``element_S`` 的插入后连接表、
``element_S_slave`` 的虚拟连接表）不是重写一遍 legacy，而是取内核 6/7 的产出 ——
它们已经各自与 legacy **逐位**比过（``test_boundaries.py``、``test_moving_mesh.py``）。
本文件比对的是 **``T`` 与装包结构这一层**：一边是被测实现，一边是照抄的 legacy 片段。

运行::

    python -m pytest tests/kernels/test_assembly.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp_sparse

from motorfem.kernels.reference import boundaries as ref_bnd
from motorfem.kernels.reference import moving_mesh as ref_mm
from motorfem.kernels.reference import assembly as ref_asm

try:  # 子任务 2/3 落地后才有
    from motorfem.kernels.fast import assembly as fast_asm
except ImportError:  # pragma: no cover
    fast_asm = None

requires_fast = pytest.mark.skipif(fast_asm is None, reason="fast/ 层尚未落地")


# =============================================================================
# 小网格（与内核 6/7 的测试同构）
# =============================================================================

SECTOR = np.pi / 2
COORD_ERR_ABS = 1e-6
R_RINGS = (0.030, 0.040, 0.050, 0.060)      # 转子内 / 转子外 / 接口圆 / 定子外


def make_interface_mesh(n_ang=7, radii=R_RINGS):
    """四层同心圆环扇网格：区域 0 = 转子铁芯、1 = 转子气隙、2 = 定子气隙。

    节点编号 ``= 层序号 * n_ang + 角度序号``，角度 ``linspace(0, 90°, n_ang)``
    —— 含两端点，于是 θ=90° 射线上的节点成为从属节点（内核 6 的前提）。
    """
    angles = np.linspace(0.0, SECTOR, n_ang)
    xy = np.array(
        [[r * np.cos(t), r * np.sin(t)] for r in radii for t in angles], dtype=float
    )
    element_S = []
    for k in range(len(radii) - 1):
        for l in range(n_ang - 1):
            a = k * n_ang + l
            b = (k + 1) * n_ang + l
            c = (k + 1) * n_ang + l + 1
            d = k * n_ang + l + 1
            element_S.append([len(element_S), a, b, c, k])
            element_S.append([len(element_S), a, c, d, k])
    element_rotor_gap = set(e[0] for e in element_S if e[4] == 1)
    element_stator = set(e[0] for e in element_S if e[4] == 2)
    element_rotor = set(e[0] for e in element_S if e[4] in (0, 1))
    return xy, element_S, element_rotor_gap, element_stator, element_rotor


def interface_nodes_legacy(xy, r_interface=0.050, coord_err_abs=COORD_ERR_ABS):
    """照抄 ``MMA10:339-340``：半径**平方差**判据（返回 set，迭代顺序与 legacy 同源）。"""
    return set(
        i for i in range(xy.shape[0])
        if abs((xy[i, 0] ** 2 + xy[i, 1] ** 2) - r_interface ** 2) < coord_err_abs
    )


class Fixture:
    """一份装好的网格 + 内核 6/7 产出 + 供 legacy 片段使用的全局量替身。"""

    def __init__(self, n_ang=7, a_eq=False):
        self.xy, self.element_S, self.element_rotor_gap, self.element_stator, \
            self.element_rotor = make_interface_mesh(n_ang=n_ang)
        self.tri = np.array([e[1:4] for e in self.element_S])

        pairing0 = ref_bnd.pair_master_slave(self.xy, parts=4,
                                             coord_err_abs=COORD_ERR_ABS)
        self.layer = ref_mm.insert_moving_mesh_layer(
            self.xy, self.tri, tuple(interface_nodes_legacy(self.xy)),
            self.element_rotor_gap, pairing0,
            element_stator=self.element_stator, element_rotor=self.element_rotor,
        )
        # 内核 6 的从属虚拟单元表：用**重编号后的 tri + 重绑后的 pairing**重算
        self.virtual = ref_bnd.slave_virtual_elements(self.layer.tri, self.layer.pairing)
        self.a_eq = a_eq

        # --- 供 legacy 片段使用的替身 ---
        self.N_node = self.layer.n_node
        self.N_element = self.layer.tri.shape[0]
        self.node_master = self.layer.pairing.master_to_slave      # legacy 的 node_master（dict）
        self.EM_MS_boundary_A_eq = a_eq
        self.element_S_post = [list(e) for e in self.element_S]
        for i_elem in self.element_rotor_gap:
            for k in (1, 2, 3):
                node = self.element_S_post[i_elem][k]
                if node in self.layer.stator_to_rotor:
                    self.element_S_post[i_elem][k] = self.layer.stator_to_rotor[node]
        self.element_S_slave_post = {
            i_elem: [i_elem, *self.virtual.virtual_tri[i_elem], 0]
            for i_elem in self.virtual.virtual_tri
        }

    def rotating(self, pos):
        """内核 7 的旋转边界（legacy 片段的输入）。"""
        return ref_mm.rotating_boundary(self.layer, parts=4, pos=pos, a_eq=self.a_eq)

    def context(self):
        return ref_asm.TopologyContext.from_boundaries(
            self.layer.tri, self.virtual, self.layer.pairing,
            self.element_rotor_gap, a_eq=self.a_eq, n_node=self.layer.n_node,
        )


@pytest.fixture(scope="module")
def fx() -> Fixture:
    return Fixture(n_ang=7)


@pytest.fixture(scope="module", params=[False, True], ids=["antiperiodic", "periodic"])
def fx_both(request) -> Fixture:
    return Fixture(n_ang=7, a_eq=request.param)


# =============================================================================
# legacy 片段（照抄 MMA10:1693-1773，不复用被测实现）
# =============================================================================

def legacy_transformation_mat_for_elem(g, K_rotating_boundary, node_rotating_boundary, i_elem):
    """照抄 ``MMA10:1693-1773``。``g`` 提供 ``element_S``/``element_S_slave``/``node_master``
    /``EM_MS_boundary_A_eq``/``N_node`` 等全局量替身。"""
    element_rotor_gap = g.element_rotor_gap
    element_S = g.element_S_post
    element_S_slave = g.element_S_slave_post
    node_master = g.node_master
    EM_MS_boundary_A_eq = g.EM_MS_boundary_A_eq
    N_node = g.N_node

    if i_elem in element_rotor_gap:
        if i_elem not in element_S_slave:
            node_i, node_j, node_m = element_S[i_elem][1:4]
            _ki = 1
            _kj = 1
            _km = 1
        else:
            node_i, node_j, node_m = element_S_slave[i_elem][1:4]

            _ki = 1 if node_i in node_master else (1 if EM_MS_boundary_A_eq else -1)
            _kj = 1 if node_j in node_master else (1 if EM_MS_boundary_A_eq else -1)
            _km = 1 if node_m in node_master else (1 if EM_MS_boundary_A_eq else -1)

        _T_temp = np.zeros((3, N_node))
        _index = set()
        if node_i in node_rotating_boundary:
            _node_1, _node_2 = node_rotating_boundary[node_i]

            _T_temp[0, _node_1] += -K_rotating_boundary[node_i, _node_1] * _ki
            _T_temp[0, _node_2] += -K_rotating_boundary[node_i, _node_2] * _ki

            _index.add(_node_1)
            _index.add(_node_2)
        else:
            _T_temp[0, node_i] += _ki

            _index.add(node_i)

        if node_j in node_rotating_boundary:
            _node_1, _node_2 = node_rotating_boundary[node_j]

            _T_temp[1, _node_1] += -K_rotating_boundary[node_j, _node_1] * _kj
            _T_temp[1, _node_2] += -K_rotating_boundary[node_j, _node_2] * _kj

            _index.add(_node_1)
            _index.add(_node_2)
        else:
            _T_temp[1, node_j] += _kj

            _index.add(node_j)

        if node_m in node_rotating_boundary:
            _node_1, _node_2 = node_rotating_boundary[node_m]

            _T_temp[2, _node_1] += -K_rotating_boundary[node_m, _node_1] * _km
            _T_temp[2, _node_2] += -K_rotating_boundary[node_m, _node_2] * _km

            _index.add(_node_1)
            _index.add(_node_2)
        else:
            _T_temp[2, node_m] += _km

            _index.add(node_m)

        _index = list(_index)

        _T = _T_temp[:, _index]
    elif i_elem in element_S_slave:
        node_i, node_j, node_m = element_S_slave[i_elem][1:4]

        _ki = 1 if node_i in node_master else (1 if EM_MS_boundary_A_eq else -1)
        _kj = 1 if node_j in node_master else (1 if EM_MS_boundary_A_eq else -1)
        _km = 1 if node_m in node_master else (1 if EM_MS_boundary_A_eq else -1)

        _T = np.diag([_ki, _kj, _km])
        _index = [node_i, node_j, node_m]
    else:
        node_i, node_j, node_m = element_S[i_elem][1:4]

        _T = np.eye(3)
        _index = [node_i, node_j, node_m]

    return _T, _index


# =============================================================================
# 来源 A：三个分支的解析性质
# =============================================================================

def _plain_elements(fx):
    """既不在转子气隙、也不含从属节点的单元。"""
    return [
        e for e in range(fx.layer.tri.shape[0])
        if e not in fx.element_rotor_gap and e not in fx.virtual.elements
    ]


def _rotor_gap_rotating_elements(fx):
    """转子气隙里含动节点的单元（分支 1 的插值行会真的出现）。"""
    return [
        e for e in fx.element_rotor_gap
        if any(int(n) in fx.rotating(0.31).brackets for n in fx.layer.tri[e])
    ]


def test_plain_element_is_identity(fx):
    """普通单元：``T = I``，``index`` 就是它的三个节点（顺序即连接表顺序）。"""
    ctx = fx.context()
    plain = _plain_elements(fx)
    assert plain, "夹具里应当存在普通单元"
    for i_elem in plain:
        tr = ref_asm.transformation_matrix(i_elem, ctx, fx.rotating(0.31))
        assert np.array_equal(tr.matrix, np.eye(3)), i_elem
        assert tr.index == tuple(int(n) for n in fx.layer.tri[i_elem])


def test_slave_element_is_diagonal_with_signs_from_kernel_6(fx_both):
    """从属单元：``T = diag(符号)``，且符号与内核 6 的 ``virtual_element_signs`` 一致。

    （本内核**复用**内核 6 的那个函数，所以这条同时确认「没有第二套符号逻辑」。）
    """
    ctx = fx_both.context()
    slave_only = [
        e for e in fx_both.virtual.elements if e not in fx_both.element_rotor_gap
    ]
    assert slave_only, "夹具里应当存在「只从属、非气隙」的单元"
    for i_elem in slave_only:
        tr = ref_asm.transformation_matrix(i_elem, ctx, fx_both.rotating(0.31))
        nodes = fx_both.virtual.virtual_tri[i_elem]
        expected = ref_bnd.virtual_element_signs(
            np.asarray([nodes]), ctx.master_nodes, fx_both.a_eq
        )[0]
        assert np.array_equal(tr.matrix, np.diag(expected)), i_elem
        assert tr.index == tuple(int(n) for n in nodes)


def _local_nodes(fx, i_elem):
    """单元局部自由度的节点号（与实现同一判据）：从属单元用**虚拟连接表**。"""
    if i_elem in fx.virtual.elements:
        return tuple(int(n) for n in fx.virtual.virtual_tri[i_elem])
    return tuple(int(n) for n in fx.layer.tri[i_elem])


def _slot_signs(fx, ctx, i_elem):
    """该单元三个槽位的符号（非从属单元恒 +1，与 legacy 的硬编码一致）。"""
    if i_elem in fx.virtual.elements:
        return ref_bnd.virtual_element_signs(
            np.asarray([ctx.virtual_tri[i_elem]]), ctx.master_nodes, fx.a_eq
        )[0]
    return np.ones(3)


def test_rotor_gap_rotating_row_is_minus_the_rotating_boundary_row(fx):
    """转子气隙单元的插值行 = 内核 7 那一行的「去对角、取负」，再乘该槽位的符号。

    即 ``T[s, c] = −K_rotating_boundary[node_s, c] · sign_s`` —— 这正是
    ``A_local_s = Σ_c K_rb[node_s, c]·A_c``（含对角项 ``A_node_s``）在消去
    ``A_node_s`` 之后的写法；``sign_s`` 来自内核 6 的符号向量（只有「本单元同时是从属
    单元」时才可能不是 +1）。``node_s`` 取**局部节点**（从属单元即虚拟节点）。
    """
    ctx = fx.context()
    res = fx.rotating(0.31)
    KB = res.matrix
    checked = 0
    for i_elem in _rotor_gap_rotating_elements(fx):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        signs = _slot_signs(fx, ctx, i_elem)
        for s, node in enumerate(_local_nodes(fx, i_elem)):
            if node not in res.brackets:
                continue
            up, low = res.brackets[node]
            for bracket in (up, low):
                col = tr.index.index(int(bracket))
                assert tr.matrix[s, col] == -KB[node, bracket] * float(signs[s])
                checked += 1
            # 动节点自身不出现在活跃自由度里（它被消去了）
            assert int(node) not in tr.index
    assert checked > 0, "没有取到任何插值行"


def exact_field(fx, pos, c=3.7):
    """沿角度轴定义、**在每个插值括号区间上线性**的自洽场（复刻内核 7 测试里的那条）。

    ``a_eq=False``：逐扇区改号的锯齿（与反周期自洽）；
    ``a_eq=True``：连续三角波（与周期自洽，折点须落在接口节点上 —— 本夹具的
    15° 等角网格满足）。

    动节点取它在**旋转后位置**上的值，于是它同时是插值行的不动点
    （内核 7 已证）。返回 ``(A, field)``。
    """
    def field(psi):
        k = int(np.floor(psi / SECTOR))
        phi = psi - k * SECTOR
        if fx.a_eq:
            return c * abs(1 - 2 * phi / SECTOR)
        sign = 1.0 if k % 2 == 0 else -1.0
        return sign * c * (1 - 2 * phi / SECTOR)

    A = np.zeros(fx.layer.n_node)
    for n in fx.layer.interface_nodes:
        A[n] = field(np.arctan2(fx.layer.xy[n, 1], fx.layer.xy[n, 0]))
    for i in fx.layer.rotor_nodes:
        A[i] = field(np.arctan2(fx.layer.xy[i, 1], fx.layer.xy[i, 0]) + pos)
    return A, field


def test_transform_maps_active_dofs_to_local_dofs(fx_both):
    """``T`` 的语义，分两族单元：

    **(1) 非从属单元**（普通单元、气隙里非从属的单元）::

        T · A_active == A[单元的真实节点]

    即 ``T`` 把活跃自由度映成单元局部自由度：恒等行照抄节点值，插值行给出
    ``A[动节点] = Σ k·A[括号节点]``。用内核 7 已证的解析场（``a_eq`` 分别取锯齿 / 帐篷）
    代入，动节点取它在**旋转后位置**上的值 —— 这条把列序、系数、消元一起钉住。

    **(2) 从属单元**（含从属槽位）::

        T · A_active == signs ⊙ A[虚拟节点]

    ``signs`` 是内核 6 的符号向量（``+1`` 于主动节点、``(1 if a_eq else −1)`` 其余）。
    这里**不**是「``A[真实节点]``」：legacy 的符号约定整体取负于理论文档
    （内核 6 docstring 第四节那个 ★待核对★ 的 ``T_legacy = −T_theory``），而 ``TᵀK0T``
    对全局负号不变、只有源项 ``TᵀR`` 会变号；该潜伏项在真实网格上已由
    ``tools/probe_slave_current.py`` 判定未激活。

    顺带记录一个实测事实：反周期（``a_eq=False``）下，**含主射线节点**的从属单元的
    符号向量是**混合**的（如 ``[−1, −1, +1]``），恰好与理论文档的 ``p_s``
    （在主动边界槽位取 ``−1``）互补 —— 这也说明「混合」不等于内核 6 §四注意 3 那种
    「前缀不满足」的退化，因为那一族单元里主动射线上的节点都配对成功了。
    """
    ctx = fx_both.context()
    pos = 0.31
    res = fx_both.rotating(pos)
    A, _field = exact_field(fx_both, pos)

    seen = {"plain": 0, "slave": 0, "mixed_signs": 0}
    for i_elem in range(fx_both.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        local = tr.matrix @ A[list(tr.index)]
        if i_elem in fx_both.virtual.elements:
            signs = ref_bnd.virtual_element_signs(
                np.asarray([ctx.virtual_tri[i_elem]]), ctx.master_nodes, fx_both.a_eq
            )[0]
            if len(set(signs.tolist())) > 1:
                seen["mixed_signs"] += 1
            expect = signs * A[[int(n) for n in ctx.virtual_tri[i_elem]]]
            seen["slave"] += 1
        else:
            expect = A[[int(n) for n in fx_both.layer.tri[i_elem]]]
            seen["plain"] += 1
        assert np.allclose(local, expect, rtol=1e-12, atol=1e-14), i_elem

    assert seen["plain"] > 0 and seen["slave"] > 0, seen
    if not fx_both.a_eq:
        assert seen["mixed_signs"] > 0, "反周期下应当出现混合符号的从属单元"
    else:
        assert seen["mixed_signs"] == 0, "周期下符号应恒为 +1"


def test_zero_weight_column_is_kept_in_index(fx):
    """``pos = 0`` 时插值系数恰为 0，但**该列仍必须留在 ``index`` 里**。

    内核 7 的装配会丢掉显式零（``lil`` 语义），而 legacy 收集 ``_index`` 用的是
    「用到哪些节点」而不是「哪些系数非零」⇒ 单元块大小不变、``T`` 里该列为 0。
    改用稀疏行的非零元列表会漏掉这一列、块大小改变 ⇒ 结果逐位不同。
    """
    ctx = fx.context()
    res = fx.rotating(0.0)
    KB = res.matrix
    found = 0
    for i_elem in _rotor_gap_rotating_elements(fx):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        nodes = tuple(int(n) for n in fx.layer.tri[i_elem])
        for s, node in enumerate(nodes):
            if node not in res.brackets:
                continue
            up, low = res.brackets[node]
            cols = [tr.index.index(int(up)), tr.index.index(int(low))]
            values = [tr.matrix[s, c] for c in cols]
            if 0.0 in values:
                found += 1
                # 权重为 0 的那一列确实存在于矩阵里（不是被丢掉的列）
                assert tr.matrix.shape[1] == len(tr.index)
                assert KB[node, up] == 0.0 or KB[node, low] == 0.0
    assert found > 0, "pos = 0 时应当出现零权重列，否则这条测试没测到东西"


def test_rotor_gap_branch_wins_over_slave_branch(fx):
    """同时属于两个区域的单元走**转子气隙**分支（判据优先级，与内核 7 第四节注②一致）。"""
    ctx = fx.context()
    both = sorted(e for e in fx.element_rotor_gap if e in fx.virtual.elements)
    assert both, "夹具里应当存在「既气隙又从属」的单元"
    res = fx.rotating(0.31)
    for i_elem in both:
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        if len(tr.index) > 3:
            # 分支 1 的特征：活跃自由度可能多于 3（插值行贡献两个括号节点）
            return
    pytest.fail("既气隙又从属的单元里没有出现块大小 > 3 的，说明没走到转子气隙分支")


def test_active_index_is_set_order_not_sorted(fx):
    """``index`` 的顺序 = legacy 的 ``list(set(...))``，**既不是排序序也不是插入序**。

    这条是**红线**（模块 docstring 第五节）：列序参与 ``Bxcoef@T`` 与 ``TᵀK0T`` 的
    浮点求和次序。Python 的 ``set`` 按**哈希槽**顺序迭代 —— 实测本夹具里
    ``{7, 16, 15, 17}`` 迭代出 ``(16, 17, 15, 7)``（不是插入序 ``7,16,15,17``，
    也不是排序序 ``7,15,16,17``）⇒ 用「插入去重」或 ``sorted`` 建模都会错。
    """
    ctx = fx.context()
    res = fx.rotating(0.31)
    checked_nonsorted = 0
    for i_elem in _rotor_gap_rotating_elements(fx):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        used = []
        for node in _local_nodes(fx, i_elem):
            if node in res.brackets:
                used.extend(int(b) for b in res.brackets[node])
            else:
                used.append(node)
        assert tr.index == tuple(set(used)), i_elem        # legacy 的语义
        if tr.index != tuple(sorted(tr.index)):
            checked_nonsorted += 1
    assert checked_nonsorted > 0, (
        "夹具里应当至少有一个单元的 index 与排序序不同，否则这条测试证明不了"
        "「没有偷偷排序」"
    )


# =============================================================================
# 来源 B：与 legacy 逐位一致
# =============================================================================

@pytest.mark.parametrize("pos", [0.0, 0.31, 1.0, 2.4])
def test_matches_legacy_transformation_matrix(fx_both, pos):
    """逐元素、逐位比对 ``(T, index)``，覆盖全网格与两个 ``a_eq`` 分支。"""
    ctx = fx_both.context()
    res = fx_both.rotating(pos)
    KB, brackets = res.matrix, res.brackets
    n_elem = fx_both.layer.tri.shape[0]
    for i_elem in range(n_elem):
        got = ref_asm.transformation_matrix(i_elem, ctx, res)
        exp_T, exp_index = legacy_transformation_mat_for_elem(fx_both, KB, brackets, i_elem)
        assert got.index == tuple(exp_index), (i_elem, got.index, exp_index)
        assert np.array_equal(got.matrix, exp_T), i_elem


def test_all_three_branches_are_exercised(fx_both):
    """防退化：夹具必须真的覆盖三个分支，否则上面那条逐位比对是空转的。"""
    ctx = fx_both.context()
    res = fx_both.rotating(0.31)
    sizes = set()
    for i_elem in range(fx_both.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        if i_elem in fx_both.element_rotor_gap:
            sizes.add(("rotor_gap", tr.n_active))
        elif i_elem in fx_both.virtual.elements:
            sizes.add(("slave", tr.n_active))
        else:
            sizes.add(("plain", tr.n_active))
    kinds = {k for (k, _n) in sizes}
    assert kinds == {"rotor_gap", "slave", "plain"}, kinds
    assert ("rotor_gap", 4) in sizes, "夹具里应当出现块大小 4 的转子气隙单元（两个动节点）"


# =============================================================================
# 来源 C：ref vs fast
# =============================================================================

@requires_fast
@pytest.mark.parametrize("pos", [0.0, 0.31, 1.0])
def test_ref_fast_parity_transformation_matrix(fx_both, pos):
    ctx = fx_both.context()
    res = fx_both.rotating(pos)
    for i_elem in range(fx_both.layer.tri.shape[0]):
        ref = ref_asm.transformation_matrix(i_elem, ctx, res)
        fast = fast_asm.transformation_matrix(i_elem, ctx, res)
        assert ref.index == fast.index
        assert np.array_equal(ref.matrix, fast.matrix)


# =============================================================================
# 子任务 2：装配族 18 项
# =============================================================================

def legacy_element_operators(g, K_rotating_boundary, node_rotating_boundary,
                             K0elem, Bxcoef, Bycoef, element_iron,
                             element_topo_opti, use_nonlinear_iron):
    """照抄 ``MMA10:1872-2072``（装配部分，不含 ``:2073-2112`` 的驱动行）。

    返回与 ``ElementOperators`` 对应的 18 项（列表/数组原样返回，供逐位比对）。
    """
    N_element = g.N_element
    N_node = g.N_node

    K0_data = []
    K0_row = []
    K0_col = []
    Kconst_mat_data = []
    Kconst_mat_row = []
    Kconst_mat_col = []
    NK0 = 0
    Bxcoef_mat = sp_sparse.lil_matrix((N_element, N_node))
    Bycoef_mat = sp_sparse.lil_matrix((N_element, N_node))
    is_vaccum = np.zeros(N_element)
    is_SIMP = np.zeros(N_element)
    is_non_SIMP = np.zeros(N_element)
    is_linear_iron = np.zeros(N_element)
    g0i_mat_data = []
    g0i_mat_row = []
    g0i_mat_col = []
    Nrow_g0i = 0
    gconst_mat_data = []
    gconst_mat_row = []
    gconst_mat_col = []
    gswap1_mat_data = []
    gswap1_mat_row = []
    gswap1_mat_col = []
    gswap2_mat_data = []
    gswap2_mat_row = []
    gswap2_mat_col = []
    Ngswap = 0
    Jabobi2_row = []
    Jabobi2_col = []
    dK_p_dgamma_A_row = []
    dK_p_dgamma_A_col = []

    for i_elem in range(N_element):
        _T, _index = legacy_transformation_mat_for_elem(
            g, K_rotating_boundary, node_rotating_boundary, i_elem)

        Bxcoef_mat[i_elem, _index] = Bxcoef[i_elem] @ _T
        Bycoef_mat[i_elem, _index] = Bycoef[i_elem] @ _T

        if i_elem in element_iron:
            # ★ 2026-10-17（D14）：**有意偏离** legacy 的掩码顺序。
            #   legacy 先判 ``not use_nonlinear_iron``，于是线性档下拓扑区
            #   也被打成 ``is_linear_iron``（μ=2000μ₀、与 ρ 无关）⇒ 设计变量
            #   完全失效（实测改 ρ 对残差的影响恰为 0）。现在拓扑区**始终**
            #   走 ``is_SIMP``，线性档由材料层给出线性 μ（μ₀·μr·ρ₁）。
            if i_elem in element_topo_opti:
                is_SIMP[i_elem] = 1
            elif not use_nonlinear_iron:
                is_linear_iron[i_elem] = 1
            else:
                is_non_SIMP[i_elem] = 1
        else:
            is_vaccum[i_elem] = 1

        _K0elem = np.array([
            [K0elem[i_elem, 0], K0elem[i_elem, 1], K0elem[i_elem, 2]],
            [K0elem[i_elem, 1], K0elem[i_elem, 3], K0elem[i_elem, 4]],
            [K0elem[i_elem, 2], K0elem[i_elem, 4], K0elem[i_elem, 5]]
        ])

        _K0 = _T.T @ (_K0elem @ _T)
        for i_row, _row in enumerate(_index):
            for i_col, _col in enumerate(_index):
                K0_data.append(_K0[i_row, i_col])
                K0_row.append(_row)
                K0_col.append(_col)

                Kconst_mat_data.append(1)
                Kconst_mat_row.append(NK0)
                Kconst_mat_col.append(i_elem)

                g0i_mat_data.append(_K0[i_row, i_col])
                g0i_mat_row.append(Nrow_g0i + i_row)
                g0i_mat_col.append(_col)

                gswap1_mat_data.append(1)
                gswap1_mat_row.append(Ngswap)
                gswap1_mat_col.append(Nrow_g0i + i_row)

                gswap2_mat_data.append(1)
                gswap2_mat_row.append(Ngswap)
                gswap2_mat_col.append(Nrow_g0i + i_col)

                Jabobi2_row.append(_row)
                Jabobi2_col.append(_col)

                NK0 += 1
                Ngswap += 1

            gconst_mat_data.append(1)
            gconst_mat_row.append(Nrow_g0i + i_row)
            gconst_mat_col.append(i_elem)

            dK_p_dgamma_A_row.append(i_elem)
            dK_p_dgamma_A_col.append(_row)

        Nrow_g0i += len(_index)

    Kconst_mat = sp_sparse.csr_matrix(
        (Kconst_mat_data, (Kconst_mat_row, Kconst_mat_col)), shape=(NK0, N_element))
    g0i_mat = sp_sparse.csr_matrix(
        (g0i_mat_data, (g0i_mat_row, g0i_mat_col)), shape=(Nrow_g0i, N_node))
    gconst_mat = sp_sparse.csr_matrix(
        (gconst_mat_data, (gconst_mat_row, gconst_mat_col)), shape=(Nrow_g0i, N_element))
    gswap1_mat = sp_sparse.csr_matrix(
        (gswap1_mat_data, (gswap1_mat_row, gswap1_mat_col)), shape=(Ngswap, Nrow_g0i))
    gswap2_mat = sp_sparse.csr_matrix(
        (gswap2_mat_data, (gswap2_mat_row, gswap2_mat_col)), shape=(Ngswap, Nrow_g0i))

    return dict(
        K0_data=K0_data, K0_row=K0_row, K0_col=K0_col, Kconst_mat=Kconst_mat,
        Bxcoef_mat=Bxcoef_mat.tocsr(), Bycoef_mat=Bycoef_mat.tocsr(),
        is_vaccum=is_vaccum, is_SIMP=is_SIMP, is_non_SIMP=is_non_SIMP,
        is_linear_iron=is_linear_iron, g0i_mat=g0i_mat, gconst_mat=gconst_mat,
        gswap1_mat=gswap1_mat, gswap2_mat=gswap2_mat,
        Jabobi2_row=Jabobi2_row, Jabobi2_col=Jabobi2_col,
        dK_p_dgamma_A_row=dK_p_dgamma_A_row, dK_p_dgamma_A_col=dK_p_dgamma_A_col,
        NK0=NK0, Ngswap=Ngswap, Nrow_g0i=Nrow_g0i,
    )


def legacy_material_elements(fx):
    """夹具里的铁磁 / 拓扑优化区单元（legacy 的两个区域集合）。"""
    iron = sorted(fx.element_rotor | fx.element_stator)
    topo = sorted(fx.element_rotor)
    return iron, topo


def _operators(fx, pos, use_nonlinear_iron=True):
    """本实现的 18 项 + 输入量（K0elem/Bxcoef/Bycoef 由内核 4 造）。"""
    from motorfem.kernels.reference.element_matrices import electromagnetic_elements

    em = electromagnetic_elements(fx.layer.xy, fx.layer.tri)
    iron, topo = legacy_material_elements(fx)
    ops = ref_asm.element_operators(
        fx.context(), fx.rotating(pos),
        n_node=fx.layer.n_node, k0elem=em.K0, bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        element_iron=iron, element_topo_opti=topo,
        use_nonlinear_iron=use_nonlinear_iron,
    )
    return ops, em, iron, topo


# --- 来源 A：装包结构的语义恒等式 ---

def test_k0_triplets_reassemble_the_elementwise_sum(fx):
    """``csr((K0_data·(Kconst@γ), (K0_row, K0_col)))`` = ``Σ_e γ_e T_eᵀK0_eT_e``。

    这是「位置矩阵族」的全部意义（理论文档 :3298、:3325）：位置存一次、值随 γ 换。
    右边用**独立的**逐单元直接装配算，容差按两者求和次序不同给（``rtol=1e-12``）。
    """
    pos = 0.31
    ops, em, _iron, _topo = _operators(fx, pos)
    rng = np.random.default_rng(0)
    gamma = rng.uniform(0.5, 2.0, size=fx.layer.tri.shape[0])

    assembled = sp_sparse.csr_matrix(
        (ops.k0_data * (ops.kconst_mat @ gamma), (ops.k0_row, ops.k0_col)),
        shape=(fx.layer.n_node, fx.layer.n_node),
    )

    direct = np.zeros((fx.layer.n_node, fx.layer.n_node))
    ctx = fx.context()
    res = fx.rotating(pos)
    for i_elem in range(fx.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        k = em.K0[i_elem]
        k0 = np.array([[k[0], k[1], k[2]], [k[1], k[3], k[4]], [k[2], k[4], k[5]]])
        block = gamma[i_elem] * (tr.matrix.T @ (k0 @ tr.matrix))
        idx = np.asarray(tr.index)
        direct[np.ix_(idx, idx)] += block
    assert np.allclose(assembled.toarray(), direct, rtol=1e-12, atol=1e-14)


def test_g0i_mat_yields_element_stiffness_times_A(fx):
    """``g0i_mat @ A`` 的第 ``(e, r)`` 行 = ``(TᵀK0_eT_e A_e)[r]``（不含 γ）。"""
    pos = 0.31
    ops, em, _iron, _topo = _operators(fx, pos)
    rng = np.random.default_rng(1)
    A = rng.standard_normal(fx.layer.n_node)

    g0 = ops.g0i_mat @ A
    ctx = fx.context()
    res = fx.rotating(pos)
    start = 0
    for i_elem in range(fx.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        k = em.K0[i_elem]
        k0 = np.array([[k[0], k[1], k[2]], [k[1], k[3], k[4]], [k[2], k[4], k[5]]])
        k0t = tr.matrix.T @ (k0 @ tr.matrix)
        expected = k0t @ A[list(tr.index)]
        assert np.allclose(g0[start:start + len(tr.index)], expected, rtol=1e-12,
                           atol=1e-14), i_elem
        start += len(tr.index)
    assert start == ops.n_row_g0i


def test_gswap_pair_reproduces_the_outer_product(fx):
    """``(gswap1@g) ⊙ (gswap2@g)`` 的第 n 项 = ``g_e[i_row]·g_e[i_col]``。

    即理论文档 :3345-3347 的「交换矩阵」确实在铺 ``g_e g_eᵀ`` 的全部元素。
    """
    pos = 0.31
    ops, em, _iron, _topo = _operators(fx, pos)
    rng = np.random.default_rng(2)
    gamma = rng.uniform(0.5, 2.0, size=fx.layer.tri.shape[0])
    g = (ops.g0i_mat @ rng.standard_normal(fx.layer.n_node)) * (ops.gconst_mat @ gamma)

    left = ops.gswap1_mat @ g
    right = ops.gswap2_mat @ g
    # 逐单元重建 g gᵀ 的展开
    expected = []
    start = 0
    for i_elem in range(fx.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, fx.context(), fx.rotating(pos))
        block = g[start:start + len(tr.index)]
        expected.extend((block[:, None] * block[None, :]).ravel())
        start += len(tr.index)
    assert np.array_equal(left * right, np.asarray(expected))


def test_bxcoef_mat_gives_every_elements_curl_coefficients(fx):
    """``bxcoef_mat @ A`` 的第 e 行 = ``Bxcoef[e] @ T_e @ A_e``（内核 4 的 Bx 系数）。"""
    pos = 0.31
    ops, em, _iron, _topo = _operators(fx, pos)
    rng = np.random.default_rng(3)
    A = rng.standard_normal(fx.layer.n_node)

    got_bx = ops.bxcoef_mat @ A
    got_by = ops.bycoef_mat @ A
    ctx = fx.context()
    res = fx.rotating(pos)
    for i_elem in range(fx.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        local = A[list(tr.index)]
        assert got_bx[i_elem] == pytest.approx(float(em.Bxcoef[i_elem] @ (tr.matrix @ local)),
                                               rel=1e-13, abs=1e-15), i_elem
        assert got_by[i_elem] == pytest.approx(float(em.Bycoef[i_elem] @ (tr.matrix @ local)),
                                               rel=1e-13, abs=1e-15), i_elem


def test_masks_partition_the_elements(fx):
    """四个掩码互斥且穷尽（每个单元恰好属于一类），并与分类逻辑一致。"""
    iron, topo = legacy_material_elements(fx)
    n_elem = fx.layer.tri.shape[0]
    for use_nonlinear in (True, False):
        ops, _em, _i, _t = _operators(fx, 0.31, use_nonlinear_iron=use_nonlinear)
        total = ops.is_vaccum + ops.is_simp + ops.is_non_simp + ops.is_linear_iron
        assert np.array_equal(total, np.ones(n_elem))
        expect_vacuum = np.array([1.0 if e not in iron else 0.0 for e in range(n_elem)])
        assert np.array_equal(ops.is_vaccum, expect_vacuum)
        if not use_nonlinear:
            # ★ D14：线性档下**只有非拓扑的纯铁芯**是线性铁
            expect_linear = np.array([1.0 if (e in iron and e not in topo) else 0.0
                                      for e in range(n_elem)])
            assert np.array_equal(ops.is_linear_iron, expect_linear)
            # 拓扑区仍必须是 SIMP（否则 ρ 失效）
            expect_simp = np.array([1.0 if (e in iron and e in topo) else 0.0
                                    for e in range(n_elem)])
            assert np.array_equal(ops.is_simp, expect_simp), (
                "线性档下拓扑区必须保持 is_SIMP，否则设计变量失效（D14）"
            )
        else:
            expect_simp = np.array([1.0 if e in topo else 0.0 for e in range(n_elem)])
            assert np.array_equal(ops.is_simp, expect_simp)
            assert np.array_equal(ops.is_non_simp,
                                  np.array([1.0 if (e in iron and e not in topo) else 0.0
                                            for e in range(n_elem)]))


def test_jabobi2_positions_duplicate_k0_positions(fx):
    """实测事实：``Jabobi2_row/col`` 与 ``K0_row/col`` 逐项相同（legacy 存了两份）。"""
    ops, _em, _i, _t = _operators(fx, 0.31)
    assert np.array_equal(ops.jabobi2_row, ops.k0_row)
    assert np.array_equal(ops.jabobi2_col, ops.k0_col)


def test_counter_relations(fx):
    """``NK0 = Ngswap = Σ_e |index_e|²``；``Nrow_g0i = Σ_e |index_e|``；矩阵形状自洽。"""
    pos = 0.31
    ops, _em, _i, _t = _operators(fx, pos)
    ctx = fx.context()
    res = fx.rotating(pos)
    sizes = [len(ref_asm.transformation_matrix(e, ctx, res).index)
             for e in range(fx.layer.tri.shape[0])]
    assert ops.n_k0 == ops.n_gswap == sum(k * k for k in sizes)
    assert ops.n_row_g0i == sum(sizes)
    n_elem, n_node = fx.layer.tri.shape[0], fx.layer.n_node
    assert ops.k0_data.shape == (ops.n_k0,)
    assert ops.kconst_mat.shape == (ops.n_k0, n_elem)
    assert ops.bxcoef_mat.shape == (n_elem, n_node)
    assert ops.g0i_mat.shape == (ops.n_row_g0i, n_node)
    assert ops.gconst_mat.shape == (ops.n_row_g0i, n_elem)
    assert ops.gswap1_mat.shape == ops.gswap2_mat.shape == (ops.n_gswap, ops.n_row_g0i)
    assert ops.dk_p_dgamma_a_row.shape == (ops.n_row_g0i,)


def test_zero_bx_coefficient_is_dropped_like_lil(fx):
    """``pos = 0`` 时 ``T`` 有一列恰为 0 ⇒ ``Bxcoef@T`` 那一列是精确零 ⇒ 必须丢掉。

    与内核 7 docstring 第四节的 ``lil`` 丢零语义同源；其余矩阵由 COO 装出、保留显式零。
    """
    ops, _em, _i, _t = _operators(fx, 0.0)
    ctx = fx.context()
    res = fx.rotating(0.0)
    dropped = 0
    for i_elem in range(fx.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        stored = ops.bxcoef_mat.getrow(i_elem).nnz
        if stored < len(tr.index):
            dropped += 1
    assert dropped > 0, "pos = 0 时应当出现被丢弃的零系数"


# --- 来源 B：与 legacy 逐位一致 ---

@pytest.mark.parametrize("pos", [0.0, 0.31, 1.0])
@pytest.mark.parametrize("use_nonlinear_iron", [True, False])
def test_matches_legacy_element_operators(fx_both, pos, use_nonlinear_iron):
    """18 项逐项与 legacy 逐位一致（稀疏项比结构 + 数据）。

    ``a_eq`` 由夹具自身的参数提供（``fx_both`` 已按它参数化），不在用例里再改，
    以免同一 module 作用域的夹具被相邻用例污染。
    """
    ops, em, iron, topo = _operators(fx_both, pos, use_nonlinear_iron=use_nonlinear_iron)
    res = fx_both.rotating(pos)

    exp = legacy_element_operators(
        fx_both, res.matrix, res.brackets, em.K0, em.Bxcoef, em.Bycoef,
        set(iron), set(topo), use_nonlinear_iron,
    )

    assert np.array_equal(ops.k0_data, np.asarray(exp["K0_data"]))
    assert np.array_equal(ops.k0_row, np.asarray(exp["K0_row"]))
    assert np.array_equal(ops.k0_col, np.asarray(exp["K0_col"]))
    assert ops.n_k0 == exp["NK0"] and ops.n_gswap == exp["Ngswap"]
    assert ops.n_row_g0i == exp["Nrow_g0i"]
    for name, got, want in (
        ("Kconst_mat", ops.kconst_mat, exp["Kconst_mat"]),
        ("g0i_mat", ops.g0i_mat, exp["g0i_mat"]),
        ("gconst_mat", ops.gconst_mat, exp["gconst_mat"]),
        ("gswap1_mat", ops.gswap1_mat, exp["gswap1_mat"]),
        ("gswap2_mat", ops.gswap2_mat, exp["gswap2_mat"]),
    ):
        assert np.array_equal(got.indptr, want.indptr), name
        assert np.array_equal(got.indices, want.indices), name
        assert np.array_equal(got.data, want.data), name
    for name, got, want in (
        ("is_vaccum", ops.is_vaccum, exp["is_vaccum"]),
        ("is_simp", ops.is_simp, exp["is_SIMP"]),
        ("is_non_simp", ops.is_non_simp, exp["is_non_SIMP"]),
        ("is_linear_iron", ops.is_linear_iron, exp["is_linear_iron"]),
        ("jabobi2_row", ops.jabobi2_row, exp["Jabobi2_row"]),
        ("jabobi2_col", ops.jabobi2_col, exp["Jabobi2_col"]),
        ("dk_row", ops.dk_p_dgamma_a_row, exp["dK_p_dgamma_A_row"]),
        ("dk_col", ops.dk_p_dgamma_a_col, exp["dK_p_dgamma_A_col"]),
    ):
        assert np.array_equal(got, np.asarray(want)), name

    # Bx/By 这两项：**结构逐位相同、数值差 ≤ 1 ULP**（见下一条测试的说明）
    for name, got, want in (
        ("Bxcoef_mat", ops.bxcoef_mat, exp["Bxcoef_mat"]),
        ("Bycoef_mat", ops.bycoef_mat, exp["Bycoef_mat"]),
    ):
        assert np.array_equal(got.indptr, want.indptr), name
        assert np.array_equal(got.indices, want.indices), name
        assert np.abs(got.data - want.data).max() <= 1e-13 * max(
            1.0, float(np.abs(want.data).max())), name


def test_bx_by_deviation_from_legacy_is_one_ulp(fx):
    """记录并守住 ``Bxcoef_mat``/``Bycoef_mat`` 与 legacy 的 **1 ULP** 级偏差。

    **原因（实测）**：这两个矩阵是 ``Bxcoef[e] @ T_e`` 的乘积，两边 ``T`` 的**数值逐位
    相同**、形状也相同 ``(3, k)``，差的是**内存布局**：

    .. code-block:: text

        legacy:  _T_temp[:, _index]   → strides (8, 24)   F 连续
        本实现:  np.zeros((3, k)) 填充  → strides (32, 8)   C 连续

    numpy 的花式索引 ``arr[:, idx]`` 返回 F 连续结果，而 BLAS gemv 对 C/F 连续的操作数
    走不同内核 ⇒ 末位差 ≤ 1 ULP（实测最大绝对差 7.1e-15、相对 2e-16，值域 ~100）。
    下面对 ``ascontiguousarray`` 后的乘积做断言 —— 它证明**差异纯粹来自布局**，不是算法。
    """
    ops, em, iron, topo = _operators(fx, 0.31)
    res = fx.rotating(0.31)
    exp = legacy_element_operators(fx, res.matrix, res.brackets, em.K0, em.Bxcoef,
                                   em.Bycoef, set(iron), set(topo), True)
    for got, want in ((ops.bxcoef_mat, exp["Bxcoef_mat"]),
                      (ops.bycoef_mat, exp["Bycoef_mat"])):
        scale = max(1.0, float(np.abs(want.data).max()))
        deviation = np.abs(got.data - want.data).max()
        assert deviation <= 2e-13 * scale, deviation
        assert not np.array_equal(got.data, want.data), (
            "若这里变成逐位相同，说明 reference 的 T 已经与 legacy 同布局"
            "（例如被改成了 ``_T_temp[:, _index]`` 式的花式索引），"
            "那就该把上面的容差收成 array_equal"
        )

    # 布局归因的直接证据，两半都断言：
    #   (a) 布局不同（花式索引分支）⇒ 统一成 C 连续后乘积逐位相同（差异纯粹来自布局）
    #   (b) 布局本来就相同（分支 2/3 返回 eye/diag）⇒ 乘积逐位相同（差异完全不出现）
    from motorfem.kernels.reference.element_matrices import electromagnetic_elements
    em2 = electromagnetic_elements(fx.layer.xy, fx.layer.tri)
    ctx = fx.context()
    n_layout_diff = 0
    for i_elem in range(fx.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        exp_T, _index = legacy_transformation_mat_for_elem(fx, res.matrix, res.brackets, i_elem)
        assert np.array_equal(tr.matrix, exp_T), i_elem
        if exp_T.flags["C_CONTIGUOUS"]:
            assert np.array_equal(em2.Bxcoef[i_elem] @ tr.matrix,
                                  em2.Bxcoef[i_elem] @ exp_T), i_elem
            continue
        n_layout_diff += 1
        assert tr.matrix.flags["C_CONTIGUOUS"], i_elem      # reference 走 C 连续
        assert np.array_equal(
            em2.Bxcoef[i_elem] @ np.ascontiguousarray(tr.matrix),
            em2.Bxcoef[i_elem] @ np.ascontiguousarray(exp_T)), i_elem
    assert n_layout_diff > 0, "夹具里应当出现 F 连续的 legacy T（气隙分支）"


# =============================================================================
# 子任务 3：右端项 R 与 legacy 元组
# =============================================================================

def legacy_get_R_vector(g, K_rotating_boundary, node_rotating_boundary, R0elem,
                        element_current, element_phaseAp, element_phaseAn,
                        element_phaseBp, element_phaseBn, element_phaseCp,
                        J_phaseA, J_phaseB, J_phaseC):
    """照抄 ``MMA10:2121-2219``。"""
    N_node = g.N_node
    N_element = g.N_element
    R = np.zeros(N_node)

    for i_elem in range(N_element):
        _T, _index = legacy_transformation_mat_for_elem(
            g, K_rotating_boundary, node_rotating_boundary, i_elem)

        T_R0elem = _T.T @ np.array([R0elem[i_elem], R0elem[i_elem], R0elem[i_elem]])

        if i_elem in element_current:
            if i_elem in element_phaseAp:
                R[_index] += T_R0elem * J_phaseA
            elif i_elem in element_phaseAn:
                R[_index] += -T_R0elem * J_phaseA
            elif i_elem in element_phaseBp:
                R[_index] += T_R0elem * J_phaseB
            elif i_elem in element_phaseBn:
                R[_index] += -T_R0elem * J_phaseB
            elif i_elem in element_phaseCp:
                R[_index] += T_R0elem * J_phaseC
            else:
                R[_index] += -T_R0elem * J_phaseC

    return R


def synthetic_phases(fx):
    """夹具里的「绕组」：把定子区单元按序号模 6 分成六个相带（互斥且穷尽）。

    合成网格没有线圈实体，所以这里**造**一组相带 —— 目的是让
    ``signed_phase_current`` 的正常路径（六个集合构成划分）被真正走到。
    """
    elements = sorted(fx.element_stator)
    groups = [set(elements[i::6]) for i in range(6)]
    currents = (5.0, -5.0, 7.0, -7.0, 9.0, -9.0)      # (Ap, An, Bp, Bn, Cp, Cn)
    return set(elements), groups, currents


def _r0elem(fx):
    from motorfem.kernels.reference.element_matrices import electromagnetic_elements
    return electromagnetic_elements(fx.layer.xy, fx.layer.tri).R0


def test_phase_sets_partition_the_current_elements(fx):
    """六个相带集合**互斥且穷尽** ``element_current`` ⇒ legacy 的 else 分支永不触发。

    这条是防「静默算错」的守卫：若某个载流单元掉在六个集合之外，legacy 会把它当作
    Cn（``-J_phaseC``）而**不报错**。真实网格上也应满足（``MMA10:270-277`` 的相带表）。
    """
    current, groups, _c = synthetic_phases(fx)
    union = set().union(*groups)
    assert union == current
    for i in range(6):
        for j in range(i + 1, 6):
            assert not (groups[i] & groups[j])


def test_signed_phase_current_matches_the_legacy_chain(fx):
    """逐单元与 legacy 的 if/elif 链一致（含符号）。"""
    current, groups, currents = synthetic_phases(fx)
    table = ref_asm.signed_phase_current(current, groups, currents)
    Ap, An, Bp, Bn, Cp, Cn = groups
    ja, jb, jc = currents[0], currents[2], currents[4]

    def legacy_value(i_elem):
        if i_elem not in current:
            return None
        if i_elem in Ap:
            return ja
        if i_elem in An:
            return -ja
        if i_elem in Bp:
            return jb
        if i_elem in Bn:
            return -jb
        if i_elem in Cp:
            return jc
        return -jc

    for e in range(fx.layer.tri.shape[0]):
        expected = legacy_value(e)
        if expected is None:
            assert e not in table
        else:
            assert table[e] == expected, e


def test_fallback_branch_is_the_cn_value(fx):
    """防退化：落在 ``element_current`` 里但六个集合都没命中的单元取 Cn 的值。

    legacy 就是这么静默处理的（``else`` 分支）；本测试把它固化成**已知语义**而不是
    意外行为 —— 上面那条划分测试保证正常路径不会走到这里。
    """
    current, groups, currents = synthetic_phases(fx)
    orphan = min(current)
    groups_wo = [set(g) - {orphan} for g in groups]
    table = ref_asm.signed_phase_current({orphan}, groups_wo, currents)
    assert table[orphan] == currents[-1]


def test_row_sums_are_pm1_except_at_the_antiperiodic_seam(fx_both):
    """``T`` 的行和**除反周期接缝外**都是 ±1。

    - 单位行 / 对角行：``±1`` ✓
    - 插值行：``±(kp + kn) = ±1`` —— ``kp + (1−kp) == 1`` 在 IEEE 下**精确**成立 ✓
    - **接缝行**：两个括号里有一个是**从属节点**（其系数已乘 ``σ = −1``）⇒ 两个系数
      **反号** ⇒ 行和 = ``±(kp − kn) ≠ ±1``。

    这正是内核 7 docstring 里说的「接缝处插值不连续」在装配里的可见后果：载荷在这种
    单元上**不再守恒**（分给两个括号的值有一部分抵消掉了）。真实网格上「从属 ∩ 载流」
    为空（``tools/probe_slave_current.py``），所以本模型不触发。
    """
    ctx = fx_both.context()
    res = fx_both.rotating(0.31)
    n_seam = 0
    for i_elem in range(fx_both.layer.tri.shape[0]):
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        for s in range(tr.matrix.shape[0]):
            row = tr.matrix[s]
            nz = np.nonzero(row)[0]
            row_sum = float(row.sum())
            if nz.size < 2:
                assert abs(abs(row_sum) - 1.0) < 1e-15, (i_elem, s)
                continue
            coefficient_signs = np.sign(row[nz])
            if len(set(coefficient_signs.tolist())) == 1:
                assert abs(abs(row_sum) - 1.0) < 1e-15, (i_elem, s, row)
            else:
                # 接缝行：|和| = |kp − kn| < 1
                n_seam += 1
                assert abs(row_sum) < 1.0 + 1e-15
    if fx_both.a_eq:
        assert n_seam == 0, "周期边界下符号恒 +1，不应出现接缝行"
    else:
        assert n_seam > 0, "反周期下应当出现接缝行"


def test_load_distribution_follows_the_row_sums(fx_both):
    """单元贡献之和 = ``R0_e · J_e · Σ_s(第 s 行和)``；反周期下出现**整体变号**。

    这一族的变号正是内核 6 第四节说的**源项 ``TᵀR`` 变号**（★待核对★ 那条关系在数值
    上的形态）：``R0 > 0``、``J > 0`` 时单元总量却是负的。
    """
    current, groups, currents = synthetic_phases(fx_both)
    table = ref_asm.signed_phase_current(current, groups, currents)
    r0 = _r0elem(fx_both)
    ctx = fx_both.context()
    res = fx_both.rotating(0.31)

    flipped = 0
    for i_elem in sorted(current):
        R = ref_asm.get_R_vector(ctx, res, n_node=fx_both.layer.n_node, r0elem=r0,
                                 current_of_element={i_elem: table[i_elem]})
        tr = ref_asm.transformation_matrix(i_elem, ctx, res)
        row_sum_total = float(tr.matrix.sum(axis=1).sum())
        expected = row_sum_total * r0[i_elem] * table[i_elem]
        assert R.sum() == pytest.approx(expected, rel=1e-13), i_elem
        if row_sum_total < 0:
            flipped += 1
    if not fx_both.a_eq:
        assert flipped > 0, "反周期下应当出现源项整体变号的载流单元"
    else:
        assert flipped == 0, "周期下符号恒 +1，不应出现变号"


def test_rotor_gap_source_never_reaches_the_moving_nodes(fx):
    """含插值行的单元：源项只落到**括号节点**上，动节点收不到任何源项。"""
    r0 = _r0elem(fx)
    gap_elem = sorted(fx.element_rotor_gap)[0]
    ctx = fx.context()
    res = fx.rotating(0.31)
    R = ref_asm.get_R_vector(ctx, res, n_node=fx.layer.n_node, r0elem=r0,
                             current_of_element={gap_elem: 3.0})
    moving = [int(n) for n in fx.layer.tri[gap_elem] if int(n) in res.brackets]
    assert moving, "该单元应当含动节点"
    for n in moving:
        assert R[n] == 0.0, n
    assert np.count_nonzero(R) > 0


@pytest.mark.parametrize("pos", [0.0, 0.31, 1.0, 2.4])
def test_matches_legacy_get_R_vector(fx_both, pos):
    """与 legacy 的 ``get_R_vector`` 逐位一致（两个 ``a_eq``、四个转子位置）。"""
    current, groups, currents = synthetic_phases(fx_both)
    table = ref_asm.signed_phase_current(current, groups, currents)
    r0 = _r0elem(fx_both)
    res = fx_both.rotating(pos)

    got = ref_asm.get_R_vector(fx_both.context(), res, n_node=fx_both.layer.n_node,
                               r0elem=r0, current_of_element=table)
    Ap, An, Bp, Bn, Cp, Cn = groups
    want = legacy_get_R_vector(
        fx_both, res.matrix, res.brackets, r0, current, Ap, An, Bp, Bn, Cp,
        currents[0], currents[2], currents[4])
    assert np.array_equal(got, want)


def test_as_legacy_tuple_layout(fx):
    """23 元组的**位置**与 ``MMA10:2114-2118`` 一致（内核 9 的解包靠它）。

    ⚠ **22 → 23（2026-09-24）**：末位新增 ``driving.col_mask``（对称 Dirichlet
    的列投影，见 ``DrivingRows.col_mask`` —— 有意改动，非 legacy 容器）。
    """
    ops, _em, iron, topo = _operators(fx, 0.31)
    res = fx.rotating(0.31)
    driving = ref_bnd.driving_rows(
        fx.layer.n_node,
        dirichlet=[],
        slave_to_master=fx.layer.pairing.slave_to_master,
        a_eq=fx.a_eq,
        row_replacements={i: res.matrix.getrow(i).toarray().ravel() for i in res.brackets},
    )
    tup = ref_asm.as_legacy_tuple(ops, driving)
    assert len(tup) == 23
    assert tup[0] is ops.k0_data and tup[1] is ops.k0_row and tup[2] is ops.k0_col
    assert tup[3] is ops.kconst_mat
    assert tup[4] is driving.is_driven and tup[5] is driving.mat_driven
    assert tup[6] is ops.bxcoef_mat and tup[7] is ops.bycoef_mat
    assert list(tup[8:12]) == [ops.is_vaccum, ops.is_simp, ops.is_non_simp,
                               ops.is_linear_iron]
    assert list(tup[12:16]) == [ops.g0i_mat, ops.gconst_mat, ops.gswap1_mat,
                                ops.gswap2_mat]
    assert list(tup[16:20]) == [ops.jabobi2_row, ops.jabobi2_col,
                                ops.dk_p_dgamma_a_row, ops.dk_p_dgamma_a_col]
    assert tup[20] is driving.j_driven and tup[21] is driving.j_mat_driven
    assert tup[22] is driving.col_mask


# =============================================================================
# fast 层：严格 B 级（fast vs legacy 逐位）+ C 级（ref vs fast）
# =============================================================================

@requires_fast
@pytest.mark.parametrize("pos", [0.0, 0.31, 1.0])
@pytest.mark.parametrize("use_nonlinear_iron", [True, False])
def test_fast_matches_legacy_element_operators_bitwise(fx_both, pos, use_nonlinear_iron):
    """**fast 的 18 项与 legacy 逐位一致，包括 ``Bxcoef_mat``/``Bycoef_mat``。**

    这是 fast 层存在的理由：它保留 legacy 的稠密 ``_T_temp`` 与 ``[:, _index]`` 花式
    索引切片，于是 ``T`` 的内存布局（F 连续）与 legacy 相同 ⇒ 乘积逐位相同。
    ``reference`` 用 C 连续的 ``(3, k)``，那两项只承诺 ``rtol=1e-12``。
    """
    ops, em, iron, topo = _fast_operators(fx_both, pos, use_nonlinear_iron)
    res = fx_both.rotating(pos)
    exp = legacy_element_operators(fx_both, res.matrix, res.brackets, em.K0, em.Bxcoef,
                                   em.Bycoef, set(iron), set(topo), use_nonlinear_iron)

    assert np.array_equal(ops.k0_data, np.asarray(exp["K0_data"]))
    assert np.array_equal(ops.k0_row, np.asarray(exp["K0_row"]))
    assert np.array_equal(ops.k0_col, np.asarray(exp["K0_col"]))
    assert ops.n_k0 == exp["NK0"] and ops.n_gswap == exp["Ngswap"]
    assert ops.n_row_g0i == exp["Nrow_g0i"]
    for name, got, want in (
        ("Kconst_mat", ops.kconst_mat, exp["Kconst_mat"]),
        ("Bxcoef_mat", ops.bxcoef_mat, exp["Bxcoef_mat"]),
        ("Bycoef_mat", ops.bycoef_mat, exp["Bycoef_mat"]),
        ("g0i_mat", ops.g0i_mat, exp["g0i_mat"]),
        ("gconst_mat", ops.gconst_mat, exp["gconst_mat"]),
        ("gswap1_mat", ops.gswap1_mat, exp["gswap1_mat"]),
        ("gswap2_mat", ops.gswap2_mat, exp["gswap2_mat"]),
    ):
        assert np.array_equal(got.indptr, want.indptr), name
        assert np.array_equal(got.indices, want.indices), name
        assert np.array_equal(got.data, want.data), name
    for name, got, want in (
        ("is_vaccum", ops.is_vaccum, exp["is_vaccum"]),
        ("is_simp", ops.is_simp, exp["is_SIMP"]),
        ("is_non_simp", ops.is_non_simp, exp["is_non_SIMP"]),
        ("is_linear_iron", ops.is_linear_iron, exp["is_linear_iron"]),
        ("jabobi2_row", ops.jabobi2_row, exp["Jabobi2_row"]),
        ("jabobi2_col", ops.jabobi2_col, exp["Jabobi2_col"]),
        ("dk_row", ops.dk_p_dgamma_a_row, exp["dK_p_dgamma_A_row"]),
        ("dk_col", ops.dk_p_dgamma_a_col, exp["dK_p_dgamma_A_col"]),
    ):
        assert np.array_equal(got, np.asarray(want)), name


@requires_fast
def test_fast_matches_legacy_transformation_matrix_layout(fx_both):
    """fast 返回的 ``T`` 与 legacy **连内存布局都一样**（这是逐位一致的前提）。"""
    ctx = fx_both.context()
    res = fx_both.rotating(0.31)
    n_layout_diff = 0
    for i_elem in range(fx_both.layer.tri.shape[0]):
        got = fast_asm.transformation_matrix(i_elem, ctx, res)
        exp, exp_index = legacy_transformation_mat_for_elem(
            fx_both, res.matrix, res.brackets, i_elem)
        assert got.index == tuple(exp_index), i_elem
        assert np.array_equal(got.matrix, exp), i_elem
        assert got.matrix.flags["F_CONTIGUOUS"] == exp.flags["F_CONTIGUOUS"], i_elem
        if exp.flags["F_CONTIGUOUS"]:
            n_layout_diff += 1
    assert n_layout_diff > 0, "夹具里应当出现 F 连续的 legacy T（气隙分支）"


@requires_fast
@pytest.mark.parametrize("pos", [0.0, 0.31, 1.0])
def test_ref_fast_parity_element_operators(fx_both, pos):
    """ref vs fast：除 ``Bxcoef/Bycoef``（1 ULP，布局所致）外**逐位**一致。"""
    ref_ops, em, iron, topo = _operators(fx_both, pos)
    fast_ops = fast_asm.element_operators(
        fx_both.context(), fx_both.rotating(pos),
        n_node=fx_both.layer.n_node, k0elem=em.K0, bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        element_iron=iron, element_topo_opti=topo, use_nonlinear_iron=True,
    )
    assert np.array_equal(ref_ops.k0_data, fast_ops.k0_data)
    assert np.array_equal(ref_ops.k0_row, fast_ops.k0_row)
    assert np.array_equal(ref_ops.k0_col, fast_ops.k0_col)
    assert np.array_equal(ref_ops.kconst_mat.indptr, fast_ops.kconst_mat.indptr)
    assert np.array_equal(ref_ops.kconst_mat.indices, fast_ops.kconst_mat.indices)
    assert np.array_equal(ref_ops.kconst_mat.data, fast_ops.kconst_mat.data)
    for name, a, b in (
        ("g0i_mat", ref_ops.g0i_mat, fast_ops.g0i_mat),
        ("gconst_mat", ref_ops.gconst_mat, fast_ops.gconst_mat),
        ("gswap1_mat", ref_ops.gswap1_mat, fast_ops.gswap1_mat),
        ("gswap2_mat", ref_ops.gswap2_mat, fast_ops.gswap2_mat),
    ):
        assert np.array_equal(a.indptr, b.indptr), name
        assert np.array_equal(a.indices, b.indices), name
        assert np.array_equal(a.data, b.data), name
    for name, a, b in (
        ("is_vaccum", ref_ops.is_vaccum, fast_ops.is_vaccum),
        ("is_simp", ref_ops.is_simp, fast_ops.is_simp),
        ("jabobi2_row", ref_ops.jabobi2_row, fast_ops.jabobi2_row),
        ("dk_row", ref_ops.dk_p_dgamma_a_row, fast_ops.dk_p_dgamma_a_row),
    ):
        assert np.array_equal(a, b), name
    # 只有这两项：1 ULP（布局）
    for name, a, b in (("Bxcoef_mat", ref_ops.bxcoef_mat, fast_ops.bxcoef_mat),
                       ("Bycoef_mat", ref_ops.bycoef_mat, fast_ops.bycoef_mat)):
        assert np.array_equal(a.indptr, b.indptr), name
        assert np.array_equal(a.indices, b.indices), name
        scale = max(1.0, float(np.abs(b.data).max()))
        assert np.allclose(a.data, b.data, rtol=1e-12, atol=1e-13 * scale), name


@requires_fast
def test_fast_load_distribution_and_tuple_are_shared(fx_both):
    """``get_R_vector`` / ``signed_phase_current`` / ``as_legacy_tuple`` 两层共用一份实现。

    理由（写在 fast 的模块 docstring 里）：``R`` 的向量化**不可能**保持逐位 ——
    花式索引的 ``+=`` 会丢重复贡献、``np.add.at`` 会改变同一节点的累加次序（红线 5）。
    这条测试确认两层确实指向同一份代码（而非各自一份悄悄分叉的实现）。
    """
    assert fast_asm.get_R_vector is ref_asm.get_R_vector
    assert fast_asm.signed_phase_current is ref_asm.signed_phase_current
    assert fast_asm.as_legacy_tuple is ref_asm.as_legacy_tuple


def _fast_operators(fx, pos, use_nonlinear_iron=True):
    """fast 层的 18 项 + 输入量。"""
    from motorfem.kernels.reference.element_matrices import electromagnetic_elements

    em = electromagnetic_elements(fx.layer.xy, fx.layer.tri)
    iron, topo = legacy_material_elements(fx)
    ops = fast_asm.element_operators(
        fx.context(), fx.rotating(pos),
        n_node=fx.layer.n_node, k0elem=em.K0, bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        element_iron=iron, element_topo_opti=topo,
        use_nonlinear_iron=use_nonlinear_iron,
    )
    return ops, em, iron, topo


@requires_fast
def test_fast_layer_is_not_slower_than_reference(fx):
    """性能守卫（宽松）：位置矩阵搬移不应慢于逐单元朴素实现。

    ⚠ 本内核的 fast **故意**保留 legacy 的稠密 ``(3, N_node)`` 缓冲（那是逐位一致的
    前提），所以它并不指望更快；这条只要求「不显著更慢」，并打印实测值。真正的性能
    收益不在这里 —— 见内核 8 的探针与 ``cProfileReport`` 的历史结论。
    """
    import timeit

    from motorfem.kernels.reference.element_matrices import electromagnetic_elements

    em = electromagnetic_elements(fx.layer.xy, fx.layer.tri)
    iron, topo = legacy_material_elements(fx)
    ctx = fx.context()
    res = fx.rotating(0.31)
    kwargs = dict(n_node=fx.layer.n_node, k0elem=em.K0, bxcoef=em.Bxcoef,
                  bycoef=em.Bycoef, element_iron=iron, element_topo_opti=topo,
                  use_nonlinear_iron=True)

    def bench(fn):
        return min(timeit.repeat(lambda: fn(ctx, res, **kwargs), number=2, repeat=5)) / 2

    t_ref = bench(ref_asm.element_operators)
    t_fast = bench(fast_asm.element_operators)
    print(f"\n[性能守卫·内核 8] 18 项装包：reference {t_ref*1e3:.2f} ms，"
          f"fast（legacy 搬移）{t_fast*1e3:.2f} ms，比值 {t_fast/t_ref:.2f}")
    assert t_fast < 3.0 * t_ref, (t_ref, t_fast)
