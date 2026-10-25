"""内核 6 测试：驱动行矩阵 / 主从周期 / 从属虚拟单元。

测试三级来源：
  来源 A 解析性质      —— 配对必须等于「绕原点转 2π/parts」；虚拟单元节点全部
                          非从属（理论文档明文要求）；驱动行约束的一步 Newton
                          消除；`j_driven` 的行缩放无浮点累加；角点同时命中
                          Dirichlet 与从属时的覆盖顺序；变换矩阵的符号关系
  来源 B 与 legacy 一致 —— 复刻 MMA10:305-321（电磁配对）、402-419（力学配对，
                          作用在另一张节点表上）、323-335（虚拟单元）、
                          2073-2102（驱动行）、2104-2112（整行替换）逐位比对
  来源 C ref vs fast   —— 合成极坐标网格 + 随机角度网格上逐位相同

运行::

    python -m pytest tests/kernels/test_boundaries.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp_sparse

from motorfem.kernels.fast import boundaries as fast_bnd
from motorfem.kernels.reference import boundaries as ref_bnd


# =============================================================================
# 网格夹具
# =============================================================================

#: 合成网格的半径（与真实模型同量级：轴、转子外径、气隙中线、定子外径）
RADII = (0.015, 0.030, 0.04475, 0.045125, 0.0725)

#: 扇区角（parts = 4 → 90°）
SECTOR_DEG = 90.0

#: 默认角度序列：0..90 每 10°，必然含 0 与 90
DEFAULT_ANGLES = np.arange(0.0, SECTOR_DEG + 1e-9, 10.0)


def make_polar_mesh(radii=RADII, angles_deg=None) -> tuple[np.ndarray, np.ndarray]:
    """造一个极坐标环扇三角形网格，三角剖分全部逆时针。

    节点角度精确取整（``np.deg2rad``），使 θ=0 与 θ=90° 射线上的节点坐标落线；
    实测 |x| 或 |y| 的残差约 1e-18，远小于 ``coord_err_abs = 1e-6``。

    节点编号规则（测试里要用到）：``节点号 = 半径序号 * n_angles + 角度序号``。

    Returns
    -------
    (xy, tri) ：``xy`` 为 (N, 2) 坐标；``tri`` 为 (M, 3) 逆时针连接表。
    """
    if angles_deg is None:
        angles_deg = DEFAULT_ANGLES
    angles_deg = np.asarray(angles_deg, dtype=float)
    radii = np.asarray(radii, dtype=float)

    xy = np.array(
        [
            [r * np.cos(np.deg2rad(t)), r * np.sin(np.deg2rad(t))]
            for r in radii
            for t in angles_deg
        ]
    )

    n_ang = angles_deg.size
    tri = []
    for k in range(len(radii) - 1):
        for l in range(n_ang - 1):
            a = k * n_ang + l
            b = (k + 1) * n_ang + l
            c = (k + 1) * n_ang + l + 1
            d = k * n_ang + l + 1
            tri += [[a, b, c], [a, c, d]]
    return xy, np.array(tri, dtype=int)


def make_polar_mesh_p2(radii=RADII, angles_deg=None) -> tuple[np.ndarray, np.ndarray]:
    """P2 版极坐标环扇网格：6 节点三角形（边中点共享、gmsh 列序）。

    先造 P1 的 3 列网格，再把每条边配一个中点号（追加在顶点之后），
    gmsh 列序 ``[v0, v1, v2, m01, m12, m20]``。顶点编号与 P1 一致，
    因此 θ=90° 射线上的**顶点和边中点**都会从属（``pair_master_slave``
    按坐标判定，与列角色无关）。
    """
    xy, tri3 = make_polar_mesh(radii=radii, angles_deg=angles_deg)
    n_vertex = xy.shape[0]
    mids: list[tuple[float, float]] = []
    edge_to_mid: dict[tuple[int, int], int] = {}

    def _mid_of(va, vb):
        key = (min(int(va), int(vb)), max(int(va), int(vb)))
        if key not in edge_to_mid:
            edge_to_mid[key] = n_vertex + len(mids)
            mids.append(((xy[va, 0] + xy[vb, 0]) / 2.0,
                         (xy[va, 1] + xy[vb, 1]) / 2.0))
        return edge_to_mid[key]

    tri6 = np.zeros((tri3.shape[0], 6), dtype=int)
    for r, (a, b, c) in enumerate(tri3):
        tri6[r] = [a, b, c, _mid_of(a, b), _mid_of(b, c), _mid_of(c, a)]
    xy6 = np.concatenate([xy, np.array(mids, dtype=float)], axis=0)
    return xy6, tri6


def rotate(points, angle):
    """绕原点旋转。独立实现，不复用被测模块的私有 ``_rot_x/_rot_y``。"""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack(
        [points[:, 0] * c - points[:, 1] * s, points[:, 0] * s + points[:, 1] * c]
    )


def legacy_rot_x(x, y, angle):
    """照抄 MMA10:177-178，供来源 B 使用。"""
    return x * np.cos(angle) - y * np.sin(angle)


def legacy_rot_y(x, y, angle):
    """照抄 MMA10:181-182，供来源 B 使用。"""
    return x * np.sin(angle) + y * np.cos(angle)


def slave_node_ids(xy, parts=4, coord_err_abs=1e-6):
    """θ=±2π/parts 射线上的节点号（从属候选），升序。"""
    angle = 2 * np.pi / parts
    rx = xy[:, 0] * np.cos(-angle) - xy[:, 1] * np.sin(-angle)
    ry = xy[:, 0] * np.sin(-angle) + xy[:, 1] * np.cos(-angle)
    return np.flatnonzero((np.abs(ry) < coord_err_abs) & (coord_err_abs < np.abs(rx)))


@pytest.fixture(scope="module")
def mesh_one_sector() -> tuple[np.ndarray, np.ndarray]:
    """单扇区 [0°, 90°]：与真实模型同构（主动在 0°，从属在 90°）。"""
    return make_polar_mesh()


@pytest.fixture(scope="module")
def mesh_two_sector() -> tuple[np.ndarray, np.ndarray]:
    """跨 [0°, 180°]：主动候选出现在 θ=0 **与** θ=180，但只有 θ=0 一侧有从属。"""
    return make_polar_mesh(angles_deg=np.arange(0.0, 2 * SECTOR_DEG + 1e-9, 10.0))


@pytest.fixture(scope="module")
def pairing(mesh_one_sector) -> ref_bnd.PeriodicBoundary:
    xy, _ = mesh_one_sector
    return ref_bnd.pair_master_slave(xy, parts=4)


@pytest.fixture(scope="module")
def pairing_two_sector(mesh_two_sector) -> ref_bnd.PeriodicBoundary:
    xy, _ = mesh_two_sector
    return ref_bnd.pair_master_slave(xy, parts=4)


@pytest.fixture(scope="module")
def virtual(mesh_one_sector, pairing) -> ref_bnd.SlaveVirtualElements:
    _, tri = mesh_one_sector
    return ref_bnd.slave_virtual_elements(tri, pairing)


# =============================================================================
# 来源 A：配对（解析性质）
# =============================================================================

def test_pairing_matches_rotation_by_sector_angle(pairing, mesh_one_sector):
    """配对的解析定义：把主动节点绕原点转 2π/parts，必须落到从属节点上。

    这是配对的**定义**，不依赖任何实现细节：同一个物理点在不同扇区里的像，
    坐标必须差一个精确的扇区旋转。顺带断言半径相同（旋转是等距变换）。
    """
    xy, _ = mesh_one_sector
    angle = 2 * np.pi / 4

    assert len(pairing.slave_to_master) > 0, "夹具网格里没有配成任何一对，测试无意义"
    for i_slave, i_master in pairing.slave_to_master.items():
        rotated = rotate(xy[i_master], angle)[0]
        err = np.abs(rotated - xy[i_slave]).max()
        assert err < 1e-15, (
            f"从属节点 {i_slave} 与主动节点 {i_master} 的旋转像不符，最大差 {err:.3e}"
        )
        assert np.isclose(
            np.hypot(*xy[i_slave]), np.hypot(*xy[i_master]), rtol=1e-14
        ), "主从节点半径不同，旋转关系不成立"

    # 从属节点必须确实在 θ=±2π/parts 上，主动节点必须确实在 θ=0 上
    for i_slave in pairing.slave_to_master:
        assert abs(xy[i_slave, 0]) < 1e-15, "从属节点的 x 分量不为 0（不在 90° 射线上）"
    for i_master in pairing.master_to_slave:
        assert abs(xy[i_master, 1]) < 1e-15, "主动节点的 y 分量不为 0（不在 0° 射线上）"


def test_pairing_is_a_bijection(pairing):
    """``slave_to_master`` 单射，``master_to_slave`` 是它的逆。"""
    s2m = pairing.slave_to_master
    m2s = pairing.master_to_slave

    assert len(set(s2m.values())) == len(s2m), "两个从属节点配到了同一个主动节点"
    assert set(m2s.keys()) == set(s2m.values()), "反向表的键不是正向表的值集合"
    assert set(m2s.values()) == set(s2m.keys()), "反向表的值不是正向表的键集合"
    for i_slave, i_master in s2m.items():
        assert m2s[i_master] == i_slave
    assert pairing.master_nodes == frozenset(s2m.values())


def test_pairing_completeness(pairing, mesh_one_sector):
    """每个从属候选都必须配上，且从属集合恰是 θ=90° 射线上的节点。"""
    xy, _ = mesh_one_sector
    expected = set(slave_node_ids(xy).tolist())
    assert set(pairing.slave_to_master) == expected, (
        "配对表覆盖的从属节点与 θ=90° 射线上的节点不一致"
    )


def test_master_candidate_without_slave_is_not_a_master(pairing_two_sector, mesh_two_sector):
    """主动**候选**与主动**节点**是两个不同的集合，不能混用。

    跨 [0°, 180°] 的网格上，θ=180 射线上的节点也满足主动候选判据
    ``|y| < err < |x|``，但它们没有对应从属节点（网格里没有 θ=270° 的节点），
    所以不会进入配对表。

    ``virtual_element_signs`` 的成员测试用的是 ``master_to_slave`` 的**键**
    （legacy 写 ``node_i in node_master``，而那是 dict 的键测试），因此这个区别
    有实际后果——若谁把「候选」当「主动节点」用，符号向量就会算错。
    """
    xy, _ = mesh_two_sector
    candidates = np.flatnonzero(
        (np.abs(xy[:, 1]) < 1e-6) & (1e-6 < np.abs(xy[:, 0]))
    )

    assert candidates.size > len(pairing_two_sector.master_to_slave), (
        "夹具网格没有造出「主动候选多于主动节点」的情形，本条测试失去意义"
    )
    assert set(pairing_two_sector.master_nodes) < set(candidates.tolist()), (
        "主动节点集合应是主动候选集合的真子集"
    )
    for i_master in pairing_two_sector.master_to_slave:
        assert xy[i_master, 0] > 0, f"θ=180 侧的候选 {i_master} 不该成为主动节点"


# =============================================================================
# 来源 A：从属虚拟单元
# =============================================================================

def test_virtual_element_nodes_are_never_slave(virtual, pairing, mesh_one_sector):
    """**虚拟单元的所有节点必须是非从属的真实节点**。

    理论文档 ``磁障设计/同步磁阻电机磁障结构分析.md`` 约 2842 行明文要求：
    「注意，**虚拟单元的节点只能为非从属边界上的真实节点**」。

    这是本内核最有力的外部判据：若替换写错（例如把从属节点又映射到另一个从属
    节点、或漏替一个槽位），虚拟单元就会残留从属节点。同时断言不退化（三个节点
    互不相同）——退化的虚拟单元会让单元刚度矩阵奇异。
    """
    _, tri = mesh_one_sector
    slave_set = set(pairing.slave_to_master)
    assert len(virtual.elements) > 0, "夹具网格里没有从属单元，测试无意义"

    for i_elem, nodes in virtual.virtual_tri.items():
        for n in nodes:
            assert n not in slave_set, (
                f"虚拟单元 {i_elem} 的节点 {n} 仍是从属节点 —— 违反理论文档要求"
            )
        assert len(set(nodes)) == 3, f"虚拟单元 {i_elem} 退化：{nodes}"

    # 防退化：夹具里必须真的出现「一个单元有两个从属节点」的情形，否则
    # 「全部非从属」这条在单替换情形下是平凡的。
    replaced_counts = [
        sum(1 for n in tri[i_elem] if n in slave_set) for i_elem in virtual.elements
    ]
    assert max(replaced_counts) >= 2, (
        "夹具网格里每个从属单元都只含一个从属节点，测试强度不足"
    )


def test_virtual_element_replaces_exactly_the_slave_slots(virtual, pairing, mesh_one_sector):
    """逐槽位核对：虚拟单元 = 原单元，把从属槽位换成对应主动节点、其余不动。"""
    _, tri = mesh_one_sector
    s2m = pairing.slave_to_master

    for i_elem, nodes in virtual.virtual_tri.items():
        original = tri[i_elem]
        assert any(n in s2m for n in original), (
            f"单元 {i_elem} 不含从属节点，不该出现在从属单元表里"
        )
        for k in range(3):
            expected = s2m[original[k]] if original[k] in s2m else original[k]
            assert nodes[k] == expected, (
                f"单元 {i_elem} 第 {k} 槽位应为 {expected}，实得 {nodes[k]}"
            )

    # 不含从属节点的单元一个都不能进表
    s2m_keys = set(s2m)
    for i_elem in range(tri.shape[0]):
        touches = any(n in s2m_keys for n in tri[i_elem])
        assert (i_elem in virtual.elements) == touches


# =============================================================================
# 来源 A：驱动行装置
# =============================================================================

def test_constraint_row_entries_are_exact(mesh_one_sector, pairing):
    """主从约束行的系数必须**精确**是 ``A_s + σ·A_m`` 的两项。

    周期（``a_eq=True``）: 行 = ``A_s − A_m`` → 系数 ``(1, −1)``
    反周期（``a_eq=False``）: 行 = ``A_s + A_m`` → 系数 ``(1, +1)``

    断言逐位相等：这两个系数是构造出来的 ±1，不经过任何浮点运算。
    """
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    for a_eq, sigma in ((True, -1.0), (False, 1.0)):
        rows = ref_bnd.driving_rows(
            n_node, dirichlet=[], slave_to_master=pairing.slave_to_master, a_eq=a_eq
        )
        dense = rows.mat_driven.toarray()
        for i_s, i_m in pairing.slave_to_master.items():
            assert dense[i_s, i_s] == 1.0, f"a_eq={a_eq} 时 [{i_s},{i_s}] 不为 1"
            assert dense[i_s, i_m] == sigma, (
                f"a_eq={a_eq} 时 [{i_s},{i_m}] 应为 {sigma}，实得 {dense[i_s, i_m]}"
            )
            assert np.count_nonzero(dense[i_s]) == 2, "约束行应恰有两个非零元"


def test_dirichlet_rows_are_unit_rows(mesh_one_sector, pairing):
    """纯 Dirichlet 行必须恰是 ``e_i``（齐次条件 ``A_i = 0``）。"""
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    dirichlet = [i for i in (0, 5) if i not in pairing.slave_to_master]
    assert len(dirichlet) == 2, "夹具选点撞上了从属节点，测试需要换点"

    rows = ref_bnd.driving_rows(
        n_node, dirichlet=dirichlet, slave_to_master=pairing.slave_to_master, a_eq=False
    )
    dense = rows.mat_driven.toarray()
    identity = np.eye(n_node)
    for i in dirichlet:
        assert np.array_equal(dense[i], identity[i]), "Dirichlet 行不是 e_i"
        assert rows.is_driven[i] == 1.0
        assert rows.j_driven[i, i] == 0.0


def test_one_newton_step_drives_constraints_to_zero(mesh_one_sector, pairing):
    """**一步 Newton 就把约束打到机器零**。

    Newton 步满足 ``J δ = −F``。受约束行的 ``J`` 行恰是约束行本身（``j_driven``
    把该行清零、``j_mat_driven`` 提供约束行），所以

        δ_s = −(A_s + σ A_m)     →     更新后 A_s + σ A_m = 0

    容差依据：约束行只有两个非零元，解出的 δ 与真值的偏差只来自线性方程求解的
    舍入（良态系统上约 ``eps·‖A‖``），因此按 ``‖A‖`` 缩放取 ``1e-13``（比机器
    eps 宽几个数量级，但仍能抓住任何一处符号或系数写错——写错会给出 O(1) 残差）。
    """
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    rng = np.random.default_rng(2024)

    for a_eq, sigma in ((True, -1.0), (False, 1.0)):
        rows = ref_bnd.driving_rows(
            n_node, [3], pairing.slave_to_master, a_eq
        )

        A = rng.normal(size=n_node)
        F = rng.normal(size=n_node)
        K = rng.normal(size=(n_node, n_node))
        J = sp_sparse.csr_matrix(K @ K.T + n_node * np.eye(n_node))  # 对称正定

        # 受约束行上读到的值必须精确等于约束表达式（乘 1.0 是精确运算）
        assert (rows.mat_driven @ A)[3] == A[3], "Dirichlet 行的值不是精确的 A_i"
        for i_s, i_m in pairing.slave_to_master.items():
            assert (rows.mat_driven @ A)[i_s] == A[i_s] + sigma * A[i_m], (
                "约束行的值不是精确的 A_s + σ·A_m"
            )

        F_driven = ref_bnd.driving_row_residual(F, A, rows)
        J_driven = ref_bnd.driving_row_jacobian(J, rows).toarray()

        delta = np.linalg.solve(J_driven, -F_driven)
        A_new = A + delta

        scale = np.abs(A).max()
        worst = max(
            abs(A_new[i_s] + sigma * A_new[i_m])
            for i_s, i_m in pairing.slave_to_master.items()
        )
        assert worst < 1e-13 * scale, (
            f"a_eq={a_eq} 时一步 Newton 后约束残差 {worst:.3e} 过大（尺度 {scale:.3e}）"
        )
        assert abs(A_new[3]) < 1e-13 * scale, (
            f"Dirichlet 节点未被一步打到零，实得 {A_new[3]:.3e}"
        )


def test_free_rows_are_bit_identical(mesh_one_sector, pairing):
    """自由行的**非 Dirichlet 列**逐位不变；Dirichlet 列被零（对称修复）。

    ⚠ **2026-09-24 对称 Dirichlet**：``driving_row_jacobian`` 在行替换之外再零
    齐次 Dirichlet 列（``col_mask``）—— 自由行的物理条目只在 Dirichlet 列上被
    清零（乘 A_dirich=0，解不变），其余列**逐位**不变。残差路径不受列影响，自由行
    仍逐位不变。
    """
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    rng = np.random.default_rng(99)

    for a_eq in (True, False):
        dirichlet = [1, 2]
        rows = ref_bnd.driving_rows(
            n_node, dirichlet, pairing.slave_to_master, a_eq
        )
        free = np.flatnonzero(rows.is_driven == 0)

        F = rng.normal(size=n_node)
        A = rng.normal(size=n_node)
        F_driven = ref_bnd.driving_row_residual(F, A, rows)
        assert np.array_equal(F_driven[free], F[free]), "自由行的残差被改动了"

        J = sp_sparse.csr_matrix(rng.normal(size=(n_node, n_node)))
        J_driven = ref_bnd.driving_row_jacobian(J, rows).toarray()
        # 自由行的非 Dirichlet 列逐位不变；Dirichlet 列被清零（对称修复）
        keep = np.ones(n_node, dtype=bool)
        keep[dirichlet] = False
        assert np.array_equal(
            J_driven[np.ix_(free, keep)], J.toarray()[np.ix_(free, keep)]
        ), "自由行的非 Dirichlet 列被改动了"
        assert np.all(J_driven[np.ix_(free, dirichlet)] == 0.0), (
            "自由行的 Dirichlet 列没被清零 —— 对称修复没落地"
        )


def test_j_driven_is_diagonal_row_scaling(mesh_one_sector, pairing):
    """``j_driven @ J`` 必须逐位等于「按行缩放」，证明 lil→csr 是无害的。

    ``j_driven`` 是对角 0/1 矩阵，每个结果项只有一个非零乘积（乘 1.0 是精确
    运算），没有浮点累加，所以存储格式与稀疏求和顺序都不参与结果。
    """
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    rng = np.random.default_rng(5)
    rows = ref_bnd.driving_rows(n_node, {0}, pairing.slave_to_master, False)

    dense = rows.j_driven.toarray()
    assert np.array_equal(dense, np.diag(np.diag(dense))), "j_driven 不是对角阵"
    assert np.array_equal(np.diag(dense), 1 - rows.is_driven), (
        "j_driven 的对角与 is_driven 不符"
    )

    J = rng.normal(size=(n_node, n_node))
    scaled = rows.j_driven @ J                      # 稀疏 @ 稠密 → np.matrix
    assert np.array_equal(
        np.asarray(scaled), J * (1 - rows.is_driven)[:, None]
    ), "j_driven @ J 不等于按行缩放"


def test_corner_node_is_driven_by_slave_after_dirichlet(mesh_one_sector, pairing):
    """角点 ``(0, m_R)`` 同时命中 Dirichlet 与从属，最终必须由**主从行**生效。

    这不是假想情形：角点 ``(0, R)`` 既在外圆上（Dirichlet）又在 θ=90° 射线上
    （从属）。legacy 的写入顺序是 Dirichlet 先、主从后，主从分支还会先清空该行
    （MMA10:2092），所以最终该行是约束 ``A_s + σ·A_m = 0``。
    """
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    corner = [
        i for i in range(n_node)
        if abs(xy[i, 0]) < 1e-15 and abs(np.hypot(*xy[i]) - max(RADII)) < 1e-12
    ]
    assert len(corner) == 1, "夹具网格里找不到外圆与 90° 射线的交点"
    corner = corner[0]
    assert corner in pairing.slave_to_master, "角点应当是从属节点"

    i_master = pairing.slave_to_master[corner]
    for a_eq, sigma in ((True, -1.0), (False, 1.0)):
        rows = ref_bnd.driving_rows(
            n_node, [corner], pairing.slave_to_master, a_eq
        )
        dense = rows.mat_driven.toarray()
        assert dense[corner, corner] == 1.0
        assert dense[corner, i_master] == sigma, (
            f"角点所在行被 Dirichlet 覆盖了（a_eq={a_eq}）：应为主从约束行"
        )
        assert np.count_nonzero(dense[corner]) == 2
        assert rows.is_driven[corner] == 1.0
        assert rows.j_driven[corner, corner] == 0.0


def test_row_replacements_hook(mesh_one_sector, pairing):
    """``row_replacements`` 钩子（内核 7 用）：整行替换、并清空该行原有内容。

    legacy 对动网格行写 ``F_mat_driven[i, :] = K_rotating_boundary[i, :]``，
    这是 ``lil_matrix`` 的整行赋值（**清空后写入**）。这里验证：注入行逐位等于
    给定向量，且该行原有的 Dirichlet 条目被清掉。
    """
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    target = 7
    assert target not in pairing.slave_to_master, "夹具选点撞上从属节点，需换点"

    row_vector = np.zeros(n_node)
    row_vector[0] = -1.0
    row_vector[2] = 0.5
    row_vector[target] = 1.0

    for impl in (ref_bnd, fast_bnd):
        rows = impl.driving_rows(
            n_node,
            dirichlet=[target],          # 先写 Dirichlet 行，再被注入行覆盖
            slave_to_master=pairing.slave_to_master,
            a_eq=False,
            row_replacements={target: row_vector},
        )
        assert np.array_equal(rows.mat_driven.toarray()[target], row_vector), (
            "注入行不等于给定向量（原有 Dirichlet 条目可能没被清空）"
        )
        assert np.array_equal(rows.j_mat_driven.toarray()[target], row_vector)
        assert rows.is_driven[target] == 1.0
        assert rows.j_driven[target, target] == 0.0


@pytest.mark.parametrize("bad_len", [3, 0])
def test_row_replacements_rejects_wrong_length(pairing, bad_len):
    """注入行长度不对时必须报错，不能静默截断。"""
    n_node = max(pairing.slave_to_master) + 1
    with pytest.raises(ValueError, match="长度"):
        ref_bnd.driving_rows(
            n_node, [], pairing.slave_to_master, False,
            row_replacements={0: np.zeros(bad_len)},
        )


def test_unmatched_slave_raises():
    """从属节点找不到主动节点时必须报错（legacy 是 assert）。

    构造：只有一个 θ=90° 的从属节点，但 θ=0° 上没有同半径的节点。
    """
    xy = np.array([[0.01, 0.0], [0.02, 0.0], [0.0, 0.05]])
    with pytest.raises(ValueError, match="找不到"):
        ref_bnd.pair_master_slave(xy, parts=4)
    with pytest.raises(ValueError, match="找不到"):
        fast_bnd.pair_master_slave(xy, parts=4)


# =============================================================================
# 来源 A：变换矩阵的符号关系（★ 潜伏项，见 reference/boundaries.py 第四节 ★）
# =============================================================================

def test_transform_sign_is_minus_theory_when_antiperiodic(virtual, pairing):
    """固化 ``T_legacy = −T_theory`` 这个事实（**反周期**分支，即默认分支）。

    理论文档的 ``p_s``（约 2844 行）是「节点在主动边界上取 −1、否则取 +1」——
    注意该公式是在**反周期假设**下写的（同节开头：「假设主从边界为反周期边界，
    即主从边界的场量大小相等符号相反」）。legacy 的 ``_ki/_kj/_km``
    （MMA10:1761）**逐槽位恰好相反**，即整体差一个负号。

    这条测试不是「承认 legacy 正确」，而是把现状钉住：谁若把它「简化」成
    ``T_theory``，这里会立刻失败并指回 reference 模块第四节的解释。
    """
    # 防退化：必须有虚拟单元含主动槽位，否则符号向量恒为常数、测试平凡
    assert any(
        any(n in pairing.master_nodes for n in nodes)
        for nodes in virtual.virtual_tri.values()
    ), "夹具里没有含主动槽位的虚拟单元，本条测试失去意义"

    saw_negative = False
    for i_elem, nodes in virtual.virtual_tri.items():
        legacy_signs = ref_bnd.virtual_element_signs(
            nodes, pairing.master_nodes, a_eq=False
        )
        theory_signs = np.array(
            [-1.0 if n in pairing.master_nodes else 1.0 for n in nodes]
        )
        assert np.array_equal(legacy_signs, -theory_signs), (
            f"单元 {i_elem}：legacy 符号不再是理论符号的相反数"
        )
        saw_negative |= bool((legacy_signs < 0).any())
    assert saw_negative, "反周期分支下符号从未出现 −1，测试平凡"


def test_signs_are_all_one_when_periodic(virtual, pairing):
    """**周期**分支下符号恒为 +1 → 变换矩阵退化为单位阵（两个实现一致）。

    理论文档那条 ``p_s = −1`` 只适用于反周期；周期边界是 ``A_slave = +A_master``，
    把主动值映回单元局部向量时不必取反，所以符号恒为 +1。
    """
    # 防退化：与上一条共用同一个「含主动槽位」的前提
    assert any(
        any(n in pairing.master_nodes for n in nodes)
        for nodes in virtual.virtual_tri.values()
    ), "夹具里没有含主动槽位的虚拟单元，本条测试失去意义"

    for nodes in virtual.virtual_tri.values():
        assert np.array_equal(
            ref_bnd.virtual_element_signs(nodes, pairing.master_nodes, True),
            np.ones(3),
        )
        assert np.array_equal(
            fast_bnd.virtual_element_signs(nodes, pairing.master_nodes, True),
            np.ones(3),
        )


def test_transform_sign_invariance_and_sensitivity():
    """量化符号翻转的影响范围：二次型**不变**，一次型**变号**。

    对任意对称 ``K`` 与任意对角 ±1 的 ``T``：

        (−T)ᵀ K (−T) = Tᵀ K T          逐位相同（两次取反精确抵消）
        (−T)ᵀ r      = −(Tᵀ r)          逐位相反
        coef @ (−T)  = −(coef @ T)      逐位相反

    这正是「刚度/饱和项不受影响、而源项受影响」在浮点层面的原因，也是
    ``reference/boundaries.py`` 第四节那张表的依据。取反是精确运算，所以这里
    用 ``np.array_equal`` 而非容差比较。
    """
    rng = np.random.default_rng(11)
    signs = rng.choice([-1.0, 1.0], size=3)
    T = np.diag(signs)
    K = rng.normal(size=(3, 3))
    K = K + K.T                                  # 对称（单元刚度的性质）
    r = rng.normal(size=3)
    coef = rng.normal(size=3)

    assert np.array_equal(T.T @ K @ T, (-T).T @ K @ (-T)), "二次型在 T → −T 下改变了"
    assert np.array_equal(T.T @ r, -((-T).T @ r)), "一次型未在 T → −T 下变号"
    assert np.array_equal(coef @ T, -(coef @ (-T))), "系数行未在 T → −T 下变号"


# =============================================================================
# 来源 A：rebind（内核 7 的重编号入口）
# =============================================================================

def test_rebind_maps_pairs_through_node_remap(pairing, mesh_one_sector):
    """``rebind`` 必须为每个落在映射里的从属节点补一对「像」。

    对应 MMA10:465-479：插入动网格层后，气隙单元里的定子侧节点号被换成新造的
    转子侧节点号，主从表要跟着迁移。映射覆盖的正是气隙环上的节点，所以**从属与
    主动两侧都在映射里**（`node_gap_stator` 含环上全部节点）。旧条目按 legacy
    保真保留。
    """
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    involved = set(pairing.slave_to_master) | set(pairing.master_to_slave)
    node_remap = {i: i + n_node for i in involved}

    new_pairing = pairing.rebind(node_remap)

    assert len(new_pairing.slave_to_master) == 2 * len(pairing.slave_to_master), (
        "新表的条目数应为「旧条目 + 每个从属节点一条」"
    )
    for i_slave, i_master in pairing.slave_to_master.items():
        assert new_pairing.slave_to_master[node_remap[i_slave]] == node_remap[i_master]
        assert new_pairing.master_to_slave[node_remap[i_master]] == node_remap[i_slave]
        assert new_pairing.slave_to_master[i_slave] == i_master, "旧条目应当保真保留"

    assert set(new_pairing.slave_to_master) - set(pairing.slave_to_master) == {
        node_remap[i_slave] for i_slave in pairing.slave_to_master
    }, "不该产生映射之外的条目"


def test_rebind_raises_when_master_missing(pairing):
    """主动节点不在映射里时必须报错（legacy 此处会 KeyError）。"""
    i_slave = next(iter(pairing.slave_to_master))
    with pytest.raises(ValueError, match="不在 node_remap"):
        pairing.rebind({i_slave: 10_000})


# =============================================================================
# 来源 B：与 legacy 逐位一致
# =============================================================================

def test_matches_legacy_em_pairing(mesh_one_sector):
    """复刻 MMA10:305-321（电磁侧配对）逐位比对。"""
    xy, _ = mesh_one_sector
    coord_err_abs = 1e-6
    parts = 4

    _node = np.column_stack([np.arange(xy.shape[0]), xy])

    node_master = [
        int(n[0]) for n in _node
        if np.abs(n[2]) < coord_err_abs < np.abs(n[1])
    ]
    node_slave = {}
    for i_node_s in [
        int(n[0]) for n in _node
        if np.abs(legacy_rot_y(n[1], n[2], -2 * np.pi / parts)) < coord_err_abs
        < np.abs(legacy_rot_x(n[1], n[2], -2 * np.pi / parts))
    ]:
        for i_node_m in node_master:
            if (
                np.abs(
                    legacy_rot_x(_node[i_node_m][1], _node[i_node_m][2], 2 * np.pi / parts)
                    - _node[i_node_s][1]
                ) < coord_err_abs
                and np.abs(
                    legacy_rot_y(_node[i_node_m][1], _node[i_node_m][2], 2 * np.pi / parts)
                    - _node[i_node_s][2]
                ) < coord_err_abs
            ):
                node_slave[i_node_s] = i_node_m
                break
        assert i_node_s in node_slave
    node_master_dict = dict((v, k) for (k, v) in node_slave.items())

    assert len(node_slave) > 0, "legacy 片段在夹具网格上没配出任何一对，测试无意义"
    for got in (
        ref_bnd.pair_master_slave(xy, parts=parts),
        fast_bnd.pair_master_slave(xy, parts=parts),
    ):
        assert got.slave_to_master == node_slave, "与 legacy 的 node_slave 不一致"
        assert got.master_to_slave == node_master_dict, "与 legacy 的 node_master 不一致"


def test_matches_legacy_mech_pairing_on_another_table(mesh_one_sector):
    """复刻 MMA10:402-419（力学侧配对）—— 作用在**另一张**节点表上。

    电磁侧与力学侧是同一个算法、不同的节点表（力学表是压缩重编号后的）。
    这里把最外两层半径上的节点单独取出来重编号成一张新表，验证实现与节点表无关，
    从而支持「两侧共用同一份实现」这个结论。
    """
    xy, _ = mesh_one_sector
    coord_err_abs = 1e-6
    parts = 4
    n_ang = DEFAULT_ANGLES.size

    keep = [i for i in range(xy.shape[0]) if i // n_ang in (3, 4)]
    xy_mech = xy[keep]
    _node_mech = np.column_stack([np.arange(xy_mech.shape[0]), xy_mech])

    node_mech_master = [
        int(n[0]) for n in _node_mech
        if np.abs(n[2]) < coord_err_abs < np.abs(n[1])
    ]
    node_mech_slave = {}
    for i_node_s in [
        int(n[0]) for n in _node_mech
        if np.abs(legacy_rot_y(n[1], n[2], -2 * np.pi / parts)) < coord_err_abs
        < np.abs(legacy_rot_x(n[1], n[2], -2 * np.pi / parts))
    ]:
        for i_node_m in node_mech_master:
            if (
                np.abs(
                    legacy_rot_x(
                        _node_mech[i_node_m][1], _node_mech[i_node_m][2], 2 * np.pi / parts
                    )
                    - _node_mech[i_node_s][1]
                ) < coord_err_abs
                and np.abs(
                    legacy_rot_y(
                        _node_mech[i_node_m][1], _node_mech[i_node_m][2], 2 * np.pi / parts
                    )
                    - _node_mech[i_node_s][2]
                ) < coord_err_abs
            ):
                node_mech_slave[i_node_s] = i_node_m
                break
        assert i_node_s in node_mech_slave

    assert len(node_mech_slave) > 0, "力学子表里没有从属节点，测试无意义"
    got = ref_bnd.pair_master_slave(xy_mech, parts=parts)
    assert got.slave_to_master == node_mech_slave, "力学节点表上与 legacy 不一致"


def test_matches_legacy_slave_elements(mesh_one_sector, pairing):
    """复刻 MMA10:323-335（从属虚拟单元）逐位比对。

    legacy 的 ``element_S`` 是 5 列 ``[单元号, n1, n2, n3, 区域号]``；
    ``element_S_slave`` 保留区域号（虽然全文件没有消费者读它）。
    """
    _, tri = mesh_one_sector
    element_S = np.column_stack(
        [np.arange(tri.shape[0]), tri, np.zeros(tri.shape[0], dtype=int)]
    )
    node_slave = pairing.slave_to_master

    element_slave = set(
        [_elem[0] for _elem in element_S
         if (_elem[1] in node_slave or _elem[2] in node_slave or _elem[3] in node_slave)]
    )
    element_S_slave = {}
    for i_elem in element_slave:
        element_S_slave[i_elem] = [
            element_S[i_elem][0],
            node_slave[element_S[i_elem][1]] if element_S[i_elem][1] in node_slave
            else element_S[i_elem][1],
            node_slave[element_S[i_elem][2]] if element_S[i_elem][2] in node_slave
            else element_S[i_elem][2],
            node_slave[element_S[i_elem][3]] if element_S[i_elem][3] in node_slave
            else element_S[i_elem][3],
            element_S[i_elem][4],
        ]

    assert len(element_slave) > 0, "legacy 片段在夹具网格上没找出从属单元，测试无意义"
    for got in (
        ref_bnd.slave_virtual_elements(tri, pairing),
        fast_bnd.slave_virtual_elements(tri, pairing),
    ):
        assert got.elements == frozenset(element_slave), "从属单元集合与 legacy 不一致"
        assert set(got.virtual_tri) == set(element_S_slave)
        for i_elem, nodes in got.virtual_tri.items():
            assert list(nodes) == element_S_slave[i_elem][1:4], (
                f"单元 {i_elem} 的虚拟连接与 legacy 不一致"
            )


def test_matches_legacy_driving_rows(mesh_one_sector, pairing):
    """复刻 MMA10:2073-2102（驱动行）逐位比对。

    legacy 用 ``lil_matrix`` 逐项赋值；本内核用 COO 一次装配。值只有 0/±1，
    没有浮点累加，因此两种装配方式必须给出**逐位**相同的矩阵。
    """
    xy, _ = mesh_one_sector
    N_node = xy.shape[0]
    node_Dirichlet = {0, 1}
    node_slave = pairing.slave_to_master
    EM_MS_boundary_A_eq = False

    F_is_driven = np.zeros(N_node)
    F_mat_driven = sp_sparse.lil_matrix((N_node, N_node))
    J_driven = sp_sparse.eye(N_node).tolil()
    J_mat_driven = sp_sparse.lil_matrix((N_node, N_node))

    for i_node in node_Dirichlet:
        F_is_driven[i_node] = 1
        F_mat_driven[i_node, i_node] = 1

        J_driven[i_node, i_node] = 0
        J_mat_driven[i_node, i_node] = 1

    for i_node_s, i_node_m in node_slave.items():
        F_is_driven[i_node_s] = 1
        F_mat_driven[i_node_s, :] = 0
        F_mat_driven[i_node_s, i_node_s] = 1
        F_mat_driven[i_node_s, i_node_m] = (-1 if EM_MS_boundary_A_eq else 1)

        J_driven[i_node_s, i_node_s] = 0
        J_mat_driven[i_node_s, i_node_s] = 1
        J_mat_driven[i_node_s, i_node_m] = (-1 if EM_MS_boundary_A_eq else 1)

    for got in (
        ref_bnd.driving_rows(N_node, node_Dirichlet, node_slave, EM_MS_boundary_A_eq),
        fast_bnd.driving_rows(N_node, node_Dirichlet, node_slave, EM_MS_boundary_A_eq),
    ):
        assert np.array_equal(got.is_driven, F_is_driven), "is_driven 与 legacy 不一致"
        assert np.array_equal(
            got.mat_driven.toarray(), F_mat_driven.toarray()
        ), "mat_driven 与 legacy 不一致"
        assert np.array_equal(
            got.j_driven.toarray(), J_driven.toarray()
        ), "j_driven 与 legacy 不一致"
        assert np.array_equal(
            got.j_mat_driven.toarray(), J_mat_driven.toarray()
        ), "j_mat_driven 与 legacy 不一致"


@pytest.mark.parametrize("a_eq", [True, False])
def test_matches_legacy_driving_rows_both_branches(mesh_one_sector, pairing, a_eq):
    """``EM_MS_boundary_A_eq`` 的 True（周期）/ False（反周期）分支都要逐位一致。"""
    xy, _ = mesh_one_sector
    N_node = xy.shape[0]
    node_Dirichlet = {0}
    node_slave = pairing.slave_to_master

    F_mat_driven = sp_sparse.lil_matrix((N_node, N_node))
    for i_node in node_Dirichlet:
        F_mat_driven[i_node, i_node] = 1
    for i_node_s, i_node_m in node_slave.items():
        F_mat_driven[i_node_s, :] = 0
        F_mat_driven[i_node_s, i_node_s] = 1
        F_mat_driven[i_node_s, i_node_m] = (-1 if a_eq else 1)

    for got in (
        ref_bnd.driving_rows(N_node, node_Dirichlet, node_slave, a_eq),
        fast_bnd.driving_rows(N_node, node_Dirichlet, node_slave, a_eq),
    ):
        assert np.array_equal(got.mat_driven.toarray(), F_mat_driven.toarray())


def test_matches_legacy_row_replacement(mesh_one_sector, pairing):
    """复刻 MMA10:2104-2112 的动网格行写法（整行替换）逐位比对。

    内核 7 会真正用 ``K_rotating_boundary`` 填这一行；这里用一个人造行验证
    ``row_replacements`` 与 legacy 的 ``lil[i, :] = row`` 逐位等价。
    """
    xy, _ = mesh_one_sector
    N_node = xy.shape[0]
    node_slave = pairing.slave_to_master
    node_Dirichlet = {0}
    gap_nodes = [2, 3]
    assert all(i not in node_slave for i in gap_nodes)

    K_rotating_boundary = np.zeros((N_node, N_node))
    for i in gap_nodes:                     # 让它像插值行：对角 −1 与两个权重
        K_rotating_boundary[i, i] = -1.0
        K_rotating_boundary[i, 1] = 0.125
        K_rotating_boundary[i, 4] = 0.875

    # legacy 逐行照抄
    F_mat_driven = sp_sparse.lil_matrix((N_node, N_node))
    J_driven = sp_sparse.eye(N_node).tolil()
    J_mat_driven = sp_sparse.lil_matrix((N_node, N_node))
    for i_node in node_Dirichlet:
        F_mat_driven[i_node, i_node] = 1
        J_driven[i_node, i_node] = 0
        J_mat_driven[i_node, i_node] = 1
    for i_node_s, i_node_m in node_slave.items():
        F_mat_driven[i_node_s, :] = 0
        F_mat_driven[i_node_s, i_node_s] = 1
        F_mat_driven[i_node_s, i_node_m] = 1
        J_driven[i_node_s, i_node_s] = 0
        J_mat_driven[i_node_s, i_node_s] = 1
        J_mat_driven[i_node_s, i_node_m] = 1
    for i_node_gap_rotor in gap_nodes:
        F_mat_driven[i_node_gap_rotor, :] = K_rotating_boundary[i_node_gap_rotor, :]
        J_driven[i_node_gap_rotor, i_node_gap_rotor] = 0
        J_mat_driven[i_node_gap_rotor, :] = K_rotating_boundary[i_node_gap_rotor, :]

    for impl in (ref_bnd, fast_bnd):
        got = impl.driving_rows(
            N_node,
            node_Dirichlet,
            node_slave,
            a_eq=False,
            row_replacements={i: K_rotating_boundary[i] for i in gap_nodes},
        )
        assert np.array_equal(got.mat_driven.toarray(), F_mat_driven.toarray()), (
            f"{impl.__name__}: mat_driven 与 legacy 不一致"
        )
        assert np.array_equal(got.j_driven.toarray(), J_driven.toarray()), (
            f"{impl.__name__}: j_driven 与 legacy 不一致"
        )
        assert np.array_equal(got.j_mat_driven.toarray(), J_mat_driven.toarray()), (
            f"{impl.__name__}: j_mat_driven 与 legacy 不一致"
        )


# =============================================================================
# 来源 C：ref vs fast 逐位一致
# =============================================================================

@pytest.mark.parametrize(
    "angles",
    [
        np.arange(0.0, 90.0 + 1e-9, 10.0),      # 单扇区
        np.arange(0.0, 180.0 + 1e-9, 10.0),     # 跨两个扇区
        np.arange(0.0, 90.0 + 1e-9, 5.0),       # 加密
    ],
    ids=["单扇区", "跨扇区", "加密"],
)
def test_ref_fast_parity_on_polar_meshes(angles):
    """极坐标网格上 ref 与 fast 逐位相同（配对、虚拟单元、符号、驱动行）。"""
    xy, tri = make_polar_mesh(angles_deg=angles)
    n_node = xy.shape[0]

    p_ref = ref_bnd.pair_master_slave(xy, parts=4)
    p_fast = fast_bnd.pair_master_slave(xy, parts=4)
    assert p_ref.slave_to_master == p_fast.slave_to_master
    assert p_ref.master_to_slave == p_fast.master_to_slave

    v_ref = ref_bnd.slave_virtual_elements(tri, p_ref)
    v_fast = fast_bnd.slave_virtual_elements(tri, p_fast)
    assert v_ref.elements == v_fast.elements
    assert v_ref.virtual_tri == v_fast.virtual_tri

    nodes = list(v_ref.virtual_tri.values())
    assert len(nodes) > 0, "夹具网格里没有从属单元，测试无意义"
    for a_eq in (True, False):
        assert np.array_equal(
            ref_bnd.virtual_element_signs(nodes, p_ref.master_nodes, a_eq),
            fast_bnd.virtual_element_signs(nodes, p_fast.master_nodes, a_eq),
        )
        dirichlet = {0, 3}
        r = ref_bnd.driving_rows(n_node, dirichlet, p_ref.slave_to_master, a_eq)
        f = fast_bnd.driving_rows(n_node, dirichlet, p_fast.slave_to_master, a_eq)
        assert np.array_equal(r.is_driven, f.is_driven)
        assert np.array_equal(r.mat_driven.toarray(), f.mat_driven.toarray())
        assert np.array_equal(r.j_driven.toarray(), f.j_driven.toarray())
        assert np.array_equal(r.j_mat_driven.toarray(), f.j_mat_driven.toarray())


def test_ref_fast_parity_on_random_angles():
    """**随机角度**网格上逐位相同 —— 规则网格的规律性可能掩盖索引错误。"""
    rng = np.random.default_rng(20240922)
    angles = np.sort(rng.uniform(0.0, 90.0, 9))
    angles[0], angles[-1] = 0.0, 90.0
    xy, tri = make_polar_mesh(angles_deg=angles)
    n_node = xy.shape[0]

    p_ref = ref_bnd.pair_master_slave(xy, parts=4)
    p_fast = fast_bnd.pair_master_slave(xy, parts=4)
    assert len(p_ref.slave_to_master) > 0, "随机网格里没有配成对，测试无意义"
    assert p_ref.slave_to_master == p_fast.slave_to_master

    v_ref = ref_bnd.slave_virtual_elements(tri, p_ref)
    v_fast = fast_bnd.slave_virtual_elements(tri, p_fast)
    assert v_ref.elements == v_fast.elements
    assert v_ref.virtual_tri == v_fast.virtual_tri

    # 同一组随机行向量，两个实现必须给出逐位相同的驱动行矩阵
    row_vector = rng.normal(size=n_node)
    dirichlet = {0, 1, 2}
    r = ref_bnd.driving_rows(
        n_node, dirichlet, p_ref.slave_to_master, False,
        row_replacements={5: row_vector},
    )
    f = fast_bnd.driving_rows(
        n_node, dirichlet, p_fast.slave_to_master, False,
        row_replacements={5: row_vector},
    )
    assert np.array_equal(r.is_driven, f.is_driven)
    assert np.array_equal(r.mat_driven.toarray(), f.mat_driven.toarray())
    assert np.array_equal(r.j_mat_driven.toarray(), f.j_mat_driven.toarray())


def test_ref_fast_parity_of_row_scaling_after_lil_to_csr(mesh_one_sector, pairing):
    """``j_driven @ J``（lil→csr 的等价性）在 ref 与 fast 上也逐位相同。"""
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    rng = np.random.default_rng(4)
    J = sp_sparse.csr_matrix(rng.normal(size=(n_node, n_node)))

    r = ref_bnd.driving_rows(n_node, {0}, pairing.slave_to_master, False)
    f = fast_bnd.driving_rows(n_node, {0}, pairing.slave_to_master, False)
    assert np.array_equal(
        ref_bnd.driving_row_jacobian(J, r).toarray(),
        fast_bnd.driving_row_jacobian(J, f).toarray(),
    )


# =============================================================================
# 来源 B（续）：legacy 消费端表达式在浮点层面的符号后果
# =============================================================================

def test_matches_legacy_consumer_expressions_at_float_level(pairing, virtual):
    """照抄 legacy 三个**消费端**表达式，把符号关系钉死在浮点层。

    legacy::

        _K0 = _T.T @ (_K0elem @ _T)                # MMA10:2021
        Bxcoef_mat[e, _index] = Bxcoef[e] @ _T     # MMA10:2000-2001
        T_R0elem = _T.T @ np.array([R0, R0, R0])   # MMA10:2203

    把 ``_T`` 换成整体取负的 ``−T``：**偶次**表达式逐位不变，**奇次**逐位反号。
    取反是精确运算，所以全部用 ``np.array_equal``。这条把 reference 模块第四节
    那张表从「推理」变成「可执行的浮点事实」。
    """
    rng = np.random.default_rng(13)
    assert len(virtual.virtual_tri) > 0, "没有虚拟单元，测试无意义"

    for i_elem in list(virtual.virtual_tri)[:4]:
        nodes = virtual.virtual_tri[i_elem]
        signs = ref_bnd.virtual_element_signs(nodes, pairing.master_nodes, a_eq=False)
        _T = np.diag(signs)

        # legacy 的对称打包存储：0 kii, 1 kij, 2 kim, 3 kjj, 4 kjm, 5 kmm（MMA10:2015-2019）
        K0elem = rng.normal(size=6)
        _K0elem = np.array(
            [
                [K0elem[0], K0elem[1], K0elem[2]],
                [K0elem[1], K0elem[3], K0elem[4]],
                [K0elem[2], K0elem[4], K0elem[5]],
            ]
        )
        Bxcoef = rng.normal(size=3)
        R0elem = float(rng.normal())

        _K0 = _T.T @ (_K0elem @ _T)
        _K0_negated = (-_T).T @ (_K0elem @ (-_T))
        assert np.array_equal(_K0, _K0_negated), "TᵀKT 在整体取负下改变了"

        bx = Bxcoef @ _T
        bx_negated = Bxcoef @ (-_T)
        assert np.array_equal(bx, -bx_negated), "系数行未在整体取负下反号"

        t_r = _T.T @ np.array([R0elem, R0elem, R0elem])
        t_r_negated = (-_T).T @ np.array([R0elem, R0elem, R0elem])
        assert np.array_equal(t_r, -t_r_negated), "源项未在整体取负下反号"


# =============================================================================
# 定义域与接口边界
# =============================================================================

def test_signs_must_not_be_used_on_non_slave_elements(mesh_one_sector, pairing, virtual):
    """★ 符号函数的定义域是**从属单元**；误用到非从属单元会**真算错**。★

    legacy 的非从属分支是硬编码 ``_T = np.eye(3)``（MMA10:1767-1771、1863），
    **不调用**符号公式。若对含「已配对主动节点」的非从属单元调用
    ``virtual_element_signs``，会得到**混合**符号模式（既有 +1 又有 −1）。

    混合模式 != 整体取负 ⇒ ``TᵀKT`` **会改变** —— 这不是末位差异，是数值错误。
    真实网格上这样的单元有 134 个，**全部**是混合模式（无一幸免）。
    """
    _, tri = mesh_one_sector

    target = None
    for e in range(tri.shape[0]):
        if e in virtual.elements:
            continue
        if any(n in pairing.master_nodes for n in tri[e]):
            target = e
            break
    assert target is not None, "夹具网格里找不到这样的单元，测试无意义"

    elem_nodes = tri[target]
    signs = ref_bnd.virtual_element_signs(elem_nodes, pairing.master_nodes, a_eq=False)
    assert (signs < 0).any() and (signs > 0).any(), (
        f"误用后应得到混合符号模式，实得 {signs}"
    )

    rng = np.random.default_rng(8)
    K = rng.normal(size=(3, 3))
    K = K + K.T
    T_misuse = np.diag(signs)
    T_correct = np.eye(3)
    assert not np.array_equal(
        T_misuse.T @ K @ T_misuse, T_correct.T @ K @ T_correct
    ), "混合符号竟然没有改变 TᵀKT？那这条警示的推导要重查"

    # 对照：整体取负（从属单元的正确形态）确实不改变二次型
    T_negated = -np.eye(3)
    assert np.array_equal(T_negated.T @ K @ T_negated, T_correct.T @ K @ T_correct)


def test_legacy_equals_minus_theory_requires_all_masters_paired():
    """★ ``T_legacy = −T_theory`` 需要「主动射线上每个节点都配对成功」这个前提。★

    造一个 [0°, 90°, 180°] 的网格：θ=180 射线上的节点也是主动**候选**，但没有
    从属节点与之配对（网格里没有 θ=270° 的节点）。于是跨 90°~180° 的虚拟单元会
    含「在主动射线上但没配上」的槽位：

        legacy: 节点不在配对表里 → −1
        理论  : 节点在主动射线上 → −1
                                 ↑ 同号

    此时既不是整体取负，``TᵀKT`` 的不变性也一并失效。真实网格上该前提恰好满足
    （主动射线 66 个节点全部配对），所以这个反例只能靠合成网格构造。
    """
    xy, tri = make_polar_mesh(radii=(0.02, 0.05), angles_deg=[0.0, 90.0, 180.0])
    pairing = ref_bnd.pair_master_slave(xy, parts=4)
    virtual = ref_bnd.slave_virtual_elements(tri, pairing)

    master_ray = {
        i for i in range(xy.shape[0])
        if abs(xy[i, 1]) < 1e-6 and abs(xy[i, 0]) > 1e-6
    }
    unpaired = master_ray - set(pairing.master_to_slave)
    assert len(unpaired) > 0, "夹具没有造出「主动射线上未配对」的节点，测试无意义"
    assert len(pairing.master_to_slave) < len(master_ray)

    found = None
    for e, nodes in virtual.virtual_tri.items():
        if any(n in unpaired for n in nodes):
            found = (e, nodes)
            break
    assert found is not None, "夹具里没有含未配对主动射线槽位的虚拟单元"
    _, nodes = found

    legacy_signs = ref_bnd.virtual_element_signs(nodes, pairing.master_nodes, False)
    theory_signs = np.array([-1.0 if n in master_ray else 1.0 for n in nodes])

    assert not np.array_equal(legacy_signs, -theory_signs), (
        "前提被破坏时等式竟然仍成立？推导或夹具需要重查"
    )
    diverging = [k for k in range(3) if legacy_signs[k] != -theory_signs[k]]
    assert all(
        legacy_signs[k] == -1.0 and theory_signs[k] == -1.0 for k in diverging
    ), "分歧点应当是「两侧同为 −1」，否则说明我对前提的理解有误"

    rng = np.random.default_rng(21)
    K = rng.normal(size=(3, 3))
    K = K + K.T
    T_legacy = np.diag(legacy_signs)
    T_theory = np.diag(theory_signs)
    assert not np.array_equal(
        T_legacy.T @ K @ T_legacy, T_theory.T @ K @ T_theory
    ), "前提破坏时 TᵀKT 竟然仍不变，与推导矛盾"


def test_virtual_tri_keeps_duplicate_nodes():
    """虚拟三节点允许重复 —— legacy 不去重，本内核也不去重。

    去重会改变单元自由度个数的语义（``_T`` 恒为 3×3），属静默的行为改动。
    """
    xy, tri = make_polar_mesh(radii=(0.02, 0.05), angles_deg=[0.0, 90.0])
    pairing = ref_bnd.pair_master_slave(xy, parts=4)
    virtual = ref_bnd.slave_virtual_elements(tri, pairing)

    degenerated = [
        nodes for nodes in virtual.virtual_tri.values() if len(set(nodes)) < 3
    ]
    assert len(degenerated) > 0, "夹具没有造出退化虚拟单元，测试无意义"

    for e, nodes in virtual.virtual_tri.items():
        expected = tuple(
            pairing.slave_to_master[n] if n in pairing.slave_to_master else n
            for n in tri[e]
        )
        assert nodes == expected, f"单元 {e} 的虚拟连接与「只替换不去重」不符"


def test_tolerance_criterion_and_origin_exclusion():
    """判据的严格性、原点排除，以及**经旋转那一侧在 ``= err`` 处的浮点模糊带**。

    - 原点 ``(0,0)`` 被 ``err < |x|`` 排除：``|y| < err`` 成立但 ``err < 0`` 不成立。
      若不排除，它会同时「在两条射线上」，成为病态节点。
    - ``|y|``（主动判据）与 ``|x|``（从属判据里 ``err < |rot_x|`` 那一半）是
      **直接比较**，严格性可精确验证：恰好 ``= err`` 的节点被排除。
    - ⚠ **但 ``|rot_y| < err`` 那一半在 ``|x| = err`` 处是模糊的**，因为旋转本身
      不精确：``cos(π/2) = 6.123e-17 ≠ 0``，于是
      ``|rot_y| = ||x| − y·6.123e-17|``。实测 ``|x| = err``、``y = 0.05`` 时
      ``|rot_y| = err − 3.06e-18``（相对 3.1e-10），**仍小于 err** → 该节点被判为
      从属。要到 ``|x| ≥ 1.001·err`` 才被排除。

    因此**不能声称「恰好偏移 err 就被排除」**——那条只对直接比较的那两级成立。
    模糊带只有约 ``1e-3·err`` 宽，工程上无关紧要，但重写判据的人必须知道它存在。
    """
    err = 1e-6
    angle = 2 * np.pi / 4
    xy = np.array(
        [
            [0.0, 0.0],        # 0 原点
            [0.05, 0.0],       # 1 正常主动
            [0.05, err],       # 2 |y| 恰好 = err → 不算主动（直接比较，严格）
            [err, 0.0],        # 3 |x| 恰好 = err → 不算主动（直接比较，严格）
            [0.0, 0.05],       # 4 正常从属
            [err, 0.05],       # 5 `|x|` 恰好 = err，但旋转不精确 → 仍算从属
        ]
    )
    pairing = ref_bnd.pair_master_slave(xy, parts=4, coord_err_abs=err)

    assert pairing.slave_to_master == {4: 1, 5: 1}, (
        f"从属集合应为 {{4, 5}}（节点 5 在模糊带内），实得 {pairing.slave_to_master}"
    )
    # 两个从属映到同一主动 → 反向表按「后写覆盖」取节点号大的（与上一条测试同源）
    assert pairing.master_to_slave == {1: 5}

    for i_node in (0, 2, 3):
        assert i_node not in pairing.slave_to_master, f"节点 {i_node} 不该成为从属"
        assert i_node not in pairing.master_to_slave, f"节点 {i_node} 不该成为主动"

    # 直接比较的那两级：严格性可精确断言
    assert not abs(xy[2, 1]) < err, "|y| = err 竟然通过了 |y| < err"
    assert not err < abs(xy[3, 0]), "|x| = err 竟然通过了 err < |x|"
    assert not err < abs(xy[0, 0]), "原点竟然通过了 err < |x|"

    # 旋转那一级：确认模糊带存在且很窄
    ry = xy[5, 0] * np.sin(-angle) + xy[5, 1] * np.cos(-angle)
    assert abs(ry) < err, "旋转后的 |rot_y| 应略小于 err（即落在模糊带内）"
    assert err - abs(ry) < 1e-3 * err, (
        f"模糊带宽度 {err - abs(ry):.3e} 超出预期的 1e-3·err"
    )
    # 把 |x| 放大 0.1% 即被排除
    xy_scaled = xy.copy()
    xy_scaled[5, 0] = 1.001 * err
    excluded = ref_bnd.pair_master_slave(xy_scaled, parts=4, coord_err_abs=err)
    assert 5 not in excluded.slave_to_master, (
        "|x| = 1.001·err 应当被排除，说明模糊带比预期宽得多"
    )


def test_master_to_slave_last_write_wins_on_duplicate_slaves():
    """两个从属节点配对到同一个主动节点时，反向表保留**后写**的那个。

    legacy: ``node_master = dict((v, k) for (k, v) in node_slave.items())``
    （MMA10:321）—— dict 构造是「后写覆盖」，而 ``node_slave`` 的插入顺序是节点号
    升序，所以**节点号大的从属胜出**。向量化实现必须保留这个语义。
    """
    xy = np.array([[0.05, 0.0], [0.0, 0.05], [0.0, 0.05]])   # 后两个坐标完全相同
    for impl in (ref_bnd, fast_bnd):
        pairing = impl.pair_master_slave(xy, parts=4)
        assert pairing.slave_to_master == {1: 0, 2: 0}, "两个从属应都配到主动节点 0"
        assert pairing.master_to_slave == {0: 2}, (
            f"{impl.__name__}: 反向表未按「后写覆盖」取节点号大的从属"
        )


def test_row_replacements_wins_over_slave_row(mesh_one_sector, pairing):
    """同一行既在 ``slave_to_master`` 又在 ``row_replacements`` 时，注入行胜出。

    这是内核 7 组成正确性的守卫：legacy 的动网格行写在主从行之后，且是整行赋值
    （清空后写入）。顺序反了就会让插值行被主从行覆盖。
    """
    xy, _ = mesh_one_sector
    n_node = xy.shape[0]
    i_slave = sorted(pairing.slave_to_master)[0]

    row_vector = np.zeros(n_node)
    row_vector[1] = 2.0
    row_vector[i_slave] = 1.0

    for impl in (ref_bnd, fast_bnd):
        rows = impl.driving_rows(
            n_node, [], pairing.slave_to_master, False,
            row_replacements={i_slave: row_vector},
        )
        assert np.array_equal(rows.mat_driven.toarray()[i_slave], row_vector), (
            f"{impl.__name__}: 注入行未覆盖从属约束行"
        )
        assert np.array_equal(rows.j_mat_driven.toarray()[i_slave], row_vector)
        assert rows.is_driven[i_slave] == 1.0
        assert rows.j_driven[i_slave, i_slave] == 0.0


@pytest.mark.parametrize("impl", [ref_bnd, fast_bnd], ids=["ref", "fast"])
def test_slave_virtual_elements_rejects_wrong_column_count(impl, mesh_one_sector, pairing):
    """传 5 列的 legacy ``element_S`` 进来必须报错。

    否则两层的**同一个**错误会让来源 C（ref vs fast）也通过 —— 只有显式的形状
    检查能拦住。
    """
    _, tri = mesh_one_sector
    bad = np.column_stack(
        [np.arange(tri.shape[0]), tri, np.zeros(tri.shape[0], dtype=int)]
    )
    assert bad.shape[1] == 5
    with pytest.raises(ValueError, match="连接表"):
        impl.slave_virtual_elements(bad, pairing)


# =============================================================================
# P2（6 列）从属虚拟单元（2026-10-02，路线 A）
# =============================================================================


@pytest.fixture(scope="module")
def pairing_p2() -> tuple[np.ndarray, np.ndarray, ref_bnd.PeriodicBoundary]:
    """P2 极坐标网格 + 配对（θ=0 主动、θ=90° 从属）。"""
    xy, tri6 = make_polar_mesh_p2()
    return xy, tri6, ref_bnd.pair_master_slave(xy, parts=4)


def test_p2_slave_virtual_elements_remap_vertices_and_midpoints(pairing_p2):
    """6 列：从属判定与替换覆盖**全部槽位** —— θ=90° 射线上有顶点和边中点从属。"""
    xy, tri6, pairing = pairing_p2
    v = ref_bnd.slave_virtual_elements(tri6, pairing)
    slave_ids = set(pairing.slave_to_master)

    # 角色分类：顶点（列 0-2）vs 中点（列 3-5），合法 P2 网格互斥
    vertex_cols = {int(n) for row in tri6 for n in row[:3]}
    mid_cols = {int(n) for row in tri6 for n in row[3:]}
    assert not (vertex_cols & mid_cols)
    slave_verts = slave_ids & vertex_cols
    slave_mids = slave_ids & mid_cols
    assert slave_verts, "θ=90° 射线上应有从属顶点"
    assert slave_mids, "θ=90° 射线上应有从属边中点（列 3-5）"
    # 顶点数 = 环数；射线上边的中点 = 环数 − 1（最外环没有往外的边）
    assert len(slave_mids) == len(slave_verts) - 1

    # 每个含从属节点的单元：6 个槽位全部按从属→主动替换（非从属原样保留）
    for e in v.elements:
        assert any(n in slave_ids for n in tri6[e]), e
        virt = v.virtual_tri[e]
        assert len(virt) == 6, e
        for k in range(6):
            if tri6[e, k] in slave_ids:
                assert virt[k] == pairing.slave_to_master[tri6[e, k]], (e, k)
            else:
                assert virt[k] == tri6[e, k], (e, k)


def test_p2_virtual_element_signs_on_6_nodes(pairing_p2):
    """6 节点虚拟单元的符号逐槽位：主动 +1、非主动 −1（反周期 a_eq=False）。"""
    _xy, tri6, pairing = pairing_p2
    v = ref_bnd.slave_virtual_elements(tri6, pairing)
    nodes = list(v.virtual_tri.values())
    assert all(len(n) == 6 for n in nodes)
    signs = ref_bnd.virtual_element_signs(nodes, pairing.master_nodes, a_eq=False)
    assert signs.shape == (len(nodes), 6)
    for i, virt in enumerate(nodes):
        for k in range(6):
            expect = 1.0 if virt[k] in pairing.master_nodes else -1.0
            assert signs[i, k] == expect, (i, k)


def test_p2_ref_fast_parity_slave_virtual_elements(pairing_p2):
    """ref 与 fast 在 6 列 tri 上逐项相同（含中点从属的替换）。"""
    _xy, tri6, pairing = pairing_p2
    v_ref = ref_bnd.slave_virtual_elements(tri6, pairing)
    v_fast = fast_bnd.slave_virtual_elements(tri6, pairing)
    assert v_ref.elements == v_fast.elements
    assert v_ref.virtual_tri == v_fast.virtual_tri
