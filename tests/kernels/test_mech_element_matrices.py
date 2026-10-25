"""内核 5 测试：力学单元矩阵（CST，平面应力）。

测试三级来源：
  来源 A 解析性质      —— **分片试验**（线性位移场精确复现）、刚体模态、
                          平面应力**而非平面应变**的判定、K 对称半正定、
                          R0 倍率固化
  来源 B 与 legacy 一致 —— 复刻 MMA10:557-603 逐位比对
  来源 C ref vs fast   —— 批量实现与循环实现**逐位**相同

运行::

    python -m pytest tests/kernels/test_mech_element_matrices.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from motorfem.kernels.fast import mech_element_matrices as fast_mech
from motorfem.kernels.reference import mech_element_matrices as ref_mech
from motorfem.kernels.reference.mech_element_matrices import MechMaterial


def make_rectangle_mesh(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
    xs, ys = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    node = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    return node, np.array(tris, dtype=int)


MAT = MechMaterial(E=200e9, nu=0.30, rho=7850.0)


@pytest.fixture(scope="module")
def mesh():
    return make_rectangle_mesh(4, 3)


@pytest.fixture(scope="module")
def mech(mesh):
    node, tri = mesh
    return ref_mech.mech_element_matrices(node, tri, MAT)


# =============================================================================
# 来源 A：解析性质
# =============================================================================

def test_patch_test_linear_displacement_field(mesh, mech):
    """**分片试验**：常应变三角形对线性位移场必须**精确**复现应变。

    取 ``u = a0 + a1·x + a2·y``、``v = b0 + b1·x + b2·y``，则
    ``εx = a1``、``εy = b2``、``γxy = a2 + b1`` 都是常数。

    这是有限元单元正确性的**标准判据**（patch test）：CST 的形函数能张成
    线性空间，所以应变恢复必须是精确的。任何一处 ``b``/``c`` 混淆、
    ``1/(2Δ)`` 写错、或 L 的行列装配错位，都会让本条失败。

    用机器精度级容差（相对 ``max|应变|``）。
    """
    node, tri = mesh
    rng = np.random.default_rng(11)
    a0, a1, a2, b0, b1, b2 = rng.normal(scale=1e-3, size=6)

    u = a0 + a1 * node[:, 0] + a2 * node[:, 1]
    v = b0 + b1 * node[:, 0] + b2 * node[:, 1]

    # 组装单元自由度 (u_i,v_i,u_j,v_j,u_m,v_m)
    dof = np.empty((tri.shape[0], 6))
    dof[:, 0::2] = u[tri]
    dof[:, 1::2] = v[tri]

    strain = np.einsum("eij,ej->ei", mech.L, dof)     # (N, 3)
    expect = np.array([a1, b2, a2 + b1])
    scale = np.abs(expect).max()
    assert np.abs(strain - expect).max() < 1e-12 * scale, (
        f"分片试验失败：应变与解析值不符\n实得 {strain[:3]}\n期望 {expect}"
    )


def test_rigid_body_modes_give_zero_strain(mesh, mech):
    """刚体平移不产生应变，且单元刚度乘刚体模态为零。

    二维刚体模态有 3 个：x 平移、y 平移、绕原点转动。
    零特征值个数必须**恰好**是 3。
    """
    node, tri = mesh
    dof = np.empty((tri.shape[0], 6))

    # 两个平移模态
    for comp in (0, 1):
        dof[:] = 0.0
        dof[:, comp::2] = 1.0
        strain = np.einsum("eij,ej->ei", mech.L, dof)
        assert np.abs(strain).max() < 1e-14, f"平移模态产生了应变 {strain}"

    # 三个零特征值（2 平移 + 1 转动）
    for e in range(min(4, mech.n_elem)):
        w = np.linalg.eigvalsh(mech.K[e])
        n_zero = int(np.sum(np.abs(w) < 1e-10 * np.abs(w).max()))
        assert n_zero == 3, (
            f"单元 {e} 的零特征值个数应为 3（二维刚体模态），实得 {n_zero}"
        )


def test_K_is_symmetric_positive_semidefinite(mech):
    """单元刚度对称且半正定（刚体模态使其非正定）。"""
    K = mech.K
    assert np.allclose(K, np.transpose(K, (0, 2, 1)), rtol=0, atol=0)
    for e in range(min(8, mech.n_elem)):
        w = np.linalg.eigvalsh(K[e])
        assert w.min() > -1e-10 * np.abs(w).max(), f"单元 {e} 出现负特征值"


def test_D_is_plane_stress_not_plane_strain(mech):
    """弹性矩阵必须是**平面应力**形式，不是平面应变。

    判据：单向应力下的应变响应。对 ``σ = (σ0, 0, 0)``，平面应力给出::

        ε = (σ0/E, −ν·σ0/E, 0)

    而平面应变会给出 ``ε_y = 0``（被约束）与不同的 ``ε_x``。
    这条能明确区分两种形式——只看 D 的对角元数值不容易分辨。
    """
    E, nu = MAT.E, MAT.nu
    sig0 = 1e6
    eps = np.linalg.solve(mech.D, np.array([sig0, 0.0, 0.0]))

    assert eps[0] == pytest.approx(sig0 / E, rel=1e-12), "εx 不等于 σ/E"
    assert eps[1] == pytest.approx(-nu * sig0 / E, rel=1e-12), "εy 不等于 −νσ/E"
    assert eps[2] == pytest.approx(0.0, abs=1e-20), "γxy 应为 0"

    # 平面应变下 εy 会被约束为 0，所以上面第二条已经排除了平面应变
    assert abs(eps[1]) > 0, "若 εy == 0 则说明用的是平面应变形式"


def test_D_explicit_formula(mech):
    """D 的每个元素与平面应力公式逐位相符。"""
    E, nu = MAT.E, MAT.nu
    expect = np.array([
        [E / (1 - nu ** 2), E * nu / (1 - nu ** 2), 0.0],
        [E * nu / (1 - nu ** 2), E / (1 - nu ** 2), 0.0],
        [0.0, 0.0, E / 2 / (1 + nu)],
    ])
    assert np.array_equal(mech.D, expect)


def test_R0_factor_is_documented(mech):
    """把 legacy 的 ``Rmech10elem = −Δ/2`` 倍率**固化**下来。

    这不是在承认它正确——而是把现状钉住。三角形一致体积力载荷应为每节点
    ``f·Δ/3``（总和 ``f·Δ``），legacy 给的是每节点 ``f·(−Δ/2)``
    （总和 ``−1.5·f·Δ``）。符号可能被残差约定吸收，但 **1.5 倍幅值不能**。

    若将来核对后决定修正，本条会失败并提醒同步更新 ``MECH_LOAD_NOTE``
    与相关标定（应力约束的增广拉格朗日参数）。
    """
    assert np.array_equal(mech.R0, -mech.area / 2), "R0 倍率已不再是 −Δ/2"

    # 显式记录「实际总和 / 一致载荷总和」的比值
    total_actual = 3 * mech.R0                 # 每单元三个节点的总和
    total_consistent = mech.area               # 一致载荷总和 = f·Δ
    ratio = total_actual / total_consistent
    assert np.allclose(ratio, -1.5, rtol=1e-14), (
        f"实际/一致 载荷比应为 −1.5，实得 {ratio[0]:.6f}"
    )


def test_matches_legacy_implementation(mesh):
    """复刻 legacy ``MMA10:557-603`` 逐位比对。"""
    node, tri = mesh
    E, nu = MAT.E, MAT.nu
    N = tri.shape[0]

    Kmech0elem = np.zeros((N, 6, 6))
    Rmech10elem = np.zeros(N)
    Lmechcoef = np.zeros((N, 3, 6))
    E0mechcoef = np.zeros((N, 3, 3))
    Amechelem = np.zeros(N)

    for i_elem in range(N):
        node_i, node_j, node_m = tri[i_elem]
        xi, yi = node[node_i]
        xj, yj = node[node_j]
        xm, ym = node[node_m]
        bi = yj - ym
        bj = ym - yi
        bm = yi - yj
        ci = -xj + xm
        cj = -xm + xi
        cm = -xi + xj
        Delta = 1 / 2 * (bi * cj - bj * ci)
        assert Delta > 0

        Lmechcoef[i_elem, 0, 0] = 1 / (2 * Delta) * bi
        Lmechcoef[i_elem, 0, 2] = 1 / (2 * Delta) * bj
        Lmechcoef[i_elem, 0, 4] = 1 / (2 * Delta) * bm
        Lmechcoef[i_elem, 1, 1] = 1 / (2 * Delta) * ci
        Lmechcoef[i_elem, 1, 3] = 1 / (2 * Delta) * cj
        Lmechcoef[i_elem, 1, 5] = 1 / (2 * Delta) * cm
        Lmechcoef[i_elem, 2, 0] = 1 / (2 * Delta) * ci
        Lmechcoef[i_elem, 2, 1] = 1 / (2 * Delta) * bi
        Lmechcoef[i_elem, 2, 2] = 1 / (2 * Delta) * cj
        Lmechcoef[i_elem, 2, 3] = 1 / (2 * Delta) * bj
        Lmechcoef[i_elem, 2, 4] = 1 / (2 * Delta) * cm
        Lmechcoef[i_elem, 2, 5] = 1 / (2 * Delta) * bm

        E0mechcoef[i_elem, 0, 0] = E / (1 - nu ** 2)
        E0mechcoef[i_elem, 0, 1] = E * nu / (1 - nu ** 2)
        E0mechcoef[i_elem, 1, 0] = E * nu / (1 - nu ** 2)
        E0mechcoef[i_elem, 1, 1] = E / (1 - nu ** 2)
        E0mechcoef[i_elem, 2, 2] = E / 2 / (1 + nu)

        Kmech0elem[i_elem] = (Delta * Lmechcoef[i_elem].T
                              @ E0mechcoef[i_elem] @ Lmechcoef[i_elem])
        Rmech10elem[i_elem] = -Delta / 2
        Amechelem[i_elem] = Delta

    got = ref_mech.mech_element_matrices(node, tri, MAT)
    assert np.array_equal(got.L, Lmechcoef), "L 与 legacy 不逐位相同"
    assert np.array_equal(got.K, Kmech0elem), "K 与 legacy 不逐位相同"
    assert np.array_equal(got.R0, Rmech10elem), "R0 与 legacy 不逐位相同"
    assert np.array_equal(got.area, Amechelem), "area 与 legacy 不逐位相同"

    # legacy 把 D 逐单元复制；本模块只存一份。数值必须完全一致。
    assert np.all(np.array_equal(E0mechcoef[e], got.D) for e in range(N)), (
        "单份 D 与 legacy 逐单元复制的 D 不一致"
    )


def test_left_associativity_discipline(mesh):
    """锁定 ``Delta * L.T @ D @ L`` 的**左结合**语义，并量化其实际影响。

    Python 里 ``*`` 与 ``@`` 同优先级、左结合，故 legacy 实际算的是
    ``((Δ·Lᵀ) @ D) @ L``，而不是 ``Δ·(Lᵀ @ D @ L)``。本模块采用前者——
    这是与 legacy 位级一致的**必要条件**，故严格断言。

    ⚠ 但**不能**断言两种形式必然给出不同结果：在规则网格上，Δ 与 Lᵀ 的
    因子可以精确交换，两者会逐位相同（本文件的矩形网格就是这种情况）。
    差异是否出现取决于坐标的具体浮点值。因此这里只在**随机坐标网格**上
    量化差异，而不断言其非零。

    这条纪律的价值不是"总能看出差别"，而是"在与 legacy 对比时不会因为
    一次看似无害的整理而引入末位差异"。
    """
    node, tri = mesh
    got = ref_mech.mech_element_matrices(node, tri, MAT)
    e = 0
    Dlt = got.area[e]
    left_assoc = (Dlt * got.L[e].T) @ got.D @ got.L[e]
    right_assoc = Dlt * (got.L[e].T @ got.D @ got.L[e])

    assert np.array_equal(got.K[e], left_assoc), "实现未采用 legacy 的左结合形式"

    # 在随机坐标网格上量化差异（不要求非零）
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(17)
    rnode = rng.uniform(0, 1, size=(200, 2))
    rtri = Delaunay(rnode).simplices
    x, y = rnode[rtri, 0], rnode[rtri, 1]
    d = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    rtri = np.where(d[:, None] < 0, rtri[:, ::-1], rtri)
    rg = ref_mech.mech_element_matrices(rnode, rtri, MAT)

    n_diff = 0
    max_rel = 0.0
    for k in range(rg.n_elem):
        a = (rg.area[k] * rg.L[k].T) @ rg.D @ rg.L[k]
        b = rg.area[k] * (rg.L[k].T @ rg.D @ rg.L[k])
        if not np.array_equal(a, b):
            n_diff += 1
            max_rel = max(max_rel, np.abs(a - b).max() / np.abs(a).max())

    # 记录事实，不做断言——差异的有无依网格而定
    print(f"\n[left-assoc] 随机网格 {rg.n_elem} 个单元中 {n_diff} 个出现差异，"
          f"最大相对差 {max_rel:.3e}")


# =============================================================================
# 来源 C：ref vs fast 逐位一致
# =============================================================================

def test_batched_matches_loop_bitexact(mesh):
    """批量实现必须与循环实现**逐位**相同。"""
    node, tri = mesh
    r = ref_mech.mech_element_matrices(node, tri, MAT)
    f = fast_mech.mech_element_matrices(node, tri, MAT)
    for name in ("L", "D", "K", "R0", "area"):
        assert np.array_equal(getattr(r, name), getattr(f, name)), (
            f"{name} 批量版与循环版不逐位相同"
        )


def test_batched_matches_loop_on_random_mesh():
    """不规则网格上也逐位一致——结构化网格的规律性可能掩盖索引错误。"""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(5)
    node = rng.uniform(0, 1, size=(300, 2))
    tri = Delaunay(node).simplices
    x, y = node[tri, 0], node[tri, 1]
    d = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    tri = np.where(d[:, None] < 0, tri[:, ::-1], tri)

    r = ref_mech.mech_element_matrices(node, tri, MAT)
    f = fast_mech.mech_element_matrices(node, tri, MAT)
    for name in ("L", "D", "K", "R0", "area"):
        assert np.array_equal(getattr(r, name), getattr(f, name)), (
            f"{name} 在不规则网格上不逐位相同"
        )


def test_patch_test_on_random_mesh():
    """在随机 Delaunay 网格上重跑分片试验——排除结构化网格的特殊性。"""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(9)
    node = rng.uniform(0, 1, size=(250, 2))
    tri = Delaunay(node).simplices
    x, y = node[tri, 0], node[tri, 1]
    d = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    tri = np.where(d[:, None] < 0, tri[:, ::-1], tri)

    mech = ref_mech.mech_element_matrices(node, tri, MAT)
    a1, a2, b1, b2 = rng.normal(scale=1e-3, size=4)
    u = a1 * node[:, 0] + a2 * node[:, 1]
    v = b1 * node[:, 0] + b2 * node[:, 1]
    dof = np.empty((tri.shape[0], 6))
    dof[:, 0::2] = u[tri]
    dof[:, 1::2] = v[tri]

    strain = np.einsum("eij,ej->ei", mech.L, dof)
    expect = np.array([a1, b2, a2 + b1])
    assert np.abs(strain - expect).max() < 1e-11 * np.abs(expect).max()


# =============================================================================
# 载荷：离心体积力与边界面力（内核 5 的后半，批 5 补齐）
# =============================================================================
#
# 来历：内核 5 的模块 docstring 第 4~6 条**一直**写着这三个载荷的公式，但第一版只交付
# 了单元矩阵 —— 而内核 16 的装配族（`get_Kmech_vectorized`）要把它们当输入读。
# 批 5 开工时发现这个缺口，在此补齐。

def make_quad_offset(lx=1.0, ly=1.0, offset=(1.0, 1.0)):
    """两个三角形的单位方块（平移到 ``offset``）外加四条边界线元。

    ⚠ 平移到第一象限是**有意的**：``mech_line_loads`` 里要除以
    ``r̄ = |(x_i+x_j, y_i+y_j)|/2``，位于原点的线元会让它为零。
    """
    ox, oy = offset
    node = np.array([[ox, oy], [ox + lx, oy], [ox + lx, oy + ly], [ox, oy + ly]])
    tri = np.array([[0, 1, 2], [0, 2, 3]])          # 逆时针
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    return node, tri, lines


def test_body_force_matches_the_centrifugal_integral():
    """来源 A：**合力 = ρω²·(体积 × 形心偏移)** —— 体积力的解析积分恒等式。

        Σ_e Δ_e·f_x,e == ρω²·(Σ_e Δ_e·x̄_e − x_c·Σ_e Δ_e)

    左边逐单元算（含 legacy 的 ``xi/3+xj/3+xm/3`` 写法），右边是面积加权和。
    两边只有在形心真的取在形心、且 ``ρω²`` 因子没写错时才相等。
    """
    node, tri, _ = make_quad_offset()
    mech = ref_mech.mech_element_matrices(node, tri, MAT)
    w = 300 * np.pi / 2
    center = (0.25, -0.5)

    fx, fy = ref_mech.mech_body_force(
        node, tri, MAT, angular_speed=w, rot_center=center)

    rho = MAT.rho
    centroid = node[tri].sum(axis=1) / 3
    area_total = mech.area.sum()

    lhs_x = float(np.sum(mech.area * fx))
    rhs_x = rho * w ** 2 * (float(np.sum(mech.area * centroid[:, 0]))
                            - center[0] * area_total)
    assert lhs_x == pytest.approx(rhs_x, rel=1e-13)

    lhs_y = float(np.sum(mech.area * fy))
    rhs_y = rho * w ** 2 * (float(np.sum(mech.area * centroid[:, 1]))
                            - center[1] * area_total)
    assert lhs_y == pytest.approx(rhs_y, rel=1e-13)


def test_body_force_is_zero_at_the_rotation_center():
    """旋转中心取在形心上时体积力恒为零（解析性质，挡住「ρ/ω 量纲写错」）。

    ⚠ 残余**不是精确的零**，因为 legacy 的形心写法 ``xi/3 + xj/3 + xm/3``
    （三项各自除 3 再相加）与 ``(xi+xj+xm)/3`` 在浮点上差末位 —— 内核 5 的
    ``test_body_force_matches_the_legacy_fragment_bitwise`` 正是把这一点钉住。
    所以这里按**相对另一单元的力**判定量级（残余应在坐标的舍入量级 1e-16 附近），
    而不是拿一个绝对阈值去卡。
    """
    node, tri, _ = make_quad_offset()
    centroid = node[tri].sum(axis=1) / 3
    fx, fy = ref_mech.mech_body_force(
        node, tri, MAT, angular_speed=123.0, rot_center=tuple(centroid[0]))

    scale = max(abs(fx[1]), abs(fy[1]))
    assert scale > 0.0, "第二个单元的力为零 ⇒ 测试无意义"
    assert abs(fx[0]) <= 1e-12 * scale
    assert abs(fy[0]) <= 1e-12 * scale


def test_line_load_coefficient_is_minus_half_the_length():
    """``R0 = −L/2``，且**只在加载线元上非零**（legacy 先置零、再逐个赋值）。"""
    node, _, lines = make_quad_offset()
    loads = ref_mech.mech_line_loads(node, lines, [0, 2], Fload=1.0)
    assert loads.R0.shape == (lines.shape[0],)

    for i in (0, 2):
        ni, nj = lines[i]
        ell = np.sqrt((node[ni, 0] - node[nj, 0]) ** 2
                      + (node[ni, 1] - node[nj, 1]) ** 2)
        assert loads.R0[i] == pytest.approx(-ell / 2, rel=1e-15)
    for i in (1, 3):
        assert loads.R0[i] == 0.0


def test_line_load_is_a_scaled_unit_direction_rotated_45_degrees():
    """来源 A：``(x_i+x_j)/2·Fload/r̄·L`` 里两个 ``|s|`` 因子**恰好约掉**。

    令 ``s = (x_i+x_j, y_i+y_j)``、``r̄ = |s|/2``::

        p = (s/2)·Fload/r̄·L = Fload·L·(s/|s|) = Fload·L·û

    即 ``p`` 是**单位方向**乘 ``Fload·L`` —— 这是两次约分的结果。任一处写错
    （例如把 ``r̄`` 换成「形心到原点的距离」）都会破坏它。随后整体绕原点转 45°、
    再除以**全部**加载线元的总长。
    """
    node, _, lines = make_quad_offset()
    Fload, deg = 3.5e6, 45.0
    loaded = [0, 2]
    loads = ref_mech.mech_line_loads(node, lines, loaded, Fload=Fload)

    rot = deg * np.pi / 180
    ells = []
    for i in loaded:
        ni, nj = lines[i]
        ells.append(np.sqrt((node[ni, 0] - node[nj, 0]) ** 2
                            + (node[ni, 1] - node[nj, 1]) ** 2))
    total_len = float(np.sum(ells))

    for k, i in enumerate(loaded):
        ni, nj = lines[i]
        s = np.array([node[ni, 0] + node[nj, 0], node[ni, 1] + node[nj, 1]])
        p = Fload * ells[k] * (s / np.linalg.norm(s))
        expect = np.array([p[0] * np.cos(rot) - p[1] * np.sin(rot),
                           p[0] * np.sin(rot) + p[1] * np.cos(rot)]) / total_len
        assert loads.fx[k] == pytest.approx(expect[0], rel=1e-13)
        assert loads.fy[k] == pytest.approx(expect[1], rel=1e-13)
        # ★ 解析性质：力与 ``s`` 方向的夹角**恰好是 45°**（防退化 —— 挡住「忘了旋转」
        # 这类错误：那时夹角为 0）
        cos_ang = float(expect @ s) / (np.linalg.norm(expect) * np.linalg.norm(s))
        assert cos_ang == pytest.approx(np.cos(rot), rel=1e-12)

    assert loads.boundary_length == pytest.approx(total_len, rel=1e-15)


def test_line_loads_with_no_loaded_lines():
    """没有加载线元时：``R0`` 全零、``fx/fy`` 为空、总长为 0（不做 ``/0``）。"""
    node, _, lines = make_quad_offset()
    loads = ref_mech.mech_line_loads(node, lines, [], Fload=1.0)
    assert np.all(loads.R0 == 0.0)
    assert loads.fx.shape == (0,) and loads.fy.shape == (0,)
    assert loads.boundary_length == 0.0


# ---- 来源 B：与 legacy 片段逐位 ----

def legacy_mech_line_loads(node, line_tri, loaded_lines, Fload, rot_angle_deg=45.0):
    """照抄 ``MMA10:605-613``（``Rmech20elem``）与 ``:639-661``（``Fmechload``）。

    ⚠ 形心/端点和都按 legacy 的 ``node_mech[i, 1:3]`` 取（本测试里 ``node`` 只有
    两列，所以直接索引 0:2）。
    """

    def _rot_x(x, y, angle):
        return x * np.cos(angle) - y * np.sin(angle)

    def _rot_y(x, y, angle):
        return x * np.sin(angle) + y * np.cos(angle)

    Rmech20elem = np.zeros(len(line_tri))
    for i_elem_L in loaded_lines:
        node_i, node_j = line_tri[i_elem_L]
        xi, yi = node[node_i]
        xj, yj = node[node_j]
        _L = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        Rmech20elem[i_elem_L] = -_L / 2

    Fmechloadx = np.zeros(len(loaded_lines))
    Fmechloady = np.zeros(len(loaded_lines))
    _len_boundary = 0
    for i_load, i_elem_L in enumerate(loaded_lines):
        node_i, node_j = line_tri[i_elem_L]
        x_i, y_i = node[node_i]
        x_j, y_j = node[node_j]
        _l = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
        _r = np.sqrt((x_i + x_j) ** 2 + (y_i + y_j) ** 2) / 2
        _px = (x_i + x_j) / 2 * Fload / _r * _l
        _py = (y_i + y_j) / 2 * Fload / _r * _l
        _a = rot_angle_deg * np.pi / 180
        Fmechloadx[i_load] = _rot_x(_px, _py, _a)
        Fmechloady[i_load] = _rot_y(_px, _py, _a)
        _len_boundary += _l
    Fmechloadx /= _len_boundary
    Fmechloady /= _len_boundary

    return Rmech20elem, Fmechloadx, Fmechloady


def test_line_loads_match_the_legacy_fragment_bitwise():
    """来源 B：``Rmech20elem`` / ``Fmechload`` 与照抄片段**逐位**一致。"""
    node, _, lines = make_quad_offset()
    loaded = [0, 2]
    Fload = 7.5e6

    R0_l, fx_l, fy_l = legacy_mech_line_loads(node, lines, loaded, Fload)
    got = ref_mech.mech_line_loads(node, lines, loaded, Fload=Fload)

    assert np.array_equal(got.R0, R0_l)
    assert np.array_equal(got.fx, fx_l)
    assert np.array_equal(got.fy, fy_l)


def test_body_force_matches_the_legacy_fragment_bitwise():
    """来源 B：``fmechload`` 与照抄片段**逐位**一致。

    ⚠ 片段里形心写成 ``xi/3 + xj/3 + xm/3``（三项各自除 3），**不是**
    ``(xi+xj+xm)/3`` —— 两种写法浮点不同，这条用 ``array_equal`` 把它钉住。
    """
    node, tri, _ = make_quad_offset()
    w = 300 * np.pi / 2
    center = (0.25, -0.5)

    n_elem = tri.shape[0]
    fmechloadx = np.zeros(n_elem)
    fmechloady = np.zeros(n_elem)
    for i_elem in range(n_elem):
        node_i, node_j, node_m = tri[i_elem]
        xi, yi = node[node_i]
        xj, yj = node[node_j]
        xm, ym = node[node_m]
        _m = MAT.rho
        fmechloadx[i_elem] = _m * w ** 2 * (xi / 3 + xj / 3 + xm / 3 - center[0])
        fmechloady[i_elem] = _m * w ** 2 * (yi / 3 + yj / 3 + ym / 3 - center[1])

    fx, fy = ref_mech.mech_body_force(
        node, tri, MAT, angular_speed=w, rot_center=center)
    assert np.array_equal(fx, fmechloadx)
    assert np.array_equal(fy, fmechloady)


# ---- 来源 C：ref vs fast ----

def test_ref_fast_parity_on_loads():
    """两层的载荷**逐位**相同（含 ``boundary_length`` 的顺序累加）。"""
    node, tri, lines = make_quad_offset()
    loaded = [0, 2, 3]
    w = 300 * np.pi / 2
    center = (0.25, -0.5)

    fx_r, fy_r = ref_mech.mech_body_force(
        node, tri, MAT, angular_speed=w, rot_center=center)
    fx_f, fy_f = fast_mech.mech_body_force(
        node, tri, MAT, angular_speed=w, rot_center=center)
    assert np.array_equal(fx_r, fx_f)
    assert np.array_equal(fy_r, fy_f)

    l_r = ref_mech.mech_line_loads(node, lines, loaded, Fload=1e7)
    l_f = fast_mech.mech_line_loads(node, lines, loaded, Fload=1e7)
    assert np.array_equal(l_r.R0, l_f.R0)
    assert l_r.boundary_length == l_f.boundary_length
    assert np.array_equal(l_r.fx, l_f.fx)
    assert np.array_equal(l_r.fy, l_f.fy)


def test_fast_loads_keeps_the_sequential_sum_order():
    """★ 总长必须是**顺序累加**：``np.sum`` 的成对求和与 legacy 的 ``+=`` 末位不同。

    「顺手向量化」在这里最容易踩的坑 —— 设计文档 §8 的向量化补齐欠着
    ``get_Kmech_vectorized`` 的债，批 5 一并还，但**还债不许改末位**。

    ⚠ **这条测试要挑数据**：成对求和与顺序求和是否分歧取决于长度序列，实测同一
    构造下 seed ∈ {1, 5} 分歧、{0, 2, 3, 4, 6, 7} 相同（n = 64/128/512 都试过）。
    所以下面的 ``!=`` 断言是**防退化**的：它保证这条测试确实在区分两种求和，
    而不是恰好撞上等值。换种子时若它失败，说明该换一组数据，不是实现错了。
    """
    rng = np.random.default_rng(1)
    n = 512
    steps = 10.0 ** rng.uniform(-9, 2, size=n)
    y = np.concatenate([[0.0], np.cumsum(steps)])
    node = np.column_stack([np.ones_like(y), y])
    lines = np.array([[i, i + 1] for i in range(n)])

    got = fast_mech.mech_line_loads(node, lines, list(range(n)), Fload=1.0)

    lengths = [float(np.sqrt((node[a, 0] - node[b, 0]) ** 2
                             + (node[a, 1] - node[b, 1]) ** 2))
               for a, b in lines]
    seq = 0.0
    for ell in lengths:
        seq += ell

    assert got.boundary_length == seq
    assert got.boundary_length != float(np.sum(lengths)), (
        "这组长度下成对求和与顺序求和竟然相同 ⇒ 这条测试抓不到东西，换个种子/规模"
    )
