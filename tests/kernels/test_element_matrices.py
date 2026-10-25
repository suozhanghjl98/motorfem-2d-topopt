"""内核 4 测试：电磁单元矩阵。

测试三级来源：
  来源 A 解析性质      —— K0 行和恒零（常值零空间）、对称半正定、
                          R0 三节点和等于面积、Bxcoef/Bycoef 行和为零、
                          与 magnetostatic 的 |B| 及 S_e 交叉一致
  来源 B 与 legacy 一致 —— 复刻 MMA10:509-542 逐位比对
  来源 C ref vs fast   —— 批量实现必须与循环实现**逐位**相同

运行::

    python -m pytest tests/kernels/test_element_matrices.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from motorfem.kernels.fast import element_matrices as fast_em
from motorfem.kernels.reference import element_matrices as ref_em
from motorfem.kernels.reference.magnetostatic import (
    element_stiffness,
    flux_density_magnitude,
)


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


@pytest.fixture(scope="module")
def mesh():
    return make_rectangle_mesh(4, 3)


@pytest.fixture(scope="module")
def elems(mesh):
    node, tri = mesh
    return ref_em.electromagnetic_elements(node, tri)


# =============================================================================
# 来源 A：解析性质
# =============================================================================

def test_stiffness_rows_sum_to_zero(elems):
    """单元刚度矩阵的**行和恒为零**。

    解析依据：``S_e = (1/(4Δ))(b bᵀ + c cᵀ)``，而由 b/c 的定义
    ``b_i + b_j + b_m = 0``、``c_i + c_j + c_m = 0``，故::

        S_e · 1 = (1/(4Δ))(b·(bᵀ1) + c·(cᵀ1)) = (1/(4Δ))(b·0 + c·0) = 0

    物理含义：**A = 常数 ⇒ B = 0 ⇒ 无能量**，所以常值向量落在零空间里。
    这是单元矩阵正确性的最强解析判据——任何系数写错（例如某处的
    ``1/4Δ`` 与 ``1/2Δ`` 混淆）都会破坏行和为零。

    容差说明：该性质在**精确算术下恒成立**，但浮点下三个大数相消会留下
    与元素量级成正比的舍入残差（K 元素量级 ~1/(4Δ) ~ 6，故残差 ~1e-16）。
    因此容差按 ``|K|`` 的量级缩放，而不是取绝对值——用绝对小容差会把
    正常的舍入当成错误，用大而固定的容差又会让真正的系数错误溜过去。
    """
    K = ref_em.expand_K0(elems.K0)
    row_sums = K.sum(axis=2)                       # (N, 3)
    tol = 1e-14 * np.abs(K).max()
    assert np.abs(row_sums).max() <= tol, (
        f"单元刚度行和应为 0（容差 {tol:.2e}），实得最大 {np.abs(row_sums).max():.3e}"
    )


def test_stiffness_is_symmetric_positive_semidefinite(elems):
    """S_e 对称且半正定（不是正定——常值向量给零特征值）。"""
    K = ref_em.expand_K0(elems.K0)
    n = K.shape[0]

    assert np.allclose(K, np.transpose(K, (0, 2, 1)), rtol=0, atol=0), "不满足对称"

    for e in range(n):
        w = np.linalg.eigvalsh(K[e])
        assert w.min() > -1e-15 * max(1.0, np.abs(w).max()), (
            f"单元 {e} 出现负特征值 {w.min():.3e}，非半正定"
        )
        # 恰好一个零特征值（常值零空间）
        assert np.sum(np.abs(w) < 1e-12 * np.abs(w).max()) == 1, (
            f"单元 {e} 的零特征值个数应为 1，实得 {np.sum(np.abs(w) < 1e-12*np.abs(w).max())}"
        )


def test_R0_gives_total_unit_source(elems):
    """``R0 = Δ/3``：三角形面积均分到 3 个节点，故三节点之和恰为 Δ。

    物理含义：单元内均匀源 J 的总"载荷"∫J dΩ = J·Δ，一致分配后
    节点载荷之和必须守恒。
    """
    assert np.array_equal(elems.R0, elems.area / 3)
    assert np.allclose(3 * elems.R0, elems.area, rtol=0, atol=0)


def test_flux_coefficients_have_zero_row_sum(elems):
    """``Bxcoef`` / ``Bycoef`` 的行和为零。

    因为 Bxcoef 正比于 b、Bycoef 正比于 c，而 b、c 的分量和都是 0。
    含义：A = 常数时 B = 0，与刚度矩阵的零空间一致。

    容差按系数自身量级缩放（同 ``test_stiffness_rows_sum_to_zero``）。
    """
    tol_x = 1e-14 * np.abs(elems.Bxcoef).max()
    tol_y = 1e-14 * np.abs(elems.Bycoef).max()
    assert np.abs(elems.Bxcoef.sum(axis=1)).max() <= tol_x
    assert np.abs(elems.Bycoef.sum(axis=1)).max() <= tol_y


def test_matches_magnetostatic_module(mesh, elems):
    """与本项目 magnetostatic 内核交叉一致（跨模块判据）。

    两条独立路径应当给出同一个量：

      ``|B|`` : 由 Bxcoef/Bycoef 算出的 ``hypot(Bx,By)``
                应等于 ``magnetostatic.flux_density_magnitude``
                的 ``sqrt((b·A)²+(c·A)²)/(2Δ)``

      ``S_e`` : ``expand_K0(K0)`` 应等于 ``element_stiffness(geom)``

    这条能抓住「两个模块各自自洽、但约定不一致」的情况——单看任一模块
    的测试都发现不了。
    """
    node, tri = mesh
    A = np.random.default_rng(0).normal(scale=1e-2, size=node.shape[0])
    A_e = A[tri]

    Bx = np.einsum("es,es->e", elems.Bxcoef, A_e)
    By = np.einsum("es,es->e", elems.Bycoef, A_e)
    B_from_coef = np.hypot(Bx, By)

    B_ref = flux_density_magnitude(A, elems.geom)
    assert np.allclose(B_from_coef, B_ref, rtol=1e-14, atol=1e-18), (
        f"两模块的 |B| 不一致，最大偏差 {np.abs(B_from_coef - B_ref).max():.3e}"
    )

    S_ref = element_stiffness(elems.geom)
    K = ref_em.expand_K0(elems.K0)
    assert np.allclose(K, S_ref, rtol=1e-14, atol=1e-18), (
        f"两模块的单元刚度不一致，最大偏差 {np.abs(K - S_ref).max():.3e}"
    )


def test_current_area_sums_element_areas(elems):
    """``current_area`` 是给定单元集合的面积之和。"""
    ids = [0, 1, 5, 7]
    expect = float(np.sum(elems.area[ids]))
    assert ref_em.current_area(elems.area, ids) == expect
    assert ref_em.current_area(elems.area, []) == 0.0
    # 全部单元 -> 总面积
    assert np.isclose(
        ref_em.current_area(elems.area, range(elems.n_elem)), elems.area.sum()
    )


def test_orientation_assertion_still_fires(mesh):
    """单元定向错误必须立刻报错（由 element_geometry 保证，本内核继承）。"""
    node, tri = mesh
    with pytest.raises(ValueError, match="逆时针"):
        ref_em.electromagnetic_elements(node, tri[:, ::-1].copy())


# =============================================================================
# 来源 B：与 legacy 逐位一致
# =============================================================================

def test_matches_legacy_implementation(mesh):
    """复刻 legacy ``MMA10:509-542`` 逐位比对。

    把 legacy 的循环体照抄进测试（含其 ``1/4/Delta*`` 的左结合形式），
    确认重构后的实现产出完全相同的浮点值。
    """
    node, tri = mesh
    N_element = tri.shape[0]

    K0elem = np.zeros((N_element, 6))
    R0elem = np.zeros(N_element)
    Bxcoef = np.zeros((N_element, 3))
    Bycoef = np.zeros((N_element, 3))
    Aelem = np.zeros(N_element)

    for i_elem in range(N_element):
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

        K0elem[i_elem, 0] = 1 / 4 / Delta * (bi * bi + ci * ci)
        K0elem[i_elem, 1] = 1 / 4 / Delta * (bi * bj + ci * cj)
        K0elem[i_elem, 2] = 1 / 4 / Delta * (bi * bm + ci * cm)
        K0elem[i_elem, 3] = 1 / 4 / Delta * (bj * bj + cj * cj)
        K0elem[i_elem, 4] = 1 / 4 / Delta * (bj * bm + cj * cm)
        K0elem[i_elem, 5] = 1 / 4 / Delta * (bm * bm + cm * cm)

        R0elem[i_elem] = Delta / 3

        Bxcoef[i_elem, 0] = 1 / 2 / Delta * bi
        Bxcoef[i_elem, 1] = 1 / 2 / Delta * bj
        Bxcoef[i_elem, 2] = 1 / 2 / Delta * bm

        Bycoef[i_elem, 0] = -1 / 2 / Delta * ci
        Bycoef[i_elem, 1] = -1 / 2 / Delta * cj
        Bycoef[i_elem, 2] = -1 / 2 / Delta * cm

        Aelem[i_elem] = Delta

    got = ref_em.electromagnetic_elements(node, tri)
    assert np.array_equal(got.K0, K0elem), "K0 与 legacy 不逐位相同"
    assert np.array_equal(got.R0, R0elem), "R0 与 legacy 不逐位相同"
    assert np.array_equal(got.Bxcoef, Bxcoef), "Bxcoef 与 legacy 不逐位相同"
    assert np.array_equal(got.Bycoef, Bycoef), "Bycoef 与 legacy 不逐位相同"
    assert np.array_equal(got.area, Aelem), "area 与 legacy 不逐位相同"


def test_delta_computation_matches_legacy(mesh):
    """Δ 的计算顺序也要一致：legacy 写 ``1/2*(bi*cj - bj*ci)``。

    ``element_geometry`` 内部写的是 ``(...)/2.0``，与 ``1/2*(...)`` 在
    浮点下是否相同需要实测确认——这条就是为了锁定它。
    """
    node, tri = mesh
    geom = ref_em.element_geometry(node, tri)
    for e in range(tri.shape[0]):
        ni, nj, nm = tri[e]
        xi, yi = node[ni]
        xj, yj = node[nj]
        xm, ym = node[nm]
        bi, bj = yj - ym, ym - yi
        ci, cj = -xj + xm, -xm + xi
        assert geom.delta[e] == 1 / 2 * (bi * cj - bj * ci), (
            f"单元 {e} 的 Δ 计算顺序与 legacy 不一致"
        )


# =============================================================================
# 来源 C：ref vs fast 逐位一致
# =============================================================================

def test_batched_matches_loop_bitexact(mesh):
    """批量实现必须与循环实现**逐位**相同。

    这是本内核唯一的 ref/fast 差异（实现方式），所以判据是 ``array_equal``。
    若有人把 ``1/4/Delta*(...)`` "整理"成 ``(...)/(4*Delta)``，本条会失败。
    """
    node, tri = mesh
    r = ref_em.electromagnetic_elements(node, tri)
    f = fast_em.electromagnetic_elements(node, tri)

    for name in ("K0", "R0", "Bxcoef", "Bycoef", "area"):
        assert np.array_equal(getattr(r, name), getattr(f, name)), (
            f"{name} 批量版与循环版不逐位相同"
        )


def test_batched_matches_loop_on_large_random_mesh():
    """在不规则网格上也逐位一致（结构化矩形网格可能掩盖索引错误）。"""
    rng = np.random.default_rng(3)
    n = 400
    node = rng.uniform(0, 1, size=(n, 2))
    # 用 Delaunay 生成合法三角剖分
    from scipy.spatial import Delaunay
    tri = Delaunay(node).simplices
    # Delaunay 不保证逆时针，按有向面积翻转
    x, y = node[tri, 0], node[tri, 1]
    d = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    tri = np.where(d[:, None] < 0, tri[:, ::-1], tri)

    r = ref_em.electromagnetic_elements(node, tri)
    f = fast_em.electromagnetic_elements(node, tri)
    for name in ("K0", "R0", "Bxcoef", "Bycoef", "area"):
        assert np.array_equal(getattr(r, name), getattr(f, name)), (
            f"{name} 在不规则网格上不逐位相同"
        )
