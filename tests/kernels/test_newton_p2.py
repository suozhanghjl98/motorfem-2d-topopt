"""P2 求解测试（路线 A：`newton_p2`）。

来源 A 解析性质：

- `field_geometry_p2` 构造正确（transforms/k0/B 系数、逐高斯点系数）；
- P2 残差（真空）与手动 `(1/μ₀)K₀·A − R` 一致；
- **积分点材料**（2026-10-02）：非线性铁单元上 Jacobian 对称、单铁单元
  `newton_solve_p2` 收敛到解析可查的场、逐高斯点 B 与中心 B 的关系；
- P1 的 `field_system` 饱和项依赖 P1 恒等式 `∂B/∂A = K0A/(BΔ)`，对 P2 不成立
  （真实网格实测发散）——本文件用真空单元钉住两层的真空一致性，非线性走
  `residual_p2`/`newton_solve_p2`。

运行::

    python -m pytest tests/kernels/test_newton_p2.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scipy.sparse as sp
from motorfem.kernels.reference.newton import (
    FieldMaterial,
    MaterialMasks,
)
from motorfem.kernels.reference.boundaries import DrivingRows

from motorfem.kernels.reference.bh_curve import BHCurve
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection
from motorfem.kernels.reference.element_matrices_p2 import electromagnetic_elements_p2
from motorfem.kernels.reference.newton_p2 import (
    field_geometry_p2,
    flux_density_p2,
    residual_p2,
    jacobian_p2,
    newton_solve_p2,
)

_CSV = str(Path(__file__).resolve().parents[2] / "legacy" / "pytorch" / "50CS470.csv")


def _empty_driving(n):
    return DrivingRows(
        is_driven=np.zeros(n),
        mat_driven=sp.csr_matrix((n, n)),
        j_driven=sp.eye(n).tocsr(),
        j_mat_driven=sp.csr_matrix((n, n)),
        col_mask=sp.eye(n).tocsr(),
    )


def _ctx(tri, element_rotor_gap=frozenset(), element_slave=frozenset(),
         master_nodes=frozenset(), virtual_tri=None):
    from motorfem.kernels.reference.assembly import TopologyContext
    return TopologyContext(
        tri=np.asarray(tri, dtype=int),
        virtual_tri=virtual_tri or {},
        element_rotor_gap=frozenset(element_rotor_gap),
        element_slave=frozenset(element_slave),
        master_nodes=frozenset(master_nodes),
        a_eq=False, n_node=20,
    )


class _FakeRotating:
    def __init__(self):
        self.brackets = {}
        self.matrix = np.zeros((20, 20))


def _single_unit():
    node = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
    ])
    tri6 = np.array([[0, 1, 2, 3, 4, 5]])
    return node, tri6


def _build_field(em, masks, tri6, n_node=6):
    return field_geometry_p2(
        _ctx(tri6), _FakeRotating(), n_node=n_node, k0elem=em.K0,
        bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        bxcoef_g=em.Bxcoef_g, bycoef_g=em.Bycoef_g, gauss_w=em.gauss_w,
        aelem=em.area, masks=masks)


def _material():
    return FieldMaterial(
        curve=BHCurve.from_csv(_CSV), simp=SIMPProjection(SIMPParams()))


def test_field_geometry_p2_construction():
    node, tri6 = _single_unit()
    em = electromagnetic_elements_p2(node, tri6)
    masks = MaterialMasks(is_vacuum=np.array([1.0]), is_simp=np.array([0.0]),
                          is_non_simp=np.array([0.0]), is_linear_iron=np.array([0.0]))
    field = _build_field(em, masks, tri6)
    assert field.n_elem == 1
    assert np.allclose(field.transforms[0].matrix, np.eye(6))     # 普通单元
    assert field.k0.shape == (1, 6, 6)
    K = np.zeros((6, 6)); idx = 0
    for a in range(6):
        for b in range(a, 6):
            K[a, b] = K[b, a] = em.K0[0, idx]; idx += 1
    assert np.allclose(field.k0[0], K)
    assert np.allclose(field.bxcoef[0], em.Bxcoef[0])
    # 逐高斯点系数：形状 + 加权和 = 刚度（K0 = Σ_g c_g K0_g）
    assert field.bxcoef_g.shape == (1, 3, 6)
    assert np.allclose(field.gauss_w, em.gauss_w)
    K_from_g = sum(
        field.gauss_w[gg] * em.area[0]
        * (np.outer(field.bxcoef_g[0, gg], field.bxcoef_g[0, gg])
           + np.outer(field.bycoef_g[0, gg], field.bycoef_g[0, gg]))
        for gg in range(3)
    )
    assert np.allclose(K_from_g, K, rtol=1e-12)


def test_p2_residual_vacuum_matches_manual():
    """P2 残差（真空，积分点口径）：F = (1/μ₀)K₀·A − R（与单点口径同——真空 γ 常数）。"""
    node, tri6 = _single_unit()
    em = electromagnetic_elements_p2(node, tri6)
    masks = MaterialMasks(is_vacuum=np.array([1.0]), is_simp=np.array([0.0]),
                          is_non_simp=np.array([0.0]), is_linear_iron=np.array([0.0]))
    field = _build_field(em, masks, tri6)
    rho = np.zeros(1)
    A = np.array([0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    R = np.array([1.0, 2.0, -1.0, 0.5, 0.0, -0.5])
    K = np.zeros((6, 6)); idx = 0
    for a in range(6):
        for b in range(a, 6):
            K[a, b] = K[b, a] = em.K0[0, idx]; idx += 1
    mu0 = 4e-7 * np.pi
    F, aux = residual_p2(A, rho, field, _material(), R, driving=_empty_driving(6))
    assert np.allclose(F, (1.0 / mu0) * (K @ A) - R, rtol=1e-10)
    # aux.B 逐高斯点（3 个值）
    assert aux.B.shape == (1, 3)
    assert np.all(aux.B[0] > 0)
    # 高斯点的 B 与中心 B 一致（线性场 A ⇒ B 常数）—— 用线性 A 检查
    Alin = np.array([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
    B_g = flux_density_p2(Alin, field)[0]
    Bx = float(em.Bxcoef[0] @ Alin); By = float(em.Bycoef[0] @ Alin)
    assert np.allclose(B_g, np.hypot(Bx, By), rtol=1e-9)


def test_p2_field_reuses_p1_residual_vacuum():
    """P2 field 喂给 P1 residual（真空）与手动 `(1/μ₀)TᵀK₀T·A − R` 一致。

    ⚠ 仅限真空：P1 `field_system` 的饱和项依赖 P1 恒等式，非线性铁单元必须走
    `residual_p2`（模块 docstring 已记录该限制）。
    """
    from motorfem.kernels.reference.newton import residual as residual_p1
    node, tri6 = _single_unit()
    em = electromagnetic_elements_p2(node, tri6)
    masks = MaterialMasks(is_vacuum=np.array([1.0]), is_simp=np.array([0.0]),
                          is_non_simp=np.array([0.0]), is_linear_iron=np.array([0.0]))
    field = _build_field(em, masks, tri6)
    rho = np.zeros(1)
    A = np.array([0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    R = np.array([1.0, 2.0, -1.0, 0.5, 0.0, -0.5])
    K = np.zeros((6, 6)); idx = 0
    for a in range(6):
        for b in range(a, 6):
            K[a, b] = K[b, a] = em.K0[0, idx]; idx += 1
    mu0 = 4e-7 * np.pi
    F_p1, aux_p1 = residual_p1(A, rho, field, _material(), R, driving=_empty_driving(6))
    F_p2, aux_p2 = residual_p2(A, rho, field, _material(), R, driving=_empty_driving(6))
    assert np.allclose(F_p1, (1.0 / mu0) * (K @ A) - R, rtol=1e-10)
    assert np.allclose(F_p2, F_p1, rtol=1e-10)     # 真空：单点与积分点同
    # 二次场 A：高斯点 B 在单元内变化（与中心 B 不同）—— 这正是积分点求值的意义
    assert aux_p2.B.shape == (1, 3)
    assert abs(float(aux_p2.B[0].max()) - float(aux_p1.B[0])) > 1e-3


def _iron_field(use_linear_iron=False):
    """单个非线性铁单元（非 SIMP、非真空）的 field + masks。"""
    node, tri6 = _single_unit()
    em = electromagnetic_elements_p2(node, tri6)
    masks = MaterialMasks(is_vacuum=np.array([0.0]), is_simp=np.array([0.0]),
                          is_non_simp=np.array([1.0]),
                          is_linear_iron=np.array([1.0 if use_linear_iron else 0.0]))
    return _build_field(em, masks, tri6)


def test_p2_jacobian_is_symmetric_with_iron():
    """非线性铁单元上 P2 Jacobian 对称（饱和项 `outer(g0,g0)` 对称 —— P1 恒等式
    失效的证明：P1 式饱和项给非对称、这里给对称）。"""
    field = _iron_field()
    rho = np.ones(1)
    A = np.array([0.02, 0.03, -0.01, 0.01, 0.02, -0.005])
    F, aux = residual_p2(A, rho, field, _material(), np.zeros(6), _empty_driving(6))
    J = jacobian_p2(A, field, aux, _empty_driving(6)).toarray()
    assert np.allclose(J, J.T, atol=1e-12)
    # 防退化：确实有非线性（μ ≠ μ₀）
    assert not np.allclose(aux.mu, 4e-7 * np.pi)


def test_p2_iron_element_newton_converges():
    """单非线性铁单元 + 电流源：`newton_solve_p2` 收敛，|B| 物理量级。"""
    field = _iron_field()
    rho = np.ones(1)
    R = np.zeros(6)
    R[3:] = 1e-5        # 边中点一致载荷（Δ/3 = 1/6 ⇒ 总电流 ≈ 6e-5/Δ）
    A, aux, J, hist = newton_solve_p2(
        None, rho, field, _material(), R, _empty_driving(6),
        solve_abs_err=1e-8, max_iter=50, iter_relaxation=20,
        relaxation_ratio=0.5, output=False)
    assert np.isfinite(A).all() and hist[-1] < 1e-8
    assert np.abs(aux.B).max() < 5.0, "B 超物理范围"
    # J 对称（收敛那一轮的雅可比）
    Jd = J.toarray()
    assert np.allclose(Jd, Jd.T, atol=1e-10)


def test_p2_linear_iron_linear_problem():
    """线性铁（use_linear_iron）：γ = 1/(μ0·μr) 常数 ⇒ 一步线性解，残差精确为零。"""
    field = _iron_field(use_linear_iron=True)
    rho = np.ones(1)
    R = np.zeros(6)
    R[3:] = 1e-5
    A, aux, J, hist = newton_solve_p2(
        None, rho, field, _material(), R, _empty_driving(6),
        solve_abs_err=1e-10, max_iter=10, iter_relaxation=20,
        relaxation_ratio=0.5, output=False)
    assert hist[-1] < 1e-10
    assert np.abs(aux.B).max() < 5.0
