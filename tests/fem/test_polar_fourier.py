# -*- coding: utf-8 -*-
"""极坐标傅里叶参数化的解析性质与梯度（A 类）测试。

这一层不依赖真实网格/FEM：用解析采样点与假 ``FieldTopology`` 验证

- 基的构造（列数、常值/低阶三角场/多项式的精确重构）；
- θ 方向三角基在均匀采样上的正交性；
- 网格无关性（同一组系数在两组采样点上给出同一个连续场）；
- ``project`` 与有限差分一致（含从属单元导数叠加）；
- ``sigmoid`` 映射的值域与导数。

真实网格上的集成（``FieldSystem(polar_spec=...)``）见
``tests/fem/test_polar_fourier_system.py``。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from motorfem.fem.polar_fourier import (
    PolarFourierParam, PolarFourierSpec, basis_matrix, radial_basis,
)

R_RANGE = (0.015, 0.0445)
TH_RANGE = (0.0, np.pi / 4.0)


def _grid(nr: int = 6, nt: int = 12):
    """规则采样点（r, θ）—— 覆盖整个窗口。"""
    r = np.linspace(R_RANGE[0], R_RANGE[1], nr)
    th = np.linspace(TH_RANGE[0], TH_RANGE[1], nt)
    R, T = np.meshgrid(r, th, indexing="ij")
    return R.ravel(), T.ravel()


def _param(spec=None, nr=6, nt=12):
    r, th = _grid(nr, nt)
    p = PolarFourierParam(r, th, spec or PolarFourierSpec(K=4, N_r=4),
                          R_RANGE, TH_RANGE)
    n = p.B.shape[0]
    # 无从属单元的假拓扑（梯度测试只需要 project 的从属叠加为空）
    p.attach_topology(SimpleNamespace(
        element_topo_opti_m=set(range(n)), element_topo_opti_s={}, n_element=n))
    return p


# ---------------------------------------------------------------- 基
def test_basis_shape_and_column_order():
    spec = PolarFourierSpec(K=3, N_r=2)
    B = basis_matrix(*_grid(), spec, R_RANGE, TH_RANGE)
    assert B.shape == (_grid()[0].size, (2 + 1) * (2 * 3 + 1))


def test_radial_basis_known_values():
    u = np.array([0.0, 0.5, 1.0])
    cos = radial_basis(u, 2, "cos")
    assert cos.shape == (3, 3)
    np.testing.assert_allclose(cos[:, 0], 1.0)
    np.testing.assert_allclose(cos[:, 1], np.cos(np.pi * u))
    leg = radial_basis(u, 2, "legendre")
    np.testing.assert_allclose(leg[:, 0], 1.0)
    np.testing.assert_allclose(leg[:, 1], 2.0 * u - 1.0)


def test_theta_basis_orthogonality_uniform():
    """均匀采样上**同族**三角基两两正交。

    ⚠ 半区间 ``[0,1]`` 上 cos 与 sin **不正交**（``∫cos(kπv)sin(lπv)dv ≠ 0``
    当 ``k±l`` 为奇数），所以只能断言 cos–cos、sin–sin 两类内部正交。
    基仍然线性无关（Gram 矩阵满秩），这正是张量基成立的前提。
    """
    K = 5
    v = (np.arange(200) + 0.5) / 200.0          # 中点采样，避开端点
    cos = [np.cos(k * np.pi * v) for k in range(1, K + 1)]
    sin = [np.sin(k * np.pi * v) for k in range(1, K + 1)]
    for family in (cos, sin):
        G = np.array([[np.dot(a, b) for b in family] for a in family]) / v.size
        off = G - np.diag(np.diag(G))
        assert np.max(np.abs(off)) < 1e-12

    # 线性无关：整套（1 + cos + sin）Gram 矩阵满秩
    allc = [np.ones_like(v)] + cos + sin
    G = np.array([[np.dot(a, b) for b in allc] for a in allc]) / v.size
    assert np.linalg.matrix_rank(G) == len(allc)


# ---------------------------------------------------------------- 精确重构
def test_constant_field_exact():
    p = _param(PolarFourierSpec(K=2, N_r=2))
    rho = np.full(p.B.shape[0], 0.5)
    c = p.initial_coeff(rho)
    np.testing.assert_allclose(p.expand(c), rho, atol=1e-12)


def test_low_order_trig_field_exact():
    """ρ = 0.5 + 0.3·cos(2πv) 能被 k=2 的余弦项精确表示。"""
    p = _param(PolarFourierSpec(K=3, N_r=0))
    v = (p.theta - TH_RANGE[0]) / (TH_RANGE[1] - TH_RANGE[0])
    rho = 0.5 + 0.3 * np.cos(2.0 * np.pi * v)
    c = p.initial_coeff(rho)
    np.testing.assert_allclose(p.expand(c), rho, atol=1e-11)


def test_radial_linear_field_exact_with_legendre():
    """ρ = u（径向线性）能被 Legendre 一次项精确表示。"""
    p = _param(PolarFourierSpec(K=0, N_r=1, radial="legendre"))
    u = (p.r - R_RANGE[0]) / (R_RANGE[1] - R_RANGE[0])
    rho = u
    c = p.initial_coeff(rho)
    np.testing.assert_allclose(p.expand(c), rho, atol=1e-12)


# ---------------------------------------------------------------- 网格无关性
def test_mesh_independence_same_coefficients():
    """同一组系数在两组采样点上给出同一个连续场（公共点逐位一致）。"""
    spec = PolarFourierSpec(K=3, N_r=3)
    r1, th1 = _grid(4, 6)
    p1 = PolarFourierParam(r1, th1, spec, R_RANGE, TH_RANGE)
    rng = np.random.default_rng(0)
    c = rng.normal(scale=0.05, size=p1.n_coef)

    # 第二组：把第一组点也包含进去 + 额外随机点（成对，不是两个独立数组）
    rng2 = np.random.default_rng(10)
    r_extra = rng2.uniform(R_RANGE[0], R_RANGE[1], 17)
    th_extra = rng2.uniform(TH_RANGE[0], TH_RANGE[1], 17)
    p2 = PolarFourierParam(np.concatenate([r1, r_extra]),
                           np.concatenate([th1, th_extra]),
                           spec, R_RANGE, TH_RANGE)

    np.testing.assert_allclose(p2.expand(c)[:r1.size], p1.expand(c), atol=1e-14)


# ---------------------------------------------------------------- 梯度
def test_project_matches_finite_difference():
    """∂(w·ρ)/∂c 与中心差分一致（线性映射、内部区域）。"""
    p = _param(PolarFourierSpec(K=2, N_r=2), nr=5, nt=8)
    rng = np.random.default_rng(1)
    c = np.zeros(p.n_coef)
    c[0] = 0.5 * np.sqrt(p._n_ref)      # 常数模态 ⇒ g≈0.5，远离 clip 边界
    c += rng.normal(scale=0.002, size=c.size)
    rho = p.expand(c)
    assert np.all((rho > 1e-3) & (rho < 1 - 1e-3)), "FD 需在 clip 内部"
    w = rng.normal(size=rho.size)

    g = p.project(w[None, :], c)[0]
    eps = 1e-7
    fd = np.zeros_like(c)
    for j in range(c.size):
        e = np.zeros_like(c)
        e[j] = eps
        fd[j] = (w @ p.expand(c + e) - w @ p.expand(c - e)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-9)


def test_project_accumulates_slave_gradient():
    """从属单元的 ∂f/∂ρ 必须叠加到其主单元（与 gradient_to_x 同一步）。"""
    p = _param(PolarFourierSpec(K=1, N_r=1), nr=4, nt=4)
    n = p.B.shape[0]
    # 假拓扑：前 n 个是主动单元（编号 0..n-1），再加两个从属单元
    top = SimpleNamespace(
        element_topo_opti_m=set(range(n)),
        element_topo_opti_s={n: 0, n + 1: 1},
        n_element=n + 2,
    )
    p.attach_topology(top)
    w = np.zeros(n + 2)
    w[0] = 1.0          # 主动单元 0
    w[n] = 2.0          # 从属单元 → 主单元 0
    c = np.zeros(p.n_coef)
    c[0] = 0.5 * np.sqrt(p._n_ref)   # 让 g 在 clip 内部（dρ/dg = 1）
    g = p.project(w[None, :], c)[0]
    # 常数项列在正交基 Q 中全 1/√N ⇒ 主单元 0 的有效权重 3.0
    np.testing.assert_allclose(g[0], 3.0 * p.Bphys[0, 0], rtol=1e-12)


# ---------------------------------------------------------------- 映射
def test_sigmoid_mapping_range_and_derivative():
    p = _param(PolarFourierSpec(K=1, N_r=1, mapping="sigmoid", gain=4.0), nr=4, nt=4)
    rng = np.random.default_rng(2)
    c = rng.normal(scale=0.5, size=p.n_coef)
    rho = p.expand(c)
    assert np.all(rho > 0) and np.all(rho < 1)

    w = rng.normal(size=rho.size)
    g = p.project(w[None, :], c)[0]
    eps = 1e-7
    fd = np.zeros_like(c)
    for j in range(c.size):
        e = np.zeros_like(c)
        e[j] = eps
        fd[j] = (w @ p.expand(c + e) - w @ p.expand(c - e)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-9)


def test_sigmoid_initial_coeff_inverts_mapping():
    p = _param(PolarFourierSpec(K=2, N_r=2, mapping="sigmoid", gain=3.0))
    v = (p.theta - TH_RANGE[0]) / (TH_RANGE[1] - TH_RANGE[0])
    rho = 1.0 / (1.0 + np.exp(-(0.5 + 2.0 * np.cos(np.pi * v))))
    c = p.initial_coeff(rho)
    np.testing.assert_allclose(p.expand(c), rho, atol=1e-10)


# ---------------------------------------------------------------- 诊断
def test_fit_reports_identity_for_representable_field():
    p = _param(PolarFourierSpec(K=2, N_r=2))
    rng = np.random.default_rng(3)
    c_true = np.zeros(p.n_coef)
    c_true[0] = 0.5 * np.sqrt(p._n_ref)
    c_true += rng.normal(scale=0.001, size=c_true.size)
    assert np.all((p.Bphys @ c_true > 1e-3) & (p.Bphys @ c_true < 1 - 1e-3)), \
        "需在 clip 内部，否则 rho 不在列空间内、无法精确重构"
    rho = p.expand(c_true)
    rep = p.fit(rho)
    assert rep["r2"] > 1 - 1e-10
    assert rep["iou"] == pytest.approx(1.0)
    assert rep["n_coef"] == p.n_coef
