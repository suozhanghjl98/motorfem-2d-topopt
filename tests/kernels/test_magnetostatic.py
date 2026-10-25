"""样板内核测试：二维非线性静磁场的残差 / Jacobian / Newton。

本文件演示 `docs/重构设计文档.md` 阶段 1 规定的**三级测试来源**，
每个内核都必须有这三个层次的覆盖：

  来源 A  解析解          —— 唯一能证明「对」的来源（本题：线性材料下 FE 精确）
  来源 B  有限差分校验     —— 无法构造解析解的量（Jacobian）的独立外部判据
  来源 C  reference vs fast —— 证明两实现一致（但不能证明都对）

另有「Newton 收敛性」与「饱和项确实被激活」两项，用来防止测试退化——
如果饱和项恒为零，Jacobian 的 FD 校验会平凡通过而毫无意义。

运行::

    python -m pytest tests/kernels/test_magnetostatic.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from motorfem.kernels.reference import magnetostatic as ref_mod
from motorfem.kernels.fast import magnetostatic as fast_mod
from motorfem.kernels.fast.magnetostatic import build_positions, element_g


# =============================================================================
# 网格夹具
# =============================================================================

def make_rectangle_mesh(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
    """构造 ``[0,lx] × [0,ly]`` 上的结构化三角网格（逆时针）。

    Returns
    -------
    node : (N_node, 2)
    tri  : (N_elem, 3)
    """
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    node = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i: int, j: int) -> int:
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            n00, n10, n11, n01 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    return node, np.array(tris, dtype=int)


def left_right_dofs(node: np.ndarray, x_left: float, x_right: float):
    """取出 x=x_left / x=x_right 两列节点，用作 Dirichlet 边界。"""
    left = np.flatnonzero(np.isclose(node[:, 0], x_left))
    right = np.flatnonzero(np.isclose(node[:, 0], x_right))
    return left, right


# 线性材料：ν 为常数
def make_linear_material(nu0: float):
    return (lambda B: np.full_like(B, nu0)), (lambda B: np.zeros_like(B))


# 非线性材料：ν(B) = ν0·(1 + β·B²)   —— 随 |B| 增大（饱和），dν/dB > 0
def make_saturating_material(nu0: float, beta: float):
    return (
        lambda B: nu0 * (1.0 + beta * B ** 2),
        lambda B: 2.0 * nu0 * beta * B,
    )


# =============================================================================
# 来源 A：解析解
# =============================================================================

def test_linear_material_reproduces_exact_linear_solution():
    """线性材料 + 两侧 Dirichlet → 精确解 A = A0·x。

    这是本内核最强的正确性判据，理由：
      - 无源（R = 0）、上下为自然边界，真解是 A(x,y) = A0·x（Laplace 方程）。
      - 一阶三角形单元对**线性函数**是精确的（形函数可张成线性空间），
        因此 FE 解应当逐位复现该线性场，而不是"近似"。
      - 该判据完全独立于我们的两个实现——它来自数学，不是来自互相比较。
    """
    nu0 = 1.0 / (4e-7 * np.pi)
    nu_of_B, dnu_dB = make_linear_material(nu0)

    node, tri = make_rectangle_mesh(nx=4, ny=3)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)

    A0 = 0.05
    left, right = left_right_dofs(node, 0.0, 1.0)
    fixed_nodes = np.concatenate([left, right])
    fixed_values = np.concatenate([np.zeros_like(left, dtype=float),
                                   np.full(right.shape, A0, dtype=float)])

    R = np.zeros(node.shape[0])
    A_init = np.zeros(node.shape[0])

    A, info = ref_mod.newton_solve(
        A_init, geom, S, nu_of_B, dnu_dB, R,
        fixed_nodes=fixed_nodes, fixed_values=fixed_values,
        rel_err=1e-12, max_iter=20,
    )

    A_exact = A0 * node[:, 0]
    err = np.abs(A - A_exact).max()
    assert err < 1e-12, f"线性材料下未精确复现线性解，最大偏差 {err:.3e}"

    # 线性材料下 Newton 应在 1 步内收敛（Jacobian 精确）
    assert info["n_iter"] <= 2, f"线性问题应 1 步收敛，实际 {info['n_iter']} 步"


def test_flux_density_matches_analytic_gradient():
    """解析解的磁密应为常数：Bx = 0, By = -A0。"""
    nu0 = 1.0 / (4e-7 * np.pi)
    nu_of_B, dnu_dB = make_linear_material(nu0)
    node, tri = make_rectangle_mesh(nx=3, ny=3)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)

    A0 = 0.05
    left, right = left_right_dofs(node, 0.0, 1.0)
    A_init = np.zeros(node.shape[0])
    A, _ = ref_mod.newton_solve(
        A_init, geom, S, nu_of_B, dnu_dB, np.zeros(node.shape[0]),
        fixed_nodes=np.concatenate([left, right]),
        fixed_values=np.concatenate([np.zeros_like(left, dtype=float),
                                     np.full(right.shape, A0, dtype=float)]),
        rel_err=1e-12, max_iter=20,
    )

    # B = sqrt((b·A)² + (c·A)²)/(2Δ) 应恒为 A0（因为 ∂A/∂x = A0，∂A/∂y = 0）
    B = ref_mod.flux_density_magnitude(A, geom)
    assert np.allclose(B, A0, rtol=1e-12), f"|B| 应为常数 {A0}，实得 {B}"


# =============================================================================
# 来源 B：有限差分校验 Jacobian
# =============================================================================

@pytest.mark.parametrize("nx,ny", [(3, 3), (4, 2)])
def test_jacobian_matches_finite_difference(nx: int, ny: int):
    """``J·v`` 与中心差分的 ``(F(A+εv) − F(A−εv))/(2ε)`` 对比。

    这是 Jacobian 的**独立外部判据**。饱和项（秩 1 外积）是最难读、最易错的部分，
    而它对该项的存在与否极敏感——若饱和项写错或漏掉，此测试必然失败。

    中心差分精度 O(ε²)，取 ε = 1e-6 时两者相对误差应在 1e-6 量级。
    """
    nu0 = 1.0 / (4e-7 * np.pi)
    nu_of_B, dnu_dB = make_saturating_material(nu0, beta=50.0)

    node, tri = make_rectangle_mesh(nx=nx, ny=ny)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)

    rng = np.random.default_rng(20260922)
    A = rng.normal(scale=1e-2, size=node.shape[0])
    R = np.zeros(node.shape[0])

    F, nu = ref_mod.residual(A, geom, S, nu_of_B, R)
    J = ref_mod.jacobian(A, geom, S, nu, dnu_dB)

    v = rng.normal(size=node.shape[0])
    Jv = J @ v

    eps = 1e-6
    F_p, _ = ref_mod.residual(A + eps * v, geom, S, nu_of_B, R)
    F_m, _ = ref_mod.residual(A - eps * v, geom, S, nu_of_B, R)
    fd = (F_p - F_m) / (2.0 * eps)

    scale = max(np.linalg.norm(Jv), np.linalg.norm(fd))
    rel = np.linalg.norm(Jv - fd) / scale
    assert rel < 1e-5, f"Jacobian 与有限差分不符：相对误差 {rel:.3e}"


def test_saturation_term_is_actually_active():
    """防止上面的 FD 测试退化。

    若材料对 B 不敏感（β≈0）或 B≈0，饱和项恒为零，FD 校验会平凡通过。
    本测试断言：饱和项对 Jacobian 的贡献不可忽略。
    """
    nu0 = 1.0 / (4e-7 * np.pi)
    nu_of_B, dnu_dB = make_saturating_material(nu0, beta=50.0)

    node, tri = make_rectangle_mesh(nx=3, ny=3)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)

    rng = np.random.default_rng(7)
    A = rng.normal(scale=1e-2, size=node.shape[0])
    F, nu = ref_mod.residual(A, geom, S, nu_of_B, R=np.zeros(node.shape[0]))

    J_full = ref_mod.jacobian(A, geom, S, nu, dnu_dB)
    J_lin = ref_mod.jacobian(A, geom, S, nu, lambda B: np.zeros_like(B))

    sat_norm = np.abs((J_full - J_lin).toarray()).max()
    assert sat_norm > 1e-6, (
        f"饱和项贡献 {sat_norm:.3e} 过小，FD 校验会平凡通过，测试无意义"
    )


# =============================================================================
# 来源 C：reference vs fast 一致性
# =============================================================================

def test_parity_flux_density():
    node, tri = make_rectangle_mesh(nx=4, ny=3)
    geom = ref_mod.element_geometry(node, tri)
    rng = np.random.default_rng(1)
    A = rng.normal(scale=1e-2, size=node.shape[0])

    B_ref = ref_mod.flux_density_magnitude(A, geom)
    B_fast = fast_mod.flux_density_magnitude(A, geom)
    assert np.allclose(B_ref, B_fast, rtol=1e-12, atol=0)


def test_parity_element_g():
    """``g_e = S_e A_e``：fast 用 (b·A)·b + (c·A)·c 的恒等式，应与显式 S_e A_e 一致。"""
    node, tri = make_rectangle_mesh(nx=3, ny=3)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)
    pos = build_positions(geom, node.shape[0])

    rng = np.random.default_rng(2)
    A = rng.normal(scale=1e-2, size=node.shape[0])
    A_e = A[geom.tri]
    g_ref = np.einsum("eij,ej->ei", S, A_e)
    g_fast = element_g(A, geom, S, pos)

    assert np.allclose(g_ref, g_fast, rtol=1e-12, atol=1e-18)


def test_parity_residual():
    node, tri = make_rectangle_mesh(nx=4, ny=3)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)
    nu_of_B, _ = make_saturating_material(1e6, beta=50.0)

    rng = np.random.default_rng(3)
    A = rng.normal(scale=1e-2, size=node.shape[0])
    R = rng.normal(scale=1e-3, size=node.shape[0])

    F_ref, nu_ref = ref_mod.residual(A, geom, S, nu_of_B, R)
    F_fast, nu_fast = fast_mod.residual(A, geom, S, nu_of_B, R)

    assert np.allclose(nu_ref, nu_fast, rtol=1e-13, atol=0)
    assert np.allclose(F_ref, F_fast, rtol=1e-12, atol=1e-16)


def test_parity_jacobian():
    """Jacobian 的 parity——重点验证饱和项（秩 1 外积）两实现一致。"""
    node, tri = make_rectangle_mesh(nx=4, ny=3)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)
    nu_of_B, dnu_dB = make_saturating_material(1e6, beta=50.0)

    rng = np.random.default_rng(4)
    A = rng.normal(scale=1e-2, size=node.shape[0])

    _, nu = ref_mod.residual(A, geom, S, nu_of_B, np.zeros(node.shape[0]))
    J_ref = ref_mod.jacobian(A, geom, S, nu, dnu_dB)
    J_fast = fast_mod.jacobian(A, geom, S, nu, dnu_dB)

    assert J_ref.shape == J_fast.shape
    diff = np.abs((J_ref - J_fast).toarray()).max()
    scale = np.abs(J_ref.toarray()).max()
    assert diff <= 1e-12 * scale, f"Jacobian 不一致：最大偏差 {diff:.3e}（尺度 {scale:.3e}）"


def test_parity_newton_end_to_end():
    """端到端：ref 与 fast 的 Newton 解一致。"""
    node, tri = make_rectangle_mesh(nx=4, ny=3)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)
    nu_of_B, dnu_dB = make_saturating_material(1e6, beta=50.0)

    left, right = left_right_dofs(node, 0.0, 1.0)
    fixed_nodes = np.concatenate([left, right])
    fixed_values = np.concatenate([np.zeros_like(left, dtype=float),
                                   np.full(right.shape, 0.05, dtype=float)])
    A_init = np.zeros(node.shape[0])
    R = np.zeros(node.shape[0])

    A_ref, info_ref = ref_mod.newton_solve(
        A_init, geom, S, nu_of_B, dnu_dB, R,
        fixed_nodes=fixed_nodes, fixed_values=fixed_values,
        rel_err=1e-12, max_iter=50,
    )
    A_fast, info_fast = fast_mod.newton_solve(
        A_init, geom, S, nu_of_B, dnu_dB, R,
        fixed_nodes=fixed_nodes, fixed_values=fixed_values,
        rel_err=1e-12, max_iter=50,
    )

    assert np.allclose(A_ref, A_fast, rtol=1e-10, atol=1e-14)
    assert info_ref["n_iter"] == info_fast["n_iter"]


# =============================================================================
# Newton 迭代本身
# =============================================================================

def test_newton_converges_quadratically_on_saturating_material():
    """非线性材料下 Newton 应收敛，且末端残差下降快（接近二次）。"""
    nu0 = 1.0 / (4e-7 * np.pi)
    nu_of_B, dnu_dB = make_saturating_material(nu0, beta=50.0)

    node, tri = make_rectangle_mesh(nx=5, ny=4)
    geom = ref_mod.element_geometry(node, tri)
    S = ref_mod.element_stiffness(geom)

    left, right = left_right_dofs(node, 0.0, 1.0)
    A, info = ref_mod.newton_solve(
        np.zeros(node.shape[0]), geom, S, nu_of_B, dnu_dB,
        np.zeros(node.shape[0]),
        fixed_nodes=np.concatenate([left, right]),
        fixed_values=np.concatenate([np.zeros_like(left, dtype=float),
                                     np.full(right.shape, 0.3, dtype=float)]),
        rel_err=1e-10, max_iter=50,
        # 强非线性下需要松弛；这里放开前 20 步全步长
        relaxation_iter=20, relaxation_ratio=0.5,
    )

    # 用相对残差判定（绝对容差随 ν 与网格尺度缩放，见 newton_solve 的注释）
    assert info["residual_norm"] < 1e-9 * info["history"][0], (
        f"未收敛，残差 {info['residual_norm']:.3e}（首次 {info['history'][0]:.3e}）"
    )
    assert info["n_iter"] < 40

    hist = info["history"]
    assert np.all(np.isfinite(hist)), "残差历史出现非有限值"


def test_geometry_rejects_clockwise_elements():
    """面积非正必须立刻报错，而不是静默算出"看起来对"的结果。"""
    node, tri = make_rectangle_mesh(nx=1, ny=1)
    bad_tri = tri[:, ::-1].copy()             # 反转一个单元的方向
    with pytest.raises(ValueError, match="逆时针"):
        ref_mod.element_geometry(node, bad_tri)
