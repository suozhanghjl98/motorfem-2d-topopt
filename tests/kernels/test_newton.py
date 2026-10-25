"""内核 9/10/11 测试：残差 / Jacobian / Newton。

测试三级来源（`docs/重构设计文档.md` §1.4 与 `CLAUDE.md` §4）：

  来源 A 解析解/解析性质
        内核 9 —— 手算 2 单元残差、``K·1 = 0``、线性材料下 ``F`` 的仿射性、
                  驱动行**逐位**等于约束表达式、掩码逐类**逐位**精确、
                  ``aelem == area`` 逐位、``0*nan`` 的加权语义、``B`` 的解析梯度
        内核 10 —— **单单元闭式解**（正面验全部数学，含恒等式 ``∂B/∂A = g/(ΔB)``）、
                   无驱动行时 ``J`` 的对称性、线性材料下 ``J == K``（**边界情形**）
        内核 11 —— **假 callable 记账**（完全解析）：松弛判据的 0 基语义、
                   平方范数判据、``max_iter=0``、打印格式的 1 基轮次、``A0`` 不被改写
  来源 B 有限差分 —— ``J·v ≈ (F(A+εv) − F(A−εv))/(2ε)``，**必须配**饱和项防退化断言
  来源 C reference vs fast —— 见本文件末节（``fast/`` 落地后启用）

**脚手架来源说明**：本文件的小网格夹具（同心圆环扇 + 内核 6/7 产出）与
``test_assembly.py`` 同构；这里不复用它的代码，而是照同样的方式重建，以便两个测试
文件可以独立修改。材料部分用 ``make_saturating_material`` —— 它是一条**解析**的
``ν(B) = ν0(1 + βB²)`` 曲线（PCHIP 插值出来的表），不是重写一遍内核 3。

运行::

    python -m pytest tests/kernels/test_newton.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse as sp_sparse

from motorfem.kernels.reference import assembly as ref_asm
from motorfem.kernels.reference import bh_curve as ref_bh
from motorfem.kernels.reference import boundaries as ref_bnd
from motorfem.kernels.reference import element_matrices as ref_em
from motorfem.kernels.reference import material as ref_mat
from motorfem.kernels.reference import moving_mesh as ref_mm
from motorfem.kernels.reference import newton as ref_nt
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection

try:  # 子任务 4 落地后才有
    from motorfem.kernels.fast import newton as fast_nt
except ImportError:  # pragma: no cover
    fast_nt = None

requires_fast = pytest.mark.skipif(fast_nt is None, reason="fast/ 层尚未落地")

# ⚠ 本文件**不读 50CS470.csv**：所有材料都用下面那条解析曲线。理由是解析测试要的是
# **闭式**对照（PCHIP 插值会带进插值误差，把断言从「实现对不对」降级成「插值器像不像」）。
# 真实 B-H 曲线的路径由 tools/probe_newton.py 在真网格上覆盖，那里才需要 CSV。


# =============================================================================
# 脚手架 1：解析材料
# =============================================================================

class AnalyticBHCurve(ref_bh.BHCurve):
    """``ν(B) = ν0·(1 + βB²)`` 的**精确**解析材料（覆盖插值，供闭式断言用）。

    由 ``ν = H(B)/B`` 反解::

        H(B)    = ν0·(B + βB³)
        dH/dB   = ν0·(1 + 3βB²)
        ν(B)    = H/B      = ν0(1 + βB²)
        dν/dB   = dH/dB/B − H/B² = ν0·2βB

    覆盖 ``H``/``dHdB`` 而不走 PCHIP，是为了让「闭式解」断言里没有插值误差。
    基类的 ``H_ext``/``B_ext`` 只是占位（本子类不会用到它们）。
    """

    def __init__(self, nu0: float, beta: float):
        self.nu0 = float(nu0)
        self.beta = float(beta)
        super().__init__(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def H(self, B):                                    # noqa: D102
        B = np.asarray(B, dtype=float)
        return self.nu0 * (B + self.beta * B ** 3)

    def dHdB(self, B):                                 # noqa: D102
        B = np.asarray(B, dtype=float)
        return self.nu0 * (1.0 + 3.0 * self.beta * B ** 2)


#: 解析材料：ν0 取接近真实铁芯的量级（ν ≈ 1/μ，μ ≈ 1e-3 ⇒ ν ≈ 1e3）；
#: β = 50 与 ``test_magnetostatic.py`` 的合成材料同量级 —— 要**明显非线性**，
#: 否则 Newton 一步就收敛，内核 11 的分段松弛永远走不到。
NU0 = 1.0e3
BETA = 50.0
#: 线性铁磁的相对磁导率（legacy 硬编码 2000）
MUR_LINEAR = 2000.0
MU0 = 4e-7 * np.pi


def make_saturating_material() -> ref_nt.FieldMaterial:
    """解析的饱和材料 + 默认 SIMP 投影。"""
    return ref_nt.FieldMaterial(
        curve=AnalyticBHCurve(NU0, BETA), simp=SIMPProjection(SIMPParams()),
        mu0=MU0, mur_linear=MUR_LINEAR,
    )


# =============================================================================
# 脚手架 2：两个小网格
# =============================================================================

def make_two_triangle_square(lx=1.0, ly=1.0):
    """``[0,lx]×[0,ly]`` 分成 2 个三角形（逆时针）。手算规模。"""
    node = np.array([[0.0, 0.0], [lx, 0.0], [lx, ly], [0.0, ly]])
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    return node, tri


SECTOR = np.pi / 2
COORD_ERR_ABS = 1e-6
R_RINGS = (0.030, 0.040, 0.050, 0.060)


def make_donut_mesh(n_ang=7, radii=R_RINGS):
    """四层同心圆环扇。节点号 ``= 层序号·n_ang + 角度序号``，含两端点。"""
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
    """照抄 ``MMA10:339-340``：半径**平方差**判据（返回 set）。"""
    return set(
        i for i in range(xy.shape[0])
        if abs((xy[i, 0] ** 2 + xy[i, 1] ** 2) - r_interface ** 2) < coord_err_abs
    )


# =============================================================================
# 脚手架 3：FieldGeometry 的两种构造方式
# =============================================================================

def simple_field(node, tri, *, element_iron=(), element_topo_opti=(),
                 use_nonlinear_iron=True):
    """``T = I``（无主从、无动网格）的 ``FieldGeometry`` + 内核 4 的产出。

    用**空集**构造 ``TopologyContext`` ⇒ 内核 8 的 ``transformation_matrix``
    对每个单元都走分支 3（``T = I₃``、``index = tri[e]``），也就完全不读
    ``rotating``。这是「手算 2 单元」类解析测试最干净的入口。
    """
    node = np.asarray(node, dtype=float)
    tri = np.asarray(tri)
    n_node = node.shape[0]
    em = ref_em.electromagnetic_elements(node, tri)
    ctx = ref_asm.TopologyContext(
        tri=tri, virtual_tri={}, element_rotor_gap=frozenset(),
        element_slave=frozenset(), master_nodes=frozenset(), a_eq=False, n_node=n_node,
    )
    ops = ref_asm.element_operators(
        ctx, None, n_node=n_node, k0elem=em.K0, bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        element_iron=set(int(e) for e in element_iron),
        element_topo_opti=set(int(e) for e in element_topo_opti),
        use_nonlinear_iron=use_nonlinear_iron,
    )
    field = ref_nt.field_geometry(
        ctx, None, n_node=n_node, k0elem=em.K0, bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        aelem=em.area, masks=ref_nt.MaterialMasks.from_operators(ops),
    )
    return field, em, ops


class DonutFixture:
    """同心圆环扇 + 内核 6/7 的完整产出（走真分支 1/2/3）。"""

    def __init__(self, n_ang=7, a_eq=False, pos=0.31, use_nonlinear_iron=True):
        self.xy, element_S, self.element_rotor_gap, self.element_stator, \
            self.element_rotor = make_donut_mesh(n_ang=n_ang)
        self.tri = np.array([e[1:4] for e in element_S])

        pairing0 = ref_bnd.pair_master_slave(
            self.xy, parts=4, coord_err_abs=COORD_ERR_ABS
        )
        self.layer = ref_mm.insert_moving_mesh_layer(
            self.xy, self.tri, tuple(interface_nodes_legacy(self.xy)),
            self.element_rotor_gap, pairing0,
            element_stator=self.element_stator, element_rotor=self.element_rotor,
        )
        self.virtual = ref_bnd.slave_virtual_elements(self.layer.tri, self.layer.pairing)
        self.a_eq = a_eq
        self.n_node, self.n_elem = self.layer.n_node, self.layer.tri.shape[0]
        # ⚠ 必须换成**插入层之后**的坐标：`n_node` 比原网格多出滑移层那一圈，
        # 用原 `xy` 索引会越界（这是本文件踩过的坑）。
        self.xy = self.layer.xy

        self.ctx = ref_asm.TopologyContext.from_boundaries(
            self.layer.tri, self.virtual, self.layer.pairing,
            self.element_rotor_gap, a_eq=a_eq, n_node=self.n_node,
        )
        self.rot = ref_mm.rotating_boundary(
            self.layer, parts=4, pos=pos, a_eq=a_eq
        )
        self.em = ref_em.electromagnetic_elements(self.layer.xy, self.layer.tri)

        # 铁磁 = 转子侧前两层；拓扑区 = 转子最内一层
        self.element_iron = set(self.element_rotor)
        self.element_topo = set(e for e in self.element_rotor if e < 12)

        self.ops = ref_asm.element_operators(
            self.ctx, self.rot, n_node=self.n_node,
            k0elem=self.em.K0, bxcoef=self.em.Bxcoef, bycoef=self.em.Bycoef,
            element_iron=self.element_iron, element_topo_opti=self.element_topo,
            use_nonlinear_iron=use_nonlinear_iron,
        )
        self.masks = ref_nt.MaterialMasks.from_operators(self.ops)
        self.field = ref_nt.field_geometry(
            self.ctx, self.rot, n_node=self.n_node,
            k0elem=self.em.K0, bxcoef=self.em.Bxcoef, bycoef=self.em.Bycoef,
            aelem=self.em.area, masks=self.masks,
        )
        self.material = make_saturating_material()

        # 内圈 Dirichlet + 外圈若干载流单元的源项
        self.dirichlet = np.array(sorted(
            i for i in range(self.n_node)
            if abs(self.xy[i, 0] ** 2 + self.xy[i, 1] ** 2 - R_RINGS[0] ** 2)
            < COORD_ERR_ABS
        ), dtype=int)
        # ⚠ 动网格插值行必须经 `row_replacements` 灌进驱动行装置（legacy
        # ``MMA10:2104-2112``）。**漏掉它，插入层的节点就成了自由节点、矩阵奇异**
        # （实测 `dgstrf info <插入层节点号>` + `Matrix is exactly singular`）——
        # 因为那些节点被 `T` 的插值行消去，在 `K` 里是零行。
        row_replacements = {
            int(n): np.asarray(self.rot.matrix[n, :].todense()).ravel()
            for n in self.layer.rotor_nodes
        }
        self.driving = ref_bnd.driving_rows(
            self.n_node, self.dirichlet, self.layer.pairing.slave_to_master, a_eq,
            row_replacements,
        )
        self.rho = np.where(
            np.isin(np.arange(self.n_elem), sorted(self.element_iron)), 1.0, 0.5
        )
        self.set_source(5.0e7)

    def set_source(self, js: float):
        """在外圈（定子侧）单元上铺源项 ``js · R0``。

        量级说明：``R0 = Δ/3``、``Δ ~ 7e-5``，而 ``K ~ ν/Δ ~ 1e7``（``ν ~ 1e3``），
        所以 ``R`` 要到 ~1e3 量级才能把 ``|A|`` 抬到 ``Δ·|B|``（``|B| ~ 1 T``），
        即真正进入饱和段。**实测（``n_ang=7`` 的玩具网格）**：

        ==========  ==========  =============  ===============
        ``js``      迭代轮数    ``\\|A\\|max``      ``\\|B\\|`` 范围
        ==========  ==========  =============  ===============
        1e5         2           1.1e-4         [1.9e-4, 5.7e-3]
        1e7         4           1.1e-2         [2.2e-2, 5.5e-1]
        **5e7**     **6**       **3.7e-2**     **[1.9e-1, 2.0]**
        ==========  ==========  =============  ===============

        取 ``5e7``：``|B|`` 真正落进饱和段，Newton 走 6 步（>1 次更新，满足
        「防退化」要求），残差历史 ``6.1e6 → 3.6e4 → 1.5e1 → 4.8e-5 → …``
        呈现二次收敛。
        """
        self.R = np.zeros(self.n_node)
        for e in sorted(self.element_stator):
            self.R[list(self.field.transforms[e].index)] += js * self.em.R0[e]

    def empty_driving(self):
        """无约束的 ``DrivingRows`` —— ``J_raw`` 的对称性要在它上面验。"""
        return ref_bnd.driving_rows(self.n_node, [], {}, a_eq=self.a_eq)

    def solved(self, **kw):
        """在当前 ``rho``/``R`` 上求解，返回 ``(A, aux, J, history)``。"""
        kwargs = dict(solve_abs_err=1e-3, max_iter=100, iter_relaxation=20,
                      relaxation_ratio=0.5, output=False)
        kwargs.update(kw)
        return ref_nt.newton_solve(
            None, self.rho, self.field, self.material, self.R, self.driving, **kwargs
        )


@pytest.fixture(scope="module")
def donut() -> DonutFixture:
    return DonutFixture()


@pytest.fixture(scope="module")
def donut_solved() -> DonutFixture:
    """已经解过一次的夹具（``.A_conv`` 是收敛的磁场）。"""
    fx = DonutFixture()
    fx.A_conv, fx.aux_conv, fx.J_conv, fx.history_conv = fx.solved()
    return fx


@pytest.fixture(scope="module")
def material() -> ref_nt.FieldMaterial:
    return make_saturating_material()


# =============================================================================
# 脚手架 4：独立的稠密单元刚度（不复用被测实现）
# =============================================================================

def kernel8_assembled_stiffness(ops, mu, n_node):
    """内核 8 的位置矩阵重装配：``csr((K0_data·(Kconst@(1/μ)), (K0_row, K0_col)))``。

    ``field_error_vectorized`` 里 ``K_data = K0_data * (Kconst_mat @ (1 / mu))``
    那一步的独立复现 —— 用来给 reference/fast 的容差定一个**有依据的尺度**。
    """
    return sp_sparse.coo_matrix(
        (ops.k0_data * (ops.kconst_mat @ (1.0 / mu)), (ops.k0_row, ops.k0_col)),
        shape=(n_node, n_node),
    ).tocsr()


def dense_element_stiffness(node, tri, mu_per_elem):
    """``Σ_e (1/μ_e)·S_e`` 的稠密矩阵，``S_e = (bbᵀ + ccᵀ)/(4Δ)``。

    **独立实现**：直接从教科书的 ``b_i/c_i/Δ`` 定义出发（`同步磁阻电机磁障结构分析.md`
    「考虑饱和的求解」），不调用内核 4 与内核 9。
    """
    node = np.asarray(node, dtype=float)
    n_node = node.shape[0]
    K = np.zeros((n_node, n_node))
    for e in range(tri.shape[0]):
        n = list(tri[e])
        x, y = node[n, 0], node[n, 1]
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([-x[1] + x[2], -x[2] + x[0], -x[0] + x[1]])
        Delta = (b[0] * c[1] - b[1] * c[0]) / 2.0
        S = (np.outer(b, b) + np.outer(c, c)) / (4.0 * Delta)
        K[np.ix_(n, n)] += S / mu_per_elem[e]
    return K


# =============================================================================
# 来源 A —— 内核 9：残差
# =============================================================================

def test_flux_density_matches_analytic_gradient():
    """``B`` 的解析值：manufactured 线性 ``A(x,y) = α·x`` ⇒ ``B`` 恒为 ``α``（legacy 约定下）。

    取 ``A = α·y``：``Bx = ∂A/∂y = α``、``By = −∂A/∂x = 0`` ⇒ ``|B| = α``。
    """
    node, tri = make_two_triangle_square(lx=2.0, ly=3.0)
    field, _, _ = simple_field(node, tri)

    alpha = 0.7
    A = alpha * node[:, 1]                         # A = α·y
    B = ref_nt.flux_density(A, field)
    assert np.allclose(B, alpha, rtol=1e-15, atol=0.0)


def test_hand_assembled_two_element_residual():
    """手算 2 单元：全真空时 ``μ = μ0`` ⇒ ``F = K A − R``，``K`` 由解析式独立装出。"""
    node, tri = make_two_triangle_square()
    field, em, _ = simple_field(node, tri)
    empty = ref_bnd.driving_rows(node.shape[0], [], {}, a_eq=False)
    mat = make_saturating_material()

    rng = np.random.default_rng(1)
    A = rng.normal(scale=1e-2, size=node.shape[0])
    R = rng.normal(scale=1e-3, size=node.shape[0])
    rho = np.full(tri.shape[0], 0.8)

    F, aux = ref_nt.residual(A, rho, field, mat, R, empty)

    # 全真空 ⇒ 逐类加权后 μ 恰好是 mu0
    assert np.all(aux.mu == mat.mu0)

    K = dense_element_stiffness(node, tri, np.full(tri.shape[0], mat.mu0))
    assert np.allclose(F, K @ A - R, rtol=1e-14, atol=0.0)


def test_stiffness_times_ones_is_zero():
    """``Σ_j K_ij = ν_e·Σ_j S_ij = 0``（因 ``Σ_s b_s = Σ_s c_s = 0``）。

    这是一条「漏装任一 3×3 分量就会破」的解析性质。容差按 ``max|K|`` 缩放 ——
    ``Σ b_s`` 在浮点下不完全为零。
    """
    node, tri = make_two_triangle_square(lx=1.3, ly=0.7)
    field, _, _ = simple_field(node, tri)
    empty = ref_bnd.driving_rows(node.shape[0], [], {}, a_eq=False)
    mat = make_saturating_material()

    A = np.ones(node.shape[0])
    _, aux = ref_nt.residual(A, np.full(tri.shape[0], 1.0), field, mat,
                             np.zeros(node.shape[0]), empty)
    J = ref_nt.jacobian(A, field, aux, empty)

    ones = np.ones(node.shape[0])
    row_sums = J @ ones
    assert np.abs(row_sums).max() <= 1e-14 * abs(J).max()


def test_residual_is_affine_for_linear_material():
    """全真空/线性铁磁 ⇒ ``F`` 对 ``A`` 严格仿射（抓「掩码误用导致非线性泄漏」）。"""
    node, tri = make_two_triangle_square()
    field, _, _ = simple_field(node, tri)
    empty = ref_bnd.driving_rows(node.shape[0], [], {}, a_eq=False)
    mat = make_saturating_material()
    rho = np.full(tri.shape[0], 0.5)
    R = np.full(node.shape[0], 1e-3)

    rng = np.random.default_rng(7)
    A1 = rng.normal(scale=1e-3, size=node.shape[0])
    A2 = rng.normal(scale=1e-3, size=node.shape[0])

    F0, _ = ref_nt.residual(np.zeros_like(A1), rho, field, mat, R, empty)
    F1, _ = ref_nt.residual(A1, rho, field, mat, R, empty)
    F2, _ = ref_nt.residual(A2, rho, field, mat, R, empty)
    F12, _ = ref_nt.residual(A1 + A2, rho, field, mat, R, empty)

    lhs = F12 - F0
    rhs = (F1 - F0) + (F2 - F0)
    scale = max(np.abs(lhs).max(), np.abs(rhs).max())
    assert np.abs(lhs - rhs).max() <= 1e-13 * scale


def test_driven_rows_are_replaced_exactly():
    """驱动行**逐位**等于约束表达式（Dirichlet 行 = ``A_i`` 本身）。"""
    node, tri = make_two_triangle_square()
    field, _, _ = simple_field(node, tri)
    fixed = [0, 3]
    driving = ref_bnd.driving_rows(node.shape[0], fixed, {}, a_eq=False)
    mat = make_saturating_material()

    rng = np.random.default_rng(3)
    A = rng.normal(scale=1e-2, size=node.shape[0])
    F, _ = ref_nt.residual(A, np.full(tri.shape[0], 0.9), field, mat,
                           np.zeros(node.shape[0]), driving)

    for i in fixed:
        assert F[i] == A[i], "受约束行必须逐位等于约束表达式"


def test_masks_select_the_right_material_per_class():
    """掩码逐类**逐位**精确：每类单元的 μ / dν/dB 恰好等于该类解析值。"""
    node, tri = make_two_triangle_square()
    rho = np.array([0.9, 0.4])
    B = np.array([0.8, 1.4])
    mat = make_saturating_material()
    mu_simp, dg_simp = ref_nt.nonlinear_material_topo_opti(
        B, rho, mat.curve, mat.simp
    )
    mu_non, dg_non = ref_nt.nonlinear_material(B, mat.curve)

    def masks_for(kind):
        z = np.zeros(2)
        o = np.ones(2)
        return ref_nt.MaterialMasks(
            is_vacuum=o if kind == "vacuum" else z,
            is_simp=o if kind == "simp" else z,
            is_non_simp=o if kind == "non_simp" else z,
            is_linear_iron=o if kind == "linear_iron" else z,
        )

    mu, dg = ref_nt.element_field_material(B, rho, masks_for("vacuum"), mat)
    assert np.array_equal(mu, np.full(2, mat.mu0))
    assert np.all(dg == 0)

    mu, dg = ref_nt.element_field_material(B, rho, masks_for("linear_iron"), mat)
    assert np.array_equal(mu, np.full(2, mat.mur_linear * mat.mu0))
    assert np.all(dg == 0)

    mu, dg = ref_nt.element_field_material(B, rho, masks_for("simp"), mat)
    assert np.array_equal(mu, mu_simp) and np.array_equal(dg, dg_simp)

    mu, dg = ref_nt.element_field_material(B, rho, masks_for("non_simp"), mat)
    assert np.array_equal(mu, mu_non) and np.array_equal(dg, dg_non)


def test_zero_times_nan_poisons_the_blend():
    """``0 * nan = nan`` —— 加权合成**不是**选择。

    改动 ``blend_class_values`` 为 ``np.where`` / 布尔索引会静默通过所有
    parity 测试（那是两层共用的代码），所以这条解析语义必须单独钉住。
    """
    n = 3
    z = np.zeros(n)
    o = np.ones(n)
    masks = ref_nt.MaterialMasks(is_vacuum=o, is_simp=z, is_non_simp=z, is_linear_iron=z)

    poisoned = np.array([np.nan, np.inf, 1.0])
    # 这里的 `invalid value encountered in multiply` 是**被测语义本身**，不是意外：
    # `0 * nan = nan` 正是要钉住的行为。故显式静音，免得它淹没真正的警告。
    with np.errstate(invalid="ignore"):
        mu, dg = ref_nt.blend_class_values(
            poisoned, poisoned, poisoned, poisoned, masks,
            mu0=MU0, mur_linear=MUR_LINEAR,
        )
    # 前两个元素：掩码为 0 的被污染类仍然把结果带成 nan
    assert np.isnan(mu[:2]).all(), "被污染的类即使掩码为 0 也必须把结果带成 nan"
    assert np.isnan(dg[:2]).all()
    # 第三个元素：只有有限值参与，结果必须干净（证明这不是「一律返回 nan」）
    assert mu[2] == MU0 and dg[2] == 0.0


def test_aelem_equals_element_area():
    """legacy 的 ``Aelem`` 就是 ``Δ``；内核 4 的 ``area`` 与它**逐位相同**。"""
    donut = DonutFixture()
    assert np.array_equal(donut.field.aelem, donut.em.area)
    # 反向不成立：不能从 R0 反推面积
    assert not np.array_equal(donut.em.area, donut.em.R0 * 3)


# =============================================================================
# 来源 A —— 内核 10：Jacobian
# =============================================================================

def _single_element_setup():
    node = np.array([[0.0, 0.0], [0.4, 0.0], [0.0, 0.3]])
    tri = np.array([[0, 1, 2]])
    field, em, _ = simple_field(node, tri)
    return node, tri, field, em


def _analytic_material_arrays(B):
    """``ν = ν0(1+βB²)`` 下的 ``(mu, dgamma)`` 闭式。"""
    nu = NU0 * (1.0 + BETA * B ** 2)
    return 1.0 / nu, 2.0 * NU0 * BETA * B


def test_single_element_closed_form_jacobian():
    """单单元闭式解 —— 内核 10 的**正面**验证。

    手算::

        S_e   = (bbᵀ + ccᵀ)/(4Δ)
        g_e   = S_e A_e                       （几何量，**不含 ν**）
        J_e   = ν_e·S_e + [ (dν/dB)/(Δ·B) ]·g_e ⊗ g_e

    这一条同时验了 ``J`` 的两项、``∂B/∂A = g/(ΔB)`` 这个恒等式，以及饱和项的
    **秩 1** 结构（任何把 ``gswap1``/``gswap2`` 弄反或写错转置的实现都会失败）。
    """
    node, tri, field, em = _single_element_setup()
    n = list(tri[0])
    x, y = node[n, 0], node[n, 1]
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([-x[1] + x[2], -x[2] + x[0], -x[0] + x[1]])
    Delta = (b[0] * c[1] - b[1] * c[0]) / 2.0
    S = (np.outer(b, b) + np.outer(c, c)) / (4.0 * Delta)

    A = np.array([0.31, -0.17, 0.09])
    B = ref_nt.flux_density(A, field)
    mu, dgamma = _analytic_material_arrays(B)
    nu = 1.0 / mu

    empty = ref_bnd.driving_rows(3, [], {}, a_eq=False)
    _, J = ref_nt.field_system(A, field, mu, dgamma, B, np.zeros(3), empty)

    g = S @ A
    coef = 1 / B[0] / Delta * dgamma[0]
    J_expected = nu[0] * S + coef * np.outer(g, g)

    assert np.allclose(J.toarray(), J_expected, rtol=1e-13, atol=1e-18)


def test_dbdA_identity_matches_the_analytic_derivative():
    """``∂B/∂A = g/(Δ·B)``，其中 ``g = S_e A_e`` —— 用中心差分独立验一次。"""
    node, tri, field, em = _single_element_setup()
    rng = np.random.default_rng(5)
    A = rng.normal(scale=1e-2, size=3)
    B = ref_nt.flux_density(A, field)[0]

    # g/(Δ·B) 的三分量
    n = list(tri[0])
    x, y = node[n, 0], node[n, 1]
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([-x[1] + x[2], -x[2] + x[0], -x[0] + x[1]])
    Delta = (b[0] * c[1] - b[1] * c[0]) / 2.0
    S = (np.outer(b, b) + np.outer(c, c)) / (4.0 * Delta)
    grad_analytic = (S @ A) / (Delta * B)

    grad_fd = np.empty(3)
    eps = 1e-7
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        grad_fd[k] = (ref_nt.flux_density(A + d, field)[0]
                      - ref_nt.flux_density(A - d, field)[0]) / (2 * eps)

    assert np.allclose(grad_analytic, grad_fd, rtol=1e-6, atol=0.0)


def test_jacobian_is_symmetric_without_driving_rows():
    """无驱动行时 ``J`` 必对称 —— 它是能量泛函 ``W(A)`` 的 Hessian。

    ``TᵀK0T`` 对称、``g⊗g`` 对称，所以任何一项写错都会破坏对称性。
    容差按 ``max|J|`` 缩放（`CLAUDE.md` §4 的容差纪律）。
    """
    donut = DonutFixture()
    empty = donut.empty_driving()
    A = 0.02 * np.random.default_rng(9).normal(size=donut.n_node)

    _, aux = ref_nt.residual(A, donut.rho, donut.field, donut.material, donut.R, empty)
    J = ref_nt.jacobian(A, donut.field, aux, empty)

    asym = np.abs((J - J.T).toarray()).max()
    assert asym <= 1e-14 * np.abs(J).max(), f"不对称 {asym:.3e}"


def test_linear_material_jacobian_equals_assembled_stiffness():
    """**边界情形**（不是本内核的来源 A）：材料对 B 不敏感时 ``J == K``。

    与 ``test_saturation_term_is_actually_active`` 成对出现才有意义 ——
    若整条饱和项被漏掉，这条测试**照样通过**。
    """
    node, tri = make_two_triangle_square(lx=1.1, ly=0.9)
    field, _, _ = simple_field(node, tri, element_iron=[0, 1], use_nonlinear_iron=False)
    empty = ref_bnd.driving_rows(node.shape[0], [], {}, a_eq=False)
    mat = make_saturating_material()

    rng = np.random.default_rng(2)
    A = rng.normal(scale=1e-2, size=node.shape[0])
    _, aux = ref_nt.residual(A, np.full(2, 0.5), field, mat, np.zeros(4), empty)
    J = ref_nt.jacobian(A, field, aux, empty)

    assert np.all(aux.dgamma == 0)
    K = dense_element_stiffness(node, tri, np.full(2, mat.mur_linear * mat.mu0))
    assert np.allclose(J.toarray(), K, rtol=1e-14, atol=0.0)


def test_jacobian_linear_part_matches_kernel8_assembly():
    """``J`` 的线性部分 == 内核 8 的位置矩阵重装配（跨内核一致性）。

    把饱和项置零（``dgamma = 0``），``J`` 就退化为
    ``csr((K0_data·(Kconst_mat @ (1/μ)), (K0_row, K0_col)))``。
    """
    donut = DonutFixture()
    empty = donut.empty_driving()
    A = 0.02 * np.random.default_rng(4).normal(size=donut.n_node)

    _, aux = ref_nt.residual(A, donut.rho, donut.field, donut.material, donut.R, empty)
    aux0 = ref_nt.FieldAux(B=aux.B, mu=aux.mu, dgamma=np.zeros_like(aux.dgamma))
    J_lin = ref_nt.jacobian(A, donut.field, aux0, empty)

    K8 = kernel8_assembled_stiffness(donut.ops, aux.mu, donut.n_node)

    diff = abs((J_lin - K8).toarray()).max()
    assert diff <= 1e-13 * abs(K8).max()


# =============================================================================
# 来源 B —— 有限差分校验 Jacobian
# =============================================================================

def test_jacobian_matches_finite_difference(donut_solved):
    """``J·v`` 与中心差分 ``(F(A+εv) − F(A−εv))/(2ε)`` 在**收敛点**上对比。

    用收敛点而不是随机点，是为了让工作点真正落在饱和段 —— 否则饱和项可能
    数值上很小，校验也就没什么力量（见下一条防退化断言）。
    """
    fx = donut_solved
    A = fx.A_conv
    empty = fx.empty_driving()

    rng = np.random.default_rng(11)
    v = rng.normal(size=fx.n_node)

    _, aux = ref_nt.residual(A, fx.rho, fx.field, fx.material, fx.R, empty)
    J = ref_nt.jacobian(A, fx.field, aux, empty)
    Jv = J @ v

    eps = 1e-6 * max(1.0, np.abs(A).max())
    Fp, _ = ref_nt.residual(A + eps * v, fx.rho, fx.field, fx.material, fx.R, empty)
    Fm, _ = ref_nt.residual(A - eps * v, fx.rho, fx.field, fx.material, fx.R, empty)
    fd = (Fp - Fm) / (2.0 * eps)

    scale = max(np.linalg.norm(Jv), np.linalg.norm(fd))
    rel = np.linalg.norm(Jv - fd) / scale
    assert rel < 1e-7, f"Jacobian 与有限差分不符：相对误差 {rel:.3e}"


def test_saturation_term_is_actually_active(donut_solved):
    """防止上面的 FD 校验退化：饱和项对 ``J`` 的贡献必须不可忽略。"""
    fx = donut_solved
    A = fx.A_conv
    empty = fx.empty_driving()

    _, aux = ref_nt.residual(A, fx.rho, fx.field, fx.material, fx.R, empty)
    J_full = ref_nt.jacobian(A, fx.field, aux, empty)
    aux0 = ref_nt.FieldAux(B=aux.B, mu=aux.mu, dgamma=np.zeros_like(aux.dgamma))
    J_lin = ref_nt.jacobian(A, fx.field, aux0, empty)

    sat = np.abs((J_full - J_lin).toarray()).max()
    frac = sat / np.abs(J_full).max()
    assert frac > 1e-3, (
        f"饱和项只占 max|J| 的 {frac:.3e}，FD 校验与闭式解测试都会平凡通过，测试无意义"
    )


# =============================================================================
# 来源 A —— 内核 11：Newton 循环的记账语义（假 callable，完全解析）
# =============================================================================

def _constant_step_fn(n_node, c=1.0, jac_value=1.0):
    """``F = c·1``（常数）、``J = jac_value·I`` ⇒ 每步精确减 ``c/jac_value``。"""
    steps = {"n": 0}

    def step_fn(A):
        steps["n"] += 1
        F = np.full(n_node, c)
        J = sp_sparse.identity(n_node, format="csr") * jac_value
        return F, ("aux", A.copy()), J

    step_fn.steps = steps
    return step_fn


def test_relaxation_switch_is_zero_based():
    """``i_epoch < iter_relaxation`` 数的是**迭代**（0 基），不是更新。

    ``iter_relaxation=3`` ⇒ 第 1~3 轮（``i_epoch`` 0,1,2）全步长、第 4 轮起减半。
    """
    n = 2
    step = _constant_step_fn(n, c=1.0)
    # 阈值取 0 ⇒ 永不判停，跑满 5 轮
    A, _, _, hist = ref_nt.newton_iterations(
        step, None, n_node=n, max_iter=5, solve_abs_err=-1.0,
        iter_relaxation=3, relaxation_ratio=0.5, output=False,
    )
    # 每轮 err = n·c² = 2，从不小于 -1，故 5 轮全走
    assert len(hist) == 5
    assert A[0] == pytest.approx(-(1.0 + 1.0 + 1.0 + 0.5 + 0.5))


def test_residual_norm_criterion_is_squared():
    """判据是 ``err = F·F``（平方范数）。构造 ``‖F‖²`` 跨越阈值两侧的假残差。"""
    n = 3
    c = 0.1
    # ‖F‖² = n·c² = 0.03
    step = _constant_step_fn(n, c=c)

    # 阈值 0.04 > 0.03 ⇒ 第 1 轮就停
    _, _, _, hist = ref_nt.newton_iterations(
        step, None, n_node=n, max_iter=10, solve_abs_err=0.04,
        iter_relaxation=20, relaxation_ratio=0.5, output=False,
    )
    assert len(hist) == 1 and hist[0] == pytest.approx(n * c * c)

    # 阈值 0.02 < 0.03 ⇒ 不会在第 1 轮停
    _, _, _, hist = ref_nt.newton_iterations(
        step, None, n_node=n, max_iter=10, solve_abs_err=0.02,
        iter_relaxation=20, relaxation_ratio=0.5, output=False,
    )
    assert len(hist) > 1


def test_max_iter_zero_returns_initial_guess():
    """``max_iter=0`` ⇒ 循环不进，返回 ``(A0 副本, None, None)``（legacy ``:2388``）。"""
    step = _constant_step_fn(2)
    A0 = np.array([0.4, -0.6])
    A, aux, J, hist = ref_nt.newton_iterations(
        step, A0, n_node=2, max_iter=0, solve_abs_err=1e-3,
        iter_relaxation=20, relaxation_ratio=0.5, output=False,
    )
    assert np.array_equal(A, A0) and aux is None and J is None and hist == []
    assert step.steps["n"] == 0


def test_initial_guess_is_not_modified():
    """``A0`` 不被改写（legacy ``:2386`` 的 ``np.copy``）。"""
    A0 = np.array([0.4, -0.6])
    A0_backup = A0.copy()
    step = _constant_step_fn(2, c=1.0)
    A, _, _, _ = ref_nt.newton_iterations(
        step, A0, n_node=2, max_iter=3, solve_abs_err=-1.0,
        iter_relaxation=20, relaxation_ratio=0.5, output=False,
    )
    assert np.array_equal(A0, A0_backup), "初始猜测被就地改写了"
    assert not np.shares_memory(A, A0)


def test_print_format_uses_one_based_epoch(capsys):
    """打印格式 ``'epoch:%d, error: %g' % (i_epoch + 1, err)``（legacy ``:2394-2395``）。"""
    n = 2
    c = 0.5
    step = _constant_step_fn(n, c=c)
    ref_nt.newton_iterations(
        step, None, n_node=n, max_iter=2, solve_abs_err=-1.0,
        iter_relaxation=20, relaxation_ratio=0.5, output=True,
    )
    out = capsys.readouterr().out
    err = n * c * c
    assert f"epoch:1, error: {err:g}" in out
    assert f"epoch:2, error: {err:g}" in out


def test_linear_problem_converges_in_one_update():
    """线性问题：恰好 2 轮（1 次更新 + 1 次判停），解 == 直接 ``spsolve``。"""
    n = 3
    rng = np.random.default_rng(21)
    M = rng.normal(size=(n, n))
    M = M @ M.T + n * np.eye(n)
    M = sp_sparse.csr_matrix(M)
    b = rng.normal(size=n)

    def step_fn(A):
        return M @ A - b, ("aux", None), M

    A, _, J, hist = ref_nt.newton_iterations(
        step_fn, None, n_node=n, max_iter=10, solve_abs_err=1e-12,
        iter_relaxation=20, relaxation_ratio=0.5, output=False,
    )
    expected = sp_sparse.linalg.spsolve(M, b)
    assert len(hist) == 2
    assert np.allclose(A, expected, rtol=1e-12, atol=1e-14)


def test_returns_converged_round_jacobian(donut_solved):
    """返回的 ``J`` 是**收敛那一轮**的（判停在 ``spsolve`` 之前，legacy ``:2396``）。"""
    fx = donut_solved
    A = fx.A_conv
    _, aux, J, hist = ref_nt.newton_solve(
        None, fx.rho, fx.field, fx.material, fx.R, fx.driving,
        solve_abs_err=1e-3, max_iter=100, iter_relaxation=20,
        relaxation_ratio=0.5, output=False,
    )
    # 收敛后按同一 A 重算，必须逐位相同 ⇒ 证明确实是收敛轮而不是最后一次求解的
    _, aux2 = ref_nt.residual(A, fx.rho, fx.field, fx.material, fx.R, fx.driving)
    J2 = ref_nt.jacobian(A, fx.field, aux2, fx.driving)
    assert np.array_equal(J.data, J2.data)
    assert np.array_equal(J.indices, J2.indices)


def test_newton_converges_on_the_real_fixture(donut_solved):
    """真实夹具上 Newton 收敛，且走了 >1 步（防退化）。"""
    fx = donut_solved
    assert len(fx.history_conv) > 1, "第 1 步就「收敛」⇒ 测试无意义"
    assert fx.history_conv[-1] < 1e-3
    assert fx.history_conv[-1] <= min(fx.history_conv) * (1 + 1e-15) + 1e-300


def test_slack_branch_converges_on_the_real_fixture(donut):
    """真实夹具上把 ``iter_relaxation`` 压到 1，松弛分支真的被走到且仍然收敛。

    默认夹具的 6 步都在 ``i_epoch < 20`` 里，松弛分支走不到 —— 这条测试补上它。
    """
    fx = donut
    # 两侧都用**紧**阈值：宽松阈值（legacy 生产用的 1e-3）下两条路径各自停在
    # 判据的裕度处，差 ~2e-8 是判据本身的分辨率，不是实现差异。
    A_slack, _, _, h_slack = fx.solved(
        iter_relaxation=1, relaxation_ratio=0.5, solve_abs_err=1e-12, max_iter=200
    )
    A_full, _, _, h_full = fx.solved(solve_abs_err=1e-12, max_iter=200)

    assert len(h_slack) > 1 and h_slack[-1] < 1e-12
    assert len(h_slack) > len(h_full), "松弛分支没有生效：轮数与全步长一样"
    scale = max(np.abs(A_slack).max(), np.abs(A_full).max())
    assert np.abs(A_slack - A_full).max() <= 1e-9 * scale


def test_masks_carry_rho_into_the_material_model(donut):
    """ρ 的变化必须经掩码传进材料模型 —— 否则掩码就是失效的。

    取全真空掩码则 ρ 不该有任何影响；本夹具里铁磁单元存在，故 ρ 变了结果就该变。
    若掩码失效（铁磁单元被判成真空），差别会**精确为零**，这条测试就会失败。
    """
    fx = donut
    empty = fx.empty_driving()
    rng = np.random.default_rng(13)
    A = rng.normal(scale=1e-2, size=fx.n_node)

    F1, _ = ref_nt.residual(A, fx.rho, fx.field, fx.material, fx.R, empty)
    F2, _ = ref_nt.residual(A, np.zeros_like(fx.rho), fx.field, fx.material, fx.R, empty)

    assert not np.array_equal(F1, F2), "掩码失效：ρ 的变化没有传进材料模型"


# =============================================================================
# 来源 B —— 照抄 legacy，与 fast 层逐位比对
# =============================================================================
#
# 下面两个函数是 ``MMA10:2222-2257`` 与 ``2380-2406`` 的**照抄**，不复用被测实现。
# ``g`` 提供 legacy 从模块全局读的那几样（``N_node`` / ``Aelem`` / B-H 曲线 / SIMP 投影）。
# 唯一替换：legacy 调模块级的 ``nonlinear_material_topo_opti`` / ``nonlinear_material``，
# 这里传内核 3 的同名函数 —— 它们已各自与 legacy **逐位**比过（`test_material.py`）。


def legacy_field_error_vectorized(_A_node, _rho_SIMP, K_vectorized, R, g):
    """照抄 ``MMA10:2222-2257``。"""
    N_node = g.n_node
    Aelem = g.area
    mu0 = g.mu0

    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    Bx = Bxcoef_mat @ _A_node
    By = Bycoef_mat @ _A_node
    B = np.sqrt(Bx ** 2 + By ** 2).clip(min=1e-20, max=np.inf)

    mu_SIMP, dgamma_SIMP = ref_mat.nonlinear_material_topo_opti(
        B, _rho_SIMP, g.curve, g.simp)
    mu_non_SIMP, dgamma_non_SIMP = ref_mat.nonlinear_material(B, g.curve)

    mu = (is_vaccum * mu0 + is_SIMP * mu_SIMP + is_non_SIMP * mu_non_SIMP
          + is_linear_iron * (2000 * mu0))
    dgamma = (is_vaccum * 0 + is_SIMP * dgamma_SIMP
              + is_non_SIMP * dgamma_non_SIMP + is_linear_iron * 0)

    K_data = K0_data * (Kconst_mat @ (1 / mu))
    K = sp_sparse.coo_matrix((K_data, (K0_row, K0_col)),
                             shape=(N_node, N_node))

    F = K @ _A_node - R
    F = F * (1 - F_is_driven) + (F_mat_driven @ _A_node) * F_is_driven

    g0i_vec = g0i_mat @ _A_node
    const_vec = gconst_mat @ (1 / B / Aelem * dgamma)
    gi_vec = g0i_vec * const_vec

    Jabobi2_data = (gswap1_mat @ g0i_vec) * (gswap2_mat @ gi_vec)
    Jabobi2_mat = sp_sparse.coo_matrix((Jabobi2_data, (Jabobi2_row, Jabobi2_col)),
                                       shape=(N_node, N_node))
    Jacobi = K + Jabobi2_mat

    # 对称 Dirichlet（2026-09-24）：与生产同步 —— 零齐次 Dirichlet 列
    Jacobi = (J_driven @ Jacobi) @ J_col_mask + J_mat_driven

    return F, B, Jacobi


def legacy_solve_field_vectorized(_rho_SIMP, K_vectorized, R, g,
                                  _A0=None, _solve_abs_err=1e-6, max_iter=100,
                                  iter_relaxation=20, relaxation_ratio=0.5,
                                  output=True):
    """照抄 ``MMA10:2380-2406``。"""
    if _A0 is None:
        _A_node = np.zeros((g.n_node,))
    else:
        _A_node = np.copy(_A0)

    _B_elem = None
    _Jacobi = None
    for i_epoch in range(max_iter):
        _F_node, _B_elem, _Jacobi = legacy_field_error_vectorized(
            _A_node, _rho_SIMP, K_vectorized, R, g)

        err = _F_node.dot(_F_node)
        if output:
            print('epoch:%d, error: %g' % (i_epoch + 1, err))
        if err < _solve_abs_err:
            break
        _Jacobi = _Jacobi.tocsr()
        dA = sp_sparse.linalg.spsolve(_Jacobi, _F_node)
        if i_epoch < iter_relaxation:
            _A_node = _A_node - dA
        else:
            # NR法可能出现振荡，需要使用松弛方法抑制振荡
            _A_node = _A_node - dA * relaxation_ratio

    return _A_node, _B_elem, _Jacobi


# =============================================================================
# 来源 C（严格级 B）—— fast 层与 legacy 逐位一致
# =============================================================================

@requires_fast
def test_fast_matches_legacy_role_of_the_shared_blend():
    """`blend_class_values` 在两层里是**同一份实现**（不是两次复制）。"""
    assert fast_nt.blend_class_values is ref_nt.blend_class_values
    assert fast_nt.MaterialMasks is ref_nt.MaterialMasks
    assert fast_nt.FieldMaterial is ref_nt.FieldMaterial


@requires_fast
def test_fast_newton_loop_is_the_same_object():
    """Newton 循环只写一份（``kernels/README.md`` 的「无差异就不建第二份」判据）。"""
    assert fast_nt.newton_iterations is ref_nt.newton_iterations


@requires_fast
def test_fast_matches_legacy_field_error_bitwise(donut):
    """``field_error_vectorized`` 与 legacy 照抄片段**逐位**一致（含稀疏结构）。"""
    fx = donut
    kv = ref_asm.as_legacy_tuple(fx.ops, fx.driving)
    g = SimpleNamespace(n_node=fx.n_node, area=fx.em.area, mu0=MU0,
                        curve=fx.material.curve, simp=fx.material.simp)

    rng = np.random.default_rng(31)
    A = 3e-2 * rng.normal(size=fx.n_node)

    F_leg, B_leg, J_leg = legacy_field_error_vectorized(A, fx.rho, kv, fx.R, g)

    cached = fast_nt.CachedMaterial(fx.material.curve, fx.material.simp,
                                    MU0, MUR_LINEAR)
    state = cached.prepare_density(fx.rho)
    F_f, B_f, J_f = fast_nt.field_error_vectorized(
        A, kv, fx.R, n_node=fx.n_node, area=fx.em.area, cached=cached, state=state
    )

    assert np.array_equal(F_leg, F_f), "残差不逐位"
    assert np.array_equal(B_leg, B_f), "B 不逐位"
    assert J_f.format == J_leg.format
    for attr in ("data", "indices", "indptr"):
        assert np.array_equal(getattr(J_leg, attr), getattr(J_f, attr)), \
            f"Jacobi.{attr} 不逐位"


def test_jacobian_is_symmetric_after_dirichlet_fix(donut):
    """★★ 对称 Dirichlet 修复（2026-09-24）：J 现为（近）对称。

    行列同时清零把 J 的相对不对称度从 0.06%（row-only）压到 ~1e-7（本夹具）
    / 0.000000（真实网格双向配对）。残余 ~1e-7 来自反周期行的 ±1 系数 vs 主行的
    0 耦合，彻底归零需主从消元（见设计文档 §11 判定）。**解不变**（齐次 A_i=0）。
    """
    fx = donut
    kv = ref_asm.as_legacy_tuple(fx.ops, fx.driving)
    cached = fast_nt.CachedMaterial(fx.material.curve, fx.material.simp,
                                    MU0, MUR_LINEAR)
    state = cached.prepare_density(fx.rho)
    A = 3e-2 * np.random.default_rng(31).normal(size=fx.n_node)
    F, B, J = fast_nt.field_error_vectorized(
        A, kv, fx.R, n_node=fx.n_node, area=fx.em.area, cached=cached, state=state)
    d = J.toarray()
    assert np.linalg.norm(d - d.T) <= 1e-6 * np.linalg.norm(d), (
        "J 修复后仍不对称（相对不对称度 "
        f"{np.linalg.norm(d - d.T) / np.linalg.norm(d):.2e}）—— 列投影没落地")


@requires_fast
def test_fast_matches_legacy_solve_bitwise(donut):
    """``solve_field_vectorized`` 与 legacy 照抄片段逐位一致（含迭代轮数）。"""
    fx = donut
    kv = ref_asm.as_legacy_tuple(fx.ops, fx.driving)
    g = SimpleNamespace(n_node=fx.n_node, area=fx.em.area, mu0=MU0,
                        curve=fx.material.curve, simp=fx.material.simp)

    A_leg, B_leg, J_leg = legacy_solve_field_vectorized(
        fx.rho, kv, fx.R, g, _A0=None, _solve_abs_err=1e-3,
        max_iter=100, iter_relaxation=20, relaxation_ratio=0.5, output=False,
    )
    A_f, B_f, J_f = fast_nt.solve_field_vectorized(
        fx.rho, kv, fx.R, None, 1e-3, 100, 20, 0.5, False,
        n_node=fx.n_node, area=fx.em.area, material=fx.material,
    )

    assert np.array_equal(A_leg, A_f), "解不逐位"
    assert np.array_equal(B_leg, B_f), "B 不逐位"
    for attr in ("data", "indices", "indptr"):
        assert np.array_equal(getattr(J_leg, attr), getattr(J_f, attr))


@requires_fast
def test_fast_reuses_the_cached_density_exactly_once(donut):
    """``prepare_density`` 在整次 Newton 求解里只调一次（缓存的前提）。"""
    fx = donut
    kv = ref_asm.as_legacy_tuple(fx.ops, fx.driving)
    fast_nt.solve_field_vectorized(
        fx.rho, kv, fx.R, None, 1e-3, 100, 20, 0.5, False,
        n_node=fx.n_node, area=fx.em.area, material=fx.material,
    )
    # 若把 prepare_density 挪进 Newton 循环，断言会立刻炸（而不是静默变慢）
    # —— 这里再直接确认一次接口：传错形状的 state 会被 _check 抓住（verify 模式）。
    cached = fast_nt.CachedMaterial(fx.material.curve, fx.material.simp,
                                    MU0, MUR_LINEAR, verify=True)
    state = cached.prepare_density(fx.rho)
    with pytest.raises(ValueError):
        cached.nonlinear(np.zeros(fx.n_elem + 1), state)


@requires_fast
def test_ref_fast_parity_residual_and_jacobian(donut_solved):
    """C 级：``reference`` vs ``fast`` 的 ``F`` / ``B`` / ``J``。

    容差 ``rtol=1e-12`` —— 装配走 COO，其内部求和顺序与逐单元 ``+=`` 累加不同
    （与内核 8 同因）。**但稀疏模式必须完全相同**：``spsolve`` 的 LU 主元顺序只取决于
    结构，模式不同会让 Newton 迭代的偏差远超 ULP。
    """
    fx = donut_solved
    A = fx.A_conv

    F_ref, aux = ref_nt.residual(A, fx.rho, fx.field, fx.material, fx.R, fx.driving)
    B_ref = aux.B
    J_ref = ref_nt.jacobian(A, fx.field, aux, fx.driving)

    kv = ref_asm.as_legacy_tuple(fx.ops, fx.driving)
    cached = fast_nt.CachedMaterial(fx.material.curve, fx.material.simp,
                                    MU0, MUR_LINEAR)
    state = cached.prepare_density(fx.rho)
    F_f, B_f, J_f = fast_nt.field_error_vectorized(
        A, kv, fx.R, n_node=fx.n_node, area=fx.em.area, cached=cached, state=state
    )

    # B **不可能**逐位相同：两层是同一个和的不同结合顺序 ——
    #   reference： (bxcoef_e @ T_e) @ A_active   （逐单元的点积）
    #   fast：      装配期就把 bxcoef@T 合成为一行，Bxcoef_mat[e] @ A
    # 两层的 Bxcoef 行本身**逐位相同**，差只来自 dense dot 与 CSR spmv 的求和次序。
    # 度量用**逐单元相对差**（不是「max|B| 处的 ULP 数」——那个指标在小 B 单元上会
    # 严重误导：小 B 处 Σ|各项| ≫ |B|，相消把末位放大几个量级）。
    # 真实网格实测的相消比 8.8e-06、逐单元最大相对差 8.4e-12，见 tools/probe_newton.py。
    # 本玩具网格（无小 B 单元）实测 2.4e-15，阈值留 4 个量级余量。
    b_rel = np.abs(B_ref - B_f) / np.maximum(np.abs(B_ref), 1e-300)
    assert b_rel.max() <= 1e-11, (
        f"B 的逐单元相对差 {b_rel.max():.3e} 超出「求和顺序不同带来的末位」量级"
    )

    # F 的容差**不能**按 |F| 缩放：残差是 K·A − R 的灾难性相消（实测各项 ~3.7e3、
    # |F| ~ 1.1e-3，相消比 3.1e-7），按 |F| 缩放会把正常舍入当成错误。
    # 按**加数**的量级缩放：实测偏差 1.24e-15 × max|K·A|（约 5 ULP）。
    K8 = kernel8_assembled_stiffness(fx.ops, aux.mu, fx.n_node)
    term_scale = max(np.abs(K8 @ A).max(), np.abs(fx.R).max())
    assert np.abs(F_ref - F_f).max() <= 1e-13 * term_scale

    # J 的稀疏模式必须完全相同：spsolve 的 LU 主元顺序只取决于结构，
    # 模式不同会让 Newton 迭代的偏差**远超 ULP**。
    assert np.array_equal(J_ref.indptr, J_f.indptr)
    assert np.array_equal(J_ref.indices, J_f.indices)
    j_scale = np.abs(J_ref.data).max()
    assert np.abs(J_ref.data - J_f.data).max() <= 1e-12 * j_scale


@requires_fast
def test_ref_fast_parity_newton_end_to_end(donut):
    """C 级：两层 Newton 到收敛。**轮数必须相等**。

    有依据的理由：收敛后 ``‖F‖²`` 的底噪约 1e-17，而阈值是 1e-3 —— 裕度 13 个量级，
    ``rtol=1e-12`` 级的末位差不可能把某一轮的 ``err`` 推过阈值。
    """
    fx = donut
    A_ref, _, _, h_ref = fx.solved()

    # fast 的 `solve_field_vectorized` 返回 legacy 的三元组（无历史），所以这里
    # 用两层**共用**的 `newton_iterations` 直接绑定 fast 的 step，把历史取出来比。
    kv = ref_asm.as_legacy_tuple(fx.ops, fx.driving)
    cached = fast_nt.CachedMaterial(fx.material.curve, fx.material.simp,
                                    MU0, MUR_LINEAR)
    state = cached.prepare_density(fx.rho)

    def step(A):
        return fast_nt.field_error_vectorized(
            A, kv, fx.R, n_node=fx.n_node, area=fx.em.area,
            cached=cached, state=state,
        )

    A_f, _B_f, _J_f, h_f = fast_nt.newton_iterations(
        step, None, n_node=fx.n_node, max_iter=100, solve_abs_err=1e-3,
        iter_relaxation=20, relaxation_ratio=0.5, output=False,
    )

    assert len(h_f) == len(h_ref), "两层的迭代轮数不同 —— 说明残差差得不止末位"
    scale = max(np.abs(A_ref).max(), np.abs(A_f).max())
    assert np.abs(A_ref - A_f).max() <= 1e-9 * scale


@requires_fast
def test_fast_layer_is_not_slower_than_reference(donut):
    """性能守卫（宽松阈值）：位置矩阵族的价值在这里 —— 下游不必逐单元重装。

    本批**不引入 `approx/` 层**，所以按 `CLAUDE.md` §4 严格说这条守卫不是必需的；
    加它是为了守住「fast 不能比 reference 还慢」这个下限 —— 否则把位置矩阵路径
    换成别的实现也不会被发现。阈值取得很松（只要不慢就行），避免抖动误报。

    小网格（35 节点）实测：reference 12.3 ms / fast 1.44 ms（8.5×）。
    **真实网格上的数字见 `tools/probe_newton.py`**，不要沿用这里的。
    """
    import timeit

    fx = donut
    kv = ref_asm.as_legacy_tuple(fx.ops, fx.driving)
    cached = fast_nt.CachedMaterial(fx.material.curve, fx.material.simp,
                                    MU0, MUR_LINEAR)
    state = cached.prepare_density(fx.rho)
    A = np.zeros(fx.n_node)

    def ref_step():
        _, aux = ref_nt.residual(A, fx.rho, fx.field, fx.material,
                                 np.zeros(fx.n_node), fx.driving)
        return ref_nt.jacobian(A, fx.field, aux, fx.driving)

    t_ref = min(timeit.repeat(ref_step, number=50, repeat=5)) / 50
    t_fast = min(timeit.repeat(
        lambda: fast_nt.field_error_vectorized(
            A, kv, np.zeros(fx.n_node), n_node=fx.n_node, area=fx.em.area,
            cached=cached, state=state),
        number=50, repeat=5)) / 50

    assert t_fast < t_ref, (
        f"fast 层比 reference 还慢：{t_fast * 1e3:.3f} ms vs {t_ref * 1e3:.3f} ms"
    )


@requires_fast
def test_fast_solve_reports_the_converged_round_jacobian(donut):
    """fast 返回的 ``(B, J)`` 也是**收敛那一轮**的，且 ``_A0`` 不被就地改写。"""
    fx = donut
    kv = ref_asm.as_legacy_tuple(fx.ops, fx.driving)
    A0 = 1e-2 * np.random.default_rng(41).normal(size=fx.n_node)
    A0_backup = A0.copy()

    A1, B1, J1 = fast_nt.solve_field_vectorized(
        fx.rho, kv, fx.R, A0, 1e-3, 100, 20, 0.5, False,
        n_node=fx.n_node, area=fx.em.area, material=fx.material,
    )
    assert np.array_equal(A0, A0_backup), "传入的 A0 被就地改写了"

    # 热启动到的是同一个解 —— 两侧都用**紧**阈值，否则各自停在 1e-3 判据的裕度处，
    # 差 ~1e-9 是判据本身的分辨率，不是实现差异。
    A2, _, _, _ = fx.solved(solve_abs_err=1e-12, max_iter=200)
    A3, _, _ = fast_nt.solve_field_vectorized(
        fx.rho, kv, fx.R, None, 1e-12, 200, 20, 0.5, False,
        n_node=fx.n_node, area=fx.em.area, material=fx.material,
    )
    scale = max(np.abs(A2).max(), np.abs(A3).max())
    assert np.abs(A2 - A3).max() <= 1e-9 * scale
