"""内核 15 测试：冻结视在磁导率法的磁链与磁链转矩。

测试三级来源（`docs/重构设计文档.md` §1.4 与 `CLAUDE.md` §4）：

  来源 A 解析解/解析性质
        - 冻结系统与内核 9/10 在**同一个 B** 上逐位相同（缝的正确性）
        - 冻结求解对源项**线性**（`B ∝ R`），而 `μ` 与源项**无关**（冻结的定义）
        - `Σ_e (Bx²+By²)/μ_e·Δ_e == Aᵀ K(1/μ) A` —— 能量积分与 FEM 二次型的恒等式
        - `Σ_e k_e/μ_e == T_flux == p·parts·[I_d I_q(L_dd−L_qq) + (I_q²−I_d²)L_dq]`
        - **有限差分**（决定性）：扰动 ρ、全非线性重解、走完整条冻结链，比 `T_flux`
  来源 B 与 legacy 一致 —— 照抄 `MMA10:2948-2957` / `:2336-2377` 等片段到本文件比对
  来源 C reference vs fast —— 见 `test_flux.py` 末节（子任务 2）

⚠ 本内核的 F4 **不活跃**：`DTorque_flux_vectorized` 用的是
`for i_elem in range(N_element)`（`:2358` 附近的逐单元循环在装配期已做掉），
`get_R_vector` 也是 `range`（内核 8 已确认逐位可复现）。所以本文件没有 set 顺序守卫。

运行::

    python -m pytest tests/kernels/test_flux.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from motorfem.kernels.reference import assembly as ref_asm
from motorfem.kernels.reference import boundaries as ref_bnd
from motorfem.kernels.reference import element_matrices as ref_em
from motorfem.kernels.reference import flux as ref_flux
from motorfem.kernels.reference import moving_mesh as ref_mm
from motorfem.kernels.reference import newton as ref_nt
from motorfem.kernels.reference import torque as ref_tq
from motorfem.kernels.reference.assembly import TopologyContext
from motorfem.kernels.reference.bh_curve import BHCurve
from motorfem.kernels.reference import material as ref_material
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection

try:
    from motorfem.kernels.fast import flux as fast_flux
    from motorfem.kernels.fast import newton as fast_nt
    from motorfem.kernels.fast.material import CachedMaterial
except ImportError:  # pragma: no cover
    fast_flux = None
    fast_nt = None
    CachedMaterial = None

requires_fast = pytest.mark.skipif(fast_flux is None, reason="fast/ 层尚未落地")

SECTOR = np.pi / 2
COORD_ERR_ABS = 1e-6
R_RINGS = (0.030, 0.040, 0.050, 0.060)
PARTS = 4
POLE_PAIRS = 2
MU0 = 4e-7 * np.pi
ROTOR_POSITION = 0.31
#: 三相电流幅值（安培）。
CURRENT_AMP = 5.0
#: 相**电流密度**幅值 —— legacy 是 ``_I_phase * N_conductor / area_coil``（``:3395``）。
#: ⚠ 这个量级不是装饰：直接传安培数会让 ``‖R‖² < 1e-3``，Newton 第一轮就判停、
#: ``A ≡ 0``、``B`` 全落在 ``clip`` 底限 —— 所有导数测试会平凡通过（本文件踩过）。
J_PHASE = 5.0e6
#: legacy 的 ``_J_D_unit = 1 * N_conductor / area_coil``（``:3398-3399``）——
#: 与 ``J_PHASE`` 的比恰好是 ``1/CURRENT_AMP``（单位电流）。
J_UNIT = J_PHASE / CURRENT_AMP


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
    element_rotor_gap = set([e[0] for e in element_S if e[4] == 1])
    element_stator = set([e[0] for e in element_S if e[4] == 2])
    element_rotor = set([e[0] for e in element_S if e[4] in (0, 1)])
    return xy, element_S, element_rotor_gap, element_stator, element_rotor


def interface_nodes_legacy(xy, r_interface=0.050, coord_err_abs=COORD_ERR_ABS):
    """照抄 ``MMA10:339-340``：半径**平方差**判据。"""
    return set(
        i for i in range(xy.shape[0])
        if abs((xy[i, 0] ** 2 + xy[i, 1] ** 2) - r_interface ** 2) < coord_err_abs
    )


# -----------------------------------------------------------------------------
# dq 变换（legacy ``MMA10:206-216``）—— 本内核把 ``I_d``/``I_q`` 当参数收，
# 所以这两个 helper 留在测试侧。**逐字照抄**，不复用被测实现。
# -----------------------------------------------------------------------------

def abc_to_dq_orthogonal(xa, xb, xc, pos):
    """照抄 ``MMA10:206-209``。"""
    xd = np.sqrt(2 / 3) * (xa * np.cos(pos) + xb * np.cos(pos - 2 / 3 * np.pi)
                           + xc * np.cos(pos - 4 / 3 * np.pi))
    xq = np.sqrt(2 / 3) * -(xa * np.sin(pos) + xb * np.sin(pos - 2 / 3 * np.pi)
                           + xc * np.sin(pos - 4 / 3 * np.pi))
    return xd, xq


def dq_to_abc_orthogonal(xd, xq, pos):
    """照抄 ``MMA10:212-216``。"""
    xa = np.sqrt(2 / 3) * (xd * np.cos(pos) - xq * np.sin(pos))
    xb = np.sqrt(2 / 3) * (xd * np.cos(pos - 2 / 3 * np.pi) - xq * np.sin(pos - 2 / 3 * np.pi))
    xc = np.sqrt(2 / 3) * (xd * np.cos(pos - 4 / 3 * np.pi) - xq * np.sin(pos - 4 / 3 * np.pi))
    return xa, xb, xc


class FluxFixture:
    """同心圆环扇 + 内核 6/7/8 产出 + 一个**非线性**工作点 + 单位 d/q 源。

    **为什么夹具要真解一次非线性场**：冻结法的输入 ``B_frozen`` 就是非线性解的磁密。
    若用零场，两次冻结解都平凡、``k_e ≡ 0``、所有导数测试都会平凡通过。

    三相绕组用「定子单元按 id 排序后三等分」造三个相带 —— 不是真实绕组排布，
    但源项空间形态互不相同，足以让 ``L_dd ≠ L_qq``、``L_dq ≠ 0``。
    """

    def __init__(self, n_ang=7, a_eq=False, r_dirichlet=0.060):
        self.xy0, element_S, self.element_rotor_gap, self.element_stator, \
            self.element_rotor = make_donut_mesh(n_ang=n_ang)
        self.a_eq = a_eq

        pairing0 = ref_bnd.pair_master_slave(self.xy0, parts=PARTS,
                                            coord_err_abs=COORD_ERR_ABS)
        self.layer = ref_mm.insert_moving_mesh_layer(
            self.xy0, np.array([e[1:4] for e in element_S]),
            tuple(interface_nodes_legacy(self.xy0)), self.element_rotor_gap, pairing0,
            element_stator=self.element_stator, element_rotor=self.element_rotor,
        )
        self.virtual = ref_bnd.slave_virtual_elements(self.layer.tri, self.layer.pairing)
        self.n_node, self.n_elem = self.layer.n_node, self.layer.tri.shape[0]
        self.xy = self.layer.xy

        self.ctx = TopologyContext.from_boundaries(
            self.layer.tri, self.virtual, self.layer.pairing,
            self.element_rotor_gap, a_eq=a_eq, n_node=self.n_node,
        )
        self.rot = ref_mm.rotating_boundary(
            self.layer, parts=PARTS, pos=ROTOR_POSITION, a_eq=a_eq)
        self.em = ref_em.electromagnetic_elements(self.layer.xy, self.layer.tri)

        # ---- 驱动行：Dirichlet + 主从 + 动网格插值行 ----
        self.dirichlet = np.array(sorted(
            i for i in range(self.n_node)
            if abs(np.hypot(*self.xy[i]) - r_dirichlet) < COORD_ERR_ABS
        ), dtype=int)
        self.row_replacements = {
            int(n): np.asarray(self.rot.matrix[n, :].todense()).ravel()
            for n in self.layer.rotor_nodes
        }
        self.driving = ref_bnd.driving_rows(
            self.n_node, self.dirichlet, self.layer.pairing.slave_to_master, a_eq,
            self.row_replacements,
        )

        # ---- 材料与密度 ----
        self.material = ref_nt.FieldMaterial(
            curve=BHCurve.from_csv(self._csv()),
            simp=SIMPProjection(SIMPParams()),
        )
        self.element_iron = set(self.element_rotor) | set(self.element_stator)
        # 拓扑区取转子**除气隙层外**的前 12 个单元，与转矩夹具同源
        self.element_topo = set(e for e in self.element_rotor if e < 12)
        all_e = np.arange(self.n_elem)
        is_iron = np.isin(all_e, np.array(sorted(self.element_iron), dtype=int))
        is_topo = np.isin(all_e, np.array(sorted(self.element_topo), dtype=int))
        # ⚠ ρ 取 0.5（SIMP 阈值）而不是 1.0：ρ=1 处 tanh 饱和、``SIMP_dfunc ≈ 0``，
        # ``dγ/dρ ≡ 0`` 会让所有导数测试平凡通过（内核 13/14 踩过这个坑）。
        self.rho = np.where(is_iron, 0.5, 0.0)
        masks = ref_nt.MaterialMasks(
            is_vacuum=(~is_iron).astype(float),
            is_simp=is_topo.astype(float),
            is_non_simp=(is_iron & ~is_topo).astype(float),
            is_linear_iron=np.zeros(self.n_elem),
        )
        self.field = ref_nt.field_geometry(
            self.ctx, self.rot, n_node=self.n_node, k0elem=self.em.K0,
            bxcoef=self.em.Bxcoef, bycoef=self.em.Bycoef, aelem=self.em.area,
            masks=masks,
        )
        self.ops = ref_asm.element_operators(
            self.ctx, self.rot, n_node=self.n_node, k0elem=self.em.K0,
            bxcoef=self.em.Bxcoef, bycoef=self.em.Bycoef,
            element_iron=self.element_iron, element_topo_opti=self.element_topo,
            use_nonlinear_iron=True,
        )

        # ---- 源项：三个相带（定子单元三等分）----
        st = sorted(self.element_stator)
        self.phase_sets = [set(st[i::3]) for i in range(3)]
        self.theta_e = ROTOR_POSITION * POLE_PAIRS

        # 源项用**电流密度**（legacy ``:3395-3397``），磁链用**安培**（``:2952``）
        self.J_abc = [J_PHASE * np.cos(k * 2 * np.pi / 3) for k in range(3)]
        self.I_abc = [CURRENT_AMP * np.cos(k * 2 * np.pi / 3) for k in range(3)]
        self.R = self._source(*self.J_abc)
        # 单位 d/q 电流**密度**的源（legacy ``:3402-3405`` 的 Rd/Rq）
        self.abc_d = dq_to_abc_orthogonal(J_UNIT, 0, self.theta_e)
        self.abc_q = dq_to_abc_orthogonal(0, J_UNIT, self.theta_e)
        self.R_d = self._source(*self.abc_d)
        self.R_q = self._source(*self.abc_q)
        self.I_d, self.I_q = abc_to_dq_orthogonal(*self.I_abc, self.theta_e)

        # ---- 非线性工作点 ----
        self.A, self.aux, self.Jacobi, self.history = ref_nt.newton_solve(
            None, self.rho, self.field, self.material, self.R, self.driving,
            solve_abs_err=1e-3, max_iter=100, iter_relaxation=20,
            relaxation_ratio=0.5, output=False,
        )
        self.B = np.asarray(self.aux.B)

    def _source(self, jA, jB, jC):
        """由三相电流密度造源项 ``R``（内核 8 的 ``get_R_vector``）。"""
        j_of_elem = ref_asm.signed_phase_current(
            self.element_stator, self.phase_sets, [jA, jB, jC])
        return ref_asm.get_R_vector(
            self.ctx, self.rot, n_node=self.n_node, r0elem=self.em.R0,
            current_of_element=j_of_elem)

    def analysis(self, rho=None, A=None):
        """在给定 ``ρ``/``A`` 上走完整条冻结链。"""
        rho = self.rho if rho is None else rho
        A = self.A if A is None else A
        B = ref_nt.flux_density(A, self.field)
        return ref_flux.frozen_flux_analysis(
            B, rho, self.field, self.material, self.R_d, self.R_q,
            self.I_d, self.I_q, self.driving,
            pole_pairs=POLE_PAIRS, parts=PARTS,
        )

    @staticmethod
    def _csv():
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for cand in (root / "磁障设计/pytorch/50CS470.csv",
                     root / "legacy/pytorch/50CS470.csv"):
            if cand.exists():
                return cand
        raise FileNotFoundError("找不到 50CS470.csv")


@pytest.fixture(scope="module")
def fx() -> FluxFixture:
    return FluxFixture()


def legacy_globals(fx):
    """legacy 片段要的模块级全局量替身。"""
    return SimpleNamespace(
        N_node=fx.n_node, N_element=fx.n_elem,
        Aelem=fx.em.area, parts=PARTS, pole_pairs=POLE_PAIRS,
        mu0=MU0, Bxcoef=fx.em.Bxcoef, Bycoef=fx.em.Bycoef,
    )


# =============================================================================
# 来源 A —— 冻结系统与内核 9/10 是同一份数学（缝）
# =============================================================================

def test_frozen_system_equals_kernel9_on_the_same_B(fx):
    """冻结束在**同一个 ``B``** 上与内核 9/10 逐位相同。

    这是「缝切对了」的可执行判据：``frozen_field_system`` 只是把
    ``element_field_material`` 与 ``field_system`` 缝起来，所以取
    ``B = flux_density(A)`` 时它必须与 ``residual``/``jacobian`` **逐位**一致。
    哪天有人在冻结束里改了算式，这条会立刻失败。
    """
    F_frozen, J_frozen, mu = ref_flux.frozen_field_system(
        fx.A, fx.B, fx.rho, fx.field, fx.material, fx.R, fx.driving)
    F_ref, aux = ref_nt.residual(
        fx.A, fx.rho, fx.field, fx.material, fx.R, fx.driving)
    J_ref = ref_nt.jacobian(fx.A, fx.field, aux, fx.driving)

    assert np.array_equal(F_frozen, F_ref)
    assert np.array_equal(np.asarray(mu), np.asarray(aux.mu))
    assert np.array_equal(J_frozen.toarray(), J_ref.toarray())


def test_frozen_field_uses_the_B_it_is_given(fx):
    """``B`` 是**输入**而不是由 ``A`` 算出来的 —— 冻结法的定义。

    把 ``B`` 人为改成常数组，``μ`` 必须跟着变；若哪天有人在内部偷偷用
    ``flux_density(A)``，这条会失败。
    """
    B_alt = np.full(fx.n_elem, 1.5)
    _, _, mu_alt = ref_flux.frozen_field_system(
        fx.A, B_alt, fx.rho, fx.field, fx.material, fx.R, fx.driving)
    _, _, mu = ref_flux.frozen_field_system(
        fx.A, fx.B, fx.rho, fx.field, fx.material, fx.R, fx.driving)
    assert not np.array_equal(mu_alt, mu), "B 被忽略了 —— 冻结束不是在用形参"


# =============================================================================
# 来源 A —— 冻结求解的性质（线性、与源无关的 μ）
# =============================================================================

def test_frozen_solve_is_linear_in_the_source(fx):
    """``B ∝ R`` —— 一次线性求解的解析性质，且 ``μ`` 与源项**无关**。

    冻结法的全部内容就是「``μ`` 冻结 ⇒ 系统线性」。所以

    - ``solve(2R) == 2·solve(R)``（对 ``Bx``、``By`` 各一次）；
    - 两次调用的 ``μ`` **逐位相同**（它只依赖 ``B_frozen`` 与 ``ρ``）。

    第二条同时是模块 docstring 第四节第 1 条的落地：legacy 把 ``μ`` 从 q 解里
    取（覆盖了 d 解的），本实现只算一次 —— 两者必须无差。
    """
    Bx1, By1, mu1 = ref_flux.solve_frozen_field(
        fx.B, fx.rho, fx.field, fx.material, fx.R_d, fx.driving)
    Bx2, By2, mu2 = ref_flux.solve_frozen_field(
        fx.B, fx.rho, fx.field, fx.material, 2.0 * fx.R_d, fx.driving)

    assert np.array_equal(mu1, mu2), "μ 竟然依赖源项 —— 冻结法失效"
    scale = max(np.abs(Bx1).max(), np.abs(By1).max())
    assert np.abs(Bx2 - 2 * Bx1).max() <= 1e-12 * scale
    assert np.abs(By2 - 2 * By1).max() <= 1e-12 * scale
    assert np.any(Bx1 != 0.0), "零场 ⇒ 测试无意义"


def test_d_and_q_solves_share_the_same_frozen_mu(fx):
    """d 与 q 两支共用同一个冻结 ``μ``（这是「电感矩阵」成立的前提）。"""
    _, _, mu_d = ref_flux.solve_frozen_field(
        fx.B, fx.rho, fx.field, fx.material, fx.R_d, fx.driving)
    _, _, mu_q = ref_flux.solve_frozen_field(
        fx.B, fx.rho, fx.field, fx.material, fx.R_q, fx.driving)
    assert np.array_equal(mu_d, mu_q)


# =============================================================================
# 来源 A —— 能量积分 == FEM 二次型（本内核的解析支点）
# =============================================================================

def test_inductance_integral_equals_the_quadratic_form(fx):
    """``Σ_e (Bx²+By²)/μ_e·Δ_e == Aᵀ K(1/μ) A``，逐元素恒等式。

    推导（模块 docstring 第一节）::

        A_eᵀK0_eA_e = ((b·A)² + (c·A)²)/(4Δ) = (Bx² + By²)·Δ

    左边由 ``flux_density_components`` 给出、右边由逐单元二次型给出 —— 两条**不同**
    的计算路径在解析上恒等。这条把「电感积分」挂到了有限元能量上：它是``L`` 有没有
    量纲正确性的唯一可执行判据。

    ⚠ 用**未驱动**的 ``K``（逐单元二次型天然就是未驱动的）：驱动行替换会把约束行的
    系数换成 O(1) 的约束表达式，恒等式在那里不成立。
    """
    Bx, By = ref_flux.flux_density_components(fx.A, fx.field)
    mu = fx.analysis().mu
    lhs = float(np.sum((Bx ** 2 + By ** 2) / mu * fx.em.area))

    rhs = 0.0
    for e in range(fx.n_elem):
        tr = fx.field.transforms[e]
        index = list(tr.index)
        k0_e_active = tr.matrix.T @ (fx.field.k0[e] @ tr.matrix)
        rhs += (1.0 / mu[e]) * (fx.A[index] @ (k0_e_active @ fx.A[index]))

    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert lhs > 0.0


def test_inductance_matrix_is_positive_semidefinite(fx):
    """``L`` 是内积 ``⟨·,·⟩_μ`` 下的 Gram 矩阵 ⇒ ``L_dd·L_qq − L_dq² ≥ 0``。

    Cauchy–Schwarz 的解析后果，与实现无关。它同时挡住「``L_dq`` 算成 ``L_dd``」
    这类装配错误（那种错会让行列式变负）。
    """
    an = fx.analysis()
    det = an.L_dd * an.L_qq - an.L_dq ** 2
    assert an.L_dd > 0.0 and an.L_qq > 0.0
    assert det >= 0.0, f"电感矩阵非半正定：det = {det}"
    # 防退化：d/q 两支必须真的不同，否则 L_dd == L_qq、行列式为 0
    assert abs(an.L_dd - an.L_qq) > 1e-6 * max(an.L_dd, an.L_qq), \
        "L_dd ≈ L_qq ⇒ d/q 源项没有区分度，测试无意义"


# =============================================================================
# 来源 A —— 磁链转矩的三条代数恒等式
# =============================================================================

def test_flux_torque_equals_sum_of_kernel_over_mu(fx):
    """``T_flux == Σ_e k_e/μ_e`` —— 显式项 ``k_e`` 与转矩值的逐位恒等式。"""
    an = fx.analysis()
    k = ref_flux.flux_torque_kernel(
        an.Bx_d, an.By_d, an.Bx_q, an.By_q, fx.I_d, fx.I_q, fx.em.area,
        pole_pairs=POLE_PAIRS, parts=PARTS)

    assert np.sum(k / an.mu) == pytest.approx(an.torque, rel=1e-12)
    assert np.any(k != 0.0), "k_e 全零 ⇒ 测试无意义"


def test_flux_torque_matches_the_lumped_form(fx):
    """``T == p·parts·[I_d I_q(L_dd−L_qq) + (I_q²−I_d²)L_dq]``。

    这是把 2 向量磁链式化成逐单元标量式的那一步（模块 docstring 第三节的钥匙）。
    两边算的是不同的东西（一个是 ``ψ×I``，一个是三个电感积分），所以它是一条
    **独立的**代数判据，而不是同义反复。
    """
    an = fx.analysis()
    want = POLE_PAIRS * PARTS * (
        fx.I_d * fx.I_q * (an.L_dd - an.L_qq)
        + (fx.I_q ** 2 - fx.I_d ** 2) * an.L_dq
    )
    assert an.torque == pytest.approx(want, rel=1e-13)


def test_flux_linkage_is_linear_in_the_currents(fx):
    """``ψ`` 对 ``I`` 线性（``L`` 与电流无关 —— 冻结法下 ``L`` 只依赖工作点）。"""
    an = fx.analysis()
    psi = ref_flux.flux_linkage(an.L_dd, an.L_dq, an.L_qq, fx.I_d, fx.I_q, PARTS)

    assert np.array_equal(psi, an.psi)
    psi2 = ref_flux.flux_linkage(an.L_dd, an.L_dq, an.L_qq,
                                 2 * fx.I_d, 2 * fx.I_q, PARTS)
    assert np.allclose(psi2, 2 * psi, rtol=1e-12, atol=0)
    assert np.any(psi != 0.0)


# =============================================================================
# 来源 B —— 与 legacy 片段逐位（L / ψ / T 那一段纯 numpy，两侧应逐位相同）
# =============================================================================

def legacy_inductance_block(g, Bx_d, By_d, Bx_q, By_q, mu, I_d, I_q):
    """照抄 ``MMA10:2948-2957``。"""
    Ldd = np.sum((Bx_d * Bx_d + By_d * By_d) / mu * g.Aelem)
    Lqq = np.sum((Bx_q * Bx_q + By_q * By_q) / mu * g.Aelem)
    Ldq = np.sum((Bx_d * Bx_q + By_d * By_q) / mu * g.Aelem)

    flux_linkage_energy = np.array([Ldd * I_d + Ldq * I_q,
                                    Ldq * I_d + Lqq * I_q]) * g.parts

    torque_flux_energy = g.pole_pairs * (flux_linkage_energy[0] * I_q
                                         - flux_linkage_energy[1] * I_d)
    return Ldd, Ldq, Lqq, flux_linkage_energy, torque_flux_energy


def test_inductance_flux_and_torque_match_legacy_bitwise(fx):
    """``L``/``ψ``/``T`` 与照抄片段**逐位**一致。"""
    g = legacy_globals(fx)
    an = fx.analysis()
    Ldd, Ldq, Lqq, psi_l, T_l = legacy_inductance_block(
        g, an.Bx_d, an.By_d, an.Bx_q, an.By_q, an.mu, fx.I_d, fx.I_q)

    assert an.L_dd == Ldd
    assert an.L_dq == Ldq
    assert an.L_qq == Lqq
    assert np.array_equal(an.psi, psi_l)
    assert an.torque == T_l
    # ``inductance`` 的返回次序是 (aa, ab, bb)，legacy 的计算次序是 (dd, qq, dq)
    assert ref_flux.inductance(an.Bx_d, an.By_d, an.Bx_q, an.By_q,
                               an.mu, fx.em.area) == (Ldd, Ldq, Lqq)


# =============================================================================
# 来源 A（决定性）—— 磁链转矩的导数 vs 有限差分
# =============================================================================

#: 有限差分用的 Newton 容差。**必须比生产值（``1e-3``）紧得多**：``dT/dρ`` 是
#: 两个 ~1e-5 项（显式项 ``k_e`` 与伴随项）近乎相消后的小量，生产的松容差会把
#: 相消后的信号埋掉 —— 实测松容差下比值散布 0.90~1.04，紧容差下收敛到 0.997~1.006。
FD_TOL = 1e-12


def torque_and_derivative_at(fx, rho_, tol=FD_TOL):
    """在给定 ``ρ`` 上走**完整生产链**：非线性解 → 冻结解 → ``T_flux``。

    返回 ``(T_flux, dTflux/dγ)``。``dT/dγ`` 用的是**该工作点**的 ``Jacobi`` 与 ``B``
    —— 与生产一致（``MMA10:3029-3033``）。
    """
    A, aux, Jacobi, _ = ref_nt.newton_solve(
        None, rho_, fx.field, fx.material, fx.R, fx.driving,
        solve_abs_err=tol, max_iter=200, iter_relaxation=20, relaxation_ratio=0.5,
        output=False,
    )
    B = np.asarray(aux.B)
    an = ref_flux.frozen_flux_analysis(
        B, rho_, fx.field, fx.material, fx.R_d, fx.R_q, fx.I_d, fx.I_q, fx.driving,
        pole_pairs=POLE_PAIRS, parts=PARTS,
    )
    dT_dgamma = ref_flux.flux_torque_derivative(
        A, fx.field, B, rho_, fx.material,
        an.Bx_d, an.By_d, an.Bx_q, an.By_q, fx.I_d, fx.I_q, Jacobi,
        sorted(fx.element_topo),
        pole_pairs=POLE_PAIRS, parts=PARTS,
    )
    return an.torque, dT_dgamma


def _fd_ratios(fx):
    """逐拓扑区单元的 ``fd / (legacy 的 dT_flux/dρ)``。"""
    _, dT_dgamma = torque_and_derivative_at(fx, fx.rho)
    _, aux_base, _, _ = ref_nt.newton_solve(
        None, fx.rho, fx.field, fx.material, fx.R, fx.driving,
        solve_abs_err=FD_TOL, max_iter=200, iter_relaxation=20,
        relaxation_ratio=0.5, output=False,
    )
    nu_drho = ref_tq.torque_derivative_wrt_rho(
        np.ones(fx.n_elem), np.asarray(aux_base.B), fx.rho,
        fx.material.curve, fx.material.simp)

    ratios = []
    for e in sorted(fx.element_topo):
        dr = 1e-4
        rp, rm = fx.rho.copy(), fx.rho.copy()
        rp[e] += dr
        rm[e] -= dr
        tp, _ = torque_and_derivative_at(fx, rp)
        tm, _ = torque_and_derivative_at(fx, rm)
        fd = (tp - tm) / (2 * dr)
        ratios.append(fd / (dT_dgamma[e] * nu_drho[e]))
    return np.array(ratios)


def test_flux_torque_derivative_is_the_true_derivative(fx):
    """★★ 这条钉住**D4 修正**（设计文档 §12，2026-09-23）：``dT_flux/dρ`` == 真值。

    验的是生产契约 ``dT/dρ = dT/dγ · dγ/dρ``（``MMA10:2964``）：伴随给 ``dT/dγ``，
    ``dγ/dρ`` 走内核 3 的 ``nonlinear_material_dnu_drho``（内核 12/13/14 的公共层）。
    有限差分是**外部**判据（扰动 ρ、全非线性重解、走完整条冻结链）。

    修前实测（本夹具，12 个拓扑区单元，``FD_TOL``）::

        fd / (legacy 的 dT_flux/dρ)  ∈  [−1.005, −0.997]，中位 −1.0005

    ⇒ 整体差一个负号（不是缺项）。**D4 判定（2026-09-23）**：
    - legacy 消费者 ``dg_torque = -mean_Tflux_p_drho``（``:3202``/``:3584``）把负号
      翻回来 ⇒ 生产梯度对、函数语义错；
    - 阶段 8 收尾 #1 的 oracle（``probe_flux_oracle.py``）「手写差 4~1429 倍」的结论
      是对错了参考系：它比的 FD 是「**B 固定**、只重解冻结场」，而手写匹配的是
      **全链** FD（扰动 ρ → 全非线性重解 → 冻结分析，``probe_flux.py`` 的取法，
      fd/手写 = −1.0000）。冻结场路径（``A_d/A_q`` 对 ρ 的隐式依赖）在全链里**抵消**，
      oracle 不是优化器要的量。⇒ 手写的量级对，只差符号。
    - 修法：``flux_torque_derivative`` 返回值翻号（reference + fast 同步）。
      ⚠ motorfem 新生产路径的目标梯度用 Arkkio 转矩，``dTflux_p_drho_vec`` 只存不消费
      （dead code）⇒ 无消费者要改；golden L3/L4 的原始 ``dTflux`` 数组翻号 ⇒ 已重标定。

    ⚠ ``ρ`` 取 0.5 而不是 1.0：ρ=1 处 tanh 饱和、``SIMP_dfunc ≈ 0``，``dT/dρ`` 恒为 0。

    本测试同时是**防退化**的：断言显式项 ``k_e`` 与伴随项量级相当，否则有限差分
    无法分辨两项中哪一项出错。
    """
    an = fx.analysis()
    elems = sorted(fx.element_topo)
    dT_dgamma = ref_flux.flux_torque_derivative(
        fx.A, fx.field, fx.B, fx.rho, fx.material,
        an.Bx_d, an.By_d, an.Bx_q, an.By_q, fx.I_d, fx.I_q, fx.Jacobi, elems,
        pole_pairs=POLE_PAIRS, parts=PARTS,
    )
    dgdr = ref_tq.torque_derivative_wrt_rho(
        dT_dgamma, fx.B, fx.rho, fx.material.curve, fx.material.simp)
    assert np.all(dgdr[elems] != 0.0), \
        "dT_flux/dρ 恒为 0 ⇒ 测试无意义（检查 ρ 是否在 SIMP 活跃区）"

    # 防退化：显式项 k_e 与伴随项必须量级相当
    k = ref_flux.flux_torque_kernel(
        an.Bx_d, an.By_d, an.Bx_q, an.By_q, fx.I_d, fx.I_q, fx.em.area,
        pole_pairs=POLE_PAIRS, parts=PARTS)
    adjoint = dT_dgamma + k          # D4 翻号后 dT = −(显式) + 伴随 ⇒ 显式项是 −k
    assert np.abs(k[elems]).max() > 1e-3 * np.abs(dT_dgamma[elems]).max()
    assert np.abs(adjoint[elems]).max() > 1e-3 * np.abs(dT_dgamma[elems]).max()

    ratios = _fd_ratios(fx)

    # (a) **D4 已修**：生产 == 有限差分（比值 ≈ +1）
    assert np.all(ratios > 0.9), (
        f"fd/生产 的比值 {np.round(ratios, 4)} 不再贴近 +1 —— D4 修后仍偏离，"
        f"重新核对设计文档 §12 的 D4 修正清单。"
    )
    # (b) 量级也对（不只符号）：|比值−1| 很小 ⇒ 是纯翻号修正
    worst = float(np.abs(ratios - 1.0).max())
    assert worst < 1e-2, f"D4 修后仍与有限差分差 {worst:.3e} —— 那就不只是符号"



def test_explicit_term_is_not_zeroed_outside_the_topo_region(fx):
    """★ 显式项 ``k_e`` **不做**拓扑区清零（与内核 13/14 的形态不同）。

    ``MMA10:2374-2377`` 只对 ``dTflux_p_dgamma1`` 清零，然后
    ``+ dTflux_p_dgamma2_vectorized``（= ``_k_vectorized``，覆盖**全部**单元）。
    所以 ``dTflux_p_dgamma`` 在拓扑区之外非零。照搬，并用这条把它钉住。

    ⚠ **D4 已修（2026-09-23）**：返回值整体翻号 ⇒ 拓扑区之外的显式项现在是 **``−k_e``**
    （仍是「不清零」的形态，只差一个符号）。
    """
    an = fx.analysis()
    elems = sorted(fx.element_topo)
    d = ref_flux.flux_torque_derivative(
        fx.A, fx.field, fx.B, fx.rho, fx.material,
        an.Bx_d, an.By_d, an.Bx_q, an.By_q, fx.I_d, fx.I_q, fx.Jacobi, elems,
        pole_pairs=POLE_PAIRS, parts=PARTS,
    )
    k = ref_flux.flux_torque_kernel(
        an.Bx_d, an.By_d, an.Bx_q, an.By_q, fx.I_d, fx.I_q, fx.em.area,
        pole_pairs=POLE_PAIRS, parts=PARTS)

    outside = [e for e in range(fx.n_elem) if e not in set(elems)]
    assert not np.any(d[outside] == 0.0), \
        "拓扑区之外全为 0 ⇒ 显式项的「不清零」形态变了"
    assert np.array_equal(d[outside], -k[outside]), \
        "拓扑区之外应当恰好等于显式项 −k_e（D4 翻号后）"


# =============================================================================
# 照抄 legacy（不复用被测实现）
# =============================================================================

class LegacyFluxGlobals(SimpleNamespace):
    """legacy 片段要的模块级全局量 + 材料函数替身。"""


def legacy_globals_full(fx):
    """``legacy_globals`` 的超集：加上材料函数与 22 元组。"""
    g = LegacyFluxGlobals(
        N_node=fx.n_node, N_element=fx.n_elem,
        Aelem=fx.em.area, parts=PARTS, pole_pairs=POLE_PAIRS, mu0=MU0,
        Bxcoef=fx.em.Bxcoef, Bycoef=fx.em.Bycoef,
        curve=fx.material.curve, simp=fx.material.simp,
        nonlinear_material_topo_opti=ref_material.nonlinear_material_topo_opti,
        nonlinear_material=ref_material.nonlinear_material,
        K_vectorized=ref_asm.as_legacy_tuple(fx.ops, fx.driving),
    )
    return g


def legacy_field_error_frozen(g, _A_node, _B_elem_frozen, _rho_SIMP, K_vectorized, R):
    """照抄 ``MMA10:2300-2333``（**含对称 Dirichlet 列投影**，2026-09-24 与生产同步）。"""
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    B = _B_elem_frozen

    mu_SIMP, dgamma_SIMP = g.nonlinear_material_topo_opti(B, _rho_SIMP, g.curve, g.simp)
    mu_non_SIMP, dgamma_non_SIMP = g.nonlinear_material(B, g.curve)

    mu = (is_vaccum * g.mu0 + is_SIMP * mu_SIMP + is_non_SIMP * mu_non_SIMP
          + is_linear_iron * (2000 * g.mu0))
    dgamma = (is_vaccum * 0 + is_SIMP * dgamma_SIMP + is_non_SIMP * dgamma_non_SIMP
              + is_linear_iron * 0)

    K_data = K0_data * (Kconst_mat @ (1 / mu))
    K = sp_sparse.coo_matrix((K_data, (K0_row, K0_col)),
                             shape=(g.N_node, g.N_node))

    F = K @ _A_node - R
    F = F * (1 - F_is_driven) + (F_mat_driven @ _A_node) * F_is_driven

    g0i_vec = g0i_mat @ _A_node
    const_vec = gconst_mat @ (1 / B / g.Aelem * dgamma)
    gi_vec = g0i_vec * const_vec

    Jabobi2_data = (gswap1_mat @ g0i_vec) * (gswap2_mat @ gi_vec)
    Jabobi2_mat = sp_sparse.coo_matrix((Jabobi2_data, (Jabobi2_row, Jabobi2_col)),
                                       shape=(g.N_node, g.N_node))
    Jacobi = K + Jabobi2_mat

    Jacobi = (J_driven @ Jacobi) @ J_col_mask + J_mat_driven

    return F, Jacobi, mu


def legacy_solve_frozen(g, _rho_SIMP, _B_elem_frozen, K_vectorized, R):
    """照抄 ``MMA10:2409-2430``。"""
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    _A_node = np.zeros((g.N_node,))

    _F_node, _Jacobi, _mu = legacy_field_error_frozen(
        g, _A_node, _B_elem_frozen, _rho_SIMP, K_vectorized, R)

    _Jacobi = _Jacobi.tocsr()
    dA = sp_sparse_linalg.spsolve(_Jacobi, _F_node)
    _A_node = _A_node - dA

    _Bx = Bxcoef_mat @ _A_node
    _By = Bycoef_mat @ _A_node

    return _Bx, _By, _mu


def legacy_DTorque_flux_vectorized(g, _A_node_frozen, _rho_SIMP,
                                   _Bx_elem_alpha, _By_elem_alpha,
                                   _Bx_elem_beta, _By_elem_beta,
                                   _I_alpha, _I_beta,
                                   _Jacobi, elements_to_compute, K_vectorized):
    """照抄 ``MMA10:2336-2377``。"""
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    Bx = Bxcoef_mat @ _A_node_frozen
    By = Bycoef_mat @ _A_node_frozen
    B = np.sqrt(Bx ** 2 + By ** 2).clip(min=1e-20, max=np.inf)

    mu_SIMP, dgamma_SIMP = g.nonlinear_material_topo_opti(B, _rho_SIMP, g.curve, g.simp)
    mu_non_SIMP, dgamma_non_SIMP = g.nonlinear_material(B, g.curve)

    dgamma = (is_vaccum * 0 + is_SIMP * dgamma_SIMP + is_non_SIMP * dgamma_non_SIMP
              + is_linear_iron * 0)

    g0i_vec = g0i_mat @ _A_node_frozen

    dK_p_dgamma_A_vectorized = sp_sparse.coo_matrix(
        (g0i_vec, (dK_p_dgamma_A_row, dK_p_dgamma_A_col)),
        shape=(g.N_element, g.N_node))
    _k_vectorized = g.pole_pairs * (
        _I_alpha * _I_beta * (_Bx_elem_alpha ** 2 + _By_elem_alpha ** 2
                              - _Bx_elem_beta ** 2 - _By_elem_beta ** 2)
        + (_I_beta ** 2 - _I_alpha ** 2) * (_Bx_elem_alpha * _Bx_elem_beta
                                            + _By_elem_alpha * _By_elem_beta)
    ) * g.Aelem * g.parts

    coef_torque_flux_vectorized = sp_sparse.coo_matrix(
        (g0i_vec * (gconst_mat @ (_k_vectorized * dgamma / (B * g.Aelem))),
         (np.zeros(len(dK_p_dgamma_A_col)), dK_p_dgamma_A_col)),
        shape=(1, g.N_node)).toarray().flatten()
    dTflux_p_dgamma2_vectorized = _k_vectorized

    _result = -dK_p_dgamma_A_vectorized @ sp_sparse_linalg.spsolve(
        _Jacobi.T, coef_torque_flux_vectorized)

    dTflux_p_dgamma1_vectorized = np.zeros(g.N_element)
    dTflux_p_dgamma1_vectorized[list(elements_to_compute)] = \
        _result[list(elements_to_compute)]

    return dTflux_p_dgamma1_vectorized + dTflux_p_dgamma2_vectorized


# =============================================================================
# 来源 B —— fast 与 legacy 逐位
# =============================================================================

@requires_fast
def test_fast_frozen_residual_matches_legacy_bitwise(fx):
    """``field_error_vectorized_frozen`` 与照抄片段**逐位**一致（含 ``F``/``Jacobi``/``μ``）。"""
    g = legacy_globals_full(fx)
    F_l, J_l, mu_l = legacy_field_error_frozen(
        g, fx.A, fx.B, fx.rho, g.K_vectorized, fx.R)

    cached = CachedMaterial(fx.material.curve, fx.material.simp,
                            fx.material.mu0, fx.material.mur_linear)
    state = cached.prepare_density(fx.rho)
    F_f, J_f, mu_f = fast_flux.field_error_vectorized_frozen(
        fx.A, fx.B, g.K_vectorized, fx.R,
        n_node=fx.n_node, area=fx.em.area, cached=cached, state=state)

    assert np.array_equal(F_l, F_f)
    assert np.array_equal(mu_l, mu_f)
    assert np.array_equal(J_l.toarray(), J_f.toarray())


@requires_fast
def test_fast_frozen_solve_matches_legacy_bitwise(fx):
    """``solve_field_vectorized_frozen`` 与照抄片段**逐位**一致。"""
    g = legacy_globals_full(fx)
    for R in (fx.R_d, fx.R_q):
        Bx_l, By_l, mu_l = legacy_solve_frozen(g, fx.rho, fx.B, g.K_vectorized, R)
        Bx_f, By_f, mu_f = fast_flux.solve_field_vectorized_frozen(
            fx.rho, fx.B, g.K_vectorized, R,
            n_node=fx.n_node, area=fx.em.area, material=fx.material)

        assert np.array_equal(mu_l, mu_f)
        assert np.array_equal(Bx_l, Bx_f)
        assert np.array_equal(By_l, By_f)


@requires_fast
def test_fast_derivative_matches_legacy_bitwise(fx):
    """``flux_torque_derivative`` 与照抄 ``DTorque_flux_vectorized`` **逐位**一致。

    ⚠ **D4 已修（2026-09-23）**：本层在返回值处翻号 ⇒ ``fast == −legacy``（legacy 的
    符号是缺陷本体）。这条仍验证「除 D4 翻号外逐位一致」。
    """
    g = legacy_globals_full(fx)
    an = fx.analysis()
    elems = sorted(fx.element_topo)

    legacy = legacy_DTorque_flux_vectorized(
        g, fx.A, fx.rho, an.Bx_d, an.By_d, an.Bx_q, an.By_q,
        fx.I_d, fx.I_q, fx.Jacobi, elems, g.K_vectorized)

    fast = fast_flux.flux_torque_derivative(
        fx.A, fx.rho, an.Bx_d, an.By_d, an.Bx_q, an.By_q, fx.I_d, fx.I_q,
        fx.Jacobi, elems, g.K_vectorized,
        n_node=fx.n_node, n_element=fx.n_elem, area=fx.em.area,
        material=fx.material, pole_pairs=POLE_PAIRS, parts=PARTS)

    assert np.array_equal(-legacy, fast)


# =============================================================================
# 来源 B' —— 把「fast 与内核 9/10 只差两件事」钉死
# =============================================================================

@requires_fast
def test_fast_frozen_residual_differs_from_kernel9_only_in_B(fx):
    """★ 取 ``B = flux_density(A)`` 时，冻结版与内核 9 的 ``F``/``Jacobi`` **逐位**相同。

    这是模块 docstring 里那条「照搬一份 + 用测试钉死相等」的落地。它比「抽公共体」
    更强：共用代码抓不到自身的 bug，而这条断言会在任一侧改动一个字符时立刻失败。
    """
    g = legacy_globals_full(fx)
    cached = CachedMaterial(fx.material.curve, fx.material.simp,
                            fx.material.mu0, fx.material.mur_linear)
    state = cached.prepare_density(fx.rho)

    F9, B9, J9 = fast_nt.field_error_vectorized(
        fx.A, g.K_vectorized, fx.R,
        n_node=fx.n_node, area=fx.em.area, cached=cached, state=state)
    F_f, J_f, mu_f = fast_flux.field_error_vectorized_frozen(
        fx.A, B9, g.K_vectorized, fx.R,
        n_node=fx.n_node, area=fx.em.area, cached=cached, state=state)

    assert np.array_equal(F9, F_f), "F 不再逐位相同 —— 两份向量化实现已经漂移"
    assert np.array_equal(J9.toarray(), J_f.toarray()), "Jacobi 不再逐位相同"

    # ⚠ ``B9`` 与 ``fx.B``（reference 的 ``flux_density``）**不逐位相同**，属已知且必然：
    # 一个是 ``Bxcoef_mat @ A``（CSR spmv）、一个是逐单元 dense 点积，只差求和顺序。
    # 设计文档 §11⑨ 实测逐单元相对差 ~7e-12（小 B 单元的相消把末位放大）。
    # ⇒ 度量必须按**逐单元相对差**，不能按 ``max|B|`` 处的 ULP 数。
    rel_B = np.abs(B9 - fx.B) / np.abs(fx.B)
    assert rel_B.max() <= 1e-9, f"B 的层间相对差 {rel_B.max():.3e} 超出记录的量级"

    # μ 与内核 9 内部算的那个必须一致（reference 侧的同一个量）
    mu_ref, _ = ref_nt.element_field_material(
        B9, fx.rho, fx.field.masks, fx.material)
    assert np.array_equal(mu_f, mu_ref)


@requires_fast
def test_fast_frozen_uses_the_B_it_is_given(fx):
    """``B`` 是形参（不是内部偷算）—— fast 侧的同一守卫。"""
    g = legacy_globals_full(fx)
    cached = CachedMaterial(fx.material.curve, fx.material.simp,
                            fx.material.mu0, fx.material.mur_linear)
    state = cached.prepare_density(fx.rho)

    _, _, mu_alt = fast_flux.field_error_vectorized_frozen(
        fx.A, np.full(fx.n_elem, 1.5), g.K_vectorized, fx.R,
        n_node=fx.n_node, area=fx.em.area, cached=cached, state=state)
    _, _, mu = fast_flux.field_error_vectorized_frozen(
        fx.A, fx.B, g.K_vectorized, fx.R,
        n_node=fx.n_node, area=fx.em.area, cached=cached, state=state)
    assert not np.array_equal(mu_alt, mu)


# =============================================================================
# 来源 C —— reference vs fast
# =============================================================================

@requires_fast
def test_ref_fast_parity_on_the_frozen_solve(fx):
    """冻结求解：两层的差别只在求和顺序（逐单元循环 vs ``Bxcoef_mat @ A``）。"""
    g = legacy_globals_full(fx)
    for R in (fx.R_d, fx.R_q):
        Bx_r, By_r, mu_r = ref_flux.solve_frozen_field(
            fx.B, fx.rho, fx.field, fx.material, R, fx.driving)
        Bx_f, By_f, mu_f = fast_flux.solve_field_vectorized_frozen(
            fx.rho, fx.B, g.K_vectorized, R,
            n_node=fx.n_node, area=fx.em.area, material=fx.material)

        assert np.array_equal(mu_r, mu_f)
        for a, b in ((Bx_r, Bx_f), (By_r, By_f)):
            scale = max(np.abs(a).max(), np.abs(b).max())
            assert np.abs(a - b).max() <= 1e-12 * scale


@requires_fast
def test_ref_fast_parity_on_the_analysis_and_derivative(fx):
    """整条链与导数：两层一致（``rtol=1e-12``）。"""
    g = legacy_globals_full(fx)
    an_r = fx.analysis()
    an_f = fast_flux.frozen_flux_analysis(
        fx.B, fx.rho, g.K_vectorized, fx.R_d, fx.R_q, fx.I_d, fx.I_q,
        n_node=fx.n_node, area=fx.em.area, material=fx.material,
        pole_pairs=POLE_PAIRS, parts=PARTS)

    # 电感/磁链/转矩走共用实现 ⇒ 逐位
    assert an_r.L_dd == an_f.L_dd
    assert an_r.L_dq == an_f.L_dq
    assert an_r.L_qq == an_f.L_qq
    assert np.array_equal(an_r.psi, an_f.psi)
    assert an_r.torque == an_f.torque

    elems = sorted(fx.element_topo)
    d_r = ref_flux.flux_torque_derivative(
        fx.A, fx.field, fx.B, fx.rho, fx.material,
        an_r.Bx_d, an_r.By_d, an_r.Bx_q, an_r.By_q, fx.I_d, fx.I_q, fx.Jacobi,
        elems, pole_pairs=POLE_PAIRS, parts=PARTS)
    d_f = fast_flux.flux_torque_derivative(
        fx.A, fx.rho, an_f.Bx_d, an_f.By_d, an_f.Bx_q, an_f.By_q,
        fx.I_d, fx.I_q, fx.Jacobi, elems, g.K_vectorized,
        n_node=fx.n_node, n_element=fx.n_elem, area=fx.em.area,
        material=fx.material, pole_pairs=POLE_PAIRS, parts=PARTS)

    scale = max(np.abs(d_r).max(), np.abs(d_f).max())
    assert np.abs(d_r - d_f).max() <= 1e-12 * scale

    # D4（已修 2026-09-23）由 reference 侧那条有限差分测试钉住；fast 值与之逐位相同、
    # 与 legacy 只差一个翻号 ⇒ 缺陷不是某一层的实现错误，而是 legacy 公式的符号。
    assert np.array_equal(
        d_f, -legacy_DTorque_flux_vectorized(
            g, fx.A, fx.rho, an_f.Bx_d, an_f.By_d, an_f.Bx_q, an_f.By_q,
            fx.I_d, fx.I_q, fx.Jacobi, elems, g.K_vectorized))

    # 阶段 8 分解复用：factor=splu(J.T)（CSC）与默认 spsolve(J.T,·) 逐位一致。
    factor = sp_sparse_linalg.splu(fx.Jacobi.T)
    d_f_reuse = fast_flux.flux_torque_derivative(
        fx.A, fx.rho, an_f.Bx_d, an_f.By_d, an_f.Bx_q, an_f.By_q,
        fx.I_d, fx.I_q, fx.Jacobi, elems, g.K_vectorized,
        n_node=fx.n_node, n_element=fx.n_elem, area=fx.em.area,
        material=fx.material, pole_pairs=POLE_PAIRS, parts=PARTS,
        factor=factor)
    assert np.array_equal(d_f, d_f_reuse)
