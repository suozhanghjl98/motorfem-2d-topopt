"""内核 17 测试：应力与应变能导数（von Mises / AL 罚项）。

测试三级来源（`docs/重构设计文档.md` §1.4 与 `CLAUDE.md` §4）：

  来源 A 解析解/解析性质 —— von Mises 的三种标准情形（单轴/等双轴/纯剪）、
         单单元手算的 `Λ`/`g`/`h`/`P`/`λ_new`、`ΣP/N` 与 `Uelastic` 的定义式、
         `g` 的跨族耦合、六支系数各自的定义式
  来源 B 与 legacy 一致 —— 照抄 `MMA10:2437-2601` 整段比对
  来源 C reference vs fast —— 见本文件末节
  来源 A（决定性）—— **有限差分**：`dU_p_drho` 与 `dALStress_p_drho`
         对「扰动 ρ → 重解 u → 重算 U / P」的中心差分

⚠ 已知偏差会影响这里的判据（**D5/D6 已修，D7/D8 仍登记**）：

- **§11⑩**：`dkE` 来自内核 2 的 `SIMPProjection.derivative`，与 legacy 的 `SIMP_dfunc`
  差 1~2 ULP（代数等价改写）⇒ 涉及 `dkE` 的比较只能用 `rtol`；
- **D5**（内核 16）：**已修（2026-09-23）**—— `dRHS` 的载荷项符号改为 `+ dR`
  （§12 修正清单第 1 条）；
- **D6**（本内核）：**已修（2026-09-23）**—— `dU_p_drho` 落地包络闭式（免伴随），
  `dU_p_du` 恢复为纯 `∂U/∂u`。本文件的有限差分判定随之从「钉住缺陷」翻转为
  「钉住修正」；
- **D7**（本内核，**已修 2026-09-23**）：AL 乘子更新从 `min(0, λ+μh)`（恒为 0 ⇒
  纯罚函数）改为 **`max(0, λ+μh)`**（真增广拉格朗日，λ ≥ 0 携带历史）。推导见 §12；
- **D8**（本内核，**已修 2026-09-23**）：`dALStress_p_drho` 与有限差分差 **~1e11 倍**
  的缺陷已修正（`d_sigma_p_du` 补 `k_E/(2σ̄)` + 显式项补 `k1·∂σ̄/∂k_E·k_E'`），
  本文件的有限差分判定已从「钉住缺陷」翻转为「钉住修正」。

运行::

    python -m pytest tests/kernels/test_mech_stress.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from motorfem.kernels.reference import boundaries as ref_bnd
from motorfem.kernels.reference import mech_element_matrices as ref_mech5
from motorfem.kernels.reference import mech_solver as ref_solver
from motorfem.kernels.reference import mech_stress as ref_stress
from motorfem.kernels.reference.simp import SIMPProjection

try:
    from motorfem.kernels.fast import mech_stress as fast_stress
except ImportError:  # pragma: no cover
    fast_stress = None

requires_fast = pytest.mark.skipif(fast_stress is None, reason="fast/ 层尚未落地")

PARTS = 4
SECTOR = np.pi / 2
COORD_ERR_ABS = 1e-6
R_RINGS = (0.030, 0.040, 0.050, 0.060)
MAT = ref_mech5.MechMaterial(E=200e9, nu=0.30, rho=7850.0)
ANGULAR_SPEED = 300.0
FLOAD = 1.0e7
MU = 2.0e6
#: 一个负的 ``λ``（生产中它恒为 0，见 D7）—— 用来验 ``h = max(g, −λ/μ)``
#: 的下界那条分支。取这个量级使 ``−λ/μ = 1.5``，与 ``g`` 的量级可比。
NEG_LAMBDA = -3.0e6


def make_sector_mesh(n_ang=7, radii=R_RINGS):
    """四层同心圆环扇 + 全部边界线元（与 ``test_mech_solver`` 同构，此处自包含）。"""
    angles = np.linspace(0.0, SECTOR, n_ang)
    node = np.array([[r * np.cos(t), r * np.sin(t)] for r in radii for t in angles],
                    dtype=float)
    tri = []
    for k in range(len(radii) - 1):
        for l in range(n_ang - 1):
            a, b = k * n_ang + l, (k + 1) * n_ang + l
            c, d = (k + 1) * n_ang + l + 1, k * n_ang + l + 1
            tri += [[a, b, c], [a, c, d]]
    tri = np.array(tri, dtype=int)

    lines, loaded = [], []
    for k in (0, len(radii) - 1):
        for l in range(n_ang - 1):
            lines.append((k * n_ang + l, k * n_ang + l + 1))
            if k == len(radii) - 1:
                loaded.append(len(lines) - 1)
    for l in (0, n_ang - 1):
        for k in range(len(radii) - 1):
            lines.append((k * n_ang + l, (k + 1) * n_ang + l))
    return node, tri, np.array(lines, dtype=int), loaded


def production_flags(max_rel: float):
    """生产配置（rel3 + type2_stress），只把 ``max_rel_stress_constraint`` 换掉。

    ⚠ 生产值是 ``50e7 * 0.8 = 4e8``，而夹具的 ``σ̄/k_E^p`` 最大只有 2.4e8
    ⇒ **约束处处满足、活跃分支根本不出现**。所以另建一个把上限收紧到 1e8 的配置，
    让「活跃/不活跃」两支都出现（否则 AL 的那半条分支完全没被测到）。
    """
    base = ref_stress.PRODUCTION_STRESS_FLAGS
    return ref_stress.StressConstraintFlags(
        use_rel_stress_constraint=base.use_rel_stress_constraint,
        use_type1_stress=base.use_type1_stress,
        use_type2_stress=base.use_type2_stress,
        use_type1_rel_stress=base.use_type1_rel_stress,
        use_type2_rel_stress=base.use_type2_rel_stress,
        use_type3_rel_stress=base.use_type3_rel_stress,
        use_type4_rel_stress=base.use_type4_rel_stress,
        max_stress_constraint=base.max_stress_constraint,
        max_rel_stress_constraint=max_rel,
        rel_stress_p_exp=base.rel_stress_p_exp,
    )


class StressFixture:
    """扇区网格 + 内核 16 的装配/求解 + AL 的 ``(λ, μ)``。"""

    def __init__(self, n_ang=7, rho_value=0.5, flags=None, lambda0=0.0):
        self.node, self.tri, self.line_tri, self.loaded_lines = make_sector_mesh(n_ang)
        self.pairing = ref_bnd.pair_master_slave(
            self.node, parts=PARTS, coord_err_abs=COORD_ERR_ABS)
        self.constrained = sorted(
            i for i in range(self.node.shape[0])
            if abs(np.hypot(*self.node[i]) - R_RINGS[0]) < COORD_ERR_ABS)

        self.mech = ref_mech5.mech_element_matrices(self.node, self.tri, MAT)
        self.body_fx, self.body_fy = ref_mech5.mech_body_force(
            self.node, self.tri, MAT, angular_speed=ANGULAR_SPEED,
            rot_center=(0.0, 0.0))
        self.line_loads = ref_mech5.mech_line_loads(
            self.node, self.line_tri, self.loaded_lines, Fload=FLOAD)

        self.n_elem = self.tri.shape[0]
        self.rho = np.full(self.n_elem, rho_value)
        self.flags = flags or ref_stress.PRODUCTION_STRESS_FLAGS

        self.ops = ref_solver.mech_operators(
            self.node, self.tri,
            k_elem=self.mech.K, l_coef=self.mech.L, d_mat=self.mech.D,
            r0_elem=self.mech.R0, r2_elem=self.line_loads.R0,
            body_fx=self.body_fx, body_fy=self.body_fy,
            line_tri=self.line_tri, loaded_lines=self.loaded_lines,
            boundary_fx=self.line_loads.fx, boundary_fy=self.line_loads.fy,
            slave_to_master=self.pairing.slave_to_master,
            constrained_nodes=self.constrained, parts=PARTS,
        )
        #: ⚠ 默认 **0** —— 那正是 legacy 的初值（``:3840`` 的 ``np.zeros``），
        #: 也是 D7（``λ ≡ 0``）的归纳起点。要验「``h = −λ/μ`` 那条下界分支」
        #: 就传一个负值（``lambda0=NEG_LAMBDA``）。
        self.al_lambda = np.full(self.n_elem, lambda0)
        self.al_mu = MU

    #: ``∂R/∂ρ_e``（dk_rho=1 ⇒ 恰是 ``r0_data``；D6 修正落地后闭式要它）
    def dR(self):
        from scipy import sparse as sp
        return sp.coo_matrix(
            (self.ops.r0_data * 1.0, (self.ops.dr_p_dth_row, self.ops.dr_p_dth_col)),
            shape=(self.ops.n_dof, self.n_elem),
        ).tocsr()

    # ---- 一次「扰动 ρ → 重解 u → 重算」的完整链 ----
    def chain(self, rho_):
        u = ref_solver.solve_mech_field(rho_, self.ops)
        K, dRHS = ref_solver.mech_sensitivity(rho_, u, self.ops)
        res = ref_stress.mech_stress_and_energy(
            u, K, dRHS, rho_, self.al_lambda, self.al_mu,
            tri=self.tri, l_coef=self.mech.L, d_mat=self.mech.D,
            area=self.mech.area, dR_p_dth=self.dR(), flags=self.flags,
        )
        return u, K, dRHS, res

    def at_production(self):
        u, K, dRHS, res = self.chain(self.rho)
        self.u, self.K, self.dRHS = u, K, dRHS
        return u, K, dRHS, res


#: 生产上限（约束处处满足 ⇒ AL 的不活跃分支）
@pytest.fixture(scope="module")
def fx() -> StressFixture:
    return StressFixture()


#: 收紧上限（部分单元违反约束 ⇒ 两支都出现）
@pytest.fixture(scope="module")
def fx_active() -> StressFixture:
    return StressFixture(flags=production_flags(1e8))


# =============================================================================
# 来源 A —— von Mises 的三种标准情形
# =============================================================================

def test_von_mises_standard_cases():
    """单轴 / 等双轴 / 纯剪 —— 教科书值，独立于实现。

    ``σ̄ = √(3σ_xy² + σ_x² + σ_y² − σ_x σ_y)`` 在三种情形下分别退化为
    ``|σ|``、``|σ|``、``√3|τ|``。任何一处 ``σ_x``/``σ_y``/``σ_xy`` 抄错都会破坏它们。
    """
    s = 3.7e6
    assert ref_stress.von_mises(np.array([s, 0.0, 0.0])) == pytest.approx(s, rel=1e-15)
    assert ref_stress.von_mises(np.array([0.0, -s, 0.0])) == pytest.approx(s, rel=1e-15)
    assert ref_stress.von_mises(np.array([s, s, 0.0])) == pytest.approx(s, rel=1e-15)
    assert ref_stress.von_mises(np.array([0.0, 0.0, s])) == pytest.approx(
        np.sqrt(3) * s, rel=1e-15)
    # 纯剪下的等效应力与坐标无关
    assert ref_stress.von_mises(np.array([0.0, 0.0, -s])) == pytest.approx(
        np.sqrt(3) * s, rel=1e-15)


def test_element_stress_matches_direct_assembly(fx):
    """``σ_e = kE_e·D·L_e·u_e`` —— 逐单元与一次装配等价。

    左边逐单元算，右边用 ``L``/``D`` 的数组形式一次算完。两条路径独立。
    """
    u = ref_solver.solve_mech_field(fx.rho, fx.ops)
    k_e = SIMPProjection(ref_solver.MECH_SIMP_PARAMS).project(fx.rho)

    sigma = ref_stress.element_stress(u, fx.tri, fx.mech.L, fx.mech.D, k_e)

    dof = np.empty((fx.n_elem, 6), dtype=np.int64)
    dof[:, 0::2] = 2 * fx.tri
    dof[:, 1::2] = 2 * fx.tri + 1
    eps = np.einsum("nij,nj->ni", fx.mech.L, u[dof])
    want = k_e[:, None] * (fx.mech.D @ eps.T).T
    assert np.abs(sigma - want).max() <= 1e-12 * np.abs(want).max()

    # 防退化：应变不能为零
    assert np.abs(sigma).max() > 0.0


# =============================================================================
# 来源 A —— 单单元手算的 AL 罚项
# =============================================================================

def test_al_terms_on_a_single_element():
    """单单元手算：``Λ`` / ``g`` / ``h`` / ``P`` / ``λ_new``（生产配置 = rel3）。

    用一个只有 1 个自由单元的退化配置直接手算 —— 不走装配。
    取 ``λ = −3e6``、``μ = 2e6`` ⇒ ``−λ/μ = 1.5``，所以 ``h = max(g, 1.5)``；
    下面选两组 ``σ̄`` 分别落到 ``g > 1.5``（活跃）与 ``g < 1.5``（不活跃）两支上。
    """
    flags = ref_stress.PRODUCTION_STRESS_FLAGS
    k_e, p_exp = 0.7, flags.rel_stress_p_exp

    for vm, expect_active in ((3.0e9, True), (1.0e6, False)):
        lam = vm / k_e ** p_exp / flags.max_rel_stress_constraint - 1
        g = lam ** 3 + lam                      # rel3 ⇒ 不带 kE 因子
        h = max(g, -NEG_LAMBDA / MU)
        pi = NEG_LAMBDA * h + MU / 2 * h ** 2
        lam_new = min(0, NEG_LAMBDA + MU * h)

        assert (g * MU + NEG_LAMBDA > 0) is expect_active, "手算的活跃判据与预期不符"
        # 罚项的两支：活跃时 ``h = g``；不活跃时 ``h = −λ/μ`` ⇒ ``P = −λ²/(2μ)``
        if expect_active:
            assert h == pytest.approx(g, rel=1e-15)
        else:
            assert h == pytest.approx(1.5, rel=1e-15)
            assert pi == pytest.approx(-NEG_LAMBDA ** 2 / (2 * MU), rel=1e-12)
        assert lam_new < 0.0 or lam_new == 0.0

    # ``g`` 的跨族耦合：只开 rel1（且 use_type2_stress=False）时 g **带** kE 因子
    rel1_only = ref_stress.StressConstraintFlags(
        use_rel_stress_constraint=True, use_type1_stress=False,
        use_type2_stress=False, use_type1_rel_stress=True,
        use_type2_rel_stress=False, use_type3_rel_stress=False,
        use_type4_rel_stress=False, max_stress_constraint=50e7,
        max_rel_stress_constraint=4e8, rel_stress_p_exp=0.5,
    )
    lam1 = 0.25
    assert ref_stress._constraint_function(lam1, k_e, rel1_only) == pytest.approx(
        k_e * (lam1 ** 3 + lam1), rel=1e-15)
    assert ref_stress._constraint_function(lam1, k_e, flags) == pytest.approx(
        lam1 ** 3 + lam1, rel=1e-15)


def test_flags_reject_multiple_branches():
    """``flags`` 必须**恰好**选一支 —— legacy 的 if/elif 链多开一个会静默走第一个。"""
    with pytest.raises(ValueError, match="恰好选一支"):
        ref_stress.StressConstraintFlags(
            use_rel_stress_constraint=True, use_type1_stress=False,
            use_type2_stress=True, use_type1_rel_stress=True,
            use_type2_rel_stress=False, use_type3_rel_stress=True,
            use_type4_rel_stress=False, max_stress_constraint=50e7,
            max_rel_stress_constraint=4e8, rel_stress_p_exp=0.5,
        )


# =============================================================================
# 来源 A —— 定义式（全场）
# =============================================================================

def test_stress_loss_is_the_normalised_sum_of_penalties(fx_active):
    """``Stress_loss_AL == Σ_e P_e / N``（定义式，用返回的 ``σ̄``/``λ`` 反算 P）。

    用**收紧上限**的夹具（``fx_active``）：生产上限下约束处处满足、活跃分支不出现。
    ⚠ 于是 `kE**p_exp` 不做乘法 -- `P_e` 只依赖 `Λ`（由 `σ̄` 显式算出）。
    这条同时把「两支都出现」变成断言（防退化）。
    """
    u, K, dRHS, res = fx_active.at_production()
    flags = fx_active.flags
    k_e_all = SIMPProjection(ref_solver.MECH_SIMP_PARAMS).project(fx_active.rho)

    total = 0.0
    active = 0
    for e in range(fx_active.n_elem):
        lam = (res.von_mises[e] / k_e_all[e] ** flags.rel_stress_p_exp
               / flags.max_rel_stress_constraint - 1)
        g = lam ** 3 + lam
        h = max(g, -fx_active.al_lambda[e] / MU)
        total += (fx_active.al_lambda[e] * h + MU / 2 * h ** 2) / fx_active.n_elem
        active += bool(g * MU + fx_active.al_lambda[e] > 0)

    assert res.stress_loss_al == pytest.approx(total, rel=1e-12)
    assert res.von_mises_max == pytest.approx(res.von_mises.max(), rel=1e-15)
    assert 0 < active < fx_active.n_elem, (
        f"活跃单元数 {active}/{fx_active.n_elem} —— 必须**两支都出现**，"
        f"否则这条测试覆盖不全（换 max_rel_stress_constraint）"
    )


def test_al_loss_uses_the_active_or_inactive_branch(fx_active):
    """``AL_loss = √(Σ [活跃: g², 不活跃: λ²])``（``:2570-2572`` 的两支）。"""
    u, K, dRHS, res = fx_active.at_production()
    flags = fx_active.flags
    k_e_all = SIMPProjection(ref_solver.MECH_SIMP_PARAMS).project(fx_active.rho)

    total = 0.0
    for e in range(fx_active.n_elem):
        lam = (res.von_mises[e] / k_e_all[e] ** flags.rel_stress_p_exp
               / flags.max_rel_stress_constraint - 1)
        g = lam ** 3 + lam
        total += g ** 2 if g * MU + fx_active.al_lambda[e] > 0 else fx_active.al_lambda[e] ** 2
    assert res.al_loss == pytest.approx(np.sqrt(total), rel=1e-12)


def test_elastic_strain_energy_is_the_element_sum(fx):
    """``Uelastic == Σ_e Δ_e·(ε_e·σ_e)/2``（定义式）。

    注意**不是** ``½uᵀKu``：``u`` 含从属自由度，而这里是逐单元按自己的连接表算的。
    两者在解上相等（约束行保证），但这条只验定义式。
    """
    u, K, dRHS, res = fx.at_production()
    flags = ref_stress.PRODUCTION_STRESS_FLAGS
    k_e_all = SIMPProjection(ref_solver.MECH_SIMP_PARAMS).project(fx.rho)

    dof = np.empty((fx.n_elem, 6), dtype=np.int64)
    dof[:, 0::2] = 2 * fx.tri
    dof[:, 1::2] = 2 * fx.tri + 1
    eps = np.einsum("nij,nj->ni", fx.mech.L, u[dof])
    sigma = k_e_all[:, None] * (fx.mech.D @ eps.T).T
    want = float(np.sum(fx.mech.area * np.einsum("ni,ni->n", eps, sigma) / 2))

    assert res.elastic_strain_energy == pytest.approx(want, rel=1e-12)
    assert res.elastic_strain_energy > 0.0
    assert flags.rel_stress_p_exp == 0.5        # 生产配置的 p_exp（下面几条依赖它）

    # 二次型一致性：½uᵀKu 与逐单元能量在解上相等（1 ULP 容差）
    K_free = sp_sparse.csr_matrix(
        (fx.ops.k0_data * (fx.ops.kconst_map
                           @ ref_solver.mech_density_factors(fx.rho)[0]),
         (fx.ops.k0_row, fx.ops.k0_col)),
        shape=(fx.ops.n_dof, fx.ops.n_dof))
    half_uku = 0.5 * float(u @ (K_free @ u))
    assert abs(half_uku - res.elastic_strain_energy) <= 1e-12 * abs(half_uku)


def test_al_lambda_update_accumulates_for_active_elements(fx_active):
    """★★ 这条钉住**D7 修正**（设计文档 §12，2026-09-23）：``λ_new = max(0, λ+μh)``。

    修前的缺陷：legacy ``:2575`` 是 ``min(0, λ + μh)`` —— 而 ``h = max(g, −λ/μ)``
    恒有 ``λ + μh ≥ 0``（两种情形都可直接验）::

        h = g       ⇒ g ≥ −λ/μ ⇒ λ + μh ≥ 0
        h = −λ/μ    ⇒ λ + μh = 0

    ``⇒ min(0, λ + μh) ≡ 0`` ⇒ ``λ ≡ 0`` ⇒ **增广拉格朗日退化为纯二次罚函数**
    （乘子不携带历史信息，``AL_mu`` 的增长成了唯一的约束力来源）。

    **D7 推导（2026-09-23）**：正确更新是 ``λ_new = max(0, λ + μh)``（即
    ``max(0, λ + μg)`` —— 因 ``h = max(g, −λ/μ) ⇒ λ+μh = max(λ+μg, 0)``）。这是
    光滑 AL ``P = λh + μ/2·h²`` 的 method-of-multipliers 更新：活跃（``g > −λ/μ``）
    时 ``λ_new = λ + μg`` 驱动乘子向 KKT 乘子收敛；不活跃时 ``λ_new = 0``（KKT
    互补性）。⚠ 理论笔记 :289 写的是同一个式子（文档与代码一致）—— 修的是公式
    本身（min → max 的投影方向）。**λ ≥ 0**（不等式 g ≤ 0 的对偶），不再是 ≤ 0。

    本条断言三件事：
    (a) ``λ_new ≥ 0`` 恒成立；
    (b) 活跃单元（``g·μ + λ > 0``）上 ``λ_new == λ + μg``（``λ=0`` 初值下 == ``μg``）；
    (c) 不活跃单元上 ``λ_new == 0``（KKT 互补性）；
    (d) 防退化：确有活跃单元（否则看不出「累加」）。
    """
    u, K, dRHS, res = fx_active.at_production()
    assert np.all(res.al_lambda_new >= 0.0), "D7 修后 λ_new 出现负值 —— max 投影没落地"
    flags = fx_active.flags
    k_e_all = SIMPProjection(ref_solver.MECH_SIMP_PARAMS).project(fx_active.rho)
    active = inactive = 0
    for e in range(fx_active.n_elem):
        lam = (res.von_mises[e] / k_e_all[e] ** flags.rel_stress_p_exp
               / flags.max_rel_stress_constraint - 1)
        g = lam ** 3 + lam
        if g * MU + fx_active.al_lambda[e] > 0:      # 活跃（λ 初值 = 0）
            assert res.al_lambda_new[e] == pytest.approx(MU * g, rel=1e-12), (
                f"单元 {e}：活跃单元 λ_new = {res.al_lambda_new[e]} 应为 μg = {MU*g}")
            active += 1
        else:
            assert res.al_lambda_new[e] == 0.0, (
                f"单元 {e}：不活跃单元 λ_new = {res.al_lambda_new[e]} 应恒为 0")
            inactive += 1
    assert active > 0, "没有活跃单元 ⇒ 看不出「累加」这件事"
    assert inactive > 0, "全是活跃单元 ⇒ 只覆盖了一支"
    print(f"  (D7 已修) 活跃 {active} / 不活跃 {inactive} 单元；"
          f"λ_new ∈ [0, {np.abs(res.al_lambda_new).max():.3e}]")


# =============================================================================
# 来源 B —— 与 legacy 逐位
# =============================================================================

def test_full_result_matches_legacy(fx):
    """照抄 ``MMA10:2437-2601`` 整段，逐项比对（**D5/D6 已修**）。

    不涉及导数的量（``σ``/``σ̄``/``Λ``/``g``/``Uelastic``/``λ_new``/``AL_loss``）
    与 legacy 逐位或近似一致 —— 它们不受 D5/D6 影响。

    ⚠ **两个梯度（``dU_p_drho``、``dALStress_p_drho``）不再与 legacy 一致**：
    D6 修正后 ``dU_p_drho`` 落地包络闭式（免伴随）、D5 修正后 ``dALStress_p_drho``
    消掉 ``2λᵀdR`` —— 两者都是**有意的数值行为变化**（§12），本测试不比对它们
    （它们的正确性由本文件的有限差分测试钉住）。
    """
    u, K, dRHS, res = fx.at_production()
    legacy = legacy_compute_mechDiff_AL(
        u, K, dRHS, fx.rho, fx.al_lambda, MU,
        tri=fx.tri, l_coef=fx.mech.L, d_mat=fx.mech.D, area=fx.mech.area,
    )

    assert np.array_equal(res.sigma, legacy['sigma'])
    assert np.array_equal(res.von_mises, legacy['von_mises'])
    assert res.von_mises_max == legacy['von_mises_max']
    assert res.rel_stress_max == legacy['rel_stress_max']
    assert np.array_equal(res.al_lambda_new, legacy['al_lambda_new'])
    assert res.elastic_strain_energy == pytest.approx(
        legacy['uelastic'], rel=1e-15)
    # ``Stress_loss_AL`` 含 ``g``（无 ``dkE``）⇒ 逐位
    assert res.stress_loss_al == legacy['stress_loss_al']
    assert res.al_loss == legacy['al_loss']


def legacy_compute_mechDiff_AL(_u_node, _Kmech, _dRHSmech_p_dth, _rho_SIMP,
                               AL_lambda, AL_mu, *, tri, l_coef, d_mat, area):
    """照抄 ``MMA10:2437-2601``（生产配置：rel3 + type2_stress）。

    ⚠ ``mech_topo_opti``/``dmech_topo_opti`` 用**照抄的公式**（不是内核 2），
    因为这一条测试的目的就是与 legacy 对齐 —— 导数那一支正是 §11⑩ 的差异所在。
    """
    from motorfem.kernels.reference import mech_solver as S

    beta, thb, p_exp_simp, mn = 8, 0.65, 3, 1e-4
    rel_p = 1 / 2
    max_stress = 50e7
    max_rel = max_stress * 0.8

    def legacy_topo(rho_):
        rho_1 = (np.tanh(beta * (rho_ - thb)) + np.tanh(beta * thb)) / (
            np.tanh(beta * (1 - thb)) + np.tanh(beta * thb))
        return mn + (1 - mn) * rho_1 ** p_exp_simp, rho_

    def legacy_dtopo(rho_):
        rho_1 = (np.tanh(beta * (rho_ - thb)) + np.tanh(beta * thb)) / (
            np.tanh(beta * (1 - thb)) + np.tanh(beta * thb))
        drho_1 = (beta * np.cosh(beta * thb) * np.cosh(beta - beta * thb)
                  / np.sinh(beta) / np.cosh(beta * (rho_ - thb)) ** 2)
        dkE = p_exp_simp * (1 - mn) * rho_1 ** (p_exp_simp - 1) * drho_1
        return dkE, np.ones_like(np.asarray(rho_, dtype=float))

    n_elem = tri.shape[0]
    n_dof = _u_node.shape[0]
    vm = np.zeros(n_elem)
    sigma = np.zeros((n_elem, 3))
    Stress_loss_AL = 0
    dALStress_pp_du = np.zeros(n_dof)
    dALStress_pp_drho = np.zeros(n_elem)
    AL_lambda_new = np.zeros(n_elem)
    AL_loss = 0
    AL_mu_new = AL_mu
    Uelastic = 0
    dU_p_du = np.zeros(n_dof)

    for e in range(n_elem):
        index = [2 * int(tri[e, 0]), 2 * int(tri[e, 0]) + 1,
                 2 * int(tri[e, 1]), 2 * int(tri[e, 1]) + 1,
                 2 * int(tri[e, 2]), 2 * int(tri[e, 2]) + 1]
        _u_elem = _u_node[index]
        epsilon_elem = l_coef[e] @ _u_elem
        _kE, _krho = legacy_topo(_rho_SIMP[e])
        _dkE, _dkrho = legacy_dtopo(_rho_SIMP[e])
        sigma[e] = _kE * (d_mat @ epsilon_elem)
        vm[e] = np.sqrt(3 * sigma[e, 2] ** 2 + sigma[e, 0] ** 2
                        + sigma[e, 1] ** 2 - sigma[e, 0] * sigma[e, 1])

        Uelastic += area[e] * (epsilon_elem @ sigma[e]) / 2
        dU_p_du[index] += area[e] * (l_coef[e].T @ sigma[e]) + Uelastic / _kE * _dkE

        dSigma_p_du = l_coef[e].T @ (d_mat.T @ np.array([
            2 * sigma[e, 0] - sigma[e, 1],
            -sigma[e, 0] + 2 * sigma[e, 1],
            6 * sigma[e, 2]]))

        _Lambda = vm[e] / _kE ** rel_p / max_rel - 1          # rel3
        _gi = _Lambda ** 3 + _Lambda                          # type2_stress 开
        _hi = max(_gi, -AL_lambda[e] / AL_mu)
        _Pi = AL_lambda[e] * _hi + AL_mu / 2 * _hi ** 2
        Stress_loss_AL += _Pi / n_elem

        if _gi * AL_mu + AL_lambda[e] > 0:
            _k1 = 1 / _kE ** rel_p * (3 * _Lambda ** 2 + 1) / max_rel
            _k2 = (-rel_p * (3 * _Lambda ** 2 + 1) * vm[e] / max_rel
                   / _kE ** (rel_p + 1)) * _dkE
            dALStress_pp_du[index] += (1 / n_elem * (AL_lambda[e] + AL_mu * _gi)
                                       * _k1 * dSigma_p_du)
            dALStress_pp_drho[e] = (1 / n_elem * (AL_lambda[e] + AL_mu * _gi) * _k2)
            AL_loss += _gi ** 2
        else:
            AL_loss += AL_lambda[e] ** 2
        AL_lambda_new[e] = min(0, AL_lambda[e] + AL_mu_new * _hi)

    von_mises_max = np.max(vm)
    _kE_all, _ = legacy_topo(_rho_SIMP)
    RelStress_max = np.max(vm / _kE_all ** rel_p)
    AL_loss = np.sqrt(AL_loss)

    K_csr = _Kmech.tocsr()
    dALStress_p_drho = (-sp_sparse_linalg.spsolve(K_csr.T, dALStress_pp_du).T
                        @ _dRHSmech_p_dth + dALStress_pp_drho)
    dU_p_drho = -sp_sparse_linalg.spsolve(K_csr.T, dU_p_du).T @ _dRHSmech_p_dth

    return dict(von_mises=vm, sigma=sigma, von_mises_max=von_mises_max,
                rel_stress_max=RelStress_max, stress_loss_al=Stress_loss_AL,
                al_loss=AL_loss, d_al_stress_p_drho=dALStress_p_drho,
                uelastic=Uelastic, d_u_p_drho=dU_p_drho,
                al_lambda_new=AL_lambda_new)


# =============================================================================
# 来源 A（决定性）—— 有限差分
# =============================================================================

def _fd(fx, func, elem, d=1e-6):
    rp, rm = fx.rho.copy(), fx.rho.copy()
    rp[elem] += d
    rm[elem] -= d
    return (func(rp) - func(rm)) / (2 * d)


class _DerivativeContext:
    """把 D6/D8 两条测试所需的中间量一次算齐（都用内核 5/16 的公开产出）。

    **D6 修正后**保留：``dU_du`` = 纯 ``∂U/∂u``（逐单元 ``Δ_e·(L_eᵀσ_e)``，
    检验生产 ``d_u_p_du`` 是否已去掉 legacy 的广播显式项）与 ``dR``/``dK_u``
    （D8 那条测试重建 legacy 形态用）。legacy 的累计 ``Uelastic`` 广播
    （``lam_legacy``）与「只修结构」对照式已随 D6 修正失效，删除。
    """

    def __init__(self, fx):
        self.fx = fx
        u, K, dRHS, res = fx.at_production()
        self.u, self.K, self.dRHS, self.res = u, K, dRHS, res
        simp = SIMPProjection(ref_solver.MECH_SIMP_PARAMS)
        self.k_e = simp.project(fx.rho)
        self.dk_e = simp.derivative(fx.rho)

        n = fx.n_elem
        self.idx_of = [[2 * int(fx.tri[e, 0]), 2 * int(fx.tri[e, 0]) + 1,
                        2 * int(fx.tri[e, 1]), 2 * int(fx.tri[e, 1]) + 1,
                        2 * int(fx.tri[e, 2]), 2 * int(fx.tri[e, 2]) + 1]
                       for e in range(n)]
        self.dU_du = np.zeros(u.shape)
        for e in range(n):
            eps = fx.mech.L[e] @ u[self.idx_of[e]]
            sigma = self.k_e[e] * (fx.mech.D @ eps)
            self.dU_du[self.idx_of[e]] += fx.mech.area[e] * (fx.mech.L[e].T @ sigma)

        # ``dRHS = dK·u + dR``（D5 已修）；``dR`` 单独取出，``dK·u = dRHS − dR``
        dR = sp_sparse.coo_matrix(
            (fx.ops.r0_data * 1.0, (fx.ops.dr_p_dth_row, fx.ops.dr_p_dth_col)),
            shape=(fx.ops.n_dof, n)).toarray()
        self.dR = dR
        self.dK_u = self.dRHS.toarray() - dR

    # ---- AL 罚项的伴随右端（内核 17 返回的中间量）----
    def _d_al_stress_pp_du(self):
        return self.res.d_al_stress_pp_du

    def _d_al_stress_pp_drho(self):
        return self.res.d_al_stress_pp_drho


def test_strain_energy_derivative_matches_finite_difference_after_fix(fx):
    """★★ ``dU_p_drho``：**D6 已修（2026-09-23，落地包络闭式）** ⇒ 与有限差分一致。

    有限差分是外部判据：``U(ρ)`` 由「解一次 → 逐单元累加能量」定义，与灵敏度公式
    完全独立。修前实测（§12 的 D6）：FD/legacy = 0.391/0.491/0.366（夹具、
    偏差不是常数倍）；两处都修后 1.0000。修后生产路径**本身**走包络闭式 ⇒
    ``res.d_u_p_drho`` 应直接 == FD。

    本测试断言三件事：
    (a) **决定性**：生产 ``d_u_p_drho`` == FD（≤5e-3，闭式的 FD 分辨率量级）；
    (b) 闭式实现与 FD 一致（`test_envelope_closed_form_matches_finite_difference`
        的夹具版，rel < 1e-4）；
    (c) 防退化：``dU_p_du`` 已恢复为**纯** ``∂U/∂u``（不再含 legacy 的广播显式项
        —— 否则说明 D6 的结构修正没落地）。
    """
    ctx = _DerivativeContext(fx)
    res = ctx.res

    def energy(rho_):
        return fx.chain(rho_)[3].elastic_strain_energy

    sample = sorted(range(fx.n_elem),
                    key=lambda e: -abs(res.d_u_p_drho[e]))[:3]
    for e in sample:
        fd = _fd(fx, energy, e)
        scale = abs(fd)
        # (a) 生产 == FD
        assert abs(fd - res.d_u_p_drho[e]) <= 5e-3 * scale, (
            f"单元 {e}：D6 修后生产值仍与 FD 差 "
            f"{abs(fd - res.d_u_p_drho[e]) / scale:.3e}")
        # (c) ``dU_p_du`` 是纯 ∂U/∂u（= Ku 的伴随右端），不含累计 Uelastic 广播
        assert np.array_equal(res.d_u_p_du, ctx.dU_du), (
            f"单元 {e}：``d_u_p_du`` 不是纯 ∂U/∂u —— D6 结构修正没落地")


def test_envelope_closed_form_matches_finite_difference(fx):
    """包络闭式（免伴随）与有限差分一致（夹具上）。

    ``strain_energy_derivative_envelope``（``dU/dρ_e = −½uᵀ(dK/dρ_e)u −
    uᵀ(dR/dρ_e)``）—— **D6 修正落地实现**（2026-09-23 起生产 ``dU_p_drho``
    就走它，见 §12 的 D6 修正清单）。本测试验证它的数学正确。
    """
    from scipy import sparse as sp

    u, K, dRHS, res = fx.at_production()
    ops = fx.ops
    # ∂R/∂ρ_e（照抄 mech_worker 的 legacy_mech_sensitivity：dk_rho=1）
    dR = sp.coo_matrix(
        (ops.r0_data * (ops.rconst_map @ np.ones(fx.n_elem)),
         (ops.dr_p_dth_row, ops.dr_p_dth_col)),
        shape=(ops.n_dof, fx.n_elem),
    ).tocsr()

    env = ref_stress.strain_energy_derivative_envelope(
        u, K, dRHS, dR, n_dof=ops.n_dof, n_elem=fx.n_elem)

    def energy(rho_):
        return fx.chain(rho_)[3].elastic_strain_energy

    # 抽样：|闭式| 最大的几个单元（FD 分辨率最好）
    sample = sorted(range(fx.n_elem),
                    key=lambda e: -abs(env[e]))[:3]
    for e in sample:
        fd = _fd(fx, energy, e)
        scale = abs(fd)
        rel = abs(env[e] - fd) / scale if scale else abs(env[e])
        assert rel < 1e-4, f"单元 {e}: 闭式 vs FD rel={rel:.2e}"

    # ★ **D6 已修**：生产 ``d_u_p_drho`` 现在**就**走包络闭式 ⇒ 应与闭式逐位相同。
    #   ⚠ 修前的防退化断言（「闭式必须与 legacy 确实不同」）已随修正失效 —— legacy
    #   （含 D6）的形态记录在 §12，不再核对。
    assert np.array_equal(res.d_u_p_drho, env), (
        "D6 修后生产 `d_u_p_drho` 与包络闭式不逐位 —— 落地有误")


def test_al_stress_derivative_state_is_pinned(fx_active):
    """★★ 钉住 ``dALStress_p_drho`` 的**实测状态**（设计文档 §12 的 D5 与 **D8**）。

    用**收紧上限**的夹具（部分单元违反约束，活跃/不活跃两支都出现）。
    两件事分开断言：

    (a) **D5 的机构可辨认**：把伴随解里的 ``dRHS`` 从 ``dK·u − dR`` 换成
        ``dK·u + dR``，结果的差**恰好**是 ``2λᵀdR``（逐单元）。这条与内核 16
        那条 D5 测试同源，但在**本内核**的伴随上再验一次（`λ` 不同）。
    (b) **D8（已修 2026-09-23）**：``dALStress_p_drho`` == 有限差分（比值 ≈ 1）。
        **D8 修正 = 两个缺失项**（§12 与 reference docstring 第四节）：
        ① ``d_sigma_p_du`` 缺 ``k_E/(2σ̄)``；② 显式项缺 ``k1·∂σ̄/∂k_E·k_E'``
        （σ̄ 经 σ=k_E·D·ε 直接依赖 k_E）。修前比值 ~1e11（夹具 1.1e11 /
        真实网格 3.4e10），修后 **1.0000**。

    有限差分的前提：`λ` 与 `μ` **固定**（笔记 :261 「优化时会固定惩罚因子 μ 和
    拉格朗日乘子 λ_i，此时进行敏感度分析」）。⚠ ``P = μ/2·max(g,0)²`` 在 ``g=0``
    处是 C¹（平方的 ReLU），FD 前提成立（``h = max(g,0)`` 本身不可微，但罚项可微）。
    """
    ctx = _DerivativeContext(fx_active)
    res = ctx.res

    def penalty(rho_):
        return fx_active.chain(rho_)[3].stress_loss_al

    elems = [e for e in range(fx_active.n_elem)
             if abs(res.d_al_stress_p_drho[e]) > 0]
    assert elems, "dALStress/dρ 恒为零（约束没被违反？）⇒ 换夹具"

    # ---- (a) D5 的机构（**已修**：生产用 ``dK·u + dR``）----
    lam_al = sp_sparse_linalg.spsolve(fx_active.K.tocsr().T.tocsc(),
                                      res.d_al_stress_pp_du)
    # legacy（D5 未修）的重装形态：``−λᵀ(dK·u − dR) + dAL_pp_drho``。生产值
    # 走已修的 ``dK·u + dR`` ⇒ 两者差**恰好** ``2λᵀdR`` —— 证明 D5 修的正是
    # 载荷项符号（机构可辨认，且 D8 的巨大偏差在两种形态里都存在）。
    delta_d5 = 2 * (lam_al @ ctx.dR)
    d_al_pp_drho = res.d_al_stress_pp_drho
    legacy_as_formula = -(
        sp_sparse_linalg.spsolve(fx_active.K.tocsr().T.tocsc(),
                                 res.d_al_stress_pp_du)
        @ (ctx.dK_u - ctx.dR)) + d_al_pp_drho
    # ① 生产值 == 已修形态 ``−λᵀ(dK·u + dR) + dAL_pp_drho``（从中间量重装）
    fixed_as_formula = -(
        sp_sparse_linalg.spsolve(fx_active.K.tocsr().T.tocsc(),
                                 res.d_al_stress_pp_du)
        @ (ctx.dK_u + ctx.dR)) + d_al_pp_drho
    assert np.abs(fixed_as_formula - res.d_al_stress_p_drho).max() <= (
        1e-9 * np.abs(res.d_al_stress_p_drho).max()),         "从中间量重装的已修形态与返回值不一致 ⇒ 中间量没取对"
    # ② ``legacy − production == 2λᵀdR`` ⇒ D5 修的正是载荷项符号
    assert np.abs((legacy_as_formula - res.d_al_stress_p_drho) - delta_d5).max() <= (
        1e-9 * np.abs(res.d_al_stress_p_drho).max()),        "legacy 与已修的差不是 2λᵀdR ⇒ D5 修的不是载荷项符号"

    sample = sorted(elems, key=lambda e: -abs(res.d_al_stress_p_drho[e]))[:3]
    for e in sample:
        fd = _fd(fx_active, penalty, e)
        ratio = res.d_al_stress_p_drho[e] / fd
        print(f"  单元 {e:3d}: FD {fd: .6e}  production {res.d_al_stress_p_drho[e]: .6e}"
              f"  比值 {ratio: .4e}  |2λᵀdR| {abs(delta_d5[e]): .3e}")
        # (b) **D8 已修**：与有限差分一致（比值 ≈ 1，FD 分辨率内）。
        #    修前形态（~1e11 倍、且不随单元变）已写入 §12，不再核对。
        assert abs(ratio - 1.0) < 1e-3, (
            f"单元 {e}: production/FD = {ratio:.4e} —— D8 修后仍偏离 FD，"
            f"重新核对 §12 的 D8 修正清单")

    # 防退化：D5 的修正量必须可分辨（否则 (a) 那条断言抓不到符号）
    assert np.abs(delta_d5).max() > 0.0
    # 防退化：D8 修正必须可分辨 —— ``d_sigma_p_du`` 的 k_E/(2σ̄) 因子确实改了
    # 生产值（不是「两边都零」的假阳性）。
    assert np.abs(res.d_al_stress_p_drho).max() > 0.0


# =============================================================================
# 来源 C —— reference vs fast（新写的向量化）
# =============================================================================

def _call(module, fx_obj, rho_):
    u, K, dRHS, _ = fx_obj.chain(rho_)
    return module.mech_stress_and_energy(
        u, K, dRHS, rho_, fx_obj.al_lambda, fx_obj.al_mu,
        tri=fx_obj.tri, l_coef=fx_obj.mech.L, d_mat=fx_obj.mech.D,
        area=fx_obj.mech.area, dR_p_dth=fx_obj.dR(), flags=fx_obj.flags,
    )


@requires_fast
def test_fast_matches_reference_bitwise(fx_active):
    """★ fast 的应力/应变能导数与 reference **逐位**相同。

    本层是「legacy 逐单元循环 + 标量 ``+=``」的向量化，所以有两处必须保住加法顺序：
    scatter 用 ``np.add.at``（按 (单元, 局部自由度) 展平顺序）、
    标量累加用 ``np.cumsum(...)[-1]``（不是 ``np.sum`` 的成对求和）。
    这条用 ``array_equal`` 把整条链钉死 —— 包括 D6 那个「累计 ``Uelastic``」
    （它依赖累加顺序，向量化最容易在这里悄悄改掉）。

    用**收紧上限**的夹具：活跃/不活跃两支都出现，分支的掩码也一起被验。
    """
    r = _call(ref_stress, fx_active, fx_active.rho)
    f = _call(fast_stress, fx_active, fx_active.rho)

    for name in ("sigma", "von_mises", "al_lambda_new",
                 "d_al_stress_p_drho", "d_u_p_drho",
                 "d_al_stress_pp_du", "d_al_stress_pp_drho", "d_u_p_du"):
        assert np.array_equal(np.asarray(getattr(r, name)),
                              np.asarray(getattr(f, name))), f"{name} 不逐位相同"

    for name in ("von_mises_max", "rel_stress_max", "stress_loss_al", "al_loss",
                 "elastic_strain_energy"):
        assert getattr(r, name) == getattr(f, name), f"{name} 不逐位相同"

    # 防退化：活跃/不活跃两支都要出现（掩码被真正用到）。
    # ``d_al_stress_pp_drho`` 只在活跃单元上非零（rel3 的 ``k2 ∝ dkE ≠ 0``）⇒ 可作代理。
    # ⚠ **不能**用 ``0 < count(λ_new == 0) < n`` —— 那在 λ=0（生产初值）下必然失败：
    #   D7 保证 ``λ_new ≡ 0`` 对**所有**单元成立。
    active_proxy = r.d_al_stress_pp_drho != 0.0
    assert np.any(active_proxy), "没有活跃单元 ⇒ 分支没覆盖到"
    assert np.any(~active_proxy), "全是活跃单元 ⇒ 只覆盖了一支"


@requires_fast
def test_fast_matches_reference_bitwise_on_the_production_limit(fx):
    """生产上限（**只有不活跃一支**）下也逐位 —— 另一套掩码。"""
    r = _call(ref_stress, fx, fx.rho)
    f = _call(fast_stress, fx, fx.rho)
    assert np.all(r.d_al_stress_pp_du == 0.0)
    assert np.array_equal(r.d_al_stress_pp_du, f.d_al_stress_pp_du)
    assert np.array_equal(r.d_u_p_du, f.d_u_p_du)
    assert r.elastic_strain_energy == f.elastic_strain_energy
    assert r.al_loss == f.al_loss


@requires_fast
def test_factor_path_matches_spsolve_bitwise(fx_active):
    """阶段 8 分解复用：``factor=splu(K.T)``（CSC）与默认 ``spsolve(K.T,·)`` 逐位。

    力学两个伴随解共用同一个 ``K`` 的转置因子。⚠ ``K.T`` 是 **CSC** —— splu 复用
    只对 CSC 成立（spsolve 对 CSR 走 SuperLU 的 CSR 路径，splu 强制转 CSC，两者
    不逐位）。力学伴随解恰好是 ``K.T``，与 EM 侧 ``J.T`` 同理。
    """
    u, K, dRHS, _ = fx_active.chain(fx_active.rho)
    f_default = fast_stress.mech_stress_and_energy(
        u, K, dRHS, fx_active.rho, fx_active.al_lambda, fx_active.al_mu,
        tri=fx_active.tri, l_coef=fx_active.mech.L, d_mat=fx_active.mech.D,
        area=fx_active.mech.area, dR_p_dth=fx_active.dR(), flags=fx_active.flags,
    )
    factor = sp_sparse_linalg.splu(K.tocsr().T)
    f_reuse = fast_stress.mech_stress_and_energy(
        u, K, dRHS, fx_active.rho, fx_active.al_lambda, fx_active.al_mu,
        tri=fx_active.tri, l_coef=fx_active.mech.L, d_mat=fx_active.mech.D,
        area=fx_active.mech.area, dR_p_dth=fx_active.dR(),
        flags=fx_active.flags, factor=factor,
    )

    # ⚠ D6 修正后：``d_u_p_drho`` 走包络闭式（免伴随、不经过 factor），两路
    # 必然相同；真正考验 factor 的是 ``d_al_stress_p_drho``。
    for name in ("d_al_stress_p_drho", "d_u_p_drho"):
        assert np.array_equal(np.asarray(getattr(f_default, name)),
                              np.asarray(getattr(f_reuse, name))), f"{name} 不逐位相同"


@requires_fast
def test_energies_are_sequential_sums_not_pairwise(fx):
    """★ 三个标量累加必须是**顺序**和：``np.sum`` 的成对求和会改末位。

    造一组量级悬殊的单元能量，验证 fast 的 ``Uelastic`` 等于顺序累加而不是 ``np.sum``。
    这是「顺手向量化」最容易踩的坑；内核 5 的载荷那一节有同型的测试。
    """
    fx_big = StressFixture(n_ang=7, rho_value=0.5,
                           flags=production_flags(1e8))
    # 人造一组量级悬殊的 ρ 让单元能量量级拉开
    rng = np.random.default_rng(1)
    rho_alt = np.clip(0.3 + 0.4 * rng.random(fx_big.n_elem), 0.0, 1.0)
    r = _call(ref_stress, fx_big, rho_alt)
    f = _call(fast_stress, fx_big, rho_alt)

    idx_of = [[2 * int(fx_big.tri[e, 0]), 2 * int(fx_big.tri[e, 0]) + 1,
               2 * int(fx_big.tri[e, 1]), 2 * int(fx_big.tri[e, 1]) + 1,
               2 * int(fx_big.tri[e, 2]), 2 * int(fx_big.tri[e, 2]) + 1]
              for e in range(fx_big.n_elem)]
    u = ref_solver.solve_mech_field(rho_alt, fx_big.ops)
    simp = SIMPProjection(ref_solver.MECH_SIMP_PARAMS)
    k_e = simp.project(rho_alt)
    u_e = np.array([
        fx_big.mech.area[e] * float(
            (fx_big.mech.L[e] @ u[idx_of[e]])
            @ (k_e[e] * (fx_big.mech.D @ (fx_big.mech.L[e] @ u[idx_of[e]])))) / 2
        for e in range(fx_big.n_elem)])
    seq = 0.0
    for v in u_e:
        seq += float(v)

    assert f.elastic_strain_energy == seq
    assert r.elastic_strain_energy == seq


@requires_fast
def test_speed_guard(fx_active):
    """性能守卫（宽松阈值）：本层是新写的向量化，必须**明显**快于逐单元循环。

    设计文档 §8 的「向量化补齐」清单点名了 `compute_mechDiff_AL`（每个 MMA 迭代
    跑的逐单元 Python 循环）。⚠ 两个伴随解（``spsolve``）在两层是同一份实现且
    占大头，所以倍速会被它拉平 —— 报出的数字是**端到端**的，不是纯循环那一段的。

    ⚠ 阈值松（1.2×）：夹具只有 36 个单元；真实网格上的倍速由
    ``tools/probe_mech.py`` 报出。**计时不可跨条件比较**（设计文档 §11⑥）。
    """
    import timeit

    def run(module):
        return lambda: _call(module, fx_active, fx_active.rho)

    t_ref = min(timeit.repeat(run(ref_stress), number=3, repeat=5))
    t_fast = min(timeit.repeat(run(fast_stress), number=3, repeat=5))
    speedup = t_ref / t_fast
    print(f"[性能守卫] 应力/应变能导数：逐单元 {t_ref * 1e3:.2f} ms，"
          f"向量化 {t_fast * 1e3:.2f} ms，加速 {speedup:.2f}×"
          f"（{fx_active.n_elem} 个单元；含两个 spsolve）")
    assert speedup > 1.2, f"向量化只快 {speedup:.2f}× ⇒ 这一层没有存在理由"
