"""阶段 4D：L4 力学段（``obj_func_worker_mech_AL`` 的内核组装版 + legacy 原式导数）。

对应 legacy ``nonlinear_solver_energy_MMA10.py:3044-3051``（下称 MMA10）：

    u = solve_mech_field_vectorized(ρ, Kmech)                → 内核 16 solve_mech_field
    K, dRHS = compute_udiff_vectorized(ρ, u, Kmech)          → 内核 16 mech_sensitivity
    VonMises, Stress_loss_AL, dALStress, Uelastic, dU, λ_new = compute_mechDiff_AL(...)
                                                             → 内核 17 mech_stress_and_energy

------------------------------------------------------------------------------
为什么这里要「legacy 原式」的力学 SIMP 导数（设计文档 §11⑩ 的后果）
------------------------------------------------------------------------------

内核 16/17 的 ``dkE`` 来自 ``reference.simp.SIMPProjection.derivative`` —— 它是
legacy ``SIMP_dfunc`` 的**代数等价但非逐位**改写（1~2 ULP，为了避开 ``/sinh(β)``
的除零）。golden 采集的是 legacy 原式的轨迹，所以 L4 的 ``dU_p_drho`` /
``dALStress_p_drho``（它们都经 ``dkE`` 进伴随解）必须用**原式**才能逐位对齐：

- ``mech_sensitivity`` 的 ``dkE`` → 本模块 ``legacy_mech_sensitivity``（照搬
  ``:2861-2897``，只把 ``dkE`` 换成原式）；
- ``mech_stress_and_energy`` 内部的 ``dk_e`` → ``LegacyMechSIMP``（覆盖
  ``derivative`` 的 ``SIMPProjection`` 子类，传给它的 ``simp`` 形参）。

``kE``（``mech_topo_opti`` 的投影）与 ``project`` 逐位一致，不需要原式。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from .. import solvers
from ..config import RunConfig
from ..elimination import build_mech_elimination
from ..kernels.fast import mech_solver as fast_mech
from ..kernels.fast import mech_stress as fast_stress
from ..kernels.reference.mech_solver import (
    MECH_SIMP_PARAMS,
    _slave_rotation,
)
from ..kernels.reference.mech_stress import StressConstraintFlags
from ..kernels.reference.simp import SIMPProjection

from .preassembly import Preassembly

__all__ = [
    "LegacyMechSIMP",
    "legacy_mech_dfunc",
    "legacy_mech_sensitivity",
    "mech_worker",
    "stress_flags_from_config",
]


def legacy_mech_dfunc(rho: np.ndarray) -> np.ndarray:
    """``dmech_topo_opti(ρ)[0]`` 的原式（``MMA10:1242-1255``）。

    ``dkE = p·(1−min)·ρ₁^(p−1)·dρ₁/dρ``，其中 ``dρ₁/dρ`` 用 legacy 的
    ``β·cosh(β·thb)·cosh(β−β·thb)/sinh(β)/cosh(β(ρ−thb))²``。⚠ 与
    ``reference.simp.derivative`` 代数等价但非逐位（1~2 ULP，§11⑩）。
    """
    p = MECH_SIMP_PARAMS
    beta, thb, power, mins = p.beta, p.thb, p.power, p.min_density
    rho = np.asarray(rho, dtype=float)
    rho_1 = ((np.tanh(beta * (rho - thb)) + np.tanh(beta * thb))
             / (np.tanh(beta * (1 - thb)) + np.tanh(beta * thb)))
    drho_1 = (beta * np.cosh(beta * thb) * np.cosh(beta - beta * thb)
              / np.sinh(beta) / np.cosh(beta * (rho - thb)) ** 2)
    return power * (1 - mins) * rho_1 ** (power - 1) * drho_1


class LegacyMechSIMP(SIMPProjection):
    """力学侧 SIMP 投影器，``derivative`` 用 legacy 原式（其余继承）。

    ``mech_stress_and_energy(simp=...)`` 用它 ⇒ ``dk_e`` 与 golden 逐位一致。
    """

    def __init__(self, params=MECH_SIMP_PARAMS):
        super().__init__(params)

    def derivative(self, rho: np.ndarray) -> np.ndarray:
        return legacy_mech_dfunc(rho)


def legacy_mech_sensitivity(
    rho: np.ndarray, u_node: np.ndarray, ops: Any
) -> Tuple[Any, Any]:
    """``compute_udiff_vectorized``（``MMA10:2861-2897``）的驱动层移植。

    与内核 16 的 ``mech_sensitivity`` 同数学，唯一差别是 ``dkE`` 用
    ``legacy_mech_dfunc``（原式）。**D5 已修**（载荷项符号翻为 ``+ dR``，与
    内核 16 同步，见 ``reference/mech_solver.py::mech_sensitivity`` 的 docstring）。
    """
    k_e, _ = fast_mech.mech_density_factors(rho)
    dk_e = legacy_mech_dfunc(rho)
    dk_rho = np.ones_like(np.asarray(rho, dtype=float))
    n_dof = ops.n_dof
    n_elem = ops.n_elem

    dR_p_dth = sp_sparse.coo_matrix(
        (ops.r0_data * (ops.rconst_map @ dk_rho), (ops.dr_p_dth_row, ops.dr_p_dth_col)),
        shape=(n_dof, n_elem),
    )
    dK_p_dth_u = sp_sparse.coo_matrix(
        (ops.k0_data * (ops.kconst_map @ dk_e) * (ops.k_u_map @ u_node),
         (ops.dk_p_dth_u_row, ops.dk_p_dth_u_col)),
        shape=(n_dof, n_elem),
    )
    K = sp_sparse.coo_matrix(
        (ops.k0_data * (ops.kconst_map @ k_e), (ops.k0_row, ops.k0_col)),
        shape=(n_dof, n_dof),
    )
    K = K.tocsr() + ops.k_boundary

    return K, dK_p_dth_u + dR_p_dth


def stress_flags_from_config(cfg: RunConfig) -> StressConstraintFlags:
    """``RunConfig.stress`` → ``StressConstraintFlags``（阶段 2 的枚举折并展开）。"""
    from ..config.enums import expand_rel_stress_type, expand_stress_type

    st = cfg.stress
    t1, t2 = expand_stress_type(st.stress_type)
    r1, r2, r3, r4 = expand_rel_stress_type(st.rel_stress_type)
    return StressConstraintFlags(
        use_rel_stress_constraint=st.use_rel_stress_constraint,
        use_type1_stress=t1, use_type2_stress=t2,
        use_type1_rel_stress=r1, use_type2_rel_stress=r2,
        use_type3_rel_stress=r3, use_type4_rel_stress=r4,
        max_stress_constraint=st.max_stress_constraint_pa,
        max_rel_stress_constraint=st.max_rel_stress_constraint_pa,
        rel_stress_p_exp=st.rel_stress_p_exp,
    )


def mech_worker(
    rho_mech: np.ndarray,
    pre: Preassembly,
    al_lambda: np.ndarray,
    al_mu: float,
    *,
    flags: StressConstraintFlags,
    solver: Optional[str] = None,
) -> Any:
    """力学 worker。对应 ``MMA10:3044-3051``。

    Returns
    -------
    ``(MechStressResult, u_node)`` —— legacy 的 7 元组里 ``u_node`` 不在
    ``MechStressResult`` 里（那 6 项 + 解一起返回），本函数补上 u。

    Parameters
    ----------
    rho_mech : (N_elem_mech,) 力学密度（``ρ[list(element_rotation)]``）。
    pre : 预组装（含 ``Kmech`` 与力学单元矩阵）。
    al_lambda : (N_elem_mech,) AL 乘子（D7 下恒为 0）。
    al_mu : float AL 罚因子。
    """
    session, owned = solvers.as_session(
        solver, pre.cfg.solver.pardiso_mtype, shared=True)
    # 主从消元（2026-09-24，替换生产路径）：K_mech 真对称 ⇒ 前向/伴随都用 TᵀKT。
    # ⚠ D12 登记的 col_mask 漏项（mech_sensitivity 缺 `@ ops.col_mask`）与消元
    #   不冲突：消元处理主从行；Dirichlet 列清零由 mech_residual 的 col_mask 做。
    elim = build_mech_elimination(
        pre.Kmech.n_dof // 2, pre.Kmech.slave_to_master or {},
        _slave_rotation(pre.cfg.geometry.parts))
    try:
        u = fast_mech.solve_mech_field(rho_mech, pre.Kmech, solver=session,
                                       elimination=elim)
        # ★ K_mech 用 **mech_residual** 的对称版（`coo @ col_mask + k_boundary`）：
        #   mech_sensitivity 返回的 K 缺 col_mask（D12 登记，非对称 0.32），直接
        #   用它做消元/伴随会因 Dirichlet 列未清零而不对称。dRHS（灵敏度项，
        #   D5 已修符号）另取 —— 与 mech_residual 的 K 解配套。
        _, K_mech = fast_mech.mech_residual(rho_mech, pre.Kmech)
        K_mech = sp_sparse.csr_matrix(K_mech)
        _, dRHS = legacy_mech_sensitivity(rho_mech, u, pre.Kmech)
        # D6 修正落地：``dU_p_drho`` 的包络闭式要 ``∂R/∂ρ_e`` 单独传（闭式重建
        # ``dK/dρ_e·u = S[:,e] − dR[:,e]``）。照 ``tools/probe_envelope.py`` 的写法
        # 从 ops（``pre.Kmech``）构造 —— dk_rho=1 ⇒ ``∂R/∂ρ_e`` 恰是 ``r0_data``。
        dR_p_dth = sp_sparse.coo_matrix(
            (pre.Kmech.r0_data * (pre.Kmech.rconst_map @ np.ones_like(rho_mech)),
             (pre.Kmech.dr_p_dth_row, pre.Kmech.dr_p_dth_col)),
            shape=(pre.Kmech.n_dof, pre.Kmech.n_elem),
        )
        # 阶段 8 分解复用：AL 罚项那一支的伴随解用 K 的转置因子（D6 修后只剩这一个
        # 伴随解 —— ``dU_p_drho`` 走包络闭式、免伴随）。只构造一次，factor.solve() 回代
        # —— 与 spsolve 逐位一致（实测，见设计文档阶段 8）。
        # ⚠ 后端经 session.factorize：superlu 返回 SuperLU 因子（逐位不变）；pardiso
        #   返回 _PardisoFactorView（引用会话实例、不重复 init）。
        factor_kt = session.factorize(elim.eliminate_matrix(K_mech.tocsr()).T)
        try:
            res = fast_stress.mech_stress_and_energy(
                u, K_mech, dRHS, rho_mech, al_lambda, al_mu,
                tri=pre.mech_tri, l_coef=pre.mech_l_coef, d_mat=pre.mech_d_mat,
                area=pre.mech_area, dR_p_dth=dR_p_dth,
                flags=flags, simp=LegacyMechSIMP(), factor=factor_kt,
                elimination=elim,
                stress_smooth=getattr(pre, "stress_smooth", None),
            )
        finally:
            close = getattr(factor_kt, "close", None)
            if close is not None:
                close()
        return res, u
    finally:
        if owned:
            session.close()
