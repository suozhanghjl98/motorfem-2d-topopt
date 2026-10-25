"""内核 17 的 fast 层：应力/应变能导数的批量实现。

数学、六支系数对照表、缺陷登记（D6/D8 已修，D7 仍待）写在
``reference/mech_stress.py`` 的模块 docstring 里，本文件不重复。

------------------------------------------------------------------------------
哪些向量化了、哪些没有
------------------------------------------------------------------------------

| 段 | 做法 | 说明 |
|---|---|---|
| 逐单元的 ``ε``/``σ``/``σ̄`` | **批量** | ⚠ 用 ``@``（matmul）**不是** ``einsum``，见下 |
| 六支的 ``Λ``/``g``/``h``/``Pi``/``k1``/``k2`` | **批量** | 分支的布尔只选一支，所以直接算那一支 |
| ``dU_p_du`` / ``dALStress_pp_du`` 的 scatter | ``np.add.at`` | ⚠ 见下 |
| ``Uelastic`` / ``Stress_loss_AL`` / ``AL_loss`` 的累加 | ``np.cumsum(...)[-1]`` | ⚠ 见下 |
| ``d_u_p_drho``（D6 修正落地） | 与 reference **共用**包络闭式 | 免伴随（``strain_energy_derivative_envelope``），无 spsolve |
| ``dALStress_p_drho`` 的伴随解 | 与 reference **共用** | 是 ``spsolve``，没有可向量化的部分 |

⚠ **三处必须保住求和/求积顺序**（本内核是「legacy 逐单元循环 + 标量 ``+=``」）：

0. **批量 matvec 用 ``@`` 不用 ``np.einsum``**：实测 ``einsum("nij,nj->ni", L, u)``
   与逐单元的 ``L[e] @ u_e`` 差 ~3e-16（求和顺序不同），而批量的
   ``(L @ u[...,None])[...,None]`` **逐位相同**。同理 ``k1``/``k2`` 的乘法**结合顺序**
   也照抄 legacy（``1/N * (λ+μg) * k1 * dSigma`` 里 ``1/N`` 在**最前**，
   写成最后除会差 1 ULP）。

1. **scatter 累加**用 ``np.add.at`` 且按 **(单元, 局部自由度)** 的展平顺序 ——
   与 legacy 的 ``for e: dU_p_du[index] += …`` 同序。
   ``CLAUDE.md`` §3 红线 5 允许用它，**前提是验证加法顺序不变** ——
   测试里有一条 ``array_equal`` 把整条链钉死。
2. **标量累加** ``Uelastic``/``Stress_loss_AL``/``AL_loss`` 在 legacy 是循环里的顺序
   累加，而 ``np.sum`` 走**成对求和** ⇒ 用 ``np.cumsum(...)[-1]``（顺序累加）。
   内核 5 的载荷那一节踩过同一个坑（那里的测试还带一条防退化断言证明两者确实不同）。

``Uelastic`` 还必须**逐单元**拿到「累计到当前」的值（`elastic_strain_energy`
的值）⇒ ``np.cumsum`` 天然给出这个序列。⚠ **D6 修正后它不再进入**
``d_u_p_drho``（legacy 曾用累计 ``Uelastic/kE·dkE`` 广播到伴随右端，已登记 D6）；
``d_u_p_drho`` 改用包络闭式（``strain_energy_derivative_envelope``，免伴随）。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp_sparse

from ..reference.mech_stress import (
    PRODUCTION_STRESS_FLAGS,
    MechStressResult,
    StressConstraintFlags,
    strain_energy_derivative_envelope,  # noqa: F401  （D6 修正落地：两层共用闭式）
    von_mises,          # noqa: F401  （纯 numpy，两层共用）
    element_stress,     # noqa: F401
)

__all__ = [
    "StressConstraintFlags",
    "PRODUCTION_STRESS_FLAGS",
    "MechStressResult",
    "von_mises",
    "element_stress",
    "strain_energy_derivative_envelope",
    "mech_stress_and_energy",
]


def _mech_simp():
    from ..reference.mech_solver import MECH_SIMP_PARAMS
    from ..reference.simp import SIMPProjection
    return SIMPProjection(MECH_SIMP_PARAMS)


def mech_stress_and_energy(
    u_node: np.ndarray,
    K_mech,
    dRHS_mech_p_dth,
    rho: np.ndarray,
    al_lambda: np.ndarray,
    al_mu: float,
    *,
    tri: np.ndarray,
    l_coef: np.ndarray,
    d_mat: np.ndarray,
    area: np.ndarray,
    dR_p_dth,
    flags: StressConstraintFlags = PRODUCTION_STRESS_FLAGS,
    simp=None,
    verbose: bool = False,
    factor=None,
    elimination=None,
    stress_smooth=None,
) -> MechStressResult:
    """批量计算应力 / 应变能的诊断量与密度导数。与 ``reference`` 版本**逐位**相同。

    对应 ``MMA10:2437-2601``。参数与返回见 ``reference.mech_stress_and_energy``。

    **D6 修正（2026-09-23）**：``dU_p_drho`` 改用包络闭式（免伴随），与
    ``reference`` 共用 ``strain_energy_derivative_envelope`` ⇒ 这一支**无
    spsolve**，只剩 ``d_al_stress_p_drho`` 一个伴随解（``factor`` 只服务它）。
    """
    from scipy.sparse import linalg as sp_sparse_linalg

    if simp is None:
        simp = _mech_simp()

    tri = np.asarray(tri, dtype=int)
    n_elem = tri.shape[0]
    n_dof = u_node.shape[0]
    rho = np.asarray(rho, dtype=float)

    # ---- 自由度表（**未替换**的连接表，见 reference 第一节）----
    dof = np.empty((n_elem, 6), dtype=np.int64)
    dof[:, 0::2] = 2 * tri
    dof[:, 1::2] = 2 * tri + 1

    # ---- 逐单元的 ε / σ / σ̄ ----
    # ⚠ **必须用 ``@``（matmul）而不是 ``np.einsum``**：实测 einsum 与逐单元
    #   ``l_coef[e] @ u_e`` 差 ~3e-16（求和顺序不同），而批量的 ``@`` **逐位相同**。
    #   本层要位级对齐，所以这一处没有选择余地。
    epsilon = (l_coef @ u_node[dof][..., None])[..., 0]
    # ⚠ 逐单元标量调用在 legacy 是 ``simp.project(rho[e])``；批量调用的结果逐位相同
    #   （SIMP 投影是逐元素的纯代数运算）
    k_e = simp.project(rho)
    dk_e = simp.derivative(rho)
    sigma = k_e[:, None] * (d_mat @ epsilon[..., None])[..., 0]
    vm_raw = von_mises(sigma)
    # 应力平滑（2026-10-20，可选）：用固定物理半径的滤波矩阵 A 把逐单元 σ̄
    # 平滑后再进 AL 约束。动机：峰值 von Mises 在尖角处是**奇异的**，随网格加密
    # 发散（实测同一设计 6s4p 82 → d030 187 → d015 303 MPa），而设计单元上的
    # 面积加权均值稳定（82.4 → 80.7 → 87.6）。A=None ⇒ 逐位走原路径。
    vm = vm_raw if stress_smooth is None else np.asarray(
        stress_smooth @ vm_raw, dtype=float).ravel()

    # ---- 应变能（顺序累加 + 逐单元的累计值）----
    # ⚠ D6 已修：``uelastic_cum`` 仍要（``elastic_strain_energy`` 的值是累计），
    #   但 ``d_u_p_du`` 不再用它 —— 恢复为纯 ``∂U/∂u``（去掉 legacy :2484 的
    #   广播显式项），显式项由包络闭式承担。
    u_e = area * (epsilon * sigma).sum(axis=1) / 2
    uelastic_cum = np.cumsum(u_e)

    d_u_p_du = np.zeros(n_dof)
    np.add.at(d_u_p_du, dof.ravel(),
              (area[:, None] * (l_coef.transpose(0, 2, 1) @ sigma[..., None])[..., 0]).ravel())

    # ---- 逐单元的 ∂σ̄/∂u（**D8 已修 2026-09-23**，见 §12）----
    # D8 修正第一项：``∂σ̄/∂u = k_E/(2σ̄)·L_eᵀ @ (Dᵀ @ v)``，v = [2σx−σy, ...]
    # （legacy :2487-2491 缺 ``k_E/(2σ̄)`` 这个标量因子 —— 实测差 ~1e11 倍）
    dvm_dsigma = np.stack([
        2 * sigma[:, 0] - sigma[:, 1],
        -sigma[:, 0] + 2 * sigma[:, 1],
        6 * sigma[:, 2],
    ], axis=1)
    _d_sigma_d_u = (d_mat.T @ dvm_dsigma[..., None])[..., 0]
    _d_sigma_p_du = (l_coef.transpose(0, 2, 1) @ _d_sigma_d_u[..., None])[..., 0]
    # ⚠ ``1/(2σ̄)`` 在 σ̄=0 处发散 —— 但活跃单元必有 σ̄ ≥ σ_0·k_E^p > 0，
    #   而 ``weight = 0`` 会把它乘成 ``0·inf = nan`` 再散进全局向量 ⇒ 必须就地置 0。
    # ⚠ 这里必须是**原始** σ̄（∂σ̄/∂u 的链式因子），不是平滑后的值
    scale = np.divide(k_e, 2 * vm_raw, out=np.zeros_like(k_e),
                      where=(vm_raw > 0))
    d_sigma_p_du = scale[:, None] * _d_sigma_p_du

    # ---- 六支的 Λ / g / h / Pi / k1 / k2（批量）----
    p_exp = flags.rel_stress_p_exp
    if not flags.use_rel_stress_constraint:
        lam = vm / flags.max_stress_constraint - 1
    elif flags.use_type1_rel_stress:
        lam = vm / k_e / flags.max_rel_stress_constraint - 1
    elif flags.use_type2_rel_stress:
        lam = vm * rho / k_e / flags.max_rel_stress_constraint - 1
    elif flags.use_type3_rel_stress:
        lam = vm / k_e ** p_exp / flags.max_rel_stress_constraint - 1
    else:
        lam = vm * rho / k_e ** p_exp / flags.max_rel_stress_constraint - 1

    g = k_e * (lam ** 3 + lam)
    if (flags.use_type2_stress or flags.use_type3_rel_stress
            or flags.use_type4_rel_stress):
        g = lam ** 3 + lam

    h = np.maximum(g, -al_lambda / al_mu)
    pi_e = al_lambda * h + al_mu / 2 * h ** 2
    stress_loss_al = float(np.cumsum(pi_e / n_elem)[-1])

    active = g * al_mu + al_lambda > 0

    k1, k2 = _weights_vectorized(vm, lam, g, k_e, dk_e, rho, flags)

    d_al_stress_pp_du = np.zeros(n_dof)
    # ⚠ 乘法**结合顺序**照抄 legacy（``:2540``）：``1/N * (λ+μg) * k1 * dSigma`` ——
    #   先除 N，不是最后除。写成 ``(λ+μg)*k1*dSigma/N`` 会差 1 ULP（实测 1.8e-16）。
    weight = np.where(active, 1 / n_elem * (al_lambda + al_mu * g) * k1, 0.0)
    # 平滑口径：约束作用在 (A·σ̄) 上 ⇒ ∂/∂u 的系数要先经 Aᵀ 回到原始单元
    if stress_smooth is not None:
        weight = np.asarray(stress_smooth.T @ weight, dtype=float).ravel()
    np.add.at(d_al_stress_pp_du, dof.ravel(),
              (weight[:, None] * d_sigma_p_du).ravel())

    d_al_stress_pp_drho = np.zeros(n_elem)
    # D8 修正第二项：补 ``k1·∂σ̄/∂k_E·k_E'``（σ̄ 经 σ = k_E·D·ε 直接依赖 k_E）。
    #   ∂σ̄/∂k_E = vᵀ·(D·ε)/(2σ̄)。⚠ 与 reference 的 ``np.dot`` 加法顺序必须一致
    #   （``np.dot`` 走 BLAS ddot，与 ``(a*b).sum()`` 可能差末位）—— 用 einsum 固定
    #   成与 reference 同型的成对求和？见 C 级测试；此处与 reference 逐位对齐。
    # ⚠ 同样用原始 σ̄（∂σ̄/∂k_E）
    dvm_dk = (dvm_dsigma * (d_mat @ epsilon[..., None])[..., 0]).sum(axis=1) / (2 * vm_raw)
    dvm_dk = np.where(vm_raw > 0, dvm_dk, 0.0)
    if stress_smooth is None:
        d_al_stress_pp_drho[active] = (
            1 / n_elem * (al_lambda[active] + al_mu * g[active])
            * (k2[active] + k1[active] * dvm_dk[active] * dk_e[active]))
    else:
        # 平滑口径的链式法则：
        #   ∂π/∂ρ_j = w_j·k2_j（k_E 的局部项）
        #           + (Aᵀ(w·k1))_j · ∂σ̄_j/∂k_E · k_E'（σ̄ 经 A 传播的那一支）
        w_mu = np.where(active, 1 / n_elem * (al_lambda + al_mu * g), 0.0)
        w_k1 = np.asarray(stress_smooth.T @ (w_mu * k1), dtype=float).ravel()
        d_al_stress_pp_drho = w_mu * k2 + w_k1 * dvm_dk * dk_e

    al_loss_e = al_lambda ** 2
    al_loss_e[active] = g[active] ** 2
    al_loss = float(np.sqrt(np.cumsum(al_loss_e)[-1]))

    # D7 修正（2026-09-23）：``min`` → ``max``（见 reference 本处的注释）。legacy 的
    # ``min(0, λ+μh)`` 让 ``λ_new ≡ 0``（D7 可证明）；正确是 ``max(0, λ+μh)``。
    al_lambda_new = np.maximum(0.0, al_lambda + al_mu * h)

    von_mises_max = float(np.max(vm))

    k_e_all = simp.project(rho)
    if flags.use_type2_rel_stress:
        rel = vm * rho / k_e_all
    elif flags.use_type3_rel_stress:
        rel = vm / k_e_all ** p_exp
    elif flags.use_type4_rel_stress:
        rel = vm * rho / k_e_all ** p_exp
    else:
        rel = vm / k_e_all
    rel_stress_max = float(np.max(rel))

    if verbose:
        print(f'AL_stress_loss={al_loss:.4g}, AL_stress_mu={al_mu:.4g}, '
              f'VonMisesStress_max={von_mises_max:.4g}, '
              f'RelStress_max={rel_stress_max:.4g}')

    K_csr = K_mech.tocsr()
    # 阶段 8 分解复用：AL 罚项那一支的伴随解用 K 的转置因子。默认 None 走原
    # spsolve；传入时回代 —— 两者逐位一致（实测，见设计文档阶段 8）。
    # ⚠ D6 已修：`d_u_p_drho` 不再走伴随解，改用包络闭式（与 reference 共用
    #   `strain_energy_derivative_envelope`，免伴随、无 spsolve）⇒ factor 只服务
    #   `d_al_stress_p_drho`。
    # ⚠ 消元（主从，2026-09-24）：factor 解 (TᵀKT)ᵀ；右端 Tᵀ 收缩、λ 经 T 恢复
    #   后再与完整 dRHS 收缩（包络闭式 dU_p_drho 用完整 u/K，不受消元影响）。
    if elimination is not None:
        lam_free = (factor.solve(elimination.eliminate_rhs(d_al_stress_pp_du))
                    if factor is not None
                    else sp_sparse_linalg.spsolve(
                        elimination.eliminate_matrix(K_csr).T,
                        elimination.eliminate_rhs(d_al_stress_pp_du)))
        lam_1 = elimination.restore(lam_free)
    else:
        lam_1 = (factor.solve(d_al_stress_pp_du) if factor is not None
                 else sp_sparse_linalg.spsolve(K_csr.T, d_al_stress_pp_du))
    d_al_stress_p_drho = (-lam_1.T @ dRHS_mech_p_dth + d_al_stress_pp_drho)
    d_u_p_drho = strain_energy_derivative_envelope(
        u_node, K_mech, dRHS_mech_p_dth, dR_p_dth,
        n_dof=n_dof, n_elem=n_elem)

    return MechStressResult(
        von_mises_max=von_mises_max,
        rel_stress_max=rel_stress_max,
        stress_loss_al=stress_loss_al,
        al_loss=al_loss,
        d_al_stress_p_drho=np.asarray(d_al_stress_p_drho),
        elastic_strain_energy=float(np.cumsum(u_e)[-1]),
        d_u_p_drho=np.asarray(d_u_p_drho),
        al_lambda_new=al_lambda_new,
        von_mises=vm,
        sigma=sigma,
        d_al_stress_pp_du=d_al_stress_pp_du,
        d_al_stress_pp_drho=d_al_stress_pp_drho,
        d_u_p_du=d_u_p_du,
    )


def _weights_vectorized(vm, lam, g, k_e, dk_e, rho, flags):
    """``(k1, k2)`` 的批量版。逐支与 ``reference._derivative_weights`` 同式。"""
    p_exp = flags.rel_stress_p_exp

    if not flags.use_rel_stress_constraint:
        if flags.use_type1_stress:
            k1 = k_e * (1 + 3 * lam ** 2) / flags.max_stress_constraint
            k2 = g / k_e * dk_e
        else:
            k1 = (1 + 3 * lam ** 2) / flags.max_stress_constraint
            k2 = np.zeros_like(k1)
        return k1, k2

    if flags.use_type1_rel_stress:
        k1 = (3 * lam ** 2 + 1) / flags.max_rel_stress_constraint
        k2 = (g / k_e - (3 * lam ** 2 + 1) * vm
              / flags.max_rel_stress_constraint / k_e) * dk_e
    elif flags.use_type2_rel_stress:
        k1 = rho * (3 * lam ** 2 + 1) / flags.max_rel_stress_constraint
        k2 = (g / k_e * dk_e
              + (3 * lam ** 2 + 1) * vm / flags.max_rel_stress_constraint
              * (1 - rho / k_e * dk_e))
    elif flags.use_type3_rel_stress:
        k1 = 1 / k_e ** p_exp * (3 * lam ** 2 + 1) / flags.max_rel_stress_constraint
        k2 = (-p_exp * (3 * lam ** 2 + 1) * vm
              / flags.max_rel_stress_constraint / k_e ** (p_exp + 1)) * dk_e
    else:
        k1 = (rho / k_e ** p_exp * (3 * lam ** 2 + 1)
              / flags.max_rel_stress_constraint)
        k2 = ((3 * lam ** 2 + 1) * vm / flags.max_rel_stress_constraint
              * (1 / k_e ** p_exp - p_exp * rho / k_e ** (p_exp + 1)))
    return k1, k2
