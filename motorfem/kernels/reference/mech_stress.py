"""应力与应变能导数（von Mises / 增广拉格朗日罚项）（内核 17）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）
    ``compute_mechDiff_AL``  ``:2437-2601``

数学出处：``docs/笔记/SynRM拓扑优化方法的笔记.md``
  - 「原文中给出的方法」（``:237``）—— 绝对应力约束（对应 ``use_type1_stress``）
  - 「将原文方法中的应力约束更改为相对应力约束」（``:293``）—— 两种相对应力定义
    （对应 ``use_type1_rel_stress`` / ``use_type2_rel_stress``）
  - 「将原文方法改为单元弹性应变能的约束」（``:343``）—— 应变能与 ``Λ``、``g`` 的关系，
    以及 ``∂U_i/∂θ`` 的两项结构

------------------------------------------------------------------------------
一、这个内核算什么
------------------------------------------------------------------------------

输入是内核 16 解出的位移 ``u``，输出是六个量：

| 输出 | 含义 |
|---|---|
| ``VonMisesStress_max`` | 全场最大等效应力（约束的诊断量） |
| ``Stress_loss_AL`` | 应力约束的 AL 罚项总量 ``Σ P_i / N`` |
| ``dALStress_p_drho`` | 罚项对密度的导数（**已经过伴随求解**） |
| ``Uelastic`` | 总弹性应变能 |
| ``dU_p_drho`` | 应变能对密度的导数（**已经过伴随求解**） |
| ``AL_lambda_new`` | 更新后的拉格朗日乘子 |

**逐单元的三段**（每一段都独立可验）：

1. **物理量**：``ε_e = L_e u_e``、``σ_e = kE_e·D ε_e``、
   ``σ̄_e = √(3σ_xy² + σ_x² + σ_y² − σ_x σ_y)`` —— 这就是笔记里**应力约束**的左端。
2. **AL 罚项**（笔记 :245-255 与 :284-289 的公式，逐字）::

       Λ_e = (约束量)/σ_0 − 1          ← 约束量的定义由 ``flags`` 选（见第三节）
       g_e = kE_e·(Λ_e³ + Λ_e)         ← 或 Λ_e³ + Λ_e，由 ``flags`` 选
       h_e = max(g_e, −λ_e/μ)
       P_e = λ_e·h_e + μ/2·h_e²
       λ_e ← min(0, λ_e + μ·h_e)

   ``μ`` 在本内核**不变**（它的更新属内核 19）。
3. **导数**（笔记的三条敏感度公式，见第三节）::

       ∂P_e/∂θ = (g_e + λ_e/μ > 0)·(λ_e + μ·g_e)·[ k1_e·∂σ̄_e/∂u + k2_e ]
       dU/dρ 的构造见第四节

   其中 ``k1``/``k2`` 是 ``flags`` 选出来的两族系数 —— **这两族系数就是本内核的全部「数学」**，
   每一条都能在笔记里找到对应的式子（对照表在第三节）。

**注意本内核与内核 16 的一处形状差异**：应力的位移索引用**未替换**的连接表
（``element_S_mech[i_elem][1:4]``，``:2462-2465``）—— 从属节点用它**自己的**位移，
而不是被消元后的主动节点位移。两者在解上一致（约束行保证 ``u_s = R·u_m``），
但读代码时不要以为内核 17 也走 ``T_e``。

------------------------------------------------------------------------------
二、六个互斥布尔 → 一个参数对象（**不改数值**）
------------------------------------------------------------------------------

legacy 把约束形式写成六个模块级布尔（``:112-122``）::

    use_type1_stress / use_type2_stress                 ← 绝对应力约束的两支
    use_type1..4_rel_stress                             ← 相对应力约束的四支
    use_rel_stress_constraint                           ← 决定走哪一族

本模块把**同样的六个布尔 + 三个常数**收进 ``StressConstraintFlags``（字段名与 legacy 的
全局量**逐字相同**），``PRODUCTION`` 复刻生产配置。这是「显式签名」的最低风险做法：
**不动分支结构**，只把全局量变成参数。设计文档 §5 阶段 2 计划的「折成两个枚举字段」
是一次**结构性**改动，且要求基线**覆盖每个分支至少一次** —— 基线还没冻结，所以现在不做。

⚠ ``max_rel_stress_constraint`` 在 legacy 是 ``max_stress_constraint * 0.8``（``:122``），
本模块把它作为**独立字段**（避免依赖字段求值顺序），``PRODUCTION`` 里写的就是那个乘积。

------------------------------------------------------------------------------
三、六支的系数对照表（本内核唯一需要小心的地方）
------------------------------------------------------------------------------

``Λ`` 的定义（约束量）：

+-------------------------------+--------------------------------------+
| flags                         | ``Λ_e``                              |
+===============================+======================================+
| ``use_type1_stress``          | ``σ̄/σ_0 − 1``                        |
| ``use_type2_stress``          | 同上（差别只在 ``g`` 与 ``k1/k2``）  |
| ``use_type1_rel_stress``      | ``σ̄/(σ_0·k_E) − 1``                  |
| ``use_type2_rel_stress``      | ``σ̄·ρ/(σ_0·k_E) − 1``                |
| ``use_type3_rel_stress`` ★    | ``σ̄/(σ_0·k_E^p) − 1``                |
| ``use_type4_rel_stress``      | ``σ̄·ρ/(σ_0·k_E^p) − 1``              |
+-------------------------------+--------------------------------------+

★ = 生产路径（``use_rel_stress_constraint=True`` + ``use_type3_rel_stress=True``）。

``g`` 的形式（``:2510-2512``）::

    g_e = kE_e·(Λ³+Λ)      默认
    g_e = Λ³+Λ             若 use_type2_stress 或 use_type3/4_rel_stress 为 True

⚠ **这条判据跨了两族**：``use_type2_stress``（绝对那族）也参与决定相对那族的 ``g``。
这是 legacy 的形态（生产配置里两个都为 True，所以看不出耦合）—— **照搬**，
并在 ``test_...`` 里用一个「只开 rel1」的配置把它固化下来。**注意 ``g`` 的 ``kE`` 因子
与 ``Λ`` 里的 ``kE`` 是两件事**。

``k1``（乘 ``∂σ̄/∂u``）与 ``k2``（显式 ``∂k_E/∂θ`` 项）逐支：

+---------------------------+--------------------------------------------+------------------------------------------------------------------+
| flags                     | ``k1``                                     | ``k2``                                                           |
+===========================+============================================+==================================================================+
| ``use_type1_stress``      | ``kE(1+3Λ²)/σ_0``                          | ``g/kE·dkE``                                                     |
| ``use_type2_stress``      | ``(1+3Λ²)/σ_0``                            | ``0``                                                            |
| ``use_type1_rel_stress``  | ``(1+3Λ²)/σ_0``                            | ``(g/kE − (1+3Λ²)σ̄/σ_0/kE)·dkE``                                 |
| ``use_type2_rel_stress``  | ``ρ(1+3Λ²)/σ_0``                           | ``g/kE·dkE + (1+3Λ²)σ̄/σ_0·(1 − ρ/kE·dkE)``                       |
| ``use_type3_rel_stress`` ★| ``(1+3Λ²)/(σ_0 kE^p)``                     | ``−p(1+3Λ²)σ̄/σ_0/kE^(p+1)·dkE``                                  |
| ``use_type4_rel_stress``  | ``ρ(1+3Λ²)/(σ_0 kE^p)``                    | ``(1+3Λ²)σ̄/σ_0·(1/kE^p − p·ρ/kE^(p+1))``                         |
+---------------------------+--------------------------------------------+------------------------------------------------------------------+

（``σ_0`` = ``max_stress_constraint`` 或 ``max_rel_stress_constraint``，按族取。）

**与笔记的对应**（逐条核过）：

- ``use_type1_stress`` = 笔记 :268-269 的原文公式（``∂kE/∂θ·g/kE + kE(1+3Λ²)/σ_0·∂σ̄/∂θ``）
  ✓ 两项分别对上 ``k2`` 与 ``k1``；
- ``use_type1_rel_stress`` = 笔记 :314-316 ✓；
- ``use_type2_rel_stress`` = 笔记 :338-340 ✓；
- ``use_type3_rel_stress`` = 笔记 :314-316 的 ``σ̄_r = σ̄/kE`` 推广到 ``kE^p``
  （``k2`` 里的 ``−p/kE^(p+1)`` 就是 ``∂kE^{-p}/∂θ``）✓；
- ``use_type4_rel_stress`` = 笔记 :338-340 的推广（``ρ/kE^p``）✓ —— 注意它的 ``k2``
  **不含** ``dkE``，因为显式 ρ 依赖已在 ``(1/kE^p − pρ/kE^{p+1})`` 里；
- ``use_type2_stress`` = 笔记 :360-364 的**应变能约束**的 ``g`` 形式（``Λ³+Λ``，无 ``kE``）
  配上应力定义的 ``Λ``，且 ``k2 = 0`` ⇒ 它**丢掉了**笔记 :377 的 ``U_i/kE·∂kE/∂θ`` 那一项。
  本模块照搬（登记为「有意为之还是笔误」待核对，见设计文档 §11）。

------------------------------------------------------------------------------
四、缺陷状态叠在这个内核里（**D5/D6/D7/D8 已修 2026-09-23**）
------------------------------------------------------------------------------

**D5（内核 16）**：**已修（2026-09-23，§12 修正清单第 1 条）**。`mech_sensitivity`
返回 ``dK·u + dR``（legacy 曾是 ``dK·u − dR``）。本内核的 ``dALStress_p_drho``
随之消掉 ``2λᵀdR`` 那支误差。

**D6（应变能导数的显式项，本内核）**：**已修（2026-09-23）**。修正方式是**落地
包络闭式**（§8b 判定 + 收尾 #2 的 ``strain_energy_derivative_envelope``）：::

    dU/dρ_e = −½·uᵀ(dK/dρ_e)u − uᵀ(dR/dρ_e)

（由平衡 ``Ku = −R`` 消 ``du/dρ``；免伴随，不 ``spsolve``）。D5 修后
``S = dK·u + dR``，且 ``dR`` 的行 ⊆ ``dK·u`` 的行 ⇒ 精确算术下展开为
``dU/dρ_e = −½·uᵀS[:,e] − ½·uᵀdR[:,e]``（两个稀疏 matvec，见
``strain_energy_derivative_envelope`` 的 docstring）。legacy 的两处错
（显式项折进 ``dU_p_du`` 的 6 个局部自由度 + 用累计 ``Uelastic``）随闭式落地
一并消失；``dU_p_du`` 恢复为**纯** ``∂U/∂u = Δ_e·(L_eᵀσ_e)``（仍是伴随右端
的诊断量，不再参与 ``dU_p_drho``）。闭式 vs FD：真实网格 ≤8.6e-8
（``probe_envelope.py``）、夹具 rtol 1e-4。详见 §12 的 D6 修正清单。

**D7（AL 乘子更新恒为 0，本内核）**：**已修（2026-09-23）**。legacy
``AL_lambda_new = min(0, λ + μh)`` 而 ``h = max(g, −λ/μ)`` 恒有 ``λ + μh ≥ 0``
⇒ ``λ_new ≡ 0``（**可证明**）⇒ 增广拉格朗日退化为纯罚函数 ``P = μ/2·max(g,0)²``。
⚠ 笔记 :289 写的是同一个式子 —— **文档与代码一致**，属**公式层面**的问题。

**D7 推导（修正依据）**：正确更新是 ``λ_new = max(0, λ + μh)``（即
``max(0, λ + μg)``：``h = max(g, −λ/μ) ⇒ λ+μh = max(λ+μg, 0)``）。这是光滑 AL
``P = λh + μ/2·h²`` 的 method-of-multipliers 更新 —— 活跃（``g > −λ/μ``）时
``λ_new = λ + μg`` 驱动乘子向 KKT 乘子收敛；不活跃时 ``λ_new = 0``（互补性）。
legacy 的 ``min`` 是**投影方向错**（min vs max）。修后 λ ≥ 0 携带历史信息，
AL 才是真增广拉格朗日；``AL_mu`` 不再是唯一的约束力来源。⚠ **mu 调度（内核 19）
暂保留 legacy 值**（0.01→1e3，min(μ0·(1+n/step)², μ_max)）—— 重标定留给
优化器收敛确认后（见 §12 的 D7 修正清单第 ④ 条）。见 §12 的 D7。

**D8（AL 罚项的密度导数差 ~1e11 倍，本内核）**：**已修（2026-09-23）**。
实测（见 §12 的 D8）legacy 与有限差分之比 **~1e11**。判定后机制是**两个缺失项**：

1. ``dSigma_p_du`` 缺标量因子 ``k_E/(2σ̄)``：正确是
   ``∂σ̄/∂u = (∂σ̄/∂σ)ᵀ·k_E·D·L = k_E/(2σ̄)·LᵀDᵀv``（``v = [2σx−σy, −σx+2σy, 6σxy]``），
   legacy 的 ``LᵀDᵀv`` 少了 ``k_E/(2σ̄)``。注：设计文档 §12 曾把「转置位置不同」
   （``LᵀDᵀv`` vs ``vᵀDL``）列为嫌疑之一 —— 对**标量** ``σ̄`` 两者是同一个列向量
   （转置后相等），不是差异。
2. 显式 ``dALStress_pp_drho`` 缺 ``k1·∂σ̄/∂k_E·k_E'``：σ̄ 经 ``σ = k_E·D·ε``
   **直接**依赖 ``k_E``（u 冻结时也有 ``∂σ̄/∂k_E = vᵀ·(D·ε)/(2σ̄)``）。legacy 的
   ``k2`` 只含 ``(3Λ²+1)·∂Λ/∂k_E·k_E'`` 那一支。有限差分会捕获这条路径。

验证：真实网格（收紧上限 1e8、活跃 2644/6362）修正/FD = **1.0000**（5 个抽样单元）；
∂σ̄/∂u 层面逐分量 FD 全 1.0000（6 单元 × 6 分量）；显式项用「u 冻结」FD 单独验过。
修正只在活跃单元上改变 ``dALStress_p_drho``；罚项**值**不变（无活跃单元的生产配置
输出逐位不变）。

⚠ 四处缺陷（D5~D8）里有三处（D5/D6/D8）都经由 ``−spsolve(Kᵀ, ·)ᵀ @ dRHS`` 这条路，
验收时**必须分开归因**（D8 的修正落在 ``dALStress_pp_du`` 的右端**与**显式项，见上）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from .simp import SIMPParams, SIMPProjection

__all__ = [
    "StressConstraintFlags",
    "PRODUCTION_STRESS_FLAGS",
    "MechStressResult",
    "von_mises",
    "element_stress",
    "strain_energy_derivative_envelope",
    "mech_stress_and_energy",
]


@dataclass(frozen=True)
class StressConstraintFlags:
    """legacy 的六个互斥布尔 + 三个常数（``MMA10:110-122``）。

    字段名与 legacy 的模块级全局量**逐字相同**（便于逐条对照）。
    见模块 docstring 第二节：这是「显式签名」的最低风险做法，
    **不动分支结构**；折成枚举留给阶段 2。

    ⚠ 生产配置是 ``use_type2_stress=True`` **且** ``use_type3_rel_stress=True``：
    前者的值参与决定 ``g`` 的形式（``:2511``），即使走的是相对应力那一族。
    """

    use_rel_stress_constraint: bool
    use_type1_stress: bool
    use_type2_stress: bool
    use_type1_rel_stress: bool
    use_type2_rel_stress: bool
    use_type3_rel_stress: bool
    use_type4_rel_stress: bool
    max_stress_constraint: float
    max_rel_stress_constraint: float
    rel_stress_p_exp: float

    def __post_init__(self) -> None:
        if self.use_rel_stress_constraint:
            chosen = [self.use_type1_rel_stress, self.use_type2_rel_stress,
                      self.use_type3_rel_stress, self.use_type4_rel_stress]
        else:
            assert self.use_type1_stress or self.use_type2_stress, (
                "绝对应力约束必须二选一（legacy :2523 / :2543 有同样的 assert）"
            )
            chosen = [self.use_type1_stress, self.use_type2_stress]
        if sum(bool(c) for c in chosen) != 1:
            raise ValueError(
                f"约束形式必须**恰好选一支**，实得 {sum(bool(c) for c in chosen)} 支："
                f"{chosen}。legacy 用 if/elif 链，多开一个不会报错而是静默走第一个分支。"
            )


#: 生产配置（照抄 ``MMA10:110-122``）
PRODUCTION_STRESS_FLAGS = StressConstraintFlags(
    use_rel_stress_constraint=True,
    use_type1_stress=False,
    use_type2_stress=True,
    use_type1_rel_stress=False,
    use_type2_rel_stress=False,
    use_type3_rel_stress=True,
    use_type4_rel_stress=False,
    max_stress_constraint=50e7,
    max_rel_stress_constraint=50e7 * 0.8,
    rel_stress_p_exp=1 / 2,
)


@dataclass(frozen=True)
class MechStressResult:
    """``mech_stress_and_energy`` 的产出（legacy 返回 6 个值，这里加诊断量）。

    Attributes
    ----------
    von_mises_max, rel_stress_max : float
        ``:2578`` 与 ``:2582-2588`` 的两个诊断量（后者按族选定义）。
    stress_loss_al : float
        ``Σ_e P_e / N``（``:2517``）。
    al_loss : float
        ``sqrt(Σ_e [g_e² 或 λ_e²])``（``:2570-2572`` 与 ``:2589``）—— legacy
        **只打印不返回**，本模块返回以便测试。
    d_al_stress_p_drho : (N_elem,) ndarray
    elastic_strain_energy : float
    d_u_p_drho : (N_elem,) ndarray
    al_lambda_new : (N_elem,) ndarray
    von_mises : (N_elem,) ndarray
        ``:2478-2480``；legacy 存了但只用来取 ``max`` 与算 ``Λ``。
    sigma : (N_elem, 3) ndarray
        ``:2475``。
    d_al_stress_pp_du : (2·N_node_mech,) ndarray
        伴随右端（``:2540`` 累加出来的那个向量）。**legacy 不返回它**；
        本模块返回是为了让「D5 的修正量」可以被分解验证（测试与探针用）。
    d_al_stress_pp_drho : (N_elem,) ndarray
        直接加上的显式项（``:2568``）。同样是为了可分解。
    d_u_p_du : (2·N_node_mech,) ndarray
        应变能的伴随右端 —— **D6 修后是纯** ``∂U/∂u``（逐单元 ``Δ_e·(L_eᵀσ_e)``，
        legacy ``:2484`` 的广播显式项已去掉）。它仍是诊断量，**不再**参与
        ``d_u_p_drho``（后者改用包络闭式，免伴随）。
    """

    von_mises_max: float
    rel_stress_max: float
    stress_loss_al: float
    al_loss: float
    d_al_stress_p_drho: np.ndarray
    elastic_strain_energy: float
    d_u_p_drho: np.ndarray
    al_lambda_new: np.ndarray
    von_mises: np.ndarray
    sigma: np.ndarray
    d_al_stress_pp_du: np.ndarray
    d_al_stress_pp_drho: np.ndarray
    d_u_p_du: np.ndarray


def von_mises(sigma: np.ndarray) -> np.ndarray:
    """逐单元 von Mises 等效应力。对应 ``MMA10:2478-2480``::

        σ̄ = √(3σ_xy² + σ_x² + σ_y² − σ_x σ_y)

    Parameters
    ----------
    sigma : (N, 3) 或 (3,) ndarray
        ``(σ_x, σ_y, σ_xy)``。

    Returns
    -------
    与输入前导维相同的数组

    Notes
    -----
    这个二次型的矩阵形式是 ``[[1, −1/2, 0], [−1/2, 1, 0], [0, 0, 3]]``
    （内核 16 的 ``von_mises_map`` 就是它）。legacy 把它**展开**写成三项之和
    （且中间两项的顺序是 ``σ_x²+σ_y²−σ_xσ_y`` 里最后一项取负）——
    本函数照抄那个展开式的**项序**，因为浮点加法不可交换。
    """
    sigma = np.asarray(sigma)
    return np.sqrt(3 * sigma[..., 2] ** 2 + sigma[..., 0] ** 2
                   + sigma[..., 1] ** 2 - sigma[..., 0] * sigma[..., 1])


def element_stress(
    u_node: np.ndarray,
    tri: np.ndarray,
    l_coef: np.ndarray,
    d_mat: np.ndarray,
    k_e: np.ndarray,
) -> np.ndarray:
    """逐单元应力 ``σ_e = kE_e·(D·L_e·u_e)``。对应 ``MMA10:2470-2475``。

    ⚠ 位移索引用**未替换**的连接表 —— 从属节点用自己的位移（见模块 docstring 第一节）。

    Returns
    -------
    (N_elem, 3) ndarray
    """
    n_elem = tri.shape[0]
    sigma = np.empty((n_elem, 3))
    for e in range(n_elem):
        index = [2 * int(tri[e, 0]), 2 * int(tri[e, 0]) + 1,
                 2 * int(tri[e, 1]), 2 * int(tri[e, 1]) + 1,
                 2 * int(tri[e, 2]), 2 * int(tri[e, 2]) + 1]
        epsilon = l_coef[e] @ u_node[index]
        sigma[e] = k_e[e] * (d_mat @ epsilon)
    return sigma


def strain_energy_derivative_envelope(
    u_node: np.ndarray,
    K: sp_sparse.csr_matrix,
    dK_u_plus_dR: sp_sparse.coo_matrix,
    dR: sp_sparse.coo_matrix,
    *,
    n_dof: int,
    n_elem: int,
) -> np.ndarray:
    """弹性应变能的**免伴随闭式**导数（阶段 8 的 8b 判定 → **D6 修正落地**）。

    对应设计文档 §阶段 8 的 8b 判定：``U = ½uᵀKu`` 是能量泛函类，由平衡
    ``Ku = −R`` 消掉 ``du/dρ`` 得包络定理闭式::

        dU/dρ_e = −½·uᵀ(dK/dρ_e)u − uᵀ(dR/dρ_e)

    Parameters
    ----------
    u_node : (2N,) 平衡位移（``solve_mech_field`` 的解）
    K : (2N, 2N) csr —— 与 ``mech_residual`` 的 K 同一个
    dK_u_plus_dR : (2N, N_elem) coo —— ``mech_sensitivity`` 的灵敏度
        ``S = dK/dρ·u + dR/dρ``（**D5 已修**，加号）。
    dR : (2N, N_elem) coo —— ``∂R/∂ρ_e``（``mech_worker`` 的
        ``dR_p_dth = coo(r0_data, …)``，``dk_rho=1``）
    n_dof, n_elem : int

    Returns
    -------
    (N_elem,) ndarray —— ``dU/dρ`` 的免伴随闭式（**无 spsolve**）。

    Notes
    -----
    ⚠ **为什么可以这样算（关键，落地后的生产形态）**：
    - 由平衡 ``Ku = −R`` 对 ρ_e 求导：``K du/dρ_e + (dK/dρ_e)u = −dR/dρ_e``
      ⇒ ``uᵀK du/dρ_e = −uᵀ(dK/dρ_e)u − uᵀ(dR/dρ_e)``。代入
      ``dU/dρ_e = uᵀK du/dρ_e + ½uᵀ(dK/dρ_e)u`` 即得闭式。
    - **D5 修后的展开**：``S = dK·u + dR``，且 ``dR`` 的行 ⊆ ``dK·u`` 的行
      （两者都是该单元的 6 个自由度行）⇒ 精确算术下
      ``uᵀ(dK/dρ_e)u = uᵀ(S[:,e] − dR[:,e])``，于是::

          dU/dρ_e = −½·uᵀS[:,e] − ½·uᵀdR[:,e]

      每个点积用**稀疏 matvec**（``(S − dR).T @ u``、``dR.T @ u``）一次算完
      所有列 —— 免伴随、无需 ``spsolve``、无稠密中间量。
    - **与 D6-fixed 伴随的等价**：``∂U/∂u = Σ_e Δ_e·(L_eᵀσ_e) = Ku``，
      ``λ = K⁻ᵀ·∂U/∂u = u`` ⇒ ``dU/dρ_e = −λᵀS + Σ_e U_e/kE_e·dkE_e``
      ``= −uᵀ(dK·u + dR) + ½·uᵀ(dK/dρ_e)u = −½·uᵀ(dK/dρ_e)u − uᵀ(dR/dρ_e)``
      —— 与闭式逐项相同（精确算术）。这是 D6 修正清单一与二「显式项直接加、
      用单元 U_e」的闭式形态；legacy 的伴随解与累计 ``Uelastic`` 两处错
      （§12 D6）随落地一并消除。
    - **与 legacy 的关系**：legacy ``dU_p_drho`` 走伴随解且含 D6（实测
      FD/legacy≈0.004，即 legacy/闭式≈250 倍，逐单元变）；本闭式正是
      **D6 修正的落地实现**（阶段 8 收尾 #2 起为验证层，2026-09-23 接线生产）。
    """
    u = np.asarray(u_node, dtype=float)
    S = sp_sparse.coo_matrix(dK_u_plus_dR).tocsr()
    dR_csr = sp_sparse.coo_matrix(dR).tocsr()
    dK_u = S - dR_csr                       # (dK/dρ_e)u 逐列（D5 修后：S − dR）
    u_dK = dK_u.T @ u                       # (N_elem,)  uᵀ(dK/dρ_e)u
    u_dR = dR_csr.T @ u                     # (N_elem,)  uᵀ(dR/dρ_e)
    return -0.5 * u_dK - u_dR


def _constraint_variable(
    von_mises_e, k_e, rho_e, flags: StressConstraintFlags
):
    """``Λ_e`` —— 约束量的定义（六支）。见模块 docstring 第三节的表。"""
    if not flags.use_rel_stress_constraint:
        return von_mises_e / flags.max_stress_constraint - 1
    if flags.use_type1_rel_stress:
        return von_mises_e / k_e / flags.max_rel_stress_constraint - 1
    if flags.use_type2_rel_stress:
        return von_mises_e * rho_e / k_e / flags.max_rel_stress_constraint - 1
    if flags.use_type3_rel_stress:
        return (von_mises_e / k_e ** flags.rel_stress_p_exp
                / flags.max_rel_stress_constraint - 1)
    # use_type4_rel_stress
    return (von_mises_e * rho_e / k_e ** flags.rel_stress_p_exp
            / flags.max_rel_stress_constraint - 1)


def _constraint_function(lam_e, k_e, flags: StressConstraintFlags):
    """``g_e``（``:2510-2512``）。⚠ 判据跨两族，见模块 docstring 第三节。"""
    g = k_e * (lam_e ** 3 + lam_e)
    if (flags.use_type2_stress or flags.use_type3_rel_stress
            or flags.use_type4_rel_stress):
        g = lam_e ** 3 + lam_e
    return g


def _derivative_weights(vm_e, lam_e, g_e, k_e, dk_e, rho_e, flags):
    """``(k1, k2)`` —— 乘 ``∂σ̄/∂u`` 与显式 ``∂k_E/∂θ`` 的两族系数。

    见模块 docstring 第三节的对照表（每一支都能在笔记里对上）。
    """
    sigma0 = (flags.max_rel_stress_constraint if flags.use_rel_stress_constraint
              else flags.max_stress_constraint)
    p_exp = flags.rel_stress_p_exp

    if not flags.use_rel_stress_constraint:
        if flags.use_type1_stress:
            k1 = k_e * (1 + 3 * lam_e ** 2) / flags.max_stress_constraint
            k2 = g_e / k_e * dk_e
        else:                                   # use_type2_stress
            k1 = (1 + 3 * lam_e ** 2) / flags.max_stress_constraint
            k2 = 0
        return k1, k2

    if flags.use_type1_rel_stress:
        k1 = (3 * lam_e ** 2 + 1) / flags.max_rel_stress_constraint
        k2 = (g_e / k_e - (3 * lam_e ** 2 + 1) * vm_e
              / flags.max_rel_stress_constraint / k_e) * dk_e
    elif flags.use_type2_rel_stress:
        k1 = rho_e * (3 * lam_e ** 2 + 1) / flags.max_rel_stress_constraint
        k2 = (g_e / k_e * dk_e
              + (3 * lam_e ** 2 + 1) * vm_e / flags.max_rel_stress_constraint
              * (1 - rho_e / k_e * dk_e))
    elif flags.use_type3_rel_stress:
        k1 = 1 / k_e ** p_exp * (3 * lam_e ** 2 + 1) / flags.max_rel_stress_constraint
        k2 = (-p_exp * (3 * lam_e ** 2 + 1) * vm_e
              / flags.max_rel_stress_constraint / k_e ** (p_exp + 1)) * dk_e
    else:                                       # use_type4_rel_stress
        k1 = (rho_e / k_e ** p_exp * (3 * lam_e ** 2 + 1)
              / flags.max_rel_stress_constraint)
        k2 = ((3 * lam_e ** 2 + 1) * vm_e / flags.max_rel_stress_constraint
              * (1 / k_e ** p_exp - p_exp * rho_e / k_e ** (p_exp + 1)))
    return k1, k2


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
    simp: Optional[SIMPProjection] = None,
    verbose: bool = False,
) -> MechStressResult:
    """应力 / 应变能的诊断量与它们的密度导数。对应 ``MMA10:2437-2601``。

    Parameters
    ----------
    u_node : (2·N_node_mech,) 位移（内核 16 的 ``solve_mech_field``）
    K_mech : (2N, 2N) 刚度（内核 16 的 ``mech_residual`` 的第二个返回）
    dRHS_mech_p_dth : (2N, N_elem)
        ``∂F/∂θ`` 的近似（内核 16 的 ``mech_sensitivity`` 的第二个返回）——
        **D5 已修**：``S = dK/dρ·u + dR/dρ``（加号）。AL 罚项那一支的伴随解
        ``−spsolve(Kᵀ, dAL_pp_du)ᵀ @ S`` 随之消掉 ``2λᵀdR``。
    dR_p_dth : (2N, N_elem)
        ``∂R/∂ρ_e``（``dRHS`` 的载荷那一支，单独传以便闭式重建
        ``dK/dρ_e·u = S[:,e] − dR[:,e]``）。**D6 修正后必填** —— 生产接线
        ``mech_worker`` 从 ``pre.Kmech``（ops）照 ``probe_envelope`` 的写法构造。
    rho : (N_elem,) 密度
    al_lambda : (N_elem,) 拉格朗日乘子
    al_mu : float 罚因子（本内核**不变**它 —— 更新属内核 19）
    tri, l_coef, d_mat, area : 内核 5 的量（``MechElements`` 的 ``tri``/``L``/``D``/``area``）
    flags : StressConstraintFlags
    simp : SIMPProjection 或 None
        默认力学侧参数（``β=8, thb=0.65, power=3, min=1e-4``）。
    verbose : bool
        是否打印 legacy 的那行诊断（``:2590``）。⚠ legacy **总是**打印；
        本模块默认关闭（打印是 I/O，不是数值行为），需要时打开。

    Returns
    -------
    MechStressResult

    Notes
    -----
    两处 ``mech_topo_opti`` 的调用形态照抄：循环内是**标量**调用
    （``mech_topo_opti(_rho_SIMP[i_elem])``，``:2473``），循环外是**数组**调用
    （``mech_topo_opti(_rho_SIMP)``，``:2580``）。两者都保留 —— 后者算 ``RelStress_max``。

    **D6 修正（2026-09-23，落地包络闭式）**：``dU_p_drho`` **不再**走伴随解
    ``−spsolve(Kᵀ, dU_p_du)ᵀ @ dRHS``，改用免伴随闭式
    ``strain_energy_derivative_envelope``（``dU/dρ_e = −½·uᵀ(dK/dρ_e)u −
    uᵀ(dR/dρ_e)``）—— 数学上与「D6 两处都修 + D5 修」的伴随式恒等
    （``∂U/∂u = Ku ⇒ λ = u``），且**省掉一个 spsolve**。``dU_p_du`` 保留
    为纯 ``∂U/∂u`` 的诊断量。AL 罚项那一支（``dALStress_p_drho``）仍走伴随解
    （``g_stress`` 不是能量泛函，§8b 判定）。
    """
    from .mech_solver import MECH_SIMP_PARAMS

    if simp is None:
        simp = SIMPProjection(MECH_SIMP_PARAMS)

    tri = np.asarray(tri, dtype=int)
    n_elem = tri.shape[0]
    n_dof = u_node.shape[0]

    von_mises_elem = np.zeros(n_elem)
    sigma_elem = np.zeros((n_elem, 3))

    stress_loss_al = 0
    d_al_stress_pp_du = np.zeros(n_dof)
    d_al_stress_pp_drho = np.zeros(n_elem)
    al_lambda_new = np.zeros(n_elem)

    al_loss = 0
    al_mu_new = al_mu

    uelastic = 0
    d_u_p_du = np.zeros(n_dof)

    for e in range(n_elem):
        index = [2 * int(tri[e, 0]), 2 * int(tri[e, 0]) + 1,
                 2 * int(tri[e, 1]), 2 * int(tri[e, 1]) + 1,
                 2 * int(tri[e, 2]), 2 * int(tri[e, 2]) + 1]
        u_elem = u_node[index]

        epsilon_elem = l_coef[e] @ u_elem

        # ⚠ 标量调用（legacy :2473）
        k_e = simp.project(rho[e])
        dk_e = simp.derivative(rho[e])
        sigma_elem[e] = k_e * (d_mat @ epsilon_elem)
        von_mises_elem[e] = von_mises(sigma_elem[e])

        # ⚠ D6 已修：``Uelastic`` 的累计仍是 legacy 语义（``elastic_strain_energy``
        #    的值），但 ``dU_p_du`` 恢复为**纯** ``∂U/∂u``（去掉 legacy :2484
        #    的广播显式项 ``+ Uelastic/kE·dkE``）—— 显式项由包络闭式承担。
        uelastic += area[e] * (epsilon_elem @ sigma_elem[e]) / 2
        d_u_p_du[index] += area[e] * (l_coef[e].T @ sigma_elem[e])

        # v = 2σ̄·(∂σ̄/∂σ)：von Mises 梯度去掉 1/(2σ̄) 与 ∂σ/∂u 后的部分
        #   （D8 修正：∂σ̄/∂u = k_E/(2σ̄)·LᵀDᵀv，见模块 docstring 第四节）
        dvm_dsigma = np.array([
            2 * sigma_elem[e, 0] - sigma_elem[e, 1],
            -sigma_elem[e, 0] + 2 * sigma_elem[e, 1],
            6 * sigma_elem[e, 2],
        ])
        d_sigma_p_du = (k_e / (2 * von_mises_elem[e])
                        * (l_coef[e].T @ (d_mat.T @ dvm_dsigma)))

        lam_e = _constraint_variable(von_mises_elem[e], k_e, rho[e], flags)
        g_e = _constraint_function(lam_e, k_e, flags)
        h_e = max(g_e, -al_lambda[e] / al_mu)
        pi_e = al_lambda[e] * h_e + al_mu / 2 * h_e ** 2
        stress_loss_al += pi_e / n_elem

        if g_e * al_mu + al_lambda[e] > 0:
            k1, k2 = _derivative_weights(
                von_mises_elem[e], lam_e, g_e, k_e, dk_e, rho[e], flags)
            d_al_stress_pp_du[index] += (1 / n_elem
                                         * (al_lambda[e] + al_mu * g_e)
                                         * k1 * d_sigma_p_du)
            # D8 修正第二项：σ̄ 经 σ = k_E·D·ε **直接**依赖 k_E（u 固定时也有
            # ∂σ̄/∂k_E = vᵀ·(D·ε)/(2σ̄)）。FD 会捕获它，代码原本漏了（只有
            # k2 = (3Λ²+1)·∂Λ/∂k_E·k_E' 那支，缺 k1·∂σ̄/∂k_E·k_E'）。
            d_sigma_p_dk = (dvm_dsigma @ (d_mat @ epsilon_elem)
                            / (2 * von_mises_elem[e]))
            d_al_stress_pp_drho[e] = (1 / n_elem
                                      * (al_lambda[e] + al_mu * g_e)
                                      * (k2 + k1 * d_sigma_p_dk * dk_e))
            al_loss += g_e ** 2
        else:
            al_loss += al_lambda[e] ** 2

        # D7 修正（2026-09-23）：``min(0, λ+μh)`` → ``max(0, λ+μh)``。legacy 的 min 让
        # ``λ_new ≡ 0``（D7 可证明），AL 退化为纯罚函数。正确更新是
        # ``λ_new = max(0, λ+μh) = max(0, λ+μg)``（因 ``h = max(g, −λ/μ) ⇒ λ+μh ≥ 0``），
        # λ ≥ 0 携带历史信息、AL 才名副其实。见模块 docstring 第四节与设计文档 §12 的 D7。
        al_lambda_new[e] = max(0.0, al_lambda[e] + al_mu_new * h_e)

    von_mises_max = np.max(von_mises_elem)

    # ⚠ 数组调用（legacy :2580）
    k_e_all = simp.project(rho)

    rel_stress_max = np.max(von_mises_elem / k_e_all)
    if flags.use_type2_rel_stress:
        rel_stress_max = np.max(von_mises_elem * rho / k_e_all)
    elif flags.use_type3_rel_stress:
        rel_stress_max = np.max(
            von_mises_elem / k_e_all ** flags.rel_stress_p_exp)
    elif flags.use_type4_rel_stress:
        rel_stress_max = np.max(von_mises_elem * rho
                                / k_e_all ** flags.rel_stress_p_exp)
    al_loss = np.sqrt(al_loss)

    if verbose:
        print(f'AL_stress_loss={al_loss:.4g}, AL_stress_mu={al_mu:.4g}, '
              f'VonMisesStress_max={von_mises_max:.4g}, '
              f'RelStress_max={rel_stress_max:.4g}')

    K_csr = K_mech.tocsr()
    d_al_stress_p_drho = (-sp_sparse_linalg.spsolve(K_csr.T, d_al_stress_pp_du).T
                          @ dRHS_mech_p_dth + d_al_stress_pp_drho)
    # ⚠ D6 已修：``dU_p_drho`` 走**包络闭式**（免伴随，不 spsolve）——
    #   ``dU/dρ_e = −½·uᵀ(dK/dρ_e)u − uᵀ(dR/dρ_e)``。由平衡 ``Ku = −R``
    #   消 ``du/dρ``（§8b 判定），与 D5+D6 都修的伴随式恒等。
    d_u_p_drho = strain_energy_derivative_envelope(
        u_node, K_mech, dRHS_mech_p_dth, dR_p_dth,
        n_dof=n_dof, n_elem=n_elem)

    return MechStressResult(
        von_mises_max=float(von_mises_max),
        rel_stress_max=float(rel_stress_max),
        stress_loss_al=float(stress_loss_al),
        al_loss=float(al_loss),
        d_al_stress_p_drho=np.asarray(d_al_stress_p_drho),
        elastic_strain_energy=float(uelastic),
        d_u_p_drho=np.asarray(d_u_p_drho),
        al_lambda_new=al_lambda_new,
        von_mises=von_mises_elem,
        sigma=sigma_elem,
        d_al_stress_pp_du=d_al_stress_pp_du,
        d_al_stress_pp_drho=d_al_stress_pp_drho,
        d_u_p_du=d_u_p_du,
    )
