"""冻结视在磁导率法的磁链与磁链转矩（内核 15）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）::

    field_error_vectorized_fronzen_appearance_conductivity   :2300-2333
    DTorque_flux_vectorized                                  :2336-2377
    solve_field_vectorized_frozen_appearance_conductivity     :2409-2430
    电感矩阵 / 磁链 / 磁链转矩（消费点）                        :2937-2964

数学出处：``磁障设计/同步磁阻电机磁障结构分析.md``
  - 「绕组磁链计算与估算」
  - 「磁链敏感度分析」

------------------------------------------------------------------------------
一、这一内核在做什么（冻结视在磁导率法 / frozen permeability）
------------------------------------------------------------------------------

非线性求解（内核 9/10/11）给出工作点 ``(A, B)`` 后，**把磁导率冻住**：

1. 用**冻结的** ``B_frozen`` 算出逐单元磁导率 ``μ_e = μ(B_frozen_e, ρ_e)``（内核 3）；
2. 用这个 ``μ`` 组一次**线性**刚度 ``K = Σ_e (1/μ_e) TᵀK0_eT``；
3. 分别以 **d 轴单位电流**与 **q 轴单位电流**为源解两次线性系统
   （``solve_frozen_field``），得到 ``(Bx_d, By_d)`` 与 ``(Bx_q, By_q)``；
4. 电感由**磁场能量密度**积出（``inductance``）::

       L_dd = Σ_e (Bx_d² + By_d²)/μ_e · Δ_e
       L_qq = Σ_e (Bx_q² + By_q²)/μ_e · Δ_e
       L_dq = Σ_e (Bx_d·Bx_q + By_d·By_q)/μ_e · Δ_e

5. 磁链与磁链转矩::

       ψ = [L_dd·I_d + L_dq·I_q,  L_dq·I_d + L_qq·I_q] · parts
       T_flux = pole_pairs · (ψ_d·I_q − ψ_q·I_d)

**为什么 ``L`` 是能量而不是 ``ψ/I``**：源项 ``Rd``/``Rq`` 用的是**单位 d/q 电流**
（legacy ``:3398-3399`` 的 ``_J_D_unit = 1 * N_conductor / area_coil``），所以
``B_d`` 就是「单位 d 电流产生的场」。于是 ``Σ_e (Bx²+By²)/μ_e·Δ_e`` 是二倍能量
（``2W = ∫|B|²/μ dV``），而 ``L = 2W``（单位电流下 ``L = 2W/I²``）。
⇒ ``L_dd`` 的物理含义是**单位电流下的磁链**，``ψ = L·I`` 的量纲才自洽。

**一条可执行的解析恒等式**（本模块的测试用它当来源 A）::

    Σ_e (1/μ_e)·(Bx² + By²)·Δ_e  ==  Aᵀ K A ,   K = Σ_e (1/μ_e) TᵀK0_eT

推导：``K0_e = (1/(4Δ))(b bᵀ + c cᵀ)``、``Bx = (b·A)/(2Δ)``、``By = −(c·A)/(2Δ)``
⇒ ``A_eᵀK0_eA_e = ((b·A)² + (c·A)²)/(4Δ) = (Bx² + By²)·Δ``。
即「单元能量密度积分」与「有限元二次型」是同一个量 —— 这是把电感积分挂到 FEM 上的
唯一支点，测试里两边各算一次比对。

⚠ 上面这条成立的前提是 ``K`` **未做驱动行替换**。驱动行替换（内核 6）把约束行的
系数换成 O(1) 的约束表达式，会破坏这个恒等式；所以测试里要另建一份未驱动的 ``K``。

------------------------------------------------------------------------------
二、与内核 9/10/11 的共用缝（本模块一行数学都不抄第二遍）
------------------------------------------------------------------------------

实测逐行 diff：``field_error_vectorized_fronzen_appearance_conductivity``
（``:2300-2333``）与 ``field_error_vectorized``（``:2222-2257``）**只差两件事**：

1. ``B`` 的来源：前者收 ``_B_elem_frozen`` 作参数（``:2307`` 的 ``B = _B_elem_frozen``），
   后者自己由 ``A`` 算（``:2229-2231``）；
2. 返回顺序：``(F, Jacobi, mu)`` vs ``(F, B, Jacobi)``。

其余每一行逐字相同。所以本模块的 ``frozen_field_system`` 就是
``reference/newton.py`` 那两个具名函数（``element_field_material`` + ``field_system``）
的三行组合，**没有任何数学被抄第二遍**：``B_frozen`` → ``element_field_material``
→ ``field_system``（``B`` 参数位传 ``B_frozen``）→ 加一层返回 ``μ``。

``fast/flux.py`` 侧不能这样切：向量化版的整段语句序列是位级对齐的对象，抽出公共体
会把「迁移可核对性」换掉。那一层的做法是**照搬一份 + 用一条逐位测试把两份钉死相等**。

------------------------------------------------------------------------------
三、磁链转矩对 γ 的导数（`DTorque_flux_vectorized`）
------------------------------------------------------------------------------

**先把转矩写成逐单元紧致形式**（这一步是读懂那 40 行的钥匙）。把 ``ψ`` 展开::

    T_flux = pole_pairs·parts·Σ_e [ I_d·I_q·(Bx_d² + By_d² − Bx_q² − By_q²)
                                    + (I_q² − I_d²)·(Bx_d·Bx_q + By_d·By_q) ] · Δ_e / μ_e

（用 ``I_q·ψ_d − I_d·ψ_q = parts[I_d I_q(L_dd − L_qq) + (I_q² − I_d²)L_dq]`` 即得。）
legacy 把这一坨命名为 ``_k_vectorized``，于是

    T_flux = Σ_e k_e / μ_e = Σ_e k_e · γ_e        （γ_e := 1/μ_e 是单元磁阻率）

导数因此有**两项**::

    dT_flux/dγ_e = k_e                          ← 显式项（∂/∂γ of Σ k_j γ_j，k 视作常数）
                 − (∂F/∂γ_e)ᵀ · J⁻ᵀ (∂T/∂A)     ← 伴随项

其中伴随项的右端量是（legacy 的 ``coef_torque_flux_vectorized``）

    (∂T/∂A)_n = Σ_{(e,r)→n} g0_e[r] · k_e · (1/B_e/Δ_e · dν_e/dB)

``g0_e = (T_eᵀK0_eT)A_active``（**不含 ν**，与内核 9/10 的饱和项同一个几何量），
而 ``Σ_r g0_e[r]·v[index_e[r]] = g0_e·v_e`` 就是 legacy 的
``dK_p_dgamma_A_vectorized @ v``。⇒ 逐元素

    (dT_flux/dγ)_e = k_e − g0_e · λ_e ,   λ := J⁻ᵀ(∂T/∂A)

⚠ **两处与内核 13/14 的差别要留意**（都是 legacy 的形态，本模块照搬）:

1. **因子是 ``−1`` 而不是 ``−2``**。内核 13/14 的 ``−2`` 里有一个因子 2 来自
   ``d(AᵀMA)/dγ`` 的乘积法则（二次型对 ``A`` 求导）；这里 ``T_flux`` 对 ``A`` 的
   依赖被写成 ``Σ_e k_e ν_e`` 的形式后，``∂T/∂A`` 只作用在那个 ``ν_e`` 因子上，
   显式项 ``k_e`` 另算，所以两项各只有一个因子。
2. **显式项 ``k_e`` 不做拓扑区清零**（``:2374-2377`` 只对 ``dTflux_p_dgamma1`` 清零，
   然后 ``+ dTflux_p_dgamma2``）。所以 ``dTflux_p_dgamma`` 在 ``elements_to_compute``
   之外**非零**，而 Arkkio / 虚位移两处是恒零。这是 legacy 的形态，照搬。

⚠ **★ D4 已修（2026-09-23）**：legacy 的上式给出的 ``dT_flux/dρ`` 与真值**符号相反**
（登记为 ``docs/重构设计文档.md`` §12 的 **D4**；实测夹具 ``fd/legacy = −1.0005``
等，即整体差一个负号、不是缺项）。本模块在 ``flux_torque_derivative`` 的返回值处
**翻号**，函数语义即真导数。

**修前它在 legacy 生产里被消费者抵消**（这是它至今没被发现的原因）::

    g_torque  = mean_Tflux
    dg_torque = -mean_Tflux_p_drho        ← MMA10:3202 与 :3584 的额外负号

对照 Arkkio 那一支是 ``dg_torque = +mean_dTArkkio_p_drho``（``:3207``）：
两个目标函数同号，所以梯度也必须同号；flux 那一支多出来的负号恰好把本函数的
符号翻回去。⇒ **legacy 生产路径上的梯度是对的**，错的是本函数的语义。

**D4 判定（2026-09-23）**：阶段 8 收尾 #1 的 oracle（``probe_flux_oracle.py``）
「手写差 4~1429 倍」是对错了参考系 —— 它比的 FD 是「**B 固定**、只重解冻结场」，
而手写匹配的是**全链** FD（``probe_flux.py``：扰动 ρ → 全非线性重解 → 冻结分析，
``fd/手写 = −1.0000``）。冻结场路径（``A_d/A_q`` 对 ρ 的隐式依赖）在全链里抵消，
oracle 不是优化器要的量 ⇒ 手写的量级对、只差符号。

⚠ motorfem 新生产路径（``objective.py``）的目标梯度用 **Arkkio** 转矩，
``dTflux_p_drho_vec`` 只存不消费（dead code）⇒ **没有消费者要改**；若将来启用
flux 目标，直接用本函数的（已修正的）返回值即可。golden L3/L4 的原始 ``dTflux``
数组随翻号已重标定。

**理论文档核对的结果（本批做过，结论出乎意料）**：推导**是有的**，在
``磁障设计/同步磁阻电机磁障结构分析.md`` 的三节里 ——

| 节 | 行 | 给了什么 |
|---|---|---|
| 绕组磁链计算与估算 | ``:3047`` | ``Ψ = (Nl/I)∫A_z j_z dS``；``W_ij = ½∫(1/μ)B_i·B_j dV`` → ``L_ij = ∫(1/μ)B_i·B_j dV`` |
| 磁链敏感度分析 | ``:3096`` | 变分法得 ``δL_ij = l∫δγ B_i·B_j dS``（``δA`` 项按 ``∇·(γ∇A_i)=0`` 与 ``δA=0``「论证为零」）；全微分 ``dγ = (∂γ/∂θ)dθ + (∂γ/∂B)dB``；**``dB^(e) = (1/(B Δ γ))(K^(e)A^(e))ᵀ(−J⁻¹(∂K/∂θ)A)dθ``**、``∂K/∂θ^(e) = K_0^(e)∂γ^(e)/∂θ^(e)`` |
| 转矩关于电感的导数 | ``:3189`` | ``∂τ/∂L_ij``（ABC 与 α-β 两套，含 ``3/2``；并注明 Clarke 必须用**等能量**正交变换） |

**逐项核对后：文档的式子与 legacy 的代码在结构与符号上一致**。把文档第二、三行的
``dB^(e)`` 展开（``K^(e)A^(e) = γ_e·g0_e``，``1/γ`` 约掉）得

    dT/dθ_j = (∂γ_j/∂θ_j)·{ k_j − [Σ_e k_e (∂γ/∂B)_e/(B_eΔ_e)·g0_e]ᵀ · J⁻¹ · g0_j }

而 legacy 正是 ``(∂γ_j/∂ρ_j)·(k_j − g0_j·λ)``，``λ = J⁻ᵀcoef``、
``coef = Σ_e g0_e·k_e·(∂γ/∂B)_e/(B_eΔ_e)`` —— 标量的转置是它自己，两式**逐字等价**。
⇒ **偏差不能用「缺项」或「文档没写」解释**（我最初正是那样猜的，见下面的 ★更正）。

**★ 更正（本批内自己推翻）**：上一提交（``3be5b44``）的 docstring 与 commit message 里
写过「``(∂T/∂A)`` 只取了 ``ν`` 随 ``A`` 变化那一部分、漏了 ``k_e`` 随 ``A`` 变化那一支，
是缺项」。**核对理论文档后该猜想不成立**：legacy 的伴随项就是文档的饱和项
（``∂γ/∂B·dB/dθ``），两项结构与符号都对得上。⇒ 剩下两种可能，**本批不判定**：

1. **文档的推导本身有一处符号错**。它把 ``δA`` 项「论证为零」用的前提是
   ``∇·(γ∇A_i) = 0`` 与 ``δA|∂Ω = 0`` —— 而冻结法里 ``A_i`` 是**单位电流激励下的解**，
   在线圈内 ``∇·(γ∇A_i) = J_i ≠ 0``，且转子域的边界（除 Dirichlet 那一小段）并不满足
   ``δA = 0``。**``D1`` 就是「有记录但记录本身错了」的先例**。
2. 有限差分那条链路与文档的前提有一处不同。

判定需要**与两者都无关的 oracle**（或者按 D3 的做法先补一遍独立推导）——
完整工作项见 ``docs/重构设计文档.md`` §12 的 D4。

**另一条已排除的错误推导路径**（两种猜想都记下来免得重走）：按
``T = Σ_e k_e(A_d(γ), A_q(γ))·γ_e``、``K_d(γ)A_a = R_a`` 做链式法则，得到
「显式项 ``k_e`` 减去**两次**冻结 ``K`` 的伴随解 ``Λ_a = K_d⁻ᵀ(∂T/∂A_a)``」。
把它与有限差分比：两者差 **1~2 个量级**（比值 0.02~1.39，逐单元无规律）⇒ 这条推导
**不成立**。

------------------------------------------------------------------------------
四、三处数值细节（照搬 legacy，勿「顺手整理」）
------------------------------------------------------------------------------

1. **``mu`` 被第二次求解覆盖**（``:2938-2942``）::

       _Bx_elem_d, _By_elem_d, mu_elem = solve(..., Rd)
       _Bx_elem_q, _By_elem_q, mu_elem = solve(..., Rq)      ← 覆盖

   两次的 ``μ`` **逐位相同**（同一个 ``B_frozen``、同一个 ``ρ``、同一个确定性材料
   函数），所以无害；但本模块的 ``frozen_flux_analysis`` 只算一次 ``μ``，
   把它同时给 d、q 两支用。
2. **``clip(min=1e-20)``**：``B`` 在 ``1/B`` 的分母上（``:2367`` 的
   ``dgamma / (B * Aelem)``）。无源区域 ``B`` 可以恒为 0 ⇒ 必须裁剪。本模块沿用
   内核 9 的裁剪值（``B_frozen`` 就是内核 11 返回的那个已裁剪数组）。
3. **``Δ > 0``**：单元定向的检查在内核 4 的 ``element_geometry``（唯一检查点），
   本模块不重复断言 —— 磁链积分里的 ``Δ_e`` 是同一个 ``Aelem``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from . import boundaries as ref_bnd
from .newton import (
    FieldGeometry,
    FieldMaterial,
    element_field_material,
    field_system,
)

__all__ = [
    "FrozenFluxAnalysis",
    "flux_density_components",
    "frozen_field_system",
    "solve_frozen_field",
    "inductance",
    "flux_linkage",
    "flux_torque",
    "frozen_flux_analysis",
    "flux_torque_kernel",
    "flux_torque_derivative",
]


# =============================================================================
# 单元磁密分量与冻结视在磁导率的系统
# =============================================================================

def flux_density_components(
    A: np.ndarray, field: FieldGeometry
) -> Tuple[np.ndarray, np.ndarray]:
    """逐单元磁密分量 ``(Bx, By)``，形状各 ``(N_elem,)``。

    对应 legacy 的 ``Bxcoef_mat @ _A_node`` / ``Bycoef_mat @ _A_node``
    （``:2347-2348`` 与 ``:2427-2428``），**未裁剪**。

    ``|B|`` 的那个版本是内核 9 的 ``flux_density``（带 ``clip(min=1e-20)``）::

        |B| = clip(sqrt(Bx² + By²), 1e-20, inf)

    ⚠ ``Bx``/``By`` 采用 **legacy 约定**（``Bxcoef = +b/(2Δ)``、``Bycoef = −c/(2Δ)``），
    与教科书 ``curl`` 相差一个反射；``|B|`` 不受影响，但 Arkkio 受影响。详见
    ``reference/element_matrices.py`` 的 ``LEGACY_BX_BY_NOTE``。本模块只做**偶次组合**
    （``Bx²+By²``、``Bx_d·Bx_q + By_d·By_q``），所以约定不影响结果。

    Notes
    -----
    单元遍历用 ``range``（README 对 ``reference/`` 的规定：累加顺序必须确定）。
    legacy 用两个 ``(N_elem, N_node)`` 稀疏矩阵一次算完全部单元，两者只差求和顺序。
    """
    Bx = np.empty(field.n_elem)
    By = np.empty(field.n_elem)
    for e in range(field.n_elem):
        tr = field.transforms[e]
        A_active = A[list(tr.index)]
        Bx[e] = field.bxcoef[e] @ tr.matrix @ A_active
        By[e] = field.bycoef[e] @ tr.matrix @ A_active
    return Bx, By


def frozen_field_system(
    A: np.ndarray,
    B_frozen: np.ndarray,
    rho: np.ndarray,
    field: FieldGeometry,
    material: FieldMaterial,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
) -> Tuple[np.ndarray, sp_sparse.csr_matrix, np.ndarray]:
    """冻结视在磁导率下的 ``(F, J, μ)``。对应 ``MMA10:2300-2333``。

    ``B`` 从参数来（**冻结**），其余与内核 9/10 完全同一条路径 —— 见模块 docstring
    第二节。所以这里只是把两个具名缝缝起来::

        μ, dν/dB ← element_field_material(B_frozen, ρ, masks, material)
        F, J     ← field_system(A, field, μ, dν/dB, B_frozen, R, driving)

    Parameters
    ----------
    A : (N_node,) 当前矢量磁位。**冻结法只解一次线性系统**，所以调用方传零向量
        （见 ``solve_frozen_field``）；传非零的 ``A`` 也算得对，只是没有生产用途。
    B_frozen : (N_elem,) 冻结的磁密幅值（内核 11 的 ``B``，**已裁剪**）
    rho : (N_elem,) 设计变量（**未裁剪**）
    field, material, R, driving :
        与内核 9 的 ``residual`` 同名同义。

    Returns
    -------
    F : (N_node,) 残差（已做驱动行替换）。``A=0`` 时即 ``−R``（自由行）/ ``0``（驱动行）
    J : (N_node, N_node) csr Jacobian（已做驱动行替换）
    mu : (N_elem,) 冻结的磁导率 ``μ(B_frozen, ρ)``

    Notes
    -----
    ⚠ 返回的是 **``μ`` 而不是 ``B``** —— 与内核 9 的 ``(F, B, J)`` 顺序不同，这是
    legacy 的形态（``:2333`` 的 ``return F, Jacobi, mu``），照搬。

    ``A=0`` 时 Jacobian 的饱和项**恒为零**（``g0_e = (TᵀK0_eT)·0 = 0``），所以
    ``J`` 就是纯线性刚度 ``K(μ)``（加驱动行替换）。legacy 照样把 ``g0i``/``gi``/
    ``gswap`` 那一串算了一遍（结果是精确的 0），本模块走同一条 ``field_system``、
    同样会算，**不是**被优化掉的 —— 保真优先，且它保证 ``F``/``J`` 与内核 9 的
    计算路径完全一致（测试里用逐位比对钉住）。
    """
    mu, dgamma = element_field_material(B_frozen, rho, field.masks, material)
    F, J = field_system(A, field, mu, dgamma, B_frozen, R, driving)
    return F, J, mu


def solve_frozen_field(
    B_frozen: np.ndarray,
    rho: np.ndarray,
    field: FieldGeometry,
    material: FieldMaterial,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一次线性求解（冻结视在磁导率）。对应 ``MMA10:2409-2430``。

    Parameters
    ----------
    B_frozen : (N_elem,) 冻结的磁密幅值（内核 11 返回的那个，**已裁剪**）
    rho, field, material, driving : 同上
    R : (N_node,) 源项。用**单位 d/q 电流**构造的那一支（legacy 的 ``Rd``/``Rq``）
        才有「电感」的含义，见模块 docstring 第一节。

    Returns
    -------
    Bx, By : (N_elem,) 该源下的磁密分量（**未裁剪**）
    mu : (N_elem,) 冻结的磁导率

    Notes
    -----
    **这不是 Newton 迭代**，是 `legacy` 的形态：``A₀ = 0`` 起步算一次
    ``(F, J, μ)``，解一次 ``spsolve``，然后 ``A = A₀ − dA``，**不做收敛判据、不循环**
    （``:2417-2424``）。它精确等价于一次 Newton 步；而 ``dν/dB`` 项在 ``A=0`` 处恒为
    零 ⇒ 系统本来就是线性的 ⇒ 一步即精确解。

    ⚠ 返回 ``μ`` 与 ``Bx``/``By`` 的「谁覆盖谁」见模块 docstring 第四节第 1 条。
    """
    A0 = np.zeros(field.n_node)
    F, J, mu = frozen_field_system(
        A0, B_frozen, rho, field, material, R, driving)

    dA = sp_sparse_linalg.spsolve(sp_sparse.csr_matrix(J), F)
    A = A0 - dA

    Bx, By = flux_density_components(A, field)
    return Bx, By, mu


# =============================================================================
# 电感矩阵、磁链、磁链转矩
# =============================================================================

def inductance(
    Bx_a: np.ndarray,
    By_a: np.ndarray,
    Bx_b: np.ndarray,
    By_b: np.ndarray,
    mu: np.ndarray,
    area: np.ndarray,
) -> Tuple[float, float, float]:
    """电感三元组 ``(L_aa, L_ab, L_bb)``。对应 ``MMA10:2948-2950``。

    逐单元::

        L_aa = Σ_e (Bx_a² + By_a²)/μ_e · Δ_e
        L_bb = Σ_e (Bx_b² + By_b²)/μ_e · Δ_e
        L_ab = Σ_e (Bx_a·Bx_b + By_a·By_b)/μ_e · Δ_e

    Parameters
    ----------
    Bx_a, By_a : (N_elem,) ``a`` 支的磁密分量（legacy 的 ``_d``）
    Bx_b, By_b : (N_elem,) ``b`` 支的磁密分量（legacy 的 ``_q``）
    mu : (N_elem,) 冻结的磁导率
    area : (N_elem,) 单元面积 Δ（内核 4 的 ``Aelem``）
    **两者用同一个 μ** —— 这是冻结法的定义（两次求解共用冻结的磁导率）。

    Returns
    -------
    ``(L_aa, L_ab, L_bb)``

    Notes
    -----
    数学与「二倍能量」的解释见模块 docstring 第一节。两条可执行的解析性质：

    - ``L_aa = A_aᵀ K A_a``（``K`` 未驱动版）—— 能量积分与 FEM 二次型的恒等式；
    - ``L_aa·L_bb − L_ab² ≥ 0`` 且 ``L_aa, L_bb > 0`` —— 2×2 电感矩阵半正定
      （Cauchy–Schwarz，因为它是 ``⟨·,·⟩_μ`` 内积下的 Gram 矩阵）。

    legacy 的计算顺序是 ``Ldd → Lqq → Ldq``（``:2948-2950``）；本函数按
    ``L_aa → L_ab → L_bb`` 返回，**三个值各自与 legacy 逐位相同**
    （``np.sum`` 在同一个数组上，顺序无关的只有返回次序）。
    """
    w = (Bx_a * Bx_a + By_a * By_a) / mu * area
    v = (Bx_b * Bx_b + By_b * By_b) / mu * area
    c = (Bx_a * Bx_b + By_a * By_b) / mu * area
    return float(np.sum(w)), float(np.sum(c)), float(np.sum(v))


def flux_linkage(
    L_aa: float,
    L_ab: float,
    L_bb: float,
    I_a: float,
    I_b: float,
    parts: int,
) -> np.ndarray:
    """磁链 ``ψ = [L_aa·I_a + L_ab·I_b, L_ab·I_a + L_bb·I_b] · parts``。对应 ``:2953-2954``。

    Parameters
    ----------
    L_aa, L_ab, L_bb : float
        由 ``inductance`` 给出。
    I_a, I_b : float
        d/q 轴电流（legacy 的 ``_I_d``/``_I_q``，由 ``abc_to_dq_orthogonal`` 从三相
        电流与电角度换算，``:2952``）。
    parts : int
        旋转对称扇区数。**乘在整个 2 向量上**（``:2954`` 的 ``* parts``）：
        这个模型只算 ``1/parts`` 的扇区，磁链要还原成整机。

    Returns
    -------
    (2,) ndarray ``[ψ_a, ψ_b]``

    Notes
    -----
    ``* parts`` 写在**数组构造之后**（legacy: ``np.array([...]) * parts``），
    本实现照抄这个次序 —— 它是 2 的幂以外的整数因子，结合序不影响浮点结果，
    但**照抄是纪律**（``CLAUDE.md`` §3 的算式形状要求）。
    """
    return np.array([L_aa * I_a + L_ab * I_b, L_ab * I_a + L_bb * I_b]) * parts


def flux_torque(
    psi: np.ndarray, I_a: float, I_b: float, pole_pairs: int
) -> float:
    """磁链转矩 ``T_flux = p·(ψ_a·I_b − ψ_b·I_a)``。对应 ``:2956-2957``。

    同步电机的经典转矩式 ``T = (3/2)p(ψ_d i_q − ψ_q i_d)`` **在这里不带 ``3/2``**，
    但**不是因为 ``parts``**（``parts = 4 ≠ 3/2``；旧注释这么说，2026-10-13 更正）。
    真正的原因是 **dq 变换的约定**：``abc_to_dq_orthogonal`` 用的是
    **``√(2/3)`` 的功率不变 Clarke+Park**（``preassembly.py:54-60``），
    它把电流相对「等幅值（2/3）」约定整体放大 ``√(3/2)`` 倍
    ⇒ 乘积 ``ψ_d·i_q`` 自带 ``3/2`` ⇒ 转矩式**本就不带 ``3/2``**。
    换成等幅值约定才需要显式补 ``3/2``。

    ⇒ **本式在两种约定下各自自洽**；``flux_linkage`` 里的 ``* parts``
    只做「1/parts 扇区 → 整机」，与 ``3/2`` 无关。

    Returns
    -------
    float
    """
    return float(pole_pairs * (psi[0] * I_b - psi[1] * I_a))


@dataclass(frozen=True)
class FrozenFluxAnalysis:
    """``frozen_flux_analysis`` 的产出：两次冻结求解 + 电感 + 磁链 + 转矩。

    Attributes
    ----------
    Bx_d, By_d, Bx_q, By_q : (N_elem,) ndarray
        d/q 单位电流源下的磁密分量（**未裁剪**）。
    mu : (N_elem,) ndarray
        冻结的磁导率（两次求解共用同一个值）。
    L_dd, L_dq, L_qq : float
    psi : (2,) ndarray
        ``[ψ_d, ψ_q]``。
    torque : float
        ``T_flux``。
    """

    Bx_d: np.ndarray
    By_d: np.ndarray
    Bx_q: np.ndarray
    By_q: np.ndarray
    mu: np.ndarray
    L_dd: float
    L_dq: float
    L_qq: float
    psi: np.ndarray
    torque: float


def frozen_flux_analysis(
    B_frozen: np.ndarray,
    rho: np.ndarray,
    field: FieldGeometry,
    material: FieldMaterial,
    R_d: np.ndarray,
    R_q: np.ndarray,
    I_d: float,
    I_q: float,
    driving: ref_bnd.DrivingRows,
    *,
    pole_pairs: int,
    parts: int,
) -> FrozenFluxAnalysis:
    """整条冻结视在磁导率链：两次线性解 → 电感 → 磁链 → 转矩。

    对应 ``MMA10:2937-2957``（worker 里那一段），并把 ``:2300-2333``/``:2409-2430``
    串起来。

    Parameters
    ----------
    B_frozen : (N_elem,) 冻结的磁密幅值（**来自非线性解**，内核 11 的返回值）
    rho : (N_elem,)
    field, material, driving :
    R_d, R_q : (N_node,)
        **单位** d/q 电流的源项（legacy 的 ``Rd``/``Rq``）。⚠ 传「当前那一时刻的
        实际电流」而不是单位电流，得到的 ``L`` 会带上电流因子、失去电感的量纲。
    I_d, I_q : float
        用于算磁链的 d/q 电流（**与被冻结的 ``B_frozen`` 同一个工作点**）。
    pole_pairs, parts : int

    Returns
    -------
    FrozenFluxAnalysis

    Notes
    -----
    两次求解的 ``μ`` **逐位相同**（见模块 docstring 第四节第 1 条），本实现只算一次、
    给两支共用；legacy 算了两次、第二次覆盖第一次。两者数值无差。
    """
    Bx_d, By_d, mu = solve_frozen_field(
        B_frozen, rho, field, material, R_d, driving)
    Bx_q, By_q, _mu_q = solve_frozen_field(
        B_frozen, rho, field, material, R_q, driving)

    # legacy 用第二次的 μ（覆盖），此处用同一个值 —— 两者逐位相同，见 Notes
    L_dd, L_dq, L_qq = inductance(Bx_d, By_d, Bx_q, By_q, mu, field.aelem)
    psi = flux_linkage(L_dd, L_dq, L_qq, I_d, I_q, parts)
    torque = flux_torque(psi, I_d, I_q, pole_pairs)

    return FrozenFluxAnalysis(
        Bx_d=Bx_d, By_d=By_d, Bx_q=Bx_q, By_q=By_q, mu=mu,
        L_dd=L_dd, L_dq=L_dq, L_qq=L_qq, psi=psi, torque=torque,
    )


# =============================================================================
# 磁链转矩对 γ 的导数（伴随 + 显式项）
# =============================================================================

def flux_torque_kernel(
    Bx_d: np.ndarray,
    By_d: np.ndarray,
    Bx_q: np.ndarray,
    By_q: np.ndarray,
    I_d: float,
    I_q: float,
    area: np.ndarray,
    *,
    pole_pairs: int,
    parts: int,
) -> np.ndarray:
    """逐单元的 ``k_e`` —— 磁链转矩的**被积核**。对应 ``MMA10:2360-2365``。

    把 ``flux_torque`` 的标量式展开到单元上（推导见模块 docstring 第三节）::

        T_flux = Σ_e k_e / μ_e

        k_e = p·[I_d·I_q·(Bx_d² + By_d² − Bx_q² − By_q²)
                 + (I_q² − I_d²)·(Bx_d·Bx_q + By_d·By_q)] · Δ_e · parts

    两个用途，都不是「为了好看」：

    1. 它是 ``dT_flux/dγ`` 的**显式项**（``k_e = ∂(Σ_j k_j γ_j)/∂γ_e``）；
    2. 它把磁链转矩与电感的三个积分钉在一起 —— ``Σ_e k_e/μ_e == flux_torque(ψ)``
       是一条**逐位**成立的代数恒等式（两侧都是有限项的和），测试用它当来源 A。

    Returns
    -------
    (N_elem,) ndarray
    """
    return (
        pole_pairs * (
            I_d * I_q * (Bx_d ** 2 + By_d ** 2 - Bx_q ** 2 - By_q ** 2)
            + (I_q ** 2 - I_d ** 2) * (Bx_d * Bx_q + By_d * By_q)
        ) * area * parts
    )


def flux_torque_derivative(
    A: np.ndarray,
    field: FieldGeometry,
    B_frozen: np.ndarray,
    rho: np.ndarray,
    material: FieldMaterial,
    Bx_d: np.ndarray,
    By_d: np.ndarray,
    Bx_q: np.ndarray,
    By_q: np.ndarray,
    I_d: float,
    I_q: float,
    Jacobi,
    elements_to_compute,
    *,
    pole_pairs: int,
    parts: int,
) -> np.ndarray:
    """磁链转矩对磁阻率的导数 ``dT_flux/dγ``。对应 ``MMA10:2336-2377``。

    数学与两项的来历见模块 docstring 第三节。逐单元::

        k_e    = p·[I_d I_q (Bx_d² + By_d² − Bx_q² − By_q²)
                    + (I_q² − I_d²)(Bx_d Bx_q + By_d By_q)]·Δ_e·parts
        (∂T/∂A)_n = Σ_{(e,r)→n} g0_e[r]·k_e·(1/B_e/Δ_e)·dν_e/dB
        λ      = J⁻ᵀ (∂T/∂A)
        out_e  = k_e − g0_e·λ_e        （e ∈ elements_to_compute，其余处清零）
        结果   = out + k               （显式项**不清零**，见模块 docstring 第三节的 ⚠2）

    Parameters
    ----------
    A : (N_node,) **非线性**工作点的矢量磁位（legacy 的 ``_A_node_frozen``）
        —— 注意它同时是 ``∂F/∂γ`` 的收缩对象（``g0_e``）与 ``B_frozen``/``dν/dB``
        的来源，两处**必须**是同一个工作点。
    field : FieldGeometry
    B_frozen : (N_elem,) 冻结的磁密幅值（= ``|B(A)|`` 裁剪后的值）
    rho : (N_elem,)
    material : FieldMaterial
    Bx_d, By_d, Bx_q, By_q : (N_elem,) d/q 冻结解的磁密分量
    I_d, I_q : float
    Jacobi : 内核 11 返回的**同一个对象**（会被 ``.T`` 求伴随）。
        与内核 13/14 同理：驱动行替换让 ``Jacobi`` **不对称**，所以伴随用转置。
    elements_to_compute : 可迭代 int
        设计变量所在的单元（legacy 的 ``element_topo_opti``）。
    pole_pairs, parts : int

    Returns
    -------
    (N_elem,) ndarray。⚠ ``elements_to_compute`` 之外**非零**（显式项），
    与内核 13/14 的恒零形态不同 —— 见模块 docstring 第三节的 ⚠2。

    ⚠ **D4 已修（2026-09-23）**：legacy 的返回值与真值符号相反（实测
    ``fd/返回值 = −1.0005``，见模块 docstring 第三节的 ★ 与设计文档 §12 的 **D4**），
    靠 legacy 消费者 ``dg_torque = -mean_Tflux_p_drho``（``:3202``）的额外负号翻回来。
    本实现**在这里翻号**（``return -out``）⇒ 函数语义即真导数。⚠ motorfem 新生产路径
    （``objective.py``）的目标梯度用 Arkkio 转矩，``dTflux_p_drho_vec`` 只存不消费
    （dead code）⇒ 没有消费者要改；若将来启用 flux 目标，直接用本函数的返回值即可。

    Notes
    -----
    一处定义域细节：``g0_e`` 与 ``B_e`` 在 ``elements_to_compute`` **之外**也要算
    （显式项 ``k_e`` 覆盖全部单元）。legacy 也是这么做的。

    另一处：``dν/dB`` 用 ``element_field_material`` 取得，它同时对两类非线性材料
    求值（SIMP 类与纯铁芯类）。legacy 在 ``:2351-2352`` 也算了同样的两路，
    只是**不用 ``μ``**；本实现沿用共用缝，多算的 ``μ`` 是无害的浪费。
    """
    n_elem = field.n_elem
    _, dgamma = element_field_material(B_frozen, rho, field.masks, material)

    k = flux_torque_kernel(
        Bx_d, By_d, Bx_q, By_q, I_d, I_q, field.aelem,
        pole_pairs=pole_pairs, parts=parts,
    )

    lam_rhs = np.zeros(field.n_node)
    g0_table = []
    for e in range(n_elem):
        tr = field.transforms[e]
        g0_e = (tr.matrix.T @ (field.k0[e] @ tr.matrix)) @ A[list(tr.index)]
        g0_table.append(g0_e)

        # ★ 算式照抄 legacy :2367： dgamma / (B * Aelem) —— 左结合的逐项除法
        scale = k[e] * dgamma[e] / (B_frozen[e] * field.aelem[e])
        for r, node in enumerate(tr.index):
            lam_rhs[int(node)] += g0_e[r] * scale

    lam = sp_sparse_linalg.spsolve(sp_sparse.csr_matrix(Jacobi).T, lam_rhs)

    out = k.copy()
    for e in elements_to_compute:
        index = list(field.transforms[e].index)
        # legacy 的因子是 −1（不是内核 13/14 的 −2）—— 见模块 docstring 第三节
        out[e] = k[e] - (g0_table[e] @ lam[index])
    # D4 修正（2026-09-23）：legacy 的 ``k − g0·λ`` 实为 **−**dT_flux/dγ（实测
    # fd/legacy = −1.0000）。翻号 ⇒ 函数语义即真导数。见本函数 docstring 与 §12 的 D4。
    return -out
