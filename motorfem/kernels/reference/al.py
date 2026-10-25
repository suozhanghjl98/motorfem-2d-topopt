"""增广拉格朗日：乘子写回与罚因子调度（内核 19）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）
    ``optimizer_func_warper_AL``   ``:3838-3980``
    ``optimizer_func_warper_AL_v2`` ``:3982-4120``
    本内核只提取这两个 wrapper 里的 **AL 状态更新**段：
    ``_v2.__call__`` 的 ``:4025-4032``（λ 写回 + μ 调度 + clip）与
    ``:4056-4058``（``AL_N_call`` 计数）。wrapper 的其余部分
    （密度滤波、从属单元传播、目标函数求值、绘图/保存）**不属于**本内核。

数学出处：``docs/笔记/MMA方法实现AL的优化.md``（「Augmented Lagrangian Method」）。

⚠ **先读这条**：笔记给的是**标准** AL —— 乘子 ``w^{k+1} = max(0, w^k − σg)``、
罚因子按 ``e^{(k+1)}/e^{(k)} ≥ β`` 的比率 ``ασ`` 调度。**legacy 用的不是那套**：

    1. μ 调度是自定义的二次式 ``μ(n) = min(μ0·(1 + n/step)², μ_max)``，
       ``n`` = **约束被违反的迭代次数**（``AL_N_call``），与笔记的
       ``e``-比率判据**无任何关系**；
    2. 乘子更新 ``λ ← λ_new`` 里，内核 17 已证 ``λ_new ≡ 0``（**D7**）⇒
       这个「增广拉格朗日」实际是**纯罚函数**（见第四节）。

所以本内核提取的是 **legacy 的实际调度**（保真搬移），**不是**笔记的公式。
将来若判定 D7 为「文档与代码同错」（而不是「有意为之」），μ 调度与乘子更新
必须**一起**重设计 —— 那属修正工作（§12 的 D7 修正清单第 ④ 条：必须重标定
``mu0``/``mu_max``/``mu_update_step``），本批**只登记、不改**。

------------------------------------------------------------------------------
一、这个内核算什么
------------------------------------------------------------------------------

每一轮 MMA 迭代（wrapper 的 ``__call__``）末尾做一次 AL 状态更新。状态是
legacy 挂在 ``optimizer_func_warper_AL`` 类上的三个类属性::

    AL_stress_lambda     (N_elem_mech,)   逐单元乘子（``:3840``，生产初值 zeros）
    AL_stress_mu         标量             罚因子（``:3842``，生产初值 = μ0 = 0.01）
    AL_N_call            int              约束被违反的累计次数（``:3845``，初值 0）

一次更新（``_v2`` 生产路径，``:4025-4032`` + ``:4057-4058``）::

    AL_stress_lambda = AL_stress_lambda_new          # 写回内核 17 的产出
    AL_stress_mu  = min( μ0 · (1 + AL_N_call / step)² ,  μ_max )
    if Stress_loss_AL > 0:  AL_N_call += 1            # 严格大于 0

**顺序纪律**（照抄，不能换）：μ 用**更新前**的 ``AL_N_call`` 计算
（``:4026`` 读旧值），计数在末尾才递增（``:4057-4058``）。换句话说：
第 ``k`` 轮迭代的 μ 由**前 ``k-1`` 轮里约束被违反的次数**决定。

**μ 无历史记忆**：公式只读 ``AL_N_call``，**不读旧 ``AL_stress_mu``**。
给定 ``AL_N_call = n``，μ 就被唯一确定；这与标准 AL（σ 依赖 ``e`` 比率）
是本质区别 —— 因为乘子恒为 0（D7），没有任何东西需要「消化」，μ 的强度
就是约束力的全部。

------------------------------------------------------------------------------
二、合并 legacy 两版（§7 规定取 ``_v2``）—— 本批核实的结论
------------------------------------------------------------------------------

CLAUDE.md §7 说「AL 与 ``_v2`` 在 ``AL_stress_mu`` 更新上差一行，合并时取
``_v2``」。**实测核对（本批）**：两版的**活动公式逐字相同** —— ``_v2``
``:4026-4032`` 与 ``AL`` ``:3888-3894`` 是同一段代码。差别只在 ``AL`` 版
上方多两行**注释掉的**死代码::

    # optimizer_func_warper_AL.AL_stress_mu *= ...AL_stress_k_penalty_update
    # ...AL_stress_k_penalty_update ** (_N_call // AL_stress_mu_update_step)

它们引用一个**不存在的属性** ``AL_stress_k_penalty_update``（两版类里都没有
定义），即使取消注释也无法运行 —— 属纯死代码。**合并决策因此是平凡的**：
取活动公式（两版相同），不搬运注释。§7 的「取 ``_v2``」自动满足，且没有
任何数值取舍。

**一个值得记录的怪癖**：``_v2.__call__`` 读写的是
``optimizer_func_warper_AL.AL_stress_lambda``（``:4020``/``:4025``）——
**不是** ``optimizer_func_warper_AL_v2`` 自己的类属性（那个 ``:3984`` 的
定义是死代码，``_v2`` 从没用它）。即两个 wrapper **共享同一份类级状态**，
都在 ``optimizer_func_warper_AL`` 上。本内核把状态做成显式的
``ALState`` 参数/返回值，这个怪癖自然消失。

**``_N_call``（总调用数）不提取**：它只用于绘图/保存的节奏
（``% save_figure_per_epoch``）与收敛曲线，**不参与任何数值**。

------------------------------------------------------------------------------
三、与内核 17 的接口（本内核的输入从哪来）
------------------------------------------------------------------------------

内核 17（``reference/mech_stress.py`` 的 ``mech_stress_and_energy``）在一次
力学求解后返回:

    al_lambda_new      →  本内核的 ``al_lambda_new``（⚠ 恒为 0，见第四节）
    stress_loss_al     →  本内核的 ``stress_loss_al``（= ``Σ P_e / N_elem``）

生产 wrapper 里 ``AL_stress_lambda_new`` 正是从 ``obj_func_parallel_AL_v2``
透传的内核 17 产出（``:3568-3569`` → ``:3752`` → ``:4018``），
``Stress_loss_AL`` 同理（``:3685`` → ``:4057``）。

------------------------------------------------------------------------------
四、★ D7 的牵连（本内核存在的理由与它的局限）
------------------------------------------------------------------------------

内核 17 已证明（设计文档 §12 的 D7）::

    h = max(g, −λ/μ)   ⇒   λ + μh ≥ 0   ⇒   λ_new = min(0, λ + μh) ≡ 0

配上初值 ``λ = np.zeros``（``:3840``）与写回，归纳得 **``λ ≡ 0`` 对每一轮
迭代成立**。于是罚项 ``P = λh + μ/2·h²`` 退化为 ``P = μ/2·max(g,0)²`` ——
乘子不携带任何历史信息，**``μ`` 的增长成了唯一的约束力来源**。

**这正是 CLAUDE.md 提示里「D7 直接决定内核 19 的做法」的落点**：本内核提取
的 μ 调度，语义上不是「常规 AL 的罚因子更新」，而是**纯罚函数的强度调度**。
它是否合理，取决于 D7 是「文档与代码同错」还是「有意为之」—— 判定需要
独立推导（正确乘子更新式 + 它与 ``λ ≤ 0`` 符号约定的关系），且改 μ 必须
重标定 ``mu0``/``mu_max``/``mu_update_step``。**按铁律与 §12 的修正清单
（「不判定不许动代码」），本批只提取 legacy 的实际调度，一处不改。**

**生产工作点**（``probe_stress`` 实测，ρ = 0.5 均匀密度）：``Stress_loss_AL
= 0``（约束处处满足）⇒ ``AL_N_call`` 永不增长 ⇒ μ **恒停留在 ``μ0 = 0.01``**。
约束不激活时 AL 完全静默；只有优化推进、ρ 不均匀到让应力约束被违反后，
μ 才开始按 ``(1 + n/step)²`` 爬升。

------------------------------------------------------------------------------
五、为什么没有 ``fast/`` 层
------------------------------------------------------------------------------

本内核是**标量/逐数组的状态机**（一次赋值 + 一个标量公式 + 一个整数计数），
没有任何可向量化的循环或可近似替换的环节。按上级 README「ref/fast 的分层
只在有真实差异时才建立」的判据，**不建 ``fast/``**，只此一份。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ALParams",
    "ALState",
    "PRODUCTION_AL_PARAMS",
    "al_initial_state",
    "al_update",
]


@dataclass(frozen=True)
class ALParams:
    """AL 罚因子调度的超参数（legacy 类属性 ``MMA10:3841-3844``）。

    Attributes
    ----------
    mu0 : float
        μ 调度的起始值（``AL_stress_mu0``，生产 ``0.01``）。
    mu_max : float
        罚因子上限（``AL_stress_mu_max``，生产 ``1e3``）。
    mu_update_step : int
        调度步长（``AL_stress_mu_update_step``，生产 ``50``）——
        公式里以 ``n/step`` 出现，``n`` 是约束被违反的累计次数。
    """

    mu0: float = 0.01
    mu_max: float = 1e3
    mu_update_step: int = 50


#: 生产配置（照抄 ``MMA10:3841-3844``）
PRODUCTION_AL_PARAMS = ALParams(mu0=0.01, mu_max=1e3, mu_update_step=50)


@dataclass(frozen=True)
class ALState:
    """AL 迭代状态（legacy 挂在 ``optimizer_func_warper_AL`` 上的三个类属性）。

    Attributes
    ----------
    al_lambda : (N_elem_mech,) ndarray
        逐单元乘子。⚠ 生产初值 ``np.zeros`` 且 D7 保证写回的 ``al_lambda_new
        ≡ 0`` ⇒ **恒为 0**（纯罚函数）。本字段保留是为「将来修 D7 后乘子有
        非零历史」留位，不因此删除。
    al_mu : float
        罚因子。⚠ **无历史记忆**：每次由 ``al_n_call`` 重算，不读旧值。
    al_n_call : int
        约束被违反的累计次数（legacy 的 ``AL_N_call``）。
    """

    al_lambda: np.ndarray
    al_mu: float
    al_n_call: int


def al_initial_state(n_elem: int,
                     params: ALParams = PRODUCTION_AL_PARAMS) -> ALState:
    """初始 AL 状态，照抄 ``MMA10:3840-3845``：``λ = zeros``、``μ = μ0``、
    ``AL_N_call = 0``。

    Parameters
    ----------
    n_elem : int
        力学子网格单元数（``N_element_mech``）。
    params : ALParams
    """
    return ALState(al_lambda=np.zeros(n_elem), al_mu=params.mu0, al_n_call=0)


def al_update(
    al_lambda_new: np.ndarray,
    stress_loss_al: float,
    state: ALState,
    params: ALParams = PRODUCTION_AL_PARAMS,
) -> ALState:
    """一步 AL 参数更新。对应 legacy ``optimizer_func_warper_AL_v2.__call__``
    的 AL 段（``MMA10:4025-4032`` 的 λ 写回 + μ 调度 + clip，
    ``:4057-4058`` 的 ``AL_N_call`` 计数）。

    Parameters
    ----------
    al_lambda_new : (N_elem_mech,) ndarray
        内核 17 的 ``mech_stress_and_energy`` 返回的 ``al_lambda_new``
        （⚠ D7：恒为 0）。**引用写回**，不拷贝 —— 与 legacy 的
        ``AL_stress_lambda = AL_stress_lambda_new`` 别名赋值语义一致。
    stress_loss_al : float
        内核 17 的 ``stress_loss_al``（``Σ P_e / N_elem``）。**严格大于 0**
        才递增 ``al_n_call``（照抄 ``:4057`` 的 ``> 0``，不等号不放大）。
    state : ALState
        当前状态。
    params : ALParams

    Returns
    -------
    ALState
        新状态。⚠ **顺序纪律**：``al_mu`` 用**更新前**的 ``al_n_call``
        计算（``:4026`` 读旧值、``:4057-4058`` 才递增）—— 换序会改变 μ
        在第 ``k`` 轮的取值。
    """
    #: μ 调度：``min(μ0·(1 + n/step)², μ_max)``（``:4026-4031``，先用旧 n）
    al_mu = params.mu0 * (1 + state.al_n_call / params.mu_update_step) ** 2
    al_mu = min(al_mu, params.mu_max)
    #: 计数：``Stress_loss_AL > 0`` 才 ``AL_N_call += 1``（``:4057-4058``）
    al_n_call = state.al_n_call + (1 if stress_loss_al > 0 else 0)
    return ALState(al_lambda=al_lambda_new, al_mu=al_mu, al_n_call=al_n_call)
