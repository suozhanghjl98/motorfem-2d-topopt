"""转矩指标注册表（阶段 7：设计文档 §阶段 7 的 AL/`_v2` 去重产物）。

legacy ``nonlinear_solver_energy_MMA10.py``（下称 MMA10）的目标函数类
（``obj_func_parallel_AL`` ``:3054`` 与 ``obj_func_parallel_AL_v2`` ``:3410``）
各自硬编码了「算哪几个转矩、每个转矩依赖哪些前置求解」。两个类约 800 行
近重复 —— 设计文档 §7 给的去重方案不是合并成带 if 的一份，而是**抽转矩
指标注册表**：每个指标是一个类，声明它的名字与它对冻结磁导率求解的依赖。

**本模块只做声明，不产生数值**。三个指标在 legacy 里的实际依赖（来源行号）：

    Arkkio  ``:2987-2991``  直接算在 Newton 解 ``A`` 上（``Torque_Arkkio_coef``）
    VD      ``:2997-3001``  直接算在 Newton 解 ``A`` 上（``Torque_VD_coef``）
    flux    ``:3006-3033``  需要额外的 **d/q 两次冻结磁导率求解**
                            （``solve_field_vectorized_frozen_appearance_conductivity``，
                            ``:2938-2942`` 的 ``Rd``/``Rq`` 源项）才能算电感→磁链→转矩

所以 ``requires_frozen_solves`` 的取值是**有 legacy 依据的依赖声明**，不是
拍脑袋：Arkkio/VD 为 ``()``（不需要），flux 为 ``("d", "q")``（需要两组）。

生产路径只消费 `_v2` 版本（阶段 4 只搬了它），本注册表把「三个指标全都算」
这个 legacy 隐含事实显式化 —— 将来若想只算某个指标（跳过冻结求解省两次
线性解），消费方按注册表逐项检查即可，不必再改 ``em_worker`` 的硬编码分支。

⚠ **顺序纪律**：注册表的元组顺序只是声明顺序，**不是** ``em_step`` 的返回
顺序（那是 ``(t, T_Arkkio, ..., T_flux, ..., T_VD, ...)``，照抄 legacy
``:3037-3041`` 的 10 元组）。涉及「发布给 MMA 的 f 向量顺序」的常量在
``published_layout.py``，与这里互相独立。
"""

from __future__ import annotations

from typing import Tuple

__all__ = [
    "TorqueMetric",
    "ArkkioTorque",
    "VirtualWorkTorque",
    "FrozenMuFluxTorque",
    "TORQUE_REGISTRY",
    "requires_frozen_solves",
]


class TorqueMetric:
    """转矩指标基类：声明名字 + 对冻结磁导率求解的依赖。

    子类只覆写两个类属性，不携带行为 —— 这是**声明层**（不是计算层）。
    ``em_step`` 的硬编码计算路径逐字不动，本类只用于回答「这个指标算不算、
    要不要先跑冻结求解」这类元问题。
    """

    #: 指标名（与 legacy 的 detail 键 / plot_torque 图例一致：Arkkio / flux / VD）
    name: str = ""
    #: 需要的冻结磁导率源项组（``Rd``/``Rq`` 的标签）。``()`` = 不需要。
    requires_frozen_solves: Tuple[str, ...] = ()


class ArkkioTorque(TorqueMetric):
    """Arkkio 转矩（``MMA10:2987-2991``）。直接算在 Newton 解上，不需要冻结求解。"""

    name = "Arkkio"
    requires_frozen_solves = ()


class VirtualWorkTorque(TorqueMetric):
    """虚位移（VD）转矩（``MMA10:2997-3001``）。直接算在 Newton 解上，不需要冻结求解。"""

    name = "VD"
    requires_frozen_solves = ()


class FrozenMuFluxTorque(TorqueMetric):
    """冻结磁导率磁链转矩（``MMA10:3006-3033``）。需要 d/q 两次冻结求解。"""

    name = "flux"
    requires_frozen_solves = ("d", "q")


#: 全部指标的声明顺序。⚠ 声明顺序，**不是** ``em_step`` 的返回顺序。
TORQUE_REGISTRY: Tuple[TorqueMetric, ...] = (
    ArkkioTorque,
    VirtualWorkTorque,
    FrozenMuFluxTorque,
)


def requires_frozen_solves(metric: TorqueMetric) -> bool:
    """这个指标需要跑冻结磁导率求解吗？``("d","q")`` ⇒ True，``()`` ⇒ False。"""
    return bool(metric.requires_frozen_solves)
