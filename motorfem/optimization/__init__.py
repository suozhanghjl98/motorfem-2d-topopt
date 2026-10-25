"""motorfem.optimization —— 阶段 7：优化层声明（转矩指标注册表 + 发布布局）。

本包**不产生数值**，只承载「优化层的事实」：

- ``torque_registry``：三个转矩指标的声明（Arkkio / VD / flux），每个标出它
  对冻结磁导率求解的依赖（设计文档 §7 的 AL/`_v2` 去重产物）。
- ``published_layout``：发布给 MMA 的 ``f``/``df`` 向量的行序常量与长度断言
  （设计文档 §7 的顺序纪律）。

数值计算路径（``motorfem/fem/``）逐字不动；本包只被它们**读**（断言/元数据）。
"""

from .published_layout import (PUBLISHED_NAMES, PUBLISHED_SIZE,
                               assert_published_layout)
from .torque_registry import (ArkkioTorque, FrozenMuFluxTorque, TorqueMetric,
                              TORQUE_REGISTRY, VirtualWorkTorque,
                              requires_frozen_solves)

__all__ = [
    "TorqueMetric",
    "ArkkioTorque",
    "VirtualWorkTorque",
    "FrozenMuFluxTorque",
    "TORQUE_REGISTRY",
    "requires_frozen_solves",
    "PUBLISHED_NAMES",
    "PUBLISHED_SIZE",
    "assert_published_layout",
]
