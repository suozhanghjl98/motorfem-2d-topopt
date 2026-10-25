"""motorfem.mesh.physical_groups —— 阶段 3：区域命名契约的常量物理组表

设计文档 §阶段 3：gmsh 侧对每个面调 ``addPhysicalGroup`` + ``setPhysicalName``，
用**显式常量物理组 tag 表**（不依赖 gmsh 自增）。本模块是求解器侧的事实来源：

* ``SYNRM_FACE_GROUPS`` / ``SYNRM_BOUNDARY_GROUPS``：SynRM（生产）的常量表。
  面组的物理组 tag 取 **= legacy elementary tag（1..26）**（``_SYNRM_FACE_TAGS``
  与 ``motorfem/mesh/mesh.py`` 的 ``DEFAULT_TAG_TABLE`` 逐值一致，测试钉住）。
  「物理组 tag == elementary tag」可行，因为 gmsh 里这是**两个独立命名空间**：
  v4.1 的 ``$Elements`` 块头写的是 elementary tag（``gmsh_reader`` 读的正是它），
  物理组归属写在 ``$Entities`` —— 互不干扰（已用 gmsh 4.13.1 真实输出验证）。
* ``PMSM_FACE_GROUPS`` / ``PMSM_BOUNDARY_GROUPS``：PMSM 侧**只定 name + tag**，
  elementary 成员未知（``model_pmsm1.msh`` 无生成器，freeze_msh 记 UNKNOWN），
  由生成器运行时通过 ``addPhysicalGroup(dim, [实际实体], tag=T)`` 填 —— 未对账。
* ``SYNRM_ROLE_TO_GROUP`` / ``SYNRM_SLOT_GROUP_NAMES``：legacy 键 → 物理组名的
  角色映射，``RegionMap.from_physical_groups`` 靠它把物理组还原成旧名映射。

本模块**不 import gmsh、不 import 其他 motorfem 模块**（纯数据），生成器脚本
（``legacy/pytorch/my_gmsh/*.py``）以自包含字面 dict 复刻同一张表，两边靠
``tests/mesh/test_physical.py`` 的静态测试钉住（防漂移）。
"""

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "PhysicalGroup",
    "SYNRM_FACE_GROUPS",
    "SYNRM_BOUNDARY_GROUPS",
    "SYNRM_ROLE_TO_GROUP",
    "SYNRM_SLOT_GROUP_NAMES",
    "PMSM_FACE_GROUPS",
    "PMSM_BOUNDARY_GROUPS",
]


@dataclass(frozen=True)
class PhysicalGroup:
    """一个物理组：名（求解器/生成器共用的词汇）+ 显式物理组 tag + 成员。

    Attributes
    ----------
    name : str
        规范名（``rotor.shaft`` / ``winding.slot{k}`` / ``bnd.outer`` …）。
    dim : int
        gmsh 维度（面 = 2，边界 = 1）。
    tag : int
        **显式**物理组 tag（``addPhysicalGroup(..., tag=T)``，不依赖 gmsh 自增）。
    entity_tags : tuple
        该物理组覆盖的 elementary entity tag（SynRM 面组已对账 legacy tag 表；
        SynRM 边界与 PMSM 全部留空，由生成器运行时填）。
    """

    name: str
    dim: int
    tag: int
    entity_tags: Tuple[int, ...] = ()


# ---------------------------------------------------------------- SynRM
#: SynRM 面组：物理组 tag == legacy elementary tag（设计决策，见模块 docstring）。
#: 与 DEFAULT_TAG_TABLE 逐值一致 —— 由 tests/mesh/test_physical.py 钉住。
_SYNRM_FACE_TAGS: Dict[str, Tuple[int, ...]] = {
    "rotor.shaft":       (1,),
    "rotor.core.main":   (2,),
    "rotor.core.mirror": (3,),
    "rotor.gap":         (4, 5),    # 转子侧气隙（2 个 elementary 面）
    "stator.gap":        (6, 7),    # 定子侧气隙（2 个 elementary 面）
    "stator.core":       (8,),
}

#: SynRM 槽数 = Z / parts = 36 / 4 = 18（每个槽 1 个 elementary 面，tag 9..26）。
_SYNRM_SLOT_COUNT = 18

SYNRM_FACE_GROUPS: Tuple[PhysicalGroup, ...] = tuple(
    PhysicalGroup(name, 2, min(tags), tags)
    for name, tags in _SYNRM_FACE_TAGS.items()
) + tuple(
    PhysicalGroup(f"winding.slot{k}", 2, 9 + k, (9 + k,))
    for k in range(_SYNRM_SLOT_COUNT)
)

#: 边界组（dim=1）：tag 取 201..206（避开 legacy elementary tag 1..185 的习惯区间，
#: 边界成员由生成器按 setPeriodic / curveloop 结构接线，运行时确认等真重生成）。
SYNRM_BOUNDARY_GROUPS: Tuple[PhysicalGroup, ...] = (
    PhysicalGroup("bnd.periodic.master", 1, 201),
    PhysicalGroup("bnd.periodic.slave", 1, 202),
    PhysicalGroup("bnd.outer", 1, 203),
    PhysicalGroup("bnd.rotor.outer", 1, 204),
    PhysicalGroup("bnd.gap.stator", 1, 205),
    PhysicalGroup("bnd.shaft.fixed", 1, 206),
)

#: legacy 区域键 → 物理组名（from_physical_groups 的还原映射；键与 DEFAULT_TAG_TABLE 一致）
SYNRM_ROLE_TO_GROUP: Dict[str, str] = {
    "element_rotor_shaft": "rotor.shaft",
    "element_rotor_core_m": "rotor.core.main",
    "element_rotor_core_s": "rotor.core.mirror",
    "element_rotor_gap": "rotor.gap",
    "element_stator_gap": "stator.gap",
    "element_stator_core": "stator.core",
}

#: 槽组名（按全局槽序；element_stator_coil 的 list 顺序由它决定）
SYNRM_SLOT_GROUP_NAMES: Tuple[str, ...] = tuple(
    f"winding.slot{k}" for k in range(_SYNRM_SLOT_COUNT)
)


# ---------------------------------------------------------------- PMSM
#: PMSM 面组：只定 name + tag，elementary 成员留空（生成器运行时填，未对账）。
#: 结构与 ``gmsh_4p6s_PMSM.py`` 一致（19 面 = 轴 + 转子 + 6 永磁 + 4 气隙 +
#: 定子 + 6 绕组；与 ``model_pmsm1`` 的 dim2=19 同构但**未确认同源**）。
#: 6 个绕组面 = 3 槽 × 2 层（内/外线圈，槽主序），命名 ``winding.layer{k}``
#: （k = 创建序）—— 与 SynRM 的 ``winding.slot{k}`` 区分，避免「每槽一面」的
#: 假设被误套到 PMSM。设计文档词汇里的 ``rotor.air.{k}`` 当前生成器无对应面，
#: 仅保留在文档词汇里，不入表。
PMSM_FACE_GROUPS: Tuple[PhysicalGroup, ...] = (
    PhysicalGroup("rotor.shaft", 2, 1),
    PhysicalGroup("rotor.core", 2, 2),
    PhysicalGroup("rotor.gap", 2, 10),
    PhysicalGroup("stator.gap", 2, 11),
    PhysicalGroup("stator.core", 2, 12),
) + tuple(
    PhysicalGroup(f"winding.layer{k}", 2, 14 + k)
    for k in range(6)                    # 3 槽 × 2 层
) + tuple(
    PhysicalGroup(f"rotor.pm.{k}", 2, 20 + k)
    for k in range(6)                    # 每对极 3 块 × 2 极
)

#: PMSM 边界组（dim=1）：名 + tag（成员由生成器运行时填，未对账）。
PMSM_BOUNDARY_GROUPS: Tuple[PhysicalGroup, ...] = (
    PhysicalGroup("bnd.periodic.master", 1, 201),
    PhysicalGroup("bnd.periodic.slave", 1, 202),
    PhysicalGroup("bnd.outer", 1, 203),
    PhysicalGroup("bnd.rotor.outer", 1, 204),
    PhysicalGroup("bnd.gap.stator", 1, 205),
    PhysicalGroup("bnd.shaft.fixed", 1, 206),
)
