"""motorfem.mesh —— 网格与区域数据契约（阶段 2 + 阶段 3）。

``MotorMesh`` 读 .msh（读取语义与 legacy 一致）+ 划分区域：默认按 tag 表
（阶段 2），网格携带 ``$PhysicalNames`` 时改由**物理组**构造（阶段 3，断掉
「tag 恰好对齐」）。``RegionMap`` 是「旧名 → 单元集合」的冻结映射（设计文档
阶段 3 的过渡支点）。三道闸（``require_regions`` / ``verify_fingerprint`` /
``RegionMap.check_partition``）在 ``gates.py`` 与 ``RegionMap`` 上。

导入本模块不得产生任何副作用（不读文件、不 import legacy）。
"""

from .gates import require_regions, verify_fingerprint
from .mesh import DEFAULT_TAG_TABLE, MotorMesh, RegionMap, gmsh_reader
from .physical import PhysicalGroups, read_physical_groups
from .physical_groups import (
    PMSM_BOUNDARY_GROUPS,
    PMSM_FACE_GROUPS,
    SYNRM_BOUNDARY_GROUPS,
    SYNRM_FACE_GROUPS,
    SYNRM_ROLE_TO_GROUP,
    SYNRM_SLOT_GROUP_NAMES,
    PhysicalGroup,
)

__all__ = [
    "MotorMesh",
    "RegionMap",
    "gmsh_reader",
    "DEFAULT_TAG_TABLE",
    "require_regions",
    "verify_fingerprint",
    "PhysicalGroups",
    "read_physical_groups",
    "PhysicalGroup",
    "SYNRM_FACE_GROUPS",
    "SYNRM_BOUNDARY_GROUPS",
    "SYNRM_ROLE_TO_GROUP",
    "SYNRM_SLOT_GROUP_NAMES",
    "PMSM_FACE_GROUPS",
    "PMSM_BOUNDARY_GROUPS",
]
