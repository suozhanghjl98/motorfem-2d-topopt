"""motorfem.config —— 阶段 2 参数对象。

``RunConfig`` 聚合 14 个 dataclass（几何/材料/电气/离散/时间/拓扑/滤波/目标/
应力/力学/求解/MMA/输出/运行时），JSON 可序列化；``RunConfig.from_legacy``
从 legacy 命名空间快照构造。设计见 ``docs/重构设计文档.md`` §阶段 2。

导入本模块不得产生任何副作用。
"""

from .config import (
    ALConfig, DiscretizationConfig, ElectricalConfig, FilterConfig,
    GeometryConfig, MaterialConfig, MMAConfig, MechConfig, ObjectiveConfig,
    OutputConfig, RunConfig, RuntimeConfig, SolverConfig, StressConfig,
    TimeConfig, TopologyConfig, WindingSpec, resolve_bh_csv,
)
from .enums import RelStressType, StressType
from .mesh_config import (
    MESH_BOUND_FIELDS, MESH_CONFIG_SUFFIX, SCHEMA_VERSION, MeshConfigError,
    apply_mesh_config, load_mesh_config, mesh_config_path, project_run_config,
)
from .resolved import ResolvedConfig, integer_slot_winding_factor

__all__ = [
    "GeometryConfig", "MaterialConfig", "WindingSpec", "ElectricalConfig",
    "DiscretizationConfig", "TimeConfig", "TopologyConfig", "FilterConfig",
    "ObjectiveConfig", "ALConfig", "StressConfig", "MechConfig",
    "SolverConfig", "MMAConfig", "OutputConfig", "RuntimeConfig", "RunConfig",
    "ResolvedConfig", "integer_slot_winding_factor", "resolve_bh_csv",
    "StressType", "RelStressType",
    "MeshConfigError", "MESH_BOUND_FIELDS", "MESH_CONFIG_SUFFIX", "SCHEMA_VERSION",
    "apply_mesh_config", "load_mesh_config", "mesh_config_path", "project_run_config",
]
