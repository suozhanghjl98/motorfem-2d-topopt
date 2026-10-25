# -*- coding: utf-8 -*-
"""motorfem.config.mesh_config —— 网格绑定量口径的**伴生配置文件**（``<stem>.motor.json``）。

把一个电机的"口径"（极槽/半径/相带生成器参数/时间轴/周期-反周期/功角）从**代码硬编码
默认**，改为**与网格同名、同目录的伴生 JSON**，在 ``FieldSystem`` 读网格时自动加载。

为什么需要它（2026-10-14）：一个电机的口径原先散落在 **≥8 份硬编码副本**里且已互相漂移
（``_apply_motor``/``_cfg6s4p``/``_cfg_6s4p``/``_probe_full``/``_probe_gap_fft``… 的
``EM_MS_boundary_A_eq``、``arkkio_parts_exponent``、功角各不相同）⇒ 漏一处就静默算错
（例：6s4p 漏设 ``A_eq=True`` ⇒ 净磁通不守恒，气隙 ``B_r`` 的 DC=0.394T，物理必须为 0）。

命名与确定性写出沿用 ``tools/freeze_msh.py`` 的 ``.model.json`` 约定（``sort_keys=True``、
LF-only）；但**文件独立**：``.model.json`` 是 freeze 从网格**再生**的指纹，本文件是**标定输入**，
二者生命周期不同（重跑 freeze 不能覆盖标定）。

---------------------------------------------------------------------------- 覆盖语义
采用**默认值比对**：只覆盖"``cfg`` 该字段仍等于 ``RunConfig()`` 类默认值"的字段，
调用方**显式设过**的（≠ 默认）一概保留。好处：零侵入（不改 ``RunConfig.__setattr__``、
不引入显式设置追踪），且探针"先改 ``cfg.electrical.winding.current_amp`` 再 ``FieldSystem``"
（饱和扫描）不会被吞掉。代价：无法区分"显式设成默认值"与"未设"——二者值相同，覆盖无害。
"""
from __future__ import annotations

import json
import warnings
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RunConfig

__all__ = [
    "MeshConfigError", "MESH_CONFIG_SUFFIX", "SCHEMA_VERSION", "MESH_BOUND_FIELDS",
    "mesh_config_path", "load_mesh_config", "project_run_config", "apply_mesh_config",
]

MESH_CONFIG_SUFFIX = ".motor.json"
SCHEMA_VERSION = 1

#: 允许被网格配置覆盖的字段（"点路径"）；**超出此集合的键一律报错**——
#: 防止伴生文件悄悄改动优化器/目标/滤波等与网格无关的量（那是 RunConfig/CLI/GUI 的职责）。
#: 注意：不收录 ``bh_curve_csv``（相对路径语义不同，仍走 ``FieldSystem(..., bh_csv)``）。
MESH_BOUND_FIELDS: tuple = (
    # 几何：极槽、周期份数、半径、Arkkio 口径
    "geometry.Z", "geometry.pole_pairs", "geometry.parts",
    "geometry.stator_inner_radius_m", "geometry.stator_outer_radius_m",
    "geometry.rotor_outer_radius_m", "geometry.shaft_radius_m",
    "geometry.arkkio_gap_m", "geometry.arkkio_parts_exponent",
    "geometry.rotor_design_half_pole",
    # 绕组：极槽、电流、star-of-slots 生成器参数
    "electrical.winding.Z", "electrical.winding.pole_pairs",
    "electrical.winding.parts", "electrical.winding.current_amp",
    "electrical.winding.N_conductor",
    "electrical.winding.inner_ref", "electrical.winding.outer_ref",
    "electrical.winding.outer_shift",
    "electrical.winding.phase_from_generator",
    # 时间轴与功角（★ 2026-10-14：易漏且直接决定跨电机可比性）
    "time.time_simulation_s", "time.N_point_simulation",
    "time.rotor_position0_deg", "time.phaseA_offset_deg",
    # 主从边界类型（★ 周期/反周期由扇区的电角跨度定，最易漏）
    "solver.EM_MS_boundary_A_eq",
)


class MeshConfigError(ValueError):
    """伴生配置损坏 / schema 不符 / 字段非法 / ``require`` 模式下缺失。"""


def mesh_config_path(msh_path) -> Path:
    """``<stem>.motor.json`` —— 与网格同目录同 stem。"""
    return Path(msh_path).with_suffix(MESH_CONFIG_SUFFIX)


def load_mesh_config(msh_path, *, check_hash: bool = True) -> Optional[Dict[str, Any]]:
    """读伴生文件并返回其 ``run_config`` 部分。

    缺失 → ``None``（不是错误：现有 15+ 网格都还没有伴生文件）。
    损坏 / schema 不符 / ``file`` 不符 / 白名单外键 → ``MeshConfigError``（**响声失败**——
    标定文件坏了却静默回落默认 = 用一个错的口径跑，最坏的结果是"安静算错"）。
    """
    p = mesh_config_path(msh_path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:  # noqa: BLE001
        raise MeshConfigError(f"{p}: 无法解析 JSON: {e}") from e
    if not isinstance(d, dict):
        raise MeshConfigError(f"{p}: 顶层必须是对象")
    for key in ("schema_version", "file", "run_config"):
        if key not in d:
            raise MeshConfigError(f"{p}: 缺键 {key!r}")
    if int(d["schema_version"]) != SCHEMA_VERSION:
        raise MeshConfigError(
            f"{p}: schema_version={d['schema_version']} != {SCHEMA_VERSION}")
    if Path(msh_path).name != d["file"]:
        raise MeshConfigError(
            f"{p}: file={d['file']!r} 与网格名 {Path(msh_path).name!r} 不符"
            "（伴生文件被复制到了别的网格？）")
    if not isinstance(d["run_config"], dict):
        raise MeshConfigError(f"{p}: run_config 必须是对象")
    _validate_paths(d["run_config"], p)
    if check_hash and d.get("sha256"):
        import hashlib
        h = hashlib.sha256(Path(msh_path).read_bytes()).hexdigest()
        if h != d["sha256"]:
            warnings.warn(f"{p}: sha256 与网格不符（网格可能已重生成）",
                          stacklevel=2)
    return dict(d["run_config"])


def project_run_config(cfg: RunConfig) -> Dict[str, Any]:
    """把 ``cfg`` 的**白名单字段**投影成嵌套 dict（供写文件 / 相等性断言）。"""
    out: Dict[str, Any] = {}
    for path in MESH_BOUND_FIELDS:
        obj: Any = cfg
        groups, leaf = path.split(".")[:-1], path.split(".")[-1]
        for g in groups:
            obj = getattr(obj, g)
        node = out
        for g in groups:
            node = node.setdefault(g, {})
        node[leaf] = getattr(obj, leaf)
    return out


def _validate_paths(d: Dict[str, Any], p: Path, prefix: str = "") -> List[str]:
    """递归校验 dict 的每个**叶子**路径都在白名单内；返回叶子路径清单。"""
    leaves: List[str] = []
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            leaves += _validate_paths(v, p, prefix=f"{name}.")
        else:
            if name not in MESH_BOUND_FIELDS:
                raise MeshConfigError(f"{p}: 字段 {name!r} 不在网格绑定量白名单内")
            leaves.append(name)
    return leaves


def apply_mesh_config(cfg: RunConfig, msh_path, *, mode: str = "auto") -> Optional[Dict]:
    """读伴生文件并按**默认值比对**就地覆盖 ``cfg``；返回应用的 ``run_config``（未加载 → None）。

    ``mode``：``auto``（缺失静默跳过）| ``ignore``（完全不加载，逃生口）| ``require``（缺失报错）。
    """
    if mode == "ignore":
        return None
    d = load_mesh_config(msh_path)
    if d is None:
        if mode == "require":
            raise MeshConfigError(f"{mesh_config_path(msh_path)} 不存在（require 模式）")
        return None
    base = RunConfig()            # 干净默认，用于"仍等于默认？"比对
    _overlay_defaultwise(cfg, base, d, prefix="")
    return d


def _overlay_defaultwise(cfg_obj, base_obj, d: Dict[str, Any], prefix: str) -> None:
    by_name = {f.name: f for f in fields(cfg_obj)}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if k not in by_name:
            raise MeshConfigError(f"未知字段 {name!r}（RunConfig 无此路径）")
        cur = getattr(cfg_obj, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                raise MeshConfigError(f"{name}: 期望对象，得到 {type(v).__name__}")
            _overlay_defaultwise(cur, getattr(base_obj, k), v, prefix=f"{name}.")
        else:
            if name not in MESH_BOUND_FIELDS:
                raise MeshConfigError(f"字段 {name!r} 不在网格绑定量白名单内")
            if cur == getattr(base_obj, k):     # 仅当仍是类默认值才覆盖
                setattr(cfg_obj, k, v)
