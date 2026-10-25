# -*- coding: utf-8 -*-
"""tools/design_gui/core.py —— 手绘结构 GUI 的纯逻辑层。

职责
----
1. **拓扑 bundle**：用与 ``FieldSystem`` 相同的前半段（``apply_mesh_config`` →
   ``MotorMesh`` → ``build_field_topology``）构建几何，但**不建预组装/求解器**。
2. **几何 payload**：把 ``top.layer.xy/tri`` + 区域色 + 主/从属映射序列化成
   前端可直接画的 JSON。
3. **滤波缓存 / 预览**：按半径取 ``compute_filter`` 矩阵，计算手绘 ρ 的
   “优化实际起点 ρ”（``W @ x0``）。
4. **NPZ 导入导出**：schema 见 ``README.md``；导出的 ``x0`` 可直接交给
   ``tools/scripts/run_auto_schedule.py --x0-npz ... --x0-key x0``。

⚠ 纪律
------
- 主单元顺序 = ``list(top.element_topo_opti_m)``（set 迭代序，见 CLAUDE.md 红线 3）。
  导出时把 ``master_cells`` 一并写入；导入时按元素 id 重映射。
- 本模块不碰任何 FEM 数值路径；滤波矩阵只用于“起点预览/前像”。
- 文件读写一律显式 ``encoding='utf-8'``。
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

#: 可选的电机网格（与 ``run_auto_schedule.MESHES`` 对齐）。
MOTORS: Dict[str, Dict[str, str]] = {
    "6s4p": {
        "label": "6s4p（6 槽 / 4 极）",
        "mesh": str(_REPO / "legacy" / "pytorch" / "my_gmsh" / "model_6s4p.msh"),
        "csv": str(_REPO / "legacy" / "pytorch" / "50CS470.csv"),
    },
    "synrm": {
        "label": "synrm（36 槽 / mirror4）",
        "mesh": str(_REPO / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh"),
        "csv": str(_REPO / "legacy" / "pytorch" / "50CS470.csv"),
    },
}

DEFAULT_MOTOR = "6s4p"
DEFAULT_OUT_DIR = _REPO / "tools" / "_out" / "design_gui" / "designs"
DEFAULT_RUN_DIR = _REPO / "tools" / "_out" / "design_gui" / "runs"

#: 设计变量下限（与 ``init_density.filter_inverse_rho`` / legacy ``min_SIMP`` 一致）。
MIN_DENSITY = 1e-6

#: 与 ``tools/optim_gui/server.py::_build_geometry`` 相同的色表。
BASE_COLORS = {0: "rgb(140,140,148)", 1: "rgb(230,184,107)", 2: "rgb(255,255,255)"}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _to_text(v: Any) -> str:
    """把 NPZ 里读出的 0 维字符串/字节数组转成 ``str``。"""
    a = np.asarray(v)
    if a.ndim == 0:
        x = a.item()
    else:
        x = a.ravel()[0]
    if isinstance(x, (bytes, np.bytes_)):
        return bytes(x).decode("utf-8", errors="replace")
    return str(x)


def _to_float(v: Any) -> Optional[float]:
    try:
        a = np.asarray(v)
        if a.size == 0:
            return None
        return float(a.ravel()[0])
    except Exception:  # noqa: BLE001
        return None


def safe_slug(name: str, *, default: str = "design", max_len: int = 60) -> str:
    """把用户输入变成安全的文件名片段（保留中文，去掉路径分隔符/控制字符）。"""
    s = (name or "").strip()
    s = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip(" .")
    if not s:
        s = default
    return s[:max_len]


def mesh_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_motor(motor: Optional[str]) -> Dict[str, str]:
    key = (motor or DEFAULT_MOTOR).strip()
    if key not in MOTORS:
        raise ValueError(f"未知电机 {key!r}；可选：{sorted(MOTORS)}")
    return MOTORS[key]


# ---------------------------------------------------------------------------
# 拓扑 bundle
# ---------------------------------------------------------------------------
@dataclass
class Bundle:
    """一次轻量几何构建的全部只读状态。"""

    motor: str
    cfg: Any
    top: Any
    mesh_path: Path
    mesh_sha256: str
    csv_path: Path

    #: 节点坐标（m）与单元连接表（``layer.xy/tri``，行号 = 单元 id）
    xy: np.ndarray
    tri: np.ndarray
    color_id: np.ndarray

    #: 主/从属/显示映射
    master_cells: np.ndarray          # (n_master,) int64，顺序 = list(element_topo_opti_m)
    slave_cells: np.ndarray           # (n_slave,) int64，顺序 = dict.keys()
    design_cells: np.ndarray          # (n_design,) = concat(master, slave)
    design_master: np.ndarray         # (n_design,) int32，显示单元 → 主索引
    slave_parent: np.ndarray          # (n_slave,) int32，从属 → 主索引

    #: 主单元几何
    centroids_master: np.ndarray      # (n_master, 2) m
    area_master: np.ndarray           # (n_master,) m²
    radii_master: np.ndarray          # (n_master,) m
    angles_master_deg: np.ndarray     # (n_master,) deg

    #: 半径小于等于它时滤波器 ≈ 单位阵（由最近邻质心距离算出）
    identity_radius_m: float

    filter_cache: Dict[float, Any] = field(default_factory=dict)

    _payload_raw: Optional[bytes] = None
    _payload_gz: Optional[bytes] = None

    # ---------------------------------------------------------------- 属性
    @property
    def n_master(self) -> int:
        return int(self.master_cells.size)

    @property
    def n_slave(self) -> int:
        return int(self.slave_cells.size)

    @property
    def n_design(self) -> int:
        return int(self.design_cells.size)

    @property
    def n_element(self) -> int:
        return int(self.tri.shape[0])

    @property
    def default_filter_radius_m(self) -> float:
        return float(self.cfg.filter.filter_radius_m)

    @property
    def master_sector_deg(self) -> Tuple[float, float]:
        """主设计域的角向范围（由质心角实测，用于前端虚线）。"""
        a = self.angles_master_deg
        if a.size == 0:
            return (0.0, 0.0)
        return (float(np.min(a)), float(np.max(a)))


def build_bundle(motor: str = DEFAULT_MOTOR, *, mesh_path: Optional[str] = None,
                 csv_path: Optional[str] = None) -> Bundle:
    """构建轻量几何 bundle（约 7~8 s）。**不建预组装/求解器。**"""
    spec = resolve_motor(motor)
    mesh = Path(mesh_path or spec["mesh"])
    csv = Path(csv_path or spec["csv"])
    if not mesh.exists():
        raise FileNotFoundError(f"网格不存在：{mesh}")
    if not csv.exists():
        raise FileNotFoundError(f"B-H 曲线不存在：{csv}")

    import sys

    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    from motorfem.config import RunConfig, apply_mesh_config
    from motorfem.fem.mesh_builder import build_field_topology
    from motorfem.mesh import MotorMesh
    from motorfem.winding.layout import star_of_slots

    cfg = RunConfig()
    apply_mesh_config(cfg, str(mesh))
    w = cfg.electrical.winding
    layout = None
    if getattr(w, "phase_from_generator", False):
        layout = star_of_slots(
            cfg.geometry.Z, cfg.geometry.pole_pairs, 3,
            parts=cfg.geometry.parts, layers=2,
            inner_ref=w.inner_ref, outer_ref=w.outer_ref,
            outer_shift=w.outer_shift,
        )
    mesh_obj = MotorMesh(str(mesh), layout=layout)
    top = build_field_topology(cfg, mesh_obj)

    xy = np.asarray(top.layer.xy, dtype=np.float64)
    tri = np.asarray(top.layer.tri, dtype=np.int64)
    n_element = int(tri.shape[0])

    master_cells = np.asarray(list(top.element_topo_opti_m), dtype=np.int64)
    slave_map = top.element_topo_opti_s
    slave_cells = np.asarray(list(slave_map.keys()), dtype=np.int64)
    n_master, n_slave = master_cells.size, slave_cells.size

    master_index = {int(e): i for i, e in enumerate(master_cells)}
    design_cells = np.concatenate([master_cells, slave_cells]).astype(np.int64)
    design_master = np.empty(design_cells.size, dtype=np.int32)
    design_master[:n_master] = np.arange(n_master, dtype=np.int32)
    for j, e in enumerate(slave_cells):
        parent = int(slave_map[int(e)])
        if parent not in master_index:
            raise RuntimeError(f"从属单元 {int(e)} 的主单元 {parent} 不在主设计域中")
        design_master[n_master + j] = master_index[parent]
    slave_parent = design_master[n_master:].copy()

    # ---- 区域色：与 optim_gui._build_geometry 同口径 ----
    color_id = np.zeros(n_element, dtype=np.int32)
    topo_all = set(int(e) for e in master_cells) | set(int(e) for e in slave_cells)
    coil = set()
    for grp in top.regions.region("element_stator_coil"):
        coil |= set(int(e) for e in grp)
    for e in coil - topo_all:
        color_id[e] = 1
    for e in top.regions.region("element_rotor_gap"):
        color_id[int(e)] = 2
    for e in topo_all:
        color_id[int(e)] = 3

    # ---- 主单元几何 ----
    centroids = xy[tri[master_cells]].mean(axis=1)
    area = np.asarray(top.em.area, dtype=np.float64)[master_cells]
    radii = np.hypot(centroids[:, 0], centroids[:, 1])
    angles = np.degrees(np.arctan2(centroids[:, 1], centroids[:, 0]))

    # ---- 恒等滤波半径：最近邻质心距离的一半 ----
    identity_r = 1e-4
    if centroids.shape[0] > 1:
        try:
            from scipy.spatial import cKDTree

            d, _ = cKDTree(centroids).query(centroids, k=2)
            nn = np.asarray(d[:, 1], dtype=float)
            nn = nn[np.isfinite(nn) & (nn > 0)]
            if nn.size:
                identity_r = 0.5 * float(np.min(nn))
        except Exception:  # noqa: BLE001
            pass

    return Bundle(
        motor=(motor or DEFAULT_MOTOR),
        cfg=cfg,
        top=top,
        mesh_path=mesh,
        mesh_sha256=mesh_sha256(mesh),
        csv_path=csv,
        xy=xy,
        tri=tri,
        color_id=color_id,
        master_cells=master_cells,
        slave_cells=slave_cells,
        design_cells=design_cells,
        design_master=design_master,
        slave_parent=slave_parent,
        centroids_master=centroids,
        area_master=area,
        radii_master=radii,
        angles_master_deg=angles,
        identity_radius_m=identity_r,
    )


# ---------------------------------------------------------------------------
# 几何 payload
# ---------------------------------------------------------------------------
def geometry_payload(bundle: Bundle) -> Dict[str, Any]:
    """前端一次性拉取的静态几何（坐标单位 mm，减少 JSON 体积）。"""
    if bundle._payload_raw is not None:
        return json.loads(bundle._payload_raw.decode("utf-8"))

    xy_mm = np.round(bundle.xy * 1e3, 5)
    cen_mm = np.round(bundle.centroids_master * 1e3, 5)
    lo, hi = bundle.master_sector_deg
    payload: Dict[str, Any] = {
        "version": 1,
        "units": "mm",
        "motor": bundle.motor,
        "mesh": {"path": str(bundle.mesh_path), "sha256": bundle.mesh_sha256},
        "n_node": int(xy_mm.shape[0]),
        "n_elem": bundle.n_element,
        "n_master": bundle.n_master,
        "n_slave": bundle.n_slave,
        "n_design": bundle.n_design,
        "xy": xy_mm.tolist(),
        "tri": bundle.tri.tolist(),
        "color_id": bundle.color_id.tolist(),
        "master_cells": bundle.master_cells.tolist(),
        "slave_cells": bundle.slave_cells.tolist(),
        "design_cells": bundle.design_cells.tolist(),
        "design_master": bundle.design_master.tolist(),
        "centroids_master": cen_mm.tolist(),
        "area_master": bundle.area_master.tolist(),
        "radii_master": (bundle.radii_master * 1e3).tolist(),
        "angles_master_deg": bundle.angles_master_deg.tolist(),
        "master_sector_deg": [lo, hi],
        "identity_filter_radius_mm": float(bundle.identity_radius_m * 1e3),
        "default_filter_radius_mm": float(bundle.default_filter_radius_m * 1e3),
        "base_colors": {str(k): v for k, v in BASE_COLORS.items()},
        "cfg_hint": {
            "max_density": float(bundle.cfg.topology.max_density),
            "min_density": float(bundle.cfg.topology.min_SIMP),
            "filter_p_exp": float(bundle.cfg.filter.filter_p_exp),
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    bundle._payload_raw = raw
    bundle._payload_gz = gzip.compress(raw, 6)
    return payload


def geometry_bytes(bundle: Bundle, *, use_gzip: bool = False) -> bytes:
    """几何 payload 的原始/压缩字节（带缓存）。"""
    if bundle._payload_raw is None:
        geometry_payload(bundle)
    assert bundle._payload_raw is not None
    if use_gzip:
        if bundle._payload_gz is None:
            bundle._payload_gz = gzip.compress(bundle._payload_raw, 6)
        return bundle._payload_gz
    return bundle._payload_raw


# ---------------------------------------------------------------------------
# 滤波 / 预览
# ---------------------------------------------------------------------------
def get_filter(bundle: Bundle, radius_m: float):
    """按半径取密度滤波矩阵（缓存）。半径必须 >0（0 会让行归一化除零）。"""
    r = float(radius_m)
    if not np.isfinite(r):
        raise ValueError("滤波半径必须是有限数")
    if r <= 0:
        raise ValueError("滤波半径必须 > 0（本网格 0.2 mm 已等价于无滤波）")
    if r > 1e-2:
        raise ValueError("滤波半径过大（>10 mm）")
    key = round(r, 12)
    W = bundle.filter_cache.get(key)
    if W is not None:
        return W
    default = float(bundle.cfg.filter.filter_radius_m)
    if abs(r - default) <= 1e-15:
        W = bundle.top.filter_SIMP
    else:
        from motorfem.kernels.fast.filter import compute_filter

        W = compute_filter(
            np.asarray(bundle.top.node_post),
            np.asarray(bundle.top.element_S_post),
            bundle.top.element_topo_opti_m,
            r,
            extend=True,
            p_filter=float(bundle.cfg.filter.filter_p_exp),
            parts=int(bundle.cfg.geometry.parts),
        )
    bundle.filter_cache[key] = W
    return W


def _validate_rho(bundle: Bundle, rho_master: Any) -> np.ndarray:
    rho = np.asarray(rho_master, dtype=np.float64).ravel()
    if rho.size != bundle.n_master:
        raise ValueError(f"rho_master 长度 {rho.size} != {bundle.n_master}")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho_master 含 NaN/Inf")
    return np.clip(rho, 0.0, 1.0)


def area_weighted_mean(bundle: Bundle, rho_master: np.ndarray) -> float:
    """与 ``objective.py`` 的 con2 同口径（面积加权）。"""
    a = bundle.area_master
    return float(np.sum(np.asarray(rho_master, float) * a) / np.sum(a))


def expand_rho(bundle: Bundle, rho_master: np.ndarray) -> np.ndarray:
    """主单元密度 → 全场密度（从属单元复制主值），其余单元为 0。"""
    rho = np.zeros(bundle.n_element, dtype=np.float64)
    rho[bundle.master_cells] = np.asarray(rho_master, dtype=np.float64)
    rho[bundle.slave_cells] = rho[bundle.master_cells[bundle.slave_parent]]
    return rho


def compute_start(bundle: Bundle, rho_master: Any, radius_m: float,
                  x_mode: str = "direct",
                  x0_override: Optional[Any] = None) -> Dict[str, Any]:
    """手绘密度 → 起点设计变量 x0 与实际起始密度 rho_eff。

    ``x_mode``
    ----------
    ``direct``（默认）：``x0 = clip(rho, 1e-6, 1)``。对二值手绘，反滤波没有
        可测收益（见 README）；实际起点 = ``W @ x0``。
    ``preimage``：解 ``W x ≈ rho`` 的前像（``filter_inverse_rho``）。仅作高级选项；
        不能恢复小于滤波尺度的细节。
    """
    mode = (x_mode or "direct").strip().lower()
    if mode not in ("direct", "preimage"):
        raise ValueError(f"x_mode 非法：{x_mode!r}")
    rho = _validate_rho(bundle, rho_master)
    W = get_filter(bundle, radius_m)

    if x0_override is not None:
        # 精确续跑：导入的 x 未编辑且半径一致时，直接复用，避免 W(Wx) 二次平滑。
        x0 = np.asarray(x0_override, dtype=np.float64).ravel()
        if x0.size != bundle.n_master:
            raise ValueError(f"x0_override 长度 {x0.size} != {bundle.n_master}")
        if not np.all(np.isfinite(x0)):
            raise ValueError("x0_override 含 NaN/Inf")
        mode = f"{mode}+override"
    elif mode == "direct":
        x0 = np.clip(rho, MIN_DENSITY, 1.0)
    else:
        from motorfem.fem.init_density import filter_inverse_rho

        full = expand_rho(bundle, rho)
        x0 = filter_inverse_rho(
            W, full, bundle.master_cells, min_density=MIN_DENSITY,
        )[bundle.master_cells]
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    rho_eff = np.asarray(W @ x0, dtype=np.float64).ravel()
    diff = rho_eff - rho
    stats = {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "frac_lost": float(np.mean(np.abs(diff) > 0.2)) if diff.size else 0.0,
        "mean_density": area_weighted_mean(bundle, rho_eff),
        "mean_density_input": area_weighted_mean(bundle, rho),
        "rho_min": float(rho.min()) if rho.size else 0.0,
        "rho_max": float(rho.max()) if rho.size else 0.0,
        "x0_min": float(x0.min()) if x0.size else 0.0,
        "x0_max": float(x0.max()) if x0.size else 0.0,
    }
    return {
        "rho_master": rho,
        "rho_full": expand_rho(bundle, rho),
        "x0": x0,
        "rho_eff_master": rho_eff,
        "rho_eff_full": expand_rho(bundle, rho_eff),
        "stats": stats,
        "x_mode": mode,
        "filter_radius_m": float(radius_m),
    }


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------
def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 1000):
        cand = path.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
    return path.with_name(f"{stem}_{int(time.time())}{suffix}")


def export_design(bundle: Bundle, rho_master: Any, radius_m: float, *,
                  x_mode: str = "direct", name: str = "design",
                  out_dir: Optional[Path | str] = None,
                  ladder: str = "S0:T=50,R=80",
                  x0_override: Optional[Any] = None) -> Dict[str, Any]:
    """把手绘设计写成 NPZ（原子替换）。返回路径与统计。"""
    res = compute_start(bundle, rho_master, radius_m, x_mode=x_mode,
                        x0_override=x0_override)
    slug = safe_slug(name)
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = _unique_path(root / f"{ts}_{slug}.npz")
    tmp = path.with_name(path.name + ".tmp")

    hint = {
        "motor": bundle.motor,
        "x0_key": "x0",
        "filter_radius_mm": float(radius_m) * 1e3,
        "ladder": ladder,
        "x_mode": res["x_mode"],
    }
    arrays: Dict[str, Any] = {
        "schema_version": np.int64(1),
        "app": np.asarray("tools/design_gui"),
        "created_utc": np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "motor": np.asarray(bundle.motor),
        "mesh_path": np.asarray(str(bundle.mesh_path)),
        "mesh_sha256": np.asarray(bundle.mesh_sha256),
        "csv_path": np.asarray(str(bundle.csv_path)),
        "n_master": np.int64(bundle.n_master),
        "n_slave": np.int64(bundle.n_slave),
        "n_design": np.int64(bundle.n_design),
        "n_element": np.int64(bundle.n_element),
        "master_cells": bundle.master_cells.astype(np.int64),
        "slave_cells": bundle.slave_cells.astype(np.int64),
        "design_cells": bundle.design_cells.astype(np.int64),
        "design_master": bundle.design_master.astype(np.int32),
        "rho_master": res["rho_master"].astype(np.float64),
        "rho_full": res["rho_full"].astype(np.float64),
        "x0": res["x0"].astype(np.float64),
        "x_mode": np.asarray(res["x_mode"]),
        "filter_radius_m": np.float64(res["filter_radius_m"]),
        "filter_p_exp": np.float64(bundle.cfg.filter.filter_p_exp),
        "rho_eff_master": res["rho_eff_master"].astype(np.float64),
        "rho_eff_full": res["rho_eff_full"].astype(np.float64),
        "rmse_master": np.float64(res["stats"]["rmse"]),
        "mean_density_master": np.float64(res["stats"]["mean_density"]),
        "config_hint": np.asarray(json.dumps(hint, ensure_ascii=False)),
    }
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return {
        "ok": True,
        "path": str(path),
        "name": path.name,
        "stats": res["stats"],
        "x_mode": res["x_mode"],
        "filter_radius_mm": float(radius_m) * 1e3,
        "motor": bundle.motor,
        "hint": hint,
    }


def list_designs(out_dir: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    root = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.glob("*.npz"), key=lambda q: q.stat().st_mtime, reverse=True):
        row: Dict[str, Any] = {
            "name": p.name,
            "path": str(p),
            "mtime": p.stat().st_mtime,
            "size": p.stat().st_size,
        }
        try:
            with np.load(p, allow_pickle=False) as z:
                for k, cast in (
                    ("filter_radius_m", lambda v: float(np.asarray(v).ravel()[0]) * 1e3),
                    ("mean_density_master", lambda v: float(np.asarray(v).ravel()[0])),
                    ("motor", _to_text),
                    ("x_mode", _to_text),
                ):
                    if k in z.files:
                        try:
                            row[k] = cast(z[k])
                        except Exception:  # noqa: BLE001
                            pass
        except Exception:  # noqa: BLE001
            pass
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
_X_KEY_HINTS = ("x", "x_best", "x_selected", "x_best_feasible", "x_calls", "x0")

#: 这些键是元数据/映射，不应出现在“选择要导入的设计”列表里。
_META_KEYS = {
    "schema_version", "master_cells", "slave_cells", "design_cells",
    "design_master", "n_master", "n_slave", "n_design", "n_element",
    "filter_radius_m", "filter_p_exp", "rmse_master",
    "mean_density_master",
}


def _looks_like_x(key: str) -> bool:
    k = key.strip().lower()
    if "rho" in k:
        return False
    return k.startswith("x")


def _num_kind(n: int, bundle: Bundle) -> Optional[str]:
    if n == bundle.n_element:
        return "rho_full"
    if n == bundle.n_master:
        return "master"
    if n == bundle.n_design:
        return "design"
    return None


def _pick_vector(a: np.ndarray, bundle: Bundle, row: Optional[int]) -> Tuple[np.ndarray, str]:
    """从 1D/2D 数组里取出一个长度匹配的向量。"""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.astype(np.float64, copy=False), "1d"
    if a.ndim != 2:
        raise ValueError(f"只支持 1D/2D 数组，实际 ndim={a.ndim}")
    lengths = {bundle.n_element, bundle.n_master, bundle.n_design}
    rows_ok = a.shape[0] in lengths
    cols_ok = a.shape[1] in lengths
    r = -1 if row is None else int(row)
    # 设计向量在“列”上（常见：x_calls 形状 (n_calls, n_master)）⇒ 取第 r 行
    if cols_ok and not rows_ok:
        return a[r, :].astype(np.float64, copy=False), "rows"
    # 设计向量在“行”上（转置：x_calls 形状 (n_master, n_calls)）⇒ 取第 r 列
    if rows_ok and not cols_ok:
        return a[:, r].astype(np.float64, copy=False), "cols"
    if rows_ok and cols_ok:
        return a[r, :].astype(np.float64, copy=False), "rows(ambiguous)"
    raise ValueError(
        f"数组形状 {a.shape} 与设计域长度不匹配（期望任一维为 "
        f"{bundle.n_master}/{bundle.n_design}/{bundle.n_element}）")


def _remap_by_cells(vec: np.ndarray, cells_in: Sequence[int],
                    master_cells: np.ndarray) -> np.ndarray:
    """按元素 id 把向量重排到当前 ``master_cells`` 顺序。"""
    cur = {int(e): i for i, e in enumerate(master_cells)}
    if len(cells_in) != len(cur) or set(int(e) for e in cells_in) != set(cur):
        raise ValueError("NPZ 的 master_cells 与当前网格的主设计域不一致")
    out = np.empty(len(cur), dtype=np.float64)
    for j, e in enumerate(cells_in):
        out[cur[int(e)]] = vec[j]
    return out


def _source_radius(bundle: Bundle, z: Any, fallback_radius_mm: Optional[float]
                   ) -> Tuple[float, bool, List[str]]:
    """导入 x 时使用的源滤波半径（m）。返回 (半径, 是否用了兜底, 警告)。"""
    warns: List[str] = []
    r: Optional[float] = None
    if "config_json" in z.files:
        try:
            cfg = json.loads(_to_text(z["config_json"]))
            if isinstance(cfg, dict):
                r = cfg.get("filter", {}).get("filter_radius_m")
        except Exception:  # noqa: BLE001
            pass
    if r is None and "filter_radius_m" in z.files:
        r = _to_float(z["filter_radius_m"])
    used_fallback = False
    if r is None:
        used_fallback = True
        if fallback_radius_mm is not None:
            r = float(fallback_radius_mm) * 1e-3
        else:
            r = float(bundle.cfg.filter.filter_radius_m)
        warns.append(f"NPZ 未记录滤波半径，按 {r * 1e3:.3f} mm 处理")
    r = float(r)
    if not (0 < r <= 1e-2):
        warns.append(f"源滤波半径 {r * 1e3:.3f} mm 非法，改用当前默认")
        r = float(bundle.cfg.filter.filter_radius_m)
        used_fallback = True
    return r, used_fallback, warns


def inspect_npz(bundle: Bundle, path: Path | str) -> Dict[str, Any]:
    """列出 NPZ 里可用于导入的数值键。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    out: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            try:
                a = z[k]
            except Exception as exc:  # noqa: BLE001  object dtype 等
                out.append({"key": k, "ok": False, "note": f"无法读取：{exc}"})
                continue
            if a.dtype.kind not in "fiub":
                continue
            shape = list(a.shape)
            kind = None
            for n in shape:
                kind = _num_kind(int(n), bundle)
                if kind is not None:
                    break
            out.append({
                "key": k,
                "ok": True,
                "shape": shape,
                "dtype": str(a.dtype),
                "kind": kind,
                "looks_like_x": _looks_like_x(k),
                "selectable": bool(kind is not None and k not in _META_KEYS),
                "rows": int(shape[0]) if len(shape) >= 1 else None,
            })
        for k in ("config_json", "filter_radius_m", "mesh_sha256", "motor",
                  "master_cells", "x_mode"):
            if k in z.files:
                try:
                    meta[k] = (_to_text(z[k]) if k in
                               ("config_json", "mesh_sha256", "motor", "x_mode")
                               else (_to_float(z[k]) if k == "filter_radius_m"
                                     else "present"))
                except Exception:  # noqa: BLE001
                    pass
    cfg_radius = None
    try:
        d = json.loads(meta.get("config_json") or "{}")
        cfg_radius = d.get("filter", {}).get("filter_radius_m")
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": True,
        "path": str(p),
        "keys": out,
        "meta": meta,
        "source_filter_radius_mm": (
            float(cfg_radius) * 1e3 if cfg_radius is not None
            else (float(meta["filter_radius_m"]) * 1e3
                  if meta.get("filter_radius_m") is not None else None)),
    }


def apply_import(bundle: Bundle, path: Path | str, key: str, *,
                 row: Optional[int] = None, as_: str = "auto",
                 fallback_radius_mm: Optional[float] = None) -> Dict[str, Any]:
    """把 NPZ 的一个键换算成主单元密度 ``rho_master``。"""
    p = Path(path)
    warns: List[str] = []
    with np.load(p, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"{p.name} 里没有键 {key!r}；可选：{list(z.files)}")
        try:
            a = np.asarray(z[key])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"无法读取键 {key!r}（可能含 object dtype）：{exc}")
        if a.dtype.kind not in "fiub":
            raise ValueError(f"键 {key!r} 不是数值数组（dtype={a.dtype}）")
        vec, orient = _pick_vector(a, bundle, row)
        cells_in = None
        if "master_cells" in z.files:
            try:
                cells_in = np.asarray(z["master_cells"], dtype=np.int64).ravel()
            except Exception:  # noqa: BLE001
                cells_in = None
        if "mesh_sha256" in z.files:
            try:
                if _to_text(z["mesh_sha256"]) != bundle.mesh_sha256:
                    warns.append("NPZ 的 mesh_sha256 与当前网格不同，仅按元素 id/长度换算")
            except Exception:  # noqa: BLE001
                pass

    if not np.all(np.isfinite(vec)):
        raise ValueError(f"键 {key!r} 含 NaN/Inf")
    n = int(vec.size)
    kind = _num_kind(n, bundle)
    if kind is None:
        raise ValueError(
            f"键 {key!r} 长度 {n} 不受支持；期望 "
            f"{bundle.n_master}（主单元）/{bundle.n_design}（显示顺序）/"
            f"{bundle.n_element}（全场密度）。"
            "（极坐标傅里叶系数本工具不支持）")

    mode = (as_ or "auto").strip().lower()
    if mode == "auto":
        if kind == "master":
            mode = "x" if _looks_like_x(key) else "rho"
        else:
            mode = "rho"
    if mode not in ("x", "rho"):
        raise ValueError(f"as 非法：{as_!r}")

    imported_x = None
    source_r = None
    if kind == "rho_full":
        rho = vec[bundle.master_cells]
        # 从属一致性检查（只警告）
        try:
            exp = vec[bundle.master_cells[bundle.slave_parent]]
            got = vec[bundle.slave_cells]
            bad = int(np.sum(np.abs(exp - got) > 1e-9))
            if bad:
                warns.append(f"全场密度里有 {bad} 个从属单元与主单元不一致；以主单元为准")
        except Exception:  # noqa: BLE001
            pass
        if mode == "x":
            warns.append("全场长度的数组按密度处理（不接受全场 x）")
        rho_master = rho
    elif kind == "design":
        rho = vec[:bundle.n_master]
        if cells_in is not None:
            try:
                rho = _remap_by_cells(rho, cells_in, bundle.master_cells)
            except Exception as exc:  # noqa: BLE001
                warns.append(f"design 顺序无法按 master_cells 重映射：{exc}")
        if mode == "x":
            warns.append("显示顺序数组按密度处理（不接受 x）")
        rho_master = rho
    else:  # master
        raw = vec
        if cells_in is not None:
            try:
                raw = _remap_by_cells(raw, cells_in, bundle.master_cells)
            except Exception as exc:  # noqa: BLE001
                warns.append(f"无法按 master_cells 重映射：{exc}")
        if mode == "x":
            source_r, used_fb, w2 = _source_radius(bundle, _NpzView(p), fallback_radius_mm)
            warns.extend(w2)
            W = get_filter(bundle, source_r)
            rho_master = np.asarray(W @ raw, dtype=np.float64).ravel()
            imported_x = raw.astype(np.float64)
            if used_fb:
                warns.append("导入 x 的源滤波半径是推测值；精确续跑请显式给出半径")
        else:
            rho_master = raw.astype(np.float64)
    rho_master = np.clip(np.asarray(rho_master, dtype=np.float64).ravel(), 0.0, 1.0)
    return {
        "ok": True,
        "rho_master": rho_master,
        "kind": kind,
        "as": mode,
        "orientation": orient,
        "imported_x": imported_x,
        "source_radius_m": source_r,
        "source_radius_mm": (None if source_r is None else source_r * 1e3),
        "stats": {
            "mean_density": area_weighted_mean(bundle, rho_master),
            "rho_min": float(rho_master.min()),
            "rho_max": float(rho_master.max()),
        },
        "warnings": warns,
    }


class _NpzView:
    """轻量上下文包装：``apply_import`` 里重复打开 NPZ 只读元数据。"""

    def __init__(self, path: Path):
        self.path = path
        self._z = None

    def __enter__(self):
        self._z = np.load(self.path, allow_pickle=False)
        return self._z

    def __exit__(self, *exc):
        if self._z is not None:
            self._z.close()
        return False

    @property
    def files(self):
        with self.__enter__() as z:
            return list(z.files)

    def __contains__(self, k):
        return k in self.files

    def __getitem__(self, k):
        with self.__enter__() as z:
            return np.asarray(z[k])
