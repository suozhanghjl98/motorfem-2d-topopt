# -*- coding: utf-8 -*-
"""motorfem/fem/init_density_strategies.py —— 初始密度的**实验性多范式**。

背景（2026-10-16）：现有 ``initial_rho_simp`` 只随机「洞撒在哪」，洞的
**尺寸/数量/间距**固定 ⇒ 各 seed 的初始密度场统计上几乎相同
（实测 ``mean_ρ=0.960±0.001``、灰度占比 **0.000**、都是单一连通块）
⇒ 多起点优化各组**殊途同归**（30 组里 73.44/73.41/73.38 三连）。

本模块提供**结构形式不同**的初始化范式，统计差异实测见
``tools/probe_init_strategies.py`` 的汇总表：

============  ========  ========  ==========  ==========
策略          mean_ρ    灰度占比  相关长度     连通块
============  ========  ========  ==========  ==========
legacy(现有)  0.960     0.000     14.0 mm     1.0
holes_var     0.939     0.000     10.3 mm     1.0
grf           0.668     0.669      2.2 mm     3.0
binary        0.462     0.000      2.2 mm     51.3
blobs         0.286     1.000      7.4 mm     2.2
============  ========  ========  ==========  ==========

⚠ **不改 legacy 路径**：``build_field_topology`` 照常产出
``rho_SIMP_initial``（golden 逐位不变），本模块只在**构造之后**按 ``spec``
覆盖它。所以 ``init_spec=None`` 时**零影响**。

用法::

    fs = FieldSystem(cfg, mesh, csv, init_spec={"mode": "grf", "seed": 7})
    # 或只给 mode，其余参数由 seed 派生（保证可复现）
    fs = FieldSystem(cfg, mesh, csv, init_spec="blobs")
"""
from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np

#: 各策略的默认参数区间（``rng`` 抽样范围）。给 ``None`` 表示"必填"。
SCHEMAS: Dict[str, Dict[str, Any]] = {
    # 洞参数全部随机：数量 / 半径 / 间距（允许重叠）
    "holes_var": {"n_holes": (15, 220), "radius": (0.25e-3, 1.6e-3),
                  "k_margin": (-0.4, 1.5)},
    # 高斯随机场（随机相位正弦和近似）：相关长度/幅度/基底随机
    "grf": {"corr_len": (1.5e-3, 12e-3), "amp": (0.08, 0.75),
            "base": (0.35, 0.9), "n_mode": (20, 80)},
    # 完全随机二值
    "binary": {"p": (0.25, 0.85)},
    # 随机尺寸圆斑（可重叠）
    "blobs": {"n_blob": (3, 14), "r_blob": (2e-3, 9e-3),
              "base": (0.4, 0.9), "dip": (0.05, 0.4)},
    # legacy（洞参数固定）—— 为对照保留
    "legacy": {"n_holes": (100, 100), "radius": (0.6e-3, 0.6e-3),
               "k_margin": (1.5, 1.5)},
}


def _sample_params(mode: str, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for k, rng_spec in SCHEMAS[mode].items():
        lo, hi = rng_spec
        if k in ("n_holes", "n_mode", "n_blob"):
            out[k] = int(rng.integers(int(lo), int(hi) + 1))
        elif "radius" in k or "len" in k or "blob" in k:
            # 跨数量级的量按对数均匀抽样
            out[k] = float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
        else:
            out[k] = float(rng.uniform(lo, hi))
    return out


def make_initial_density(mode: str, seed: int, xy: np.ndarray, *,
                         R_rotor: float, R_shaft: float, parts: int,
                         params: Dict[str, float] = None) -> np.ndarray:
    """造一个初始密度场（设计区局部，长度 = ``len(xy)``）。

    Parameters
    ----------
    mode : str
        ``holes_var`` / ``grf`` / ``binary`` / ``blobs`` / ``legacy``。
    seed : int
        决定参数抽样与随机源（同 seed 同结果）。
    xy : (n_topo, 2)
        设计区**单元中心**坐标（m）。
    params : dict, optional
        显式参数（覆盖 ``seed`` 派生的）。用于复现某个特定初值。
    """
    if mode not in SCHEMAS:
        raise ValueError(f"未知 init mode={mode!r}（可选 {list(SCHEMAS)}）")
    p = dict(_sample_params(mode, seed))
    if params:
        p.update(params)
    n = xy.shape[0]

    if mode == "binary":
        return (np.random.default_rng(seed + 7).random(n) < p["p"]).astype(float)

    if mode == "grf":
        rng = np.random.default_rng(seed + 11)
        field = np.zeros(n)
        nm = int(p["n_mode"])
        for _ in range(nm):
            th = rng.uniform(0, np.pi)
            k = 2 * np.pi / p["corr_len"] * rng.uniform(0.5, 1.8)
            ph = rng.uniform(0, 2 * np.pi)
            field += np.cos(k * (xy[:, 0] * np.cos(th) + xy[:, 1] * np.sin(th)) + ph)
        field /= max(np.sqrt(nm), 1.0)
        return np.clip(p["base"] + p["amp"] * field, 0.0, 1.0)

    if mode == "blobs":
        rng = np.random.default_rng(seed + 13)
        nb = int(p["n_blob"])
        centers = np.column_stack([rng.uniform(0, R_rotor, nb),
                                   rng.uniform(0, R_rotor, nb)])
        d = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2)
        cover = (d < p["r_blob"]).any(axis=1)
        return np.where(cover, p["base"], p["dip"])

    # --- 洞系列（legacy / holes_var 共用几何判据）---
    rng = np.random.default_rng(seed + 17)
    radius, km, nh = p["radius"], p["k_margin"], int(p["n_holes"])
    centers = []
    for _ in range(nh):
        for _try in range(300):
            x, y = rng.uniform(0, R_rotor), rng.uniform(0, R_rotor)
            r = np.hypot(x, y); th = np.arctan2(y, x)
            if r > R_rotor - radius * (1 + km / 2):
                continue
            if r < R_shaft + radius * (1 + km / 2):
                continue
            if r * th > r * (2 * np.pi / parts / 2) - radius * (1 + km / 2):
                continue
            if any((x - c[0]) ** 2 + (y - c[1]) ** 2 < (radius * (2 + km)) ** 2
                   for c in centers):
                continue
            centers.append((x, y)); break
    if not centers:
        return np.ones(n)
    c = np.asarray(centers)
    d = np.linalg.norm(xy[:, None, :] - c[None, :, :], axis=2)
    return np.where((d < radius).any(axis=1), 0.0, 1.0)


def apply_init_spec(top, spec: Union[str, Dict[str, Any]], mesh=None) -> None:
    """**就地**把 ``top.rho_SIMP_initial`` 换成指定范式的初始密度。

    只改**设计区**（主 + 从单元）的值，其余单元保持不动。调用方
    （``FieldSystem.__init__``）在 ``build_field_topology`` **之后**调用它，
    所以 ``spec=None`` 时 legacy 路径一字不改。

    ⚠ 坐标与单元表取 ``top.node_post`` / ``top.element_S_post``
    （**动网格插层之后**的表）—— 与 ``initial_rho_simp`` 的输入一致；
    用 ``mesh.node``/``mesh.element_S`` 会错位。
    """
    if isinstance(spec, str):
        spec = {"mode": spec}
    mode = spec.get("mode", "grf")
    seed = int(spec.get("seed", 0))

    cfg = top.cfg
    eS = np.asarray(top.element_S_post)
    n2 = np.asarray(top.node_post, float)[:, 1:3]
    topo_m = [int(e) for e in top.element_topo_opti_m]
    topo_s = dict(top.element_topo_opti_s)

    tri_m = np.array([eS[e][1:4] for e in topo_m], dtype=int)
    xy_m = n2[tri_m].mean(axis=1)
    rho_m = make_initial_density(
        mode, seed, xy_m,
        R_rotor=spec.get("R_rotor", cfg.geometry.rotor_outer_radius_m),
        R_shaft=spec.get("R_shaft", cfg.geometry.shaft_radius_m),
        parts=spec.get("parts", cfg.geometry.parts),
        params=spec.get("params"))

    # ⚠ 赋值顺序照抄 legacy（``initial_rho_simp`` 先写主、再从属取主的值），
    #   避免从属单元的来源与主不同。
    arr = np.asarray(top.rho_SIMP_initial, dtype=float).copy()
    arr[topo_m] = rho_m
    for i_s, i_m in topo_s.items():
        arr[int(i_s)] = arr[int(i_m)]
    top.rho_SIMP_initial = arr
