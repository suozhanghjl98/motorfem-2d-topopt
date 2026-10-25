# -*- coding: utf-8 -*-
"""两级网格：**细 FEM / 粗设计变量**（2026-10-20）。

背景
----
拓扑优化变量是**单元**上的分片常数（``rho[topo_m] = W @ x``），不是节点量。
因此“FEM 加密、设计分辨率不变”不需要节点插值，只需要一个**父映射**：

    design_parent[j] = i   # 第 j 个细设计单元属于第 i 个粗设计单元
    rho_fine[j]      = (W @ x)[design_parent[j]]      # 前向
    df_dx[i]         = Σ_{j: parent[j]=i} (Wᵀ df_fine)[j]   # 反向（严格求和）

细网格上仍按**细网格的滤波** ``W`` 作用于分片常数的粗密度场——等价于把同一个
滤波核离散在细网格上，比在粗网格上建滤波更精确，且不影响设计自由度。

⚠ 与 ``filter_inverse_rho`` 的关系：初始密度（legacy 随机孔洞）在细网格上生成，
长度为 ``n_fine_master``；本模块提供 :func:`compress_initial` 把它压成
``n_design`` 的设计变量（父单元取子单元均值），只在 ``--init legacy`` 时用到。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def master_centroids(top) -> np.ndarray:
    """细/粗拓扑的主设计单元质心（米），顺序 = ``list(top.element_topo_opti_m)``。"""
    master = np.asarray(list(top.element_topo_opti_m), int)
    xy = np.asarray(top.layer.xy, float)
    tri = np.asarray(top.layer.tri)[master]
    return xy[tri].mean(axis=1)


def _inside(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """点是否在三角形内（含边界，向量化 barycentric）。返回 (n_tri, n_pt) bool。

    用重心坐标的符号判定；对退化三角形（面积为 0）返回 False。
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0 = b - a
    v1 = c - a
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(den) > 1e-24
    den = np.where(ok, den, 1.0)
    out = np.zeros((tri.shape[0], points.shape[0]), bool)
    for k, p in enumerate(points):
        v2 = p - a
        u = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / den
        v = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / den
        out[:, k] = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & ok
    return out


def build_design_parent(fine_top, coarse_top) -> np.ndarray:
    """细设计单元 → 粗设计单元索引（长度 = 细主单元数）。

    口径：用**粗单元三角形包含细单元质心**判定；落在所有三角形之外（边界舍入）
    的少数单元回退到**最近质心**。两套网格必须覆盖同一物理设计域。
    """
    from scipy.spatial import cKDTree

    coarse_master = np.asarray(list(coarse_top.element_topo_opti_m), int)
    xy_c = np.asarray(coarse_top.layer.xy, float)
    tri_c = xy_c[np.asarray(coarse_top.layer.tri)[coarse_master]]
    cen_c = tri_c.mean(axis=1)

    cen_f = master_centroids(fine_top)
    n_c = coarse_master.size
    parent = np.full(cen_f.shape[0], -1, int)

    # 粗单元包围盒预筛，避免 5041×1207 的全量重心测试太慢
    lo, hi = tri_c.min(axis=1), tri_c.max(axis=1)
    for i, p in enumerate(cen_f):
        m = np.all((p >= lo - 1e-12) & (p <= hi + 1e-12), axis=1)
        idx = np.nonzero(m)[0]
        if idx.size:
            hit = _inside(p[None, :], tri_c[idx])
            j = np.nonzero(hit.any(axis=1))[0]
            if j.size:
                parent[i] = idx[j[0]]
    miss = parent < 0
    if miss.any():
        _, nn = cKDTree(cen_c).query(cen_f[miss], k=1)
        parent[miss] = nn
    # 粗单元里一个子单元都没有（非嵌套网格的边界效应）⇒ 把最近的细单元划给它。
    # 否则这些设计变量的梯度恒为 0，MMA 会退化（实测 d030 有 8/1207 个）。
    empty = np.nonzero(np.bincount(parent, minlength=n_c) == 0)[0]
    if empty.size:
        _, jj = cKDTree(cen_f).query(cen_c[empty], k=1)
        parent[jj] = empty
    assert parent.min() >= 0 and parent.max() < n_c
    return parent


def attach_design_parent(fine_top, coarse_top, *, verbose: bool = True) -> np.ndarray:
    """把父映射挂到 ``fine_top`` 上（``design_parent`` / ``n_design``）。"""
    parent = build_design_parent(fine_top, coarse_top)
    fine_top.design_parent = parent
    fine_top.n_design = int(parent.max()) + 1
    if verbose:
        n_f, n_c = parent.size, fine_top.n_design
        counts = np.bincount(parent, minlength=n_c)
        print(f"[two_level] 细设计单元 {n_f} → 粗设计变量 {n_c}；"
              f"每组子单元数 min/median/max = "
              f"{counts.min()}/{int(np.median(counts))}/{counts.max()}", flush=True)
    return parent


def compress_initial(top, rho_fine_full: np.ndarray) -> np.ndarray:
    """把细网格上的初始密度（全长）压成 ``n_design`` 的设计变量（父单元取均值）。"""
    master = np.asarray(list(top.element_topo_opti_m), int)
    parent = getattr(top, "design_parent", None)
    vals = np.asarray(rho_fine_full, float)[master]
    if parent is None:
        return vals
    cnt = np.bincount(parent, minlength=int(parent.max()) + 1).astype(float)
    out = np.bincount(parent, weights=vals, minlength=cnt.size) / np.maximum(cnt, 1.0)
    return out
