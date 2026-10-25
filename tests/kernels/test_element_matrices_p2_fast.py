"""P2 电磁单元预计算量的向量化 fast 层与 reference 逐位一致。

对照 `reference.element_matrices_p2.electromagnetic_elements_p2`（逐元素 Python
循环）。fast 版是同一数学的批量实现，K0 累加顺序逐一对应 ⇒ **逐位一致**
（``np.array_equal``）。

测试：
1. 真实 order2 网格（7144 单元）：K0/R0/Bxcoef/Bycoef/Bxcoef_g/Bycoef_g/area
   全字段 ``array_equal``；
2. 合成小网格（无气隙结构，纯几何）也逐位；
3. 加速断言（宽松 ≥5×）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "scripts"))

from motorfem.kernels.reference.element_matrices_p2 import (  # noqa: E402
    electromagnetic_elements_p2,
)
from motorfem.kernels.fast.element_matrices_p2 import (  # noqa: E402
    electromagnetic_elements_p2_fast,
)

_MESH = _REPO_ROOT / "tools" / "_gap_mesh_test" / "model_barrier1_gapall1.0_slot0.1_cl8.0_order2.msh"


def _read_mesh(path, ncols=6):
    import probe_p2_scan_engine as eng
    return eng.read_mesh(path, ncols=ncols)


def test_real_mesh_bitwise():
    if not _MESH.exists():
        pytest.skip("缺 order2 网格")
    xy, tri, _ = _read_mesh(_MESH, ncols=6)
    ref = electromagnetic_elements_p2(xy, tri)
    fast = electromagnetic_elements_p2_fast(xy, tri)

    for name in ("K0", "R0", "Bxcoef", "Bycoef", "Bxcoef_g", "Bycoef_g",
                 "gauss_w", "area"):
        a = getattr(ref, name)
        b = getattr(fast, name)
        assert np.array_equal(a, b), (
            f"{name}: fast 与 reference 不逐位 "
            f"(max|Δ|={np.abs(np.asarray(a)-np.asarray(b)).max():.3e})"
        )
    # geom 的 b/c/delta 也应逐位
    for name in ("b", "c", "delta"):
        assert np.array_equal(getattr(ref.geom, name), getattr(fast.geom, name))


def test_synthetic_bitwise():
    """确定性正定向三角网格（规律错开的行，保证 Δ>0）+ 弦中点。"""
    nx, ny = 12, 12
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append((i * 0.5 + (0.25 if j % 2 else 0.0), j * 0.5))
    xy = np.array(pts, dtype=float)
    n_vert = len(xy)
    tri_vert = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            tri_vert.append((a, b, d))    # 上三角形
            tri_vert.append((a, d, c))    # 下三角形
    tri = np.array(tri_vert, dtype=int)
    # 修正定向（统一逆时针）
    for e in range(len(tri)):
        i, j, m = tri[e]
        if (xy[j][0] - xy[i][0]) * (xy[m][1] - xy[i][1]) \
           - (xy[j][1] - xy[i][1]) * (xy[m][0] - xy[i][0]) < 0:
            tri[e] = (i, m, j)
    n_tri = len(tri)
    n6 = np.zeros((n_tri, 6), dtype=int)
    mids = np.zeros((3 * n_tri, 2))
    for e, (i, j, m) in enumerate(tri):
        n6[e, 0], n6[e, 1], n6[e, 2] = i, j, m
        n6[e, 3] = n_vert + 3 * e
        n6[e, 4] = n_vert + 3 * e + 1
        n6[e, 5] = n_vert + 3 * e + 2
        mids[3 * e] = (xy[i] + xy[j]) / 2
        mids[3 * e + 1] = (xy[j] + xy[m]) / 2
        mids[3 * e + 2] = (xy[m] + xy[i]) / 2
    xy = np.vstack([xy, mids])
    ref = electromagnetic_elements_p2(xy, n6)
    fast = electromagnetic_elements_p2_fast(xy, n6)
    for name in ("K0", "R0", "Bxcoef", "Bycoef", "Bxcoef_g", "Bycoef_g", "area"):
        assert np.array_equal(getattr(ref, name), getattr(fast, name)), name


def test_speedup():
    if not _MESH.exists():
        pytest.skip("缺 order2 网格")
    xy, tri, _ = _read_mesh(_MESH, ncols=6)
    n_rep = 3
    t0 = time.perf_counter()
    for _ in range(n_rep):
        electromagnetic_elements_p2(xy, tri)
    t_ref = (time.perf_counter() - t0) / n_rep
    t0 = time.perf_counter()
    for _ in range(n_rep):
        electromagnetic_elements_p2_fast(xy, tri)
    t_fast = (time.perf_counter() - t0) / n_rep
    assert t_ref / t_fast >= 5.0, (
        f"加速不足：ref={t_ref*1e3:.0f}ms fast={t_fast*1e3:.0f}ms ({t_ref/t_fast:.1f}×)"
    )
