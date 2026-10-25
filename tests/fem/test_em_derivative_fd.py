# -*- coding: utf-8 -*-
"""生产口径 EM 向量导数的 FD 核对（T_flux 与 T_ArkBE 都通过）。

2026-10-18 起 `objective.py` 的两条分支默认开启（con3 平均转矩 = `T_flux`，
con1 纹波 = `T_ArkBE`），它们的梯度原料是 `em_segment` 返回的向量与其导数。
golden 只覆盖 arkkio 口径 ⇒ 本测试补这一段。

结论（36 槽 model_mirror4 与 6s4p 一致）
----------------------------------------
- **T_flux（`dTflux_p_drho`）**：6s4p、φ∈[0.3,0.99] 的单元、ε=1e-2，
  相对误差 **8.2e-3**；
- **T_ArkBE（`dTArkBE_p_drho`）**：同口径 **1.3e-2**。
（2026-10-18 早前的"差 1e3 倍"是**口径错**：探测单元落在 SIMP 塌陷区
（φ<0.1，单元素响应 1e-7 量级）、或对 36 槽用了未传播的 `rho_SIMP_initial`。
换成中间态单元后两张表都回到 1e-2 量级。）

口径陷阱（都踩过，留在断言旁边）
-------------------------------
1. `em_segment` 会把 `A_node_last` 热启动缓存**写进传入的 dict** ⇒ 基准与扰动
   各自用新 dict。
2. 单点 FD 必须挑 **φ∈[0.3,0.99]** 的单元：SIMP 塌陷区（φ<0.1，6s4p 占 20%）
   的响应被 φ 的三次方压到 1e-7，FD 落到 Newton 停点噪声上。
3. 36 槽的 `element_topo_opti_m` 与 `element_topo_opti_s` **互不重叠**
   （3181+3181=6362 个单元，`density_from_x` 才传播），直接传
   `rho_SIMP_initial` 会让从属半边被当成真空。

运行::

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python tests/fem/test_em_derivative_fd.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MESH = _REPO / "legacy" / "pytorch" / "my_gmsh" / "model_6s4p.msh"
CSV = _REPO / "legacy" / "pytorch" / "50CS470.csv"
EPS = 1e-2
TOL = 5e-2


def _fs():
    from motorfem.config import RunConfig, apply_mesh_config
    from motorfem.fem.system import FieldSystem
    c = RunConfig.legacy()
    apply_mesh_config(c, str(MESH))
    c.solver.solver_backend = "superlu"
    c.runtime.n_jobs = 1
    c.time.sampling_mode = "uniform"
    c.time.N_point_simulation = 3
    c.time.use_validation_set = False
    return FieldSystem(c, str(MESH), str(CSV))


def _state(fs):
    from motorfem.fem.em_worker import density_from_x
    topo = list(fs.top.element_topo_opti_m)
    rho = density_from_x(np.asarray(fs.top.rho_SIMP_initial[topo], float), fs.top)
    from motorfem.kernels.fast.simp import TableSIMPProjection
    from motorfem.kernels.reference.simp import SIMPParams
    phi = np.asarray(rho, float).copy()
    phi[topo] = TableSIMPProjection(SIMPParams()).project(np.asarray(rho, float)[topo])
    return rho, phi, topo


def _em(fs, rho):
    from motorfem.fem.em_worker import em_segment
    recs = em_segment(np.asarray(rho, float), fs.pre, fs.top, fs.material,
                      A_node_last={}, n_jobs=1, solver="superlu")
    return (np.array([r[6] for r in recs]), np.array([r[10] for r in recs]),
            np.array([r[7] for r in recs]), np.array([r[11] for r in recs]))


def _compare(key: int, dkey: int, n=3, eps=EPS):
    fs = _fs()
    rho, phi, topo = _state(fs)
    c0 = _em(fs, rho)
    mid = [e for e in topo if 0.3 <= phi[e] < 0.99]
    assert mid, "没有中间态单元 —— 状态变了请重新标定口径"
    elems = sorted(mid, key=lambda e: -abs(c0[dkey][0][e]))[:n]
    num, ana = [], []
    for e in elems:
        rp = rho.copy(); rp[e] += eps
        rm = rho.copy(); rm[e] -= eps
        num.append((_em(fs, rp)[key] - _em(fs, rm)[key]) / (2 * eps))
        ana.append(c0[dkey][:, e])
    num, ana = np.asarray(num), np.asarray(ana)
    rel = float(np.max(np.abs(num - ana)) / (np.max(np.abs(ana)) + 1e-30))
    return rel, elems, num, ana


def test_flux_vector_derivative_matches_fd():
    """`dTflux_p_drho` 与 FD 一致（<5%）。"""
    rel, elems, num, ana = _compare(0, 2)
    print(f"[T_flux] rel={rel:.2e} elems={elems}")
    print(f"  FD={np.round(num[0], 5).tolist()}")
    print(f"  AN={np.round(ana[0], 5).tolist()}")
    assert rel < TOL, f"T_flux 向量导数与 FD 不符 {rel:.2e}"


def test_arkbe_vector_derivative_matches_fd():
    """`dTArkBE_p_drho` 与 FD 一致（<5%）。"""
    rel, elems, num, ana = _compare(1, 3)
    print(f"[T_ArkBE] rel={rel:.2e} elems={elems}")
    print(f"  FD={np.round(num[0], 5).tolist()}")
    print(f"  AN={np.round(ana[0], 5).tolist()}")
    assert rel < TOL, f"T_ArkBE 向量导数与 FD 不符 {rel:.2e}"


if __name__ == "__main__":
    rc = 0
    for fn in (test_flux_vector_derivative_matches_fd,
               test_arkbe_vector_derivative_matches_fd):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:                        # noqa: PERF203
            rc = 1; print(f"FAIL {fn.__name__}: {e}")
    sys.exit(rc)
