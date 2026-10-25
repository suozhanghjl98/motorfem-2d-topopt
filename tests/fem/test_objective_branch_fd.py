# -*- coding: utf-8 -*-
"""生产口径下 `objective.fun` 的**两个新增分支**的有限差分核对。

2026-10-18 起 `objective.py` 新增了两条生产分支，它们默认关闭（golden 不受
影响），所以既有的 golden 测试**覆盖不到**：

1. ``ObjectiveConfig.mean_torque_source='flux'`` —— con3 的转矩用 ``T_flux``
   均值、梯度用 ``dTflux_p_drho`` 的均值；
2. ``ObjectiveConfig.ripple_source='arkbe'`` —— con1 用 ``T_ArkBE`` 的 p 范数
   纹波、梯度用 ``dTArkBE_p_drho`` 的 p-范数链式。

本测试用**中心差分**（独立外部判据，不是自我印证）核对这两支的
``d(f)/dx``：``x`` 是 MMA 的设计变量，全链 ``x → ρ=filter_SIMP·x → 物理量``。

口径注意（`docs/关键事实速查.md` E3）
-----------------------------------
- 必须**全链** FD（对 x 扰动，走 `warper`），不能对固定 B 的中间量做 FD；
- 每次扰动后恢复 ``a_last``（热启动缓存），否则扰动之间互相污染；
- 用解析 ``df`` 的**最大分量**挑扰动单元 ⇒ 信噪比最高；再加一个随机单元
  防"只挑到好点"。
- ε=1e-3（x∈[0,1]，ρ 经滤波后平滑；再小会被 Newton 停点差淹没）。

运行：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python -m pytest tests/fem/test_objective_branch_fd.py -q
（或直接 ``python tests/fem/test_objective_branch_fd.py`` —— 非 pytest 可用。）
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


def _make(mean_torque_source: str, ripple_source: str):
    from motorfem.config import RunConfig, apply_mesh_config
    from motorfem.fem.system import FieldSystem
    c = RunConfig.legacy()                 # 逐位口径：与 golden 同一求解器路径
    apply_mesh_config(c, str(MESH))
    c.solver.solver_backend = "superlu"
    c.runtime.n_jobs = 1
    oc = c.objective
    oc.objective_role = "con3"
    oc.enable_uelastic = oc.enable_stress = False
    oc.enable_con2 = oc.enable_con3 = True
    oc.enable_con1 = True
    oc.enable_con4 = False
    oc.mean_torque_source = mean_torque_source
    oc.ripple_source = ripple_source
    oc.mean_torque_obj = 50.0
    oc.max_torque_ripple = 5.0
    c.time.sampling_mode = "uniform"
    c.time.N_point_simulation = 3          # 3 点：FD 便宜，且 p 范数仍有意义
    c.time.use_validation_set = False
    return FieldSystem(c, str(MESH), str(CSV))


def _fd_check(fs, tag: str, n_probe: int = 3, eps: float = 1e-3,
              tol: float = 5e-3) -> dict:
    """对 ``fs.warper`` 的 f 做中心差分，返回 {行名: 最大相对误差}。"""
    top = fs.top
    x0 = np.asarray(top.rho_SIMP_initial[list(top.element_topo_opti_m)], float)
    # 起点先跑一段"自由最大化"让状态离开初值（初值处 τ 几乎为 0，纹波比值无意义）
    x = x0.copy()
    a_ref = None
    for _ in range(3):
        f, df = fs.warper(x)
        fs.trace.clear()
    a_ref = {k: v.copy() for k, v in fs.a_last.items()}

    def F(xx):
        fs.a_last = {k: v.copy() for k, v in a_ref.items()}
        ff, _ = fs.warper(xx)
        fs.trace.clear()
        fs.a_last = {k: v.copy() for k, v in a_ref.items()}
        return np.asarray(ff, float)

    f0, df0 = fs.warper(x)
    fs.trace.clear()
    df0 = np.asarray(df0, float)
    rng = np.random.RandomState(3)
    ks = list(np.argsort(-np.abs(df0).max(axis=0))[:n_probe]) + list(
        rng.choice(x.size, 2, replace=False))
    errs = {}
    for row in range(f0.size):
        g_fd, g_an = [], []
        for k in ks:
            xp = x.copy(); xm = x.copy()
            step = eps * 0.5
            xp[k] = min(1.0, x[k] + step)
            xm[k] = max(0.0, x[k] - step)
            d = xp[k] - xm[k]
            if d == 0:
                continue
            g_fd.append((F(xp)[row] - F(xm)[row]) / d)
            g_an.append(df0[row, k])
        g_fd, g_an = np.asarray(g_fd), np.asarray(g_an)
        denom = np.max(np.abs(g_an)) + 1e-30
        errs[row] = float(np.max(np.abs(g_fd - g_an)) / denom)
        print(f"[{tag}] row{row}  FD={np.round(g_fd / denom, 5).tolist()}  "
              f"AN={np.round(g_an / denom, 5).tolist()}  rel={errs[row]:.2e}")
    print(f"[{tag}] f0 = {f0}")
    return errs


def test_con3_flux_gradient_matches_fd():
    fs = _make("flux", "arkkio")
    errs = _fd_check(fs, "con3=flux")
    assert errs[0] < 5e-3, f"con3(T_flux) 梯度与 FD 不符: {errs[0]:.2e}"


def test_con1_arkbe_gradient_matches_fd():
    fs = _make("arkkio", "arkbe")
    errs = _fd_check(fs, "con1=arkbe")
    # f = [obj(con3/T_ArkBE), con2, con4?...]：con1 是 row0（role=con3 ⇒ con3 是目标）
    assert errs[0] < 5e-3, f"目标(T_Arkkio) 梯度与 FD 不符: {errs[0]:.2e}"


def test_dual_source_combo_matches_fd():
    """生产组合：平均值 flux + 纹波 arkbe（两者同时开）。"""
    fs = _make("flux", "arkbe")
    errs = _fd_check(fs, "flux+arkbe")
    assert max(errs.values()) < 5e-3, f"生产组合梯度与 FD 不符: {errs}"


if __name__ == "__main__":
    rc = 0
    for fn in (test_con3_flux_gradient_matches_fd,
               test_con1_arkbe_gradient_matches_fd,
               test_dual_source_combo_matches_fd):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:                       # noqa: PERF203
            rc = 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(rc)
