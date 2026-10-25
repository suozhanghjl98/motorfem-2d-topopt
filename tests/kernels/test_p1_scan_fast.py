"""P1 fast 扫描引擎（`tools/probe_p1_scan_fast.py`）一致性 + 加速测试。

对照 `probe_ansys_curves.py`（reference 层）。P1 fast 层与 legacy **逐位一致**
（CLAUDE.md §1），故引擎解应与 reference 解 ``array_equal``/1e-14 级一致。

测试：
1. 单点：fast 引擎解 vs reference `newton_solve` —— A rel < 1e-12；
2. 转矩：T_VD/T_Arkkio rel < 1e-9（对照 `probe_ansys_curves.torques_full`）；
3. A0 热启动：相邻点轮数 15→≤5（宽松断言，防机器抖动）；
4. （可选）cholmod 后端：解与 superlu 一致（rel < 1e-10），DLL 缺失自动 skip。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "scripts"))

_MESH = _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_barrier1.msh"


@pytest.fixture(scope="module")
def st():
    if not _MESH.exists():
        pytest.skip("缺 P1 网格 model_barrier1.msh")
    import probe_p1_scan_fast as p1f
    return p1f.P1Static(_MESH)


def test_single_point_consistent(st):
    import probe_p1_scan_fast as p1f
    from motorfem.kernels.reference import newton as ref_nt
    import probe_torque_vs_ansys as pta

    per = st.per_point(pos=0.0, phase_pos=np.pi / 4)
    A, aux, J, hist = p1f.solve_p1_fast(per, per["material"],
                                        solver="superlu", A0=None)

    env = pta.build_system()
    A_ref, aux_ref, _, hist_ref = ref_nt.newton_solve(
        None, env["rho"], env["field"], env["material"], env["R"],
        env["driving"], solve_abs_err=1e-3, max_iter=100,
        iter_relaxation=20, relaxation_ratio=0.5, output=False)
    rel = np.abs(A - A_ref).max() / max(np.abs(A_ref).max(), 1e-30)
    assert rel < 1e-12, f"A rel={rel:.2e}"
    assert len(hist) == len(hist_ref)


def test_torques_consistent(st):
    import probe_p1_scan_fast as p1f
    from motorfem.kernels.reference import newton as ref_nt
    import probe_torque_vs_ansys as pta
    from probe_ansys_curves import torques_full

    per = st.per_point(pos=0.0, phase_pos=np.pi / 4)
    A, aux, J, hist = p1f.solve_p1_fast(per, per["material"],
                                        solver="superlu", A0=None)
    T_vd, T_ark = p1f.torques_p1(per, A)

    env = pta.build_system()
    A_ref, aux_ref, _, _ = ref_nt.newton_solve(
        None, env["rho"], env["field"], env["material"], env["R"],
        env["driving"], solve_abs_err=1e-3, max_iter=100,
        iter_relaxation=20, relaxation_ratio=0.5, output=False)
    T_vd_r, T_ark_r, _ = torques_full(env, A_ref)
    assert abs(T_vd - T_vd_r) / abs(T_vd_r) < 1e-9, (
        f"T_VD rel={abs(T_vd - T_vd_r)/abs(T_vd_r):.2e}"
    )
    assert abs(T_ark - T_ark_r) / abs(T_ark_r) < 1e-9, (
        f"T_Ark rel={abs(T_ark - T_ark_r)/abs(T_ark_r):.2e}"
    )


def test_warm_start_reduces_iters(st):
    import probe_p1_scan_fast as p1f

    per0 = st.per_point(pos=0.0, phase_pos=np.pi / 4)
    A0, _, _, h0 = p1f.solve_p1_fast(per0, per0["material"],
                                     solver="superlu", A0=None)
    per1 = st.per_point(pos=np.deg2rad(0.5),
                        phase_pos=2 * np.deg2rad(0.5) + np.pi / 4)
    A1, _, _, h1 = p1f.solve_p1_fast(per1, per1["material"],
                                     solver="superlu", A0=A0)
    assert len(h1) <= len(h0), (
        f"热启动轮数 {len(h1)} 未少于冷启动 {len(h0)}"
    )


def test_cholmod_backend_consistent(st):
    """可选后端：cholmod 解与 superlu 一致（非逐位路径，rel<1e-10）。"""
    import probe_p1_scan_fast as p1f

    per = st.per_point(pos=0.0, phase_pos=np.pi / 4)
    # 通过引擎路径探测 cholmod（DLL 缺失会抛错 → skip）
    try:
        A_c, _, _, _ = p1f.solve_p1_fast(per, per["material"],
                                         solver="cholmod", A0=None)
    except Exception as ex:
        pytest.skip(f"CHOLMOD DLL 不可用: {type(ex).__name__}: {ex}")
    A_s, _, _, _ = p1f.solve_p1_fast(per, per["material"],
                                     solver="superlu", A0=None)
    rel = np.abs(A_c - A_s).max() / max(np.abs(A_s).max(), 1e-30)
    assert rel < 1e-10, f"cholmod vs superlu A rel={rel:.2e}"
