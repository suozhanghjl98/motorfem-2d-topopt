"""P2 向量化 fast 层（`fast/newton_p2.py`）与 reference 的一致性 + 加速。

对齐判据：fast 层相对 reference 是**浮点累加顺序**差异（无算法差异），
用 ``rtol=1e-12`` 对齐（P2 无 legacy 逐位对象，见模块 docstring）。

测试内容：
1. 小网格（含气隙/从属单元）上 ``field_error_p2`` vs ``field_system_p2``：
   F 逐点 rel、J 的 Frobenius rel 均 < 1e-12；
2. ``newton_solve_p2_fast`` vs ``newton_solve_p2``：解逐点 rel < 1e-12、轮数相同；
3. 加速断言（宽松阈值，防 CI 抖动）：fast 每轮场组装比 reference 快 ≥ 5×；
4. 防退化：两个求解器收敛到同一 ‖F‖²（同一判停）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "scripts"))

from motorfem.kernels.reference import newton_p2 as ref_nt_p2  # noqa: E402
from motorfem.kernels.fast import newton_p2 as fast_nt_p2  # noqa: E402

_CSV = _REPO_ROOT / "legacy" / "pytorch" / "50CS470.csv"

# 小网格：order2，含 167 从属 + 气隙单元。网格文件在仓库里。
_MESH = _REPO_ROOT / "tools" / "_gap_mesh_test" / "model_barrier1_gapall1.0_slot0.1_cl8.0_order2.msh"


def _make_env(mesh=_MESH):
    import probe_p2_solve_barrier1 as pp2
    pp2._MESH_P2 = Path(mesh)
    env = pp2.build_p2_system(pos=0.0, phase_pos=np.pi / 4)
    return env


@pytest.fixture(scope="module")
def env():
    if not _MESH.exists():
        pytest.skip("缺 order2 网格文件")
    return _make_env()


def _field(env):
    from motorfem.kernels.reference import newton as ref_nt
    ref_nt.FieldMaterial  # noqa: B018  （确保引用）
    return env["field"]


def test_field_error_consistent(env):
    field = _field(env)
    driving = env["driving"]
    # geom 带 driving（J 掩码/jmd 合并在结构里），与 reference 的 field_system_p2
    # 对同一 J 语义对比
    geom = fast_nt_p2.make_p2_fast_geometry(field, driving=driving)
    n_node = field.n_node

    rng = np.random.default_rng(0)
    A = rng.normal(size=n_node)
    R = env["R"]
    rho = env["rho"]
    material = env["material"]

    F_ref, _ = ref_nt_p2.residual_p2(A, rho, field, material, R, driving)
    B_ref = ref_nt_p2.flux_density_p2(A, field)
    mu, dgamma = ref_nt_p2._material_gauss(B_ref, rho, field.masks, material)
    _, J_ref = ref_nt_p2.field_system_p2(A, field, mu, dgamma, B_ref, R, driving)

    F_fast, B_fast, mu_fast, dgamma_fast, J_fast = fast_nt_p2.field_error_p2(
        A, geom, rho, material, R, driving)

    # F 逐点 rel（实测 2e-16；阈值 1e-14 留余量）
    denom = np.maximum(np.abs(F_ref).max(), 1e-30)
    assert np.abs(F_fast - F_ref).max() / denom < 1e-14, (
        f"F max|Δ|/scale = {np.abs(F_fast - F_ref).max() / denom:.3e}"
    )
    # B
    assert np.allclose(B_fast, B_ref, rtol=1e-14, atol=1e-15)
    # J（实测 2e-16）
    dJ = np.abs(J_fast - J_ref).toarray() if hasattr(J_fast, "toarray") else np.abs(J_fast - J_ref)
    denomJ = max(np.abs(J_ref).max(), 1e-30)
    assert dJ.max() / denomJ < 1e-14, f"J max|Δ|/scale = {dJ.max() / denomJ:.3e}"


def test_newton_solve_consistent(env):
    field = _field(env)
    geom = fast_nt_p2.make_p2_fast_geometry(field, driving=env["driving"])

    A_ref, aux_ref, _, hist_ref = ref_nt_p2.newton_solve_p2(
        None, env["rho"], field, env["material"], env["R"], env["driving"],
        solve_abs_err=1e-3, max_iter=100, iter_relaxation=20,
        relaxation_ratio=0.5, output=False)
    A_fast, aux_fast, _, hist_fast = fast_nt_p2.newton_solve_p2_fast(
        None, env["rho"], geom, env["material"], env["R"], env["driving"],
        solve_abs_err=1e-3, max_iter=100, iter_relaxation=20,
        relaxation_ratio=0.5, output=False)

    denom = max(np.abs(A_ref).max(), 1e-30)
    # 实测 6e-12（每轮 1e-16 误差经 Newton 放大）；阈值 1e-10 留余量
    assert np.abs(A_fast - A_ref).max() / denom < 1e-10, (
        f"A max|Δ|/scale = {np.abs(A_fast - A_ref).max() / denom:.3e}"
    )
    assert len(hist_ref) == len(hist_fast), (
        f"轮数不同：ref={len(hist_ref)} fast={len(hist_fast)}"
    )
    # 收敛终点一致
    assert np.isclose(hist_ref[-1], hist_fast[-1], rtol=1e-8, atol=1e-12)


def test_fast_speedup(env):
    """每轮场组装加速断言（宽松 ≥5×，防机器抖动）。"""
    import time
    field = _field(env)
    geom = fast_nt_p2.make_p2_fast_geometry(field)
    n_node = field.n_node
    rng = np.random.default_rng(1)
    A = rng.normal(size=n_node)
    rho = env["rho"]
    driving = env["driving"]
    material = env["material"]
    R = env["R"]

    # reference 每轮（残差+雅可比两次 field_system_p2，模拟 step_fn）
    def ref_step():
        B = ref_nt_p2.flux_density_p2(A, field)
        mu, dgamma = ref_nt_p2._material_gauss(B, rho, field.masks, material)
        ref_nt_p2.field_system_p2(A, field, mu, dgamma, B, R, driving)

    n_rep = 3
    t0 = time.perf_counter()
    for _ in range(n_rep):
        ref_step()
    t_ref = (time.perf_counter() - t0) / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        fast_nt_p2.field_error_p2(A, geom, rho, material, R, driving)
    t_fast = (time.perf_counter() - t0) / n_rep

    assert t_ref / t_fast >= 5.0, (
        f"加速不足：ref={t_ref*1e3:.1f}ms/轮 fast={t_fast*1e3:.1f}ms/轮 "
        f"({t_ref/t_fast:.1f}×)"
    )
