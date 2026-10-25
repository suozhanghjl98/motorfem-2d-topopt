"""tests/autograd/test_oracle.py —— 阶段 6 核心验收：转矩梯度 oracle vs 手写

设计文档 §阶段 6 验收：**每个转矩项，autograd 梯度 vs 手写解析梯度，相对误差
< 1e-9**（在 |梯度| ≥ min_abs 的条目的意义下 —— 低于该量级是 ULP/舍入底噪，
相对值会虚高，见 ``compare_with_hand_written`` 的 docstring）。

验证三件事：
1. 真实工作点（冻结网格 + 生产配置）的 T_Arkkio / T_VD 梯度：direct（复用已解
   状态）与 full（EMFieldGrad Function 端到端）都 rel < 1e-9。
2. full Function 路径与 direct 路径**逐位一致**（同一套 backward 数学）。
3. 第二个时间点（转子转角 ≠ 0，旋转边界激活）同样通过 —— 旋转边界是 ρ 无关的
   位置矩阵，不影响梯度（偏差登记里写明）。

⚠ 两个已知口径（模块 docstring 详述）：
- F_lam 的**值**在滑移层环节点与受信内核不一致（旋转边界 remap 是 ρ 无关的）；
  梯度只经 is_simp 单元（转子铁芯），不受影响 —— 已由本测试的 rel 证明。
- 手写路径的 Arkkio 伴随对 |梯度|≈1e-6 的条目 rel 会到 1e-8（绝对差 ~1e-14，
  纯舍入）；min_abs 阈值把这类噪声排除。
"""
import glob
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motorfem.autograd import (
    adjoint_gradient_from_state,
    adjoint_gradients_from_state_batched,
    autograd_torque_gradient,
    compare_with_hand_written,
    oracle_from_parts,
)
from motorfem.config import RunConfig
from motorfem.fem.em_worker import legacy_dgamma_p_drho
from motorfem.fem.system import FieldSystem
from motorfem.kernels.fast import torque as fast_tq

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]
#: 相对噪声下限（见模块 docstring）
_MIN_ABS = 1e-3


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


@pytest.fixture(scope="module")
def fs():
    csvs = list(_REPO_ROOT.rglob("50CS470.csv"))
    if not csvs:
        pytest.skip("找不到 50CS470.csv")
    # 模块级建一次（preassembly ~2 s），三个测试共享
    # ★ 2026-10-13：oracle 需要**精确 PCHIP**（与查表结构互斥）⇒ 必须 legacy 配置
    return FieldSystem(RunConfig.legacy(), str(_mesh_path()), str(csvs[0]))


def _working_point(fs, i_t: int, rho):
    """解一个时间点，返回 (oracle, A, B, J, dgamma)。"""
    oracle = oracle_from_parts(fs.top, fs.pre, fs.material, i_t)
    A, B, J = oracle.solve(np.asarray(rho, dtype=float))
    dgamma = legacy_dgamma_p_drho(B, np.asarray(rho, dtype=float), fs.material)
    return oracle, A, B, J, dgamma


def _hand_gradient(fs, A, kv, J, dgamma, M):
    """手写 dT/dρ（生产路径：位置矩阵伴随 × legacy dγ/dρ）。"""
    return fast_tq.adjoint_torque_derivative(
        A, kv, J, M, list(fs.top.element_topo_opti),
        n_node=fs.top.n_node, n_element=fs.top.n_element,
    ) * dgamma


def _assert_matches(auto, hand, label):
    rel, ab, nz = compare_with_hand_written(auto, hand, min_abs=_MIN_ABS, label=label)
    assert rel < 1e-9, f"{label}: autograd vs 手写 rel={rel:.3e} ≥ 1e-9"
    return rel, ab, nz


def test_arkkio_gradient_matches_hand_written(fs):
    """T_Arkkio 梯度：oracle（direct + full）vs 手写，rel < 1e-9。"""
    i_t, rho = 0, np.full(fs.top.n_element, 0.5)
    oracle, A, B, J, dgamma = _working_point(fs, i_t, rho)
    M = fs.top.torque_arkkio_coef
    hand = _hand_gradient(fs, A, fs.pre.K_vectorized_vec[i_t], J, dgamma, M)
    gA = 2.0 * (M @ A)

    direct = adjoint_gradient_from_state(rho, A, J, gA, oracle)
    full = autograd_torque_gradient(rho, oracle, M)
    r_d, ab, nz = _assert_matches(direct, hand, "T_Arkkio(direct)")
    r_f, _, _ = _assert_matches(full, hand, "T_Arkkio(full)")
    assert np.array_equal(direct, full), "Function 路径与 direct 路径应逐位一致"
    print(f"  T_Arkkio: rel(direct)={r_d:.3e} rel(full)={r_f:.3e} max_abs={ab:.3e} nz={nz}")


def test_vd_gradient_matches_hand_written(fs):
    """T_VD 梯度：oracle vs 手写，rel < 1e-9。"""
    i_t, rho = 0, np.full(fs.top.n_element, 0.5)
    oracle, A, B, J, dgamma = _working_point(fs, i_t, rho)
    M = fs.pre.Torque_VD_coef_vec[i_t]
    hand = _hand_gradient(fs, A, fs.pre.K_vectorized_vec[i_t], J, dgamma, M)
    direct = adjoint_gradient_from_state(rho, A, J, 2.0 * (M @ A), oracle)
    r, ab, nz = _assert_matches(direct, hand, "T_VD")
    print(f"  T_VD: rel={r:.3e} max_abs={ab:.3e} nz={nz}")


def test_gradient_at_second_time_step(fs):
    """转子转角 ≠ 0（旋转边界激活）的时间点同样通过。"""
    i_t, rho = 1, np.full(fs.top.n_element, 0.5)
    oracle, A, B, J, dgamma = _working_point(fs, i_t, rho)
    M = fs.top.torque_arkkio_coef
    hand = _hand_gradient(fs, A, fs.pre.K_vectorized_vec[i_t], J, dgamma, M)
    direct = adjoint_gradient_from_state(rho, A, J, 2.0 * (M @ A), oracle)
    r, ab, nz = _assert_matches(direct, hand, "T_Arkkio(i_t=1)")
    print(f"  T_Arkkio@t1: rel={r:.3e} max_abs={ab:.3e} nz={nz}")


def test_batched_gradient_matches_single(fs):
    """批量 oracle（b 个右端一次解）== 逐列单次（固定开销摊薄的正确性前提）。"""
    i_t, rho = 0, np.full(fs.top.n_element, 0.5)
    oracle, A, B, J, dgamma = _working_point(fs, i_t, rho)
    M_ark = fs.top.torque_arkkio_coef
    M_vd = fs.pre.Torque_VD_coef_vec[i_t]
    gas = [(2.0 * (M_ark @ A)), (2.0 * (M_vd @ A)),
           (M_ark @ A) + (M_vd @ A), 0.5 * (M_ark @ A)]
    G = np.stack(gas, axis=1)
    D = adjoint_gradients_from_state_batched(rho, A, J, G, oracle)
    assert D.shape == (len(gas), fs.top.n_element)
    for k, gA in enumerate(gas):
        d = adjoint_gradient_from_state(rho, A, J, gA, oracle)
        assert np.abs(D[k] - d).max() < 1e-9, f"批量第 {k} 列与单次不一致"


def test_gradient_with_production_density(fs):
    """生产初始密度（golden 工作点）下同样通过。"""
    i_t, rho = 0, fs.top.rho_SIMP_initial
    oracle, A, B, J, dgamma = _working_point(fs, i_t, rho)
    M = fs.top.torque_arkkio_coef
    hand = _hand_gradient(fs, A, fs.pre.K_vectorized_vec[i_t], J, dgamma, M)
    direct = adjoint_gradient_from_state(rho, A, J, 2.0 * (M @ A), oracle)
    r, ab, nz = _assert_matches(direct, hand, "T_Arkkio(prod-density)")
    print(f"  T_Arkkio@prod: rel={r:.3e} max_abs={ab:.3e} nz={nz}")
