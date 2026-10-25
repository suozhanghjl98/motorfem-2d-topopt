"""tests/fem/test_init_density.py —— 阶段 4A：初始密度的 RNG 重放与滤波器前像

验证核心（``motorfem/fem/init_density.py``，legacy 驱动层初始化，非数值内核）：

  1. **重放确定性**：同一 seed + 同一表 → 逐位相同（golden L2 的前提）。
  2. **第一次调用是承重的**：删掉第一次 ``init_opti_elem_random_hole``（它被
     第二次覆盖，但仍消耗 RNG）会改变结果 —— 这正是必须照跑两次的原因
     （模块 docstring 的更正）。
  3. **滤波器前像**：``filter_inverse_rho`` 返回的 ``x`` 满足 ``W@x ≈ ρ``
     且落在 bounds 里；golden 的 ``rho_SIMP_initial`` 就是这一步之后的值。

来源 A（小合成算例）与 B（真实网格上对照 golden 的说明）并见
``test_mesh_builder.py`` 的 L2 项。
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse as sp

from motorfem.config import RunConfig
from motorfem.fem.init_density import filter_inverse_rho, initial_rho_simp
from motorfem.fem.mesh_builder import build_field_topology
from motorfem.mesh import MotorMesh

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


@pytest.fixture(scope="module")
def top():
    return build_field_topology(RunConfig.legacy(), MotorMesh(_mesh_path()))


# ---------------------------------------------------------------- RNG 重放
def test_replay_is_deterministic(top):
    """同一 seed + 同一表 → 逐位相同。"""
    cfg = RunConfig.legacy()
    a = initial_rho_simp(
        top.node_post, top.element_S_post.tolist(), top.element_topo_opti_m,
        n_element=top.n_element, seed=1002,
        rho_init_radius=cfg.topology.rho_init_radius_m,
        rho_init_distance=cfg.topology.rho_init_distance_m,
        m_Rrotor=cfg.geometry.rotor_outer_radius_m,
        m_Rirotor=cfg.geometry.shaft_radius_m,
        parts=cfg.geometry.parts,
        simp_inv=top.simp_inv.invert,
    )
    b = initial_rho_simp(
        top.node_post, top.element_S_post.tolist(), top.element_topo_opti_m,
        n_element=top.n_element, seed=1002,
        rho_init_radius=cfg.topology.rho_init_radius_m,
        rho_init_distance=cfg.topology.rho_init_distance_m,
        m_Rrotor=cfg.geometry.rotor_outer_radius_m,
        m_Rirotor=cfg.geometry.shaft_radius_m,
        parts=cfg.geometry.parts,
        simp_inv=top.simp_inv.invert,
    )
    assert np.array_equal(a, b)


def test_first_call_consumes_rng(top):
    """第一次调用（被第二次覆盖但消耗 RNG）是承重的 —— 删掉它会改变结果。"""
    cfg = RunConfig.legacy()
    kw = dict(
        n_element=top.n_element, seed=1002,
        rho_init_radius=cfg.topology.rho_init_radius_m,
        rho_init_distance=cfg.topology.rho_init_distance_m,
        m_Rrotor=cfg.geometry.rotor_outer_radius_m,
        m_Rirotor=cfg.geometry.shaft_radius_m,
        parts=cfg.geometry.parts,
        simp_inv=top.simp_inv.invert,
    )
    two_call = initial_rho_simp(
        top.node_post, top.element_S_post.tolist(), top.element_topo_opti_m, **kw)
    # 单第二次调用：RNG 起点不同 ⇒ 洞不同
    from motorfem.fem.init_density import random_hole_density
    np.random.seed(kw["seed"])
    rho = np.zeros(top.n_element)
    rho[list(top.element_topo_opti_m)] = random_hole_density(
        top.node_post, top.element_S_post.tolist(), list(top.element_topo_opti_m),
        radius=0.6e-3, k_margin=1.5, n_count=100,
        m_Rrotor=kw["m_Rrotor"], m_Rirotor=kw["m_Rirotor"], parts=kw["parts"])
    one_call = kw["simp_inv"](rho.clip(min=1e-6, max=1))
    assert not np.array_equal(two_call, one_call), (
        "删掉第一次调用结果竟然不变 —— RNG 重放已退化为可省步骤，测试失守")


# ---------------------------------------------------------------- 滤波器前像
def test_filter_inverse_reproduces_target():
    """小合成算例（来源 A）：恒等滤波器下前像 ≈ 目标。"""
    n = 3
    W = sp.identity(n, format="csr")
    rho = np.zeros(2 * n)
    topo = [0, 2, 4]
    rho[topo] = np.array([0.1, 0.5, 0.9])
    out = filter_inverse_rho(W, rho, topo, min_density=1e-6)
    # 恒等滤波器 ⇒ x ≈ 目标（L-BFGS-B 收敛到 quadratic 的精确解）
    assert np.allclose(out[topo], rho[topo], atol=1e-4)
    # 未命中的位置原样保留
    assert np.array_equal(out[[1, 3]], rho[[1, 3]])
    # bounds：x ∈ [1e-6, 1]
    assert out[topo].min() >= 1e-6 - 1e-12
    assert out[topo].max() <= 1.0 + 1e-12


def test_filter_inverse_respects_bounds(top):
    """真实滤波器：x 全部落在 [min_density, 1]，且 W@x 逼近目标。"""
    from motorfem.fem.init_density import initial_rho_simp
    cfg = RunConfig.legacy()
    raw = initial_rho_simp(
        top.node_post, top.element_S_post.tolist(), top.element_topo_opti_m,
        n_element=top.n_element, seed=1002,
        rho_init_radius=cfg.topology.rho_init_radius_m,
        rho_init_distance=cfg.topology.rho_init_distance_m,
        m_Rrotor=cfg.geometry.rotor_outer_radius_m,
        m_Rirotor=cfg.geometry.shaft_radius_m,
        parts=cfg.geometry.parts,
        simp_inv=top.simp_inv.invert,
    )
    out = filter_inverse_rho(top.filter_SIMP, raw, top.element_topo_opti_m)
    x = out[list(top.element_topo_opti_m)]
    assert x.min() >= 1e-6 - 1e-9
    assert x.max() <= 1.0 + 1e-9
    target = raw[list(top.element_topo_opti_m)]
    fitted = np.asarray(top.filter_SIMP @ x).ravel()
    # 残差是 legacy L-BFGS-B 的真实收敛水平（实测 ≈0.21·‖target‖）；
    # golden 的 rho_SIMP_initial 就是 res.x（L1/L2 已逐位对齐），所以这里
    # 只钉住「确实在逼近」与确定性，不设过紧阈值。
    assert np.linalg.norm(target - fitted) < 0.5 * np.linalg.norm(target)
    out2 = filter_inverse_rho(top.filter_SIMP, raw, top.element_topo_opti_m)
    assert np.array_equal(out, out2), "滤波器前像必须确定（同输入同输出）"
