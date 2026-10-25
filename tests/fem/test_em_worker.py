"""tests/fem/test_em_worker.py —— 阶段 4C：单步电磁（golden L3 对齐）

验证核心：注入 golden L4 的 ``call_{i}_x``，跑 ``em_segment``（3 次调用 × 11
时间点，带 ``A_node_last`` 暖启动），与 golden L3 全部 363 个数组
``np.array_equal``。这样把「EM 段正确性」与「MMA 轨迹」（阶段 4E）解耦。

**ULP 陷阱（本批踩过并已修）**：``dγ/dρ`` 里的 ``SIMP_dfunc`` 必须用 legacy 原式
（``MMA10:1142-1155``）。``reference.simp.derivative`` 是等价写法但末位不同
（实测 max|Δ| ≈ 6.7e-16），会让三条 ``dT*/dρ`` 全链差 1 ULP
（``test_legacy_simp_dfunc_differs_from_reference_in_ulp`` 把它钉住）。

运行::

    python -m pytest tests/fem/test_em_worker.py -v
"""
import os
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig
from motorfem.fem.em_worker import (
    density_from_x, em_segment, legacy_simp_dfunc, make_field_material,
)
from motorfem.fem.mesh_builder import build_field_topology
from motorfem.fem.preassembly import build_preassembly
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection
from motorfem.mesh import MotorMesh

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]
_GOLDEN_DIR = _REPO_ROOT / os.environ.get("MOTORFEM_BASELINE", "golden")
_GOLDEN_L3 = _GOLDEN_DIR / "L3_em_step.npz"
_GOLDEN_DIR = _REPO_ROOT / os.environ.get("MOTORFEM_BASELINE", "golden")
_GOLDEN_L4 = _GOLDEN_DIR / "L4_mech_mma.npz"


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


@pytest.fixture(scope="module")
def env():
    cfg = RunConfig.legacy()
    top = build_field_topology(cfg, MotorMesh(_mesh_path()))
    pre = build_preassembly(cfg, top)
    csv = _REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv
    material = make_field_material(str(csv))
    return cfg, top, pre, material


# ---------------------------------------------------------------- ULP 陷阱
def test_legacy_simp_dfunc_differs_from_reference_in_ulp():
    """reference.derivative 与 legacy 原式末位不同 —— golden 对齐必须用原式。"""
    simp = SIMPProjection(SIMPParams())
    rho = np.linspace(0, 1, 200)
    ref = simp.derivative(rho)
    leg = legacy_simp_dfunc(rho, simp)
    d = np.abs(ref - leg).max()
    assert d > 0, "两者恰好逐位相同 —— 测试失守（换材料参数后可能翻转）"
    assert d < 1e-14, f"差异 {d} 超出 ULP 量级，不是等价写法了"


# ---------------------------------------------------------------- golden 对齐
def test_l3_all_arrays_match_golden(env):
    """3 次调用 × 11 时间点全部 L3 数组 np.array_equal（注入 golden x）。"""
    if not (_GOLDEN_L3.exists() and _GOLDEN_L4.exists()):
        pytest.skip("golden 缺失 —— 基线冻结尚未入库")
    cfg, top, pre, material = env
    g3 = np.load(_GOLDEN_L3)
    g4 = np.load(_GOLDEN_L4)
    a_last: dict = {}
    for c in range(3):
        x = g4[f"call_{c:02d}_x"]
        rho = density_from_x(x, top)
        records = em_segment(rho, pre, top, material, A_node_last=a_last)
        for i_t, res in enumerate(records):
            for i, n in enumerate(("Torque_Arkkio", "dTArkkio_p_drho", "A_node",
                                   "B_elem", "flux_linkage_energy",
                                   "torque_flux_energy", "dTflux_p_drho",
                                   "Torque_VD", "dTVD_p_drho")):
                key = f"em_{c:02d}_{i_t:02d}_{n}"
                assert np.array_equal(g3[key], res[i + 1]), (
                    f"{key} 与 golden 不一致\n"
                    f"  golden {g3[key].shape} / actual {np.asarray(res[i+1]).shape}")


def test_density_from_x_matches_warper_semantics(env):
    """ρ = W@x 落到 topo_m，从属单元同步到主动值（MMA10:4008-4015）。"""
    _cfg, top, _pre, _material = env
    x = np.full(len(top.element_topo_opti_m), 0.5)
    rho = density_from_x(x, top)
    # 非拓扑区为 0
    topo = set(top.element_topo_opti_m) | set(top.element_topo_opti_s.keys())
    non_topo = [i for i in range(top.n_element) if i not in topo]
    assert np.all(rho[non_topo] == 0.0)
    # 从属 = 主动（对称合并 dict 的语义）
    for i_s, i_m in top.element_topo_opti_s.items():
        assert rho[i_s] == rho[i_m]
