"""tests/fem/test_preassembly.py —— 阶段 4B：单元级预组装（golden L2 对齐）

验证核心：``build_preassembly`` + ``FieldTopology`` 产出与 golden L2 **全部 697
个数组** ``np.array_equal``（K_vectorized_vec 22 元组 / R·Rd·Rq / TVDcoef / Kmech /
time_span / topo_opti_m·s / rho_SIMP_initial）。

**ULP 陷阱（本批踩过并已修）**：``area_current`` 必须用 legacy 的**顺序累加**
（``MMA10:544-545`` 的 ``for i in range(N): +=``），不能用 ``np.sum``（成对求和，
末位不同 ⇒ 电流密度 ⇒ R 全链差 ~1e-14）。golden 比对把它钉住：
``test_area_coil_uses_sequential_accumulation`` 直接对比两种算法。

运行::

    python -m pytest tests/fem/test_preassembly.py -v
"""
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig
from motorfem.fem.mesh_builder import build_field_topology
from motorfem.fem.preassembly import build_preassembly
from motorfem.mesh import MotorMesh

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]
_GOLDEN_L2 = _REPO_ROOT / "golden" / "L2_element.npz"


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


@pytest.fixture(scope="module")
def pre():
    top = build_field_topology(RunConfig.legacy(), MotorMesh(_mesh_path()))
    return build_preassembly(RunConfig.legacy(), top)


@pytest.fixture(scope="module")
def golden():
    if not _GOLDEN_L2.exists():
        pytest.skip(f"{_GOLDEN_L2} 不存在 —— 基线冻结尚未入库")
    return np.load(_GOLDEN_L2)


def test_l2_all_arrays_match_golden(pre, golden):
    """L2 全部数组 np.array_equal（K/R/Rd/Rq/TVDcoef/Kmech/time_span/ρ 部分）。"""
    from motorfem.fem.mesh_builder import build_field_topology
    from motorfem.mesh import MotorMesh
    top = build_field_topology(RunConfig.legacy(), MotorMesh(_mesh_path()))
    actual = dict(top.L2_snapshot_partial())
    actual.update(pre.L2_snapshot())
    for k in sorted(golden.files):
        assert k in actual, f"actual 缺 {k}"
        assert np.array_equal(golden[k], actual[k]), (
            f"{k} 与 golden 不一致\n"
            f"  golden {golden[k].shape} {golden[k].dtype} / "
            f"actual {np.asarray(actual[k]).shape} {np.asarray(actual[k]).dtype}"
        )
    assert set(actual) == set(golden.files), (
        f"actual 多出的键: {set(actual) - set(golden.files)}")


def test_area_coil_uses_sequential_accumulation(pre):
    """area_current 必须顺序累加 —— np.sum 末位不同（ULP 陷阱）。"""
    top = pre.top
    ap_an = top.regions.region("element_phaseAp") | top.regions.region("element_phaseAn")
    seq = 0.0
    for i in range(top.n_element):
        if i in ap_an:
            seq += float(top.em.area[i])
    np_sum = float(np.sum(top.em.area[list(ap_an)]))
    assert seq != np_sum, "顺序累加与 np.sum 恰好相同 —— 测试失守（换网格后可能不同）"
    # area_coil = area_current / (Z/p/3)（顺序累加版）
    assert pre.area_coil == seq / (pre.cfg.geometry.Z / pre.cfg.geometry.pole_pairs / 3)


def test_preassembly_structures_match_golden_shapes(pre, golden):
    """规模与 golden 形状一致（11 时间点、23 元组、Kmech 稀疏结构）。

    ⚠ 22 → **23**（2026-09-24 对称 Dirichlet）：末位新增 ``J_col_mask``。
    """
    assert pre.time_span.shape == (11,)
    assert len(pre.K_vectorized_vec) == 11
    assert len(pre.Torque_VD_coef_vec) == 11
    assert all(len(kv) == 23 for kv in pre.K_vectorized_vec)
    assert pre.Kmech.n_dof == 2 * pre.top.n_node_mech == 6584
    g = golden
    assert g["Kmech__K0_stress__data"].size == pre.top.n_element_mech * 3 * 6
    assert g["Kvec_00_00"].size == pre.K_vectorized_vec[0][0].size
