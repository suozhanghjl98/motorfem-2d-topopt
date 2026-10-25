"""tests/config/test_resolved.py —— 阶段 2：ResolvedConfig 测试

验证：model_hash 与 legacy_msh 的 MANIFEST.sha256 一致；区域统计与 golden L1 的
集合规模一致；winding_factor 解析值；JSON 往返。
"""
import json
import sys
from pathlib import Path

import pytest

from motorfem.config import ResolvedConfig, RunConfig, integer_slot_winding_factor

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tests"))
from config.test_config import LEGACY_GLOBALS  # noqa: E402
from mesh.test_mesh import _mesh_path  # noqa: E402


def test_winding_factor_analytic():
    """Z=36/p=2/m=3 的整数槽分布因数：q=3、α=20°、kd=sin30°/(3·sin10°)=0.95979…"""
    wf = integer_slot_winding_factor(36, 2, 3)
    expected = 0.5 / (3 * __import__("math").sin(10 * 3.141592653589793 / 180))
    assert wf == pytest.approx(expected, abs=1e-12)
    assert 0.9 < wf < 1.0


def test_model_hash_matches_manifest():
    """model_hash 必须等于 legacy_msh/MANIFEST.sha256 里的值（网格指纹锚定）。"""
    manifest = _REPO_ROOT / "legacy_msh" / "MANIFEST.sha256"
    if not manifest.exists():
        pytest.skip("legacy_msh/MANIFEST.sha256 不存在")
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    expected = {l.split("  ")[1]: l.split("  ")[0] for l in lines}
    rc = _resolved()
    assert rc.model_hash == expected["model_mirror4.msh"]


def test_region_element_counts_match_golden():
    """区域统计与 golden L1 的集合规模一致（抽查关键区域）。"""
    g = _golden()
    rc = _resolved()
    counts = rc.region_element_counts
    assert counts["element_rotor_shaft"] == 206
    assert counts["element_rotor_core_m"] == 3181
    assert counts["element_rotor_core_s"] == 3181
    assert counts["element_current"] == 144
    assert counts["element_stator_core"] == len(g["element_stator_core__sorted"])
    assert counts["element_stator_coil"] == 18  # list of sets → 槽数


def test_json_roundtrip():
    rc = _resolved()
    rc2 = ResolvedConfig.from_json(rc.to_json())
    assert rc2.model_hash == rc.model_hash
    assert rc2.region_element_counts == rc.region_element_counts
    assert rc2.winding_factor == rc.winding_factor
    assert rc2.run.to_dict() == rc.run.to_dict()


def test_winding_factor_override():
    """显式传 winding_factor 时不用解析值。"""
    from motorfem.mesh import MotorMesh
    regions = MotorMesh(_mesh_path()).regions
    rc2 = ResolvedConfig.from_legacy(LEGACY_GLOBALS, _mesh_path(), regions,
                                     winding_factor=0.85)
    assert rc2.winding_factor == 0.85


def _resolved() -> ResolvedConfig:
    """构造一个 ResolvedConfig：legacy 快照 + 真实网格 + RegionMap 区域。"""
    from motorfem.mesh import MotorMesh
    mesh = MotorMesh(_mesh_path())
    return ResolvedConfig.from_legacy(LEGACY_GLOBALS, _mesh_path(), mesh.regions)


def _golden():
    path = _REPO_ROOT / "golden" / "L1_regions.npz"
    if not path.exists():
        pytest.skip("golden/L1_regions.npz 不存在")
    import numpy as np
    return np.load(path)
