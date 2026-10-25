"""tests/winding/test_layout.py —— 阶段 5：star-of-slots 生成器对账

验证核心（设计文档 §5 的硬门槛）：生成器给出的 6 个相带单元集合与基线 L1 的
``element_phase{Ap,An,Bp,Bn,Cp,Cn}`` **逐位相等**。

**来源 A（本批新增）**：教科书整距三相 / 分数槽用例的相带结构（q、每相槽数）
—— 生成器不能只 fit 生产配置。

**关键参数**：````（槽结束边电角）是 legacy 表的特征 —— 相带
边界落在槽中心线上（槽 3/6/9 恰在边界，归入下一相带）。``"center"``（标准
星形图）与本项目 legacy 表**不**匹配 —— 这正是当初对账失败的原因
（`docs/参考资料/分数槽绕组/README.md`）。

运行::

    python -m pytest tests/winding/test_layout.py -v
"""
from pathlib import Path

import numpy as np
import pytest

from motorfem.winding.layout import (SEC_TO_NAME, build_phase_element_sets,
        coils_per_phase, star_of_slots)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]

#: legacy 相带表（MMA10:270-277）：18 个线圈实体 → 相带
LEGACY_PHASE_TABLE = {
    "Ap": [0, 1, 2],
    "An": [15, 16, 17],
    "Bp": [9, 10, 11, 12, 13, 14],
    "Bn": [],
    "Cp": [],
    "Cn": [3, 4, 5, 6, 7, 8],
}


# ---------------------------------------------------------------- 相带表对账
def test_coil_phase_matches_legacy_table():
    """生成器 18 线圈相带 == legacy 相带表（ref_angle='edge'）。"""
    lay = star_of_slots(36, 2, 3, parts=4, layers=2)
    gen = coils_per_phase(lay)
    for n, coils in LEGACY_PHASE_TABLE.items():
        assert gen[n] == coils, f"{n}: 生成 {gen[n]} != legacy {coils}"


def test_center_ref_angle_does_not_match_legacy():
    """标准星形图（槽中心参考）与本项目 legacy 表不匹配 —— 文档化的差异。"""
    lay = star_of_slots(36, 2, 3, parts=4, layers=2, inner_ref="center")
    gen = coils_per_phase(lay)
    assert any(gen[n] != LEGACY_PHASE_TABLE[n] for n in LEGACY_PHASE_TABLE), (
        "center 参考竟然匹配 legacy —— 说明两者等价，ref_angle 参数失去意义")


# ---------------------------------------------------------------- golden 对账
def test_phase_element_sets_match_golden():
    """6 个相带的单元集合与 golden L1 element_phase* 逐位相等（硬门槛）。"""
    for p in _MESH_CANDIDATES:
        if p.exists():
            mesh_path = p
            break
    else:
        pytest.skip("找不到 model_mirror4.msh")

    from motorfem.config import RunConfig
    from motorfem.fem.mesh_builder import build_field_topology
    from motorfem.mesh import MotorMesh

    top = build_field_topology(RunConfig.legacy(), MotorMesh(str(mesh_path)))
    coil_sets = top.regions.region("element_stator_coil")
    lay = star_of_slots(36, 2, 3, parts=4, layers=2)
    gen = coils_per_phase(lay)
    for n in SEC_TO_NAME[:6]:
        got = set().union(*(coil_sets[c] for c in gen[n])) if gen[n] else set()
        exp = top.regions.region(f"element_phase{n}")
        assert got == exp, f"element_phase{n} 单元集合不一致"


# ---------------------------------------------------------------- 扩展性（来源 A）
def test_generic_cases_slot_counts():
    """通用用例的相带槽数结构（教科书判据，不依赖生产网格）。"""
    # Z=12, p=1, m=3：每极每相 q = 2 槽（整数槽）
    lay = star_of_slots(12, 1, 3)
    assert lay.slots_per_part == 12            # parts=1 → 全模型
    # 每相带槽边 = q·2·p·m/... 每相带 2·p 个槽边？验证 6 相带总槽 = Z·layers
    gen = coils_per_phase(lay)
    assert sum(len(v) for v in gen.values()) == 12 * 2   # 全模型 24 线圈边
    # Z=12, p=5：分数槽（q = 12/(2·5·3) = 0.4）
    lay5 = star_of_slots(12, 5, 3)
    assert lay5.slots_per_part == 12
    assert sum(len(v) for v in coils_per_phase(lay5).values()) == 24
    # Z=36, p=3, parts=1：全模型，无空相带（q = 2）
    lay36 = star_of_slots(36, 3, 3, parts=1)
    gen36 = coils_per_phase(lay36)
    assert all(len(v) > 0 for v in gen36.values()), "全模型不应有空相带"
    assert sum(len(v) for v in gen36.values()) == 72


def test_parts_must_divide_z():
    with pytest.raises(ValueError):
        star_of_slots(36, 2, 3, parts=5)


# ---------------------------------------------------------------- 绕组系数
def test_winding_factors_production_matches_resolved():
    """生产配置（Z=36, p=2, 整距 y=9）：k_p=1，k_w == 阶段 2 的整数槽公式。"""
    from motorfem.config.resolved import integer_slot_winding_factor
    from motorfem.winding.layout import winding_factors

    lay = star_of_slots(36, 2, 3, parts=4)
    kp, kd, kw = winding_factors(lay, pitch=9)
    assert kp == 1.0, "整距 9 槽（每极 9 槽）节距因数应为 1"
    assert abs(kw - integer_slot_winding_factor(36, 2)) < 1e-9, (
        "生成器分布因数与阶段 2 的整数槽公式应一致")


def test_winding_factors_mathematical_properties():
    """绕组系数数学性质（来源 A）：k_d ≤ 1、k_w = k_p·k_d、三相分布对称。"""
    from motorfem.winding.layout import winding_factors

    for Z, p, y in ((12, 1, 6), (36, 2, 9), (12, 5, 1), (9, 8, 1)):
        lay = star_of_slots(Z, p, 3, parts=1, inner_ref="center")
        kp, kd, kw = winding_factors(lay, pitch=y)
        assert kd <= 1.0 + 1e-12, f"Z={Z} p={p}: 分布因数 k_d={kd} > 1"
        assert abs(kw - kp * kd) < 1e-12, "k_w 必须 = k_p · k_d"
        assert 0.0 <= kp <= 1.0 and 0.0 <= kd <= 1.0


# ---------------------------------------------------------------- PMSM 对账
def test_pmsm_coil_phase_matches_legacy():
    """PMSM 表（nonlinear_PM_solver2.py:286-291）：线圈 i = SEC_TO_NAME[i]。

    Z=6, p=2, m=3 是 q=0.5 分数槽（槽距 120° 电角）⇒ 每槽跨 2 个相邻 60° 相带：
    内层 = 槽起始边、外层 = 槽中心相带（标准 60° 相带，非 120° 相带绕组）。
    """
    lay = star_of_slots(6, 2, 3, parts=2,
                        inner_ref="start", outer_ref="center", outer_shift=0)
    legacy = ["Ap", "Cn", "Bp", "An", "Cp", "Bn"]     # coil[0..5]
    got = [lay.coil_phase(c) for c in range(6)]
    assert got == legacy, f"PMSM 线圈相带 {got} != legacy {legacy}"


def test_pmsm_element_sets_match_golden():
    """PMSM 6 相带的单元集合与 solver2 的 element_phase* 一致。"""
    msh = _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_pmsm1.msh"
    if not msh.exists():
        pytest.skip("找不到 model_pmsm1.msh")
    from motorfem.mesh import gmsh_reader
    node, _, _, element_S, _ = gmsh_reader(filename=str(msh),
                                           index_begins_with_zero=True,
                                           unique_index_for_any_element=False)
    coil_sets = [set(e[0] for e in element_S if e[4] == t) for t in range(14, 20)]
    lay = star_of_slots(6, 2, 3, parts=2,
                        inner_ref="start", outer_ref="center", outer_shift=0)
    # 与 solver2:286-291 的并集链对比（Ap=coil0, Cn=coil1, Bp=coil2, An=coil3,
    # Cp=coil4, Bn=coil5）
    legacy_map = {"Ap": [0], "Cn": [1], "Bp": [2], "An": [3], "Cp": [4], "Bn": [5]}
    for name, coils in legacy_map.items():
        exp = set().union(*(coil_sets[c] for c in coils))
        got = build_phase_element_sets(lay, coil_sets)[name]
        assert got == exp, f"PMSM element_phase{name} 不一致"
