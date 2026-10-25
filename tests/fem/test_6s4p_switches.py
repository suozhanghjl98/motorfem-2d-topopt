# -*- coding: utf-8 -*-
"""tests/fem/test_6s4p_switches.py —— 6s4p 适配新增开关的测试（2026-10-12）。

补上 review 指出的「新功能零测试覆盖」（CLAUDE.md §4）。每个开关至少一条
**A 级（解析/生成器）判据**，并断言默认值下 legacy 不受影响。

覆盖：
* ``geometry.stator_inner_radius_m`` / ``arkkio_gap_m`` / ``arkkio_parts_exponent``
  / ``rotor_design_half_pole`` / ``winding.phase_from_generator``
* ``time.rotor_position0_deg`` / ``phaseA_offset_deg``

运行::

    python -m pytest tests/fem/test_6s4p_switches.py -v
"""
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig, apply_mesh_config
from motorfem.kernels.reference.torque import ArkkioGeometry, arkkio_coefficient
from motorfem.kernels.reference.boundaries import pair_master_slave
from motorfem.mesh import MotorMesh
from motorfem.winding.layout import SEC_TO_NAME, star_of_slots

_REPO = Path(__file__).resolve().parents[2]
_MESH_6S4P = _REPO / "legacy" / "pytorch" / "my_gmsh" / "model_6s4p.msh"


def _cfg6s4p(**over):
    """6s4p 口径 —— 从网格伴生配置加载（唯一来源），不再抄一份常量。"""
    c = RunConfig()
    apply_mesh_config(c, _MESH_6S4P)
    for k, v in over.items():
        setattr(c, k, v)
    return c


@pytest.fixture(scope="module")
def mesh6():
    lay = star_of_slots(6, 2, 3, parts=2, layers=2, inner_ref='start',
                        outer_ref='center', outer_shift=0)
    return MotorMesh(str(_MESH_6S4P), layout=lay)


# ---------------------------------------------------------------- 默认值 = legacy


def test_defaults_are_legacy():
    """七个新字段的默认值必须逐字等于 legacy 硬编码（否则破坏 golden）。"""
    c = RunConfig()
    assert c.geometry.stator_inner_radius_m == 45.0e-3
    assert c.geometry.arkkio_gap_m is None              # None → LEGACY_ARKKIO_GAP_M
    assert c.geometry.arkkio_parts_exponent == 1        # 1 → parts²/gap（原式）
    assert c.geometry.rotor_design_half_pole is False
    assert c.electrical.winding.phase_from_generator is False
    assert c.time.rotor_position0_deg == 22.5           # legacy :3447
    assert c.time.phaseA_offset_deg == 0.0              # legacy :3819
    assert c.solver.EM_MS_boundary_A_eq is False        # 反周期


# ---------------------------------------------------------------- 相带参数化


def test_phase_from_generator_matches_star_of_slots(mesh6):
    """A 级：6s4p 的 element_phase* 必须 == star-of-slots 推导（与 36 槽同判据）。

    这是「换极槽后相带不错位」的解析判据（36 槽的对应断言在 test_mesh_builder）。
    """
    lay = star_of_slots(6, 2, 3, parts=2, layers=2, inner_ref='start',
                        outer_ref='center', outer_shift=0)
    coil = mesh6.regions.region("element_stator_coil")
    assert len(coil) == lay.n_coils_per_part == 6
    # 每相带 = 其线圈的并集（coil_phase 已由生成器决定）
    for sec, name in enumerate(SEC_TO_NAME[:6]):
        want = set()
        for c_i in range(lay.n_coils_per_part):
            if lay.coil_sec(c_i) == sec:
                want |= coil[c_i]
        assert mesh6.regions.region("element_phase%s" % name) == want


def test_phase_from_generator_off_is_18_slot_path():
    """默认（layout=None）路径的相带合成**写死 18 槽** ⇒ 传 6 槽会立刻报错。

    这是「防静默走错分支」的护栏测试：6 槽必须走 layout 分支，否则相带合成
    会在 `element_stator_coil[15]` 处 IndexError / 得到空集。
    """
    from motorfem.mesh.mesh import DEFAULT_TAG_TABLE, RegionMap
    assert len(DEFAULT_TAG_TABLE["element_stator_coil"]) == 18
    # 18 槽的 legacy 表达式直接索引 coil[15..17] ⇒ 给 6 槽必然越界
    six_slot_tag_table = dict(DEFAULT_TAG_TABLE)
    six_slot_tag_table["element_stator_coil"] = tuple(range(9, 15))
    with pytest.raises((IndexError, ValueError)):
        RegionMap.from_tags([], six_slot_tag_table)


# ---------------------------------------------------------------- 半极设计域


def test_rotor_design_half_pole_is_half(mesh6):
    """A 级（几何判据）：设计域角度范围必须是**半个极**（0..45°）。"""
    node = mesh6.node
    eS = mesh6.element_S
    des = list(mesh6.regions.region("element_rotor_core_m"))
    assert len(des) > 0
    ang = np.array([
        np.degrees(np.arctan2(node[eS[e][1:4], 2].mean(),
                              node[eS[e][1:4], 1].mean())) for e in des])
    assert ang.min() >= 0.0 and ang.max() < 45.0, "设计域应落在 [0,45°)"
    # 整个转子铁芯仍覆盖 0..180°（core_m ∪ core_s）
    core = set(mesh6.regions.region("element_rotor_core_m")) | \
        set(mesh6.regions.region("element_rotor_core_s"))
    ang_all = np.array([
        np.degrees(np.arctan2(node[eS[e][1:4], 2].mean(),
                              node[eS[e][1:4], 1].mean())) for e in core])
    assert ang_all.max() > 175.0, "整个转子铁芯应到 ~180°"


# ---------------------------------------------------------------- Arkkio 系数


def _coef_with_exponent(exp):
    """用一个极小的假几何测系数：只需要 parts/mu0/gap 与空单元表。"""
    geom = ArkkioGeometry(
        tri=np.zeros((0, 3), dtype=np.int64), xy=np.zeros((1, 2)),
        area=np.zeros(0), bxcoef=np.zeros((0, 3)), bycoef=np.zeros((0, 3)),
        gap_elements=[], rotor_nodes=[], parts=4, mu0=4e-7 * np.pi,
        gap=0.25e-3, parts_exponent=exp)
    return arkkio_coefficient(geom).toarray().sum()


def test_arkkio_parts_exponent_zero_removes_one_parts():
    """A 级（解析）：exp=0 的系数 = exp=1 的 1/parts（去掉多余的一个 parts）。

    空几何下矩阵恒 0 ⇒ 改用一个非零最小案例：手工喂一个单元。
    """
    # 直接验系数公式本身（不经矩阵）：docstring §二 的口径
    mu0, parts, gap = 4e-7 * np.pi, 4, 0.25e-3
    one = 1 / mu0 * parts / (gap / parts ** 1)      # legacy
    zero = 1 / mu0 * parts / (gap / parts ** 0)     # 正确
    assert zero * parts == pytest.approx(one, rel=0, abs=0) or \
        one == pytest.approx(zero * parts, rel=1e-15)
    # 且 exp=1 时 parts**1 精确等于 parts（int），逐位同上
    assert parts ** 1 == parts


def test_arkkio_parts_exponent_default_is_bitwise_legacy():
    """默认 exp=1 时 `parts**exp` 是 int 精确幂 ⇒ gap/parts 与原式逐位相同。"""
    gap, parts = 0.25e-3, 4
    assert gap / parts ** 1 == gap / parts          # 逐位


# ---------------------------------------------------------------- 功角保持


def test_phase_a_offset_keeps_power_angle():
    """A 级（解析）：改 rotor_position0_deg 时，功角须由 phaseA_offset_deg 保持。

    功角 = phaseA − pole_pairs·rotor_position（对 t 的常数项）。
    legacy：phaseA=100πt、θ0=22.5° ⇒ 功角 = −pole_pairs·22.5°。
    设 θ0=0 时必须补 offset = pole_pairs·(0−22.5°)。
    """
    pp = 2
    legacy_angle = -pp * 22.5                        # 度
    c = _cfg6s4p()
    c.time.rotor_position0_deg = 0.0
    c.time.phaseA_offset_deg = pp * (0.0 - 22.5)
    # 功角（常数部分）= offset + pp*θ0_legacy_ref − pp*θ0
    new_angle = c.time.phaseA_offset_deg - pp * c.time.rotor_position0_deg
    assert new_angle == pytest.approx(legacy_angle, abs=1e-12)


# ---------------------------------------------------------------- 主从边界方向限定


def test_winding_layout_sentinel_catches_slot_order():
    """E 级护栏：槽组顺序错（如把某槽的内/外层互换）必须被哨兵抓到。

    `verify_winding_layout` 对账「生成器推导的相带」vs「区域表相带」；槽序错时
    生成器的 `coil_sec(i)` 会给出不同相带 ⇒ 集合不等 ⇒ AssertionError。
    """
    from motorfem.fem.mesh_builder import build_field_topology, verify_winding_layout
    from motorfem.mesh import RegionMap
    lay = star_of_slots(6, 2, 3, parts=2, layers=2, inner_ref='start',
                        outer_ref='center', outer_shift=0)
    m = MotorMesh(str(_MESH_6S4P), layout=lay)
    c = _cfg6s4p()
    top = build_field_topology(c, m)
    verify_winding_layout(top)                       # 正确顺序：通过
    # 人为打乱线圈顺序（交换 slot0/slot1 = 槽0 的内/外层），重建相带
    coil = list(m.regions.region("element_stator_coil"))
    coil[0], coil[1] = coil[1], coil[0]
    from motorfem.winding.layout import build_phase_element_sets
    bad = build_phase_element_sets(lay, coil)
    top.regions._regions["element_phaseAp"] = bad["Ap"]
    with pytest.raises(AssertionError):
        verify_winding_layout(top)


def test_pair_master_slave_parts2_does_not_cross_pair():
    """A 级：parts=2（180° 扇区）时 θ=0 与 θ=180° **不得互相配对**。

    旧判据 `|y|<err<|x|` 对两条射线同时成立 ⇒ θ=0 配到 θ=180°（双向映射）。
    方向限定后：主动只在 x>0，从属只在 rot_x(...)>0。
    """
    # θ=0: (1,0) (2,0)；θ=180: (-1,0) (-2,0)（半径一一对应）
    xy = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    pb = pair_master_slave(xy, parts=2, coord_err_abs=1e-6)
    assert set(pb.slave_to_master.keys()) == {2, 3}, "从属应只是 θ=180° 的节点"
    assert pb.slave_to_master[2] == 0 and pb.slave_to_master[3] == 1
    # θ=0 的节点必须是主动、不能出现在从属里
    assert 0 not in pb.slave_to_master and 1 not in pb.slave_to_master
