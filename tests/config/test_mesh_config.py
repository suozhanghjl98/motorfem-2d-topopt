# -*- coding: utf-8 -*-
"""tests/config/test_mesh_config.py —— 网格伴生配置文件（``<stem>.motor.json``）。

核心判据（``CLAUDE.md`` §4 分级）：
* **A / 防漂移**：``RunConfig.legacy()`` 应用 36 槽配置后 ``to_dict()`` **逐位不变**
  （这是"纯重构不改数值"的硬保证）。
* **防漂移**：仓库里的 36 槽文件 == ``project_run_config(RunConfig())``。
* **A**：默认值比对语义（显式设过的字段不被覆盖）。
* 护栏：白名单外键 / 缺失 / 损坏 / ``file`` 不符 → 对应异常或 None。

运行::

    PYTHONPATH=. python -m pytest tests/config/test_mesh_config.py -q
"""
import json
from pathlib import Path

import pytest

from motorfem.config import (
    MESH_BOUND_FIELDS, MeshConfigError, RunConfig, apply_mesh_config,
    load_mesh_config, mesh_config_path, project_run_config,
)

_REPO = Path(__file__).resolve().parents[2]
_MSH_36 = _REPO / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh"
_MSH_36_FROZEN = _REPO / "legacy_msh" / "model_mirror4.msh"
_MSH_6S4P = _REPO / "legacy" / "pytorch" / "my_gmsh" / "model_6s4p.msh"


def _sidecar(tmp: Path, name: str, run_config: dict) -> str:
    """在 tmp 里造 ``<name>.msh`` + 伴生 json，返回 msh 路径。"""
    msh = tmp / f"{name}.msh"
    msh.write_bytes(b"")                       # 内容不参与（sha256=None）
    cfg_path = mesh_config_path(msh)
    cfg_path.write_text(json.dumps({
        "schema_version": 1, "file": msh.name, "sha256": None,
        "run_config": run_config,
    }), encoding="utf-8")
    return str(msh)


# ---------------------------------------------------------------- 逐位 / 防漂移


def test_legacy_sidecar_is_bitwise_noop():
    """A：legacy() 应用 36 槽配置后整份 to_dict() 逐位不变（纯重构的硬保证）。"""
    cfg = RunConfig.legacy()
    before = cfg.to_dict()
    apply_mesh_config(cfg, _MSH_36)
    assert cfg.to_dict() == before


def test_committed_36slot_equals_defaults():
    """防漂移：仓库 36 槽文件 == project_run_config(RunConfig())（手改即红）。"""
    for msh in (_MSH_36, _MSH_36_FROZEN):
        d = load_mesh_config(msh)
        assert d is not None, f"{mesh_config_path(msh)} 缺失"
        assert d == project_run_config(RunConfig())


# ---------------------------------------------------------------- 覆盖语义（A）


def test_explicit_setting_is_preserved():
    """A：默认值比对 —— 显式设过的字段（≠ 默认）不被网格配置覆盖。"""
    cfg = RunConfig()
    cfg.electrical.winding.current_amp = 7.7    # 探针"扫电流"的典型写法
    apply_mesh_config(cfg, _MSH_6S4P)
    assert cfg.electrical.winding.current_amp == 7.7     # 保留
    assert cfg.geometry.Z == 6                            # 其余仍被覆盖
    assert cfg.electrical.winding.phase_from_generator is True


def test_default_valued_field_is_overridden():
    """A：仍是默认值的字段会被网格配置覆盖（6s4p 的 geometry/时间轴）。"""
    cfg = RunConfig()
    apply_mesh_config(cfg, _MSH_6S4P)
    g = cfg.geometry
    assert (g.Z, g.parts, g.pole_pairs) == (6, 2, 2)
    assert g.rotor_outer_radius_m == 44.5e-3
    assert g.arkkio_gap_m == 0.5e-3
    assert g.rotor_design_half_pole is True
    assert cfg.time.time_simulation_s == pytest.approx(2 / (50.0 * 6))
    # 2026-10-14 修正：6s4p 换 star-of-slots 相带后 A 相轴线比 36 槽差 **+29.83°电角**
    # （解析 tools/_chk_axis2.py）⇒ 同一 offset 下功角差 29.83°。设为 −29.827 使功角
    # 与 36 槽**逐度相同**（FEM 功角扫描：36槽 |T| 负峰在 offset=0、6s4p 在 −30）。
    # phaseA_offset_deg 的单位是**电角度**（preassembly.py:322）。见文档 §14.2。
    assert cfg.time.phaseA_offset_deg == pytest.approx(-29.827)
    # 2026-10-14 修正：6s4p 扇区 = 180°机械 = 完整 360°电角 ⇒ 主从边界必须是**周期**。
    # 判据：气隙 B_r 的 DC 必须为 0（false → 1.08 T ✗，true → 0.015 T ✓）；
    # 见 docs/6s4p入流水线适配_2026-10-12.md §14.5。
    assert cfg.solver.EM_MS_boundary_A_eq is True


def test_apply_is_idempotent():
    """A：连续应用两次 == 一次。"""
    a = RunConfig(); apply_mesh_config(a, _MSH_6S4P)
    b = RunConfig(); apply_mesh_config(b, _MSH_6S4P); apply_mesh_config(b, _MSH_6S4P)
    assert a.to_dict() == b.to_dict()


def test_projection_keys_are_all_whitelisted():
    """白名单完整覆盖投影出的每条路径（防白名单/字段漂移）。"""
    leaves = []

    def walk(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(f"{prefix}{k}.", v)
            else:
                leaves.append(f"{prefix}{k}")
    walk("", project_run_config(RunConfig()))
    assert set(leaves) == set(MESH_BOUND_FIELDS)


# ---------------------------------------------------------------- 护栏


def test_missing_file_returns_none_and_require_raises(tmp_path):
    msh = str(tmp_path / "nope.msh")
    assert load_mesh_config(msh) is None
    assert apply_mesh_config(RunConfig(), msh) is None            # auto
    with pytest.raises(MeshConfigError):
        apply_mesh_config(RunConfig(), msh, mode="require")


def test_ignore_mode_skips_loading():
    cfg = RunConfig()
    assert apply_mesh_config(cfg, _MSH_6S4P, mode="ignore") is None
    assert cfg.geometry.Z == 36                                    # 未被覆盖


def test_non_whitelist_key_raises(tmp_path):
    msh = _sidecar(tmp_path, "foo", {"objective": {"weight_torque": 1.0}})
    with pytest.raises(MeshConfigError, match="白名单"):
        load_mesh_config(msh)


def test_unknown_field_raises(tmp_path):
    msh = _sidecar(tmp_path, "foo", {"geometry": {"not_a_field": 1}})
    with pytest.raises(MeshConfigError):
        load_mesh_config(msh)


def test_corrupt_json_raises(tmp_path):
    msh = tmp_path / "foo.msh"; msh.write_bytes(b"")
    mesh_config_path(msh).write_text("{ not json", encoding="utf-8")
    with pytest.raises(MeshConfigError, match="JSON"):
        load_mesh_config(str(msh))


def test_file_mismatch_raises(tmp_path):
    msh = _sidecar(tmp_path, "foo", {"geometry": {"Z": 6}})
    p = mesh_config_path(msh)
    d = json.loads(p.read_text(encoding="utf-8")); d["file"] = "other.msh"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(MeshConfigError, match="不符"):
        load_mesh_config(msh)


def test_schema_version_mismatch_raises(tmp_path):
    msh = _sidecar(tmp_path, "foo", {"geometry": {"Z": 6}})
    p = mesh_config_path(msh)
    d = json.loads(p.read_text(encoding="utf-8")); d["schema_version"] = 99
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(MeshConfigError, match="schema_version"):
        load_mesh_config(msh)
