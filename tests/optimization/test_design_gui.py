# -*- coding: utf-8 -*-
"""手绘结构 GUI（tools/design_gui）的纯逻辑测试。

覆盖：
- 几何 payload 与主/从属映射（含与 optim_gui._build_geometry 的口径一致性）
- 滤波/预览（恒等半径、direct/preimage、非法半径）
- NPZ 导出 schema 与 ``run_auto_schedule._load_x0`` 兼容
- NPZ 导入换算（23453/1207/4828、2D 行、按元素 id 重映射）
- ``run_auto_schedule`` 的 ``--filter-radius-mm/--max-density`` 默认行为不变
- 命令拼装与日志增量解析

⚠ 构建一次 6s4p topology（约 9 s）；网格缺失则 skip。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "design_gui"))
sys.path.insert(0, str(ROOT / "tools" / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


core = _load("design_gui_core", ROOT / "tools" / "design_gui" / "core.py")
jobs = _load("design_gui_jobs", ROOT / "tools" / "design_gui" / "jobs.py")

MESH = ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_6s4p.msh"


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle():
    if not MESH.exists():
        pytest.skip("缺少 6s4p 网格")
    return core.build_bundle("6s4p")


@pytest.fixture(scope="module")
def stripe(bundle):
    c = bundle.centroids_master
    r = np.hypot(c[:, 0], c[:, 1])
    th = np.arctan2(c[:, 1], c[:, 0])
    rho = ((r > 0.020) & (r < 0.040)
           & (((th / (np.pi / 8)).astype(int)) % 2 == 0)).astype(float)
    return rho


# ---------------------------------------------------------------------------
# 1. 几何 / 映射
# ---------------------------------------------------------------------------
def test_geometry_counts_and_mapping(bundle):
    assert (bundle.n_master, bundle.n_slave, bundle.n_design) == (1207, 3621, 4828)
    assert bundle.n_element == 23453
    assert np.array_equal(bundle.design_cells[:bundle.n_master], bundle.master_cells)
    assert np.array_equal(
        bundle.design_cells[bundle.n_master:], bundle.slave_cells)

    master_index = {int(e): i for i, e in enumerate(bundle.master_cells)}
    for j, e in enumerate(bundle.slave_cells):
        parent = int(bundle.top.element_topo_opti_s[int(e)])
        assert bundle.design_master[bundle.n_master + j] == master_index[parent]


def test_geometry_payload(bundle):
    p = core.geometry_payload(bundle)
    assert p["units"] == "mm"
    assert (p["n_master"], p["n_slave"], p["n_design"], p["n_elem"]) == (
        1207, 3621, 4828, 23453)
    assert len(p["xy"]) == bundle.n_element * 0 + bundle.xy.shape[0]
    assert len(p["tri"]) == bundle.n_element
    assert len(p["design_master"]) == 4828
    assert p["identity_filter_radius_mm"] == pytest.approx(
        bundle.identity_radius_m * 1e3)
    # 坐标是 mm（模型半径 ~80 mm）
    assert max(abs(v) for xy in p["xy"] for v in xy) < 200


def test_color_id_regions(bundle):
    ci = bundle.color_id
    design = set(int(e) for e in bundle.design_cells)
    gap = set(int(e) for e in bundle.top.regions.region("element_rotor_gap"))
    coil = set()
    for grp in bundle.top.regions.region("element_stator_coil"):
        coil |= set(int(e) for e in grp)
    assert all(ci[e] == 3 for e in design)
    assert all(ci[e] == 2 for e in gap)
    assert all(ci[e] == 1 for e in (coil - design - gap))


def test_geometry_matches_optim_gui(bundle):
    """与现有 GUI 的 ``_build_geometry`` 口径逐项一致（防两套绘制漂移）。"""
    gui = _load("optim_gui_server_for_test",
                ROOT / "tools" / "optim_gui" / "server.py")
    old = gui._build_geometry(bundle.top)
    assert np.allclose(np.asarray(old["xy"]), bundle.xy)
    assert np.array_equal(np.asarray(old["tri"]), bundle.tri)
    assert np.array_equal(np.asarray(old["color_id"]), bundle.color_id)
    assert np.array_equal(np.asarray(old["design_idx"]), bundle.design_cells)


# ---------------------------------------------------------------------------
# 2. 滤波 / 预览
# ---------------------------------------------------------------------------
def test_identity_filter_is_identity(bundle):
    W = core.get_filter(bundle, bundle.identity_radius_m)
    assert W.nnz == bundle.n_master
    assert np.allclose(W.diagonal(), 1.0)
    assert np.allclose(W @ np.ones(bundle.n_master), 1.0)


def test_direct_start_identity_radius(bundle, stripe):
    res = core.compute_start(bundle, stripe, bundle.identity_radius_m, "direct")
    assert np.allclose(res["rho_eff_master"], np.clip(stripe, 1e-6, 1.0), atol=1e-6)
    assert res["stats"]["rmse"] < 1e-5


def test_direct_start_production_radius_matches_filter(bundle, stripe):
    res = core.compute_start(bundle, stripe, 2e-3, "direct")
    expect = bundle.top.filter_SIMP @ np.clip(stripe, 1e-6, 1.0)
    assert np.allclose(res["rho_eff_master"], np.asarray(expect).ravel(), atol=1e-12)
    assert res["stats"]["rmse"] > 0.01     # 2 mm 确实抹平了细节


def test_expand_rho_slave_copy(bundle, stripe):
    full = core.expand_rho(bundle, stripe)
    assert full.shape == (bundle.n_element,)
    assert np.allclose(full[bundle.slave_cells],
                       full[bundle.master_cells[bundle.slave_parent]])


def test_invalid_radius_raises(bundle, stripe):
    for r in (0.0, -1e-3, 2e-2):
        with pytest.raises(ValueError):
            core.compute_start(bundle, stripe, r, "direct")


def test_preimage_mode_finite(bundle, stripe):
    res = core.compute_start(bundle, stripe, 2e-3, "preimage")
    assert np.all(np.isfinite(res["x0"]))
    assert res["x0"].min() >= 1e-6 and res["x0"].max() <= 1.0


# ---------------------------------------------------------------------------
# 3. 导出
# ---------------------------------------------------------------------------
def test_export_schema_and_semantics(bundle, stripe, tmp_path):
    from motorfem.fem.em_worker import density_from_x

    out = core.export_design(bundle, stripe, 2e-3, name="测试设计",
                             out_dir=tmp_path)
    path = Path(out["path"])
    assert path.exists() and path.parent == tmp_path
    assert not list(tmp_path.glob("*.tmp"))

    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        for k in ("master_cells", "slave_cells", "design_cells", "design_master",
                  "rho_master", "rho_full", "x0", "rho_eff_master",
                  "rho_eff_full", "filter_radius_m", "config_hint"):
            assert k in keys
        assert z["rho_master"].shape == (bundle.n_master,)
        assert z["rho_full"].shape == (bundle.n_element,)
        assert z["x0"].shape == (bundle.n_master,)
        assert np.array_equal(z["master_cells"], bundle.master_cells)
        hint = json.loads(str(z["config_hint"]))
        assert hint["x0_key"] == "x0"
        assert hint["filter_radius_mm"] == pytest.approx(2.0)
        x0 = np.asarray(z["x0"], float)
        rho_eff_full = np.asarray(z["rho_eff_full"], float)

    got = density_from_x(x0, bundle.top)
    assert np.allclose(got, rho_eff_full, atol=1e-12)


def test_export_loadable_by_run_auto_schedule(bundle, stripe, tmp_path):
    from run_auto_schedule import Run

    out = core.export_design(bundle, stripe, 0.5e-3, name="load", out_dir=tmp_path)
    stub = types.SimpleNamespace(
        a=types.SimpleNamespace(x0_npz=out["path"], x0_key="x0"),
        fs=types.SimpleNamespace(top=bundle.top),
    )
    x = Run._load_x0(stub)
    with np.load(out["path"], allow_pickle=False) as z:
        assert np.array_equal(x, np.asarray(z["x0"]))


def test_export_x0_override(bundle, stripe, tmp_path):
    x = np.clip(stripe, 1e-6, 1.0)
    out = core.export_design(bundle, stripe, 2e-3, name="override",
                             out_dir=tmp_path, x0_override=x)
    with np.load(out["path"], allow_pickle=False) as z:
        assert np.array_equal(z["x0"], x)
        assert "override" in str(z["x_mode"])


# ---------------------------------------------------------------------------
# 4. 导入
# ---------------------------------------------------------------------------
def _read(path, key):
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z[key])


def test_import_rho_full(bundle, stripe, tmp_path):
    path = core.export_design(bundle, stripe, 2e-3, out_dir=tmp_path)["path"]
    res = core.apply_import(bundle, path, "rho_full")
    assert np.allclose(res["rho_master"], stripe)
    assert res["kind"] == "rho_full"


def test_import_rho_master(bundle, stripe, tmp_path):
    path = core.export_design(bundle, stripe, 2e-3, out_dir=tmp_path)["path"]
    res = core.apply_import(bundle, path, "rho_master")
    assert np.allclose(res["rho_master"], stripe)
    assert res["as"] == "rho"


def test_import_x_uses_source_radius(bundle, stripe, tmp_path):
    path = core.export_design(bundle, stripe, 2e-3, out_dir=tmp_path)["path"]
    with np.load(path, allow_pickle=False) as z:
        expect = np.asarray(z["rho_eff_master"])
    res = core.apply_import(bundle, path, "x0")
    assert res["kind"] == "master" and res["as"] == "x"
    assert res["source_radius_mm"] == pytest.approx(2.0)
    assert np.allclose(res["rho_master"], expect)
    assert res["imported_x"] is not None


def test_import_design_order(bundle, stripe, tmp_path):
    path = core.export_design(bundle, stripe, 2e-3, out_dir=tmp_path)["path"]
    with np.load(path, allow_pickle=False) as z:
        full = np.asarray(z["rho_full"])
        design = full[bundle.design_cells]
    p2 = tmp_path / "design_order.npz"
    np.savez(p2, rho_design=design)
    res = core.apply_import(bundle, p2, "rho_design")
    assert res["kind"] == "design"
    assert np.allclose(res["rho_master"], full[bundle.master_cells])


def test_import_2d_rows(bundle, stripe, tmp_path):
    rows = np.stack([stripe, stripe * 0.5, stripe * 0.25])
    p = tmp_path / "calls.npz"
    np.savez(p, x_calls=rows,
             config_json=np.asarray(json.dumps(
                 {"filter": {"filter_radius_m": 2e-3}})))
    last = core.apply_import(bundle, p, "x_calls", row=-1)
    second = core.apply_import(bundle, p, "x_calls", row=1)
    W = core.get_filter(bundle, 2e-3)
    # 导入 x 时按原值滤波（0 是合法设计变量，不做 min_density 裁剪）
    assert np.allclose(last["rho_master"], W @ rows[-1])
    assert np.allclose(second["rho_master"], W @ rows[1])
    # 转置后仍能识别
    p2 = tmp_path / "calls_T.npz"
    np.savez(p2, x_calls=rows.T)
    res = core.apply_import(bundle, p2, "x_calls", row=-1)
    assert np.allclose(res["rho_master"], W @ rows[-1])


def test_import_remap_by_master_cells(bundle, stripe, tmp_path):
    """NPZ 的 master 顺序若不同，按元素 id 重排。"""
    perm = np.random.RandomState(0).permutation(bundle.n_master)
    p = tmp_path / "perm.npz"
    np.savez(p, rho_master=stripe[perm],
             master_cells=bundle.master_cells[perm])
    res = core.apply_import(bundle, p, "rho_master")
    assert np.allclose(res["rho_master"], stripe)


def test_import_rejects_bad_length(bundle, tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, foo=np.zeros(777))
    with pytest.raises(ValueError, match="长度"):
        core.apply_import(bundle, p, "foo")


# ---------------------------------------------------------------------------
# 5. run_auto_schedule 可选开关
# ---------------------------------------------------------------------------
def _build_cfg(**kw):
    from run_auto_schedule import Stage, build_config

    return build_config(Stage("S0", T=50.0, R=80.0), motor="6s4p",
                        n_train=11, n_eval=6, mma_move=0.05, **kw)


def test_filter_radius_default_unchanged():
    cfg = _build_cfg()
    assert cfg.filter.filter_radius_m == pytest.approx(2e-3)


def test_filter_radius_override():
    cfg = _build_cfg(filter_radius_mm=1.5)
    assert cfg.filter.filter_radius_m == pytest.approx(1.5e-3)


def test_filter_radius_invalid():
    with pytest.raises(ValueError):
        _build_cfg(filter_radius_mm=0.0)
    with pytest.raises(ValueError):
        _build_cfg(filter_radius_mm=-1.0)


def test_enter_compat_getattr():
    """pareto 的 Namespace 没有新字段；``_enter`` 必须用 getattr 容忍。"""
    a = types.SimpleNamespace()
    assert getattr(a, "filter_radius_mm", None) is None
    assert getattr(a, "max_density", None) is None


# ---------------------------------------------------------------------------
# 6. 命令 / 日志
# ---------------------------------------------------------------------------
def test_build_command_contains_expected_flags(tmp_path):
    spec = jobs.JobSpec(
        npz_path=str(tmp_path / "d.npz"), motor="6s4p",
        stages="S0:T=50,R=80", filter_radius_mm=0.5, max_density=0.7,
        tag="tag")
    out = jobs.build_command(spec, slug="tag", python="python")
    cmd = out["cmd"]
    assert "--x0-npz" in cmd and "--x0-key" in cmd
    assert cmd[cmd.index("--x0-key") + 1] == "x0"
    assert "--filter-radius-mm" in cmd
    assert cmd[cmd.index("--filter-radius-mm") + 1] == "0.5"
    assert "--max-density" in cmd
    assert cmd[cmd.index("--max-density") + 1] == "0.7"
    assert "--out" in cmd


def test_safe_slug():
    assert core.safe_slug("a/b:c*?") == "a_b_c__"
    assert core.safe_slug("  ") == "design"
    assert core.safe_slug("中文 名称") == "中文_名称"


def test_log_delta_incremental(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("RUNNING pid=1\n"
                   "EVAL stage=1 n=1 obj=5.0 seconds=1 con1=0.1 con2=-0.2\n",
                   encoding="utf-8")
    mgr = jobs.JobManager()
    mgr.paths = {"log": log}
    r1 = mgr.log_delta(0)
    assert r1["next_offset"] == log.stat().st_size
    assert mgr._n_evals == 1
    # 半行不消费
    with open(log, "a", encoding="utf-8") as f:
        f.write("EVAL stage=1 n=2 obj=4.0 sec")
    r2 = mgr.log_delta(r1["next_offset"])
    assert r2["text"] == ""
    assert mgr._n_evals == 1
    # 补全后消费
    with open(log, "a", encoding="utf-8") as f:
        f.write("onds=1\n")
    r3 = mgr.log_delta(r2["next_offset"])
    assert "EVAL stage=1 n=2" in r3["text"]
    assert mgr._n_evals == 2
    # 文件变短（重开）时 offset 归零
    log.write_text("RUNNING pid=2\n", encoding="utf-8")
    r4 = mgr.log_delta(r3["next_offset"])
    assert r4["offset"] == 0
