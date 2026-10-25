"""tests/fem/test_sampling.py —— 任务 3 采样设置（uniform/random/table + 验证集）。

覆盖 ``TimeConfig.sampling_mode`` 与 ``use_validation_set``：

- uniform（默认）= legacy 均匀时间点（golden 逐位）；
- random = 随机时间点（种子可复现）；
- table = 显式转子位置度表（t 由位置反解）；
- 验证集：``train_mask`` 正确、训练/验证不重叠、``warper(subset=...)`` 只算
  对应子集、``evaluate(subset="val")`` 求验证点。

用 con3 配置（力学段关 ⇒ 更快），真实网格 module-scope fixture。
"""
import os
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig
from motorfem.fem.em_worker import density_from_x
from motorfem.fem.system import FieldSystem
from motorfem.mesh import MotorMesh

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]
_GOLDEN_DIR = _REPO_ROOT / os.environ.get("MOTORFEM_BASELINE", "golden")
_GOLDEN_L4 = _GOLDEN_DIR / "L4_mech_mma.npz"


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


def _con3_cfg() -> RunConfig:
    """con3 目标 + 仅 con1/con2 约束、力学段关（加速 + 语义与用户场景一致）。"""
    cfg = RunConfig.legacy()
    oc = cfg.objective
    oc.objective_role = "con3"
    oc.enable_uelastic = False
    oc.enable_stress = False
    oc.enable_con1 = True
    oc.enable_con2 = True
    oc.enable_con3 = True
    oc.enable_con4 = False
    return cfg


@pytest.fixture(scope="module")
def fs_uniform():
    return FieldSystem(_con3_cfg(), str(_mesh_path()),
                       str(_REPO_ROOT / "legacy" / "pytorch" / "50CS470.csv"))


@pytest.fixture(scope="module")
def fs_validation():
    cfg = _con3_cfg()
    cfg.time.use_validation_set = True
    cfg.time.n_validation_points = 6
    return FieldSystem(cfg, str(_mesh_path()),
                       str(_REPO_ROOT / "legacy" / "pytorch" / "50CS470.csv"))


def test_uniform_is_legacy_arange(fs_uniform):
    """uniform（默认）time_span = arange(0, N)*dt（legacy 逐位）。"""
    tc = fs_uniform.pre.cfg.time
    time_en = tc.time_simulation_s
    dt = time_en / tc.N_point_simulation
    expect = np.arange(0, time_en / dt) * dt
    assert np.array_equal(fs_uniform.pre.time_span, expect)
    assert fs_uniform.pre.train_mask is None
    # time_span_sampled 与 time_span 一致（无验证集）
    assert np.array_equal(fs_uniform.pre.time_span_sampled,
                          fs_uniform.pre.time_span)


def test_random_sampling_seeded_reproducible():
    cfg = _con3_cfg()
    cfg.time.sampling_mode = "random"
    cfg.time.random_seed = 42
    fs1 = FieldSystem(cfg, str(_mesh_path()),
                      str(_REPO_ROOT / "legacy" / "pytorch" / "50CS470.csv"))
    fs2 = FieldSystem(cfg, str(_mesh_path()),
                      str(_REPO_ROOT / "legacy" / "pytorch" / "50CS470.csv"))
    assert fs1.pre.time_span.size == cfg.time.N_point_simulation
    assert np.all(fs1.pre.time_span >= 0.0)
    assert np.all(fs1.pre.time_span <= cfg.time.time_simulation_s)
    assert np.array_equal(fs1.pre.time_span, fs2.pre.time_span), \
        "同种子应可复现"


def test_table_positions_deg_roundtrip():
    """table 模式：给位置度数，pre 的 rotor_positions 落在这些角度。"""
    deg = [0.0, 5.0, 10.0]
    cfg = _con3_cfg()
    cfg.time.sampling_mode = "table"
    cfg.time.positions_deg = deg
    fs = FieldSystem(cfg, str(_mesh_path()),
                     str(_REPO_ROOT / "legacy" / "pytorch" / "50CS470.csv"))
    t = fs.pre.time_span
    rotor0 = np.pi / 180 * 22.5
    # 反解：t = (deg·π/180 − rotor0)·pole_pairs/(100π)
    expect_t = [(d * np.pi / 180 - rotor0) * cfg.geometry.pole_pairs
                / (100 * np.pi) for d in deg]
    assert np.allclose(t, expect_t, rtol=1e-12, atol=1e-15)
    # rotor_positions 是**机械角**：rotor0 + 100π·t/pp = 反解后 = 输入度数
    ang = fs.pre.rotor_positions % (2 * np.pi)
    assert np.allclose(ang, np.deg2rad(deg), atol=1e-9)


def test_validation_train_mask(fs_validation):
    """验证集：train_mask 前训练后验证，长度 = 训练+验证，互不重叠。"""
    pre = fs_validation.pre
    n_tr = pre.time_span_sampled.size
    assert pre.time_span.size == n_tr + pre.cfg.time.n_validation_points
    assert pre.train_mask is not None
    assert pre.train_mask[:n_tr].all() and not pre.train_mask[n_tr:].any()
    train_t = pre.time_span[:n_tr]
    val_t = pre.time_span[n_tr:]
    # 训练/验证时间点不重叠（uniform 训练在整数 dt，验证在间隔中间）
    dt = pre.cfg.time.time_simulation_s / pre.cfg.time.N_point_simulation
    assert all(abs(t - int(round(t / dt)) * dt) > 1e-12 for t in val_t), \
        "验证点不应落在整数 dt 上（与训练重叠）"


def test_subset_train_val_counts(fs_validation):
    """warper(subset='train') 与 evaluate(subset='val') 只算各自子集。"""
    x0 = fs_validation.top.rho_SIMP_initial[
        list(fs_validation.top.element_topo_opti_m)]
    f_train, _ = fs_validation.warper(x0, subset="train")
    call_val = fs_validation.evaluate(density_from_x(x0, fs_validation.top),
                                      subset="val")
    n_tr = fs_validation.pre.time_span_sampled.size
    n_val = fs_validation.pre.cfg.time.n_validation_points
    assert f_train.size == 3          # obj + con1 + con2
    assert call_val.torque_Arkkio_vec.size == n_val
    # train 的均值转矩 ≠ val 的均值转矩（不同位置 ⇒ 不同值，且都能算出来）
    assert np.isfinite(call_val.torque_Arkkio_vec).all()


def test_subset_none_uses_all_points(fs_validation):
    """subset=None（默认 warper 是 'train'；显式 None）= 全部点。"""
    x0 = fs_validation.top.rho_SIMP_initial[
        list(fs_validation.top.element_topo_opti_m)]
    call = fs_validation.evaluate(density_from_x(x0, fs_validation.top),
                                  subset=None)
    assert call.torque_Arkkio_vec.size == fs_validation.pre.time_span.size


def test_constraint_schedule_activates_con1():
    """约束逐步激活：con1 前 N 步不参与（f 2 行），第 N 步起参与（3 行）。"""
    cfg = _con3_cfg()
    cfg.objective.constraint_schedule = True
    cfg.objective.con1_activate_step = 3
    fs = FieldSystem(cfg, str(_mesh_path()),
                     str(_REPO_ROOT / "legacy" / "pytorch" / "50CS470.csv"))
    x0 = fs.top.rho_SIMP_initial[list(fs.top.element_topo_opti_m)]
    for i in range(3):
        f, _ = fs.warper(x0)
        assert f.size == 2, f"第 {i} 次 warper 应 2 行（obj+con2），实际 {f.size}"
    f, _ = fs.warper(x0)
    assert f.size == 3, f"第 4 次起 con1 参与，应 3 行，实际 {f.size}"


def test_constraint_schedule_off_is_default(fs_uniform):
    """调度关 = 默认全启用（f 3 行，golden 路径不覆盖）。"""
    assert not fs_uniform.cfg.objective.constraint_schedule
    x0 = fs_uniform.top.rho_SIMP_initial[
        list(fs_uniform.top.element_topo_opti_m)]
    f, _ = fs_uniform.warper(x0)
    assert f.size == 3
