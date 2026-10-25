"""tests/config/test_config.py —— 阶段 2 参数对象（motorfem/config）的测试

覆盖：
* 枚举折并（StressType / RelStressType）的双向一致与非法输入
* RunConfig.from_legacy 的字段映射（用 legacy 生产值字典）
* JSON 序列化往返（含嵌套 dataclass / tuple / 枚举还原）
* to_dict 可 JSON 化（无 ndarray、无枚举）
* 单位后缀规范（_m/_pa/_rpm）
* ⚠ 锚定测试：LEGACY_GLOBALS 与 golden/manifest.json 的 module_scalars 交叉验证，
  防止「测试里手写的 legacy 值」与真实生产配置漂移（golden 不存在时 skip）
"""
import json
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig, StressType, RelStressType
from motorfem.config.enums import (collapse_rel_stress_type, collapse_stress_type,
                                   expand_rel_stress_type, expand_stress_type)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: legacy 生产值（照抄 MMA10:71-170 配置块 + 推导量）。**不改这些值**——
#: 它们必须与 golden/manifest.json 的 module_scalars 一致（见锚定测试）。
LEGACY_GLOBALS = {
    # GeometryConfig
    "pole_pairs": 2, "Z": 36, "parts": 4,
    "m_R": 72.5e-3, "m_Rrotor": 44.75e-3, "m_Rirotor": 15.0e-3, "_coord_err_abs": 1e-6,
    # MaterialConfig
    "mu0": 4 * np.pi * 1e-7, "mech_E": 200e9, "mech_nu": 0.30, "mech_rho": 7850,
    # WindingSpec / ElectricalConfig
    # ⚠ area_current：MMA10:507 初值是 0 占位，模块体运行后被几何重算为 9.9e-05
    #（golden manifest 采的是运行后快照 —— 见 test_legacy_globals_match_golden_manifest）
    "current_amp": 5, "N_conductor": 20, "area_current": 9.9e-05,
    # DiscretizationConfig
    "N_node": 5117, "N_element": 9599, "N_element_mech": 6362,
    # TimeConfig
    "time_simulation": 10e-3 / 9, "N_point_simulation": 11, "N_point_sampled": 13,
    "use_random_sample_method": False,
    # TopologyConfig
    "p_SIMP": 3, "min_SIMP": 1e-6, "SIMP_beta": 8, "SIMP_thb": 0.5,
    "SIMP_linear_factor": 0.0, "max_density": 0.65,
    "rho_init_radius": 1.25e-3, "rho_init_distance": 4.0e-3,
    "use_poly_constraint": True,
    # FilterConfig
    "if_filter_rho": True, "if_filter_grad": True,
    "filter_radius": 2e-3, "filter_p_exp": 2,
    "filter_grad_radius": 0.4e-3, "filter_grad_p_exp": 0.5,
    # ObjectiveConfig
    "mean_torque_obj": 90, "max_torque_ripple": 5, "p_exp_torque_ripple": 10,
    "torque_ripple_computed_around_obj_torque": True,
    "weight_torque": 100, "weight_torque_ripple": 10, "weight_density": 1,
    "Tflux_ratio": 0.5, "use_random_mean_torque": False,
    # ALConfig
    "AL_stress_mu0": 0.01, "AL_stress_mu_max": 1e3, "AL_stress_mu_update_step": 50,
    # StressConfig
    "max_stress_constraint": 50e7, "use_rel_stress_constraint": True,
    "rel_stress_p_exp": 0.5, "max_rel_stress_constraint": 50e7 * 0.8,
    "use_type1_stress": False, "use_type2_stress": True,
    "use_type1_rel_stress": False, "use_type2_rel_stress": False,
    "use_type3_rel_stress": True, "use_type4_rel_stress": False,
    # MechConfig
    "rot_speed_rpm": 10000, "rot_center_x": 0, "rot_center_y": 0, "Fload": 0,
    "p_SIMP_mech": 3, "min_SIMP_mech": 1e-4, "SIMP_beta_mech": 8, "SIMP_thb_mech": 0.65,
    # SolverConfig
    "use_nonlinear_iron": True, "EM_MS_boundary_A_eq": False, "job_n_jobs": 14,
    # OutputConfig
    "save_numpy_data": True, "show_convergence_curve": True,
    "save_figure_per_epoch": 10, "figsize_model": (10, 9), "figsize_curve": (6, 5),
    # RuntimeConfig
    "img_path": "./imgs/result_test", "now": "2026-09-22-00-00-00",
}


# ---------------------------------------------------------------- 枚举折并
def test_collapse_stress_type_legacy_production():
    """生产配置：use_type1=False / use_type2=True → TYPE2。"""
    assert collapse_stress_type(False, True) == StressType.TYPE2
    assert collapse_stress_type(True, False) == StressType.TYPE1


def test_collapse_stress_type_rejects_invalid():
    """非法输入（0 个或 2 个 True）必须报错，不延续 legacy 的静默。"""
    with pytest.raises(ValueError):
        collapse_stress_type(False, False)
    with pytest.raises(ValueError):
        collapse_stress_type(True, True)


def test_collapse_rel_stress_type_each_branch():
    """四个互斥布尔的每个枚举分支都正确。"""
    for i in (1, 2, 3, 4):
        flags = [False] * 4
        flags[i - 1] = True
        assert collapse_rel_stress_type(*flags) == RelStressType(i)
    with pytest.raises(ValueError):
        collapse_rel_stress_type(False, False, False, False)  # 0 个 True
    with pytest.raises(ValueError):
        collapse_rel_stress_type(True, True, False, False)    # 2 个 True


def test_expand_is_inverse_of_collapse():
    """折并 ⇄ 展开双向一致。"""
    for st in StressType:
        assert collapse_stress_type(*expand_stress_type(st)) == st
    for rst in RelStressType:
        assert collapse_rel_stress_type(*expand_rel_stress_type(rst)) == rst


# ---------------------------------------------------------------- from_legacy
def test_from_legacy_fields():
    cfg = RunConfig.from_legacy(LEGACY_GLOBALS)

    assert cfg.geometry.stator_outer_radius_m == 72.5e-3
    assert cfg.geometry.rotor_outer_radius_m == 44.75e-3
    assert cfg.geometry.shaft_radius_m == 15.0e-3
    assert cfg.geometry.pole_pairs == 2 and cfg.geometry.Z == 36 and cfg.geometry.parts == 4

    assert cfg.material.mech_E_pa == 200e9 and cfg.material.mech_nu == 0.30
    assert cfg.material.mu0 == 4 * np.pi * 1e-7

    assert cfg.electrical.winding.current_amp == 5
    assert cfg.electrical.winding.N_conductor == 20

    assert cfg.discretization.N_node == 5117
    assert cfg.discretization.N_element == 9599

    assert cfg.time.N_point_simulation == 11

    assert cfg.topology.SIMP_beta == 8 and cfg.topology.max_density == 0.65

    assert cfg.filter.filter_radius_m == 2e-3

    assert cfg.objective.weight_torque == 100

    assert cfg.stress.max_stress_constraint_pa == 50e7
    assert cfg.stress.al.mu0 == 0.01 and cfg.stress.al.mu_update_step == 50

    assert cfg.mech.rot_speed_rpm == 10000

    assert cfg.solver.job_n_jobs == 14

    assert cfg.mma.max_step == 3000 and cfg.mma.MMA_move == 0.1

    assert cfg.output.figsize_model == (10, 9)

    assert cfg.runtime.img_path == "./imgs/result_test"

    # 性能钩子（阶段 7：只留开关，不实现）—— legacy 无对应物，取默认值
    assert cfg.solver.spsolve_reuse_symbolic is False
    assert cfg.runtime.omp_num_threads is None
    assert cfg.runtime.use_gpu is False


def test_from_legacy_enum_collapse():
    """生产值折并：stress_type=TYPE2、rel_stress_type=TYPE3。"""
    cfg = RunConfig.from_legacy(LEGACY_GLOBALS)
    assert cfg.stress.stress_type is StressType.TYPE2
    assert cfg.stress.rel_stress_type is RelStressType.TYPE3


# ---------------------------------------------------------------- 序列化
def test_to_dict_is_json_serializable():
    """to_dict 结果必须能直接 json.dumps —— 无 ndarray、无枚举、无 tuple 泄漏。"""
    cfg = RunConfig.from_legacy(LEGACY_GLOBALS)
    d = cfg.to_dict()
    s = json.dumps(d)
    assert isinstance(s, str)


def test_json_roundtrip():
    """to_json → from_json 往返，逐字段相等（含嵌套 dataclass / tuple / 枚举）。"""
    cfg = RunConfig.from_legacy(LEGACY_GLOBALS)
    cfg2 = RunConfig.from_json(cfg.to_json())

    import dataclasses
    assert dataclasses.asdict(cfg) == dataclasses.asdict(cfg2)


def test_perf_hooks_survive_json_roundtrip():
    """性能钩子显式开启后 JSON 往返不丢（阶段 8 通过配置层传递的前提）。"""
    cfg = RunConfig.from_legacy(LEGACY_GLOBALS)
    cfg.solver.spsolve_reuse_symbolic = True
    cfg.runtime.omp_num_threads = 4
    cfg.runtime.use_gpu = True
    cfg2 = RunConfig.from_json(cfg.to_json())
    assert cfg2.solver.spsolve_reuse_symbolic is True
    assert cfg2.runtime.omp_num_threads == 4
    assert cfg2.runtime.use_gpu is True
    assert cfg2.stress.stress_type is StressType.TYPE2
    assert cfg2.stress.rel_stress_type is RelStressType.TYPE3
    assert cfg2.output.figsize_model == (10, 9)  # tuple 还原


# ---------------------------------------------------------------- 单位后缀
def test_unit_suffixes():
    """设计文档规范：字段名单位后缀写死（_m/_pa/_rpm）。"""
    geom_fields = {f.name for f in __import__(
        "dataclasses").fields(RunConfig.from_legacy(LEGACY_GLOBALS).geometry)}
    assert {"stator_outer_radius_m", "rotor_outer_radius_m", "shaft_radius_m"} <= geom_fields
    stress = RunConfig.from_legacy(LEGACY_GLOBALS).stress
    assert stress.max_stress_constraint_pa > 0
    assert stress.max_rel_stress_constraint_pa > 0
    mech = RunConfig.from_legacy(LEGACY_GLOBALS).mech
    assert mech.rot_speed_rpm == 10000


# ---------------------------------------------------------------- 锚定 golden
def test_legacy_globals_match_golden_manifest():
    """⚠ 防漂移：测试里手写的 LEGACY_GLOBALS 必须与 golden/manifest.json 的
    module_scalars（真实 legacy 命名空间快照）一致。

    module_scalars 只含 int/float/bool（str/tuple 被采集时排除），所以
    只对其中出现的 key 断言；golden 不存在（旧 checkout）时跳过。
    """
    manifest = _REPO_ROOT / "golden" / "manifest.json"
    if not manifest.exists():
        pytest.skip("golden/manifest.json 不存在 —— 基线冻结尚未入库")
    ms = json.loads(manifest.read_text(encoding="utf-8"))["module_scalars"]

    # ⚠ golden_capture 在 MMA 入口把它置成 1e9 关画图，不是生产值 10 —— 排除
    # （其余被补丁的量：joblib 是替换属性不改变量；BLAS 线程是环境变量不进命名空间）
    EXCLUDED = {"save_figure_per_epoch"}

    mismatches = []
    for k, v in LEGACY_GLOBALS.items():
        if k in EXCLUDED or k not in ms:
            continue
        # float 逐位比较（rtol=atol=0）；int/bool 用 ==
        ok = np.isclose(ms[k], v, rtol=0, atol=0) if isinstance(v, float) else ms[k] == v
        if not ok:
            mismatches.append(f"{k}: manifest={ms[k]} vs test={v}")
    assert not mismatches, "LEGACY_GLOBALS 与 golden 基线不一致:\n" + "\n".join(mismatches)


# ---------------------------------------------------------------- 嵌套默认
def test_defaults_do_not_share_state():
    """嵌套 dataclass 用 default_factory，不同实例不共享 ALConfig/WindingSpec。"""
    a = RunConfig()
    b = RunConfig()
    assert a.stress.al is not b.stress.al
    assert a.electrical.winding is not b.electrical.winding
    a.stress.al.mu0 = 999.0
    assert b.stress.al.mu0 == 0.01
