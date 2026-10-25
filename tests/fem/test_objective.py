"""tests/fem/test_objective.py —— 阶段 4D：力学段 + 目标聚合（golden L4 注入对齐）

验证核心：注入 golden L4 的 ``call_{i}_x``，跑 ``fun``（EM + 力学 + 聚合）+ AL
状态机，与 golden L4 **除 mma_result_*** 外的全部数组 ``np.array_equal``。

**golden 采集伪影（本批发现的机制）**：``tools/golden_capture.py`` 的
``_snapshot_call`` 用 ``_put(snap, n, obj.__dict__[n])`` **存数组引用**，而
``obj_func_parallel_AL_v2`` 的 7 个 EM vec 数组在 ``__init__`` 建一次、``__call__``
原地覆盖 ⇒ 三次快照指向同一个数组，最终全是**末调用**的值（``u_node``/``detail``
等每次重新赋值 ⇒ 逐调用正确）。所以 golden L4 的 EM vec 键与 detail 的转矩键是
「末调用值」—— ``L4_snapshot`` 忠实复现它（逐调用的 EM 值在 L3 里验证）。

运行::

    python -m pytest tests/fem/test_objective.py -v
"""
import os
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig
from motorfem.fem.em_worker import density_from_x, make_field_material
from motorfem.fem.mech_worker import stress_flags_from_config
from motorfem.fem.mesh_builder import build_field_topology
from motorfem.fem.objective import L4_snapshot, fun
from motorfem.fem.preassembly import build_preassembly
from motorfem.kernels.reference.al import al_initial_state, al_update
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


@pytest.fixture(scope="module")
def env():
    cfg = RunConfig.legacy()
    top = build_field_topology(cfg, MotorMesh(_mesh_path()))
    pre = build_preassembly(cfg, top)
    csv = _REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv
    material = make_field_material(str(csv))
    flags = stress_flags_from_config(cfg)
    return cfg, top, pre, material, flags


def _golden():
    if not _GOLDEN_L4.exists():
        pytest.skip(f"{_GOLDEN_L4} 不存在 —— 基线冻结尚未入库")
    return np.load(_GOLDEN_L4)


def _run_injected(env, n_calls=3) -> dict:
    _cfg, top, pre, material, flags = env
    golden4 = _golden()
    x_calls = [np.asarray(golden4[f"call_{c:02d}_x"]) for c in range(n_calls)]
    state = al_initial_state(top.n_element_mech)
    a_last: dict = {}
    calls, states, counters, a_last_snaps = [], [], [], []
    for c, x in enumerate(x_calls):
        rho = density_from_x(x, top)
        call = fun(rho, pre, top, material, a_last,
                   state.al_lambda, state.al_mu, flags=flags)
        calls.append(call)
        state = al_update(call.al_lambda_new, call.stress_loss_al, state)
        states.append(state)
        counters.append(c + 1)
        a_last_snaps.append(dict(a_last))
    return L4_snapshot(calls, states, counters, x_calls, a_last_snaps, top)


def test_l4_injected_all_arrays_match_golden(env):
    """golden L4 除 mma_result_* 外全部数组 np.array_equal（注入 x）。"""
    golden4 = _golden()
    actual = _run_injected(env)
    for k in sorted(golden4.files):
        if k.startswith("mma_result"):
            continue                       # 阶段 4E 验证
        assert k in actual, f"actual 缺 {k}"
        assert np.array_equal(golden4[k], actual[k]), (
            f"{k} 与 golden 不一致\n"
            f"  golden {golden4[k].shape} / actual {np.asarray(actual[k]).shape}")
    assert set(actual) <= set(golden4.files), (
        f"actual 多出的键: {set(actual) - set(golden4.files)}")


def test_l4_vecs_reproduce_final_call_artifact(env):
    """golden 采集伪影：7 个 EM vec 键全部 = 末调用值（不是逐调用）。"""
    actual = _run_injected(env)
    for c in range(3):
        for n in ("torque_Arkkio_vec", "dTArkkio_p_drho_vec",
                  "flux_linkage_energy_vec", "torque_flux_energy_vec",
                  "dTflux_p_drho_vec", "torque_VD_vec", "dTVD_p_drho_vec"):
            assert np.array_equal(actual[f"call_{c:02d}_{n}"],
                                  actual["call_02_" + n]), \
                f"call_{c:02d}_{n} 不是末调用值（伪影复现失败）"
    # 对照：u_node 是逐调用正确的
    assert not np.array_equal(actual["call_00_u_node"], actual["call_02_u_node"])


def test_gradient_to_x_shape(env):
    """call_{i}_df 是 (5, N_topo) —— warper 的梯度后处理（从属叠加 + 滤波）。"""
    golden4 = _golden()
    assert golden4["call_00_df"].shape == (5, len(env[1].element_topo_opti_m))
    actual = _run_injected(env)
    assert actual["call_00_df"].shape == golden4["call_00_df"].shape


def test_fun_p_exp_override_matches_config_default(env):
    """fun 的 p_exp_torque_ripple=None（默认）与显式传 config 值逐位一致。

    p 连续化是 FieldSystem 层调度、fun 层接受覆盖值的两段式设计：默认
    ``None`` 必须走 legacy 固定值（golden 锚定的前提），显式传相同值结果
    ``np.array_equal``。
    """
    _cfg, top, pre, material, flags = env
    golden4 = _golden()
    x0 = np.asarray(golden4["call_00_x"])
    rho = density_from_x(x0, top)
    state = al_initial_state(top.n_element_mech)
    a_last: dict = {}
    cfg_p = _cfg.objective.p_exp_torque_ripple
    call_default = fun(rho, pre, top, material, dict(a_last),
                       state.al_lambda, state.al_mu, flags=flags)
    call_explicit = fun(rho, pre, top, material, dict(a_last),
                        state.al_lambda, state.al_mu, flags=flags,
                        p_exp_torque_ripple=cfg_p)
    assert np.array_equal(call_default.f, call_explicit.f)
    assert np.array_equal(call_default.df, call_explicit.df)


def test_enable_mech_off_skips_mech_worker(env, monkeypatch):
    """enable_uelastic & enable_stress 都关 ⇒ mech_worker 不被调用（来源 A）。

    用 monkeypatch 顶替 ``mech_worker`` 使其计数，验证跳过逻辑真正生效
    （不是「算了再乘 0」）—— 这是「跳过力学段」的性能语义。
    """
    _cfg, top, pre, material, flags = env
    golden4 = _golden()
    x0 = np.asarray(golden4["call_00_x"])
    rho = density_from_x(x0, top)
    state = al_initial_state(top.n_element_mech)

    import motorfem.fem.objective as obj_mod
    calls = {"n": 0}

    def fake_mech(*a, **k):
        calls["n"] += 1
        raise AssertionError("不应调用 mech_worker")

    monkeypatch.setattr(obj_mod, "mech_worker", fake_mech)
    monkeypatch.setattr(_cfg.objective, "enable_uelastic", False)
    monkeypatch.setattr(_cfg.objective, "enable_stress", False)
    call = fun(rho, pre, top, material, {}, state.al_lambda, state.al_mu,
               flags=flags)
    assert calls["n"] == 0, "mech_worker 应被跳过"
    assert call.f[0] == 0.0, "obj 应为 0（机械目标全禁用）"
    assert call.uelastic == 0.0 and call.stress_loss_al == 0.0
    assert call.al_lambda_new.shape == (top.n_element_mech,)
    assert call.u_node.shape == (pre.Kmech.n_dof,)


def test_objective_role_single_con3(env, monkeypatch):
    """objective_role=con3 ⇒ 转矩作目标、其余启用项作约束（来源 A）。

    f[0] = 转矩约束形式 (1−T/T_obj)·w（最小化 = 最大化转矩）；
    con1/con2/con4 仍是约束；应变能/应力默认启用则也作约束。
    验证 f 行数 = 1 + 参与约束数，且默认 5 行布局不受影响（已有
    ``test_l4_injected_all_arrays_match_golden`` 钉死默认路径）。
    """
    _cfg, top, pre, material, flags = env
    golden4 = _golden()
    x0 = np.asarray(golden4["call_00_x"])
    rho = density_from_x(x0, top)
    state = al_initial_state(top.n_element_mech)

    monkeypatch.setattr(_cfg.objective, "objective_role", "con3")
    call = fun(rho, pre, top, material, {}, state.al_lambda, state.al_mu,
               flags=flags)
    # 默认 enable 全开：目标 con3 + 约束 [uelastic, stress, con1, con2, con4] = 6 行
    assert call.f.shape[0] == 6
    # f[0] 是转矩形式，其余 5 行是启用项约束
    assert call.f[0] != 0.0


def test_enable_con3_off_removes_torque_constraint(env, monkeypatch):
    """enable_con3=False ⇒ 转矩约束不进 f（目标仍是 legacy 组合）。

    默认 role=uelastic+stress、enable_con3=False ⇒ f = [obj, con1, con2, con4]
    = 4 行；原 con3 槽位移除。
    """
    _cfg, top, pre, material, flags = env
    golden4 = _golden()
    x0 = np.asarray(golden4["call_00_x"])
    rho = density_from_x(x0, top)
    state = al_initial_state(top.n_element_mech)

    monkeypatch.setattr(_cfg.objective, "enable_con3", False)
    call = fun(rho, pre, top, material, {}, state.al_lambda, state.al_mu,
               flags=flags)
    # 目标(组合) + 参与约束 [con1, con2, con4] = 4 行
    assert call.f.shape[0] == 4
    # 目标仍是应变能+应力
    assert call.f[0] != 0.0


def test_fun_p_exp_override_low_p_changes_ripple(env):
    """低 p 确实改变 con1/con4（脉动范数聚合）—— 测试非平凡（防退化）。

    p=10 vs p=3：``|T−T_obj|^p`` 的范数聚合不同 ⇒ con1/con4 不同；obj/con2
    （应变能/密度）不受 p 影响，逐位一致。这同时验证 override 真的落进了
    ripple 聚合，而不是被忽略。
    """
    _cfg, top, pre, material, flags = env
    golden4 = _golden()
    x0 = np.asarray(golden4["call_00_x"])
    rho = density_from_x(x0, top)
    state = al_initial_state(top.n_element_mech)
    a_last: dict = {}
    call_p3 = fun(rho, pre, top, material, dict(a_last),
                  state.al_lambda, state.al_mu, flags=flags,
                  p_exp_torque_ripple=3.0)
    a_last = {}
    call_p10 = fun(rho, pre, top, material, dict(a_last),
                   state.al_lambda, state.al_mu, flags=flags,
                   p_exp_torque_ripple=10.0)
    assert call_p3.f[1] != call_p10.f[1], "con1 对 p 不敏感 —— 测试退化"
    assert call_p3.f[4] != call_p10.f[4], "con4 对 p 不敏感 —— 测试退化"
    # 不依赖 p 的分量逐位一致（防「全改了」的假阳性）
    assert call_p3.f[0] == call_p10.f[0], "obj 不应受 p 影响"
    assert call_p3.f[2] == call_p10.f[2], "con2（密度）不应受 p 影响"
