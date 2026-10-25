# -*- coding: utf-8 -*-
"""Contracts of the approved mechanical-first experiment (no FEM solve)."""
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('best184_runner', ROOT/'tools'/'scripts'/'run_best184_mech70.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_best184_exact_initial_design():
    x = runner.source_x()
    with np.load(runner.SOURCE, allow_pickle=True) as z:
        assert np.array_equal(x, z['mma__x_calls'][183])
    assert np.isfinite(x).all() and np.all((x >= 0) & (x <= 1))


def test_stage_contracts():
    for stage, count in [(1, 1), (2, 11)]:
        c = runner.config(stage)
        o = c.objective
        assert c.time.N_point_simulation == count
        assert c.time.sampling_mode == 'uniform'
        assert not c.time.use_validation_set
        assert o.objective_role == 'uelastic+stress'
        assert o.objective_role_phase2 is None
        assert o.enable_con2 and o.enable_con3 and o.mean_torque_obj == 70
        assert o.enable_uelastic
        assert o.enable_con1 == o.enable_con4 == (stage == 2)
        assert o.ripple_ramp_steps == (200 if stage == 2 else 0)
        assert o.mean_torque_source == 'flux' and o.ripple_source == 'arkbe'
        assert c.topology.max_density == .65 and c.mma.MMA_move == .02


def test_stress_is_off_by_default_so_objective_is_pure_strain_energy():
    """★ 默认只上应变能：应力不进目标、也不作约束。

    理由：应变能已足以把转子连成一块，且**初始设计本身就满足**
    （不需要额外找可行初值）；应力留到应变能收敛后再引入。
    """
    for stage in (1, 2):
        assert runner.config(stage).objective.enable_stress is False
    assert runner.config(1, use_stress=True).objective.enable_stress is True


def test_stage1_has_no_ripple_constraints():
    """stage 1 只有密度 + 转矩两个约束，脉动（con1/con4）留到 stage 2。"""
    o = runner.config(1).objective
    assert (o.enable_con1, o.enable_con2, o.enable_con3, o.enable_con4) == \
        (False, True, True, False)


def test_objective_does_not_inherit_single_point_schedule():
    """★ 防回归：不得从单点产物继承旧口径。

    单点优化是 ``objective_role='con3'`` / ``mean_torque_obj=25`` /
    ``enable_uelastic=False``；继承+覆盖的写法一漏项就静默带旧口径跑。
    """
    with np.load(runner.SOURCE, allow_pickle=True) as z:
        old = json.loads(str(z['_args_json']))['experiment_config']['objective']
    assert old['objective_role'] == 'con3' and old['mean_torque_obj'] == 25.0
    assert old['enable_uelastic'] is False
    o = runner.config(1).objective
    assert o.objective_role != old['objective_role']
    assert o.mean_torque_obj == 70.0 != old['mean_torque_obj']
    assert o.enable_uelastic is True


def test_initial_design_is_only_design_variables():
    """初始条件只需要设计变量 x；ρ = filter_SIMP @ x 由 FieldSystem 重建。"""
    x = runner.source_x()
    assert x.shape == (1207,), "设计区（主）单元数"
    assert x.min() >= 0 and x.max() <= 1


def test_position_pool_default_and_override():
    """位置池：默认 stage1=1（单点）、stage2=11（脉动多点）；可显式覆盖。"""
    assert runner.config(1).time.N_point_simulation == 1
    assert runner.config(2).time.N_point_simulation == 11
    assert runner.config(1, n_positions=13).time.N_point_simulation == 13
    assert runner.config(2, n_positions=7).time.N_point_simulation == 7
    # 池大 but 采样模式仍是 uniform（池 = 均匀时间点，随机化在 runner 侧做）
    assert runner.config(1, n_positions=13).time.sampling_mode == 'uniform'


def test_atomic_pairing(tmp_path):
    x = runner.source_x()
    target = tmp_path/'check.npz'
    runner.atomic(target, x=x, metric=np.array([70.]))
    with np.load(target) as z:
        assert np.array_equal(z['x'], x)
        assert z['metric'][0] == 70
    assert not Path(str(target)+'.tmp').exists()


# ---------------------------------------------------------------- 产物契约
# 生产端（RunLog）与消费端（server.LogManager）是一对接口，必须交叉验证：
# 只测一边「格式对不对」没有意义，要测「对面能不能读懂」。
def _gui():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'optim_gui_server', ROOT/'tools'/'optim_gui'/'server.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_producer_log_is_readable_by_consumer(tmp_path):
    gui = _gui()
    log_path = tmp_path/'run.txt'
    log = runner.RunLog(log_path, target=250)
    log.labels({'con1': 'ArkBE', 'con2': 'density',
                'con3': 'flux70', 'con4': 'VD'})
    log.eval(2, 1, 62.5, 3.25, [0.1, -0.2, -3.7, None])
    log.eval(2, 2, 61.0, 3.10, [0.05, -0.15, -3.6, None])
    log.checkpoint('run_stage2.npz', 25)
    log.complete()
    log.close()

    mgr = gui.LogManager(log_path, tmp_path/'run.npz', 'm.msh', 'c.csv')
    mgr.refresh()
    assert mgr.status == 'done', mgr.error or mgr.msg
    assert mgr.target == 250
    assert mgr.labels['con1'] == 'ArkBE'
    assert len(mgr.rows) == 2
    assert mgr.rows[0]['obj'] == 62.5
    assert mgr.rows[0]['cons'] == [0.1, -0.2, -3.7, None]
    assert mgr.rows[0]['seconds'] == 3.25


def test_consumer_sees_error_when_producer_never_writes_terminal(tmp_path):
    """★ 生产端进程被强杀（没走 finally）⇒ 消费者必须判 unknown，不能判成功。"""
    gui = _gui()
    log_path = tmp_path/'run.txt'
    # 手工写一个「有 pid、无终态」的日志，模拟 SIGKILL
    log_path.write_text('RUNNING pid=999999 target=10\n'
                        'EVAL stage=1 n=1 obj=5.0 seconds=1\n', encoding='utf-8')
    mgr = gui.LogManager(log_path, tmp_path/'run.npz', 'm.msh', 'c.csv')
    mgr.refresh()
    assert mgr.status == 'unknown'
    assert '不判定成功' in mgr.msg


def test_runlog_close_writes_failed_if_no_terminal(tmp_path):
    """RunLog.close() 必须补终态，否则消费者的 unknown 会掩盖真实失败。"""
    log_path = tmp_path/'run.txt'
    log = runner.RunLog(log_path, target=10)
    log.eval(1, 1, 5.0, 1.0, [None, None, None, None])
    log.close()                       # 没显式写 COMPLETE/FAILED
    text = log_path.read_text(encoding='utf-8')
    assert text.rstrip().splitlines()[-1].startswith('FAILED')


def test_runlog_complete_is_terminal(tmp_path):
    log_path = tmp_path/'run.txt'
    log = runner.RunLog(log_path, target=10)
    log.complete()
    log.close()
    lines = log_path.read_text(encoding='utf-8').splitlines()
    assert lines[-1] == 'COMPLETE', 'close() 不该在 COMPLETE 之后再补 FAILED'
