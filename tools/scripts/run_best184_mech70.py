# -*- coding: utf-8 -*-
"""Approved best184 -> mechanical -> ripple experiment; fresh MMA/AL states."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from motorfem.config import RunConfig
from motorfem.fem.system import FieldSystem
from motorfem.fem.optimizers import optimize_mma
from motorfem.fem.em_worker import density_from_x
from motorfem.fem.mech_worker import LegacyMechSIMP
from motorfem.kernels.reference.mech_stress import element_stress, von_mises
MESH = ROOT / 'legacy/pytorch/my_gmsh/model_6s4p.msh'
CSV = ROOT / 'legacy/pytorch/50CS470.csv'
SOURCE = ROOT / 'tools/_flux_single_mma250_20261016.npz'
#: 每多少次评价落一次盘。20 步 ≈ 1 分钟，兼顾「检查/恢复」的粒度和 I/O 开销
#: （NPZ 压缩写几百 MB 的话要调大；当前 x_calls 是 1207 维、很轻）。
CHECKPOINT_EVERY = 20


def config(stage=1, start=5.0, use_stress=False, n_positions=None):
    """机械阶段的完整配置。

    ⚠ **不继承单点优化的 objective**（旧写法从 ``_args_json`` 取整份配置再逐条
    覆盖，漏改一项就静默带着旧口径跑 —— 那是 ``objective_role='con3'`` /
    ``mean_torque_obj=25`` / ``enable_uelastic=False`` 的来源）。这里只借用
    单点产物的**几何/材料/求解器**等与优化口径无关的部分，优化相关的量全部显式写死。

    Parameters
    ----------
    use_stress : bool
        ``False``（默认）⇒ 目标 = **纯弹性应变能**，应力不进目标也不做约束。
        先只上应变能，是因为它已足以把转子连成一块，且初始设计**本身就满足**
        （不需要额外找可行初值）；应力约束留到应变能目标收敛后再引入。
        ``True`` ⇒ 目标 = 应变能 + 应力 AL 罚项。
    n_positions : int | None
        **位置池**大小（= ``N_point_simulation``）。``None`` ⇒ stage 1 用 1
        （单点，快）、stage 2 用 11（脉动需要多点）。显式给值时两个阶段都用它。
        ⚠ 池 = ``N_point_simulation`` 个均匀时间点；runner 每次评价从池里**随机
        取 1 个**（见 ``run()`` 的 RANDOM POSITION 段）。池只有 1 个时随机化
        无效果 —— 要真正随机看不同磁场，池要 > 1。
    """
    with np.load(SOURCE, allow_pickle=True) as z:
        c = RunConfig.from_dict(json.loads(str(z['_args_json']))['experiment_config'])
    o = c.objective

    # ---- 优化口径：全部显式，不依赖继承来的值 ----
    o.objective_role = 'uelastic+stress'   # 名字是组合目标的标签；stress 关掉即退化
    o.objective_role_phase2 = None         #   成纯应变能（走 objective.py 的组合分支）
    o.objective_role_switch_step = 0
    o.enable_uelastic = True
    o.enable_stress = bool(use_stress)
    o.enable_con2 = True                   # 密度/体积
    o.enable_con3 = True                   # 平均转矩 ≥ mean_torque_obj
    o.enable_con1 = o.enable_con4 = stage == 2
    o.mean_torque_source = 'flux'
    o.ripple_source = 'arkbe'
    o.mean_torque_obj = 70.0
    o.constraint_schedule = True
    # 应力入场点：use_stress=False 时 enable_stress 已为 False，此项不起作用；
    # 置 0 表示「一旦启用就立即参与」，避免与 stage 的语义纠缠。
    o.stress_activate_step = 0
    o.uelastic_activate_step = 0
    o.con3_activate_step = 0
    o.con1_activate_step = o.con4_activate_step = 0
    o.ripple_ramp_steps = 200 if stage == 2 else 0
    o.ripple_target_start = start
    o.max_torque_ripple = 5.0
    o.p_exp_torque_ripple = 10
    o.p_exp_continuation = o.p_exp_move_linkage = False
    c.time.N_point_simulation = (n_positions if n_positions is not None
                                 else (1 if stage == 1 else 11))
    c.time.sampling_mode = 'uniform'
    c.time.use_validation_set = False
    c.time.positions_deg = None
    c.topology.max_density = .65
    c.mma.MMA_move = .02
    return c


def source_x():
    """单点-仅平均转矩优化的结果，作为机械阶段的**初始设计变量**。

    只取 ``x``：``ρ = filter_SIMP @ x``，SIMP 滤波与从属单元映射都由
    ``FieldSystem`` 内部重建，其余状态（λ、MMA 渐近线、A 热启动）一律不继承
    —— 机械阶段是**新的优化问题**，不该带着旧问题的历史。

    ⚠ 不用 ``assert T >= 70``：72.59 是**自由优化**（只最大化转矩）的结果，
    不是约束条件。引入应变能/应力后转矩必然下降，那个断言会把正常结果判成失败。
    """
    with np.load(SOURCE, allow_pickle=True) as z:
        x = z['mma__x_best_feasible'].copy()
        idx = int(z['mma__best_feasible_index'])
        assert idx == 183 and np.array_equal(x, z['mma__x_calls'][idx])
        assert x.shape == (1207,) and np.isfinite(x).all()
        assert np.min(x) >= 0 and np.max(x) <= 1
    return x


def atomic(path, **data):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **data)
    os.replace(tmp, path)


class RunLog:
    """按 ``docs/GUI只读监控产物契约.md`` §2 写事件日志。

    GUI 只读监视只认这里的行格式；stdout 的 ``print`` 给人看，本类给
    消费者看，两者不共用（消费者不该去啃自由文本）。

    ⚠ 纪律：**终态标记必须写**。进程被 kill 时 ``finally`` 仍会补一条
    ``FAILED``/``STOP``；真被强杀则消费者靠 pid 判 ``unknown``，不判成功。
    """

    def __init__(self, path, target=None):
        self.path = Path(path)
        self.f = self.path.open('w', encoding='utf-8', newline='\n')
        self.line(f'RUNNING pid={os.getpid()}' + (f' target={target}' if target else ''))
        self._terminal = False

    def line(self, text):
        self.f.write(text + '\n')
        self.f.flush()                      # 行缓冲：半行不会被消费者读走
        print(f'LOG {text}', flush=True)

    def labels(self, mapping):
        self.line('LABELS ' + ' '.join(f'{k}={v}' for k, v in mapping.items()))

    def eval(self, stage, n, obj, seconds, cons):
        parts = [f'stage={stage}', f'n={n}', f'obj={obj:.10g}',
                 f'seconds={seconds:.6f}']
        for i, v in enumerate(cons, start=1):
            parts.append(f'con{i}=' + ('-' if v is None else f'{v:.10g}'))
        self.line('EVAL ' + ' '.join(parts))

    def checkpoint(self, name, n):
        self.line(f'CHECKPOINT {name} n={n}')

    def complete(self):
        self._terminal = True
        self.line('COMPLETE')

    def failed(self, why):
        self._terminal = True
        self.line('FAILED ' + str(why).replace('\n', ' ')[:300])

    def stopped(self, why):
        self._terminal = True
        self.line('STOP: ' + str(why).replace('\n', ' ')[:300])

    def close(self):
        # 没有终态标记就补一条 —— 「进程退出」不能等同「成功」。
        if not self._terminal:
            self.failed('进程结束但未写终态标记')
        self.f.close()


def metrics(fs, call):
    rho = call.rho_SIMP_mech
    k = LegacyMechSIMP().project(rho)
    # Independent physical stress diagnostic using the same displacement and material.
    vm = von_mises(element_stress(call.u_node, fs.pre.mech_tri,
                                  fs.pre.mech_l_coef, fs.pre.mech_d_mat, k))
    rel = vm / k ** fs.flags.rel_stress_p_exp
    flux = float(np.mean(call.torque_flux_energy_vec))
    return dict(T_flux=flux, uelastic=call.uelastic, stress_AL=call.stress_loss_al,
                von_mises_max_Pa=float(vm.max()), relative_TYPE3_max_Pa=float(rel.max()),
                ArkBE_mean=float(np.mean(call.torque_ArkBE_vec)),
                ArkBE_std=float(np.std(call.torque_ArkBE_vec)),
                ArkBE_pp=float(np.ptp(call.torque_ArkBE_vec)),
                ArkBE_norm10=float(np.mean(np.abs(call.torque_ArkBE_vec-70)**10)**.1),
                VD_norm10=float(np.mean(np.abs(call.torque_VD_vec-70)**10)**.1))


def independent(fs, x, label):
    call = fs.evaluate(density_from_x(x, fs.top), subset='train')
    m = metrics(fs, call)
    print(label, json.dumps(m), 'f=', call.f.tolist(), flush=True)
    return call, m


def run(stage, x0, steps, prefix, start=5, log=None, use_stress=False,
        rng_seed=0, n_pick=1, n_positions=None):
    cfg = config(stage, start, use_stress=use_stress, n_positions=n_positions)
    fs = FieldSystem(cfg, str(MESH), str(CSV))
    assert x0.shape == (len(fs.top.element_topo_opti_m),)
    assert np.isfinite(x0).all() and np.min(x0) >= 0 and np.max(x0) <= 1
    assert fs.cfg.electrical.winding.current_amp == 30
    assert fs.cfg.time.phaseA_offset_deg == -29.827
    assert fs.cfg.time.rotor_position0_deg == 22.5
    print('CONFIG', json.dumps(fs.cfg.to_dict(), ensure_ascii=False), flush=True)
    print(f'Fresh MMA and AL; stress in objective = {fs.cfg.objective.enable_stress}.', flush=True)
    baseline, bm = independent(fs, x0, f'STAGE{stage}_BASELINE')
    records, designs, fvals = [], [], []
    labels = ['objective', 'con2_density', 'con3_flux70'] if stage == 1 else ['objective', 'con1_ArkBE', 'con2_density', 'con3_flux70', 'con4_VD']
    # ⚠ 约束名与 f 的下标对齐：labels[1:] 对应 f[1:]，但它们要落到日志的**固定
    #   槽位** con1..con4 上。stage 1 是 [con2, con3]，stage 2 是 [con1..con4]。
    #   直接 enumerate(start=1) 会让 stage 1 的密度/转矩整体错位一格（实测踩过）。
    _order = [2, 3] if stage == 1 else [1, 2, 3, 4]
    slot = {i: s for i, s in enumerate(_order, start=1)}
    path = Path(str(prefix)+f'_stage{stage}.npz')
    if log is not None:
        log.labels({f'con{slot[i]}': labels[i].removeprefix(f'con{i}_')
                    for i in range(1, len(labels))})

    def save(**extra):
        atomic(path, x_initial=x0, x_calls=np.asarray(designs), f_calls=np.asarray(fvals),
               metrics_json=json.dumps(records), labels=np.asarray(labels),
               config_json=json.dumps(fs.cfg.to_dict()), baseline_json=json.dumps(bm),
               stage=stage, mesh_path=str(MESH), csv_path=str(CSV),
               mesh_sha256=hashlib.sha256(MESH.read_bytes()).hexdigest(),
               al_mu=fs.al_state.al_mu, al_lambda=fs.al_state.al_lambda,
               time_idx_used=np.asarray(time_idx_used, dtype=int), **extra)

    #: 每次评价**随机**取一个转子位置（方案 A：种子固定 ⇒ 整轮可复现）。
    #:
    #: 动机：flux 转矩几乎与转子位置无关，但**磁场分布**有关；固定位置会让
    #: 所有评价看到同一张磁场图。⚠ MMA 路径原本**没有**采样钩子（`system.py:239`
    #: 直接把 `self.warper` 交给 optimize_mma；随机子采样只接了 Adam），所以
    #: 这里在 runner 侧包一层，把 `time_idx` 显式传进 `warper`。
    #:
    #: 位置池 = `pre.time_span` 的**训练点索引**（uniform 下即 N_point_simulation
    #: 个均匀时间点）。stage 1 是单点，池里只有 1 个 ⇒ 随机化对它无效果；
    #: 要让随机真实生效，需 stage 1 用多点池（见 `--positions`）。
    n_pool = int(fs.pre.time_span_sampled.size)
    _rng = np.random.RandomState(rng_seed)
    time_idx_used: List[int] = []
    print(f'RANDOM POSITION: pool={n_pool} 个训练点, seed={rng_seed}, '
          f'每次评价随机取 {n_pick} 个', flush=True)

    def wrapped(x):
        mu = fs.al_state.al_mu
        t = time.perf_counter()
        if n_pool > n_pick:
            idx = sorted(_rng.choice(n_pool, size=n_pick, replace=False).tolist())
        else:
            idx = list(range(n_pool))
        time_idx_used.append(idx[0] if idx else 0)
        f, df = fs.warper(x, time_idx=idx)
        call = fs.trace[-1][1]
        m = metrics(fs, call)
        # ⚠ 这里**不能**断言 λ == 0。D7（λ ≡ 0）在 2026-09-23 已修：现在的更新是
        #   λ_new = max(0, λ + μh)，λ ≥ 0 **携带历史**，第一次评价起就非零
        #   （本工作点实测 g_e ≈ 165 ⇒ 首轮 λ ≈ 1.65）。旧断言建立在「AL 是纯罚项、
        #   stress_loss_al/μ 与 μ 无关」这个**已失效的前提**上，必然失败。
        #   重打分改用 f[0]（见下方候选选择）。
        m.update(evaluation=len(records)+1, al_mu_used=mu,
                 al_lambda_max=float(np.max(np.abs(fs.al_state.al_lambda))),
                 seconds=time.perf_counter()-t)
        records.append(m); designs.append(np.array(x, copy=True)); fvals.append(f.copy())
        print('EVAL', stage, len(records), json.dumps(m), 'constraints', dict(zip(labels[1:], f[1:].tolist())), flush=True)
        if log is not None:
            # 与上面 LABELS 用同一套 slot 映射，保证「名」和「值」落在同一槽位。
            cons = [None, None, None, None]
            for i in range(1, len(labels)):
                cons[slot[i] - 1] = float(f[i])
            log.eval(stage, len(records), float(f[0]), m['seconds'], cons)
        fs.trace.clear()
        if len(records) % CHECKPOINT_EVERY == 0:
            save()
            print('CHECKPOINT', path.name, len(records), flush=True)
            if log is not None:
                log.checkpoint(path.name, len(records))
        return f, df
    f, returned = optimize_mma(wrapped, x0.copy(), np.zeros_like(x0), np.ones_like(x0),
                              max_step=steps, mma_cfg=fs.cfg.mma,
                              move_at=fs._current_move, f_rel_change=-1, x_rel_change=-1)
    rc, rm = independent(fs, returned, f'STAGE{stage}_RETURNED_X_INDEPENDENT')
    # ⚠ 优化器返回的 x **不是**循环里评价过的点（它是最外一层的设计），必须把它
    #   也放进候选池 —— 实测 3 步 smoke 里它的应变能 28.9 比任何记录点都低，
    #   若只从 designs 里挑就会白扔一个更好的解。
    designs.append(np.array(returned, copy=True))
    fvals.append(np.asarray(rc.f, dtype=float).copy())
    records.append({**rm, 'evaluation': len(records) + 1, 'is_returned': True})
    a = np.asarray(fvals)
    # 严格可行（不给 1e-6 的松弛余地去放水 70 这个下限）。
    feasible = np.flatnonzero(np.all(a[:, 1:] <= 0, axis=1) & np.all(np.isfinite(a), axis=1))
    extra = dict(x_returned=returned, returned_metrics_json=json.dumps(rm),
                 returned_f_independent=rc.f, eligible_indices=feasible,
                 selection_rule='min f[0] among feasible (incl. optimizer-returned x)')
    selected = None
    if feasible.size:
        # 选项①：可行点里取 **f[0] 最小**。不用「固定 μ 重打分」——那是建立在
        # λ ≡ 0 的旧前提上，D7 修正后 stress_loss_al/μ 随 μ 变化，那个尺度不成立。
        # 目标里没应力时 f[0] 就是纯应变能，跨评价可比；有应力时见 config 的说明。
        idx = int(feasible[np.argmin(a[feasible, 0])])
        selected = designs[idx].copy()
        sc, sm = independent(fs, selected, f'STAGE{stage}_SELECTED_INDEPENDENT')
        assert np.isclose(sc.f[0], a[idx, 0], rtol=1e-10)
        if not np.all(sc.f[1:] <= 0):
            print('STOP: selected cold evaluation infeasible', flush=True)
            selected = None
        extra.update(x_selected=designs[idx], selected_index=idx,
                     selected_metrics_json=json.dumps(sm), selected_f_independent=sc.f,
                     selected_rho=density_from_x(designs[idx], fs.top),
                     selected_A_node=sc.detail['A_node'], selected_B_elem=sc.detail['B_elem'])
    save(**extra)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--out', default=str(ROOT/'tools/_best184_mech70_20261016'))
    ap.add_argument('--log', default=str(ROOT/'tools/_log_best184_mech70_20261016.txt'),
                    help='事件日志（GUI 只读监视读它，见 docs/GUI只读监控产物契约.md）')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--stress', action='store_true',
                    help='在 stage 1 的目标里加入应力 AL 罚项。默认关 —— 先只上'
                         '应变能（足以把转子连成一块，且初始设计即满足），'
                         '等它收敛后再决定要不要引应力。')
    ap.add_argument('--positions', type=int, default=None,
                    help='位置池大小（= N_point_simulation）。默认 stage1=1、stage2=11。'
                         '池 > 1 时每次评价从池里随机取 --n-pick 个位置。')
    ap.add_argument('--n-pick', type=int, default=1,
                    help='每次评价随机取几个位置（默认 1）。')
    ap.add_argument('--rng-seed', type=int, default=0,
                    help='随机位置种子（默认 0）。固定种子 ⇒ 整轮可复现。')
    args = ap.parse_args()
    # 上限从 250 放宽到 400：250 是**单点产物**的步数（SOURCE 只有 250 步），
    # 但机械阶段是**新的优化问题**，MMA 预算与那个数无关。初始设计只取 x，
    # 不取任何随步数的状态。
    assert 1 <= args.steps <= 400
    log = RunLog(args.log, target=args.steps)
    try:
        x = run(1, source_x(), args.steps, args.out, log=log,
                use_stress=args.stress, rng_seed=args.rng_seed,
                n_pick=args.n_pick, n_positions=args.positions)
        if args.smoke:
            print('SMOKE COMPLETE', flush=True)
            log.complete()
            return
        if x is None:
            print('STOP: no independently feasible stage1 candidate', flush=True)
            log.stopped('stage1 无可行的机械候选')
            return
        fs = FieldSystem(config(2, use_stress=args.stress,
                                n_positions=args.positions), str(MESH), str(CSV))
        call, m = independent(fs, x, 'STAGE2_11POINT_GATE')
        atomic(Path(args.out+'_stage2_gate.npz'), x=x, f=call.f,
               metrics_json=json.dumps(m), torque_flux=call.torque_flux_energy_vec,
               torque_ArkBE=call.torque_ArkBE_vec, torque_VD=call.torque_VD_vec)
        if m['T_flux'] < 70 or call.f[2] > 0:
            print('STOP: 11-point flux/density gate failed; floor remains 70', flush=True)
            log.stopped('11 点 flux/密度门槛未过，转矩下限仍是 70')
            return
        start = max(5., 1.1*max(m['ArkBE_norm10'], m['VD_norm10']))
        run(2, x, args.steps, args.out, start, log=log, use_stress=args.stress,
            rng_seed=args.rng_seed, n_pick=args.n_pick,
            n_positions=args.positions)
        log.complete()
    except Exception as exc:  # noqa: BLE001
        log.failed(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
