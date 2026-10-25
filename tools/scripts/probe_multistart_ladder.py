# -*- coding: utf-8 -*-
"""tools/probe_multistart_ladder.py —— **逐级筛选**（successive halving）多起点优化。

思路（用户 2026-10-16 提出）：
  选足够多的初值 → 短跑 → 留比较大的一批 → 接着跑 → 再留优 → 再接着跑 …
  每轮都做**收敛检测**，已收敛的组提前退出，不浪费后续步数。

与 ``probe_random_init_multistart.py``（一次性等长跑）的区别：

  1. **阶梯轮次**：每轮只给存活组追加 ``--steps-per-round`` 步；
  2. **真续跑**：用 MMA ``control.on_step`` 抓的 ``state`` 作下一轮的
     ``initial_state``，配 ``FieldSystem.checkpoint_state`` 一起恢复 ⇒
     **不重跑**已算过的步数（轨迹与"不中断地从头跑到同一步数"逐位一致）；
  3. **收敛提前退出**：``check_convergence.analyse`` 判 ``converged_feasible``
     的组不再进入下一轮（判 ``stalled_*`` 的**保留** —— 那是算法卡住，
     不是收敛，等后续轮次看它能不能脱困）；
  4. **按比例淘汰**：每轮留 ``--keep-frac``（默认 0.5）的最优组，
     下限 ``--keep-min``。

淘汰判据 = **该组迄今见到的最大可行转矩**（``best_T``）。
⚠ **不用"最后一步的转矩"**：中间步可能更高 —— 实测 30 步那次
``obj=10.0``（最优）出现在第 41 步、末步反而回升到 22.18。取"见到的最大值"。
⚠ 也要评价**优化器返回的 x**：它常是最好的点（runner 那边实测踩过 ——
返回设计 28.9 比任何记录点都低，却因为没进候选池被丢掉）。

用法::

    PYTHONIOENCODING=utf-8 python tools/probe_multistart_ladder.py \\
        --n-groups 32 --rounds 4 --steps-per-round 60 --jobs 8 \\
        --out tools/_ladder_20261016

产物：
  - ``<out>.log``          逐轮进度（可被 GUI 只读监视读）
  - ``<out>.npz``          每组每轮的摘要（画收敛曲线用）
  - ``<out>_designs.npz``  存活组的 x_best（绘图脚本用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tools" / "scripts"))

MESH = _REPO / "legacy" / "pytorch" / "my_gmsh" / "model_6s4p.msh"
CSV = _REPO / "legacy" / "pytorch" / "50CS470.csv"


def single_flux_config():
    """纯平均转矩任务口径（照抄 ``probe_optimizers._single_flux``）。"""
    from motorfem.config import RunConfig, apply_mesh_config
    cfg = RunConfig()
    apply_mesh_config(cfg, str(MESH))
    cfg.time.sampling_mode = "uniform"
    cfg.time.N_point_simulation = 1
    cfg.time.use_validation_set = False
    cfg.time.positions_deg = None
    oc = cfg.objective
    oc.objective_role = "con3"
    oc.mean_torque_source = "flux"
    cfg.topology.max_density = 0.65
    oc.enable_uelastic = oc.enable_stress = False
    oc.enable_con1 = oc.enable_con4 = False
    oc.enable_con2 = oc.enable_con3 = True
    oc.constraint_schedule = False
    oc.objective_role_phase2 = None
    oc.p_exp_continuation = oc.p_exp_move_linkage = False
    return cfg


class _StateGrab:
    """只抓 MMA 的 ``on_step`` 快照 —— ``initial_state`` 的同型格式。

    ⚠ 必须实现 ``pause/resume/step_once/stop_requested/wait_if_paused``
    （``mma_driver`` 会按鸭子类型调用它们），本类全部给"不停"的实现。
    """

    def __init__(self):
        self.last = None
        self.k = 0

    def on_step(self, i_step, state):
        self.last = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in state.items()}
        self.k = i_step

    # --- mma_driver 会调的鸭子类型接口（全部"不干预"）---
    def stop_requested(self):
        return False

    def wait_if_paused(self):
        return None

    def pause(self):
        pass

    def resume(self):
        pass

    def step_once(self):
        pass

    def stop(self):
        pass


def run_group(cfg_dict, seed, steps, n_jobs, prev, init_seed):
    """跑一组的一轮。``prev`` = 上一轮返回的状态 dict（第一轮为 None）。"""
    from motorfem.config import RunConfig
    from motorfem.fem.system import FieldSystem
    from motorfem.fem.optimizers import optimize_mma
    import check_convergence as cc

    cfg = RunConfig.from_dict(cfg_dict)
    cfg.runtime.n_jobs = n_jobs
    fs = FieldSystem(cfg, str(MESH), str(CSV), init_seed=init_seed)

    T_hist, dens_hist, designs, fvals = [], [], [], []
    if prev is not None:
        fs.restore_checkpoint_state(prev["fs_state"])
        x0 = np.asarray(prev["x_last"], float)
        mma_init = prev["mma_state"]
        T_hist = list(prev["T_hist"])
        dens_hist = list(prev["dens_hist"])
        designs = [np.asarray(v) for v in prev["designs"]]
        fvals = [np.asarray(v) for v in prev["fvals"]]
    else:
        x0 = np.asarray(fs.top.rho_SIMP_initial[
            list(fs.top.element_topo_opti_m)], float)
        mma_init = None

    def wrapped(x):
        f, df = fs.warper(x)
        call = fs.trace[-1][1]
        T_hist.append(float(np.mean(call.torque_flux_energy_vec)))
        dens_hist.append(float(f[1]))
        designs.append(np.array(x, copy=True))
        fvals.append(np.asarray(f, float).copy())
        fs.trace.clear()
        return f, df

    grab = _StateGrab()
    # ⚠ ``max_step`` 必须是**累计目标步数**，不是本轮增量。
    #   ``mma_driver`` 的循环是 ``while i_step < max_step``，而 ``i_step`` 从
    #   ``initial_state`` 恢复；若传本轮增量（如 5），续跑第二轮的 i_step 已经
    #   是 6 > 5 ⇒ 循环体一次都不进 ⇒ ``f_val`` 未绑定 ⇒ UnboundLocalError。
    #   （实测踩过：第二轮所有组都抛 UnboundLocalError。）
    i_step_prev = int(mma_init["i_step"]) if mma_init is not None else 0
    target_step = i_step_prev + steps
    _, x_ret = optimize_mma(wrapped, x0, np.zeros_like(x0), np.ones_like(x0),
                            max_step=target_step, mma_cfg=cfg.mma, control=grab,
                            move_at=fs._current_move, f_rel_change=-1,
                            x_rel_change=-1, initial_state=mma_init)
    # 优化器返回的 x 也评价一次（它常是最好的点）
    f_ret, _ = fs.warper(x_ret)
    call = fs.trace[-1][1]
    T_hist.append(float(np.mean(call.torque_flux_energy_vec)))
    dens_hist.append(float(f_ret[1]))
    designs.append(np.array(x_ret, copy=True))
    fvals.append(np.asarray(f_ret, float).copy())
    fs.trace.clear()

    t = np.asarray(T_hist); d = np.asarray(dens_hist)
    feas = np.flatnonzero(np.isfinite(t) & np.isfinite(d) & (d <= 1e-6))
    best_i = int(feas[np.argmax(t[feas])]) if feas.size else None
    best_T = float(t[best_i]) if best_i is not None else float("-inf")
    best_x = np.asarray(designs[best_i], float) if best_i is not None else None

    conv = cc.analyse(np.asarray(fvals, float), window=min(20, len(fvals)))
    return dict(
        seed=int(seed), init_seed=int(init_seed),
        best_T=best_T, T_last=float(t[-1]) if t.size else float("-inf"),
        T_max=float(t.max()) if t.size else float("-inf"),
        n_eval=len(T_hist), n_feasible=int(feas.size),
        best_index=best_i, x_best=best_x, x_last=np.asarray(x_ret, float),
        mma_state=grab.last, fs_state=fs.checkpoint_state(),
        verdict=conv["verdict"], obj_rel_range=conv["obj_rel_range"],
        T_hist=T_hist, dens_hist=dens_hist,
        designs=[np.asarray(v) for v in designs],
        fvals=[np.asarray(v) for v in fvals],
    )


def _worker(payload):
    cfg_dict, entries, steps, n_jobs = payload
    out = {}
    for seed, init_seed, prev in entries:
        try:
            out[seed] = run_group(cfg_dict, seed, steps, n_jobs, prev, init_seed)
        except Exception as exc:  # noqa: BLE001
            out[seed] = dict(seed=seed, init_seed=init_seed,
                             error=f"{type(exc).__name__}: {exc}",
                             best_T=float("-inf"), n_eval=0)
    return out


def _chunks(seq, k):
    """把 seq 均分成 k 段（并发用）。"""
    k = max(1, min(k, len(seq)))
    return [seq[i::k] for i in range(k)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-groups", type=int, default=32, help="初始组数")
    ap.add_argument("--rounds", type=int, default=4, help="轮数")
    ap.add_argument("--steps-per-round", type=int, default=60,
                    help="每轮给存活组追加多少步")
    ap.add_argument("--jobs", type=int, default=8, help="并行进程数")
    ap.add_argument("--n-jobs", type=int, default=1, help="每组内部的 n_jobs")
    ap.add_argument("--keep-frac", type=float, default=0.5, help="每轮保留比例")
    ap.add_argument("--keep-min", type=int, default=2, help="每轮至少保留几组")
    ap.add_argument("--seed-base", type=int, default=1)
    ap.add_argument("--out", default=str(_REPO / "tools" / "_ladder_20261016"))
    args = ap.parse_args()

    seeds = [s for s in (args.seed_base + i for i in range(args.n_groups * 3))
             if s != 1002][: args.n_groups]
    out = Path(args.out)
    lf = Path(str(out) + ".log").open("w", encoding="utf-8", newline="\n")

    def emit(line):
        lf.write(line + "\n"); lf.flush()
        print(line, flush=True)

    cfg_dict = single_flux_config().to_dict()
    emit(f"RUNNING pid={os.getpid()} target={args.rounds}")
    emit(f"逐级筛选：{args.n_groups} 组 -> {args.rounds} 轮 x "
         f"{args.steps_per_round} 步，并行 {args.jobs}，"
         f"每轮留 {args.keep_frac:.0%}（下限 {args.keep_min}）")
    emit(f"种子：{seeds[0]}..{seeds[-1]}（跳过 1002）")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    alive = [(s, s, None) for s in seeds]     # (seed, init_seed, prev)
    round_summary = []
    t_all = time.perf_counter()

    for rnd in range(1, args.rounds + 1):
        t0 = time.perf_counter()
        emit(f"\n=== 第 {rnd}/{args.rounds} 轮：{len(alive)} 组 x "
             f"{args.steps_per_round} 步 ===")
        results = {}
        with ctx.Pool(args.jobs) as pool:
            jobs = [(cfg_dict, ch, args.steps_per_round, args.n_jobs)
                    for ch in _chunks(alive, args.jobs) if ch]
            # ⚠ 用 ``imap_unordered`` 而非 ``map``：``map`` 要等**所有** chunk
            #   返回才给出结果，一轮里日志会长时间空白（看起来像卡死）。
            #   这里每个 chunk 完成就回收 + 打点（chunk 内是串行的，所以
            #   打点粒度 = 一个 chunk 跑完，够看出"在动"）。
            total_chunks = len(jobs)
            for ci, part in enumerate(pool.imap_unordered(_worker, jobs), 1):
                results.update(part)
                done = [s for s, _, _ in alive if s in results]
                emit(f"  [chunk {ci}/{total_chunks}] 已回 {len(done)}/{len(alive)} 组："
                     + " ".join(f"{s}:{results[s].get('best_T', float('nan')):.2f}"
                                for s in done[-6:]))

        rows = []
        for seed, init_seed, _ in alive:
            r = results.get(seed, {})
            if r.get("error"):
                emit(f"EVAL seed={seed:4d} 失败 {r['error'][:110]}")
                rows.append(dict(seed=seed, best_T=float("-inf"),
                                 verdict="error", n_eval=r.get("n_eval", 0)))
                continue
            rows.append({k: v for k, v in r.items()
                         if k not in ("designs", "fvals", "T_hist", "dens_hist",
                                      "x_best", "x_last", "mma_state", "fs_state")})
            emit(f"EVAL seed={seed:4d} 累计{r['n_eval']:4d}步 "
                 f"best_T={r['best_T']:8.4f} T_last={r['T_last']:8.4f} "
                 f"可行={r['n_feasible']:3d} 判定={r['verdict']} "
                 f"(obj极差 {r['obj_rel_range']:.2e})")

        # 排名 + 淘汰（按 best_T 降序）
        order = sorted([s for s in results if results[s].get("best_T", float("-inf")) > float("-inf")],
                       key=lambda s: -results[s]["best_T"])
        n_keep = max(args.keep_min, int(round(len(order) * args.keep_frac)))
        n_keep = min(n_keep, len(order))
        keep = order[:n_keep]
        emit(f"--- 本轮排名（前 {min(8, len(order))}）---")
        for i, s in enumerate(order[:8], 1):
            emit(f"  {i:2d}. seed={s:4d} best_T={results[s]['best_T']:.4f}")
        conv_out = [s for s in order if results[s].get("verdict") == "converged_feasible"]
        emit(f"--- 淘汰：留 {n_keep}/{len(order)}；"
             f"收敛退出 {len(conv_out)} 组 {conv_out if conv_out else ''} ---")
        emit(f"本轮用时 {time.perf_counter()-t0:.0f}s")

        round_summary.append(dict(round=rnd, n_alive=len(alive),
                                  n_keep=n_keep, kept=keep,
                                  rows=rows,
                                  spread=(results[order[0]]["best_T"]
                                          - results[order[-1]]["best_T"])
                                  if len(order) > 1 else 0.0))

        alive = [(s, results[s]["init_seed"], results[s]) for s in keep]
        if not alive:
            emit("没有存活组，提前结束"); break

    # ---- 落盘 ----
    arrays = dict(
        rounds=np.array([rs["round"] for rs in round_summary]),
        n_alive=np.array([rs["n_alive"] for rs in round_summary]),
        n_keep=np.array([rs["n_keep"] for rs in round_summary]),
        spread=np.array([rs["spread"] for rs in round_summary]),
        summary_json=np.array(json.dumps(
            [{k: v for k, v in rs.items() if k != "rows"} for rs in round_summary])),
        rows_json=np.array(json.dumps([rs["rows"] for rs in round_summary])),
    )
    tmp = Path(str(out) + ".npz.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, out.with_suffix(".npz"))

    # 存活组（含末轮）的 x_best
    dfl = {}
    for seed, init_seed, prev in alive:
        if prev and prev.get("x_best") is not None:
            dfl[f"x_{seed}"] = np.asarray(prev["x_best"], float)
    if dfl:
        tmpd = Path(str(out) + "_designs.npz.tmp")
        with tmpd.open("wb") as fh:
            np.savez_compressed(fh, **dfl)
        os.replace(tmpd, Path(str(out) + "_designs.npz"))
        emit(f"存活组最优设计 -> {out.name}_designs.npz（{len(dfl)} 组）")

    emit(f"\n全部完成，总用时 {time.perf_counter()-t_all:.0f}s")
    if alive and alive[0][2]:
        best = max((p for _, _, p in alive if p), key=lambda p: p["best_T"])
        emit(f"★ 全局最好：seed={best['seed']} best_T={best['best_T']:.4f} "
             f"（累计 {best['n_eval']} 步）")
        emit("各轮存活数：" + " -> ".join(str(rs["n_alive"]) for rs in round_summary))
        emit("各轮极差：" + " ".join(f"{rs['spread']:.3f}" for rs in round_summary))
    emit("COMPLETE")
    lf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
