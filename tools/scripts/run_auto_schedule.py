# -*- coding: utf-8 -*-
"""自动调度拓扑优化：**按优化状态**逐级引入并加严约束（链式续跑，逼近帕累托前沿）。

用户口径（2026-10-18，已确认）
------------------------------
- 目标 = **平均转矩底座**（`T_flux` 均值），力学（应变能 + 应力 AL）作**修补项**
  —— 即 `objective_role='con3'`，其余量作约束；
- 平均转矩口径 `T_flux`（同时打印 `T_ArkBE` 作对照）；应力 = 生产相对应力 TYPE3；
- 训练 11 点 + 独立评估 6 点（验证集）；逐级入场 + 回退 + `--strict` 严格终局；
- 先做**基准**，再沿"脉振上限 / 应力上限 / 转矩下限"逐档加严；未达标也交付。

与既有调度的区别
----------------
`ObjectiveConfig.constraint_schedule`（2026-10-12）是**固定步数**的门槛
（`con1_activate_step` 等）。本脚本是**状态驱动**：每步后算严格度向量

    ρ = ( T̄/R ,  R_rip/S ,  σ/Σ ,  Ḡ/Ḡ_lim  )      （只含本档启用的项）

最近窗口 `mean ρ ≥ (1−tol)·ρ_target` 才晋级；窗口里 ρ 不再上升且转矩不动 ⇒
**停滞**：把约束放松一档（放宽 `S` 与 / 或降 `R`）并把移动极限减半重试；
连续回退到"最低档仍停滞"才判失败。`--strict` 时不放松，只缩小移动极限。

引擎要点（都是踩过坑的）
------------------------
1. **一个阶段一个 `FieldSystem`**：构造成本 ~45 s，且 AL 状态 / `A` 热启动必须
   跨档延续（否则每档冷启牛顿）。本档升级时**只换 `cfg.objective` 的值**。
2. **候选池**：每次评价的 (x, f, 指标) 都入池，最优解从池里挑
   （方法学第 3 条：最优常在中间步；`x_last_step` 也要进池）。
3. **并行靠进程**：一个进程只跑一条（CHOLMOD 两个实例会打架），由
   `run_pareto_grid.py` 用 joblib 编排出 16~18 路。
4. GUI 只读契约：`<out>.log` 的 `EVAL` 行 `con1..con4` 用 **固定槽位**
   （见 `docs/GUI只读监控产物契约.md`），`<out>_stage_<名>.npz` 带 `x_calls` +
   `config_json`。

用法
----
    PYTHONIOENCODING=utf-8 python -u tools/scripts/run_auto_schedule.py \\
        --motor 6s4p --stages 'S0:T=60|S1:T=60,R=8|S2:T=65,R=6|S3:T=70,R=5' \\
        --out tools/_out/auto_20261018/r_a
（不传 `--stages` 用 `--ladder bench|rip|stress` 的默认链，见 `LADDERS`。）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

# ⚠ Windows 远程/ joblib 子进程的控制台可能是 GBK；本文件的进度行里有
#   ``ρ / σ / T̄ / Ḡ``，会抛 UnicodeEncodeError 直接中断优化。强制 UTF-8
#   （只影响控制台显示，不影响任何数值/日志文件——``RunLog`` 本来就写 UTF-8）。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  （stdout 可能是 None / 不支持 reconfigure）
    pass

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tools" / "scripts"))

from motorfem.config import RunConfig, apply_mesh_config  # noqa: E402
from motorfem.fem.em_worker import density_from_x  # noqa: E402
from motorfem.fem.init_density import filter_inverse_rho  # noqa: E402
from motorfem.fem.optimizers import optimize_mma  # noqa: E402
from motorfem.fem.system import FieldSystem  # noqa: E402
from run_best184_mech70 import RunLog  # noqa: E402
from probe_multistart_ladder import _StateGrab  # noqa: E402

MESHES = {
    "6s4p": _REPO / "legacy" / "pytorch" / "my_gmsh" / "model_6s4p.msh",
    "synrm": _REPO / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
    # 2026-10-19 加密转子网格（同几何；主设计域 5041 / 11013 单元）
    "6s4p_d030": _REPO / "tools" / "_gap_mesh_test" / "model_6s4p_d030.msh",
    "6s4p_d015": _REPO / "tools" / "_gap_mesh_test" / "model_6s4p_d015.msh",
}
CSV_PROD = _REPO / "legacy" / "pytorch" / "50CS470.csv"

#: `objective_role='con3'` 时 `f` 的行序（`objective.py` 的 `cons` 顺序）。
#: `f` 的行序 —— **必须与 `objective.py` 的 `cons` 构造同源**：
#: 它遍历 `enabled` 字典（插入序 = uelastic/stress/con1/con2/con3/con4），
#: 跳过目标角色。2026-10-18 用了字面量顺序，导致日志把应力罚项写成 con2。
_ROW_ORDER = ("uelastic", "stress", "con1", "con2", "con3", "con4")


def f_row_names(cfg, role: str = "con3") -> list:
    """`f[1:]` 对应的量名（与 `objective.py` 的 `cons` 逐项一致）。"""
    oc = cfg.objective
    en = dict(uelastic=oc.enable_uelastic, stress=oc.enable_stress,
              con1=oc.enable_con1, con2=oc.enable_con2,
              con3=oc.enable_con3, con4=oc.enable_con4)
    return [k for k in _ROW_ORDER if en[k] and k != role]
#: GUI 日志里 con1..con4 的**固定槽位**（等价 `run_best184_mech70` 的 `_order`）。
SLOT = {"stress": 1, "con2": 2, "con3": 3, "con1": 4, "con4": None, "uelastic": None}

#: 默认阶段链（逐档加严）。`T`=平均转矩下限，`R`=脉振上限(T_ArkBE 的 p=10 范数)，
#: `sigma`=相对应力上限(Pa)，`G`=平均密度上限。
LADDERS = {
    # 基准：只有密度约束（+ 应变能/应力作为"修补"约束始终在场），得到基准设计
    "bench": ["S0:T=1"],
    # ⚠ `--x0-npz` 给了种子时**不要**用 S0（T=1）——它会让转矩掉到局部解。
    # 今晚（2026-10-18）：基准 = 全局探索的种子（生产曲线 ≈52.9 N·m），
    # 直接进入"压脉振 → 抬转矩 → 压应力"。
    "night": ["S0:T=50,R=80", "S1:T=50,R=65", "S2:T=52,R=55",
              "S3:T=52,R=45", "S4:T=55,R=38", "S5:T=55,R=30"],
    # 脉振轴：先低转矩成形，再逐档抬 T̄、压脉振
    "rip": ["S0:T=1", "S1:T=40", "S2:T=50,R=10", "S3:T=60,R=8",
            "S4:T=70,R=6", "S5:T=70,R=5"],
    # 应力轴：应力上限从默认 40 MPa 下压（离心应力 ~28 MPa，留下的可动余量很小）
    "stress": ["S0:T=1", "S1:T=50", "S2:T=60,R=8", "S3:T=65,R=6,sigma=40e6",
               "S4:T=68,R=5.5,sigma=35e6", "S5:T=70,R=5,sigma=32e6"],
}


# ---------------------------------------------------------------------------
@dataclass
class Stage:
    """链上一档的约束集合。`relax` 是回退档（判据用 ρ_target<1）。

    ``role`` 是本档的**目标角色**（`objective_role`）：默认 ``con3`` = 最大化
    T_flux；``uelastic`` = 固定转矩下限下最小化弹性应变能（增刚度）。
    2026-10-20：用户要求按「平均转矩 → 应变能 → 脉振 → 应力」的顺序加约束，
    应力最后加（最严苛）。
    """

    name: str
    T: float = 1.0
    R: float = 1e9
    sigma: float = 1e30
    G: float = 1e30
    parent: Optional[str] = None
    relax: float = 1.0
    note: str = ""
    role: str = "con3"

    def values(self) -> Dict[str, float]:
        return dict(T=self.T, R=self.R, sigma=self.sigma, G=self.G,
                    relax=self.relax, role=self.role)

    def rip_on(self) -> bool:
        return self.R < 1e8

    def stress_on(self) -> bool:
        return self.sigma < 1e29

    def grid_on(self) -> bool:
        return self.G < 1e8

    def __repr__(self) -> str:
        return (f"{self.name}(T≥{self.T:g}, R≤{self.R:g}, σ≤{self.sigma:g}, "
                f"Ḡ≤{self.G:g}, ρ*={self.relax:g})")


_COARSE_TOP_CACHE: Dict[str, object] = {}


def _coarse_top(motor: str):
    """粗设计网格的拓扑（只建一次）。用于“细 FEM / 粗设计变量”。"""
    if motor not in _COARSE_TOP_CACHE:
        from motorfem.config import RunConfig, apply_mesh_config
        from motorfem.fem.mesh_builder import build_field_topology
        from motorfem.mesh import MotorMesh
        from motorfem.winding.layout import star_of_slots

        cfg = RunConfig()
        apply_mesh_config(cfg, str(MESHES[motor]))
        w = cfg.electrical.winding
        layout = None
        if getattr(w, "phase_from_generator", False):
            layout = star_of_slots(cfg.geometry.Z, cfg.geometry.pole_pairs, 3,
                                   parts=cfg.geometry.parts, layers=2,
                                   inner_ref=w.inner_ref, outer_ref=w.outer_ref,
                                   outer_shift=w.outer_shift)
        _COARSE_TOP_CACHE[motor] = build_field_topology(
            cfg, MotorMesh(str(MESHES[motor]), layout=layout))
    return _COARSE_TOP_CACHE[motor]


def _attach_coarse(fs, motor: Optional[str]) -> None:
    if not motor:
        return
    from motorfem.fem.two_level import attach_design_parent
    attach_design_parent(fs.top, _coarse_top(motor))


def seed_metrics(seeds_npz: str, key: str, *, motor: str, n_train: int,
                 filter_radius_mm: Optional[float] = None,
                 filter_grad_radius_mm: Optional[float] = None,
                 coarse_design_motor: Optional[str] = None) -> Dict[str, float]:
    """种子在 **11 点训练口径**下的实测（T_flux 均值与 T_ArkBE 的 p=10 范数）。

    ⚠ 滤波半径必须与后续 run 一致：否则种子的有效密度不同，T0/R0 会失真。
    """
    a = argparse.Namespace(motor=motor, csv=None, tag="seed", ladder=None, stages=None,
                           n_train=n_train, n_eval=1, init="legacy", init_seed=1002,
                           start="init", n_bar=4, duty=0.5, x0_npz=seeds_npz, x0_key=key,
                           mma_move=0.05, move_floor=0.005, move_decay=0.5,
                           max_eval_per_stage=1, n_min=0, window=1, advance_tol=0.02,
                           stall_tol=0.008, patience=1, relax_rho=0.85,
                           max_relax_back=0, strict=False, max_budget_shrinks=0,
                           wall_seconds=1, ckpt_every=1, jobs=1, out="/tmp/_seed")
    st = Stage("probe", T=1.0)
    cfg = build_config(st, motor=motor, n_train=n_train, n_eval=1, mma_move=0.05,
                       filter_radius_mm=filter_radius_mm,
                       filter_grad_radius_mm=filter_grad_radius_mm)
    fs = FieldSystem(cfg, str(MESHES[motor]),
                     str(CSV_PROD if a.csv is None else a.csv), init_seed=1002)
    _attach_coarse(fs, coarse_design_motor)
    with np.load(seeds_npz, allow_pickle=True) as z:
        x = np.asarray(z[key], float)
    rho = density_from_x(x, fs.top)
    call = fs.evaluate(rho, subset="train")
    # ⚠ R0 必须按**第一阶段会用的参考点**（T0−3）算，否则模板里的 `R0*1.3`
    #   与真正的 con1 不同量纲：绕自身均值算出的 R0≈2，而绕 T0−3 的真正值是
    #   ≈|3|+脉动 ⇒ 模板会给出"比现状还紧"的上限，第一档就不可行。
    m0 = metrics_of(call)
    m = metrics_of(call, m0["T_flux"] - 3.0)
    return dict(T0=m0["T_flux"], R0=m["ripple_arkbe"], S0=m["von_mises"],
                G0=m["rho_mean"])


def stage_spec_from_seeds(seeds_npz: str, keys, spec: str, *, motor: str,
                          n_train: int, n_eval: int,
                          filter_radius_mm: Optional[float] = None,
                          filter_grad_radius_mm: Optional[float] = None,
                          coarse_design_motor: Optional[str] = None) -> List[dict]:
    """把 ``--stage-spec``（占位符 T0/R0/S0/G0）按**每个种子自己的实测值**展开。"""
    with np.load(seeds_npz, allow_pickle=True) as z:
        ks = sorted(z.keys()) if not keys else (
            keys.split(",") if isinstance(keys, str) else list(keys))
    out = []
    for k in ks:
        m = seed_metrics(seeds_npz, k, motor=motor, n_train=n_train,
                         filter_radius_mm=filter_radius_mm,
                         filter_grad_radius_mm=filter_grad_radius_mm,
                         coarse_design_motor=coarse_design_motor)
        print(f"SEED {k:<32s} T0={m['T0']:9.4f} R0={m['R0']:8.4f}", flush=True)
        import re as _re
        env = {kk: float(vv) for kk, vv in m.items()}
        # ⚠ `str.format` 不解析算术（`{T0-3}` 会被当成键名 'T0-3'）⇒ 用 eval 带受限命名空间
        text = _re.sub(r"\{([^{}]+)\}",
                       lambda mm: f"{float(eval(mm.group(1), {'__builtins__': {}}, env)):g}",
                       spec)
        out.append(dict(tag=k, x0_key=k, stages=text, n_train=n_train, n_eval=n_eval,
                        motor=motor, T0=m["T0"], R0=m["R0"]))
    return out


def parse_stages(spec: str) -> List[Stage]:
    """``S0:T=52,R=30|S1:T=55,R=25`` → 阶段链。

    ⚠ 档间分隔符**同时接受 ``|`` 与 ``;``**：命令行里 ``;`` 会被 Git Bash 当成
    命令分隔符（实测 2026-10-18 崩了一轮），所以 `--stage-spec-file` 用 ``;``
    写在文件里；两种写法解析结果一致。
    """
    out: List[Stage] = []
    for seg in spec.replace(";", "|").split("|"):
        head, _, rest = seg.partition(":")
        kw: Dict[str, object] = {}
        for item in rest.split(","):
            if not item.strip():
                continue
            k, sep, v = item.partition("=")
            if not sep:
                raise ValueError(f"阶段参数应写成 k=v：{item!r}（整段 {rest!r}）")
            k = {"T": "T", "R": "R", "sigma": "sigma", "G": "G"}.get(k.strip(), k)
            k = str(k).strip()
            # role / name 是字符串；其余是数值
            kw[k] = v if k in ("name", "role") else float(v)
        out.append(Stage(head.strip(), **kw))
    return out


def build_config(st: Stage, *, motor: str, n_train: int, n_eval: int,
                 mma_move: float,
                 filter_radius_mm: Optional[float] = None,
                 filter_grad_radius_mm: Optional[float] = None,
                 stress_filter_mm: Optional[float] = None) -> RunConfig:
    """一档的 `RunConfig`。`FieldSystem` 只建一次；档间只改 objective 的值。

    ``filter_radius_mm`` / ``filter_grad_radius_mm``（2026-10-19 新增，
    默认 None ⇒ 行为逐位不变）：覆盖**密度滤波**与**梯度滤波**半径。
    加密网格上必须按单元尺寸同时缩小两者：否则 0.5 mm 在 d030 上覆盖约
    1.3 个单元（6s4p 只有 0.64 个），等效平滑更强、长跑会退化。
    """
    c = RunConfig()
    apply_mesh_config(c, str(MESHES[motor]))
    if filter_radius_mm is not None:
        r = float(filter_radius_mm)
        if not (0.0 < r <= 10.0):
            raise ValueError(
                f"--filter-radius-mm 必须在 (0, 10] 内，收到 {r}；"
                "半径 0 会让滤波器行归一化除零（本网格 0.2 mm 已等价于无滤波）")
        c.filter.filter_radius_m = r * 1e-3
    if filter_grad_radius_mm is not None:
        r = float(filter_grad_radius_mm)
        if not (0.0 < r <= 10.0):
            raise ValueError(
                f"--filter-grad-radius-mm 必须在 (0, 10] 内，收到 {r}")
        c.filter.filter_grad_radius_m = r * 1e-3
    if stress_filter_mm is not None:
        r = float(stress_filter_mm)
        if r < 0.0 or r > 10.0:
            raise ValueError(f"--stress-filter-mm 必须在 [0, 10] 内，收到 {r}")
        c.stress.stress_filter_radius_m = r * 1e-3
    oc = c.objective
    oc.objective_role = "con3"          # 目标 = 最大化平均转矩（底座）
    oc.mean_torque_source = "flux"
    oc.ripple_source = "arkbe"
    oc.mean_torque_obj = st.T           # con3 的分母 = 严格度里的 R
    oc.max_torque_ripple = st.R
    # ⚠ **应变能不能作约束**（2026-10-18 实测）：`objective.fun` 把
    #   ``g_Uelastic = elastic_strain_energy``（恒正的原始能量）当成 ``g ≤ 0``，
    #   于是它**恒不可行** ⇒ MMA 会被逼着把 U 压到 0（转子做空/做软）。
    #   legacy 里 uelastic 只在**组合目标**里当目标基座，从不作约束。
    #   本脚本的目标是 con3，机械项只保留**应力 AL 罚**（它确实是 ≤0 形式）。
    oc.enable_uelastic = False
    oc.enable_stress = True
    oc.enable_con1 = st.rip_on()
    oc.enable_con2 = True
    oc.enable_con3 = True
    oc.enable_con4 = st.rip_on()
    oc.constraint_schedule = False       # 本脚本自己调度
    oc.objective_role_phase2 = None
    oc.p_exp_continuation = oc.p_exp_move_linkage = False
    oc.p_exp_torque_ripple = 10
    c.topology.max_density = 0.65 if not st.grid_on() else st.G
    # ⚠ 未在链上给 σ 时**必须放开**，不能落回生产的 40 MPa：
    #   种子是"纯转矩"优化出来的（实测 von Mises ~76 MPa），一上来套 40 MPa 的
    #   AL 罚项会把转矩打崩（实测 obj 92 ⇒ T̄≈4 N·m）。σ 也要作为一档进梯子。
    c.stress.max_rel_stress_constraint_pa = (1e9 if not st.stress_on() else st.sigma)
    c.time.sampling_mode = "uniform"
    c.time.N_point_simulation = int(n_train)
    c.time.use_validation_set = True
    c.time.n_validation_points = int(n_eval)
    c.mma.MMA_move = float(mma_move)
    c.runtime.n_jobs = 1
    return c


def make_start(top, mode: str, n_bar: int = 4, duty: float = 0.5) -> np.ndarray:
    """**纯几何**的起点设计（不依赖任何先验优化结果）。

    动机（2026-10-18）：「让优化自己找到基底」在 6s4p 上不成立 —— 从实心起点
    出发的纯密度阶段只把转矩推到 ~10 N·m 就停住（局部解）。用已知有效的拓扑
    （平行磁障）做**基准**，再由调度逐档加严约束；优化器负责精修细节。

    mode
    ----
    ``solid``  全实心（legacy 初值那种"随便找个点"）
    ``bars``   沿 x 等分 ``n_bar`` 条磁障，在"轴↔外缘"之间交替开/闭。
               ``k`` 偶数的条是**空气**（ρ_low），奇数条是**导磁**（ρ_high）。
               每条从 ``x>0``（朝轴）或 ``x<0``（朝外缘）单端锚定，另一端留
               小间隙 ⇒ 磁障**不形成闭槽**，磁通可从轴的半圆绕过。
    ``bars2``  同上但锚定端交替（更接近 SynRM 的"相邻条交替锚定"）。

    返回 ``(N_topo,)`` 的**设计变量**（滤波前像不做 —— 滤波会把 0/1 抹成灰度，
    这在这里是想要的：让 MMA 一开始就在一个平滑的拓扑附近）。
    """
    import numpy as np
    topo = list(top.element_topo_opti_m)
    if mode == "solid":
        return np.ones(len(topo))
    eS = np.asarray(top.node_post, float)
    xy = eS[:, 1:3]
    tri = np.array([np.asarray(top.element_S_post)[e][1:4] for e in topo], dtype=int)
    cx = xy[tri].mean(axis=1)          # (N_topo, 2) 单元中心
    x, y = cx[:, 0], cx[:, 1]
    r = np.hypot(x, y)
    R_in = float(top.cfg.geometry.shaft_radius_m)
    R_out = float(top.cfg.geometry.rotor_outer_radius_m)
    s = (r - R_in) / max(R_out - R_in, 1e-12)          # 0..1 径向归一化
    xs = (x - x.min()) / max(x.max() - x.min(), 1e-12)  # 0..1 横向
    k = np.minimum((xs * n_bar).astype(int), n_bar - 1)  # 第几根条
    air = (k % 2 == 1)                                  # 偶数条 = 空气
    if mode == "bars2":
        anchor_out = (k % 4 < 2)                        # 交替锚定方向
    else:
        anchor_out = np.zeros_like(air)
    # 锚定端 → 闭；自由端 → 在 duty 处切断，留 15% 缝
    free_zone = np.where(anchor_out, s < (1 - duty), s > duty)
    closed = air & ~free_zone
    rho = np.where(closed, 0.05, 1.0)
    rho = np.where(air, np.where(closed, 0.05, 0.6), rho)
    return rho.astype(float)


# ---------------------------------------------------------------------------
def metrics_of(call, T_obj: Optional[float] = None) -> Dict[str, float]:
    """一次评价的物理量（严格度与候选打分都用它）。

    Parameters
    ----------
    T_obj : float | None
        纹波的参考点。**生产 con1 是 ``mean_torque_obj``**
        （``torque_ripple_computed_around_obj_torque=True``）⇒ 调用方传本档的
        ``st.T``。``None`` ⇒ 用该向量自身的均值（等价于标准差式的脉动，
        适合"还不知道本档目标"的场合，如种子标定与探索）。
    """
    tf = np.asarray(call.torque_flux_energy_vec, float)
    ta = np.asarray(call.torque_ArkBE_vec, float)
    tv = np.asarray(call.torque_VD_vec, float)
    rho = np.asarray(call.rho_SIMP_mech, float)
    r0 = float(np.mean(ta)) if T_obj is None else float(T_obj)
    r1 = float(np.mean(tf)) if T_obj is None else float(T_obj)
    r2 = float(np.mean(tv)) if T_obj is None else float(T_obj)

    def pnorm(v, ref, p=10.0):
        return float(np.mean(np.abs(v - ref) ** p) ** (1.0 / p))

    return dict(
        T_flux=float(np.mean(tf)), T_ArkBE=float(np.mean(ta)), T_VD=float(np.mean(tv)),
        ripple_arkbe=pnorm(ta, r0), ripple_flux=pnorm(tf, r1), ripple_vd=pnorm(tv, r2),
        arkbe_vec=[float(v) for v in ta], flux_vec=[float(v) for v in tf],
        von_mises=float(call.von_mises_max),
        uelastic=float(call.uelastic),
        stress_AL=float(call.stress_loss_al),
        rho_mean=float(np.mean(rho)),
    )


def strictness(m: Dict[str, float], st: Stage) -> Dict[str, float]:
    """严格度 ρ —— **每一维都统一成「越大越好」，1 ⇔ 恰好落在本档硬指标上**。

    ⚠ 上限型约束（脉振 R、应力 σ、密度 Ḡ）必须取**倒数的方向**
    （``目标 / 实际``），不能写 ``实际 / 目标``。2026-10-18 写成后者时：
    脉振恶化 ⇒ ρ 变大 ⇒ 判"更严格 ⇒ 可晋级"，而脉振变好 ⇒ ρ 变小 ⇒ 永远
    晋级不了 —— 判据与物理**完全反向**，整条链会卡在第一档。
    """
    r = {"T": m["T_flux"] / st.T}                       # 下限型：越大越好
    if st.rip_on():
        r["R"] = st.R / max(m["ripple_arkbe"], 1e-30)   # 上限型：越小越好 ⇒ 取倒数
    if st.stress_on():
        r["sigma"] = st.sigma / max(m["von_mises"], 1e-30)
    if st.grid_on():
        r["G"] = st.G / max(m["rho_mean"], 1e-30)
    return r


def score(m: Dict[str, float], st: Stage) -> float:
    """加严得分 = 最严格的那一维（瓶颈）。1 ⇔ 恰好落在本档的全部指标上。"""
    return float(min(strictness(m, st).values()))


def is_feasible(m: Dict[str, float], st: Stage) -> bool:
    """**严格可行**（与 `objective.py` 的 g≤0 同义，仅留 1e-9 浮点余量）。"""
    eps = 1e-9
    if m["T_flux"] < st.T * (1 - eps):
        return False
    if st.rip_on() and m["ripple_arkbe"] > st.R * (1 + eps):
        return False
    if st.stress_on() and m["von_mises"] > st.sigma * (1 + eps):
        return False
    if st.grid_on() and m["rho_mean"] > st.G * (1 + eps):
        return False
    return True


# ---------------------------------------------------------------------------
class _StopStage(Exception):
    def __init__(self, verdict: str):
        super().__init__(verdict)
        self.verdict = verdict


class Run:
    def __init__(self, args, chain: List[Stage], log: RunLog):
        self.a = args
        self.chain = chain
        self.log = log
        self.out = Path(args.out)
        self.pool: List[dict] = []
        self.fs: Optional[FieldSystem] = None
        self.cfg: Optional[RunConfig] = None
        self.n_eval = 0
        self.stage_log: List[dict] = []
        self.x = None

    # ------------------------------------------------------------ 阶段
    def _enter(self, st: Stage, move: float) -> None:
        """进入（或重进）本档：新建/复用 `FieldSystem`，只改 objective 的值。"""
        a = self.a
        if self.fs is None:
            # ⚠ ``getattr`` 是必需的：``run_pareto_grid.py`` 构造的 Namespace
            #   没有新字段（它 monkey-patch 了 build_config 后直接跑 Run）。
            self.cfg = build_config(st, motor=a.motor, n_train=a.n_train,
                                    n_eval=a.n_eval, mma_move=move,
                                    filter_radius_mm=getattr(
                                        a, "filter_radius_mm", None),
                                    filter_grad_radius_mm=getattr(
                                        a, "filter_grad_radius_mm", None),
                                    stress_filter_mm=getattr(
                                        a, "stress_filter_mm", None))
            csv = str(CSV_PROD if a.csv is None else a.csv)
            self.fs = FieldSystem(self.cfg, str(MESHES[a.motor]), csv,
                                  init_seed=a.init_seed,
                                  init_spec=None if a.init == "legacy"
                                  else {"mode": a.init, "seed": a.init_seed})
            _attach_coarse(self.fs, getattr(a, "coarse_design_motor", None))
            self.x = self._load_x0()
            # 池里放一个起点冷评（后面的"严格可行"判定要用独立值）
            self._cold(self.x, "start", st)
        oc = self.cfg.objective
        oc.mean_torque_obj = st.T
        oc.max_torque_ripple = st.R
        oc.enable_con1 = st.rip_on()
        oc.enable_con4 = st.rip_on()
        # 目标角色按档切换（2026-10-20）：con3 = 最大化 T_flux；
        # uelastic = 在 T≥T_obj 下最小化弹性应变能。⚠ uelastic **只能当目标**：
        # 当约束时 g=U≥0 恒不可行（见 build_config 的注释），会把转子做空。
        oc.objective_role = st.role
        oc.enable_uelastic = (st.role == "uelastic")
        oc.enable_con3 = True
        # 密度的 con2 上限：阶段给了 G 时以 G 为准；否则用 CLI 覆盖（手绘 GUI
        # 常用），再否则保持生产 0.65。getattr 同样兼容 pareto 的 Namespace。
        _max_d = getattr(a, "max_density", None)
        self.cfg.topology.max_density = (
            st.G if st.grid_on()
            else (0.65 if _max_d is None else float(_max_d)))
        self.cfg.stress.max_rel_stress_constraint_pa = (1e9 if not st.stress_on()
                                                        else st.sigma)
        self.cfg.mma.MMA_move = float(move)
        # ⚠ `flags` 在 `FieldSystem.__init__` 里从 cfg 快照一次 ⇒ 改 σ 必须重建，
        #   否则本档的应力上限**根本不生效**（静默用档外的旧值）。
        from motorfem.fem.mech_worker import stress_flags_from_config
        self.fs.flags = stress_flags_from_config(self.cfg)
        self._row_names = f_row_names(self.cfg, st.role)   # f[1:] 的量名（日志/落盘用）
        self.al_at_entry = self.fs.al_state      # 档间**保留** AL（μ 是纯罚强度）
        self.pool_at_entry = len(self.pool)

    def _load_x0(self) -> np.ndarray:
        a = self.a
        topo = list(self.fs.top.element_topo_opti_m)
        n_design = int(getattr(self.fs.top, "n_design", None) or len(topo))
        if a.x0_npz:
            with np.load(a.x0_npz, allow_pickle=True) as z:
                # ⚠ 不能按后缀猜（种子文件里的键就是设计名，如 mu1_b0.85_s29）：
                #   一个文件装多个种子时必须显式给 --x0-key。
                if a.x0_key:
                    key = a.x0_key
                else:
                    cand = [k for k in z.keys()
                            if k.endswith("x_selected") or k.endswith("x_best_feasible")
                            or k.endswith("x_best") or k == "x"]
                    if len(cand) != 1:
                        raise ValueError(
                            f"{a.x0_npz} 里有 {len(z.keys())} 个键、无法唯一确定初值，"
                            f"请用 --x0-key 指定（候选：{sorted(z.keys())[:6]}…）")
                    key = cand[0]
                x = np.asarray(z[key], float)
        elif a.start != "init":
            x = make_start(self.fs.top, a.start, n_bar=a.n_bar, duty=a.duty)
        else:
            # ⚠ **`rho_SIMP_initial` 是密度，不是设计变量**（2026-10-18 实测）：
            #   它是 `filter_inverse_rho` 的**返回值**，但不满足 `ρ = filter_SIMP@x`
            #   （MSE 0.131，而 `density_from_x(rho_SIMP_initial)` 的 ρ̄=0.874 而非
            #   0.777），当 x 传进去密度约束会直接飙到 300+。它的用法是
            #   `initial_state["x"]`（MMA 续跑语义）——那个路径是另一个问题，
            #   这里按 v7「先跑起来」的原则**取逆投影**：
            x = filter_inverse_rho(self.fs.top.filter_SIMP, np.asarray(
                self.fs.top.rho_SIMP_initial, float), topo, min_density=1e-6)[topo]
        if n_design != len(topo):
            # 两级网格：把细单元上的初值压成粗设计变量（父单元取均值）
            from motorfem.fem.two_level import compress_initial
            full = np.zeros(int(self.fs.top.n_element))
            full[np.asarray(topo, int)] = x
            x = compress_initial(self.fs.top, full)
        if x.shape != (n_design,):
            raise ValueError(f"初值维度不符：{x.shape} vs {(n_design,)}"
                             f"（细设计单元 {len(topo)}，设计变量 {n_design}）")
        return x

    def _cold(self, x: np.ndarray, where: str, st: Stage) -> Dict[str, float]:
        rho = density_from_x(np.asarray(x, float), self.fs.top)
        call = self.fs.evaluate(rho, subset="train")
        m = metrics_of(call, st.T)
        self.pool.append(dict(x=np.array(x, copy=True), m=m, where=where))
        return m

    # ------------------------------------------------------------ 一步
    def _step(self, st: Stage, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        f, df = self.fs.warper(x)
        m = metrics_of(self.fs.trace[-1][1], st.T)
        sec = time.perf_counter() - t0
        self.fs.trace.clear()
        self.n_eval += 1
        self.pool.append(dict(x=np.array(x, copy=True), m=dict(m), where=st.name))
        cons = [None, None, None, None]
        for i, name in enumerate(self._row_names, start=1):
            s = SLOT.get(name)
            if s:
                cons[s - 1] = float(f[i])
        self.log.eval(len(self.stage_log) + 1, self.n_eval, float(f[0]), sec, cons)
        r = strictness(m, st)
        print(f"  [{self.n_eval:4d}] ρ={min(r.values()):.3f} T̄={m['T_flux']:7.3f} "
              f"R={m['ripple_arkbe']:6.2f}/{st.R:g} σ={m['von_mises']/1e6:6.2f}MPa "
              f"g={np.round(f[1:], 4).tolist()} {sec:.1f}s", flush=True)
        if self.n_eval % self.a.ckpt_every == 0:
            p = self._save(st, x)
            self.log.checkpoint(p.name, self.n_eval)
        return f, df, m

    # ------------------------------------------------------------ 一档
    def _run_stage(self, st: Stage, move: float) -> str:
        """跑本档直到 advance / stalled / hit_budget。"""
        a = self.a
        self.n_stage = 0
        rho_hist: List[float] = []
        T_hist: List[float] = []
        verdict = "continue"
        grab = _StateGrab()

        def wrapped(xv):
            nonlocal verdict
            f, df, m = self._step(st, xv)
            self.n_stage += 1
            rho_hist.append(min(strictness(m, st).values()))
            T_hist.append(m["T_flux"])
            verdict = self._judge(st, rho_hist, T_hist)
            if verdict != "continue":
                grab.on_step(self.n_stage, dict(x=np.copy(xv)))   # 末步也入池
                raise _StopStage(verdict)
            return f, df

        # MMA 状态：换档时从池里最好点**重新开始**（渐近线按新 x 重建），
        # 但 A 热启动缓存与 AL 状态随 `self.fs` 延续。
        init = None
        if getattr(self, "_mma_state", None) is not None and self._mma_from_stage == st.name:
            init = self._mma_state
        x0 = self.x if init is None else np.asarray(init["x"], float)
        try:
            optimize_mma(wrapped, x0, np.zeros_like(x0), np.ones_like(x0),
                         max_step=a.max_eval_per_stage + 2, mma_cfg=self.cfg.mma,
                         control=grab, move_at=lambda i: move,
                         f_rel_change=-1, x_rel_change=-1, initial_state=init)
            verdict = "hit_budget"
        except _StopStage as e:
            verdict = e.verdict
        self._mma_state = grab.last
        self._mma_from_stage = st.name
        return verdict

    def _judge(self, st: Stage, rho_hist, T_hist) -> str:
        a = self.a
        if self.n_stage < a.n_min:
            return "continue"
        nw = min(len(rho_hist), a.window)
        r_now = float(np.mean(rho_hist[-nw:]))
        if r_now >= st.relax * (1 - a.advance_tol):
            return "advance"
        if len(rho_hist) >= a.patience:
            rec = rho_hist[-a.patience:]
            tw = T_hist[-a.patience:]
            if (max(rec) - min(rec) < a.stall_tol
                    and abs(tw[-1] - tw[0]) / (abs(tw[-1]) + 1e-20) < 1e-3):
                return "stalled"
        if self.n_stage >= a.max_eval_per_stage:
            return "hit_budget"
        return "continue"

    # ------------------------------------------------------------ 候选与写盘
    def _best(self, st: Stage) -> Tuple[np.ndarray, Dict[str, float], dict]:
        best, key = None, None
        for p in self.pool[self.pool_at_entry:]:
            k = (1 if is_feasible(p["m"], st) else 0, score(p["m"], st))
            if key is None or k > key:
                best, key = p, k
        if best is None:
            best = self.pool[-1]
        return np.array(best["x"], copy=True), best["m"], best

    def _save(self, st: Stage, x: np.ndarray) -> Path:
        path = Path(str(self.out) + f"_stage_{st.name}.npz")
        tmp = Path(str(path) + ".tmp")
        xs = np.asarray([p["x"] for p in self.pool])
        ms = [p["m"] for p in self.pool]
        with tmp.open("wb") as f:
            np.savez_compressed(
                f, x_calls=xs, config_json=json.dumps(self.cfg.to_dict()),
                labels=np.asarray(["objective", *self._row_names]),
                metrics_json=json.dumps(ms), where=np.asarray([p["where"] for p in self.pool]),
                x_best=np.asarray(x), stage=st.name, stage_values=json.dumps(st.values()),
                mesh_path=str(MESHES[self.a.motor]),
                csv_path=str(CSV_PROD if self.a.csv is None else self.a.csv),
                mesh_sha256=hashlib.sha256(
                    MESHES[self.a.motor].read_bytes()).hexdigest(),
                al_mu=float(self.fs.al_state.al_mu))
        os.replace(tmp, path)
        return path

    # ------------------------------------------------------------ 主线
    def run(self) -> dict:
        a = self.a
        st = self.chain[0]
        self._enter(st, a.mma_move)
        self._cur_name = st.name      # ⚠ 必须设：否则主循环首轮会**重进**本档（再冷评一次）
        self._mma_state = None
        print(f"\n=== {st} move={a.mma_move:g} ===", flush=True)
        self.log.line(f"STAGE {st.name} {st.values()}")
        i = 0
        move = a.mma_move
        relax_stack: List[Stage] = []      # 回退用（按需压栈）
        consecutive = 0                    # 连续回退次数
        self._budget_shrinks = 0           # 本档因预算用尽而缩移动限的次数（有上限，防死循环）
        t0 = time.time()
        while i < len(self.chain) and time.time() - t0 < a.wall_seconds:
            st = self.chain[i]
            if st.name != getattr(self, "_cur_name", None):
                self._enter(st, move)
                self._cur_name = st.name
                self._mma_state = None
                self.n_stage = 0
                self._budget_shrinks = 0
            v = self._run_stage(st, move)
            x, m, p = self._best(st)
            self.x = x
            rec = dict(stage=st.name, values=st.values(), verdict=v, move=move,
                       n_stage=self.n_stage, n_total=self.n_eval,
                       best=m, best_where=p["where"],
                       feasible=is_feasible(m, st), elapsed=time.time() - t0)
            self.stage_log.append(rec)
            print(f"--- {st.name}: {v} (evals={self.n_stage}) "
                  f"feasible={rec['feasible']} T̄={m['T_flux']:.3f} "
                  f"R={m['ripple_arkbe']:.2f} σ={m['von_mises']/1e6:.2f}MPa "
                  f"ρ̄={m['rho_mean']:.3f}", flush=True)
            self._save(st, x)
            # 校验（独立空缓存冷算，验证集另算）。⚠ 这里**不能带 v==... 的条件**：
            # 测试里 max_eval_per_stage=3 时永远到不了 advance，流程分叉压根没被覆盖过。
            val = self._validate(x, st)
            rec["validate"] = val
            print(f"    validate(6pt) {json.dumps({k: round(v, 4) for k, v in val.items()})}",
                  flush=True)
            if v == "advance":
                consecutive = 0
                relax_stack.clear()
                i += 1
                move = a.mma_move
            elif v in ("stalled", "hit_budget"):
                # 预算用尽 ≠ 停滞：只有"判据标定的停滞"才允许放松；预算用尽只缩移动限。
                if a.strict or v == "hit_budget":
                    if move > a.move_floor * 1.0001 and self._budget_shrinks < a.max_budget_shrinks:
                        self._budget_shrinks += 1
                        move = max(a.move_floor, move * a.move_decay)
                        self._mma_state = None
                        print(f"    (budget) 缩小移动限 → {move:g}（{self._budget_shrinks}/"
                              f"{a.max_budget_shrinks}），重跑本档", flush=True)
                        continue
                # 放松一档：脉振上限 × ripple_relax（如有），否则降 T̄ 下限
                relaxed = self._relax(st)
                body = Stage(f"{st.name}_r{len(relax_stack) + 1}", parent=st.name,
                             relax=a.relax_rho,
                             **{k: getattr(relaxed, k)
                                for k in ("T", "R", "sigma", "G", "role")})
                print(f"    停滞 → 放松为 {body}（父档 {st.name}）", flush=True)
                relax_stack.append(st)
                if len(relax_stack) > a.max_relax_back:
                    print("STOP: 连续放松已达上限", flush=True)
                    self.log.stopped("连续放松达上限")
                    break
                self._enter(body, move * a.move_decay)
                self._cur_name = body.name
                self._mma_state = None
                self.n_stage = 0
                self._budget_shrinks = 0
                self.chain.insert(i + 1, st)      # 放松档跑完后重试原档
                self.chain.insert(i, body)
                self.stage_log[-1]["relaxed_into"] = body.name
                continue
            if v == "advance":
                consecutive = 0
        if i >= len(self.chain):
            self.log.complete()
        return self._summary(t0)

    def _relax(self, st: Stage) -> Stage:
        """放松一档：先抬脉振上限（×1.5），再降 T̄ 下限（×0.9）；Δσ 同理。"""
        T, R, S = st.T, st.R, st.sigma
        if st.rip_on():
            R = R * 1.5
        elif S < 1e29:
            S = S * 1.1
        else:
            T = T * 0.9
        return Stage(st.name, T=T, R=R, sigma=S, G=st.G, role=st.role)

    def _validate(self, x, st: Stage) -> Dict[str, float]:
        rho = density_from_x(np.asarray(x, float), self.fs.top)
        call = self.fs.evaluate(rho, subset="val")
        m = metrics_of(call, st.T)
        return dict(T_flux=m["T_flux"], T_ArkBE=m["T_ArkBE"],
                    ripple_arkbe=m["ripple_arkbe"], von_mises=m["von_mises"],
                    rho_mean=m["rho_mean"])

    def _summary(self, t0) -> dict:
        pts = [p for p in self.pool if np.isfinite(p["m"]["T_flux"])]
        front = _pareto([p["m"] for p in pts])
        out = dict(chain=[c.values() for c in self.chain], stages=self.stage_log,
                   n_evals=self.n_eval, pool=len(self.pool), pareto=front,
                   elapsed_s=time.time() - t0,
                   config_json=self.cfg.to_dict() if self.cfg else None)
        Path(str(self.out) + "_summary.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8")
        return out


def _pareto(ms: List[Dict[str, float]]) -> List[dict]:
    """(T↑, R↓, σ↓, Ḡ↓) 非支配；池小，O(n²) 足够。"""
    keep = []
    for a in ms:
        dom = False
        for b in ms:
            if b is a:
                continue
            if (b["T_flux"] >= a["T_flux"] and b["ripple_arkbe"] <= a["ripple_arkbe"]
                    and b["von_mises"] <= a["von_mises"] and b["rho_mean"] <= a["rho_mean"]
                    and (b["T_flux"] > a["T_flux"]
                         or b["ripple_arkbe"] < a["ripple_arkbe"]
                         or b["von_mises"] < a["von_mises"]
                         or b["rho_mean"] < a["rho_mean"])):
                dom = True
                break
        if not dom:
            keep.append({k: float(a[k]) for k in
                         ("T_flux", "ripple_arkbe", "von_mises", "rho_mean",
                          "uelastic", "T_ArkBE")})
    keep.sort(key=lambda d: -d["T_flux"])
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motor", choices=sorted(MESHES), default="6s4p")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--ladder", choices=sorted(LADDERS), default="rip")
    ap.add_argument("--stages", default=None, help="覆盖 --ladder 的自定义阶段链")
    ap.add_argument("--n-train", type=int, default=11)
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--init", default="legacy", choices=["legacy", "grf", "binary", "blobs"])
    ap.add_argument("--init-seed", type=int, default=1002)
    ap.add_argument("--start", default="init",
                    choices=["init", "solid", "bars", "bars2"],
                    help="起点：init=legacy/init_spec；solid=全实心；bars=等宽平行磁障（纯几何）")
    ap.add_argument("--n-bar", type=int, default=4, help="bars 起点的条数")
    ap.add_argument("--duty", type=float, default=0.5, help="bars 起点磁障占径向的比例")
    ap.add_argument("--x0-npz", default=None)
    ap.add_argument("--x0-key", default=None)
    ap.add_argument("--filter-radius-mm", type=float, default=None,
                    help="覆盖密度滤波半径（mm）；默认 None = 生产 2.0 mm。"
                         "半径必须 >0（0 会让滤波器行归一化除零）。")
    ap.add_argument("--filter-grad-radius-mm", type=float, default=None,
                    help="覆盖梯度滤波半径（mm）；默认 None = 生产 0.4 mm。"
                         "加密网格上应与密度滤波一起按单元尺寸缩小。")
    ap.add_argument("--max-density", type=float, default=None,
                    help="con2 的平均密度上限；默认 None = 0.65。"
                         "阶段给了 G 时以 G 为准。")
    ap.add_argument("--mma-move", type=float, default=0.05)
    ap.add_argument("--move-floor", type=float, default=0.005)
    ap.add_argument("--move-decay", type=float, default=0.5)
    ap.add_argument("--max-eval-per-stage", type=int, default=100)
    ap.add_argument("--n-min", type=int, default=10)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--advance-tol", type=float, default=0.02)
    ap.add_argument("--stall-tol", type=float, default=0.008)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--relax-rho", type=float, default=0.85,
                    help="放松档的判据阈值（ρ≥relax_rho 即算过）")
    ap.add_argument("--max-relax-back", type=int, default=3)
    ap.add_argument("--max-budget-shrinks", type=int, default=2,
                    help="同一档因评价预算用尽而缩小移动限的最多次数（防死循环）")
    ap.add_argument("--strict", action="store_true", help="卡住只缩移动限，不放松")
    ap.add_argument("--wall-seconds", type=float, default=3.5 * 3600)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--stress-filter-mm", type=float, default=None,
                    help="应力平滑半径（mm）：把逐单元 σ̄ 平滑后再进 AL 约束；"
                         "默认 None/0 = 关闭（逐位不变）")
    ap.add_argument("--coarse-design-motor", default=None,
                    help="两级网格：FEM 用 --motor 的细网格，设计变量取该电机的粗设计"
                         "单元（如 6s4p）。默认 None = 单级（行为不变）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.jobs == 1, "本脚本只跑单条；并行由 run_pareto_grid.py 编排"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    spec = a.stages or "|".join(LADDERS[a.ladder])
    chain = parse_stages(spec)
    print(f"tag={a.tag} chain={[str(s) for s in chain]}", flush=True)
    log = RunLog(str(a.out) + ".log", target=a.max_eval_per_stage)
    log.labels({"con1": "stress", "con2": "density", "con3": "con3_flux", "con4": "con1_arkbe"})
    try:
        s = Run(a, chain, log).run()
        print(f"COMPLETE tag={a.tag} evals={s['n_evals']} {s['elapsed_s']:.0f}s",
              flush=True)
        return 0
    except BaseException:                                   # noqa: BLE001
        log.failed(traceback.format_exc()[-800:])
        raise


if __name__ == "__main__":
    sys.exit(main())
