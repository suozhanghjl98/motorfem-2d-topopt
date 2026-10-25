"""阶段 4D：目标聚合（``obj_func_parallel_AL_v2.__call__`` 的聚合段）。

对应 legacy ``nonlinear_solver_energy_MMA10.py:3575-3752``（下称 MMA10）：

- 平均转矩（Arkkio，生产用）、转矩波动（p=10，绕 ``mean_torque_obj``）、
  密度约束（多项式）、应力约束（AL）、弹性应变能；
- ``f = [obj, con1..4]``、``df = [dobj, dcon1..4]``、``AL_stress_lambda_new``、
  ``detail`` dict —— 与 ``:3750-3752`` 的返回逐一对应。

**本模块不产生新数值**：EM 段（``em_worker.em_segment``）与力学段
（``mech_worker.mech_worker``）都是已逐位验证的内核/驱动层移植；这里只把
legacy 的标量聚合算式照搬。``L4_snapshot`` 把一次 MMA 轨迹（3 次 fun 调用 +
AL 状态）铺成 golden L4 命名的快照。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..kernels.reference.al import ALState
from ..kernels.reference.mech_stress import StressConstraintFlags
from ..optimization.published_layout import assert_published_layout
from .em_worker import em_segment
from .mesh_builder import FieldTopology
from .mech_worker import mech_worker
from .preassembly import Preassembly

__all__ = ["CallResult", "NullMechResult", "fun", "L4_snapshot"]


@dataclass
class NullMechResult:
    """跳过力学段时的占位 ``mres``（``enable_uelastic & enable_stress`` 都关）。

    只提供 ``fun`` 聚合段消费的力学字段：全零，形状与真实 ``MechStressResult``
    一致（``objective.fun`` 只读这些键，见 ``objective.fun`` 力学段）。
    """

    n_elem_mech: int

    @property
    def stress_loss_al(self) -> float:
        return 0.0

    @property
    def elastic_strain_energy(self) -> float:
        return 0.0

    @property
    def d_al_stress_p_drho(self) -> np.ndarray:
        return np.zeros(self.n_elem_mech)

    @property
    def d_u_p_drho(self) -> np.ndarray:
        return np.zeros(self.n_elem_mech)

    @property
    def al_lambda_new(self) -> np.ndarray:
        return np.zeros(self.n_elem_mech)

    @property
    def von_mises_max(self) -> float:
        return 0.0

    @property
    def von_mises(self) -> np.ndarray:
        # `fun` 的聚合段无条件读 `mres.von_mises`（写进 CallResult.von_mises_elem），
        # 跳过力学时必须给同形状的全零数组，否则 AttributeError。
        return np.zeros(self.n_elem_mech)


@dataclass
class CallResult:
    """一次 fun(x) 调用的全部产出（golden L4 的 call_{i}_* 与 mech_{i}_* 的来源）。"""

    f: np.ndarray                  # (size,)  [obj, con1, con2, con3, con4]
    df: np.ndarray                 # (size, N_element) 对应的梯度
    al_lambda_new: np.ndarray      # (N_elem_mech,) AL 乘子写回（D7 下恒为 0）
    detail: Dict[str, object]
    # ---- EM 向量（golden call_{i}_*_vec）----
    torque_Arkkio_vec: np.ndarray        # (11,)
    dTArkkio_p_drho_vec: np.ndarray      # (11, N_element)
    flux_linkage_energy_vec: np.ndarray  # (11, 2)
    torque_flux_energy_vec: np.ndarray   # (11,)
    dTflux_p_drho_vec: np.ndarray        # (11, N_element)
    torque_VD_vec: np.ndarray            # (11,)
    dTVD_p_drho_vec: np.ndarray          # (11, N_element)
    # ★ 2026-10-14 新增：体边积分 Maxwell 应力（T_ArkBE）—— 生产默认的**纹波口径**。
    #   ⚠ **故意不进 `L4_snapshot.final_vecs` / `detail`**：那两个是 golden 契约
    #     （多一个键会被 `verify_golden` 报"多出"）。这里只作为 `CallResult` 字段
    #     供报告/诊断使用。
    torque_ArkBE_vec: np.ndarray         # (11,)
    dTArkBE_p_drho_vec: np.ndarray       # (11, N_element)
    # ---- 力学（golden call_{i}_*）----
    rho_SIMP_mech: np.ndarray            # (N_elem_mech,)
    u_node: np.ndarray                   # (2·N_node_mech,)
    von_mises_max: float
    stress_loss_al: float
    d_al_stress_p_drho: np.ndarray       # (N_elem_mech,)
    uelastic: float
    d_u_p_drho: np.ndarray               # (N_elem_mech,)
    #: 逐力学单元的 von Mises（诊断/网格敏感性研究用；不参与 golden 快照）
    von_mises_elem: Optional[np.ndarray] = None


def fun(
    rho: np.ndarray,
    pre: Preassembly,
    top: FieldTopology,
    material,
    a_last: Dict[int, np.ndarray],
    al_lambda: np.ndarray,
    al_mu: float,
    *,
    flags: StressConstraintFlags,
    n_jobs: Optional[int] = None,
    solver: Optional[str] = None,
    p_exp_torque_ripple: Optional[float] = None,
    subset: Optional[str] = None,
    time_idx: Optional[List[int]] = None,
    active_override: Optional[Dict[str, bool]] = None,
    ripple_target: Optional[float] = None,
    objective_role_override: Optional[str] = None,
    residual_tol: Optional[float] = None,
) -> CallResult:
    """一次完整 fun 调用（EM 段 + 力学段 + 聚合）。对应 ``MMA10:3487-3752``。

    Parameters
    ----------
    rho : (N_element,) 全场密度（warper 滤波后的）。
    pre, top : 预组装与拓扑。
    material : ``em_worker.make_field_material`` 的产物。
    a_last : ``A_node_last`` 缓存（跨调用，本函数写回）。
    al_lambda, al_mu : 力学 AL 乘子 / 罚因子（当前调用值）。
    flags : 应力约束形式。
    n_jobs : Optional[int]
        阶段 8 的 8c 并行度，透传给 ``em_segment``（None = 串行 = golden 对齐）。
    solver : Optional[str]
        线性求解后端（``motorfem.solvers``），透传给 EM 与力学段。
        ``None`` ⇒ ``pre.cfg.solver.solver_backend``（默认 ``'superlu'``）。
    p_exp_torque_ripple : Optional[float]
        脉动范数 p 的**本次调用值**（p 连续化的落点，见 ``ObjectiveConfig``）。
        ``None`` ⇒ ``cfg.objective.p_exp_torque_ripple``（= legacy 固定值，逐位不变）。
    subset : Optional[str]
        2026-10-11 新增：``None``/``"train"``/``"val"``。开验证集时
        （``pre.train_mask`` 非 None）``"train"`` 只求值训练点（优化路径，
        高效：只跑训练子集）、``"val"`` 只求值验证点（评估用）。默认
        ``None`` = 全部点（无验证集时与 legacy 逐位一致）。
    time_idx : Optional[List[int]]
        2026-10-11 新增：只求值这些时间索引（Adam 随机子采样传入）。``None``
        = 默认路径。与 ``subset`` 同时给时 ``time_idx`` 优先。
    """
    cfg = pre.cfg
    n_element = top.n_element
    solver = solver if solver is not None else cfg.solver.solver_backend

    # ---- EM 段（MMA10:3527-3564）----
    # 训练/验证/子采样过滤：默认（无验证集、无 time_idx）走 legacy 全量路径。
    _time_idx = time_idx
    if _time_idx is None and subset is not None and pre.train_mask is not None:
        _want_train = (subset == "train")
        _time_idx = [j for j, m in enumerate(pre.train_mask)
                     if bool(m) == _want_train]
    records = em_segment(rho, pre, top, material, A_node_last=a_last,
                         n_jobs=n_jobs, solver=solver, time_idx=_time_idx,
                         residual_tol=residual_tol)
    torque_Arkkio_vec = np.array([r[1] for r in records])
    dTArkkio_p_drho_vec = np.array([r[2] for r in records])
    flux_linkage_energy_vec = np.array([r[5] for r in records])
    torque_flux_energy_vec = np.array([r[6] for r in records])
    dTflux_p_drho_vec = np.array([r[7] for r in records])
    torque_VD_vec = np.array([r[8] for r in records])
    dTVD_p_drho_vec = np.array([r[9] for r in records])
    # ★ 2026-10-14 新增（record 末尾两项）：体边积分 Maxwell 应力转矩（T_ArkBE）。
    #   生产默认的**纹波口径**。
    torque_ArkBE_vec = np.array([r[10] for r in records])
    dTArkBE_p_drho_vec = np.array([r[11] for r in records])
    _A_node_last, _B_elem_last = records[-1][3], records[-1][4]

    # ---- 力学段（MMA10:3534-3536 + :3044-3051）----
    # ⚠ 目标使能（2026-10-10）：应变能与应力**都禁用**时跳过整个力学段
    #   （mech_worker 不调用 ⇒ 省一次力学求解 + 2 个伴随 spsolve）。此时
    #   g_stress/g_Uelastic 恒 0、AL 状态冻结（无应力约束不需要 AL）。
    rho_mech = rho[list(top.element_rotation)]
    _mech_on = cfg.objective.enable_uelastic or cfg.objective.enable_stress
    if _mech_on:
        mres, u_mech = mech_worker(rho_mech, pre, al_lambda, al_mu, flags=flags,
                                   solver=solver)
    else:
        n_mech_dof = pre.Kmech.n_dof
        u_mech = np.zeros(n_mech_dof)
        mres = NullMechResult(top.n_element_mech)

    # ---- 平均转矩与梯度（MMA10:3575-3595）----
    # ⚠ legacy 还算了 mean_Tflux/mean_TVD（:3576-3577），但只在 :3730 的
    # print 里消费、不进数值路径 —— 阶段 4 没搬 print ⇒ motorfem 侧它们是
    # 死代码，阶段 7 清理（见 design doc §7 死代码清理）。
    mean_TArkkio = float(np.mean(torque_Arkkio_vec))

    if cfg.objective.use_random_mean_torque:
        raise NotImplementedError("use_random_mean_torque 非生产路径，阶段 4 不搬运")
    mean_dTArkkio_p_drho = np.mean(dTArkkio_p_drho_vec, axis=0)
    # ★ 2026-10-14：**平均转矩口径可选**。
    #   'flux'（生产默认）—— `T_flux` 的平均与 ANSYS 差 **0.06%**（最准），
    #       且伴随 `dTflux_p_drho_vec` 本来就有（`frozen_flux_analysis` 一起算的）。
    #   'arkkio'（legacy）—— 逐位保 golden；但它的 h 标定在细网格上偏 45%。
    _mts = cfg.objective.mean_torque_source
    if _mts == "flux":
        g_torque = float(np.mean(torque_flux_energy_vec))
        dg_torque = np.mean(dTflux_p_drho_vec, axis=0)
    elif _mts == "arkkio":
        g_torque = mean_TArkkio
        dg_torque = mean_dTArkkio_p_drho
    else:
        raise ValueError(f"mean_torque_source 非法: {_mts!r}（'flux' | 'arkkio'）")

    # ---- 转矩波动（MMA10:3597-3659）----
    p_exp = cfg.objective.p_exp_torque_ripple if p_exp_torque_ripple is None else p_exp_torque_ripple
    mean_torque_obj = cfg.objective.mean_torque_obj
    n_sampled = len(records)      # 训练/验证/子采样后实际参与的点数
    if cfg.objective.torque_ripple_computed_around_obj_torque:
        ripple_TArkkio = (np.mean(np.abs(torque_Arkkio_vec - mean_torque_obj) ** p_exp)) ** (1 / p_exp)
        ripple_dTArkkio_p_dTi = (1 / ripple_TArkkio ** (p_exp - 1) * np.abs(
            torque_Arkkio_vec - mean_torque_obj) ** (p_exp - 1)
            * np.sign(torque_Arkkio_vec - mean_torque_obj) / n_sampled)
        ripple_dTArkkio_p_dTi -= np.mean(ripple_dTArkkio_p_dTi)

        ripple_TVD = (np.mean(np.abs(torque_VD_vec - mean_torque_obj) ** p_exp)) ** (1 / p_exp)
        ripple_dTVD_p_dTi = (1 / ripple_TVD ** (p_exp - 1) * np.abs(
            torque_VD_vec - mean_torque_obj) ** (p_exp - 1)
            * np.sign(torque_VD_vec - mean_torque_obj) / n_sampled)
        ripple_dTVD_p_dTi -= np.mean(ripple_dTVD_p_dTi)

        # ★ 2026-10-14：T_ArkBE 的纹波（生产默认的纹波口径）。
        #   同一 p-范数与链式结构，与上面两项逐式同形。
        ripple_TArkBE = (np.mean(np.abs(torque_ArkBE_vec - mean_torque_obj)
                                 ** p_exp)) ** (1 / p_exp)
        ripple_dTArkBE_p_dTi = (1 / ripple_TArkBE ** (p_exp - 1) * np.abs(
            torque_ArkBE_vec - mean_torque_obj) ** (p_exp - 1)
            * np.sign(torque_ArkBE_vec - mean_torque_obj) / n_sampled)
        ripple_dTArkBE_p_dTi -= np.mean(ripple_dTArkBE_p_dTi)
    else:
        raise NotImplementedError("torque_ripple_computed_around_obj_torque=False 非生产路径")

    ripple_dTArkkio_p_drho = np.zeros(n_element)
    for i_t in range(n_sampled):
        ripple_dTArkkio_p_drho += ripple_dTArkkio_p_dTi[i_t] * dTArkkio_p_drho_vec[i_t]
    ripple_dTVD_p_drho = np.zeros(n_element)
    for i_t in range(n_sampled):
        ripple_dTVD_p_drho += ripple_dTVD_p_dTi[i_t] * dTVD_p_drho_vec[i_t]
    ripple_dTArkBE_p_drho = np.zeros(n_element)
    for i_t in range(n_sampled):
        ripple_dTArkBE_p_drho += (ripple_dTArkBE_p_dTi[i_t]
                                  * dTArkBE_p_drho_vec[i_t])

    # 脉振目标连续化（2026-10-12）：warper 按步数传当前脉振上限（约束调度开时
    # 从 ripple_target_start 收到 max_torque_ripple）。默认 None = cfg 值，逐位不变。
    _ripple_target = (cfg.objective.max_torque_ripple
                      if ripple_target is None else ripple_target)
    g_ripple_Arkkio = (ripple_TArkkio / _ripple_target - 1) * cfg.objective.weight_torque_ripple
    dg_ripple_Arkkio = (ripple_dTArkkio_p_drho / _ripple_target) * cfg.objective.weight_torque_ripple
    g_ripple_VD = (ripple_TVD / _ripple_target - 1) * cfg.objective.weight_torque_ripple
    dg_ripple_VD = (ripple_dTVD_p_drho / _ripple_target) * cfg.objective.weight_torque_ripple
    g_ripple_ArkBE = (ripple_TArkBE / _ripple_target - 1) * cfg.objective.weight_torque_ripple
    dg_ripple_ArkBE = (ripple_dTArkBE_p_drho / _ripple_target) * cfg.objective.weight_torque_ripple

    # ★ 2026-10-14：**con1 的纹波口径可选**（`ObjectiveConfig.ripple_source`）。
    #   'arkbe'（生产默认）—— 网格稳健、脉动准（波形相关 0.996），与 T_VD 互吻 0.34%；
    #   'vd'    —— 用 con4 那套 T_VD 的纹波（同样是"对 θ 求导"口径，须 cubic）；
    #   'arkkio'（legacy）—— 逐位保 golden；但 h 标定在细网格上偏 45%。
    #   ⚠ con4 恒为 T_VD 的纹波（legacy 契约），**不随本开关变**。
    _rs = cfg.objective.ripple_source
    if _rs == "arkbe":
        _g_rip, _dg_rip = g_ripple_ArkBE, dg_ripple_ArkBE
    elif _rs == "vd":
        _g_rip, _dg_rip = g_ripple_VD, dg_ripple_VD
    elif _rs == "arkkio":
        _g_rip, _dg_rip = g_ripple_Arkkio, dg_ripple_Arkkio
    else:
        raise ValueError(f"ripple_source 非法: {_rs!r}（'arkbe' | 'vd' | 'arkkio'）")

    # ---- 平均密度（MMA10:3661-3681，多项式约束）----
    topo_m = list(top.element_topo_opti_m)
    area = top.em.area
    mean_density = float(np.sum(rho[topo_m] * area[topo_m]) / np.sum(area[topo_m]))
    if not cfg.topology.use_poly_constraint:
        raise NotImplementedError("use_poly_constraint=False 非生产路径")
    Gamma_density = mean_density / cfg.topology.max_density - 1
    g_density = Gamma_density * (Gamma_density ** 2 + 1)
    dg_density = np.zeros(n_element)
    dg_density[topo_m] = ((3 * Gamma_density ** 2 + 1)
                          / (np.sum(area[topo_m]) * cfg.topology.max_density) * area[topo_m])
    g_density *= cfg.objective.weight_density
    dg_density *= cfg.objective.weight_density

    # ---- 应力与应变能（MMA10:3683-3697）----
    g_stress = mres.stress_loss_al if cfg.objective.enable_stress else 0.0
    dg_stress = np.zeros(n_element)
    if cfg.objective.enable_stress:
        for i_elem_mech, i_elem in top.element_mech2raw.items():
            dg_stress[i_elem] = mres.d_al_stress_p_drho[i_elem_mech]

    # ⚠⚠ 临时实验开关（2026-09-30）：去掉**应力约束 AL 项**，目标退化成纯弹性应变能。
    #   目的：验证上一轮 3000 步不收敛、终态偏灰是否由应力约束引起。
    #   **默认关** —— 不设环境变量时逐字走上面的 legacy 路径，
    #   L4/golden 比对（工具 verify_golden.py）不受任何影响。
    #   用完即删；这不是生产路径的一部分，不要留在代码里。
    if os.environ.get("MF_NO_STRESS_CON") == "1":
        g_stress = 0.0
        dg_stress = np.zeros_like(dg_stress)

    g_Uelastic = mres.elastic_strain_energy if cfg.objective.enable_uelastic else 0.0
    dg_Uelastic = np.zeros(n_element)
    if cfg.objective.enable_uelastic:
        for i_elem_mech, i_elem in top.element_mech2raw.items():
            dg_Uelastic[i_elem] = mres.d_u_p_drho[i_elem_mech]

    # ---- 目标与约束（MMA10:3699-3723，2026-10-10 加角色分配）----
    # 6 个参与项（2 机械 + 4 电磁）各有一个「最小化形式」g_i（≤0 约束形式）：
    #   uelastic/stress/con1(Arkkio 脉动)/con2(密度)/con3(转矩)/con4(VD 脉动)。
    # `objective_role` 指定哪项进 f[0] 作目标（minimize），其余启用项进约束（≤0），
    # 禁用项不进 f。默认 "uelastic+stress" = legacy 组合目标（逐位不变）。
    _o = cfg.objective
    g_all = {
        "uelastic": (g_Uelastic, dg_Uelastic),
        "stress": (g_stress, dg_stress),
        # ★ con1 的纹波口径由 `ObjectiveConfig.ripple_source` 选（生产的 'arkbe' /
        #   legacy 的 'arkkio'）；见上面 `_g_rip/_dg_rip` 的赋值。
        "con1": (_g_rip, _dg_rip),
        "con2": (g_density, dg_density),
        "con3": ((1 - g_torque / mean_torque_obj) * _o.weight_torque,
                 (-dg_torque / mean_torque_obj) * _o.weight_torque),
        "con4": (g_ripple_VD, dg_ripple_VD),
    }
    enabled = {
        "uelastic": _o.enable_uelastic,
        "stress": _o.enable_stress,
        "con1": _o.enable_con1,
        "con2": _o.enable_con2,
        "con3": _o.enable_con3,
        "con4": _o.enable_con4,
    }
    # ⚠ 约束逐步激活调度（2026-10-12）：warper 按步数覆盖某些约束的参与标志
    #   （如 con1 前 N 步不参与）。None = 用 cfg 默认（golden 逐位不变）。
    if active_override is not None:
        for _k, _v in active_override.items():
            enabled[_k] = bool(_v)
    role = _o.objective_role if objective_role_override is None else objective_role_override
    if role != "uelastic+stress" and role not in g_all:
        raise ValueError(f"objective_role 非法: {role}（可用 "
                         f"{'uelastic+stress' + '/' + '/'.join(g_all)}）")
    if role != "uelastic+stress" and not enabled[role]:
        raise ValueError(f"objective_role={role} 但该参与项被 enable 关掉 —— 目标必须参与")

    if role == "uelastic+stress":
        # 6 量调度（2026-10-12）：uelastic 恒作目标基座；stress 可按
        # `stress_activate_step` 延迟加入（前 N 步目标 = 纯应变能）。默认
        # enabled["stress"]=True ⇒ obj = uelastic+stress = legacy 组合目标，逐位不变。
        obj = g_all["uelastic"][0]
        dobj = g_all["uelastic"][1]
        if enabled["stress"]:
            obj = obj + g_all["stress"][0]
            dobj = dobj + g_all["stress"][1]
        # 组合目标下，约束 = 4 个电磁约束中**启用**的（禁用项不进 f）
        cons = [k for k in ("con1", "con2", "con3", "con4") if enabled[k]]
    else:
        obj, dobj = g_all[role]
        cons = [k for k in enabled if k != role and enabled[k]]

    f = np.array([obj] + [g_all[c][0] for c in cons])
    df = np.array([dobj] + [g_all[c][1] for c in cons])

    # ---- detail（MMA10:3737-3748）----
    rotor_position0 = np.pi / 180 * cfg.time.rotor_position0_deg
    _rotor_position = rotor_position0 + 100 * np.pi * records[-1][0] / cfg.geometry.pole_pairs
    detail = {
        "time_span": pre.time_span_sampled,
        "torque_Arkkio": torque_Arkkio_vec,
        "torque_flux": torque_flux_energy_vec,
        "torque_VD": torque_VD_vec,
        "A_node": _A_node_last,
        "B_elem": _B_elem_last,
        "rotor_position": _rotor_position,
        "u_node": u_mech,
        "rho_SIMP_mech": rho_mech,
    }

    # 顺序纪律：默认布局 f/df 行序必须与 legacy MMA10:3750-3752 一致（阶段 7）。
    # 自定义布局（目标/约束使能）由 objective_role + enable_* 决定，行数可变；
    # 默认全启用时 = 5 行 = golden 契约。
    assert_published_layout(f, df, size=f.shape[0])

    return CallResult(
        f=f,
        df=df,
        al_lambda_new=mres.al_lambda_new,
        detail=detail,
        torque_Arkkio_vec=torque_Arkkio_vec,
        dTArkkio_p_drho_vec=dTArkkio_p_drho_vec,
        flux_linkage_energy_vec=flux_linkage_energy_vec,
        torque_flux_energy_vec=torque_flux_energy_vec,
        dTflux_p_drho_vec=dTflux_p_drho_vec,
        torque_VD_vec=torque_VD_vec,
        dTVD_p_drho_vec=dTVD_p_drho_vec,
        torque_ArkBE_vec=torque_ArkBE_vec,
        dTArkBE_p_drho_vec=dTArkBE_p_drho_vec,
        rho_SIMP_mech=rho_mech,
        u_node=u_mech,
        von_mises_max=float(mres.von_mises_max),
        stress_loss_al=float(mres.stress_loss_al),
        d_al_stress_p_drho=mres.d_al_stress_p_drho,
        uelastic=float(mres.elastic_strain_energy),
        d_u_p_drho=mres.d_u_p_drho,
        von_mises_elem=np.asarray(mres.von_mises, float),
    )


def gradient_to_x(df0: np.ndarray, top: FieldTopology,
                  mode: str = "legacy") -> np.ndarray:
    """``(5, N_element)`` → ``(5, N_topo)``：warper 的梯度后处理（``MMA10:4034-4045``）。

    从属单元导数叠加到主动单元，再滤波。golden 的
    ``call_{i}_df`` 是这一步之后的值（形状 ``(5, 3181)``）。

    ``mode``（设计文档 §11①）：
    - ``"legacy"``：``filter_SIMP_grad @`` 前向滤波（MMA10:4042 原样，默认保
      golden）。⚠ 这不是被求值目标（``ρ = filter_SIMP @ x``）的伴随链式 ——
      legacy 头部注释声明「有意为之」，判定见 §11①。
    - ``"adjoint"``：``filter_SIMP.T @`` 伴随一致 = ``df/dx`` 真梯度
      （``df/dx = filter_SIMP.T @ (Pᵀ·∂f/∂ρ)[m]``，P 为从属复制映射；
      从属叠加就是 ``Pᵀ·∂f/∂ρ``）。
    """
    df0 = np.array(df0, dtype=float, copy=True)
    for i_elem_s in top.element_topo_opti_s:
        df0[:, top.element_topo_opti_s[i_elem_s]] += df0[:, i_elem_s]
    filt = top.filter_SIMP_grad if mode == "legacy" else top.filter_SIMP.T
    n_fine = len(top.element_topo_opti_m)
    # ⚠ `n_design` 的类属性默认是 **None**（mesh_builder:168），不是缺失 ⇒
    #   `getattr(top, "n_design", n_fine)` 会返回 None 并让 int(None) 崩溃。
    #   与 `run_auto_schedule._load_x0` 的写法保持一致：None → 退回细网格数。
    n_design = int(getattr(top, "n_design", None) or n_fine)
    parent = getattr(top, "design_parent", None)
    out = np.zeros((df0.shape[0], n_design))
    for i_con in range(df0.shape[0]):
        g = filt @ df0[i_con, list(top.element_topo_opti_m)]
        if parent is None:
            out[i_con, :] = g
        else:
            # 两级网格：同一父单元的所有子单元共享一个设计变量 ⇒ 梯度**求和**。
            # 顺序按父索引升序（np.bincount 的累加序由输入下标顺序决定，确定）。
            out[i_con, :] = np.bincount(parent, weights=np.asarray(g, float),
                                        minlength=n_design)
    return out


def L4_snapshot(
    calls: List[CallResult],
    states: List[ALState],
    n_call_counters: List[int],
    x_calls: List[np.ndarray],
    a_last_per_call: List[Dict[int, np.ndarray]],
    top: FieldTopology,
    gradient_mode: str = "legacy",
) -> Dict[str, np.ndarray]:
    """把一次 MMA 轨迹（3 次 fun 调用 + AL 状态 + x）铺成 golden L4 命名。

    ``states[i]`` = 第 i 次调用**之后**的 AL 状态（含更新后的 λ/μ/AL_N_call）。
    ``n_call_counters[i]`` = 第 i 次调用**之后**的 ``_N_call``（= i+1）。
    ``a_last_per_call[i]`` = 第 i 次调用**结束时**的 ``A_node_last`` 快照。

    ⚠ **golden 采集伪影的复现**（``tools/golden_capture.py`` 的
    ``_snapshot_call`` 用 ``_put(snap, n, obj.__dict__[n])`` **存数组引用**，
    而 ``obj_func_parallel_AL_v2`` 的 EM vec 数组在 ``__init__`` 里建一次、
    ``__call__`` 里**原地**覆盖 ⇒ 三次快照指向同一个数组，最终全是**末调用**
    的值。相反，``u_node``/``detail``/``AL_stress_lambda_new`` 等每次
    ``__call__`` 重新赋值（新数组）⇒ 逐调用正确）。

    因此这里对 7 个 EM vec 键与 detail 里的 3 个转矩键，**所有调用**都用
    ``calls[-1]`` 的值 —— 忠实复现 golden 的内容（golden 的逐调用 EM 值在
    L3 里，本函数不负责）。``fun`` 本身的返回值（``CallResult``）不受影响，
    仍是逐调用正确的。
    """
    out: Dict[str, np.ndarray] = {}
    df_x = [gradient_to_x(c.df, top, mode=gradient_mode) for c in calls]   # golden call_{i}_df 是 (5, N_topo)
    final = calls[-1]
    # 末调用值（用于复现「引用捕获」的伪影）
    final_vecs = {
        "torque_Arkkio_vec": final.torque_Arkkio_vec,
        "torque_flux_energy_vec": final.torque_flux_energy_vec,
        "torque_VD_vec": final.torque_VD_vec,
        "flux_linkage_energy_vec": final.flux_linkage_energy_vec,
        "dTArkkio_p_drho_vec": final.dTArkkio_p_drho_vec,
        "dTflux_p_drho_vec": final.dTflux_p_drho_vec,
        "dTVD_p_drho_vec": final.dTVD_p_drho_vec,
    }
    for i, (call, st, ncall, x, a_last) in enumerate(
            zip(calls, states, n_call_counters, x_calls, a_last_per_call)):
        out[f"call_{i:02d}_x"] = np.asarray(x)
        out[f"call_{i:02d}_f"] = np.asarray(call.f)
        out[f"call_{i:02d}_df"] = np.asarray(df_x[i])
        for n, v in final_vecs.items():       # ⚠ 伪影：全部用末调用值
            out[f"call_{i:02d}_{n}"] = np.asarray(v)
        for n, v in (("u_node", call.u_node),
                     ("VonMisesStress_max", call.von_mises_max),
                     ("Stress_loss_AL", call.stress_loss_al),
                     ("dALStress_p_drho", call.d_al_stress_p_drho),
                     ("Uelastic", call.uelastic),
                     ("dU_p_drho", call.d_u_p_drho),
                     ("AL_stress_lambda_new", call.al_lambda_new),
                     ("rho_SIMP_mech", call.rho_SIMP_mech)):
            out[f"call_{i:02d}_{n}"] = np.asarray(v)
        out[f"call_{i:02d}_AL_stress_lambda"] = np.asarray(st.al_lambda)
        out[f"call_{i:02d}_AL_stress_mu"] = np.asarray(st.al_mu)
        out[f"call_{i:02d}_AL_N_call"] = np.asarray(st.al_n_call)
        out[f"call_{i:02d}__N_call"] = np.asarray(ncall)
        for k, v in call.detail.items():
            if v is None:
                continue
            # ⚠ 伪影：detail 的转矩键也是 obj 的原地数组 ⇒ 末调用值
            if k in ("torque_Arkkio", "torque_flux", "torque_VD"):
                v = final.detail[k]
            out[f"call_{i:02d}_detail__{k}"] = np.asarray(v)
        for i_t, a in a_last.items():
            out[f"call_{i:02d}_A_node_last_{i_t}"] = np.asarray(a)
        # mech_{i:02d}_*（力学 worker 的解包顺序，MMA10:3568-3569）
        for n, v in (("u_node", call.u_node),
                     ("VonMisesStress_max", call.von_mises_max),
                     ("Stress_loss_AL", call.stress_loss_al),
                     ("dALStress_p_drho", call.d_al_stress_p_drho),
                     ("Uelastic", call.uelastic),
                     ("dU_p_drho", call.d_u_p_drho),
                     ("AL_stress_lambda_new", call.al_lambda_new)):
            out[f"mech_{i:02d}_{n}"] = np.asarray(v)
    return out
