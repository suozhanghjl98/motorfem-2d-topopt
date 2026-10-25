"""阶段 4E：``FieldSystem`` —— 求解器链的 orchestrator（warper + MMA 驱动）。

把 L1~L4 各段（``mesh_builder`` / ``preassembly`` / ``em_worker`` /
``mech_worker`` / ``objective``）组装成一条**可调用**的优化链路，对应 legacy
``optimizer_func_warper_AL_v2``（``MMA10:3982-4120``）＋ ``MMA_optimize``
（``optimization_method/MMA.py:10-173``）：

    FieldSystem(config, mesh, bh_csv)      # 预组装（L1 + L2）
    fs.warper(x)  → (f, df)                # 一次 fun 调用（L3 EM + L4 力学 + 聚合）
    fval, x = fs.optimize(max_step=3, ...) # MMA 驱动（3 步轨迹）
    fs.L4_snapshot()                       # golden L4 命名 + mma_result

------------------------------------------------------------------------------
warper 与 legacy 的对应（``MMA10:4003-4060``）
------------------------------------------------------------------------------

    ``ρ = density_from_x(x)``          ``:4008-4015``
    ``fun(ρ, λ, μ)``                    ``:4018-4022``
    ``al_update(λ_new, Stress_loss)``   ``:4025-4032 + :4057-4058``（内核 19）
    ``gradient_to_x(df0)``              ``:4034-4045``
    ``_N_call += 1``                    ``:4056``

⚠ legacy 的 v2 warper 读写的是 **v1 类的类属性**（``optimizer_func_warper_AL``，
``:4020/:4025-4032``）—— 本实现用一个 ``ALState`` 对象承载（语义相同，
``kernels/reference/al.py`` 已按此建模）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config import RunConfig, apply_mesh_config, resolve_bh_csv
from ..mesh import MotorMesh, require_regions
from ..kernels.reference.al import ALState, al_initial_state, al_update
from .em_worker import density_from_x, make_field_material
from .mech_worker import stress_flags_from_config
from .mesh_builder import FieldTopology, build_field_topology, verify_winding_layout
from ..winding.layout import star_of_slots
from .objective import L4_snapshot, fun, gradient_to_x
from .optimizers import (optimize_adam, optimize_gcmma, optimize_mma,
                         optimize_slsqp)
from .preassembly import Preassembly, build_preassembly

__all__ = ["FieldSystem", "REQUIRED_EM_REGIONS"]

#: EM driver 声明必需的区域（阶段 3 三道闸之一的启动断言清单）。
#: tag 表路径下 7 个键都由 ``RegionMap.from_tags`` 保证存在；物理组路径下由
#: ``from_physical_groups`` 的 role_map 校验 —— 这里再声明一次是**契约**：
#: 未来区域映射换了来源，缺区域会立刻在驱动构造处报出来，而不是深处 KeyError。
REQUIRED_EM_REGIONS = (
    "element_rotor_shaft",
    "element_rotor_core_m",
    "element_rotor_core_s",
    "element_rotor_gap",
    "element_stator_gap",
    "element_stator_core",
    "element_stator_coil",
)


class FieldSystem:
    """一条完整的 SynRM 拓扑优化求解链（生产配置 = golden 复现）。

    构造即跑阶段 3 三道闸：``MotorMesh`` 的覆盖性检查（闸③）＋ 这里的
    ``require_regions``（闸①）；指纹校验（闸②）由调用方在 resume 时用
    ``ResolvedConfig.verify_mesh`` 触发。
    """

    def __init__(self, config: RunConfig, msh_path, bh_csv: str = None,
                 verify_winding: bool = True, *, mesh_config: str = "auto",
                 init_seed: int = 1002, init_spec=None, polar_spec=None):
        # ⚠ ``bh_csv=None``（2026-10-16 新增）⇒ 从 ``config.material.bh_curve_csv``
        #   解析（``resolve_bh_csv``：绝对路径直接用，相对路径拼 legacy/pytorch/）。
        #   这样**换 B-H 曲线只需改 config 一个字段**，不必每处手传路径。
        #   显式传值仍然优先（旧调用点逐字不变）。
        if bh_csv is None:
            bh_csv = resolve_bh_csv(config)
        # ★ 网格绑定口径：读 ``<stem>.motor.json`` 并**就地**覆盖 config（默认值比对，
        #   调用方显式设过的字段保留）。**必须在这里**（早于 layout/topology 等一切
        #   几何/绕组消费者）。`mesh_config="ignore"` 是探针"故意覆盖网格绑定量"的逃生口。
        apply_mesh_config(config, msh_path, mode=mesh_config)
        self.cfg = config
        # 相带参数化（换极槽/换电机）：``phase_from_generator=True`` 时由
        # star-of-slots 推导 WindingLayout 并交给 MotorMesh（region 相带随之参数化）。
        # False（默认）= legacy 18 槽硬编码路径，逐位不变。
        layout = None
        _w = config.electrical.winding
        if _w.phase_from_generator:
            layout = star_of_slots(
                config.geometry.Z, config.geometry.pole_pairs, 3,
                parts=config.geometry.parts, layers=2,
                inner_ref=_w.inner_ref, outer_ref=_w.outer_ref,
                outer_shift=_w.outer_shift,
            )
        self.mesh = MotorMesh(msh_path, layout=layout)  # 阶段 3：构造即跑覆盖性检查（闸③）
        # ⚠ ``init_seed``（2026-10-16 新增，默认 1002 = legacy）：初始密度的
        #   RNG 种子（``initial_rho_simp`` 的随机打孔）。**默认值逐位不变**；
        #   显式传别的值 ⇒ 换一个随机初始设计（"随机初始化"多起点搜索用）。
        # ⚠ ``init_spec``（2026-10-16 新增，默认 None = legacy 路径）：
        #   换**初始化范式**（洞参数随机 / 高斯随机场 / 二值 / 圆斑…）。
        #   None 时逐字走 ``initial_rho_simp``（golden 路径，逐位不变）。
        #   见 ``fem/init_density.py::make_initial_density``。
        self.top: FieldTopology = build_field_topology(config, self.mesh,
                                                       seed=init_seed)
        if init_spec is not None:
            from .init_density_strategies import apply_init_spec
            apply_init_spec(self.top, init_spec)
        # ★ 极坐标低阶参数化（2026-10-18）：给出 ``polar_spec`` 时，设计变量变成
        #   **傅里叶系数**（不再是每单元密度）。``None``（默认）⇒ 走原路径，
        #   ``density_from_x``/``gradient_to_x`` 逐位不变。见 ``fem/polar_fourier.py``。
        self.polar = None
        if polar_spec is not None:
            from .polar_fourier import PolarFourierParam
            self.polar = PolarFourierParam.from_topology(self.top, polar_spec)
            self.polar.attach_topology(self.top)
        require_regions(self.top.regions, REQUIRED_EM_REGIONS, who="FieldSystem(EM)")  # 闸①
        if verify_winding:
            verify_winding_layout(self.top)   # 生成器推导相带 vs tag 表（阶段 5 哨兵）
        self.pre: Preassembly = build_preassembly(config, self.top)
        self.material = make_field_material(
            bh_csv, config.material.bh_lut_n, config.solver.mur_linear,
            # ★ 2026-10-17：线性档下 SIMP 区也走线性材料（见 assembly 的材料分类）
            linear_simp=not config.solver.use_nonlinear_iron,
        )
        # CSR 装配的「两次归约融合」开关（进程级；每次构造都重置 ⇒ 测试间不串）
        from ..kernels.fast.sparsity import set_csr_fusion
        set_csr_fusion(config.solver.csr_fuse)
        self.flags = stress_flags_from_config(config)
        # 应力平滑（2026-10-20，默认 0 = 关闭）：把逐单元 σ̄ 用固定物理半径的
        # 滤波器平滑后再进 AL，抑制尖角应力奇异导致的网格相关峰值。
        _r_s = float(getattr(config.stress, "stress_filter_radius_m", 0.0) or 0.0)
        if _r_s > 0:
            from ..kernels.fast.filter import compute_filter
            _n_mech = len(self.top.element_S_mech)
            self.pre.stress_smooth = compute_filter(
                self.top.node_mech, self.top.element_S_mech, range(_n_mech), _r_s,
                extend=False, p_filter=float(config.filter.filter_p_exp),
                parts=int(config.geometry.parts))
        self.al_state: ALState = al_initial_state(self.top.n_element_mech)
        self.a_last: Dict[int, np.ndarray] = {}
        self.n_call = 0
        self.trace: List[tuple] = []       # 每次 warper 调用: (x, CallResult, ALState, a_last 快照)
        self.last_f: Optional[np.ndarray] = None
        self.last_x: Optional[np.ndarray] = None

    # ------------------------------------------------------------ warper
    def _current_p_exp(self) -> float:
        """本次 fun 调用的脉动范数 p（p 连续化的当前值）。

        - ``cfg.objective.p_exp_continuation=False``（默认）：返回
          ``p_exp_torque_ripple``，与 legacy 固定值逐位一致。
        - ``True``：p 从 ``p_exp_start`` 线性爬升到 ``p_exp_torque_ripple``，
          ``p_exp_ramp_steps`` 步内完成，之后固定（拓扑优化 continuation
          套路：先低 p 平滑收敛、再高 p 收紧脉动约束）。步数用 ``self.n_call``
          （= 已完成的 warper 调用次数），续跑恢复 checkpoint 后自动接续。
        """
        oc = self.cfg.objective
        if not oc.p_exp_continuation:
            return oc.p_exp_torque_ripple
        frac = min(1.0, self.n_call / max(1, oc.p_exp_ramp_steps))
        return oc.p_exp_start + frac * (oc.p_exp_torque_ripple - oc.p_exp_start)

    def _current_move(self, i_step: int) -> float:
        """MMA 移动极限的当前值（``move_at`` 调度器）。

        - ``p_exp_move_linkage=False``（默认）：恒为 ``cfg.mma.MMA_move``，
          与 legacy 固定移动极限逐字一致（回调存在但返回值恒等于默认 ⇒
          golden 逐位不变）。
        - ``True``（仅 ``p_exp_continuation=True`` 时有意义）：move 随 p 爬升
          同步收紧 —— p 从 ``p_exp_start`` 起每升 ``p_exp_move_step``，
          move × ``p_exp_move_factor``。高 p 时约束梯度 ∝|T−T_obj|^(p−1)
          对设计极敏感，收紧移动极限防 overshoot（2026-10-10 探针结论：
          连续化爬到 p≥7 后 con1 反弹 26.6）。
        """
        oc = self.cfg.objective
        if not oc.p_exp_move_linkage:
            return self.cfg.mma.MMA_move
        frac = min(1.0, self.n_call / max(1, oc.p_exp_ramp_steps))
        p_now = oc.p_exp_start + frac * (oc.p_exp_torque_ripple - oc.p_exp_start)
        steps = max(0.0, p_now - oc.p_exp_start) / max(1.0, oc.p_exp_move_step)
        return max(1e-4, self.cfg.mma.MMA_move * (oc.p_exp_move_factor ** steps))

    def warper(self, x: np.ndarray, *, subset: str = "train",
               time_idx: Optional[List[int]] = None):
        """对应 ``optimizer_func_warper_AL_v2.__call__``（``MMA10:4003-4060``）。

        Returns
        -------
        ``(f, df)`` —— ``f`` (5,)、``df`` (5, N_topo)。golden 的 ``call_{i}_f/df``。

        ``subset``/``time_idx``（2026-10-11 新增，task 3 采样）：
        - ``subset="train"``（默认）：无验证集时 = 全部点，**与 legacy 逐位
          一致**；开验证集时只用训练点（优化路径）。
        - ``subset="val"``：只用验证点（``FieldSystem.evaluate`` 用）。
        - ``time_idx``：显式指定时间索引子集（Adam 随机子采样传入）。

        约束逐步激活（2026-10-12 新增）：``constraint_schedule`` 开时按
        ``self.n_call`` 覆盖 con1/con4 的参与标志（前 N 步不参与，避免 x0
        处反平行死锁；见 ``ObjectiveConfig.constraint_schedule``）。默认关 =
        不覆盖，golden 逐位不变。
        """
        rho = (density_from_x(x, self.top) if self.polar is None
               else self.polar.rho_full(x, self.top))
        active_override = None
        ripple_target = None
        role_override = None
        if self.cfg.objective.constraint_schedule:
            oc = self.cfg.objective
            # 目标角色切换（用户方案：先转矩 → 再机械 → 最后脉动）：
            # 前 N 步用 objective_role（如 con3 最大化转矩），之后切 phase2
            # （如 uelastic+stress 组合目标，转矩变 ≥90 约束保持）。
            if oc.objective_role_phase2 is not None \
                    and self.n_call >= oc.objective_role_switch_step:
                role_override = oc.objective_role_phase2
            # 6 量逐个进场：uelastic/stress 可延迟（con3 目标阶段后组合目标阶段
            # 参与）；con3 作约束可延迟（作目标时必须始终参与）；con1/con4 可延迟。
            active_override = {
                "uelastic": bool(oc.enable_uelastic
                                 and self.n_call >= oc.uelastic_activate_step),
                "stress": bool(oc.enable_stress
                               and self.n_call >= oc.stress_activate_step),
                "con1": bool(oc.enable_con1
                             and self.n_call >= oc.con1_activate_step),
                "con4": bool(oc.enable_con4
                             and self.n_call >= oc.con4_activate_step),
                "con3": bool(oc.enable_con3 and (
                    (oc.objective_role == "con3"
                     and role_override is None)
                    or self.n_call >= oc.con3_activate_step)),
            }
            # 脉振目标连续化：con1 激活后上限从宽收到 5（防过冲，2026-10-12）
            if oc.ripple_ramp_steps > 0 and oc.enable_con1:
                n_after = self.n_call - oc.con1_activate_step
                frac = min(1.0, n_after / oc.ripple_ramp_steps)
                ripple_target = (oc.ripple_target_start
                                 + frac * (oc.max_torque_ripple
                                           - oc.ripple_target_start))
        call = fun(rho, self.pre, self.top, self.material, self.a_last,
                   self.al_state.al_lambda, self.al_state.al_mu, flags=self.flags,
                   n_jobs=self.cfg.runtime.n_jobs,
                   solver=self.cfg.solver.solver_backend,
                   p_exp_torque_ripple=self._current_p_exp(),
                   subset=subset, time_idx=time_idx,
                   active_override=active_override,
                   ripple_target=ripple_target,
                   objective_role_override=role_override)
        self.al_state = al_update(call.al_lambda_new, call.stress_loss_al, self.al_state)
        self.n_call += 1
        self.trace.append((np.array(x, dtype=float), call, self.al_state, dict(self.a_last)))
        if self.polar is not None:
            # 参数化：∂f/∂ρ → ∂f/∂c（从属叠加 + 映射导数 + Bᵀ）
            return call.f, self.polar.project(call.df, x)
        return call.f, gradient_to_x(call.df, self.top,
                                     mode=self.cfg.runtime.gradient_filter)

    # ------------------------------------------------------------ MMA
    def optimize(self, max_step: int = 3000, *, control=None, initial_state=None,
                 optimizer: Optional[str] = None, x0=None, **mma_kw):
        """跑优化（默认 MMA）。返回 ``(fval, x)``。

        golden 对齐用 ``max_step=3`` + ``f_rel_change=x_rel_change=-1``（关退出）；
        生产默认 3000 步。``mma_kw`` 覆盖 ``MMAConfig`` 的其余参数（MMA 路径）。

        ``optimizer`` ∈ {mma, slsqp, adam, gcmma}（``fem/optimizers.py``）；
        ``None`` ⇒ ``cfg.runtime.optimizer``（默认 "mma"，逐位走 legacy）。

        ``control``/``initial_state`` 是 GUI（``tools/optim_gui``）的
        暂停/续跑钩子，透传给各优化器（MMA 见 ``mma_driver`` docstring）；
        默认 ``None`` 时行为逐字不变。续跑前需先 ``restore_checkpoint_state``
        恢复 AL 状态与 A 热启动缓存，轨迹才会与「不中断的从头跑」逐位一致。

        ``x0``（2026-10-18 新增，默认 ``None``）：显式起点。**语义随参数化而变** ——
        ``polar is None`` 时是每单元设计变量（``[0,1]``）；否则是**傅里叶系数**。
        用于"从已有最优设计热启动"（先用 ``polar.initial_coeff`` 投影）。
        """
        m = self.cfg.mma
        optimizer = optimizer or self.cfg.runtime.optimizer
        if initial_state is not None:
            x0 = np.asarray(initial_state["x"], dtype=float)
        elif x0 is not None:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (self.polar.n_coef if self.polar is not None
                            else len(self.top.element_topo_opti_m),):
                raise ValueError(
                    f"x0 形状 {x0.shape} 与当前参数化不符（期望 "
                    f"{(self.polar.n_coef if self.polar is not None else len(self.top.element_topo_opti_m)),})")
        elif self.polar is None:
            x0 = self.top.rho_SIMP_initial[list(self.top.element_topo_opti_m)]
        else:
            # 参数化：初值 = 现有初始密度的最小二乘投影（同一设计的低阶近似）
            x0 = self.polar.initial_coeff(
                self.top.rho_SIMP_initial[list(self.top.element_topo_opti_m)])
        if self.polar is None:
            x_upper = np.ones(x0.shape)
            x_lower = np.zeros(x0.shape)
        else:
            x_lower, x_upper = self.polar.bounds(x0)

        if optimizer == "mma":
            fval, x = optimize_mma(
                self.warper, x0, x_lower, x_upper,
                max_step=max_step, mma_cfg=m,
                control=control, initial_state=initial_state,
                move_at=self._current_move, **mma_kw)
        elif optimizer == "adam":
            # 随机子采样（task 3）：warper 的 time_idx 用**训练点全量索引**；
            # 无验证集时训练索引 = 0..n_train-1（_n_time_points 告知 Adam）。
            def adam_fun(x, sampled_idx=None):
                return self.warper(x, time_idx=sampled_idx)
            adam_fun._n_time_points = int(self.pre.time_span_sampled.size)
            fval, x = optimize_adam(
                adam_fun, x0, x_lower, x_upper,
                max_step=max_step, control=control, initial_state=initial_state,
                **mma_kw)
        elif optimizer == "slsqp":
            fval, x = optimize_slsqp(
                self.warper, x0, x_lower, x_upper,
                max_step=max_step, control=control, initial_state=initial_state,
                **mma_kw)
        elif optimizer == "gcmma":
            fval, x = optimize_gcmma(
                self.warper, x0, x_lower, x_upper,
                max_step=max_step, control=control, initial_state=initial_state,
                **mma_kw)
        else:
            raise ValueError(
                f"未知优化器 {optimizer!r}（可选 mma/slsqp/adam/gcmma，"
                f"见 fem/optimizers.py）")
        self.last_f, self.last_x = fval, x
        return fval, x

    def evaluate(self, rho: np.ndarray, *, subset: str = "val",
                 solver: Optional[str] = None) -> "CallResult":
        """验证集/指定子集求值（task 3 的 test-train 分离评估）。

        不更新 AL 状态、不写轨迹；``A_node_last`` 用独立空缓存（每次独立冷求解）。
        用于优化**之后**在未参与训练的位置上评估真实转矩/脉动。
        """
        return fun(rho, self.pre, self.top, self.material, {},
                   self.al_state.al_lambda, self.al_state.al_mu,
                   flags=self.flags, n_jobs=self.cfg.runtime.n_jobs,
                   solver=solver or self.cfg.solver.solver_backend,
                   subset=subset)

    # ------------------------------------------------------------ 检查点
    def checkpoint_state(self) -> Dict[str, object]:
        """可 JSON 化的续跑状态：AL 乘子/罚因子/计数 + A 热启动缓存 + 调用数。

        与 ``mma_driver.MMA_optimize`` 的 ``initial_state`` 一起构成一次完整
        续跑所需的最小状态（``tools/optim_gui`` 的保存/加载用它）。``a_last``
        是 Newton 热启动缓存：不恢复则续跑后的轨迹与中断前**不一致**（不是
        错，只是不同）。
        """
        return {
            "al_lambda": self.al_state.al_lambda.tolist(),
            "al_mu": float(self.al_state.al_mu),
            "al_n_call": int(self.al_state.al_n_call),
            "a_last": {str(k): v.tolist() for k, v in self.a_last.items()},
            "n_call": int(self.n_call),
        }

    def restore_checkpoint_state(self, d: Dict[str, object]) -> None:
        """``checkpoint_state`` 的逆：续跑前调用，使 AL 状态与 A 热启动
        逐值恢复到中断时刻（golden 对齐路径不需要它）。"""
        self.al_state = ALState(
            al_lambda=np.asarray(d["al_lambda"], dtype=float),
            al_mu=float(d["al_mu"]),
            al_n_call=int(d["al_n_call"]),
        )
        self.a_last = {int(k): np.asarray(v, dtype=float)
                       for k, v in d["a_last"].items()}
        self.n_call = int(d["n_call"])

    # ------------------------------------------------------------ 快照
    def L4_snapshot(self) -> Dict[str, np.ndarray]:
        """golden L4 命名 + ``mma_result_*``（须先跑 ``optimize``）。"""
        if not self.trace:
            raise RuntimeError("先跑 optimize() 才有轨迹可快照")
        x_calls = [t[0] for t in self.trace]
        calls = [t[1] for t in self.trace]
        states = [t[2] for t in self.trace]
        a_last_snaps = [t[3] for t in self.trace]
        counters = [i + 1 for i in range(len(self.trace))]
        out = L4_snapshot(calls, states, counters, x_calls, a_last_snaps, self.top,
                          gradient_mode=self.cfg.runtime.gradient_filter)
        out["mma_result_f"] = np.asarray(self.last_f)
        out["mma_result_x"] = np.asarray(self.last_x)
        return out
