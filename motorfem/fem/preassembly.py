"""阶段 4B：golden L2 的单元级预组装链（``Preassembly``）。

对应 legacy ``obj_func_parallel_AL_v2.__init__`` 的预组装段（``MMA10:3410-3484``）
加 ``get_Kmech_vectorized``（``MMA10:2604-2814``，内核 16）：

- 时间序列：``time_span``（11 点，``:3415-3418``）；
- 逐时间点：``rotating_boundary`` → ``element_operators`` → ``as_legacy_tuple``
  （22 元组 K_vectorized）+ ``R/Rd/Rq``（``:3814-3835`` 的 ``_compute_K_vectorized``）；
- ``Torque_VD_coef_vec``：``rotating_boundary_with_derivative`` →
  ``vd_coefficient``（``:3477-3482``）；
- ``Kmech_vectorized``：``mech_operators``（内核 16）按 legacy dict 契约导出。

golden L2 的 697 个数组里，与 ρ 无关的部分（K/R/Rd/Rq/TVDcoef/Kmech/time_span/
topo_opti_m/s/rho_SIMP_initial）全部由本模块 + ``FieldTopology`` 提供 ——
``rho_SIMP_initial`` 已在 4A 提供，本批补其余。

------------------------------------------------------------------------------
与 legacy 的对应关系（F4 纪律）
------------------------------------------------------------------------------

- ``element_operators`` / ``as_legacy_tuple`` / ``get_R_vector`` 全部是内核 8 的
  保真搬移，逐时间点重算（``rot`` 随转子位置变 ⇒ 插值行与系数变）。
- ``signed_phase_current`` 的相带顺序照抄 legacy（Ap, An, Bp, Bn, Cp, Cn）。
- ``Kmech`` 用 legacy 编号的力学子网格（``FieldTopology`` 提供）—— 探针用
  ``sorted`` 编号，数值解不变但**位置数组不同**，这里必须用 legacy 序。
- ``_flatten`` 的命名语义照抄 ``tools/golden_capture.py``，使快照键名与 golden
  npz 完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import sparse as sp

from ..config import RunConfig
from ..kernels.fast import mech_element_matrices as fast_mech5
from ..kernels.fast import mech_solver as fast_mech
from ..kernels.fast import moving_mesh as fast_mm
from ..kernels.fast import torque as fast_tq
from ..kernels.fast.assembly import element_operators
from ..kernels.reference import assembly as asm
from ..kernels.reference import boundaries as bnd
from ..mesh import RegionMap

from .mesh_builder import FieldTopology

__all__ = ["Preassembly", "build_preassembly", "abc_to_dq_orthogonal",
           "dq_to_abc_orthogonal", "flatten_to_golden_names"]


def abc_to_dq_orthogonal(xa, xb, xc, pos):
    """Clarke + Park 正交变换（等幅值，无 1/√3 缩放）。照抄 ``MMA10:206-211``。"""
    xd = np.sqrt(2 / 3) * (xa * np.cos(pos) + xb * np.cos(pos - 2 / 3 * np.pi)
                           + xc * np.cos(pos - 4 / 3 * np.pi))
    xq = np.sqrt(2 / 3) * -(xa * np.sin(pos) + xb * np.sin(pos - 2 / 3 * np.pi)
                            + xc * np.sin(pos - 4 / 3 * np.pi))
    return xd, xq


def dq_to_abc_orthogonal(xd, xq, pos):
    """Park + 逆 Clarke。照抄 ``MMA10:212-217``。"""
    xa = np.sqrt(2 / 3) * (xd * np.cos(pos) - xq * np.sin(pos))
    xb = np.sqrt(2 / 3) * (xd * np.cos(pos - 2 / 3 * np.pi) - xq * np.sin(pos - 2 / 3 * np.pi))
    xc = np.sqrt(2 / 3) * (xd * np.cos(pos - 4 / 3 * np.pi) - xq * np.sin(pos - 4 / 3 * np.pi))
    return xa, xb, xc


def flatten_to_golden_names(obj: Any, name: str, arrays: Dict[str, np.ndarray]):
    """递归展开 tuple/list/dict/sparse 到扁平的『名字→数组』。

    命名语义与 ``tools/golden_capture.py`` 的 ``_flatten`` 逐字一致，使
    ``Preassembly.L2_snapshot`` 的键名与 golden npz 完全对应：
    元组/列表按位置加 ``_{i:02d}``，dict 按键加 ``__{k}``，稀疏矩阵拆成
    COO 三元组（data/row/col/shape）。
    """
    if isinstance(obj, np.ndarray):
        arrays[name] = obj
    elif isinstance(obj, (np.generic, int, float, bool, str)):
        arrays[name] = np.asarray(obj)
    elif hasattr(obj, "tocoo"):
        c = obj.tocoo()
        arrays[f"{name}__data"] = c.data
        arrays[f"{name}__row"] = c.row
        arrays[f"{name}__col"] = c.col
        arrays[f"{name}__shape"] = np.asarray(c.shape, dtype=np.int64)
    elif isinstance(obj, (tuple, list)):
        try:
            arr = np.asarray(obj)
            if arr.dtype != object:
                arrays[name] = arr
                return
        except (ValueError, TypeError):
            pass
        for i, v in enumerate(obj):
            flatten_to_golden_names(v, f"{name}_{i:02d}", arrays)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten_to_golden_names(v, f"{name}__{k}", arrays)
    else:
        raise TypeError(f"{name}: 无法展开的类型 {type(obj)}")


def mech_operators_to_legacy_dict(ops, n_node: int, n_elem: int) -> Dict[str, Any]:
    """``MechOperators`` → legacy ``get_Kmech_vectorized`` 的 dict（``:2792-2814``）。

    键名与顺序照抄 legacy；``K0_stress``（csr）由 ``stress_data/row/col`` 现场
    装配（legacy ``:2789-2790`` 同式）。⚠ ``_R0_map``（``:2622``）是零消费者，
    内核不搬运、这里也没有（golden 也无此键）。
    """
    return {
        "K0mech_data": ops.k0_data,
        "K0mech_row": ops.k0_row,
        "K0mech_col": ops.k0_col,
        "Kconst_map": ops.kconst_map,
        "K_u_map": ops.k_u_map,
        "K_boundary": ops.k_boundary,
        "R0_data": ops.r0_data,
        "R0_row": ops.r0_row,
        "R0_col": ops.r0_col,
        "Rconst_map": ops.rconst_map,
        "R2": ops.r2,
        "dK_p_dth_u_row": ops.dk_p_dth_u_row,
        "dK_p_dth_u_col": ops.dk_p_dth_u_col,
        "dR_p_dth_row": ops.dr_p_dth_row,
        "dR_p_dth_col": ops.dr_p_dth_col,
        "K0_stress_kE_map": ops.stress_k_e_map,
        "VonMisesStress_map": ops.von_mises_map,
        "K0_stress": sp.csr_matrix(
            (ops.stress_data, (ops.stress_row, ops.stress_col)),
            shape=(3 * n_elem, 2 * n_node),
        ),
        "K0_stress_data": ops.stress_data,
        "K0_stress_row": ops.stress_row,
        "K0_stress_col": ops.stress_col,
    }


@dataclass
class Preassembly:
    """golden L2 的单元级预组装状态（与 ρ 无关）+ EM worker 的输入。

    所有 11 长度列表按 ``time_span`` 索引对齐。
    """

    cfg: RunConfig
    top: FieldTopology

    time_span: np.ndarray            # (11,) 全部时间点（训练+验证合并，若开验证）
    time_span_sampled: np.ndarray
    area_coil: float                 # 每极每相载流区面积（m²）
    j_unit: float                    # 单位 d/q 电流的电流密度
    rotor_positions: np.ndarray      # (11,) 每个时间点的转子机械位置
    currents: np.ndarray             # (11, 3) I_A/I_B/I_C（A）
    j_phase: np.ndarray              # (11, 3) J_A/J_B/J_C（A/m²）

    K_vectorized_vec: List[tuple]    # 11 × 22 元组
    R_vec: List[np.ndarray]          # 11 × (N_node,)
    Rd_vec: List[np.ndarray]
    Rq_vec: List[np.ndarray]
    Torque_VD_coef_vec: List         # 11 × csr
    Kmech: Any                       # MechOperators
    Kmech_legacy_dict: Dict[str, Any]
    mech_tri: np.ndarray             # (N_elem_mech, 3) 力学连接表（legacy 编号）
    mech_l_coef: np.ndarray          # (N_elem_mech, 3, 6) 力学 L 矩阵
    mech_d_mat: np.ndarray           # (3, 3) 力学 D 矩阵
    mech_area: np.ndarray            # (N_elem_mech,) 力学单元面积

    # ⚠ 训练/验证分离（2026-10-11 新增；默认 None = 无验证集，全部点都是训练点，
    # 行为与 legacy 逐位一致）。train_mask[i] = 该时间点是训练点；优化只用训练
    # 点，验证点由 FieldSystem.evaluate(subset="val") 求值。
    train_mask: Optional[np.ndarray] = None

    #: 应力平滑矩阵 A（N_mech×N_mech，行归一化）。None = 关闭（默认，逐位不变）。
    #: 由 ``FieldSystem`` 在 ``cfg.stress.stress_filter_radius_m > 0`` 时构造。
    stress_smooth: Any = None

    # ---------------------------------------------------------------- 快照
    def L2_snapshot(self) -> Dict[str, np.ndarray]:
        """golden L2 命名下的快照（除 ``rho_SIMP_initial``/``element_topo_opti*``，
        那 4 项在 ``FieldTopology`` 的 L2 部分里）。"""
        out: Dict[str, np.ndarray] = {}
        out["time_span"] = np.asarray(self.time_span)
        out["time_span_sampled"] = np.asarray(self.time_span_sampled)
        for i_t, kv in enumerate(self.K_vectorized_vec):
            flatten_to_golden_names(kv, f"Kvec_{i_t:02d}", out)
        for i_t, r in enumerate(self.R_vec):
            out[f"R_vec_{i_t:02d}"] = np.asarray(r)
        for i_t, r in enumerate(self.Rd_vec):
            out[f"Rd_vec_{i_t:02d}"] = np.asarray(r)
        for i_t, r in enumerate(self.Rq_vec):
            out[f"Rq_vec_{i_t:02d}"] = np.asarray(r)
        for i_t, vd in enumerate(self.Torque_VD_coef_vec):
            flatten_to_golden_names(vd, f"TVDcoef_{i_t:02d}", out)
        flatten_to_golden_names(self.Kmech_legacy_dict, "Kmech", out)
        return out


def build_preassembly(config: RunConfig, top: FieldTopology) -> Preassembly:
    """预组装 golden L2 的全部单元级量。构造顺序照抄 legacy ``__init__``。"""
    cfg = config
    layer, ctx, em = top.layer, top.ctx, top.em
    parts = cfg.geometry.parts
    pole_pairs = cfg.geometry.pole_pairs
    n_node = layer.n_node
    err = cfg.geometry.coord_err_abs
    regions: RegionMap = top.regions

    # ---- 时间序列（legacy :3415-3418；2026-10-11 加采样/验证集）----
    time_en = cfg.time.time_simulation_s
    time_step = time_en / cfg.time.N_point_simulation
    rotor_position0 = np.pi / 180 * cfg.time.rotor_position0_deg  # legacy :3447（22.5°）
    _tc = cfg.time
    if _tc.sampling_mode == "uniform":
        train_t = np.arange(0, time_en / time_step) * time_step
    elif _tc.sampling_mode == "random":
        _rng = np.random.RandomState(_tc.random_seed)
        train_t = np.sort(_rng.uniform(0.0, time_en, _tc.N_point_simulation))
    elif _tc.sampling_mode == "table":
        if not _tc.positions_deg:
            raise ValueError("sampling_mode='table' 需要 TimeConfig.positions_deg")
        train_t = np.array([
            (d * np.pi / 180 - rotor_position0) * cfg.geometry.pole_pairs
            / (100 * np.pi) for d in _tc.positions_deg], dtype=float)
    else:
        raise ValueError(f"未知 sampling_mode={_tc.sampling_mode!r}（可选 "
                         f"uniform/random/table）")
    if _tc.use_validation_set:
        if _tc.sampling_mode == "uniform":
            # 验证点 = 训练间隔的中间（确定性半周期偏移，与训练点不重叠）
            _nv = _tc.n_validation_points
            val_t = (np.arange(_nv) + 0.5) * (time_en / _nv)
        else:
            _rng = np.random.RandomState(_tc.validation_seed)
            val_t = np.sort(_rng.uniform(0.0, time_en, _tc.n_validation_points))
        time_span = np.concatenate([train_t, val_t])
        time_span_sampled = train_t.copy()
        train_mask = np.array([True] * train_t.size + [False] * val_t.size,
                              dtype=bool)
    else:
        time_span = train_t
        time_span_sampled = train_t.copy()
        train_mask = None

    # ---- 常量（legacy :3444-3447）----
    current_amp = cfg.electrical.winding.current_amp
    n_conductor = cfg.electrical.winding.N_conductor
    # ⚠ area_current 用**顺序累加**（legacy :509-545 的 ``area_current += Delta``，
    # ``for i_elem in range(N_element)`` 顺序）。``current_area`` 走 ``np.sum``
    #（成对求和），末位不同 ⇒ 电流密度 ⇒ R 差 ~1e-14（golden 比对实测）。
    _ap_an = regions.region("element_phaseAp") | regions.region("element_phaseAn")
    _area_current = 0.0
    for i_elem in range(em.area.shape[0]):
        if i_elem in _ap_an:
            _area_current += float(em.area[i_elem])
    # area_coil = 单位电流密度用的「A 相单实体面积」= _area_current / (A 相在扇区
    # 内的绕组实体数)。实体数 = 每极每相槽数 q × 层数 2 × 扇区极数 (2p/parts)。
    # legacy（36 槽，扇区 = 1 个极）时 = Z/(p·m)，逐位不变；6s4p 扇区 = 2 个极
    # （parts=2），需再乘 2 —— 否则 area_coil 偏大一倍、J 偏小、转矩偏小 ~4 倍。
    _n_sector_poles = 2.0 * pole_pairs / parts
    area_coil = _area_current / (cfg.geometry.Z / pole_pairs / 3 * _n_sector_poles)
    j_unit = 1 * n_conductor / area_coil

    # ---- 相带集合（legacy 书写顺序，F4：Ap/An/Bp/Bn/Cp/Cn）----
    phase_sets = [
        regions.region("element_phaseAp"), regions.region("element_phaseAn"),
        regions.region("element_phaseBp"), regions.region("element_phaseBn"),
        regions.region("element_phaseCp"), regions.region("element_phaseCn"),
    ]
    element_current = regions.region("element_current")

    # Dirichlet 圆（定子外径，legacy :288-289）
    # ★ 半径由**网格最大节点 r** 自动确定（用户 2026-10-12：换电机不必手填
    #   stator_outer_radius_m）。legacy 网格外圆节点在圆弧上 ⇒ max_r == m_R，
    #   SynRM golden 逐位不变。判据仍是 ``|r² − m_R²| < err``（err = coord_err_abs）。
    m_R = float(np.sqrt((top.node_raw[:, 1] ** 2
                         + top.node_raw[:, 2] ** 2).max()))
    node_dirichlet = set(
        int(_n[0]) for _n in top.node_raw
        if abs((_n[1] ** 2 + _n[2] ** 2) - m_R ** 2) < err
    )
    # 护栏（2026-10-12 review）：自动检测的半径若选错（如网格含远场/包围盒节点
    # 使 max r 偏大），Dirichlet 集会**静默为空** —— 求解随即奇异。这里立刻报错，
    # 并提示与 config 的差（config 值不强制一致，但相差过大通常是网格有问题）。
    if not node_dirichlet:
        raise ValueError(
            "自动检测的 Dirichlet 圆（最大节点 r = %.6g m）上没有任何节点；"
            "检查网格是否含超出定子外径的节点（cfg.stator_outer_radius_m = %.6g m）"
            % (m_R, cfg.geometry.stator_outer_radius_m))

    def _source(rot, jA, jB, jC) -> np.ndarray:
        """带符号电流密度 → R（legacy :2205-2217 + :2121-2219）。"""
        j_of_elem = asm.signed_phase_current(
            element_current, phase_sets, [jA, -jA, jB, -jB, jC, -jC],
        )
        return asm.get_R_vector(ctx, rot, n_node=n_node, r0elem=em.R0,
                                current_of_element=j_of_elem)

    rotor_positions = np.zeros(time_span.size)
    currents = np.zeros((time_span.size, 3))
    j_phase = np.zeros((time_span.size, 3))
    K_vectorized_vec, R_vec, Rd_vec, Rq_vec, Torque_VD_coef_vec = [], [], [], [], []

    for i_t, _t in enumerate(time_span):
        rotor_position = rotor_position0 + 100 * np.pi * _t / pole_pairs  # legacy :3815
        rotor_positions[i_t] = rotor_position

        # ---- 逐时间点的 K_vectorized（22 元组）+ R/Rd/Rq（legacy :3814-3835）----
        rot = fast_mm.rotating_boundary(layer, parts, rotor_position,
                                        a_eq=cfg.solver.EM_MS_boundary_A_eq,
                                        interp=cfg.solver.moving_mesh_interp)
        ops_t = element_operators(
            ctx, rot, n_node=n_node, k0elem=em.K0, bxcoef=em.Bxcoef, bycoef=em.Bycoef,
            element_iron=regions.region("element_iron"),
            element_topo_opti=top.element_topo_opti,
            use_nonlinear_iron=cfg.solver.use_nonlinear_iron,
        )
        row_rep = {int(n): np.asarray(rot.matrix[n, :].todense()).ravel()
                   for n in layer.rotor_nodes}
        driving_t = bnd.driving_rows(n_node, node_dirichlet,
                                     layer.pairing.slave_to_master, False, row_rep)
        K_vectorized_vec.append(asm.as_legacy_tuple(ops_t, driving_t))

        phaseA_position = (np.pi / 180 * cfg.time.phaseA_offset_deg
                           + 100 * np.pi * _t)  # legacy :3819（偏移默认 0）
        _I_A = current_amp * np.cos(phaseA_position - 0 * np.pi / 3)
        _I_B = current_amp * np.cos(phaseA_position - 2 * np.pi / 3)
        _I_C = current_amp * np.cos(phaseA_position - 4 * np.pi / 3)
        currents[i_t] = (_I_A, _I_B, _I_C)
        _J_A = _I_A * n_conductor / area_coil
        _J_B = _I_B * n_conductor / area_coil
        _J_C = _I_C * n_conductor / area_coil
        j_phase[i_t] = (_J_A, _J_B, _J_C)

        R_vec.append(_source(rot, _J_A, _J_B, _J_C))
        Rd_vec.append(_source(rot, *dq_to_abc_orthogonal(j_unit, 0.0,
                                                         rotor_position * pole_pairs)))
        Rq_vec.append(_source(rot, *dq_to_abc_orthogonal(0.0, j_unit,
                                                         rotor_position * pole_pairs)))

        # ---- 虚位移（VD）系数矩阵（legacy :3477-3482，rotating_boundary_v2）----
        rot_v2 = fast_mm.rotating_boundary_with_derivative(
            layer, parts, rotor_position, a_eq=cfg.solver.EM_MS_boundary_A_eq,
            interp=cfg.solver.moving_mesh_interp)
        Torque_VD_coef_vec.append(fast_tq.vd_coefficient(
            ctx, rot_v2, n_node=n_node, n_element=ctx.tri.shape[0],
            bxcoef=em.Bxcoef, bycoef=em.Bycoef, area=em.area,
            parts=parts, mu0=cfg.material.mu0,
        ))

    # ---- Kmech（legacy :3460-3473 的 get_Kmech_vectorized；内核 16）----
    node_mech_xy = top.node_mech[:, 1:3]
    tri_mech = np.array([e[1:4] for e in top.element_S_mech], dtype=np.int64)
    line_tri = np.array([e[1:3] for e in top.element_L_mech], dtype=np.int64)
    loaded_lines = top.element_mech_L_load
    constrained = sorted(top.node_mech_constraint)   # content 集；顺序不影响装配
    mech_pairing = top.mech_pairing

    mat = fast_mech5.MechMaterial(
        E=cfg.material.mech_E_pa, nu=cfg.material.mech_nu, rho=cfg.material.mech_rho_kg_m3,
    )
    mech = fast_mech5.mech_element_matrices(node_mech_xy, tri_mech, mat)
    omega = cfg.mech.rot_speed_rpm / 30 * np.pi
    body_fx, body_fy = fast_mech5.mech_body_force(
        node_mech_xy, tri_mech, mat, angular_speed=omega,
        rot_center=(cfg.mech.rot_center_x_m, cfg.mech.rot_center_y_m),
    )
    line_loads = fast_mech5.mech_line_loads(
        node_mech_xy, line_tri, loaded_lines, Fload=cfg.mech.Fload_n,
    )
    ops_mech = fast_mech.mech_operators(
        node_mech_xy, tri_mech,
        k_elem=mech.K, l_coef=mech.L, d_mat=mech.D,
        r0_elem=mech.R0, r2_elem=line_loads.R0,
        body_fx=body_fx, body_fy=body_fy,
        line_tri=line_tri, loaded_lines=loaded_lines,
        boundary_fx=line_loads.fx, boundary_fy=line_loads.fy,
        slave_to_master=mech_pairing.slave_to_master,
        constrained_nodes=constrained, parts=parts,
    )
    kmech_legacy = mech_operators_to_legacy_dict(ops_mech, n_node=top.n_node_mech,
                                                 n_elem=top.n_element_mech)

    return Preassembly(
        cfg=cfg, top=top,
        time_span=time_span, time_span_sampled=time_span_sampled,
        area_coil=area_coil, j_unit=j_unit, train_mask=train_mask,
        rotor_positions=rotor_positions, currents=currents, j_phase=j_phase,
        K_vectorized_vec=K_vectorized_vec, R_vec=R_vec, Rd_vec=Rd_vec, Rq_vec=Rq_vec,
        Torque_VD_coef_vec=Torque_VD_coef_vec,
        Kmech=ops_mech, Kmech_legacy_dict=kmech_legacy,
        mech_tri=tri_mech, mech_l_coef=mech.L, mech_d_mat=mech.D, mech_area=mech.area,
    )
