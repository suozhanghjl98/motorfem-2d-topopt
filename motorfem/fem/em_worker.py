"""阶段 4C：L3 单步电磁（``obj_func_worker_vectorized_v2`` 的内核组装版）。

对应 legacy ``nonlinear_solver_energy_MMA10.py:2971-3041``（下称 MMA10）。纯内核
组合调用 —— Newton（内核 11 fast）、Arkkio / 虚位移转矩与伴随导数（内核 12/13/14）、
冻结磁导率磁链 / 转矩 / 导数（内核 15）。**本模块不产生新数值**，只负责按 legacy
的调用顺序把已逐位验证的内核串起来。

------------------------------------------------------------------------------
三件事的 legacy 出处
------------------------------------------------------------------------------

1. ``em_step``：单时间点 worker，返回与 ``:3037-3041`` 完全相同的 10 元组
   ``(t, T_Arkkio, dT_Arkkio/dρ, A, B, ψ, T_flux, dT_flux/dρ, T_VD, dT_VD/dρ)``。
2. ``density_from_x``：warper 把设计变量 ``x`` 变回全场密度
   （``:4008-4015``：``filter_SIMP @ x`` + 从属单元同步）。这是 L3/L4 验收时
   注入 golden ``call_{i}_x`` 的入口。
3. ``em_segment``：一次 fun 调用的完整 EM 段（11 时间点，带 ``A_node_last``
   暖启动缓存，``:3527-3564``）。

------------------------------------------------------------------------------
F4 注意
------------------------------------------------------------------------------

``elements_to_compute`` 传 ``list(element_topo_opti)`` —— 顺序由 set 的迭代
顺序决定（``FieldTopology.element_topo_opti`` 按 legacy 构造式建立），**禁止
sorted**。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import linalg as sp_sparse_linalg

from .. import solvers
from ..elimination import build_em_elimination
from ..kernels.approx.bh_lut import TabulatedHcurve
from ..kernels.fast import flux as fast_flux
from ..kernels.fast import newton as fast_nt
from ..kernels.fast import torque as fast_tq
from ..kernels.reference.bh_curve import BHCurve
from ..kernels.reference.newton import FieldMaterial
from ..kernels.reference.simp import SIMPParams, SIMPProjection

from .mesh_builder import FieldTopology
from .preassembly import Preassembly, abc_to_dq_orthogonal

__all__ = [
    "FieldMaterial",
    "make_field_material",
    "density_from_x",
    "em_step",
    "em_segment",
]


def legacy_simp_dfunc(rho: np.ndarray, simp: SIMPProjection) -> np.ndarray:
    """``SIMP_dfunc`` 的 legacy 原式（``MMA10:1142-1155``）。

    ⚠ ``reference.simp.SIMPProjection.derivative`` 用的是**等价**写法
    （``β·sech²/A``），两者代数相等、浮点末位不同（实测 max|Δ| ≈ 6.7e-16）。
    ``reference/simp.py`` 的 docstring 明确写的是「数值一致」而非逐位一致 ——
    它服务于 β→0 的稳健性。**golden 对齐必须用本函数**（dγ/dρ 链的所有消费点）。
    """
    p = simp.params
    beta, thb, lin, mins = p.beta, p.thb, p.linear_factor, p.min_density
    rho_1 = ((np.tanh(beta * (rho - thb)) + np.tanh(beta * thb))
             / (np.tanh(beta * (1 - thb)) + np.tanh(beta * thb)))
    drho_1 = (beta * np.cosh(beta * thb) * np.cosh(beta - beta * thb)
              / np.sinh(beta) / np.cosh(beta * (rho - thb)) ** 2)
    rho_2 = rho
    drho_2 = 1.0
    rho_c = rho_1 * (1 - lin) + rho_2 * lin
    drho_c = drho_1 * (1 - lin) + drho_2 * lin
    return (1 - mins) * rho_c ** (p.power - 1) * p.power * drho_c


def legacy_dgamma_p_drho(B: np.ndarray, rho: np.ndarray, material: FieldMaterial) -> np.ndarray:
    """``dγ/dρ`` 的 legacy 原式（``MMA10:1196-1204``，逐行照搬）。

    ``-H′(B/ρ₁)/ρ₁² · SIMP_dfunc(ρ)``。⚠ ``material`` 的 ``simp`` 提供
    ``SIMP_func``（``project`` 逐位一致）与 ``curve.dHdB``（内核 1 逐位一致）；
    ``SIMP_dfunc`` 用上面的 ``legacy_simp_dfunc``（reference.derivative 差 1 ULP）。
    """
    Bc = np.clip(B, a_min=1e-20, a_max=np.inf)
    rhoc = np.clip(rho, a_min=0, a_max=1)
    rho1 = material.simp.project(rhoc)          # = SIMP_func（逐位一致）
    return -material.curve.dHdB(Bc / rho1) / rho1 ** 2 * legacy_simp_dfunc(rhoc, material.simp)


#: BLAS 线程环境变量名（OMP/OPENBLAS/MKL —— 阶段 8 的 8c 并行用）
_BLAS_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _build_em_elimination(kv, top, a_eq):
    """从该时间点的 22 元组构造消元算子（滑移层插值行取自 J_mat_driven）。

    滑移层插值随转子位置变（rotating_boundary 每时间点不同）⇒ 每个时间点各自的
    elimination。主从配对与 rotor_nodes 来自 ``top.layer``（插层后不变）。
    """
    Jmd = kv[21].tocsr()
    rotor_nodes = top.layer.rotor_nodes
    interp = {}
    for n in rotor_nodes:
        row = Jmd[n].tocoo()
        coeffs = {int(c): float(v) for c, v in zip(row.col, row.data) if c != n}
        interp[n] = coeffs
    return build_em_elimination(
        top.layer.n_node, top.layer.pairing.slave_to_master, interp, a_eq)


#: 消元算子缓存（性能优化，2026-09-30 实测）：``EliminationOperator`` 只依赖
#: 时间点常量（``J_mat_driven`` / ``top.layer`` / ``a_eq``），与 ρ、与 MMA 步无关
#: —— ``em_segment`` 每步每点重建是纯浪费（实测 ~18 ms/点 × 3000 步）。按
#: ``(id(top), i_t)`` 缓存，用 ``top`` 引用校验 id 复用。数值零影响（确定性纯函数）。
_ELIM_CACHE: dict = {}


def _cached_em_elimination(i_t: int, kv, top, a_eq):
    """按时间点缓存 ``_build_em_elimination`` 的产物（跨 MMA 步复用）。

    ⚠ key 必须含 ``a_eq``（2026-10-12 补）：它参与消元算子、却被原 key 漏掉 ⇒
    同一 ``top`` 上改 ``a_eq`` 会命中旧缓存（生产路径里 a_eq 恒定、不出错，但探针
    里临时改会静默拿错结果）。
    """
    key = (id(top), i_t, bool(a_eq))
    ent = _ELIM_CACHE.get(key)
    if ent is not None and ent[0] is top:
        return ent[1]
    elim = _build_em_elimination(kv, top, a_eq)
    _ELIM_CACHE[key] = (top, elim)
    return elim


def _set_blas_threads(n: int) -> None:
    """把 BLAS 线程环境变量设成 ``n``（8c 并行时设 1，避免超订）。"""
    for var in _BLAS_THREAD_ENV:
        os.environ[var] = str(n)


def _restore_blas_threads() -> None:
    """恢复 BLAS 线程环境变量到本进程的**调用前**状态。

    阶段 8 的 8c 纪律：并行是 `RuntimeConfig.n_jobs` 显式开启的局部行为，
    不得改变进程级环境（golden 对齐路径依赖 BLAS 单线程，`tests/conftest.py`
    在 pytest 里设了它；此处不恢复会污染后续调用）。
    """
    for var in _BLAS_THREAD_ENV:
        os.environ.pop(var, None)


def make_field_material(bh_csv: str, bh_lut_n: int = 0,
                        mur_linear: float = 2000.0,
                        linear_simp: bool = False) -> FieldMaterial:
    """生产路径的材料束：``BHCurve.from_csv`` + ``SIMPProjection(SIMPParams())``。

    ⚠ 必须用 ``from_csv``（含 ``extend_bh_table`` 的对称化与尾段外推），不能用裸表
    ``load_bh_table`` —— 裸表在低 B 处的斜率不同，收敛余量差 ~100 倍
    （``docs/重构设计文档.md`` §11⑨ 的 ★更正）。

    Parameters
    ----------
    bh_lut_n : int
        ``0``（默认）= 用 scipy PCHIP 的 ``BHCurve``，与 legacy 位级一致。
        ``>0`` = 换成 ``approx.TabulatedHcurve``（表点数 = ``bh_lut_n``），
        **放弃位级一致**（见 ``kernels/approx/bh_lut.py``）。

        **切换方式是换注入的曲线对象，``fast/`` 与 ``reference/`` 一行不改**
        —— 这正是这两层把 ``curve`` 当参数注入的收益。``Wm``/``dWmdB`` 在
        查表版里原样委托给原曲线，故非 Newton 路径不受影响。
    mur_linear : float
        线性铁磁相对磁导率（默认 **2000 = legacy 硬编码值，逐位不变**）。
        ⚠ 只在 ``cfg.solver.use_nonlinear_iron=False`` 时参与 μ 合成
        （见 ``newton.py`` 的 ``is_linear_iron`` 支），与那个开关**配对使用**。
    """
    curve = (TabulatedHcurve.from_csv(bh_csv, n=bh_lut_n) if bh_lut_n > 0
             else BHCurve.from_csv(bh_csv))
    return FieldMaterial(curve=curve, simp=SIMPProjection(SIMPParams()),
                         mur_linear=float(mur_linear),
                         linear_simp=bool(linear_simp))


def density_from_x(x: np.ndarray, top: FieldTopology) -> np.ndarray:
    """设计变量 ``x`` → 全场密度 ``ρ``。对应 ``MMA10:4008-4015``（warper 的头部）。

    ``ρ[topo_m] = filter_SIMP @ x``，再把从属单元设成主动单元的值
    （``element_topo_opti_s`` 是合并后的 dict，``:4014-4015``）。
    """
    rho = np.zeros(top.n_element)
    # 两级网格（2026-10-20）：``design_parent`` 存在时，x 的每个分量对应一个**粗**
    # 设计单元，先按父映射展开成细设计单元上的分片常数，再过细网格滤波。
    parent = getattr(top, "design_parent", None)
    if parent is not None:
        rho[list(top.element_topo_opti_m)] = top.filter_SIMP @ np.asarray(x, float)[parent]
    else:
        rho[list(top.element_topo_opti_m)] = top.filter_SIMP @ x
    for i_elem_s in top.element_topo_opti_s:
        rho[i_elem_s] = rho[top.element_topo_opti_s[i_elem_s]]
    return rho


def em_step(
    t: float,
    i_t: int,
    rho: np.ndarray,
    kv,
    R: np.ndarray,
    Rd: np.ndarray,
    Rq: np.ndarray,
    currents: Tuple[float, float, float],
    vd_coef,
    rotor_position: float,
    a0: Optional[np.ndarray],
    *,
    top: FieldTopology,
    material: FieldMaterial,
    pole_pairs: int,
    parts: int,
    solver: Optional[str] = None,
    elimination: Optional[object] = None,
    solve_abs_err: float = 1e-3,
    residual_tol: Optional[float] = None,
) -> tuple:
    """单时间点电磁 worker。对应 ``MMA10:2971-3041`` 的 10 元组返回。

    三个转矩指标的冻结求解依赖在这里是**硬编码**的（Arkkio/VD 不需要、
    flux 需要 d/q）—— 与 ``optimization.torque_registry`` 的
    ``requires_frozen_solves`` 声明一一对应（阶段 7 把它显式化）。

    Parameters
    ----------
    t : float
        时间点（``pre.time_span[i_t]``，用于返回首元）。
    i_t : int
        时间索引（仅用于注解，不参与数值）。
    rho : (N_element,) 全场密度（warper 滤波后的）。
    kv : 22 元组（``pre.K_vectorized_vec[i_t]``）。
    R, Rd, Rq : (N_node,) 源项（三相 / d / q 单位电流）。
    currents : (I_A, I_B, I_C) 相电流（A）。
    vd_coef : 虚位移系数矩阵（``pre.Torque_VD_coef_vec[i_t]``）。
    rotor_position : 转子机械位置（rad）。
    a0 : 暖启动（``A_node_last[i_t]``，None = 冷启动）。
    """
    n_node = top.n_node
    n_elem = top.n_element
    area = top.em.area

    # 求解会话：一次构造（em_segment 传入则复用，此处 owned=False）。
    # ⚠ Newton 循环 + 三个伴随解 + 两次冻结求解共用一个会话 —— pardiso 的 MKL
    # 实例只初始化一次（实测每实例 ~2s，循环里新建会卡死）。
    session, owned = solvers.as_session(solver)
    try:
        # ---- Newton 求解（MMA10:2978-2981）----
        A, B, J = fast_nt.solve_field_vectorized(
            rho, kv, R, _A0=a0, _solve_abs_err=solve_abs_err, max_iter=100,
            iter_relaxation=20, relaxation_ratio=0.5, output=False,
            n_node=n_node, area=area, material=material, solver=session,
            elimination=elimination, residual_tol=residual_tol,
        )

        # 公共链：dT/dρ = dT/dγ · dγ/dρ（legacy :2991/:3001/:3033，三处相同）
        # ⚠ dγ/dρ 用 legacy 原式 —— 内核的 nonlinear_material_dnu_drho 差 1 ULP
        #（SIMP_dfunc 的等价写法，见 legacy_simp_dfunc 的 docstring）。
        dgamma_p_drho = legacy_dgamma_p_drho(B, rho, material)
        elements_to_compute = list(top.element_topo_opti)

        # 阶段 8 分解复用：三个伴随解（Arkkio/VD/flux）共用收敛点 J 的转置因子。
        # ⚠ 只构造一次，逐个右端 factor.solve() 回代 —— 与 spsolve 逐位一致（实测）；
        #   禁止把右端拼成矩阵一次 factor.solve(B)（批量回代不逐位一致）。
        # ⚠ 后端经 session.factorize：superlu 返回 SuperLU 因子（逐位不变）；pardiso
        #   返回 _PardisoFactorView（引用会话实例、不重复 init —— 伴随在 Newton
        #   之后、冻结之前，覆盖 Newton 最后一步的因子无碍）。
        # ⚠ 消元模式下 J 已是 Tᵀ·J_full·T（对称，newton_iterations 返回）⇒
        #   factor 解 J_free.T；伴随内核用 elimination 收缩右端/投影（见内核）。
        factor_jt = session.factorize(J.T)
        try:
            # ---- Arkkio（MMA10:2987-2991）----
            T_arkkio = fast_tq.torque_quadratic(A, top.torque_arkkio_coef)
            dT_arkkio = (fast_tq.adjoint_torque_derivative(
                A, kv, J, top.torque_arkkio_coef, elements_to_compute,
                n_node=n_node, n_element=n_elem, factor=factor_jt,
                elimination=elimination,
            ) * dgamma_p_drho)

            # ---- 虚位移 VD（MMA10:2997-3001）----
            T_vd = fast_tq.torque_quadratic(A, vd_coef)
            dT_vd = (fast_tq.adjoint_torque_derivative(
                A, kv, J, vd_coef, elements_to_compute,
                n_node=n_node, n_element=n_elem, factor=factor_jt,
                elimination=elimination,
            ) * dgamma_p_drho)

            # ---- ★ T_ArkBE（体边积分 Maxwell 应力；2026-10-14 新增）----
            #   生产默认的**纹波口径**（`ObjectiveConfig.ripple_source='arkbe'`）。
            #   与上面三者**完全同构**：`T = Aᵀ Q A` 的二次型 ⇒ 值用 `torque_quadratic`、
            #   伴导用 `adjoint_torque_derivative`（**不必另写伴导**）。
            #   为什么用它：`T_Arkkio` 的 h 标定在细网格上偏 45%（见 §4.4），
            #   而 `T_ArkBE` 网格稳健、且与 T_VD 互吻 0.34%。
            T_arkbe = fast_tq.torque_quadratic(A, top.torque_arkbe_coef)
            dT_arkbe = (fast_tq.adjoint_torque_derivative(
                A, kv, J, top.torque_arkbe_coef, elements_to_compute,
                n_node=n_node, n_element=n_elem, factor=factor_jt,
                elimination=elimination,
            ) * dgamma_p_drho)

            # ---- 冻结磁导率（MMA10:3006-3033）----
            I_A, I_B, I_C = currents
            I_d, I_q = abc_to_dq_orthogonal(I_A, I_B, I_C, rotor_position * pole_pairs)
            frozen = fast_flux.frozen_flux_analysis(
                B, rho, kv, Rd, Rq, I_d, I_q,
                n_node=n_node, area=area, material=material,
                pole_pairs=pole_pairs, parts=parts, solver=session,
                elimination=elimination,
            )
            dT_flux = fast_flux.flux_torque_derivative(
                A, rho, frozen.Bx_d, frozen.By_d, frozen.Bx_q, frozen.By_q,
                I_d, I_q, J, elements_to_compute, kv,
                n_node=n_node, n_element=n_elem, area=area, material=material,
                pole_pairs=pole_pairs, parts=parts, factor=factor_jt,
                elimination=elimination,
            )
            # ✅ 2026-10-14 复核结论：**本函数（手写路径）在全链口径下正确** ——
            #   四口径的伴随经「全链 FD + 收紧 Newton 容差」独立验证，相对误差
            #   ≤3e-3 且随 ε 减小而减小（真误差 ~1e-4）：
            #     tol=1e-3,ε=2e-2: Arkkio 6.3e-2 / flux 1.3e-2 / VD 1.3e-2 / ArkBE 2.6e-2
            #     tol=1e-12,ε=1e-3:          7.4e-4 /      4.7e-4 /    4.5e-4 /   2.9e-3
            #     tol=1e-12,ε=1e-4:          3.0e-4 /      2.7e-4 /    2.8e-4 /   1.4e-3
            #   （`tools/probe_fd_torque_adjoint.py --tol 1e-12 --eps 1e-4`）
            #
            # ⚠ 两处曾把我误导向"它错了"的**口径陷阱**（记下来免得重犯）：
            #   ① **随机 ρ 的 FD**：`ρ~U(0.05,0.95)` ⇒ `ρ₁=project(ρ)` 跑到 1e-6
            #      ⇒ `dγ/dρ=−H′/ρ₁²·SIMP′` 爆到 1e14 ⇒ 假"错 1e6 倍"。**必须用
            #      生产分布 ρ**（`density_from_x`）。
            #   ② **"固定 B"口径的 FD**：`tests/autograd/test_flux_oracle.py` 的 oracle
            #      把 `B_frozen` 当**常数输入** ⇒ 它 = ①显式项 + ③经冻结线性解的链，
            #      **不含 ②（γ 经主场 B 的链）**；而生产手写是 ①+②。**两者测的不是
            #      同一个量** ⇒ 那个"max_rel=1431"是**口径差异**，不是缺陷。
            #      ⇒ 判生产路径必须用**全链** FD（重解主场）。
            #   ③ 全链 FD 的 ε 不能小到被 `solve_abs_err=1e-3` 的噪声淹没
            #      （噪声 ∝1/ε）⇒ 探针已把容差做成参数 `solve_abs_err`。
            dT_flux = dT_flux * dgamma_p_drho

            return (t, T_arkkio, dT_arkkio, A, B, frozen.psi, frozen.torque,
                    dT_flux, T_vd, dT_vd, T_arkbe, dT_arkbe)
        finally:
            close = getattr(factor_jt, "close", None)
            if close is not None:
                close()
    finally:
        if owned:
            session.close()


def em_segment(
    rho: np.ndarray,
    pre: Preassembly,
    top: FieldTopology,
    material: FieldMaterial,
    *,
    A_node_last: Dict[int, np.ndarray],
    n_jobs: Optional[int] = None,
    solver: Optional[str] = None,
    time_idx: Optional[List[int]] = None,
    solve_abs_err: float = 1e-3,
    residual_tol: Optional[float] = None,
) -> List[tuple]:
    """一次 fun 调用的完整 EM 段（时间点，带暖启动）。

    对应 ``MMA10:3527-3564`` 的 EM 任务段。``A_node_last`` 是跨调用缓存
    （legacy 的模块级 dict ``:2910``），本函数原地写回 —— 调用方持有它。

    Parameters
    ----------
    n_jobs : Optional[int]
        阶段 8 的 8c 并行度。``None`` = 串行（**默认**，golden 对齐路径，数值
        与 legacy 逐位一致）；整数 >1 = 用 ``ThreadPoolExecutor`` 并行跑时间点
        （任务并行：每个时间点独立的 Newton + 伴随解，互不共享可变状态 —— 只
        读各自的 ``pre.*[i_t]``、写各自的 ``A_node_last[i_t]`` 槽，线程安全）。
        ⚠ 并行时**自动**设 ``OMP/OPENBLAS/MKL_NUM_THREADS=1`` 并在结束时恢复
        —— 任务并行维度下不给单次求解开 BLAS 线程（避免超订，设计文档
        §阶段 8 的并行项：legacy 的 ``n_jobs=14`` vs ``N_point_simulation=11``
        错配 + ``OMP_NUM_THREADS`` 未设 ⇒ 超订）。
    time_idx : Optional[List[int]]
        2026-10-11 新增：只跑这些时间索引（Adam 随机子采样 / 验证集评估用）。
        ``None``（默认）= 全部时间点，路径与 legacy 逐位一致。返回的 records
        按 ``time_idx`` 的**顺序**排列（即 ``records[j]`` 对应
        ``time_idx[j]``）。

    Returns
    -------
    ``records`` = ``em_step`` 的 10 元组列表（长度 = 全部时间点或 ``len(time_idx)``）。
    """
    pole_pairs = pre.cfg.geometry.pole_pairs
    parts = pre.cfg.geometry.parts
    if time_idx is None:
        idx_list = list(range(pre.time_span.size))
    else:
        idx_list = list(time_idx)
    n_t = len(idx_list)

    def _run(i_t: int, sess) -> tuple:
        a0 = A_node_last.get(i_t)
        kv = pre.K_vectorized_vec[i_t]
        elim = _cached_em_elimination(
            i_t, kv, top, pre.cfg.solver.EM_MS_boundary_A_eq)
        return em_step(
            pre.time_span[i_t], i_t, rho, kv,
            pre.R_vec[i_t], pre.Rd_vec[i_t], pre.Rq_vec[i_t],
            tuple(pre.currents[i_t]), pre.Torque_VD_coef_vec[i_t], pre.rotor_positions[i_t],
            a0, top=top, material=material, pole_pairs=pole_pairs, parts=parts,
            solver=sess, elimination=elim,
            solve_abs_err=solve_abs_err, residual_tol=residual_tol,
        )

    # 求解会话：串行用**进程级单例**（pardiso 的 MKL 实例跨 fun 调用只初始化
    # 一次 —— 2026-09-24 singleton）；并行时每个任务独立构造（会话非线程安全，
    # MKL 句柄不能跨线程共享）。mtype 来自配置（2 = 对称，消元后可用）。
    mtype = pre.cfg.solver.pardiso_mtype
    if n_jobs is None or n_jobs <= 1:
        session, owned = solvers.as_session(solver, mtype, shared=True)
        records = [_run(i_t, session) for i_t in idx_list]
    else:
        _set_blas_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
                def run_one(i_t):
                    sess, own = solvers.as_session(solver, mtype, shared=False)
                    try:
                        return _run(i_t, sess)
                    finally:
                        if own:
                            sess.close()

                records = list(ex.map(run_one, idx_list))
        finally:
            _restore_blas_threads()
    for j, i_t in enumerate(idx_list):
        A_node_last[i_t] = records[j][3]
    return records
