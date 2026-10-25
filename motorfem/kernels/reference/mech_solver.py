"""力学系统：装配族 / 残差 / 求解 / 灵敏度（内核 16）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）::

    get_Kmech_vectorized()                 :2604-2814   ← 装配族（21 个产出）
    mech_field_error_vectorized()          :2817-2858   ← 残差 (R, K)
    compute_udiff_vectorized()             :2861-2897   ← 灵敏度 (K, dK/dθ·u − dR/dθ)
    solve_mech_field_vectorized()          :2900-2903   ← 求解 u

数学出处：``磁障设计/弹性力学与有限元.md``（「mech_solver_rotor4」一族）。

⚠ **记号**：本模块与 legacy 一致地用 ``θ`` 表示**密度**（不是角度）——
理论文档也是这样写的（``∂γ/∂θ`` 里的 θ 就是设计变量）。所以 ``dth`` 读作 ``dρ``。
``σ``/``ε`` 才是应力/应变。

------------------------------------------------------------------------------
一、这个内核在解什么
------------------------------------------------------------------------------

**平面应力线弹性**（材料与单元矩阵是内核 5 的 ``mech_element_matrices``）::

    K(ρ)·u = −R(ρ)

``K`` 是 ``(2·N_node_mech, 2·N_node_mech)``（每节点 2 个笛卡尔自由度），
``R`` 有三支：

1. **体积力**（离心）:math:`R^{body}`：逐单元 ``T_eᵀ[(Rmech10_e·krho_e)·f_e ⊗ 𝟙₃]``；
2. **边界面力**:math:`R^{bc}`（legacy 的 ``_R2``）：逐加载线元在两端点各加
   ``Rmech20_e·(F_x, F_y)``；
3. **约束清零**：Dirichlet 节点的两个分量置零。

**没有非线性**——所以「Jacobian」就是 ``K`` 本身，求解是一次 ``spsolve``
（对比内核 9/10/11 的 Newton）。这是设计与文档 §2 那张表里「电磁难、机械易」的第一条。

------------------------------------------------------------------------------
二、三条约束机制（本内核最容易读错的地方）
------------------------------------------------------------------------------

**(1) 主从旋转消元**（``:2651-2682``、``:2763-2773``）

从属节点 ``s`` 与主动节点 ``m`` 的位移满足（**笛卡尔分量**上的旋转）::

    u_s = R(2π/parts)·u_m ,   R(a) = [[cos a, −sin a], [sin a, cos a]]

legacy 的做法**不是**在矩阵上加约束行就完事，而是**把从属自由度换成主动自由度**：
单元局部节点表里，从属节点被**替换**成它的主动节点（``_node``），
同时 6×6 变换 ``T_e`` 在该节点的 2×2 对角块上放 ``R``::

    T_e = blkdiag(R or I, R or I, R or I)     （逐局部节点：从属取 R、其余取 I）
    K_e ← T_eᵀ K_e T_e ,   DOF 位置 ← 主动节点的自由度

于是装配出来的 ``K`` 在主动自由度上是奇异的（从属自由度不再出现），
由 ``K_boundary`` 补上约束行::

    K_boundary[2s, 2s] = 1 ,  K_boundary[2s, 2m : 2m+2] = −R[0, :]
    K_boundary[2s+1, 2s+1] = 1 ,  K_boundary[2s+1, 2m : 2m+2] = −R[1, :]

即 ``u_s − R·u_m = 0`` ✓（与 ``boundaries.py`` 第五节记的
``u_slave = R(+2π/parts)·u_master`` 一致）。

⚠ ``_r11.._r22`` 在 legacy 用 ``cos(0 − 2π/parts)`` 等四个式子写出（``:2653-2656``），
展开后**正是** ``R(+2π/parts)``：``_r11 = cos a``、``_r12 = −sin a``、
``_r21 = +sin a``、``_r22 = cos a``。照抄式子、不要「化简」。

**(2) 从属的二次判定用的 ``_node`` 是「替换后」的节点**（``:2737-2749``）

这两组清零::

    if _node[s] in node_mech_constraint:  K0 的第 s 个 2×2 块置零
    if _node[s] in node_mech_slave:       K0 的第 s 个 2×2 块置零

⚠ ``_node`` 在上一段已经被**替换**过了，所以对 ``s`` 本身是从属节点的情形，
``_node[s]`` 是**主动**节点 —— 第二组条件检查的是「主动节点自己也是从属节点」
（链式从属）。这个语义很绕，**照搬**，并由测试固化。

**(3) Dirichlet 约束**（``:2777-2784``）

``K_boundary`` 的对应行先整行置零、再放单位元（``:2781-2784`` 的 ``.lil`` 赋值
是**后写覆盖**）；``_R2`` 的对应分量置零。⇒ ``u = 0`` 在那两个分量上。

⚠ 顺序是「先主从、后 Dirichlet」（``:2763`` 在前、``:2777`` 在后），且 Dirichlet
是**整行清零**，所以两套机制作用在同一节点上时 **Dirichlet 赢**。本模块保持这个顺序。

------------------------------------------------------------------------------
三、21 个产出与「位置一次、值每次重算」的设计
------------------------------------------------------------------------------

legacy 把几何量固化成矩阵位置、把材料相关量留成每次重算的向量
（``同步磁阻电机磁障结构分析.md``「静磁场计算的向量化数值实现」的同一思想）::

    K = coo( K0_data · (Kconst_map @ kE),  (K0_row, K0_col) ).tocsr() + K_boundary
    R = coo( R0_data · (Rconst_map @ krho), (R0_row, R0_col) ).toarray().flatten() + R2

``kE``/``krho`` 来自 ``mech_topo_opti(ρ)``，``dkE``/``dkrho`` 来自 ``dmech_topo_opti(ρ)``
（灵敏度用）。**这四个量就是内核 2 的 ``SIMPProjection``**（力学参数
``β=8, thb=0.65, power=3, min=1e-4``）—— 逐项对应见 ``MECH_SIMP_PARAMS`` 的 docstring。

⚠ **一处在 §7 主判据上不严格成立的地方**（已登记进设计文档 §11⑩）：
``SIMPProjection.derivative`` 相对 legacy 的 ``SIMP_dfunc`` 是**代数等价但非逐位**的
改写（1~2 ULP，是为了避开 legacy 的 ``/sinh(β)`` 在 β→0 时的除零）。所以凡涉及
``dkE`` 的 parity 测试只能用 ``rtol``。**本模块有意复用内核 2，不另写一份 legacy 形式
的导数** —— 同一份数学两种写法比 1~2 ULP 更糟。

------------------------------------------------------------------------------
四、与相邻内核的接口
------------------------------------------------------------------------------

- **上游**：内核 5 的 ``MechElements``（``K``/``L``/``D``/``R0``/``area``，载荷另见
  ``mech_body_force``/``mech_line_loads``）；内核 6 的 ``pair_master_slave``（力学侧
  逐字同构，可直接用）；内核 2 的 ``SIMPProjection``。
- **下游**：内核 17 消费 ``stress_data``/``stress_k_e_map``/``von_mises_map``
  （逐单元应力与 von Mises 的位置矩阵）与 ``compute_udiff`` 的产出。
- **死代码**（设计文档 §11④）：``node_mech_master``（转 dict 后）、
  ``element_mech_slave``、``element_S_mech_slave``（``:419-436``）**零消费者**，
  本模块**不搬运**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from ... import solvers
from .simp import SIMPParams, SIMPProjection

__all__ = [
    "MECH_SIMP_PARAMS",
    "MechOperators",
    "mech_simp",
    "mech_density_factors",
    "mech_density_derivatives",
    "boundary_part",
    "mech_operators",
    "assemble_mech_system",
    "mech_residual",
    "solve_mech_field",
    "mech_sensitivity",
]

#: 力学侧的 SIMP 参数。legacy ``MMA10:130-133``::
#:
#:     p_SIMP_mech = 3 ; min_SIMP_mech = 1e-4
#:     SIMP_beta_mech = 8 ; SIMP_thb_mech = 0.65
#:
#: ⚠ 与电磁侧（``:124-128``，``thb = 0.5``、``min = 1e-6``）**不同** ——
#: ``thb`` 与 ``min`` 两处都不一样，别把电磁的参数搬过来。
MECH_SIMP_PARAMS = SIMPParams(
    beta=8.0, thb=0.65, power=3.0, min_density=1e-4, linear_factor=0.0,
)


def mech_simp() -> SIMPProjection:
    """力学侧 SIMP 投影器（``MECH_SIMP_PARAMS``）。

    不用模块级单例：``mech_topo_opti``/``dmech_topo_opti`` 在 legacy 里是**函数**
    （每次调用重算 ``tanh``），这里保持同样的调用形态。
    """
    return SIMPProjection(MECH_SIMP_PARAMS)


def mech_density_factors(rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``mech_topo_opti(ρ)`` → ``(kE, krho)``。对应 ``MMA10:1231-1240``。

    ``kE`` 是**刚度**的密度因子（用于 ``K``），``krho`` 是**载荷**的密度因子（用于 ``R``）::

        kE   = mech_simp().project(ρ)          ← SIMP 投影
        krho = ρ                               ← 恒等（legacy ``krho = _rho``）

    ⚠ ``krho`` 在 legacy 里就是 ``_rho`` **本身**（不是副本）。本函数返回同一个对象，
    以保持「载荷与密度线性」这一支的可核对性；调用方不要就地改写它。
    """
    return mech_simp().project(rho), rho


def mech_density_derivatives(rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``dmech_topo_opti(ρ)`` → ``(dkE, dkrho)``。对应 ``MMA10:1242-1255``。

    ``dkrho`` 在 legacy 是 ``np.ones(shape)``（密度对自身的导数）—— 所以
    ``Rconst_map @ dkrho`` 恒为 1，``dR/dθ`` 就**恰是** ``R0_data`` 本身。
    """
    return mech_simp().derivative(rho), np.ones_like(np.asarray(rho, dtype=float))


# =============================================================================
# 装配族（legacy ``get_Kmech_vectorized``）
# =============================================================================

@dataclass(frozen=True)
class MechOperators:
    """力学装配族的 21 个产出。

    分成三组：刚度、载荷、应力（应力那一组是**内核 17** 的输入）。

    Attributes
    ----------
    k0_data, k0_row, k0_col : (N_elem·36,) ndarray
        逐单元 ``T_eᵀ K_e T_e`` 展开成三元组，按 (单元, 局部行, 局部列) 排列。
        **同一个全局 ``(row, col)`` 会在不同单元上重复** —— 这是有意的：
        下游 ``coo((k0_data·(Kconst_map@kE), (k0_row, k0_col)))`` 靠 COO 的重复项
        求和完成装配。
        ⚠ 被约束/从属节点的 2×2 块已被**置零**（见模块 docstring 第二节第 2 条），
        所以这里有**显式零**。
    kconst_map : (N_elem·36, N_elem) csr（整数 1）
        ``Kconst_map @ kE`` 把每个展开项映射到它所属单元的 ``kE``。
    k_u_map : (N_elem·36, 2·N_node) csr（整数 1）
        ``K_u_map @ u`` 的第 ``(e, 局部行 r, 局部列 c)`` 行给出 ``u[DOF(e, r, c)]``，
        于是 ``k0_data · (k_u_map @ u)`` 逐项就是 ``(TᵀK_eT·u_e)`` 的分量。
    k_boundary : (2·N_node, 2·N_node) csr
        约束行（主从旋转 + Dirichlet）。见模块 docstring 第二节。
    col_mask : (2·N_node, 2·N_node) csr
        **对称 Dirichlet 列投影（2026-09-24 新增）**：单位阵把**齐次 Dirichlet
        列**（``constrained`` 节点的两个分量）清零。``K = (coo @ col_mask) +
        k_boundary``。只零 Dirichlet 列 —— 力学主从的 2×2 旋转块本就对称（实测
        K_mech 4.3% 不对称 100% 来自 Dirichlet 列；只零 Dirichlet 列 → 0.000000，
        max|Δu| = 7.7e-20，解不变）。
    r0_data, r0_row, r0_col : (N_elem·6,) ndarray
        逐单元体积力 ``T_eᵀ[fx, fy]×3·Rmech10_e`` 展开成三元组。
        ⚠ **``r0_data`` 与 ``R0_data`` 不是同一个东西**：legacy 的 ``_R0_map`` 用
        未清零的版本、``_R0_data`` 用清零后的版本；本模块只保留**清零后**的
        ``r0_data``（``_R0_map`` 零消费者，见 Notes）。
    rconst_map : (N_elem·6, N_elem) csr（整数 1）
    r2 : (2·N_node,) ndarray
        直接叠加的载荷项：加载线元的面力 + 约束节点的清零（``:2751-2784``）。
    dk_p_dth_u_row, dk_p_dth_u_col : (N_elem·36,) ndarray
        ``(全局自由度, 单元)`` —— ``∂K/∂θ_e`` 收缩 ``u`` 之后每个分量的位置。
    dr_p_dth_row, dr_p_dth_col : (N_elem·6,) ndarray
        ``(全局自由度, 单元)`` —— ``∂R/∂θ_e`` 每个分量的位置。
    stress_data, stress_row, stress_col : (N_elem·18,) ndarray
        逐单元 ``D·L_e·T_e``（``:2708`` 的 ``_K0_stress``）展开成三元组，
        行为 ``3e + 分量``、列为全局自由度。
    stress_k_e_map : (N_elem·3, N_elem) csr（整数 1）
        ``stress_k_e_map @ kE`` 把每个应变分量映射到它单元的 ``kE``。
    von_mises_map : (N_elem·3, N_elem·3) csr
        逐单元的 von Mises 二次型 ``[[1, −1/2, 0], [−1/2, 1, 0], [0, 0, 3]]``。

    Notes
    -----
    legacy 的 ``get_Kmech_vectorized`` 还返回一个 ``_R0_map``（``(2N, N_elem)``，
    未清零的体积力映射）与四个只被注释掉的中间量（``_K_u_map`` 之外的三个）。
    ``_R0_map`` 的调查结论：**零消费者**（消费点 ``:2885`` 被注释掉了）——
    本模块不搬运它（设计文档 §11④ 的同一处理：只搬有消费者的东西）。
    """

    k0_data: np.ndarray
    k0_row: np.ndarray
    k0_col: np.ndarray
    kconst_map: object
    k_u_map: object
    k_boundary: object
    col_mask: object
    r0_data: np.ndarray
    r0_row: np.ndarray
    r0_col: np.ndarray
    rconst_map: object
    r2: np.ndarray
    dk_p_dth_u_row: np.ndarray
    dk_p_dth_u_col: np.ndarray
    dr_p_dth_row: np.ndarray
    dr_p_dth_col: np.ndarray
    stress_data: np.ndarray
    stress_row: np.ndarray
    stress_col: np.ndarray
    stress_k_e_map: object
    von_mises_map: object
    #: 力学主从配对（2026-09-24 消元用；构造时已传入，补存便于组消元算子）
    slave_to_master: Optional[Mapping[int, int]] = None

    @property
    def n_elem(self) -> int:
        return int(self.kconst_map.shape[1])

    @property
    def n_dof(self) -> int:
        return int(self.k_boundary.shape[0])


def _slave_rotation(parts: int) -> np.ndarray:
    """主从旋转矩阵 ``R(+2π/parts)``，2×2。对应 ``MMA10:2653-2656``。

    legacy 用 ``cos(0 − 2π/parts)`` 等四个式子写出，展开后正是 ``R(+2π/parts)``。
    本函数直接给 ``R``，四个元素的**符号**与 legacy 一致（测试逐位比对）。
    """
    a = 2 * np.pi / parts
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def boundary_part(
    n_node: int,
    rot: np.ndarray,
    slave: Mapping[int, int],
    constrained: Iterable[int],
    line_tri: np.ndarray,
    loaded_lines: Sequence[int],
    r2_elem: np.ndarray,
    boundary_fx: np.ndarray,
    boundary_fy: np.ndarray,
    r2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``r2`` 与 ``k_boundary`` 的非零元。对应 ``MMA10:2751-2784``。

    **两层共用**（``reference`` 与 ``fast``）：这一段是逐**节点**的循环、与单元循环
    无关，向量化收益为零，且它的语义（``+=`` 累加 + 「整行后写覆盖」）在共享一份
    实现时最不容易被改坏。

    Parameters
    ----------
    n_node : int
    rot : (2, 2) ndarray
        ``R(+2π/parts)``。
    slave, constrained : 见 ``mech_operators``
    line_tri, loaded_lines, r2_elem, boundary_fx, boundary_fy :
        见 ``mech_operators``。
    r2 : (2·n_node,) ndarray
        **会被就地修改**（加载线元的 ``+=`` 与约束节点的清零）。

    Returns
    -------
    r2 : 同一个数组（便于链式调用）
    rows, cols, vals : k_boundary 的三元组

    Notes
    -----
    ``line_tri`` 上按 ``loaded_lines`` 的顺序作 ``+=`` —— 与 legacy 的
    ``_R2[_index] += ...`` 同序（花式索引的 ``+=`` 会累加重复项）。
    """
    # ---- 加载线元（:2751-2759）----
    for i_load, i_line in enumerate(loaded_lines):
        node_i, node_j = int(line_tri[int(i_line), 0]), int(line_tri[int(i_line), 1])
        index = [2 * node_i, 2 * node_i + 1, 2 * node_j, 2 * node_j + 1]
        values = (np.array([boundary_fx[i_load], boundary_fy[i_load]] * 2)
                  * r2_elem[int(i_line)])
        for k, d in enumerate(index):
            r2[d] += values[k]

    # ---- 约束行（:2761-2784）：先主从、后 Dirichlet ----
    # legacy 在 ``lil`` 上先写主从行、再对 Dirichlet 行**整行清零**后写单位元
    # （赋值是后写覆盖）⇒ Dirichlet 赢。用「主从条目 + 被覆盖的行集合」表示同一语义。
    slave_entries = {}
    for i_node_s, i_node_m in slave.items():
        slave_entries[(2 * i_node_s, 2 * i_node_s)] = 1.0
        slave_entries[(2 * i_node_s, 2 * i_node_m)] = -rot[0, 0]
        slave_entries[(2 * i_node_s, 2 * i_node_m + 1)] = -rot[0, 1]
        slave_entries[(2 * i_node_s + 1, 2 * i_node_s + 1)] = 1.0
        slave_entries[(2 * i_node_s + 1, 2 * i_node_m)] = -rot[1, 0]
        slave_entries[(2 * i_node_s + 1, 2 * i_node_m + 1)] = -rot[1, 1]
        r2[2 * i_node_s] = 0.0
        r2[2 * i_node_s + 1] = 0.0

    constraint_rows = set()
    for i_node in constrained:
        constraint_rows.add(2 * i_node)
        constraint_rows.add(2 * i_node + 1)
        r2[2 * i_node] = 0.0
        r2[2 * i_node + 1] = 0.0

    bnd = dict((k, v) for k, v in slave_entries.items() if k[0] not in constraint_rows)
    for row in constraint_rows:
        bnd[(row, row)] = 1.0

    b_keys = sorted(bnd)
    return (
        r2,
        np.array([k[0] for k in b_keys], dtype=np.int64),
        np.array([k[1] for k in b_keys], dtype=np.int64),
        np.array([bnd[k] for k in b_keys], dtype=float),
    )


def mech_operators(
    node: np.ndarray,
    tri: np.ndarray,
    *,
    k_elem: np.ndarray,
    l_coef: np.ndarray,
    d_mat: np.ndarray,
    r0_elem: np.ndarray,
    r2_elem: np.ndarray,
    body_fx: np.ndarray,
    body_fy: np.ndarray,
    line_tri: np.ndarray,
    loaded_lines: Sequence[int],
    boundary_fx: np.ndarray,
    boundary_fy: np.ndarray,
    slave_to_master: Mapping[int, int],
    constrained_nodes: Iterable[int],
    parts: int,
) -> MechOperators:
    """力学装配族的 21 项。对应 ``MMA10:2604-2814``。

    Parameters
    ----------
    node : (N_node_mech, 2) ndarray
        力学子网格的节点坐标（**不是**电磁网格的 ``node`` 表）。
    tri : (N_elem, 3) ndarray
        力学单元连接表。
    k_elem : (N_elem, 6, 6) ndarray
        内核 5 的 ``MechElements.K``（单元刚度 ``Δ·LᵀDL``）。
    l_coef : (N_elem, 3, 6) ndarray
        内核 5 的 ``MechElements.L``。
    d_mat : (3, 3) ndarray
        内核 5 的 ``MechElements.D``（平面应力，**只存一份**）。
    r0_elem : (N_elem,) ndarray
        内核 5 的 ``MechElements.R0``（体积力系数 ``−Δ/2``，见 D1）。
    r2_elem : (N_line,) ndarray
        ``Rmech20elem``（线元系数 ``−L/2``），来自 ``mech_line_loads``。
    body_fx, body_fy : (N_elem,) ndarray
        离心体积力 ``fmechloadx/y``，来自 ``mech_body_force``。
    line_tri : (N_line, 2) ndarray
        全部线元的端点节点号。
    loaded_lines : Sequence[int]
        加载线元的行号（legacy 的 ``element_mech_L_load``）。
    boundary_fx, boundary_fy : (N_load,) ndarray
        边界面力 ``Fmechloadx/y``（已归一化），来自 ``mech_line_loads``。
    slave_to_master : Mapping[int, int]
        主导/从属配对（**力学子网格**的节点号）。内核 6 的
        ``pair_master_slave(node, parts=parts, coord_err_abs=...)`` 可直接用
        —— 力学侧的判定与电磁侧**逐字同构**（``MMA10:402-419``）。
    constrained_nodes : 可迭代 int
        Dirichlet 节点（位移约束）。
    parts : int
        旋转对称扇区数。

    Returns
    -------
    MechOperators

    Notes
    -----
    逐单元做四件事（与 legacy 同序）：取 ``T_e`` 与（替换后的）DOF 表、写体积力项、
    写刚度与灵敏度的位置、写应力位置。随后两组边界处理：加载线元 → ``r2``，
    主从 → Dirichlet → ``k_boundary``。

    ⚠ ``r0_data`` 的**清零**与 ``k0_data`` 的**清零**用的是同一张（替换后的）
    ``node_of_local`` 表，但前者只清 Dirichlet、后者清 Dirichlet **与**从属
    （``:2692-2697`` vs ``:2737-2749``）—— 这个不对称是 legacy 的形态，照搬。
    """
    node = np.asarray(node, dtype=float)
    tri = np.asarray(tri, dtype=int)
    line_tri = np.asarray(line_tri, dtype=int)
    n_elem = tri.shape[0]
    n_node = node.shape[0]
    rot = _slave_rotation(parts)
    constrained = set(int(n) for n in constrained_nodes)
    slave = dict((int(k), int(v)) for k, v in slave_to_master.items())

    n_k0 = n_elem * 36
    n_r0 = n_elem * 6

    k0_data = np.zeros(n_k0)
    k0_row = np.zeros(n_k0, dtype=np.int64)
    k0_col = np.zeros(n_k0, dtype=np.int64)
    k_u_map_rows, k_u_map_cols = [], []
    kconst_rows, kconst_cols = [], []

    r0_data = np.zeros(n_r0)
    r0_row = np.zeros(n_r0, dtype=np.int64)
    r0_col = np.zeros(n_r0, dtype=np.int64)
    rconst_rows, rconst_cols = [], []

    dk_row = np.zeros(n_k0, dtype=np.int64)
    dk_col = np.zeros(n_k0, dtype=np.int64)
    dr_row = np.zeros(n_r0, dtype=np.int64)
    dr_col = np.zeros(n_r0, dtype=np.int64)

    stress_data = np.zeros(n_elem * 18)
    stress_row = np.zeros(n_elem * 18, dtype=np.int64)
    stress_col = np.zeros(n_elem * 18, dtype=np.int64)
    stress_k_e_rows, stress_k_e_cols = [], []

    von_rows, von_cols, von_data = [], [], []

    for e in range(n_elem):
        nodes = [int(tri[e, 0]), int(tri[e, 1]), int(tri[e, 2])]

        # ---- 变换矩阵 T_e 与（从属替换后的）自由度表 ----
        T = np.eye(6)
        dof = [2 * nodes[0], 2 * nodes[0] + 1,
               2 * nodes[1], 2 * nodes[1] + 1,
               2 * nodes[2], 2 * nodes[2] + 1]
        node_of_local = list(nodes)
        for s in range(3):
            if nodes[s] in slave:
                T[2 * s:2 * s + 2, 2 * s:2 * s + 2] = rot
                master = slave[nodes[s]]
                dof[2 * s:2 * s + 2] = [2 * master, 2 * master + 1]
                node_of_local[s] = master

        # ---- 体积力（:2684-2697）----
        body = np.array([body_fx[e], body_fy[e]] * 3) * r0_elem[e]
        load_e = T.T @ body
        r0_data[6 * e:6 * e + 6] = load_e
        for s in range(3):
            if node_of_local[s] in constrained:
                r0_data[6 * e + 2 * s:6 * e + 2 * s + 2] = 0.0

        r0_row[6 * e:6 * e + 6] = dof
        rconst_rows.extend(6 * e + i for i in range(6))
        rconst_cols.extend([e] * 6)
        dr_row[6 * e:6 * e + 6] = dof
        dr_col[6 * e:6 * e + 6] = e

        # ---- 刚度与灵敏度的位置（:2707-2721、:2734-2749）----
        J = T.T @ k_elem[e] @ T
        stress_e = d_mat @ l_coef[e] @ T

        for i_r in range(6):
            base = 36 * e + 6 * i_r
            k0_data[base:base + 6] = J[i_r, :]
            k0_row[base:base + 6] = dof[i_r]
            k0_col[base:base + 6] = dof
            k_u_map_rows.extend(range(base, base + 6))
            k_u_map_cols.extend(dof)
            kconst_rows.extend(range(base, base + 6))
            kconst_cols.extend([e] * 6)
            dk_row[base:base + 6] = dof[i_r]
            dk_col[base:base + 6] = e

        for i_r in range(3):
            base = 18 * e + 6 * i_r
            stress_data[base:base + 6] = stress_e[i_r, :]
            stress_row[base:base + 6] = 3 * e + i_r
            stress_col[base:base + 6] = dof
        stress_k_e_rows.extend(range(3 * e, 3 * e + 3))
        stress_k_e_cols.extend([e] * 3)

        von = np.zeros((3, 3))
        von[:, 0] = [1.0, -1.0 / 2.0, 0.0]
        von[:, 1] = [-1.0 / 2.0, 1.0, 0.0]
        von[:, 2] = [0.0, 0.0, 3.0]
        for k in range(3):
            for l in range(3):
                if von[k, l] != 0.0:
                    von_rows.append(3 * e + k)
                    von_cols.append(3 * e + l)
                    von_data.append(von[k, l])

        # ---- 约束/从属的 2×2 块清零（:2737-2749）----
        for s in range(3):
            if node_of_local[s] in constrained or node_of_local[s] in slave:
                k0_data[36 * e + 12 * s:36 * e + 12 * s + 12] = 0.0

    # ---- 载荷与约束行：与 fast 层共用同一份实现 ----
    r2 = np.zeros(2 * n_node)
    r2, b_rows, b_cols, b_vals = boundary_part(
        n_node, rot, slave, constrained, line_tri, loaded_lines, r2_elem,
        boundary_fx, boundary_fy, r2,
    )

    # 对称 Dirichlet（2026-09-24）：齐次 u=0 ⇒ 行换 e_i 之外还要**零列**。
    # 只零 Dirichlet 列（力学主从 2×2 旋转块本就对称，实测 0 贡献；K 严格对称、解不变）。
    col_mask = sp_sparse.identity(2 * n_node).tolil()
    for i_node in constrained:
        col_mask[:, 2 * i_node] = 0.0
        col_mask[:, 2 * i_node + 1] = 0.0
    col_mask = col_mask.tocsr()

    return MechOperators(
        k0_data=k0_data, k0_row=k0_row, k0_col=k0_col,
        kconst_map=sp_sparse.csr_matrix(
            (np.ones(len(kconst_rows)), (kconst_rows, kconst_cols)),
            shape=(n_k0, n_elem)),
        k_u_map=sp_sparse.csr_matrix(
            (np.ones(len(k_u_map_rows)), (k_u_map_rows, k_u_map_cols)),
            shape=(n_k0, 2 * n_node)),
        k_boundary=sp_sparse.csr_matrix(
            (b_vals, (b_rows, b_cols)), shape=(2 * n_node, 2 * n_node)),
        col_mask=col_mask,
        r0_data=r0_data, r0_row=r0_row, r0_col=r0_col,
        rconst_map=sp_sparse.csr_matrix(
            (np.ones(len(rconst_rows)), (rconst_rows, rconst_cols)),
            shape=(n_r0, n_elem)),
        r2=r2,
        dk_p_dth_u_row=dk_row, dk_p_dth_u_col=dk_col,
        dr_p_dth_row=dr_row, dr_p_dth_col=dr_col,
        stress_data=stress_data, stress_row=stress_row, stress_col=stress_col,
        stress_k_e_map=sp_sparse.csr_matrix(
            (np.ones(len(stress_k_e_rows)), (stress_k_e_rows, stress_k_e_cols)),
            shape=(3 * n_elem, n_elem)),
        von_mises_map=sp_sparse.csr_matrix(
            (np.asarray(von_data), (np.asarray(von_rows, dtype=np.int64),
                                    np.asarray(von_cols, dtype=np.int64))),
            shape=(3 * n_elem, 3 * n_elem)),
        slave_to_master=dict(slave_to_master) if slave_to_master else None,
    )


# =============================================================================
# 残差与求解
# =============================================================================

def assemble_mech_system(
    rho: np.ndarray,
    node: np.ndarray,
    tri: np.ndarray,
    *,
    k_elem: np.ndarray,
    r0_elem: np.ndarray,
    body_fx: np.ndarray,
    body_fy: np.ndarray,
    r2: np.ndarray,
    k_boundary,
    slave_to_master: Mapping[int, int],
    constrained_nodes: Iterable[int],
    parts: int,
) -> Tuple[np.ndarray, sp_sparse.csr_matrix]:
    """**直接逐单元装配** ``(R, K)`` —— 本内核的权威数学表达。

    与 ``mech_residual`` 算的是同一个东西，但**不经过位置矩阵**：逐单元算
    ``T_eᵀK_eT_e`` 与 ``T_eᵀ(body)`` 后直接 scatter。它存在的理由是让
    ``reference/`` 真的写出公式，而不是让读者逆推位置矩阵
    （``kernels/README.md`` 的纪律）；测试里两条路线**逐位**比对。

    数学（``mech_solver_rotor4``；``同步磁阻电机磁障结构分析.md`` 的力学侧记号）::

        K = Σ_e  scatter(T_eᵀ K_e T_e, dof_e)          + K_boundary
            （对每个局部槽位 s，若替换后的节点 ∈ 约束集 ∪ 从属集，
              则把该 6×6 块的**第 2s、2s+1 两行**置零 —— 见下面 ⚠）
        R = Σ_e  scatter(T_eᵀ (Rmech10_e·krho_e·[f_e,f_e,f_e]), dof_e)
            （Dirichlet 节点的两个分量清零）
            + R2

    ⚠ **两处不同的清零**（legacy ``:2692-2697`` vs ``:2737-2749``，照搬）：

    - 体积力：只清 **Dirichlet**；
    - 刚度块：清 Dirichlet **与**「替换后的节点在从属集里」。

    后者的第二组条件对「槽位本身是从属节点」的情形检查的其实是**主动节点自己是否
    从属**（因为 ``node_of_local`` 已替换）—— 这与模块 docstring 第二节第 2 条是同一件事。

    两条清零都**只清行、不清列**（legacy 按 36 个局部行的偏移置零），所以 ``K``
    在结构上不对称 —— 这是 legacy 的形态，不是 bug。

    ⚠ 与 ``mech_residual`` 的**唯一**差别是累加顺序：这里是逐单元 scatter、
    那边是 ``coo(...).tocsr()`` 的重复项求和。两者在浮点上可能差末位 ——
    **这是 F4 那一族**（设计文档第二节），照搬 legacy 的路线是 ``mech_residual``。
    """
    node = np.asarray(node, dtype=float)
    tri = np.asarray(tri, dtype=int)
    n_elem = tri.shape[0]
    n_node = node.shape[0]
    rot = _slave_rotation(parts)
    constrained = set(int(n) for n in constrained_nodes)
    slave = dict((int(k), int(v)) for k, v in slave_to_master.items())

    k_e, k_rho = mech_density_factors(rho)

    rows, cols, vals = [], [], []
    R = np.zeros(2 * n_node)
    for e in range(n_elem):
        nodes = [int(tri[e, 0]), int(tri[e, 1]), int(tri[e, 2])]
        T = np.eye(6)
        dof = [2 * nodes[0], 2 * nodes[0] + 1,
               2 * nodes[1], 2 * nodes[1] + 1,
               2 * nodes[2], 2 * nodes[2] + 1]
        node_of_local = list(nodes)
        for s in range(3):
            if nodes[s] in slave:
                T[2 * s:2 * s + 2, 2 * s:2 * s + 2] = rot
                master = slave[nodes[s]]
                dof[2 * s:2 * s + 2] = [2 * master, 2 * master + 1]
                node_of_local[s] = master

        block = T.T @ (k_e[e] * k_elem[e]) @ T
        # 与 legacy ``:2737-2749`` 一致：这两类节点的 2×2 局部行置零（**只清行**）
        for s in range(3):
            if node_of_local[s] in constrained or node_of_local[s] in slave:
                block[2 * s:2 * s + 2, :] = 0.0

        for i_r in range(6):
            for i_c in range(6):
                rows.append(dof[i_r])
                cols.append(dof[i_c])
                vals.append(block[i_r, i_c])

        body = T.T @ (np.array([body_fx[e], body_fy[e]] * 3) * r0_elem[e])
        for s in range(3):
            if node_of_local[s] in constrained:
                body[2 * s:2 * s + 2] = 0.0
        for k, d in enumerate(dof):
            R[d] += body[k] * k_rho[e]

    K = sp_sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows, dtype=np.int64),
                            np.asarray(cols, dtype=np.int64))),
        shape=(2 * n_node, 2 * n_node),
    )
    # 对称 Dirichlet（2026-09-24）：列投影零齐次 Dirichlet 列（与 mech_residual 同款，
    # 保持两条路线逐位一致）。约束行的对角 1 由 k_boundary 自带，不被 col_mask 波及。
    col_mask = sp_sparse.identity(2 * n_node).tolil()
    for i_node in constrained:
        col_mask[:, 2 * i_node] = 0.0
        col_mask[:, 2 * i_node + 1] = 0.0
    K = K @ col_mask.tocsr() + sp_sparse.csr_matrix(k_boundary)
    R = R + np.asarray(r2, dtype=float)
    return R, K


def mech_residual(
    rho: np.ndarray, ops: MechOperators
) -> Tuple[np.ndarray, sp_sparse.csr_matrix]:
    """残差 ``(R, K)``。对应 ``MMA10:2817-2858``。

    ``legacy`` 的两行（位置矩阵路线）::

        R = coo(R0_data·(Rconst_map @ krho), (R0_row, R0_col)).toarray().flatten() + R2
        K = coo(K0_data·(Kconst_map @ kE),   (K0_row, K0_col)).tocsr()      + K_boundary

    ⚠ 两处必须照抄：``R`` 先装成 ``(2N, 1)`` 再 ``.toarray().flatten()``
    （不是直接装成 ``(2N,)``）；``K`` 是 ``coo(...).tocsr() @ col_mask + K_boundary``
    （COO 的重复项在 ``tocsr`` 里求和，**列投影在加约束矩阵之前** —— 对称 Dirichlet，
    2026-09-24；约束行的对角 1 由 ``k_boundary`` 自带，不被 ``col_mask`` 波及）。
    """
    k_e, k_rho = mech_density_factors(rho)
    n_dof = ops.n_dof

    R = sp_sparse.coo_matrix(
        (ops.r0_data * (ops.rconst_map @ k_rho), (ops.r0_row, ops.r0_col)),
        shape=(n_dof, 1),
    ).toarray().flatten() + ops.r2

    K = sp_sparse.coo_matrix(
        (ops.k0_data * (ops.kconst_map @ k_e), (ops.k0_row, ops.k0_col)),
        shape=(n_dof, n_dof),
    )
    K = K.tocsr() @ ops.col_mask + ops.k_boundary
    return R, K


def solve_mech_field(
    rho: np.ndarray, ops: MechOperators, solver: Optional[str] = None,
    elimination=None,
) -> np.ndarray:
    """求解位移 ``u``。对应 ``MMA10:2900-2903``。

    ``K u = −R``（一次 ``spsolve``，**没有迭代** —— 线弹性）。

    Notes
    -----
    legacy 写成 ``spsolve(_Kmech.tocsr(), -_Rmech)``；``mech_residual`` 返回的已是
    ``csr``，``.tocsr()`` 是 no-op（与内核 9 记过的同一件事），本实现省掉它。
    ``solver``：线性求解后端（``motorfem.solvers``），None = superlu（生产现状）。
    ``elimination``：主从消元算子（力学，``build_mech_elimination``）。消元模式
    下解 ``TᵀKT`` 并恢复 ``u = T·x_free``（K_sym 对称）。
    """
    session, owned = solvers.as_session(solver)
    try:
        R, K = mech_residual(rho, ops)
        K = K.tocsr()
        if elimination is not None:
            return elimination.restore(session.solve(
                elimination.eliminate_matrix(K),
                elimination.eliminate_rhs(-R)))
        return session.solve(K, -R)
    finally:
        if owned:
            session.close()


def mech_sensitivity(
    rho: np.ndarray, u_node: np.ndarray, ops: MechOperators
) -> Tuple[sp_sparse.csr_matrix, sp_sparse.csr_matrix]:
    """刚度与灵敏度 ``(K, dK/dθ·u + dR/dθ)``。对应 ``MMA10:2861-2897``。

    Returns
    -------
    K : (2N, 2N) csr
        与 ``mech_residual`` 的 ``K`` 同一个（重算一遍 —— legacy 如此）。
    dK_p_dth_u_plus_dR : (2N, N_elem) coo
        ``∂K/∂θ_e·u + ∂R/∂θ_e``，逐单元一列。

    Notes
    -----
    逐项::

        (∂K/∂θ_e·u) 的位置 = (全局自由度, e)，值 = K0_data · (Kconst_map @ dkE) · (K_u_map @ u)
        (∂R/∂θ_e)   的位置 = (全局自由度, e)，值 = R0_data · (Rconst_map @ dkrho)

    ``dkrho = 1`` ⇒ ``∂R/∂θ_e`` 就恰是 ``r0_data``。

    ⚠ ``dkE`` 来自内核 2 的 ``SIMPProjection.derivative``，它与 legacy 的
    ``SIMP_dfunc`` 是**代数等价但非逐位**（1~2 ULP，设计文档 §11⑩）⇒ 本函数的
    返回值与 legacy 的对齐只到 ``rtol``（``K`` 那一支仍是逐位）。

    ⚠ **★ D5 已修（2026-09-23，§12 修正清单第 1 条）**：legacy 返回
    ``∂K/∂θ_e·u − ∂R/∂θ_e``（``:2897``），而它自己的求解约定是 ``K u = −R``
    （``:2902``）⇒ 残差 ``F = Ku + R`` 的灵敏度应当是 ``∂K/∂θ_e·u **+** ∂R/∂θ_e``。
    legacy 的减号曾让 ``dU_p_drho`` 与 ``dALStress_p_drho`` 都带 ``2λᵀdR`` 的误差
    （实测 ``fd − legacy == 2·∂R/∂θ_e``）。本实现**翻号**为加号。
    """
    k_e, _ = mech_density_factors(rho)
    dk_e, dk_rho = mech_density_derivatives(rho)
    n_dof = ops.n_dof

    dR_p_dth = sp_sparse.coo_matrix(
        (ops.r0_data * (ops.rconst_map @ dk_rho), (ops.dr_p_dth_row, ops.dr_p_dth_col)),
        shape=(n_dof, ops.n_elem),
    )
    dK_p_dth_u = sp_sparse.coo_matrix(
        (ops.k0_data * (ops.kconst_map @ dk_e) * (ops.k_u_map @ u_node),
         (ops.dk_p_dth_u_row, ops.dk_p_dth_u_col)),
        shape=(n_dof, ops.n_elem),
    )
    K = sp_sparse.coo_matrix(
        (ops.k0_data * (ops.kconst_map @ k_e), (ops.k0_row, ops.k0_col)),
        shape=(n_dof, n_dof),
    )
    K = K.tocsr() + ops.k_boundary

    return K, dK_p_dth_u + dR_p_dth
    k_e, _ = mech_density_factors(rho)
    dk_e, dk_rho = mech_density_derivatives(rho)
    n_dof = ops.n_dof

    dR_p_dth = sp_sparse.coo_matrix(
        (ops.r0_data * (ops.rconst_map @ dk_rho), (ops.dr_p_dth_row, ops.dr_p_dth_col)),
        shape=(n_dof, ops.n_elem),
    )
    dK_p_dth_u = sp_sparse.coo_matrix(
        (ops.k0_data * (ops.kconst_map @ dk_e) * (ops.k_u_map @ u_node),
         (ops.dk_p_dth_u_row, ops.dk_p_dth_u_col)),
        shape=(n_dof, ops.n_elem),
    )
    K = sp_sparse.coo_matrix(
        (ops.k0_data * (ops.kconst_map @ k_e), (ops.k0_row, ops.k0_col)),
        shape=(n_dof, n_dof),
    )
    K = K.tocsr() + ops.k_boundary

    return K, dK_p_dth_u - dR_p_dth
