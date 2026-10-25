"""内核 16 的 fast 层：力学装配族的**真向量化**（legacy 是逐单元 Python 循环）。

数学、三条约束机制、21 个产出的含义与 D5 的登记写在
``reference/mech_solver.py`` 的模块 docstring 里，本文件不重复。

------------------------------------------------------------------------------
本层与既有内核的一个关键不同：legacy **没有**向量化版本可搬
------------------------------------------------------------------------------

前 6 个内核的 ``fast/`` 都是「把 legacy 已有的位置矩阵搬过来」。内核 7
（``moving_mesh``）第一次遇到「legacy 本身就是逐节点 ``for`` 循环」，
本内核是第二次：``get_Kmech_vectorized`` 是 6362 个单元上的 Python 循环 +
每单元 36 次数组切片赋值。所以本层是**新写的向量化**，按
``docs/重构设计文档.md`` §8「向量化补齐」的清单还债
（该清单点名了 ``get_Kmech_vectorized`` 的 lil 循环）。

**它因此多背一条纪律**（与内核 7 同）：必须与 legacy **逐位**一致，
**而且包括稀疏结构**（``indptr``/``indices``/``data``）—— 稠密比较看不出
「显式存储的零」。测试用 ``np.array_equal`` 逐项钉住六个 CSR 的三段。

------------------------------------------------------------------------------
哪些能向量化、哪些不
------------------------------------------------------------------------------

| 段 | 向量化 | 理由 |
|---|---|---|
| 单元循环（``T_e``/``dof``/``J``/应力/体积力/清零） | **是** | 6362 个单元的纯批量运算，且 legacy 的切片赋值是性能热点 |
| 加载线元 → ``r2`` | 否（**共用 reference**） | 逐线元的 ``+=`` 累加，向量化需要 ``np.add.at``（红线 5：除非验证加法顺序不变）；量小 |
| ``k_boundary`` | 否（**共用 reference**） | ``lil`` 的「整行后写覆盖」语义，向量化后反而更难核对 |
| ``mech_residual`` / ``solve_mech_field`` / ``mech_sensitivity`` | 否（**共用 reference**） | 全是稀疏矩阵运算，没有可向量化的部分（先例：内核 12 的 ``torque_quadratic``） |

------------------------------------------------------------------------------
向量化的三处「必须照抄」
------------------------------------------------------------------------------

1. **``J = Tᵀ K_e T`` 的结合顺序**：legacy 写 ``_T.T @ Kmech0elem[i_elem] @ _T``，
   Python 的 ``@`` 左结合 ⇒ ``(Tᵀ·K_e)·T``。批量形式写 ``(Tt @ k_elem) @ T``
   （**不是** ``Tt @ (k_elem @ T)``）。
2. **体积力的形心/端点写法**在内核 5 已照抄（``xi/3 + xj/3 + xm/3``），本层只做乘法。
3. **清零**：``k0_data`` 清的是「替换后节点 ∈ 约束集 ∪ 从属集」的**两行**，
   ``r0_data`` 只清「替换后节点 ∈ 约束集」的两行 —— 两组掩码**不同**，
   不能合并成一个（本层用两个独立的掩码）。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp_sparse

from ..reference.mech_solver import (
    MECH_SIMP_PARAMS,
    MechOperators,
    _slave_rotation,
    boundary_part,
    mech_density_derivatives,   # noqa: F401  （两层共用，纯 numpy）
    mech_density_factors,       # noqa: F401
    mech_residual,              # noqa: F401  （全是稀疏运算，无可向量化部分）
    mech_sensitivity,           # noqa: F401
    mech_simp,                  # noqa: F401
    solve_mech_field,           # noqa: F401
)

__all__ = [
    "MECH_SIMP_PARAMS",
    "MechOperators",
    "mech_simp",
    "mech_density_factors",
    "mech_density_derivatives",
    "mech_residual",
    "solve_mech_field",
    "mech_sensitivity",
    "mech_operators",
]


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
    loaded_lines,
    boundary_fx: np.ndarray,
    boundary_fy: np.ndarray,
    slave_to_master,
    constrained_nodes,
    parts: int,
) -> MechOperators:
    """批量装配力学位置矩阵族。与 ``reference`` 版本**逐位**相同。

    对应 ``MMA10:2604-2814``。参数含义见 ``reference.mech_operators``。
    """
    node = np.asarray(node, dtype=float)
    tri = np.asarray(tri, dtype=int)
    line_tri = np.asarray(line_tri, dtype=int)
    n_elem = tri.shape[0]
    n_node = node.shape[0]
    rot = _slave_rotation(parts)
    constrained = set(int(n) for n in constrained_nodes)
    slave = dict((int(k), int(v)) for k, v in slave_to_master.items())

    # ---- 从属槽位与「替换后」的节点表 ----
    # master_of[k] = 主动节点；非从属节点用 -1 占位（下面按掩码取用）
    master_of = np.full(n_node, -1, dtype=np.int64)
    if slave:
        keys = np.fromiter(slave.keys(), dtype=np.int64, count=len(slave))
        vals = np.fromiter(slave.values(), dtype=np.int64, count=len(slave))
        master_of[keys] = vals

    is_slave_of_node = master_of >= 0
    slot_is_slave = is_slave_of_node[tri]                       # (N, 3) bool
    repl = np.where(slot_is_slave, master_of[tri], tri)          # (N, 3)

    # ---- 逐单元的 T_e（(N, 6, 6)，从属槽位放 R）----
    T = np.broadcast_to(np.eye(6), (n_elem, 6, 6)).copy()
    for s in range(3):
        T[slot_is_slave[:, s], 2 * s:2 * s + 2, 2 * s:2 * s + 2] = rot

    # ---- 自由度表（(N, 6)，顺序 u_i, v_i, u_j, v_j, u_m, v_m）----
    dof = np.empty((n_elem, 6), dtype=np.int64)
    dof[:, 0::2] = 2 * repl
    dof[:, 1::2] = 2 * repl + 1

    # ---- 刚度块 TᵀK_eT 与应力映射 D·L_e·T_e ----
    Tt = T.transpose(0, 2, 1)
    J = (Tt @ k_elem) @ T                                        # ⚠ 左结合，见模块 docstring
    stress_e = (d_mat @ l_coef) @ T
    stress_full = stress_e.reshape(n_elem, 18)

    # ---- 体积力（:2684-2697）：只清 Dirichlet ----
    body = np.empty((n_elem, 6))
    body[:, 0::2] = body_fx[:, None]
    body[:, 1::2] = body_fy[:, None]
    body = body * r0_elem[:, None]
    load_e = np.einsum("nij,nj->ni", Tt, body)

    mask_body = np.zeros((n_elem, 6), dtype=bool)
    is_constrained_node = np.zeros(n_node, dtype=bool)
    if constrained:
        is_constrained_node[np.fromiter(sorted(constrained), dtype=np.int64,
                                        count=len(constrained))] = True
    for s in range(3):
        mask_body[:, 2 * s:2 * s + 2] = is_constrained_node[repl[:, s]][:, None]
    load_e[mask_body] = 0.0

    # ---- 刚度块的清零（:2737-2749）：清 Dirichlet **与**从属 ----
    mask_k0_rows = np.zeros((n_elem, 6), dtype=bool)
    bad_node = is_constrained_node | is_slave_of_node
    for s in range(3):
        mask_k0_rows[:, 2 * s:2 * s + 2] = bad_node[repl[:, s]][:, None]

    k0_blocks = J.copy()
    k0_blocks[mask_k0_rows] = 0.0

    # ---- 位置三元组（形状与书写顺序严格照抄 legacy）----
    k0_data = k0_blocks.reshape(n_elem, 36)
    k0_row = np.repeat(dof, 6, axis=1)
    k0_col = np.tile(dof, (1, 6))
    dk_row = np.repeat(dof, 6, axis=1)
    dk_col = (np.arange(n_elem, dtype=np.int64)[:, None]
              * np.ones(36, dtype=np.int64)).reshape(n_elem, 36)

    r0_data = load_e.reshape(n_elem, 6)
    r0_row = dof.copy()
    r0_col = np.zeros((n_elem, 6), dtype=np.int64)
    dr_row = dof.copy()
    dr_col = (np.arange(n_elem, dtype=np.int64)[:, None]
              * np.ones(6, dtype=np.int64)).reshape(n_elem, 6)

    # ---- 与 reference 共用的边界/载荷段 ----
    r2 = np.zeros(2 * n_node)
    r2, b_rows, b_cols, b_vals = boundary_part(
        n_node, rot, slave, constrained, line_tri, loaded_lines, r2_elem,
        boundary_fx, boundary_fy, r2,
    )

    n_k0 = n_elem * 36
    n_r0 = n_elem * 6
    ones_k0 = np.ones(n_k0)
    ones_r0 = np.ones(n_r0)

    # 对称 Dirichlet（2026-09-24）：只零齐次 Dirichlet 列（与 reference 逐位同构）。
    col_mask = sp_sparse.identity(2 * n_node).tolil()
    for i_node in constrained:
        col_mask[:, 2 * i_node] = 0.0
        col_mask[:, 2 * i_node + 1] = 0.0
    col_mask = col_mask.tocsr()

    return MechOperators(
        # ⚠ 2026-09-24 消元：必须存 slave_to_master（mech_worker 组消元算子用；
        #   漏存 ⇒ 消元退化单位 T ⇒ 未消元矩阵 + Pardiso mtype=2 会解错）。
        slave_to_master=dict(slave_to_master) if slave_to_master else None,
        k0_data=k0_data.ravel(), k0_row=k0_row.ravel(), k0_col=k0_col.ravel(),
        kconst_map=sp_sparse.csr_matrix(
            (ones_k0, (np.arange(n_k0, dtype=np.int64),
                       np.repeat(np.arange(n_elem, dtype=np.int64), 36))),
            shape=(n_k0, n_elem)),
        k_u_map=sp_sparse.csr_matrix(
            (ones_k0, (np.arange(n_k0, dtype=np.int64), k0_col.ravel())),
            shape=(n_k0, 2 * n_node)),
        k_boundary=sp_sparse.csr_matrix(
            (b_vals, (b_rows, b_cols)), shape=(2 * n_node, 2 * n_node)),
        col_mask=col_mask,
        r0_data=r0_data.ravel(), r0_row=r0_row.ravel(), r0_col=r0_col.ravel(),
        rconst_map=sp_sparse.csr_matrix(
            (ones_r0, (np.arange(n_r0, dtype=np.int64),
                       np.repeat(np.arange(n_elem, dtype=np.int64), 6))),
            shape=(n_r0, n_elem)),
        r2=r2,
        dk_p_dth_u_row=dk_row.ravel(), dk_p_dth_u_col=dk_col.ravel(),
        dr_p_dth_row=dr_row.ravel(), dr_p_dth_col=dr_col.ravel(),
        stress_data=stress_full.ravel(),
        stress_row=_stress_rows(n_elem),
        stress_col=np.tile(dof, (1, 3)).ravel(),
        stress_k_e_map=sp_sparse.csr_matrix(
            (np.ones(3 * n_elem),
             (np.arange(3 * n_elem, dtype=np.int64),
              np.repeat(np.arange(n_elem, dtype=np.int64), 3))),
            shape=(3 * n_elem, n_elem)),
        von_mises_map=_von_mises_map(n_elem),
    )


def _stress_rows(n_elem: int) -> np.ndarray:
    """``K0_stress_row``：每个单元 3 行 ×6 列，行号 = ``3e + 分量``。"""
    return np.repeat((3 * np.arange(n_elem, dtype=np.int64))[:, None]
                     + np.arange(3, dtype=np.int64), 6, axis=1).ravel()


def _von_mises_map(n_elem: int) -> sp_sparse.csr_matrix:
    """逐单元的 von Mises 二次型 ``[[1, −1/2, 0], [−1/2, 1, 0], [0, 0, 3]]``。

    ⚠ **零元不存**：legacy 用 ``lil`` 逐列赋值（`:2730-2732`），``lil`` 会丢弃零
    ⇒ 每单元只有 **5** 个非零元 ``(0,0)/(0,1)/(1,0)/(1,1)/(2,2)``。照搬这个结构。
    """
    rows, cols, vals = [], [], []
    pattern = ((0, 0, 1.0), (0, 1, -1.0 / 2.0), (1, 0, -1.0 / 2.0),
               (1, 1, 1.0), (2, 2, 3.0))
    for e in range(n_elem):
        for k, l, v in pattern:
            rows.append(3 * e + k)
            cols.append(3 * e + l)
            vals.append(v)
    return sp_sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows, dtype=np.int64),
                            np.asarray(cols, dtype=np.int64))),
        shape=(3 * n_elem, 3 * n_elem))
