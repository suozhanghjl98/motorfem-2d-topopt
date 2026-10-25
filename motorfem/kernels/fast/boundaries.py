"""主从周期边界 / 从属虚拟单元 / 驱动行矩阵的批量实现（内核 6 的 fast 层）。

与 ``reference/boundaries.py`` **逐位相同**，只是把逐节点/逐单元循环换成 numpy
批量运算。数据结构直接从 ``reference`` 导入（沿用内核 5 的分层约定）。

数学表述、``a_eq`` 的语义、以及那个 ★待核对★ 的变换矩阵符号关系，全部写在
``reference/boundaries.py`` 的模块 docstring 里，本文件不重复。

------------------------------------------------------------------------------
本层与 legacy 的对应关系
------------------------------------------------------------------------------

===============================  ==================  ==============================
本模块函数                        legacy 行号          向量化手法
===============================  ==================  ==============================
``pair_master_slave``            ``304-321``          广播出 (N_master, N_slave) 的
                                                     匹配掩码 + ``argmax``
``slave_virtual_elements``       ``323-335``          ``np.isin`` 掩码 + ``arange``
                                                     查表替换
``virtual_element_signs``        ``1706-1708`` 等     ``np.isin`` + ``np.where``
``driving_rows``                 ``2073-2102``        分段数组拼接 + 末次写入生效
===============================  ==================  ==============================

------------------------------------------------------------------------------
两处必须保留的 legacy 语义（向量化下最容易丢的两点）
------------------------------------------------------------------------------

1. **主从配对取「列表顺序中的第一个匹配」**。legacy 是

   .. code-block:: python

       for i_node_m in node_master:
           if 匹配: node_slave[i_node_s] = i_node_m; break

   而 ``node_master`` 是**升序**的节点号列表，所以语义是「最小节点号的匹配者」。
   本层用 ``argmax`` 沿**主动轴**取第一个 ``True``——前提是主动候选数组按升序
   生成（``np.flatnonzero`` 保证）。若几何有多个匹配者，换成
   ``argmin(距离)`` 之类的写法就会与 legacy 不一致。

2. **从属行会先被清空**。legacy 写作 ``F_mat_driven[i_node_s, :] = 0``
   （MMA10:2092），而 ``j_mat_driven`` 的主从分支**没有**清空动作。本层用
   ``_drop_rows`` 显式复现这个差别，并在最后做一次「末次写入生效」去重，
   避免 COO 组装把同一 (行, 列) 的两个相同值**相加**成 2.0。

------------------------------------------------------------------------------
红线清单
------------------------------------------------------------------------------

``fast/`` 必须完整遵守 ``CLAUDE.md`` §3：本模块**不调用** ``sorted()``、
**不改动**输入集合的构造与书写顺序、**不合并**任何 ``for i_elem in
range(N_element)`` 的嵌套顺序。所有区域集合都是**只读输入**，本层只做成员测试
（``np.isin``）与升序展平（``np.flatnonzero``），不迭代集合本身。
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp_sparse

from ..reference.boundaries import (
    DrivingRows,
    PeriodicBoundary,
    SlaveVirtualElements,
    driving_row_jacobian,
    driving_row_residual,
)

__all__ = [
    "PeriodicBoundary",
    "SlaveVirtualElements",
    "DrivingRows",
    "pair_master_slave",
    "slave_virtual_elements",
    "virtual_element_signs",
    "driving_rows",
    "driving_row_residual",
    "driving_row_jacobian",
]


# =============================================================================
# 主从配对
# =============================================================================

def pair_master_slave(
    xy: np.ndarray, parts: int, coord_err_abs: float = 1e-6
) -> PeriodicBoundary:
    """批量版主从配对。与 ``reference`` 版本逐位/逐项相同。

    对应 legacy ``MMA10:304-321``。

    ``node_master``（主动候选）在 legacy 里是列表推导生成的**升序**列表，本层用
    ``np.flatnonzero`` 得到同一顺序；配对用 ``argmax(hit, axis=0)`` 取主动轴上的
    第一个匹配，等价于 legacy 的 ``break``。
    """
    xy = np.asarray(xy, dtype=float)
    angle = 2 * np.pi / parts
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x = xy[:, 0]
    y = xy[:, 1]

    # 主动候选：θ=0 正射线（|y| < err 且 x > err）★ 方向限定，见 reference docstring
    master_mask = np.abs(y) < coord_err_abs
    master_mask &= coord_err_abs < x
    node_master = np.flatnonzero(master_mask)

    # 从属候选：转 −2π/parts 后落在 θ=0 **正**射线上
    rx_neg = x * np.cos(-angle) - y * np.sin(-angle)
    ry_neg = x * np.sin(-angle) + y * np.cos(-angle)
    slave_mask = np.abs(ry_neg) < coord_err_abs
    slave_mask &= coord_err_abs < rx_neg
    slave_candidates = np.flatnonzero(slave_mask)

    if slave_candidates.size:
        # 主动候选转 +2π/parts 后的坐标，形状 (N_master, 1)
        mx = xy[node_master, 0][:, None]
        my = xy[node_master, 1][:, None]
        rotated_mx = mx * cos_a - my * sin_a
        rotated_my = mx * sin_a + my * cos_a

        # 从属候选坐标，形状 (1, N_slave)
        sx = xy[slave_candidates, 0][None, :]
        sy = xy[slave_candidates, 1][None, :]

        hit = np.abs(rotated_mx - sx) < coord_err_abs
        hit &= np.abs(rotated_my - sy) < coord_err_abs

        unmatched = ~hit.any(axis=0)
        if unmatched.any():
            i_bad = int(slave_candidates[int(np.flatnonzero(unmatched)[0])])
            raise ValueError(
                f"从属节点 {i_bad}（坐标 {xy[i_bad]}）在 {node_master.size} 个主动"
                f"候选里找不到转 +2π/{parts} 后与之重合的节点"
                f"（容差 {coord_err_abs:g}）；legacy 此处是 assert 失败"
            )

        first_match = np.argmax(hit, axis=0)          # 主动轴上的第一个 True
        matched_master = node_master[first_match]
    else:
        matched_master = np.empty(0, dtype=node_master.dtype)

    slave_to_master: Dict[int, int] = {
        int(s): int(m) for s, m in zip(slave_candidates, matched_master)
    }
    master_to_slave = dict((v, k) for (k, v) in slave_to_master.items())

    return PeriodicBoundary(slave_to_master, master_to_slave)


# =============================================================================
# 从属虚拟单元
# =============================================================================

def slave_virtual_elements(
    tri: np.ndarray, pairing: PeriodicBoundary
) -> SlaveVirtualElements:
    """批量版从属虚拟单元表。与 ``reference`` 版本逐位/逐项相同。

    对应 legacy ``MMA10:323-335``。

    替换用「``arange`` 查表 + 覆盖从属槽位」实现：先造 ``map_arr = arange``，
    再把 ``map_arr[从属节点] = 主动节点``，随后直接 ``map_arr[tri]`` 得到整张
    虚拟连接表——等价于逐节点做 ``node_slave[n] if n in node_slave else n``。

    P2（6 列）：核心逻辑本就 shape 无关（``np.isin(tri, …)`` 覆盖全槽位、按节点号
    查表），只有校验与 virtual_tri 的取值放宽到 6 列。P1 3 列行为逐位不变。
    """
    tri = np.asarray(tri)
    if tri.ndim != 2 or tri.shape[1] not in (3, 6):
        raise ValueError(
            f"tri 必须是 (N_elem, 3) 或 (N_elem, 6) 的连接表，实得形状 {tri.shape}。"
            f"注意 legacy 的 element_S 是 5 列 [单元号, n1, n2, n3, 区域号]，"
            f"传它进来会让两个实现一起错（来源 C 的 ref/fast 对比抓不到这种错）"
        )
    slave_keys = np.fromiter(
        pairing.slave_to_master.keys(), dtype=np.int64,
        count=len(pairing.slave_to_master),
    )

    touch = np.isin(tri, slave_keys)
    elements = np.flatnonzero(touch.any(axis=1))

    max_id = -1
    if tri.size:
        max_id = max(max_id, int(tri.max()))
    if slave_keys.size:
        max_id = max(
            max_id,
            int(slave_keys.max()),
            int(max(pairing.slave_to_master.values())),
        )

    map_arr = np.arange(max_id + 1, dtype=np.int64)
    if slave_keys.size:
        map_arr[slave_keys] = np.fromiter(
            pairing.slave_to_master.values(), dtype=np.int64, count=slave_keys.size,
        )

    virtual = map_arr[tri[elements]]

    virtual_tri: Dict[int, Tuple[int, ...]] = {
        int(e): tuple(int(x) for x in virtual[i])
        for i, e in enumerate(elements)
    }

    return SlaveVirtualElements(
        frozenset(int(e) for e in elements), virtual_tri
    )


# =============================================================================
# 虚拟单元的符号向量
# =============================================================================

def virtual_element_signs(
    virtual_nodes: Sequence[int],
    master_nodes: Iterable[int],
    a_eq: bool,
) -> np.ndarray:
    """批量版虚拟单元符号向量。与 ``reference`` 版本逐位相同。

    对应 legacy ``MMA10:1706-1708`` 等。

    ★ **定义域：只能用于从属单元（虚拟单元）。** ★
    legacy 的非从属分支是**硬编码 ``+1``**（``_T = np.eye(3)``），不调用符号
    公式；对一个含「已配对主动节点」的非从属单元误用本函数会得到**混合**符号
    模式，``TᵀKT`` 会真被改变。真实网格上有 134 个这样的单元（其中 4 个在
    ``element_rotor_gap`` 里，属内核 7 的风险面）。完整论证、实测数字与正确用法
    见 ``reference/boundaries.py`` 同名函数的 Notes。
    """
    arr = np.asarray(virtual_nodes)
    master_arr = np.fromiter(master_nodes, dtype=np.int64)
    other = 1.0 if a_eq else -1.0
    if master_arr.size == 0:
        return np.full(arr.shape, other, dtype=float)
    return np.where(np.isin(arr, master_arr), 1.0, other)


# =============================================================================
# 驱动行
# =============================================================================

def _drop_rows(rows, cols, vals, drop_rows_arr):
    """丢掉 ``rows`` 落在 ``drop_rows_arr`` 里的条目（对应 legacy 的整行清零）。"""
    if rows.size == 0 or drop_rows_arr.size == 0:
        return rows, cols, vals
    keep = ~np.isin(rows, drop_rows_arr)
    return rows[keep], cols[keep], vals[keep]


def _dedupe_last_wins(rows, cols, vals, n_node):
    """同一 (行, 列) 只保留**最后一次**写入。

    必须做这一步：COO 组装会把同一坐标的重复条目**相加**，而 legacy 用
    ``lil_matrix`` 逐项赋值是**覆盖**语义。漏掉它会让同时属于 Dirichlet 与主从的
    角点节点（真实网格里 ``(0, ±m_R)``）得到 2.0。
    """
    if rows.size == 0:
        return rows, cols, vals
    key = rows.astype(np.int64) * np.int64(n_node) + cols.astype(np.int64)
    reversed_order = np.arange(key.size - 1, -1, -1)
    _, index_in_reversed = np.unique(key[reversed_order], return_index=True)
    keep = np.sort(reversed_order[index_in_reversed])
    return rows[keep], cols[keep], vals[keep]


def driving_rows(
    n_node: int,
    dirichlet: Iterable[int],
    slave_to_master: Mapping[int, int],
    a_eq: bool,
    row_replacements: Optional[Mapping[int, np.ndarray]] = None,
) -> DrivingRows:
    """批量版驱动行装置。与 ``reference`` 版本逐位相同。

    对应 legacy ``MMA10:2073-2112``。

    写入顺序与 legacy 一致：Dirichlet → 主从 → ``row_replacements``。
    ``F_mat_driven`` 在主从行与注入行上会先清空整行，``J_mat_driven`` 只在注入行
    上清空（主从分支在 legacy 里没有清空动作，见 ``reference`` 的同名函数）。
    """
    n_node = int(n_node)
    sigma = -1.0 if a_eq else 1.0

    dirichlet_rows = np.fromiter(dirichlet, dtype=np.int64)
    slave_rows = np.fromiter(
        slave_to_master.keys(), dtype=np.int64, count=len(slave_to_master)
    )
    slave_masters = np.fromiter(
        slave_to_master.values(), dtype=np.int64, count=len(slave_to_master)
    )

    ones_d = np.ones(dirichlet_rows.size, dtype=float)

    # --- 第 0 段：齐次 Dirichlet 行 ---
    f_rows, f_cols, f_vals = dirichlet_rows.copy(), dirichlet_rows.copy(), ones_d.copy()
    j_rows, j_cols, j_vals = dirichlet_rows.copy(), dirichlet_rows.copy(), ones_d.copy()

    # --- 第 1 段：主从行。F 先清空该行，J 不清空 ---
    f_rows, f_cols, f_vals = _drop_rows(f_rows, f_cols, f_vals, slave_rows)
    f_rows = np.concatenate([f_rows, slave_rows, slave_rows])
    f_cols = np.concatenate([f_cols, slave_rows, slave_masters])
    f_vals = np.concatenate(
        [f_vals, np.ones(slave_rows.size), np.full(slave_rows.size, sigma)]
    )

    j_rows = np.concatenate([j_rows, slave_rows, slave_rows])
    j_cols = np.concatenate([j_cols, slave_rows, slave_masters])
    j_vals = np.concatenate(
        [j_vals, np.ones(slave_rows.size), np.full(slave_rows.size, sigma)]
    )

    # --- 第 2 段：额外整行替换（内核 7 的动网格插值行）---
    if row_replacements:
        repl_rows = np.fromiter(row_replacements.keys(), dtype=np.int64)
        r_rows, r_cols, r_vals = [], [], []
        for i_row, row_vector in row_replacements.items():
            row_vector = np.asarray(row_vector, dtype=float).ravel()
            if row_vector.shape[0] != n_node:
                raise ValueError(
                    f"row_replacements 中第 {i_row} 行的长度是 {row_vector.shape[0]}，"
                    f"应为 n_node = {n_node}"
                )
            nz = np.nonzero(row_vector)[0]
            r_rows.append(np.full(nz.size, i_row, dtype=np.int64))
            r_cols.append(nz.astype(np.int64))
            r_vals.append(row_vector[nz])
        r_rows = np.concatenate(r_rows) if r_rows else np.empty(0, dtype=np.int64)
        r_cols = np.concatenate(r_cols) if r_cols else np.empty(0, dtype=np.int64)
        r_vals = np.concatenate(r_vals) if r_vals else np.empty(0, dtype=float)

        f_rows, f_cols, f_vals = _drop_rows(f_rows, f_cols, f_vals, repl_rows)
        j_rows, j_cols, j_vals = _drop_rows(j_rows, j_cols, j_vals, repl_rows)

        f_rows = np.concatenate([f_rows, r_rows])
        f_cols = np.concatenate([f_cols, r_cols])
        f_vals = np.concatenate([f_vals, r_vals])
        j_rows = np.concatenate([j_rows, r_rows])
        j_cols = np.concatenate([j_cols, r_cols])
        j_vals = np.concatenate([j_vals, r_vals])
    else:
        repl_rows = np.empty(0, dtype=np.int64)

    f_rows, f_cols, f_vals = _dedupe_last_wins(f_rows, f_cols, f_vals, n_node)
    j_rows, j_cols, j_vals = _dedupe_last_wins(j_rows, j_cols, j_vals, n_node)

    mat_driven = sp_sparse.coo_matrix(
        (f_vals, (f_rows, f_cols)), shape=(n_node, n_node)
    ).tocsr()
    j_mat_driven = sp_sparse.coo_matrix(
        (j_vals, (j_rows, j_cols)), shape=(n_node, n_node)
    ).tocsr()

    driven_rows = np.concatenate([dirichlet_rows, slave_rows, repl_rows])
    is_driven = np.zeros(n_node, dtype=float)
    is_driven[driven_rows] = 1.0
    diagonal = np.ones(n_node, dtype=float)
    diagonal[driven_rows] = 0.0
    j_driven = sp_sparse.diags(diagonal, format="csr")

    # 对称 Dirichlet（2026-09-24）：只零齐次 Dirichlet 列（反周期/动网格行本就对称）。
    col_mask = sp_sparse.identity(n_node).tolil()
    for i_node in dirichlet_rows:
        col_mask[:, i_node] = 0.0
    col_mask = col_mask.tocsr()

    return DrivingRows(
        is_driven=is_driven,
        mat_driven=mat_driven,
        j_driven=j_driven,
        j_mat_driven=j_mat_driven,
        col_mask=col_mask,
    )


# ``driving_row_residual`` / ``driving_row_jacobian`` 是**消费端的两行公式**
# （MMA10:2244 与 2255），不含任何向量化余地，因此直接从 ``reference`` 再导出，
# 不在这里复制一份 —— 复制只会制造双份维护与漂移风险。数据结构同样从
# ``reference`` 导入（沿用内核 5 的分层约定）。
