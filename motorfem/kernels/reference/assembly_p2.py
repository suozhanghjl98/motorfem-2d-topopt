"""P2（6 节点）单元变换矩阵 —— 路线 A 框架接入的第一步。

规划：`docs/重构设计文档.md` §13。对照 P1 的 `assembly.transformation_matrix`
（内核 8，三个分支）。P2 单元 6 节点 = 3 顶点 (i,j,m) + 3 边中点
（k=ij, l=jm, n=mi）。

三分支的 P2 推广：

- **分支 3（普通单元）**：`T = I₆`，`index = 6 节点`。
- **分支 2（从属单元，非气隙）**：`T = diag(signs)`，`signs` 是 6 个节点的
  反周期符号（边中点符号与其所在边的符号一致，`virtual_element_signs` 逐节点
  判定即可）。
- **分支 1（转子气隙单元）**：**全部 6 个槽位统一处理**，与 P1 同构 ——
  槽位节点在 `rotating.brackets` 里 → 走内核 7 的插值约束
  （`T[s, 括号节点] = −K_rotating_boundary[node_s, 括号节点]·sign_s`，动节点本身
  被消去）；否则 → `sign_s` 的单位行（独立自由度）。

⚠ Design B（2026-10-02 定稿，与 moving_mesh 配套）：**接口圆边中点也是动节点、
走自己的插值约束**（`moving_mesh` 给它建 bracket 行），不是早期过渡写的
「两端点插值平均」。理由：pos=0 恒等（`A_中点 = A_定子中点`）、界面保 P2 阶、
驱动行与 T 消元一致（扭矩读回 `A[动节点]`）。气隙内部边/跨接口边的中点不在
brackets ⇒ 独立自由度 —— 即「全局 P2、无悬挂节点」的路线 A 处理，满足
§13.5 的可调阶数设计。

用法：与 `assembly.ElementTransform` 接口平行，供 P2 的 `element_operators`
（后续内核）消费。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

import numpy as np

from . import boundaries as ref_bnd
from .assembly import (
    ElementOperators,
    TopologyContext,
    _active_index,
    _coo,
)

__all__ = [
    "P2ElementTransform",
    "transformation_matrix_p2",
    "element_operators_p2",
    "get_R_vector_p2",
]


@dataclass(frozen=True)
class P2ElementTransform:
    """P2 单元变换矩阵 ``T``（6 节点 → active 自由度）。

    Attributes
    ----------
    matrix : (6, k) ndarray
        ``A_local6 = T · A_active``。
    index : tuple[int, ...]
        活跃自由度全局节点号（列序），顺序同 P1 的 `_active_index` 语义。
    """

    matrix: np.ndarray
    index: Tuple[int, ...]


def transformation_matrix_p2(
    i_elem: int, ctx: TopologyContext, rotating
) -> P2ElementTransform:
    """P2 单元变换矩阵。对应 P1 ``MMA10:1693-1773`` 的 6 节点推广。

    Parameters
    ----------
    i_elem : int
    ctx : TopologyContext
        **P2 拓扑**：``ctx.tri`` 每行 6 列（顶点 + 边中点）、``virtual_tri``
        同；``element_rotor_gap``、``element_slave`` 等集合按单元号判定。
    rotating : RotatingBoundary
        内核 7 产出：``.matrix``、``.brackets``。

    Notes
    -----
    分支 1 全部 6 槽位统一走 `_vertex_entry`（brackets → 插值行；否则单位行），
    设计见模块 docstring 的 Design B 段。``used``/``entries`` 的结构与 P1
    相同：先定列序（``_active_index``），每个位置至多写一次。
    """
    i_elem = int(i_elem)

    # --- 分支 1：转子气隙单元（判据优先于从属分支）---
    if i_elem in ctx.element_rotor_gap:
        if i_elem not in ctx.element_slave:
            nodes = tuple(int(n) for n in ctx.tri[i_elem])
            signs = (1.0,) * 6
        else:
            nodes = tuple(int(n) for n in ctx.virtual_tri[i_elem])
            signs = tuple(
                float(s)
                for s in ref_bnd.virtual_element_signs(
                    np.asarray([nodes]), ctx.master_nodes, ctx.a_eq
                )[0]
            )

        used = []
        entries = []          # entries[s] = [(节点, 系数), ...]

        def _vertex_entry(node, sign):
            if node in rotating.brackets:
                node_up, node_low = rotating.brackets[node]
                return (
                    (int(node_up), -rotating.matrix[node, node_up]),
                    (int(node_low), -rotating.matrix[node, node_low]),
                )
            return ((node, 1.0),)

        # 全部 6 槽位统一：在 brackets 里 → 自己的插值行；否则 → 单位行。
        # 接口圆边中点动节点也在 brackets（moving_mesh 配套的 Design B：中点走
        # 自己的插值约束，不是端点平均），见模块 docstring 与 moving_mesh 第七节。
        for s in range(6):
            entries.append(list(_vertex_entry(nodes[s], signs[s])))
            used.extend(n for n, _ in entries[s])

        index = _active_index(used)
        column_of = {node: col for col, node in enumerate(index)}
        matrix = np.zeros((6, len(index)))
        for s, row_entries in enumerate(entries):
            sign_s = signs[s]
            for node, coefficient in row_entries:
                matrix[s, column_of[node]] += coefficient * sign_s
        return P2ElementTransform(matrix=matrix, index=index)

    # --- 分支 2：从属单元（非气隙）---
    if i_elem in ctx.element_slave:
        nodes = tuple(int(n) for n in ctx.virtual_tri[i_elem])
        signs = ref_bnd.virtual_element_signs(
            np.asarray([nodes]), ctx.master_nodes, ctx.a_eq
        )[0]
        return P2ElementTransform(matrix=np.diag(signs), index=nodes)

    # --- 分支 3：普通单元 ---
    nodes = tuple(int(n) for n in ctx.tri[i_elem])
    return P2ElementTransform(matrix=np.eye(6), index=nodes)


def _expand21_to_matrix(k0e):
    """P2 压缩 21 值 → 6×6 对称矩阵。"""
    K = np.zeros((6, 6)); idx = 0
    for a in range(6):
        for b in range(a, 6):
            K[a, b] = K[b, a] = k0e[idx]; idx += 1
    return K


def element_operators_p2(
    ctx: TopologyContext,
    rotating,
    *,
    n_node: int,
    k0elem: np.ndarray,
    bxcoef: np.ndarray,
    bycoef: np.ndarray,
    element_iron: Iterable[int],
    element_topo_opti: Iterable[int],
    use_nonlinear_iron: bool,
) -> ElementOperators:
    """P2 装配族（对照 P1 `assembly.element_operators`，内核 8，6 节点推广）。

    结构与 P1 完全同构，差异只在：
    - 变换矩阵用 `transformation_matrix_p2`（6 节点）；
    - `k0elem` 是 P2 的 21 值（→ 6×6）、`bxcoef/bycoef` 是 (N,6)；
    - `k0_t = Tᵀ K0_e T` 的尺寸 (k,k)（k = 活跃自由度列数）。

    返回的 `ElementOperators` 字段与 P1 相同（`bxcoef_mat`/`bycoef_mat` 形状
    (N_elem, n_node)，n_node 是 **P2** 节点数）。

    ⚠ `bxcoef_mat`/`bycoef_mat` 丢零（`lil` 语义），其余矩阵保留显式零 —— 与 P1 同。
    """
    tri = np.asarray(ctx.tri)
    n_elem = tri.shape[0]
    iron = set(int(e) for e in element_iron)
    topo = set(int(e) for e in element_topo_opti)

    k0_data, k0_row, k0_col = [], [], []
    kconst_rows, kconst_cols = [], []
    bx_rows, bx_cols, bx_vals = [], [], []
    by_rows, by_cols, by_vals = [], [], []
    g0i_rows, g0i_cols, g0i_vals = [], [], []
    gconst_rows, gconst_cols = [], []
    gswap1_rows, gswap1_cols = [], []
    gswap2_rows, gswap2_cols = [], []
    jabobi2_row, jabobi2_col = [], []
    dk_rows, dk_cols = [], []

    is_vaccum = np.zeros(n_elem)
    is_simp = np.zeros(n_elem)
    is_non_simp = np.zeros(n_elem)
    is_linear_iron = np.zeros(n_elem)

    n_k0 = 0
    n_gswap = 0
    n_row_g0i = 0

    for i_elem in range(n_elem):
        tr = transformation_matrix_p2(i_elem, ctx, rotating)
        matrix, index = tr.matrix, tr.index

        # --- Bx / By 系数行（丢零）---
        bx = np.asarray(bxcoef[i_elem], dtype=float) @ matrix
        by = np.asarray(bycoef[i_elem], dtype=float) @ matrix
        for col, node in zip(range(len(index)), index):
            if bx[col] != 0.0:
                bx_rows.append(i_elem); bx_cols.append(int(node)); bx_vals.append(bx[col])
            if by[col] != 0.0:
                by_rows.append(i_elem); by_cols.append(int(node)); by_vals.append(by[col])

        # --- 材料分类 ---
        if i_elem in iron:
            if not use_nonlinear_iron:
                is_linear_iron[i_elem] = 1
            elif i_elem in topo:
                is_simp[i_elem] = 1
            else:
                is_non_simp[i_elem] = 1
        else:
            is_vaccum[i_elem] = 1

        # --- 单元刚度（21 值 → 6×6 → TᵀK0T）---
        k0_t = matrix.T @ (_expand21_to_matrix(k0elem[i_elem]) @ matrix)

        for i_row in range(len(index)):
            for i_col in range(len(index)):
                value = k0_t[i_row, i_col]
                k0_data.append(value)
                k0_row.append(int(index[i_row]))
                k0_col.append(int(index[i_col]))

                kconst_rows.append(n_k0); kconst_cols.append(i_elem)

                g0i_rows.append(n_row_g0i + i_row)
                g0i_cols.append(int(index[i_col]))
                g0i_vals.append(value)

                gswap1_rows.append(n_gswap); gswap1_cols.append(n_row_g0i + i_row)
                gswap2_rows.append(n_gswap); gswap2_cols.append(n_row_g0i + i_col)
                jabobi2_row.append(int(index[i_row]))
                jabobi2_col.append(int(index[i_col]))

                n_k0 += 1
                n_gswap += 1

            gconst_rows.append(n_row_g0i + i_row)
            gconst_cols.append(i_elem)
            dk_rows.append(i_elem)
            dk_cols.append(int(index[i_row]))

        n_row_g0i += len(index)

    return ElementOperators(
        k0_data=np.asarray(k0_data, dtype=float),
        k0_row=np.asarray(k0_row, dtype=np.int64),
        k0_col=np.asarray(k0_col, dtype=np.int64),
        kconst_mat=_coo(kconst_rows, kconst_cols, [1] * len(kconst_rows),
                        (n_k0, n_elem), dtype=np.int64),
        bxcoef_mat=_coo(bx_rows, bx_cols, bx_vals, (n_elem, n_node)),
        bycoef_mat=_coo(by_rows, by_cols, by_vals, (n_elem, n_node)),
        is_vaccum=is_vaccum,
        is_simp=is_simp,
        is_non_simp=is_non_simp,
        is_linear_iron=is_linear_iron,
        g0i_mat=_coo(g0i_rows, g0i_cols, g0i_vals, (n_row_g0i, n_node)),
        gconst_mat=_coo(gconst_rows, gconst_cols, [1] * len(gconst_rows),
                        (n_row_g0i, n_elem), dtype=np.int64),
        gswap1_mat=_coo(gswap1_rows, gswap1_cols, [1] * len(gswap1_rows),
                        (n_gswap, n_row_g0i), dtype=np.int64),
        gswap2_mat=_coo(gswap2_rows, gswap2_cols, [1] * len(gswap2_rows),
                        (n_gswap, n_row_g0i), dtype=np.int64),
        jabobi2_row=np.asarray(jabobi2_row, dtype=np.int64),
        jabobi2_col=np.asarray(jabobi2_col, dtype=np.int64),
        dk_p_dgamma_a_row=np.asarray(dk_rows, dtype=np.int64),
        dk_p_dgamma_a_col=np.asarray(dk_cols, dtype=np.int64),
        n_k0=n_k0,
        n_gswap=n_gswap,
        n_row_g0i=n_row_g0i,
    )


def get_R_vector_p2(
    ctx: TopologyContext,
    rotating,
    *,
    n_node: int,
    r0elem: np.ndarray,
    current_of_element: Mapping[int, float],
) -> np.ndarray:
    """P2 源项（右端项）``R``。对照 P1 ``assembly.get_R_vector`` 的 6 节点版。

    Parameters 与 P1 ``get_R_vector`` 同，只是 ``r0elem`` 是 (N_elem, 6) 的 P2
    一致载荷（顶点 0、边中点 Δ/3，``element_matrices_p2``）。

    Notes
    -----
    数学：``R = Σ_{e ∈ 载流} T_eᵀ (R0_e · J_e)``，``T_eᵀ`` 在**左** ⇒ 与 P1 相同
    的关键语义：含插值行的单元（转子气隙）源项只落到**括号节点**，动节点本身
    收不到。P2 下顶点 R0 恰为 0，所以动**顶点**的 bracket 拉回恒为 0；动**中点**
    的 R0 = Δ/3 会拉到它的括号节点上（测试 ``test_get_R_p2_moving_nodes_get_nothing``
    钉住）。
    """
    R = np.zeros(int(n_node))
    for i_elem in range(np.asarray(ctx.tri).shape[0]):
        j = current_of_element.get(i_elem)
        if j is None:
            continue
        tr = transformation_matrix_p2(i_elem, ctx, rotating)
        contribution = tr.matrix.T @ np.asarray(r0elem[i_elem], dtype=float)
        R[list(tr.index)] += contribution * j
    return R
