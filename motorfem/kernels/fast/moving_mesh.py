"""动网格滑移层与旋转边界权重矩阵的批量实现（内核 7 的 fast 层）。

与 ``reference/moving_mesh.py`` **逐位相同**，只是把逐节点循环换成 numpy 批量运算。
数据结构与两个不含算法的辅助函数直接从 ``reference`` 导入（沿用内核 5、6 的分层
约定）。数学、八个符号分支的统一式、实测事实与红线讨论全部写在
``reference/moving_mesh.py`` 的模块 docstring 里，本文件不重复。

------------------------------------------------------------------------------
与 reference/legacy 的对应关系：每个环节的等价性依据
------------------------------------------------------------------------------

=============================  ========================  ==============================
本模块                          逐节点版                   逐位相同的依据
=============================  ========================  ==============================
角度                            ``arctan2(y, x) + pos``    同一个 ufunc，逐元素同运算
                                                           同次序（批量不改变单元素
                                                           的求值结果）
折叠 ``φ``                      ``mod(θ, w)``              同
奇扇区 ``s``                    ``mod(θ // w, 2) > .5``    同
上界下标                        ``while a[i] ≤ φ: i += 1`` ``np.searchsorted(a, φ,
                                                           side='right')`` 的定义就是
                                                           「第一个严格大于 φ 的
                                                           下标」⇒ 整数完全一致
权重 ``kp/kn``                  同一表达式                 同（批量不改变结合顺序）
从属替换 + 符号 ``−s·σ·k``       同                         同（乘 ±1 是精确的）
稀疏装配                        ``dict`` 去重后装 COO      ``_dedupe_last_wins`` 复现
                                                           「后写覆盖」；再过滤零元
                                                           复现 ``lil`` 丢弃零元
角度排序                        ``sorted(key=θ)``（稳定）  ``np.argsort(kind='stable')``
                                                           + 同一输入顺序 ⇒ 平局一致
=============================  ========================  ==============================

------------------------------------------------------------------------------
两处最容易在向量化时丢掉的 legacy 语义
------------------------------------------------------------------------------

1. **「后写覆盖」而不是「相加」**。同一 ``(i, j)`` 在一次求值里可能被写两次（下侧
   括号节点与上侧括号节点映射到同一个主动节点 —— 退化几何下会发生）。COO 组装会
   把重复项**相加**，必须先用 ``_dedupe_last_wins`` 去重。注意本层的写入顺序是
   「先全部对角、再全部上侧、再全部下侧」，而 reference 是逐节点
   「对角→上侧→下侧」；两者只在同一行内比较先后，而下侧块整体在上侧块之后、对角
   元不可能与插值列同列（新节点号 ≥ 原节点数，接口节点号 < 原节点数），所以
   **胜者一致**。
2. **恰好为零的元要被丢弃**（``lil`` 的语义）。``pos = 0`` 时 ``kp = 0``。

------------------------------------------------------------------------------
红线清单
------------------------------------------------------------------------------

``fast/`` 必须完整遵守 ``CLAUDE.md`` §3。本模块对区域集合 ``element_rotor_gap``
只做一次**保序展平**（``np.fromiter``，不排序、不改构造表达式）与逐元素查表替换；
该展平顺序不进入任何浮点累加（替换是逐元素赋值），因此不影响位级结果。本模块
**不调用** ``sorted()``，也不改动任何输入集合的书写顺序。
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp_sparse

from ..reference import moving_mesh as ref_mm
from ..reference.moving_mesh import (
    MovingMeshLayer,
    RotatingBoundary,
    RotatingBoundaryWithDerivative,
    nodes_of_elements,
)
from .boundaries import _dedupe_last_wins

__all__ = [
    "MovingMeshLayer",
    "RotatingBoundary",
    "RotatingBoundaryWithDerivative",
    "nodes_of_elements",
    "insert_moving_mesh_layer",
    "rotating_boundary",
    "rotating_boundary_with_derivative",
]


# =============================================================================
# 装配
# =============================================================================

def _assemble(rows, cols, vals, n_node: int):
    """三元组 → CSR，复现 ``lil`` 的两条语义：后写覆盖 + 丢弃零元。"""
    rows, cols, vals = _dedupe_last_wins(rows, cols, vals, n_node)
    keep = vals != 0.0
    if not np.any(keep):
        return sp_sparse.csr_matrix((n_node, n_node))
    return sp_sparse.coo_matrix(
        (vals[keep], (rows[keep], cols[keep])), shape=(n_node, n_node)
    ).tocsr()


# =============================================================================
# 动网格节点层的插入
# =============================================================================

def insert_moving_mesh_layer(
    xy: np.ndarray,
    tri: np.ndarray,
    interface_nodes: Sequence[int],
    element_rotor_gap: Iterable[int],
    pairing,
    element_stator: Optional[Iterable[int]] = None,
    element_rotor: Optional[Iterable[int]] = None,
) -> MovingMeshLayer:
    """批量版 ``insert_moving_mesh_layer``。与 ``reference`` 版本逐位相同。"""
    xy = np.asarray(xy, dtype=float)
    tri = np.asarray(tri).copy()
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"tri 必须是 (N_elem, 3) 的连接表，实得形状 {tri.shape}")

    ordered = np.asarray(interface_nodes, dtype=np.int64)
    n_node = xy.shape[0]
    new_ids = n_node + np.arange(ordered.size, dtype=np.int64)

    # 新节点层：坐标逐位复制（fancy index 是纯复制，不改变数值）
    xy_new = np.concatenate([xy, xy[ordered]], axis=0)

    # 旧接口节点 → 新节点号；其余节点映射到自身（legacy 的成员测试 + 条件替换）
    remap = np.arange(xy_new.shape[0], dtype=np.int64)
    remap[ordered] = new_ids

    gap_index = np.fromiter(
        element_rotor_gap, dtype=np.int64, count=len(element_rotor_gap)
    )
    if gap_index.size:
        tri[gap_index] = remap[tri[gap_index]]

    rotor_to_stator = dict(zip(new_ids.tolist(), ordered.tolist()))
    stator_to_rotor = dict(zip(ordered.tolist(), new_ids.tolist()))
    pairing = pairing.rebind(stator_to_rotor)

    return MovingMeshLayer(
        xy=xy_new,
        tri=tri,
        stator_to_rotor=stator_to_rotor,
        rotor_to_stator=rotor_to_stator,
        interface_nodes=tuple(ordered.tolist()),
        rotor_nodes=tuple(new_ids.tolist()),
        pairing=pairing,
        node_stator=(
            nodes_of_elements(tri, element_stator)
            if element_stator is not None
            else None
        ),
        node_rotor=(
            nodes_of_elements(tri, element_rotor)
            if element_rotor is not None
            else None
        ),
    )


# =============================================================================
# 旋转边界
# =============================================================================

def _sorted_angle_table(layer: MovingMeshLayer):
    """``(角度升序数组, 对应节点号数组)``，稳定排序（= legacy 的 ``sorted(key=θ)``）。"""
    nodes = np.asarray(layer.interface_nodes, dtype=np.int64)
    angles = np.arctan2(layer.xy[nodes, 1], layer.xy[nodes, 0])
    order = np.argsort(angles, kind="stable")
    return angles[order], nodes[order]


def _rotating_boundary_terms(layer, parts, pos, a_eq, with_derivative):
    """批量算出插值行的三元组。对应 ``reference._rotating_boundary_terms``。"""
    xy = layer.xy
    n_node = xy.shape[0]
    sector = 2 * np.pi / parts

    angles_sorted, nodes_sorted = _sorted_angle_table(layer)
    rotor = np.asarray(layer.rotor_nodes, dtype=np.int64)

    # 逐元素与逐节点版同运算、同次序 —— 批量不改变任何单元素的取值
    theta = np.arctan2(xy[rotor, 1], xy[rotor, 0]) + pos
    phi = np.mod(theta, sector)
    other_region = np.mod(theta // sector, 2) > 0.5

    # 「第一个角度严格大于 φ 的下标」—— 与 reference 的 while 循环同一个定义
    index = np.searchsorted(angles_sorted, phi, side="right")
    bad_left = np.flatnonzero(index == 0)
    if bad_left.size:
        i_bad = int(rotor[bad_left[0]])
        raise ValueError(
            f"动节点 {i_bad} 的折叠角 φ = {phi[bad_left[0]]!r} 小于接口节点的最小角度 "
            f"{angles_sorted[0]!r}；接口节点角必须覆盖 [0, {sector!r}] 的左端点"
            f"（legacy 此处 assert index_stator > 0 失败）"
        )
    bad_right = np.flatnonzero(index == angles_sorted.size)
    if bad_right.size:
        i_bad = int(rotor[bad_right[0]])
        raise ValueError(
            f"动节点 {i_bad} 的折叠角 φ = {phi[bad_right[0]]!r} 超出接口节点的最大角度 "
            f"{angles_sorted[-1]!r}；接口节点角必须覆盖右端点"
            f"（legacy 此处会 IndexError）"
        )

    i_up = nodes_sorted[index]
    i_low = nodes_sorted[index - 1]
    theta_up = angles_sorted[index]
    theta_low = angles_sorted[index - 1]

    kp = (phi - theta_low) / (theta_up - theta_low)
    kn = 1.0 - kp
    s = np.where(np.logical_and(not a_eq, other_region), -1.0, 1.0)

    slave_to_master = layer.pairing.slave_to_master
    sigma_value = 1.0 if a_eq else -1.0
    slave_keys = np.fromiter(
        slave_to_master.keys(), dtype=np.int64, count=len(slave_to_master)
    )
    masters = np.fromiter(
        slave_to_master.values(), dtype=np.int64, count=len(slave_to_master)
    )
    is_slave = np.isin(np.concatenate([i_up, i_low]), slave_keys)
    sigma_up = np.where(is_slave[: i_up.size], sigma_value, 1.0)
    sigma_low = np.where(is_slave[i_up.size:], sigma_value, 1.0)

    # 从属 → 主动的替换：查表（非从属节点映射到自身）
    lookup = np.arange(n_node, dtype=np.int64)
    lookup[slave_keys] = masters
    idx_up = lookup[i_up]
    idx_low = lookup[i_low]

    coef_up = -s * sigma_up * kp
    coef_low = -s * sigma_low * kn

    ones = np.ones(rotor.size, dtype=float)
    rows = np.concatenate([rotor, rotor, rotor])
    cols = np.concatenate([rotor, idx_up, idx_low])
    vals = np.concatenate([ones, coef_up, coef_low])

    d_entries = None
    if with_derivative:
        dkp = 1.0 / (theta_up - theta_low)
        dkn = -dkp
        d_entries = (
            np.concatenate([rotor, rotor]),
            np.concatenate([idx_up, idx_low]),
            np.concatenate([-s * sigma_up * dkp, -s * sigma_low * dkn]),
        )

    brackets = dict(
        (int(r), (int(u), int(l))) for (r, u, l) in zip(rotor, idx_up, idx_low)
    )
    return (rows, cols, vals), d_entries, brackets


def rotating_boundary(
    layer: MovingMeshLayer, parts: int, pos: float, a_eq: bool,
    interp: str = "linear",
) -> RotatingBoundary:
    """批量版 ``rotating_boundary``。**``interp='linear'``（默认）与 ``reference`` 逐位相同。**

    ``interp='cubic'``（新方案，非逐位）**委托给 ``reference``** —— 三次插值不在
    生产热路径上（只在显式开启时使用），没有必要再写一份向量化实现，也不该让
    新分支有机会污染那条逐位路径。
    """
    if interp != "linear":
        return ref_mm.rotating_boundary(layer, parts, pos, a_eq, interp=interp)
    (rows, cols, vals), _d, brackets = _rotating_boundary_terms(
        layer, parts, pos, a_eq, with_derivative=False
    )
    return RotatingBoundary(
        matrix=_assemble(rows, cols, vals, layer.xy.shape[0]), brackets=brackets
    )


def rotating_boundary_with_derivative(
    layer: MovingMeshLayer, parts: int, pos: float, a_eq: bool,
    interp: str = "linear",
) -> RotatingBoundaryWithDerivative:
    """批量版 ``rotating_boundary_with_derivative``。**``'linear'`` 与 ``reference`` 逐位相同。**"""
    if interp != "linear":
        return ref_mm.rotating_boundary_with_derivative(
            layer, parts, pos, a_eq, interp=interp)
    (rows, cols, vals), d_entries, brackets = _rotating_boundary_terms(
        layer, parts, pos, a_eq, with_derivative=True
    )
    n_node = layer.xy.shape[0]
    d_rows, d_cols, d_vals = d_entries
    return RotatingBoundaryWithDerivative(
        matrix=_assemble(rows, cols, vals, n_node),
        brackets=brackets,
        d_matrix=_assemble(d_rows, d_cols, d_vals, n_node),
    )
