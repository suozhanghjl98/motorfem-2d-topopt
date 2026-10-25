"""位置矩阵族装配的保真搬移（内核 8 的 fast 层）。

本层是 ``get_K_vectorized``（``MMA10:1693-1773``、``1872-2072``）与
``get_R_vector``（``:2121-2219``）的**逐位**搬移：容器、写入顺序、乘法结合顺序、
甚至 ``T`` 的**内存布局**都照原样保留。数学、18 项的含义、F4 讨论与实测事实全部写在
``reference/assembly.py`` 的模块 docstring 里，本文件不重复。

------------------------------------------------------------------------------
为什么这一层「照抄」比「写清楚」重要
------------------------------------------------------------------------------

内核 8 的产出是**位置矩阵族**：位置（``K0_row/col``、``gswap1/2_mat`` 的 0/1 结构、
``_index``、``Jabobi2_row/col``）与值（``K0_data``、``gxi_mat`` 的数据）分开存，
下游靠 ``csr((K0_data * (Kconst_mat @ γ), (K0_row, K0_col)))`` 一类式子重装矩阵。
因此**任何**对构造顺序、容器类型或布局的「整理」都会渗进下游的浮点结果。

三处必须原样保留的细节（都有实测支撑）：

1. **``_T_temp = np.zeros((3, N_node))`` + ``_T_temp[:, _index]``**。这个花式索引返回
   **Fortran 连续**的 ``(3, k)`` 数组（numpy 的布局规则），而 ``Bxcoef[e] @ T`` 在 F
   连续与 C 连续两种操作数上走**不同的 BLAS 内核** ⇒ 末位不同（实测 1 ULP）。
   ``reference`` 直接建 ``(3, k)``（C 连续）便于阅读，所以这两项在 ref/fast 之间
   只承诺 ``rtol=1e-12``；fast 与 legacy 之间是逐位一致。
   ⇒ **不要**把这里的 ``[:, _index]`` 改写成先建小数组再填值。
2. **``_index`` 的收集顺序**（``set`` + ``add`` + ``list``）决定 ``T`` 的列序 ⇒ 决定
   ``Bxcoef@T``、``TᵀK0T`` 的求和次序（``reference`` 模块 docstring 第五节）。
   两个实现共用 ``reference`` 里的 ``_active_index`` —— 这是本仓库唯一一处
   **``reference`` 必须与 ``fast`` 共用代码**的地方。
3. **``Bxcoef_mat``/``Bycoef_mat`` 用 ``lil`` 逐行赋值**（会丢弃恰好为零的元），
   其余矩阵用 ``csr_matrix((data, (row, col)))``（保留显式零、对重复项求和）。
   两种语义都不能混。

------------------------------------------------------------------------------
本层复用 ``reference`` 的三件东西（不是偷懒，是「无差异就不建第二份」）
------------------------------------------------------------------------------

- ``ElementOperators`` 等数据结构（与内核 6/7 的分层约定一致）；
- ``_active_index``（理由见上）；
- ``signed_phase_current`` / ``get_R_vector`` / ``as_legacy_tuple``：``R`` 的装配
  legacy 本身就是逐单元循环，**向量化不可能保持逐位**（花式索引的 ``+=`` 会丢重复
  贡献，``np.add.at`` 会改变同一节点的累加次序 —— 红线 5）；分类与元组布局是纯记账。
  故这三件直接复用 ``reference`` 的实现（等价于内核 1「B-H 曲线」只写一份的先例）。

------------------------------------------------------------------------------
红线
------------------------------------------------------------------------------

本层完整遵守 ``CLAUDE.md`` §3：不 ``sorted()`` 区域集合、不改 ``set`` 构造表达式、
不合并 ``for i_elem in range(N_element)`` 的嵌套顺序、不把 ``+=`` 改成 ``np.add.at``。
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse as sp_sparse

# 数据结构与「无差异」的三件：直接复用 reference（见模块 docstring）
from ..reference.assembly import (
    ElementOperators,
    ElementTransform,
    TopologyContext,
    _active_index,          # F4 关键：必须与 reference 用同一个 helper
    as_legacy_tuple,        # noqa: F401  （公开接口一致性：两层的 __all__ 相同）
    get_R_vector,           # noqa: F401
    signed_phase_current,   # noqa: F401
)

__all__ = [
    "TopologyContext",
    "ElementTransform",
    "ElementOperators",
    "transformation_matrix",
    "element_operators",
    "signed_phase_current",
    "get_R_vector",
    "as_legacy_tuple",
]


# =============================================================================
# 单元变换矩阵（legacy 的稠密 _T_temp + 花式索引）
# =============================================================================

def _dense_transformation(i_elem: int, ctx: TopologyContext, rotating, n_node: int):
    """legacy 的 ``_get_transformation_mat_for_elem``：先建 ``(3, N_node)`` 再切片。

    返回 ``(T, index)``，其中 ``T`` 是**花式索引切片**后的 ``(3, k)`` —— 它按 numpy 的
    布局规则是 **F 连续**的，这正是下游 ``Bxcoef@T`` 与 legacy 逐位一致的前提（见模块
    docstring 第 1 条）。
    """
    i_elem = int(i_elem)

    if i_elem in ctx.element_rotor_gap:
        if i_elem not in ctx.element_slave:
            node_i, node_j, node_m = (int(n) for n in ctx.tri[i_elem])
            _ki = 1
            _kj = 1
            _km = 1
        else:
            node_i, node_j, node_m = (int(n) for n in ctx.virtual_tri[i_elem])
            _ki = 1 if node_i in ctx.master_nodes else (1 if ctx.a_eq else -1)
            _kj = 1 if node_j in ctx.master_nodes else (1 if ctx.a_eq else -1)
            _km = 1 if node_m in ctx.master_nodes else (1 if ctx.a_eq else -1)

        _T_temp = np.zeros((3, n_node))
        index_set = set()
        for s, node, sign in ((0, node_i, _ki), (1, node_j, _kj), (2, node_m, _km)):
            if node in rotating.brackets:
                # ⚠ brackets 长度随插值阶数变（linear 2 / cubic 4）⇒ 遍历。
                # linear 下与原先的 (_node_1, _node_2) 写法逐位相同（同序、同两项）。
                for _c in rotating.brackets[node]:
                    _c = int(_c)
                    _T_temp[s, _c] += -rotating.matrix[node, _c] * sign
                    index_set.add(_c)
            else:
                _T_temp[s, node] += sign
                index_set.add(node)

        index = list(index_set)
        return _T_temp[:, index], index

    if i_elem in ctx.element_slave:
        node_i, node_j, node_m = (int(n) for n in ctx.virtual_tri[i_elem])
        _ki = 1 if node_i in ctx.master_nodes else (1 if ctx.a_eq else -1)
        _kj = 1 if node_j in ctx.master_nodes else (1 if ctx.a_eq else -1)
        _km = 1 if node_m in ctx.master_nodes else (1 if ctx.a_eq else -1)
        return np.diag([_ki, _kj, _km]), [node_i, node_j, node_m]

    node_i, node_j, node_m = (int(n) for n in ctx.tri[i_elem])
    return np.eye(3), [node_i, node_j, node_m]


def transformation_matrix(
    i_elem: int, ctx: TopologyContext, rotating
) -> ElementTransform:
    """与 ``reference`` 同签名同结果（数值逐位相同；``T`` 的**内存布局**不同）。

    ``reference`` 建 C 连续的 ``(3, k)``；本层沿用 legacy 的
    ``_T_temp[:, index]``（F 连续）。数值逐位相同，但下游乘积可能差末位 —— 见模块
    docstring 第 1 条。C 级测试对 ``matrix`` 用 ``np.array_equal``（值相同），
    对**乘积**用 ``rtol=1e-12``。
    """
    matrix, index = _dense_transformation(i_elem, ctx, rotating, _width(ctx))
    return ElementTransform(matrix=matrix, index=tuple(index))


def _width(ctx: TopologyContext) -> int:
    """legacy ``_T_temp`` 的列数 = 节点数（``TopologyContext.n_node``）。

    只要宽度 ≥ 所有被写入的列号，切片后的 ``T`` 就完全一致；这里用真实节点数，
    与 legacy 逐字一致以便对拍。
    """
    if ctx.n_node is None:
        raise ValueError(
            "TopologyContext.n_node 缺失：fast 层的稠密 T 需要它（legacy 用 (3, N_node)）。"
            "请用 TopologyContext.from_boundaries(..., n_node=layer.n_node) 构造。"
        )
    return int(ctx.n_node)


def element_operators(
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
    """与 ``reference.element_operators`` 同签名。本层是 ``MMA10:1872-2072`` 的搬移。"""
    n_element = np.asarray(ctx.tri).shape[0]
    n_node = int(n_node)
    iron = set(int(e) for e in element_iron)
    topo = set(int(e) for e in element_topo_opti)

    K0_data, K0_row, K0_col = [], [], []
    Kconst_mat_data, Kconst_mat_row, Kconst_mat_col = [], [], []
    NK0 = 0

    Bxcoef_mat = sp_sparse.lil_matrix((n_element, n_node))
    Bycoef_mat = sp_sparse.lil_matrix((n_element, n_node))

    is_vaccum = np.zeros(n_element)
    is_SIMP = np.zeros(n_element)
    is_non_SIMP = np.zeros(n_element)
    is_linear_iron = np.zeros(n_element)

    g0i_mat_data, g0i_mat_row, g0i_mat_col = [], [], []
    Nrow_g0i = 0
    gconst_mat_data, gconst_mat_row, gconst_mat_col = [], [], []
    gswap1_mat_data, gswap1_mat_row, gswap1_mat_col = [], [], []
    gswap2_mat_data, gswap2_mat_row, gswap2_mat_col = [], [], []
    Ngswap = 0

    Jabobi2_row, Jabobi2_col = [], []
    dK_p_dgamma_A_row, dK_p_dgamma_A_col = [], []

    for i_elem in range(n_element):
        _T, _index = _dense_transformation(i_elem, ctx, rotating, n_node)

        Bxcoef_mat[i_elem, _index] = bxcoef[i_elem] @ _T
        Bycoef_mat[i_elem, _index] = bycoef[i_elem] @ _T

        # --- 材料分类 ---
        # ★ 2026-10-17 修正：线性铁磁开关**只影响非拓扑区的纯铁芯**。
        #   拓扑区必须保持 ``is_SIMP`` —— 否则 ρ 完全失效（实测线性档下把
        #   ρ 从 1.0 改到 0.1，残差的相对差是 **0.000e+00**），拓扑优化
        #   在线性材料档下根本挖不出磁障。
        #   线性档下 SIMP 支走线性材料（``μ = μ₀·μ_r·ρ₁``，``dν/dB ≡ 0``），
        #   见 ``fast/material.CachedMaterial.linear_simp``。
        if i_elem in iron:
            if i_elem in topo:
                is_SIMP[i_elem] = 1
            elif not use_nonlinear_iron:
                is_linear_iron[i_elem] = 1
            else:
                is_non_SIMP[i_elem] = 1
        else:
            is_vaccum[i_elem] = 1

        _K0elem = np.array([
            [k0elem[i_elem, 0], k0elem[i_elem, 1], k0elem[i_elem, 2]],
            [k0elem[i_elem, 1], k0elem[i_elem, 3], k0elem[i_elem, 4]],
            [k0elem[i_elem, 2], k0elem[i_elem, 4], k0elem[i_elem, 5]]
        ])

        _K0 = _T.T @ (_K0elem @ _T)
        for i_row, _row in enumerate(_index):
            for i_col, _col in enumerate(_index):
                K0_data.append(_K0[i_row, i_col])
                K0_row.append(_row)
                K0_col.append(_col)

                Kconst_mat_data.append(1)
                Kconst_mat_row.append(NK0)
                Kconst_mat_col.append(i_elem)

                g0i_mat_data.append(_K0[i_row, i_col])
                g0i_mat_row.append(Nrow_g0i + i_row)
                g0i_mat_col.append(_col)

                gswap1_mat_data.append(1)
                gswap1_mat_row.append(Ngswap)
                gswap1_mat_col.append(Nrow_g0i + i_row)

                gswap2_mat_data.append(1)
                gswap2_mat_row.append(Ngswap)
                gswap2_mat_col.append(Nrow_g0i + i_col)

                Jabobi2_row.append(_row)
                Jabobi2_col.append(_col)

                NK0 += 1
                Ngswap += 1

            gconst_mat_data.append(1)
            gconst_mat_row.append(Nrow_g0i + i_row)
            gconst_mat_col.append(i_elem)

            dK_p_dgamma_A_row.append(i_elem)
            dK_p_dgamma_A_col.append(_row)

        Nrow_g0i += len(_index)

    Kconst_mat = sp_sparse.csr_matrix(
        (Kconst_mat_data, (Kconst_mat_row, Kconst_mat_col)), shape=(NK0, n_element))
    g0i_mat = sp_sparse.csr_matrix(
        (g0i_mat_data, (g0i_mat_row, g0i_mat_col)), shape=(Nrow_g0i, n_node))
    gconst_mat = sp_sparse.csr_matrix(
        (gconst_mat_data, (gconst_mat_row, gconst_mat_col)), shape=(Nrow_g0i, n_element))
    gswap1_mat = sp_sparse.csr_matrix(
        (gswap1_mat_data, (gswap1_mat_row, gswap1_mat_col)), shape=(Ngswap, Nrow_g0i))
    gswap2_mat = sp_sparse.csr_matrix(
        (gswap2_mat_data, (gswap2_mat_row, gswap2_mat_col)), shape=(Ngswap, Nrow_g0i))

    return ElementOperators(
        k0_data=np.asarray(K0_data),
        k0_row=np.asarray(K0_row),
        k0_col=np.asarray(K0_col),
        kconst_mat=Kconst_mat,
        bxcoef_mat=Bxcoef_mat.tocsr(),
        bycoef_mat=Bycoef_mat.tocsr(),
        is_vaccum=is_vaccum,
        is_simp=is_SIMP,
        is_non_simp=is_non_SIMP,
        is_linear_iron=is_linear_iron,
        g0i_mat=g0i_mat,
        gconst_mat=gconst_mat,
        gswap1_mat=gswap1_mat,
        gswap2_mat=gswap2_mat,
        jabobi2_row=np.asarray(Jabobi2_row),
        jabobi2_col=np.asarray(Jabobi2_col),
        dk_p_dgamma_a_row=np.asarray(dK_p_dgamma_A_row),
        dk_p_dgamma_a_col=np.asarray(dK_p_dgamma_A_col),
        n_k0=NK0,
        n_gswap=Ngswap,
        n_row_g0i=Nrow_g0i,
    )
