"""二维非线性静磁场的向量化实现（生产路径）。

与 `motorfem.kernels.reference.magnetostatic` **公开接口完全一致**，
只有内部实现不同：这里把三阶张量收缩压平成稀疏矩阵乘法，以换取速度。

设计原则
--------
1. 位置/选择矩阵**只构造一次**（在 ``PositionMatrices`` 里），全程复用。
   这与 legacy 的 ``get_K_vectorized`` 思路相同——那里也是把所有不变的
   结构一次性展平成 COO/CSR（``MMA10:1872-2119``）。
2. 每个位置矩阵的 docstring 必须写清**它在做什么收缩**，指向 `reference/` 里
   对应的那行朴素代码。这是本目录唯一的可读性要求，也是它能被维护的前提。
3. 除位置矩阵外，不引入任何新的数学。

红线
----
本模块**必须**遵守 `docs/重构设计文档.md` 第八节的红线清单（它参与位级对齐）。
与 `reference/` 不同，这里不允许用 `sorted()` 或改变区域集合的构造方式。

当前 parity 目标
----------------
``rtol = 1e-12``（不是位级一致）。原因是装配走 COO，其内部求和顺序与
``reference/`` 的逐单元 ``np.ix_`` 累加不同。**阶段 1 的验收标准就是这个**。
若将来需要位级一致，装配需改回按 legacy 的累加顺序写入——那时应单独记录。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp_sparse

from ..reference.magnetostatic import ElementGeometry, element_geometry, element_stiffness

__all__ = [
    "ElementGeometry",
    "element_geometry",
    "element_stiffness",
    "flux_density_magnitude",
    "PositionMatrices",
    "build_positions",
    "assemble_stiffness",
    "residual",
    "jacobian",
    "newton_solve",
]


# =============================================================================
# 位置矩阵族
# =============================================================================

@dataclass(frozen=True)
class PositionMatrices:
    """预计算的稀疏「位置/选择」矩阵。全部只依赖网格几何，与 A 和材料无关。

    它们是本模块速度的来源，也是可读性的代价——所以每一个都写明作用。

    Attributes
    ----------
    gather : (3·N_elem, N_node) csr
        ``gather @ A`` 把 A 摊平成「扩展空间」向量 ``A_exp``，其中
        ``A_exp[3e + s] = A[tri[e, s]]``。
        **作用**：让后续所有单元级运算都变成对长度 3·N_elem 的向量做逐元素运算。

        对应 reference: ``A_e = A[geom.tri]``

    dot_b : (N_elem, N_node) csr
        ``dot_b @ A`` 一次算出所有单元的 ``b·A_e``（reference 里的 ``gS``）。
        **作用**：取代 reference 的 ``np.einsum("es,es->e", geom.b, A_e)``。

    dot_c : (N_elem, N_node) csr
        同上，算 ``c·A_e``（reference 里的 ``gC``）。

    scatter_9 : (N_node², 9·N_elem) csr
        ``scatter_9 @ vals_9`` 把每个单元的 9 个局部块值累加进全局矩阵，
        返回 COO 的 data 数组。行索引已按 ``(i, j)`` 展平。
        **作用**：取代 reference 的
        ``coo_matrix((vals.reshape(n,9).ravel(), (rows.ravel(), cols.ravel())))``。

    coo_rows, coo_cols : (9·N_elem,) int
        对应的 COO 行/列索引。

    sel_i : (9·N_elem, 3·N_elem) csr
        ``sel_i @ g_exp`` 取出每个 (单元, 局部位置 k) 组合的 **行** 分量
        ``g_e[i]``，其中 i = k // 3。

    sel_j : (9·N_elem, 3·N_elem) csr
        同上，取 **列** 分量 ``g_e[j]``，其中 j = k % 3。

    Notes
    -----
    ``sel_i`` / ``sel_j`` 这一对就是 legacy 的 ``gswap1_mat`` / ``gswap2_mat``
    （``MMA10:2250``）。它们的唯一作用是「取第 i 个 / 取第 j 个」，从而让
    ``(sel_i @ g) * (sel_j @ g)`` 逐元素相乘得到 ``g_e[i]·g_e[j]``——也就是
    reference 里那个 ``np.outer(g, g)`` 展平后的 9 个分量。

    **这就是「把三阶张量 (单元, 局部i, 局部j) 压平成稀疏矩阵」的全部含义。**
    """

    gather: sp_sparse.csr_matrix
    dot_b: sp_sparse.csr_matrix
    dot_c: sp_sparse.csr_matrix
    scatter_9: sp_sparse.csr_matrix
    sel_i: sp_sparse.csr_matrix
    sel_j: sp_sparse.csr_matrix
    coo_rows: np.ndarray
    coo_cols: np.ndarray
    n_elem: int
    n_node: int


def build_positions(geom: ElementGeometry, n_node: int) -> PositionMatrices:
    """构造 ``PositionMatrices``。每个网格只调用一次。

    Parameters
    ----------
    geom : ElementGeometry
    n_node : int

    Notes
    -----
    本函数内部的 Python 循环只用于**构造索引**，不参与数值计算，因此可以随意
    使用 ``np.repeat`` / ``np.tile`` 等展开写法，不影响任何浮点求和顺序。
    真正参与数值计算的只有返回的那几个稀疏矩阵的 ``@``。
    """
    tri = geom.tri
    n_elem = tri.shape[0]

    # ---- 扩展空间的索引 ----
    # 扩展序号：A_exp[3e + s] = A[tri[e, s]]
    exp_rows = np.arange(3 * n_elem)
    exp_cols = tri.ravel()

    gather = sp_sparse.coo_matrix(
        (np.ones(3 * n_elem), (exp_rows, exp_cols)), shape=(3 * n_elem, n_node)
    ).tocsr()

    # ---- b·A_e 与 c·A_e ----
    def _dot_matrix(coef: np.ndarray) -> sp_sparse.csr_matrix:
        """行 e、列 tri[e, s]、值 coef[e, s] -> coef @ A 给出 Σ_s coef[e,s]·A[tri[e,s]]。"""
        rows = np.repeat(np.arange(n_elem), 3)
        cols = tri.ravel()
        vals = coef.ravel()
        return sp_sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n_elem, n_node)
        ).tocsr()

    dot_b = _dot_matrix(geom.b)
    dot_c = _dot_matrix(geom.c)

    # ---- 单元 9 个块位置 ----
    coo_rows = np.repeat(tri, 3, axis=1).ravel()          # (9·N_elem,) 全局行索引
    coo_cols = np.tile(tri, (1, 3)).ravel()               # (9·N_elem,) 全局列索引

    # scatter_9：把 (9·N_elem,) 的块值累加进全局矩阵。
    # 行号用「展平后的位置序号」k = 9e + t，这样多个单元写入同一 (i,j) 时
    # 由 COO 求和自动完成。
    block_row_index = np.arange(9 * n_elem)
    scatter_9 = sp_sparse.coo_matrix(
        (np.ones(9 * n_elem), (block_row_index, block_row_index + 0)),
        shape=(9 * n_elem, 9 * n_elem),
    ).tocsr()

    # ---- sel_i / sel_j（= legacy 的 gswap1 / gswap2）----
    # 对第 k = 9e + t 个块位置，局部行 t//3、局部列 t%3
    t_idx = np.arange(9 * n_elem) % 9
    local_i = t_idx // 3
    local_j = t_idx % 3
    elem_of_block = np.repeat(np.arange(n_elem), 9)

    sel_i = sp_sparse.coo_matrix(
        (np.ones(9 * n_elem), (np.arange(9 * n_elem), 3 * elem_of_block + local_i)),
        shape=(9 * n_elem, 3 * n_elem),
    ).tocsr()
    sel_j = sp_sparse.coo_matrix(
        (np.ones(9 * n_elem), (np.arange(9 * n_elem), 3 * elem_of_block + local_j)),
        shape=(9 * n_elem, 3 * n_elem),
    ).tocsr()

    return PositionMatrices(
        gather=gather, dot_b=dot_b, dot_c=dot_c,
        scatter_9=scatter_9, sel_i=sel_i, sel_j=sel_j,
        coo_rows=coo_rows, coo_cols=coo_cols,
        n_elem=n_elem, n_node=n_node,
    )


# =============================================================================
# 单元量
# =============================================================================

def flux_density_magnitude(
    A: np.ndarray, geom: ElementGeometry, pos: PositionMatrices | None = None
) -> np.ndarray:
    """单元磁密幅值 ``|B|``，形状 ``(N_elem,)``。

    对应 reference::

        B = sqrt((b·A_e)² + (c·A_e)²) / (2Δ)

    向量化路径：把两个点积换成两次稀疏矩阵乘法（``dot_b`` / ``dot_c``），
    其余逐元素。数学完全一致。
    """
    if pos is None:
        pos = build_positions(geom, A.shape[0])
    b_dot_A = pos.dot_b @ A                               # (N_elem,)
    c_dot_A = pos.dot_c @ A
    B = np.sqrt(b_dot_A ** 2 + c_dot_A ** 2) / (2.0 * geom.delta)
    return np.clip(B, 1e-20, np.inf)


def assemble_stiffness(
    S: np.ndarray, nu: np.ndarray, tri: np.ndarray, n_node: int,
    pos: PositionMatrices | None = None,
) -> sp_sparse.csr_matrix:
    """装配 ``Σ_e ν_e S_e``。与 reference 同数学，走 COO 一次成阵。

    这里不依赖位置矩阵（只需一次 ``reshape`` + ``ravel``），所以保留与
    reference 相同的实现风格，便于比对。
    """
    n_elem = tri.shape[0]
    rows = np.repeat(tri, 3, axis=1)
    cols = np.tile(tri, (1, 3))
    vals = (nu[:, None, None] * S).reshape(n_elem, 9)
    K = sp_sparse.coo_matrix(
        (vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n_node, n_node)
    )
    return K.tocsr()


def element_g(
    A: np.ndarray, geom: ElementGeometry, S: np.ndarray, pos: PositionMatrices
) -> np.ndarray:
    """扩展空间里的 ``g_e = S_e A_e``，形状 ``(N_elem, 3)``。

    对应 reference 的 ``g = np.einsum("eij,ej->ei", S, A_e)``。

    向量化路径利用了 ``S_e = (1/(4Δ))·(b bᵀ + c cᵀ)`` 的显式结构::

        S_e A_e = (1/(4Δ))·( b·(b·A_e) + c·(c·A_e) )

    于是只需两个点积（``dot_b`` / ``dot_c``）加逐元素运算，**不必显式构造
    ``S`` 的 3×3 块再做批量矩阵乘法**。这是数学恒等式带来的加速，不是近似。
    """
    b_dot_A = pos.dot_b @ A                               # (N_elem,)
    c_dot_A = pos.dot_c @ A
    inv4d = 1.0 / (4.0 * geom.delta)                      # (N_elem,)
    g = inv4d[:, None] * (geom.b * b_dot_A[:, None] + geom.c * c_dot_A[:, None])
    return g


# =============================================================================
# 残差与 Jacobian
# =============================================================================

def residual(
    A: np.ndarray,
    geom: ElementGeometry,
    S: np.ndarray,
    nu_of_B: Callable[[np.ndarray], np.ndarray],
    R: np.ndarray,
    pos: PositionMatrices | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """残差 ``F(A) = K(A)A − R``。与 reference 数值等价。"""
    if pos is None:
        pos = build_positions(geom, A.shape[0])
    B = flux_density_magnitude(A, geom, pos)
    nu = nu_of_B(B)
    K = assemble_stiffness(S, nu, geom.tri, A.shape[0], pos)
    return K @ A - R, nu


def jacobian(
    A: np.ndarray,
    geom: ElementGeometry,
    S: np.ndarray,
    nu: np.ndarray,
    dnu_dB: Callable[[np.ndarray], np.ndarray],
    pos: PositionMatrices | None = None,
) -> sp_sparse.csr_matrix:
    """Jacobian。

    对应 reference::

        g     = S_e A_e                                  # (N_elem, 3)
        coef  = dν/dB / (Δ·B)                            # (N_elem,)
        J_sat = Σ_e coef_e · g_e ⊗ g_e

    向量化路径（**这就是 legacy ``gswap1/gswap2`` 的全部内容**）::

        g_exp  = gather 后的 g，长度 3·N_elem        （扩展空间）
        J_sat_data = (sel_i @ g_exp) * (sel_j @ coef_rep * g_exp)

    ``sel_i @ g_exp`` 取每个块位置的 ``g_e[i]``，``sel_j @ (coef·g)_exp`` 取
    ``coef_e·g_e[j]``，逐元素相乘即得 9 个分量 ``coef_e·g_e[i]·g_e[j]``，
    正好是 ``coef_e · g_e ⊗ g_e`` 的展平。最后 scatter 进全局矩阵。
    """
    if pos is None:
        pos = build_positions(geom, A.shape[0])

    B = flux_density_magnitude(A, geom, pos)
    n_elem, n_node = pos.n_elem, pos.n_node

    K = assemble_stiffness(S, nu, geom.tri, n_node, pos)

    g = element_g(A, geom, S, pos)                        # (N_elem, 3)
    coef = dnu_dB(B) / (geom.delta * B)                   # (N_elem,)

    g_exp = g.ravel()                                     # 3·N_elem，已按 (e, s) 排布
    coef_rep = np.repeat(coef, 3)                         # 扩展到每个局部槽

    # ★ 秩 1 外积的展平：取第 i 个 × 取第 j 个
    sat_data = (pos.sel_i @ g_exp) * (pos.sel_j @ (coef_rep * g_exp))

    J_sat = sp_sparse.coo_matrix(
        (sat_data, (pos.coo_rows, pos.coo_cols)), shape=(n_node, n_node)
    ).tocsr()

    return (K + J_sat).tocsr()


def newton_solve(*args, **kwargs):
    """与 reference 同实现（Newton 迭代本身没有可向量化的部分，
    ``spsolve`` 已是外部库调用）。直接复用 reference 的实现以保证行为一致。"""
    from ..reference.magnetostatic import newton_solve as _ref_newton

    return _ref_newton(*args, **kwargs)
