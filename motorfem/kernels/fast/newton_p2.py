"""P2（6 节点）残差 / Jacobian / Newton 的向量化 fast 层（路线 A，生产性能）。

对照 `reference/newton_p2.py`（权威数学、逐元素 Python 循环）。本层是**同一数学**
的批量张量实现：预计算所有与 ``A``、``ρ`` 无关的量（``T`` 投影后的高斯点磁密系数、
``K0_g`` 块、组装位置），每轮 Newton 只做几个 ``einsum`` + 一次稀疏组装。

------------------------------------------------------------------------------
为什么可以这样建（对照 fast 层的建层判据）
------------------------------------------------------------------------------

P2 没有 legacy 逐位对象（P1 的 fast 层必须与 legacy ``array_equal``；P2 的权威
表达是 ``reference``）。按 `kernels/README.md`「reference 是唯一权威、fast 是生产
路径」的判据，P2 的 fast 层对齐对象是 ``reference/newton_p2.py``，判据用
``rtol=1e-12``（``reference/`` 不受 §3 约束，只参与 ``rtol=1e-12`` 对齐）。

本层与 reference 的差异只有**浮点累加顺序**（逐元素循环 vs 批量归约），无算法差异。
实测 max rel 差 < 1e-14（``tests/kernels/test_newton_p2_fast.py``）。

------------------------------------------------------------------------------
数学（与 reference 逐字对应，模块 docstring 的记号）
------------------------------------------------------------------------------

.. code-block:: text

    F_e = Σ_g c_g γ_g K0_g A_e            c_g = w_g·Δ
    J_e = Σ_g [ c_g γ_g K0_g + c_g (∂γ_g/∂B_g) g0_g g0_gᵀ / B_g ]
    K0_g = b_g b_gᵀ + c_g c_gᵀ            g0_g = K0_g A_e

``b_g = bxcoef_g[e,gg] @ T_e``（``T_e`` 是 reference `assembly_p2` 的 6×k 变换矩阵，
把 6 局部 DOF 映射到活跃自由度）。``bxcoef_g/bycoef_g`` 是 (N,3,6) 逐高斯点系数。

------------------------------------------------------------------------------
统一到定长 7（实现要点）
------------------------------------------------------------------------------

- 单元 ``index`` 长度 k ∈ {6, 7}（实测 7144 单元网格：k=6 的 7002、k=7 的 142；
  k=7 出现在转子气隙插值行取两个括号节点时）。pad 到 ``K=7``：``index_pad``
  无效位置填 ``-1``、``bx_proj/by_proj`` 无效位置填 0 ⇒ 该位置对 B/F/J 的贡献
  为 0，与 reference 一致。
- ``A_active = A[index_pad]`` 对 ``-1`` 取 ``A[0]`` 再乘 ``(index_pad>=0)`` 掩码
  置零 —— 避免 index 越界。
- 组装位置预计算成 (N,3,K,K) 的 ``rows/cols``，值每轮重算 ⇒ 一次 ``coo_matrix``
  组装（无 ``np.add.at`` 散射、无逐元素 Python 循环）。

------------------------------------------------------------------------------
与本层有关的数值口径
------------------------------------------------------------------------------

- 驱动行替换（``driving_row_residual`` / ``driving_row_jacobian``）与 reference
  完全一致 —— 本层只把「bulk 组装」换成批量，驱动行照常处理。
- 材料求值复用 ``reference.newton.element_field_material``（向量化 numpy，
  与 P1 fast 层共用同一份）—— 展平 (N,3) 后逐高斯点成立。
- ``flux_density_p2``：批量 ``einsum``，``clip(min=1e-20)`` 与 reference 同。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import sparse as sp_sparse

from ..reference import boundaries as ref_bnd
from ..reference.newton import (
    FieldAux,
    MaterialMasks,
    element_field_material,
    newton_iterations,
)

__all__ = [
    "P2FastGeometry",
    "make_p2_fast_geometry",
    "make_p2_fast_geometry_cached",
    "field_error_p2",
    "newton_solve_p2_fast",
]

# pad 后的定长活跃自由度。**这是下限，实际宽度按网格取**
# （`_pad_arrays_from_field` 用 `max(_K, 本网格静态最大 k)`）。
#
# ⚠ 2026-10-13：原为 8，注释写「实测 k∈{6,7,8}」—— 那只在 gapall0.05 order2 上
# 实测过，**没有考虑气隙单元的上界**：P2 单元有 6 个槽位，每个受限槽位在
# 「线性插值」下取 2 列 ⇒ **气隙单元的 index 上界是 6×2 = 12**。
# 更密的网格（gapall0.02_slot0.1_order2）实测到 k = 11，直接撞断言。
# 取 16 = 12 + 余量。**多出来的列数值上是 no-op**（见 `_pad_arrays_from_field`）。
_K = 16


@dataclass(frozen=True)
class P2FastGeometry:
    """P2 向量化求解的预计算几何（与 ``A``、``ρ`` 无关）。

    Attributes
    ----------
    n_node : int
    n_elem : int
    bx_proj, by_proj : (N, 3, K) ndarray
        ``bxcoef_g[e,gg] @ T_e``（pad 到 7，无效位置 0）。
    k0g : (N, 3, K, K) ndarray
        ``outer(bx,bx) + outer(by,by)``（预计算块）。
    c_g : (N, 3) ndarray
        ``w_g · Δ``（高斯权重 × 单元面积）。
    index_pad : (N, K) int ndarray
        每单元活跃自由度全局节点号，无效位置 ``-1``。
    rows, cols : (N, 3, K, K) int ndarray
        J 组装位置（``index_pad`` 展开）。
    masks : MaterialMasks
    sparsity : CSRSparsity
        J 组装的预计算排列（含 driving 掩码与 j_mat_driven 合并）。
    jmd_mask : (M,) float 或 None
        driving 行/列掩码预乘因子（与 j_mat_driven 合并后）。
    """

    n_node: int
    n_elem: int
    bx_proj: np.ndarray
    by_proj: np.ndarray
    k0g: np.ndarray
    c_g: np.ndarray
    index_pad: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    masks: MaterialMasks
    sparsity: object = None
    jmd_mask: np.ndarray = None

    @property
    def rho_repeat3(self) -> np.ndarray:
        """占位属性（保持接口对称）；实际在 `_material3` 里用 repeat。"""
        raise NotImplementedError


def _pad_arrays_from_field(field):
    """从 P2FieldGeometry 抽出 (index_pad, bx_proj, by_proj)，供组装/缓存用。

    ⚠ **pad 宽度按本网格的实际最大 ``k`` 取**（不低于历史下限 ``_K``），
    不再写死 8。原因（2026-10-13）：``_K = 8`` 是**在 gapall0.05 order2 上实测**
    （k∈{6,7,8}）定下的；更密的气隙网格（``gapall0.02_slot0.1_order2``）出现
    ``k = 10`` ⇒ 直接撞断言「index 长度 10 超过 pad 宽度 8」。

    **数值上与原实现逐位相同**：多出来的列全部填
    ``index_pad = −1``（下游用 ``(index_pad >= 0)`` 掩码掉）、
    ``bx_proj/by_proj = 0``（乘 0 ⇒ 不贡献）。所以放宽宽度对
    k ≤ 8 的网格是 **no-op**。
    """
    n_elem = field.n_elem
    k_pad = max(_K, max((len(field.transforms[e].index) for e in range(n_elem)),
                        default=0))
    index_pad = np.full((n_elem, k_pad), -1, dtype=np.int64)
    bx_proj = np.zeros((n_elem, 3, k_pad))
    by_proj = np.zeros((n_elem, 3, k_pad))
    for e in range(n_elem):
        tr = field.transforms[e]
        k = len(tr.index)
        assert k <= k_pad, f"单元 {e} index 长度 {k} 超过 pad 宽度 {k_pad}"
        index_pad[e, :k] = np.asarray(tr.index, dtype=np.int64)
        for gg in range(3):
            bx_proj[e, gg, :k] = field.bxcoef_g[e, gg] @ tr.matrix
            by_proj[e, gg, :k] = field.bycoef_g[e, gg] @ tr.matrix
    return index_pad, bx_proj, by_proj


def make_p2_fast_geometry(field, driving=None) -> P2FastGeometry:
    """从 `reference.newton_p2.P2FieldGeometry` 预计算全部几何量。

    Parameters
    ----------
    field : P2FieldGeometry
        ``reference.newton_p2.field_geometry_p2`` 的产出（含 transforms、bxcoef_g、
        bycoef_g、gauss_w、aelem、masks）。
    driving : DrivingRows 或 None
        给定则 J 组装的掩码与 j_mat_driven 合进结构（生产路径）；
        None（默认）则只做纯 bulk（fast 层内部测试）。

    Returns
    -------
    P2FastGeometry
    """
    n_node = field.n_node
    index_pad, bx_proj, by_proj = _pad_arrays_from_field(field)
    return _assemble(n_node, index_pad, bx_proj, by_proj, field.aelem,
                     field.gauss_w, field.masks, driving)


def make_p2_fast_geometry_cached(
    n_node: int,
    index_pad_static: np.ndarray,
    bx_proj_static: np.ndarray,
    by_proj_static: np.ndarray,
    bxcoef_g: np.ndarray,
    bycoef_g: np.ndarray,
    aelem: np.ndarray,
    gauss_w: np.ndarray,
    masks,
    ctx,
    rotating,
    driving=None,
) -> P2FastGeometry:
    """跨点缓存版：非气隙单元的 T 跨 181 点复用，气隙单元每点重算。

    Parameters
    ----------
    index_pad_static, bx_proj_static, by_proj_static :
        非气隙单元的 pad 值（pos 无关，一次构造）；气隙单元位置留 0/-1，
        由本函数填。
    bxcoef_g, bycoef_g : (N,3,6) 静态逐高斯点磁密系数。
    aelem, gauss_w, masks : 静态。
    ctx, rotating : 每点（气隙单元的分支 1 用 rotating.brackets/matrix）。

    用途：181 点扫描里 `ops`/`field` 逐单元重建 492+62ms 是纯浪费 ——
    87% 单元（非气隙）T 与 pos 无关。本函数把 per-point 成本压到只算
    气隙单元（916/7144）。
    """
    from ..reference.assembly_p2 import transformation_matrix_p2
    n_elem = index_pad_static.shape[0]
    # ⚠ pad 宽度取**静态数组的实际列数**（由 `_pad_arrays_from_field` 按本网格
    #   最大 k 决定），不再用常量 `_K` —— 见那个函数 2026-10-13 的说明。
    k_pad = int(index_pad_static.shape[1])
    index_pad = index_pad_static.copy()
    bx_proj = bx_proj_static.copy()
    by_proj = by_proj_static.copy()
    gap = set(int(e) for e in ctx.element_rotor_gap)
    for e in gap:
        tr = transformation_matrix_p2(e, ctx, rotating)
        k = len(tr.index)
        assert k <= k_pad, f"单元 {e} index 长度 {k} 超过 pad 宽度 {k_pad}"
        index_pad[e, :k] = np.asarray(tr.index, dtype=np.int64)
        for gg in range(3):
            bx_proj[e, gg, :k] = bxcoef_g[e, gg] @ tr.matrix
            by_proj[e, gg, :k] = bycoef_g[e, gg] @ tr.matrix
    return _assemble(n_node, index_pad, bx_proj, by_proj, aelem, gauss_w, masks,
                     driving)


def _assemble(n_node, index_pad, bx_proj, by_proj, aelem, gauss_w, masks,
              driving=None):
    """由已填好的 pad 数组组装 P2FastGeometry（K0g、c_g、rows/cols + driving）。

    ``driving`` 给定（生产/引擎路径）时，把 J 组装的**位置**预计算成
    「bulk 位置 + j_mat_driven 位置」的合并结构，掩码预乘进 ``jmd_mask``，
    每轮一次 ``sparsity.build`` 完成 bulk+driving（见模块 docstring 浮点口径）。
    ``driving`` 为 None（纯 fast 层内部测试）时只做 bulk。
    """
    from .sparsity import CSRSparsity

    n_elem = index_pad.shape[0]
    k0g = np.einsum("nqk,nql->nqkl", bx_proj, bx_proj) \
        + np.einsum("nqk,nql->nqkl", by_proj, by_proj)
    c_g = gauss_w[None, :] * aelem[:, None]                     # (N,3)
    rows = np.repeat(index_pad, _K, axis=1).reshape(n_elem, 1, _K, _K)
    rows = np.broadcast_to(rows, (n_elem, 3, _K, _K)).copy()
    cols = np.tile(index_pad, (1, _K)).reshape(n_elem, 1, _K, _K)
    cols = np.broadcast_to(cols, (n_elem, 3, _K, _K)).copy()

    sparsity = None
    jmd_mask = None
    if driving is not None:
        # bulk 有效位置（rows/cols 都 ≥ 0）
        r_flat = rows.reshape(-1)
        c_flat = cols.reshape(-1)
        bulk_valid = (r_flat >= 0) & (c_flat >= 0)
        r_b = r_flat[bulk_valid]
        c_b = c_flat[bulk_valid]
        n_b = r_b.size
        # driving 掩码（对角 0/1）：j_driven[i] * col_mask[j]
        rm = np.asarray(driving.j_driven.diagonal())
        cm = np.asarray(driving.col_mask.diagonal())
        # j_mat_driven 的 (row, col, data)
        jmd = driving.j_mat_driven.tocoo()
        jmd_r = jmd.row
        jmd_c = jmd.col
        jmd_v = jmd.data
        # 合并结构：bulk + jmd
        r_all = np.concatenate([r_b, jmd_r])
        c_all = np.concatenate([c_b, jmd_c])
        sparsity = CSRSparsity(r_all, c_all, n_node)
        # 掩码：bulk 乘 rm[ri]*cm[ci]；jmd 不乘（reference 语义 + 在掩码之后）
        jmd_mask = np.concatenate([
            rm[r_b] * cm[c_b],
            np.ones(jmd_r.size),
        ])
        # 每轮 data_all = concat(bulk_data[bulk_valid], jmd_v)；bulk_data 需按
        # bulk_valid 过滤。为让 build 拿到同长度，存 n_b 供 field_error_p2 切片。
        # jmd 的 data 每轮固定 ⇒ 预拼进「常量部分」，由 field_error 合并。
        geom_obj = P2FastGeometry(
            n_node=n_node, n_elem=n_elem,
            bx_proj=bx_proj, by_proj=by_proj, k0g=k0g, c_g=c_g,
            index_pad=index_pad, rows=rows, cols=cols,
            masks=masks, sparsity=sparsity, jmd_mask=jmd_mask,
        )
        object.__setattr__(geom_obj, "_jmd_v", jmd_v)
        object.__setattr__(geom_obj, "_bulk_valid", bulk_valid)
        object.__setattr__(geom_obj, "_n_bulk", n_b)
        return geom_obj

    return P2FastGeometry(
        n_node=n_node, n_elem=n_elem,
        bx_proj=bx_proj, by_proj=by_proj, k0g=k0g, c_g=c_g,
        index_pad=index_pad, rows=rows, cols=cols,
        masks=masks,
    )

    # K0_g = outer(bx,bx) + outer(by,by)，批量
    k0g = np.einsum("nqk,nql->nqkl", bx_proj, bx_proj) \
        + np.einsum("nqk,nql->nqkl", by_proj, by_proj)

    c_g = field.gauss_w[None, :] * field.aelem[:, None]          # (N,3)

    # J 组装位置：每单元每高斯点 (K,K) 块展开
    rows = np.repeat(index_pad, _K, axis=1).reshape(n_elem, 1, _K, _K)
    rows = np.broadcast_to(rows, (n_elem, 3, _K, _K)).copy()
    cols = np.tile(index_pad, (1, _K)).reshape(n_elem, 1, _K, _K)
    cols = np.broadcast_to(cols, (n_elem, 3, _K, _K)).copy()

    return P2FastGeometry(
        n_node=n_node, n_elem=n_elem,
        bx_proj=bx_proj, by_proj=by_proj, k0g=k0g, c_g=c_g,
        index_pad=index_pad, rows=rows, cols=cols,
        masks=field.masks,
    )


def _material3(Bg, rho, geom: P2FastGeometry, material):
    """逐高斯点 μ/∂γ/∂B：展平 (N,3) 后复用 P1 的 `element_field_material`。"""
    n_elem = geom.n_elem
    flat = Bg.reshape(-1)
    masks = geom.masks
    mu, dgamma = element_field_material(
        flat,
        np.repeat(rho, 3),
        MaterialMasks(
            is_vacuum=np.repeat(masks.is_vacuum, 3),
            is_simp=np.repeat(masks.is_simp, 3),
            is_non_simp=np.repeat(masks.is_non_simp, 3),
            is_linear_iron=np.repeat(masks.is_linear_iron, 3),
        ),
        material,
    )
    return mu.reshape(n_elem, 3), dgamma.reshape(n_elem, 3)


def field_error_p2(
    A: np.ndarray,
    geom: P2FastGeometry,
    rho: np.ndarray,
    material,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, sp_sparse.csr_matrix]:
    """P2 残差 ``F`` + 磁密 ``B`` + μ/∂γ/∂B + Jacobian ``J``（一次算出）。

    Parameters
    ----------
    A : (N_node,)
    geom : P2FastGeometry
    rho : (N_elem,)
    material : FieldMaterial
    R : (N_node,)
    driving : DrivingRows

    Returns
    -------
    (F, B, mu, dgamma, J) ：
        F (N_node,)、B (N_elem, 3)（裁剪后）、mu/dgamma (N_elem, 3)、
        J (N_node, N_node) csr。
    """
    A_active = A[geom.index_pad.clip(min=0)] * (geom.index_pad >= 0)   # (N,K)

    # B（逐高斯点幅值）
    Bx = np.einsum("nqk,nk->nq", geom.bx_proj, A_active)
    By = np.einsum("nqk,nk->nq", geom.by_proj, A_active)
    B = np.sqrt(Bx ** 2 + By ** 2).clip(min=1e-20, max=np.inf)         # (N,3)

    mu, dgamma = _material3(B, rho, geom, material)

    g0 = np.einsum("nqkl,nl->nqk", geom.k0g, A_active)                 # (N,3,K)

    # F = Σ c·γ·g0（每单元每高斯点）
    contrib = (1.0 / mu)[:, :, None] * geom.c_g[:, :, None] * g0        # (N,3,K)
    F = np.zeros(geom.n_node)
    # index 有效位 scatter；bincount 求和顺序与 reference 不同但浮点末位差异 < 1e-14
    idx = geom.index_pad[:, None, :].repeat(3, axis=1).reshape(-1)      # (N*3*K,)
    w = contrib.reshape(-1)
    valid = idx >= 0
    F += np.bincount(idx[valid], weights=w[valid], minlength=geom.n_node)

    F = F - R
    F = ref_bnd.driving_row_residual(F, A, driving)

    # J：线性项 + 饱和项 → 一次预计算结构组装（bulk + driving 掩码 + j_mat_driven）
    lin = (1.0 / mu)[:, :, None, None] * geom.c_g[:, :, None, None] * geom.k0g
    sat = (geom.c_g * dgamma / B)[:, :, None, None] * np.einsum(
        "nqk,nql->nqkl", g0, g0)
    data = (lin + sat).reshape(-1)
    if geom.sparsity is not None:
        # bulk 有效位置 + jmd 常量 data，一起乘掩码后一次 build
        bulk_data = data[geom._bulk_valid]
        all_data = np.concatenate([bulk_data, geom._jmd_v])
        all_data = all_data * geom.jmd_mask
        J = geom.sparsity.build(all_data)
    else:
        # 无 driving（测试路径）：纯 bulk
        r = geom.rows.reshape(-1)
        c = geom.cols.reshape(-1)
        valid = (r >= 0) & (c >= 0)
        J = sp_sparse.coo_matrix(
            (data[valid], (r[valid], c[valid])),
            shape=(geom.n_node, geom.n_node),
        ).tocsr()
    return F, B, mu, dgamma, J


def newton_solve_p2_fast(
    A0: np.ndarray,
    rho: np.ndarray,
    geom: P2FastGeometry,
    material,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
    **kwargs,
) -> Tuple[np.ndarray, FieldAux, sp_sparse.csr_matrix, list]:
    """P2 Newton 求解（fast 层）：复用 P1 的 `newton_iterations` 循环。

    返回值同 `reference.newton_p2.newton_solve_p2`：
    ``(A, FieldAux(B=(N,3), mu, dgamma), Jacobi, history)``。
    """
    def step_fn(A):
        F, B, mu, dgamma, J = field_error_p2(A, geom, rho, material, R, driving)
        return F, FieldAux(B=B, mu=mu, dgamma=dgamma), J

    return newton_iterations(
        step_fn, A0, n_node=geom.n_node, **kwargs
    )
