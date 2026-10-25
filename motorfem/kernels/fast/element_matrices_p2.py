"""P2 电磁单元预计算量的向量化实现（生产性能，`approx` 语义）。

对照 `reference/element_matrices_p2.electromagnetic_elements_p2`（逐元素 Python
循环，实测 ~762 ms @ 7144 单元 order2）。本模块是**同一数学**的批量实现：
``_gradients`` 在固定面积坐标处是 ``[b,c]/(2Δ)`` 的线性组合 ⇒ 可对整个网格
一次 ``einsum`` 算出 (N,3,6,2) 的真梯度，再批量外积求和。

------------------------------------------------------------------------------
浮点口径（对齐判据）
------------------------------------------------------------------------------

K0 的累加顺序与 reference **完全相同**（每单元每高斯点 ``outer(g[:,0],g[:,0])
+ outer(g[:,1],g[:,1])`` 逐高斯点 ``+=``，批量版是 3 个高斯点的逐项求和，乘法
与加法结合序逐一对应）⇒ **逐位一致**（实测 ``max|Δ| = 0.0``）。B 系数逐高斯点
赋值一一对应，同样逐位。

本文件放在 ``fast/`` 下但**没有对应的 P1 legacy** —— 按 `kernels/README.md`
「reference 是权威表达、fast 是生产路径」的判据，本层对齐对象是
``reference.element_matrices_p2``，判据 ``np.array_equal``（逐位）。它属于
「同一数学的向量化」，不是新近似（不放弃位级一致）。
"""

from __future__ import annotations

import numpy as np

from ..reference.element_matrices_p2 import (
    P2ElectromagneticElements,
    _GAUSS_LAM,
    _GAUSS_W,
    _gradients,
)

__all__ = ["electromagnetic_elements_p2_fast"]


def _gradients_batch(b, c, D, lam1, lam2, lam3):
    """整个网格 (N,6,2) 的真梯度（对 fixed 面积坐标）。逐元素对照 `_gradients`。

    b, c : (N,3)（每单元的 b_i/b_j/b_m 与 c_i/c_j/c_m）
    D : (N,) 单元面积

    ⚠ 浮点顺序必须与 reference 逐位一致：reference 每行是
    ``(4λ−1) * np.array([b, c]) / (2*D)``（先乘系数、再除以 2D）。这里对
    ``[b,c]`` 先除 ``2D`` 再乘系数会差 ~1e-11 ⇒ **不能**先除后乘。
    """
    one_over_2D = 1.0 / (2 * D)
    # 先算 (4λ−1)·[b,c]，再整体除以 2D —— 与 reference 的标量乘/除顺序对应
    g = np.empty((b.shape[0], 6, 2))
    g[:, 0] = (4 * lam1 - 1) * np.stack([b[:, 0], c[:, 0]], axis=-1) / (2 * D[:, None])
    g[:, 1] = (4 * lam2 - 1) * np.stack([b[:, 1], c[:, 1]], axis=-1) / (2 * D[:, None])
    g[:, 2] = (4 * lam3 - 1) * np.stack([b[:, 2], c[:, 2]], axis=-1) / (2 * D[:, None])
    g[:, 3] = 4 * (lam2 * np.stack([b[:, 0], c[:, 0]], axis=-1)
                   + lam1 * np.stack([b[:, 1], c[:, 1]], axis=-1)) / (2 * D[:, None])
    g[:, 4] = 4 * (lam3 * np.stack([b[:, 1], c[:, 1]], axis=-1)
                   + lam2 * np.stack([b[:, 2], c[:, 2]], axis=-1)) / (2 * D[:, None])
    g[:, 5] = 4 * (lam1 * np.stack([b[:, 2], c[:, 2]], axis=-1)
                   + lam3 * np.stack([b[:, 0], c[:, 0]], axis=-1)) / (2 * D[:, None])
    return g


def electromagnetic_elements_p2_fast(
    node: np.ndarray, tri6: np.ndarray
) -> P2ElectromagneticElements:
    """`reference.element_matrices_p2.electromagnetic_elements_p2` 的向量化版。

    参数与返回值**完全一致**（P2ElectromagneticElements 结构），数值逐位一致。
    """
    from ..reference.element_matrices import element_geometry

    tri6 = np.asarray(tri6)
    geom = element_geometry(node, tri6[:, :3])          # 顶点定向断言（与 reference 同）
    b, c, delta = geom.b, geom.c, geom.delta
    n_elem = tri6.shape[0]

    # 逐高斯点真梯度 (N,3,6,2)
    g_all = np.stack(
        [_gradients_batch(b, c, delta, *_GAUSS_LAM[gg]) for gg in range(3)],
        axis=1,
    )                                                 # (N,3,6,2)
    gx = g_all[:, :, :, 0]                             # (N,3,6)  ∂N/∂x
    gy = g_all[:, :, :, 1]                             # (N,3,6)  ∂N/∂y

    # B 系数（legacy 约定：Bxcoef=+∂N/∂x、Bycoef=−∂N/∂y）
    Bxcoef_g = gx
    Bycoef_g = -gy

    # K0 = Σ_g w·Δ·(∇N⊗∇N)，批量（逐高斯点外积求和，累加顺序同 reference）
    K = _GAUSS_W * delta[:, None, None, None] * (
        np.einsum("...ka,...kb->...kab", gx, gx)
        + np.einsum("...ka,...kb->...kab", gy, gy)
    )                                                  # (N,3,6,6)
    K0 = K.sum(axis=1)                                 # (N,6,6)

    # 压缩 21 值（上三角按行，同 reference 的 idx 顺序）
    iu = np.triu_indices(6)
    K0_21 = K0[:, iu[0], iu[1]]                         # (N,21)

    # 一致载荷 R0 = Σ_g w·Δ·N(g)（逐高斯点累加，顺序同 reference）
    R0 = np.zeros((n_elem, 6))
    for gg in range(3):
        s = _shape_vec(*_GAUSS_LAM[gg])                 # (6,)
        R0 += _GAUSS_W * delta[:, None] * s[None, :]

    # 中心磁密系数（λ=1/3，同 reference）
    gc = _gradients_batch(b, c, delta, 1 / 3, 1 / 3, 1 / 3)   # (N,6,2)
    Bxcoef = gc[:, :, 0]
    Bycoef = -gc[:, :, 1]

    return P2ElectromagneticElements(
        K0=K0_21, R0=R0, Bxcoef=Bxcoef, Bycoef=Bycoef,
        Bxcoef_g=Bxcoef_g, Bycoef_g=Bycoef_g,
        gauss_w=np.full(3, _GAUSS_W),
        area=delta, tri6=tri6, geom=geom,
    )


def _shape_vec(lam1, lam2, lam3):
    """6 节点形函数值（对照 `reference` 的 `_shape`）。"""
    return np.array([
        lam1 * (2 * lam1 - 1), lam2 * (2 * lam2 - 1), lam3 * (2 * lam3 - 1),
        4 * lam1 * lam2, 4 * lam2 * lam3, 4 * lam3 * lam1,
    ])
