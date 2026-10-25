"""P2（二阶 6 节点三角形）电磁单元矩阵 —— 路线 A（全局高阶单元）。

规划：`docs/重构设计文档.md` §13。P1（3 节点，内核 4）保持逐位不变；
本模块是**新增**的 P2 路径。

------------------------------------------------------------------------------
数学（对照内核 4 的 P1）
------------------------------------------------------------------------------

P2 三角形：6 节点 = 3 顶点 (i, j, m) + 3 边中点（k=ij, l=jm, n=mi），逆时针。
面积坐标 λ₁,λ₂,λ₃（``λ_s = (a_s + b_s x + c_s y)/(2Δ)``，`b`/`c`/`Δ` 同 P1）：

    顶点形函数   N_s  = λ_s(2λ_s − 1)        梯度 ∂N_s/∂x = (4λ_s−1)·b_s/(2Δ)
    边中点形函数 N_k  = 4λ_iλ_j              梯度 ∂N_k/∂x = 4λ_j·b_i/(2Δ) + 4λ_i·b_j/(2Δ)
    （k = ij 之间，l = jm 之间，n = mi 之间）

单元刚度（几何，不含材料 ν）用 **3 点高斯数值积分**（面积坐标
`{1/6,1/6,2/3}` 循环，权重 1/3，对二次精确）：:

    K_e = Σ_g w_g·Δ·(∇N ⊗ ∇N)|_g

压缩存 6×6 对称上三角 **21 个**独立值（顺序按行）。

磁密：B 在单元内**线性**（一阶是常数）。本实现按**单元中心**（λ=1/3）
求值 Bxcoef/Bycoef（形函数梯度在中心，常数 6 值），接口与 P1 兼容
（`Bx = Bxcoef[e] @ A_local6`）。⚠ B 约定沿用 legacy
（`Bx = +b/2Δ` 体系），见内核 4 的 `LEGACY_BX_BY_NOTE` —— 对 Arkkio
的影响（反射与旋转不对易）在 P2 同样成立。

源项：一致载荷 ``R_e = Σ_g w_g·Δ·N(g)``。P2 形函数的面积积分是
**顶点 ∫N_s dV = 0**（顶点形函数有负区域，正负抵消）、**边中点 ∫N_k dV = Δ/3**
——3 点高斯精确给出这一结果。这意味 P2 的一致电流载荷全部落在边中点。

------------------------------------------------------------------------------
单元定向 / 一致性
------------------------------------------------------------------------------

`Δ > 0` 的定向断言复用 `element_geometry`（用顶点 `tri6[:, :3]`）。
`area = Δ`（顶点围成面积），与 P1 相同。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .magnetostatic import ElementGeometry, element_geometry

__all__ = [
    "P2ElectromagneticElements",
    "electromagnetic_elements_p2",
]

#: 3 点高斯（面积坐标），权重 1/3
_GAUSS_LAM = np.array([
    [1.0 / 6, 1.0 / 6, 2.0 / 3],
    [1.0 / 6, 2.0 / 3, 1.0 / 6],
    [2.0 / 3, 1.0 / 6, 1.0 / 6],
])
_GAUSS_W = 1.0 / 3


@dataclass(frozen=True)
class P2ElectromagneticElements:
    """P2 单元级预计算量。全部只依赖网格，与 A、材料无关。

    Attributes
    ----------
    K0 : (N, 21)
        压缩存储的 6×6 对称单元几何刚度（上三角按行）。
        真实刚度 = ``ν_e · K0[e]``（展开后）。
    R0 : (N, 6)
        一致载荷节点系数（`B_e = J_e · R0[e]`）。顶点 = 0、边中点 = Δ/3。
    Bxcoef, Bycoef : (N, 6)
        **单元中心**（λ=1/3）的磁密系数 ``Bx = Bxcoef[e] @ A_local6``。
        legacy 约定。
    Bxcoef_g, Bycoef_g : (N, 3, 6)
        **逐高斯点**的磁密系数（3 点规则，`_GAUSS_LAM`）—— 非线性材料在积分点
        求值用（`newton_p2` 的 `field_system_p2`），P1 的单元中心值是它的单点特例。
    gauss_w : (3,)
        高斯权重（本规则均匀 = 1/3）。
    area : (N,)
        单元面积 Δ。
    tri6 : (N, 6)
        P2 连接表（顶点 + 边中点），供装配/下游使用。
    geom : ElementGeometry
        底层几何量（顶点 `b`/`c`/`delta`/`tri`）。
    """

    K0: np.ndarray
    R0: np.ndarray
    Bxcoef: np.ndarray
    Bycoef: np.ndarray
    Bxcoef_g: np.ndarray
    Bycoef_g: np.ndarray
    gauss_w: np.ndarray
    area: np.ndarray
    tri6: np.ndarray
    geom: ElementGeometry

    @property
    def n_elem(self) -> int:
        return int(self.area.shape[0])


def _gradients(b, c, D, lam1, lam2, lam3):
    """6 节点形函数**真梯度**在面积坐标 (λ1,λ2,λ3) 处（∂N/∂x, ∂N/∂y）。

    ⚠ **2026-10-02 修复（P2 转矩偏低根因）**：早期实现把 legacy B 约定
    （`Bycoef = −c/(2Δ)`）的负号塞进了梯度（y 分量用 `−c`），导致：
    - 刚度 `K0 = Σ w·Δ·(∇N⊗∇N)` 的 y 分量反号 —— 线性场（常数 B）能量是
      平方量不受影响（抽查假阴性），但**二次场能量错 0.5 倍**（实测）；
    - B 系数（`Bycoef`）因此正确（恰好匹配 legacy）—— 修复后由调用方显式
      取 `−g[:,1]` 恢复 legacy 约定。
    真梯度：`∂λ_i/∂x = b_i/(2Δ)`、`∂λ_i/∂y = c_i/(2Δ)`（`element_geometry`
    的定义 `c_i = −x_j + x_m`，`∂λ/∂y = c/(2Δ)` 为正）。
    """
    bi, bj, bm = b
    ci, cj, cm = c
    g = np.empty((6, 2))
    # 顶点（N_s = λ_s(2λ_s−1)，∂/∂x = (4λ_s−1)·b_s/(2Δ)，∂/∂y = (4λ_s−1)·c_s/(2Δ)）
    g[0] = (4 * lam1 - 1) * np.array([bi, ci]) / (2 * D)
    g[1] = (4 * lam2 - 1) * np.array([bj, cj]) / (2 * D)
    g[2] = (4 * lam3 - 1) * np.array([bm, cm]) / (2 * D)
    # 边中点（N_k = 4λ_iλ_j）
    g[3] = 4 * (lam2 * np.array([bi, ci]) + lam1 * np.array([bj, cj])) / (2 * D)
    g[4] = 4 * (lam3 * np.array([bj, cj]) + lam2 * np.array([bm, cm])) / (2 * D)
    g[5] = 4 * (lam1 * np.array([bm, cm]) + lam3 * np.array([bi, ci])) / (2 * D)
    return g


def _shape(lam1, lam2, lam3):
    """6 节点形函数值在面积坐标处（用于一致载荷积分）。"""
    return np.array([
        lam1 * (2 * lam1 - 1), lam2 * (2 * lam2 - 1), lam3 * (2 * lam3 - 1),
        4 * lam1 * lam2, 4 * lam2 * lam3, 4 * lam3 * lam1,
    ])


def electromagnetic_elements_p2(
    node: np.ndarray, tri6: np.ndarray
) -> P2ElectromagneticElements:
    """P2（6 节点）电磁单元预计算量。

    Parameters
    ----------
    node : (N_node, 2) 节点坐标
    tri6 : (N_elem, 6) 连接表：**顶点 (i,j,m) 在前，边中点 (k=ij, l=jm, n=mi) 在后**，
        顶点逆时针（定向断言用顶点）。

    Returns
    -------
    P2ElectromagneticElements

    Notes
    -----
    刚度用 3 点高斯精确积分（对二次）。磁密系数取**单元中心**（λ=1/3），
    接口与 P1 兼容（`Bxcoef` 行向量点积局部 A）。
    """
    tri6 = np.asarray(tri6)
    geom = element_geometry(node, tri6[:, :3])     # 顶点定向断言
    b, c, delta = geom.b, geom.c, geom.delta
    n_elem = tri6.shape[0]

    K0 = np.zeros((n_elem, 21))
    R0 = np.zeros((n_elem, 6))
    Bxcoef = np.zeros((n_elem, 6))
    Bycoef = np.zeros((n_elem, 6))
    Bxcoef_g = np.zeros((n_elem, 3, 6))
    Bycoef_g = np.zeros((n_elem, 3, 6))
    area = np.zeros(n_elem)

    for e in range(n_elem):
        D = delta[e]
        # 刚度：Σ_g w·Δ·(∇N ⊗ ∇N)（真梯度，2026-10-02 修复）；同时记录逐高斯点
        # B 系数（legacy 约定：Bxcoef = +∂N/∂x、Bycoef = −∂N/∂y）
        K = np.zeros((6, 6))
        for gg in range(3):
            g_ = _gradients(b[e], c[e], D, *_GAUSS_LAM[gg])
            Bxcoef_g[e, gg] = g_[:, 0]
            Bycoef_g[e, gg] = -g_[:, 1]
            K += _GAUSS_W * D * (
                np.outer(g_[:, 0], g_[:, 0]) + np.outer(g_[:, 1], g_[:, 1])
            )
        # 压缩 21 值（上三角按行）
        idx = 0
        for a in range(6):
            for b_ in range(a, 6):
                K0[e, idx] = K[a, b_]
                idx += 1

        # 一致载荷：Σ_g w·Δ·N(g)
        R0e = np.zeros(6)
        for gg in range(3):
            R0e += _GAUSS_W * D * _shape(*_GAUSS_LAM[gg])
        R0[e] = R0e

        # 中心磁密系数（λ=1/3，legacy 约定：Bycoef = −∂N/∂y）
        gc = _gradients(b[e], c[e], D, 1 / 3, 1 / 3, 1 / 3)
        Bxcoef[e] = gc[:, 0]
        Bycoef[e] = -gc[:, 1]

        area[e] = D

    return P2ElectromagneticElements(
        K0=K0, R0=R0, Bxcoef=Bxcoef, Bycoef=Bycoef,
        Bxcoef_g=Bxcoef_g, Bycoef_g=Bycoef_g,
        gauss_w=np.full(3, _GAUSS_W),
        area=area, tri6=tri6, geom=geom,
    )
