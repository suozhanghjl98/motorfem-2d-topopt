"""P2 求解（路线 A）：`P2FieldGeometry` + `field_geometry_p2` + **积分点求值**。

规划：`docs/重构设计文档.md` §13。结构上复用 P1 的牛顿循环
（`newton.newton_iterations`），但**残差/雅可比不能直接复用 P1 的
`field_system`** —— 它的饱和项 `1/B/Δ·∂γ/∂B·(K0A)(K0A)ᵀ` 依赖 P1 的恒等式
``∂B/∂A = K0A/(BΔ)``（只有常数 B 才成立）。P2 的 B 在单元内线性变化，沿用
该式会得到**不一致的雅可比** ⇒ 牛顿发散（真实电机网格实测：‖F‖² 冲到 1e9）。

本模块的 P2 求解把**材料 μ/∂γ/∂B 在积分点求值**（§13.3「newton」行的设计）：
每个高斯点贡献自己的 γ(B_g)、饱和项对称：

.. code-block:: text

    F_e = Σ_g c_g γ_g K0_g A_e            c_g = w_g·Δ
    J_e = Σ_g [ c_g γ_g K0_g + c_g (∂γ_g/∂B_g) g0_g g0_gᵀ / B_g ]
    其中 K0_g = b_g b_gᵀ + c_g c_gᵀ（高斯点梯度）、g0_g = K0_g A_e

这是「P1 单元中心值」在积分点上的直接推广（P1 = 1 个点）；J 对称
（``outer(g0,g0)`` 对称），与 P1 的对称雅可比机制、以及下游的
Pardiso mtype=2 / 消元路径兼容。磁密系数（``Bxcoef_g``/``Bycoef_g``）来自
`element_matrices_p2`（3 点高斯规则 `_GAUSS_LAM`）。

`flux_density_p2` 返回 (N_elem, 3)（逐高斯点），不是 P1 的 (N_elem,) ——
`FieldAux` 字段形状不同，消费方（伴随解、扭矩）必须按 P2 口径处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import sparse as sp_sparse

from . import boundaries as ref_bnd
from .assembly_p2 import P2ElementTransform, transformation_matrix_p2
from .newton import (
    FieldAux,
    MaterialMasks,
    element_field_material,
    newton_iterations,
)

__all__ = [
    "P2FieldGeometry",
    "field_geometry_p2",
    "flux_density_p2",
    "field_system_p2",
    "residual_p2",
    "jacobian_p2",
    "newton_solve_p2",
]


@dataclass(frozen=True)
class P2FieldGeometry:
    """P2 求解输入束（对照 `newton.FieldGeometry`，6 节点）。

    Attributes
    ----------
    n_node : int
        P2 节点数。
    k0 : (N_elem, 6, 6) ndarray
        P2 几何单元刚度（不含 ν），由 21 值展开。
    bxcoef, bycoef : (N_elem, 6) ndarray
        P2 磁密系数（**未经 T**），单元**中心**值（legacy 约定，供单点求值）。
    bxcoef_g, bycoef_g : (N_elem, 3, 6) ndarray
        逐高斯点磁密系数（**未经 T**）—— 非线性材料积分点求值的主口径。
    gauss_w : (3,) ndarray
        高斯权重（本规则均匀 1/3）。
    aelem : (N_elem,) 单元面积。
    masks : MaterialMasks
    transforms : tuple[P2ElementTransform, ...]
    """

    n_node: int
    k0: np.ndarray
    bxcoef: np.ndarray
    bycoef: np.ndarray
    bxcoef_g: np.ndarray
    bycoef_g: np.ndarray
    gauss_w: np.ndarray
    aelem: np.ndarray
    masks: MaterialMasks
    transforms: Tuple[P2ElementTransform, ...]

    @property
    def n_elem(self) -> int:
        return len(self.transforms)


def _expand21_to_matrix(k0e):
    K = np.zeros((6, 6)); idx = 0
    for a in range(6):
        for b in range(a, 6):
            K[a, b] = K[b, a] = k0e[idx]; idx += 1
    return K


def field_geometry_p2(
    ctx,
    rotating,
    *,
    n_node: int,
    k0elem: np.ndarray,
    bxcoef: np.ndarray,
    bycoef: np.ndarray,
    bxcoef_g: np.ndarray,
    bycoef_g: np.ndarray,
    gauss_w: np.ndarray,
    aelem: np.ndarray,
    masks: MaterialMasks,
) -> P2FieldGeometry:
    """构造 `P2FieldGeometry`：逐单元取一次 P2 `T`，之后反复复用。

    Parameters 与 `newton.field_geometry` 同，只是 ``k0elem`` 是 P2 的 21 值、
    ``bxcoef/bycoef`` 是 (N, 6)；``bxcoef_g/bycoef_g`` 是 (N, 3, 6) 的逐高斯点
    磁密系数、``gauss_w`` 是 (3,) 权重（来自 `element_matrices_p2`）。
    """
    ctx = ctx
    n_elem = np.asarray(ctx.tri).shape[0]
    transforms = tuple(
        transformation_matrix_p2(i, ctx, rotating) for i in range(n_elem)
    )
    k0 = np.asarray(
        [_expand21_to_matrix(k0elem[i]) for i in range(n_elem)], dtype=float
    )
    return P2FieldGeometry(
        n_node=n_node,
        k0=k0,
        bxcoef=np.asarray(bxcoef, dtype=float),
        bycoef=np.asarray(bycoef, dtype=float),
        bxcoef_g=np.asarray(bxcoef_g, dtype=float),
        bycoef_g=np.asarray(bycoef_g, dtype=float),
        gauss_w=np.asarray(gauss_w, dtype=float),
        aelem=np.asarray(aelem, dtype=float),
        masks=masks,
        transforms=transforms,
    )


# =============================================================================
# 积分点求值的残差 / 雅可比 / 牛顿
# =============================================================================

def flux_density_p2(A: np.ndarray, field: P2FieldGeometry) -> np.ndarray:
    """逐高斯点磁密幅值 ``|B|``（裁剪后），形状 ``(N_elem, 3)``。

    每个高斯点的 ``B_g = sqrt((b_g·A)² + (c_g·A)²)``，``b_g/c_g`` 是
    `field.bxcoef_g/field.bycoef_g` 经 `T` 投影到活跃自由度的行。
    ``clip(min=1e-20)`` 与 P1 同（防除零）。
    """
    B = np.empty((field.n_elem, 3))
    for e in range(field.n_elem):
        tr = field.transforms[e]
        A_active = A[list(tr.index)]
        for gg in range(3):
            Bx = field.bxcoef_g[e, gg] @ tr.matrix @ A_active
            By = field.bycoef_g[e, gg] @ tr.matrix @ A_active
            B[e, gg] = np.sqrt(Bx ** 2 + By ** 2)
    return np.clip(B, 1e-20, np.inf)


def _material_gauss(Bg, rho, masks, material):
    """逐高斯点的 μ/∂γ/∂B：把 (N,3) 的 B 展平、ρ 与掩码各重复 3 倍，
    复用 P1 的 `element_field_material`（元素级函数，展平后逐点成立）。"""
    n_elem = Bg.shape[0]
    flat = Bg.reshape(-1)
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


def field_system_p2(
    A: np.ndarray,
    field: P2FieldGeometry,
    mu: np.ndarray,
    dgamma: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
) -> Tuple[np.ndarray, sp_sparse.csr_matrix]:
    """P2 残差 + Jacobian（已做驱动行替换），材料**逐高斯点**求值。

    Parameters
    ----------
    mu, dgamma, B : (N_elem, 3) ndarray
        由 `_material_gauss` 产出（逐高斯点）。``mu`` 是磁导率（γ = 1/μ）。
    R : (N_node,) 源项。
    driving : DrivingRows

    Returns
    -------
    (F, J) ：F (N_node,)，J (N_node, N_node) csr。J 对称（不含驱动行部分）。

    Notes
    -----
    每个高斯点贡献（模块 docstring 的记号）：

    .. code-block:: text

        K0_g = b_g b_gᵀ + c_g c_gᵀ          （活跃空间，b_g = bxcoef_g[e,gg] @ T）
        g0_g = K0_g A_e                     （几何向量，不含权重）
        F_e  += c_g γ_g g0_g
        J_e  += c_g γ_g K0_g + c_g (∂γ_g/∂B_g) g0_g g0_gᵀ / B_g
    """
    n_node, n_elem = field.n_node, field.n_elem
    rows, cols, vals = [], [], []
    F = np.zeros(n_node)

    for e in range(n_elem):
        tr = field.transforms[e]
        index = list(tr.index)
        T = tr.matrix
        A_active = A[index]

        for gg in range(3):
            bx = field.bxcoef_g[e, gg] @ T
            by = field.bycoef_g[e, gg] @ T
            c = field.gauss_w[gg] * field.aelem[e]
            K0g = np.outer(bx, bx) + np.outer(by, by)
            g0 = K0g @ A_active

            F[index] += (1.0 / mu[e, gg]) * (c * g0)
            sat = c * dgamma[e, gg] * np.outer(g0, g0) / B[e, gg]
            block = (1.0 / mu[e, gg]) * (c * K0g) + sat

            rows.extend(np.repeat(index, len(index)))
            cols.extend(np.tile(index, len(index)))
            vals.extend(block.ravel())

    F = F - R
    F = ref_bnd.driving_row_residual(F, A, driving)
    J = ref_bnd.driving_row_jacobian(
        sp_sparse.coo_matrix(
            (np.asarray(vals, dtype=float), (np.asarray(rows, dtype=np.int64),
                                             np.asarray(cols, dtype=np.int64))),
            shape=(n_node, n_node),
        ),
        driving,
    )
    return F, J.tocsr()


def residual_p2(
    A: np.ndarray,
    rho: np.ndarray,
    field: P2FieldGeometry,
    material,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
) -> Tuple[np.ndarray, FieldAux]:
    """P2 残差 ``F(A) = K(A)A − R``（积分点材料）。``aux.B/mu/dgamma`` 是 (N,3)。"""
    B = flux_density_p2(A, field)
    mu, dgamma = _material_gauss(B, rho, field.masks, material)
    F, _ = field_system_p2(A, field, mu, dgamma, B, R, driving)
    return F, FieldAux(B=B, mu=mu, dgamma=dgamma)


def jacobian_p2(
    A: np.ndarray,
    field: P2FieldGeometry,
    aux: FieldAux,
    driving: ref_bnd.DrivingRows,
) -> sp_sparse.csr_matrix:
    """P2 Jacobian（积分点材料，对称）。``aux`` 由 `residual_p2` 返回。"""
    _, J = field_system_p2(
        A, field, aux.mu, aux.dgamma, aux.B, np.zeros_like(A), driving
    )
    return J


def newton_solve_p2(
    A0: np.ndarray,
    rho: np.ndarray,
    field: P2FieldGeometry,
    material,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
    **kwargs,
) -> Tuple[np.ndarray, FieldAux, sp_sparse.csr_matrix, list]:
    """P2 Newton 求解：绑定 `residual_p2`/`jacobian_p2`，复用 P1 的
    `newton_iterations` 循环（消元/松弛/判停全同）。"""
    def step_fn(A):
        F, aux = residual_p2(A, rho, field, material, R, driving)
        return F, aux, jacobian_p2(A, field, aux, driving)

    return newton_iterations(
        step_fn, A0, n_node=field.n_node, **kwargs
    )
