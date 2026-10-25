"""motorfem.autograd.oracle —— 阶段 6：转矩梯度的 autograd oracle 与验收

入口：::

    auto = autograd_torque_gradient(rho, oracle, M, device=None)      # 端到端（Function）
    auto = adjoint_gradient_from_state(rho, A, B, J, gA, oracle)       # 复用已解状态
    rel = compare_with_hand_written(auto, hand, ...)                   # rel < 1e-9 验收

对 ``T = AᵀMA`` 类的转矩（Arkkio / 虚位移），手写路径是
``dT/dρ = -2·dK_p_dgamma·A·spsolve(Jᵀ, M·A) · dγ/dρ``（位置矩阵实现）；autograd
路径走残差式隐式伴随。两者数学恒等 —— 差异只在实现路径，正是验证点。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sp_sparse_linalg
import torch

from .field import EMFieldGrad, OracleContext
from .residual import residual_lam

__all__ = [
    "autograd_torque_gradient",
    "adjoint_gradient_from_state",
    "adjoint_gradients_from_state_batched",
    "compare_with_hand_written",
]


def sparse_to_torch(M, device=None) -> torch.Tensor:
    """scipy csr → torch COO 稀疏张量（float64）。"""
    M = M.tocoo()
    idx = np.vstack([M.row, M.col])
    return torch.sparse_coo_tensor(
        torch.as_tensor(idx, dtype=torch.int64),
        torch.as_tensor(M.data, dtype=torch.float64),
        M.shape, device=device,
    )


def autograd_torque_gradient(
    rho: np.ndarray,
    oracle: OracleContext,
    M,
    *,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """端到端 autograd：``A = EMFieldGrad(rho)`` → ``T = AᵀMA`` → ``dT/dρ``。

    通过 ``EMFieldGrad.apply`` 走完整隐式伴随（backward 即残差式伴随）。
    ``M``：转矩系数矩阵（Arkkio ``top.torque_arkkio_coef`` 或 VD
    ``pre.Torque_VD_coef_vec[i_t]``）。
    """
    rho_t = torch.tensor(np.asarray(rho, dtype=float), dtype=torch.float64,
                         requires_grad=True, device=device)
    A = EMFieldGrad.apply(rho_t, oracle)
    M_t = sparse_to_torch(M, device)
    T = A @ (M_t @ A)                       # 二次型 AᵀMA（torch 自动算 dT/dA = 2MA）
    (dT,) = torch.autograd.grad(T, rho_t)
    return dT.detach().cpu().numpy()


def adjoint_gradient_from_state(
    rho: np.ndarray,
    A: np.ndarray,
    J,
    gA: np.ndarray,
    oracle: OracleContext,
    *,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """复用已解状态（``(A, B, J)`` + ``gA = dT/dA``）的残差式伴随。

    与 ``EMFieldGrad.backward`` 相同的数学（λ = spsolve(Jᵀ, gA) →
    −∂(λᵀF)/∂ρ），但不重复 Newton 求解 —— 测试对比手写时用这个（一次求解）。
    """
    lam_np = sp_sparse_linalg.spsolve(J.T, np.asarray(gA, dtype=float))
    rho_t = torch.tensor(np.asarray(rho, dtype=float), dtype=torch.float64,
                         requires_grad=True, device=device)
    A_t = torch.as_tensor(np.asarray(A, dtype=float), dtype=torch.float64, device=device)
    lam_t = torch.as_tensor(lam_np, dtype=torch.float64, device=device)
    v = oracle.residual_views(A, device=device)
    F_lam = residual_lam(A_t, rho_t, lam_t, v)
    (grho,) = torch.autograd.grad(-F_lam, rho_t)
    return grho.detach().cpu().numpy()


def adjoint_gradients_from_state_batched(
    rho: np.ndarray,
    A: np.ndarray,
    J,
    G: np.ndarray,
    oracle: OracleContext,
    *,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """b 个伴随向量**一次**求解 + 一次 batched torch 收缩 → ``(b, N_elem)`` 梯度。

    3 转矩 × 11 时间点 = 33 个梯度向量可一批算：

    * ``spsolve(Jᵀ, G)``：G 是 (N_node, b) —— SuperLU 因子只分解一次、解 b 个右端；
    * ``residual_lam`` 带批量维（λ 为 (N_node, b)），torch 收缩一次完成；
    * ``torch.autograd.grad(F_lam, rho, grad_outputs=I_b)`` 给逐列的 VJP（b, N_elem）。

    固定开销（kernel launch / 数据搬移 / 因子分解）被 b 摊薄 —— 这是评估
    「torch 在大批量下是否追上 numpy」的路径。
    """
    G = np.asarray(G, dtype=float)
    if G.ndim != 2:
        raise ValueError(f"G 应为 (N_node, b)，实得 {G.shape}")
    b = G.shape[1]
    lam_np = sp_sparse_linalg.spsolve(J.T, G)          # (N_node,) 或 (N_node, b)
    if lam_np.ndim == 1:
        lam_np = lam_np[:, None]                       # scipy 会压缩单列右端
    rho_t = torch.tensor(np.asarray(rho, dtype=float), dtype=torch.float64,
                         requires_grad=True, device=device)
    A_t = torch.as_tensor(np.asarray(A, dtype=float), dtype=torch.float64, device=device)
    lam_t = torch.as_tensor(lam_np, dtype=torch.float64, device=device)
    v = oracle.residual_views(A, device=device)
    F_lam = residual_lam(A_t, rho_t, lam_t, v)          # (b,)
    # batched VJP：grad_outputs=eye(b) + is_grads_batched ⇒ 逐列的 dF_lam[k]/dρ
    eye = torch.eye(b, dtype=torch.float64, device=device)
    (J_lam,) = torch.autograd.grad(F_lam, rho_t, grad_outputs=eye,
                                   is_grads_batched=True)
    # J_lam (b, N_elem)：每列的 dF_lam[k]/dρ ⇒ 残差式伴随梯度 = −J_lam
    return -J_lam.detach().cpu().numpy()


def compare_with_hand_written(
    auto: np.ndarray,
    hand: np.ndarray,
    *,
    rtol: float = 1e-9,
    min_abs: float = 0.0,
    label: str = "dT/dρ",
) -> tuple:
    """验收比较：autograd vs 手写。

    Parameters
    ----------
    min_abs : float
        **相对噪声下限**：|手写值| < min_abs 的条目不参与 rel（它们量级接近
        ULP/舍入底噪，rel 会虚高）。真实网格建议 ``1e-3``（实测 |手写|≈1e-6 的
        条目 rel 会到 1e-8，而 |手写|≥1e-3 时最大 5e-10）。

    Returns
    -------
    ``(max_rel, max_abs, n_nonzero)`` —— ``max_rel < rtol`` 即通过。
    """
    auto = np.asarray(auto, dtype=float)
    hand = np.asarray(hand, dtype=float)
    if auto.shape != hand.shape:
        raise ValueError(f"{label}: 形状不一致 {auto.shape} vs {hand.shape}")
    nz = np.abs(hand) > min_abs
    if not nz.any():
        raise ValueError(f"{label}: 手写梯度在 |·|>{min_abs} 内全零 —— 无法比较")
    rel = np.abs(auto[nz] - hand[nz]) / np.abs(hand[nz])
    max_rel = float(rel.max())
    max_abs = float(np.abs(auto - hand).max())
    return max_rel, max_abs, int(nz.sum())
