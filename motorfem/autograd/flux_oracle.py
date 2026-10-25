"""motorfem.autograd.flux_oracle —— 阶段 8 收尾 #1：磁链转矩的全导数 oracle

手写 ``flux_torque_derivative``（``fast/flux.py``）把**冻结求解当常数**（冻结
近似，不微分 ``K_frozen(ρ)·A = R``）。本模块给出穿过冻结系统的**真全导数**
oracle，判定 D4 的冻结约定是否与全导数一致（设计文档 §阶段 6 偏差登记①）。

------------------------------------------------------------------------------
数学
------------------------------------------------------------------------------

冻结链（``reference/frozen_flux_analysis``）::

    A_d = K_frozen(ρ)⁻¹ R_d,   A_q = K_frozen(ρ)⁻¹ R_q        （线性，一步精确）
    T_flux = Σ_e k_e/μ_e
    k_e    = p·[I_d·I_q·(Bx_d²+By_d²−Bx_q²−By_q²)
                + (I_q²−I_d²)·(Bx_d·Bx_q + By_d·By_q)]·Δ·parts

``μ_e`` 冻结（经 ``blend_class_values`` 的四掩码加权，ρ 经 ``is_SIMP`` 分支进入）。
``Bx_d = Bxcoef·A_d``（Bxcoef 与 ρ 无关）。

``T_flux`` 对 ``ρ`` 的依赖有**两条路**，都要可微:

1. **``A_d``/``A_q`` 的隐式依赖**（经 ``K_frozen(ρ)``）—— 用残差式隐式伴随:
      ``dA/dρ`` 项 = ``−λᵀ·(∂K_frozen/∂ρ)·A``，其中 ``J_frozenᵀλ = gA``
2. **``μ`` 的显式依赖**（``1/μ(ρ)``）—— ``torch_nu(B_frozen, ρ, …)`` 恰好
   重建冻结 μ 的倒数（``mu_simp = B/H(B/ρ₁)`` 分支 + 四掩码加权，见
   ``autograd/material.py``）

端到端 torch 图::

    A_d = FrozenSolveGrad.apply(rho, ctx)      # forward = numpy 冻结解
    A_q = FrozenSolveGrad.apply(rho, ctx)      #           backward = 隐式伴随
    Bx_d = Bxcoef·A_d …（torch，自动可微）
    k_e = flux_torque_kernel_torch(Bx_d…, …)   # 纯 torch 代数
    nu  = torch_nu(B_frozen, rho, …)           # 冻结 μ 的倒数，ρ 可微
    T   = Σ k_e·ν_e
    dT/dρ = torch.autograd.grad(T, rho)

------------------------------------------------------------------------------
与手写的差异（D4 判定）
------------------------------------------------------------------------------

手写 ``flux_torque_derivative`` 把 ``A_d``/``A_q`` 当**常数**，只取
``∂T/∂ρ|_A``（显式 ``_k`` 项）＋ 主场伴随（``∂T/∂A`` 经 ``J⁻ᵀ`` 的项）。
本 oracle 额外让梯度穿过 ``A_d``/``A_q`` 对 ``ρ`` 的隐式依赖。两者若不一致，
说明冻结近似漏项 ⇒ 与 D4（符号相反、被消费者抵消）一起构成对「磁链转矩梯度
语义」的完整判定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse.linalg as sp_sparse_linalg
import torch

from .material import TorchSIMP, torch_nu
from .pchip import TorchPCHIP
from .residual import ResidualContext, residual_lam_frozen

__all__ = [
    "FluxFrozenContext",
    "FrozenSolveGrad",
    "flux_torque_gradient",
    "flux_torque_full_derivative",
]


def _field_components(A, Bxcoef, Bycoef, tri):
    """逐单元磁密分量 ``(Bx, By) = (Bxcoef·A_e, Bycoef·A_e)``（torch）。

    ``Bxcoef``/``Bycoef`` 是 ``(N_elem, 3)`` 逐单元系数；``A`` 是 ``(N_node,)``。
    ``tri`` 是逐单元的全局节点表（``(N_elem, 3)``，**原始** tri —— 从属节点用
    自己的值，与 ``reference`` 的 ``flux_density_components`` 同构）。
    """
    Ae = A[tri]                                # (N_elem, 3)
    return (Bxcoef * Ae).sum(-1), (Bycoef * Ae).sum(-1)


def _flux_torque_kernel_torch(
    Bx_d, By_d, Bx_q, By_q,
    I_d: float,
    I_q: float,
    area,
    *,
    pole_pairs: int,
    parts: int,
):
    """``flux_torque_kernel``（``reference/flux.py:534``）的 torch 版。

    纯逐元素代数（乘加），与 numpy 版同一算式、同一结合序。
    """
    return (
        pole_pairs * (
            I_d * I_q * (Bx_d ** 2 + By_d ** 2 - Bx_q ** 2 - By_q ** 2)
            + (I_q ** 2 - I_d ** 2) * (Bx_d * Bx_q + By_d * By_q)
        ) * torch.as_tensor(area, dtype=torch.float64, device=Bx_d.device) * parts
    )


@dataclass
class FluxFrozenContext:
    """冻结链的全导数 oracle 工作点：numpy 求解 + torch 收缩共用同一批输入。

    Parameters
    ----------
    top : FieldTopology
        ``top.em``（单元矩阵）、``top.layer``（tri/pairing）、
        ``top.element_rotor_gap``。
    kv : 22 元组（``pre.K_vectorized_vec[i_t]``）
    B_frozen : (N_elem,) 冻结磁密幅值（内核 11 的返回值，**已裁剪**）
    material : FieldMaterial
    R_d, R_q : (N_node,) 单位 d/q 电流源项
    I_d, I_q : float 该工作点的 d/q 电流
    pole_pairs, parts : int
    """

    top: object
    kv: tuple
    B_frozen: np.ndarray
    material: object
    R_d: np.ndarray
    R_q: np.ndarray
    I_d: float
    I_q: float
    pole_pairs: int
    parts: int

    # ------------------------------------------------------------ 求解
    def solve_frozen(self, rho: np.ndarray) -> tuple:
        """两次冻结线性求解（fast 层组合，numpy 一行不改）。

        Returns ``(A_d, A_q, J_frozen, mu)`` —— ``J_frozen`` 是冻结系统的
        Jacobian（与 ``A`` 无关、只依赖 ``ρ``；d/q 两支逐位相同 ⇒ 构造一次，
        两次回代 —— 阶段 8 的 8a 分解复用在这里天然适用）。
        """
        from ..kernels.fast import flux as fast_flux
        from ..kernels.fast.material import CachedMaterial

        cached = CachedMaterial(
            self.material.curve, self.material.simp,
            self.material.mu0, self.material.mur_linear,
        )
        state = cached.prepare_density(np.asarray(rho, dtype=float))
        A0 = np.zeros(self.top.n_node)

        def _solve(R):
            # A=0 处 dν/dB 项恒为零 ⇒ F 是 R 的线性函数、J 与 R 无关（内核 15 测试钉住）
            F, J, mu = fast_flux.field_error_vectorized_frozen(
                A0, self.B_frozen, self.kv, R,
                n_node=self.top.n_node, area=self.top.em.area,
                cached=cached, state=state,
            )
            J_csr = J.tocsr()
            A = A0 - sp_sparse_linalg.spsolve(J_csr, F)
            return A, J_csr, mu

        A_d, J_frozen, _ = _solve(np.asarray(self.R_d, dtype=float))
        A_q, _, mu = _solve(np.asarray(self.R_q, dtype=float))
        return A_d, A_q, J_frozen, mu

    # ------------------------------------------------------------ 残差视图
    def residual_views(self, A: np.ndarray, device=None):
        """冻结残差收缩的 torch 视图（B_frozen 显式传入）。

        与 ``OracleContext.residual_views`` 同构，但 ``ResidualContext.build``
        的 ``A`` 用**解出的冻结解**（``F_mat_driven_A = F_mat_driven @ A``
        依赖具体 A —— A_d 与 A_q 各自不同，调用方传各自的）。
        """
        rctx = ResidualContext.build(
            self.top, self.kv, self.R_d, self.material, A,
            curve=self._torch_curve(), simp=self._torch_simp(),
        )
        return rctx.to_torch(device)

    def _torch_curve(self) -> TorchPCHIP:
        return TorchPCHIP(self.material.curve._H_of_B.c,
                          self.material.curve._H_of_B.x)

    def _torch_simp(self) -> TorchSIMP:
        return TorchSIMP(self.material.simp)


class FrozenSolveGrad(torch.autograd.Function):
    """``A = solve_frozen(ρ)`` 的隐式伴随 Function（阶段 8 收尾 #1）。

    forward：现有 numpy 冻结求解（一行不改）；backward：冻结系统的残差式
    隐式伴随 —— ``λ = spsolve(J_frozenᵀ, gA)``，
    ``dL/dρ = −∂(λᵀF_frozen)/∂ρ``（``residual_lam_frozen``，B 用冻结值）。

    ⚠ 只返回 ``A_d``；``A_q`` 是另一次 ``apply``。``ctx_frozen`` 非张量。
    """

    @staticmethod
    def forward(ctx, rho, ctx_frozen: FluxFrozenContext):
        rho_np = np.asarray(rho.detach().cpu().numpy(), dtype=float)
        A_d, A_q, J_frozen, mu = ctx_frozen.solve_frozen(rho_np)
        ctx.ctx_frozen = ctx_frozen
        ctx.J_frozen = J_frozen
        ctx.mu = mu
        ctx.save_for_backward(
            rho.detach().to(torch.float64),
            torch.as_tensor(A_d, dtype=torch.float64),
            torch.as_tensor(ctx_frozen.B_frozen, dtype=torch.float64),
        )
        return torch.as_tensor(A_d, dtype=torch.float64, device=rho.device)

    @staticmethod
    def backward(ctx, gA_d):
        rho, A_d, B_frozen = ctx.saved_tensors
        ctx_frozen: FluxFrozenContext = ctx.ctx_frozen
        gA_d_np = gA_d.detach().cpu().numpy()
        lam_np = sp_sparse_linalg.spsolve(ctx.J_frozen.T, gA_d_np)
        with torch.enable_grad():
            rho_t = rho.detach().clone().requires_grad_(True).to(torch.float64)
            v = ctx_frozen.residual_views(A_d.detach().cpu().numpy(),
                                          device=rho.device)
            lam_t = torch.as_tensor(lam_np, dtype=torch.float64, device=rho.device)
            F_lam = residual_lam_frozen(
                A_d.detach().to(torch.float64), rho_t, lam_t, v, B_frozen)
            (grho,) = torch.autograd.grad(-F_lam, rho_t)
        return grho.to(rho.dtype), None


def flux_torque_gradient(
    rho: np.ndarray,
    ctx_frozen: FluxFrozenContext,
    *,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """端到端全导数：穿过冻结系统（A_d/A_q 隐式依赖）＋ μ 显式依赖。

    数学与 ``flux_torque_full_derivative`` 相同；本函数自行解冻结场（forward），
    再委托给 ``flux_torque_full_derivative`` 计算梯度 —— 消除两套逻辑漂移。
    """
    A_d, A_q, J_frozen, _ = ctx_frozen.solve_frozen(np.asarray(rho, dtype=float))
    return flux_torque_full_derivative(
        rho, ctx_frozen, A_d, A_q, J_frozen, device=device)


def flux_torque_full_derivative(
    rho: np.ndarray,
    ctx_frozen: FluxFrozenContext,
    A_d: np.ndarray,
    A_q: np.ndarray,
    J_frozen,
    *,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """复用已解状态的实现（供测试/探针用，一次求解）。

    与 ``flux_torque_gradient`` 数学相同，但不重复冻结求解 —— 手写路径已经
    解过 d/q，本函数接收解好的 ``A_d``/``A_q``/``J_frozen``。
    """
    rho_t = torch.tensor(np.asarray(rho, dtype=float), dtype=torch.float64,
                         requires_grad=True, device=device)
    A_d_t = torch.as_tensor(np.asarray(A_d, dtype=float),
                            dtype=torch.float64, device=device
                            ).requires_grad_(True)
    A_q_t = torch.as_tensor(np.asarray(A_q, dtype=float),
                            dtype=torch.float64, device=device
                            ).requires_grad_(True)

    Bxcoef = torch.as_tensor(np.asarray(ctx_frozen.top.em.Bxcoef),
                             dtype=torch.float64, device=device)
    Bycoef = torch.as_tensor(np.asarray(ctx_frozen.top.em.Bycoef),
                             dtype=torch.float64, device=device)
    tri = torch.as_tensor(np.asarray(ctx_frozen.top.layer.tri, dtype=np.int64),
                          device=device)
    Bx_d, By_d = _field_components(A_d_t, Bxcoef, Bycoef, tri)
    Bx_q, By_q = _field_components(A_q_t, Bxcoef, Bycoef, tri)

    k_e = _flux_torque_kernel_torch(
        Bx_d, By_d, Bx_q, By_q, ctx_frozen.I_d, ctx_frozen.I_q,
        np.asarray(ctx_frozen.top.em.area),
        pole_pairs=ctx_frozen.pole_pairs, parts=ctx_frozen.parts,
    )
    # 显式项梯度：μ(ρ) 的显式依赖（T = Σ k_e·ν_e 对 ρ 的部分，A 固定）—— torch 自动
    Bf_t = torch.as_tensor(ctx_frozen.B_frozen, dtype=torch.float64, device=device)
    rctx = ctx_frozen.residual_views(A_d, device=device)
    nu = torch_nu(
        Bf_t, rho_t, rctx.masks,
        ctx_frozen._torch_curve(), ctx_frozen._torch_simp(),
        ctx_frozen.material.mu0, ctx_frozen.material.mur_linear,
    )
    T = (k_e * nu).sum()
    # 同一张 T 图要对 rho_t / A_d_t / A_q_t 三次求导 ⇒ retain_graph=True
    (g_expl,) = torch.autograd.grad(T, rho_t, retain_graph=True)

    # 隐式依赖：dA/dρ 经冻结系统 —— 需 (∂T/∂A_d)·(dA_d/dρ) + (∂T/∂A_q)·(dA_q/dρ)。
    # 用残差式伴随：λ_i = J_frozen⁻ᵀ gA_i，贡献 = −∂(λᵀF_frozen)/∂ρ
    (gA_d,) = torch.autograd.grad(T, A_d_t, retain_graph=True)
    (gA_q,) = torch.autograd.grad(T, A_q_t)
    lam_d = sp_sparse_linalg.spsolve(J_frozen.T, gA_d.detach().cpu().numpy())
    lam_q = sp_sparse_linalg.spsolve(J_frozen.T, gA_q.detach().cpu().numpy())

    v_d = ctx_frozen.residual_views(A_d, device=device)
    v_q = ctx_frozen.residual_views(A_q, device=device)
    with torch.enable_grad():
        rho_imp = rho_t.detach().clone().requires_grad_(True).to(torch.float64)
        F_d = residual_lam_frozen(
            A_d_t, rho_imp,
            torch.as_tensor(lam_d, dtype=torch.float64, device=device), v_d, Bf_t)
        F_q = residual_lam_frozen(
            A_q_t, rho_imp,
            torch.as_tensor(lam_q, dtype=torch.float64, device=device), v_q, Bf_t)
        (g_impl,) = torch.autograd.grad(-(F_d + F_q), rho_imp)

    return (g_expl + g_impl).detach().cpu().numpy()
