"""motorfem.autograd.field —— 阶段 6：隐式伴随的 ``EMFieldGrad`` Function

设计文档 §阶段 6 的目标代码形态（F5⑥ 已实测）：``A = solve_field(rho)`` 包成
torch ``autograd.Function``，backward 用**残差式隐式伴随**:

    forward : A = solve_field_vectorized(rho)   # 现有 numpy Newton，一行不改
    backward: λ = spsolve(Jᵀ, gA)               # scipy，唯一手写处
              dL/dρ = −∂(λᵀF(A,ρ))/∂ρ           # torch 自然写法，autograd 管收缩

``OracleContext`` 捆绑一个时间点的工作点（22 元组 / 源项 / 材料 / 几何），供
forward 的 numpy 求解与 backward 的 torch 残差共用。生产路径（``em_step`` 等）
一行不改 —— 本模块是**只读验证层**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sp_sparse_linalg
import torch

from ..kernels.fast import newton as fast_nt
from ..kernels.reference.bh_curve import BHCurve
from .material import TorchSIMP
from .pchip import TorchPCHIP
from .residual import ResidualContext, _TorchViews, residual_lam

__all__ = ["OracleContext", "EMFieldGrad"]


@dataclass
class OracleContext:
    """一个时间点的 oracle 工作点：numpy 求解 + torch 残差共用同一批输入。

    Parameters
    ----------
    top : FieldTopology
    kv : 22 元组（``pre.K_vectorized_vec[i_t]``）
    R : (N_node,) 该时间点的电流源项（``pre.R_vec[i_t]``）
    material : FieldMaterial
    i_t : int（注解用，不参与数值）
    """

    top: object
    kv: tuple
    R: np.ndarray
    material: object
    i_t: int

    # ------------------------------------------------------------ 求解
    def solve(self, rho: np.ndarray) -> tuple:
        """现有 numpy Newton（``fast.newton.solve_field_vectorized``，一行不改）。

        Returns ``(A, B, J)`` —— 与 ``em_step`` 内部同一调用形态。
        """
        top = self.top
        return fast_nt.solve_field_vectorized(
            rho, self.kv, self.R,
            _A0=None, _solve_abs_err=1e-3, max_iter=100,
            iter_relaxation=20, relaxation_ratio=0.5, output=False,
            n_node=top.n_node, area=top.em.area, material=self.material,
        )

    # ------------------------------------------------------------ 残差
    def residual_views(self, A: np.ndarray, device=None) -> _TorchViews:
        """构造残差收缩的 torch 视图（含 ``F_mat_driven @ A`` 常量项）。"""
        # ⚠ oracle 必须是**精确 PCHIP**：它读 scipy 插值器的分段系数在 torch 里重建，
        #   作为「手写路径」的独立参考。查表曲线（``approx.TabulatedHcurve``）没有
        #   ``_H_of_B``，而且就算补上也是**近似** —— oracle 与手写路径用的是同一份
        #   近似就失去参考意义。所以这里**明确拒绝**而不是抛 ``AttributeError``：
        #   oracle 层要跑，必须 ``config.material.bh_lut_n = 0``。
        curve = self.material.curve
        if not isinstance(curve, BHCurve):
            raise TypeError(
                "autograd oracle 需要精确 PCHIP 的 BHCurve（它读 ._H_of_B 的系数重建"
                f" torch PCHIP），实得 {type(curve).__name__}。oracle 是独立精确参考，"
                "不能用近似替身。请设 config.material.bh_lut_n = 0（查表见 "
                "kernels/approx/bh_lut.py，只用于性能路径）。"
            )
        curve = TorchPCHIP(curve._H_of_B.c, curve._H_of_B.x)
        simp = TorchSIMP(self.material.simp)
        rctx = ResidualContext.build(
            self.top, self.kv, self.R, self.material, A,
            curve=curve, simp=simp,
        )
        return rctx.to_torch(device)


class EMFieldGrad(torch.autograd.Function):
    """``A = solve_field(rho)`` 的隐式伴随 Function（阶段 6 oracle 核心）。

    用法（torch 可微）：:

        A = EMFieldGrad.apply(rho_t, oracle_ctx)     # rho_t requires_grad
        T = A @ (M_torch @ A)                        # 下游任何可微量
        dT = torch.autograd.grad(T, rho_t)[0]        # 自动走 backward

    ``backward`` 的残差式伴随：``λ = spsolve(Jᵀ, gA)``，``dL/dρ = −∂(λᵀF)/∂ρ``。
    ``oracle_ctx`` 是非张量参数（backward 返回 None）。
    """

    @staticmethod
    def forward(ctx, rho, oracle: OracleContext):
        rho_np = np.asarray(rho.detach().cpu().numpy(), dtype=float)
        A, B, J = oracle.solve(rho_np)
        ctx.oracle = oracle
        ctx.J = J
        ctx.save_for_backward(
            rho.detach().to(torch.float64),
            torch.as_tensor(A, dtype=torch.float64),
        )
        return torch.as_tensor(A, dtype=torch.float64, device=rho.device)

    @staticmethod
    def backward(ctx, gA):
        rho, A = ctx.saved_tensors
        oracle: OracleContext = ctx.oracle
        # 唯一手写处：一次伴随解（scipy，CPU）
        gA_np = gA.detach().cpu().numpy()
        lam_np = sp_sparse_linalg.spsolve(ctx.J.T, gA_np)

        # torch 自然写法：−λᵀF 对 ρ 的梯度。
        # ⚠ backward 在 torch.no_grad() 下执行 —— 必须 enable_grad 才有图
        #（设计文档 §阶段 6 的代码形态正是这么写的）。
        with torch.enable_grad():
            rho_t = rho.detach().clone().requires_grad_(True).to(torch.float64)
            v = oracle.residual_views(A.detach().cpu().numpy(), device=rho.device)
            lam_t = torch.as_tensor(lam_np, dtype=torch.float64, device=rho.device)
            F_lam = residual_lam(A.detach().to(torch.float64), rho_t, lam_t, v)
            (grho,) = torch.autograd.grad(-F_lam, rho_t)
        return grho.to(rho.dtype), None


# --------------------------------------------------------------------- 便捷
def oracle_from_parts(top, pre, material, i_t) -> OracleContext:
    """从 FieldTopology + Preassembly + FieldMaterial 打包一个时间点的 oracle。

    ``pre`` 提供 22 元组、源项（与 ``em_segment`` 同源）。
    """
    return OracleContext(
        top=top,
        kv=pre.K_vectorized_vec[i_t],
        R=np.asarray(pre.R_vec[i_t]),
        material=material,
        i_t=i_t,
    )
