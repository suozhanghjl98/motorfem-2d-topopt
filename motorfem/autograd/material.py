"""motorfem.autograd.material —— 阶段 6：torch 可微的 SIMP 投影与材料链

oracle 的残差式伴随需要 `∂F/∂ρ`。F 的 ρ 依赖全部经由单元磁阻率 `ν(ρ)`（B 固定），
而 `ν` 的链是：`ρ → clip → SIMP 投影(tanh+幂律) → ν = H(B/ρ₁)/B`。本模块在 torch
里重建这条链（`TorchSIMP` + `torch_nu`），与 numpy 侧（`CachedMaterial.nonlinear` +
`blend_class_values`）逐值核对（1e-12 级）。

四材料掩码的混合照抄 legacy（``MMA10:2236-2237``）::

    mu = is_vacuum·mu0 + is_SIMP·mu_SIMP + is_non_SIMP·mu_non_SIMP
         + is_linear_iron·(mur_linear·mu0)
    ν = 1/mu

⚠ 残差路径里**线性铁芯是常数**（`mur_linear·mu0`，无 ρ₁）—— ρ 只经 `is_SIMP`
分支进入。这与 ``CachedMaterial.linear()``（含 ρ₁）不同：后者是冻结分析用的另一条
链，不在本模块范围。
"""

from __future__ import annotations

import torch

from ..kernels.fast.material import (
    B_CLIP_MIN,
    RHO_CLIP_MAX,
    RHO_CLIP_MIN,
    DEFAULT_MUR_LINEAR,
)
from ..kernels.reference.newton import MaterialMasks
from ..kernels.reference.simp import SIMPProjection
from .pchip import TorchPCHIP

__all__ = ["TorchSIMP", "torch_nu", "torch_nu_of_rho"]

_MASKS_NAMES = ("is_vacuum", "is_simp", "is_non_simp", "is_linear_iron")


class TorchSIMP:
    """``SIMPProjection`` 的 torch 重建（``project`` 整条链可微）。

    与 ``SIMPProjection.project`` 数学同一（tanh 阈值投影 → 线性混合 → 幂律），
    常数（`_T`/`_A`/参数）在构造时从 numpy 版本提取，避免两边参数漂移。
    """

    def __init__(self, simp: SIMPProjection):
        p = simp.params
        self.min_density = float(p.min_density)
        self.power = float(p.power)
        self.linear_factor = float(p.linear_factor)
        self.beta = float(p.beta)
        self.thb = float(p.thb)
        self._T = float(simp._T)
        self._A = float(simp._A)

    def project(self, rho: torch.Tensor) -> torch.Tensor:
        """完整投影 ``ρ_p(ρ)``（torch 可微；与 ``SIMPProjection.project`` 一致）。"""
        rho = rho.to(torch.float64)
        fc = self.linear_factor
        rho1 = (torch.tanh(self.beta * (rho - self.thb)) + self._T) / self._A
        rho_c = rho1 * (1.0 - fc) + rho * fc
        return self.min_density + (1.0 - self.min_density) * rho_c ** self.power


def torch_nu(
    B: torch.Tensor,
    rho: torch.Tensor,
    masks: MaterialMasks,
    curve: TorchPCHIP,
    simp: TorchSIMP,
    mu0: float,
    mur_linear: float = DEFAULT_MUR_LINEAR,
) -> torch.Tensor:
    """四类混合的磁阻率 ``ν = 1/μ``（torch 可微，B 固定 ⇒ ρ 只经 SIMP 分支）。

    算术与 legacy ``blend_class_values`` 逐字一致（加权不是选择），ρ 依赖只出现在
    ``is_simp * mu_SIMP`` 项。``B`` 应为**已 clip**（≥1e-20）的收敛磁密（常量）。
    """
    rho1 = simp.project(torch.clamp(rho, RHO_CLIP_MIN, RHO_CLIP_MAX))
    Bc = torch.clamp(B, min=B_CLIP_MIN)

    # 照 legacy：mu_SIMP / mu_non 对**全部**单元求值（掩码算术加权）。
    # 掩码可能是 numpy（调用方直接传）或 torch（经 ResidualContext.to_torch），统一转。
    is_vacuum = torch.as_tensor(masks.is_vacuum, device=B.device, dtype=torch.float64)
    is_simp = torch.as_tensor(masks.is_simp, device=B.device, dtype=torch.float64)
    is_non_simp = torch.as_tensor(masks.is_non_simp, device=B.device, dtype=torch.float64)
    is_linear_iron = torch.as_tensor(masks.is_linear_iron, device=B.device, dtype=torch.float64)

    mu_simp = Bc / curve.eval(Bc / rho1)
    mu_non = Bc / curve.eval(Bc)

    mu = (
        is_vacuum * mu0
        + is_simp * mu_simp
        + is_non_simp * mu_non
        + is_linear_iron * (mur_linear * mu0)
    )
    return 1.0 / mu


def torch_nu_of_rho(
    rho: torch.Tensor,
    B: torch.Tensor,
    masks: MaterialMasks,
    curve: TorchPCHIP,
    simp: TorchSIMP,
    mu0: float,
    mur_linear: float = DEFAULT_MUR_LINEAR,
) -> torch.Tensor:
    """同 ``torch_nu``，参数顺序颠倒以便 ``rho.requires_grad`` 时写法自然。

    仅在调用方想把 ρ 放首位时用；数学完全一致。
    """
    return torch_nu(B, rho, masks, curve, simp, mu0, mur_linear)
