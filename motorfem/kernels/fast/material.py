"""非线性材料模型的带缓存实现（内核 3 的 fast 层）。

与 ``reference/material.py`` **数学同一、结果逐位相同**，唯一区别是
把「只依赖 ρ 的量」显式缓存起来，不在 Newton 迭代里重复计算。

------------------------------------------------------------------------------
为什么需要缓存（实测数据）
------------------------------------------------------------------------------

``SIMP_func(ρ)`` 与 ``SIMP_dfunc(ρ)`` **只依赖 ρ**。而 ρ 是设计变量，
在整个 Newton 求解过程中**固定不变**——变的是 A，进而变的是 B。
legacy 却在每次 ``field_error_vectorized`` 调用（=每个 Newton 迭代）里
重算 ``SIMP_func(ρ)`` 与 ``SIMP_dfunc(ρ)``。

实测（11160 个单元）::

    每步重算 tanh 版   : 1516 µs
    缓存中间量后       :  672 µs      2.24x
    两者的最大绝对差   : 0.000e+00    ← 逐位相同

这是**纯优化，零数值代价**——但前提是运算顺序与 reference 完全一致
（例如 ``x/B/rho1`` 不能写成 ``x/(B*rho1)``，浮点结果会不同）。
本模块刻意逐字保留 reference 的算式结构，并由 parity 测试以
``np.array_equal`` 锁定。

------------------------------------------------------------------------------
正确使用方式
------------------------------------------------------------------------------

::

    mat = CachedMaterial(curve, simp)

    # 每次 rho 改变时调用一次（即每个 MMA 迭代一次）
    state = mat.prepare_density(rho)

    # Newton 内循环里反复调用，state 复用
    for _ in range(max_iter):
        B = flux_density_magnitude(A, geom)
        mu, dnu_dB = mat.nonlinear(B, state)
        J = jacobian(A, geom, S, nu_of_B=..., ...)

**契约**：``state`` 只在其 ``rho`` 未变时有效。传入与 state 不符的 ρ
会得到错误结果且**不会报错**（这是性能与安全性的取舍）。
``DensityState.rho`` 字段保留原始输入，便于在调试时人工核对；
``CachedMaterial(..., verify=True)`` 会开启逐次校验（慢，仅调试用）。

------------------------------------------------------------------------------
顺带消除的 tanh 调用
------------------------------------------------------------------------------

缓存后，``tanh`` 从 Newton 内循环里**完全消失**——它只在 ``prepare_density``
里执行一次。因此 ``approx/trig.py`` 的 Padé 近似在这条路径上收益归零。
若将来某处无法缓存、又需要 tanh，Padé 仍然可用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..reference.bh_curve import BHCurve
from ..reference.material import (
    B_CLIP_MIN,
    RHO_CLIP_MAX,
    RHO_CLIP_MIN,
    _clip_B,
    _clip_rho,
    material_dnu_drho,
    material_dnu_drho_linear,
    material_values,
    material_values_linear,
)
from ..reference.simp import SIMPProjection

__all__ = ["DensityState", "CachedMaterial", "B_CLIP_MIN"]

#: linear 材料默认的相对磁导率——legacy 硬编码在 ``MMA10:1213``
DEFAULT_MUR_LINEAR = 2000.0


@dataclass(frozen=True)
class DensityState:
    """只依赖 ρ 的量的缓存快照。

    Attributes
    ----------
    rho : (N,) 裁剪到 [0,1] 后的**原始**设计变量。仅用于核对，不参与计算。
    rho1 : (N,) ``SIMP_func(ρ)`` —— 投影后的物理密度
    drho1_drho : (N,) ``SIMP_dfunc(ρ)`` —— 整条链的导数 dρ₁/dρ

    Notes
    -----
    本快照对**任何 B** 都有效；它只依赖 ρ。这正是它能在 Newton 内循环里
    被复用的原因。
    """

    rho: np.ndarray
    rho1: np.ndarray
    drho1_drho: np.ndarray


class CachedMaterial:
    """非线性（与线性）材料模型，带 ρ 相关量缓存。

    Parameters
    ----------
    curve : BHCurve
    simp : SIMPProjection
    mu0 : float 真空磁导率，仅线性材料用
    mur_linear : float 线性材料的相对磁导率（legacy 硬编码 2000）
    verify : bool
        为 True 时每次调用都校验 ``B`` 与 ``state.rho1`` 的形状一致，
        并重新计算 ``SIMP_func`` 做比对。**仅供调试**，会抵消缓存收益。

    Examples
    --------
    >>> mat = CachedMaterial(curve, simp)        # doctest: +SKIP
    >>> state = mat.prepare_density(rho)         # doctest: +SKIP
    >>> mu, dnu_dB = mat.nonlinear(B, state)     # doctest: +SKIP
    """

    def __init__(
        self,
        curve: BHCurve,
        simp: SIMPProjection,
        mu0: float = 4e-7 * np.pi,
        mur_linear: float = DEFAULT_MUR_LINEAR,
        verify: bool = False,
        linear_simp: bool = False,
    ):
        self.curve = curve
        self.simp = simp
        self.mu0 = mu0
        self.mur_linear = mur_linear
        self.verify = verify
        # ★ 2026-10-17：``use_nonlinear_iron=False`` 时 SIMP 支走**线性**
        #   材料 ``μ = μ₀·μ_r·ρ₁``（``dν/dB ≡ 0``）。默认 False = 走非线性
        #   曲线支，**逐位不变**（golden 不受影响）。
        self.linear_simp = linear_simp
        self._n_prepare = 0          # 诊断用：prepare 被调用了几次

    # ---- 缓存建立 ----

    def prepare_density(self, rho: np.ndarray) -> DensityState:
        """计算并缓存所有只依赖 ρ 的量。

        每次 ρ 改变时调用一次即可（每个 MMA 迭代一次），不要在 Newton 内循环里调用。
        """
        rho_c = _clip_rho(rho)
        self._n_prepare += 1
        return DensityState(
            rho=rho_c,
            rho1=self.simp.project(rho_c),
            drho1_drho=self.simp.derivative(rho_c),
        )

    @property
    def n_prepare(self) -> int:
        """``prepare_density`` 被调用的次数。用于验证缓存确实生效。"""
        return self._n_prepare

    # ---- 非线性材料 ----

    def _check(self, state: DensityState, B: np.ndarray) -> None:
        if not self.verify:
            return
        rho1_fresh = self.simp.project(state.rho)
        if not np.array_equal(rho1_fresh, state.rho1):
            raise ValueError("DensityState 与传入的 ρ 不符——缓存被误用")
        if B.shape != state.rho1.shape:
            raise ValueError(
                f"B 形状 {B.shape} 与 state.rho1 形状 {state.rho1.shape} 不一致"
            )

    def nonlinear(
        self, B: np.ndarray, state: DensityState
    ) -> tuple[np.ndarray, np.ndarray]:
        """μ 与 ∂ν/∂B。★ 2026-10-17 补真空项（D13）并同步导数，与
        ``reference.nonlinear_material_topo_opti`` 逐位相同。

        legacy ``nonlinear_material_topo_opti``（``MMA10:1183-1193``）的
        ``mu = _B / H(_B/_rho1)`` 等价于 ``rho1 * mu_iron(B/rho1)``，
        **漏了体积平均的真空项** ``(1-rho1)*mu0``：
        ``rho1 → min_SIMP`` 时 legacy 给 ``mu ~ mu0*min_SIMP < mu0``
        （违反「任何介质 μ ≥ 真空」）。

        修正：``mu = B/H(B/rho1) + mu0*(1 - rho1)``
        —— ``rho1→1`` 时该项 → 0（与 legacy 一致）；``rho1→0`` 时 → ``mu0``。

        ★★ **``dnu_dB`` 必须由新的 μ 求导**（``ν = 1/μ``）::

            dμ/dB = (1 − x·H'(x)/H(x)) / H(x)      x = B/ρ₁
            dν/dB = −(dμ/dB) / μ²

        ⚠ 只改 μ 而不改导数会让 Jacobian 不再是残差的真导数：实测 36 槽
        同一设计 **60 轮不收敛**（同步后 31 轮收敛）。详见
        ``reference/material.py`` 同名函数的表。
        ``rho1 = 1`` 时本式与 legacy 原式 ``H'/B − H/B²`` 恒等（rel 3e-09）。
        """
        self._check(state, B)
        B = _clip_B(B)
        # ★ 公式本体在 ``reference.material.material_values``（**唯一实现**）；
        #   本方法只负责「用缓存的 ρ₁」+「取曲线」。
        if self.linear_simp:
            # 线性档：μ 与 B 无关、dν/dB ≡ 0 ⇒ 方程真线性、Newton 一轮收敛。
            return self.linear(B, state)
        rho1 = state.rho1
        x = B / rho1
        return material_values(B, rho1, self.curve.H(x), self.curve.dHdB(x),
                               self.mu0)

    def nonlinear_dnu_drho(
        self, B: np.ndarray, state: DensityState
    ) -> np.ndarray:
        """∂ν/∂ρ。与 ``reference.nonlinear_material_dnu_drho`` 逐位相同。

        对应 legacy ``nonlinear_material_dgamma_p_drho``（``MMA10:1196-1204``），
        但 ★★ 2026-10-17 起**必须由加了真空项的 μ 求导**（ν = 1/μ）::

            dμ/dρ = [ B²·H'(x)/(ρ₁²·H(x)²) − μ₀ ] · dρ₁/dρ      x = B/ρ₁
            ∂ν/∂ρ = −(dμ/dρ) / μ²

        ``−μ₀`` 项就是真空项的贡献。照抄 legacy 会在低密度处差到 1e12 量级。
        """
        self._check(state, B)
        B = _clip_B(B)
        if self.linear_simp:
            return self.linear_dnu_drho(B, state)
        rho1 = state.rho1
        x = B / rho1
        return material_dnu_drho(B, rho1, self.curve.H(x), self.curve.dHdB(x),
                                 self.mu0, state.drho1_drho)

    # ---- 线性材料 ----

    def linear(
        self, B: np.ndarray, state: DensityState
    ) -> tuple[np.ndarray, np.ndarray]:
        """线性材料的 μ 与 ∂ν/∂B（后者恒为 0）。

        对应 legacy ``linear_material_topo_opti``（``MMA10:1207-1216``）。
        """
        self._check(state, B)
        B = _clip_B(B)
        mu, _ = material_values_linear(state.rho1, self.mu0, self.mur_linear)
        return mu, np.zeros_like(B)

    def linear_dnu_drho(self, B: np.ndarray, state: DensityState) -> np.ndarray:
        """线性材料的 ∂ν/∂ρ。

        对应 legacy ``linear_material_dgamma_p_drho``（``MMA10:1219-1227``）。
        """
        self._check(state, B)
        _ = _clip_B(B)                                # 保持与 reference 相同的形参与语义
        return material_dnu_drho_linear(state.rho1, self.mu0, self.mur_linear,
                                        state.drho1_drho)

    def __repr__(self) -> str:
        return (
            f"CachedMaterial(mur_linear={self.mur_linear:g}, "
            f"verify={self.verify}, n_prepare={self._n_prepare})"
        )
