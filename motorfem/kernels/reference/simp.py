"""SIMP 密度投影与反演（内核 2）。

来源：``legacy/pytorch/nonlinear_solver_energy_MMA10.py:1118-1155``

本内核**存在** ref/fast 差异，但只在一处：``invert``。
legacy 把解析反演**注释掉**、改用 1000 点查表（``MMA10:1128-1139``），
那是一个真实的数值近似。``project`` 与 ``derivative`` 两层数学同一，
``fast`` 直接复用本模块（见 kernels/README.md 的分层判据）。

------------------------------------------------------------------------------
数学表述
------------------------------------------------------------------------------

SIMP 投影把设计变量 ρ ∈ [0, 1] 映射为「物理密度」，分三步：

1) tanh 阈值投影
   ::

       ρ₁(ρ) = [ tanh(β(ρ − thb)) + tanh(β·thb) ]
               ───────────────────────────────────────
               [ tanh(β(1 − thb)) + tanh(β·thb) ]

   两条边界性质（分母正是为此选取）::

       ρ = 0  →  分子 = −tanh(β·thb) + tanh(β·thb) = 0          → ρ₁ = 0
       ρ = 1  →  分子 =  tanh(β(1−thb)) + tanh(β·thb) = 分母    → ρ₁ = 1

   故 ``ρ₁ ∈ [0, 1]`` 且单调递增。β 越大越接近在 ρ = thb 处的阶跃
   （阈值化），β → 0 时退化为恒等映射。

2) 与原密度的线性混合
   ::

       ρ_c = ρ₁ · (1 − fc) + ρ · fc

   fc（legacy 的 ``SIMP_linear_factor``）控制「多大程度上保留原始线性插值」：
   fc = 0 → 纯投影；fc = 1 → 完全退化为 SIMP 的原始线性密度。
   生产配置取 fc = 0（``MMA10:128``）。

3) SIMP 幂律
   ::

       ρ_p = min_SIMP + (1 − min_SIMP) · ρ_c^p

   ``min_SIMP`` 是密度下界（人工防止刚度矩阵奇异），p 是惩罚指数。

导数（链式）::

    dρ₁/dρ =  β · sech²(β(ρ − thb)) / [ tanh(β(1−thb)) + tanh(β·thb) ]

``derivative`` 返回的是**整条链**的导数 ``dρ_p/dρ``，而不是 ``dρ₁/dρ``。
注意上游拿到它时要的是 ``dρ_p/dρ`` 没错，但命名容易误读——本模块显式区分。

反演（``invert``）::

    ρ = thb + arctanh( ρ₁ · A − T ) / β

    其中 A = tanh(β(1−thb)) + tanh(β·thb) ,  T = tanh(β·thb)

   推导：把 ρ₁ 的定义式解出 tanh(β(ρ−thb)) = ρ₁·A − T，再取 arctanh。

   定义域保证：ρ₁ ∈ [0,1] 时 ``ρ₁·A − T ∈ [−T, T1] ⊂ (−1, 1)``，故
   arctanh 不会溢出。fc ≠ 0 时因 ρ_c 里同时含 ρ₁(ρ) 与 ρ，无法闭式求解，
   退回 ``brentq`` 数值求根（精确但慢）。

------------------------------------------------------------------------------
出处与文档缺口
------------------------------------------------------------------------------

理论文档 ``同步磁阻电机磁障结构分析.md`` 的
「基于密度函数的最大转矩拓扑优化方法」一节（约第 1633 行）记录了 **SIMP
的幂律部分**::

    μ = μ₀[1 + (μr − 1)·ρ^p]  ≈  μ₀[1 + μr·ρ^p]

但**没有**记录本模块实现的 **tanh 阈值投影**（``ρ₁(ρ)`` 那一层）。
全文检索 ``tanh`` / ``投影`` / ``阈值`` 均为 **0 处命中**。

也就是说：从「简单幂律」到「tanh 阈值投影 + 线性混合 fc」这一步演化
**没有写进理论文档**。本模块的数学是从 legacy 代码（``MMA10:1118-1155``）
与 tanh 投影的定义式反推并独立验证的（边界值、单调性、反演往返一致、
与 legacy 逐位比对的测试见 ``tests/kernels/test_simp.py``），
但缺少原始的设计意图记录——例如 ``β = 8``、``thb = 0.5`` 是怎么定的。

**建议**：若这一步演化有推导或文献依据，补进理论文档；
否则至少记录参数取值的理由。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize as sp_optimize

__all__ = ["SIMPParams", "SIMPProjection"]


@dataclass(frozen=True)
class SIMPParams:
    """SIMP 投影参数。默认值取自 legacy ``MMA10:124-128`` 的电磁侧配置。

    Attributes
    ----------
    beta : float
        阈值锐度。legacy 的 ``SIMP_beta``。
    thb : float
        阈值位置。legacy 的 ``SIMP_thb``。
    power : float
        SIMP 惩罚指数。legacy 的 ``p_SIMP``。
    min_density : float
        密度下界。legacy 的 ``min_SIMP``。
    linear_factor : float
        与原始线性密度的混合比。legacy 的 ``SIMP_linear_factor``。0 = 纯投影。
    """

    beta: float = 8.0
    thb: float = 0.5
    power: float = 3.0
    min_density: float = 1e-6
    linear_factor: float = 0.0


class SIMPProjection:
    """SIMP 投影 ``ρ → ρ_p`` 及其导数与反演。

    Parameters
    ----------
    params : SIMPParams

    Examples
    --------
    >>> proj = SIMPProjection(SIMPParams())
    >>> proj.project(np.array([0.0, 0.5, 1.0]))       # doctest: +SKIP
    array([1.000000e-06, 5.000000e-01, 1.000000e+00])
    """

    def __init__(self, params: SIMPParams = SIMPParams()):
        self.params = params
        # 分母 A 与偏移 T 在构造时算好——它们是常数，且反复出现在
        # project / derivative / invert 里。
        p = params
        self._T = np.tanh(p.beta * p.thb)
        self._A = np.tanh(p.beta * (1.0 - p.thb)) + self._T

    # ---- 前向 ----

    def _rho1(self, rho: np.ndarray) -> np.ndarray:
        """第 1 步：tanh 阈值投影 ``ρ₁(ρ)``。"""
        return (np.tanh(self.params.beta * (rho - self.params.thb)) + self._T) / self._A

    def _rho_c(self, rho: np.ndarray) -> np.ndarray:
        """第 2 步：与原始密度线性混合 ``ρ_c``。"""
        fc = self.params.linear_factor
        return self._rho1(rho) * (1.0 - fc) + rho * fc

    def project(self, rho: np.ndarray) -> np.ndarray:
        """完整投影 ``ρ_p(ρ)``，形状与输入相同。

        对应 legacy ``SIMP_func``（``MMA10:1118-1125``）。
        """
        p = self.params
        rho_c = self._rho_c(np.asarray(rho, dtype=float))
        return p.min_density + (1.0 - p.min_density) * rho_c ** p.power

    def derivative(self, rho: np.ndarray) -> np.ndarray:
        """``dρ_p/dρ``——**整条链**的导数（含幂律那一层）。

        对应 legacy ``SIMP_dfunc``（``MMA10:1142-1155``）。

        关于 ``dρ₁/dρ`` 的等价写法：legacy 写成::

            β · cosh(β·thb) · cosh(β − β·thb) / sinh(β) / cosh(β(ρ−thb))²

        本模块写成 ``β·sech²(β(ρ−thb)) / A``。两者等价，依据是恒等式::

            tanh(a) + tanh(b) = sinh(a+b) / (cosh a · cosh b)

        取 a = β(1−thb)、b = β·thb 得 ``A = sinh(β)/(cosh(β(1−thb))·cosh(β·thb))``，
        代回即得 legacy 的形式（其 ``cosh(β−β·thb)`` 就是 ``cosh(β(1−thb))``）。

        ⚠ legacy 的写法含 ``/ sinh(β)``，β → 0 时除零。本模块的写法无此问题，
        故在 β 很小时更稳健（生产 β = 8，两者数值一致）。
        """
        p = self.params
        rho = np.asarray(rho, dtype=float)

        sech2 = 1.0 / np.cosh(p.beta * (rho - p.thb)) ** 2
        drho_1 = p.beta * sech2 / self._A

        fc = p.linear_factor
        drho_c = drho_1 * (1.0 - fc) + 1.0 * fc        # d(rho·fc)/dρ = fc

        rho_c = self._rho_c(rho)
        return (1.0 - p.min_density) * p.power * rho_c ** (p.power - 1.0) * drho_c

    # ---- 反演 ----

    def _rho_c_from_rho_p(self, rho_p: np.ndarray) -> np.ndarray:
        """幂律层的反演：``ρ_c = ((ρ_p − min)/(1 − min))^(1/p)``。"""
        p = self.params
        return ((rho_p - p.min_density) / (1.0 - p.min_density)) ** (1.0 / p.power)

    def invert(self, rho_p: np.ndarray) -> np.ndarray:
        """投影的反演 ``ρ_p → ρ``。

        fc = 0 时走**闭式解**::

            ρ = thb + arctanh( ρ_c · A − T ) / β

        （legacy 的注释代码 ``MMA10:1134-1137`` 给的正是这个式子，
        但它被 1000 点查表 ``SIMP_invfunc_interp`` 取代了。）

        fc ≠ 0 时 ``ρ_c`` 同时含 ``ρ₁(ρ)`` 与 ``ρ``，无法闭式求解，
        对每个输入用 ``brentq`` 在 [0, 1] 上求根。精确但慢。

        Parameters
        ----------
        rho_p : 投影后的密度，应落在 ``[min_density, 1]``

        Returns
        -------
        rho : 设计变量，落在 [0, 1]

        Raises
        ------
        ValueError
            输入越界。**不静默裁剪**——越界通常意味着上游算错了，
            静默裁剪会把它藏起来。
        """
        p = self.params
        rho_p = np.asarray(rho_p, dtype=float)
        lo = p.min_density - 1e-12
        if np.any(rho_p < lo) or np.any(rho_p > 1.0 + 1e-12):
            raise ValueError(
                f"ρ_p 越界：应在 [{p.min_density:.3g}, 1]，"
                f"实得 [{rho_p.min():.6g}, {rho_p.max():.6g}]"
            )
        rho_p = np.clip(rho_p, p.min_density, 1.0)
        rho_c = self._rho_c_from_rho_p(rho_p)

        if p.linear_factor == 0.0:
            # 闭式解
            arg = np.clip(rho_c * self._A - self._T, -1.0 + 1e-15, 1.0 - 1e-15)
            return np.clip(p.thb + np.arctanh(arg) / p.beta, 0.0, 1.0)

        # fc ≠ 0：逐元素求根
        flat = rho_c.ravel()
        out = np.empty_like(flat)
        for k, target in enumerate(flat):
            out[k] = sp_optimize.brentq(
                lambda r: float(self._rho_c(np.array([r]))[0]) - target,
                0.0, 1.0, xtol=1e-15, rtol=1e-15, maxiter=200,
            )
        return out.reshape(rho_c.shape)

    # ---- 诊断 ----

    def __repr__(self) -> str:
        p = self.params
        return (
            f"SIMPProjection(beta={p.beta}, thb={p.thb}, p={p.power}, "
            f"min={p.min_density:.3g}, fc={p.linear_factor})"
        )
