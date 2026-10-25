"""B-H 曲线的查表求值（approx 层）—— 替换 PCHIP，主动放弃位级一致。

本模块回答的问题是：**Newton 内循环里 ``curve.H`` / ``curve.dHdB`` 值不值得换成查表。**

------------------------------------------------------------------------------
1. 被替换的量与调用点
------------------------------------------------------------------------------

生产路径上曲线只有两个消费点（``grep curve\\.\\(H\\|dHdB\\)``）::

    reference/material.py:181,183   nonlinear_material(B, curve)      ← 纯铁支，x = B
    fast/material.py:182,184        CachedMaterial.nonlinear(B, st)   ← SIMP 支，x = B/rho1

两支的自变量**都是密度缩放磁密** ``x``：纯铁支取 ``rho1 = 1``。所以本层只需要一个
关于 x 的一维函数 ``H(x)``，外加它的导数。

------------------------------------------------------------------------------
2. 表怎么建（每一条都是实测定的，不是拍的）
------------------------------------------------------------------------------

**(a) 采样变量用 x，不用 x²。** 方案的原始设想是「按 B² 等间距」。实测否掉：

    ================  ====  ==============
    表（n=2000）      ms    max|ΔH|/H
    ================  ====  ==============
    uniform-x         0.372 2.36e-03
    **uniform-x²**    0.174 **6.12e-01**
    uniform-log-x     0.413 3.72e-03
    ================  ====  ==============

x² 采样把点堆在 x→0（曲线最平处），而 H 曲率最大的是膝点 ``x≈0.5~2``，那里恰好最稀。
差 250 倍。**故采用 uniform-x。**

**(b) 表域上界取曲线自己的末节点，域外走解析仿射支。**
实测自变量 ``x = B/rho1`` 在真实网格（9599 单元、收敛解）上的分布::

    min 0.025  |  50% 1.14  |  99% 1.79e6  |  max 2.08e6       ← 跨 8 个数量级
    33.7% 的单元落在曲线定义域 [0, 18.814] 之外

uniform 表铺不到 2e6，也不需要铺：曲线末段本来就按 μ0 线性延拓，实测
``curve.H(1e6)`` 与解析直线吻合到 12 位（斜率两边都是 ``1/μ0 = 795775``，
即 PCHIP 末段的「三次外推」实际退化成这条直线）。故::

    x > x_max:   H = H_end + (x − x_max)·tail_slope
                 H' = tail_slope
    tail_slope = (Hg[-1] − Hg[-2]) / dx      ← 从表自己算，不另传 mu0，避免不一致

**(c) 导数用割线斜率，不用真导数。**
线性插值的函数，其导数就是割线。存「真导数」再线性插值会让 **Jacobian 与残差
不自洽**（Jacobian 不再是残差里那个 H 的导数），Newton 从二次退化成准牛顿。
用割线则 Newton 仍是对「插值后的问题」的精确 Newton。**副作用：只需存 H 一个数组。**

------------------------------------------------------------------------------
3. 实测精度（表 n=65536，域 [0, 18.813863]，割线导数）
------------------------------------------------------------------------------

探针取真实 x 分布（6000 点对数均匀 [1e-2, 2] + 3599 点对数均匀 [1e3, 2e6]）::

    n        表 MB   max|ΔH|/H    max|Δν|/ν    max|Δ(dν/dB)|   /尺度     割线−真导
    8192     0.1     3.97e-04     3.97e-04     4.364e+02       8.40e-03   3.46e-02
    65536    0.5     6.30e-06     6.30e-06     4.867e+01       9.36e-04   4.41e-03
    262144   2.1     4.05e-07     4.05e-07     1.323e+01       2.54e-04   1.19e-03
    1048576  8.4     2.51e-08     2.51e-08     3.764e+00       7.24e-05   2.95e-04

（``尺度 = max|dν/dB| = 5.198e+04``。``dν/dB`` 会过零 ⇒ **相对误差无意义**，
只能看绝对误差 / 尺度。末列是割线导数与真 PCHIP 导数的最大相对差 —— 它**不是误差**，
它就是 (c) 里那个「换上插值后问题变了」的量，Newton 一致性由它保证。）

------------------------------------------------------------------------------
4. 实测加速比（BLAS 单线程 / AC 电源）
------------------------------------------------------------------------------

9599 个单元，一次 ``curve.H`` + ``curve.dHdB``::

    scipy PCHIP（现状）      0.302 ms       1.00×
    numpy np.interp LUT      0.372 ms       1.23×  ← **更慢**
    numba LUT n=65536        0.022 ms      13.7×
    numba LUT n=1048576      0.063 ms       4.8×

numpy 版输在中间数组的内存流量 —— 与 ``docs/已否定的优化方向.md`` §4 否掉
「合并两次 scipy 调用」时给出的根因**同一个**（``PPoly._evaluate`` 是单个 C 循环，
无中间数组）。所以 **numpy 版没有意义，本模块必须有 numba**。

表大小与 cache：1M 表（8.4 MB/列）比 64K 表（0.5 MB/列）慢 2.8× —— **随机访问的
cache 命中**才是约束，不是容量。故默认取 65536。

**整步口径（`tools/probe_step_profile.py --bh-lut` 的 A/B，10 步 / AC / BLAS 单线程）**::

    cholmod：curve.H 0.198→0.031 ms/次，curve.dHdB 0.213→0.021 ms/次
             曲线合计占全局 11.8% → 1.7%；步总计 1487.5 → 1322.9 ms/步   = 1.124×
    superlu：整轮 solve 250.0 → 241.7 ms                                  = 1.034×

⚠ **收益强依赖后端**：后端越慢（superlu 一轮 ~19 ms，分解就占 14 ms），曲线的相对
份额越低、本模块越不值；cholmod 一轮只有 ~6 ms，曲线占 14.6%。
**报这个收益必须注明后端**（同 §11⑥「必须带电源状态」的道理）。

------------------------------------------------------------------------------
5. 使用契约
------------------------------------------------------------------------------

``TabulatedHcurve`` 是 ``BHCurve`` 的 drop-in（同 ``H`` / ``dHdB`` 签名），
唯一的生产切换点是 ``fem/em_worker.make_field_material``。**``fast/`` 与
``reference/`` 一行不改** —— 换的是注入进去的曲线对象，不是调用点。

``Wm`` / ``dWmdB`` **不近似**，直接委托给被包裹的原曲线（见 ``Wm`` 的说明）。

⚠ 使用本模块即放弃与 legacy 的位级一致（approx 层的定义）。它不在 golden 对齐
路径上，只由显式配置开启（``config.material.bh_lut_n``，**默认 0 = 关**）。

⚠⚠ **与 autograd oracle 层结构互斥（实测，2026-09-29）**：``motorfem/autograd/field.py``
读 ``curve._H_of_B.c/.x``（scipy 插值器的**分段系数**）在 torch 里重建精确 PCHIP ——
oracle 的价值就是当「手写路径」的**独立精确参考**，用近似替身就失去意义。
故把 ``bh_lut_n`` 默认打开会挂 8 个测试（7 个 autograd + 1 个 golden 轨迹）；
``field.py`` 已改成**明确报 TypeError**（不再是难懂的 ``AttributeError``）。
⇒ **要跑 autograd 层就必须 ``bh_lut_n = 0``。**
"""

from __future__ import annotations

import numpy as np

from ..reference.bh_curve import BHCurve, DEFAULT_CSV, TailExtension
from ..reference.material import _clip_B
from ..reference.material import material_scalar as _material_scalar

__all__ = [
    "TabulatedHcurve",
    "TabulatedCachedMaterial",
    "DEFAULT_N_TABLE",
]

#: 默认表点数。见 docstring §4：64K 是「精度已到 6e-6、且整表落在 cache 里」的取值。
DEFAULT_N_TABLE = 65536

#: 导数的下界（与 reference.material 的 B 下界同源，防止 /B² 除零）
_B_CLIP_MIN = 1e-20


def _numba():
    """延迟取 numba（本模块是唯一需要它的地方；缺失时给明确报错而不是静默降级）。"""
    try:
        from numba import njit
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "approx/bh_lut 需要 numba。没有 numba 时请用由 scipy PCHIP 支撑的 "
            "reference.bh_curve.BHCurve —— numpy 版的查表实测比 PCHIP 慢 1.23×，"
            "见本模块 docstring §4，故不提供降级路径。"
        ) from exc
    return njit


# =============================================================================
# numba 内核
# =============================================================================
#
# ⚠ 内核里**不出现**任何 numpy 临时数组运算：整个循环逐元素标量算完，
#   H / dH 直接写进出参。这是本模块相对 numpy 版唯一的速度来源。

def _build_kernels():
    """构造内核。放在函数里是为了让 numba 成为**惰性**依赖。"""
    njit = _numba()

    # ★ 材料公式的标量实现：``reference.material.material_scalar`` 是
    #   **唯一来源**（与 numpy 路径的 ``material_values`` 同形）。
    #   这里用同一个函数对象编译成 numba 版本 —— 不在本文件重抄算式。
    material_scalar = njit(cache=True, fastmath=False)(_material_scalar)

    @njit(cache=True, fastmath=False)
    def eval_lut(x, Hg, dHg, inv_dx, kmax, x_max, H_end, tail_slope):
        """H(x) 与 H'(x)（割线）。x 任意维；返回两个同形数组。

        x < 0 走奇对称（H 奇、H' 偶）—— 生产路径不会出现负值（调用方已
        ``clip(min=1e-20)``），但 ``BHCurve.H`` 的文档承诺「B 可为负」，故保留。
        """
        n = x.size
        H = np.empty(n, dtype=np.float64)
        dH = np.empty(n, dtype=np.float64)
        for i in range(n):
            xi = x[i]
            s = 1.0
            if xi < 0.0:
                xi = -xi
                s = -1.0
            if xi >= x_max:
                H[i] = s * (H_end + (xi - x_max) * tail_slope)
                dH[i] = tail_slope
            else:
                f = xi * inv_dx
                k = int(f)
                if k < 0:
                    k = 0
                elif k > kmax:
                    k = kmax
                w = f - k
                h = Hg[k] + w * (Hg[k + 1] - Hg[k])
                H[i] = s * h
                dH[i] = dHg[k]
        return H, dH

    @njit(cache=True, fastmath=False)
    def fused_nonlinear(B, rho1, Hg, dHg, inv_dx, kmax, x_max, H_end, tail_slope,
                        mu0):
        """把 ``CachedMaterial.nonlinear`` 整条链融合成一次遍历。

        ``B`` 与 ``rho1`` 同形；返回 ``(mu, dnu_dB)``::

            x    = clip(B, 1e-20) / rho1
            H    = lut_H(x)                H' = lut_dH(x)   ← 割线
            mu    = B / H + mu0 * (1 − rho1)          ← ★ 真空项（D13, 2026-10-17）
            dmu   = (1 − x·H'/H) / H                  ← ★ 与 mu 同步
            dnu   = −dmu / mu²

        ★ **公式单点化（2026-10-17）**：``(μ, ∂ν/∂B)`` 的两行算式不再写在这里，
        而是逐元素调用 ``reference.material.material_scalar`` —— 那个函数是
        标量版公式的**唯一实现**，numpy 路径的 ``material_values`` 与它同形。
        ``test_bh_lut`` 逐点比对两条路径，公式漂移会立刻失败。
        （numba 不接受 Python 调用 ⇒ ``material_scalar`` 用 ``njit`` 编译。）
        approx 层仍不承诺位级一致，但差异现在只剩「插值」这一个来源。

        ★ ``mu0`` 参数与真空项是 2026-10-17 补的（D13）：legacy 的
        ``mu = B/H`` 漏了体积平均的真空项 ``(1-rho1)*mu0``，``rho1→0`` 时
        ``mu < mu0``（物理非法）。见 ``reference/material.py`` 同名函数。

        ★★ ``dnu`` 必须由新的 ``mu`` 求导（``ν = 1/μ``）：只改 ``mu`` 不改
        ``dnu`` 会让 Jacobian 不再是残差的真导数 —— 实测 36 槽 60 轮不收敛，
        同步后收敛。``rho1 = 1`` 时与 legacy 原式 ``H'/B − H/B²`` 恒等。
        """
        n = B.size
        mu = np.empty(n, dtype=np.float64)
        dnu = np.empty(n, dtype=np.float64)
        for i in range(n):
            b = B[i]
            if b < 1e-20:
                b = 1e-20
            r1 = rho1[i]
            x = b / r1
            if x >= x_max:
                H = H_end + (x - x_max) * tail_slope
                dH = tail_slope
            else:
                f = x * inv_dx
                k = int(f)
                if k < 0:
                    k = 0
                elif k > kmax:
                    k = kmax
                w = f - k
                H = Hg[k] + w * (Hg[k + 1] - Hg[k])
                dH = dHg[k]
            mu[i], dnu[i] = material_scalar(b, r1, H, dH, mu0)
        return mu, dnu

    return eval_lut, fused_nonlinear


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _build_kernels()
    return _KERNELS


# =============================================================================
# 表
# =============================================================================

class TabulatedHcurve:
    """``BHCurve`` 的查表替身（approx 层）。

    Parameters
    ----------
    curve : BHCurve
        被逼近的原曲线。表的节点值直接用它采样，**尾段斜率也从它算**
        （见模块 docstring §2b），因此不需要另传 ``mu0``。
    n : int
        表点数（含端点）。默认 65536，见模块 docstring §4。

    Examples
    --------
    >>> ref = BHCurve.from_csv("磁障设计/pytorch/50CS470.csv")   # doctest: +SKIP
    >>> lut = TabulatedHcurve(ref, n=65536)                    # doctest: +SKIP
    >>> float(lut.H(np.array([1.5]))[0])                       # doctest: +SKIP
    """

    def __init__(self, curve: BHCurve, n: int = DEFAULT_N_TABLE):
        if n < 4:
            raise ValueError(f"表点数至少 4，实得 {n}")
        self._curve = curve
        self._n = int(n)

        # 域上界 = 曲线自己的末节点。用 table 而不是私有属性，避免依赖 BHCurve 内部。
        H_tab, B_tab = curve.table
        x_max = float(B_tab[-1])

        xg = np.linspace(0.0, x_max, self._n)
        Hg = np.ascontiguousarray(curve.H(xg), dtype=np.float64)
        dx = x_max / (self._n - 1)
        # 割线斜率（长度 n-1）—— 见模块 docstring §2c
        dHg = np.ascontiguousarray(np.diff(Hg) / dx, dtype=np.float64)

        self._xg = xg
        self._Hg = Hg
        self._dHg = dHg
        self._inv_dx = 1.0 / dx
        self._kmax = self._n - 2
        self._x_max = x_max
        self._H_end = float(Hg[-1])
        self._tail_slope = float(dHg[-1])       # = 1/μ0（实测与解析斜率一致到 12 位）

    # ---- 构造 ----

    @classmethod
    def from_csv(
        cls,
        csv_path=DEFAULT_CSV,
        mu0: float = 4e-7 * np.pi,
        tail: TailExtension = TailExtension(),
        n: int = DEFAULT_N_TABLE,
    ) -> "TabulatedHcurve":
        """与 ``BHCurve.from_csv`` 同签名，先建原曲线再建表。"""
        return cls(BHCurve.from_csv(csv_path, mu0=mu0, tail=tail), n=n)

    # ---- 求值（BHCurve 的 drop-in 接口）----

    def H(self, x: np.ndarray) -> np.ndarray:
        """``H(x)``，单位 A/m。与 ``BHCurve.H`` 同签名（不做裁剪，可为负）。"""
        x = np.asarray(x, dtype=np.float64)
        eval_lut, _ = _kernels()
        H, _ = eval_lut(x.ravel(), self._Hg, self._dHg, self._inv_dx,
                        self._kmax, self._x_max, self._H_end, self._tail_slope)
        return H.reshape(x.shape)

    def dHdB(self, x: np.ndarray) -> np.ndarray:
        """``dH/dx``（**插值函数的割线导数**，见模块 docstring §2c）。"""
        x = np.asarray(x, dtype=np.float64)
        eval_lut, _ = _kernels()
        _, dH = eval_lut(x.ravel(), self._Hg, self._dHg, self._inv_dx,
                         self._kmax, self._x_max, self._H_end, self._tail_slope)
        return dH.reshape(x.shape)

    def fused_nonlinear(self, B: np.ndarray, rho1: np.ndarray):
        """一次遍历给出 ``(mu, dnu_dB)``。等价于 ``CachedMaterial.nonlinear``。

        ⚠ 真空项 ``mu0`` 从 ``_tail_slope`` 反推（``mu0 = 1/tail_slope``）——
        尾段斜率按构造就是 ``1/μ0``（见 ``__init__`` 的注释），故不需要新属性。
        """
        _, fused = _kernels()
        mu0 = 1.0 / self._tail_slope
        return fused(np.ascontiguousarray(B, dtype=np.float64),
                     np.ascontiguousarray(rho1, dtype=np.float64),
                     self._Hg, self._dHg, self._inv_dx,
                     self._kmax, self._x_max, self._H_end, self._tail_slope,
                     mu0)

    # ---- 不近似的部分：原样委托 ----

    def Wm(self, B: np.ndarray) -> np.ndarray:
        """**不近似**，直接走原曲线。

        ``W_m`` 有 1.2e-8 的绝对精度底噪（积分常数 ULP，见 ``bh_curve.BHCurve.Wm``），
        查表会再叠一层插值误差且无处比对；而它不在生产 Newton 路径上
        （``nonlinear_energy`` 已不使用）。故原样委托，宁慢不错。
        """
        return self._curve.Wm(B)

    def dWmdB(self, B: np.ndarray) -> np.ndarray:
        """**不近似**，同上。"""
        return self._curve.dWmdB(B)

    # ---- 诊断 ----

    @property
    def n_table(self) -> int:
        return self._n

    @property
    def x_max(self) -> float:
        """表域上界（= 原曲线的末节点 B）。域外走解析仿射支。"""
        return self._x_max

    @property
    def tail_slope(self) -> float:
        """末段斜率，实测等于 ``1/μ0``。"""
        return self._tail_slope

    @property
    def table_bytes(self) -> int:
        """表占的字节数（两列 float64）。"""
        return self._Hg.nbytes + self._dHg.nbytes

    def max_rel_error_H(self, x_probe: np.ndarray) -> float:
        """在给定的探测点上，相对原曲线的 ``max|ΔH|/H``。精度验收用。"""
        x_probe = np.asarray(x_probe, dtype=np.float64)
        ref = self._curve.H(x_probe)
        return float(np.max(np.abs(self.H(x_probe) - ref) / np.abs(ref)))

    @property
    def curve(self) -> BHCurve:
        """被包裹的原曲线（``Wm`` 等未近似量的来源）。"""
        return self._curve

    def __repr__(self) -> str:
        return (
            f"TabulatedHcurve(n={self._n}, x∈[0, {self._x_max:.6g}], "
            f"tail_slope={self._tail_slope:.6g}, {self.table_bytes / 1e6:.2f} MB)"
        )


# =============================================================================
# 材料（SIMP 支的融合路径）
# =============================================================================

def _tabulated_cached_material_cls():
    """构造 ``CachedMaterial`` 的子类。

    子类只在 ``__init__`` 里要求 ``curve`` 是 ``TabulatedHcurve``，并把 ``nonlinear``
    换成融合内核；其余方法（``nonlinear_dnu_drho`` / ``linear`` …）原样继承 ——
    它们走的 ``self.curve.dHdB`` 已经是查表实现了。

    放在函数里是为了让 ``fast.material`` 的导入不因 numba 缺失而失败。
    """
    from ..fast.material import CachedMaterial

    class TabulatedCachedMaterial(CachedMaterial):
        """``CachedMaterial`` 的查表版：``nonlinear`` 走融合内核（一次遍历）。

        ⚠ 与 ``CachedMaterial.nonlinear`` **不逐位相同**（approx 层定义如此）。
        差值来源唯一：``H`` / ``dH/dx`` 由插值给出，见模块 docstring §3。
        """

        def nonlinear(self, B, state):
            self._check(state, B)
            return self.curve.fused_nonlinear(
                np.asarray(B, dtype=np.float64), state.rho1
            )

        def __repr__(self) -> str:
            return (
                f"TabulatedCachedMaterial(curve={self.curve!r}, "
                f"verify={self.verify}, n_prepare={self._n_prepare})"
            )

    return TabulatedCachedMaterial


_CLS = None


def TabulatedCachedMaterial(curve, simp, mu0=4e-7 * np.pi,
                            mur_linear=2000.0, verify=False):  # noqa: N802 - 工厂，故意用类名
    """``TabulatedCachedMaterial`` 工厂（惰性建类，见 ``_tabulated_cached_material_cls``）。"""
    global _CLS
    if _CLS is None:
        _CLS = _tabulated_cached_material_cls()
    if not isinstance(curve, TabulatedHcurve):
        raise TypeError(
            "TabulatedCachedMaterial 需要 TabulatedHcurve（否则融合内核没有表可用）；"
            f"实得 {type(curve).__name__}"
        )
    return _CLS(curve, simp, mu0, mur_linear, verify)
