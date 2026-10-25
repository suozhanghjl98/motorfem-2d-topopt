"""非线性铁磁材料模型（内核 3）。

来源：``legacy/pytorch/nonlinear_solver_energy_MMA10.py:924-936, 1183-1227``

本模块是**纯函数**形式，保持 legacy 的签名以便逐位比对。
``fast/material.py`` 提供带缓存态的同数学实现——两者结果**逐位相同**，
差别只在于 ρ 相关量是否在 Newton 迭代中被重复计算（见 ``fast`` 的说明）。

------------------------------------------------------------------------------
数学表述
------------------------------------------------------------------------------

记号：ν 为**磁阻率**（reluctivity），ν = 1/μ。
legacy 里这个量叫 ``gamma``（所以导数叫 ``dgamma``），本模块统一叫 ν，
但在返回值说明里标注 legacy 的名字以便对照。

1) 无密度（纯铁芯）

   由 B-H 曲线 H(B)::

       ν(B) = H(B) / B
       ∂ν/∂B = H'(B)/B − H(B)/B²

   ``∂ν/∂B`` 就是 legacy 的 ``dgamma``，它进入 Newton 的 Jacobian。

2) 带密度（拓扑优化）

   设 SIMP 投影后的密度为 ``ρ₁ = SIMP_func(ρ)``，材料模型为「密度缩放磁密」::

       ν(B, ρ) = H(B/ρ₁) / B

   即**密度为 ρ₁ 的单元，等效于实心铁芯在磁密 B/ρ₁ 下的工作点**。
   低密度（ρ₁→0）时 B/ρ₁ → ∞，落在 B-H 曲线深饱和段，ν 趋于 μ₀ 级别——
   物理上正确地表现为「空气」。

   对 B 求导::

       ∂ν/∂B = H'(B/ρ₁)/(B·ρ₁) − H(B/ρ₁)/B²

   对 ρ 求导（链式，注意还要乘 ∂ρ₁/∂ρ = SIMP_dfunc）::

       ∂ν/∂ρ = − H'(B/ρ₁) / ρ₁² · (dρ₁/dρ)

   **出处**：理论文档 ``同步磁阻电机磁障结构分析.md`` 的
   「磁链敏感度分析」一节（约第 3171-3183 行）给出了本模型的全微分链式::

       ∂γ/∂θ = −(1/θ²)·f'(B/θ)
       ∂γ/∂B = 1/(B·θ)·f'(B/θ) − 1/B²·f(B/θ)

   其中 ``H = f(B)`` 被明确定义为「密度变量为 1 时的非线性导磁材料的
   B-H 特性函数」。把 ``ν = f(B/θ)/B`` 求导即得上述两式，**与 legacy 代码
   逐字对应**。所以本模型在理论文档里是有完整推导的。

   文档还明确区分了两项的物理含义（原文）：第一部分是「单元密度变化时
   材质的变化直接导致的磁阻率变化」，第二部分是「场域发生变化导致磁密变化、
   进而导致饱和程度变化」——正好对应该模型的 ``∂ν/∂ρ`` 与 ``∂ν/∂B``。

   该模型有一条**精确的标度律**（代数恒等式，已在测试中锁定）::

       ν(B, ρ₁) · ρ₁  =  ν_iron(B/ρ₁)

   等价说法：**固定 B/ρ₁ 时，μ 严格正比于 ρ₁**。也可以把这个模型理解为
   「磁导率按密度线性缩放，工作点按密度反向平移」。这条性质是本模型
   相对其他插值形式最可辨认的特征——若哪天模型被改动，测试会立刻发现。

   两条极限（均已实测）::

       ρ₁ = 1   → 与纯铁芯模型逐位相同
       ρ₁ → 0   → μ → 0（即 ν → ∞），单元几乎不过磁通 —— 这正是磁障
                  能起作用的前提

   ⚠ **文档与代码的一处差异（供核对，非缺陷断言）**

   理论文档在「基于密度函数的最大转矩拓扑优化方法」一节（约第 1633 行）
   给出的是**早期的简单 SIMP**，用磁导率线性插值::

       μ = μ₀[1 + (μr − 1)·ρ^p]  ≈  μ₀[1 + μr·ρ^p]

   该式在 ρ = 0 时给出 **μ = μ₀（空气）**。而本模块（= legacy 代码）用的
   密度缩放模型在 ρ₁ → 0 时给出 **μ → 0**（磁阻率 → ∞），即把空隙当作
   **理想磁绝缘体**而非空气。两者在 ρ = 0 的行为**不同**。

   这不是文档内部矛盾——文档同时记录了早期形式与后来的密度缩放模型
   （后者在敏感度一节有完整链式推导），代码采用的是后者。需要核对的是
   「空隙按 μ→0 而非 μ₀ 处理」这一建模选择是否与后续标定一致：
   它会让优化器高估磁障的隔磁效果。

   另外，代码里 ρ₁ 来自 **tanh 阈值投影**（``SIMP_func``），而理论文档
   只记录了幂律 ``ρ^p``（全文 0 处提到 tanh / 投影 / 阈值）。
   即：**从幂律到 tanh 投影这一步演化没有记进文档**。

3) 线性材料（用于对比与调试）

   ``μ = μ₀ · 2000 · ρ₁``，即相对磁导率固定 2000、不随 B 变化::

       ν(B, ρ) = 1 / (μ₀ · 2000 · ρ₁)
       ∂ν/∂B   = 0
       ∂ν/∂ρ   = − 1/(μ₀·2000) / ρ₁² · (dρ₁/dρ)

------------------------------------------------------------------------------
关于此处的裁剪（clip）
------------------------------------------------------------------------------

legacy 在**每个函数开头**都做::

    _B = np.clip(_B, a_min=1e-20, a_max=np.inf)
    _rho = np.clip(_rho, a_min=0, a_max=1)

**这里才是裁剪该在的地方**，不是 B-H 曲线那边（见 ``bh_curve`` 的说明）：

  - ``B`` 的下界 1e-20 是为 ``ν = H(B)/B`` 与 ``∂ν/∂B`` 里的 ``/B``、``/B²`` 服务。
    空载或无源区域 B 可以恒为 0，不设下界就会除零得到 nan。
  - ``ρ`` 裁剪到 [0,1] 是设计变量的物理范围。SIMP 投影里的 ``ρ_c^p`` 在
    ρ<0 时对非整数 p 会产生 nan。

裁剪放在曲线层会破坏曲线的奇对称（``H(-0.5)`` 被夹成 ``H(1e-20)``）
并让纯数学对象带上调用方的假设；放在这里则语义明确。

------------------------------------------------------------------------------
调用频率与缓存
------------------------------------------------------------------------------

``SIMP_func(ρ)`` 与 ``SIMP_dfunc(ρ)`` **只依赖 ρ**，而 ρ 在整个 Newton
求解过程中固定不变，legacy 却对每次调用都重算。实测把中间量提出循环外可省
2.24×（1516 µs → 672 µs）且结果逐位相同——见 ``fast/material.py``。
"""

from __future__ import annotations

import numpy as np

from .bh_curve import BHCurve
from .simp import SIMPProjection

__all__ = [
    "B_CLIP_MIN",
    "RHO_CLIP_MIN",
    "RHO_CLIP_MAX",
    "nonlinear_material",
    "nonlinear_energy",
    "nonlinear_material_topo_opti",
    "nonlinear_material_dnu_drho",
    "linear_material_topo_opti",
    "linear_material_dnu_drho",
    # ★ 单点化后的公式内核（fast / approx 也调它）
    "material_values",
    "material_values_linear",
    "material_dnu_drho",
    "material_dnu_drho_linear",
]

#: B 的下界，防止 ``H(B)/B`` 与 ``/B²`` 除零（legacy ``MMA10:925``）
B_CLIP_MIN = 1e-20
RHO_CLIP_MIN = 0.0
RHO_CLIP_MAX = 1.0


def _clip_B(B: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(B, dtype=float), B_CLIP_MIN, np.inf)


def _clip_rho(rho: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(rho, dtype=float), RHO_CLIP_MIN, RHO_CLIP_MAX)


# =============================================================================
# ★ 材料公式的**唯一**实现（2026-10-17 单点化）
# =============================================================================
#
# 背景：同一组公式曾经在三个地方各写一遍 ——
#   ``reference/material.py``（纯函数）、
#   ``fast/material.CachedMaterial``（类方法，多了 ρ 缓存）、
#   ``approx/bh_lut.fused_nonlinear``（numba 融合内核）。
# 三份实现漂移过一次，代价很大：D13 加真空项时只改了 μ 的公式而没同步
# ``dν/dB``，导致 Jacobian 不是残差的真导数、牛顿 60 轮不收敛（详见 D13/D14）。
#
# 现在的分工是**公式 vs 缓存**，不是 reference vs fast：
#
#   ``material_values``   纯公式，输入是**已经算好的** H、dH、ρ₁（无任何缓存/
#                         求值逻辑）。所有层都调它。
#   ``material_values_linear``  同上，线性铁芯那一支。
#
#   ``reference`` 的四个公开函数 = 取曲线 + 调内核；
#   ``fast.CachedMaterial``      = 缓存 ρ₁/dρ₁ + 取曲线 + 调内核；
#   ``approx.fused_nonlinear``   = numba 里**逐元素内联**同一组算式（为了省
#                                  中间数组），其正确性由 ``test_bh_lut`` 用
#                                  ``material_values`` 当参照钉死。
#
# ⚠ **改公式只能改这里**。``fast`` 与 ``approx`` 里若再出现 ``B / H`` 之类的
#   算式，就是重复，应当删掉改调内核。


def material_scalar(b, r1, H, dH, mu0):
    """标量版 ``(μ, ∂ν/∂B)`` —— 供 numba 融合内核**逐元素调用**。

    与 ``material_values`` 是**同一个公式**的标量形式（后者对数组向量化，
    这里对单点求值）。两条路径由 ``test_bh_lut`` 的逐点比对钉死 —— 改动任一方
    而未同步另一方，测试立刻失败。

    ⚠ 必须保持与 ``material_values`` **逐字同形**（``b / H``、``x * dH / H``、
    ``-dmu / m ** 2`` 的结合顺序），否则两者在末位上会分叉。
    """
    m = b / H + mu0 * (1.0 - r1)
    dmu = (1.0 - (b / r1) * dH / H) / H
    return m, -dmu / m ** 2


def material_values(B, rho1, H, dH, mu0):
    """非线性铁芯 + SIMP 密度的 ``(μ, ∂ν/∂B)`` —— **公式的唯一实现**。

    Parameters
    ----------
    B : (N,) 单元磁密幅值（**未裁剪**，本函数内部裁剪）
    rho1 : (N,) 投影后的密度 ``ρ₁ = SIMP_func(ρ)``（**已算好**）
    H : (N,) ``H(B/ρ₁)``（**已算好**，调用方负责取曲线）
    dH : (N,) ``H'(B/ρ₁)``（**已算好**）
    mu0 : float 真空磁导率

    Returns
    -------
    mu : (N,) μ = B/H + μ₀(1−ρ₁)
    dnu_dB : (N,) ∂ν/∂B = −(dμ/dB)/μ²，``dμ/dB = (1 − x·H'/H)/H``

    Notes
    -----
    ``x = B/ρ₁`` 是 B-H 曲线的自变量（「密度缩放磁密」，legacy ``MMA10:1183``）。

    ★ 真空项：体积平均 ``μ = ρ₁·μ_铁(B/ρ₁) + (1−ρ₁)·μ₀``，因
    ``ρ₁·(B/ρ₁)/H = B/H``，等价于 ``B/H + μ₀(1−ρ₁)``。legacy 漏了这一项，
    ``ρ₁→0`` 时 μ→0（低于真空，物理非法）。

    ★ 导数必须由**这个** μ 求：``ν = 1/μ`` ⇒ ``∂ν/∂B = −(∂μ/∂B)/μ²``。
    2026-10-17 曾只改 μ 不改导数，36 槽 60 轮不收敛；同步后 31 轮收敛。
    ``ρ₁ = 1`` 时本式与 legacy ``H'/B − H/B²`` 代数恒等（实测 rel 3e-09）。
    """
    B = _clip_B(B)
    x = B / rho1
    mu = B / H + mu0 * (1.0 - rho1)
    dmu_dB = (1.0 - x * dH / H) / H
    dnu_dB = -dmu_dB / mu ** 2
    return mu, dnu_dB


def material_values_linear(rho1, mu0, mur_linear):
    """线性铁芯 + SIMP 密度的 ``(μ, ∂ν/∂B)`` —— **公式的唯一实现**。

    ``μ = μ₀·μ_r·ρ₁``，``∂ν/∂B ≡ 0``（与 B 无关）。此时整个磁场方程是
    **真线性**的：``J`` 与 A 无关，Newton 一轮即精确收敛。

    ⚠ 与 ``blend_class_values`` 里 ``is_linear_iron`` 那一支的
    ``mur_linear * mu0``（**与 ρ 无关**）是**两个不同的模型**：这里乘了 ρ₁，
    那里没有。前者用于 SIMP 区、后者用于非拓扑的纯铁芯，二者不是同一个东西，
    不要合并。详见 ``reference/newton.FieldMaterial`` 的 Notes。
    """
    rho1 = np.asarray(rho1, dtype=float)
    return mu0 * mur_linear * rho1, np.zeros_like(rho1)


def material_dnu_drho(B, rho1, H, dH, mu0, drho1_drho):
    """非线性 SIMP 支的 ``∂ν/∂ρ`` —— **公式的唯一实现**。

    ``dμ/dρ = [ B²·H'(x)/(ρ₁²·H(x)²) − μ₀ ] · dρ₁/dρ``，``∂ν/∂ρ = −(dμ/dρ)/μ²``。
    方括号里的 ``−μ₀`` 是真空项的贡献（legacy 没有；漏掉会差 1e12 量级）。
    """
    B = _clip_B(B)
    x = B / rho1
    mu = B / H + mu0 * (1.0 - rho1)
    dmu_drho1 = B ** 2 * dH / (rho1 ** 2 * H ** 2) - mu0
    return -dmu_drho1 * drho1_drho / mu ** 2


def material_dnu_drho_linear(rho1, mu0, mur_linear, drho1_drho):
    """线性 SIMP 支的 ``∂ν/∂ρ``：``ν = 1/(μ₀μ_rρ₁)`` ⇒ ``−1/(μ₀μ_r)/ρ₁²·dρ₁/dρ``。"""
    return -1.0 / (mu0 * mur_linear) / rho1 ** 2 * drho1_drho



# =============================================================================
# 纯铁芯（无密度）——  legacy MMA10:924-936
# =============================================================================

def nonlinear_material(B: np.ndarray, curve: BHCurve) -> tuple[np.ndarray, np.ndarray]:
    """纯铁芯的磁导率与磁阻率导数。

    对应 legacy ``nonlinear_material``（``MMA10:924-929``）。

    Returns
    -------
    mu : (N,) 磁导率 μ = B/H(B)，单位 H/m
    dnu_dB : (N,) 磁阻率对 B 的导数 ∂ν/∂B（legacy 称 ``dgamma``）

    Notes
    -----
    ``mu`` 在深饱和段会趋近 μ₀，在低 B 段则很大（可达 1e4·μ₀ 量级）。
    上游用 ``1/mu``（即 ν）组装刚度矩阵。
    """
    B = _clip_B(B)
    H = curve.H(B)
    mu = B / H
    dnu_dB = curve.dHdB(B) / B - H / B ** 2
    return mu, dnu_dB


def nonlinear_energy(B: np.ndarray, curve: BHCurve) -> tuple[np.ndarray, np.ndarray]:
    """磁场能量密度及其对 B 的导数。

    对应 legacy ``nonlinear_energy``（``MMA10:932-936``）。

    Returns
    -------
    Wm : (N,) 能量密度 ∫₀^B H db，单位 J/m³
    dWm : (N,) H(B)，即 dW_m/dB

    Notes
    -----
    本函数当前**不在 MMA10 的主流程里**——``nonlinear_solver_energy*.py``
    时代用它构造能量型目标，后来目标改成了机械应变能。保留它是为了
    完整性，以及将来若回到能量法可直接取用。

    ``Wm`` 的绝对精度下限是积分常数的 ULP（见 ``bh_curve.BHCurve.Wm``）。
    """
    B = _clip_B(B)
    return curve.Wm(B), curve.H(B)


# =============================================================================
# 带密度（拓扑优化）——  legacy MMA10:1183-1227
# =============================================================================

def nonlinear_material_topo_opti(
    B: np.ndarray, rho: np.ndarray, curve: BHCurve, simp: SIMPProjection,
    mu0: float = 4e-7 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """非线性铁芯 + SIMP 密度的磁导率与磁阻率导数。

    对应 legacy ``nonlinear_material_topo_opti``（``MMA10:1183-1193``）。

    Parameters
    ----------
    B : (N,) 单元磁密幅值
    rho : (N,) 设计变量（**未投影**的密度）
    curve : BHCurve
    simp : SIMPProjection

    Returns
    -------
    mu : (N,) μ = B/H(B/ρ₁) + μ₀(1−ρ₁)   ← ★ 2026-10-17 补真空项（D13）
    dnu_dB : (N,) ∂ν/∂B = −(dμ/dB)/μ²     ← ★ 与 μ 同步（见 Notes）

    Notes
    -----
    ``ρ₁ = SIMP_func(ρ)`` 是投影后的密度。legacy 用 ``_B / _rho1`` 作为 B-H
    曲线的自变量（「密度缩放磁密」假设，见模块开头的说明）。

    ★ **真空项修正的由来**：按体积平均，单元的等效磁导率应为
    ``μ_eff = ρ₁·μ_铁(B/ρ₁) + (1−ρ₁)·μ₀``——因为 ``ρ₁·(B/ρ₁)/H = B/H``，
    等价于 ``B/H(B/ρ₁) + μ₀(1−ρ₁)``。legacy 只保留了铁的一支，于是
    ``ρ₁→0`` 时 ``μ→0``（低于真空），既违反物理约束又让 ``1/μ`` 爆炸。

    ★★ **``dnu_dB`` 必须与 ``mu`` 同步（2026-10-17 修）**

    2026-10-17 加真空项时 ``dnu_dB`` 照抄了 legacy 原式，导致
    ``dnu_dB ≠ ∂(1/μ)/∂B``。**实测后果是牛顿不改收敛性反而坏掉**：

    | ρ₁ | legacy 自洽误差 | 只改 μ、不改 dnu |
    |---|---|---|
    | 1.0 | 3e-09 | 3e-09 |
    | 0.2 | 8e-10 | **3.7e-03 ~ 1e+01** |
    | 1e-2 | 7e-07 | **9e-02 ~ 1e+04** |
    | 1e-6 | 5e-08 | **1e+12** |

    36 槽同一设计：legacy 15 轮收敛 / 不改 dnu 60 轮不收敛 / 同步后 31 轮收敛。

    推导（``ν = 1/μ``，记 ``x = B/ρ₁``）::

        μ       = B/H(x) + μ₀(1−ρ₁)
        dμ/dB   = d/dB [ B/H(B/ρ₁) ] = (1 − x·H'(x)/H(x)) / H(x)
                  （真空项与 B 无关 ⇒ 不贡献；``dx/dB = 1/ρ₁`` 与
                    ``d/dB[H] = H'/ρ₁`` 相消，故**没有** ``1/ρ₁`` 因子）
        dν/dB   = −(dμ/dB) / μ²

    ``ρ₁ = 1`` 时 ``μ = B/H``、``dμ/dB = (1 − xH'/H)/H``，代入化简即
    ``H'/B − H/B²``——**与 legacy 原式恒等**（实测 rel 3e-09），故本式是
    legacy 的严格推广，不是另一种模型。

    ⚠ 本函数每次调用都会重算 ``SIMP_func(ρ)``。在 Newton 迭代里 ρ 不变，
    因此这是可避免的开销——``fast/material.py`` 把它提到了循环外。

    ★ 公式本体在 ``material_values``（**唯一实现**），本函数只负责
    取 ρ₁ 与曲线。
    """
    rho1 = simp.project(_clip_rho(rho))
    x = _clip_B(B) / rho1
    return material_values(B, rho1, curve.H(x), curve.dHdB(x), mu0)


def nonlinear_material_dnu_drho(
    B: np.ndarray, rho: np.ndarray, curve: BHCurve, simp: SIMPProjection,
    mu0: float = 4e-7 * np.pi,
) -> np.ndarray:
    """非线性材料的 ∂ν/∂ρ。

    对应 legacy ``nonlinear_material_dgamma_p_drho``（``MMA10:1196-1204``）。

    Returns
    -------
    dnu_drho : (N,) ∂ν/∂ρ  =  −(dμ/dρ) / μ²

    Notes
    -----
    ★★ **必须由加了真空项的 μ 求导（2026-10-17）**。legacy 推导的是
    ``ν = H(B/ρ₁)/B``（即 ``1/μ`` 且 ``μ=B/H``）那一支::

        ∂ν/∂ρ = − H'(B/ρ₁) / ρ₁² · dρ₁/dρ        ← legacy（μ=B/H）

    补了真空项之后 ``μ = B/H(B/ρ₁) + μ₀(1−ρ₁)``，``ν = 1/μ``，于是::

        dμ/dρ = [ B²·H'(x) / (ρ₁²·H(x)²)  −  μ₀ ] · dρ₁/dρ      x = B/ρ₁
        ∂ν/∂ρ = −(dμ/dρ) / μ²

    方括号里的 ``−μ₀`` 就是真空项的贡献（legacy 没有）。实测与中心差分
    吻合 rel ≤ 2e-9；若这里照抄 legacy，∂ν/∂ρ 会差到 **1e12** 量级
    （低密度处 μ 被抬高、ν 变小，导数量级完全不同）。

    ⚠ 本函数是 ``dγ_p/dρ`` 链的数学来源；生产路径的同名量在
    ``fast/material.CachedMaterial.nonlinear_dnu_drho``（逐位一致），而
    ``em_worker.legacy_dgamma_p_drho`` 才是 golden 对齐用的 legacy 原式。
    """
    rhoc = _clip_rho(rho)
    rho1 = simp.project(rhoc)
    x = _clip_B(B) / rho1
    return material_dnu_drho(B, rho1, curve.H(x), curve.dHdB(x), mu0,
                             simp.derivative(rhoc))


def linear_material_topo_opti(
    B: np.ndarray, rho: np.ndarray, mu0: float, mur_linear: float,
    simp: SIMPProjection,
) -> tuple[np.ndarray, np.ndarray]:
    """线性铁芯 + SIMP 密度。

    对应 legacy ``linear_material_topo_opti``（``MMA10:1207-1216``）。

    Returns
    -------
    mu : (N,) μ = μ₀·μ_r·ρ₁，其中 μ_r = ``mur_linear``（legacy 硬编码 2000）
    dnu_dB : (N,) 恒为 0（线性材料与 B 无关）

    Notes
    -----
    legacy 把 ``2000`` 直接写在代码里（``MMA10:1213``）。这里提升为参数
    ``mur_linear``，默认值由调用方给出——**不改变数值**，只是把魔数外提。

    ★ 公式本体在 ``material_values_linear``（**唯一实现**）。
    """
    mu, _ = material_values_linear(simp.project(_clip_rho(rho)), mu0, mur_linear)
    return mu, np.zeros_like(_clip_B(B))


def linear_material_dnu_drho(
    B: np.ndarray, rho: np.ndarray, mu0: float, mur_linear: float,
    simp: SIMPProjection,
) -> np.ndarray:
    """线性材料的 ∂ν/∂ρ。

    对应 legacy ``linear_material_dgamma_p_drho``（``MMA10:1219-1227``）。

    Notes
    -----
    ν = 1/(μ₀·μ_r·ρ₁) ⇒::

        ∂ν/∂ρ = − 1/(μ₀·μ_r) / ρ₁² · dρ₁/dρ

    ★ 公式本体在 ``material_dnu_drho_linear``（**唯一实现**）。
    """
    rhoc = _clip_rho(rho)
    rho1 = simp.project(rhoc)
    return material_dnu_drho_linear(rho1, mu0, mur_linear, simp.derivative(rhoc))
