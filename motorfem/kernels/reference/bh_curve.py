"""B-H 磁化曲线的构建、插值与积分（内核 1）。

来源：``legacy/pytorch/nonlinear_solver_energy_MMA10.py:906-936``

本内核**没有** `fast/` 对应实现——legacy 的做法与「直接照公式写」在数学上完全
同一，唯一差别是它把插值器建在模块导入期（缓存）。见上级 README 的
「ref/fast 的分层只在有真实差异时才建立」。缓存改由 `BHCurve` 对象承担，
从而满足「导入不得产生副作用」的硬性要求。

------------------------------------------------------------------------------
数学表述
------------------------------------------------------------------------------

1) 原始数据
   ``50CS470.csv`` 是 0.5 mm 厚硅钢片的磁化曲线，26 个 (H, B) 测量点，
   从 (0, 0) 单调递增到 (794130 A/m, 3.1059 T)。

2) 尾段外推（这一步是**物理建模**，不是数值技巧）
   实测曲线到 3.1 T 为止，但优化过程中局部磁密可能被推到更高。深饱和时
   材料已接近真空，故按真空磁导率 μ0 线性延长::

       H_{k+1} = H_k + ΔH_k
       B_{k+1} = B_k + ΔH_k · μ0          ΔH_k = linspace(1e6, 4e6, 5)_k

   注意 ΔH 是**累加的**（每一点在上一点基础上再加一个步长），不是等间距。

   ⚠ 这里的 ``linspace(1e6, 4e6, 5)`` 是 legacy 的魔法数（``MMA10:913``），
   没有任何推导记录。它的作用只是把曲线拖到足够远以覆盖优化中的过冲，
   本身不敏感。重构时**保留原值**以维持与 legacy 的位级一致，但标注为待考证。

3) 奇对称化
   磁化曲线是奇函数（H(-B) = -H(B)），但 CSV 只给了 B ≥ 0 的一半。
   把实测段关于原点镜像再拼接::

       H = concat([-H[::-1], H[1:]])
       B = concat([-B[::-1], B[1:]])

   ``H[1:]`` 去掉重复的原点，因此 (0, 0) 只出现一次。

4) 插值
   用 **PCHIP**（分段三次 Hermite，保单调）以 B 为自变量插值 H：

       H = f(B)

   PCHIP 的关键性质：**不会引入过冲**。这对磁化曲线是必需的——普通三次样条
   在数据拐点附近会振荡，导致 H(B) 非单调，进而使 Newton 迭代发散。

   之所以以 B 为自变量而不是 H，是因为求解器里 B 由 A 直接算出，需要反过来
   求 H（从而求 1/μ）。

5) 磁场能量密度
   ::

       W_m(B) = ∫₀^B H(b) db

   由 ``PchipInterpolator.antiderivative(1)`` 精确得到（PCHIP 的分段三次原函数
   仍是分段的解析式），**不是数值积分**。使用时要减去 ``W_m(0)`` 去掉积分常数。

   本模块提供它，但注意当前 MMA10 的主流程只用到 ``H`` 与 ``dH/dB``，
   ``W_m`` 是 ``nonlinear_solver_energy*.py`` 时代的遗产（见上级 README 的说明）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate as sp_interp

__all__ = [
    "TailExtension",
    "load_bh_table",
    "extend_bh_table",
    "BHCurve",
    "DEFAULT_CSV",
]

# 默认数据文件。**在仓库根目录下运行的相对路径**，且只在显式调用时才读取
# （不在此处求值，以满足「导入无副作用」）。
DEFAULT_CSV = Path("磁障设计/pytorch/50CS470.csv")


@dataclass(frozen=True)
class TailExtension:
    """尾段外推参数。

    Attributes
    ----------
    n_points : int
        追加的点数。
    H_step_lo, H_step_hi : float
        累加步长的上下限，对应 legacy 的 ``np.linspace(1e6, 4e6, 5)``。
        这是**魔法数**，原始推导未保留；保留原值以维持位级一致。
    """

    n_points: int = 5
    H_step_lo: float = 1e6
    H_step_hi: float = 4e6


def load_bh_table(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 (H, B) 数据表。

    Parameters
    ----------
    csv_path : 路径。文件为两列 CSV，首行为表头 ``H (A_per_meter), B (tesla)``。

    Returns
    -------
    H : (N,) 磁场强度 A/m，单调递增，首点为 0
    B : (N,) 磁通密度 T，单调递增，首点为 0

    Notes
    -----
    用 ``np.loadtxt`` 而非 legacy 的 ``pd.read_csv``——本内核不需要 DataFrame，
    去掉 pandas 依赖可减少重构后的包体积。文件含 UTF-8 BOM，用 ``utf-8-sig``
    读取以免首列名被污染（虽然本函数只取数值列，但显式编码符合项目规范）。
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, encoding="utf-8-sig")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{csv_path} 应有两列 (H, B)，实得形状 {data.shape}")
    H = np.ascontiguousarray(data[:, 0], dtype=float)
    B = np.ascontiguousarray(data[:, 1], dtype=float)
    return H, B


def extend_bh_table(
    H: np.ndarray,
    B: np.ndarray,
    mu0: float,
    tail: TailExtension = TailExtension(),
) -> tuple[np.ndarray, np.ndarray]:
    """按 μ0 斜率延长曲线尾段，并关于原点奇对称化。

    Parameters
    ----------
    H, B : (N,) 实测表，须单调递增且首点近似为原点
    mu0 : float 真空磁导率
    tail : TailExtension

    Returns
    -------
    H_ext, B_ext : (2N-1+2·n_tail,) 奇对称、单调递增的表

    Notes
    -----
    对应 legacy ``MMA10:913-917``::

        for H_ext in np.linspace(1e6, 4e6, 5):
            H = np.append(H, H[-1] + H_ext)
            B = np.append(B, B[-1] + H_ext * mu0)
        H = np.concatenate([-H[::-1], H[1:]])
        B = np.concatenate([-B[::-1], B[1:]])

    ⚠ **步长是累加的**：第 k 个追加点在**前一个追加点**的基础上再加 μ0·ΔH_k，
    而不是从最后一个实测点起算。这一点从 ``H[-1]`` 的写法可以确认，
    但很容易看错——本函数显式写成循环以消除歧义。
    """
    steps = np.linspace(tail.H_step_lo, tail.H_step_hi, tail.n_points)

    H = list(H)
    B = list(B)
    for step in steps:
        H.append(H[-1] + step)
        B.append(B[-1] + step * mu0)
    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)

    H_sym = np.concatenate([-H[::-1], H[1:]])
    B_sym = np.concatenate([-B[::-1], B[1:]])

    if not np.all(np.diff(H_sym) > 0):
        raise ValueError("延长+对称化后 H 不再严格单调递增，曲线不可用")
    return H_sym, B_sym


class BHCurve:
    """以 B 为自变量的磁化曲线 ``H(B)`` 及其导数、原函数。

    本类承担 legacy 里「模块导入期构建插值器」的角色，但改为**显式构造**，
    以保证导入本包不产生副作用。

    Parameters
    ----------
    H_ext, B_ext : 已延长并对称化的表（``extend_bh_table`` 的输出）

    Examples
    --------
    >>> curve = BHCurve.from_csv("磁障设计/pytorch/50CS470.csv", mu0=4e-7*np.pi)
    >>> float(curve.H(np.array([1.5]))[0])       # 1.5 T 处的 H
    """

    def __init__(self, H_ext: np.ndarray, B_ext: np.ndarray):
        self._H = H_ext
        self._B = B_ext
        # PCHIP 保单调、不过冲——对磁化曲线是必需的，普通三次样条会在
        # 拐点附近振荡出非单调的 H(B)，直接导致 Newton 迭代发散。
        self._H_of_B = sp_interp.PchipInterpolator(
            B_ext, H_ext, extrapolate=True
        )
        self._Wm_of_B = self._H_of_B.antiderivative(1)
        self._Wm0 = float(self._Wm_of_B(0.0))

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path = DEFAULT_CSV,
        mu0: float = 4e-7 * np.pi,
        tail: TailExtension = TailExtension(),
    ) -> "BHCurve":
        """从 CSV 构建。数据读取与插值器构建都发生在此调用，不在导入期。"""
        H, B = load_bh_table(csv_path)
        H_ext, B_ext = extend_bh_table(H, B, mu0, tail)
        return cls(H_ext, B_ext)

    # ---- 曲线本身 ----
    #
    # 本类**不做任何裁剪**。legacy 在 ``nonlinear_material`` 里对 B 做了
    # ``np.clip(_B, 1e-20, inf)``（MMA10:925），但那是为了 ``mu = B/H(B)`` 与
    # ``dgamma = H'(B)/B − H(B)/B²`` 里的除法，属于**材料内核**的守卫，
    # 不属于曲线本身。
    #
    # 把裁剪放在这里会造成两个后果：
    #   1. 破坏奇对称——H(-0.5) 会被夹成 H(1e-20)，而磁化曲线本应满足
    #      H(-B) = -H(B)。legacy 之所以没暴露这个问题，是因为它只用
    #      B ≥ 0（由 sqrt 得到的幅值）调用本曲线。
    #   2. 让「曲线」这个纯数学对象带上调用方的假设，测试无法独立验证。
    #
    # 故裁剪统一放在内核 3（材料）里，并在那里注明它对应的除法。

    def H(self, B: np.ndarray) -> np.ndarray:
        """``H(B)``，单位 A/m。B 为实数（可为负）；不做裁剪。"""
        return self._H_of_B(np.asarray(B, dtype=float))

    def dHdB(self, B: np.ndarray) -> np.ndarray:
        """``dH/dB``（PCHIP 的解析一阶导，不是差分）。"""
        return self._H_of_B(np.asarray(B, dtype=float), 1)

    def Wm(self, B: np.ndarray) -> np.ndarray:
        """磁场能量密度 ``W_m(B) = ∫₀^B H db``，单位 J/m³。

        已减去 ``W_m(0)`` 去掉积分常数。由 PCHIP 的解析原函数给出，非数值积分。
        因 H 是奇函数，``W_m`` 是偶函数。

        ⚠ **精度注意（实测结论，勿按"抵消"直觉去"修复"）**

        原函数的定义域起点是最外侧的负尾段 ``B ≈ -18.81``，故积分常数
        ``W_m(0) ≈ -1.11e8``。本方法用两个 ≈1.1e8 的数相减得到结果，
        因此 ``W_m`` 的**绝对精度下限约 1.2e-8**（float64 在 1.1e8 处的间距），
        与 B 无关。

        这个抵消**不是**缺陷：Wm 在低 B 处约 0.5、高 B 处约 3e5，
        绝对误差 1.2e-8 对应相对误差 2e-8 ~ 4e-14，可忽略。
        实测与高精度数值积分的相对偏差：B=0.05 时 1.5e-8，B=3.0 时 3.7e-10。

        **反面尝试（已实测，勿重复）**：改为「只在非负半支上建插值器、从 0 起积分」
        看似能消掉抵消，实际反而差 7 个数量级（B=0.05 时相对误差 1.3e-1）。
        原因是**丢掉负半支会改变 PCHIP 在 B=0 处的端点条件**——插值器本身变了，
        不再是同一个函数。要保持偶数性，必须保留整条奇对称曲线。
        """
        return self._Wm_of_B(np.asarray(B, dtype=float)) - self._Wm0

    def dWmdB(self, B: np.ndarray) -> np.ndarray:
        """``dW_m/dB``，**解析**求导（不经过差分），恒等于 ``H(B)``。

        单独暴露它有两个用途：
          1. 供上游按需取用，避免用有限差分（见下）；
          2. 作为测试判据——它能查出「原函数被错误地对另一个插值器求积分」
             这类错误（此时 ``dW_m/dB ≠ H``），而只比较 ``Wm`` 的数值查不出。

        ⚠ 不要用有限差分验证 ``W_m`` 的导数：``W_m`` 的绝对精度下限是 1.2e-8
        （见 ``Wm`` 的说明），而 eps=1e-6 的中心差分分子只有 ~4e-5，
        会把底噪放大到 ~3e-4 的相对误差，得出"实现有错"的错误结论。
        """
        return self._Wm_of_B(np.asarray(B, dtype=float), 1)

    # ---- 只读属性，便于测试与诊断 ----

    @property
    def table(self) -> tuple[np.ndarray, np.ndarray]:
        """延长并对称化后的 (H, B) 表。"""
        return self._H.copy(), self._B.copy()

    @property
    def Wm_precision_floor(self) -> float:
        """``W_m`` 的**绝对**精度下限（单位 J/m³）。

        来源：``W_m`` 由两个 ≈1.1e8 的数相减得到，误差被量化在积分常数的
        float64 间距（ULP）上。实测偏差是 ULP 的 1~7 倍（本数据集 ULP =
        1.490e-8），故取 8 ULP 作为安全上界。

        调用方在比较 ``W_m`` 时应使用本值作为容差，**不要**用有限差分去验
        它的导数（见 ``dWmdB`` 的说明）。
        """
        return 8.0 * abs(float(np.spacing(self._Wm0)))

    @property
    def integration_constant(self) -> float:
        """原函数在 B=0 处的值（即被减掉的积分常数），供诊断用。

        它之所以很大（≈-1.1e8），是因为 PCHIP 的定义域起点在最外侧的负尾段
        ``B ≈ -18.81``，原函数从那里开始累积。这正是 ``Wm`` 精度下限的来源。
        """
        return self._Wm0

    @property
    def n_measured(self) -> int:
        """实测点数（对称化 + 延长之前）。"""
        return int(np.sum(self._B > 0) )  # 正向一半的点数

    def __repr__(self) -> str:
        return (
            f"BHCurve(n_points={self._H.size}, "
            f"B=[{self._B[0]:.3g}, {self._B[-1]:.3g}] T, "
            f"H=[{self._H[0]:.3g}, {self._H[-1]:.3g}] A/m)"
        )
