"""非线性静磁场的残差 / Jacobian / Newton（内核 9+10+11）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）

    ``field_error_vectorized``  ``:2222-2257``   ← 内核 9（残差）+ 内核 10（Jacobian）
    ``solve_field_vectorized``  ``:2380-2406``   ← 内核 11（Newton + 分段松弛）

数学出处：``磁障设计/同步磁阻电机磁障结构分析.md``
  - 「使用Newton-Raphson方法进行求解」
  - 「N-R 的 Jacobi 矩阵推导」
  - 「计算步骤」

与样板内核 ``magnetostatic.py`` 的关系
``magnetostatic.py`` 是**通用骨架**（一个网格 + 一个材料 callable，无 ``T``、无掩码），
演示了残差 / Jacobian / Newton 的数学。本模块是 **legacy 保真版**：把 legacy 真正在跑的
那套东西——四个材料掩码、单元变换矩阵 ``T``、驱动行装置——用同一套朴素写法表达出来。
**两者数学同一**（``magnetostatic.jacobian`` 的 docstring 已经把 legacy 的
``gswap1/gswap2`` 认成秩 1 外积），差别只在输入的组织方式。样板内核一行不动。

------------------------------------------------------------------------------
一、残差（内核 9）
------------------------------------------------------------------------------

1) 磁密。单元的局部矢量磁位是 ``A^(e) = T_e · A_active``（``T_e`` 由内核 8 给出）::

       Bx_e = Bxcoef_e · A^(e)          By_e = Bycoef_e · A^(e)
       B_e  = sqrt(Bx_e² + By_e²)      然后 clip(min=1e-20, max=np.inf)

   ``clip`` 的下界 1e-20 是必要的：无源区域 ``B`` 可以恒为 0，而下游的
   ``1/B``（残差里的 ``ν``、Jacobian 里的饱和系数）会除零。legacy 在 ``:2231``
   有同样的 clip。**本模块返回裁剪后的 B**——下游（磁链、``dγ/dρ`` 链）消费的都是它。

2) 材料：四类单元的**算术加权**合成（legacy ``:2236-2237``）::

       mu     = is_vacuum·mu0 + is_SIMP·mu_SIMP + is_non_SIMP·mu_non_SIMP + is_linear_iron·(2000·mu0)
       dgamma = is_vacuum·0   + is_SIMP·dgamma_SIMP + is_non_SIMP·dgamma_non_SIMP + is_linear_iron·0

   ⚠ **这是加权而不是选择**，四条都参与浮点加法：
   ``is_vacuum`` 等是 **float64 的 0/1 数组**（内核 8 的 ``np.zeros(N_element)``），
   不是索引数组。因此 ``0 * nan = nan``——某一类的值若为 ``inf/nan``，结果**会被污染成 nan**。
   ⇒ 严禁改写成 ``np.where`` 或布尔索引：那既改了 nan 语义（改行为），
   也改了四项加法的结合顺序（改末位）。同族的还有 ``1 - F_is_driven``（``:2244``）。

3) 刚度重装配与残差::

       nu_e = 1 / mu_e
       F    = (Σ_e nu_e · T_eᵀ K0_e T_e) A  −  R
       F    ← 驱动行替换（内核 6 的 ``driving_row_residual``）

   ``K0_e`` 是内核 4 的**几何**刚度（不含 ν）。``Σ_e nu_e TᵀK0T`` 就是 legacy 的
   ``csr((K0_data * (Kconst_mat @ (1/mu)), (K0_row, K0_col)))``：位置不变、值每次重算。

   ``Δ > 0`` 的断言在内核 4 的 ``element_geometry`` 里（单元定向的唯一检查点）。

------------------------------------------------------------------------------
二、Jacobian（内核 10）—— 全项目最难读的一处
------------------------------------------------------------------------------

从 ``B² = ((b·A^(e))² + (c·A^(e))²)/(4Δ²)`` 出发可以推出（推导见
``magnetostatic.py`` 的模块 docstring 第 5 节）::

       ∂B_e/∂A = g_e / (Δ_e · B_e)          其中 g_e = K0_e A^(e)

于是在活跃自由度空间里::

       J = Σ_e nu_e · (T_eᵀ K0_e T_e)                 ← 线性部分（= 残差里的 K）
         + Σ_e [ dnu_e/dB / (Δ_e · B_e) ] · g0_e ⊗ g0_e
           └──────────── 饱和项（秩 1 外积）────────────┘
       其中 g0_e = (T_eᵀ K0_e T_e) · A_active

**关键观察：外积的两个因子是同一个向量。** 这正是 legacy
（``:2246-2252``）::

       g0i_vec    = g0i_mat @ _A_node
       const_vec  = gconst_mat @ (1 / B / Aelem * dgamma)
       gi_vec     = g0i_vec * const_vec
       Jabobi2_data = (gswap1_mat @ g0i_vec) * (gswap2_mat @ gi_vec)

里 ``g0i_vec`` 出现两次、以及 ``gi_vec = g0i_vec * const_vec`` 的原因：
``gswap1`` / ``gswap2`` 只是**对同一个扩展向量做「取第 i 个 / 取第 j 个」两种选择**，
逐元素相乘即得 9 个分量 ``g0[i]·g0[j]``。那两个矩阵唯一的作用是**把三阶张量
(单元, 局部 i, 局部 j) 展平成稀疏矩阵** —— 数学上就是一个 ``np.outer(g0, g0)``。

⚠ **中缀细节**：``const_vec`` 的算式是 ``1 / B / Aelem * dgamma``（**左结合**）。
写成 ``dgamma / (B * Aelem)`` 浮点结果不同（见 ``CLAUDE.md`` §3 的算式形状纪律）。

------------------------------------------------------------------------------
三、Newton + 分段松弛（内核 11）
------------------------------------------------------------------------------

逐字对应 legacy ``:2380-2406``::

       A ← A0 的副本（A0=None 则零向量）
       每轮： (F, B, J) ← 残差/Jacobian
              err = F·F                        ← ★ 平方范数，不是范数
              打印 'epoch:%d, error: %g' % (轮次+1, err)   ← 轮次 1-based
              err < solve_abs_err 则 break      ← 判停在 spsolve 之前
              dA ← spsolve(J, F)
              步长： i_epoch < iter_relaxation ? 1 : relaxation_ratio   ← ★ 0 基

三处必须留意的地方：

1. **``err = F·F`` 是平方范数**（``:2393``）。阈值 ``solve_abs_err`` 因此是
   「平方残差的绝对阈值」，与常见的 L2 判据差一个平方。legacy 的默认值 ``1e-6``
   在 MMA10 里**从未被实际用到**，生产调用传的是 ``1e-3``（``:2921`` / ``:2980``）。
   默认值与调用值不同**是接口的一部分**，不要「统一」。
2. **判停发生在 ``spsolve`` 之前**（``:2396-2399``）⇒ 返回的 ``(B, J)`` 是**收敛那一轮**
   的残差产物，不是最后一次求解的。本模块用 aux 透传保证这一点。
3. **松弛判据是 0 基**（``if i_epoch < iter_relaxation``，``:2400``）：数的是
   **迭代**而不是更新。换算成 1 基的迭代序号即「第 1..iter_relaxation 次取 α=1」。

⚠ 样板内核 ``magnetostatic.newton_solve`` 用的是**相对**判据（并注释解释了为何
固定绝对容差在本问题尺度下没有意义）。**本模块不采用它**——内核 11 的职责是
逐位复现 legacy 的绝对平方判据，判据的改进属「改行为」，留给基线冻结之后。

### ★ 2026-10-17 实测：牛顿**跑满 max_iter 仍不收敛**（已定位为 D13）

对 h200（μr=33，``_bh_h200.csv``）的一个设计，实测：

- **每次调用都跑满 100 轮**（``max_iter``），**从未触发 break** ⇒ ``err`` 始终
  ``>= solve_abs_err``；残差**冻结**在 2.3e-3（40 轮零变化），甚至**首轮之后暴涨**
  （线性材料下 3.84e4 → 6.35e12）。
- **任何初值都不收敛**：``A0=0``、``A0=常数 1e-3``（结果与零解**逐位相同** —— 因为
  ``∇(常数)=0``，与零解在梯度算子上等价）、``A0=随机 1e-4/1e-3/1e-2``（首轮 err
  高达 1e19~1e23）全部跑满 100 轮。
- **同一个 ρ 的转矩随调用历史漂移**：连算 6 次的 T = 1.51692 / 1.51842 / 1.51230 /
  1.51272 / 1.51355 / 1.51336（**极差 0.0061**）；曾观测 0.03 量级。

**原因（D13，见 ``docs/重构设计文档.md`` §12）**：不在本模块，而在材料层 ——
SIMP 支的 ``μ = B/H(B/ρ₁)`` 在 ``ρ₁→0``（下界 ``min_SIMP=1e-6``）时把自变量
``B/ρ₁`` 放大到曲线外推区，``μ`` 塌陷到 **2.68e-12**（比 ``μ₀`` 还小 47 万倍）；
``μ`` 在设计单元上跨 10 个数量级 ⇒ ``1/μ`` 动态范围 **3.1e9 倍** ⇒
``K``/Jacobian 病态。

**推论**：``best_T`` 这类「历史最大目标值」的统计里混有这种**历史依赖噪声**。
**判「某档优化是否有效」前，先确认牛顿确实收敛**（数 ``history`` 长度是否等于
``max_iter``）。

已知与 ``solve_abs_err`` 相关的其它坑：全链 FD 的差分步长 ε 会被 ``solve_abs_err``
噪声淹没（``em_worker`` 第 334 行附近有注）。

------------------------------------------------------------------------------
四、与本层相邻的接口
------------------------------------------------------------------------------

- **上游**：内核 8 的 ``transformation_matrix``（逐单元 ``T``）与 ``ElementOperators``
  （四个掩码）；内核 4 的 ``K0elem``/``Bxcoef``/``Bycoef``/``area``；
  内核 6 的 ``DrivingRows`` + ``driving_row_residual``/``driving_row_jacobian``；
  内核 3 的材料函数。
- **下游**：内核 12~17 全部消费本模块的 ``(F, B, J)`` 三元组。返回的 ``J`` 会被
  ``spsolve(J.T, …)`` 消费（内核 13/14 的伴随解）⇒ **同一个对象**，不重建、不拷贝。
- **内核 15**（冻结视在磁导率的磁链与转矩，``:2300-2333``）：实测逐行 diff 显示它与
  内核 9/10 **只差「B 的来源」与返回顺序**（其余每一行逐字相同）。所以本模块把缝
  切在两个具名函数上：``field_material``（B → μ, dν/dB）与 ``field_system``
  （A, μ, dν/dB, B → F, J）。内核 15 = ``B_frozen`` → ``field_material`` →
  ``field_system``，再包一层返回 ``(F, J, mu)``。**不用 callable、不用布尔开关**：
  前者把缝藏进间接调用，后者在一个函数里造出两条路径。

------------------------------------------------------------------------------
五、分层：为什么 ``fast`` 的签名与这里不同（**bundle 型签名**）
------------------------------------------------------------------------------

本模块的输入是**内核 8 的入参**（``k0elem``/``bxcoef``/``bycoef`` + ``T``），
``fast/newton.py`` 的输入是内核 8 的**出参**（``ElementOperators`` 18 项 / 22 元组）。
这是**不可约的分叉**，理由不是「懒得凑」：

``ElementOperators`` **只装结果**（``k0_data`` 三元组、``bxcoef_mat = Bxcoef@T``、
``g0i_mat``、``gswap1/2``…），里面**没有**每单元 ``T``、没有原始 ``bxcoef``、
没有 ``k0elem``。本模块若改吃它，就只能**从向量化形式反推数学** —— 正是
``motorfem/kernels/README.md`` 明令禁止的「禁止从 ``fast/`` 反推」。
反过来 ``fast`` 必须吃 ``ElementOperators``，否则每次残差调用都要重建位置矩阵，
而「位置存一次、值每次重算」正是 ``cProfileReport.txt`` 那 61% 的历史性能结论。

**一句话**：本模块的输入 = ``element_operators()`` 的**入参**；``fast`` 的输入 = 它的**出参**。
两层的其余部分（参数个数、位置顺序、关键字尾）保持同构，``reference/magnetostatic.py``
与本模块共用 ``newton_iterations`` 这一个循环实现（Newton 循环没有可向量化的部分）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from ... import solvers
from . import boundaries as ref_bnd
from .assembly import ElementTransform, TopologyContext, transformation_matrix
from .bh_curve import BHCurve
from .element_matrices import expand_K0
from .material import nonlinear_material, nonlinear_material_topo_opti
from .simp import SIMPProjection

__all__ = [
    "MaterialMasks",
    "FieldMaterial",
    "FieldGeometry",
    "FieldAux",
    "field_geometry",
    "blend_class_values",
    "element_field_material",
    "flux_density",
    "field_system",
    "residual",
    "jacobian",
    "newton_iterations",
    "newton_solve",
]


# =============================================================================
# 输入装包
# =============================================================================

@dataclass(frozen=True)
class MaterialMasks:
    """单元分类掩码。四个字段**互斥且穷尽**，取值是 ``float64`` 的 0/1 **数组**。

    对应 legacy ``MMA10:1888-1891`` 的四条 ``np.zeros(N_element)``，由内核 8 的
    ``ElementOperators`` 逐个单元填出（``:2003-2012``）::

        if i_elem in element_iron:
            if not use_nonlinear_iron:   is_linear_iron[i] = 1
            elif i_elem in element_topo_opti: is_SIMP[i] = 1
            else:                        is_non_SIMP[i] = 1
        else:                            is_vaccum[i] = 1

    ⚠ **它们是数组而不是索引集合**，因为下游要把四项**算术加权**相加
    （见模块 docstring 第一节第 2 条）。别「顺手」改成布尔索引。
    """

    is_vacuum: np.ndarray
    is_simp: np.ndarray
    is_non_simp: np.ndarray
    is_linear_iron: np.ndarray

    @classmethod
    def from_operators(cls, ops) -> "MaterialMasks":
        """由内核 8 的 ``ElementOperators`` 取出（字段名大小写不同，此处一次性对齐）。"""
        return cls(
            is_vacuum=np.asarray(ops.is_vaccum),
            is_simp=np.asarray(ops.is_simp),
            is_non_simp=np.asarray(ops.is_non_simp),
            is_linear_iron=np.asarray(ops.is_linear_iron),
        )

    @property
    def n_elem(self) -> int:
        return int(self.is_vacuum.shape[0])

    def partition_error(self) -> float:
        """``|四类之和 − 1|`` 的最大值。合法掩码下应为 0（浮点精确）。"""
        total = self.is_vacuum + self.is_simp + self.is_non_simp + self.is_linear_iron
        return float(np.abs(total - 1.0).max()) if total.size else 0.0


@dataclass(frozen=True)
class FieldMaterial:
    """材料模型的参数束（内核 3 的两个对象 + 两个常数）。

    Attributes
    ----------
    curve : BHCurve
        B-H 曲线（内核 1）。
    simp : SIMPProjection
        tanh 投影（内核 2）。只有 ``is_simp`` 那一类用到它。
    mu0 : float
        真空磁导率。legacy 写 ``4 * np.pi * 1e-7``（``MMA10:80``）——
        **已实测与 ``4e-7 * np.pi`` 逐位相同**，所以沿用 ``CachedMaterial`` 的默认值
        是安全的。不要把它「修正」成另一种拼法。
    mur_linear : float
        线性铁磁的相对磁导率，legacy 硬编码 ``2000``。

    Notes
    -----
    ⚠ **待核对（已登记进 ``docs/重构设计文档.md`` §11，本批不改）**：
    四类合成里的线性铁磁项是 ``is_linear_iron * (2000 * mu0)``（``:2236``），
    **与 ρ 无关**；而内核 3 的 ``linear_material_topo_opti`` 给的是
    ``μ0·2000·ρ1``，**与 ρ 有关**。两者是不同的模型，本模块照搬 legacy 的那一个。
    """

    curve: BHCurve
    simp: SIMPProjection
    mu0: float = 4e-7 * np.pi
    mur_linear: float = 2000.0
    #: ★ 2026-10-17：``use_nonlinear_iron=False`` 时 SIMP 支是否走**线性**
    #: 材料 ``μ = μ₀·μ_r·ρ₁``（``dν/dB ≡ 0``）。默认 False = 走非线性曲线
    #: （legacy 口径，逐位不变）。
    linear_simp: bool = False


@dataclass(frozen=True)
class FieldGeometry:
    """本模块的输入束：内核 8 的**入参**（见模块 docstring 第五节）。

    Attributes
    ----------
    n_node : int
        节点数。
    k0 : (N_elem, 3, 3) ndarray
        内核 4 的**几何**单元刚度（不含 ν），由 ``expand_K0`` 从 ``(N, 6)`` 展开。
    bxcoef, bycoef : (N_elem, 3) ndarray
        内核 4 的磁密系数（**未经 T**）。``Bx_e = bxcoef_e · (T_e A_active)``。
    aelem : (N_elem,) ndarray
        单元面积 Δ。**已实测与 legacy 的 ``Aelem`` 逐位相同**
        （``Δ = ½(b_i c_j − b_j c_i)`` 的 ``*0.5`` 与 ``/2.0`` 都是 2 的幂缩放）。
        ⚠ 反向不成立：``aelem ≠ R0 * 3``（差 1 ULP），别从 ``R0`` 反推。
    masks : MaterialMasks
    transforms : tuple[ElementTransform, ...]
        逐单元的 ``T`` 与活跃自由度 ``index``（内核 8 的 ``transformation_matrix``）。
        **持了它之后本模块完全不需要 ``ctx``/``rotating``** —— 少两个参数、
        少一条与内核 6/7 的耦合。
    """

    n_node: int
    k0: np.ndarray
    bxcoef: np.ndarray
    bycoef: np.ndarray
    aelem: np.ndarray
    masks: MaterialMasks
    transforms: Tuple[ElementTransform, ...]

    @property
    def n_elem(self) -> int:
        return len(self.transforms)


def field_geometry(
    ctx: TopologyContext,
    rotating,
    *,
    n_node: int,
    k0elem: np.ndarray,
    bxcoef: np.ndarray,
    bycoef: np.ndarray,
    aelem: np.ndarray,
    masks: MaterialMasks,
) -> FieldGeometry:
    """构造 ``FieldGeometry``：逐单元取一次 ``T``，之后反复复用。

    Parameters
    ----------
    ctx, rotating :
        内核 6/7 的拓扑与旋转边界（只在本函数里用于取 ``T``）。
    n_node : int
    k0elem : (N_elem, 6) ndarray
        内核 4 的 ``ElectromagneticElements.K0``。
    bxcoef, bycoef : (N_elem, 3) ndarray
    aelem : (N_elem,) ndarray
        内核 4 的 ``ElectromagneticElements.area``。
    masks : MaterialMasks

    Notes
    -----
    ``T`` 只依赖网格与转子位置，与 ``A``、``ρ`` 无关 —— 这与 legacy 把位置信息
    固化在 ``K_vectorized`` 里的动机完全一致（``同步磁阻电机磁障结构分析.md``
    「静磁场计算的向量化数值实现」：*元素位置不变，因此只需要管理数值即可*）。
    """
    n_elem = int(np.asarray(k0elem).shape[0])
    transforms = tuple(
        transformation_matrix(e, ctx, rotating) for e in range(n_elem)
    )
    return FieldGeometry(
        n_node=int(n_node),
        k0=expand_K0(np.asarray(k0elem)),
        bxcoef=np.asarray(bxcoef, dtype=float),
        bycoef=np.asarray(bycoef, dtype=float),
        aelem=np.asarray(aelem, dtype=float),
        masks=masks,
        transforms=transforms,
    )


@dataclass(frozen=True)
class FieldAux:
    """残差算出的、Jacobian 要复用的单元级量。全部形状 ``(N_elem,)``。

    Attributes
    ----------
    B : (N_elem,) ndarray
        **裁剪后**的 ``|B|``（下界 1e-20）。
    mu : (N_elem,) ndarray
        四掩码加权后的磁导率 μ。
    dgamma : (N_elem,) ndarray
        ``∂ν/∂B``（ν = 1/μ）。legacy 叫它 ``dgamma``，此处沿用 legacy 名以便对照。
    """

    B: np.ndarray
    mu: np.ndarray
    dgamma: np.ndarray


# =============================================================================
# 单元材料：四掩码加权（内核 9 的公式，**两层共用同一份**）
# =============================================================================

def blend_class_values(
    mu_simp: np.ndarray,
    dgamma_simp: np.ndarray,
    mu_non_simp: np.ndarray,
    dgamma_non_simp: np.ndarray,
    masks: MaterialMasks,
    mu0: float,
    mur_linear: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """四类单元的 μ 与 ∂ν/∂B 的**算术加权**合成。

    逐字对应 legacy ``MMA10:2236-2237``::

        mu = (is_vaccum * mu0 + is_SIMP * mu_SIMP + is_non_SIMP * mu_non_SIMP
              + is_linear_iron * (2000 * mu0))
        dgamma = (is_vaccum * 0 + is_SIMP * dgamma_SIMP + is_non_SIMP * dgamma_non_SIMP
                  + is_linear_iron * 0)

    Parameters
    ----------
    mu_simp, dgamma_simp : (N_elem,)
        非线性 + SIMP 密度类（``nonlinear_material_topo_opti``）。**对全部单元求值**，
        不只对 ``is_simp`` 为 1 的单元 —— legacy 就是这么算的。
    mu_non_simp, dgamma_non_simp : (N_elem,)
        非线性纯铁芯类（``nonlinear_material``）。同样对全部单元求值。
    masks : MaterialMasks
    mu0, mur_linear : float

    Notes
    -----
    ⚠ **本函数是 ``reference`` 必须与 ``fast`` 共用代码的第二处**（第一处是内核 8 的
    ``_active_index``）。原因是它是全内核唯一一处**既对浮点结合序敏感、又对 nan 语义
    敏感**的算式：

    - 四项按书写顺序相加，改顺序即改末位；
    - 是**加权**不是选择 ⇒ ``0 * nan = nan``。改写成 ``np.where`` 会同时改掉这两件事。

    共用代码意味着 parity 测试抓不到它自身的错，所以必须有**独立的解析测试**兜底：
    ``tests/kernels/test_newton.py`` 里「掩码逐类精确」与「``0*nan`` 语义」两条。
    """
    is_vacuum = masks.is_vacuum
    is_simp = masks.is_simp
    is_non_simp = masks.is_non_simp
    is_linear_iron = masks.is_linear_iron

    mu = (is_vacuum * mu0
          + is_simp * mu_simp
          + is_non_simp * mu_non_simp
          + is_linear_iron * (mur_linear * mu0))
    dgamma = (is_vacuum * 0
              + is_simp * dgamma_simp
              + is_non_simp * dgamma_non_simp
              + is_linear_iron * 0)
    return mu, dgamma


def element_field_material(
    B: np.ndarray, rho: np.ndarray, masks: MaterialMasks, material: FieldMaterial
) -> Tuple[np.ndarray, np.ndarray]:
    """由 ``B`` 与 ``ρ`` 算单元级的 μ 与 ∂ν/∂B。对应 legacy ``:2233-2237``。

    Parameters
    ----------
    B : (N_elem,) 裁剪后的磁密幅值
    rho : (N_elem,) 设计变量（**未裁剪**——裁剪在 ``nonlinear_material_topo_opti``
        内部做，与 legacy 同。残差层不要再裁一次）
    masks : MaterialMasks
    material : FieldMaterial

    Returns
    -------
    mu, dgamma : (N_elem,) ``mu`` 是磁导率，``dgamma`` 是 ``∂ν/∂B``

    Notes
    -----
    两类非线性材料都**对全部单元**求值（legacy 如此），再由 ``blend_class_values``
    加权取用。这是有意的浪费：省掉它是「改结构」，不是「搬家」。
    """
    mu_simp, dgamma_simp = nonlinear_material_topo_opti(
        B, rho, material.curve, material.simp
    )
    mu_non_simp, dgamma_non_simp = nonlinear_material(B, material.curve)
    return blend_class_values(
        mu_simp, dgamma_simp, mu_non_simp, dgamma_non_simp,
        masks, material.mu0, material.mur_linear,
    )


# =============================================================================
# 单元量与残差（内核 9）
# =============================================================================

def flux_density(A: np.ndarray, field: FieldGeometry) -> np.ndarray:
    """逐单元磁密幅值 ``|B|``（**裁剪后**），形状 ``(N_elem,)``。对应 legacy ``:2229-2231``。

    ``Bx = Bxcoef·T·A_active``，``By = Bycoef·T·A_active``，``B = sqrt(Bx²+By²)``，
    再 ``clip(min=1e-20, max=np.inf)``。

    Notes
    -----
    ``clip`` 的必要性见模块 docstring 第一节第 1 条。**返回的是裁剪后的值**：
    legacy 的 ``B_elem`` 就是这样一路传给磁链、转矩与 ``dγ/dρ`` 链的。
    """
    B = np.empty(field.n_elem)
    for e in range(field.n_elem):
        tr = field.transforms[e]
        A_active = A[list(tr.index)]
        Bx = field.bxcoef[e] @ tr.matrix @ A_active
        By = field.bycoef[e] @ tr.matrix @ A_active
        B[e] = np.sqrt(Bx ** 2 + By ** 2)
    return np.clip(B, 1e-20, np.inf)


def field_system(
    A: np.ndarray,
    field: FieldGeometry,
    mu: np.ndarray,
    dgamma: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
) -> Tuple[np.ndarray, sp_sparse.csr_matrix]:
    """由**已算好的** μ / dν/dB / B 组装 ``(F, J)``。对应 legacy ``:2239-2255``。

    这是模块 docstring 第四节说的那条「缝」：内核 9 与内核 15（冻结视在磁导率）
    的**唯一**差异是 ``B`` 从哪来，所以把它作为参数而不是在函数里算。
    两个内核共用本函数，谁都不需要抄第二遍。

    Parameters
    ----------
    A : (N_node,) 活跃自由度上的矢量磁位
    field : FieldGeometry
    mu, dgamma, B : (N_elem,) 由 ``element_field_material`` 产出
    R : (N_node,) 源项
    driving : DrivingRows

    Returns
    -------
    F : (N_node,) 残差（已做驱动行替换）
    J : (N_node, N_node) csr Jacobian（已做驱动行替换）

    Notes
    -----
    数学见模块 docstring 第一、二节。两处照搬 legacy 的写法：

    - 饱和系数 ``1 / B / aelem * dgamma``（**左结合**）；
    - 线性部分是**同一个** ``TᵀK0T`` 既进 ``F`` 又进 ``J``（legacy 的 ``K_data`` 一次算、
      两处用），所以本函数把 ``(TᵀK0T)`` 算一次存进局部表。
    """
    n_node = field.n_node
    n_elem = field.n_elem

    rows, cols, vals = [], [], []
    F = np.zeros(n_node)

    for e in range(n_elem):
        tr = field.transforms[e]
        index = list(tr.index)
        T = tr.matrix

        # 活跃自由度空间里的几何单元刚度： (Tᵀ K0_e T)
        k0_e_active = T.T @ (field.k0[e] @ T)

        A_active = A[index]
        # 残差贡献： ν_e · (Tᵀ K0_e T) · A_active
        F[index] += (1.0 / mu[e]) * (k0_e_active @ A_active)

        # 线性部分（= 残差里的 K）
        block = (1.0 / mu[e]) * k0_e_active

        # 饱和项： coef_e · g0_e ⊗ g0_e，其中 g0_e = (Tᵀ K0_e T) A_active（**不含 ν**）
        g0_e = k0_e_active @ A_active
        coef_e = 1 / B[e] / field.aelem[e] * dgamma[e]
        sat = coef_e * np.outer(g0_e, g0_e)

        rows.extend(np.repeat(index, len(index)))
        cols.extend(np.tile(index, len(index)))
        vals.extend((block + sat).ravel())

    F = F - R
    F = ref_bnd.driving_row_residual(F, A, driving)

    J_raw = sp_sparse.coo_matrix(
        (np.asarray(vals, dtype=float), (np.asarray(rows, dtype=np.int64),
                                         np.asarray(cols, dtype=np.int64))),
        shape=(n_node, n_node),
    )
    J = ref_bnd.driving_row_jacobian(J_raw, driving)
    return F, J.tocsr()


def residual(
    A: np.ndarray,
    rho: np.ndarray,
    field: FieldGeometry,
    material: FieldMaterial,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
) -> Tuple[np.ndarray, FieldAux]:
    """残差 ``F(A) = K(A)A − R``（已做驱动行替换）+ Jacobian 要复用的单元量。

    参数顺序对齐 ``fast.field_error_vectorized`` 的 ``(A_node, rho, K_vectorized, R)``：
    ``A`` 在最前、``rho`` 紧随其后，``driving`` 是 ``reference`` 侧额外显式的一件。

    Parameters
    ----------
    A : (N_node,) 当前矢量磁位
    rho : (N_elem,) 设计变量（**未裁剪**，见 ``element_field_material``）
    field : FieldGeometry
    material : FieldMaterial
    R : (N_node,) 源项（电流 + 永磁体）
    driving : DrivingRows

    Returns
    -------
    F : (N_node,) 残差
    aux : FieldAux
        ``(B, mu, dgamma)`` —— 供 ``jacobian`` 复用，避免重算材料。
    """
    B = flux_density(A, field)
    mu, dgamma = element_field_material(B, rho, field.masks, material)
    F, _ = field_system(A, field, mu, dgamma, B, R, driving)
    return F, FieldAux(B=B, mu=mu, dgamma=dgamma)


def jacobian(
    A: np.ndarray,
    field: FieldGeometry,
    aux: FieldAux,
    driving: ref_bnd.DrivingRows,
) -> sp_sparse.csr_matrix:
    """Jacobian ``J = ∂F/∂A``（已做驱动行替换）。

    Parameters
    ----------
    A : (N_node,)
    field : FieldGeometry
    aux : FieldAux
        由 ``residual`` 返回。``B`` / ``mu`` / ``dgamma`` 三个量直接复用 ——
        ``B`` 若重算会得到逐位相同的值（同一算式、同一输入），复用只是省时间。
    driving : DrivingRows

    Returns
    -------
    J : (N_node, N_node) csr

    Notes
    -----
    数学见模块 docstring 第二节。两处易错：

    1. 饱和项用的 ``g0_e`` 是**几何量** ``(TᵀK0_eT) A_active``（**不含 ν**）——
       它是 ``∂B/∂A`` 的那个因子，与 ``∂ν/∂B`` 无关。
    2. ``J`` 与源项 ``R`` 无关，所以这里给零源项；``field_system`` 返回的 ``F``
       被丢弃（参考实现里这点浪费可以忽略）。
    """
    _, J = field_system(
        A, field, aux.mu, aux.dgamma, aux.B, np.zeros_like(A), driving
    )
    return J


# =============================================================================
# Newton（内核 11）—— 唯一的循环实现，两层共用
# =============================================================================

def newton_iterations(
    step_fn: Callable[[np.ndarray], Tuple[np.ndarray, object, sp_sparse.csr_matrix]],
    A0: Optional[np.ndarray],
    *,
    n_node: int,
    max_iter: int = 100,
    solve_abs_err: float = 1e-6,
    iter_relaxation: int = 20,
    relaxation_ratio: float = 0.5,
    output: bool = True,
    solver: Optional[str] = None,
    elimination: Optional[object] = None,
    residual_tol: Optional[float] = None,
) -> Tuple[np.ndarray, object, Optional[sp_sparse.csr_matrix], list]:
    """Newton–Raphson 主循环（含分段松弛）。逐字对应 legacy ``MMA10:2380-2406``。

    Parameters
    ----------
    step_fn : callable
        ``step_fn(A) -> (F, aux, J)``。这是两层唯一的接口差异点：

        - ``reference``：``lambda A: (F, aux, jacobian(A, field, aux, driving))``
          （先用 ``residual`` 拿 ``(F, aux)``，再算 ``J``）；
        - ``fast``：直接就是 legacy 的 ``field_error_vectorized``（它一次返回三元组）。

    A0 : (N_node,) 或 None
        初值。``None`` ⇒ 零向量（legacy ``:2383-2384``）。**``A0`` 不会被改写** ——
        legacy 用 ``np.copy``（``:2386``）。
    n_node : int
        节点数（``A0=None`` 时用来定初值形状；legacy 在这里读模块全局 ``N_node``）。
    max_iter, solve_abs_err, iter_relaxation, relaxation_ratio, output :
        同名照抄 legacy 的形参与默认值（``:2380-2382``）。⚠ ``solve_abs_err`` 的
        默认值 ``1e-6`` 在 MMA10 里从未被用到，生产传 ``1e-3`` —— 两者不同是接口的一部分。
    residual_tol : float | None
        **L2 残差容差**（``‖F‖₂``，不是平方）。给了它就**接管**收敛判据，
        ``solve_abs_err`` 被忽略（内部按 ``‖F‖₂² < residual_tol²`` 判，故语义关系
        是 ``solve_abs_err ≡ residual_tol²``）。

        用途：把 legacy 那个**平方范数口径**（易与 L2 判据混淆）在接口上显式化。

        ⚠ **实测（2026-10-17）它不改变本问题的结果**：对 h200（μr=33）的一个
        设计，``residual_tol`` 从 1e-4 收到 1e-10，转矩**逐位不变**
        （1.516918…）。原因见下条 —— **牛顿根本达不到这个容差**。

        ``None``（默认）= 逐字走 legacy 判据，**逐位一致不受影响**。

    Returns
    -------
    A : (N_node,) 解
    aux : step_fn 第二次返回的那个对象（``reference`` 是 ``FieldAux``，``fast`` 是 ``B``）
    J : csr 或 None
        **收敛那一轮**的 Jacobian（不是最后一次求解的 —— 判停在 ``spsolve`` 之前）。
        ``max_iter=0`` 时为 ``None``。
    history : list[float]
        每一轮的 ``err = F·F``（含收敛那一轮）。``len(history)`` 即迭代轮数。

    Notes
    -----
    三处必须留意的语义（模块 docstring 第三节）：``err`` 是**平方**范数、
    打印的轮次是 **1-based**（``i_epoch + 1``）、松弛判据是 **0 基**
    （``i_epoch < iter_relaxation``，数的是迭代而不是更新）。
    """
    if A0 is None:
        A_node = np.zeros((int(n_node),))
    else:
        A_node = np.copy(A0)

    aux = None
    Jacobi = None
    history = []

    # 求解会话在循环外构造一次（pardiso 时 MKL 实例只初始化一次 —— 每步新建
    # 实例实测 ~2s/步会卡死）；本函数构造的会话（owned）由本函数负责 close()。
    session, owned = solvers.as_session(solver)
    try:
        for i_epoch in range(max_iter):
            _F_node, aux, Jacobi = step_fn(A_node)

            err = _F_node.dot(_F_node)
            history.append(float(err))
            if output:
                print('epoch:%d, error: %g' % (i_epoch + 1, err))
            # ⚠ 判据口径（见模块 docstring 第三节 1）：legacy 是**平方**残差阈值。
            #   ``residual_tol`` 给了就按 L2 残差判（把它平方后与原判据同义比较），
            #   避免「阈值忘了平方」这类口径混淆。默认 None ⇒ 逐字走 legacy。
            _tol = (residual_tol * residual_tol) if residual_tol is not None \
                else solve_abs_err
            if err < _tol:
                break
            Jacobi = Jacobi.tocsr()
            if elimination is not None:
                # 消元：Tᵀ·J·T（约束行自动消掉 ⇒ 对称）+ Tᵀ·F，解后恢复 A = T·x_free
                J_free = elimination.eliminate_matrix(Jacobi)
                F_free = elimination.eliminate_rhs(_F_node)
                dA = elimination.restore(session.solve(J_free, F_free))
            else:
                dA = session.solve(Jacobi, _F_node)
            if i_epoch < iter_relaxation:
                A_node = A_node - dA
            else:
                # NR法可能出现振荡，需要使用松弛方法抑制振荡   （legacy :2403 原注释）
                A_node = A_node - dA * relaxation_ratio

        if elimination is not None and Jacobi is not None:
            # 消元模式返回**对称**的 J_free（TᵀJT，约束行已消掉）—— 下游伴随
            # 解的就是消元系统，不再消费含驱动行的 J_full。
            Jacobi = elimination.eliminate_matrix(Jacobi)
        return A_node, aux, Jacobi, history
    finally:
        if owned:
            session.close()


def newton_solve(
    A0: Optional[np.ndarray],
    rho: np.ndarray,
    field: FieldGeometry,
    material: FieldMaterial,
    R: np.ndarray,
    driving: ref_bnd.DrivingRows,
    **kwargs,
) -> Tuple[np.ndarray, FieldAux, Optional[sp_sparse.csr_matrix], list]:
    """``reference`` 层的 Newton 求解：绑定 ``residual`` 与 ``jacobian``。

    Returns
    -------
    ``newton_iterations`` 的四元组 ``(A, aux, J, history)``。

    Notes
    -----
    与 ``fast.solve_field_vectorized`` 的差别只有 ``step_fn`` 的构造方式与返回值
    第二项的语义（这里是 ``FieldAux``，那里是 ``B``）—— Newton 循环本身是**同一份实现**。
    """

    def step_fn(A):
        F, aux = residual(A, rho, field, material, R, driving)
        return F, aux, jacobian(A, field, aux, driving)

    return newton_iterations(
        step_fn, A0, n_node=field.n_node, **kwargs
    )
