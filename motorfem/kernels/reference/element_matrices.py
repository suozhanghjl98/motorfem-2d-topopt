"""电磁单元矩阵（内核 4）。

来源：``legacy/pytorch/nonlinear_solver_energy_MMA10.py:500-545``

------------------------------------------------------------------------------
数学表述
------------------------------------------------------------------------------

一阶三角形单元（3 节点线性）的静磁场单元量。记节点 (i, j, m) 逆时针，
局部形函数梯度 ``∇N_s = (b_s, c_s)/(2Δ)``。

1) 单元刚度（几何部分，不含材料）
   ::

       S_e = (1/(4Δ))·(b bᵀ + c cᵀ)        ∈ R^{3×3}，对称

   legacy 利用对称性只存 **6 个独立值**，顺序为::

       K0[e] = [S_ii, S_ij, S_im, S_jj, S_jm, S_mm]
                (0,0) (0,1) (0,2) (1,1) (1,2) (2,2)

   真实单元刚度是 ``ν_e · S_e``（ν = 1/μ），在装配时才乘上去。

2) 磁密系数
   ::

       Bxcoef[e] =  b/(2Δ)        →  Bx = Bxcoef[e] @ A_local
       Bycoef[e] = -c/(2Δ)        →  By = Bycoef[e] @ A_local

   ⚠ **符号约定很关键，且不影响 |B|**。见下方专节。

3) 源项
   ::

       R0[e] = Δ/3

   这是单元内均匀电流密度 J 的一致（consistent）节点分配——三角形面积 Δ
   均分到 3 个节点。真实右端项是 ``R0[e]·J_e``。

4) 单元面积
   ::

       area[e] = Δ

------------------------------------------------------------------------------
关于 Bx/By 的符号约定（重要）
------------------------------------------------------------------------------

标准矢量恒等式给出 ``B = curl A``，二维下 ``Bx = ∂A/∂y``、``By = -∂A/∂x``，
即::

    教科书约定:  Bxcoef = c/(2Δ) ,  Bycoef = -b/(2Δ)

而 **legacy 用的是**::

    legacy 约定: Bxcoef = b/(2Δ) ,  Bycoef = -c/(2Δ)

两者关系是 **``(Bx, By)_textbook = (-By, -Bx)_legacy``**——一个**反射**
（互换后再取负），不是旋转。

实测该差异对各下游量的影响（同一网格、同一 A）::

    |B|                         两约定**完全相同**（反射不改变模长）
    VD 项 (Bx⊗dBx + By⊗dBy)     两约定**完全相同**（实测 0.000e+00）
    Arkkio 系数外积              两约定**不同，相对差 1.26**
    Arkkio 转矩 (AᵀMA)           两约定**不同，相对差 1.39**

**原因**：Arkkio 要把系数向量按转子位置角 θ 旋转::

    (Bx·cosθ − By·sinθ) ⊗ (Bx·sinθ + By·cosθ)

而反射与旋转**不对易**，旋转后的外积在反射下不再相同。VD 那边是
``Bx⊗dBx + By⊗dBy`` 这样的**和**，反射下两项互换、和不变，故不受影响。

**结论**：`|B|` 与 VD 对约定不敏感，所以写错也不会立刻暴露——**只有 Arkkio
会错，且是静默算错**。因此本模块**严格采用 legacy 约定**，
并把该约定与实测影响记录在此，供 Arkkio 内核（12/13）比对。

------------------------------------------------------------------------------
实现说明
------------------------------------------------------------------------------

本模块是 ``reference/`` 层：逐单元 Python 循环，直接转写 legacy，
便于逐行核对。``fast/element_matrices.py`` 给出批量化版本，两者逐位一致。

几何量（``b``/``c``/``Δ``）复用 ``magnetostatic.element_geometry``，
避免重复实现——那里已包含 ``Δ > 0`` 的定向断言。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .magnetostatic import ElementGeometry, element_geometry

__all__ = [
    "ElectromagneticElements",
    "electromagnetic_elements",
    "expand_K0",
    "current_area",
    "LEGACY_BX_BY_NOTE",
]

#: 本模块采用的 Bx/By 约定的说明（供其它内核引用，避免约定漂移）
LEGACY_BX_BY_NOTE = (
    "Bxcoef = +b/(2Δ), Bycoef = -c/(2Δ) —— legacy 约定。"
    "与教科书 curl 约定差一个反射 (Bx,By)->(-By,-Bx)；"
    "|B| 与 VD 项不受影响，但 Arkkio 会静默算错（相对差约 1.26）。见模块文档。"
)


@dataclass(frozen=True)
class ElectromagneticElements:
    """静磁场的单元级预计算量。全部只依赖网格，与 A、材料无关。

    Attributes
    ----------
    K0 : (N, 6)
        压缩存储的单元几何刚度，顺序 ``[S_ii, S_ij, S_im, S_jj, S_jm, S_mm]``。
        真实刚度 = ``ν_e · K0[e]``（展开后）。
    R0 : (N,)
        ``Δ/3``，源项的一致节点分配系数。
    Bxcoef, Bycoef : (N, 3)
        磁密分量 ``Bx = Σ_s Bxcoef[e,s]·A[tri[e,s]]``。
        **采用 legacy 约定**，见 ``LEGACY_BX_BY_NOTE``。
    area : (N,)
        单元面积 Δ。
    geom : ElementGeometry
        底层几何量（``b``/``c``/``delta``/``tri``），供下游复用。
    """

    K0: np.ndarray
    R0: np.ndarray
    Bxcoef: np.ndarray
    Bycoef: np.ndarray
    area: np.ndarray
    geom: ElementGeometry

    @property
    def n_elem(self) -> int:
        return self.area.shape[0]


def electromagnetic_elements(
    node: np.ndarray, tri: np.ndarray
) -> ElectromagneticElements:
    """计算静磁场的全部单元级预计算量。

    Parameters
    ----------
    node : (N_node, 2) 节点坐标
    tri : (N_elem, 3) 三角形连接表，**必须逆时针**

    Returns
    -------
    ElectromagneticElements

    Notes
    -----
    逐单元循环，与 legacy ``MMA10:509-542`` 一一对应。定向检查（``Δ > 0``）
    由 ``element_geometry`` 完成并会抛 ``ValueError``。

    三种约定的等价写法都写在这里以便核对::

        legacy                本模块                     等价形式
        -------------------   ------------------------   ----------------------
        1/4/Delta*(bi*bi+ci*ci)  (b·b + c·c)/(4Δ)        S 的 (i,i) 分量
        1/2/Delta*bi            b_i/(2Δ)                 Bxcoef 的 i 分量
        -1/2/Delta*ci           -c_i/(2Δ)                Bycoef 的 i 分量
    """
    geom = element_geometry(node, tri)              # 含 Δ > 0 断言
    b, c, delta = geom.b, geom.c, geom.delta
    n_elem = tri.shape[0]

    K0 = np.zeros((n_elem, 6))
    R0 = np.zeros(n_elem)
    Bxcoef = np.zeros((n_elem, 3))
    Bycoef = np.zeros((n_elem, 3))
    area = np.zeros(n_elem)

    for e in range(n_elem):
        bi, bj, bm = b[e]
        ci, cj, cm = c[e]
        D = delta[e]

        # 单元几何刚度，压缩存 6 个独立值（对上三角按行展开）
        K0[e, 0] = 1 / 4 / D * (bi * bi + ci * ci)      # (i,i)
        K0[e, 1] = 1 / 4 / D * (bi * bj + ci * cj)      # (i,j)
        K0[e, 2] = 1 / 4 / D * (bi * bm + ci * cm)      # (i,m)
        K0[e, 3] = 1 / 4 / D * (bj * bj + cj * cj)      # (j,j)
        K0[e, 4] = 1 / 4 / D * (bj * bm + cj * cm)      # (j,m)
        K0[e, 5] = 1 / 4 / D * (bm * bm + cm * cm)      # (m,m)

        # 源项：Δ/3 均分到 3 个节点
        R0[e] = D / 3

        # 磁密系数 —— legacy 约定（Bx=+b, By=-c），勿改成教科书 curl，见模块文档
        Bxcoef[e, 0] = 1 / 2 / D * bi
        Bxcoef[e, 1] = 1 / 2 / D * bj
        Bxcoef[e, 2] = 1 / 2 / D * bm

        Bycoef[e, 0] = -1 / 2 / D * ci
        Bycoef[e, 1] = -1 / 2 / D * cj
        Bycoef[e, 2] = -1 / 2 / D * cm

        area[e] = D

    return ElectromagneticElements(
        K0=K0, R0=R0, Bxcoef=Bxcoef, Bycoef=Bycoef, area=area, geom=geom
    )


def expand_K0(K0: np.ndarray) -> np.ndarray:
    """把压缩的 6 值刚度展开为 ``(N, 3, 3)`` 对称矩阵。

    Parameters
    ----------
    K0 : (N, 6) 顺序 ``[S_ii, S_ij, S_im, S_jj, S_jm, S_mm]``

    Returns
    -------
    (N, 3, 3) 每单元一个对称矩阵

    Notes
    -----
    压缩存储只在装配时需要（避免为对称矩阵存两份）。
    需要逐单元矩阵（例如测试、诊断、力学耦合）时用本函数展开。
    展开是纯粹的数据重排，无浮点运算，故无精度损失。
    """
    K0 = np.asarray(K0)
    n = K0.shape[0]
    K = np.empty((n, 3, 3), dtype=K0.dtype)
    K[:, 0, 0] = K0[:, 0]
    K[:, 0, 1] = K[:, 1, 0] = K0[:, 1]
    K[:, 0, 2] = K[:, 2, 0] = K0[:, 2]
    K[:, 1, 1] = K0[:, 3]
    K[:, 1, 2] = K[:, 2, 1] = K0[:, 4]
    K[:, 2, 2] = K0[:, 5]
    return K


def current_area(area: np.ndarray, element_ids) -> float:
    """给定单元集合的总面积——legacy 用它统计「每极每相载流区域面积」。

    对应 legacy ``MMA10:544-545``::

        if i_elem in element_phaseAp or i_elem in element_phaseAn:
            area_current += Delta

    Parameters
    ----------
    area : (N,) 由 ``electromagnetic_elements`` 给出
    element_ids : 可迭代的单元编号

    Returns
    -------
    float 总面积

    Notes
    -----
    求和顺序会影响浮点末位。这里按**传入顺序**累加（``sum`` 的语义），
    调用方若需要与 legacy 位级一致，应按 legacy 的集合迭代顺序传入。
    详见 ``docs/重构设计文档.md`` 的 F4（set 迭代顺序）。
    """
    ids = list(element_ids)
    return float(np.sum(area[ids])) if ids else 0.0
