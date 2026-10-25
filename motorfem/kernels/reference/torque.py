"""Arkkio 转矩系数与求值（内核 12）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）
     ``get_Arkkio_coef``   ``:666-701``
     求值                   ``:2928``（``_A_node.T @ (Torque_Arkkio_coef @ _A_node)``）

数学出处：``磁障设计/同步磁阻电机磁障结构分析.md``
  - 「Arkkio转矩计算」（约 ``:2947-3027``）
  - 「Arkkio转矩关于参数的导数」（约 ``:3029-3045``，内核 13）

------------------------------------------------------------------------------
一、公式
------------------------------------------------------------------------------

Arkkio 法用气隙圆周上的磁密算转矩（理论文档 :2952）::

    T_Arkkio = (r²/μ0) ∮ B_r B_θ dθ

离散到单元上（理论文档 :2967、:3010-3026）::

    T_Arkkio = Aᵀ [K_Arkkio] A

    [K_Arkkio^(e)] = (r̄²·Δθ_e/μ0)·(k_Bx cos θ̄ − k_By sin θ̄)(k_Bxᵀ sin θ̄ + k_Byᵀ cos θ̄)
    [K_Arkkio]     = ½ Σ_(e) ([K_Arkkio^(e)] + [K_Arkkio^(e)ᵀ])

其中 ``A`` 是**未经约束消元的完整自由度向量**（动节点保留、其值由驱动行的插值约束
确定），``k_Bx``/``k_By`` 是**局部**磁密系数（**不经** ``T``）。

**注意径向/切向分量是怎么来的**：``B_r = Bx cos θ̄ − By sin θ̄``、
``B_θ = Bx sin θ̄ + By cos θ̄`` —— 这就是理论文档那个旋转矩阵
``[[cosθ, −sinθ], [sinθ, cosθ]]``。⚠ ``Bx``/``By`` 采用 **legacy 约定**
（``Bxcoef = +b/(2Δ)``、``Bycoef = −c/(2Δ)``），与教科书 curl 相差一个反射；
``|B|`` 与虚位移项不受影响，但 **Arkkio 会静默算错**——详见
``reference/element_matrices.py`` 的 ``LEGACY_BX_BY_NOTE``。本模块只是把 legacy 的
约定照搬下来，**不改**。

------------------------------------------------------------------------------
二、哪三个量只依赖几何、与 A 无关
------------------------------------------------------------------------------

1. **单元中心到原点的距离** ``r̄``：legacy 写
   ``sqrt((Σx)²/9 + (Σy)²/9)`` = ``|(x̄, ȳ)|``（**形心**到原点的距离）。
2. **单元的平均角** ``θ̄``：legacy 写 ``(θ_i + θ_j + θ_m)/3`` —— 是三个节点
   ``arctan2(y_s, x_s)`` 的**算术平均**，**不是** ``arctan2(ȳ, x̄)``。两者在单元
   跨过 ``θ = 0`` 分支切割时会差 ``2π/3``。本模型（一个象限 [0°, 90°]）不会跨切割，
   但这个写法要照搬。
3. **面积** ``Δ``：legacy 用 ``Aelem``（= 内核 4 的 ``area``，**已实测逐位相同**）。

⚠ **与理论文档的因子差异（★待核对★，已登记进 ``docs/重构设计文档.md`` §11，本批不改）**：
理论文档的面积分版本是 ``1/(μ0(R₂−R₁))·∫∫ r B_r B_θ dS``，配上「扇区 → 整机」的
``parts`` 因子，自然给出 ``(parts/gap)·r̄·Δ/(μ0)·…``。而 legacy 末尾写的是

    _Torque_Arkkio_coef *= 1 / mu0 * parts / (0.25e-3 / 4)

其中 ``0.25e-3`` 是气隙宽度（44.75 → 45.0 mm）、``/4`` 就是 ``/parts`` ⇒ 因子是
``parts²/gap``，**比上面的自然形式多一个 ``parts``（= 4）**。本模块照搬 legacy 的
表达式（``gap`` 与 ``parts`` 都是显式参数，默认值即上述两个数），把这个 4 倍登记为
待核对 —— 修正它属于「改数值行为」，要等基线冻结后与文档、标定一起动。

------------------------------------------------------------------------------
三、与 ``reference/assembly.py``（内核 8）的一处输入差异
------------------------------------------------------------------------------

``get_Arkkio_coef`` 看到的是**插入动网格层之后**的连接表：legacy 在 ``:461-465``
**就地改写** ``element_S``（转子气隙单元的接口节点换成新节点），而 ``get_Arkkio_coef``
在 ``:666`` 才跑。所以本模块的输入是 ``TopologyContext.tri``（= ``layer.tri``）与
``layer.xy``，判据 ``node ∈ rotor_nodes`` 用的是**新**节点号。

它**不需要** ``T``：Arkkio 矩阵直接在（插入后的）节点空间里装配，不经过约束消元。

------------------------------------------------------------------------------
四、与 ``fast`` 层的分层理由
------------------------------------------------------------------------------

本内核**没有「手工向量化」**——legacy 本身就是逐单元 ``for`` 循环加 ``lil`` 逐项累加。
但两层仍然要分开，因为**迭代顺序与累加容器参与数值**（F4）：

- legacy 是 ``for i_elem in element_rotor_gap:``，而 ``element_rotor_gap`` 是一个
  **``set``**（``MMA10:251``）；``lil_matrix`` 的 ``+=`` 逐项累加，故同一个 ``(i, j)``
  的加数顺序由 **set 迭代顺序**决定 ⇒ 末位不同。这是设计文档 F4 那条的**原始案例**。
- ``reference/`` 按 README 的规定顺序必须确定（用 ``range`` 而不是 ``set``），
  因此它与 legacy **不可能**逐位一致；``fast/`` 侧则逐字保留 set 迭代与 lil 累加。

另：legacy 在 ``:695`` 做 ``(M + Mᵀ)/2`` 的**整体**对称化，而理论文档的式子是
**逐单元**对称化 ``½Σ(K^(e) + K^(e)ᵀ)``。两者代数上相等（``½(ΣK + (ΣK)ᵀ)``），
浮点上不同 —— 本模块照公式逐单元写，``fast`` 照 legacy 整体写。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from . import boundaries as ref_bnd
from . import assembly as ref_asm
from .assembly import TopologyContext

__all__ = [
    "ArkkioGeometry",
    "arkkio_coefficient",
    "torque_quadratic",
    "arkkio_torque",
    "ElementTransformWithDerivative",
    "transformation_matrix_v2",
    "vd_coefficient",
    "vd_torque",
    "transformation_matrix_v2_p2",
    "vd_coefficient_p2",
    "vd_torque_p2",
    "adjoint_torque_derivative",
    "torque_derivative_wrt_rho",
]

#: legacy 硬编码在 ``MMA10:693`` 的分母里的气隙宽度（米）。44.75 → 45.0 mm。
LEGACY_GAP_M = 0.25e-3


@dataclass(frozen=True)
class ArkkioGeometry:
    """Arkkio 系数所需的全部输入（与 ``A`` 无关的不变量）。

    Attributes
    ----------
    tri : (N_elem, 3) ndarray
        **插入动网格层之后**的连接表。
    xy : (N_node, 2) ndarray
        节点坐标（含插入的动网格层节点）。
    area : (N_elem,) ndarray
        单元面积 Δ（内核 4 的 ``area``）。
    bxcoef, bycoef : (N_elem, 3) ndarray
        **局部**磁密系数（内核 4 的原值，**未经** ``T``）。
    gap_elements : 可迭代 int
        转子气隙单元（legacy 的 ``element_rotor_gap``，是个 ``set``）。
    rotor_nodes : 可迭代 int
        动网格层节点（legacy 的 ``node_gap_rotor`` 的**键**）。只有局部三个节点里
        **至少一个**在此集合中的单元才参与。
    parts : int
        旋转对称扇区数（真实模型 4）。
    mu0 : float
        真空磁导率。
    gap : float
        气隙宽度（米）。legacy 硬编码 ``0.25e-3``。
    """

    tri: np.ndarray
    xy: np.ndarray
    area: np.ndarray
    bxcoef: np.ndarray
    bycoef: np.ndarray
    gap_elements: Iterable[int]
    rotor_nodes: Iterable[int]
    parts: int = 4
    mu0: float = 4e-7 * np.pi
    gap: float = LEGACY_GAP_M
    # ★ 2026-10-12：`parts²/gap` 里那个多余 `parts` 的开关（见模块 docstring §二）。
    #   1 = legacy 原式 ``parts/(gap/parts) = parts²/gap``（默认，36 槽 golden 逐位不变）；
    #   0 = 理论自然式 ``parts/gap``（6s4p 用 —— 去掉多余的 parts 后，同极对数下
    #       槽数对转矩的影响降到 ~与气隙磁密² 一致，见 tools/_probe_strut.py）。
    #   实现：分母写 ``gap / parts**parts_exponent`` ⇒ exp=1 得 parts²/gap（逐位同原式）。
    parts_exponent: int = 1

    @property
    def n_elem(self) -> int:
        return int(np.asarray(self.tri).shape[0])

    @property
    def n_node(self) -> int:
        return int(np.asarray(self.xy).shape[0])


def _centroid_radius_and_mean_angle(xy: np.ndarray, tri: np.ndarray, e: int) -> Tuple[float, float]:
    """单元 e 的 ``(r̄, θ̄)``。见模块 docstring 第二节。

    ``r̄ = |形心|``、``θ̄ = mean(arctan2(y_s, x_s))`` —— 后者**不是**形心的角度，
    照搬 legacy（``MMA10:673-674`` / ``:676-679``）。
    """
    nodes = tri[e]
    x, y = xy[nodes, 0], xy[nodes, 1]
    r_bar = np.sqrt((x.sum() ** 2) / 9 + (y.sum() ** 2) / 9)
    th_i, th_j, th_m = np.arctan2(y[0], x[0]), np.arctan2(y[1], x[1]), np.arctan2(y[2], x[2])
    return float(r_bar), float((th_i + th_j + th_m) / 3)


def arkkio_coefficient(geom: ArkkioGeometry) -> sp_sparse.csr_matrix:
    """Arkkio 转矩矩阵 ``[K_Arkkio]``，形状 ``(N_node, N_node)``。对应 ``MMA10:666-697``。

    Parameters
    ----------
    geom : ArkkioGeometry

    Returns
    -------
    (N_node, N_node) csr

    Notes
    -----
    按理论文档 :3026 的**逐单元**对称化形式装配::

        [K_Arkkio] = ½ Σ_(e) ([K_Arkkio^(e)] + [K_Arkkio^(e)ᵀ])

    注意 legacy 是「先累加、再整体对称化」（``:695``），两者代数相等、浮点不同。
    ``fast/`` 侧照 legacy 写。

    单元遍历用 ``range`` 并跳过非气隙单元 —— **顺序确定为 element id 升序**，
    不用 ``set`` 迭代（README 对 ``reference/`` 的规定）。因此本函数与 legacy
    只能做到 ``rtol=1e-12``，逐位一致是 ``fast/`` 的职责。
    """
    tri = np.asarray(geom.tri)
    n_node = geom.n_node
    in_gap = set(int(e) for e in geom.gap_elements)
    rotor_nodes = set(int(n) for n in geom.rotor_nodes)

    scale = 1 / geom.mu0 * geom.parts / (geom.gap / geom.parts ** geom.parts_exponent)

    entries = {}
    for e in range(geom.n_elem):
        if e not in in_gap:
            continue
        nodes = [int(n) for n in tri[e]]
        if not any(n in rotor_nodes for n in nodes):
            continue

        r_bar, th_bar = _centroid_radius_and_mean_angle(geom.xy, tri, e)
        cos_t, sin_t = np.cos(th_bar), np.sin(th_bar)

        # 径向 / 切向磁密系数（理论文档 :2991-2998 的旋转矩阵）
        k_br = geom.bxcoef[e] * cos_t - geom.bycoef[e] * sin_t
        k_bt = geom.bxcoef[e] * sin_t + geom.bycoef[e] * cos_t

        # K^(e) = (r̄·Δ/μ0 部分) · k_Br ⊗ k_Bθ，逐单元对称化后再叠加
        block = (r_bar * geom.area[e] * scale) * np.outer(k_br, k_bt)
        block = (block + block.T) / 2

        for a, ni in enumerate(nodes):
            for b, nj in enumerate(nodes):
                key = (ni, nj)
                entries[key] = entries.get(key, 0.0) + block[a, b]

    keys = sorted(entries)
    rows = np.array([k[0] for k in keys], dtype=np.int64)
    cols = np.array([k[1] for k in keys], dtype=np.int64)
    vals = np.array([entries[k] for k in keys], dtype=float)
    return sp_sparse.coo_matrix((vals, (rows, cols)), shape=(n_node, n_node)).tocsr()


def torque_quadratic(A: np.ndarray, coef: sp_sparse.spmatrix) -> float:
    """二次型 ``Aᵀ (coef @ A)``。对应 ``MMA10:2928`` / ``:2997``。

    Notes
    -----
    Arkkio 与虚位移两套转矩的**求值**是同一个二次型，只有系数矩阵不同
    （Arkkio 用本模块的 ``arkkio_coefficient``，虚位移用内核 14 的 ``vd_coefficient``）。
    所以求值只写一份。

    ⚠ 二次型**只看见系数矩阵的对称部分**（``AᵀMA = Aᵀ M_sym A``）—— 这一点在
    内核 14 的导数上会重新出现，见 ``docs/重构设计文档.md`` §11。
    """
    A = np.asarray(A, dtype=float)
    return float(A @ (coef @ A))


def arkkio_torque(A: np.ndarray, geom: ArkkioGeometry) -> float:
    """Arkkio 转矩的便捷入口：装配系数 + 求值。"""
    return torque_quadratic(A, arkkio_coefficient(geom))


# =============================================================================
# 虚位移法（内核 14）
# =============================================================================
#
# 理论文档「虚功法计算转矩」（约 :2906-2945）::
#
#     W_tot = ½ Aᵀ K A − Rᵀ A = −½ Rᵀ A
#     T_VD  = −dW_tot/dθ_m = ½ Rᵀ (dA/dθ_m)
#     F = KA − R = 0  ⇒  dA/dθ_m = −J⁻¹ (∂K/∂θ_m) A
#
# 即转矩是「刚度矩阵对转子位置的导数」的二次型。legacy 直接算 ``∂K/∂θ``（下称 ``M``），
# 然后 ``T_VD = Aᵀ M A``。
#
# **★ D3 判定（2026-09-23）**：代码 ``M_e = 2(Bx⊗dBx + By⊗dBy)/(2μ0)·Δ`` 恰是
# ``∂K/∂θ`` 的**一半**（``∂K_e/∂θ = (1/μ0)(dTᵀK0T + TᵀK0dT) = block_e + block_eᵀ``，
# 真实网格 6 单元验证 rel ~3e-16）。二次型 ``Aᵀ(∂K/∂θ)A = 2Aᵀ·block·A`` 会把另一半
# 自动补回 ⇒ **``T_VD`` 的值与理论 ``−½Aᵀ(∂K/∂θ)A`` 一致**（含 ``−parts`` 模型缩放），
# 理论文档那节的「此处存疑」就此可部分消解 —— 差异不在虚位移**值**的系数，而在
# Arkkio 的 Bx/By 约定（设计文档 §7）。⇒ **D3 = 纯梯度修正，值不变，无需重标定**。
#
# ⚠ **``M`` 原本一般不对称**：``M_e = 2(Bx⊗dBx + By⊗dBy)/(2μ0)·Δ`` 是两个**不同**秩 1 项
# 之和（实测相对不对称度 1.08）。二次型 ``AᵀMA`` 只看见对称部分 —— **值**不受影响，
# 但伴随导数 ``dT/dγ = 2(∂A)ᵀ Q_sym A`` **必须用对称部分**（D3 的根源）。现已对称化
# （见 ``vd_coefficient``），值不变。
#
# ⚠ **F4 在本内核不活跃**：legacy 用的是 ``for i_elem in range(N_element)``
# （``:711``），不是 ``set`` 迭代。所以这里的单元遍历顺序与 legacy 一致，
# 两层只有累加容器的差别（字典 vs ``lil``）。

@dataclass(frozen=True)
class ElementTransformWithDerivative:
    """``T`` 加上对转子位置的解析导数 ``dT``。

    Attributes
    ----------
    matrix : (3, k) ndarray
        ``T``：``A^(e)_local = T · A_active``（与内核 8 的 ``ElementTransform.matrix`` 同）。
    index : tuple[int, ...]
        活跃自由度的全局节点号，顺序即列序。
    d_matrix : (3, k) ndarray
        ``dT``：``∂T/∂pos``。**只在转子气隙单元的插值行上非零**
        （``T`` 的插值系数含 ``pos``；对角元与 ``diag`` 分支都与 ``pos`` 无关）。
    """

    matrix: np.ndarray
    index: Tuple[int, ...]
    d_matrix: np.ndarray


def transformation_matrix_v2(
    i_elem: int, ctx: TopologyContext, rotating,
) -> ElementTransformWithDerivative:
    """单元变换矩阵 ``T`` 与 ``dT``。对应 ``MMA10:1776-1869``。

    Parameters
    ----------
    i_elem : int
    ctx : TopologyContext
    rotating :
        内核 7 的 ``RotatingBoundaryWithDerivative`` —— 要它的 ``matrix``（``T`` 的插值
        系数）、``brackets`` 与 ``d_matrix``（``dKrb_p_dth``）。

    Notes
    -----
    ``T`` 部分与内核 8 的 ``transformation_matrix`` **完全同（逐位）**；
    本函数多出来的只有 ``dT``。三分支逻辑在这里重写一遍（而不是调用内核 8 那个函数）
    是因为 ``dT`` 要用到每个槽位的**符号** ``sign_s``，而内核 8 的接口不暴露它 ——
    与其从 ``matrix`` 反推符号（脆弱），宁可显式重写这十几行。

    ``dT`` 的公式（与 ``T`` 的插值行逐项对应）::

        dT[s, 括号节点] = −dKrb_p_dth[node_s, 括号节点] · sign_s      （node_s 是转子动节点）
        其余位置        = 0

    （``T`` 那边是 ``−K_rb[node_s, 括号节点] · sign_s``。见内核 8 的模块 docstring 第二节。）
    """
    i_elem = int(i_elem)
    dT = None

    # --- 分支 1：转子气隙单元（判据优先于从属分支）---
    if i_elem in ctx.element_rotor_gap:
        if i_elem not in ctx.element_slave:
            nodes = tuple(int(n) for n in ctx.tri[i_elem])
            signs = (1.0, 1.0, 1.0)
        else:
            nodes = tuple(int(n) for n in ctx.virtual_tri[i_elem])
            signs = tuple(
                float(s) for s in ref_bnd.virtual_element_signs(
                    np.asarray([nodes]), ctx.master_nodes, ctx.a_eq
                )[0]
            )

        index = ref_asm._active_index(
            [n for node in nodes
             for n in (rotating.brackets[node] if node in rotating.brackets else (node,))]
        )
        column_of = {node: col for col, node in enumerate(index)}
        matrix = np.zeros((3, len(index)))
        dT = np.zeros((3, len(index)))
        for s, node in enumerate(nodes):
            sign_s = signs[s]
            if node in rotating.brackets:
                # ⚠ brackets 长度随插值阶数变（linear 2 项 / cubic 4 项）⇒ 遍历。
                # linear 下与原先的 (node_up, node_low) 写法逐位相同（同序、同两项）。
                for bracket in (int(c) for c in rotating.brackets[node]):
                    matrix[s, column_of[bracket]] += (
                        -rotating.matrix[node, bracket] * sign_s)
                    dT[s, column_of[bracket]] += (
                        -rotating.d_matrix[node, bracket] * sign_s)
            else:
                matrix[s, column_of[node]] += sign_s
        return ElementTransformWithDerivative(
            matrix=matrix, index=index, d_matrix=dT)

    # --- 分支 2：从属单元（非气隙）---
    if i_elem in ctx.element_slave:
        nodes = tuple(int(n) for n in ctx.virtual_tri[i_elem])
        signs = ref_bnd.virtual_element_signs(
            np.asarray([nodes]), ctx.master_nodes, ctx.a_eq
        )[0]
        return ElementTransformWithDerivative(
            matrix=np.diag(signs), index=nodes, d_matrix=np.zeros((3, 3)))

    # --- 分支 3：普通单元 ---
    nodes = tuple(int(n) for n in ctx.tri[i_elem])
    return ElementTransformWithDerivative(
        matrix=np.eye(3), index=nodes, d_matrix=np.zeros((3, 3)))


def vd_coefficient(
    ctx: TopologyContext,
    rotating,
    *,
    n_node: int,
    bxcoef: np.ndarray,
    bycoef: np.ndarray,
    area: np.ndarray,
    parts: int,
    mu0: float = 4e-7 * np.pi,
) -> sp_sparse.csr_matrix:
    """虚位移转矩矩阵 ``M = ∂K/∂θ_m``（含 ``−parts`` 因子）。对应 ``MMA10:707-736``。

    逐单元（理论文档 :2939-2945 的 ``∂K/∂θ`` 展开，取 ``∂T`` 那一支）::

        M_e = 2 (Bx_e ⊗ dBx_e + By_e ⊗ dBy_e) / (2μ0) · Δ_e
        其中 Bx_e = Bxcoef_e · T_e      dBx_e = Bxcoef_e · dT_e
        ⇒ M = −parts · Σ_e M_e         （末尾的 *= -parts，``:735``）

    Parameters
    ----------
    ctx, rotating :
        内核 6 的拓扑与内核 7 的 ``RotatingBoundaryWithDerivative``。
    n_node : int
        legacy 在这里读模块全局 ``N_node``。
    bxcoef, bycoef : (N_elem, 3)
        内核 4 的**局部**系数（本函数负责乘 ``T`` / ``dT``）。
    area : (N_elem,) 单元面积 Δ
    parts : int 扇区数
    mu0 : float

    Returns
    -------
    (N_node, N_node) csr

    Notes
    -----
    **D3 已修（2026-09-23，对称化）**：legacy 不对称化（``:735`` 直接 ``*= -parts``），
    但那会让伴随导数错 —— 二次型 ``T = AᵀMA`` 只看见对称部分（**值**不受影响），
    而导数 ``dT/dγ = 2(∂A)ᵀ Q_sym A`` **必须用对称部分**。见模块 docstring 第二节
    与设计文档 §12 的 D3。

    对称化做在**装配后**（``(M + Mᵀ)/2``），保持逐单元累加与 legacy 一致
    （``:711`` 就是 ``range(N_element)``，F4 不活跃），只改最后一步。

    单元遍历用 ``range``，与 legacy 一致（``:711`` 就是 ``range(N_element)``），
    所以 F4 在这里不活跃。
    """
    n_elem = int(np.asarray(bxcoef).shape[0])
    entries = {}

    for e in range(n_elem):
        if e not in ctx.element_rotor_gap:
            continue
        tr = transformation_matrix_v2(e, ctx, rotating)
        bx = bxcoef[e] @ tr.matrix
        by = bycoef[e] @ tr.matrix
        dbx = bxcoef[e] @ tr.d_matrix
        dby = bycoef[e] @ tr.d_matrix
        block = (2 * (np.outer(bx, dbx) + np.outer(by, dby))
                 / (2 * mu0) * area[e])
        for a, ni in enumerate(tr.index):
            for b, nj in enumerate(tr.index):
                entries[(int(ni), int(nj))] = entries.get((int(ni), int(nj)), 0.0) + block[a, b]

    # ⚠ 丢掉**恰好为 0** 的累加项：legacy 用 ``lil_matrix`` 的 ``+=``，而 lil 的赋值
    # 会丢弃零（内核 8 也记过这族语义：lil 丢零 / COO 保零）。保留它们会让稀疏结构
    # 与 legacy 不同（本文件实测过 84 vs 48 个非零元），parity 测试会失败。
    keys = sorted(k for k in entries if entries[k] != 0.0)
    rows = np.array([k[0] for k in keys], dtype=np.int64) if keys else np.zeros(0, np.int64)
    cols = np.array([k[1] for k in keys], dtype=np.int64) if keys else np.zeros(0, np.int64)
    vals = np.array([entries[k] for k in keys], dtype=float) if keys else np.zeros(0)
    M = sp_sparse.coo_matrix((vals, (rows, cols)), shape=(n_node, n_node)).tocsr()
    # D3 修正（2026-09-23）：对称化再做 −parts。值不变（二次型只看对称部分），
    # 伴随导数用对称部分才正确。见本函数 docstring 与设计文档 §12 的 D3。
    return ((M + M.T) / 2) * (-parts)


def vd_torque(A: np.ndarray, ctx, rotating, **kwargs) -> float:
    """虚位移转矩的便捷入口：装配系数 + 求值。"""
    return torque_quadratic(A, vd_coefficient(ctx, rotating, **kwargs))


# =============================================================================
# P2（6 节点）虚位移转矩 —— 路线 A，2026-10-02
# =============================================================================

def transformation_matrix_v2_p2(
    i_elem: int, ctx: TopologyContext, rotating,
) -> ElementTransformWithDerivative:
    """P2 单元变换矩阵 ``T`` 与 ``dT``（对照 P1 `transformation_matrix_v2`）。

    ``T`` 部分与 `assembly_p2.transformation_matrix_p2` 完全同（6 槽位统一：
    brackets → 插值行、否则单位行，Design B）。``dT`` 只在插值行非零：:

        dT[s, 括号节点] = −dKrb_p_dth[node_s, 括号节点] · sign_s      （转子动节点）
        其余位置 = 0

    见 P1 `transformation_matrix_v2` 的 Notes（``dT`` 要用每个槽位的符号 ``sign_s``，
    内核 8 的接口不暴露它，故显式重写而非复用）。
    """
    i_elem = int(i_elem)

    # --- 分支 1：转子气隙单元（判据优先于从属分支）---
    if i_elem in ctx.element_rotor_gap:
        if i_elem not in ctx.element_slave:
            nodes = tuple(int(n) for n in ctx.tri[i_elem])
            signs = (1.0,) * 6
        else:
            nodes = tuple(int(n) for n in ctx.virtual_tri[i_elem])
            signs = tuple(
                float(s) for s in ref_bnd.virtual_element_signs(
                    np.asarray([nodes]), ctx.master_nodes, ctx.a_eq
                )[0]
            )

        index = ref_asm._active_index(
            [n for node in nodes
             for n in (rotating.brackets[node] if node in rotating.brackets else (node,))]
        )
        column_of = {node: col for col, node in enumerate(index)}
        matrix = np.zeros((6, len(index)))
        dT = np.zeros((6, len(index)))
        for s, node in enumerate(nodes):
            sign_s = signs[s]
            if node in rotating.brackets:
                # ⚠ brackets 长度随插值阶数变（linear 2 项 / cubic 4 项）⇒ 遍历。
                # linear 下与原先的 (node_up, node_low) 写法逐位相同（同序、同两项）。
                for bracket in (int(c) for c in rotating.brackets[node]):
                    matrix[s, column_of[bracket]] += (
                        -rotating.matrix[node, bracket] * sign_s)
                    dT[s, column_of[bracket]] += (
                        -rotating.d_matrix[node, bracket] * sign_s)
            else:
                matrix[s, column_of[node]] += sign_s
        return ElementTransformWithDerivative(
            matrix=matrix, index=index, d_matrix=dT)

    # --- 分支 2：从属单元（非气隙）---
    if i_elem in ctx.element_slave:
        nodes = tuple(int(n) for n in ctx.virtual_tri[i_elem])
        signs = ref_bnd.virtual_element_signs(
            np.asarray([nodes]), ctx.master_nodes, ctx.a_eq
        )[0]
        return ElementTransformWithDerivative(
            matrix=np.diag(signs), index=nodes, d_matrix=np.zeros((6, 6)))

    # --- 分支 3：普通单元 ---
    nodes = tuple(int(n) for n in ctx.tri[i_elem])
    return ElementTransformWithDerivative(
        matrix=np.eye(6), index=nodes, d_matrix=np.zeros((6, 6)))


def vd_coefficient_p2(
    ctx: TopologyContext,
    rotating,
    *,
    n_node: int,
    bxcoef_g: np.ndarray,
    bycoef_g: np.ndarray,
    gauss_w: np.ndarray,
    area: np.ndarray,
    parts: int,
    mu0: float = 4e-7 * np.pi,
) -> sp_sparse.csr_matrix:
    """P2 虚位移转矩矩阵 ``M = ∂K/∂θ_m``（含 ``−parts``），积分点口径。

    对照 P1 `vd_coefficient`。P2 磁能逐高斯点求值，虚功转矩相应为（推导见
    `docs/重构设计文档.md` §13.3）：:

        M_e = Σ_g (w_g·Δ_e/μ0)·(Bx_g ⊗ dBx_g + By_g ⊗ dBy_g)
        Bx_g = bxcoef_g[e, gg] @ T_e        dBx_g = bxcoef_g[e, gg] @ dT_e

    只对转子气隙单元（∂T/∂θ 非零）贡献。末尾 ``×(−parts)`` 与 D3 对称化
    ``(M + Mᵀ)/2`` 同 P1（二次型只看见对称部分，伴随导数要对称部分）。
    """
    n_elem = int(np.asarray(bxcoef_g).shape[0])
    n_gauss = int(bxcoef_g.shape[1])
    entries = {}

    for e in range(n_elem):
        if e not in ctx.element_rotor_gap:
            continue
        tr = transformation_matrix_v2_p2(e, ctx, rotating)
        for gg in range(n_gauss):
            bx = bxcoef_g[e, gg] @ tr.matrix
            by = bycoef_g[e, gg] @ tr.matrix
            dbx = bxcoef_g[e, gg] @ tr.d_matrix
            dby = bycoef_g[e, gg] @ tr.d_matrix
            block = (gauss_w[gg] * area[e] / mu0
                     * (np.outer(bx, dbx) + np.outer(by, dby)))
            for a, ni in enumerate(tr.index):
                for b, nj in enumerate(tr.index):
                    entries[(int(ni), int(nj))] = (
                        entries.get((int(ni), int(nj)), 0.0) + block[a, b]
                    )

    keys = sorted(k for k in entries if entries[k] != 0.0)
    rows = np.array([k[0] for k in keys], dtype=np.int64) if keys else np.zeros(0, np.int64)
    cols = np.array([k[1] for k in keys], dtype=np.int64) if keys else np.zeros(0, np.int64)
    vals = np.array([entries[k] for k in keys], dtype=float) if keys else np.zeros(0)
    M = sp_sparse.coo_matrix((vals, (rows, cols)), shape=(n_node, n_node)).tocsr()
    return ((M + M.T) / 2) * (-parts)


def vd_torque_p2(A: np.ndarray, ctx, rotating, **kwargs) -> float:
    """P2 虚位移转矩便捷入口：装配系数 + 求值。"""
    return torque_quadratic(A, vd_coefficient_p2(ctx, rotating, **kwargs))


# =============================================================================
# 对 γ 的导数（内核 13）—— 伴随解
# =============================================================================

def adjoint_torque_derivative(
    A: np.ndarray,
    field,
    Jacobi,
    coef: sp_sparse.spmatrix,
    elements_to_compute,
) -> np.ndarray:
    """转矩对**磁阻率** γ 的导数 ``dT/dγ``。

    **内核 13 与内核 14 共用本函数** —— legacy 的 ``DTorque_Arkkio_vectorized``
    （``:2260-2277``）与 ``DTorque_VD_vectorized``（``:2280-2297``）**函数体逐字相同**，
    只差系数矩阵从哪里来（前者用模块全局 ``Torque_Arkkio_coef``、后者是形参）。
    所以这里只写一份，系数作为参数。

    Parameters
    ----------
    A : (N_node,) 收敛的矢量磁位（**完整自由度向量**，含动节点）
    field : FieldGeometry
        内核 9/10 的输入束（要 ``transforms`` 与 ``k0``）。
    Jacobi : (N_node, N_node)
        内核 11 返回的**同一个对象**（不重建、不拷贝）。
    coef : (N_node, N_node) 系数矩阵
        Arkkio 用 ``arkkio_coefficient``，虚位移用内核 14 的 ``vd_coefficient``。
    elements_to_compute : 可迭代 int
        设计变量所在的单元（legacy 的 ``element_topo_opti``）。

    Returns
    -------
    (N_elem,) ndarray，``elements_to_compute`` 之外恒为 0

    数学（理论文档 :3044，形状与内核 9/10 的残差形式伴随一致）::

        T(γ) = A(γ)ᵀ M A(γ)          M = 系数矩阵（Arkkio 或虚位移）
        F(A, γ) = K(γ)A − R = 0   ⇒  dA/dγ = −J⁻¹ (∂K/∂γ) A
        dT/dγ   = 2 (dA/dγ)ᵀ M A  = −2 · [(∂K/∂γ) A]ᵀ · J⁻ᵀ(M A)
                                     └──────────┘     └────────┘
                                     legacy 的           伴随解 λ
                                     dK_p_dgamma_A

    即：**先解一次伴随 ``λ = J⁻ᵀ(M A)``，再把 ``(∂K/∂γ_e)A`` 与它收缩**。

    逐单元的收缩有闭式（这是本模块唯一需要 ``field`` 的地方）::

        (∂K/∂γ_e) A 在单元 e 的活跃自由度上 = (TᵀK0_eT) · A_active = g0_e
        ⇒ (dT/dγ)_e = −2 · (g0_e · λ_active)

    ``g0_e`` **不含** ν —— 与内核 9/10 的饱和项用的是同一个几何量。

    ⚠ **一处必须写清的语义**：伴随右端 ``coef @ A`` 里的 ``coef`` 必须是**对称**的
    （``dT/dγ = 2(∂A)ᵀ Q_sym A``，见模块 docstring 第二节）。Arkkio 的 ``M`` 已被
    legacy ``:695`` 显式对称化；**虚位移的 ``M`` 现已同步对称化（D3 修正，
    2026-09-23）** —— ``vd_coefficient`` 返回的就是对称矩阵。本实现不再需要
    调用方操心这件事。

    Notes
    -----
    ``λ`` 用 ``Jacobi.T`` 求解（``:2272``）：驱动行替换让 ``Jacobi`` **不对称**，
    所以伴随用的是转置而不是同一个矩阵。

    ``elements_to_compute`` 之外的分量**清零**（``:2274-2275``）—— 设计变量只在
    拓扑优化区，这个清零是它唯一的落地处。清零与遍历顺序无关（每个元素写自己的槽位）。
    """
    lam = sp_sparse_linalg.spsolve(sp_sparse.csr_matrix(Jacobi).T, coef @ A)

    out = np.zeros(field.n_elem)
    for e in elements_to_compute:
        tr = field.transforms[e]
        index = list(tr.index)
        k0_e_active = tr.matrix.T @ (field.k0[e] @ tr.matrix)
        g0_e = k0_e_active @ A[index]
        out[e] = -2 * (g0_e @ lam[index])
    return out


def torque_derivative_wrt_rho(
    dT_d_gamma: np.ndarray,
    B: np.ndarray,
    rho: np.ndarray,
    curve,
    simp,
) -> np.ndarray:
    """``dT/dρ = dT/dγ · dγ/dρ`` —— §7 说的**三处重复链式环节**的公共层。

    对应 legacy 三处逐字相同的写法（``MMA10:2932`` / ``:2992`` / ``:2964`` 一族）::

        dTArkkio_p_drho = dTArkkio_p_dgamma * nonlinear_material_dgamma_p_drho(B_elem, _rho_SIMP)
        dTVD_p_drho     = dTVD_p_dgamma     * nonlinear_material_dgamma_p_drho(B_elem, _rho_SIMP)
        dTflux_p_drho   = dTflux_p_dgamma   * nonlinear_material_dgamma_p_drho(B_elem, _rho_SIMP)

    三处**逐字相同**，只是 ``dT/dγ`` 来自不同的转矩项。所以提到这里一份，
    内核 14/15 直接复用（设计文档 §1.8 批 3 的要求）。

    ⚠ ``dγ/dρ`` 对**全部**单元求值（不掩码），依赖下游的 ``elements_to_compute``
    清零把非设计区压掉 —— 这条依赖关系见 ``reference/material.py`` 与
    设计文档 §11。
    """
    from .material import nonlinear_material_dnu_drho
    return dT_d_gamma * nonlinear_material_dnu_drho(B, rho, curve, simp)
