"""二维非线性静磁场的参考实现（朴素、逐单元循环）。

本模块是**数学的权威表达**。它故意写得慢、写得直白——每个 `for e in range(N_elem)`
都对应理论文档里的一条公式。生产用的向量化实现在 `motorfem.kernels.fast.magnetostatic`，
两者的等价性由 `tests/kernels/` 保证。

出处：``同步磁阻电机磁障结构分析.md``
  - 「完整的拓扑优化方法 › 考虑饱和的求解」
  - 「完整的拓扑优化方法 › 使用Newton-Raphson方法进行求解」
  - 「完整的拓扑优化方法 › 计算步骤」

------------------------------------------------------------------------------
数学表述
------------------------------------------------------------------------------

求解二维平行平面场中的矢量磁位 A_z（下称 A），用一阶三角形单元（3 节点线性）。

1) 单元几何（节点 i, j, m 按逆时针编号）
   设 (x_s, y_s) 为节点 s 的坐标，定义::

       b_i = y_j - y_m      c_i = -x_j + x_m
       b_j = y_m - y_i      c_j = -x_m + x_i
       b_m = y_i - y_j      c_m = -x_i + x_j

   形函数梯度为 ∇N_s = (1/(2Δ))·(b_s, c_s)，其中::

       Δ = (b_i·c_j - b_j·c_i) / 2 = 三角形的有向面积

   Δ > 0 等价于节点逆时针排列。本模块不做修正，直接断言，以便暴露网格方向错误。

2) 磁密
   B = curl A = (∂A/∂y, -∂A/∂x)，在单元内为常数::

       Bx = (b·A_e) / (2Δ)          b·A_e = Σ_s b_s A_s
       By = -(c·A_e) / (2Δ)         c·A_e = Σ_s c_s A_s
       B  = sqrt((b·A_e)² + (c·A_e)²) / (2Δ)

   注意 b/c 与 Bx/By 的配对是个符号约定；|B| 的表达式不受该约定影响，而下面所有
   公式只通过 |B| 依赖材料，所以约定不影响结果。

3) 单元刚度矩阵
   取 ν = 1/μ 为磁阻率（reluctivity）。单元刚度::

       S_e = (1/(4Δ))·(b bᵀ + c cᵀ)          ∈ R^{3×3}，只由几何决定
       K_e = ν_e · S_e

4) 残差（弱形式的离散）
   记右端源项（电流密度与永磁体贡献）装配为 R::

       F(A) = Σ_e ν_e S_e A_e − R = K(A)·A − R

   A 满足 F(A) = 0。

5) Jacobian
   定义辅助向量（**这就是 legacy 代码里 ``g0i_vec`` 的数学含义**）::

       g_e = S_e A_e          ∈ R³

   由 B² = ((b·A_e)² + (c·A_e)²)/(4Δ²) 求导可得::

       ∂B_e/∂A_m = (gS·b_m + gC·c_m) / (4Δ²·B_e)          其中 gS = b·A_e, gC = c·A_e

   而 S_e A_e = (1/(4Δ))·(b·gS + c·gC)，代入得::

       ∂B_e/∂A = g_e / (Δ·B_e)

   于是::

       J = ∂F/∂A = Σ_e ν_e S_e  +  Σ_e (dν_e/dB) · (∂B_e/∂A) ⊗ g_e
                 = Σ_e ν_e S_e  +  Σ_e [ dν_e/dB / (Δ_e·B_e) ] · g_e ⊗ g_e
                                     └──────────────────────────────────┘
                                              饱和项（秩 1 外积）

   **关键观察**：外积的两个因子是**同一个**向量 g_e。这正是 legacy 代码里
   ``g0i_vec`` 出现两次、以及 ``gi_vec = g0i_vec * const_vec`` 的原因——
   ``gswap1/gswap2`` 只是对同一个扩展向量做「取第 i 个 / 取第 j 个」两种选择。

   本模块把这一项直接写成 ``np.outer(g, g)``，因为这就是它的数学。
   `fast` 版本把它压成 ``(gswap1 @ g_exp) * (gswap2 @ (coef * g_exp))`` 以换速度。

6) Newton–Raphson 迭代
   ::

       J(A^k)·ΔA = F(A^k)
       A^{k+1} = A^k − α·ΔA

   前 ``relaxation_iter`` 步取 α = 1，之后取 ``relaxation_ratio`` —— 因为 NR 在
   强非线性下可能振荡，需要用松弛抑制（legacy 注释原话）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp_sparse

__all__ = [
    "ElementGeometry",
    "element_geometry",
    "element_stiffness",
    "flux_density_magnitude",
    "assemble_stiffness",
    "residual",
    "jacobian",
    "newton_solve",
]


# =============================================================================
# 单元几何
# =============================================================================

@dataclass(frozen=True)
class ElementGeometry:
    """预计算的单元几何量。全部形状为 ``(N_elem, 3)`` 或 ``(N_elem,)``。

    Attributes
    ----------
    b, c : ndarray, (N_elem, 3)
        形函数梯度系数，``b[e, s]`` / ``c[e, s]`` 对应单元 e 局部节点 s。
        与 ``tri`` 的列顺序一致。
    delta : ndarray, (N_elem,)
        三角形有向面积 Δ，恒为正（``element_geometry`` 会断言）。
    tri : ndarray, (N_elem, 3)
        单元-节点连接表，与输入相同，便于下游使用。
    """

    b: np.ndarray
    c: np.ndarray
    delta: np.ndarray
    tri: np.ndarray


def element_geometry(node: np.ndarray, tri: np.ndarray) -> ElementGeometry:
    """由节点坐标与连接表计算 ``b`` / ``c`` / ``Δ``。

    Parameters
    ----------
    node : (N_node, 2) 节点坐标 (x, y)，单位 SI 米
    tri : (N_elem, 3) 三角形连接表，**必须逆时针**

    Returns
    -------
    ElementGeometry

    Notes
    -----
    对应理论文档「考虑饱和的求解」中的::

        b_i = y_j - y_m ,  b_j = y_m - y_i ,  b_m = y_i - y_j
        c_i = -x_j + x_m,  c_j = -x_m + x_i,  c_m = -x_i + x_j
        Δ  = (b_i·c_j - b_j·c_i)/2

    这里用 ``np.roll`` 的等价写法：局部节点 (i, j, m) 固定取 (0, 1, 2)，
    ``(i,j,m)`` 与 ``(j,m,i)`` 的循环置换关系由 roll 实现。

    之所以断言 ``Δ > 0`` 而不取绝对值：面积符号是**网格方向的唯一检查点**。
    一旦某个单元被反向编号，后续所有 ``b``/``c`` 的符号约定都会静默失效，
    而结果看起来"像是对的"。legacy 代码里有同样的断言（``MMA10:583``）。
    """
    n_elem = tri.shape[0]

    # x[e, s] = 单元 e 局部节点 s 的 x 坐标
    x = node[tri, 0]          # (N_elem, 3)
    y = node[tri, 1]

    # (i, j, m) = (0, 1, 2)，故
    #   b_i = y_j - y_m  ->  np.roll(y, -1) - np.roll(y, -2)
    #   c_i = -x_j + x_m -> -np.roll(x, -1) + np.roll(x, -2)
    # 用 roll 一次算出全部三个分量。
    b = np.roll(y, -1, axis=1) - np.roll(y, -2, axis=1)
    c = -np.roll(x, -1, axis=1) + np.roll(x, -2, axis=1)

    delta = (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]) / 2.0

    if np.any(delta <= 0.0):
        bad = np.flatnonzero(delta <= 0.0)[:5]
        raise ValueError(
            f"有 {int(np.sum(delta <= 0.0))} 个单元面积非正（须逆时针编号）。"
            f"前几个单元编号: {bad.tolist()}"
        )

    return ElementGeometry(b=b, c=c, delta=delta, tri=tri)


# =============================================================================
# 单元量
# =============================================================================

def element_stiffness(geom: ElementGeometry) -> np.ndarray:
    """单元几何刚度 ``S_e = (1/(4Δ))·(b bᵀ + c cᵀ)``，形状 ``(N_elem, 3, 3)``。

    只依赖几何。真实单元刚度是 ``K_e = ν_e · S_e``，ν 由材料给出。

    对应理论文档::

        K^{(e)} = (1/(4μΔ))·[[b_i b_i + c_i c_i,  b_i b_j + c_i c_j,  b_i b_m + c_i c_m],
                             [ ... ,                  ... ,                 ...        ],
                             [ ... ,                  ... ,                 ...        ]]

    即 ``b bᵀ + c cᵀ`` 的外积展开，只差一个 ``1/(4Δ)`` 与材料因子 ``1/μ = ν``。

    实现说明：``b[:, :, None] * b[:, None, :]`` 是 ``b_e ⊗ b_e`` 的批量外积写法。
    这里用批量写法而不是逐单元循环，是因为外积本身无累加顺序问题（每个单元
    独立写入自己的 3×3 块），不影响任何浮点求和顺序。
    """
    inv4d = 1.0 / (4.0 * geom.delta)                      # (N_elem,)
    bb = geom.b[:, :, None] * geom.b[:, None, :]          # (N_elem, 3, 3)
    cc = geom.c[:, :, None] * geom.c[:, None, :]          # (N_elem, 3, 3)
    return inv4d[:, None, None] * (bb + cc)


def flux_density_magnitude(A: np.ndarray, geom: ElementGeometry) -> np.ndarray:
    """单元内磁密幅值 |B|，形状 ``(N_elem,)``。

    对应理论文档::

        B = (1/Δ)·sqrt( A^{(e)T} k^{(e)} A^{(e)} )

    与之等价且更易读的写法是直接展开::

        B = sqrt((b·A_e)² + (c·A_e)²) / (2Δ)

    ``clip`` 的下界 1e-20 是必要的：线性材料下 B 可以恒为 0（空载、无源），
    而下游的 ``∂B/∂A = g/(Δ·B)`` 与 ``1/B`` 会除零。legacy 代码在
    ``MMA10:2231`` 有同样的 ``clip(min=1e-20)``。
    """
    A_e = A[geom.tri]                                     # (N_elem, 3)
    b_dot_A = np.einsum("es,es->e", geom.b, A_e)          # (N_elem,)
    c_dot_A = np.einsum("es,es->e", geom.c, A_e)
    B = np.sqrt(b_dot_A ** 2 + c_dot_A ** 2) / (2.0 * geom.delta)
    return np.clip(B, 1e-20, np.inf)


# =============================================================================
# 装配
# =============================================================================

def assemble_stiffness(
    S: np.ndarray, nu: np.ndarray, tri: np.ndarray, n_node: int
) -> sp_sparse.csr_matrix:
    """把 ``ν_e · S_e`` 装配为全局稀疏矩阵。形状 ``(N_node, N_node)``。

    Parameters
    ----------
    S : (N_elem, 3, 3) 几何刚度（``element_stiffness`` 的输出）
    nu : (N_elem,) 单元磁阻率 ν = 1/μ
    tri : (N_elem, 3) 连接表
    n_node : int 节点总数

    Notes
    -----
    实现用 COO 三元组一次性构造，而不是 ``lil_matrix`` 逐项赋值。
    legacy 在这一步用的是 ``lil``，而 ``cProfileReport.txt`` 显示 **153 s 里有
    94 s（61%）花在 lil 的构造与逐项赋值上**——这正是 ``get_K_vectorized``
    改走 COO 预组装的原因。

    ⚠ 注意：本函数是 ``reference/`` 层，允许重排装配顺序（COO 内部的求和顺序
    与 lil 不同），因此**不参与位级对齐**，只参与 ``rtol=1e-12`` 的内核级对齐。
    见本目录 README 的「红线豁免」。
    """
    n_elem = tri.shape[0]

    # 单元 3×3 块 -> 9 个 (i, j) 位置
    rows = np.repeat(tri, 3, axis=1)                      # (N_elem, 9)
    cols = np.tile(tri, (1, 3))                           # (N_elem, 9)
    vals = (nu[:, None, None] * S).reshape(n_elem, 9)

    K = sp_sparse.coo_matrix(
        (vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n_node, n_node)
    )
    return K.tocsr()


# =============================================================================
# 残差与 Jacobian
# =============================================================================

def residual(
    A: np.ndarray,
    geom: ElementGeometry,
    S: np.ndarray,
    nu_of_B: Callable[[np.ndarray], np.ndarray],
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """残差 ``F(A) = K(A)·A − R``。

    Parameters
    ----------
    A : (N_node,) 当前矢量磁位
    geom, S : 由 ``element_geometry`` / ``element_stiffness`` 预计算
    nu_of_B : callable, (N_elem,) → (N_elem,)
        磁阻率 B → ν。之所以传函数而不是数组，是为了让「材料模型」与
        「装配」解耦——将来换材料、换 autograd 路径都只替换这个 callable。
    R : (N_node,) 源项（电流 + 永磁体）

    Returns
    -------
    F : (N_node,) 残差
    nu : (N_elem,) 本次用到的磁阻率（供 Jacobian 复用，避免重复求值）
    """
    B = flux_density_magnitude(A, geom)
    nu = nu_of_B(B)
    K = assemble_stiffness(S, nu, geom.tri, A.shape[0])
    return K @ A - R, nu


def jacobian(
    A: np.ndarray,
    geom: ElementGeometry,
    S: np.ndarray,
    nu: np.ndarray,
    dnu_dB: Callable[[np.ndarray], np.ndarray],
) -> sp_sparse.csr_matrix:
    """Jacobian ``J = ∂F/∂A``。

    对应理论文档「使用Newton-Raphson方法进行求解」::

        J = Σ_e [K_st^{(e)} + (1/γ²)·(g_s^{(e)} g_t^{(e)})/(B·Δ)·(∂γ/∂B)]

    本模块的记法下（ν 即 1/μ，取 γ ↔ ν）::

        J = Σ_e ν_e S_e  +  Σ_e [ dν_e/dB / (Δ_e·B_e) ] · g_e ⊗ g_e

    其中 ``g_e = S_e A_e``。

    Parameters
    ----------
    A : (N_node,)
    geom, S : 预计算的几何
    nu : (N_elem,) 由 ``residual`` 返回，避免重复求 B 与 ν
    dnu_dB : callable, (N_elem,) → (N_elem,) 磁阻率对 |B| 的导数

    Returns
    -------
    J : (N_node, N_node) csr 矩阵

    Notes
    -----
    **这是本模块最核心的一段，也是 legacy 代码最不可读的一段。**
    legacy 把它写成（``MMA10:2246-2252``）::

        g0i_vec    = g0i_mat @ _A_node
        const_vec  = gconst_mat @ (1 / B / Aelem * dgamma)
        gi_vec     = g0i_vec * const_vec
        Jabobi2_data = (gswap1_mat @ g0i_vec) * (gswap2_mat @ gi_vec)

    对照本模块：``g0i_vec`` ↔ 扩展空间里的 ``g_e``；``gswap1 @ ·`` 取 ``g_e[i]``、
    ``gswap2 @ ·`` 取 ``g_e[j]``，两者逐元素相乘即得 9 个位置的 ``g_e[i]·g_e[j]``。
    也就是说那两个矩阵只是「把 3 阶张量展平成稀疏矩阵」的工具，**数学上就是一个
    秩 1 外积**。
    """
    B = flux_density_magnitude(A, geom)
    n_elem = geom.tri.shape[0]
    n_node = A.shape[0]

    K = assemble_stiffness(S, nu, geom.tri, n_node)

    # g_e = S_e A_e  —— (N_elem, 3)
    A_e = A[geom.tri]                                     # (N_elem, 3)
    g = np.einsum("eij,ej->ei", S, A_e)                   # (N_elem, 3)

    # 饱和项系数:  dν/dB / (Δ·B)
    coef = dnu_dB(B) / (geom.delta * B)                   # (N_elem,)

    # 逐单元把 coef_e · g_e ⊗ g_e 叠加进全局矩阵
    rows = np.repeat(geom.tri, 3, axis=1)                 # (N_elem, 9)
    cols = np.tile(geom.tri, (1, 3))                      # (N_elem, 9)
    outer = g[:, :, None] * g[:, None, :]                 # (N_elem, 3, 3) = g_e ⊗ g_e
    vals = coef[:, None, None] * outer                    # (N_elem, 3, 3)

    J_sat = sp_sparse.coo_matrix(
        (vals.reshape(n_elem, 9).ravel(), (rows.ravel(), cols.ravel())),
        shape=(n_node, n_node),
    ).tocsr()

    return (K + J_sat).tocsr()


# =============================================================================
# Newton–Raphson
# =============================================================================

def newton_solve(
    A0: np.ndarray,
    geom: ElementGeometry,
    S: np.ndarray,
    nu_of_B: Callable[[np.ndarray], np.ndarray],
    dnu_dB: Callable[[np.ndarray], np.ndarray],
    R: np.ndarray,
    *,
    fixed_nodes: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
    abs_err: float = 0.0,
    rel_err: float = 1e-10,
    max_iter: int = 100,
    relaxation_iter: int = 20,
    relaxation_ratio: float = 0.5,
    callback: Callable[[int, float], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Newton–Raphson 迭代求解 ``F(A) = 0``。

    Parameters
    ----------
    A0 : (N_node,) 初值
    geom, S : 预计算几何
    nu_of_B, dnu_dB : 材料模型 callable
    R : (N_node,) 源项
    fixed_nodes : 第一类（Dirichlet）边界节点索引
    fixed_values : 对应给定值
    abs_err : 绝对收敛判据，``||F|| < abs_err``
    rel_err : 相对收敛判据，``||F|| < rel_err · ||F(A⁰)||``
    max_iter : 最大迭代数
    relaxation_iter : 前多少步取全步长
    relaxation_ratio : 之后使用的松弛因子 α
    callback : ``callback(iter, residual_norm)``，用于观测收敛过程

    Returns
    -------
    A : (N_node,) 解
    info : dict，含 ``n_iter`` / ``residual_norm`` / ``history``

    Notes
    -----
    边界条件的施加方式是**驱动行替换**：把残差与 Jacobian 中受约束的行替换成
    ``A = 给定值``，而不是消元。legacy 用 ``F_is_driven`` / ``F_mat_driven`` /
    ``J_driven`` / ``J_mat_driven`` 四个矩阵实现同一件事（``MMA10:2086-2110``）。

    松弛策略对应理论文档「计算步骤」与 legacy ``MMA10:2405`` 的注释：
    NR 在强非线性下会振荡，前 ``relaxation_iter`` 步用全步长，之后减半。
    """
    A = np.array(A0, dtype=float, copy=True)
    n_node = A.shape[0]
    history: list[float] = []

    if fixed_nodes is None:
        fixed_nodes = np.zeros(0, dtype=int)
    if fixed_values is None:
        fixed_values = np.zeros(0, dtype=float)
    fixed_nodes = np.asarray(fixed_nodes, dtype=int)
    fixed_values = np.asarray(fixed_values, dtype=float)

    A[fixed_nodes] = fixed_values

    n_iter = 0
    res_norm = np.inf
    f_scale = None                                    # 首次迭代的残差范数，作为相对判据的分母

    for n_iter in range(1, max_iter + 1):
        F, nu = residual(A, geom, S, nu_of_B, R)
        J = jacobian(A, geom, S, nu, dnu_dB)

        # --- 施加 Dirichlet：替换受约束的行（驱动行替换，不消元）---
        # 第 i 行改成 A_i - 给定值 = 0，Jacobian 第 i 行改成单位行。
        # 这样残差与 Jacobian 的维数不变，便于后续向量化。
        J = J.tolil()
        for i_node, val in zip(fixed_nodes, fixed_values):
            J.rows[i_node] = [i_node]
            J.data[i_node] = [1.0]
            F[i_node] = A[i_node] - val
        J = J.tocsr()

        res_norm = float(np.linalg.norm(F))
        history.append(res_norm)
        if callback is not None:
            callback(n_iter, res_norm)

        # 相对判据的分母：取首次迭代的残差范数。
        # ⚠ 为什么需要相对判据（这是本实现与 legacy 的一处有意差异）：
        #   残差 F = K·A − R 的量纲随 ν = 1/μ 与几何尺度缩放。本问题里
        #   ν ≈ 7.96e5、单元 Δ ≈ 4e-2，K 的量级 ~2e7，于是 F 的**浮点下限**
        #   （K·A 与 R 相消后的舍入残留）约 1e-10 —— 绝对容差 1e-12 永远达不到，
        #   Newton 会在解已经精确的情况下空转满 max_iter 步。
        #   legacy 用的是硬编码 `_solve_abs_err = 1e-3`（MMA10:2921），
        #   一旦改 current_amp、改单位或改网格尺度，这个数就失去意义。
        #   相对判据对所有尺度都成立。
        if f_scale is None:
            f_scale = res_norm if res_norm > 0.0 else 1.0
        if res_norm < abs_err + rel_err * f_scale:
            break

        dA = sp_sparse.linalg.spsolve(J, F)

        alpha = 1.0 if n_iter <= relaxation_iter else relaxation_ratio
        A = A - alpha * dA
        A[fixed_nodes] = fixed_values

    return A, {
        "n_iter": n_iter,
        "residual_norm": res_norm,
        "history": np.array(history),
    }
