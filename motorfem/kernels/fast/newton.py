"""残差 / Jacobian / Newton 的 legacy 保真搬移（内核 9/10/11 的 fast 层）。

本层是 ``field_error_vectorized``（``MMA10:2222-2257``）与 ``solve_field_vectorized``
（``:2380-2406``）的**逐位**搬移：容器、写入顺序、乘法结合顺序、函数边界全部照原样保留。
数学、18 项的含义、F4 讨论与实测事实写在 ``reference/newton.py`` 的模块 docstring 里，
本文件不重复。

------------------------------------------------------------------------------
为什么这一层「照抄」比「写清楚」重要
------------------------------------------------------------------------------

内核 8 的产出是**位置矩阵族**：位置（``K0_row/col``、``gswap1/2_mat`` 的 0/1 结构、
``Jabobi2_row/col``）与值（``K0_data``、``g0i_mat`` 的数据）分开存，下游靠
``csr((K0_data * (Kconst_mat @ γ), (K0_row, K0_col)))`` 一类式子重装矩阵。因此**任何**对
构造顺序、容器类型或布局的「整理」都会渗进下游的浮点结果。

------------------------------------------------------------------------------
三条容器级红线（本机 scipy 1.12 实测确认）
------------------------------------------------------------------------------

与内核 8 的 ``_T_temp[:, _index]`` 同类：看着无害、末位不同、**不报错**。

1. **``K = coo_matrix(...)`` 之后直接 ``K @ A``，不许「顺手」加 ``.tocsr()``。**
   实测 ``coo @ x`` 与 ``coo.tocsr() @ x`` **不逐位相同**（最大差 1.1e-16）——
   两者走不同的 matvec 内核。legacy 的 ``:2240-2243`` 正是先建 COO 再用它做 matvec。
2. **``K + Jabobi2_mat`` 不许改写成「合并两个 COO 的 data/row/col 后一次 tocsr」。**
   实测 ``coo + coo`` 的结果是 **csr** 且重复项**在那一步**就已求和（实测 24 → 19），
   即「各自先 ``tocsr()``（各自治内重复项求和）再相加」。合并后一次求和 = 不同的浮点
   累加顺序。（附带结论：``:2399`` 的 ``_Jacobi.tocsr()`` 是 no-op，因为 ``csr.tocsr()
   is self`` 为真；留着无害，但别以为它在做归一化。）
3. **四掩码合成是算术加权，不许改 ``np.where``。** ``0 * nan = nan`` 与四项加法的
   结合序都参与数值。见 ``reference/newton.py`` 的 ``blend_class_values``。

------------------------------------------------------------------------------
本层复用 ``reference`` 的四件东西（不是偷懒，是「无差异就不建第二份」）
------------------------------------------------------------------------------

- ``FieldMaterial`` / ``MaterialMasks`` 等数据结构（与内核 6/7/8 的分层约定一致）；
- ``blend_class_values``：**第二处「reference 必须与 fast 共用代码」的地方**
  （第一处是内核 8 的 ``_active_index``）。理由：它是全内核唯一「浮点结合序敏感 +
  nan 语义敏感」的算式；共用代码意味着 parity 抓不到它自身的错，故
  ``tests/kernels/test_newton.py`` 为它单列了两条解析测试（掩码逐类、``0*nan``）。
- ``newton_iterations``：Newton 循环体是标量循环 + 一次 ``spsolve``，**没有任何可
  向量化的部分**，按 ``kernels/README.md``「无差异就不建第二份」的判据只写一份
  （先例：``fast/magnetostatic.py`` 的 ``newton_solve``、内核 8 的 ``get_R_vector``）。
- ``element_field_material`` 的**两类非线性材料求值**：`nonlinear_material` 直接来自
  ``reference.material``；SIMP 那一类走 ``CachedMaterial``（与 reference **逐位相同**，
  实测省 2.24×，见 ``fast/material.py``）。

------------------------------------------------------------------------------
与 ``reference`` 的签名差异（bundle 型签名，见 reference 模块 docstring 第五节）
------------------------------------------------------------------------------

本层吃内核 8 的**出参**（``as_legacy_tuple`` 复原的 **22 元组**），reference 吃它的
**入参**。差异不可约，理由写在 reference 侧；这里只强调一点：
**本层保留 legacy 的函数边界** —— 内核 9 与内核 10 在 legacy 里本来就在**同一个函数**
里（``:2246-2255`` 是 ``field_error_vectorized`` 的内存块），一次返回 ``(F, B, Jacobi)``。
拆成两个函数是本仓库的结构选择（reference 侧为了能单独 FD 校验），不是 legacy 的结构。

------------------------------------------------------------------------------
红线
------------------------------------------------------------------------------

本层完整遵守 ``CLAUDE.md`` §3：不 ``sorted()`` 区域集合、不改 ``set`` 构造表达式、
不合并 ``for i_elem in range(N_element)`` 的嵌套顺序、不把 ``+=`` 改成 ``np.add.at``、
``lil`` 逐项赋值不换 COO，外加本文件开头那三条。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse as sp_sparse

from ... import solvers
from ..reference.material import nonlinear_material
from ..reference.newton import (
    FieldMaterial,
    MaterialMasks,
    blend_class_values,
    newton_iterations,
)
from .material import CachedMaterial, DensityState
from .sparsity import assemble_csr, assemble_csr_sum, project_dirichlet

__all__ = [
    "FieldMaterial",
    "MaterialMasks",
    "newton_iterations",     # noqa: F401  （公开接口一致性：两层的 __all__ 相同）
    "field_error_vectorized",
    "solve_field_vectorized",
]


# =============================================================================
# 残差 + Jacobian（内核 9 + 内核 10，legacy 的同一个函数）
# =============================================================================

def field_error_vectorized(
    _A_node: np.ndarray,
    K_vectorized: tuple,
    R: np.ndarray,
    *,
    n_node: int,
    area: np.ndarray,
    cached: CachedMaterial,
    state: DensityState,
) -> tuple:
    """残差 ``F``、磁密幅值 ``B``、Jacobian ``Jacobi``。对应 ``MMA10:2222-2257``。

    Parameters
    ----------
    _A_node : (N_node,) 当前矢量磁位
    K_vectorized : 22 元组
        ``reference.assembly.as_legacy_tuple(ops, driving)`` 的产出，顺序逐项照抄
        ``MMA10:2114-2118``。**下面那句解包是从 legacy 原样保留的**（设计文档 §4 的
        「核心兼容装置」），所以元组的项数与顺序都不能动。
    R : (N_node,) 源项
    n_node : int
        legacy 在这里读模块全局 ``N_node``（``:2241`` / ``:2251`` 的 ``shape=``）；
        本实现提成显式参数（``kernels/README.md`` 的「命名与签名约定」第 1 条）。
    area : (N_element,) 单元面积 Δ
        legacy 的 ``Aelem``（模块全局）。**已实测与内核 4 的 ``area`` 逐位相同。**
    cached : CachedMaterial
        ρ 相关量的缓存器。由调用方持有，以便校验 ``prepare_density`` 只被调了一次。
    state : DensityState
        ``cached.prepare_density(ρ)`` 的产出。**这是本函数与 legacy 唯一的一处形参差异**：
        legacy 在这里收 ``_rho_SIMP`` 并现算 ``SIMP_func``/``SIMP_dfunc``（``:2233``），
        本实现把「只依赖 ρ 的量」提进 ``state`` 缓存（逐位相同，实测 2.24×）。
        所以形参里没有 ρ —— 它已经折进 ``state`` 了。调用方是
        ``solve_field_vectorized``，那里持有 ρ 并负责制备 ``state``。

    Returns
    -------
    F : (N_node,) 残差（已做驱动行替换）
    B : (N_element,) **裁剪后**的磁密幅值（下界 1e-20）
    Jacobi : (N_node, N_node) csr
        **收敛那一轮**的对象，会被下游 ``spsolve(Jacobi.T, …)`` 消费（内核 13/14）。

    Notes
    -----
    两处与 legacy 的容器差异（都不影响数值）：

    1. 元组里的 ``F_mat_driven`` / ``J_mat_driven`` / ``Bxcoef_mat`` / ``Bycoef_mat``
       legacy 在返回前已 ``tocsr()``（``:2114-2118``），本实现同样；``J_driven``
       legacy 是 ``lil``、本实现是 ``csr``（对角 0/1，逐元素乘积无累加，逐位无差）。
    2. ``is_vaccum`` 等四个掩码 legacy 是 ``np.zeros(N_element)`` 的 float64 数组，
       本实现相同 —— **它是数组不是索引**，见 ``blend_class_values``。
    """
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    Bx = Bxcoef_mat @ _A_node
    By = Bycoef_mat @ _A_node
    B = np.sqrt(Bx ** 2 + By ** 2).clip(min=1e-20, max=np.inf)

    # ρ 相关量走缓存（与 reference 逐位相同，实测 2.24×）；纯铁芯那一类无 ρ，直接算
    mu_SIMP, dgamma_SIMP = cached.nonlinear(B, state)
    mu_non_SIMP, dgamma_non_SIMP = nonlinear_material(B, cached.curve)

    # ★ 四掩码算术加权 —— 与 reference 共用同一份实现（见模块 docstring）
    mu, dgamma = blend_class_values(
        mu_SIMP, dgamma_SIMP, mu_non_SIMP, dgamma_non_SIMP,
        MaterialMasks(
            is_vacuum=is_vaccum, is_simp=is_SIMP,
            is_non_simp=is_non_SIMP, is_linear_iron=is_linear_iron,
        ),
        cached.mu0, cached.mur_linear,
    )

    # ★ 红线 1：这里**不能**加 .tocsr()
    K_data = K0_data * (Kconst_mat @ (1 / mu))
    K = sp_sparse.coo_matrix((K_data, (K0_row, K0_col)),
                             shape=(n_node, n_node))

    F = K @ _A_node - R
    F = F * (1 - F_is_driven) + (F_mat_driven @ _A_node) * F_is_driven

    g0i_vec = g0i_mat @ _A_node
    const_vec = gconst_mat @ (1 / B / area * dgamma)
    gi_vec = g0i_vec * const_vec

    # ★ 内核 10：秩 1 外积的「交换矩阵」形式（数学见 reference 模块 docstring 第二节）
    Jabobi2_data = (gswap1_mat @ g0i_vec) * (gswap2_mat @ gi_vec)
    # ★ 红线 2：``K + Jabobi2_mat`` 的语义 = ``csr(K) + csr(Jabobi2)``（scipy 的
    #   ``coo + coo`` 内部就是两次 ``tocsr()`` 再相加），重复项在该步求和；
    #   **不许改写成「合并两个 COO 的 data/row/col 后一次 tocsr」**（会改浮点累加序）。
    #   ★ 性能优化（逐位一致，2026-09-30）：这里仍走「两个独立 csr 相加」，只把
    #   两次 ``tocsr()`` 换成**预计算稀疏结构**（``fast/sparsity.py``，1.05→0.10 ms/次）。
    #   排序/合并仍由 scipy 的 ``sum_duplicates`` 完成 ⇒ 与 ``coo.tocsr()`` 逐位一致
    #   （``tests/kernels/test_newton.py`` 有逐位断言钉死）。
    #   ★ 实测：``K0_row/col`` 与 ``Jabobi2_row/col`` 稀疏模式**逐位相同** ⇒ 相加可走
    #   「逐元素 data 相加」快速路径（与 ``csr + csr`` 逐位一致，12×）。不同结构自动回退。
    Jacobi = assemble_csr_sum(K0_row, K0_col, K_data,
                              Jabobi2_row, Jabobi2_col, Jabobi2_data, n_node)

    # 对称 Dirichlet（2026-09-24）：``(J_driven @ J_bulk) @ J_col_mask + J_mat_driven``
    # —— 列投影零齐次 Dirichlet 列（对角 1 由 J_mat_driven 自带），J 严格对称、解不变。
    Jacobi = project_dirichlet(Jacobi, J_driven, J_col_mask, J_mat_driven)

    return F, B, Jacobi


# =============================================================================
# Newton + 分段松弛（内核 11）
# =============================================================================

def solve_field_vectorized(
    _rho_SIMP: np.ndarray,
    K_vectorized: tuple,
    R: np.ndarray,
    _A0: Optional[np.ndarray] = None,
    _solve_abs_err: float = 1e-6,
    max_iter: int = 100,
    iter_relaxation: int = 20,
    relaxation_ratio: float = 0.5,
    output: bool = True,
    *,
    n_node: int,
    area: np.ndarray,
    material: FieldMaterial,
    solver: Optional[str] = None,
    elimination: Optional[object] = None,
    residual_tol: Optional[float] = None,
) -> tuple:
    """Newton 求解。对应 ``MMA10:2380-2406``。

    Parameters
    ----------
    _rho_SIMP, K_vectorized, R, _A0, _solve_abs_err, max_iter, iter_relaxation,
    relaxation_ratio, output :
        逐字照抄 legacy 的形参名与默认值（``:2380-2382``）。⚠ ``_solve_abs_err`` 的
        默认值 ``1e-6`` 在 MMA10 里**从未被实际用到**，生产调用传的是 ``1e-3``
        （``:2921`` / ``:2980``）—— 两者不同**是接口的一部分**，不要「统一」。
    n_node : int
        legacy 读模块全局，本实现提成参数。
    area : (N_element,) 单元面积 Δ
    material : FieldMaterial
        材料模型的参数束；``CachedMaterial`` 由本函数构造。

    Returns
    -------
    A : (N_node,) 解
    B : (N_element,) **收敛那一轮**的磁密幅值（判停在 ``spsolve`` 之前，``:2396``）
    Jacobi : csr
        **收敛那一轮**的 Jacobian，同一个对象（不重建、不拷贝）。

    Notes
    -----
    ``_A0`` 是**跨 MMA 迭代的时间点级热启动缓存**：在 legacy 里它由模块全局
    ``A_node_last``（``:2910``）给出。本实现不搬运那个全局（它从不被清空 ⇒ 跨 run
    不可复现；采样开启时读写键空间不自洽 ⇒ 静默退回冷启动。两条已登记进
    ``docs/重构设计文档.md`` §11），只保留 legacy 已有的 ``_A0`` 形参。

    ``prepare_density`` 在本函数里**只被调用一次** —— ρ 在整个 Newton 求解中固定，
    而 legacy 每轮都重算 ``SIMP_func``/``SIMP_func``（``:2233``）。这是纯优化、
    零数值代价（``fast/material.py`` 实测逐位相同）；函数末尾有断言守住它。
    """
    cached = CachedMaterial(
        material.curve, material.simp, material.mu0, material.mur_linear,
        linear_simp=bool(getattr(material, "linear_simp", False)),
    )
    state = cached.prepare_density(_rho_SIMP)

    def step_fn(_A_node):
        return field_error_vectorized(
            _A_node, K_vectorized, R,
            n_node=n_node, area=area, cached=cached, state=state,
        )

    # Newton 循环是**同一份实现**（reference 的 newton_iterations）—— 见模块 docstring
    A_node, B_elem, Jacobi, _history = newton_iterations(
        step_fn, _A0, n_node=n_node,
        max_iter=max_iter, solve_abs_err=_solve_abs_err,
        iter_relaxation=iter_relaxation, relaxation_ratio=relaxation_ratio,
        output=output, solver=solver, elimination=elimination,
        residual_tol=residual_tol,
    )

    # ρ 固定的前提被违反了就会在这里炸，而不是静默给出偏慢的结果
    assert cached.n_prepare == 1, (
        f"prepare_density 被调了 {cached.n_prepare} 次；Newton 循环里 ρ 固定，应为 1 次"
    )
    return A_node, B_elem, Jacobi
