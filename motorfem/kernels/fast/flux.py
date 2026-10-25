"""内核 15 的 legacy 保真搬移（fast 层）。

本层是下面三段的**逐位**搬移，容器、写入顺序、乘法结合顺序、函数边界全部照原样：

    field_error_vectorized_fronzen_appearance_conductivity   MMA10:2300-2333
    DTorque_flux_vectorized                                  MMA10:2336-2377
    solve_field_vectorized_frozen_appearance_conductivity     MMA10:2409-2430

数学、公式出处、解析恒等式与 D4 的登记写在 ``reference/flux.py`` 的模块 docstring 里，
本文件不重复。**电感/磁链/转矩的求值与 ``k_e`` 与 ``reference`` 共用同一份实现**
（它们本来就是纯 numpy，没有可向量化的部分；先例：内核 12/13/14 的 ``torque_quadratic``
与 ``torque_derivative_wrt_rho``）。

------------------------------------------------------------------------------
与 ``fast/newton.field_error_vectorized`` 的关系：照搬一份，再用测试钉死相等
------------------------------------------------------------------------------

实测逐行 diff：冻结残差版与 ``field_error_vectorized``（``:2222-2257``）**只差两件事**
（``B`` 的来源、返回顺序），其余每一行逐字相同。``reference`` 侧因此把缝切在
``element_field_material`` / ``field_system`` 两个具名函数上、**一行数学都不抄第二遍**。

本层**不那样做**，理由是「迁移可核对性」：向量化版的整段语句序列是位级对齐的对象，
把公共体抽出来会让「搬移是否原样」失去可核对的形式。替代方案更强：
``tests/kernels/test_flux.py`` 用一条**逐位**断言把两份钉死相等 ——

    取 ``B = flux_density(A)`` 时，``field_error_vectorized_frozen(A, B, …)`` 的
    ``F``/``Jacobi`` 必须与 ``field_error_vectorized(A, …)`` **逐位**相同

共用代码抓不到自身的 bug（``blend_class_values`` 就是先例，它另有两条独立解析测试
兜底），钉死重复项反而更强：哪天有人在任一侧改了一个字符，这条会立刻失败。

------------------------------------------------------------------------------
三处照抄的细节
------------------------------------------------------------------------------

1. **``B`` 是形参**（``:2307`` 的 ``B = _B_elem_frozen``）：本层不内部偷算
   ``Bxcoef_mat @ A``。测试里有一条「把 B 换成常数组，μ 必须跟着变」的守卫。
2. **``coef`` 的算式形状**（``:2367``）::

       g0i_vec * (gconst_mat @ (_k_vectorized * dgamma / (B * Aelem)))

   ``(B * Aelem)`` 的括号照抄（``CLAUDE.md`` §3 的算式形状纪律）。装配成
   ``(1, N_node)`` 的 COO 再 ``.toarray().flatten()`` —— 重复项求和顺序与 legacy 相同。
3. **显式项不清零**（``:2374-2377``）：只对 ``dTflux_p_dgamma1`` 做花式索引赋值，
   然后 ``+ dTflux_p_dgamma2_vectorized``。所以结果在 ``elements_to_compute`` 之外**非零**。

------------------------------------------------------------------------------
红线
------------------------------------------------------------------------------

本层完整遵守 ``CLAUDE.md`` §3：不 ``sorted()`` 区域集合、不改 ``set`` 构造表达式、
不合并 ``for i_elem in range(N_element)`` 的嵌套顺序、不把 ``+=`` 改成 ``np.add.at``、
``lil`` 逐项赋值不换 COO。本内核**没有** ``set`` 迭代（两处循环都是 ``range``），
所以 F4 在这里不活跃。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from ... import solvers
from ..reference.flux import (
    FrozenFluxAnalysis,
    flux_linkage,
    flux_torque,
    flux_torque_kernel,
    inductance,
)
from ..reference.material import nonlinear_material, nonlinear_material_topo_opti
from ..reference.newton import FieldMaterial, MaterialMasks, blend_class_values
from .material import CachedMaterial, DensityState
from .sparsity import assemble_csr, assemble_csr_sum, project_dirichlet

# 两层名字的对应关系（两份 ``__all__`` 不逐字相同，与 ``fast/newton.py`` /
# ``fast/torque.py`` 同一形态：本层只声明自己的公开接口 + 转出共用的那几个）:
#
#   reference                       fast
#   ------------------------------  ---------------------------------------
#   frozen_field_system             field_error_vectorized_frozen
#   solve_frozen_field              solve_field_vectorized_frozen
#   flux_density_components         （内联在 solve 里：Bxcoef_mat @ A）
#   frozen_flux_analysis            frozen_flux_analysis（bundle 型签名）
#   flux_torque_derivative          flux_torque_derivative（bundle 型签名）
#   inductance / flux_linkage / flux_torque / flux_torque_kernel
#                                   → **共用 reference 的同一份**（纯 numpy）
__all__ = [
    "FieldMaterial",
    "MaterialMasks",
    "FrozenFluxAnalysis",
    # ---- 纯 numpy，两层共用（本内核的求值没有可向量化的部分）----
    "inductance",
    "flux_linkage",
    "flux_torque",
    "flux_torque_kernel",
    # ---- 本层独有的向量化实现 ----
    "field_error_vectorized_frozen",
    "solve_field_vectorized_frozen",
    "frozen_flux_analysis",
    "flux_torque_derivative",
]


# =============================================================================
# 冻结视在磁导率的残差 + Jacobian（legacy 的同一个函数）
# =============================================================================

def field_error_vectorized_frozen(
    _A_node: np.ndarray,
    _B_elem_frozen: np.ndarray,
    K_vectorized: tuple,
    R: np.ndarray,
    *,
    n_node: int,
    area: np.ndarray,
    cached: CachedMaterial,
    state: DensityState,
) -> tuple:
    """残差 ``F``、Jacobian ``Jacobi``、磁导率 ``mu``。对应 ``MMA10:2300-2333``。

    与 ``fast/newton.field_error_vectorized`` 的差别**只有两处**（见模块 docstring）:
    ``B`` 从 ``_B_elem_frozen`` 来、返回值第二项是 ``mu`` 而不是 ``B``。

    Parameters
    ----------
    _A_node : (N_node,) 当前矢量磁位
    _B_elem_frozen : (N_element,) **冻结**的磁密幅值（内核 11 返回的那个，**已裁剪**）
    K_vectorized : 22 元组
        ``reference.assembly.as_legacy_tuple(ops, driving)`` 的产出。**下面那句解包是从
        legacy 原样保留的**，项数与顺序都不能动。
    R : (N_node,) 源项
    n_node, area, cached, state :
        与 ``fast/newton.field_error_vectorized`` 同名同义（``area`` 是 legacy 的
        ``Aelem``，``cached``/``state`` 是 ρ 相关量的缓存器，**逐位相同**）。

    Returns
    -------
    F : (N_node,) 残差（已做驱动行替换）
    Jacobi : (N_node, N_node) csr（已做驱动行替换）
    mu : (N_element,) 冻结的磁导率

    Notes
    -----
    ⚠ ``A = 0`` 时饱和项恒为零，但本函数**照样把 ``g0i``/``gi``/``gswap`` 那一串算一遍**
    （结果是精确的 0）—— 与 legacy 同。不要「顺手」短路掉：那样 ``F``/``Jacobi`` 与
    内核 9/10 的计算路径就不再是同一条，钉死两份相等的测试也就失去意义。
    """
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    # ★ 与内核 9 的唯一差别之一：B 从形参来（legacy :2307）
    B = _B_elem_frozen

    mu_SIMP, dgamma_SIMP = cached.nonlinear(B, state)
    mu_non_SIMP, dgamma_non_SIMP = nonlinear_material(B, cached.curve)

    # ★ 四掩码算术加权 —— 与 reference 共用同一份实现（见内核 9 的说明）
    mu, dgamma = blend_class_values(
        mu_SIMP, dgamma_SIMP, mu_non_SIMP, dgamma_non_SIMP,
        MaterialMasks(
            is_vacuum=is_vaccum, is_simp=is_SIMP,
            is_non_simp=is_non_SIMP, is_linear_iron=is_linear_iron,
        ),
        cached.mu0, cached.mur_linear,
    )

    # ★ 红线 1（内核 9）：这里**不能**加 .tocsr()
    K_data = K0_data * (Kconst_mat @ (1 / mu))
    K = sp_sparse.coo_matrix((K_data, (K0_row, K0_col)),
                             shape=(n_node, n_node))

    F = K @ _A_node - R
    F = F * (1 - F_is_driven) + (F_mat_driven @ _A_node) * F_is_driven

    g0i_vec = g0i_mat @ _A_node
    const_vec = gconst_mat @ (1 / B / area * dgamma)
    gi_vec = g0i_vec * const_vec

    # ★ 红线 2（内核 9）：``K + Jabobi2_mat`` 语义 = ``csr(K) + csr(Jabobi2)``
    #   （scipy 的 ``coo + coo`` 内部即两次 ``tocsr()`` 再相加），重复项在该步求和。
    #   ★ 性能优化（逐位一致，2026-09-30）：两次 ``tocsr()`` 换预计算稀疏结构
    #   （``fast/sparsity.py``）—— 与 ``fast/newton.field_error_vectorized`` 同一处
    #   优化（冻结版与它只差 B 来源与返回顺序，见模块 docstring）。
    Jabobi2_data = (gswap1_mat @ g0i_vec) * (gswap2_mat @ gi_vec)
    #   ★ 同 newton：两结构逐位相同 ⇒ 走逐元素相加快速路径（不同结构自动回退）
    Jacobi = assemble_csr_sum(K0_row, K0_col, K_data,
                              Jabobi2_row, Jabobi2_col, Jabobi2_data, n_node)

    # 对称 Dirichlet（2026-09-24）：列投影零齐次 Dirichlet 列，J 严格对称、解不变。
    Jacobi = project_dirichlet(Jacobi, J_driven, J_col_mask, J_mat_driven)

    # ★ 与内核 9 的唯一差别之二：返回顺序（legacy :2333 的 ``return F, Jacobi, mu``）
    return F, Jacobi, mu


# =============================================================================
# 一次线性冻结求解
# =============================================================================

def solve_field_vectorized_frozen(
    _rho_SIMP: np.ndarray,
    _B_elem_frozen: np.ndarray,
    K_vectorized: tuple,
    R: np.ndarray,
    *,
    n_node: int,
    area: np.ndarray,
    material: FieldMaterial,
    solver: Optional[str] = None,
    elimination=None,
) -> tuple:
    """冻结视在磁导率的一次线性求解。对应 ``MMA10:2409-2430``。

    ⚠ **阶段 8 的分解复用不适用于本函数**：``_Jacobi`` 是 **CSR**，``splu`` 内部
    强制转 CSC，与 ``spsolve(CSR, ·)`` 的 SuperLU CSR 路径**不**逐位一致（实测差
    1 ULP，见设计文档阶段 8）。``splu`` 复用只对 **CSC 矩阵**（如 ``J.T``）成立。

    ⚠ **本函数保持单右端形态不变**（供独立调用/测试）；d/q 两次求解的合并版在
    ``frozen_flux_analysis`` 里实现（2026-09-30，一次多列 spsolve，逐位一致）。

    ⚠ **这不是 Newton 迭代**：``A₀ = 0`` 起步算一次，解一次 ``spsolve``，
    ``A = A₀ − dA``，**不判收敛、不循环**（legacy 的形态，``:2417-2424``）。
    ``A=0`` 处 ``dν/dB`` 项恒为零 ⇒ 系统本来就是线性的 ⇒ 一步即精确解。

    Parameters
    ----------
    _rho_SIMP, _B_elem_frozen, K_vectorized, R :
        逐字照抄 legacy 的形参名与顺序（``:2409``）。
    n_node : int
        legacy 在该函数里读模块全局 ``N_node``（``:2417`` 的 ``np.zeros((N_node,))``）。
    area : (N_element,) 单元面积 Δ
    material : FieldMaterial
        ``CachedMaterial`` 由本函数构造（与 ``fast/newton.solve_field_vectorized`` 同）。

    Returns
    -------
    Bx, By : (N_element,) 该源下的磁密分量（**未裁剪**，由 ``Bxcoef_mat @ A`` 给出）
    mu : (N_element,) 冻结的磁导率

    Notes
    -----
    末尾的 ``Bx``/``By`` 用的是**矩阵形式**（``:2427-2428``），不是逐单元循环 ——
    这就是本层与 ``reference.solve_frozen_field`` 的差别所在（后者走
    ``flux_density_components`` 的逐单元循环），两者只差求和顺序。
    """
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    cached = CachedMaterial(
        material.curve, material.simp, material.mu0, material.mur_linear
    )
    state = cached.prepare_density(_rho_SIMP)

    # 计算冻结视在磁导率情况下的磁场分布   （legacy :2416 原注释）
    _A_node = np.zeros((int(n_node),))

    _F_node, _Jacobi, _mu = field_error_vectorized_frozen(
        _A_node, _B_elem_frozen, K_vectorized, R,
        n_node=n_node, area=area, cached=cached, state=state,
    )

    _Jacobi = _Jacobi.tocsr()
    # 求解会话：一次构造（传入外部会话则复用），一次 solve。
    session, owned = solvers.as_session(solver)
    try:
        if elimination is not None:
            # 消元：Tᵀ·J·T + Tᵀ·F，解后恢复
            dA = elimination.restore(session.solve(
                elimination.eliminate_matrix(_Jacobi),
                elimination.eliminate_rhs(_F_node)))
        else:
            dA = session.solve(_Jacobi, _F_node)
        _A_node = _A_node - dA

        # 计算此时的磁场分布
        _Bx = Bxcoef_mat @ _A_node
        _By = Bycoef_mat @ _A_node

        return _Bx, _By, _mu
    finally:
        if owned:
            session.close()


# =============================================================================
# 整条链：两次冻结解 → 电感 → 磁链 → 转矩
# =============================================================================

def frozen_flux_analysis(
    _B_elem_frozen: np.ndarray,
    _rho_SIMP: np.ndarray,
    K_vectorized: tuple,
    R_d: np.ndarray,
    R_q: np.ndarray,
    I_d: float,
    I_q: float,
    *,
    n_node: int,
    area: np.ndarray,
    material: FieldMaterial,
    pole_pairs: int,
    parts: int,
    solver: Optional[str] = None,
    elimination=None,
) -> FrozenFluxAnalysis:
    """``fast`` 层的整条冻结链。对应 ``MMA10:2937-2957`` 的 worker 段。

    参数对应关系：``reference.frozen_flux_analysis`` 吃 ``FieldGeometry`` +
    ``driving``，本层吃 22 元组（bundle 型签名，理由与内核 9/10/11 同）。

    Notes
    -----
    legacy 调两次 ``solve_field_vectorized_frozen_appearance_conductivity``、第二次的
    ``mu_elem`` **覆盖**第一次（``:2938-2942``）。两次的 ``μ`` 逐位相同（同一个
    ``B_frozen``、同一个 ``ρ``）。

    ★ 性能优化（2026-09-30 实测）：本实现改为 **Jacobi 只建一次、消元只做一次、
    d/q 两次求解合并成一次多列 spsolve**（superlu 后端），数值与逐列求解
    **逐位一致**（``np.array_equal`` + 真实网格 max|Δ|=0.0，见函数体内注释）。
    ``splu`` 复用仍不适用于 CSR 的 ``_Jacobi``（1 ULP），所以走「同一矩阵多列
    spsolve」而非「分解复用」—— 两者对逐位一致的要求不同（多列 spsolve 是
    SuperLU 在同一次分解内逐列回代，结果与逐列调用一致）。

    电感/磁链/转矩走与 ``reference`` **共用**的实现（纯 numpy，无向量化余地）。
    """
    # ★ 性能优化（2026-09-30 实测，逐位一致）：d/q 两次冻结求解共用同一个
    # ``_Jacobi``（同一冻结 B、同一 ρ、同一 K_vectorized —— 已在真实网格上实测
    # ``np.array_equal`` 为真）。legacy 调两次 ``solve_field_vectorized_frozen``
    # 每次都重建 Jacobi / 重消元 / 重新分解。这里改为：Jacobi 只建一次、
    # 消元只做一次、两次 spsolve 合并成**一次多列 spsolve**（``spsolve(Ed,
    # [Fd, Fq])`` 与逐列 ``spsolve`` 逐位一致，实测 max|Δ| = 0.0 —— SuperLU 在
    # 同一矩阵上分解一次、逐列回代）。mu 只算一次（d/q 两次逐位相同）。
    # ⚠ 多列批量只对 scipy ``spsolve``（superlu 后端）用：``session.solve`` 的
    # 因子缓存走矩阵对象身份，批量右端拼列会失去逐位一致（阶段 8a 纪律）。
    #   该优化改变函数边界（不再逐次调用 solve_field_vectorized_frozen），
    #   数值路径与 legacy 逐位一致（本文件测试钉死）。
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    cached = CachedMaterial(
        material.curve, material.simp, material.mu0, material.mur_linear
    )
    state = cached.prepare_density(_rho_SIMP)

    # 冻结视在磁导率情况下的磁场分布   （legacy :2416 原注释，同 solve_field_vectorized_frozen）
    _A_node = np.zeros((int(n_node),))
    _F_d, _Jacobi, _mu = field_error_vectorized_frozen(
        _A_node, _B_elem_frozen, K_vectorized, R_d,
        n_node=n_node, area=area, cached=cached, state=state,
    )
    _F_q, _J2, _m2 = field_error_vectorized_frozen(
        _A_node, _B_elem_frozen, K_vectorized, R_q,
        n_node=n_node, area=area, cached=cached, state=state,
    )
    _Jacobi = _Jacobi.tocsr()

    session, owned = solvers.as_session(solver)
    try:
        if elimination is not None:
            # 消元：Tᵀ·J·T + Tᵀ·F，解后恢复（与 solve_field_vectorized_frozen 同路径）
            Ed = elimination.eliminate_matrix(_Jacobi)
            Fd = elimination.eliminate_rhs(_F_d)
            Fq = elimination.eliminate_rhs(_F_q)
            if session.backend == solvers.SUPERLU:
                X = sp_sparse_linalg.spsolve(Ed, np.column_stack([Fd, Fq]))
                x_d, x_q = X[:, 0], X[:, 1]
            else:
                x_d = session.solve(Ed, Fd)
                x_q = session.solve(Ed, Fq)
            dA_d = elimination.restore(x_d)
            dA_q = elimination.restore(x_q)
        else:
            if session.backend == solvers.SUPERLU:
                X = sp_sparse_linalg.spsolve(_Jacobi, np.column_stack([_F_d, _F_q]))
                x_d, x_q = X[:, 0], X[:, 1]
            else:
                x_d = session.solve(_Jacobi, _F_d)
                x_q = session.solve(_Jacobi, _F_q)
            dA_d = x_d
            dA_q = x_q

        _A_d = _A_node - dA_d
        _A_q = _A_node - dA_q

        # 计算此时的磁场分布（同 solve_field_vectorized_frozen :2427-2428）
        Bx_d = Bxcoef_mat @ _A_d
        By_d = Bycoef_mat @ _A_d
        Bx_q = Bxcoef_mat @ _A_q
        By_q = Bycoef_mat @ _A_q
    finally:
        if owned:
            session.close()

    L_dd, L_dq, L_qq = inductance(Bx_d, By_d, Bx_q, By_q, _mu, area)
    psi = flux_linkage(L_dd, L_dq, L_qq, I_d, I_q, parts)
    torque = flux_torque(psi, I_d, I_q, pole_pairs)

    out = FrozenFluxAnalysis(
        Bx_d=Bx_d, By_d=By_d, Bx_q=Bx_q, By_q=By_q, mu=_mu,
        L_dd=L_dd, L_dq=L_dq, L_qq=L_qq, psi=psi, torque=torque,
    )
    # ★ 2026-10-14：挂上**诊断用**中间量（`object.__setattr__` 不动 dataclass 定义
    #   ⇒ golden 契约不变）。用于 `tools/probe_chain_fd.py` 在导数传递链的**每一环**
    #   上做独立 FD 验证（用户要求：不看终点，看中间量）。
    #   - `_A_d`/`_A_q`：两次冻结线性解（`J_frozen·A = R`）
    #   - `_J_frozen` ：`K(μ_frozen)` + 驱动行
    object.__setattr__(out, "_A_d", _A_d)
    object.__setattr__(out, "_A_q", _A_q)
    object.__setattr__(out, "_J_frozen", _Jacobi)
    return out


# =============================================================================
# 磁链转矩对 γ 的导数
# =============================================================================

def flux_torque_derivative(
    _A_node_frozen: np.ndarray,
    _rho_SIMP: np.ndarray,
    _Bx_elem_alpha: np.ndarray,
    _By_elem_alpha: np.ndarray,
    _Bx_elem_beta: np.ndarray,
    _By_elem_beta: np.ndarray,
    _I_alpha: float,
    _I_beta: float,
    _Jacobi,
    elements_to_compute,
    K_vectorized: tuple,
    *,
    n_node: int,
    n_element: int,
    area: np.ndarray,
    material: FieldMaterial,
    pole_pairs: int,
    parts: int,
    factor=None,
    elimination=None,
) -> np.ndarray:
    """``dT_flux/dγ``。对应 ``MMA10:2336-2377``，逐字搬移。

    ``factor`` : 可选 ``splu(Jacobi.T)`` 的 SuperLU 因子 —— 阶段 8 的分解复用
    （与 Arkkio/VD 的伴随解共用同一个 ``J`` 的转置因子）。默认 ``None`` 走原
    ``spsolve``；传入时用 ``factor.solve(rhs)`` 逐个右端回代 —— 两者**逐位一致**
    （实测，见设计文档阶段 8）。⚠ 禁止把多个右端拼成矩阵一次 ``factor.solve(B)``。

    Parameters
    ----------
    _A_node_frozen : (N_node,) **非线性**工作点的矢量磁位（legacy 的
        ``_A_node_frozen``）。它同时是 ``∂F/∂γ`` 的收缩对象与 ``B``/``dν/dB`` 的来源。
    _rho_SIMP : (N_element,) 设计变量
    _Bx_elem_alpha, _By_elem_alpha : (N_element,)
        d 轴（legacy 的 ``alpha``）冻结解的磁密分量。
    _Bx_elem_beta, _By_elem_beta : (N_element,)
        q 轴（legacy 的 ``beta``）冻结解的磁密分量。
    _I_alpha, _I_beta : float
        d/q 轴电流（legacy 的 ``_I_d``/``_I_q``）。
    _Jacobi : 内核 11 返回的**同一个对象**（会被 ``.T`` 求伴随）
    elements_to_compute : 可迭代 int
    K_vectorized : 22 元组。**下面那句解包是从 legacy 原样保留的。**
    n_node : int
        legacy 在 ``:2369`` 的 ``shape=(1, N_node)`` 里读模块全局。
    n_element : int
        单元数（22 元组里没有它）；legacy 读模块全局 ``N_element``。
    area : (N_element,) 单元面积 Δ（legacy 的 ``Aelem``）
    material : FieldMaterial
        ⚠ 这里**不用** ``CachedMaterial``：本函数每次时间点只调一次、不在 Newton 循环里，
        缓存的收益为零，所以照 legacy 直接调 ``nonlinear_material_topo_opti``
        （与 ``field_error_vectorized_frozen`` 的选择不同，那里是循环内）。
    pole_pairs, parts : int

    Returns
    -------
    (N_element,) ndarray，``elements_to_compute`` 之外**非零**（显式项）

    ⚠ **返回值与真值符号相反**（已确认缺陷 D4，见设计文档 §12）。**照搬 legacy，
    不在这里翻号** —— legacy 的消费者 ``dg_torque = -mean_Tflux_p_drho``（``:3202``）
    会把它翻回来。修正必须两侧同时做。

    Notes
    -----
    ``_k_vectorized`` 与 ``reference.flux_torque_kernel`` 是**同一个数组表达式**
    （乘除次序逐字相同），所以直接调用共用实现；测试里有逐位断言钉住这一点。
    """
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    Bx = Bxcoef_mat @ _A_node_frozen
    By = Bycoef_mat @ _A_node_frozen
    B = np.sqrt(Bx ** 2 + By ** 2).clip(min=1e-20, max=np.inf)

    # 只需要 dgamma（mu 被 legacy 算了但没用）—— 与 field_error 那一支的差别
    mu_SIMP, dgamma_SIMP = nonlinear_material_topo_opti(
        B, _rho_SIMP, material.curve, material.simp)
    mu_non_SIMP, dgamma_non_SIMP = nonlinear_material(B, material.curve)

    _mu, dgamma = blend_class_values(
        mu_SIMP, dgamma_SIMP, mu_non_SIMP, dgamma_non_SIMP,
        MaterialMasks(
            is_vacuum=is_vaccum, is_simp=is_SIMP,
            is_non_simp=is_non_SIMP, is_linear_iron=is_linear_iron,
        ),
        material.mu0, material.mur_linear,
    )

    g0i_vec = g0i_mat @ _A_node_frozen

    dK_p_dgamma_A_vectorized = sp_sparse.coo_matrix(
        (g0i_vec, (dK_p_dgamma_A_row, dK_p_dgamma_A_col)),
        shape=(n_element, n_node),
    )
    _k_vectorized = flux_torque_kernel(
        _Bx_elem_alpha, _By_elem_alpha, _Bx_elem_beta, _By_elem_beta,
        _I_alpha, _I_beta, area, pole_pairs=pole_pairs, parts=parts,
    )

    # ★ 算式形状照抄 legacy :2367（含 (B * Aelem) 的括号）
    coef_torque_flux_vectorized = sp_sparse.coo_matrix(
        (g0i_vec * (gconst_mat @ (_k_vectorized * dgamma / (B * area))),
         (np.zeros(len(dK_p_dgamma_A_col)), dK_p_dgamma_A_col)),
        shape=(1, n_node),
    ).toarray().flatten()
    dTflux_p_dgamma2_vectorized = _k_vectorized

    if elimination is not None:
        # 消元（同 adjoint_torque_derivative）：coef 经 Tᵀ 收缩、λ_free 解
        # J_freeᵀ、dK_p_dgamma_A 经 T 投影。
        coef_free = elimination.eliminate_rhs(coef_torque_flux_vectorized)
        lam = (factor.solve(coef_free) if factor is not None
               else sp_sparse_linalg.spsolve(_Jacobi.T, coef_free))
        _result = -(dK_p_dgamma_A_vectorized @ elimination.T) @ lam
    else:
        lam = (factor.solve(coef_torque_flux_vectorized) if factor is not None
               else sp_sparse_linalg.spsolve(_Jacobi.T, coef_torque_flux_vectorized))
        _result = -dK_p_dgamma_A_vectorized @ lam

    dTflux_p_dgamma1_vectorized = np.zeros(n_element)
    dTflux_p_dgamma1_vectorized[list(elements_to_compute)] = \
        _result[list(elements_to_compute)]

    # D4 修正（2026-09-23）：legacy 返回值与真值符号相反（fd/legacy = −1.0000）。
    # 翻号 ⇒ 函数语义即真导数。见 reference/flux.py 本函数 docstring 与 §12 的 D4。
    return -(dTflux_p_dgamma1_vectorized + dTflux_p_dgamma2_vectorized)
