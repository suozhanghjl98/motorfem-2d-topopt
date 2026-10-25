"""Arkkio 转矩系数的 legacy 保真搬移（内核 12 的 fast 层）。

本层是 ``get_Arkkio_coef``（``MMA10:666-701``）的**逐位**搬移：``set`` 迭代顺序、
``lil`` 逐项 ``+=``、标量缩放的时机、整体对称化的写法全部照原样。数学、公式出处、
以及那处 ★因子待核对★ 写在 ``reference/torque.py`` 的模块 docstring 里，本文件不重复。

------------------------------------------------------------------------------
★ 本内核的 F4 是**原始案例**（设计文档第二节）
------------------------------------------------------------------------------

``get_Arkkio_coef`` 遍历的是 ``for i_elem in element_rotor_gap:``，而 ``element_rotor_gap``
是一个 **``set``**（``MMA10:258``）；循环体里 ``lil_matrix`` 的 ``+=`` 逐项累加。
⇒ 同一个 ``(i, j)`` 位置的**浮点加数顺序**由 set 的迭代顺序决定 ⇒ 末位不同。
这正是设计文档 F4 那条（「位级一致的最大威胁不是算法，是 set 的迭代顺序」）的出处。

**三条红线**（都不许「整理」）：

1. **``gap_elements`` 必须是真正的 ``set``，且构造表达式与 legacy 逐字相同**
   （``MMA10:258``）::

       element_rotor_gap = set([_elem[0] for _elem in element_S if _elem[4] in entity_rotor_gap])

   本函数带一道显式守卫（见下），因为**``frozenset`` 与 ``set`` 的迭代顺序不保证一致**
   —— 实测 500 组随机整数集合里有 **176 组**顺序不同。把 ``set`` 换成 ``frozenset``
   （或反过来）会静默改变末位。⚠ 内核 8 的 ``TopologyContext.element_rotor_gap``
   是个 ``frozenset``，它在那里**只做成员测试**所以无害；**不要**把它传进来迭代。
2. **不许 ``sorted()``、不许 ``list(...)``、不许换容器。** 循环头就是
   ``for i_elem in gap_elements:``。
3. **``lil`` 的逐项 ``+=`` 不许改成 COO 组装**（``CLAUDE.md`` §3 红线 6）。
   COO 的重复项求和顺序与 lil 的逐次 ``+=`` 不同。

------------------------------------------------------------------------------
两处已实测的容器事实（都不影响数值，写在这里免得后人重测）
------------------------------------------------------------------------------

- ``lil @ x`` 与 ``csr @ x`` **逐位相同**（实测 0 位差），所以返回 csr 是安全的。
- legacy 的 ``_Torque_Arkkio_coef = (_M + _M.T) / 2`` 里，``lil + lil.T`` 的结果
  numpy/scipy 给的是 **``csr``** ⇒ ``Torque_Arkkio_coef`` 在 legacy 里**本来就是 csr**，
  不是 lil。所以本层与 legacy 在容器上也一致。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from ..reference.torque import (
    ArkkioGeometry,
    ElementTransformWithDerivative,
    torque_quadratic,          # noqa: F401  （公开接口一致性：两层的 __all__ 相同）
    torque_derivative_wrt_rho,  # noqa: F401  （三处重复链式环节的公共层，见 reference）
)

__all__ = [
    "ArkkioGeometry",
    "ElementTransformWithDerivative",
    "arkkio_coefficient",
    "torque_quadratic",
    "arkkio_torque",
    "transformation_matrix_v2",
    "vd_coefficient",
    "vd_torque",
    "adjoint_torque_derivative",
    "torque_derivative_wrt_rho",
]


def arkkio_coefficient(geom: ArkkioGeometry) -> sp_sparse.csr_matrix:
    """Arkkio 转矩矩阵。对应 ``MMA10:666-697``，逐字搬移。

    Parameters
    ----------
    geom : ArkkioGeometry
        ⚠ ``gap_elements`` 必须是**真正的 ``set``**（见模块 docstring 红线 1）；
        传 ``frozenset`` 会被显式拒绝，因为它会静默改变累加顺序。

    Returns
    -------
    (N_node, N_node) csr

    Notes
    -----
    与 legacy 的三处写法对应关系::

        legacy                                   本实现
        --------------------------------------   --------------------------------
        for i_elem in element_rotor_gap:         for i_elem in geom.gap_elements:
        _node_gap = node_gap_rotor               rotor_nodes（成员测试用）
        element_S[i_elem][1:-1]                  tri[i_elem]
        node[node_i, 2] / node[node_i, 1]        xy[node_i, 1] / xy[node_i, 0]
        _Torque_Arkkio_coef[.., ..] +=           _M[.., ..] +=
        *= 1/mu0 * parts / (0.25e-3/4)           *= 1/mu0 * parts / (gap/parts)
        (M + M.T)/2                              (M + M.T)/2
    """
    gap_elements = geom.gap_elements
    if isinstance(gap_elements, frozenset):
        raise TypeError(
            "gap_elements 必须是与 legacy 同构的 set，不能是 frozenset —— "
            "两者的迭代顺序实测不保证一致（500 组随机里 176 组不同），"
            "换容器会静默改变 lil 累加的浮点顺序。"
            "legacy 的构造式见 MMA10:258。"
        )
    if not isinstance(gap_elements, set):
        raise TypeError(
            f"gap_elements 必须是 set（实得 {type(gap_elements).__name__}）—— "
            f"本内核的 F4 风险就在它的迭代顺序上，不接受任何其它容器。"
        )

    tri = np.asarray(geom.tri)
    xy = np.asarray(geom.xy)
    n_node = geom.n_node
    _node_gap = set(int(n) for n in geom.rotor_nodes)

    _Torque_Arkkio_coef = sp_sparse.lil_matrix((n_node, n_node))

    for i_elem in gap_elements:
        node_i, node_j, node_m = (int(n) for n in tri[i_elem])
        if not (node_i in _node_gap or node_j in _node_gap or node_m in _node_gap):
            continue

        _th_i = np.arctan2(xy[node_i, 1], xy[node_i, 0])
        _th_j = np.arctan2(xy[node_j, 1], xy[node_j, 0])
        _th_m = np.arctan2(xy[node_m, 1], xy[node_m, 0])
        _th_0 = (_th_i + _th_j + _th_m) / 3

        _r0 = np.sqrt(
            (xy[node_i, 0] + xy[node_j, 0] + xy[node_m, 0]) ** 2 / 9
            + (xy[node_i, 1] + xy[node_j, 1] + xy[node_m, 1]) ** 2 / 9
        )

        # 计算矩阵元素（径向/切向 = 理论文档 :2991-2998 的旋转矩阵）
        _K_Arkkio = (
            _r0 * geom.area[i_elem]
            * (geom.bxcoef[i_elem] * np.cos(_th_0) - geom.bycoef[i_elem] * np.sin(_th_0)).reshape((-1, 1))
            @ (geom.bxcoef[i_elem] * np.sin(_th_0) + geom.bycoef[i_elem] * np.cos(_th_0)).reshape((1, -1))
        )

        for _ki, node_index1 in enumerate([node_i, node_j, node_m]):
            for _kj, node_index2 in enumerate([node_i, node_j, node_m]):
                _Torque_Arkkio_coef[node_index1, node_index2] += _K_Arkkio[_ki, _kj]

    _Torque_Arkkio_coef *= 1 / geom.mu0 * geom.parts / (
        geom.gap / geom.parts ** geom.parts_exponent)

    # 该矩阵用于计算转矩关于单元磁导率的导数   （legacy :695 原注释）
    _Torque_Arkkio_coef = (_Torque_Arkkio_coef + _Torque_Arkkio_coef.T) / 2

    return _Torque_Arkkio_coef


def arkkio_torque(A: np.ndarray, geom: ArkkioGeometry) -> float:
    """装配 + 求值。求值本身无可向量化的部分，故与 ``reference`` 共用同一份。"""
    return torque_quadratic(A, arkkio_coefficient(geom))


# =============================================================================
# ★ 体边积分版 Maxwell 应力转矩（T_ArkBE）—— 2026-10-14 新增，生产默认口径
# =============================================================================
#
# **无 legacy 对应物**（本轮新方法）。数学与 `tools/probe_fine_scan.py::body_edge_arkkio`
# **完全同式**，这里只是把它装配成稀疏系数矩阵，以便复用
# `torque_quadratic`（值）与 `adjoint_torque_derivative`（伴随），**不必另写伴导**。
#
# 与 `arkkio_coefficient`（生产 `T_Arkkio`）的**本质区别**（详见
# `docs/转矩口径等价链核对与动网格插值_2026-10-13.md` §4.4）：
#
#   T_Arkkio = ∫_气隙环 (1/μ0) r·B_r·B_θ **dS**   ← **面积分**，带 h 标定因子
#                `gap / parts**parts_exponent`，该标定**在细网格上失效**（实测偏 45%）
#   T_ArkBE  = ∮_气隙中线 (1/μ0) r·B_n·B_t **dl** ← **线积分**（Maxwell 应力张量），
#                **无经验标定**，网格稳健（细网格贴 ANSYS 0.69%、与 T_VD 互吻 0.34%）
#
# `range(n_elem)` 顺序确定（同 `body_edge_arkkio`），与 legacy 的 `set` 迭代无关。

def arkbe_coefficient(geom: ArkkioGeometry) -> sp_sparse.csr_matrix:
    """**体边积分版** Maxwell 应力转矩矩阵 ``Q``：``T_ArkBE = Aᵀ Q A``。

    逐式（照 `tools/probe_fine_scan.py::body_edge_arkkio`）::

        T_ArkBE = parts · Σ_e  |mid| · (Bn·Bt/μ0) · l
        e 遍历气隙单元，只取该单元**跨接口圆 r_c 的那条边** (i,j)：
          Bx = (−Bycoef_e)·A_loc ,  By = (−Bxcoef_e)·A_loc    （legacy Bx/By 约定）
          Bn = Bx·n0 + By·n1 ,     Bt = Bx·t0 + By·t1
          t = (xy_j − xy_i)/l ,    n = (−t1, t0)（按 mid·n>0 定向）
        ⇒ 装配成 3×3 块 `(|mid|·l/μ0)·(k1 ⊗ k2)`，k1/k2 是 Bn/Bt 的局部系数向量

    ``r_c`` 从 ``geom.rotor_nodes`` 的半径中位数**自动推出**（不硬编码，与
    `body_edge_arkkio` 一致；硬编码 44.875e-3 在别的电机上会静默返回 0）。

    **Q 只在几何上依赖**（θ 无关、ρ 无关）⇒ 与 `torque_arkkio_coef` 一样**预组装一次**。
    """
    tri = np.asarray(geom.tri)
    xy = np.asarray(geom.xy)
    n_node = geom.n_node
    rotor = set(int(n) for n in geom.rotor_nodes)
    _rn = np.asarray(list(geom.rotor_nodes), dtype=np.int64)
    r_c = float(np.median(np.hypot(xy[_rn, 0], xy[_rn, 1])))
    Bx_c = -np.asarray(geom.bycoef)
    By_c = -np.asarray(geom.bxcoef)
    mu0 = geom.mu0

    rows, cols, vals = [], [], []
    for e in range(int(geom.n_elem)):
        nodes = tri[e]
        circ = [s for s in range(3)
                if int(nodes[s]) in rotor
                and abs(np.hypot(xy[nodes[s], 0], xy[nodes[s], 1]) - r_c) < 1e-6]
        if len(circ) < 2:
            continue
        i, j = circ[0], circ[1]
        e_ij = xy[nodes[j]] - xy[nodes[i]]
        l = float(np.hypot(e_ij[0], e_ij[1]))
        if l == 0.0:
            continue
        th_i = np.arctan2(xy[nodes[i], 1], xy[nodes[i], 0])
        th_j = np.arctan2(xy[nodes[j], 1], xy[nodes[j], 0])
        if (th_j - th_i + np.pi) % (2 * np.pi) - np.pi < 0:
            i, j = j, i
            e_ij = -e_ij
        t = e_ij / l
        n = np.array([-t[1], t[0]])
        mid = (xy[nodes[i]] + xy[nodes[j]]) / 2
        if mid @ n < 0:
            n = -n
        k1 = Bx_c[e] * n[0] + By_c[e] * n[1]
        k2 = Bx_c[e] * t[0] + By_c[e] * t[1]
        w = (float(np.hypot(mid[0], mid[1])) * l / mu0) * np.outer(k1, k2)
        for a in range(3):
            for b in range(3):
                v = w[a, b]
                if v != 0.0:
                    rows.append(int(nodes[a]))
                    cols.append(int(nodes[b]))
                    vals.append(v)
    Q = sp_sparse.coo_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=(n_node, n_node)).tocsr()
    # ⚠ **必须对称化**（同 `arkkio_coefficient` 的 `(M + M.T)/2`）：
    #   边贡献 `k1 ⊗ k2` 天然不对称，但二次型 `AᵀQA` 只看见对称部分 ⇒ **值不变**。
    #   而 `adjoint_torque_derivative` 用 `Q·A` 作右端（真实导数需要 `(Q+Qᵀ)A`）
    #   ⇒ **不对称会让伴导差一倍**（实测相对误差 0.88 → 修后 ~1e-3，见
    #   `tools/probe_fd_torque_adjoint.py`）。
    return ((Q + Q.T) / 2) * geom.parts


def arkbe_torque(A: np.ndarray, geom: ArkkioGeometry) -> float:
    """装配 + 求值（便捷入口）。值应与 `body_edge_arkkio(A, geom)` 一致（相对差 ~1e-15）。"""
    return torque_quadratic(A, arkbe_coefficient(geom))


# =============================================================================
# 虚位移法（内核 14）：v2 变换矩阵 + 系数
# =============================================================================
#
# ⚠ 本内核 F4 **不活跃**：legacy 的 ``get_VD_coef`` 用的是
# ``for i_elem in range(N_element)``（``:711``），不是 ``set`` 迭代。
# 所以这里的遍历顺序与 legacy 天然一致，两层的差别只在累加容器（``lil`` vs 字典）。

def transformation_matrix_v2(i_elem: int, ctx, rotating, *, n_node: int):
    """``T`` 与 ``dT``。对应 ``MMA10:1776-1869``，逐字搬移。

    ⚠ 与内核 8 的 fast 层同一个理由：``_T_temp[:, _index]`` 这个**花式索引**返回
    **Fortran 连续**的 ``(3, k)``（numpy 的布局规则），而 ``Bxcoef @ T`` 在 F/C 连续
    两种操作数上走**不同的 BLAS 内核** ⇒ 末位不同。``dT`` 同样要保证 F 连续，
    因为 ``Bxcoef @ dT`` 也是矩阵乘法。**不要**改写成先建小数组再填值。
    """
    i_elem = int(i_elem)

    if i_elem in ctx.element_rotor_gap:
        if i_elem not in ctx.element_slave:
            node_i, node_j, node_m = (int(n) for n in ctx.tri[i_elem])
            _ki = _kj = _km = 1
        else:
            node_i, node_j, node_m = (int(n) for n in ctx.virtual_tri[i_elem])
            _ki = 1 if node_i in ctx.master_nodes else (1 if ctx.a_eq else -1)
            _kj = 1 if node_j in ctx.master_nodes else (1 if ctx.a_eq else -1)
            _km = 1 if node_m in ctx.master_nodes else (1 if ctx.a_eq else -1)

        _T_temp = np.zeros((3, n_node))
        _dT_temp = np.zeros((3, n_node))
        _index = set()
        for s, node, sign in ((0, node_i, _ki), (1, node_j, _kj), (2, node_m, _km)):
            if node in rotating.brackets:
                # ⚠ brackets 长度随插值阶数变（linear 2 / cubic 4）⇒ 遍历。
                # linear 下与原先的 (_node_1, _node_2) 写法逐位相同（同序、同两项）。
                for _c in rotating.brackets[node]:
                    _c = int(_c)
                    _T_temp[s, _c] += -rotating.matrix[node, _c] * sign
                    _dT_temp[s, _c] += -rotating.d_matrix[node, _c] * sign
                    _index.add(_c)
            else:
                _T_temp[s, node] += sign
                _index.add(node)

        _index = list(_index)
        return ElementTransformWithDerivative(
            matrix=_T_temp[:, _index], index=tuple(_index),
            d_matrix=_dT_temp[:, _index],
        )

    if i_elem in ctx.element_slave:
        node_i, node_j, node_m = (int(n) for n in ctx.virtual_tri[i_elem])
        _ki = 1 if node_i in ctx.master_nodes else (1 if ctx.a_eq else -1)
        _kj = 1 if node_j in ctx.master_nodes else (1 if ctx.a_eq else -1)
        _km = 1 if node_m in ctx.master_nodes else (1 if ctx.a_eq else -1)
        return ElementTransformWithDerivative(
            matrix=np.diag([_ki, _kj, _km]), index=(node_i, node_j, node_m),
            d_matrix=np.diag([0, 0, 0]),
        )

    node_i, node_j, node_m = (int(n) for n in ctx.tri[i_elem])
    return ElementTransformWithDerivative(
        matrix=np.eye(3), index=(node_i, node_j, node_m),
        d_matrix=np.eye(3) * 0,
    )


def vd_coefficient(
    ctx,
    rotating,
    *,
    n_node: int,
    n_element: int,
    bxcoef: np.ndarray,
    bycoef: np.ndarray,
    area: np.ndarray,
    parts: int,
    mu0: float = 4e-7 * np.pi,
) -> sp_sparse.csr_matrix:
    """虚位移转矩矩阵。对应 ``MMA10:707-736``，逐字搬移。

    ``Bxcoef``/``Bycoef``/``Aelem``/``mu0``/``parts``/``N_element`` 在 legacy 里都是
    模块全局，本实现提成显式参数。``_dKe`` 的算式与结合顺序照抄，含末尾的
    ``*= -parts``。

    Notes
    -----
    与 legacy 的**唯一**容器差异：legacy 返回 ``lil``（``:735`` 的 ``*=`` 不换类型），
    本实现返回 ``csr``。数值上中性 —— ``lil @ x`` 与 ``csr @ x`` 实测**逐位相同**
    （见 ``fast/torque.py`` 顶部 ``arkkio_coefficient`` 那里的同类记录）。

    ⚠ **D3 已修（2026-09-23）**：legacy 不做对称化（``:735``），本层与 reference
    同步在装配后对称化（``(M + Mᵀ)/2``）—— 见设计文档 §12 的 D3 与
    ``reference/torque.py`` 第二节。**值不变**（二次型只看对称部分）。

    ⚠ **阶段 8 的 8c 判定：本函数不向量化（登记）**。设计文档 §阶段 8 的
    「向量化补齐」点过 ``get_VD_coef``（= 本函数），但实测（2026-09-23，
    ``probe_reuse.py`` 的姊妹实验）：
    - **不逐位**：legacy 用 ``lil`` 稀疏 ``+=``（每时间点一次），改 ``np.add.at``
      散射后 141/1880 个位置差 max 9.5e-7 —— 加法顺序不同，违反 fast/ 红线；
    - **ROI 为零**：它在 ``Preassembly`` 构造时跑 **11 次/运行**（不在 MMA 迭代
      内），一次 38 ms、合计 0.42 s —— 相对 3000 迭代的分钟级总时程可忽略。
    ⇒ 维持逐字搬移，不引入近似/重排。真正在 MMA 迭代内的逐单元循环
      （``compute_mechDiff_AL``）已在内核 17 的 fast 层向量化（见其 docstring）。
    """
    _dKe_p_dth = sp_sparse.lil_matrix((n_node, n_node))

    for i_elem in range(n_element):
        if i_elem not in ctx.element_rotor_gap:
            continue

        tr = transformation_matrix_v2(i_elem, ctx, rotating, n_node=n_node)
        _T, _index, _dT = tr.matrix, list(tr.index), tr.d_matrix

        _Bxcoef = bxcoef[i_elem] @ _T
        _Bycoef = bycoef[i_elem] @ _T
        _dBxcoef = bxcoef[i_elem] @ _dT
        _dBycoef = bycoef[i_elem] @ _dT

        _dKe = 2 * (_Bxcoef.reshape((-1, 1)) @ _dBxcoef.reshape((1, -1))
                    + _Bycoef.reshape((-1, 1)) @ _dBycoef.reshape((1, -1))
                    ) / (2 * mu0) * area[i_elem]

        for i, node_i in enumerate(_index):
            for j, node_j in enumerate(_index):
                _dKe_p_dth[node_i, node_j] += _dKe[i, j]

    _dKe_p_dth *= -parts
    M = _dKe_p_dth.tocsr()
    # D3 修正（2026-09-23）：对称化。与 reference 的 ``(M+Mᵀ)/2 * (-parts)`` 逐位一致
    # （parts 是整数、±缩放精确；``(v1+v2)/2·c == (v1·c+v2·c)/2`` 对 2 的幂 c 成立）。
    return (M + M.T) / 2


def vd_torque(A, ctx, rotating, **kwargs) -> float:
    """装配 + 求值（求值无可向量化的部分，与 ``reference`` 共用）。"""
    return torque_quadratic(A, vd_coefficient(ctx, rotating, **kwargs))


# =============================================================================
# 对 γ 的导数（内核 13 + 内核 14 的导数，legacy 把它们写成两个同构函数）
# =============================================================================

def adjoint_torque_derivative(
    _A_node: np.ndarray,
    K_vectorized: tuple,
    _Jacobi,
    Torque_coef,
    elements_to_compute,
    *,
    n_node: int,
    n_element: int,
    factor=None,
    elimination=None,
) -> np.ndarray:
    """``dT/dγ``（伴随解）。对应 ``MMA10:2260-2277`` 与 ``:2280-2297``。

    ``factor`` : 可选 ``splu(Jacobi.T)`` 的 SuperLU 因子 —— 阶段 8 的分解复用
    （EM 三个伴随解共用收敛点同一个 ``J`` 的转置因子）。默认 ``None`` 走原
    ``spsolve``；传入时用 ``factor.solve(rhs)`` 逐个右端回代 —— 两者**逐位一致**
    （实测，见 ``docs/重构设计文档.md`` 阶段 8）。⚠ 禁止把多个右端拼成矩阵一次
    ``factor.solve(B)``：批量回代与逐列 spsolve **不**逐位一致。

    ⚠ **legacy 有两个函数体逐字相同的版本**：``DTorque_Arkkio_vectorized``
    （用模块全局 ``Torque_Arkkio_coef``）与 ``DTorque_VD_vectorized``（系数是形参）。
    本实现只写一份、系数作参数（``kernels/README.md``：无差异就不建第二份）。

    Parameters
    ----------
    _A_node : (N_node,) 收敛的矢量磁位
    K_vectorized : 22 元组
        ``reference.assembly.as_legacy_tuple(ops, driving)``。**下面那句解包是从 legacy
        原样保留的**（设计文档 §4 的兼容装置），项数与顺序都不能动。
    _Jacobi : 内核 11 返回的**同一个对象**（会被 ``.T`` 求伴随）
    Torque_coef : (N_node, N_node) 系数矩阵（Arkkio 或虚位移）
    elements_to_compute : 可迭代 int 设计变量所在的单元
    n_node : int
        legacy 在这里读模块全局 ``N_node``（``:2270`` 的 ``shape=``）；本实现提成显式参数。
    n_element : int
        单元数。22 元组里没有它（legacy 从模块全局读 ``N_element``）；本实现显式传。

    Returns
    -------
    (N_elem,) ndarray，``elements_to_compute`` 之外恒为 0

    Notes
    -----
    三处照抄的细节：

    1. ``dK_p_dgamma_A_vectorized`` 是 **``(N_element, N_node)``** 的 COO —— 元素号为行、
       全局节点号为列，值与 ``g0i_vec`` 同源。所以 ``@ v`` 给出每个单元上的
       ``g0_e · v_active``（``g0_e`` **不含** ν，与内核 9/10 的饱和项同一几何量）。
    2. ``_result = -2 * (...)``：**``-2`` 同时含理论文档的负号与乘积法则的 2**
       （``d(AᵀMA)/dγ = 2 (dA/dγ)ᵀ M A``）。理论文档 :3044 写的是无 2 的单项，
       但它把 ``∂T/∂θ`` 直接写成了 ``Aᵀ M dA/dθ``，2 已被吸收进约定。
    3. ``dT[...] = _result[...]`` 的**清零**：设计变量只在拓扑优化区。
       ⚠ ``list(elements_to_compute)`` 的调用点照抄 —— 花式索引赋值的**顺序**
       在这里不影响数值（每个元素写自己的槽位），但红线 3 要求不动它。
    """
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    g0i_vec = g0i_mat @ _A_node

    dK_p_dgamma_A_vectorized = sp_sparse.coo_matrix(
        (g0i_vec, (dK_p_dgamma_A_row, dK_p_dgamma_A_col)),
        shape=(n_element, n_node),
    )

    rhs = Torque_coef @ _A_node
    if elimination is not None:
        # 消元（约束代入）：A = T·x_free ⇒
        #   rhs_free = Tᵀ·rhs，λ_free 解 J_freeᵀ（= (TᵀJT)ᵀ），
        #   dK_p_dgamma_A 经 T 投影（N_elem,N_node)@T → (N_elem,N_free)。
        rhs = elimination.eliminate_rhs(rhs)
        lam = (factor.solve(rhs) if factor is not None
               else sp_sparse_linalg.spsolve(_Jacobi.T, rhs))
        _result = -2 * (dK_p_dgamma_A_vectorized @ elimination.T) @ lam
    else:
        lam = (factor.solve(rhs) if factor is not None
               else sp_sparse_linalg.spsolve(_Jacobi.T, rhs))
        _result = -2 * dK_p_dgamma_A_vectorized @ lam

    dT_p_dgamma_vectorized = np.zeros(n_element)
    dT_p_dgamma_vectorized[list(elements_to_compute)] = _result[list(elements_to_compute)]

    return dT_p_dgamma_vectorized
