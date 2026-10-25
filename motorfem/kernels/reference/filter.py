"""密度滤波器（内核 18）—— 朴素 O(N²) 实现。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）
   ``compute_filter``             ``:1259-1379``
   ``compute_filter_variable``    ``:1382-1505``
   ``compute_filter_neighbour``   ``:1508-1639``
另：``磁障设计/pytorch/nonlinear_PM_solver2.py`` 有与 MMA10 **逐字相同**的
``compute_filter`` / ``compute_filter_variable``（本批已用文本 diff 核实；
``compute_filter_neighbour`` 只存在于 MMA10）。

数学出处：``磁障设计/同步磁阻电机磁障结构分析.md``
   「拓扑优化方法 › 密度滤波」相关章节（设计文档 §1.2 登记为
   「拓扑优化·方形网格的拓扑优化实现」）。

------------------------------------------------------------------------------
一、这个内核算什么（三种核 + 一个 bucket 网格）
------------------------------------------------------------------------------

三种核**共享同一个 bucket 网格**（设计文档 §6 阶段 4 的原话：不是三种实现，
而是一个 bucket 网格 + 三种核）。本模块是它的**朴素表达**：每种核直接对
全单元两两配对求和（O(N²)），不建 bucket —— bucket 只是加速手段，不是数学
（§ 本文档第二节证明它是**精确**加速，不是近似，因此本模块与 fast 层可以
做到 ``np.array_equal``）。

三种核的输出都是 ``(N_topo, N_topo)`` 的**行归一化**稀疏矩阵（每行和为 1），
行/列顺序由 ``list(elements)`` 决定（这就是 ``filter_SIMP`` 的行列顺序，F4
红线 § 设计文档第 8 节第 3 条）。上游用法：

- ``filter_SIMP @ x``（``MMA10:3865``/``:4009``）：把设计变量平滑成物理密度；
- ``filter_SIMP_grad @ dobj/dρ``（``MMA10:3904``/``:4042``）：对梯度做**前向**
  滤波（⚠ 不是伴随链式 —— 见设计文档 §11①）。**判定完成（2026-09-28，
  ``tools/probe_filter_grad_decision.py``）：保留为默认** —— 真梯度
  （``filter_SIMP.T @``）已由方向性 FD 坐实，但它 300 步收敛实证差于 legacy
  窄滤波的灵敏度正则（obj 1.69 vs 5.07）。开关：``RuntimeConfig.gradient_filter``
  （"legacy" 默认 / "adjoint" 伴随一致），本内核不改。

**三种核的权重公式**（``i`` 行 ``j`` 列的元，``d_ij`` = 单元质心距离）：

1. **固定半径**（``compute_filter``，AT 线生产路径，``MMA10:1661``）::

       w_ij = Σ_k  (r − d_ij⁽ᵏ⁾)^p · 1[d_ij⁽ᵏ⁾ < r]

   其中 ``k`` 遍历「本尊 + 两个镜像副本」（见第三节），``p = p_filter``。

2. **变半径**（``compute_filter_variable``，PMSM 线生产路径，``solver2:1447``）::

       w_ij = Σ_k  (r_i − d_ij⁽ᵏ⁾)^p · 1[d_ij⁽ᵏ⁾ < r_i]

   半径**逐单元**取 ``r_i = radius_func(x_i, y_i)``（PMSM 线用它做「越靠转子
   外缘半径越大」的非均匀滤波）。``max_radius`` 只用于 bucket 网格的尺寸
   （加速参数，不进入权重）。

3. **邻居核**（``compute_filter_neighbour``，纯 MMA10，**全仓库零调用者**）::

       w_ij = 1[ 单元 i、j 共享一条边 ]        （无距离判据、无 p_filter）

   ⚠ 它的 ``radius`` 参数**只用于 bucket 网格**（决定网格尺寸），**不进入**
   权重 —— 这是它与前两核的本质区别。前两核里 ``radius`` 既进网格又进权重。

归一化（三种核相同）::

       W_ij ← W_ij / Σ_j W_ij

   （legacy 用 ``_filter[i_row, :] /= _filter[i_row, :].sum()``，逐行除以行和，
   见 ``MMA10:1374-1375`` 等三处。）

------------------------------------------------------------------------------
二、bucket 网格为什么是精确加速（本模块与 fast 逐位一致的前提）
------------------------------------------------------------------------------

legacy 用「多叉树算法」（``MMA10:1295-1331``）把单元质心按坐标分桶，
每个桶尺寸 = ``radius``，只搜索本桶 + 8 邻桶（3×3）。本模块直接全对求和。

**为什么两者逐位一致**：对任意质心距 ``d_ij < radius`` 的两个单元，它们在
x/y 两个方向的坐标差都 `< radius = 桶尺寸`，所以桶号差 ≤ 1（每个方向）——
即必然落在 3×3 邻域内。因此 bucket 搜索**不会漏掉任何一对** radius 内的
单元；搜到但距离超限的会被 ``if _dist < radius`` 滤掉（不会误加）。这是
「不漏不误」，所以 bucket 版与全对版结果相同。

**累加顺序**：``extend=False`` 时每个 ``(i, j)`` 元只累加一次（本尊），两种写法
逐位相同。``extend=True`` 时每个元最多累加三项（本尊 + 镜像1 + 镜像2），
**顺序由 bucket 网格的遍历顺序决定**（fast 里三项落在不同桶、按桶遍历先后累加，
不是固定的「本尊→镜像1→镜像2」）—— 朴素 O(N²) 无法复现该顺序，实测末位差
**1 ulp**（相对差 ~1e-13）。因此 C 级对齐用 ``rtol=1e-12``（README 标准），
不承诺 ``extend=True`` 时逐位；``extend=False`` 时逐位。

------------------------------------------------------------------------------
三、镜像副本（``extend=True`` 时的周期延拓）
------------------------------------------------------------------------------

设计域是电机的**周期扇区**（``parts`` 分之一）。为避免滤波在扇区边界把密度
「拉偏」，legacy 把每个单元复制到两个镜像位置（``MMA10:1291-1293``）::

    镜像1 = _mirror(x, y, 2π/parts)        ← 绕扇区右边界
    镜像2 = _mirror(x, y, 0)               ← 绕扇区左边界（x 轴）

``_mirror``（``MMA10:189-190``，由 ``:177-187`` 的 ``_rot_x/_rot_y/_rot``
组合）::

    _rot_x(x, y, a) =  x·cos a − y·sin a
    _rot_y(x, y, a) =  x·sin a + y·cos a
    _mirror(x, y, a) = ( _rot_x(x, y, −a), _rot_y(x, −y, a) )

展开得（记 ``c = cos a, s = sin a``）::

    mirror_a(x, y) = ( x·c + y·s ,  x·s − y·c )

即关于「与 x 轴夹角 ``a/2``」的直线的镜像。``a = 2π/parts`` 时镜像到相邻
扇区、``a = 0`` 时镜像过 x 轴 —— 两者共同把扇区「翻」成完整周期。
**生产 ``parts = 4``**（``MMA10:83``）：镜像1 过 45° 线、镜像2 过 x 轴。

权重里镜像的角色：``w_ij`` 额外加上「单元 ``i`` 到单元 ``j`` 的镜像副本
的距离」的贡献。若 ``j`` 本身不在 ``i`` 的 radius 内、但 ``j`` 的镜像副本在，
则边界附近的 ``i`` 也能「看到」扇区外的邻居 —— 这就是周期延拓。

⚠ ``extend=False`` 时无镜像，两种核退化为普通（非周期）滤波。

------------------------------------------------------------------------------
四、★ 已登记事项（本批核实，只登记不改）
------------------------------------------------------------------------------

1. **``_x_min = max(np.min(...))`` 用 ``max`` 疑似笔误**（应为 ``min``），
   三处核里逐字相同（``MMA10:1274-1275`` 等）。它**只影响 bucket 网格的原点**
   不影响任何权重（权重只依赖质心距）。真实网格上实测：
   ``max`` 版与 ``min`` 版输出**逐位一致**（含 CSR 结构）—— 因为 bucket 网格
   的精确性不依赖原点（第二节的论证只用了「桶尺寸 = radius」）。但
   ``extend=False`` 时 ``_x_min`` 偏大会让部分单元桶号为负 → ``lil_matrix``
   负索引 → **静默错**。生产 ``extend=True`` 不受影响。登记为待核对，
   修正归基线后（§11 或 §12 的 D 系列）。

2. **``compute_filter_neighbour`` 的 ``is_neighbour`` 疑似索引错位**：它用
   **矩阵索引** ``element_S[i_elem1]``（``MMA10:1597-1599``）取单元节点，
   而 ``element_S`` 的第 ``i`` 行是**单元 id = i**，矩阵索引 ``i`` 对应的单元
   id 是 ``list(elements)[i]``。真实网格上拓扑区单元 id 从 206 开始
   （``element_topo_opti_m`` 是转子铁芯，单元 id 从 shaft 之后），所以
   ``element_S[i_elem1]`` 与真正参与滤波的单元**不是同一个**。该核全仓库
   **零调用者**（死代码），故不发作。登记为待核对；本模块按**正确的语义**
   写（用 ``list(elements)[i]`` 取单元），fast 层照抄 legacy（含错位），
   C 级对齐测试用小网格（单元 id == 矩阵索引，错位不激活）。

3. **三种核在 MMA10 与 PM_solver2 之间逐字相同**（本批文本 diff 核实），
   但 ``compute_filter_neighbour`` 只存在于 MMA10。两边的 ``_mirror`` 与
    ``_rot_x/_rot_y/_rot`` 也逐字相同。

4. **行归一化的遍历范围是 ``range(_filter.shape[1])``**（列数，与行数相等的
   方阵）—— 三种核都是 ``(N_topo, N_topo)`` 方阵，所以等价于逐行归一化。
   本模块与 fast 照抄（``:1374``、``:1500``、``:1634``）。

------------------------------------------------------------------------------
五、为什么 reference 是 O(N²)（与 fast 的分层依据）
------------------------------------------------------------------------------

legacy 的 bucket 网格是**手工加速**（手写「多叉树」避免全对配对），属
``kernels/README.md`` 分层判据里「手工向量化」一类 ⇒ 有真实的 ref/fast
差异：本模块是直白的全对求和（数学权威），fast 层是 legacy 的 bucket 搬移。
两者**逐位一致**（第二节），所以 C 级对齐可以用 ``np.array_equal``
（比常规 ``rtol=1e-12`` 更强）。

⚠ 性能提示：O(N²) 只在测试夹具规模（≤ 数百单元）下使用；真实网格（6362 个
拓扑单元）必须用 fast 层或 bucket 版探针，本模块不应对真实网格直接调用。
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy import sparse as sp_sparse

__all__ = [
    "mirror",
    "element_centroids",
    "row_normalize",
    "compute_filter",
    "compute_filter_variable",
    "compute_filter_neighbour",
]


def mirror(
    x: np.ndarray,
    y: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """镜像变换（legacy 的 ``_mirror``，``MMA10:189-190`` 的展开式）。

    关于「与 x 轴夹角 ``angle/2``」的直线镜像::

        mirror_a(x, y) = ( x·cos a + y·sin a ,  x·sin a − y·cos a )

    Parameters
    ----------
    x, y : (N,) ndarray
        坐标（单位 m）。
    angle : float
        镜像角（弧度）。生产取值 ``2π/parts`` 与 ``0``。

    Returns
    -------
    (x', y') : ((N,), (N,)) ndarray
        镜像后的坐标。
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return x * c + y * s, x * s - y * c


def element_centroids(
    node: np.ndarray,
    element_S: Sequence[Sequence[int]],
    elements: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """拓扑单元质心（legacy 的 ``_x = (_x1+_x2+_x3)/3``，``MMA10:1288-1289``）。

    Parameters
    ----------
    node : (N_node, 4) ndarray
        节点表 ``[节点号, x, y, z]``（单位 m）。
    element_S : sequence
        单元表，每行 ``[单元id, n1, n2, n3, 区域号]``（与 legacy 同构）。
    elements : iterable of int
        拓扑优化单元 id 集合（``element_topo_opti_m``）。**返回顺序由
        ``list(elements)`` 决定** —— 与 fast 层一致，从而与 legacy 的
        ``filter_SIMP`` 行列顺序一致。

    Returns
    -------
    (x, y) : ((N,), (N,)) ndarray
        质心坐标。
    """
    elem = list(elements)
    _S = np.array(element_S).astype(int)
    x1 = node[_S[elem, 1], 1]
    y1 = node[_S[elem, 1], 2]
    x2 = node[_S[elem, 2], 1]
    y2 = node[_S[elem, 2], 2]
    x3 = node[_S[elem, 3], 1]
    y3 = node[_S[elem, 3], 2]
    return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3


def row_normalize(W: sp_sparse.csr_matrix) -> sp_sparse.csr_matrix:
    """逐行归一化（legacy 的 ``_filter[i_row, :] /= _filter[i_row, :].sum()``）。

    ⚠ 遍历范围照抄 legacy 的 ``range(_filter.shape[1])``（方阵，列数 == 行数）。
    """
    for i_row in range(W.shape[1]):
        W[i_row, :] /= W[i_row, :].sum()
    return W


def compute_filter(
    node: np.ndarray,
    element_S: Sequence[Sequence[int]],
    elements: Iterable[int],
    radius: float,
    *,
    extend: bool = True,
    p_filter: float = 2,
    parts: int = 4,
) -> sp_sparse.csr_matrix:
    """固定半径密度滤波器（legacy ``compute_filter``，``MMA10:1259-1379``）。

    .. math::

        W_{ij} = \\sum_{k\\in\\{本尊,\\ mirror_{2π/parts},\\ mirror_0\\}}
                 (r - d_{ij}^{(k)})^{p} \\cdot [d_{ij}^{(k)} < r]
        \\tilde W = \\mathrm{row\\_normalize}(W)

    ``r`` = ``radius``，``p`` = ``p_filter``，``d_ij⁽ᵏ⁾`` 是单元 ``i`` 质心到
    单元 ``j``（或 ``j`` 的镜像副本 ``k``）质心的欧氏距离。权重只依赖质心距，
    与单元形状无关。

    Parameters
    ----------
    node : (N_node, 4) ndarray
        节点表（单位 m）。
    element_S : sequence
        单元表，每行 ``[单元id, n1, n2, n3, 区域号]``。
    elements : iterable of int
        拓扑优化单元 id 集合；行/列顺序由 ``list(elements)`` 决定。
    radius : float
        滤波半径（单位 m，与坐标同单位）。
    extend : bool
        是否做周期延拓（镜像副本参与求和）。生产 ``True``。
    p_filter : float
        权重指数。生产 AT 线 ``2``（``filter_p_exp``，``MMA10:143``）；
        PMSM 线 ``1``（``solver2:149``）。
    parts : int
        周期扇区数，决定镜像1 的角度 ``2π/parts``。生产 ``4``。

    Returns
    -------
    csr_matrix
        ``(N_topo, N_topo)`` 行归一化滤波器（每行和为 1）。
    """
    N = len(list(elements))
    x, y = element_centroids(node, element_S, elements)
    if extend:
        x_ext1, y_ext1 = mirror(x, y, 2 * np.pi / parts)
        x_ext2, y_ext2 = mirror(x, y, 0)

    W = np.zeros((N, N))
    for i in range(N):
        xi, yi = x[i], y[i]
        for j in range(N):
            d = np.sqrt((xi - x[j]) ** 2 + (yi - y[j]) ** 2)
            if d < radius:
                W[i, j] += (radius - d) ** p_filter
            if extend:
                d = np.sqrt((xi - x_ext1[j]) ** 2 + (yi - y_ext1[j]) ** 2)
                if d < radius:
                    W[i, j] += (radius - d) ** p_filter
                d = np.sqrt((xi - x_ext2[j]) ** 2 + (yi - y_ext2[j]) ** 2)
                if d < radius:
                    W[i, j] += (radius - d) ** p_filter
    return row_normalize(sp_sparse.csr_matrix(W))


def compute_filter_variable(
    node: np.ndarray,
    element_S: Sequence[Sequence[int]],
    elements: Iterable[int],
    radius_func: Callable[[float, float], float],
    max_radius: float,
    *,
    extend: bool = True,
    p_filter: float = 2,
    parts: int = 4,
) -> sp_sparse.csr_matrix:
    """变半径密度滤波器（legacy ``compute_filter_variable``，``MMA10:1382-1505``）。

    与 ``compute_filter`` 唯一区别：半径逐单元取 ``r_i = radius_func(x_i, y_i)``::

        W_{ij} = Σ_k (r_i − d_ij⁽ᵏ⁾)^p · [d_ij⁽ᵏ⁾ < r_i]

    ``max_radius`` 只用于 bucket 网格尺寸（加速参数，不进权重），本模块不消费
    （保留它只为与 fast 层签名一致）。⚠ **调用方必须保证**
    ``radius_func`` 返回值 ≤ ``max_radius``，否则 bucket 网格会漏对
    （fast 层的精确性依赖桶尺寸 ≥ 最大半径）。

    Parameters
    ----------
    radius_func : callable(x, y) -> float
        逐单元滤波半径（生产 PMSM 线 ``solver2:1438-1444`` 的 ``_radius_func``：
        越靠转子外缘半径越大，实现「非均匀滤波」）。
    max_radius : float
        桶尺寸 = 最大可能半径（用于 bucket 网格）。

    其余参数同 ``compute_filter``。
    """
    elem = list(elements)
    N = len(elem)
    x, y = element_centroids(node, element_S, elements)
    if extend:
        x_ext1, y_ext1 = mirror(x, y, 2 * np.pi / parts)
        x_ext2, y_ext2 = mirror(x, y, 0)

    W = np.zeros((N, N))
    for i in range(N):
        r_i = radius_func(x[i], y[i])
        xi, yi = x[i], y[i]
        for j in range(N):
            d = np.sqrt((xi - x[j]) ** 2 + (yi - y[j]) ** 2)
            if d < r_i:
                W[i, j] += (r_i - d) ** p_filter
            if extend:
                d = np.sqrt((xi - x_ext1[j]) ** 2 + (yi - y_ext1[j]) ** 2)
                if d < r_i:
                    W[i, j] += (r_i - d) ** p_filter
                d = np.sqrt((xi - x_ext2[j]) ** 2 + (yi - y_ext2[j]) ** 2)
                if d < r_i:
                    W[i, j] += (r_i - d) ** p_filter
    return row_normalize(sp_sparse.csr_matrix(W))


def compute_filter_neighbour(
    node: np.ndarray,
    element_S: Sequence[Sequence[int]],
    elements: Iterable[int],
    radius: float,
    *,
    extend: bool = True,
    parts: int = 4,
) -> sp_sparse.csr_matrix:
    """邻居滤波器（legacy ``compute_filter_neighbour``，``MMA10:1508-1639``）。

    权重是**纯邻接**：::

        W_ij = 1[ 单元 i、j 共享一条边 ]      （两个公共节点）

    无距离判据、无 ``p_filter``。``radius`` 只决定 bucket 网格尺寸（加速参数，
    本模块不消费，保留只为与 fast 层签名一致）。

    ⚠ **正确语义 vs legacy 的差异（已登记，见第四节第 2 条）**：本模块按正确
    语义写 —— 用 ``list(elements)[i]`` 取单元 id 再查 ``element_S`` 的节点；
    legacy 用矩阵索引 ``element_S[i]``（真实网格上单元 id ≠ 矩阵索引，是疑似
    错位）。C 级对齐测试用小网格（单元 id == 矩阵索引）保证两者一致。

    Parameters
    ----------
    radius : float
        只用于 bucket 网格尺寸（legacy 行为），不进入权重。
    其余参数同 ``compute_filter``（无 ``p_filter``）。
    """
    elem = list(elements)
    N = len(elem)
    #: 正确语义：矩阵索引 i → 单元 id = elem[i] → element_S 行的节点。
    nodes = [set(element_S[e][1:4]) for e in elem]

    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if len(nodes[i].intersection(nodes[j])) == 2:
                W[i, j] = 1
    return row_normalize(sp_sparse.csr_matrix(W))
