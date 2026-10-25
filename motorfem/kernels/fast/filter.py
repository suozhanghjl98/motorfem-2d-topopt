"""密度滤波器的 bucket 网格保真实现（内核 18 的 fast 层）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）
   ``compute_filter``           ``:1259-1379``
   ``compute_filter_variable``  ``:1382-1505``
   ``compute_filter_neighbour`` ``:1508-1639``
   （``compute_filter`` 与 ``compute_filter_variable`` 在
   ``nonlinear_PM_solver2.py`` 里逐字相同，见 ``reference/filter.py`` 第一节。）

数学、镜像公式、已登记事项全部写在 ``reference/filter.py`` 的模块 docstring
里，本文件不重复。本层是 legacy 的**逐字保真**实现：bucket 网格（多叉树）照抄
legacy ``:1260-1331`` / ``:1385-1455`` / ``:1511-1581``，权重累加循环照抄
``:1348-1371`` / ``:1471-1496`` / ``:1602-1631``。

------------------------------------------------------------------------------
结构与 legacy 的对应
------------------------------------------------------------------------------

三种核共享一个 bucket 网格（设计文档 §6：不是三种实现，而是一个 bucket 网格
+ 三种核）。legacy 把这段代码在三个函数里**各自复制一遍**；本层按设计文档的
目标结构抽成 ``build_bucket_grid`` 公共函数，三种核各自消费。**抽公共不影响
逐位结果**：bucket 构造只含整数/布尔运算与列表 append（无浮点累加），
抽出来与内联逐位相同。权重累加循环（浮点 ``+=`` 的累加顺序）则**逐字照抄**
legacy，一根手指都不动。

与 legacy 的映射：

==========================  ===========================================
legacy（MMA10 行号）         本模块
==========================  ===========================================
``:1260-1331`` 质心/bbox/桶   ``build_bucket_grid``
``:1348-1371`` 权重循环       ``compute_filter`` 的内层循环
``:1471-1496`` 权重循环       ``compute_filter_variable`` 的内层循环
``:1602-1631`` 权重循环       ``compute_filter_neighbour`` 的内层循环
``:1373-1375`` 行归一化        ``reference.filter.row_normalize``
==========================  ===========================================

------------------------------------------------------------------------------
红线清单逐条
------------------------------------------------------------------------------

1. ``list(element_topo_opti_m)`` 的调用点 → 本层统一在 ``build_bucket_grid``
   开头 ``_elems = list(elements)`` 取一次。行列顺序由这一次 ``list()`` 决定，
   与 legacy 逐次调用 ``list(element_topo_opti_m)`` **顺序相同**（同一 set 的
   迭代顺序在进程内稳定），缓存只是消除重复调用，不改变顺序。⚠ 若要改集合的
   构造表达式，改这里 —— 其他核函数不再出现 ``list(elements)``。
2. ``set(...)`` 构造表达式 → ``is_neighbour`` 里照抄
   ``set(element_S[i_elem1][1:4])``（``:1597-1598``）。
3. 循环嵌套顺序 → 权重循环 ``for i_x → i_y → i_elem1 → di_x → di_y → i_elem2``
   照抄（``:1349-1358`` 等），``di_x``/``di_y`` 都按 ``[-1, 0, 1]`` 顺序。
4. ``+=`` 累加顺序 → ``_filter[i_elem1, i_elem2] += ...`` 照抄；三种核都是
   「本尊、镜像1、镜像2」同序（``:1361`` → ``:1367`` → ``:1371``）。
5. ``lil_matrix`` 逐项赋值 → 照抄（``:1348`` 等）。
6. ``_x_min = max(np.min(...))`` 疑似笔误 → **照抄**（含 ``max``），
   已在 reference 第四节登记。

------------------------------------------------------------------------------
★ 与 reference 层的差异（唯一一处）
------------------------------------------------------------------------------

``compute_filter_neighbour`` 的邻居判定：legacy 用**矩阵索引**
``element_S[i_elem1]``（``:1597``），reference 用**正确语义**（单元 id）。
本层照抄 legacy（含错位）；C 级对齐测试用小网格（单元 id == 矩阵索引）验证
两者一致。见 ``reference/filter.py`` 第四节第 2 条。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import sparse as sp_sparse

from ..reference.filter import (
    mirror,             # noqa: F401  （纯 numpy，两层共用）
    row_normalize,      # noqa: F401
)

__all__ = [
    "mirror",
    "element_centroids",
    "row_normalize",
    "compute_filter",
    "compute_filter_variable",
    "compute_filter_neighbour",
]


@dataclass
class _BucketGrid:
    """bucket 网格（legacy 多叉树，``MMA10:1260-1331`` 的产物）。

    Attributes
    ----------
    x, y : (N,) ndarray
        拓扑单元质心（矩阵索引序）。
    x_ext1, y_ext1, x_ext2, y_ext2 : (N,) ndarray
        镜像副本质心（``extend=True`` 时有效）。
    region : list[list[list[int]]]
        ``region[ix][iy]`` = 落在该桶的单元矩阵索引列表（append 序 = 矩阵索引升序）。
    region_ext1, region_ext2 : list[list[list[int]]]
        镜像副本的桶（只含落在网格范围内的副本）。
    n_x_region, n_y_region : int
        桶网格尺寸。
    """

    x: np.ndarray
    y: np.ndarray
    x_ext1: np.ndarray
    y_ext1: np.ndarray
    x_ext2: np.ndarray
    y_ext2: np.ndarray
    region: list
    region_ext1: list
    region_ext2: list
    n_x_region: int
    n_y_region: int


def build_bucket_grid(
    node: np.ndarray,
    element_S: Sequence[Sequence[int]],
    elements: Iterable[int],
    dl: float,
    *,
    extend: bool,
    parts: int,
) -> _BucketGrid:
    """构造 bucket 网格 —— 照抄 legacy ``:1260-1331``（三种核的公共段）。

    Parameters
    ----------
    node : (N_node, 4) ndarray
        节点表 ``[节点号, x, y, z]``（单位 m）。
    element_S : sequence
        单元表，每行 ``[单元id, n1, n2, n3, 区域号]``。
    elements : iterable of int
        拓扑优化单元 id 集合。**矩阵行列顺序 = ``list(elements)`` 的顺序**
        （F4 红线第 3 条）。
    dl : float
        桶尺寸。legacy 三处分别取 ``radius`` / ``max_radius`` / ``radius``。
    extend : bool
        是否构造镜像副本桶。
    parts : int
        周期扇区数（镜像1 的角度 ``2π/parts``）。

    ⚠ 照抄 legacy 的 ``_x_min = max(np.min(...))`` / ``_y_min = max(...)``
    （疑似 ``min`` 笔误，见 reference 第四节第 1 条）。
    """
    N_elem_topo_opti = len(elements)
    _elems = list(elements)                     # ← 行列顺序的唯一来源（F4）
    _element_S_array = np.array(element_S).astype(int)

    _x1 = node[_element_S_array[_elems, 1], 1]
    _y1 = node[_element_S_array[_elems, 1], 2]
    _x2 = node[_element_S_array[_elems, 2], 1]
    _y2 = node[_element_S_array[_elems, 2], 2]
    _x3 = node[_element_S_array[_elems, 3], 1]
    _y3 = node[_element_S_array[_elems, 3], 2]

    # 计算坐标范围（照抄；_x_min/_y_min 的 max 疑似笔误，只登记不改）
    _x_max = max(np.max(_x1), np.max(_x2), np.max(_x3))
    _y_max = max(np.max(_y1), np.max(_y2), np.max(_y3))
    _x_min = max(np.min(_x1), np.min(_x2), np.min(_x3))
    _y_min = max(np.min(_y1), np.min(_y2), np.min(_y3))

    if extend:
        _x_extend = dl * 2
        _y_extend = dl * 2
    else:
        _x_extend = 0
        _y_extend = 0

    # 计算单元中心位置
    _x = (_x1 + _x2 + _x3) / 3
    _y = (_y1 + _y2 + _y3) / 3

    if extend:
        _x_ext1, _y_ext1 = mirror(_x, _y, 2 * np.pi / parts)
        _x_ext2, _y_ext2 = mirror(_x, _y, 0)

    # 多叉树算法（照抄 :1295-1331）
    N_x_region = int(np.ceil((_x_max - _x_min + _x_extend * 2) / dl))
    N_y_region = int(np.ceil((_y_max - _y_min + _y_extend * 2) / dl))

    _region = [[[] for j in range(N_y_region)] for i in range(N_x_region)]

    _x_index = ((_x - (_x_min - _x_extend)) // dl).astype(int)
    _y_index = ((_y - (_y_min - _y_extend)) // dl).astype(int)

    if extend:
        _region_ext1 = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
        _region_ext2 = [[[] for j in range(N_y_region)] for i in range(N_x_region)]
        _x_rot1_index = ((_x_ext1 - (_x_min - _x_extend)) // dl).astype(int)
        _y_rot1_index = ((_y_ext1 - (_y_min - _y_extend)) // dl).astype(int)
        _x_rot2_index = ((_x_ext2 - (_x_min - _x_extend)) // dl).astype(int)
        _y_rot2_index = ((_y_ext2 - (_y_min - _y_extend)) // dl).astype(int)

    for i_elem in range(N_elem_topo_opti):
        _region[_x_index[i_elem]][_y_index[i_elem]].append(i_elem)
        if extend:
            if 0 <= _x_rot1_index[i_elem] < N_x_region and 0 <= _y_rot1_index[i_elem] < N_y_region:
                _region_ext1[_x_rot1_index[i_elem]][_y_rot1_index[i_elem]].append(i_elem)
            if 0 <= _x_rot2_index[i_elem] < N_x_region and 0 <= _y_rot2_index[i_elem] < N_y_region:
                _region_ext2[_x_rot2_index[i_elem]][_y_rot2_index[i_elem]].append(i_elem)

    return _BucketGrid(
        x=_x,
        y=_y,
        x_ext1=_x_ext1 if extend else np.empty(0),
        y_ext1=_y_ext1 if extend else np.empty(0),
        x_ext2=_x_ext2 if extend else np.empty(0),
        y_ext2=_y_ext2 if extend else np.empty(0),
        region=_region,
        region_ext1=_region_ext1 if extend else None,
        region_ext2=_region_ext2 if extend else None,
        n_x_region=N_x_region,
        n_y_region=N_y_region,
    )


def _dist_p(x1, y1, x2, y2):
    """距离函数（legacy ``dist_p``，``MMA10:1335-1336``）。"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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
    """固定半径密度滤波器 —— legacy ``compute_filter`` 的 bucket 保真实现。

    权重累加循环照抄 ``MMA10:1348-1371``。矩阵元 ``(i, j)`` 的累加顺序是
    「本尊、镜像1、镜像2」（``:1361`` → ``:1367`` → ``:1371``），与
    ``reference.compute_filter`` 同序 ⇒ 两者逐位一致。
    """
    N = len(elements)
    g = build_bucket_grid(node, element_S, elements, radius, extend=extend, parts=parts)
    _x, _y = g.x, g.y
    _region = g.region
    N_x_region, N_y_region = g.n_x_region, g.n_y_region

    _filter = sp_sparse.lil_matrix((N, N))
    for i_x in range(N_x_region):
        for i_y in range(N_y_region):
            for i_elem1 in _region[i_x][i_y]:
                for di_x in [-1, 0, 1]:
                    if di_x + i_x < 0 or di_x + i_x >= N_x_region:
                        continue
                    for di_y in [-1, 0, 1]:
                        if di_y + i_y < 0 or di_y + i_y >= N_y_region:
                            continue
                        for i_elem2 in _region[i_x + di_x][i_y + di_y]:
                            _dist = _dist_p(_x[i_elem1], _y[i_elem1], _x[i_elem2], _y[i_elem2])
                            if _dist < radius:
                                _filter[i_elem1, i_elem2] += (radius - _dist) ** p_filter

                        if extend:
                            for i_elem2 in g.region_ext1[i_x + di_x][i_y + di_y]:
                                _dist_ext1 = _dist_p(_x[i_elem1], _y[i_elem1],
                                                     g.x_ext1[i_elem2], g.y_ext1[i_elem2])
                                if _dist_ext1 < radius:
                                    _filter[i_elem1, i_elem2] += (radius - _dist_ext1) ** p_filter
                            for i_elem2 in g.region_ext2[i_x + di_x][i_y + di_y]:
                                _dist_ext2 = _dist_p(_x[i_elem1], _y[i_elem1],
                                                     g.x_ext2[i_elem2], g.y_ext2[i_elem2])
                                if _dist_ext2 < radius:
                                    _filter[i_elem1, i_elem2] += (radius - _dist_ext2) ** p_filter

    _filter = _filter.tocsr()
    return row_normalize(_filter)


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
    """变半径密度滤波器 —— legacy ``compute_filter_variable`` 的 bucket 保真实现。

    与 ``compute_filter`` 唯一区别：半径逐单元取 ``radius_func(_x[i_elem1],
    _y[i_elem1])``（照抄 ``MMA10:1475``），桶尺寸取 ``max_radius``
    （``:1420``）。权重累加循环照抄 ``:1471-1496``。
    """
    N = len(elements)
    g = build_bucket_grid(node, element_S, elements, max_radius, extend=extend, parts=parts)
    _x, _y = g.x, g.y
    _region = g.region
    N_x_region, N_y_region = g.n_x_region, g.n_y_region

    _filter = sp_sparse.lil_matrix((N, N))
    for i_x in range(N_x_region):
        for i_y in range(N_y_region):
            for i_elem1 in _region[i_x][i_y]:
                radius = radius_func(_x[i_elem1], _y[i_elem1])
                for di_x in [-1, 0, 1]:
                    if di_x + i_x < 0 or di_x + i_x >= N_x_region:
                        continue
                    for di_y in [-1, 0, 1]:
                        if di_y + i_y < 0 or di_y + i_y >= N_y_region:
                            continue
                        for i_elem2 in _region[i_x + di_x][i_y + di_y]:
                            _dist = _dist_p(_x[i_elem1], _y[i_elem1], _x[i_elem2], _y[i_elem2])
                            if _dist < radius:
                                _filter[i_elem1, i_elem2] += (radius - _dist) ** p_filter

                        if extend:
                            for i_elem2 in g.region_ext1[i_x + di_x][i_y + di_y]:
                                _dist_ext1 = _dist_p(_x[i_elem1], _y[i_elem1],
                                                     g.x_ext1[i_elem2], g.y_ext1[i_elem2])
                                if _dist_ext1 < radius:
                                    _filter[i_elem1, i_elem2] += (radius - _dist_ext1) ** p_filter
                            for i_elem2 in g.region_ext2[i_x + di_x][i_y + di_y]:
                                _dist_ext2 = _dist_p(_x[i_elem1], _y[i_elem1],
                                                     g.x_ext2[i_elem2], g.y_ext2[i_elem2])
                                if _dist_ext2 < radius:
                                    _filter[i_elem1, i_elem2] += (radius - _dist_ext2) ** p_filter

    _filter = _filter.tocsr()
    return row_normalize(_filter)


def compute_filter_neighbour(
    node: np.ndarray,
    element_S: Sequence[Sequence[int]],
    elements: Iterable[int],
    radius: float,
    *,
    extend: bool = True,
    parts: int = 4,
) -> sp_sparse.csr_matrix:
    """邻居滤波器 —— legacy ``compute_filter_neighbour`` 的 bucket 保真实现。

    ⚠ ``radius`` 只用于桶尺寸（``:1546``），**不进入权重**（距离判据被注释掉，
    ``:1613-1615``）。权重是纯邻接：两单元共享一条边（2 个公共节点）则置 1。

    ⚠ **照抄 legacy 的邻居判定**：``is_neighbour`` 用**矩阵索引**
    ``element_S[i_elem1]``（``:1596-1599``）取单元节点 —— 不是「矩阵索引 →
    单元 id」的正确语义（见 reference 第四节第 2 条）。本层保真照抄。
    """
    N = len(elements)
    g = build_bucket_grid(node, element_S, elements, radius, extend=extend, parts=parts)
    _x, _y = g.x, g.y
    _region = g.region
    N_x_region, N_y_region = g.n_x_region, g.n_y_region

    def is_neighbour(i_elem1, i_elem2):
        #: 照抄 MMA10:1596-1599（含矩阵索引的疑似错位，只登记不改）
        nodes_1 = set(element_S[i_elem1][1:4])
        nodes_2 = set(element_S[i_elem2][1:4])
        return len(nodes_1.intersection(nodes_2)) == 2

    _filter = sp_sparse.lil_matrix((N, N))
    for i_x in range(N_x_region):
        for i_y in range(N_y_region):
            for i_elem1 in _region[i_x][i_y]:
                for di_x in [-1, 0, 1]:
                    if di_x + i_x < 0 or di_x + i_x >= N_x_region:
                        continue
                    for di_y in [-1, 0, 1]:
                        if di_y + i_y < 0 or di_y + i_y >= N_y_region:
                            continue
                        for i_elem2 in _region[i_x + di_x][i_y + di_y]:
                            if is_neighbour(i_elem1, i_elem2):
                                _filter[i_elem1, i_elem2] = 1

                        if extend:
                            for i_elem2 in g.region_ext1[i_x + di_x][i_y + di_y]:
                                if is_neighbour(i_elem1, i_elem2):
                                    _filter[i_elem1, i_elem2] = 1
                            for i_elem2 in g.region_ext2[i_x + di_x][i_y + di_y]:
                                if is_neighbour(i_elem1, i_elem2):
                                    _filter[i_elem1, i_elem2] = 1

    _filter = _filter.tocsr()
    return row_normalize(_filter)
