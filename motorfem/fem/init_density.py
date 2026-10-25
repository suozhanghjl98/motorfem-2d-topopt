"""初始密度的 legacy 保真移植（阶段 4A：golden L2 的 ``rho_SIMP_initial``）。

来源：``legacy/pytorch/nonlinear_solver_energy_MMA10.py``

    ``init_opti_elem_random_hole``   ``:1038-1115``
    两次调用 + SIMP 反演            ``:1174-1179``
    滤波器前像（L-BFGS-B 求逆）    ``:1672-1686``
    ``np.random.seed(1002)``        ``:42``

------------------------------------------------------------------------------
golden 的 ``rho_SIMP_initial`` 是**滤波器前像**，不是原始空洞密度
------------------------------------------------------------------------------

legacy 模块体在 ``:1179`` 造出随机空洞密度后，又在 ``:1672-1686`` 解
``filter_SIMP @ x ≈ ρ``（L-BFGS-B，``x0=ones``，bounds=[min_SIMP, 1]），并把
``rho_SIMP[list(element_topo_opti_m)]`` 换成 ``res.x``。于是 MMA 的起点
``x0 = ρ[list(element_topo_opti_m)]`` = 这个前像；``filter_SIMP @ x0`` 才是
（近似）初始空洞密度。`verify_golden` 若发现 ``rho_SIMP_initial`` 对不上，
先查这一步是否漏掉 —— 它最容易漏，因为与空洞密度「看着无关」。

这是**驱动层初始化**，不是数值内核 —— 它只决定 MMA 的起点 ``x0``，不参与
求解器迭代。因此不建 reference/fast 两层，只写一份逐行照搬的实现。

------------------------------------------------------------------------------
为什么必须重放 RNG
------------------------------------------------------------------------------

golden L2 的 ``rho_SIMP_initial`` 由 ``seed(1002)`` 之后**两次**
``init_opti_elem_random_hole`` 产生：第一次调用（radius=1.25e-3、k_margin=1.2）
的洞**全部被第二次（radius=0.6e-3、k_margin=1.5）覆盖**，但第一次仍然消耗随机
数流。少调一次或换调用顺序都会让第二次的洞不同 —— 进而 L4 里 MMA 的起点
``x0 = ρ[list(element_topo_opti_m)]`` 与整条轨迹都不同。所以这里必须
「seed → 两次调用 → SIMP 反演」**原样**重放。

------------------------------------------------------------------------------
数值注意
------------------------------------------------------------------------------

- 空洞判据用 ``np.arctan(_y / _x)``（不是 ``arctan2``）—— 照抄 legacy ``:1067``。
- 单元覆盖率是 ``交集面积 / 洞面积``（``:1106-1108``）；shapely 的浮点结果
  依赖版本（golden 在本机 shapely 2.0.1 采集，verify 须在同一环境跑）。
- 反演用 ``TableSIMPProjection``（内核 2 fast 层）—— legacy 的 1000 点查表
  + 线性插值（``:1128-1139``），**不是**闭式反演。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import optimize as sp_optimize
from shapely.geometry import Polygon

__all__ = ["random_hole_density", "initial_rho_simp", "filter_inverse_rho"]


def random_hole_density(
    node: np.ndarray,
    element_S: Sequence[Sequence[int]],
    elements: Iterable[int],
    *,
    radius: float,
    k_margin: float,
    n_count: int,
    m_Rrotor: float,
    m_Rirotor: float,
    parts: int,
    n_interp: int = 20,
    neg_film: bool = False,
    max_trial: float = 1e6,
) -> np.ndarray:
    """随机圆孔 → 逐单元被覆盖比例。对应 ``MMA10:1038-1115``，逐行照搬。

    Parameters
    ----------
    node : (N_node, 4)
        节点表 ``[节点号, x, y, z]``（单位 m）。
    element_S : sequence
        单元表，每行 ``[单元id, n1, n2, n3, 区域号]``。
    elements : iterable of int
        拓扑优化单元 id。**行序 = ``list(elements)``**（与滤波器共用 F4 约定）。
    radius : float
        洞半径（m）。
    k_margin : float
        边界 / 洞间安全系数。生产两次调用分别为 1.2 与 1.5
        （``:1175-1178``）。
    n_count : int
        尝试放置的洞数上限。生产 100。
    m_Rrotor, m_Rirotor, parts : float, float, int
        转子外径、轴径、周期份数（决定允许的圆心域）。
    n_interp : int
        圆的多边形逼近点数（``:1089``）。默认 20，与 legacy 一致。
    neg_film : bool
        取反（``:1112-1113``，生产 False）。
    max_trial : float
        拒收重试上限（``:1042-1058``）。默认 1e6，与 legacy 一致。

    Returns
    -------
    (len(elements),) ndarray
        逐单元覆盖比例，clip 到 [0, 1]。
    """
    _hole_polygons = []
    _hole_centers = []

    x_min = 0.0
    x_max = m_Rrotor
    y_min = 0.0
    y_max = m_Rrotor

    exit_flag = False
    for _i in range(n_count):
        if exit_flag:
            break

        while True:
            max_trial -= 1
            if max_trial <= 0:
                print("init_opti_elem_random_hole: 超过最大尝试次数，退出尝试")
                exit_flag = True
                break

            # 随机产生圆心（legacy :1062-1063 —— 逐 draw 重放，次序不能动）
            _x = np.random.rand() * (x_max - x_min) + x_min
            _y = np.random.rand() * (y_max - y_min) + y_min

            # 判断圆心是否位于区域内，并且加入了 margin（legacy :1065-1072）
            _r = np.sqrt(_x ** 2 + _y ** 2)
            _th = np.arctan(_y / _x)
            _t = _r * _th
            if _r > m_Rrotor - radius * (1 + k_margin / 2) or _r < m_Rirotor + radius * (1 + k_margin / 2):
                continue
            if _t < radius * (1 + k_margin / 2) or _t > _r * (2 * np.pi / parts / 2) - radius * (1 + k_margin / 2):
                continue

            # 判断与之前的洞是否相交，并且加入一个半径长度的 margin（:1078-1085）
            flag_intersected = False
            for _hole_center in _hole_centers:
                if (_x - _hole_center[0]) ** 2 + (_y - _hole_center[1]) ** 2 < (radius * (2 + k_margin)) ** 2:
                    flag_intersected = True
                    break
            if flag_intersected:
                continue

            break

        _th_vec = np.linspace(0, 2 * np.pi, n_interp)

        _hole_polygons.append(
            Polygon([(_x + radius * np.cos(_th), _y + radius * np.sin(_th)) for _th in _th_vec])
        )
        _hole_centers.append([_x, _y])

    # 分别计算每一个三角形单元内的平均值作为初始值（legacy :1094-1115）
    _SIMP_rho_initial = np.ones(len(elements))
    elements_list = list(elements)
    for i_elem_opti, i_elem in enumerate(elements_list):
        _i, _j, _m = element_S[i_elem][1:4]
        xi, xj, xm = node[[_i, _j, _m], 1]
        yi, yj, ym = node[[_i, _j, _m], 2]

        _elem_polygon = Polygon([(xi, yi), (xj, yj), (xm, ym)])

        for _hole_polygon in _hole_polygons:
            if _hole_polygon.intersects(_elem_polygon):
                _area_intersect = _hole_polygon.intersection(_elem_polygon).area
                _area = _hole_polygon.area
                _SIMP_rho_initial[i_elem_opti] = _area_intersect / _area
                break

    _SIMP_rho_initial = _SIMP_rho_initial.clip(min=0, max=1)
    if neg_film:
        _SIMP_rho_initial = 1 - _SIMP_rho_initial

    return _SIMP_rho_initial


def initial_rho_simp(
    node: np.ndarray,
    element_S: Sequence[Sequence[int]],
    element_topo_opti_m: Iterable[int],
    *,
    n_element: int,
    seed: int,
    rho_init_radius: float,
    rho_init_distance: float,
    m_Rrotor: float,
    m_Rirotor: float,
    parts: int,
    simp_inv,
    min_density: float = 1e-6,
    n_holes: int = 100,
) -> np.ndarray:
    """重放 legacy 的初始密度。对应 ``MMA10:42 + :1174-1179``。

    Parameters
    ----------
    node : (N_node, 4) 节点表（单位 m）。生产路径用**动网格插层之后**的表
        （legacy 在 ``:455`` 扩节点、``:1174`` 才做初始化）。
    element_S : 单元表（同样取插层之后的）。
    element_topo_opti_m : 拓扑优化主动单元集合（``list(set)`` 即行序）。
    n_element : 全模型单元数（9599）。
    seed : ``np.random.seed`` 的值（生产 1002）。
    rho_init_radius, rho_init_distance : 生产 1.25e-3 / 4.0e-3 —— 第一次调用的
        ``k_margin = rho_init_distance / rho_init_radius - 2``（legacy ``:1176``）。
    m_Rrotor, m_Rirotor, parts : 见 ``random_hole_density``。
    simp_inv : 可调用 ``(rhop) -> rho``。生产用
        ``TableSIMPProjection(SIMPParams()).invert``（legacy 的 1000 点查表）。
    min_density : clip 下界（``:1179``，生产 ``min_SIMP = 1e-6``）。
    n_holes : 每次调用放置的洞数上限（``:1176-1178``，生产 100）。

    Returns
    -------
    (n_element,) ndarray —— **原始空洞密度**（还没做滤波器前像）。
        与 golden L2 ``rho_SIMP_initial`` 可比的前提是后续再过
        ``filter_inverse_rho``（见模块 docstring 的更正）。
    """
    np.random.seed(seed)
    rho = np.zeros(n_element)
    elems = list(element_topo_opti_m)

    # 第一次调用：radius=rho_init_radius, k_margin=1.2。结果被下一次覆盖，
    # 但**随机数流被消耗** —— 少这一次会让第二次的洞不同，必须照跑。
    rho[elems] = random_hole_density(
        node, element_S, elems,
        radius=rho_init_radius,
        k_margin=rho_init_distance / rho_init_radius - 2,
        n_count=n_holes,
        m_Rrotor=m_Rrotor, m_Rirotor=m_Rirotor, parts=parts,
    )
    # 第二次调用：radius=0.6e-3, k_margin=1.5（生产生效值，``:1178``）
    rho[elems] = random_hole_density(
        node, element_S, elems,
        radius=0.6e-3, k_margin=1.5, n_count=n_holes,
        m_Rrotor=m_Rrotor, m_Rirotor=m_Rirotor, parts=parts,
    )
    rho = simp_inv(rho.clip(min=min_density, max=1))
    return rho


def filter_inverse_rho(
    filter_SIMP,
    rho: np.ndarray,
    element_topo_opti_m: Iterable[int],
    *,
    min_density: float = 1e-6,
) -> np.ndarray:
    """滤波器前像：解 ``filter_SIMP @ x ≈ ρ[topo]`` 并写回。对应 ``MMA10:1672-1686``。

    Parameters
    ----------
    filter_SIMP : (N_topo, N_topo) 稀疏矩阵
        ``compute_filter`` 的行归一化滤波器（golden L1 的 ``filter_SIMP``）。
    rho : (N_element,) ndarray
        随机空洞密度（``initial_rho_simp`` 的产出）。只读，返回新数组。
    element_topo_opti_m : iterable of int
        拓扑优化主动单元集合；``list(set)`` 顺序 = 滤波器行列序。
    min_density : float
        变量下界（legacy ``:1682`` 的 ``min_SIMP = 1e-6``）。

    Returns
    -------
    (N_element,) ndarray —— ``rho[topo] = res.x``，其余位置与输入相同。

    Notes
    -----
    - 目标与梯度（``:1676-1677``）：``err = ‖ρ − W·x‖``、
      ``derr = −2 Wᵀ(ρ − W·x)``。⚠ 是**标量范数**目标，不是逐分量。
    - ``sp_optimize.minimize`` 用默认方法（有 bounds ⇒ L-BFGS-B）、
      ``x0 = ones``、``jac=True`` —— 三个选择都影响轨迹，照抄 legacy。
      L-BFGS-B 确定性强，同环境同输入必得同一 ``res.x``。
    - golden L2 ``rho_SIMP_initial`` 就是这一步之后的值（模块 docstring 的更正）。
    """
    target = rho[list(element_topo_opti_m)]

    def _inv_obj(x):
        err = np.linalg.norm(target - filter_SIMP @ x)
        derr = -2 * filter_SIMP.T @ (target - filter_SIMP @ x)
        return err, derr

    bounds = np.zeros((target.size, 2))
    bounds[:, 0] = min_density
    bounds[:, 1] = 1.0
    res = sp_optimize.minimize(_inv_obj, x0=np.ones(target.size), jac=True, bounds=bounds)

    out = np.array(rho, dtype=float, copy=True)
    out[list(element_topo_opti_m)] = res.x
    return out
