"""磁障几何基元（阶段 5F：几何参数化）。

从 ``legacy/pytorch/my_gmsh/barrier_test.py`` 提取的**纯几何**函数（磁障参数 →
节点位置），加平面旋转/镜像基元（原分散在多个 gmsh 脚本与 MMA10:177-190）。

**为什么独立成包**：这些函数不依赖 gmsh（生成脚本会弹窗阻塞，CLAUDE.md 有记录），
是「磁障参数 → 几何」的可测纯函数（设计文档 §5 说的 `get_barrier_nodes` 天然就是
这条映射，改造量最小）。

------------------------------------------------------------------------------
`get_barrier_nodes` 的几何（对照 ``barrier_test.py:44-109``）
------------------------------------------------------------------------------

输入 ``Rrotor/drib/dc1/R1/dc2/R2/r`` 描述一个磁障段（外侧弧圆心 ``(dc1,dc1)/√2``
半径 ``R1``、内侧弧圆心 ``(dc2,dc2)/√2`` 半径 ``R2``、磁桥厚 ``drib``、倒角半径
``r``、转子半径 ``Rrotor``）：

1. 外侧弧 + 磁桥的倒角切点：两圆交点（圆心 ``(dc1,dc1)/√2`` 半径 ``R1+r`` 与
   原点半径 ``Rrotor-drib-r``）；
2. 倒角圆心记录 + 弧上节点（``get_circle_point`` 把点投到圆上）；
3. 对称轴（y=x）镜像补齐剩余节点；
4. 返回 8 个节点 + 6 个圆心（带圆心是为了后续画倒角圆弧）。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "rot_x",
    "rot_y",
    "mirror",
    "get_circle_intersection",
    "get_circle_point",
    "get_barrier_nodes",
]


# ---------------------------------------------------------------- 旋转 / 镜像
def rot_x(x, y, angle):
    """绕原点旋转的 x 分量（MMA10:177-178 的 ``_rot_x``）。"""
    return x * np.cos(angle) - y * np.sin(angle)


def rot_y(x, y, angle):
    """绕原点旋转的 y 分量（MMA10:181-182 的 ``_rot_y``）。"""
    return x * np.sin(angle) + y * np.cos(angle)


def mirror(x, y, angle):
    """关于角平分线 ``angle/2`` 的镜像（MMA10:189-190 的 ``_mirror``）。

    ``mirror(x, y, θ) = (rot(−θ), rot_y(x, −y, θ))`` —— 注意 y 取负号，这是
    legacy 的约定（非普通旋转），对称单元配对靠它。
    """
    return rot_x(x, y, -angle), rot_y(x, -y, angle)


# ---------------------------------------------------------------- 圆几何
def get_circle_intersection(x1, y1, r1, x2, y2, r2) -> Tuple[float, float, float, float]:
    """两圆交点。返回 ``(xc1, yc1, xc2, yc2)``。对应 ``barrier_test.py:14-34``。

    余弦定理求夹角 + 圆心连线角度 ± 偏移。⚠ 前提 ``d < r1+r2``（断言）。
    """
    assert r1 > 0
    assert r2 > 0

    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    assert d < r1 + r2

    th1 = np.arccos((r2 ** 2 - d ** 2 - r1 ** 2) / (-2 * d * r1))
    th0 = np.arctan2((y2 - y1), (x2 - x1))

    xc1 = x1 + r1 * np.cos(th0 + th1)
    yc1 = y1 + r1 * np.sin(th0 + th1)
    xc2 = x1 + r1 * np.cos(th0 - th1)
    yc2 = y1 + r1 * np.sin(th0 - th1)
    return xc1, yc1, xc2, yc2


def get_circle_point(xc, yc, rc, xp, yp) -> Tuple[float, float]:
    """把点 ``(xp,yp)`` 沿径向投到圆 ``(xc,yc,rc)`` 上。``barrier_test.py:37-41``。"""
    dx = xp - xc
    dy = yp - yc
    d = np.sqrt(dx ** 2 + dy ** 2)
    return xc + rc * dx / d, yc + rc * dy / d


def get_barrier_nodes(
    Rrotor: float, drib: float, dc1: float, R1: float,
    dc2: float, R2: float, r: float,
) -> Tuple[List[List[float]], List[List[float]]]:
    """计算转子磁障段的所有节点位置（含圆心，画倒角弧用）。

    对应 ``barrier_test.py:44-109``，逐段照搬。

    Parameters
    ----------
    Rrotor : 转子半径
    drib : 磁桥厚度
    dc1, R1 : 外侧磁障弧圆心距离 / 半径
    dc2, R2 : 内侧磁障弧圆心距离 / 半径
    r : 倒角半径

    Returns
    -------
    ``(node_vec, center_vec)`` —— 8 个节点（对称补齐）+ 8 个圆心（画倒角弧用，
        含原点与两侧磁障圆心）。
    """
    node_vec = []
    center_vec = []

    # 外侧磁障与磁桥的倒角切点
    xc1, yc1, xc2, yc2 = get_circle_intersection(
        dc1 / np.sqrt(2), dc1 / np.sqrt(2), R1 + r, 0, 0, Rrotor - drib - r)
    if xc1 < yc1:
        # 令 (xc1,yc1) 处于对称轴下侧
        xc1, yc1 = xc2, yc2

    center_vec.append([xc1, yc1])
    _x, _y = get_circle_point(dc1 / np.sqrt(2), dc1 / np.sqrt(2), R1, xc1, yc1)
    node_vec.append([_x, _y])
    _x, _y = get_circle_point(0, 0, Rrotor - drib, xc1, yc1)
    node_vec.append([_x, _y])

    center_vec.append([0, 0])

    # 内侧磁障与磁桥的倒角切点
    xc1, yc1, xc2, yc2 = get_circle_intersection(
        dc2 / np.sqrt(2), dc2 / np.sqrt(2), R2 - r, 0, 0, Rrotor - drib - r)
    if xc1 < yc1:
        xc1, yc1 = xc2, yc2

    center_vec.append([xc1, yc1])
    _x, _y = get_circle_point(0, 0, Rrotor - drib, xc1, yc1)
    node_vec.append([_x, _y])
    _x, _y = get_circle_point(dc2 / np.sqrt(2), dc2 / np.sqrt(2), R2, xc1, yc1)
    node_vec.append([_x, _y])

    center_vec.append([dc2 / np.sqrt(2), dc2 / np.sqrt(2)])

    # 对称性补齐剩余节点
    node_vec.append([node_vec[3][1], node_vec[3][0]])
    node_vec.append([node_vec[2][1], node_vec[2][0]])
    node_vec.append([node_vec[1][1], node_vec[1][0]])
    node_vec.append([node_vec[0][1], node_vec[0][0]])

    center_vec.append([center_vec[2][1], center_vec[2][0]])
    center_vec.append([center_vec[1][1], center_vec[1][0]])
    center_vec.append([center_vec[0][1], center_vec[0][0]])

    center_vec.append([dc1 / np.sqrt(2), dc1 / np.sqrt(2)])

    return node_vec, center_vec
