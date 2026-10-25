"""tests/geometry/test_geometry.py —— 阶段 5F：几何参数化

验证核心（来源 A：解析性质 / 手算）：

- ``base.get_circle_intersection`` / ``get_circle_point``：解析几何性质（交点在
  两圆上、投点距离 = 半径）；
- ``base.get_barrier_nodes``：8 节点关于 y=x 对称（对称补齐正确）；
- ``periodic.periodic_affine``：两套 legacy 旋转矩阵（SynRM R(+α) / PMSM
  R(−α)，与实测一致，设计文档 §5 记载相反）；
- ``design.barrier_design_sizes``：design.m:23-47 公式的手算对照。

运行::

    python -m pytest tests/geometry/test_geometry.py -v
"""

import math

import numpy as np
import pytest

from motorfem.geometry.base import (
    get_barrier_nodes, get_circle_intersection, get_circle_point, mirror, rot_x, rot_y,
)
from motorfem.geometry.design import barrier_design_sizes
from motorfem.geometry.periodic import (
    ROTATION_SIGN_PMSM, ROTATION_SIGN_SYNRM, periodic_affine,
)


# ---------------------------------------------------------------- base
def test_circle_intersection_points_lie_on_both_circles():
    """两圆交点必须在两个圆上（解析性质）。"""
    x1, y1, r1 = 0.0, 0.0, 5.0
    x2, y2, r2 = 6.0, 0.0, 5.0          # 圆心距 6 < r1+r2=10
    xa, ya, xb, yb = get_circle_intersection(x1, y1, r1, x2, y2, r2)
    for (px, py) in ((xa, ya), (xb, yb)):
        assert abs(math.hypot(px - x1, py - y1) - r1) < 1e-12
        assert abs(math.hypot(px - x2, py - y2) - r2) < 1e-12


def test_circle_intersection_rejects_disjoint():
    with pytest.raises(AssertionError):
        get_circle_intersection(0, 0, 1, 10, 0, 1)   # 相离 d=10 > 2


def test_circle_point_projects_to_circle():
    """get_circle_point 把任意点投到圆上（|返回点 − 圆心| = 半径）。"""
    x, y = get_circle_point(1.0, 2.0, 3.0, 10.0, -5.0)
    assert abs(math.hypot(x - 1.0, y - 2.0) - 3.0) < 1e-12


def test_barrier_nodes_symmetric_about_diagonal():
    """get_barrier_nodes 的 8 个节点关于 y=x 对称（对称补齐正确）。

    参数取 barrier_test.py 的真实示例（Rrotor=44.75, r=0.5, drib=0.4,
    dc1=97.6, R1=55.0, dc2=97.6, R2=57.2）—— 保证圆几何可相交。
    """
    nodes, centers = get_barrier_nodes(
        Rrotor=44.75, drib=0.4, dc1=97.6, R1=55.0, dc2=97.6, R2=57.2, r=0.5)
    assert len(nodes) == 8 and len(centers) == 8
    for i in range(4):
        # 节点 i 与节点 7-i 关于 y=x 对称（barrier_test 的补齐逻辑）
        assert abs(nodes[i][0] - nodes[7 - i][1]) < 1e-9
        assert abs(nodes[i][1] - nodes[7 - i][0]) < 1e-9


def test_rotation_mirror_basics():
    """rot_x/rot_y/mirror 的解析性质。"""
    # rot 90° 逆时针：(1,0) → (0,1)
    assert abs(rot_x(1, 0, math.pi / 2) - 0) < 1e-12
    assert abs(rot_y(1, 0, math.pi / 2) - 1) < 1e-12
    # mirror(·, θ=π/2) = 关于 45° 线对称：点 (a,b) → (b,a)
    mx, my = mirror(3.0, 5.0, math.pi / 2)
    assert abs(mx - 5.0) < 1e-12 and abs(my - 3.0) < 1e-12


# ---------------------------------------------------------------- periodic
def test_periodic_affine_matches_legacy_signs():
    """两套 legacy 旋转符号（矩阵按列向量作用）：SynRM R(−α)、PMSM R(+α)。

    ⚠ 设计文档 §5 记载（SynRM R(−α)、PMSM R(+α)）与实测一致 —— 只看矩阵形式
    会把 [[cos,sin],[−sin,cos]] 误判为 R(+α)，实际列向量作用是 R(−α)。
    """
    a = 2 * math.pi / 4
    # SynRM（sign=−1）：块 = [[cos, sin], [−sin, cos]] = R(−α)（gmsh_motor.py）
    syn = periodic_affine(4, ROTATION_SIGN_SYNRM)
    assert syn[0] == pytest.approx(math.cos(a)) and syn[1] == pytest.approx(math.sin(a))
    assert syn[4] == pytest.approx(-math.sin(a)) and syn[5] == pytest.approx(math.cos(a))
    # PMSM（sign=+1）：块 = [[cos, −sin], [sin, cos]] = R(+α)（gmsh_4p6s_PMSM.py）
    pmsm = periodic_affine(4, ROTATION_SIGN_PMSM)
    assert pmsm[1] == pytest.approx(-math.sin(a)) and pmsm[4] == pytest.approx(math.sin(a))
    assert not np.array_equal(syn[:4], pmsm[:4])


def test_periodic_affine_rotation_action():
    """旋转作用（列向量）：SynRM R(−90°)(1,0) → (0,−1)；PMSM R(+90°)(1,0) → (0,1)。"""
    syn = np.array(periodic_affine(4, ROTATION_SIGN_SYNRM)).reshape(4, 4)[:2, :2]
    pmsm = np.array(periodic_affine(4, ROTATION_SIGN_PMSM)).reshape(4, 4)[:2, :2]
    v = np.array([1.0, 0.0])
    assert np.allclose(syn @ v, [0.0, -1.0])
    assert np.allclose(pmsm @ v, [0.0, 1.0])


# ---------------------------------------------------------------- design
def test_barrier_design_sizes_hand_calculation():
    """design.m:23-47 公式手算对照（kb=1 单层磁障）。"""
    th = [math.pi / 4 * 0.7]
    np_, kB, Ri2 = 2, 0.5, 0.4
    out = barrier_design_sizes(th, np_, kB=kB, Ri2=Ri2)

    t = th[0]
    d0 = kB / np_ * (1 - math.cos(np_ * t))
    d1 = kB / np_ * math.cos(np_ * t)
    w0 = 1 - math.sin(math.pi / 2 / np_ - t) - Ri2
    w1 = math.sin(math.pi / 2 / np_ - t)
    L0 = 1 - Ri2 / math.cos(math.pi / 2 / np_) + Ri2 * math.tan(math.pi / 2 / np_)
    L1 = Ri2 * math.tan(math.pi / 2 / np_) / math.tan(math.pi / 4 / np_)
    l0 = L0 * (1 - 1 / L1 * math.sin(math.pi / 2 / np_ - t))

    assert out["d_vec"] == pytest.approx([d0, d1])
    assert out["w_vec"] == pytest.approx([w0, w1])
    assert out["air_vec"] == pytest.approx([w0 - d0, w1 - d1])
    assert out["l_vec"] == pytest.approx([l0])
    # 物理性质：厚度非负、空气厚度非负
    assert all(v >= 0 for v in out["d_vec"] + out["w_vec"] + out["air_vec"] + out["l_vec"])


def test_barrier_design_sizes_positive_for_default():
    """design.m 示例参数（4 层磁障）全部尺寸为正且随层数合理。"""
    th = [math.pi / 4 * x for x in (0.7, 0.8, 0.85, 0.9)]
    out = barrier_design_sizes(th, np_=2)
    assert len(out["d_vec"]) == 5 and len(out["l_vec"]) == 4
    assert all(v > 0 for v in out["d_vec"] + out["w_vec"] + out["air_vec"] + out["l_vec"])
    # 全部 > 0 已断言；空气层厚度分布由 design.m 公式决定，不做拍脑袋的单调断言
