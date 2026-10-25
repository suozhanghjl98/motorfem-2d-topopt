"""力学单元矩阵的批量实现（内核 5 的 fast 层）。

与 ``reference/mech_element_matrices.py`` **逐位相同**，只是把逐单元循环
换成 numpy 批量运算。

------------------------------------------------------------------------------
保持 legacy 的运算顺序
------------------------------------------------------------------------------

legacy 写的是::

    Kmech0elem[i_elem] = Delta * Lmechcoef[i_elem].T @ E0mechcoef[i_elem] @ Lmechcoef[i_elem]

Python 里 ``*`` 与 ``@`` **同优先级、左结合**，所以这实际是::

    ((Delta * L.T) @ E) @ L

**不是** ``Delta * (L.T @ E @ L)``。两者在浮点下不同（前者先把 Lᵀ 整体缩放，
后者最后缩放）。本模块用 ``(delta[:,None,None] * LT) @ D @ L`` 复现左结合形式，
由 parity 测试以 ``np.array_equal`` 锁定。
"""

from __future__ import annotations

import numpy as np

from ..reference.mech_element_matrices import (
    MECH_LOAD_NOTE,
    R0ELEM_FACTOR,
    R2ELEM_FACTOR,
    MechElements,
    MechLineLoads,
    MechMaterial,
    mech_element_matrices as _ref_mech_element_matrices,
)

__all__ = [
    "MechMaterial",
    "MechElements",
    "MechLineLoads",
    "mech_element_matrices",
    "mech_body_force",
    "mech_line_loads",
    "MECH_LOAD_NOTE",
    "R0ELEM_FACTOR",
    "R2ELEM_FACTOR",
]


def mech_element_matrices(
    node: np.ndarray, tri: np.ndarray, material: MechMaterial = MechMaterial()
) -> MechElements:
    """批量计算力学单元矩阵。与 ``reference`` 版本逐位相同。

    对应 legacy ``MMA10:557-603``。
    """
    n_elem = tri.shape[0]
    E, nu = material.E, material.nu

    x = node[tri, 0]
    y = node[tri, 1]
    b = np.roll(y, -1, axis=1) - np.roll(y, -2, axis=1)
    c = -np.roll(x, -1, axis=1) + np.roll(x, -2, axis=1)
    delta = (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]) / 2.0

    if np.any(delta <= 0.0):
        raise ValueError(
            f"力学网格有 {int(np.sum(delta <= 0.0))} 个单元面积非正（须逆时针编号）"
        )

    D = np.zeros((3, 3))
    D[0, 0] = E / (1 - nu ** 2)
    D[0, 1] = E * nu / (1 - nu ** 2)
    D[1, 0] = E * nu / (1 - nu ** 2)
    D[1, 1] = E / (1 - nu ** 2)
    D[2, 2] = E / 2 / (1 + nu)

    inv2d = 1 / (2 * delta)                          # (N,)，与 legacy 的 1/(2*Delta) 同式

    L = np.zeros((n_elem, 3, 6))
    # 每列赋值：用 inv2d[:,None]*列，与 legacy 的 1/(2*Delta)*b 逐位一致
    L[:, 0, 0] = inv2d * b[:, 0]
    L[:, 0, 2] = inv2d * b[:, 1]
    L[:, 0, 4] = inv2d * b[:, 2]
    L[:, 1, 1] = inv2d * c[:, 0]
    L[:, 1, 3] = inv2d * c[:, 1]
    L[:, 1, 5] = inv2d * c[:, 2]
    L[:, 2, 0] = inv2d * c[:, 0]
    L[:, 2, 1] = inv2d * b[:, 0]
    L[:, 2, 2] = inv2d * c[:, 1]
    L[:, 2, 3] = inv2d * b[:, 1]
    L[:, 2, 4] = inv2d * c[:, 2]
    L[:, 2, 5] = inv2d * b[:, 2]

    # ⚠ 左结合：((delta * Lᵀ) @ D) @ L，与 legacy 的 Delta * L.T @ E @ L 同序
    LT = L.transpose(0, 2, 1)                        # (N, 6, 3)
    K = (delta[:, None, None] * LT) @ D @ L          # (N, 6, 6)

    R0 = -delta / 2                                  # 见 MECH_LOAD_NOTE

    return MechElements(L=L, D=D, K=K, R0=R0, area=delta, tri=tri)


# =============================================================================
# 载荷（离心体积力与边界面力）——  kernel 5 的 fast 层
# =============================================================================
#
# 与 ``reference`` 版本**逐位相同**，把逐单元/逐线元循环换成批量运算。
# ⚠ 只有一处必须小心：``boundary_length`` 在 legacy 里是循环里的 ``+=`` 顺序累加，
# 而 ``np.sum`` 走**成对求和**、结果末位不同 ⇒ 这里用 ``np.cumsum(...)[-1]``
# （顺序累加），parity 测试用 ``np.array_equal`` 而不是 ``allclose`` 钉住。
#
# 这两个量在 legacy 里**只算一次**（模块级 ``if True:`` 块），不在任何迭代里 ——
# 所以本内核的性能守卫（成员：``get_Kmech_vectorized`` 的装配）不覆盖它们，
# 这里也**不加**性能断言（无意义的 ~1×）。写两层是为了与内核 5 的其余部分同形。

def mech_body_force(
    node: np.ndarray,
    tri: np.ndarray,
    material: MechMaterial = MechMaterial(),
    *,
    angular_speed: float,
    rot_center=(0.0, 0.0),
) -> tuple:
    """批量计算离心体积力。与 ``reference`` 版本逐位相同。对应 ``MMA10:620-632``。"""
    m = material.rho
    xc, yc = rot_center

    x, y = node[:, 0], node[:, 1]
    # legacy 的 (xi/3 + xj/3 + xm/3)：三项各自除 3 再相加，**不是** (xi+xj+xm)/3
    x_c3 = x[tri[:, 0]] / 3 + x[tri[:, 1]] / 3 + x[tri[:, 2]] / 3
    y_c3 = y[tri[:, 0]] / 3 + y[tri[:, 1]] / 3 + y[tri[:, 2]] / 3

    fx = m * angular_speed ** 2 * (x_c3 - xc)
    fy = m * angular_speed ** 2 * (y_c3 - yc)
    return fx, fy


def mech_line_loads(
    node: np.ndarray,
    line_tri: np.ndarray,
    loaded_lines,
    *,
    Fload: float,
    rot_angle_deg: float = 45.0,
) -> MechLineLoads:
    """批量计算边（线元）载荷。与 ``reference`` 版本逐位相同。对应 ``MMA10:605-661``。"""
    line_tri = np.asarray(line_tri, dtype=int)
    loaded = [int(i) for i in loaded_lines]
    idx = np.asarray(loaded, dtype=np.int64) if loaded else np.zeros(0, np.int64)

    R0 = np.zeros(line_tri.shape[0])
    if idx.size == 0:
        return MechLineLoads(R0=R0, fx=np.zeros(0), fy=np.zeros(0),
                             boundary_length=0.0)

    ni, nj = line_tri[idx, 0], line_tri[idx, 1]
    xi, yi = node[ni, 0], node[ni, 1]
    xj, yj = node[nj, 0], node[nj, 1]

    lengths = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
    # 花式索引赋值 = 后写覆盖，与 legacy 的逐个 ``Rmech20elem[i] = -_L/2`` 同语义
    R0[idx] = R2ELEM_FACTOR * lengths

    r_bar = np.sqrt((xi + xj) ** 2 + (yi + yj) ** 2) / 2
    px = (xi + xj) / 2 * Fload / r_bar * lengths
    py = (yi + yj) / 2 * Fload / r_bar * lengths

    rot_angle = rot_angle_deg * np.pi / 180
    # ⚠ np.cumsum 而不是 np.sum：顺序累加，与 legacy 的 ``_len_boundary += _l`` 同序
    boundary_length = float(np.cumsum(lengths)[-1])
    fx = (px * np.cos(rot_angle) - py * np.sin(rot_angle)) / boundary_length
    fy = (px * np.sin(rot_angle) + py * np.cos(rot_angle)) / boundary_length

    return MechLineLoads(R0=R0, fx=fx, fy=fy, boundary_length=boundary_length)
