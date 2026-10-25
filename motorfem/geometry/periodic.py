"""周期边界仿射矩阵（阶段 5F：统一两套 gmsh 的符号约定）。

------------------------------------------------------------------------------
★ 两套 legacy 约定（2026-09-23 核实，与设计文档 §5 一致）
------------------------------------------------------------------------------

``gmsh.model.mesh.setPeriodic`` 需要把从属边界的点映射到主动边界的仿射变换。
两个 gmsh 生成器的写法（2×2 旋转块，**按列向量作用** ``x' = A·x``）：

- ``gmsh_motor.py``（SynRM）：``[[cos α, sin α], [−sin α, cos α]]`` = **R(−α)**
  （顺时针，α = 2π/parts）。
- ``gmsh_4p6s_PMSM.py``（PMSM）：``[[cos α, −sin α], [sin α, cos α]]`` = **R(+α)**
  （逆时针）。

⚠ **注意矩阵形式 ≠ 旋转方向**：gmsh 存的是行主序，但变换按列向量作用，所以
``[[cos, sin], [−sin, cos]]`` 是 R(−α) 而非 R(+α)（只看矩阵形式会判反）。

**为什么要统一**：周期方向与主从配对的旋转方向必须一致，否则从属节点配错、
动网格插值符号翻转 —— 这是 gmsh 侧「最易踩的坑」之一（设计文档 §5 原话）。

------------------------------------------------------------------------------
统一接口
------------------------------------------------------------------------------

``periodic_affine(parts, sign)`：返回 gmsh 的 4×4 仿射矩阵（行主序 16 元素）。
``sign=+1`` = R(+2π/parts)（逆时针，PMSM 约定）；``sign=−1`` = R(−2π/parts)
（顺时针，SynRM 约定）。**新代码一律显式传 sign**，不再靠注释区分。
"""

from __future__ import annotations

import math
from typing import List

__all__ = ["periodic_affine", "ROTATION_SIGN_SYNRM", "ROTATION_SIGN_PMSM"]

#: SynRM 生成器（gmsh_motor.py）用 R(−α)（顺时针）
ROTATION_SIGN_SYNRM = -1
#: PMSM 生成器（gmsh_4p6s_PMSM.py）用 R(+α)（逆时针）
ROTATION_SIGN_PMSM = +1


def periodic_affine(parts: int, sign: int = +1) -> List[float]:
    """gmsh 周期边界的 4×4 仿射矩阵（行主序 16 元素列表）。

    Parameters
    ----------
    parts : int 周期份数（旋转角 = 2π/parts）。
    sign : +1 = R(+2π/parts)（逆时针，PMSM 约定）；−1 = R(−2π/parts)
        （顺时针，SynRM 约定）。

    Returns
    -------
    16 元素列表（gmsh 的 affineTransform 行主序；列向量作用 x' = A·x）。

    Notes
    -----
    对应两套 legacy：SynRM（sign=−1）的 2×2 块 = ``[[cos, sin], [−sin, cos]]``
    = R(−α)；PMSM（sign=+1）的块 = ``[[cos, −sin], [sin, cos]]`` = R(+α)。
    """
    a = sign * 2.0 * math.pi / parts
    c, s = math.cos(a), math.sin(a)
    return [
        c, -s, 0, 0,
        s, c, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
