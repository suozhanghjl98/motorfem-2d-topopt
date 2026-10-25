"""三段式平行直线磁障的尺寸估算（阶段 5F：参数预填层）。

移植 ``legacy/design.m:23-47`` 的「等磁路宽」公式 —— 给定磁障角度序列与
气隙磁密比，估算各导磁块厚度 / 层总厚度 / 空气厚度 / 磁障长度，作为 GUI
参数预填层（不用手算磁障圆心/半径）。

------------------------------------------------------------------------------
几何约定（design.m:7-17）
------------------------------------------------------------------------------

转子外半径为 **1**（归一化）：``Ri2`` = 转子内半径比，``th_vec`` = 各磁障的
半极角（从**最外侧**磁障开始，增序），``np`` = 极对数，``kB`` = 气隙磁密比。
``kb`` = 磁障层数。

------------------------------------------------------------------------------
公式（等磁路宽：相邻磁障间导磁块的磁路宽度相等）
------------------------------------------------------------------------------

::

    d_vec(1)   = kB/np · (1 − cos(np·th1))                # 最外导磁块
    d_vec(kb+1)= kB/np · cos(np·th_kb)                    # 最内导磁块
    d_vec(i)   = kB/np · (cos(np·th_{i-1}) − cos(np·th_i))
    w_vec(1)   = 1 − sin(π/2np − th1) − Ri2
    w_vec(kb+1)= sin(π/2np − th_kb)
    w_vec(i)   = sin(π/2np − th_{i-1}) − sin(π/2np − th_i)
    air_vec    = w_vec − d_vec
    l_vec(i)   = L0 · (1 − (1/L1)·sin(π/2np − th_i))
    L0 = 1 − Ri2/cos(π/2np) + Ri2·tan(π/2np)
    L1 = Ri2·tan(π/2np) / tan(π/4np)

（``design.m:24-47``，``th_vec`` 用 MATLAB 1 基索引 —— 这里 0 基：``th_vec[0]``
对应 ``design.m`` 的 ``th_vec(1)``。）
"""

from __future__ import annotations

import math
from typing import List, Sequence

__all__ = ["barrier_design_sizes"]


def barrier_design_sizes(
    th_vec: Sequence[float],
    np_: int,
    kB: float = 0.5,
    Ri2: float = 0.4,
) -> dict:
    """估算三段式磁障的尺寸。对应 ``design.m:23-47``。

    Parameters
    ----------
    th_vec : 磁障半极角序列（rad，**从最外侧**开始，增序）。
    np_ : 极对数。
    kB : 气隙磁密比（默认 0.5，design.m 的输入）。
    Ri2 : 转子内半径比（外半径 = 1；默认 0.4）。

    Returns
    -------
    dict —— ``d_vec``（导磁块厚度，kb+1 个）、``w_vec``（层总厚度）、
    ``air_vec``（空气厚度）、``l_vec``（磁障长度，kb 个）。
    全部以转子外半径归一化（乘 Rrotor 得米）。
    """
    kb = len(th_vec)
    th = [float(t) for t in th_vec]

    # 导磁块最小厚度（design.m:23-28）
    d_vec = [0.0] * (kb + 1)
    d_vec[0] = kB * 1 / np_ * (1 - math.cos(np_ * th[0]))
    d_vec[kb] = kB * 1 / np_ * (math.cos(np_ * th[kb - 1]))
    for i in range(1, kb):
        d_vec[i] = kB * 1 / np_ * (math.cos(np_ * th[i - 1]) - math.cos(np_ * th[i]))

    # 各层总厚度（design.m:31-36）
    w_vec = [0.0] * (kb + 1)
    w_vec[0] = 1 - math.sin(math.pi / 2 / np_ - th[0]) - Ri2
    w_vec[kb] = math.sin(math.pi / 2 / np_ - th[kb - 1])
    for i in range(1, kb):
        w_vec[i] = (math.sin(math.pi / 2 / np_ - th[i - 1])
                    - math.sin(math.pi / 2 / np_ - th[i]))

    air_vec = [w - d for w, d in zip(w_vec, d_vec)]

    # 磁障长度（design.m:42-47）
    L0 = 1 - Ri2 / math.cos(math.pi / 2 / np_) + Ri2 * math.tan(math.pi / 2 / np_)
    L1 = Ri2 * math.tan(math.pi / 2 / np_) / math.tan(math.pi / 4 / np_)
    l_vec = [L0 * (1 - 1 / L1 * math.sin(math.pi / 2 / np_ - t)) for t in th]

    return {"d_vec": d_vec, "w_vec": w_vec, "air_vec": air_vec, "l_vec": l_vec}
