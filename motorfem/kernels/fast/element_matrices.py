"""电磁单元矩阵的批量实现（内核 4 的 fast 层）。

与 ``reference/element_matrices.py`` **逐位相同**，只是把逐单元 Python 循环
换成 numpy 批量运算。

------------------------------------------------------------------------------
算式形状必须与 legacy 逐字一致
------------------------------------------------------------------------------

本层唯一的风险点是**浮点结合顺序**。以下写法看着等价，实际浮点结果不同::

    legacy / reference                错误但"看起来等价"的写法
    -------------------------------   -----------------------------
    1 / 4 / Delta * (bi*bi + ci*ci)   (bi*bi + ci*ci) / (4 * Delta)
    1 / 2 / Delta * bi                bi / (2 * Delta)
    -1 / 2 / Delta * ci               -ci / (2 * Delta)

因为 IEEE 浮点下 ``(1/4)/D`` 与 ``1/(4*D)`` 不保证相等（D 为有限浮点数时
前者是先做精确的 0.25 除法再做一次除法，后者是先算 4*D 再除）。
批量实现刻意保留 legacy 的左结合形式，并由 parity 测试以
``np.array_equal`` 锁定——一旦有人"顺手整理"成更自然的写法，测试会失败。

------------------------------------------------------------------------------
为什么值得批量化
------------------------------------------------------------------------------

本内核在 legacy 里是 ``for i_elem in range(N_element)`` 的纯 Python 循环
（``MMA10:509-542``），在 11160 个单元上每次都要跑一遍。虽然它只在中途
调用（不在 Newton 内循环），但批量化的收益以实测为准（见测试中的性能守卫）。
"""

from __future__ import annotations

import numpy as np

from ..reference.element_matrices import (
    ElectromagneticElements,
    current_area,
    expand_K0,
)
from ..reference.magnetostatic import element_geometry

__all__ = [
    "ElectromagneticElements",
    "electromagnetic_elements",
    "expand_K0",
    "current_area",
]


def electromagnetic_elements(
    node: np.ndarray, tri: np.ndarray
) -> ElectromagneticElements:
    """批量计算静磁场单元量。与 ``reference`` 版本逐位相同。

    对应 legacy ``MMA10:509-542``（逐单元循环）。

    Notes
    -----
    与 reference 版的差别仅在实现方式：这里把 3 个局部节点的循环展开成
    ``(N_elem, 3)`` 的整列运算。**算式形状保持 legacy 的左结合形式**，
    见模块文档。
    """
    geom = element_geometry(node, tri)              # 含 Δ > 0 断言
    b, c, delta = geom.b, geom.c, geom.delta
    n_elem = tri.shape[0]

    # 保持 legacy 的左结合：(1/4)/delta，不是 1/(4*delta)
    inv4d = 1 / 4 / delta                            # (N,)
    b_inv2d = 1 / 2 / delta                          # (N,)

    bb = b * b                                       # (N, 3)
    cc = c * c

    K0 = np.empty((n_elem, 6), dtype=float)
    # 顺序 [S_ii, S_ij, S_im, S_jj, S_jm, S_mm]
    K0[:, 0] = inv4d * (bb[:, 0] + cc[:, 0])
    K0[:, 1] = inv4d * (b[:, 0] * b[:, 1] + c[:, 0] * c[:, 1])
    K0[:, 2] = inv4d * (b[:, 0] * b[:, 2] + c[:, 0] * c[:, 2])
    K0[:, 3] = inv4d * (bb[:, 1] + cc[:, 1])
    K0[:, 4] = inv4d * (b[:, 1] * b[:, 2] + c[:, 1] * c[:, 2])
    K0[:, 5] = inv4d * (bb[:, 2] + cc[:, 2])

    R0 = delta / 3
    area = delta.copy()

    # legacy 约定：Bxcoef = +b/(2Δ)，Bycoef = -c/(2Δ)
    # -inv2d * c 与 legacy 的 -1/2/D*ci 逐位相同（IEEE 取负是精确的，
    # 且除法对符号对称：-(0.5/D) == (-0.5)/D）。
    Bxcoef = b_inv2d[:, None] * b                    # (N, 3)
    Bycoef = -b_inv2d[:, None] * c

    return ElectromagneticElements(
        K0=K0, R0=R0, Bxcoef=Bxcoef, Bycoef=Bycoef, area=area, geom=geom
    )
