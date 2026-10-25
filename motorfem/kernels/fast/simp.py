"""SIMP 投影的 legacy 保真实现（内核 2 的 fast 层）。

来源：``legacy/pytorch/nonlinear_solver_energy_MMA10.py:1128-1139``

本模块与 ``reference.simp`` **只在 ``invert`` 上不同**：

  - ``reference``：闭式反演 ``ρ = thb + arctanh(ρ_c·A − T)/β``
  - ``fast``（本模块）：legacy 的 **1000 点查表 + 线性插值**

``project`` 与 ``derivative`` 两层数学完全同一（legacy 没有在那里引入近似），
故本模块**直接继承** ``reference`` 的实现，不复制代码。这符合
``kernels/README.md`` 的分层判据：只在有真实差异处才分两层。

------------------------------------------------------------------------------
查表反演的误差来源与量级
------------------------------------------------------------------------------

legacy 用::

    _x = np.linspace(0, 1, 1000)
    SIMP_invfunc_interp = sp_interp.interp1d(SIMP_func(_x), _x)

以投影值为自变量做**分段线性**插值。误差有两个来源：

1. **表的分辨率**：``_x`` 的步长是 1/999。投影函数在阈值附近被 β = 8 锐化，
   所以 ``SIMP_func(_x)`` 的结点在投影值上**分布极不均匀**——阈值附近稀疏、
   两端密。误差因此不是均匀的，在投影值接近两端时最大。
2. **线性插值对被反演曲线的曲率误差**：投影函数的反演在 ρ→0 与 ρ→1 处
   曲率很大（幂律 + arctanh 的复合），分段线性在那里误差最大。

实测（见 tests/kernels/test_simp.py）：最大绝对误差约 **4e-4**（在 ρ 上），
相对误差在 ρ 中等区域约 1e-3。**这不是可忽略的量级**——它直接进入设计
变量的初始化环节（``MMA10:1179``），会让「用户指定的初始空洞半径」
与实际得到的密度场有 ~0.04% 的偏差。

保留理由：保真搬移是阶段 1 的目标（迁移期要以 legacy 为基准做位级对齐）。
是否改用闭式反演，留待迁移完成后的优化阶段决策——届时本模块可整体替换为
``reference.simp.SIMPProjection``，只要 parity 测试放宽到表误差量级。
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate as sp_interp

from ..reference.simp import SIMPParams, SIMPProjection

__all__ = ["SIMPParams", "SIMPProjection", "TableSIMPProjection", "LEGACY_TABLE_SIZE"]

#: legacy 的表分辨率（``MMA10:1128`` 的 ``np.linspace(0, 1, 1000)``）
LEGACY_TABLE_SIZE = 1000


class TableSIMPProjection(SIMPProjection):
    """legacy 的查表反演。``project`` / ``derivative`` 继承自 reference。

    Parameters
    ----------
    params : SIMPParams
    table_size : int
        表的点数。默认 1000，与 legacy 一致。

    Notes
    -----
    表在**构造时**建立（对应 legacy 在模块导入期建立 ``SIMP_invfunc_interp``），
    从而满足「导入不得产生副作用」——legacy 把这一步放在模块级，
    本实现改为显式构造的副作用。
    """

    def __init__(
        self,
        params: SIMPParams = SIMPParams(),
        table_size: int = LEGACY_TABLE_SIZE,
    ):
        super().__init__(params)
        self.table_size = int(table_size)

        _x = np.linspace(0.0, 1.0, self.table_size)
        self._table = sp_interp.interp1d(self.project(_x), _x)

    def invert(self, rho_p: np.ndarray) -> np.ndarray:
        """查表反演。

        与 legacy 的行为一致：输入落在表的取值范围之外时 ``interp1d`` 抛
        ``ValueError``（``bounds_error`` 默认为 True）。**不做静默裁剪**，
        理由同 ``reference.invert``——越界通常意味着上游算错了。

        注意表的取值范围是 ``[project(0), project(1)] = [min_density, 1]``，
        与 ``reference.invert`` 的合法域一致。
        """
        rho_p = np.asarray(rho_p, dtype=float)
        return self._table(rho_p)

    def __repr__(self) -> str:
        return f"TableSIMPProjection({super().__repr__()}, n={self.table_size})"
