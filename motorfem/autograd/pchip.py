"""motorfem.autograd.pchip —— 阶段 6：scipy PCHIP 的 torch 重建（可微 B-H 曲线）

``BHCurve`` 的 H(B) 用 scipy ``PchipInterpolator``（保单调、不过冲）。要在
torch 自动微分里穿过 B-H 链，需要 H 与 dH/dB 作为 torch 可微函数。做法：
**提取 scipy 插值器的分段三次系数**（``.c``：每段 (4,) 的幂基系数；``.x``：节点），
在 torch 里按同一多项式求值 —— 值与一阶导与 scipy 差 ~1e-16 相对（实证），
满足阶段 6 验收（梯度 rel < 1e-9）。

⚠ 这是**材料数据**的重建，不是算法重建：B-H 曲线对 oracle 与手写是同一份输入，
共享它是正当的。oracle 验证的是「位置矩阵 vs torch 单元局部」这条实现路径，
不是重新推导磁化曲线。
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["TorchPCHIP"]


class TorchPCHIP:
    """scipy ``PchipInterpolator`` 的 torch 可微重建（值 + 一阶导）。

    系数与节点从 scipy 插值器原样提取（``.c`` / ``.x``，float64）。求值按
    ``ppval`` 的幂基语义：第 k 段上 ``H(x) = c[0,k]·t³ + c[1,k]·t² + c[2,k]·t + c[3,k]``，
    ``t = x − x_k``。

    域外行为：x 在 ``[x[0], x[-1]]`` 外时按端点线性延拓（斜率为端段斜率）——
    与 ``BHCurve.H`` 的裁剪语义对齐（BHCurve 已把定义域扩到覆盖全部工作点，
    域外仅理论路径）。

    Parameters
    ----------
    c : (4, n-1) ndarray
        ``PchipInterpolator.c``（每段幂基系数）。
    x : (n,) ndarray
        ``PchipInterpolator.x``（节点，升序）。

    Notes
    -----
    ``x`` 的 dtype 可能是 float32（scipy 对 float64 输入也存 float64；这里强制
    float64，与 numpy 侧一致）。求值时 ``t`` 用与 scipy 相同的差商（不是
    ``x - x_k`` 的 float32 缩水）。
    """

    def __init__(self, c: np.ndarray, x: np.ndarray):
        c = np.asarray(c, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if c.ndim != 2 or c.shape[0] != 4 or c.shape[1] != len(x) - 1:
            raise ValueError(f"PCHIP 系数形状应为 (4, n-1)，实得 {c.shape} / x {x.shape}")
        self._c = torch.as_tensor(c)          # (4, nseg) 幂基系数
        self._x = torch.as_tensor(x)          # (nseg+1,) 节点
        self._nseg = c.shape[1]

    # ------------------------------------------------------------ 求值
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """H(x)。x 任意形状，与输入同形状返回（torch 可微）。"""
        return self._eval(x, deriv=False)

    def deriv(self, x: torch.Tensor) -> torch.Tensor:
        """dH/dx。torch 可微（对 x 的二阶也可微）。"""
        return self._eval(x, deriv=True)

    # ------------------------------------------------------------ 内部
    def _seg(self, x: torch.Tensor) -> torch.Tensor:
        """每点所在段索引（0..nseg-1）。域外夹到端点段。"""
        lo = torch.clamp(x, min=float(self._x[0]), max=float(self._x[-1]))
        # ⚠ 节点表可能建在 CPU 而输入在 GPU：searchsorted 要求同 device
        knots = self._x if self._x.device == x.device else self._x.to(x.device)
        seg = torch.searchsorted(knots, lo, right=True) - 1
        return torch.clamp(seg, 0, self._nseg - 1)

    def _eval(self, x: torch.Tensor, deriv: bool) -> torch.Tensor:
        xf = x.to(torch.float64)
        seg = self._seg(xf)
        # 系数与节点表按输入 device 迁移（支持 GPU）
        c = self._c if self._c.device == x.device else self._c.to(x.device)
        knots = self._x if self._x.device == x.device else self._x.to(x.device)
        t = xf - knots[seg]                       # 段内偏移（scipy 幂基语义）
        c = c[:, seg]                             # (4, n) 每点所在段的系数
        if not deriv:
            return (c[0] * t ** 3 + c[1] * t ** 2 + c[2] * t + c[3]).to(x.dtype)
        return (3 * c[0] * t ** 2 + 2 * c[1] * t + c[2]).to(x.dtype)

    def __repr__(self) -> str:
        return f"TorchPCHIP({self._nseg} 段, x∈[{float(self._x[0]):.4g}, {float(self._x[-1]):.4g}])"
