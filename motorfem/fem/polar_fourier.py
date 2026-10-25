# -*- coding: utf-8 -*-
"""极坐标二维傅里叶低阶参数化（转子设计域）。

------------------------------------------------------------------------------
为什么需要它
------------------------------------------------------------------------------

现有设计变量是转子主动设计域**每个单元一个密度**（6s4p：1207 个）。它有三个
结构性问题：

1. **维度随网格增长** —— 换细网格 ⇒ 变量数随之膨胀，最优解不可迁移；
2. **允许任意微小结构** —— 只靠密度滤波（半径 2 mm）约束特征尺寸，优化器仍会
   产生锯齿/小岛，`docs/自动加严调度与帕累托扫描_2026-10-18.md` §8 观察到
   "实心单元数差 6.7 倍而性能几乎相同"；
3. **设计空间过大** —— 多起点实验（`docs/随机初始化实验汇总_2026-10-16.md`）
   显示 30 组殊途同归，说明大空间里的有效自由度远小于单元数。

本模块把设计变量换成**极坐标下的低阶张量傅里叶系数**：变量数从 O(单元数) 降到
O(K·N_r)，且基函数由**物理坐标**定义 ⇒ 换网格不改变参数维度（网格无关）。

------------------------------------------------------------------------------
数学定义
------------------------------------------------------------------------------

设计域是一个环形扇区 ``r ∈ [r0, r1]``、``θ ∈ [θ0, θ1]``。归一化

    u = (r − r0) / (r1 − r0) ∈ [0, 1]
    v = (θ − θ0) / (θ1 − θ0) ∈ [0, 1]

基函数取张量积

    φ_{j,k}(r,θ) = R_j(u) · cos(kπ v)        j = 0..N_r,  k = 0..K
    ψ_{j,k}(r,θ) = R_j(u) · sin(kπ v)        j = 0..N_r,  k = 1..K

``cos(kπv)`` / ``sin(kπv)`` 是 **θ 方向的傅里叶级数**（扇区上的半区间三角基；
若设计域是完整环，调用方把 ``θ`` 改用 0..2π 归一即可，基形式不变）。
径向 ``R_j`` 由 ``radial`` 选择：

- ``"cos"``（默认）：``R_j(u) = cos(jπu)`` —— [0,1] 上的偶延拓傅里叶级数；
- ``"legendre"`` / ``"cheb"``：Legendre / Chebyshev 多项式。

> 径向为什么不用傅里叶？``r`` 不是周期方向。``cos(jπu)`` 等价于把 ``[0,1]``
> 偶延拓到 ``[−1,1]`` 后做标准傅里叶余弦展开，与 θ 方向同构；实验（2026-10-18）
> 显示三种径向基的拟合优度接近（cos 略好），因此默认 cos。

密度映射（``mapping``）：

- ``"linear"``（默认）：``ρ = clip(Σ c·φ, 0, 1)``；
- ``"sigmoid"``：``ρ = σ(gain · Σ c·φ)``，恒在 (0,1)，适合需要严格内点的场景。

------------------------------------------------------------------------------
拟合能力（6s4p 实测，2026-10-18，`tools/scripts/probe_polar_param.py`）
------------------------------------------------------------------------------

对今晚前沿最优设计 `mu1_b1_s25__S1_r1` 的**物理密度 ρ**（1207 个主动单元）：

=========  =======  =========  =======  =======
K           N_r      变量数      R²       IoU
=========  =======  =========  =======  =======
2           2        15         0.651    0.765
4           6        63         0.921    0.945
6           6        91         0.939    0.950
6           8        117        0.959    0.959
8           8        153        0.967    0.960
=========  =======  =========  =======  =======

（对**设计变量 x** 做同样拟合只有 R²≈0.86/363 变量 —— 因为 x 已收敛到 0/1 双峰；
⇒ 参数化必须放在 **ρ 空间**，跳过 ``filter_inverse_rho``。）

------------------------------------------------------------------------------
与现有流程的接口
------------------------------------------------------------------------------

- ``rho_full(c, top)``：系数 → 全场密度（主动单元用基展开，从属单元复制）。
  它**取代** ``density_from_x``（不再调用 ``filter_SIMP @ x``）。
- ``project(df, top)``：``∂f/∂ρ`` → ``∂f/∂c``。先把从属单元的导数叠加到主动
  单元（与 ``gradient_to_x`` 同一步），再左乘 ``Bᵀ``（含映射导数）。
- ``initial_coeff(rho_seed)``：把已有密度投影成系数（最小二乘），用于
  **从现有最优设计热启动**。
- ``bounds()``：MMA 变量边界。

⚠ 参数化后 ``x`` 的语义变成**系数向量**，不再是 ``[0,1]`` 密度。任何假定
``x ∈ [0,1]`` 的下游（报告脚本、GUI 快照）都要用 ``rho_full`` 取密度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["PolarFourierSpec", "PolarFourierParam"]


@dataclass(frozen=True)
class PolarFourierSpec:
    """参数化的阶数与映射选择。"""

    K: int = 6                 # θ 方向最高谐波（cos 与 sin 各 K 项）
    N_r: int = 6               # 径向多项式阶数（R_0..R_N_r，共 N_r+1 项）
    radial: str = "cos"        # cos | legendre | cheb
    radial_map_alpha: float = 1.0
    #: 径向归一化映射 ``u ← u^alpha``。``alpha>1`` 把基函数分辨率推向
    #: **转子表面**（u=1）——电机里靠近气隙的径向结构对转矩影响最大。
    theta_periodic: bool = False
    #: ``False``（默认）：扇区半区间基 ``cos/sin(kπv)``（端点 Neumann）。
    #: ``True``：整周期基 ``cos/sin(2πk v)``（以所在扇区的全宽为周期），
    #: 适合"设计域 = 一个极距"或要求 θ 两端连续的情形。
    mapping: str = "linear"    # linear | sigmoid
    gain: float = 1.0          # sigmoid 的斜率（mapping="sigmoid" 时用）
    coeff_bound: float = 4.0   # MMA 变量边界 [−b, b]


# ---------------------------------------------------------------- 基
def radial_basis(u: np.ndarray, n: int, kind: str) -> np.ndarray:
    """``u ∈ [0,1]`` 上的 n 阶径向基，返回 ``(N, n+1)``。"""
    if kind == "cos":
        return np.stack([np.cos(j * np.pi * u) for j in range(n + 1)], axis=1)
    if kind == "legendre":
        return np.polynomial.legendre.legvander(2.0 * u - 1.0, n)
    if kind == "cheb":
        return np.polynomial.chebyshev.chebvander(2.0 * u - 1.0, n)
    raise ValueError(f"未知径向基 {kind!r}（可选 cos/legendre/cheb）")


def basis_matrix(r: np.ndarray, theta: np.ndarray, spec: PolarFourierSpec,
                 r_range: Tuple[float, float],
                 th_range: Tuple[float, float]) -> np.ndarray:
    """张量傅里叶基矩阵，形状 ``(N, (N_r+1)(2K+1))``。

    列顺序：``cos(k=0..K)`` 块在前、``sin(k=1..K)`` 块在后；每块内径向阶由低到高。
    ``r_range`` / ``th_range`` 必须由**几何**给出（不是样本 min/max）——
    这是网格无关性的前提：同一组系数在不同网格上定义同一个连续密度场。
    """
    r0, r1 = r_range
    t0, t1 = th_range
    if not (r1 > r0 and t1 > t0):
        raise ValueError(f"非法窗口 r_range={r_range}, th_range={th_range}")
    u = (r - r0) / (r1 - r0)
    if spec.radial_map_alpha != 1.0:
        u = np.power(np.clip(u, 0.0, 1.0), spec.radial_map_alpha)
    v = (theta - t0) / (t1 - t0)
    rad = radial_basis(u, spec.N_r, spec.radial)
    freq = 2.0 * np.pi if spec.theta_periodic else np.pi
    ang = ([np.ones_like(v)]
           + [np.cos(k * freq * v) for k in range(1, spec.K + 1)]
           + [np.sin(k * freq * v) for k in range(1, spec.K + 1)])
    return np.concatenate([rad * a[:, None] for a in ang], axis=1)


# ---------------------------------------------------------------- 参数化
class PolarFourierParam:
    """极坐标低阶参数化：系数向量 ``c`` ↔ 设计域密度 ``ρ``。"""

    def __init__(self, r: np.ndarray, theta: np.ndarray,
                 spec: PolarFourierSpec,
                 r_range: Tuple[float, float],
                 th_range: Tuple[float, float]):
        self.spec = spec
        self.r = np.asarray(r, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.r_range = (float(r_range[0]), float(r_range[1]))
        self.th_range = (float(th_range[0]), float(th_range[1]))
        self.B = basis_matrix(self.r, self.theta, spec,
                              self.r_range, self.th_range)
        # ★ 固定参考网格上的正交化（2026-10-18 实测必需）。
        #   问题：θ/r 张量基在**非均匀单元质心**上 cond(B) 可达 1e4~1e5，
        #   直接最小二乘会得到范数极大的系数（实测 sigmoid 初值 |c|max=491）
        #   ⇒ MMA 边界卡死、梯度尺度失真。
        #   ⚠ 不能对**当前网格**的 B 做 QR：Q 依赖采样点 ⇒ 同一系数在不同网格上
        #   给出不同的场，网格无关性就没了。
        #   做法：在窗口内的**固定规则网格**上做 QR（与真实网格无关），
        #   得到可逆变换 R；定义 ``Bphys = B · R⁻¹ · diag(sign)``：
        #   - 在参考网格上列两两正交且 **L2 范数 = 1**；
        #   - 列仍是**固定连续函数**的线性组合 ⇒ 换网格只换采样点，同一 c 同场。
        #   物理意义：变量是"正交模态系数"（``c ≈ Q_refᵀ ρ``），与原始傅里叶
        #   系数差一个可逆线性变换。
        #
        # ⚠ 归一化必须是 **L2=1，不能乘 √N_ref**（2026-10-18 实测）：乘 √N_ref
        #   会让系数变小（0.27）很"好看"，但同时把 ``∂f/∂c = Bᵀ·∂f/∂ρ`` 放大
        #   √N_ref≈46 倍 ⇒ MMA 的对偶子问题尺度失衡，移动极限形同虚设：第一步
        #   就把 T_flux 从 54 打到 22，之后再也回不来。L2=1 时梯度与密度设计
        #   变量的量级一致（O(1)），系数尺度 = O(√N_ref)，由 ``bounds(c0)``
        #   自适应覆盖。
        r_ref, th_ref = self._reference_grid(spec, self.r_range, self.th_range)
        B_ref = basis_matrix(r_ref, th_ref, spec, self.r_range, self.th_range)
        _q, R_ref = np.linalg.qr(B_ref, mode="reduced")
        # ⚠ QR 的符号是 LAPACK 任意的（实测常数模态被翻成 −1，c[0]=0.5 ⇒ g=−0.5
        #   ⇒ clip 成全 0）。用 diag(R) 的符号统一，保证常数模态恒为 +1。
        sign = np.sign(np.diag(R_ref))
        sign[sign == 0.0] = 1.0
        self._n_ref = int(r_ref.size)
        self.T = np.linalg.inv(R_ref) @ np.diag(sign)
        self.Bphys = self.B @ self.T
        # 采样点是否落在窗口内（落在外的单元会被基外推，仅作诊断用）
        u = (self.r - self.r_range[0]) / (self.r_range[1] - self.r_range[0])
        v = (self.theta - self.th_range[0]) / (self.th_range[1] - self.th_range[0])
        self.n_outside_window = int(np.sum((u < -0.02) | (u > 1.02)
                                           | (v < -0.02) | (v > 1.02)))

    # ------------------------------------------------------------ 构造
    @classmethod
    def from_topology(cls, top, spec: Optional[PolarFourierSpec] = None,
                      *, r_range: Optional[Tuple[float, float]] = None,
                      th_range: Optional[Tuple[float, float]] = None
                      ) -> "PolarFourierParam":
        """从 ``FieldTopology`` 构造：取主动设计域单元质心 + 几何窗口。

        ``r_range`` 默认 ``[shaft_radius_m, rotor_outer_radius_m]``；
        ``th_range`` 默认 ``[0, π/(2·pole_pairs)]``（半极设计域）。
        两者都可显式覆盖（例如完整环模型传 ``(0, 2π)``）。
        """
        node = np.asarray(top.node_post, dtype=float)
        es = np.asarray(top.element_S_post, dtype=np.int64)
        topo = np.asarray(list(top.element_topo_opti_m), dtype=np.int64)
        tri = es[topo, 1:4]
        xy = node[tri, 1:3].mean(axis=1)
        r = np.hypot(xy[:, 0], xy[:, 1])
        th = np.arctan2(xy[:, 1], xy[:, 0])
        th = np.where(th < 0.0, th + 2.0 * np.pi, th)

        geo = top.cfg.geometry
        if r_range is None:
            r_range = (float(geo.shaft_radius_m), float(geo.rotor_outer_radius_m))
        if th_range is None:
            half = np.pi / (2.0 * int(geo.pole_pairs))
            th_range = (0.0, float(half))

        param = cls(r, th, spec or PolarFourierSpec(), r_range, th_range)
        if param.n_outside_window:
            print(f"[polar_fourier] ⚠ {param.n_outside_window} 个单元质心落在几何窗口外"
                  f"（r_range={r_range}, th_range={th_range}），将发生基外推")
        return param

    # ------------------------------------------------------------ 属性
    @staticmethod
    def _reference_grid(spec: PolarFourierSpec, r_range, th_range):
        """窗口内的固定规则参考网格（只依赖阶数，不依赖真实网格）。"""
        nr = max(33, 4 * (spec.N_r + 1))
        nt = max(65, 8 * (2 * spec.K + 1))
        r = np.linspace(r_range[0], r_range[1], nr)
        th = np.linspace(th_range[0], th_range[1], nt)
        R, T = np.meshgrid(r, th, indexing="ij")
        return R.ravel(), T.ravel()

    @property
    def n_coef(self) -> int:
        return int(self.Bphys.shape[1])

    @property
    def topo(self) -> np.ndarray:
        """主动单元编号（与 ``B`` 的行一一对应）。"""
        return self._topo if hasattr(self, "_topo") else np.arange(self.B.shape[0])

    def attach_topology(self, top) -> None:
        """绑定 ``FieldTopology``（从属单元复制与梯度叠加需要它）。"""
        self._top = top
        self._topo = np.asarray(list(top.element_topo_opti_m), dtype=np.int64)

    # ------------------------------------------------------------ 正问题
    def _map(self, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """平滑映射 ``g → ρ`` 与其导数 ``dρ/dg``。"""
        if self.spec.mapping == "linear":
            rho = np.clip(g, 0.0, 1.0)
            drdg = ((g > 0.0) & (g < 1.0)).astype(float)
            return rho, drdg
        if self.spec.mapping == "sigmoid":
            z = self.spec.gain * g
            rho = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
            return rho, self.spec.gain * rho * (1.0 - rho)
        raise ValueError(f"未知映射 {self.spec.mapping!r}（linear | sigmoid）")

    def expand(self, c: np.ndarray) -> np.ndarray:
        """系数 ``c`` → 设计域密度 ``ρ_topo``（长度 = 单元数）。"""
        g = self.Bphys @ np.asarray(c, dtype=float)
        return self._map(g)[0]

    def rho_full(self, c: np.ndarray, top) -> np.ndarray:
        """系数 → **全场**密度（主动单元展开、从属单元复制）。"""
        rho = np.zeros(top.n_element)
        rho[list(top.element_topo_opti_m)] = self.expand(c)
        for i_elem_s, i_elem_m in top.element_topo_opti_s.items():
            rho[i_elem_s] = rho[i_elem_m]
        return rho

    # ------------------------------------------------------------ 梯度
    def project(self, df: np.ndarray, c: np.ndarray) -> np.ndarray:
        """``∂f/∂ρ``（形状 ``(n_con, n_element)``）→ ``∂f/∂c``（``(n_con, n_coef)``）。

        ``c`` 必须是**本次前向使用的同一系数**（映射导数 ``dρ/dg`` 依赖它）。

        两步：
        1. 从属单元导数叠加到主动单元（``df[:, m] += df[:, s]``，与
           ``gradient_to_x`` 的从属叠加同一步——从属单元的值是主动单元的复制，
           所以 ``d/dρ_m`` 必须累加 ``d/dρ_s``）；
        2. ``∂f/∂c = (∂f/∂ρ_topo · dρ/dg) @ B``。
        """
        if not hasattr(self, "_top"):
            raise RuntimeError("请先调用 attach_topology(top) 再求梯度")
        top = self._top
        df0 = np.array(df, dtype=float, copy=True)
        for i_elem_s, i_elem_m in top.element_topo_opti_s.items():
            df0[:, i_elem_m] += df0[:, i_elem_s]

        topo = list(top.element_topo_opti_m)
        g = self.Bphys @ np.asarray(c, dtype=float)
        drdg = self._map(g)[1]
        # (n_con, n_topo) * (n_topo,) @ (n_topo, n_coef)
        return (df0[:, topo] * drdg[None, :]) @ self.Bphys

    # ------------------------------------------------------------ 初值
    def initial_coeff(self, rho_seed: np.ndarray) -> np.ndarray:
        """把已有密度投影成系数（最小二乘）。

        ``linear``：直接拟合 ρ；
        ``sigmoid``：拟合 ``logit(ρ) / gain``（否则 σ 压缩会让初值系统性偏低）。
        """
        rho = np.asarray(rho_seed, dtype=float)
        if self.spec.mapping == "linear":
            c, *_ = np.linalg.lstsq(self.Bphys, rho, rcond=None)
            return c
        # sigmoid：对 logit 做最小二乘。eps=0.02 限制 logit 幅度（±3.9），
        # 否则 ρ 精确 0/1 的单元会把初值推到 ±∞（实测 |c|max=491）。
        eps = 0.02
        target = np.log(np.clip(rho, eps, 1.0 - eps)
                        / np.clip(1.0 - rho, eps, 1.0)) / self.spec.gain
        c, *_ = np.linalg.lstsq(self.Bphys, target, rcond=None)
        return c

    def bounds(self, c0: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """MMA 变量边界 ``[−b, b]``。

        系数是正交模态系数（``Bphys`` 列 L2=1）⇒ 常数模态约 ``0.5·√N_ref``
        （6s4p 实测 |c0|max≈23），固定的 ``coeff_bound=4`` 会把起点卡在边界上。
        给定 ``c0`` 时自动放大到 ``1.5·max|c0|``（至少 ``coeff_bound``）。
        """
        b = self.spec.coeff_bound
        if c0 is not None and np.size(c0):
            b = max(b, 1.5 * float(np.max(np.abs(c0))))
        lo = -np.full(self.n_coef, b)
        hi = np.full(self.n_coef, b)
        return lo, hi

    # ------------------------------------------------------------ 诊断
    def fit(self, rho: np.ndarray) -> dict:
        """对给定密度做投影并报告拟合指标（不参与优化，仅诊断）。"""
        c = self.initial_coeff(rho)
        rho_hat = self.expand(c)
        resid = rho - rho_hat
        denom = max(float(np.sum((rho - rho.mean()) ** 2)), 1e-30)
        a = rho > 0.5
        b = rho_hat > 0.5
        return {
            "r2": 1.0 - float(np.sum(resid ** 2)) / denom,
            "iou": float(np.sum(a & b) / max(np.sum(a | b), 1)),
            "n_coef": self.n_coef,
            "rho_min": float(rho_hat.min()),
            "rho_max": float(rho_hat.max()),
            "coef_abs_max": float(np.abs(c).max()),
        }
