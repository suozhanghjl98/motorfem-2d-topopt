"""motorfem.autograd.residual —— 阶段 6：残差式伴随的单元局部 torch 收缩

oracle 的 backward 需要 ``λᵀF``（F = 残差，λ = 伴随解），且要对 ρ 可微。
本模块把 F 的收缩写成**单元局部、自然写法**的 torch 表达式:

    F_lam = Σ_e (λ·s·(1−d))_vtriᵀ · (S·K0·S·A_vtri)_e − λᵀ(1−d)R + λᵀ d ⊙ (F_mat_driven·A)

* ``K_e = ν_e · K0_e``（ν 经 ``torch_nu``，ρ 可微；B 固定）
* **反周期主从**（实测确认）：22 元组 ``K0_data`` 是 ``TᵀK0T`` 的展开（内核 6/8 的
  位置矩阵），对从属单元等价于 ``gather 到 master 节点 + 从属槽位反号``
  （``A_slave = −A_master``）⇒ ``S·K0·S``，``S = diag(±1)``。
* 驱动行（``d`` 按节点）替换为线性 ``F_mat_driven·A``（ρ 无关，常数项）；R 是
  全局向量，**只减一次**（不逐元素减）。
* A、λ、R 都是**常量**（收敛点/伴随解），只有 ρ 走 autograd

数值关系：``F_lam`` 在收敛点应 ≈ 0（Newton 收敛 ⇒ F ≈ 0）—— 这是对 torch 重建
正确性的**独立 sanity check**（测试里断言）。梯度由 ``torch.autograd.grad(−F_lam, ρ)``
给出，等价于手写的 ``−λᵀ·∂K/∂ρ·A``（位置矩阵实现）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..kernels.fast.material import B_CLIP_MIN, DEFAULT_MUR_LINEAR
from ..kernels.reference.newton import MaterialMasks
from .material import TorchSIMP, torch_nu
from .pchip import TorchPCHIP

__all__ = ["ResidualContext", "residual_lam", "residual_lam_frozen"]


@dataclass
class ResidualContext:
    """残差收缩所需的常量数据（收敛点 A、伴随 λ、几何、材料），一次打包。

    全部从 ``FieldTopology`` / 22 元组 / ``FieldMaterial`` 提取一次；torch 张量
    在 ``to_torch(device)`` 时按需构造（保证 device 一致性）。
    """

    # 几何（单元局部，元素序 = element_S 序）
    tri: np.ndarray          # (N, 3) 每单元的 3 个全局节点号（插层后，**原始**）
    tri_v: np.ndarray        # (N, 3) 从属虚拟节点表（slave→master 替换；非从属 = tri）
    sign: np.ndarray         # (N, 3) 从属槽位符号（反周期 A_slave=−A_master ⇒ −1，否则 +1）
    Bxcoef: np.ndarray       # (N, 3)
    Bycoef: np.ndarray       # (N, 3)
    K0: np.ndarray           # (N, 6) 对称刚度 6 元组（真实刚度 = ν_e · K0_e）
    masks: MaterialMasks     # 四类 0/1 数组（算术加权）
    # 源与驱动
    R: np.ndarray            # (N_node,) 电流源项
    f_driven: np.ndarray     # (N_node,) 0/1 驱动行掩码
    F_mat_driven_A: np.ndarray  # (N_node,) F_mat_driven @ A（A 固定后的常量）
    # 材料
    curve: TorchPCHIP
    simp: TorchSIMP
    mu0: float
    mur_linear: float = DEFAULT_MUR_LINEAR

    # ------------------------------------------------------------ 打包
    @classmethod
    def build(
        cls,
        top,
        kv: tuple,
        R: np.ndarray,
        material,
        A: np.ndarray,
        curve: TorchPCHIP,
        simp: TorchSIMP,
    ) -> "ResidualContext":
        """从 FieldTopology + 22 元组 + FieldMaterial + 收敛 A 构造。

        Parameters
        ----------
        top : FieldTopology（``top.em`` 的单元矩阵、``top.layer.tri`` 的节点表）
        kv : 22 元组（``pre.K_vectorized_vec[i_t]``；掩码与驱动在其中）
        R : (N_node,) 该时间点的电流源项
        material : FieldMaterial（``mu0``/``mur_linear``）
        A : (N_node,) 收敛的矢量磁位（用于 ``F_mat_driven @ A``）
        """
        (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
         Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
         *_) = kv
        em = top.em
        tri = np.asarray(top.layer.tri, dtype=np.int64)
        # 从属虚拟节点表 + 反周期符号（A_slave = −A_master ⇒ 从属槽位 −1）。
        # ⚠ 必须用 **layer.pairing**（滑移层插入后的主从映射）：虚拟单元表
        # （``slave_virtual_elements(layer.tri, layer.pairing)``）用的是它，插入前
        # 的 ``pairing0`` 节点号已失效。
        # 22 元组的 K0_data 是 TᵀK0T 的展开（内核 6/8 的位置矩阵），对从属单元
        # 等价于「gather 到 master 节点 + 槽位反号」—— 见设计文档 §阶段 6 的偏差记录。
        s2m = top.layer.pairing.slave_to_master
        vtri = np.array([
            [s2m.get(n, n) for n in row] for row in tri
        ], dtype=np.int64)
        sign = np.where(
            np.array([[n in s2m for n in row] for row in tri]), -1.0, 1.0,
        )
        return cls(
            tri=tri,
            tri_v=vtri,
            sign=sign,
            Bxcoef=np.asarray(em.Bxcoef),
            Bycoef=np.asarray(em.Bycoef),
            K0=np.asarray(em.K0),
            masks=MaterialMasks(
                is_vacuum=np.asarray(is_vaccum, dtype=float),
                is_simp=np.asarray(is_SIMP, dtype=float),
                is_non_simp=np.asarray(is_non_SIMP, dtype=float),
                is_linear_iron=np.asarray(is_linear_iron, dtype=float),
            ),
            R=np.asarray(R),
            f_driven=np.asarray(F_is_driven, dtype=float),
            F_mat_driven_A=F_mat_driven @ A,
            curve=curve,
            simp=simp,
            mu0=float(material.mu0),
            mur_linear=float(material.mur_linear),
        )

    # ------------------------------------------------------------ torch 视图
    def to_torch(self, device: Optional[torch.device] = None) -> "_TorchViews":
        """一次性构造全部 torch 张量（在同一 device 上）。"""
        return _TorchViews(
            tri=self.tri,
            tri_v=self.tri_v,
            sign=torch.as_tensor(self.sign, device=device),
            Bxcoef=torch.as_tensor(self.Bxcoef, device=device),
            Bycoef=torch.as_tensor(self.Bycoef, device=device),
            K0=torch.as_tensor(self.K0, device=device),
            masks=MaterialMasks(
                is_vacuum=torch.as_tensor(self.masks.is_vacuum, device=device),
                is_simp=torch.as_tensor(self.masks.is_simp, device=device),
                is_non_simp=torch.as_tensor(self.masks.is_non_simp, device=device),
                is_linear_iron=torch.as_tensor(self.masks.is_linear_iron, device=device),
            ),
            R=torch.as_tensor(self.R, device=device),
            f_driven=torch.as_tensor(self.f_driven, device=device),
            F_mat_driven_A=torch.as_tensor(self.F_mat_driven_A, device=device),
            mu0=self.mu0,
            mur_linear=self.mur_linear,
            curve=self.curve,
            simp=self.simp,
        )


@dataclass
class _TorchViews:
    tri: np.ndarray
    tri_v: np.ndarray
    sign: torch.Tensor
    Bxcoef: torch.Tensor
    Bycoef: torch.Tensor
    K0: torch.Tensor
    masks: MaterialMasks
    R: torch.Tensor
    f_driven: torch.Tensor
    F_mat_driven_A: torch.Tensor
    mu0: float
    mur_linear: float
    curve: TorchPCHIP
    simp: TorchSIMP


def _expand_k0(K0: torch.Tensor) -> torch.Tensor:
    """(N, 6) 对称刚度 → (N, 3, 3) 完整矩阵 [ii,ij,im; ij,jj,jm; im,jm,mm]。"""
    return torch.stack([
        torch.stack([K0[:, 0], K0[:, 1], K0[:, 2]], dim=-1),
        torch.stack([K0[:, 1], K0[:, 3], K0[:, 4]], dim=-1),
        torch.stack([K0[:, 2], K0[:, 4], K0[:, 5]], dim=-1),
    ], dim=-2)


def residual_lam(
    A: torch.Tensor,
    rho: torch.Tensor,
    lam: torch.Tensor,
    v: _TorchViews,
) -> torch.Tensor:
    """``λᵀF(A, ρ)`` 的 torch 收缩（对 ρ 可微，其余是常量）。

    Parameters
    ----------
    A : (N_node,) 收敛矢量磁位（**常量**，detach）
    rho : (N_elem,) 密度（**requires_grad**）
    lam : (N_node,) 伴随解（常量）
    v : ``ResidualContext.to_torch`` 的产物

    Returns
    -------
    torch 标量 ``λᵀF`` —— ``torch.autograd.grad(−F_lam, rho)`` 即
    残差式伴随梯度 ``dL/dρ = −λᵀ ∂F/∂ρ``。
    """
    # 局部矢量统一用**原始 tri**（slave 节点值）：收敛点上 A_slave = −A_master，
    # 即 S·A_vtri = A_tri。numpy 的 B 用 em.Bxcoef·A_tri，K 用 S·K0·S·A_vtri =
    # S·K0·A_tri —— 两者都以 A_tri 为局部矢量。
    Ae = A[v.tri]                              # (N, 3)
    Bx = (v.Bxcoef * Ae).sum(-1)
    By = (v.Bycoef * Ae).sum(-1)
    B = torch.sqrt(Bx ** 2 + By ** 2).clamp(min=B_CLIP_MIN)

    nu = torch_nu(B, rho, v.masks, v.curve, v.simp, v.mu0, v.mur_linear)
    # 单元矩阵 K_e = ν_e·K0_e；反周期主从带行符号（22 元组 K0_data = TᵀK0T 展开的
    # 等价形式）：S·(ν·K0)·S·A_vtri = sign[:, :, None]·(ν·K0_33 @ A_tri)
    # ⚠ 不要用 sign[:, None, None] —— 2D 张量的该索引是 (N,1,1,3) 不是 (N,1,1)。
    K0_33 = _expand_k0(v.K0)                   # (N, 3, 3)
    K0_Ae = (nu[:, None, None] * K0_33) @ Ae[..., None]   # (N, 3, 1) = ν·K0·A_tri
    KeAe = v.sign[:, :, None] * K0_Ae          # (N, 3, 1) = S·ν·K0·S·A_vtri

    # 非驱动行的单元收缩（散射到 vtri 行；行符号已在 KeAe 里）；R 是全局一次减法。
    # λ 可带批量维（N_node, b）：对多个伴随向量一次收缩 —— 固定开销摊薄的关键
    #（3 转矩 × 11 时间点 = 33 个梯度向量可一批算）。返回 (b,) 或标量。
    batched = lam.ndim == 2
    mask = 1.0 - v.f_driven[v.tri_v]               # (N, 3)
    lam_e = lam[v.tri_v] * (mask[..., None] if batched else mask)
    KeAe_b = KeAe[..., 0] if not batched else KeAe[..., 0][..., None]
    F_lam = (lam_e * KeAe_b).sum(dim=(0, 1)) if batched else (lam_e * KeAe_b).sum()
    r_term = (1.0 - v.f_driven) * v.R               # (N,) 或 (N,1)
    driven_term = v.f_driven * v.F_mat_driven_A
    if batched:
        F_lam = F_lam - (lam * r_term[:, None]).sum(0) + (lam * driven_term[:, None]).sum(0)
    else:
        F_lam = F_lam - (lam * r_term).sum() + (lam * driven_term).sum()
    return F_lam


def residual_lam_frozen(
    A: torch.Tensor,
    rho: torch.Tensor,
    lam: torch.Tensor,
    v: _TorchViews,
    B_frozen: torch.Tensor,
) -> torch.Tensor:
    """冻结系统（``K_frozen(ρ)·A = R``）的 ``λᵀF`` 收缩 —— 阶段 8 收尾 #1。

    与 ``residual_lam`` **唯一差别**：磁阻率链的 ``B`` 来自形参 ``B_frozen``
    （冻结的磁密幅值，**已裁剪**），而不是从 ``A`` 现算 ``|Bx,By|``。

    冻结求解是**线性**的（``A=0`` 处饱和项为零 ⇒ ``dν/dB`` 项为零 ⇒
    ``K_frozen`` 与 ``A`` 无关，一步即精确解，内核 15 的测试钉住），所以
    ``F_frozen(A; ρ) = K_frozen(ρ)·A − R`` 里 ``K_frozen`` 只依赖 ``ρ``
    （经冻结 ``B_frozen`` 与 SIMP 密度）。本收缩对 ``ρ`` 可微 —— 这正是
    ``FrozenSolveGrad.backward`` 需要的项（残差式隐式伴随穿过冻结系统）。

    ``B_frozen`` : (N_elem,) 冻结磁密幅值（numpy 或 torch；torch 时须与
    ``v``/``A`` 同 device）。
    """
    Bf = torch.as_tensor(B_frozen, device=A.device, dtype=torch.float64)
    nu = torch_nu(Bf, rho, v.masks, v.curve, v.simp, v.mu0, v.mur_linear)
    # 以下与 residual_lam 的 K 收缩段完全相同（S·K0·S 反周期、驱动行、R 一次减）
    Ae = A[v.tri]                              # (N, 3)
    K0_33 = _expand_k0(v.K0)                   # (N, 3, 3)
    K0_Ae = (nu[:, None, None] * K0_33) @ Ae[..., None]   # (N, 3, 1) = ν·K0·A_tri
    KeAe = v.sign[:, :, None] * K0_Ae          # (N, 3, 1) = S·ν·K0·S·A_vtri

    batched = lam.ndim == 2
    mask = 1.0 - v.f_driven[v.tri_v]               # (N, 3)
    lam_e = lam[v.tri_v] * (mask[..., None] if batched else mask)
    KeAe_b = KeAe[..., 0] if not batched else KeAe[..., 0][..., None]
    F_lam = (lam_e * KeAe_b).sum(dim=(0, 1)) if batched else (lam_e * KeAe_b).sum()
    r_term = (1.0 - v.f_driven) * v.R
    driven_term = v.f_driven * v.F_mat_driven_A
    if batched:
        F_lam = F_lam - (lam * r_term[:, None]).sum(0) + (lam * driven_term[:, None]).sum(0)
    else:
        F_lam = F_lam - (lam * r_term).sum() + (lam * driven_term).sum()
    return F_lam
