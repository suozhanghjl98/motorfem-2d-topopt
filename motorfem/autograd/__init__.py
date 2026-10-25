"""motorfem.autograd —— 阶段 6：张量/autograd 梯度 oracle（第二条并行路径）

设计文档 §阶段 6：手写伴随**不删**（变成 oracle）；本包提供隐式伴随的
torch 实现，验证三套转矩的手写解析梯度（验收 rel < 1e-9）。

组件：
- ``pchip.py``：scipy PCHIP 的 torch 可微重建（B-H 曲线，值/导 ~1e-16 相对）。
- ``material.py``：torch SIMP 投影 + 四类掩码混合的磁阻率链。
- ``residual.py``：残差式伴随的单元局部收缩 ``λᵀF``（对 ρ 可微）。
- ``field.py``：``EMFieldGrad`` torch Function（forward = 现有 numpy Newton；
  backward = ``λ = spsolve(Jᵀ, gA)`` + ``−∂(λᵀF)/∂ρ``）。
- ``flux_oracle.py``（阶段 8 收尾 #1）：冻结求解的**全导数** oracle ——
  ``FrozenSolveGrad`` Function（backward 穿过冻结系统）+ ``residual_lam_frozen``
  + ``T=Σk_e·ν_e`` 的端到端图。判定 D4 的冻结约定（§阶段 6 偏差登记①的下一片）。
- ``oracle.py``：转矩梯度的 autograd 入口 + 与手写比较（rel < 1e-9）。

**纪律**：本包不修改任何生产路径（``motorfem/fem/``、``kernels/fast/``）；
torch 缺失时相关测试 ``skip``。梯度不与手写位级一致（浮点顺序不同），
验收是相对误差 < 1e-9（设计文档 §阶段 6 明确）。
"""

from .field import EMFieldGrad, OracleContext, oracle_from_parts
from .flux_oracle import (
    FluxFrozenContext,
    FrozenSolveGrad,
    flux_torque_full_derivative,
    flux_torque_gradient,
)
from .oracle import (
    adjoint_gradient_from_state,
    adjoint_gradients_from_state_batched,
    autograd_torque_gradient,
    compare_with_hand_written,
)
from .pchip import TorchPCHIP
from .residual import ResidualContext, residual_lam, residual_lam_frozen

__all__ = [
    "TorchPCHIP",
    "ResidualContext",
    "residual_lam",
    "residual_lam_frozen",
    "OracleContext",
    "EMFieldGrad",
    "oracle_from_parts",
    "autograd_torque_gradient",
    "adjoint_gradient_from_state",
    "adjoint_gradients_from_state_batched",
    "compare_with_hand_written",
    "FluxFrozenContext",
    "FrozenSolveGrad",
    "flux_torque_gradient",
    "flux_torque_full_derivative",
]
