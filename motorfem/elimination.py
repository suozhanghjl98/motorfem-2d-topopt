"""主从 / 动网格插值约束的**消元算子**：把不对称的驱动行装置替换成对称收缩。

背景（2026-09-24 拆解实测，设计文档 D11）
----------------------------------------
生产矩阵（J / K_frozen / K_mech 前向）的 O(1) 绝对不对称 **100% 来自驱动行的
单向写入**（`J_mat_driven` / `k_boundary`：主从约束行 `A_s=±A_m`、动网格滑移层
插值行 `A_n=Σc_j·A_j` —— 只写行、不写列）。D11 的 `col_mask` 只对齐次 Dirichlet
做了行列对称；主从/插值是**非齐次**约束，列不能零化。**正确的去不对称 = 消元**
（把约束代入能量：`A = T·x_free` ⇒ `K_sym = TᵀKT` 对称），这是 D11 登记的
「主从消元」structural 项 —— 也是 Pardiso `mtype=2`（Cholesky）可用的前提。

消元的数学
----------
把自由度分成 free（非 slave/非插值节点）与 constrained（slave / 滑移层插值
节点）。消元矩阵 ``T``（N × N_free）：free 行单位元；constrained 行 = 约束展开
系数（``A_s = Σ_j T[s,j]·A_j``）。于是::

    K_sym = Tᵀ·K·T      R_sym = Tᵀ·R      A = T·x_free

**关键性质（实测）**：对**生产形态** ``J_full``（含驱动行）做 ``Tᵀ·J_full·T``
就自动得到对称系统 —— 约束行被 T 消成恒等 0（约束代入后约束方程 ≡ 0），
Dirichlet 的列清零/单位行由 ``col_mask``+``J_mat_driven`` 已处理好，无需额外
逻辑。浮点路径与「TᵀK_bulkT+Dirichlet」完全一致（实测差 0）。

两类约束（同构，统一构造）：
- 标量主从：``A_s = c·A_m``（EM 反周期 c=−1 / 周期 c=+1）。
- 插值行：``A_s = Σ_j w_j·A_j``（动网格滑移层，``rot.matrix`` 的对侧插值）。
- 旋转主从：``u_s = R(2×2)·u_m``（力学，两自由度成对）。

⚠ 消元**替换**生产路径（用户 2026-09-24 决策）：矩阵真对称、golden 重标定、
fast 层红线从「逐位 legacy」变「逐位消元版」。解与 legacy 差在浮点噪声
（真实网格消元 Newton vs 生产 max|ΔA| = 4.4e-15）。
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
from scipy import sparse as sp_sparse

__all__ = ["EliminationOperator", "build_em_elimination", "build_mech_elimination"]


class EliminationOperator:
    """约束消元算子：``A = T·x_free``，``K_sym = Tᵀ·K·T``（对称）。

    Parameters
    ----------
    T : (N, N_free) csr
        消元矩阵。``T[i, k]`` = 自由度 i 对自由自由度 k 的系数。
    free_idx : Mapping[int, int]
        节点 → free 索引（EM 标量）或自由度 → free 索引（力学，见工厂函数）。

    Methods
    -------
    ``eliminate_matrix(A)``   ``Tᵀ·A·T``（tocsr；约束行自动消掉 ⇒ 对称）
    ``eliminate_rhs(b)``      ``Tᵀ·b``
    ``restore(x_free)``       ``T·x_free``（解恢复成完整 N 维）
    """

    def __init__(self, T, free_idx: Mapping[int, int]):
        self._T = T.tocsr()
        self.free_idx = dict(free_idx)
        self.n_free = self._T.shape[1]

    @property
    def T(self) -> sp_sparse.csr_matrix:
        return self._T

    def eliminate_matrix(self, A):
        """``Tᵀ·A·T`` —— 生产形态矩阵（含驱动行）直接可用（约束行自动消掉）。"""
        return (self._T.T @ A @ self._T).tocsr()

    def eliminate_rhs(self, b: np.ndarray) -> np.ndarray:
        return self._T.T @ b

    def restore(self, x_free: np.ndarray) -> np.ndarray:
        return self._T @ x_free


def _build_T(
    n_dof: int,
    free: list,
    constrained_expansion: Dict[int, Dict[int, float]],
) -> sp_sparse.csr_matrix:
    """按「free 单位 + constrained 展开系数」构造 T。"""
    free_idx = {i: k for k, i in enumerate(free)}
    rows, cols, vals = [], [], []
    for k, i in enumerate(free):
        rows.append(i); cols.append(k); vals.append(1.0)
    for node, expansion in constrained_expansion.items():
        for k, v in expansion.items():
            rows.append(node); cols.append(k); vals.append(v)
    T = sp_sparse.coo_matrix((vals, (rows, cols)),
                             shape=(n_dof, len(free))).tocsr()
    return T, free_idx


def build_em_elimination(
    n_node: int,
    slave_to_master: Mapping[int, int],
    rotor_interp_rows: Mapping[int, Mapping[int, float]],
    a_eq: bool,
) -> EliminationOperator:
    """EM 侧消元（标量主从 + 滑移层插值）。

    Parameters
    ----------
    n_node : 节点数（自由度 = 节点，标量 A）。
    slave_to_master : 主从配对（``pairing.slave_to_master``）。
    rotor_interp_rows : {插值节点: {目标节点: 系数}}，滑移层插值
        （``rot.matrix[n, :]`` 的非对角部分；``A_n = Σ −v_j·A_j``）。
    a_eq : 周期（True）/ 反周期（False）⇒ slave 系数 ``c = ∓1``。

    Notes
    -----
    constrained = slave ∪ 插值节点；free = 其余。链式展开（slave 的 master 或
    插值目标本身也是 constrained）会递归展开，约束图必须无环（实测无环）。
    """
    sigma = -1.0 if a_eq else 1.0   # 行 A_s + σ·A_m = 0 ⇒ A_s = −σ·A_m
    slave = set(slave_to_master.keys())
    rotor = set(rotor_interp_rows.keys())
    constrained = slave | rotor
    free = [i for i in range(n_node) if i not in constrained]
    free_idx = {i: k for k, i in enumerate(free)}

    def _expand(node, coef, out, stack):
        if node in free_idx:
            out[free_idx[node]] = out.get(free_idx[node], 0.0) + coef
            return
        if node in stack:
            raise ValueError(f"约束图有环：{node}")
        stack.add(node)
        if node in slave:
            _expand(slave_to_master[node], coef * (-sigma), out, stack)
        elif node in rotor:
            for tgt, w in rotor_interp_rows[node].items():
                _expand(tgt, coef * (-w), out, stack)
        else:
            raise ValueError(f"未知约束节点 {node}")
        stack.discard(node)

    expansion = {}
    for node in constrained:
        out = {}
        _expand(node, 1.0, out, set())
        expansion[node] = out

    T, free_idx = _build_T(n_node, free, expansion)
    return EliminationOperator(T, free_idx)


def build_mech_elimination(
    n_node: int,
    slave_to_master: Mapping[int, int],
    rot: np.ndarray,
) -> EliminationOperator:
    """力学侧消元（旋转主从，两自由度成对）。

    Parameters
    ----------
    n_node : 力学子网格节点数（自由度 = 2·n_node）。
    slave_to_master : 力学主从配对（节点号）。
    rot : (2, 2) 旋转矩阵 ``R(+2π/parts)``（``_slave_rotation``）。

    Notes
    -----
    k_boundary 的主从行内容（``mech_solver.py``）：``u_s = R·u_m``，
    T[2s, 2m..] = R 的行；free 自由度 = 非 slave 节点的两个分量。
    """
    slave = set(slave_to_master.keys())
    free = [2 * i + c for i in range(n_node) if i not in slave for c in (0, 1)]
    free_idx = {d: k for k, d in enumerate(free)}
    expansion = {}
    for s in slave:
        m = slave_to_master[s]
        for a in (0, 1):
            row = {free_idx[2 * m + b]: rot[a, b] for b in (0, 1)}
            expansion[2 * s + a] = row
    T, free_idx = _build_T(2 * n_node, free, expansion)
    return EliminationOperator(T, free_idx)
