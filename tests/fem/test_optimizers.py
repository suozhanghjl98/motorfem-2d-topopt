"""tests/fem/test_optimizers.py —— 替代优化器（SLSQP / Adam / GCMMA）。

在带不等式约束的二次型上验证各优化器**能收敛到解析最优附近**（来源 A）：

    min f0 = x0² + 2·x1²
    s.t.  f1 = 0.5 − (x0+x1) ≤ 0   ⇒ x0+x1 ≥ 0.5
          f2 = −0.7 + (x0+x1) ≤ 0   ⇒ x0+x1 ≤ 0.7（非活跃）

活跃约束 f1（取下界）：x0+x1 = 0.5。KKT ⇒ x0 = 2x1 ⇒ x1 = 0.5/3 ≈ 0.1667、
x0 ≈ 0.3333、f0* = x0²+2x1² ≈ 0.1667。MMA 的解析解测试同款问题（legacy
``MMA.py __main__``）。

**判据**：约束 ≤ 容差 + obj 显著低于初值（初值 x=[1,1] 处 obj=3）。各优化器
收敛速度不同，容差分开给（adam 最慢、给最宽）。
"""
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig
from motorfem.fem.optimizers import (OPTIMIZER_NAMES, optimize_adam,
                                     optimize_gcmma, optimize_mma,
                                     optimize_slsqp)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _quad_fun(x: np.ndarray):
    y0 = x[0] ** 2 + 2 * x[1] ** 2
    y1 = 0.5 - (x[0] + x[1])
    y2 = -0.7 + (x[0] + x[1])
    dy0 = np.array([2 * x[0], 4 * x[1]])
    dy1 = np.array([-1.0, -1.0])
    dy2 = np.array([1.0, 1.0])
    return np.array([y0, y1, y2]), np.array([dy0, dy1, dy2])


_OPTIMAL_F0 = 0.1666667


def _assert_solved(name, f_val, x, *, f_tol, con_tol):
    f = np.asarray(f_val, dtype=float)
    x = np.asarray(x, dtype=float)
    assert f.size == 3, f"{name}: f 行数应为 3，实际 {f.size}"
    # 目标显著下降（初值 3.0）
    assert f[0] < 1.5, f"{name}: obj 未下降：{f[0]:.4f}（最优 {_OPTIMAL_F0:.4f}）"
    # 约束可行
    assert np.all(f[1:] <= con_tol), f"{name}: 约束仍违规 {f[1:]}"
    # 目标接近解析最优
    assert abs(f[0] - _OPTIMAL_F0) <= f_tol, (
        f"{name}: obj {f[0]:.4f} 距解析最优 {_OPTIMAL_F0:.4f} 太远")
    # 活跃约束贴合（x0+x1 = 0.5 取下界）
    assert abs((x[0] + x[1]) - 0.5) <= con_tol + 1e-2, f"{name}: x 不合约束"


def _mma_cfg():
    c = RunConfig().mma
    c.MMA_move = 0.5     # 与 legacy 测试同款（见 mma_driver 签名默认）
    return c


# ---------------------------------------------------------------- 各自收敛
def test_slsqp_converges_to_analytic():
    f_val, x = optimize_slsqp(_quad_fun, np.array([1.0, 1.0]),
                              np.zeros(2), np.ones(2),
                              max_step=50, maxiter_per_step=30)
    _assert_solved("SLSQP", f_val, x, f_tol=1e-4, con_tol=1e-6)


def test_adam_converges_to_analytic():
    f_val, x = optimize_adam(_quad_fun, np.array([1.0, 1.0]),
                             np.zeros(2), np.ones(2),
                             max_step=800, lr=0.2, mu0=1.0,
                             mu_update_step=50)
    f = np.asarray(f_val, dtype=float)
    # 罚函数法在约束边界振荡：断言 obj 接近最优 + 约束可行即可（不要求 x 精确贴合）
    assert f[0] < 0.30, f"Adam obj 未到最优附近：{f[0]:.4f}（最优 {_OPTIMAL_F0:.4f}）"
    assert np.all(f[1:] <= 1e-3), f"Adam 约束仍违规 {f[1:]}"


def test_gcmma_converges_to_analytic():
    f_val, x = optimize_gcmma(_quad_fun, np.array([1.0, 1.0]),
                              np.zeros(2), np.ones(2),
                              max_step=100, f_rel_change=-1.0,
                              x_rel_change=-1.0)
    _assert_solved("GCMMA", f_val, x, f_tol=1e-3, con_tol=1e-4)


def test_mma_unchanged_reference():
    """mma 分发逐位走 MMA_optimize（黄金锚点不受替代优化器影响）。"""
    f_val, x = optimize_mma(_quad_fun, np.array([1.0, 1.0]),
                            np.zeros(2), np.ones(2),
                            max_step=200, mma_cfg=_mma_cfg(),
                            f_rel_change=-1.0, x_rel_change=-1.0)
    _assert_solved("MMA", f_val, x, f_tol=1e-4, con_tol=1e-6)


def test_dispatch_unknown_optimizer():
    with pytest.raises(ValueError):
        from motorfem.fem.optimizers import optimize
        optimize("nosuch", _quad_fun, np.array([1.0, 1.0]),
                 np.zeros(2), np.ones(2), max_step=1, mma_cfg=_mma_cfg())


def test_optimizer_names_cover_runtime_config():
    """config 的 runtime.optimizer 合法值 = OPTIMIZER_NAMES（GUI 下拉同源）。"""
    for name in OPTIMIZER_NAMES:
        c = RunConfig()
        c.runtime.optimizer = name
        assert c.runtime.optimizer == name


# ---------------------------------------------------------------- adam 子采样
def test_adam_sampled_idx_passthrough():
    """Adam 的 sample_size 每步把随机时间索引子集传给 fun。"""
    seen = []

    def fun(x, sampled_idx=None):
        seen.append(None if sampled_idx is None else list(sampled_idx))
        return _quad_fun(x)

    optimize_adam(fun, np.array([1.0, 1.0]), np.zeros(2), np.ones(2),
                  max_step=5, sample_size=1, rng_seed=0, n_time_points=11)
    assert len(seen) == 5
    assert all(len(s) == 1 for s in seen), f"应每步传 1 个索引，实际 {seen}"
