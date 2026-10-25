# -*- coding: utf-8 -*-
"""收敛判据的契约测试（纯合成数据，不跑 FEM）。

覆盖 `tools/check_convergence.py::analyse` 的四种判定。这些判据是用来
**决定要不要停优化**的，判错方向的代价（白跑 400 步 / 丢掉更优解）很高，
所以每种情形都钉一条。
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_convergence", ROOT / "tools" / "scripts" / "check_convergence.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def _f(obj, cons):
    """把 obj 序列 + 约束序列拼成 (n, 1+n_con)。"""
    o = np.asarray(obj, float).reshape(-1, 1)
    c = np.asarray(cons, float)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    return np.hstack([o, c])


def test_still_moving_when_objective_descends():
    n = 60
    f = _f(np.linspace(100, 10, n), np.full(n, -0.5))
    r = cc.analyse(f, window=30)
    assert r["verdict"] == "still_moving"
    assert not r["plateau"]


def test_converged_feasible_is_stoppable():
    """平台 + 约束宽松（不贴边）⇒ 正常收敛，可以停。"""
    n = 60
    obj = np.concatenate([np.linspace(100, 20, n - 30), np.full(30, 20.0)])
    f = _f(obj, np.full(n, -0.5))          # |g|=0.5 远离边界
    r = cc.analyse(f, window=30)
    assert r["verdict"] == "converged_feasible"
    assert r["plateau"] and r["feasible_now"]
    assert r["cons_at_bound_frac"] == 0.0


def test_stalled_at_bound_is_not_stoppable():
    """★ 平台 + 骑在约束边界 ⇒ 停滞，不该停。

    实测形态：con3 从 +0.6 收到 +0.00028（贴边），obj 停在 22.18 不动。
    此时停会错过中途 obj=11.26 的更优可行解。
    """
    n = 60
    obj = np.concatenate([np.linspace(100, 22.18, n - 30), np.full(30, 22.18)])
    cons = np.concatenate([np.full(n - 30, -0.5), np.full(30, 2.8e-4)])
    r = cc.analyse(_f(obj, cons), window=30)
    assert r["verdict"] == "stalled_at_bound"
    assert r["rides_bound"]


def test_stalled_at_bound_from_feasible_side():
    """贴边在**可行侧**（g 略 < 0）同样是停滞 —— 不该因符号不同就换个结论。"""
    n = 60
    obj = np.full(n, 22.18)
    cons = np.concatenate([np.full(n - 30, -0.5), np.full(30, -3.0e-4)])
    r = cc.analyse(_f(obj, cons), window=30)
    assert r["verdict"] == "stalled_at_bound"


def test_stalled_infeasible_when_constraint_stays_violated():
    """平台 + 明确不可行（违反量远大于容差）⇒ 该约束组合下无解。"""
    n = 60
    obj = np.full(n, 50.0)
    cons = np.concatenate([np.full(n - 30, -0.5), np.full(30, 0.8)])
    r = cc.analyse(_f(obj, cons), window=30)
    assert r["verdict"] == "stalled_infeasible"
    assert not r["feasible_now"]


def test_feas_tol_does_not_call_bound_riding_infeasible():
    """★ 防回归：1e-4 量级的贴边不能被判成「不可行无解」。

    这是本判据初版的实际 bug（实测 con3=+0.00028 被判 stalled_infeasible）。
    ``rides_bound`` 分支**优先于**可行性分支，所以即使 feas_tol 调小，贴边
    仍然判 ``stalled_at_bound`` —— 这正是我们要的：符号不该改变结论。
    """
    n = 60
    f = _f(np.full(n, 22.18), np.full(n, 2.8e-4))
    for ft in (1e-3, 0.0):
        assert cc.analyse(f, window=30, feas_tol=ft)["verdict"] == "stalled_at_bound"
    # 但违反量**远大于** bound_tol 时，就不算贴边了，该判不可行
    f_big = _f(np.full(n, 22.18), np.full(n, 0.5))
    assert cc.analyse(f_big, window=30)["verdict"] == "stalled_infeasible"


def test_best_feasible_reports_index_and_gap_signal():
    """全程最优可行点：应能指出中途更好的解（实测第 48 步 vs 末步差 97%）。"""
    n = 100
    obj = np.full(n, 22.0)
    obj[47] = 11.0                          # 中途有个更好的可行点
    cons = np.full(n, -0.5)
    r = cc.analyse(_f(obj, cons), window=30)
    b = r["best_feasible"]
    assert b is not None and b["step"] == 48 and b["obj"] == 11.0
    assert b["n_feasible"] == n


def test_best_feasible_none_when_nothing_feasible():
    f = _f(np.full(50, 30.0), np.full(50, 0.5))
    r = cc.analyse(f, window=30)
    assert r["best_feasible"] is None
    assert r["verdict"] == "stalled_infeasible"


def test_multi_constraint_window_uses_max_violation():
    """多约束时看**最大违反**，不能只看某一个。"""
    n = 60
    c1 = np.full(n, -0.5)
    c2 = np.concatenate([np.full(n - 30, -0.5), np.full(30, 0.8)])
    r = cc.analyse(_f(np.full(n, 10.0), np.hstack([c1.reshape(-1, 1),
                                                   c2.reshape(-1, 1)])),
                   window=30)
    assert r["verdict"] == "stalled_infeasible"


def test_empty_input_is_safe():
    assert cc.analyse(np.zeros((0, 3)))["verdict"] == "empty"


def test_window_larger_than_data_is_clamped():
    f = _f(np.full(5, 10.0), np.full(5, -0.5))
    r = cc.analyse(f, window=30)
    assert r["window"] == 5 and r["verdict"] == "converged_feasible"
