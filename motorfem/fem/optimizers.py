"""优化器层：MMA + 替代优化器（SLSQP / Adam 罚函数 / GCMMA）。

``mma_driver.MMA_optimize`` 是自研 MM 渐近线驱动（legacy 逐字移植，golden 钉死）。
本模块在其上加一层**可替换优化器**：统一入口 ``optimize`` 按名字分发，默认
``"mma"`` 时**逐位走原 MMA_optimize**（golden 契约不破）。

------------------------------------------------------------------------------
为什么需要替代优化器
------------------------------------------------------------------------------

§11② 与 2026-09-29 全量记录确认：MM 的 ``|T−T_obj|^p`` 非凸脉动约束下 MMA
3000 步仍不收敛（dx_rel 6.7e-3）。2026-10-11 又确认一个新死锁：``con3`` 作
目标（最大化平均转矩）时，x0 处目标梯度 ∝ −mean_dT、con1（绕固定 T_obj 的
p=10 脉振）梯度 ∝ +mean_dT，两者反平行（cos≈−0.999）⇒ 一阶方法局部无
可行下降方向。替代优化器提供**不同的下降机制**，值得平行对比：

- ``mma``    —— MM 渐近线（现状，逐位）。
- ``slsqp``  —— scipy SLSQP（序列二次规划，BFGS 近 Hessian，处理非线性约束
  的方式与 MM 完全不同：直接解 KKT 意义上的 QP 子问题）。
- ``adam``   —— Adam + 增广拉格朗日罚函数。无约束、纯一阶、随机子采样
  （``sample_size``）⇒ 天然 SGD 效应，配合「随机转子位置采样」（task 3）
  使用；μ 按 ``mu_update_step`` 步翻倍逼近约束。
- ``gcmma``  —— 移植 legacy ``GCMMA_optimize``（mmapy ``gcmmasub``，全局收敛
  版 MM：内层迭代把近似拉成保守，每外层步 2~3 次 fun）。

------------------------------------------------------------------------------
GUI 钩子（control / initial_state）约定
------------------------------------------------------------------------------

与 ``mma_driver`` 相同：``control`` 鸭子类型（``stop_requested()`` /
``wait_if_paused()`` / ``on_step(i_step, state)``），每步完成调用一次；
``initial_state`` 携带 ``x`` 即可（非 MMA 优化器无渐近线/对偶状态，续跑 =
从 ``x`` 重新起跑；MMA 仍是完整快照）。``state`` 统一含
``x/x_last_1/x_last_2/MMA_l/MMA_u/f_last/f_val/i_step``（非 MMA 优化器的
``MMA_l/u`` 填当前设计的 asyinit 渐近线，仅供 GUI 保存/续跑结构兼容）。

``fun(x)`` 返回 ``(f, df)``，``f[0]`` 目标、``f[1:]`` 约束（≤0）。Adam 额外
约定：``fun(x, sampled_idx=None)`` 可选位置子采样（task 3）。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .mma_driver import MMA_optimize

__all__ = [
    "OPTIMIZER_NAMES",
    "optimize",
    "optimize_mma",
    "optimize_slsqp",
    "optimize_adam",
    "optimize_gcmma",
]

#: 可选的优化器名（GUI 下拉 / config runtime.optimizer 的合法值）。
OPTIMIZER_NAMES = ("mma", "slsqp", "adam", "gcmma")


# ---------------------------------------------------------------- 状态快照
def _snapshot(x: np.ndarray, x_last_1: np.ndarray, x_last_2: np.ndarray,
              mma_l: np.ndarray, mma_u: np.ndarray, f_last: float,
              f_val: np.ndarray, i_step: int) -> dict:
    """统一 on_step state（与 mma_driver 的续跑快照键一致）。"""
    return {
        "x": np.asarray(x, dtype=float).copy(),
        "x_last_1": np.asarray(x_last_1, dtype=float).copy(),
        "x_last_2": np.asarray(x_last_2, dtype=float).copy(),
        "MMA_l": np.asarray(mma_l, dtype=float).copy(),
        "MMA_u": np.asarray(mma_u, dtype=float).copy(),
        "f_last": float(f_last),
        "f_val": np.asarray(f_val, dtype=float).copy(),
        "i_step": int(i_step),
    }


def _emit(control, i_step: int, x: np.ndarray, x_last_1: np.ndarray,
          x_last_2: np.ndarray, mma_l: np.ndarray, mma_u: np.ndarray,
          f_last: float, f_val: np.ndarray) -> None:
    if control is not None:
        control.on_step(i_step, _snapshot(x, x_last_1, x_last_2, mma_l, mma_u,
                                          f_last, f_val, i_step))


def _restore_x(initial_state) -> Tuple[np.ndarray, int]:
    if initial_state is None:
        return None, 0
    return np.asarray(initial_state["x"], dtype=float), int(initial_state["i_step"])


# ---------------------------------------------------------------- mma
def optimize_mma(fun: Callable, x0: np.ndarray, x_lower: np.ndarray,
                 x_upper: np.ndarray, *, max_step: int, mma_cfg,
                 control=None, initial_state=None, move_at=None,
                 f_rel_change=None, x_rel_change=None) -> Tuple[np.ndarray, np.ndarray]:
    """MMA（legacy 逐字路径）。``mma_cfg`` 是 ``MMAConfig``；``f/x_rel_change``
    可覆盖（golden 用 -1 关退出）。"""
    m = mma_cfg
    return MMA_optimize(
        fun, x0, x_upper, x_lower,
        f_rel_change=m.f_rel_change if f_rel_change is None else f_rel_change,
        x_rel_change=m.x_rel_change if x_rel_change is None else x_rel_change,
        max_step=max_step,
        MMA_move=m.MMA_move, MMA_dgrad=m.MMA_dgrad, MMA_rra0=m.MMA_rra0,
        MMA_albefa=m.MMA_albefa, MMA_asyinit=m.MMA_asyinit,
        MMA_asydecr=m.MMA_asydecr, MMA_asyincr=m.MMA_asyincr,
        control=control, initial_state=initial_state, move_at=move_at,
    )


# ---------------------------------------------------------------- slsqp
def optimize_slsqp(
    fun: Callable, x0: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray,
    *, max_step: int = 100, control=None, initial_state=None,
    ftol: float = 1e-6, maxiter_per_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """scipy SLSQP：序列二次规划，非线性不等式约束 + 解析 jac。

    一次「外层步」= 一次 ``minimize``（内部迭代数受 ``maxiter_per_step`` 限制）。
    每内部迭代调一次 ``fun``（记忆化避免 f/jac 双调）。SLSQP 无渐近线状态，
    ``initial_state`` 只恢复 ``x``。

    Parameters
    ----------
    maxiter_per_step : int
        每次 minimize 允许的 SLSQP 迭代数。1 = 每外层步一次 QP 迭代
        （对「优化器对比」的口径更接近 MMA 的一步一次 fun）；大值 = 让
        SLSQP 一次跑到底。默认 1 ⇒ 外层步数 ≈ fun 调用次数。
    """
    from scipy import optimize as sp_optimize

    x, _ = _restore_x(initial_state)
    if x is None:
        x = np.asarray(x0, dtype=float).copy()

    memo: dict = {}

    def _eval(xk: np.ndarray):
        key = bytes(np.asarray(xk, dtype=float).tobytes())
        hit = memo.get(key)
        if hit is None:
            f, df = fun(xk)
            memo[key] = (np.asarray(f), np.asarray(df))
        return memo[key]

    def f_obj(xk):
        return _eval(xk)[0][0]

    def j_obj(xk):
        return _eval(xk)[1][0]

    n_con = None
    cons = []

    def _build_cons(xk):
        nonlocal n_con, cons
        if n_con is not None:
            return
        f = _eval(xk)[0]
        n_con = f.size - 1
        for i in range(n_con):
            cons.append({
                "type": "ineq",
                "fun": (lambda ii: lambda xx: -_eval(xx)[0][1 + ii])(i),
                "jac": (lambda ii: lambda xx: -_eval(xx)[1][1 + ii])(i),
            })

    x_last_1 = x.copy()
    x_last_2 = x.copy()
    f_last = np.inf
    i_step = int(initial_state["i_step"]) if initial_state else 0
    l = x - 0.5 * (x_upper - x_lower)
    u = x + 0.5 * (x_upper - x_lower)

    for _ in range(max_step):
        if control is not None:
            if control.stop_requested():
                break
            control.wait_if_paused()
        _build_cons(x)
        res = sp_optimize.minimize(
            f_obj, x, jac=j_obj, method="SLSQP",
            bounds=[(float(lo), float(hi)) for lo, hi in
                    zip(x_lower, x_upper)],
            constraints=cons,
            options={"maxiter": maxiter_per_step, "ftol": ftol,
                     "disp": False},
            callback=lambda xk: _emit(control, i_step, xk, x_last_1, x_last_2,
                                      l, u, f_last, _eval(xk)[0]),
        )
        x_new = np.asarray(res.x, dtype=float).clip(x_lower, x_upper)
        f_val, df_val = _eval(x_new)
        df_rel = np.abs(f_val[0] - f_last) / (np.abs(f_val[0]) + 1e-20)
        dx_rel = np.linalg.norm(x_new - x) / (np.linalg.norm(x_new) + 1e-20)
        x_last_2 = x_last_1.copy()
        x_last_1 = x.copy()
        x = x_new
        i_step += 1
        if (np.abs(f_val[0] - f_last) <= np.abs(f_val[0]) * 1e-6
                and dx_rel <= 1e-6 and np.all(f_val[1:] <= 0)):
            f_last = f_val[0]
            break
        f_last = f_val[0]
        print(f'opti(SLSQP): i_step={i_step}, df_rel={df_rel:.6g}, '
              f'dx_rel={dx_rel:.6g}, f={f_val}\n')
        _emit(control, i_step, x, x_last_1, x_last_2, l, u, f_last, f_val)

    return f_val, x


# ---------------------------------------------------------------- adam
def optimize_adam(
    fun: Callable, x0: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray,
    *, max_step: int = 100, control=None, initial_state=None,
    lr: float = 0.1, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, mu0: float = 1.0, mu_update_step: int = 50,
    mu_max: float = 1e6, lambda0: float = 1.0,
    sample_size: Optional[int] = None, rng_seed: int = 0,
    n_time_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Adam + 增广拉格朗日罚函数（无约束一阶，配合随机子采样 = SGD 效应）。

    - 梯度：``g = df[0] + Σ_i (λ_i + μ·max(f_i,0))·df_i``（活跃约束加罚）；
    - AL 乘子：``λ_i ← max(λ_i + μ·max(f_i,0), 0)``（D7 同一形式）；
    - μ 每 ``mu_update_step`` 步 ×10（逼近约束）；上限 ``mu_max``；
    - 每步 ``Adam(x, g)`` 后 clip 到 ``[x_lower, x_upper]``；
    - ``sample_size``：每步随机抽 N 个时间索引传 ``fun(x, sampled_idx=...)``
      ⇒ 随机转子位置子采样（task 3 的 SGD 效应）。默认 None = 全点。
    - ``n_time_points``：时间索引总数（子采样范围）。``None`` = 从
      ``fun._n_time_points`` 探测；探测不到 ⇒ 子采样停用（全点）。

    注意：``fun`` 必须接受可选 ``sampled_idx`` 关键字（FieldSystem 的 warper
    已支持）；不支持的 fun 会在 sample_size 非 None 时 TypeError。
    """
    rng = np.random.RandomState(rng_seed)
    n_t = (n_time_points if n_time_points is not None
           else getattr(fun, "_n_time_points", None))
    x, i_step = _restore_x(initial_state)
    if x is None:
        x = np.asarray(x0, dtype=float).copy()

    # 约束数未知 ⇒ 首次 fun 后初始化 AL 向量
    lam = None
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    mu = mu0
    x_last_1 = x.copy()
    x_last_2 = x.copy()
    f_last = np.inf
    l = x - 0.5 * (x_upper - x_lower)
    u = x + 0.5 * (x_upper - x_lower)

    for _ in range(max_step):
        if control is not None:
            if control.stop_requested():
                break
            control.wait_if_paused()
        kw = {}
        if sample_size is not None and n_t:
            kw["sampled_idx"] = sorted(
                rng.choice(n_t, size=min(sample_size, n_t), replace=False)
                .astype(int).tolist())
        f_val, df_val = fun(x, **kw)
        if lam is None:
            lam = np.full(f_val.size - 1, lambda0)

        g = df_val[0].copy()
        for i in range(f_val.size - 1):
            if f_val[1 + i] > 0:
                g += (lam[i] + mu * f_val[1 + i]) * df_val[1 + i]
                lam[i] = max(lam[i] + mu * f_val[1 + i], 0.0)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mhat = m / (1 - beta1 ** (i_step + 1))
        vhat = v / (1 - beta2 ** (i_step + 1))
        x_new = (x - lr * mhat / (np.sqrt(vhat) + eps)).clip(x_lower, x_upper)

        if (i_step + 1) % mu_update_step == 0:
            mu = min(mu * 10.0, mu_max)

        # 先算 dx_rel 再更新 x_last_1（顺序纪律：dx 是「本次实际移动」）
        df_rel = np.abs(f_val[0] - f_last) / (np.abs(f_val[0]) + 1e-20)
        dx_rel = np.linalg.norm(x_new - x) / (np.linalg.norm(x_new) + 1e-20)
        x_last_2 = x_last_1.copy()
        x_last_1 = x.copy()
        x = x_new
        i_step += 1
        if (np.abs(f_val[0] - f_last) <= np.abs(f_val[0]) * 1e-6
                and dx_rel <= 1e-6 and np.all(f_val[1:] <= 0)):
            f_last = f_val[0]
            break
        f_last = f_val[0]
        print(f'opti(Adam): i_step={i_step}, df_rel={df_rel:.6g}, '
              f'dx_rel={dx_rel:.6g}, mu={mu:.1e}, f={f_val}\n')
        _emit(control, i_step, x, x_last_1, x_last_2, l, u, f_last, f_val)

    return f_val, x


def _probe_n_time(fun: Callable) -> int:
    """探测 fun 的时间点数（用于 sample_size 子采样；仅当 fun 暴露计数属性失败时）。"""
    for name in ("_n_time_points", "n_time_points"):
        v = getattr(fun, name, None)
        if v is not None:
            return int(v)
    return 0


# ---------------------------------------------------------------- gcmma
def optimize_gcmma(
    fun: Callable, x0: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray,
    *, max_step: int = 100, control=None, initial_state=None,
    f_rel_change: float = 1e-6, x_rel_change: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """GCMMA（mmapy ``gcmmasub``）—— 移植 legacy ``MMA.py:287-422`` 逐行。

    每外层步可能 2~3 次 ``fun``（xmma 处保守性检查 + xval 处重算梯度）。
    非凸约束的全局收敛版 MM。``initial_state`` 只恢复 ``x``。
    """
    from mmapy import asymp, concheck, gcmmasub, kktcheck, raaupdate

    x, _ = _restore_x(initial_state)
    if x is None:
        x = np.asarray(x0, dtype=float).copy()

    epsimin = 1e-7
    xval = x.reshape((-1, 1))
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = np.asarray(x_lower).reshape((-1, 1))
    xmax = np.asarray(x_upper).reshape((-1, 1))
    low = xmin.copy()
    upp = xmax.copy()
    raa0 = 0.01
    raa0eps = 1e-6
    outeriter = 0
    maxoutit = max_step
    kkttol = 0

    f_val, f_grad = fun(xval.flatten())
    f0val = f_val[0]
    df0dx = f_grad[0].reshape((-1, 1))
    fval = f_val[1:].reshape((-1, 1))
    dfdx = f_grad[1:, :]

    m = f_val.size - 1
    n = x.size
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    c = 1000 * eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    raa = 0.01 * eeem
    raaeps = 1e-6 * eeem

    kktnorm = kkttol + 10
    outit = 0
    x_last = x.copy()
    f_last = np.inf

    while kktnorm > kkttol and outit < maxoutit:
        if control is not None:
            if control.stop_requested():
                break
            control.wait_if_paused()
        outit += 1
        outeriter += 1

        low, upp, raa0, raa = asymp(
            outeriter, n, xval, xold1, xold2, xmin, xmax, low, upp,
            raa0, raa, raa0eps, raaeps, df0dx, dfdx)
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
            m, n, outeriter, epsimin, xval, xmin, xmax, low, upp, raa0, raa,
            f0val, df0dx, fval, dfdx, a0, a, c, d)

        f_val, f_grad = fun(xmma.flatten())
        f0valnew = f_val[0]
        fvalnew = f_val[1:].reshape((-1, 1))

        conserv = concheck(m, epsimin, f0app, f0valnew, fapp, fvalnew)
        innerit = 0
        if conserv == 0:
            while conserv == 0 and innerit <= 15:
                innerit += 1
                raa0, raa = raaupdate(
                    xmma, xval, xmin, xmax, low, upp, f0valnew, fvalnew,
                    f0app, fapp, raa0, raa, raa0eps, raaeps, epsimin)
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
                    m, n, outeriter, epsimin, xval, xmin, xmax, low, upp,
                    raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d)
                f_val, f_grad = fun(xmma.flatten())
                f0valnew = f_val[0]
                fvalnew = f_val[1:].reshape((-1, 1))
                conserv = concheck(m, epsimin, f0app, f0valnew, fapp, fvalnew)

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        f_last = f0val

        f_val, f_grad = fun(xval.flatten())
        f0val = f_val[0]
        df0dx = f_grad[0].reshape((-1, 1))
        fval = f_val[1:].reshape((-1, 1))
        dfdx = f_grad[1:, :]

        residu, kktnorm, residumax = kktcheck(
            m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s,
            xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)

        x = xval.flatten()
        df_rel = np.abs(f0val - f_last) / (np.abs(f0val) + 1e-20)
        dx_rel = np.linalg.norm(x - x_last) / (np.linalg.norm(x) + 1e-20)
        x_last = x.copy()
        if (np.abs(f0val - f_last) <= np.abs(f0val) * f_rel_change
                and dx_rel <= x_rel_change and np.all(fval <= 0)):
            break
        print(f'opti(GCMMA): i_step={outit}, df_rel={df_rel:.6g}, '
              f'dx_rel={dx_rel:.6g}, f={f_val}\n')
        _emit(control, outit, x, xold1.flatten(), xold2.flatten(),
              low.flatten(), upp.flatten(), f_last, f_val)

    return f_val.flatten(), xval.flatten()


# ---------------------------------------------------------------- 分发
def optimize(
    optimizer: str, fun: Callable, x0: np.ndarray, x_lower: np.ndarray,
    x_upper: np.ndarray, *, max_step: int = 100, mma_cfg=None,
    control=None, initial_state=None, move_at=None, **kw,
) -> Tuple[np.ndarray, np.ndarray]:
    """统一优化器入口。

    ``optimizer`` ∈ ``OPTIMIZER_NAMES``；``"mma"`` 时逐位走 legacy
    ``MMA_optimize``（golden 契约）。其余参数透传给各优化器：
    - ``mma``：``f_rel_change/x_rel_change`` 可覆盖；``mma_cfg`` 必须给。
    - ``slsqp``：``ftol/maxiter_per_step``。
    - ``adam``：``lr/mu0/mu_update_step/sample_size/rng_seed`` 等。
    - ``gcmma``：``f_rel_change/x_rel_change``。
    """
    if optimizer == "mma":
        if mma_cfg is None:
            raise ValueError("optimizer='mma' 需要 mma_cfg（MMAConfig）")
        return optimize_mma(fun, x0, x_lower, x_upper, max_step=max_step,
                            mma_cfg=mma_cfg, control=control,
                            initial_state=initial_state, move_at=move_at, **kw)
    if optimizer == "slsqp":
        return optimize_slsqp(fun, x0, x_lower, x_upper, max_step=max_step,
                              control=control, initial_state=initial_state, **kw)
    if optimizer == "adam":
        return optimize_adam(fun, x0, x_lower, x_upper, max_step=max_step,
                             control=control, initial_state=initial_state, **kw)
    if optimizer == "gcmma":
        return optimize_gcmma(fun, x0, x_lower, x_upper, max_step=max_step,
                              control=control, initial_state=initial_state, **kw)
    raise ValueError(f"未知优化器 {optimizer!r}（可选 {OPTIMIZER_NAMES}）")
