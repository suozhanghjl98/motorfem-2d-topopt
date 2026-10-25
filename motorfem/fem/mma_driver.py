"""阶段 4E：MMA 驱动的 legacy 逐字移植（``optimization_method/MMA.py:10-173``）。

legacy 的 ``MMA_optimize`` 是**自研驱动**（2024-10-17 重写版）：外层循环构造
渐近线子问题，内层用 ``scipy.optimize.minimize``（有 bounds ⇒ L-BFGS-B）解对偶
子问题。**不依赖 mmapy**（``MMA_optimize_v2``/``GCMMA_optimize`` 才用 mmapy，
非生产路径，不移植）。

------------------------------------------------------------------------------
为什么是「驱动」不是「内核」
------------------------------------------------------------------------------

它没有独立可测的数学（子问题就是对偶 + L-BFGS-B），且与求解器耦合（调 ``fun``）。
按 ``kernels/README.md`` 的判据不建 reference/fast 两层：只写一份逐行照搬的
实现，正确性由「golden L4 的 3 步 x/f 轨迹逐位对齐」钉死。

------------------------------------------------------------------------------
移植纪律
------------------------------------------------------------------------------

- 浮点结合顺序、``clip`` 调用、``np.where`` 掩码全部照抄（换写法会改变
  L-BFGS-B 的轨迹）。
- ``x0 = ones(N_con-1)``、``bounds = [(0, 1e4)]*(N_con-1)``、``jac=True``
  三个选择决定 L-BFGS-B 的路径，不能动。
- 退出条件在 ``x_last_2/x_last_1/x`` 更新**之后**、``f_last/i_step`` 更新
  **之前**（顺序影响轨迹）。golden 用 ``f_rel_change=x_rel_change=-1``
  ⇒ 条件恒假 ⇒ 跑满 ``max_step``。

------------------------------------------------------------------------------
GUI 钩子（默认 None，逐字不变）
------------------------------------------------------------------------------

网页 GUI（``tools/optim_gui/server.py``）需要暂停/单步/停止/续跑，但驱动本身
是**无状态函数**，不允许为此改结构。因此加两个可选参数，默认 ``None`` 时
代码路径与 legacy 逐字相同：

- ``control`` —— 鸭子类型对象，每次迭代的循环顶部调用
  ``control.stop_requested()``、``control.wait_if_paused()``，每次迭代**完成**
  （``i_step`` 已递增）后调用 ``control.on_step(i_step, state)``。``state`` 是
  续跑所需的完整快照（``x/x_last_1/x_last_2/MMA_l/MMA_u/f_last/i_step`` +
  当次 ``f_val``），与 ``initial_state`` 的格式一致。
- ``initial_state`` —— 续跑入口：用快照里保存的算法中间变量替代「从头初始化」
  （``x=copy(x0)``、``x_last_* = 0``、``f_last = inf``、``i_step = 0``）。必须与
  ``FieldSystem.restore_checkpoint_state``（AL/A 热启动缓存）一起用，轨迹才会
  与「不中断的从头跑」逐位一致（见 ``tests/fem/test_system.py`` 的续跑测试）。

这两个钩子只**读取/传递**驱动内部变量，不参与任何浮点运算，也不改迭代次序。
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy import optimize as sp_optimize

__all__ = ["MMA_optimize"]


def MMA_optimize(
    fun: Callable,
    x0: np.ndarray,
    x_upper: np.ndarray,
    x_lower: np.ndarray,
    f_rel_change: float = 1e-3,
    x_rel_change: float = 1e-3,
    max_step: int = 100,
    MMA_dgrad: float = 1e-3,
    MMA_rra0: float = 1e-5,
    MMA_albefa: float = 0.1,
    MMA_move: float = 0.5,
    MMA_asyinit: float = 0.5,
    MMA_asydecr: float = 0.7,
    MMA_asyincr: float = 1.2,
    control=None,
    initial_state=None,
    move_at: Optional[Callable[[int], float]] = None,
):
    """MM 渐近线法。返回 ``(f_val, x)``，与 ``MMA.py:10-173`` 逐行一致。

    ``fun(x)`` 返回 ``(f_val, f_grad)``（f_val 首元素是目标，其余是约束）。
    参数默认值 = ``MMA.py`` 签名；生产覆盖来自 ``MMAConfig``。

    ``control``/``initial_state`` 是 GUI 钩子（见模块 docstring）：默认
    ``None`` 时本函数逐字走 legacy 路径。

    ``move_at`` 是移动极限调度器：``move_at(i_step) -> float``，每步顶部
    把 ``MMA_move_cur`` 重算为该返回值（再经 ``assert 0<move<1`` 校验）。
    ``None``（默认）= ``MMA_move_cur`` 恒为 ``MMA_move``，逐字走 legacy。
    用途：p 连续化爬升到高 p 时约束变陡（梯度 ∝|T−T_obj|^9），需同步收紧
    移动极限防 overshoot（2026-10-10 探针结论，见 ``FieldSystem._current_move``）。
    """
    f_last = np.inf
    i_step = 0

    # 配置MMA算法的参数
    assert 0 <= MMA_dgrad < 1
    assert 0 < MMA_rra0 < 1
    assert 0 < MMA_albefa < 1
    assert 0 < MMA_move < 1
    assert 0 < MMA_asyinit < 1
    assert 0 < MMA_asydecr < 1
    assert 1 < MMA_asyincr

    # 避免除零错误
    MMA_eps = 1e-6

    # 初始化算法中间变量（续跑时用快照替换，见模块 docstring）
    if initial_state is not None:
        x = np.asarray(initial_state["x"], dtype=float)
        x_last_1 = np.asarray(initial_state["x_last_1"], dtype=float)
        x_last_2 = np.asarray(initial_state["x_last_2"], dtype=float)
        MMA_l = np.asarray(initial_state["MMA_l"], dtype=float)
        MMA_u = np.asarray(initial_state["MMA_u"], dtype=float)
        f_last = float(initial_state["f_last"])
        i_step = int(initial_state["i_step"])
    else:
        x = np.copy(x0)
        x_last_1 = np.zeros(x.shape)
        x_last_2 = np.zeros(x.shape)

    MMA_move_cur = MMA_move

    while i_step < max_step:
        # GUI 钩子：停止/暂停检查（默认 control=None 时无开销、行为不变）
        if control is not None:
            if control.stop_requested():
                break
            control.wait_if_paused()
        # 移动极限调度（默认 None = 固定 MMA_move，逐字 legacy）
        if move_at is not None:
            MMA_move_cur = move_at(i_step)
            assert 0 < MMA_move_cur < 1, f"move_at({i_step}) 非法: {MMA_move_cur}"
        # 计算函数值及其梯度
        t0 = time.time()
        f_val, f_grad = fun(x)
        print(f'MMA: compute fun: {time.time() - t0:.4f}s')
        t0 = time.time()

        # 目标函数个数
        N_con = f_val.size
        assert N_con > 1

        # 构造MMA子问题
        if i_step >= 2:
            MMA_gamma = (MMA_asydecr * ((x - x_last_1) * (x_last_1 - x_last_2) < 0)
                         + MMA_asyincr * ((x - x_last_1) * (x_last_1 - x_last_2) > 0)
                         + 1.0 * ((x - x_last_1) * (x_last_1 - x_last_2) == 0)
                         )
            MMA_l = x - MMA_gamma * (x_last_1 - MMA_l)
            MMA_u = x + MMA_gamma * (MMA_u - x_last_1)
        else:
            MMA_l = x - MMA_asyinit * (x_upper - x_lower)
            MMA_u = x + MMA_asyinit * (x_upper - x_lower)

        MMA_alpha = MMA_l + MMA_albefa * (x - MMA_l)
        MMA_alpha = MMA_alpha.clip(min=x_lower)
        MMA_alpha = MMA_alpha.clip(min=x - MMA_move_cur * (x_upper - x_lower))

        MMA_beta = MMA_u - MMA_albefa * (MMA_u - x)
        MMA_beta = MMA_beta.clip(max=x_upper)
        MMA_beta = MMA_beta.clip(max=x + MMA_move_cur * (x_upper - x_lower))

        f_grad_pos = f_grad.clip(min=0)
        f_grad_neg = (-f_grad).clip(min=0)

        MMA_p = np.zeros((N_con, x.size))
        MMA_q = np.zeros((N_con, x.size))

        for i_con in range(N_con):
            MMA_p[i_con] = (MMA_u - x) ** 2 * (
                    (1 + MMA_dgrad) * f_grad_pos[i_con] + MMA_dgrad * f_grad_neg[i_con] + MMA_rra0 / (
                    x_upper - x_lower))
            MMA_q[i_con] = (x - MMA_l) ** 2 * (
                    MMA_dgrad * f_grad_pos[i_con] + (1 + MMA_dgrad) * f_grad_neg[i_con] + MMA_rra0 / (
                    x_upper - x_lower))

        MMA_r = f_val[1:] - np.sum(MMA_p[1:, :] / (MMA_u - x + MMA_eps) + MMA_q[1:, :] / (x - MMA_l + MMA_eps), axis=1)

        # 求解MMA子问题（对偶，L-BFGS-B）
        def MMA_subproblem(y):
            MMA_pl_alpha = ((MMA_p[0] + y @ MMA_p[1:, :]) / (MMA_u - MMA_alpha + MMA_eps) ** 2
                            - (MMA_q[0] + y @ MMA_q[1:, :]) / (MMA_l - MMA_alpha + MMA_eps) ** 2)

            MMA_pl_beta = ((MMA_p[0] + y @ MMA_p[1:, :]) / (MMA_u - MMA_beta + MMA_eps) ** 2
                           - (MMA_q[0] + y @ MMA_q[1:, :]) / (MMA_l - MMA_beta + MMA_eps) ** 2)

            MMA_x_new = ((MMA_l * np.sqrt(MMA_p[0] + y @ MMA_p[1:, :])
                          + MMA_u * np.sqrt(MMA_q[0] + y @ MMA_q[1:, :]))
                         / (np.sqrt(MMA_p[0] + y @ MMA_p[1:, :])
                            + np.sqrt(MMA_q[0] + y @ MMA_q[1:, :])))

            MMA_x_new[np.where(MMA_pl_alpha >= 0)] = MMA_alpha[np.where(MMA_pl_alpha >= 0)]
            MMA_x_new[np.where(MMA_pl_beta <= 0)] = MMA_beta[np.where(MMA_pl_beta <= 0)]

            MMA_W = y @ MMA_r + np.sum((MMA_p[0] + y @ MMA_p[1:, :]) / (MMA_u - MMA_x_new + MMA_eps)
                                       + (MMA_q[0] + y @ MMA_q[1:, :]) / (MMA_x_new - MMA_l + MMA_eps))

            MMA_dW = MMA_r + np.sum(MMA_p[1:, :] / (MMA_u - MMA_x_new + MMA_eps) + MMA_q[1:, :] / (MMA_x_new - MMA_l + MMA_eps), axis=1)

            return -MMA_W, -MMA_dW

        # 使用通用求解器求解子问题
        t1 = time.time()
        res = sp_optimize.minimize(MMA_subproblem, x0=np.ones(N_con - 1), jac=True,
                                   bounds=[(0, 1e4)] * (N_con - 1))
        print(f'MMA: solve sub-problem: {time.time() - t1:.4f}s')

        MMA_y = res.x
        assert np.all(MMA_y >= 0)

        # 计算求解结果对应的x取值
        MMA_pl_alpha = ((MMA_p[0] + MMA_y @ MMA_p[1:, :]) / (MMA_u - MMA_alpha + MMA_eps) ** 2
                        - (MMA_q[0] + MMA_y @ MMA_q[1:, :]) / (MMA_l - MMA_alpha + MMA_eps) ** 2)
        MMA_pl_beta = ((MMA_p[0] + MMA_y @ MMA_p[1:, :]) / (MMA_u - MMA_beta + MMA_eps) ** 2
                       - (MMA_q[0] + MMA_y @ MMA_q[1:, :]) / (MMA_l - MMA_beta + MMA_eps) ** 2)
        MMA_x_new = ((MMA_l * np.sqrt(MMA_p[0] + MMA_y @ MMA_p[1:, :]) + MMA_u * np.sqrt(
            MMA_q[0] + MMA_y @ MMA_q[1:, :]))
                     / (np.sqrt(MMA_p[0] + MMA_y @ MMA_p[1:, :]) + np.sqrt(MMA_q[0] + MMA_y @ MMA_q[1:, :])))
        MMA_x_new[np.where(MMA_pl_alpha >= 0)] = MMA_alpha[np.where(MMA_pl_alpha >= 0)]
        MMA_x_new[np.where(MMA_pl_beta <= 0)] = MMA_beta[np.where(MMA_pl_beta <= 0)]

        # ⚠ 上下界必须用**实际 bounds**，不能写死 [0,1]（2026-10-18 实测）：
        #   密度设计变量是 [0,1]，legacy 写死等价、逐位不变；但极坐标傅里叶
        #   参数化的变量是**系数**（可正可负、范围 ±coeff_bound），写死 [0,1]
        #   会把负系数全部截成 0 —— 实测第一步即把 T_flux 从 54 打到 26，
        #   优化器再也回不来。默认路径 x_lower=0/x_upper=1 ⇒ 与旧写法逐位一致。
        MMA_x_new = MMA_x_new.clip(min=x_lower, max=x_upper)

        # 更新变量
        x_last_2 = np.copy(x_last_1)
        x_last_1 = np.copy(x)
        x = MMA_x_new

        # 检查是否满足退出条件
        df_rel = np.abs(f_val[0] - f_last) / (np.abs(f_val[0]) + 1e-20)
        dx_rel = np.linalg.norm(x - x_last_1) / (np.linalg.norm(x) + 1e-20)
        if (np.abs(f_val[0] - f_last) <= np.abs(f_val[0]) * f_rel_change
                and np.linalg.norm(x - x_last_1) <= np.linalg.norm(x) * x_rel_change
                and np.all(f_val[1:] <= 0)):
            break

        f_last = f_val[0]
        print(f'MMA: optimize: {time.time() - t0:.4f}s')
        print(f'opti: i_step={i_step}, df_rel={df_rel:.6g}, dx_rel={dx_rel:.6g}, f={f_val}\n')

        i_step += 1

        # GUI 钩子：每步完成后的回调（携带续跑快照；默认 control=None 时不调用）
        if control is not None:
            control.on_step(i_step, {
                "i_step": i_step,
                "x": x.copy(),
                "x_last_1": x_last_1.copy(),
                "x_last_2": x_last_2.copy(),
                "MMA_l": MMA_l.copy(),
                "MMA_u": MMA_u.copy(),
                "f_last": float(f_last),
                "f_val": np.asarray(f_val),
            })

    return f_val, x
