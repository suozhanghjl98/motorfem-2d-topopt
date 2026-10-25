"""发布向量的顺序纪律（阶段 7：设计文档 §7 的 ``published_layout.py``）。

「published」指一次 ``fun(x)`` 调用发布给 MMA 驱动（及 golden L4 快照）的
目标与约束向量：

    ``f``  形状 ``(5,)``        golden 的 ``call_{i:02d}_f``
    ``df`` 形状 ``(5, N_topo)`` golden 的 ``call_{i:02d}_df``

**顺序纪律**：这两个向量的行序必须与 legacy ``MMA10:3750-3752`` 完全一致
（先目标、后四个约束，约束按 Arkkio 转矩波动 → 密度 → 转矩 → VD 转矩波动）。
legacy 的顺序只在 ``np.array([obj, con1, ...])`` 的字面量里存在，改错一个
位置 MMA 子问题与 golden 全链都会对不上 —— 本模块把顺序常量集中到一处，
让「顺序错了」在构造处就被断言拦下。

设计文档 §7 原文（``docs/重构设计文档.md`` §5 阶段 7）：

    「**顺序纪律**：``published`` 向量元素顺序必须与 legacy 完全一致，
    把涉及顺序的常量集中到 ``optimization/published_layout.py`` 一个函数，
    并断言与基线 L4 的 ``f`` 向量长度一致。」

**本模块只提供常量与断言，不产生数值**。消费方（``objective.fun``）在
构造 ``f``/``df`` 后调用 ``assert_published_layout`` —— 纯断言，不改数值。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "PUBLISHED_NAMES",
    "PUBLISHED_SIZE",
    "assert_published_layout",
]

#: ``f`` 向量行序的语义名（照抄 ``MMA10:3750`` 的 ``np.array([obj, con1..4])``）。
#: 顺序即 legacy 顺序，**不可调换**。
PUBLISHED_NAMES = (
    "obj",            # g_Uelastic + g_stress（弹性应变能 + AL 应力罚）
    "con1",           # Arkkio 转矩波动约束
    "con2",           # 平均密度约束
    "con3",           # 平均转矩约束（1 - g_torque / mean_torque_obj）
    "con4",           # VD 转矩波动约束
)

#: ``f`` / ``df`` 的行数（= golden L4 ``call_{i}_f`` 的尺寸）。
PUBLISHED_SIZE = len(PUBLISHED_NAMES)


def assert_published_layout(f: np.ndarray, df: np.ndarray = None,
                            *, size: int = PUBLISHED_SIZE) -> None:
    """断言 ``f``（以及可选的 ``df``）符合发布布局的顺序纪律。

    Parameters
    ----------
    f : (size,) ndarray
        一次 fun 调用的目标 + 约束向量。
    df : (size, N) ndarray 或 None
        对应的梯度（行数必须一致；列数不限）。
    size : int
        期望行数。默认 ``PUBLISHED_SIZE``（5）= 全启用布局（golden 契约）；
        目标/约束使能自定义时（`objective.fun` 按 ``objective_role`` +
        ``enable_*`` 组装）传入实际行数 = 1 目标 + 参与约束数。

    Raises
    ------
    AssertionError
        行数 ≠ ``size``，或 ``df`` 行数与 ``f`` 不一致。
        行序本身无法在此校验（语义是标签的事），长度是防错的第一道闸。
    """
    assert f.shape[0] == size, (
        f"published f 应为 {size} 行，实际 {f.shape[0]} —— "
        f"顺序纪律被破坏，先查 objective.fun 的 f 构造"
    )
    if df is not None:
        assert df.shape[0] == f.shape[0], (
            f"published df 行数 {df.shape[0]} 与 f 的 {f.shape[0]} 不一致"
        )
