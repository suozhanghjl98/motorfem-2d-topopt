"""motorfem.config.enums —— 阶段 2：互斥布尔标志的枚举折并

legacy（``MMA10:112-120``）用两组互斥布尔选择应力约束形式：

* ``use_type1_stress`` / ``use_type2_stress``
* ``use_type1_rel_stress`` … ``use_type4_rel_stress``

使用模式（``MMA10:2494-2511``）是 ``assert 至少一个为 True`` + ``if/elif`` 链，
即**两两互斥**。把它们折成两个枚举是阶段 2 唯一改结构的参数改动
（设计文档 §阶段 2）：只影响分支选择、不影响数值。

**折并纪律**：collapse_* 要求**恰好一个** True。legacy 的 assert 只保证「至少一个」，
两个同时 True 时 legacy 静默走第一个分支 —— 折并选择在此报错，把这种非法配置
暴露出来而不是延续静默（这正是枚举相对布尔的价值）。

每个枚举带 ``use_type`` 到 legacy 布尔的下标映射，反向展开（``expand_*``）
供测试验证「折并 ⇄ 展开」的双向一致性。
"""
from __future__ import annotations

from enum import Enum
from typing import List, Tuple

__all__ = ["StressType", "RelStressType", "collapse_stress_type",
           "collapse_rel_stress_type", "expand_stress_type", "expand_rel_stress_type"]


class StressType(Enum):
    """折并 ``use_type1_stress`` / ``use_type2_stress`` 的枚举。"""

    TYPE1 = 1
    TYPE2 = 2


class RelStressType(Enum):
    """折并 ``use_type{1..4}_rel_stress`` 的枚举。"""

    TYPE1 = 1
    TYPE2 = 2
    TYPE3 = 3
    TYPE4 = 4


def collapse_stress_type(use_type1: bool, use_type2: bool) -> StressType:
    """两个互斥布尔 → 枚举。非法输入（0 个或 2 个 True）抛 ValueError。

    **Why 报错而非容忍**：legacy 的 ``assert use_type1_stress or use_type2_stress``
    只保证至少一个；两个同时 True 时 legacy 静默走 ``if use_type1_stress`` 分支。
    阶段 2 的折并选择在此显式报错，避免把非法配置静默带进新代码。
    """
    flags = [bool(use_type1), bool(use_type2)]
    if sum(flags) != 1:
        raise ValueError(
            f"use_type1_stress / use_type2_stress 必须恰好一个为 True（互斥），实为 {flags}"
        )
    return StressType.TYPE1 if use_type1 else StressType.TYPE2


def collapse_rel_stress_type(t1: bool, t2: bool, t3: bool, t4: bool) -> RelStressType:
    """四个互斥布尔 → 枚举。非法输入（≠1 个 True）抛 ValueError。"""
    flags = [bool(t1), bool(t2), bool(t3), bool(t4)]
    if sum(flags) != 1:
        raise ValueError(
            f"use_type{{1..4}}_rel_stress 必须恰好一个为 True（互斥），实为 {flags}"
        )
    for i, flag in enumerate(flags):
        if flag:
            return RelStressType(i + 1)
    raise AssertionError("unreachable")  # sum==1 保证上面必有 return


def expand_stress_type(st: StressType) -> Tuple[bool, bool]:
    """枚举 → (use_type1_stress, use_type2_stress)。供测试做双向一致性验证。"""
    return (st == StressType.TYPE1, st == StressType.TYPE2)


def expand_rel_stress_type(st: RelStressType) -> List[bool]:
    """枚举 → [use_type1..4_rel_stress]。"""
    return [st.value == i for i in (1, 2, 3, 4)]
