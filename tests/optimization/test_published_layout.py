"""tests/optimization/test_published_layout.py —— 阶段 7：发布向量的顺序纪律。

来源 A（解析性质）：``assert_published_layout`` 对正确的 ``(5,)`` / ``(5, N)``
通过、对错误长度抛 AssertionError —— 长度是顺序纪律的第一道闸（行序本身
是标签语义，长度错是最常见的破坏方式）。

来源 B（与 legacy / golden 逐位比对）：对**真实 golden L4** 的 ``call_{i}_f``
（形状 ``(5,)``）与 ``call_{i}_df``（形状 ``(5, N_topo)``）断言通过 —— 这
证明发布布局常量与基线冻结的 f 向量长度一致（设计文档 §7 原文要求：
「断言与基线 L4 的 ``f`` 向量长度一致」）。

来源 C（reference vs fast 的同类纪律）：``objective.fun`` 构造的 ``f``/``df``
通过断言（golden 注入路径已由 ``tests/fem/test_objective.py`` 覆盖，这里只
确认断言本身接进了 fun —— 见 ``test_fun_constructs_published_f``）。

运行::

    python -m pytest tests/optimization/test_published_layout.py -v
"""
import os
from pathlib import Path

import numpy as np
import pytest

from motorfem.optimization import (PUBLISHED_NAMES, PUBLISHED_SIZE,
                                   assert_published_layout)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO_ROOT / os.environ.get("MOTORFEM_BASELINE", "golden")
_GOLDEN_L4 = _GOLDEN_DIR / "L4_mech_mma.npz"


# ---------------------------------------------------------------- 来源 A
def test_published_size_is_five():
    """生产路径 f = [obj, con1..con4]，5 行（legacy MMA10:3750）。"""
    assert PUBLISHED_SIZE == 5
    assert len(PUBLISHED_NAMES) == PUBLISHED_SIZE
    assert PUBLISHED_NAMES[0] == "obj"
    assert PUBLISHED_NAMES[-1] == "con4"


def test_assert_passes_correct_shapes():
    assert_published_layout(np.zeros(5))
    assert_published_layout(np.zeros(5), np.zeros((5, 3181)))
    assert_published_layout(np.zeros((1, 5))[0])  # (5,) 不管列序


def test_assert_rejects_wrong_f_length():
    with pytest.raises(AssertionError, match="published f"):
        assert_published_layout(np.zeros(4))


def test_assert_rejects_df_row_mismatch():
    with pytest.raises(AssertionError, match="df 行数"):
        assert_published_layout(np.zeros(5), np.zeros((4, 3181)))


# ---------------------------------------------------------------- 来源 B
def test_golden_L4_f_matches_published_layout():
    """golden L4 的 call_{i}_f 形状 == (5,) —— 与发布布局长度一致。"""
    if not _GOLDEN_L4.exists():
        pytest.skip(f"{_GOLDEN_L4} 不存在 —— 基线冻结尚未入库")
    g = np.load(_GOLDEN_L4)
    for c in range(3):
        f = np.asarray(g[f"call_{c:02d}_f"])
        df = np.asarray(g[f"call_{c:02d}_df"])
        assert_published_layout(f, df)
        assert f.shape == (PUBLISHED_SIZE,)


# ---------------------------------------------------------------- 来源 C
def test_fun_constructs_published_f():
    """objective.fun 的 f/df 构造经过 assert_published_layout（防顺序漂移）。

    ⚠ 2026-10-10 目标使能后调用带 ``size=f.shape[0]``（自定义布局行数可变），
    前缀匹配即可（防「换了别的断言/漏掉断言」都拦得住）。
    """
    import inspect
    from motorfem.fem import objective
    src = inspect.getsource(objective.fun)
    assert "assert_published_layout(f, df" in src
