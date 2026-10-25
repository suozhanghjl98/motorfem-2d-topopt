"""内核 19 测试：AL 乘子写回与罚因子调度。

测试三级来源（`docs/重构设计文档.md` §1.4 与 `CLAUDE.md` §4）：

  来源 A 解析解/解析性质 —— μ 调度的手算点、clip、**顺序纪律**
         （μ 用更新前的 ``AL_N_call``）、μ 无历史记忆、λ 引用写回、
         ``AL_N_call`` 递增判据（严格 ``> 0``）
  来源 B 与 legacy 一致 —— 照抄 ``MMA10:4025-4032 + 4056-4058``
         （``optimizer_func_warper_AL_v2.__call__`` 的 AL 段）逐位比对
  来源 C —— **不适用**：本内核无 ``fast/`` 层（标量状态机，无向量化空间，
         见 `reference/al.py` 模块 docstring 第五节与上级 README 的判据）

⚠ 本内核与 D7 的牵连（设计文档 §12，**只登记不改**）：
  ``λ_new = min(0, λ+μh)`` 恒为 0 ⇒ AL 退化为**纯罚函数** ⇒ 本内核的 μ 调度
  是「约束力强度的唯一来源」。判定 D7 是「文档与代码同错」还是「有意为之」
  **不属于本批**；提取时照抄 legacy 的实际调度（``_v2`` 生产路径）。

运行::

    python -m pytest tests/kernels/test_al.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from motorfem.kernels.reference import al as ref_al

#: 生产参数（照抄 ``MMA10:3841-3844``）
MU0, MU_MAX, STEP = 0.01, 1e3, 50


# =============================================================================
# 来源 A —— μ 调度的手算点（解析性质）
# =============================================================================

def test_mu_schedule_hand_computed_points():
    """``μ(n) = min(μ0·(1 + n/step)², μ_max)`` 的手算点。

    取 ``λ = 0``（生产初值）、``stress_loss_al = 0``（不递增计数），
    则一次更新后 ``al_mu`` 完全由旧的 ``al_n_call`` 决定。手算::

        n=0   → 0.01·(1+0/50)²   = 0.01
        n=50  → 0.01·(1+1)²      = 0.04
        n=100 → 0.01·(1+2)²      = 0.09
        n=500 → 0.01·(1+10)²     = 1.21
    """
    zero_lam = np.zeros(3)
    for n, want in ((0, 0.01), (50, 0.04), (100, 0.09), (500, 1.21)):
        state = ref_al.ALState(al_lambda=zero_lam, al_mu=0.0, al_n_call=n)
        new = ref_al.al_update(zero_lam, stress_loss_al=0.0, state=state)
        assert new.al_mu == pytest.approx(want, rel=1e-15), f"n={n}"
        # 防退化：μ 确实是 n 的增函数（下一档必须更大）
        if n < 500:
            nxt = MU0 * (1 + (n + STEP) / STEP) ** 2
            assert new.al_mu < nxt


def test_mu_clips_at_mu_max():
    """``min(..., μ_max)`` 封顶。n 足够大时 μ 恒等于 ``μ_max``。

    解 ``0.01·(1+n/50)² = 1e3`` 得 ``n ≈ 15762``，所以 ``n = 20000`` 一定
    封顶；``n = 5000`` 时 ``0.01·101² = 102.01 < 1e3`` 不封顶。
    """
    zero_lam = np.zeros(3)
    state = ref_al.ALState(al_lambda=zero_lam, al_mu=0.0, al_n_call=20000)
    new = ref_al.al_update(zero_lam, 0.0, state)
    assert new.al_mu == MU_MAX
    # 防退化：封顶前的原值应大于 μ_max（证明是 clip 生效、不是巧合相等）
    assert MU0 * (1 + 20000 / STEP) ** 2 > MU_MAX

    state = ref_al.ALState(al_lambda=zero_lam, al_mu=0.0, al_n_call=5000)
    new = ref_al.al_update(zero_lam, 0.0, state)
    assert new.al_mu == pytest.approx(MU0 * (1 + 5000 / STEP) ** 2, rel=1e-15)
    assert new.al_mu < MU_MAX


def test_mu_uses_the_pre_update_al_n_call():
    """★★ **顺序纪律**：μ 用**更新前**的 ``AL_N_call``（legacy ``:4026``
    先读旧值、``:4057-4058`` 后递增）。

    从 ``al_n_call = 50`` 出发且 ``stress_loss_al > 0``（本轮违反约束）：
    若用旧值 ``μ = 0.01·(1+1)² = 0.04``；若误用更新后的 ``n=51`` 则
    ``μ = 0.01·(1+51/50)² ≈ 0.0408`` —— 两种写法**浮点不同**，可分辨。
    """
    zero_lam = np.zeros(3)
    state = ref_al.ALState(al_lambda=zero_lam, al_mu=0.0, al_n_call=50)
    new = ref_al.al_update(zero_lam, stress_loss_al=1.0, state=state)
    assert new.al_mu == pytest.approx(0.04, rel=1e-15)
    assert new.al_mu == pytest.approx(MU0 * (1 + 50 / STEP) ** 2, rel=1e-15)
    assert new.al_n_call == 51                        # 计数递增了，但 μ 用的旧值
    assert new.al_mu != pytest.approx(MU0 * (1 + 51 / STEP) ** 2, rel=1e-15)


def test_mu_has_no_memory_of_the_old_mu():
    """μ **无历史记忆**：公式只读 ``al_n_call``，不读旧 ``al_mu``。

    对比标准 AL（笔记《MMA方法实现AL的优化.md》的 ``σ`` 按 ``e``-比率调度），
    这是本内核「纯罚函数强度」的签名性质 —— 因为乘子恒为 0（D7），
    没有任何量需要被「消化」。
    """
    zero_lam = np.zeros(3)
    state_a = ref_al.ALState(al_lambda=zero_lam, al_mu=0.5, al_n_call=100)
    state_b = ref_al.ALState(al_lambda=zero_lam, al_mu=999.0, al_n_call=100)
    new_a = ref_al.al_update(zero_lam, 0.0, state_a)
    new_b = ref_al.al_update(zero_lam, 0.0, state_b)
    assert new_a.al_mu == new_b.al_mu == pytest.approx(0.09, rel=1e-15)


# =============================================================================
# 来源 A —— λ 写回与计数判据
# =============================================================================

def test_lambda_writeback_is_reference_not_copy():
    """λ **引用写回**（legacy ``:4025`` 的别名赋值，不拷贝）。

    传入的 ``al_lambda_new`` 就是新状态的 ``al_lambda``（同一对象）。
    """
    lam_new = np.linspace(-1.0, 2.0, 5)   # 任意非零值（生产中恒 0，见 D7）
    state = ref_al.ALState(al_lambda=np.zeros(5), al_mu=1.0, al_n_call=0)
    new = ref_al.al_update(lam_new, 0.0, state)
    assert new.al_lambda is lam_new
    assert np.array_equal(new.al_lambda, lam_new)


def test_al_n_call_increments_only_when_stress_strictly_positive():
    """``AL_N_call`` 判据是**严格 ``> 0``**（``:4057``）：0 与负数都不递增。"""
    zero_lam = np.zeros(3)
    for stress, want in ((1e-8, 7), (0.0, 6), (-2.5, 6)):
        state = ref_al.ALState(al_lambda=zero_lam, al_mu=1.0, al_n_call=6)
        new = ref_al.al_update(zero_lam, stress_loss_al=stress, state=state)
        assert new.al_n_call == want, f"stress={stress}"


def test_lambda_stays_zero_under_production_initial_state():
    """★★ **D7 的归纳起点**：生产初值 ``λ = 0``（``:3840``）+ 写回 ⇒
    ``λ ≡ 0`` 对每一轮成立 —— 所以「这个 AL」是纯罚函数。

    这条是本内核与设计文档 §12 的 D7 之间的**可执行检查**：哪天内核 17
    修了 D7（乘子开始非零），``al_update`` 的 λ 也会随之非零 —— 届时这里
    的断言要改写，且 §12 的标定（``mu0``/``mu_max``/``step``）要重做。
    """
    state = ref_al.al_initial_state(n_elem=7)
    assert np.all(state.al_lambda == 0.0)
    new = ref_al.al_update(state.al_lambda, stress_loss_al=1.0, state=state)
    assert np.all(new.al_lambda == 0.0)
    # 防退化：即便用任意非零 λ_new（模拟 D7 修好后），更新也要原样带上它
    lam_new = np.arange(7, dtype=float)
    new2 = ref_al.al_update(lam_new, 0.0, state)
    assert np.array_equal(new2.al_lambda, lam_new)


def test_initial_state_matches_legacy_class_attributes():
    """初值照抄 ``MMA10:3840-3845``：``λ=zeros``、``μ=μ0=0.01``、``n=0``。"""
    s = ref_al.al_initial_state(n_elem=4)
    assert np.array_equal(s.al_lambda, np.zeros(4))
    assert s.al_mu == MU0
    assert s.al_n_call == 0


def test_production_params_match_legacy():
    """``PRODUCTION_AL_PARAMS`` 照抄 ``:3841-3844``。"""
    assert ref_al.PRODUCTION_AL_PARAMS.mu0 == MU0
    assert ref_al.PRODUCTION_AL_PARAMS.mu_max == MU_MAX
    assert ref_al.PRODUCTION_AL_PARAMS.mu_update_step == STEP


# =============================================================================
# 来源 B —— 与 legacy 逐位（照抄 ``_v2`` 的 AL 段）
# =============================================================================

def _legacy_al_step(al_lambda, al_mu, al_n_call, al_lambda_new, stress_loss_al):
    """照抄 ``optimizer_func_warper_AL_v2.__call__`` 的 AL 段
    （``MMA10:4025-4032`` + ``:4056-4058``；注释掉的两行不抄 ——
    它们引用不存在的 ``AL_stress_k_penalty_update``，见模块 docstring 第二节）。
    """
    al_lambda = al_lambda_new
    al_mu = 0.01 * (1 + al_n_call / 50) ** 2
    al_mu = min(al_mu, 1e3)
    if stress_loss_al > 0:
        al_n_call += 1
    return al_lambda, al_mu, al_n_call


def test_matches_legacy_v2_bitwise():
    """与 legacy ``_v2`` 的 AL 段**逐位**一致（``np.array_equal`` 级）。

    随机扫 ``(λ, μ, n, stress)`` 的组合，覆盖：计数递增/不递增、clip 封顶/
    未封顶、``λ_new`` 非零（生产中恒 0，此处为对抗性取值）。
    """
    rng = np.random.default_rng(42)
    n_calls = [0, 1, 50, 51, 5000, 15761, 20000]
    stresses = [0.0, -1e-8, 1e-12, 1.0, 1e3]
    for trial in range(60):
        n = int(rng.choice(n_calls))
        stress = float(rng.choice(stresses))
        size = int(rng.integers(1, 9))
        lam = rng.normal(size=size)
        lam_new = rng.normal(size=size)
        mu = float(rng.uniform(0.0, 2e3))     # 旧 μ 任意（公式不读它）

        state = ref_al.ALState(al_lambda=lam, al_mu=mu, al_n_call=n)
        new = ref_al.al_update(lam_new, stress_loss_al=stress, state=state)

        want_lam, want_mu, want_n = _legacy_al_step(
            lam, mu, n, lam_new, stress)
        assert np.array_equal(new.al_lambda, want_lam), f"trial {trial}（λ）"
        assert new.al_mu == want_mu, (
            f"trial {trial}（μ）：n={n} stress={stress}  "
            f"got {new.al_mu!r} want {want_mu!r}")
        assert new.al_n_call == want_n, f"trial {trial}（n）"
