"""内核 2 测试：SIMP 密度投影、导数与反演。

测试三级来源：
  来源 A 解析性质      —— 边界值、单调性、往返一致、导数与差分一致
  来源 B 与 legacy 一致 —— 复刻 MMA10:1118-1155 逐位比对
  来源 C ref vs fast   —— 查表反演相对闭式反演的误差量化

另含 ``approx/trig.py``（tanh 的 Padé 近似）的测试。

运行::

    python -m pytest tests/kernels/test_simp.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from motorfem.kernels.approx.trig import (
    K_MAX_EXACT,
    PADE_4_4_P,
    PADE_4_4_Q,
    tanh_pade,
)
from motorfem.kernels.fast.simp import LEGACY_TABLE_SIZE, TableSIMPProjection
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection

# legacy MMA10:124-128 的电磁侧配置
LEGACY = SIMPParams(
    beta=8.0, thb=0.5, power=3.0, min_density=1e-6, linear_factor=0.0
)


@pytest.fixture(scope="module")
def proj() -> SIMPProjection:
    return SIMPProjection(LEGACY)


# =============================================================================
# 来源 A：解析性质
# =============================================================================

def test_boundary_values(proj: SIMPProjection):
    """两条边界必须精确：ρ=0 → min_density，ρ=1 → 1。

    这两条成立是因为 tanh 投影的分母正是按此选取的
    （见 reference/simp.py 的推导）。若分母被改动，本条立刻失败。
    """
    got = proj.project(np.array([0.0, 1.0]))
    assert got[0] == pytest.approx(LEGACY.min_density, rel=1e-15)
    assert got[1] == pytest.approx(1.0, rel=1e-15)


def test_projection_is_monotonic(proj: SIMPProjection):
    """投影必须严格单调递增——否则密度场与设计变量的对应关系被破坏。"""
    rho = np.linspace(0.0, 1.0, 5001)
    assert np.all(np.diff(proj.project(rho)) > 0)


def test_projection_stays_in_range(proj: SIMPProjection):
    """对所有 ρ ∈ [0,1]，投影值落在 [min_density, 1]——这是密度下界的作用。"""
    rho = np.linspace(0.0, 1.0, 5001)
    rp = proj.project(rho)
    assert rp.min() >= LEGACY.min_density - 1e-18
    assert rp.max() <= 1.0 + 1e-15


def test_derivative_matches_finite_difference(proj: SIMPProjection):
    """``dρ_p/dρ`` 与中心差分一致。投影处处光滑（tanh + 幂律），无结点问题。"""
    rho = np.linspace(0.01, 0.99, 200)
    eps = 1e-7
    fd = (proj.project(rho + eps) - proj.project(rho - eps)) / (2 * eps)
    ana = proj.derivative(rho)
    rel = np.abs(fd - ana) / np.maximum(np.abs(ana), 1e-12)
    assert rel.max() < 1e-5, f"导数与中心差分不符，最大相对误差 {rel.max():.3e}"


def test_derivative_is_nonnegative(proj: SIMPProjection):
    """导数非负（单调的必然结果）；非负性也保证 MMA 收到的梯度符号正确。"""
    rho = np.linspace(0.0, 1.0, 2001)
    assert np.all(proj.derivative(rho) >= 0)


@pytest.mark.parametrize("linear_factor", [0.0, 0.25, 0.5, 1.0])
def test_invert_is_exact_roundtrip(linear_factor: float):
    """``invert(project(ρ))`` 必须还原 ρ。

    fc = 0 走闭式反演、fc ≠ 0 走 brentq——两条路径都要验，
    因为它们实现完全不同（fc≠0 那条还带逐元素 Python 循环）。
    """
    params = SIMPParams(**{**LEGACY.__dict__, "linear_factor": linear_factor})
    proj = SIMPProjection(params)
    rho = np.array([0.0, 0.01, 0.2, 0.35, 0.5, 0.65, 0.8, 0.99, 1.0])
    got = proj.invert(proj.project(rho))
    assert np.allclose(got, rho, rtol=1e-10, atol=1e-12), (
        f"fc={linear_factor} 往返不一致：最大偏差 {np.abs(got - rho).max():.3e}"
    )


def test_invert_rejects_out_of_range(proj: SIMPProjection):
    """越界输入必须报错，**不静默裁剪**——越界通常意味着上游算错了。"""
    with pytest.raises(ValueError, match="越界"):
        proj.invert(np.array([1.5]))
    with pytest.raises(ValueError, match="越界"):
        proj.invert(np.array([-1e-3]))


def test_closed_form_inverse_matches_derivation(proj: SIMPProjection):
    """闭式反演与手写公式逐位一致（防止实现漂移）。

    手写式（即 legacy 注释掉的 ``MMA10:1134-1137``）::

        ρ = thb + arctanh(ρ_c·A − T)/β
    """
    rho = np.linspace(0.02, 0.98, 97)
    rp = proj.project(rho)
    rho_c = ((rp - LEGACY.min_density) / (1 - LEGACY.min_density)) ** (1 / LEGACY.power)
    T = np.tanh(LEGACY.beta * LEGACY.thb)
    A = np.tanh(LEGACY.beta * (1 - LEGACY.thb)) + T
    manual = LEGACY.thb + np.arctanh(rho_c * A - T) / LEGACY.beta
    assert np.allclose(proj.invert(rp), manual, rtol=1e-13, atol=1e-15)


# =============================================================================
# 来源 B：与 legacy 逐位一致
# =============================================================================

def test_matches_legacy_forward_and_derivative(proj: SIMPProjection):
    """复刻 legacy ``MMA10:1118-1155`` 的 SIMP_func / SIMP_dfunc 逐位比对。

    这是「保真搬移」的验证：把 legacy 原始代码片段照抄进测试，
    确认重构后的实现产出完全相同的数值。
    注意 legacy 的导数写法含 ``/ sinh(β)``，本实现改用了等价但无除零风险的
    形式；两者在 β=8 下应当一致到浮点精度。
    """
    SIMP_beta, SIMP_thb, p_SIMP = 8.0, 0.5, 3.0
    min_SIMP, SIMP_linear_factor = 1e-6, 0.0

    def legacy_SIMP_func(rho):
        rho_1 = (np.tanh(SIMP_beta * (rho - SIMP_thb)) + np.tanh(SIMP_beta * SIMP_thb)
                 ) / (np.tanh(SIMP_beta * (1 - SIMP_thb)) + np.tanh(SIMP_beta * SIMP_thb))
        rho_2 = rho
        rho_c = rho_1 * (1 - SIMP_linear_factor) + rho_2 * SIMP_linear_factor
        return min_SIMP + (1 - min_SIMP) * rho_c ** p_SIMP

    def legacy_SIMP_dfunc(rho):
        rho_1 = (np.tanh(SIMP_beta * (rho - SIMP_thb)) + np.tanh(SIMP_beta * SIMP_thb)
                 ) / (np.tanh(SIMP_beta * (1 - SIMP_thb)) + np.tanh(SIMP_beta * SIMP_thb))
        drho_1 = (SIMP_beta * np.cosh(SIMP_beta * SIMP_thb)
                  * np.cosh(SIMP_beta - SIMP_beta * SIMP_thb)
                  / np.sinh(SIMP_beta) / np.cosh(SIMP_beta * (rho - SIMP_thb)) ** 2)
        rho_2 = rho
        drho_2 = 1
        rho_c = rho_1 * (1 - SIMP_linear_factor) + rho_2 * SIMP_linear_factor
        drho_c = drho_1 * (1 - SIMP_linear_factor) + drho_2 * SIMP_linear_factor
        return (1 - min_SIMP) * rho_c ** (p_SIMP - 1) * p_SIMP * drho_c

    rho = np.linspace(0.0, 1.0, 501)
    assert np.allclose(proj.project(rho), legacy_SIMP_func(rho), rtol=1e-14, atol=1e-18)
    assert np.allclose(proj.derivative(rho), legacy_SIMP_dfunc(rho), rtol=1e-12, atol=1e-18)


# =============================================================================
# 来源 C：查表反演 vs 闭式反演
# =============================================================================

def test_table_inverse_error_is_bounded_and_quantified(proj: SIMPProjection):
    """量化 legacy 查表反演的误差，并断言其界。

    这是**有意的近似**（legacy 把闭式解注释掉了），不是 bug。本测试的作用是
    把误差固化下来，这样将来若要改用闭式反演，能立刻看到收益有多大。
    实测最大绝对误差约 4e-4（在 ρ 上），见 fast/simp.py 的说明。
    """
    table = TableSIMPProjection(LEGACY)
    rho = np.linspace(0.0, 1.0, 2001)
    rp = proj.project(rho)

    ref = proj.invert(rp)              # 闭式
    tbl = table.invert(rp)             # 查表
    err = np.abs(tbl - ref)
    abs_err = err.max()
    assert abs_err < 1e-3, f"查表反演误差 {abs_err:.3e} 超出预期量级"
    assert np.allclose(tbl, ref, atol=1e-3)


def test_table_size_matches_legacy(proj: SIMPProjection):
    """表的分辨率必须与 legacy 一致（1000 点），否则 parity 会漂移。"""
    table = TableSIMPProjection(LEGACY)
    assert table.table_size == LEGACY_TABLE_SIZE == 1000


def test_table_forward_is_identical_to_reference(proj: SIMPProjection):
    """查表版的前向与导数必须与 reference **完全相同**——它们继承自同一实现，
    只有 invert 被覆写。若哪天有人复制了一份，本条能查出来。"""
    table = TableSIMPProjection(LEGACY)
    rho = np.linspace(0.0, 1.0, 501)
    assert np.array_equal(table.project(rho), proj.project(rho))
    assert np.array_equal(table.derivative(rho), proj.derivative(rho))


# =============================================================================
# approx/trig.py：tanh 的 Padé 近似
# =============================================================================

def test_tanh_pade_accuracy_on_derivation_domain():
    """在推导域 [-4,4] 上，Padé [4/4] 的最大绝对误差必须 < 4e-8。

    实测 3.75e-8。这个界是**从解析系数推导出来的**，不是拟合出来的——
    若系数被改动，本条立刻失败。
    """
    x = np.linspace(-K_MAX_EXACT, K_MAX_EXACT, 20001)
    err = np.abs(tanh_pade(x) - np.tanh(x)).max()
    assert err < 4e-8, f"Padé 误差 {err:.3e} 超出预期（应为 3.75e-8 量级）"


def test_tanh_pade_saturates_outside_domain():
    """|x| > 4 时输入被夹取，输出不会跑飞也不会超过 1。

    ``clip`` 是逐元素 min/max，不引入 Python 分支，不破坏「无分支」。
    """
    x = np.array([-1e6, -5.0, -4.0, 0.0, 4.0, 5.0, 1e6])
    y = tanh_pade(x)
    assert np.all(np.abs(y) <= 1.0 + 1e-12), f"输出越界: {y}"
    assert np.all(np.isfinite(y))

    # 夹取行为必须**逐位**等价于「先夹到 ±4 再求值」——这是 clip 的定义。
    # 注意不要拿 np.tanh(4) 来比：Padé 在域边界自身有 3.75e-8 的误差，
    # 比 clamp 的语义误差大得多，混在一起会写出过紧的容差。
    assert y[1] == tanh_pade(np.array([-4.0]))[0], "负向夹取应等价于在 -4 处求值"
    assert y[5] == tanh_pade(np.array([4.0]))[0], "正向夹取应等价于在 4 处求值"
    assert y[0] == y[1] and y[6] == y[5], "夹取区外的点应给出同一值"

    # 精度：夹取区外的真值误差上限 = 1 − tanh(4) ≈ 6.7e-4
    assert abs(y[5] - np.tanh(5.0)) < 1e-3


def test_tanh_pade_is_odd_and_zero_at_origin():
    """Padé 有理式只含奇次项，故是奇函数且过原点——与 tanh 的对称性一致。"""
    x = np.linspace(0.0, 4.0, 301)
    assert np.allclose(tanh_pade(-x), -tanh_pade(x), rtol=1e-15, atol=0)
    assert tanh_pade(np.array([0.0]))[0] == 0.0


def test_tanh_pade_coefficients_match_derivation():
    """系数必须是解析推导给出的精确有理数（回归守卫）。

    这条防止有人"顺手优化"成数值拟合的系数——那样精度会失去可控性。
    """
    from fractions import Fraction
    assert PADE_4_4_P == (Fraction(1), Fraction(7, 51), Fraction(1, 255),
                          Fraction(2, 69615), Fraction(1, 34459425))
    assert PADE_4_4_Q == (Fraction(1), Fraction(8, 17), Fraction(7, 255),
                          Fraction(4, 9945), Fraction(1, 765765))


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Padé 的加速主张只在低功耗态成立：记录值（np.tanh 66 ns/元素）是笔记本**离电**"
        "被限到小核时测的；接电复测（2026-09-22）np.tanh 为 4.06 ns/元素、Padé 仍"
        "20~24 ns/元素 ⇒ Padé 反而**慢约 5×**。本层「已确认前提失效、待决策」，见 "
        "approx/trig.py 的 warning 与 docs/重构设计文档.md §11 疑点⑥。"
        "离电跑这条会 XPASS —— 那正是记录值成立的条件。"
    ),
)
def test_tanh_pade_is_faster_than_numpy_tanh():
    """性能回归守卫：Padé 在小核/低功耗态下应快于 ``np.tanh``。

    **这条测试的前提依赖电源状态**，所以标了 ``xfail``（离电时 XPASS）：

    - 离电 / 小核：``np.tanh`` 的 SIMD 路径退化到标量量级（66 ns/元素），
      Padé 只做数组四则运算、退化较小 ⇒ Padé 快 2~3 倍，本测试通过。
    - 接电 / 大核：``np.tanh`` 4.06 ns/元素（SIMD 生效），Padé 20~24 ns/元素
      ⇒ Padé 慢约 5×，本测试失败。

    也就是说**它在「机器快」的时候失败、在「机器慢」的时候通过** —— 作为性能守卫
    它是失效的。真正的判据不是本测试，而是「这一层还要不要」这个决策（见 §11 疑点⑥）。
    """
    import time
    n = 20000
    x = np.linspace(-4, 4, n)

    def bench(fn, reps=200):
        fn()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return time.perf_counter() - t

    t_np = bench(lambda: np.tanh(x))
    t_pade = bench(lambda: tanh_pade(x))
    assert t_pade < t_np, (
        f"Padé 未快于 np.tanh（{t_pade*1e3:.1f} ms vs {t_np*1e3:.1f} ms）；"
        "接电时这是预期结果，见本测试 docstring 与 §11 疑点⑥"
    )
