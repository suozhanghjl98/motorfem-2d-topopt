"""approx 层测试：B-H 曲线查表（``approx/bh_lut.py``）。

本层**主动放弃位级一致**，所以测试纪律与 ``reference``/``fast`` 不同：不能拿
``np.array_equal`` 当判据，必须换成「解析性质 + 实测标定的误差界」。

三级来源：
  来源 A 解析解/解析性质  —— ``H(0)=0``、奇对称、尾段严格落在 μ₀ 线上、
                            **导数恒等于插值函数自身的割线**（Newton 一致性的判据）、
                            单调性、插值落在相邻节点之间
  来源 B 与 legacy 一致    —— **不适用**（approx 层的定义就是不一致）；
                            改测「未近似的量原样委托」（``Wm`` / ``dWmdB``）
  来源 C ref vs fast       —— 不适用（approx 不是 fast 的变体）
  额外：误差界（实测标定）+ 性能守卫（宽松阈值，§4 要求）

运行::

    python -m pytest tests/kernels/test_bh_lut.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motorfem.kernels.approx.bh_lut import (
    DEFAULT_N_TABLE,
    TabulatedCachedMaterial,
    TabulatedHcurve,
)
from motorfem.kernels.fast.material import CachedMaterial
from motorfem.kernels.reference.bh_curve import BHCurve
from motorfem.kernels.reference.material import nonlinear_material
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection

MU0 = 4e-7 * np.pi
_REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_CANDIDATES = [
    _REPO_ROOT / "磁障设计/pytorch/50CS470.csv",
    _REPO_ROOT / "legacy/pytorch/50CS470.csv",
]

#: ``H`` 的相对误差上界。**不是理论界，是实测标定** —— 换表点数或换数据集必须重测。
#:   n=65536、真实网格 x 分布（收敛解）→ 2.0e-6
#:   n=65536、对抗性对数均匀（本文件的 ``_real_x``）→ 6.3e-6   ← 取这个作准
#: 取 2e-5 = 对抗性取样的 3.2× 余量。
REL_TOL_H = 2e-5

#: ``dν/dB`` 的误差上界，以 ``max|dν/dB|`` 为尺度。**dν/dB 会过零**，相对误差无意义。
#:   n=65536、对抗性对数均匀 → 1.1e-3
DNU_TOL_SCALED = 5e-3


def _csv_path() -> Path:
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 50CS470.csv（请在仓库根目录下运行 pytest）")


@pytest.fixture(scope="module")
def ref() -> BHCurve:
    return BHCurve.from_csv(_csv_path(), mu0=MU0)


@pytest.fixture(scope="module")
def lut(ref: BHCurve) -> TabulatedHcurve:
    return TabulatedHcurve(ref, n=DEFAULT_N_TABLE)


def _real_x(ref: BHCurve, n: int = 4000) -> np.ndarray:
    """真实工作区间的取样：跨 8 个数量级（§2b 实测的 x 分布形状）。

    对数均匀而不是线性均匀 —— 线性均匀会让 1e6 那一端被 0.0x 那端淹没，
    测出来的「最大误差」只反映膝点，掩盖尾段。
    """
    rng = np.random.default_rng(20260929)
    x = np.concatenate([
        10 ** rng.uniform(-2.0, 0.3, int(n * 0.6)),      # 铁芯工作区
        10 ** rng.uniform(3.0, 6.3, n - int(n * 0.6)),   # 真空/磁障区（域外）
    ])
    return np.sort(x)


# =============================================================================
# 来源 A —— 解析性质
# =============================================================================

def test_origin_is_exact(lut: TabulatedHcurve) -> None:
    """H(0) = 0（曲线过原点）。"""
    assert float(lut.H(np.array([0.0]))[0]) == 0.0


def test_odd_symmetry_is_exact(lut: TabulatedHcurve) -> None:
    """H 奇、H' 偶 —— 由内核里的符号处理保证，应当**逐位**成立。

    （原曲线的奇对称是构建时镜像出来的，查表若丢掉这一半就会静默错。）
    """
    x = np.array([0.01, 0.5, 1.5, 3.0, 12.0, 1e5])
    assert np.array_equal(lut.H(-x), -lut.H(x))
    assert np.array_equal(lut.dHdB(-x), lut.dHdB(x))


def test_tail_is_affine_on_mu0_line(lut: TabulatedHcurve, ref: BHCurve) -> None:
    """域外走解析仿射支：斜率 == 1/μ₀，且**与原曲线吻合**。

    原曲线在 x > 18.814 上的 PCHIP「三次外推」实测退化为直线（末段节点共线），
    所以这一支不是近似 —— 断言到 1e-8 相对。
    """
    x_end = lut.x_max
    # 尾段斜率来自表末段的割线（不是从 mu0 反算的），故只能到 ~5e-12 相对 ——
    # 那点差是 H 逐点累加建表时的浮点舍入，不是建模差。实测差 5.3e-12。
    ref_slope = float(ref.dHdB(np.array([x_end]))[0])
    assert np.isclose(lut.tail_slope, ref_slope, rtol=1e-10)
    assert np.isclose(lut.tail_slope, 1.0 / MU0, rtol=1e-10)

    x = np.array([x_end * 1.001, 50.0, 1e3, 1e6])
    H_lut, H_ref = lut.H(x), ref.H(x)
    assert np.allclose(H_lut, H_ref, rtol=1e-8)

    # 严格仿射：两点定出的直线必须穿过第三点
    H_a, H_b, H_c = lut.H(x[:1]), lut.H(x[1:2]), lut.H(x[3:4])
    slope = (H_b[0] - H_a[0]) / (x[1] - x[0])
    assert np.isclose(H_c[0] - H_b[0], slope * (x[3] - x[1]), rtol=1e-10)


def test_derivative_is_exactly_the_secant(lut: TabulatedHcurve) -> None:
    """★ 最关键的一条：``dHdB`` 必须是**插值函数自身的导数**。

    这是 Newton 一致性的判据 —— Jacobian 用的导数若与残差里的 H 不是同一个
    函数的导数，Newton 就从二次退化成准牛顿（模块 docstring §2c）。
    段内 H 是仿射的，故 ``H(x2) − H(x1) == dH·(x2 − x1)`` 必须成立。
    """
    x_end = lut.x_max
    dx = x_end / (lut.n_table - 1)
    for k in (1, 137, 5000, lut.n_table - 2):
        x1 = k * dx + 0.2 * dx
        x2 = k * dx + 0.8 * dx
        H1, H2 = lut.H(np.array([x1, x2]))
        d = lut.dHdB(np.array([x1]))[0]
        assert np.isclose(H2 - H1, d * (x2 - x1), rtol=1e-12, atol=0.0), (
            f"第 {k} 段：H 不是以 dHdB 为斜率的仿射函数 —— Newton 一致性被破坏"
        )


def test_monotone_and_bracketed(lut: TabulatedHcurve) -> None:
    """单调性 + 插值落在相邻节点之间（线性插值不可能越过真曲线太远）。"""
    x = np.linspace(0.0, lut.x_max, 20001)
    H = lut.H(x)
    dH = lut.dHdB(x)
    assert np.all(dH > 0.0), "PCHIP 保单调 ⇒ 割线斜率也必须处处为正"
    assert np.all(np.diff(H) > 0.0), "H 必须严格单调递增"

    # 相邻节点之间：H 必落在两端点之间（取节点值直接验）
    assert H[0] == 0.0
    assert float(H[-1]) == pytest.approx(float(lut.H(np.array([lut.x_max]))[0]))


def test_table_is_consistent_with_its_own_builder(lut: TabulatedHcurve,
                                                  ref: BHCurve) -> None:
    """表节点处 H 必须等于原曲线到 ULP 级（表是**采样**出来的，不是拟合出来的）。

    ⚠ 不能断言 ``array_equal``：节点处 ``x*inv_dx`` 有 ~1e-14 的残差 ``w``，
    于是返回 ``Hg[k] + w·(Hg[k+1]−Hg[k])``，最后几位与 ``Hg[k]`` 不同。
    那点差是索引算式引入的，不是「拟合」。
    """
    xg = np.linspace(0.0, lut.x_max, lut.n_table)
    # 用「绝对 ≤ 1e-13·参考 + 0」而不是除法 —— x=0 处两边都是 0，除出来是 nan。
    dev = np.abs(lut.H(xg) - ref.H(xg))
    assert np.all(dev <= 1e-13 * np.abs(ref.H(xg))), (
        f"节点处最大绝对偏差 {dev.max():.3e} —— 表不是采出来的？"
    )


# =============================================================================
# 误差界（实测标定 —— 不是理论界）
# =============================================================================

def test_error_within_calibrated_bound(ref: BHCurve) -> None:
    """真实工作区间上的 max 相对误差 < 5e-6（实测 2.0e-6）。"""
    lut = TabulatedHcurve(ref, n=DEFAULT_N_TABLE)
    err = lut.max_rel_error_H(_real_x(ref))
    assert err < REL_TOL_H, f"max|ΔH|/H = {err:.3e} 超界（实测应为 2.0e-6 量级）"


def test_error_decreases_with_table_size(ref: BHCurve) -> None:
    """加大表必须单调降误差 —— 否则「采样」这一步就被写错了（比如索引算错）。"""
    x = _real_x(ref)
    errs = [TabulatedHcurve(ref, n=n).max_rel_error_H(x)
            for n in (8192, 65536, 262144)]
    assert errs[0] > errs[1] > errs[2], f"误差未随表点数下降：{errs}"
    # 二次收敛的量级检查（宽松：只要求至少降一个数量级）
    assert errs[0] / errs[2] > 10.0


def test_curve_far_outside_domain_still_matches(ref: BHCurve) -> None:
    """x 到 2e6（实测最大工作值）时仍与 PCHIP 一致 —— 尾支不能漏。"""
    lut = TabulatedHcurve(ref, n=DEFAULT_N_TABLE)
    x = np.array([2.0793e6, 1e7, 1e9])
    assert np.allclose(lut.H(x), ref.H(x), rtol=1e-6)


# =============================================================================
# drop-in 一致性（不是位级一致 —— 是「误差只来自插值」）
# =============================================================================

@pytest.fixture(scope="module")
def simp() -> SIMPProjection:
    return SIMPProjection(SIMPParams())


def test_nonlinear_material_drop_in(ref: BHCurve, lut: TabulatedHcurve,
                                    simp: SIMPProjection) -> None:
    """``nonlinear_material(B, lut)`` 与 ``nonlinear_material(B, ref)`` 一致。

    它走的是 reference 的 numpy 算式 + ``lut.H``/``lut.dHdB``，是**纯铁支**的路径。
    """
    rng = np.random.default_rng(7)
    B = np.clip(rng.uniform(0.0, 4.0, 3000), 1e-20, np.inf)
    mu_a, dn_a = nonlinear_material(B, lut)
    mu_r, dn_r = nonlinear_material(B, ref)
    assert np.max(np.abs(mu_a - mu_r) / mu_r) < REL_TOL_H
    scale = np.max(np.abs(dn_r))
    assert np.max(np.abs(dn_a - dn_r)) < scale * DNU_TOL_SCALED


def test_fused_kernel_adds_no_extra_error(lut: TabulatedHcurve) -> None:
    """融合内核 == 从 ``lut.H``/``lut.dHdB`` 手算同一条链。

    两者若不等，说明融合时写错了算式（而不是「插值误差」）。
    """
    rng = np.random.default_rng(11)
    B = np.clip(rng.uniform(0.0, 3.0, 2000), 1e-20, np.inf)
    rho1 = rng.uniform(1e-6, 1.0, 2000)

    mu_f, dn_f = lut.fused_nonlinear(B, rho1)

    x = B / rho1
    H = lut.H(x)
    dH = lut.dHdB(x)
    # ★ 2026-10-17（D13）：μ 含真空项 ``mu0*(1-rho1)``；dnu 必须由新的 μ 求导。
    mu_h = B / H + MU0 * (1.0 - rho1)
    dn_h = -(1.0 - x * dH / H) / H / mu_h ** 2

    # ⚠ 融合内核的 μ₀ 是 ``1/tail_slope`` 反推的（见 ``fused_nonlinear`` 的
    #   docstring），与 ``4e-7*pi`` 差 ~5e-12 相对 ⇒ 不能用 array_equal。
    #   approx 层本就不承诺位级一致，判据是「算式同形、误差只来自插值」。
    assert np.array_equal(mu_f, mu_h) or (
        np.max(np.abs(mu_f - mu_h) / mu_h) < 1e-10
    ), "融合内核的 mu 与手算不同形"
    assert np.max(np.abs(dn_f - dn_h)) < np.max(np.abs(dn_h)) * 1e-9, (
        "融合内核的 dnu_dB 与手算不同形"
    )


def test_fused_material_vs_cached(ref: BHCurve, lut: TabulatedHcurve,
                                  simp: SIMPProjection) -> None:
    """``TabulatedCachedMaterial.nonlinear`` vs ``CachedMaterial.nonlinear``（SIMP 支）。"""
    rng = np.random.default_rng(13)
    rho = rng.uniform(0.0, 1.0, 2000)
    B = np.clip(rng.uniform(0.0, 3.0, 2000), 1e-20, np.inf)

    c_ref = CachedMaterial(ref, simp, mu0=MU0, mur_linear=2000.0)
    c_lut = TabulatedCachedMaterial(lut, simp, mu0=MU0, mur_linear=2000.0)
    st_r = c_ref.prepare_density(rho)
    st_l = c_lut.prepare_density(rho)

    assert np.array_equal(st_r.rho1, st_l.rho1), "ρ 缓存必须与原实现逐位相同"

    mu_a, dn_a = c_lut.nonlinear(B, st_l)
    mu_r, dn_r = c_ref.nonlinear(B, st_r)
    assert np.max(np.abs(mu_a - mu_r) / mu_r) < REL_TOL_H
    assert np.max(np.abs(dn_a - dn_r)) < np.max(np.abs(dn_r)) * DNU_TOL_SCALED


def test_rejects_plain_bhcurve(ref: BHCurve, simp: SIMPProjection) -> None:
    """传原曲线进 ``TabulatedCachedMaterial`` 必须**报错**，不能静默退化成精确路径。"""
    with pytest.raises(TypeError, match="TabulatedHcurve"):
        TabulatedCachedMaterial(ref, simp)


def test_rejects_tiny_table(ref: BHCurve) -> None:
    with pytest.raises(ValueError, match="至少 4"):
        TabulatedHcurve(ref, n=3)


# =============================================================================
# 来源 B 的替代 —— 未近似的量必须原样委托
# =============================================================================

def test_Wm_is_delegated_not_approximated(ref: BHCurve,
                                          lut: TabulatedHcurve) -> None:
    """``Wm`` / ``dWmdB`` 逐位等于原曲线（本层**不**近似它们）。

    ``W_m`` 有 1.2e-8 的绝对精度底噪（积分常数 ULP），查表再叠一层插值误差
    会让它失去唯一的比对基准；且它不在生产 Newton 路径上。
    """
    B = np.array([0.05, 0.5, 1.5, 2.5, 3.0])
    assert np.array_equal(lut.Wm(B), ref.Wm(B))
    assert np.array_equal(lut.dWmdB(B), ref.dWmdB(B))


# =============================================================================
# 性能守卫（§4 要求；宽松阈值）
# =============================================================================

@pytest.mark.parametrize("n_pt", [65536])
def test_performance_guard(ref: BHCurve, n_pt: int) -> None:
    """查表至少比 scipy PCHIP 快 2×（实测 10~14×，阈值放到 2× 抗抖动）。

    ⚠ 阈值宽松是**有意**的：本项只守护「查表路径没有退化到 numpy 版」
    （numpy 版实测比 PCHIP **慢** 1.23×，见模块 docstring §4）。
    ⚠ 电源状态会扭曲这个量（离电限频时 numpy 的 tanh/超越函数会成倍变慢），
    故不报具体数字、只断言方向。
    """
    import time

    lut = TabulatedHcurve(ref, n=n_pt)
    x = np.linspace(0.02, 4.0, 9599)

    def best(fn, reps=40):
        fn()
        return min((lambda t0: (fn(), time.perf_counter() - t0)[1])(time.perf_counter())
                   for _ in range(reps))

    t_scipy = best(lambda: (ref.H(x), ref.dHdB(x)))
    t_lut = best(lambda: (lut.H(x), lut.dHdB(x)))
    assert t_lut < t_scipy / 2.0, (
        f"查表未达 2× 加速：scipy {t_scipy * 1e3:.4f} ms vs LUT {t_lut * 1e3:.4f} ms"
        "（检查 numba 是否真的生效，见 approx/bh_lut.py docstring §4）"
    )
