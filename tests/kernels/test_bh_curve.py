"""内核 1 测试：B-H 磁化曲线。

测试三级来源：
  来源 A 解析解/解析性质  —— 奇对称、尾段落在 μ₀ 线上、微积分基本定理、
                            PCHIP 不过冲、单调性
  来源 B 与 legacy 一致    —— 复刻 legacy MMA10:910-917 的构建过程逐位比对
  来源 C ref vs fast       —— 不适用（本内核无 fast 变体，见 kernels/README.md）

运行::

    python -m pytest tests/kernels/test_bh_curve.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motorfem.kernels.reference.bh_curve import (
    BHCurve,
    TailExtension,
    extend_bh_table,
    load_bh_table,
)

MU0 = 4e-7 * np.pi

# 数据文件路径**相对测试文件定位**，不依赖当前工作目录——
# 否则从 tests/kernels 或其它目录跑 pytest 时会找不到文件而 skip。
_REPO_ROOT = Path(__file__).resolve().parents[2]        # tests/kernels/x.py -> 仓库根

# 兼容两种目录布局：阶段 0（迁移到 legacy/）执行前后都能跑
CSV_CANDIDATES = [
    _REPO_ROOT / "磁障设计/pytorch/50CS470.csv",         # 阶段 0 之前
    _REPO_ROOT / "legacy/pytorch/50CS470.csv",           # 阶段 0 之后
]


def _csv_path() -> Path:
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 50CS470.csv（请在仓库根目录下运行 pytest）")


@pytest.fixture(scope="module")
def curve() -> BHCurve:
    return BHCurve.from_csv(_csv_path(), mu0=MU0)


# =============================================================================
# 来源 A：解析性质
# =============================================================================

def test_curve_interpolates_measured_points_exactly(curve: BHCurve):
    """PCHIP 是插值而非拟合——在所有实测结点上必须精确穿过。

    这是「数据没被弄错」的最直接判据：任何加载/对称化/插值环节的偏差
    都会让结点值不再精确。
    """
    H_csv, B_csv = load_bh_table(_csv_path())
    H_got = curve.H(B_csv)
    # 相对误差：H 跨越 0 ~ 8e5，用相对判据
    scale = np.maximum(np.abs(H_csv), 1.0)
    rel = np.abs(H_got - H_csv) / scale
    assert rel.max() < 1e-12, f"实测结点未被精确穿过，最大相对误差 {rel.max():.3e}"


def test_curve_is_odd_symmetric(curve: BHCurve):
    """磁化曲线是奇函数：H(-B) = -H(B)，W_m(-B) = W_m(B)。

    容差选取依据（**来自实测，不是随手放宽**）：
      - H：实测相对偏差 3.2e-16，取 rtol=1e-14 即可。
      - W_m：偏差被量化在积分常数的 float64 间距（ULP）上，实测是 ULP 的
        1~7 倍。故直接用 ``curve.Wm_precision_floor``（8 ULP）作容差——
        它是**从数据推导出来的**，不是魔数。
    """
    B = np.linspace(0.05, 3.0, 40)
    assert np.allclose(curve.H(-B), -curve.H(B), rtol=1e-14, atol=1e-9)
    # 能量密度是偶函数（∫₀^{-B} 与 ∫₀^{B} 对称）
    assert np.allclose(
        curve.Wm(-B), curve.Wm(B), rtol=1e-14, atol=curve.Wm_precision_floor
    ), f"W_m 的偶对称性超出精度下限 {curve.Wm_precision_floor:.3e}"


def test_Wm_quantization_is_ulp_of_integration_constant(curve: BHCurve):
    """把「W_m 的偏差量化在积分常数的 ULP 上」这一实测事实固化成回归测试。

    这条同时做了两件事：
      1. 证明 Wm_precision_floor 的**推导依据**成立（偏差确实是 ULP 的整数倍）；
      2. 若将来有人改掉积分常数的计算方式（例如改成从 0 起积分），
         ULP 会骤降、量化关系不再成立，本条会失败并提示复核。
    """
    B = np.linspace(0.05, 3.0, 40)
    diff = np.abs(curve.Wm(-B) - curve.Wm(B))
    ulp = abs(float(np.spacing(curve.integration_constant)))

    nonzero = diff[diff > 0]
    assert nonzero.size > 0, "偏差全为 0，说明积分常数不再是那个大数，需复核"
    ratio = nonzero / ulp
    assert np.allclose(ratio, np.round(ratio), rtol=1e-6), (
        f"偏差并非 ULP 的整数倍：比值样本 {ratio[:5]}"
    )
    assert ratio.max() <= 8.0, f"实测最大偏差达 {ratio.max():.1f} ULP，超出预估上界 8"


def test_tail_lies_on_vacuum_line(curve: BHCurve):
    """尾段每一点的斜率必须精确等于 μ₀。

    这是尾段外推的**定义**，也是最容易搞错的地方——legacy 的步长是**累加**的
    （``H[-1] + H_ext``），若误写成从最后一个实测点起算，本测试会失败。
    """
    H_tab, B_tab = curve.table
    tail = TailExtension()
    n_tail = tail.n_points

    # 正向尾段的最后 n_tail 个点
    positive = B_tab > 0
    H_pos, B_pos = H_tab[positive], B_tab[positive]
    H_tail, B_tail = H_pos[-n_tail:], B_pos[-n_tail:]

    # 与前一点的差分应严格是 μ₀
    H_prev = np.concatenate([[H_pos[-n_tail - 1]], H_tail[:-1]])
    B_prev = np.concatenate([[B_pos[-n_tail - 1]], B_tail[:-1]])
    slope = (B_tail - B_prev) / (H_tail - H_prev)
    assert np.allclose(slope, MU0, rtol=1e-12), f"尾段斜率应为 μ₀，实得 {slope}"

    # 步长应是累加的：第 k 段是 linspace 的第 k 个值
    steps = np.linspace(tail.H_step_lo, tail.H_step_hi, tail.n_points)
    assert np.allclose(H_tail - H_prev, steps, rtol=1e-12)


def test_h_is_strictly_monotonic(curve: BHCurve):
    """H(B) 必须严格单调递增。PCHIP 不过冲是这条得以成立的前提；
    普通三次样条会在拐点附近振荡出非单调段，直接导致 Newton 发散。"""
    B = np.linspace(-4.0, 4.0, 4001)
    H = curve.H(B)
    assert np.all(np.diff(H) > 0), "H(B) 出现非单调段"


def test_dHdB_matches_finite_difference(curve: BHCurve):
    """``dH/dB`` 是 PCHIP 的解析导数，应与中心差分一致。

    这是对导数的**独立外部判据**——导数由 scipy 的解析实现给出，
    我们用数值微分反查它。
    """
    # 取点**避开 PCHIP 的结点**：结点处二阶导跳变，中心差分在那里退化到 O(eps)
    # 精度而非 O(eps²)，会掩盖真实的导数误差。CSV 的 B 结点在 0.1 的整数倍上
    # （0.0, 0.1, 0.2, ...），故取其中点。
    B = np.arange(0.05, 2.0, 0.1)
    eps = 1e-6
    fd = (curve.H(B + eps) - curve.H(B - eps)) / (2 * eps)
    ana = curve.dHdB(B)
    rel = np.abs(fd - ana) / np.abs(ana)
    assert rel.max() < 1e-6, f"dH/dB 与中心差分不符，最大相对误差 {rel.max():.3e}"


def test_Wm_derivative_equals_H_analytically(curve: BHCurve):
    """微积分基本定理，用**解析导数**验证：``dW_m/dB ≡ H``。

    ⚠ 不能用有限差分验这一条。``W_m`` 的绝对精度下限是 1.2e-8（积分常数
    ≈-1.11e8 的浮点间距），而 eps=1e-6 的中心差分分子只有 ~4e-5，
    底噪会被放大到 ~3e-4 的相对误差，得出"实现有错"的错误结论。
    这个坑已经踩过一次，实测记录见 ``BHCurve.Wm`` 的文档。

    解析导数恒等于 H 是**构造上必然**的（原函数的导数就是被积函数），
    所以它本身不构成"正确性"证明；它的价值是**回归守卫**：
    若 Wm 被错误地对另一个插值器求积分（例如只在非负半支上建的那个），
    本条会失败而只比较 Wm 数值的测试查不出来。真正的正确性判据是下面
    与高精度数值积分的对比。
    """
    B = np.arange(0.05, 2.0, 0.1)
    assert np.array_equal(curve.dWmdB(B), curve.H(B)), "dW_m/dB 与 H 不逐位相等"


def test_Wm_matches_high_precision_quadrature(curve: BHCurve):
    """``W_m(B)`` 与高精度数值积分 ``∫₀^B H`` 对比。

    这是 Wm 的**独立外部判据**：用 scipy.integrate.quad 在 PCHIP 的结点处分段
    积分，与解析原函数对比。容差按实测精度下限设定（见 ``BHCurve.Wm``）。

    实测相对偏差：B=0.05 → 1.5e-8，B=1.0 → 2.4e-10，B=3.0 → 3.7e-10。
    """
    from scipy.integrate import quad

    H_tab, B_tab = curve.table
    for b in [0.05, 0.5, 1.0, 2.0, 3.0]:
        # 把 B 区间内的结点喂给 quad 作为分段点，避免在 C¹ 拐点处积分
        breaks = B_tab[(B_tab > 0.0) & (B_tab < b)]
        ref = quad(
            lambda t: float(curve.H(np.array([t]))[0]),
            0.0, b, points=breaks, limit=200,
        )[0]
        got = float(curve.Wm(np.array([b]))[0])
        tol = 1e-7 * max(1.0, abs(ref))
        assert abs(got - ref) < tol, (
            f"W_m({b}) = {got:.10g} 与高精度积分 {ref:.10g} 相差 "
            f"{abs(got - ref):.3e}，超出容差 {tol:.3e}"
        )


def test_deep_saturation_approaches_vacuum(curve: BHCurve):
    """深饱和时微分磁导率应趋近 μ₀。

    这是尾段外推的物理意义所在：材料饱和后行为接近真空。若尾段外推写错
    （例如用了 μr 而不是 μ0），本测试会失败。
    """
    tail = TailExtension()
    H_tab, B_tab = curve.table
    B_tail = B_tab[B_tab > 0][-1]
    # 尾段中后部的微分磁导率
    mu_diff = 1.0 / curve.dHdB(np.array([B_tail * 0.98]))[0]
    assert 0.8 * MU0 < mu_diff < 1.2 * MU0, f"深饱和微分磁导率 {mu_diff:.4e} 未趋近 μ₀={MU0:.4e}"


# =============================================================================
# 来源 B：与 legacy 逐位一致
# =============================================================================

def test_matches_legacy_table_construction(curve: BHCurve):
    """复刻 legacy ``MMA10:910-917`` 的构建过程，断言逐位一致。

    这是「保真搬移」的验证：把 legacy 的原始代码片段照抄到测试里，
    确认重构后的实现产出完全相同的表。任何语义漂移（例如把累加步长
    改成等间距）都会被抓住。

    注意：本测试**故意**照抄 legacy 写法（含其魔法数与累加语义），
    不复用被测实现——否则就是自我印证。
    """
    import pandas as pd

    data = pd.read_csv(_csv_path()).to_numpy()
    H = data[:, 0]
    B = data[:, 1]
    for H_ext in np.linspace(1e6, 4e6, 5):
        H = np.append(H, H[-1] + H_ext)
        B = np.append(B, B[-1] + H_ext * MU0)
    H = np.concatenate([-H[::-1], H[1:]])
    B = np.concatenate([-B[::-1], B[1:]])

    H_got, B_got = curve.table
    assert H_got.shape == H.shape, f"表长不符：{H_got.shape} vs {H.shape}"
    assert np.array_equal(H_got, H), "H 表与 legacy 不一致"
    assert np.array_equal(B_got, B), "B 表与 legacy 不一致"


# =============================================================================
# 输入校验
# =============================================================================

def test_rejects_non_monotonic_table():
    """非单调输入必须立刻报错，而不是静默产出一条不可用的曲线。"""
    H = np.array([0.0, 1.0, 2.0, 1.5, 3.0])       # 第 4 个点回退
    B = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="单调"):
        extend_bh_table(H, B, MU0)


def test_csv_loading_edge_cases(tmp_path):
    """CSV 读入的边界：缺文件必须报错，单列文件必须报错。"""
    # 文件不存在 -> OSError（np.loadtxt 抛 FileNotFoundError）
    with pytest.raises(OSError):
        load_bh_table(tmp_path / "nonexistent.csv")

    # 只有一列 -> ValueError（找不到 B 列）
    one_col = tmp_path / "one_col.csv"
    one_col.write_text("H\n0\n1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="两列"):
        load_bh_table(one_col)
