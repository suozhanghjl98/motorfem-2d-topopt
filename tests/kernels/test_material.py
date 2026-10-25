"""内核 3 测试：非线性铁磁材料模型。

测试三级来源：
  来源 A 解析性质      —— ν 的精确标度律（代数恒等式）、ρ₁=1 退化、void 极限、
                          偏导数与有限差分一致
  来源 B 与 legacy 一致 —— 复刻 MMA10:924-936, 1183-1227 逐位比对
  来源 C ref vs fast   —— 缓存实现必须与纯函数实现**逐位**相同

运行::

    python -m pytest tests/kernels/test_material.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motorfem.kernels.fast.material import CachedMaterial, DEFAULT_MUR_LINEAR
from motorfem.kernels.reference import material as ref_mat
from motorfem.kernels.reference.bh_curve import BHCurve
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection

MU0 = 4e-7 * np.pi

# 数据文件路径**相对测试文件定位**，不依赖当前工作目录
_REPO_ROOT = Path(__file__).resolve().parents[2]

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


@pytest.fixture(scope="module")
def simp() -> SIMPProjection:
    return SIMPProjection(SIMPParams())


@pytest.fixture(scope="module")
def mat(curve, simp) -> CachedMaterial:
    return CachedMaterial(curve, simp, mu0=MU0)


# =============================================================================
# 来源 A：解析性质
# =============================================================================

def test_rho1_unity_reduces_to_bulk_iron(curve, simp):
    """ρ₁ = 1 时，带密度的模型必须**逐位**退化为纯铁芯模型。

    这是一个精确恒等：``ν(B,1) = H(B/1)/B = H(B)/B``，两条代码路径应当
    给出完全相同的浮点数。任何多余/缺失的中间运算都会破坏这一点，
    所以用 ``np.array_equal`` 而非 allclose。

    ★ 2026-10-17（D13 真空项）后 ``μ`` 多了一项 ``μ₀(1−ρ₁)``，ρ₁=1 时
    该项**精确为 0**（``μ₀*0.0``），但浮点上 ``B/H + 0.0`` 与 ``B/H`` 逐位
    相同 ⇒ **μ 仍逐位退化**。``∂ν/∂B`` 在新式下是 ``−(dμ/dB)/μ²``，与
    纯铁芯的 ``H'/B − H/B²`` 代数恒等但不逐位（结合顺序不同），故对导数
    用 rel 容差；**导数与纯铁芯解析式的一致性另由
    ``test_dnu_dB_matches_finite_difference`` 独立守住**。
    """
    B = np.linspace(0.05, 3.0, 200)
    ones = np.ones_like(B)

    mu_pure, dnu_pure = ref_mat.nonlinear_material(B, curve)
    mu_topo, dnu_topo = ref_mat.nonlinear_material_topo_opti(B, ones, curve, simp)

    assert np.array_equal(mu_pure, mu_topo), "ρ₁=1 时 μ 未退化为纯铁芯"
    # B=0.05 附近 ∂ν/∂B 本身≈0（纯式给精确 0），相对判据在那里无意义 ⇒ 用绝对判据
    scale = np.max(np.abs(dnu_pure))
    assert np.max(np.abs(dnu_topo - dnu_pure)) <= 1e-9 * scale, (
        f"ρ₁=1 时 ∂ν/∂B 未退化为纯铁芯，绝对差 "
        f"{np.max(np.abs(dnu_topo - dnu_pure)):.3e}（尺度 {scale:.3e}）"
    )


def test_mu_is_volume_average_of_iron_and_vacuum(curve, simp):
    """μ 的精确标度律（D13 后的正确形式）。

    推导（代数恒等式，非近似）::

        μ(B, ρ₁) = B/H(B/ρ₁) + μ₀(1−ρ₁)
                 = ρ₁·μ_iron(B/ρ₁) + (1−ρ₁)·μ₀

    即「铁的一支按 ρ₁ 体积加权 + 真空按 (1−ρ₁) 加权」。这是体积平均的
    直接推论；legacy 的 ``ν·ρ₁ = ν_iron`` 是它的 ``μ₀`` 项缺失版本。

    ⚠ 这条替换了 2026-09-22 的 ``test_reluctivity_scaling_law``——那条断言
    的正是「μ₀ 项缺失」，D13 补上后它必然失败。
    """
    rng = np.random.default_rng(20260922)
    rho = rng.uniform(0.15, 0.95, 60)          # 避开 ρ=0/1 的裁剪端点
    rho1 = simp.project(rho)
    B = rng.uniform(0.05, 2.0, 60)

    mu_topo, _ = ref_mat.nonlinear_material_topo_opti(B, rho, curve, simp, MU0)

    x = B / rho1
    mu_iron = x / curve.H(x)                    # 纯铁芯在 B=x 处的 μ
    mu_expect = rho1 * mu_iron + (1.0 - rho1) * MU0
    assert np.allclose(mu_topo, mu_expect, rtol=1e-13, atol=0), (
        "μ 应等于 ρ₁·μ_iron(B/ρ₁) + (1−ρ₁)·μ₀"
    )
    # 物理下限：任何介质都不比真空更难磁化
    assert np.all(mu_topo >= MU0), "存在 μ < μ₀ 的单元（违反物理下限）"


def test_void_limit_gives_vacuum_permeability(curve, simp):
    """ρ₁ → 0（void）时 μ → μ₀（真空），即 ν → 1/μ₀。

    ★ 2026-10-17（D13）改判：legacy 的 ``μ→0``（比真空还难磁化）是
    **物理非法**的——void 里就是空气，空气的磁导率是 μ₀。旧测试
    ``test_void_limit_gives_near_zero_permeability`` 断言的正是那个非法极限。

    磁障能起作用的机制不依赖「μ→0」：磁障是低密度区，μ≈μ₀，与铁芯的
    2000μ₀ 相比已足够低，磁通照样绕开。
    """
    rho_tiny = np.array([1e-6, 1e-4, 1e-3])
    B = np.array([0.05, 0.05, 0.05])
    mu, _ = ref_mat.nonlinear_material_topo_opti(B, rho_tiny, curve, simp, MU0)
    assert np.allclose(mu, MU0, rtol=1e-12), f"void 的 μ 应为 μ₀，实得 {mu}"
    # 单调：ρ 越大 μ 越大。⚠ ρ ≤ 0.01 时 ρ₁ 全部钉在 min_density=1e-6
    #   （tanh 投影 thb=0.5 + p=3 的塌陷区）⇒ μ 精确等于 μ₀、差分为 0，
    #   必须跨出塌陷区才能看到单调性。
    rho_more = np.array([1e-6, 0.3, 0.4, 0.6, 0.8])
    B4 = np.full_like(rho_more, 0.05)
    mu_more, _ = ref_mat.nonlinear_material_topo_opti(B4, rho_more, curve, simp, MU0)
    assert np.all(np.diff(mu_more) > 0), f"μ 对 ρ 不单调：{mu_more}"


def test_dnu_dB_matches_finite_difference(curve, simp):
    """``∂ν/∂B`` 与 ν(B) 的中心差分一致。

    这是 ∂ν/∂B 的**独立外部判据**。它进入 Newton 的 Jacobian，
    写错会让迭代不收敛或收敛到错解。
    """
    rho = np.array([0.35, 0.6, 0.85])
    B = np.linspace(0.1, 2.5, 40)
    Bg, rhog = np.meshgrid(B, rho, indexing="ij")
    Bf, rhof = Bg.ravel(), rhog.ravel()

    eps = 1e-7
    nu_p = 1.0 / ref_mat.nonlinear_material_topo_opti(Bf + eps, rhof, curve, simp)[0]
    nu_m = 1.0 / ref_mat.nonlinear_material_topo_opti(Bf - eps, rhof, curve, simp)[0]
    fd = (nu_p - nu_m) / (2 * eps)

    _, ana = ref_mat.nonlinear_material_topo_opti(Bf, rhof, curve, simp)
    # 在深饱和区 ∂ν/∂B 趋于 0，用相对判据会放大噪声，故用混合判据
    assert np.allclose(fd, ana, rtol=1e-4, atol=1e-3), (
        f"∂ν/∂B 与中心差分不符，最大偏差 {np.abs(fd - ana).max():.3e}"
    )


def test_dnu_drho_matches_finite_difference(curve, simp):
    """``∂ν/∂ρ`` 与 ν(ρ) 的中心差分一致。

    ⚠ 取 ρ 时须避开 [0,1] 的裁剪端点——裁剪会引入拐点使差分失真。
    ⚠ **还要避开 ρ₁ 塌陷区**（``ρ₁ → min_density``）：那里 ``dμ/dρ`` 是
    「两个 ~1e12 量级项的差」，FD 本身已无分辨力（实测 code 与 FD 中位数
    一致到 8e-10，尾部却因 FD 噪声差到 O(1)）。判定塌陷区用
    ``simp.project(rho) > 100*min_density``。
    """
    B = np.array([0.3, 0.8, 1.5, 2.0])
    rho = np.linspace(0.25, 0.75, 26)
    Bg, rhog = np.meshgrid(B, rho, indexing="ij")
    Bf, rhof = Bg.ravel(), rhog.ravel()

    eps = 1e-7
    nu_p = 1.0 / ref_mat.nonlinear_material_topo_opti(Bf, rhof + eps, curve, simp)[0]
    nu_m = 1.0 / ref_mat.nonlinear_material_topo_opti(Bf, rhof - eps, curve, simp)[0]
    fd = (nu_p - nu_m) / (2 * eps)

    ana = ref_mat.nonlinear_material_dnu_drho(Bf, rhof, curve, simp)
    ok = simp.project(rhof) > 100.0 * simp.params.min_density
    assert ok.sum() >= len(rhof) // 2, "塌陷区外可用的点太少，本测试失去意义"
    scale = np.max(np.abs(fd[ok]))
    assert np.max(np.abs(fd[ok] - ana[ok])) <= 1e-5 * scale, (
        f"∂ν/∂ρ 与中心差分不符，最大偏差 {np.abs(fd[ok] - ana[ok]).max():.3e}"
        f"（尺度 {scale:.3e}）"
    )
    # 防退化：塌陷区确实存在，说明上面那个掩码不是空条件
    assert (~ok).sum() > 0


def test_dnu_drho_includes_vacuum_term(curve, simp):
    """``∂ν/∂ρ`` 必须含真空项贡献 ``−μ₀·dρ₁/dρ``（D13 后新增）。

    推导（见 ``nonlinear_material_dnu_drho`` docstring）::

        dμ/dρ = [ B²·H'(x)/(ρ₁²·H(x)²) − μ₀ ] · dρ₁/dρ

    legacy 的 ``−H'(x)/ρ₁²·dρ₁/dρ`` 只有方括号里的第一项。本测试用
    ``ν = 1/μ`` 的中心差分独立确认「第二项（−μ₀）确实在场」：去掉它
    后在**正常密度区**就会差到 O(1) 量级。
    """
    B = np.array([0.3, 0.8, 1.5])
    rho = np.array([0.6, 0.7, 0.8])
    eps = 1e-7
    nu_p = 1.0 / ref_mat.nonlinear_material_topo_opti(B, rho + eps, curve, simp)[0]
    nu_m = 1.0 / ref_mat.nonlinear_material_topo_opti(B, rho - eps, curve, simp)[0]
    fd = (nu_p - nu_m) / (2 * eps)

    rho1 = simp.project(rho)
    x = B / rho1
    H = curve.H(x)
    mu = B / H + MU0 * (1.0 - rho1)
    drho1 = simp.derivative(rho)
    without_vac = -(-B ** 2 * curve.dHdB(x) / (rho1 ** 2 * H ** 2)) * drho1 / mu ** 2
    got = ref_mat.nonlinear_material_dnu_drho(B, rho, curve, simp)

    assert np.allclose(got, fd, rtol=1e-6, atol=0), "∂ν/∂ρ 与 FD 不符"
    # 少了真空项就无法复现 FD（防「真空项被误删而测试仍通过」）
    assert not np.allclose(without_vac, fd, rtol=1e-3, atol=0), (
        "去掉真空项仍与 FD 一致 ⇒ 本测试没有分辨力"
    )


def test_dnu_drho_is_negative(curve, simp):
    """∂ν/∂ρ < 0：提高材料密度应当降低磁阻率（增大磁导率）。

    符号错误的直接后果是 MMA 收到反号的梯度、优化朝反方向走。
    """
    B = np.linspace(0.1, 2.5, 50)
    rho = np.full_like(B, 0.5)
    assert np.all(ref_mat.nonlinear_material_dnu_drho(B, rho, curve, simp) < 0)


def test_linear_material_is_exact(curve, simp):
    """线性材料：``μ = μ₀·μ_r·ρ₁``，且 ∂ν/∂B ≡ 0。

    这两条都是闭式，可逐位校验。
    """
    rho = np.linspace(0.1, 1.0, 10)
    rho1 = simp.project(rho)
    B = np.linspace(0.1, 2.0, 10)

    mu, dnu_dB = ref_mat.linear_material_topo_opti(B, rho, MU0, 2000.0, simp)
    assert np.array_equal(mu, MU0 * 2000.0 * rho1)
    assert np.array_equal(dnu_dB, np.zeros_like(B))

    dnu_drho = ref_mat.linear_material_dnu_drho(B, rho, MU0, 2000.0, simp)
    assert np.array_equal(dnu_drho, -1.0 / (MU0 * 2000.0) / rho1 ** 2 * simp.derivative(rho))


def test_energy_derivative_is_H(curve):
    """``dW_m/dB ≡ H(B)``（微积分基本定理，经材料层的包装后仍成立）。"""
    B = np.linspace(0.05, 3.0, 100)
    _, dWm = ref_mat.nonlinear_energy(B, curve)
    assert np.array_equal(dWm, curve.H(B))


# =============================================================================
# 来源 B：与 legacy 逐位一致
# =============================================================================

def test_matches_legacy_implementation(curve, simp):
    """复刻 legacy ``MMA10:924-936, 1183-1227`` 逐位比对。

    这是保真搬移验证：把 legacy 原始代码照抄进测试（含其裁剪与算式顺序），
    确认重构后的实现产出完全相同的数值。
    """
    from scipy import interpolate as sp_interp

    # --- 重建 legacy 的插值器（照抄 MMA10:910-921） ---
    import pandas as pd
    data = pd.read_csv(_csv_path()).to_numpy()
    Ht, Bt = data[:, 0], data[:, 1]
    for H_ext in np.linspace(1e6, 4e6, 5):
        Ht = np.append(Ht, Ht[-1] + H_ext)
        Bt = np.append(Bt, Bt[-1] + H_ext * MU0)
    Ht = np.concatenate([-Ht[::-1], Ht[1:]])
    Bt = np.concatenate([-Bt[::-1], Bt[1:]])
    H_interp = sp_interp.PchipInterpolator(Bt, Ht, extrapolate=True)

    SIMP_beta, SIMP_thb, p_SIMP = 8.0, 0.5, 3.0
    min_SIMP, SIMP_linear_factor = 1e-6, 0.0

    def legacy_SIMP_func(rho):
        rho_1 = (np.tanh(SIMP_beta * (rho - SIMP_thb)) + np.tanh(SIMP_beta * SIMP_thb)
                 ) / (np.tanh(SIMP_beta * (1 - SIMP_thb)) + np.tanh(SIMP_beta * SIMP_thb))
        rho_c = rho_1 * (1 - SIMP_linear_factor) + rho * SIMP_linear_factor
        return min_SIMP + (1 - min_SIMP) * rho_c ** p_SIMP

    def legacy_nonlinear_material(_B):
        _B = np.clip(_B, a_min=1e-20, a_max=np.inf)
        mu = _B / H_interp(_B)
        dgamma = H_interp(_B, 1) / _B - H_interp(_B) / _B ** 2
        return mu, dgamma

    def legacy_topo(_B, _rho):
        _B = np.clip(_B, a_min=1e-20, a_max=np.inf)
        _rho = np.clip(_rho, a_min=0, a_max=1)
        _rho1 = legacy_SIMP_func(_rho)
        mu = _B / H_interp(_B / _rho1)
        dgamma = (H_interp(_B / _rho1, 1) / _B / _rho1
                  - H_interp(_B / _rho1) / _B ** 2)
        return mu, dgamma

    def legacy_SIMP_dfunc(rho):
        rho_1 = (np.tanh(SIMP_beta * (rho - SIMP_thb)) + np.tanh(SIMP_beta * SIMP_thb)
                 ) / (np.tanh(SIMP_beta * (1 - SIMP_thb)) + np.tanh(SIMP_beta * SIMP_thb))
        drho_1 = (SIMP_beta * np.cosh(SIMP_beta * SIMP_thb)
                  * np.cosh(SIMP_beta - SIMP_beta * SIMP_thb)
                  / np.sinh(SIMP_beta) / np.cosh(SIMP_beta * (rho - SIMP_thb)) ** 2)
        rho_c = rho_1 * (1 - SIMP_linear_factor) + rho * SIMP_linear_factor
        drho_c = drho_1 * (1 - SIMP_linear_factor) + 1 * SIMP_linear_factor
        return (1 - min_SIMP) * rho_c ** (p_SIMP - 1) * p_SIMP * drho_c

    def legacy_dgamma_p_drho(_B, _rho):
        _B = np.clip(_B, a_min=1e-20, a_max=np.inf)
        _rho = np.clip(_rho, a_min=0, a_max=1)
        _rho1 = legacy_SIMP_func(_rho)
        return -H_interp(_B / _rho1, 1) / _rho1 ** 2 * legacy_SIMP_dfunc(_rho)

    B = np.linspace(0.02, 2.5, 120)
    rho = np.linspace(0.1, 1.0, 120)

    # 无密度
    mu_l, dnu_l = legacy_nonlinear_material(B)
    mu_r, dnu_r = ref_mat.nonlinear_material(B, curve)
    assert np.array_equal(mu_l, mu_r), "无密度的 μ 与 legacy 不一致"
    assert np.array_equal(dnu_l, dnu_r), "无密度的 ∂ν/∂B 与 legacy 不一致"

    # 带密度：★ 2026-10-17（D13）起**有意偏离** legacy —— 补了体积平均的
    # 真空项 ``μ₀(1−ρ₁)``，并让 ∂ν/∂B、∂ν/∂ρ 同步由 ν=1/μ 求导。
    # 见 ``docs/重构设计文档.md`` §12 D13。因此这里比对的不再是逐位相等，
    # 而是「与 legacy 的差恰好等于真空项」——这既证明偏离是**受控的**，
    # 也保证 legacy 的铁那一支没有被动过。
    mu_l2, dnu_l2 = legacy_topo(B, rho)
    mu_r2, dnu_r2 = ref_mat.nonlinear_material_topo_opti(B, rho, curve, simp)
    rho1 = simp.project(rho)
    # 低密度处 legacy 的 μ 只有 ~1e-12（=ρ₁·μ_iron），纯相对判据会被舍入放大
    # ⇒ 用相对 + 绝对混合判据（绝对项 = 真空项本身的 1e-12，即 float64 ULP 量级）
    vac = MU0 * (1.0 - rho1)
    assert np.allclose(mu_r2 - mu_l2, vac, rtol=1e-9, atol=MU0 * 1e-9), (
        "带密度的 μ 与 legacy 的差应恰为真空项 μ₀(1−ρ₁)"
    )
    # ρ₁ = 1 时真空项为 0 ⇒ 必须逐位回到 legacy
    full = rho1 >= 1.0
    assert full.any()
    assert np.array_equal(mu_l2[full], mu_r2[full]), (
        "ρ₁=1 处的 μ 未逐位回到 legacy"
    )

    # ∂ν/∂ρ：★ D13 后与新 μ 同步，不再是 legacy 原式（见
    # ``nonlinear_material_dnu_drho`` docstring 的推导）。此处只守一条
    # 受控关系：**ρ₁→1 时两者趋于一致**（真空项把 μ 抬高 (1−ρ₁)μ₀，
    # 经 1/μ² 放大后仍留下与 (1−ρ₁) 同阶的相对偏差）。
    # 实测（B∈[0.02,2.5]、ρ∈[0.1,1] 网格）：ρ₁>0.9 时相对差中位 2.4e-03、
    # 最大 2.6e-02 —— 这就是**模型差异的量级本身**，不是浮点噪声。
    # 低密度塌陷区按设计差别更大（legacy 到 −1e11，新式有界），故不在此比。
    dnu_drho_l = legacy_dgamma_p_drho(B, rho)
    dnu_drho_r = ref_mat.nonlinear_material_dnu_drho(B, rho, curve, simp)
    near_full = rho1 > 0.9
    assert near_full.sum() >= 5, "ρ₁≈1 的样本太少，本测试失去意义"
    rel = np.abs(dnu_drho_r[near_full] - dnu_drho_l[near_full]) / np.abs(dnu_drho_l[near_full])
    assert np.median(rel) < 1e-2 and rel.max() < 5e-2, (
        f"ρ₁→1 处 ∂ν/∂ρ 与 legacy 的相对差应随 ρ₁→1 收敛到 O(1e-2)，"
        f"实得 median={np.median(rel):.3e} max={rel.max():.3e}"
    )


# =============================================================================
# 来源 C：ref vs fast（缓存）逐位一致
# =============================================================================

def test_cached_matches_reference_bitexact(mat, curve, simp):
    """缓存实现必须与纯函数实现**逐位**相同。

    缓存本身是纯优化（无近似），所以判据是 ``array_equal`` 而非 allclose。
    若有人"顺手"调整了算式顺序（如把 ``x/B/rho1`` 写成 ``x/(B*rho1)``），
    本条会失败——这正是我们想要的保护。
    """
    rng = np.random.default_rng(7)
    rho = rng.uniform(0.1, 1.0, 500)
    B = rng.uniform(0.02, 2.5, 500)
    state = mat.prepare_density(rho)

    mu_f, dnu_f = mat.nonlinear(B, state)
    mu_r, dnu_r = ref_mat.nonlinear_material_topo_opti(B, rho, curve, simp)
    assert np.array_equal(mu_f, mu_r), "缓存的 μ 与 reference 不逐位相同"
    assert np.array_equal(dnu_f, dnu_r), "缓存的 ∂ν/∂B 与 reference 不逐位相同"

    assert np.array_equal(
        mat.nonlinear_dnu_drho(B, state),
        ref_mat.nonlinear_material_dnu_drho(B, rho, curve, simp),
    )

    assert np.array_equal(
        mat.linear(B, state)[0],
        ref_mat.linear_material_topo_opti(B, rho, MU0, DEFAULT_MUR_LINEAR, simp)[0],
    )
    assert np.array_equal(
        mat.linear_dnu_drho(B, state),
        ref_mat.linear_material_dnu_drho(B, rho, MU0, DEFAULT_MUR_LINEAR, simp),
    )


def test_cache_is_prepared_once_per_density(mat, curve, simp):
    """缓存确实生效：``prepare_density`` 只应被显式调用，不在内层循环里。

    模拟真实的 Newton 内循环：ρ 固定、B 每步变。断言 SIMP 投影只算一次。
    """
    rho = np.linspace(0.2, 0.9, 300)
    n_before = mat.n_prepare
    state = mat.prepare_density(rho)

    rng = np.random.default_rng(1)
    for _ in range(50):                       # 模拟 50 次 Newton 迭代
        B = rng.uniform(0.05, 2.0, 300)
        mu, dnu = mat.nonlinear(B, state)
        assert np.all(np.isfinite(mu))

    assert mat.n_prepare == n_before + 1, (
        f"prepare_density 被调用 {mat.n_prepare - n_before} 次，应恰好 1 次"
    )


def test_verify_mode_catches_stale_state(curve, simp):
    """``verify=True`` 时，把 A 的 state 配 B 的 ρ 必须报错。

    这是对「契约：state 只在其 rho 未变时有效」的可选强制。默认关闭
    （逐次校验会抵消缓存收益），但调试时应当能打开并抓到误用。
    """
    strict = CachedMaterial(curve, simp, mu0=MU0, verify=True)
    state = strict.prepare_density(np.full(10, 0.4))

    # 先用正确形状探一次（应通过）
    strict.nonlinear(np.full(10, 1.0), state)

    # 形状不符 -> 报错
    with pytest.raises(ValueError, match="形状"):
        strict.nonlinear(np.full(7, 1.0), state)


def test_clip_is_applied_here_not_in_curve(curve, simp):
    """裁剪在本层生效：B=0 不得产生 nan。

    这是 kernel 1 的结论——裁剪属于「除法发生的地方」，即本内核。
    B=0 时若不做下界裁剪，``H(B)/B`` 与 ``/B²`` 会给出 nan。
    """
    B = np.array([0.0, 1e-30, 1e-25])
    mu, dnu = ref_mat.nonlinear_material(B, curve)
    assert np.all(np.isfinite(mu)), f"B=0 处 μ 出现非有限值: {mu}"
    assert np.all(np.isfinite(dnu)), f"B=0 处 ∂ν/∂B 出现非有限值: {dnu}"
