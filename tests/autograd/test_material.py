"""tests/autograd/test_material.py —— 阶段 6：torch SIMP 投影与材料链

验证：``TorchSIMP.project`` 与 numpy ``SIMPProjection.project`` 逐值一致；
``torch_nu`` 与受信四类混合（``blend_class_values``）一致；autograd 的
∂ν/∂ρ == legacy ``nonlinear_material_dgamma_p_drho``（is_simp 区，1e-12 级）。
"""
import glob
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motorfem.autograd.material import TorchSIMP, torch_nu
from motorfem.autograd.pchip import TorchPCHIP
from motorfem.fem.em_worker import legacy_dgamma_p_drho
from motorfem.kernels.fast import newton as fast_nt
from motorfem.kernels.fast.material import CachedMaterial
from motorfem.kernels.reference.bh_curve import BHCurve
from motorfem.kernels.reference.newton import MaterialMasks, blend_class_values
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def simp():
    return SIMPProjection(SIMPParams())


@pytest.fixture(scope="module")
def curve():
    csvs = list(Path(_REPO_ROOT).rglob("50CS470.csv"))
    if not csvs:
        pytest.skip("找不到 50CS470.csv")
    return BHCurve.from_csv(str(csvs[0]))


def _masks(n, simp_frac=0.5):
    is_vacuum = np.zeros(n)
    is_simp = np.zeros(n)
    is_non_simp = np.zeros(n)
    is_linear_iron = np.zeros(n)
    is_simp[: int(n * simp_frac)] = 1.0
    is_vacuum[int(n * simp_frac):] = 1.0
    return MaterialMasks(is_vacuum=is_vacuum, is_simp=is_simp,
                         is_non_simp=is_non_simp, is_linear_iron=is_linear_iron)


def test_torch_simp_project_matches_numpy(simp):
    rho = np.linspace(-0.2, 1.2, 2000)
    ts = TorchSIMP(simp)
    assert np.abs(ts.project(torch.tensor(rho)).numpy() - simp.project(rho)).max() < 1e-14


def test_torch_simp_derivative_via_autograd_matches_numpy(simp):
    rho = np.linspace(0.0, 1.0, 500)
    ts = TorchSIMP(simp)
    rho_t = torch.tensor(rho, requires_grad=True, dtype=torch.float64)
    (g,) = torch.autograd.grad(ts.project(rho_t).sum(), rho_t)
    # numpy 的 derivative 是解析整条链；torch 是 autograd —— 应 1e-14 级一致
    assert np.abs(g.numpy() - simp.derivative(rho)).max() < 1e-12


def test_torch_nu_matches_blend(curve, simp):
    """torch_nu == 受信四类混合的 1/μ（任意 B/ρ，掩码算术加权）。"""
    tchip = TorchPCHIP(curve._H_of_B.c, curve._H_of_B.x)
    n = 300
    rng = np.random.default_rng(0)
    B = 10 ** rng.uniform(-4, 0.3, n)          # 0.0001 .. ~2
    rho = rng.uniform(0.0, 1.0, n)
    masks = _masks(n, simp_frac=0.5)
    # numpy 受信（照 field_error 的 blend）
    cached = CachedMaterial(curve, simp, mu0=4e-7 * np.pi, mur_linear=2000.0)
    state = cached.prepare_density(rho)
    mu_simp, dg = cached.nonlinear(B, state)
    mu_non, _ = fast_nt.nonlinear_material(B, curve)
    mu_blend, _ = blend_class_values(
        mu_simp, dg, mu_non, np.zeros_like(mu_non), masks,
        cached.mu0, cached.mur_linear)
    nu_np = 1.0 / mu_blend
    # torch
    nu_t = torch_nu(torch.tensor(B, dtype=torch.float64),
                    torch.tensor(rho, dtype=torch.float64), masks, tchip,
                    TorchSIMP(simp), cached.mu0, cached.mur_linear)
    assert np.abs(nu_t.numpy() - nu_np).max() / np.abs(nu_np).max() < 1e-12


def test_autograd_dnu_drho_matches_legacy_chain(curve, simp):
    """∂ν/∂ρ（autograd）== legacy 链（is_simp 区；非 simp 区为 0）。"""
    tchip = TorchPCHIP(curve._H_of_B.c, curve._H_of_B.x)
    n = 300
    rng = np.random.default_rng(1)
    B = 10 ** rng.uniform(-4, 0.3, n)
    rho = rng.uniform(0.0, 1.0, n)
    masks = _masks(n, simp_frac=0.4)
    cached = CachedMaterial(curve, simp, mu0=4e-7 * np.pi, mur_linear=2000.0)
    rho_t = torch.tensor(rho, dtype=torch.float64, requires_grad=True)
    nu_t = torch_nu(torch.tensor(B, dtype=torch.float64), rho_t, masks, tchip,
                    TorchSIMP(simp), cached.mu0, cached.mur_linear)
    (g,) = torch.autograd.grad(nu_t.sum(), rho_t)
    g = g.numpy()
    # legacy 链用一个临时材料束（只用到 curve/simp）
    import types
    material = types.SimpleNamespace(curve=curve, simp=simp)
    legacy = legacy_dgamma_p_drho(B, rho, material)
    is_simp = np.asarray(masks.is_simp) == 1.0
    assert np.abs(g[~is_simp]).max() < 1e-20          # 非 simp 区 ν 与 ρ 无关
    assert (np.abs(g[is_simp] - legacy[is_simp]) / np.abs(legacy[is_simp])).max() < 1e-10
