"""tests/autograd/test_pchip.py —— 阶段 6：TorchPCHIP 重建 scipy PCHIP

验证：torch 分段三次求值与 scipy ``PchipInterpolator`` 的值、一阶导一致
（相对 1e-12 级），且 torch autograd 对 x 的梯度 == scipy 的解析导。
这是 oracle 的 B-H 链基础 —— 值/导都对齐，梯度验收（1e-9）才有意义。
"""
import glob
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motorfem.autograd.pchip import TorchPCHIP
from motorfem.kernels.reference.bh_curve import BHCurve

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def curve():
    csvs = list(Path(_REPO_ROOT).rglob("50CS470.csv"))
    if not csvs:
        pytest.skip("找不到 50CS470.csv")
    return BHCurve.from_csv(str(csvs[0]))


@pytest.fixture(scope="module")
def tchip(curve):
    return TorchPCHIP(curve._H_of_B.c, curve._H_of_B.x)


def test_value_matches_scipy(curve, tchip):
    """H(x)：torch 重建 vs scipy，相对 1e-12。"""
    x = np.linspace(float(curve._H_of_B.x[0]) + 1e-9, float(curve._H_of_B.x[-1]) - 1e-9,
                    4000)[1:-1]
    H_ref = curve.H(x)
    H_t = tchip.eval(torch.tensor(x)).numpy()
    assert np.abs(H_t - H_ref).max() / np.abs(H_ref).max() < 1e-12


def test_derivative_matches_scipy(curve, tchip):
    """dH/dx：torch 重建 vs scipy 解析导，相对 1e-12。"""
    x = np.linspace(float(curve._H_of_B.x[0]) + 1e-9, float(curve._H_of_B.x[-1]) - 1e-9,
                    4000)[1:-1]
    dH_ref = curve.dHdB(x)
    dH_t = tchip.deriv(torch.tensor(x)).numpy()
    assert np.abs(dH_t - dH_ref).max() / np.abs(dH_ref).max() < 1e-12


def test_autograd_grad_equals_analytic_deriv(curve, tchip):
    """torch.autograd 对 x 的梯度 == scipy dHdB（oracle 的 B-H 链可微性）。"""
    x = np.linspace(float(curve._H_of_B.x[0]) + 1e-9, float(curve._H_of_B.x[-1]) - 1e-9,
                    300)[1:-1]
    x_t = torch.tensor(x, requires_grad=True, dtype=torch.float64)
    (g,) = torch.autograd.grad(tchip.eval(x_t).sum(), x_t)
    assert np.abs(g.numpy() - curve.dHdB(x)).max() / np.abs(curve.dHdB(x)).max() < 1e-12


def test_eval_is_second_order_differentiable(tchip):
    """eval 对 x 的二阶可微（dH/dx 再求导不炸）。"""
    x = torch.tensor([0.5, 1.0, 2.0], requires_grad=True, dtype=torch.float64)
    d = tchip.deriv(x)
    (dd,) = torch.autograd.grad(d.sum(), x)
    assert torch.isfinite(dd).all()
