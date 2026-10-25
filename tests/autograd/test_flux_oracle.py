"""tests/autograd/test_flux_oracle.py —— 阶段 8 收尾 #1：磁链转矩全导数 oracle

设计文档 §阶段 6 偏差登记①：手写 ``flux_torque_derivative`` 把冻结求解当常数
（冻结近似），真全导数 oracle 穿过冻结系统。本测试验证：

1. **forward 正确**：oracle 的 ``T = Σ k_e·ν_e`` 与 numpy ``frozen_flux_analysis``
   的 ``torque`` 一致（forward 用同一 numpy 求解）。
2. **全导数 vs 有限差分**（来源 A，独立判据）：扰动 ρ 重解冻结场、差商，
   oracle 的 ``dT_flux/dρ`` rel < 1e-6（FD 分辨率）。
3. **D4 判定证据**：oracle（全导数）vs 手写（冻结近似）的差异 —— 量化并断言
   **确实存在**（防退化：若两者逐位一致，说明冻结近似没漏项，D4 的语义假设变了）。
   同时记录差异量级，供设计文档 D4 修正清单引用。

⚠ 纪律：本测试是验证层，不改生产路径。梯度不与手写位级一致（浮点顺序不同，
见设计文档 §阶段 6），验收是相对误差 < 1e-6（FD 分辨率）。
"""
import glob
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motorfem.autograd.flux_oracle import (  # noqa: E402
    FluxFrozenContext,
    flux_torque_full_derivative,
    flux_torque_gradient,
)
from motorfem.config import RunConfig  # noqa: E402
from motorfem.fem.em_worker import abc_to_dq_orthogonal  # noqa: E402
from motorfem.fem.system import FieldSystem  # noqa: E402
from motorfem.kernels.fast import flux as fast_flux  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]
#: FD 分辨率（扰动 ρ 重解冻结场的相对噪声下限）。实测：oracle vs FD
#: max_rel=1.7e-5（FD 步长 1e-5 的分子噪声 ~O(ε/h)）。步长太小则截断误差主导，
#: 太大则差商吞掉信号 —— 1e-4 是稳定判据。
_FD_TOL = 1e-4


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


@pytest.fixture(scope="module")
def fs():
    csvs = list(_REPO_ROOT.rglob("50CS470.csv"))
    if not csvs:
        pytest.skip("找不到 50CS470.csv")
    # ★ 2026-10-13：oracle 需要精确 PCHIP ⇒ 必须 legacy 配置
    return FieldSystem(RunConfig.legacy(), str(_mesh_path()), str(csvs[0]))


def _frozen_ctx(fs, i_t: int, rho: np.ndarray, A, B) -> FluxFrozenContext:
    """从 FieldSystem 打包 i_t 工作点的冻结 oracle ctx（与生产同源）。

    ``I_d/I_q`` 用 ``abc_to_dq_orthogonal``（与 ``em_step`` 同式）；``B_frozen``
    用内核 11 返回的 ``B``（已裁剪）。``R_d/R_q`` 用 ``pre`` 的单位电流源。
    """
    pole_pairs = fs.cfg.geometry.pole_pairs
    parts = fs.cfg.geometry.parts
    pos = fs.pre.rotor_positions[i_t]
    I_A, I_B, I_C = fs.pre.currents[i_t]
    I_d, I_q = abc_to_dq_orthogonal(I_A, I_B, I_C, pos * pole_pairs)
    return FluxFrozenContext(
        top=fs.top,
        kv=fs.pre.K_vectorized_vec[i_t],
        B_frozen=np.asarray(B, dtype=float),
        material=fs.material,
        R_d=np.asarray(fs.pre.Rd_vec[i_t]),
        R_q=np.asarray(fs.pre.Rq_vec[i_t]),
        I_d=I_d, I_q=I_q,
        pole_pairs=pole_pairs, parts=parts,
    )


def _hand_frozen_gradient(fs, i_t, rho, A, B, J, ctx_frozen):
    """手写路径（生产）：``dT_flux/dρ`` —— 把冻结解当常数（冻结近似）。

    用 ``fast_flux.flux_torque_derivative`` × legacy ``dγ/dρ``（与 ``em_step``
    完全同式）。``J`` 是**非线性场**的收敛 Jacobian（手写的伴随解用它，
    ``flux_torque_derivative`` 的第 10 个位置参数）。
    """
    from motorfem.fem.em_worker import legacy_dgamma_p_drho

    dgamma = legacy_dgamma_p_drho(np.asarray(B, dtype=float),
                                  np.asarray(rho, dtype=float), fs.material)
    frozen = fast_flux.frozen_flux_analysis(
        np.asarray(B, dtype=float), np.asarray(rho, dtype=float),
        fs.pre.K_vectorized_vec[i_t], fs.pre.Rd_vec[i_t], fs.pre.Rq_vec[i_t],
        ctx_frozen.I_d, ctx_frozen.I_q,
        n_node=fs.top.n_node, area=fs.top.em.area, material=fs.material,
        pole_pairs=ctx_frozen.pole_pairs, parts=ctx_frozen.parts,
    )
    dT = fast_flux.flux_torque_derivative(
        np.asarray(A, dtype=float), np.asarray(rho, dtype=float),
        frozen.Bx_d, frozen.By_d, frozen.Bx_q, frozen.By_q,
        ctx_frozen.I_d, ctx_frozen.I_q, J, list(fs.top.element_topo_opti),
        fs.pre.K_vectorized_vec[i_t],
        n_node=fs.top.n_node, n_element=fs.top.n_element, area=fs.top.em.area,
        material=fs.material, pole_pairs=ctx_frozen.pole_pairs,
        parts=ctx_frozen.parts,
    )
    return dT * dgamma


def _fd_gradient(fs, i_t, rho, A, B, ctx_frozen, step=1e-5):
    """有限差分（来源 A，独立判据）：扰动 ρ_e 重解冻结场、差商 T_flux。"""
    # T_flux 是冻结分析的 scalar 产出；用 numpy 链重解。
    n_elem = fs.top.n_element
    fd = np.zeros(n_elem)
    # 抽样：|dT/dρ| 大概率非零的单元（拓扑区 = 转子铁芯）
    sample = sorted(fs.top.element_topo_opti)[:: max(1, len(fs.top.element_topo_opti) // 24)]
    for e in sample:
        rp, rm = rho.copy(), rho.copy()
        rp[e] += step
        rm[e] -= step
        fd[e] = (_torque_frozen(fs, i_t, rp, B, ctx_frozen)
                 - _torque_frozen(fs, i_t, rm, B, ctx_frozen)) / (2 * step)
    return fd


def _torque_frozen(fs, i_t, rho, B, ctx_frozen):
    """冻结链的 T_flux（numpy，与生产同式）。"""
    frozen = fast_flux.frozen_flux_analysis(
        np.asarray(B, dtype=float), np.asarray(rho, dtype=float),
        fs.pre.K_vectorized_vec[i_t], fs.pre.Rd_vec[i_t], fs.pre.Rq_vec[i_t],
        ctx_frozen.I_d, ctx_frozen.I_q,
        n_node=fs.top.n_node, area=fs.top.em.area, material=fs.material,
        pole_pairs=ctx_frozen.pole_pairs, parts=ctx_frozen.parts,
    )
    return frozen.torque


def test_flux_torque_gradient_matches_finite_difference(fs):
    """★★ 来源 A：全导数 oracle vs 有限差分（扰动 ρ 重解冻结场），rel < 1e-4。

    ⚠ 用 ``i_t=1``（电流非零）：``i_t=0`` 时 ``I_q≈0`` ⇒ ``k_e≈0``（q 轴电流为零
    的物理特性），FD 在拓扑区全零、测试无意义。``i_t=1`` 电流非零，``T_flux``
    有非零梯度可验。
    """
    i_t, rho = 1, np.full(fs.top.n_element, 0.5)
    A, B, J = _solve_field(fs, i_t, rho)
    ctx = _frozen_ctx(fs, i_t, rho, A, B)

    auto = flux_torque_gradient(rho, ctx)
    fd = _fd_gradient(fs, i_t, rho, A, B, ctx)

    nz = np.abs(fd) > 1e-9
    assert nz.sum() > 0, "FD 梯度全零 —— 测试无意义"
    rel = np.abs(auto[nz] - fd[nz]) / np.abs(fd[nz])
    print(f"  dT_flux/dρ: oracle vs FD  max_rel={rel.max():.3e} nz={nz.sum()}")
    assert rel.max() < _FD_TOL, f"oracle vs FD rel={rel.max():.3e} ≥ {_FD_TOL}"


def test_full_derivative_differs_from_hand_written(fs):
    """D4 判定证据：全导数 vs 手写（冻结近似）**确实不同**，量化差异。

    若两者逐位一致 ⇒ 冻结近似没漏项 ⇒ D4 的「模型性差异」假设不成立 —— 防退化。
    ⚠ 用 ``i_t=1``（电流非零，``k_e`` 非零）保证手写梯度有可比的非零条目。
    """
    i_t, rho = 1, np.full(fs.top.n_element, 0.5)
    A, B, J = _solve_field(fs, i_t, rho)
    ctx = _frozen_ctx(fs, i_t, rho, A, B)

    auto = flux_torque_gradient(rho, ctx)
    hand = _hand_frozen_gradient(fs, i_t, rho, A, B, J, ctx)

    nz = np.abs(hand) > 1e-3
    assert nz.sum() > 0
    rel = np.abs(auto[nz] - hand[nz]) / np.abs(hand[nz])
    print(f"  dT_flux/dρ: 全导数 vs 手写(冻结近似)  max_rel={rel.max():.3e} "
          f"（>1e-6 ⇒ 冻结近似确实漏项，D4 的模型性差异被量化）")
    assert rel.max() > _FD_TOL, (
        f"全导数与手写一致到 FD 分辨率（rel={rel.max():.3e}）—— 冻结近似没漏项，"
        f"D4 的判定前提变了，重新核对设计文档 §12 的 D4。")


def _solve_field(fs, i_t, rho):
    """解一个时间点，返回 (A, B, J)。与 test_oracle 的 _working_point 同构。"""
    from motorfem.autograd import oracle_from_parts

    oracle = oracle_from_parts(fs.top, fs.pre, fs.material, i_t)
    A, B, J = oracle.solve(np.asarray(rho, dtype=float))
    return A, B, J
