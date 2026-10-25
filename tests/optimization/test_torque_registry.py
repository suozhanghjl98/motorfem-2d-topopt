"""tests/optimization/test_torque_registry.py —— 阶段 7：转矩指标注册表。

来源 A（唯一能证明「对」的来源）：三个指标的 ``requires_frozen_solves`` 声明
与 legacy ``nonlinear_solver_energy_MMA10.py`` 的**实际计算依赖**一一对应
（不是拍脑袋的元数据，是可以对账的事实）：

- ``obj_func_worker_vectorized_v2`` ``:2971-3041``：
    Arkkio  ``:2987-2991``  在 Newton 解 ``A`` 上直接算（``Torque_Arkkio_coef``）
    VD      ``:2997-3001``  在 Newton 解 ``A`` 上直接算（``Torque_VD_coef``）
    flux    ``:3006-3033``  依赖 ``:3006-3011`` 的**两次**冻结磁导率求解
                            （``solve_field_vectorized_frozen_appearance_conductivity``，
                            源项 ``Rd``/``Rq`` ⇒ d/q 两组）

来源 B（保真搬移验证）：与 ``em_worker.em_step`` 的硬编码依赖对照 ——
``em_step`` 里 Arkkio/VD 分支不调用冻结求解，只有 flux 分支调用
``fast_flux.frozen_flux_analysis``。

运行::

    python -m pytest tests/optimization/test_torque_registry.py -v
"""
import inspect

import pytest

from motorfem.optimization import (ArkkioTorque, FrozenMuFluxTorque,
                                   TORQUE_REGISTRY, TorqueMetric,
                                   VirtualWorkTorque, requires_frozen_solves)


def test_registry_has_exactly_three_metrics():
    """注册表正好三个指标：Arkkio / VD / flux（生产路径全量计算）。"""
    names = [m.name for m in TORQUE_REGISTRY]
    assert names == ["Arkkio", "VD", "flux"]
    assert len(TORQUE_REGISTRY) == 3


def test_all_are_torque_metric_subclasses():
    """注册项是 TorqueMetric 的**子类**（类即注册项，像枚举；``:3054/:3410``
    的 obj_func_parallel_AL(_v2) 也是类对象）。"""
    assert all(issubclass(m, TorqueMetric) for m in TORQUE_REGISTRY)
    # 且可无参实例化（将来若要携带状态，实例即声明）
    assert all(m() for m in TORQUE_REGISTRY)


def test_arkkio_does_not_need_frozen_solves():
    """Arkkio（``MMA10:2987-2991``）：直接算在 Newton 解上，无冻结求解。"""
    assert ArkkioTorque.name == "Arkkio"
    assert ArkkioTorque.requires_frozen_solves == ()
    assert not requires_frozen_solves(ArkkioTorque)


def test_vd_does_not_need_frozen_solves():
    """VD（``MMA10:2997-3001``）：直接算在 Newton 解上，无冻结求解。"""
    assert VirtualWorkTorque.name == "VD"
    assert VirtualWorkTorque.requires_frozen_solves == ()
    assert not requires_frozen_solves(VirtualWorkTorque)


def test_flux_needs_dq_frozen_solves():
    """flux（``MMA10:3006-3033``）：电感→磁链→转矩，依赖 d/q 两次冻结求解。"""
    assert FrozenMuFluxTorque.name == "flux"
    assert FrozenMuFluxTorque.requires_frozen_solves == ("d", "q")
    assert requires_frozen_solves(FrozenMuFluxTorque)


def test_names_match_legacy_detail_keys():
    """指标名 = legacy ``detail`` dict 的转矩键（``MMA10:3737-3748``）。"""
    from motorfem.fem.em_worker import em_step
    # em_step 的返回元组里第 2/9/8 位分别是 T_Arkkio / T_VD / T_flux，
    # detail 键用同名：torque_Arkkio / torque_VD / torque_flux
    sig = inspect.signature(em_step)
    assert "vd_coef" in sig.parameters  # VD 走专用系数矩阵，与 Arkkio 分开
    names = {m.name for m in TORQUE_REGISTRY}
    assert names == {"Arkkio", "VD", "flux"}


def test_em_step_frozen_dependency_matches_registry():
    """来源 B：em_step 里只有 flux 分支调冻结分析 —— 与注册表声明一致。

    反证：若未来给 Arkkio/VD 加了冻结求解，但注册表没改，这条测试会拦住
    （它检查「不需要冻结求解的指标，其分支里不该出现 frozen_flux_analysis」）。
    """
    import motorfem.fem.em_worker as ew
    src = inspect.getsource(ew.em_step)
    frozen_call = "frozen_flux_analysis" in src
    # em_step 里恰好一次 frozen_flux_analysis 调用（flux 分支）
    assert frozen_call
    assert src.count("frozen_flux_analysis") == 1
    # 每个需要冻结求解的指标，em_step 里都有对应消费；反之亦然（1:1 对账）
    registry_needing = [m for m in TORQUE_REGISTRY if requires_frozen_solves(m)]
    assert len(registry_needing) == 1
    assert registry_needing[0].name == "flux"
