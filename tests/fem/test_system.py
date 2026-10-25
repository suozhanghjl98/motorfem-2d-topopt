"""tests/fem/test_system.py —— 阶段 4E：FieldSystem（warper + MMA 驱动）

验证核心：``FieldSystem.optimize(max_step=3, ...)`` 的**完整 3 步 MMA 轨迹**
（含 ``mma_result_*``）与 golden L4 全部 149 个数组 ``np.array_equal``。

**来源 A（本批新增）**：``MMA_optimize`` 在带不等式约束的二次型上收敛到解析解
（``test_mma_driver_converges_to_analytic``）—— 驱动不是「照抄即对」，这是它
唯一与 golden 无关的正确性证据。

运行::

    python -m pytest tests/fem/test_system.py -v
"""
import os
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig
from motorfem.fem.mma_driver import MMA_optimize
from motorfem.fem.system import FieldSystem
from motorfem.mesh import MotorMesh

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]
_GOLDEN_DIR = _REPO_ROOT / os.environ.get("MOTORFEM_BASELINE", "golden")
_GOLDEN_L4 = _GOLDEN_DIR / "L4_mech_mma.npz"


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


# ---------------------------------------------------------------- 来源 A
def test_mma_driver_converges_to_analytic():
    """MMA 驱动在带不等式约束的二次型上收敛到解析解（来源 A）。

    ``min (x−0.5)²`` s.t. ``x ≤ 0.5``、``x ≥ −0.5`` ⇒ ``x* = 0.5``。
    目标与约束：``f = [obj, x−0.5, −x−0.5]``，梯度 ``df = [2(x−0.5), 1, −1]``。
    """
    def fun(x):
        f = np.array([(x[0] - 0.5) ** 2, x[0] - 0.5, -x[0] - 0.5])
        df = np.array([[2 * (x[0] - 0.5)], [1.0], [-1.0]])
        return f, df

    fval, x = MMA_optimize(fun, x0=np.array([0.0]), x_upper=np.array([1.0]),
                           x_lower=np.array([0.0]), max_step=200)
    assert abs(x[0] - 0.5) < 1e-4, f"MMA 未收敛到 0.5，实得 {x[0]}"


# ---------------------------------------------------------------- golden 对齐
@pytest.fixture(scope="module")
def fs():
    cfg = RunConfig.legacy()
    csv = _REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv
    return FieldSystem(cfg, str(_mesh_path()), str(csv))


def test_l4_full_trajectory_matches_golden(fs):
    """完整 3 步 MMA 轨迹（含 mma_result_*）与 golden L4 全部数组对齐。"""
    if not _GOLDEN_L4.exists():
        pytest.skip(f"{_GOLDEN_L4} 不存在 —— 基线冻结尚未入库")
    fs.optimize(max_step=3, f_rel_change=-1.0, x_rel_change=-1.0)
    actual = fs.L4_snapshot()
    golden = np.load(_GOLDEN_L4)
    for k in sorted(golden.files):
        assert k in actual, f"actual 缺 {k}"
        assert np.array_equal(golden[k], actual[k]), (
            f"{k} 与 golden 不一致\n"
            f"  golden {golden[k].shape} / actual {np.asarray(actual[k]).shape}")
    assert set(actual) == set(golden.files)


def test_warper_return_shape(fs):
    """warper 返回 (5,) 与 (5, N_topo)（golden 的 call_{i}_f/df 形状）。"""
    x0 = fs.top.rho_SIMP_initial[list(fs.top.element_topo_opti_m)]
    f, df = fs.warper(x0)
    assert f.shape == (5,)
    assert df.shape == (5, len(fs.top.element_topo_opti_m))


def test_optimize_reproducible():
    """两个全新实例跑同样配置：轨迹逐位相同（L-BFGS-B 确定性 + BLAS 单线程）。"""
    cfg = RunConfig.legacy()
    csv = str(_REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv)
    msh = str(_mesh_path())
    a = FieldSystem(cfg, msh, csv)
    b = FieldSystem(cfg, msh, csv)
    a.optimize(max_step=3, f_rel_change=-1.0, x_rel_change=-1.0)
    b.optimize(max_step=3, f_rel_change=-1.0, x_rel_change=-1.0)
    assert np.array_equal(a.last_x, b.last_x)
    assert np.array_equal(a.last_f, b.last_f)


# ---------------------------------------------------------------- GUI 续跑钩子
class _Recorder:
    """mma_driver 的 control 鸭子类型：记录每步 f_val + 保留最后快照。"""

    def __init__(self):
        self.states = []
        self.last_state = None

    def stop_requested(self):
        return False

    def wait_if_paused(self):
        return None

    def on_step(self, i_step, state):
        self.last_state = state
        self.states.append(state)


def test_mma_resume_fresh_instances_match():
    """续跑（检查点 → 新实例恢复）与从头跑**逐位一致**。

    覆盖 GUI 的保存/加载路径：``a`` 从头跑 3 步；``b`` 跑 1 步后取
    ``MMA 快照 + fs 检查点``（``checkpoint_state``），``c``（全新实例）
    恢复后继续 2 步。三段 obj 轨迹与 ``a`` 逐位相同 —— 这同时验证了
    ``initial_state``（MMA 渐近线变量）与 ``a_last``（Newton 热启动）都
    被无损恢复。
    """
    cfg = RunConfig.legacy()
    csv = str(_REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv)
    msh = str(_mesh_path())

    a = FieldSystem(cfg, msh, csv)
    ra = _Recorder()
    a.optimize(max_step=3, f_rel_change=-1.0, x_rel_change=-1.0, control=ra)

    b = FieldSystem(cfg, msh, csv)
    rb = _Recorder()
    b.optimize(max_step=1, f_rel_change=-1.0, x_rel_change=-1.0, control=rb)
    assert rb.last_state is not None and rb.last_state["i_step"] == 1
    snap = rb.last_state
    b_fs_state = b.checkpoint_state()

    c = FieldSystem(cfg, msh, csv)
    c.restore_checkpoint_state(b_fs_state)
    rc = _Recorder()
    c.optimize(max_step=3, f_rel_change=-1.0, x_rel_change=-1.0,
               control=rc, initial_state=snap)

    obj_a = [s["f_val"][0] for s in ra.states]
    obj_bc = [s["f_val"][0] for s in rb.states] + [s["f_val"][0] for s in rc.states]
    assert len(obj_a) == len(obj_bc) == 3
    assert np.array_equal(np.asarray(obj_a), np.asarray(obj_bc)), (
        "续跑轨迹与从头跑不一致\n"
        f"  从头跑 {obj_a}\n"
        f"  1+续2   {obj_bc}")
    assert np.array_equal(a.last_x, c.last_x)
    assert np.array_equal(a.last_f, c.last_f)


# ---------------------------------------------------------------- p 连续化
def test_p_continuation_schedule():
    """p 连续化的调度：默认关 = 固定；开 = 线性爬升 + 步数饱和（来源 A）。

    直接测 ``FieldSystem._current_p_exp``（真实方法，不是复刻），但只改
    ``n_call`` 手动驱动，不跑 warper/优化 —— 纯调度逻辑，无需真实网格。
    """
    cfg = RunConfig.legacy()
    fs = object.__new__(FieldSystem)      # 不跑 __init__（避免构造网格）
    fs.cfg = cfg

    # 默认关：p 恒为 legacy 值 10
    for n in (0, 1, 5, 100):
        fs.n_call = n
        assert fs._current_p_exp() == cfg.objective.p_exp_torque_ripple == 10

    # 开：3 → 10，300 步内线性爬升，之后固定
    cfg.objective.p_exp_continuation = True
    cfg.objective.p_exp_start = 3.0
    cfg.objective.p_exp_ramp_steps = 300
    cases = [
        (0, 3.0),        # 起始步 = p_start
        (150, 6.5),      # 中途线性
        (300, 10.0),     # 到达终值
        (1000, 10.0),    # 饱和
    ]
    for n, expect in cases:
        fs.n_call = n
        got = fs._current_p_exp()
        assert abs(got - expect) < 1e-12, f"n={n}: 期望 {expect}，实得 {got}"

    # ramp_steps=1 时首个非零步即饱和到终值（防除零）
    cfg.objective.p_exp_ramp_steps = 1
    fs.n_call = 1
    assert fs._current_p_exp() == 10.0

    # 默认关时 move 恒为 MMA_move（golden 逐位的前提）
    cfg.objective.p_exp_continuation = False
    for n in (0, 50, 300):
        fs.n_call = n
        assert fs._current_move(n) == cfg.mma.MMA_move


def test_p_continuation_move_linkage():
    """p 连续化开启时 move 随 p 爬升同步收紧（来源 A，调度纯逻辑）。

    p 每升 p_exp_move_step，move × p_exp_move_factor。起点 = MMA_move，
    终点收敛到 MMA_move·factor^((end−start)/step)。关闭时恒等于 MMA_move。
    """
    cfg = RunConfig.legacy()
    fs = object.__new__(FieldSystem)
    fs.cfg = cfg

    # 关（默认）：即使 p 连续化开，move 也恒等于 MMA_move（= cont 行为）
    cfg.objective.p_exp_continuation = True
    cfg.objective.p_exp_start = 3.0
    cfg.objective.p_exp_ramp_steps = 300
    base = cfg.mma.MMA_move
    for n in (0, 150, 300):
        fs.n_call = n
        assert fs._current_move(n) == base

    # 开：p 3→10，ramp 300 步，move_step=1、factor=0.7（= contm 行为）
    cfg.objective.p_exp_move_linkage = True
    cfg.objective.p_exp_move_step = 1.0
    cfg.objective.p_exp_move_factor = 0.7

    fs.n_call = 0          # p=3.0
    assert abs(fs._current_move(0) - base) < 1e-12
    fs.n_call = 300        # p=10.0，爬了 7 档
    expect_end = base * (0.7 ** 7)
    assert abs(fs._current_move(300) - expect_end) < 1e-12, \
        f"期望 move {expect_end:.6g}，实得 {fs._current_move(300):.6g}"
    fs.n_call = 150        # p=6.5，爬了 3.5 档
    expect_mid = base * (0.7 ** 3.5)
    assert abs(fs._current_move(150) - expect_mid) < 1e-12

    # 下限保护：极端收窄不会到 0 或负数
    fs.n_call = 300
    assert fs._current_move(300) >= 1e-4
