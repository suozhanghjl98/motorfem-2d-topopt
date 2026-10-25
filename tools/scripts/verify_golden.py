#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/verify_golden.py —— golden L1~L4 对齐验收（设计文档 §1.7 第 2 条 / §7 第 1 条）

**目标**：新代码跑同样输入（同一 ``model_mirror4.msh`` + 生产配置），golden 四层
L1~L4 **全部** ``np.array_equal``。本脚本逐数组比对并给出可定位的失败报告。

**环境纪律**（与 ``tools/golden_capture.py`` 采集时同款）：
``OMP/OPENBLAS/MKL_NUM_THREADS=1`` —— 多线程 BLAS 改变归约顺序，是位级一致的
硬前提。必须在 import numpy **之前**设置。

**用法**：

    cd <仓库根>
    PYTHONIOENCODING=utf-8 python tools/verify_golden.py [--layer L1|L2|L3|L4|all]
        [--inject-x]        # L3/L4 用 golden 的 call_{i}_x 注入，跳过 MMA 轨迹

**退出码**：0 = 所选层全部通过；1 = 有失败；2 = 目标层尚未实现/环境不满足。

**阶段进度**：L1 + L2 的 ρ 无关部分已实现（阶段 4A）；L2 预组装、L3、L4 随
子阶段 B~E 逐步点亮。未实现的层以「TODO」列出并返回 2。
"""

import os

# ---- 位级一致的环境（必须在 import numpy/scipy 之前）----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
_GOLDEN_DIR = _REPO_ROOT / "golden"
_MESH = _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh"

_GOLDEN_FILES = {
    "L1": "L1_regions.npz",
    "L2": "L2_element.npz",
    "L3": "L3_em_step.npz",
    "L4": "L4_mech_mma.npz",
}


def _load_layer(name: str) -> np.lib.npyio.NpzFile:
    path = _GOLDEN_DIR / _GOLDEN_FILES[name]
    return np.load(path)


def _diagnose(g: np.ndarray, a: np.ndarray) -> str:
    """失败的定位信息：形状差 / 首差索引与两边的值。"""
    if g.shape != a.shape:
        return f"shape: golden {g.shape} vs actual {a.shape}"
    if g.ndim == 0:                       # 标量数组：unravel_index 对空 shape 会崩
        return (f"max|Δ| = {abs(float(g) - float(a)):.6g}  "
                f"golden {float(g)!r}  actual {float(a)!r}")
    d = np.abs(g.astype(np.float64) - a.astype(np.float64))
    idx = np.unravel_index(np.argmax(d), d.shape)
    gv, av = g[idx], a[idx]
    return (f"max|Δ| = {d.max():.6g} @ index {idx}  "
            f"golden {gv!r}  actual {av!r}")


def compare_layer(name: str, golden, actual: dict):
    """逐数组比对。返回 (results, missing, extra)。"""
    g_keys = set(golden.files)
    a_keys = set(actual.keys())
    missing = sorted(g_keys - a_keys)
    extra = sorted(a_keys - g_keys)

    results = []
    for k in sorted(g_keys & a_keys):
        g, a = golden[k], np.asarray(actual[k])
        if np.array_equal(g, a):
            results.append((k, "PASS", ""))
        else:
            results.append((k, "FAIL", _diagnose(g, a)))
    return results, missing, extra


# ---------------------------------------------------------------- actual 侧
_TOPOLOGY_CACHE = {}


def _build_topology():
    if "top" not in _TOPOLOGY_CACHE:
        sys.path.insert(0, str(_REPO_ROOT))
        from motorfem.config import RunConfig
        from motorfem.mesh import MotorMesh

        from motorfem.fem.mesh_builder import build_field_topology

        mesh = MotorMesh(str(_MESH))
        _TOPOLOGY_CACHE["top"] = build_field_topology(RunConfig.legacy(), mesh)
    return _TOPOLOGY_CACHE["top"]


def _build_preassembly():
    if "pre" not in _TOPOLOGY_CACHE:
        from motorfem.config import RunConfig
        from motorfem.fem.preassembly import build_preassembly

        _TOPOLOGY_CACHE["pre"] = build_preassembly(RunConfig.legacy(), _build_topology())
    return _TOPOLOGY_CACHE["pre"]


def _build_L1_actual() -> dict:
    return dict(_build_topology().L1_snapshot())


def _build_L4_actual(n_calls: int) -> dict:
    """注入 golden 的 call_{i}_x，跑力学段 + 目标聚合 + AL 状态机（阶段 4D）。

    与 MMA 轨迹解耦（mma_result_* 不在此验证，属阶段 4E）。
    """
    from motorfem.config import RunConfig
    from motorfem.fem.em_worker import density_from_x, make_field_material
    from motorfem.fem.mech_worker import stress_flags_from_config
    from motorfem.fem.objective import L4_snapshot, fun
    from motorfem.kernels.reference.al import al_initial_state, al_update

    top = _build_topology()  # 先调用：内部把 _REPO_ROOT 插入 sys.path
    pre = _build_preassembly()
    cfg = RunConfig.legacy()
    csv = _REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv
    # ⚠ **显式钉死 scipy PCHIP**（``bh_lut_n=0``）：golden 是**legacy 位级对齐**路径，
    #   必须走原曲线。这里原来只传一个参数、靠函数默认值拿到 0 —— 行为一样但**是隐式的**：
    #   一旦 ``RunConfig.material.bh_lut_n`` 的默认值改了，读代码的人会以为 golden 会跟着走
    #   （其实不会，因为它根本没读 config），两边对不上又没人说得清。
    material = make_field_material(str(csv), bh_lut_n=0)
    flags = stress_flags_from_config(cfg)

    golden4 = _load_layer("L4")
    x_calls = [np.asarray(golden4[f"call_{c:02d}_x"]) for c in range(n_calls)]

    state = al_initial_state(top.n_element_mech)
    a_last: dict = {}
    calls, states, counters, a_last_snaps = [], [], [], []
    for c, x in enumerate(x_calls):
        rho = density_from_x(x, top)
        call = fun(rho, pre, top, material, a_last,
                   state.al_lambda, state.al_mu, flags=flags)
        calls.append(call)
        state = al_update(call.al_lambda_new, call.stress_loss_al, state)
        states.append(state)
        counters.append(c + 1)              # _N_call 在每次调用后 +1
        a_last_snaps.append(dict(a_last))

    out = L4_snapshot(calls, states, counters, x_calls, a_last_snaps, top)
    # 排除 MMA 轨迹（阶段 4E 验证）
    out = {k: v for k, v in out.items() if not k.startswith("mma_result")}
    return out


def _build_L3_actual(n_calls: int) -> dict:
    """注入 golden 的 call_{i}_x，跑完整 EM 段（阶段 4C）。

    与 MMA 轨迹解耦：x 从 golden L4 取，跳过 MMA 更新。A_node_last 暖启动
    按 legacy 语义跨调用缓存。
    """
    from motorfem.fem.em_worker import density_from_x, em_segment, make_field_material
    from motorfem.config import RunConfig

    top = _build_topology()  # 先调用：内部把 _REPO_ROOT 插入 sys.path
    pre = _build_preassembly()
    cfg = RunConfig.legacy()
    csv = _REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv
    # ⚠ **显式钉死 scipy PCHIP**（``bh_lut_n=0``）：golden 是**legacy 位级对齐**路径，
    #   必须走原曲线。这里原来只传一个参数、靠函数默认值拿到 0 —— 行为一样但**是隐式的**：
    #   一旦 ``RunConfig.material.bh_lut_n`` 的默认值改了，读代码的人会以为 golden 会跟着走
    #   （其实不会，因为它根本没读 config），两边对不上又没人说得清。
    material = make_field_material(str(csv), bh_lut_n=0)

    golden4 = _load_layer("L4")
    out = {}
    a_last: dict = {}
    for c in range(n_calls):
        x = golden4[f"call_{c:02d}_x"]
        rho = density_from_x(x, top)
        records = em_segment(rho, pre, top, material, A_node_last=a_last)
        for i_t, res in enumerate(records):
            for i, n in enumerate(("Torque_Arkkio", "dTArkkio_p_drho", "A_node",
                                   "B_elem", "flux_linkage_energy",
                                   "torque_flux_energy", "dTflux_p_drho",
                                   "Torque_VD", "dTVD_p_drho")):
                out[f"em_{c:02d}_{i_t:02d}_{n}"] = np.asarray(res[i + 1])
            out[f"em_{c:02d}_{i_t:02d}_call"] = np.asarray(c)
            out[f"em_{c:02d}_{i_t:02d}_t"] = np.asarray(res[0])
    return out


def _build_L2_actual() -> dict:
    top = _build_topology()
    out = dict(top.L2_snapshot_partial())
    out.update(_build_preassembly().L2_snapshot())
    return out


def _report(results, missing, extra) -> int:
    n_pass = sum(1 for _k, s, _d in results if s == "PASS")
    n_fail = len(results) - n_pass
    for k, s, d in results:
        print(f"  [{s:4s}] {k}" + (f"    ← {d}" if s == "FAIL" else ""))
    for k in missing:
        print(f"  [缺失] {k}    ← actual 没有这个键")
    for k in extra:
        print(f"  [多出] {k}    ← golden 没有这个键")
    print(f"\n  {n_pass} 通过 / {n_fail} 失败 / {len(missing)} 缺失 / {len(extra)} 多出")
    if n_pass == len(results) and not missing and not extra:
        print("  ✓ 全对齐")
        return 0
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", choices=("L1", "L2", "L3", "L4", "all"), default="L1")
    ap.add_argument("--inject-x", action="store_true",
                    help="L3/L4 用 golden 的 call_{i}_x 注入 x（跳过 MMA 轨迹验证）")
    ap.add_argument("--baseline", choices=("golden", "golden2"), default="golden",
                    help="比对哪套基线。golden = legacy 冻结基线；"
                         "golden2 = motorfem 当前口径（D13/D14 之后）。默认 golden。")
    args = ap.parse_args()

    global _GOLDEN_DIR
    _GOLDEN_DIR = _REPO_ROOT / args.baseline
    print(f"基线目录: {_GOLDEN_DIR}")

    if not _MESH.exists():
        print(f"[失败] 找不到网格文件：{_MESH}")
        return 2

    # 版本记录（golden 在同环境采集；换环境必须重验位级承诺）
    import scipy
    try:
        import shapely
        shapely_v = shapely.__version__
    except ImportError:
        shapely_v = "?"
    print(f"环境: numpy {np.__version__} / scipy {scipy.__version__} / "
          f"shapely {shapely_v} / BLAS 单线程=1")

    layers = ("L1", "L2", "L3", "L4") if args.layer == "all" else (args.layer,)
    exit_codes = []
    for name in layers:
        print("=" * 74)
        print(f"层 {name}")
        print("=" * 74)
        golden = _load_layer(name)

        if name == "L1":
            actual = _build_L1_actual()
            results, missing, extra = compare_layer(name, golden, actual)
            exit_codes.append(_report(results, missing, extra))
        elif name == "L2":
            actual = _build_L2_actual()
            results, missing, extra = compare_layer(name, golden, actual)
            exit_codes.append(_report(results, missing, extra))
        elif name == "L3":
            if not args.inject_x:
                print("  L3 需要 --inject-x（用 golden call_x 注入 x，跳过 MMA）")
                exit_codes.append(2)
                continue
            n_calls = 3
            actual = _build_L3_actual(n_calls)
            results, missing, extra = compare_layer(name, golden, actual)
            exit_codes.append(_report(results, missing, extra))
        elif name == "L4":
            if args.inject_x:
                # 阶段 4D：注入 golden call_x，跳过 MMA（mma_result_* 不算缺失）
                actual = _build_L4_actual(3)
                expected_missing = {"mma_result_x", "mma_result_f"}
                actual = {k: v for k, v in actual.items()
                          if k not in expected_missing}
                results, missing, extra = compare_layer(name, golden, actual)
                missing = [k for k in missing if k not in expected_missing]
                exit_codes.append(_report(results, missing, extra))
            else:
                # 阶段 4E：完整 3 步 MMA 轨迹（含 mma_result_*）
                from motorfem.config import RunConfig
                from motorfem.fem.system import FieldSystem

                cfg = RunConfig.legacy()
                csv = _REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv
                fs = FieldSystem(cfg, str(_MESH), str(csv))
                fs.optimize(max_step=3, f_rel_change=-1.0, x_rel_change=-1.0)
                actual = fs.L4_snapshot()
                results, missing, extra = compare_layer(name, golden, actual)
                exit_codes.append(_report(results, missing, extra))
        else:  # pragma: no cover
            exit_codes.append(2)

    return 1 if any(c == 1 for c in exit_codes) else (2 if any(c == 2 for c in exit_codes) else 0)


if __name__ == "__main__":
    raise SystemExit(main())
