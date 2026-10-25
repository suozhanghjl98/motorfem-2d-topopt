# -*- coding: utf-8 -*-
"""verify_env.py —— 提交包环境自检：能不能真的跑起来。

判据（README「怎样判断结果成立」第 0 步）：
  1. 依赖齐全（numba 可选，缺则自动关闭 B-H 查表）
  2. 能构造 FieldSystem（网格 / 伴生配置 / B-H 曲线都能解析）
  3. 能完成一次评价并给出 T_flux
  4. golden 基线文件在位（L1/L3）

CHOLMOD 一项允许 FAIL —— 包内默认走 superlu，结果与 golden **逐位一致**。
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools" / "scripts"))

ok = True


def check(name, fn, fatal=True):
    global ok
    try:
        r = fn()
        print(f"  [OK]   {name}: {r}")
    except Exception as e:
        if fatal:
            ok = False
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        else:
            print(f"  [SKIP] {name}: {type(e).__name__}: {e}")


print("=== 1. 依赖 ===")
for m in ("numpy", "scipy", "matplotlib", "shapely", "sympy", "joblib", "gmsh"):
    def f(m=m):
        mod = __import__(m)
        return getattr(mod, "__version__", "?")
    check(m, f)


def numba_v():
    import numba
    return numba.__version__


check("numba（B-H 查表用）", numba_v, fatal=False)


print("\n=== 2. 求解后端 ===")
def backend():
    from _bundle_env import make_config, describe
    cfg = make_config()
    return describe(cfg)


check("后端选择", backend)


print("\n=== 3. golden 基线 ===")
for g in ("L1_regions.npz", "L3_em_step.npz", "L4_mech_mma.npz",
          "manifest.json"):
    check(g, lambda g=g: f"{(HERE / 'golden' / g).stat().st_size/1e6:.1f} MB"
          if (HERE / "golden" / g).exists() else (_ for _ in ()).throw(FileNotFoundError(g)))


print("\n=== 4. 建 FieldSystem（6s4p）===")
MESH = HERE / "legacy" / "pytorch" / "my_gmsh" / "model_6s4p.msh"
CSV = HERE / "legacy" / "pytorch" / "50CS470.csv"


def build():
    from motorfem.config import apply_mesh_config
    from motorfem.fem.system import FieldSystem
    from _bundle_env import make_config
    assert MESH.exists(), f"缺网格 {MESH}"
    cfg = make_config()
    apply_mesh_config(cfg, str(MESH))
    fs = FieldSystem(cfg, str(MESH), str(CSV))
    return f"{len(fs.top.element_topo_opti_m)} 个主动设计单元"


check("构造", build)


print("\n=== 5. 单次评价（约 5~20 s）===")
def evaluate():
    import numpy as np
    from motorfem.config import apply_mesh_config
    from motorfem.fem.system import FieldSystem
    from motorfem.fem.em_worker import density_from_x
    from motorfem.fem.init_density import filter_inverse_rho
    from _bundle_env import make_config, describe
    cfg = make_config()
    apply_mesh_config(cfg, str(MESH))
    fs = FieldSystem(cfg, str(MESH), str(CSV))
    # ⚠ `rho_SIMP_initial` 是**密度**，不是设计变量（2026-10-18 实测）：
    #   直接当 x 传进去，密度会错乱（T_flux 掉到 0.16 N·m）。
    #   必须取逆投影：x 满足 ρ = filter_SIMP @ x。
    topo = list(fs.top.element_topo_opti_m)
    x = filter_inverse_rho(
        fs.top.filter_SIMP,
        np.asarray(fs.top.rho_SIMP_initial, float),
        topo, min_density=1e-6)[topo]
    call = fs.evaluate(density_from_x(x, fs.top), subset="train")
    T = float(np.mean(call.torque_flux_energy_vec))
    return f"T_flux = {T:.3f} N·m（{describe(cfg)}）"


check("评价", evaluate)

print("\n=== 6. 算力 ===")
print(f"  CPU 逻辑核 = {os.cpu_count()}")

print("\n" + ("=== 全部通过，可以跑 demo_smoke.py ===" if ok else
              "=== 有失败项，先解决再跑（见上面 [FAIL]）==="))
sys.exit(0 if ok else 1)
