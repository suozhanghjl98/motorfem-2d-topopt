"""CHOLMOD（ctypes 稀疏 Cholesky）后端的正确性与符号复用测试。

⚠ **本文件刻意在子进程里跑全部断言**。原因：加载 ``cholmod.dll`` 会把其目录**前置
进进程 PATH**（延迟加载的 MKL 依赖需要），这会让同进程的 scipy/SuperLU 也改绑 MKL
—— 与 pytest 主进程里其它测试的 BLAS 混用会 **`Fatal Python error: Aborted`**（实测
全套测试崩在第 69 个用例）。子进程隔离后既保住正确性断言，又零污染主进程。

⚠ 没有 ``cholmod.dll`` 时整个文件 skip（默认路径 superlu 不受影响）。

验收口径：与 SuperLU 的解相对差 < 1e-8（**不逐位** —— CHOLMOD 只读矩阵一个三角，
且算法不同；实测真实 J_free 上 rel ~5e-14）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from motorfem.cholmod_ctypes import CHOLMOD_AVAILABLE

pytestmark = pytest.mark.skipif(
    not CHOLMOD_AVAILABLE(),
    reason="cholmod.dll 不存在（设 CHOLMOD_DLL_DIR 或部署 D:\\cholmod）",
)

_ROOT = Path(__file__).resolve().parents[2]

_SUB_BODY = r'''
import ctypes
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

from motorfem.cholmod_ctypes import (CHOLMOD_AVAILABLE, CholmodContext,
                                     CholmodNotPositiveDefinite)
from motorfem.cholmod_ctypes import _pack_sparse

assert CHOLMOD_AVAILABLE(), "DLL 不可用"


def _spd(n=60, density=0.12, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, format="csc", random_state=rng, dtype=float)
    A = (A + A.T) * 0.5
    return (A + sp.eye(n, format="csc") * (abs(A).sum(axis=1).max() + 1.0)).tocsc()


# ---- 1) 与 SuperLU 一致 ----
A = _spd(); n = A.shape[0]; b = np.arange(1.0, n + 1)
ctx = CholmodContext()
try:
    ctx.factorize(A)
    x = ctx.solve(b)
finally:
    ctx.close()
x_ref = spsolve(A, b)
assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < 1e-8, "与 superlu 不一致"
assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-10
print("  [OK] 1) 与 SuperLU 一致，残差", np.linalg.norm(A @ x - b) / np.linalg.norm(b))

# ---- 2) 符号复用（同模式新数值）----
A = _spd(seed=1); b = np.linspace(0.5, 3.0, A.shape[0])
ctx = CholmodContext()
try:
    ctx.factorize(A)
    sig1 = ctx._sig
    x1 = ctx.solve(b)
    A2 = (A * 1.7).tocsc()
    ctx.factorize(A2)
    assert ctx._sig == sig1, "同结构必须复用符号缓存"
    x2 = ctx.solve(b)
finally:
    ctx.close()
assert np.linalg.norm(A2 @ x2 - b) / np.linalg.norm(b) < 1e-10
assert not np.allclose(x1, x2), "防退化：数值确实变了"
print("  [OK] 2) 同模式复用符号，数值确实变化")

# ---- 3) 模式变了必须重新符号分析 ----
A = _spd(n=40, seed=2)
ctx = CholmodContext()
try:
    ctx.factorize(A)
    sig1 = ctx._sig
    B = _spd(n=40, density=0.3, seed=3)
    ctx.factorize(B)
    assert ctx._sig != sig1, "模式变了必须重分析"
    bb = np.arange(1.0, 41)
    xb = ctx.solve(bb)
finally:
    ctx.close()
assert np.linalg.norm(B @ xb - bb) / np.linalg.norm(bb) < 1e-10
print("  [OK] 3) 模式变化触发重新符号分析")

# ---- 4) 多右端 ----
A = _spd(seed=4); n = A.shape[0]
ctx = CholmodContext()
try:
    ctx.factorize(A)
    for k in range(3):
        bk = np.full(n, float(k + 1))
        xk = ctx.solve(bk)
        assert np.linalg.norm(A @ xk - bk) / np.linalg.norm(bk) < 1e-10
finally:
    ctx.close()
print("  [OK] 4) 多右端")

# ---- 5) 非正定必须抛异常（判据 = 因子头部 minor != n）----
Ai = sp.eye(30, format="lil"); Ai[0, 0] = -1.0; Ai = sp.csc_matrix(Ai)
ctx = CholmodContext()
try:
    try:
        ctx.factorize(Ai)
        raise AssertionError("非正定未抛 CholmodNotPositiveDefinite")
    except CholmodNotPositiveDefinite:
        pass
finally:
    ctx.close()
print("  [OK] 5) 非正定抛 CholmodNotPositiveDefinite")

# ---- 6) 真实网格 J_free：与 superlu 相对差 < 1e-8 ----
root = Path.cwd()
mesh = root / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh"
if mesh.exists():
    from motorfem.config import RunConfig
    from motorfem.fem.system import FieldSystem
    from motorfem.fem.em_worker import density_from_x, _build_em_elimination
    from motorfem.kernels.fast import newton as fast_nt
    from motorfem.kernels.fast.newton import field_error_vectorized
    from motorfem.kernels.fast.material import CachedMaterial

    cfg = RunConfig.legacy()
    csv = root / "legacy" / "pytorch" / cfg.material.bh_curve_csv
    fs = FieldSystem(cfg, str(mesh), str(csv))
    top, pre, material = fs.top, fs.pre, fs.material
    rho = density_from_x(top.rho_SIMP_initial[list(top.element_topo_opti_m)], top)
    i_t = pre.time_span.size // 2
    kv = pre.K_vectorized_vec[i_t]
    elim = _build_em_elimination(kv, top, cfg.solver.EM_MS_boundary_A_eq)
    A_sol, B, J = fast_nt.solve_field_vectorized(
        rho, kv, pre.R_vec[i_t], _A0=None, _solve_abs_err=1e-3, max_iter=100,
        iter_relaxation=20, relaxation_ratio=0.5, output=False,
        n_node=top.n_node, area=top.em.area, material=material, elimination=elim)
    cached = CachedMaterial(material.curve, material.simp, material.mu0,
                            material.mur_linear)
    state = cached.prepare_density(rho)
    F, _, _ = field_error_vectorized(A_sol, kv, pre.R_vec[i_t], n_node=top.n_node,
                                     area=top.em.area, cached=cached, state=state)
    bf = elim.eliminate_rhs(F)
    J_free = J.tocsc()
    ctx = CholmodContext()
    try:
        ctx.factorize(J_free)
        xc = ctx.solve(bf)
    finally:
        ctx.close()
    xl = splu(J_free).solve(bf)
    rel = np.linalg.norm(xc - xl) / np.linalg.norm(xl)
    assert rel < 1e-8, f"真实 J_free 相对差 {rel:.3e}"
    assert np.linalg.norm(J_free @ xc - bf) / np.linalg.norm(bf) < 1e-10
    print(f"  [OK] 6) 真实 J_free vs superlu 相对差 {rel:.2e}")
else:
    print("  [SKIP] 6) 真实网格缺失")

print("SUB OK")
'''


def test_cholmod_backend_in_subprocess():
    """全部断言在子进程里跑（隔离 DLL/PATH 污染，见模块 docstring）。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, "-c", _SUB_BODY],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, cwd=str(_ROOT), timeout=900,
    )
    assert r.returncode == 0, (
        f"子进程退出码 {r.returncode}\n--- stdout ---\n{r.stdout[-2000:]}\n"
        f"--- stderr ---\n{r.stderr[-3000:]}")
    assert "SUB OK" in r.stdout, r.stdout[-2000:]


# =============================================================================
# 符号因子**多槽位**缓存（2026-09-29）
# =============================================================================
#
# 背景（实测）：真实 Jacobian 的稀疏模式**逐时间点不同**（11 个时间点 = 11 个唯一模式），
# 而 MMA 每步都要把所有时间点走一遍。单槽位缓存 ⇒ 每步每个时间点都重做一次
# ``cholmod_l_analyze``；改成按签名缓存多个后第 2 步起 analyze 降到 0。
# 实测 5/5 轮一致：**1369.4 → 1284.7 ms/步（1.066×）**，且 3 步 MMA 的解 ``np.array_equal``。
#
# 本测试锁两件事：
#   ① analyze 次数 == **模式数**（不是 factorize 次数）；
#   ② ``close()`` 不 double free —— 缓存里已含当前因子，再 free 一次会把手
#      CHOLMOD 堆搞坏（实测表现为子进程 0xC0000374 崩溃，正是本文件抓到的）。

_CACHE_BODY = r'''
import numpy as np
import scipy.sparse as sp

from motorfem.cholmod_ctypes import CholmodContext

rng = np.random.default_rng(0)
N = 300


def make(seed):
    A = sp.random(N, N, density=0.02, random_state=seed, format="csr")
    A = (A + A.T + sp.eye(N) * (N + 1)).tocsc()      # 严格对称正定
    return A


ctx = CholmodContext()
try:
    pats = [make(s) for s in (1, 2, 3)]
    for _pass in range(4):                            # 轮着来 4 遍
        for P in pats:
            ctx.factorize(P)
            b = np.asarray(P.sum(axis=1)).ravel()
            x = ctx.solve(b)
            rel = np.linalg.norm(P @ x - b) / np.linalg.norm(b)
            assert rel < 1e-10, f"相对残差 {rel:.3e}"
    assert ctx.n_analyze == len(pats), (
        f"符号分析 {ctx.n_analyze} 次，应为 {len(pats)}（= 模式数）—— 多槽位缓存失效")
    assert ctx.n_sym_hit == 4 * len(pats) - len(pats), (
        f"缓存命中 {ctx.n_sym_hit} 次，应为 {4 * len(pats) - len(pats)}")
finally:
    ctx.close()                                       # ← double free 会在这里崩
print("CACHE OK")
'''


def test_symbolic_cache_is_keyed_by_pattern():
    """符号因子按**模式签名**缓存多个；``close()`` 不 double free。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-c", _CACHE_BODY],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=str(_ROOT), timeout=600)
    assert r.returncode == 0, (
        f"子进程退出码 {r.returncode}（0xC0000374 = 堆损坏 ⇒ 多半是 double free）\n"
        f"--- stdout ---\n{r.stdout[-2000:]}\n--- stderr ---\n{r.stderr[-2000:]}")
    assert "CACHE OK" in r.stdout, r.stdout[-2000:]
