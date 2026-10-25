"""线性求解器抽象层：可插拔后端（superlu 默认 / pardiso 可选）。

对应阶段 8 的求解链，把「解线性系统」从 scipy 解耦：

- ``solve(A, b, solver=None)``：一次求解（分解 + 回代）。
- ``factorize(A, solver=None)``：分解，返回 factor 对象，接口 ``.solve(rhs)``
  —— 与 SuperLU 因子一致，阶段 8a 的分解复用路径直接可用。

``solver`` 参数接受 ``None`` | ``'superlu'`` | ``'pardiso'``。

后端
----
- ``'superlu'``（默认）：``scipy.sparse.linalg``。**零包装**：直接转发 ``spsolve`` /
  ``splu``，保留 scipy 的格式语义 —— CSR 走 SuperLU 的 CSR 路径、CSC 走 splu 的
  CSC 路径（阶段 8a 的关键边界：只有 CSC 才逐位一致，本层**不做**任何格式转换）⇒
  默认路径与生产现状**逐位一致**，golden 不动。
- ``'pardiso'``：MKL Pardiso（``pypardiso``，需 ``pip install mkl pypardiso``）。
  **只用 ``mtype=11``（非对称 LU）**。⚠ 本程序的主从/驱动行有 O(1) 绝对不对称
  （D11 残余），Pardiso 对称模式（mtype=2）按上三角对称补齐会解**另一个系统**
  （``probe_pardiso.py``：残差 J 3.1e-3 / K_mech 7.5e-8）⇒ 生产路径禁止对称模式，
  真正利用对称要等主从消元（设计文档 D11）。
  ⚠ **pypardiso 的关键坑（2026-09-24 实测根因，见 ``probe_pardiso.py`` docstring）**：
  ① ``PyPardisoSolver()`` **每次构造都递归 glob 全盘**（``<sys.prefix>/lib/**/mkl_rt*``，
  实测 ~1.9s/次，与 MKL 无关）——本模块在 import 前设 ``PYPARDISO_MKL_RT`` 绕过
  （定位一次、缓存 env），构造降到 ~0.1ms；② 模块级 ``spsolve(matrix_type=…)``
  静默失效；③ 对称 mtype 必须传上三角（满矩阵 OSError）；④ **多实例存活 MKL
  OpenMP 打架**。⇒ 生产路径**必须用 ``SolverSession``**（一次构造多次求解 + 用完
  ``free_memory()``——注意要在 factorize **之后**调，空实例直接 free 会报 error -1）。

⚠ 数值性质：superlu 与 pardiso 的**解不逐位一致**（不同引擎）。切换后端即改变
数值行为 —— 这是显式选择（``RunConfig.solver.solver_backend``），不是重构默认。
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Optional

from scipy import sparse as sp_sparse
from scipy.sparse import linalg as _sp_linalg

__all__ = ["solve", "factorize", "SUPERLU", "PARDISO", "CHOLMOD"]

SUPERLU = "superlu"
PARDISO = "pardiso"
CHOLMOD = "cholmod"      # SuiteSparse 稀疏 Cholesky（经 ctypes，见 motorfem/cholmod_ctypes.py）

#: 零包装：直接引用 scipy 的函数（不包 lambda/装饰器，避免任何间接层的浮点风险）
_spsolve = _sp_linalg.spsolve
_splu = _sp_linalg.splu


def _resolve_backend(solver: Optional[str]) -> str:
    backend = solver if solver is not None else SUPERLU
    if backend not in (SUPERLU, PARDISO, CHOLMOD):
        raise ValueError(
            f"未知求解后端 {backend!r}；可选 'superlu'（默认）| 'pardiso' | 'cholmod'")
    return backend


# =====================================================================
# pypardiso 库定位的递归 glob 绕过（2026-09-24 实测根因）
# =====================================================================

_RT_GUIDED = False


def _ensure_pardiso_rt_env() -> None:
    """定位 ``mkl_rt`` 并设 ``PYPARDISO_MKL_RT``，跳过 pypardiso 的递归 glob。

    ★ 根因（2026-09-24 实测）：``PyPardisoSolver()`` 在 ``find_library('mkl_rt')``
    返回 ``None``（Windows）时走 ``glob.glob('<prefix>/lib/**/mkl_rt*',
    recursive=True)`` —— **每次构造**递归扫整个 ``site-packages`` 树
    （~几十万文件），实测 ~1.9s/次，与 MKL 无关。pypardiso 的 ``__init__`` 优先
    读 ``PYPARDISO_MKL_RT``（``pardiso_wrapper.py:63-66``），设了即跳过 glob。
    本函数用**一次性** glob 定位路径并缓存进 env，之后构造降到 ~0.1ms。
    """
    global _RT_GUIDED
    if _RT_GUIDED:
        return
    _RT_GUIDED = True
    if os.environ.get("PYPARDISO_MKL_RT"):
        return
    # 窄位置优先（毫秒级命中，避免 `**` 全盘扫 site-packages）；最后全盘兜底。
    candidates = (
        f"{sys.prefix}/Library/bin/mkl_rt*",          # mkl 包在 Windows 的落点
        f"{sys.prefix}/Library/lib/mkl_rt*",
        f"{sys.prefix}/lib/site-packages/mkl/lib/mkl_rt*",
        f"{sys.prefix}/[Ll]ib*/**/*mkl_rt*",          # fallback 全盘
    )
    for pattern in candidates:
        hits = sorted(glob.glob(pattern, recursive=True), key=len)
        if hits:
            os.environ["PYPARDISO_MKL_RT"] = hits[0]
            return


def _load_pypardiso():
    """导入 pypardiso（先做 env 引导）。"""
    _ensure_pardiso_rt_env()
    import pypardiso  # noqa: F401

    return pypardiso


def solve(A, b, solver: Optional[str] = None):
    """解 ``A·x = b``（分解 + 回代）。``solver=None`` ⇒ superlu（生产现状，逐位一致）。

    ⚠ 高频/循环用途（Newton 每步、多时间点）**必须用 ``SolverSession``**：本函数的
    pardiso 路径每次新建 MKL 实例（``pardisoinit`` 实测 ~2s），循环里新建会卡死。
    """
    backend = _resolve_backend(solver)
    if backend == SUPERLU:
        return _spsolve(A, b)
    return _pardiso_solve(A, b)


def factorize(A, solver: Optional[str] = None):
    """分解 ``A``，返回 factor 对象（接口 ``.solve(rhs)``）。

    - superlu：直接返回 ``splu(A)`` 的 SuperLU 因子（与阶段 8a 复用路径同一对象）。
    - pardiso：``_PardisoFactor``（独立实例，构造即分解；用完 ``.close()``）。
      ⚠ 生产伴随路径用 ``SolverSession.factorize``（复用会话实例，只付一次 init）。
    """
    backend = _resolve_backend(solver)
    if backend == SUPERLU:
        return _splu(A)
    return _PardisoFactor(A)


# =====================================================================
# 求解会话：一次构造、多次求解（pardiso 的 init 只付一次）
# =====================================================================

class SolverSession:
    """可插拔求解会话 —— Newton 循环/多时间点/伴随解共用一个实例。

    构造一次（pardiso 时 ``PyPardisoSolver(mtype=11)`` 初始化一次，实测 ~2s），
    之后多次 ``solve``：矩阵变了重新 ``factorize``、不变直接回代。

    接口
    ----
    ``solve(A, b)``       解 ``A·x=b``（矩阵变了自动重新分解）。
    ``factorize(A)``      返回 factor 对象（接口 ``.solve(rhs)``）：superlu →
                          ``splu(A)``（独立因子）；pardiso → ``_PardisoFactorView``
                          （**引用本会话的实例**，不重复 init，也不释放 —— 会话
                          由持有者 ``close()``）。⚠ 会话在 factorize 之后到 factor
                          用完期间**不能再 solve 别的矩阵**（因子会被覆盖）。
    ``close()``           释放 MKL 内存（pypardiso 无析构，必须显式调用）。
    """

    def __init__(self, solver: Optional[str] = None, mtype: int = 11):
        self.backend = _resolve_backend(solver)
        # ⚠ mtype 只在 pardiso 后端生效：11 = 非对称 LU；2 = 实对称正定 LDLᵀ/Cholesky
        #   （**要求矩阵真对称** —— 消元后（`motorfem/elimination.py`）才可用；
        #   对称模式按「只存上三角」解释 CSR，本会话对 mtype=2 自动 ``sp.triu``）。
        self.mtype = 2 if (self.backend == PARDISO and mtype == 2) else 11
        self._pardiso = None
        self._chol = None
        self._last_A = None
        if self.backend == PARDISO:
            pypardiso = _load_pypardiso()  # 惰性 + env 引导（跳过递归 glob）

            self._pardiso = pypardiso.PyPardisoSolver(mtype=self.mtype)
        elif self.backend == CHOLMOD:
            # SuiteSparse 稀疏 Cholesky（ctypes）。⚠ 只在显式选 cholmod 时加载 ——
            # 加载会把 DLL 目录前置进 PATH，影响同进程的 scipy/SuperLU（见模块 docstring）。
            from .cholmod_ctypes import CholmodContext

            self._chol = CholmodContext()

    def _maybe_triu(self, A):
        if self.mtype == 2:
            return sp_sparse.triu(A, format="csr")
        return A

    def solve(self, A, b):
        if self.backend == SUPERLU:
            return _spsolve(A, b)
        if self.backend == CHOLMOD:
            if self._last_A is not A:
                self._chol.factorize(A)
                self._last_A = A
            return self._chol.solve(b)
        A = self._maybe_triu(A)
        if self._last_A is not A:
            self._pardiso.factorize(A)
            self._last_A = A
        return self._pardiso.solve(A, b)

    def factorize(self, A):
        if self.backend == SUPERLU:
            return _splu(A)
        if self.backend == CHOLMOD:
            self._chol.factorize(A)
            self._last_A = A
            return _CholmodFactorView(self._chol, A)
        A = self._maybe_triu(A)
        self._pardiso.factorize(A)
        self._last_A = A
        return _PardisoFactorView(self._pardiso, A)

    def close(self) -> None:
        if self._chol is not None:
            try:
                self._chol.close()
            except Exception:  # noqa: BLE001
                pass
            self._chol = None
        p = self._pardiso
        if p is not None:
            try:
                p.free_memory()
            except Exception:  # noqa: BLE001
                pass
            self._pardiso = None

    def __del__(self):  # noqa: D105
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class _PardisoFactorView:
    """引用 ``SolverSession`` 的 Pardiso 因子：接口与 SuperLU 因子一致（``.solve(rhs)``）。

    不拥有 MKL 实例（会话持有者负责 ``close()``）；``close()`` 是 no-op。
    """

    def __init__(self, solver, A):
        self._solver = solver
        self._A = A

    def solve(self, b):
        return self._solver.solve(self._A, b)

    def close(self) -> None:
        pass


class _CholmodFactorView:
    """引用 ``CholmodContext`` 的 CHOLMOD 因子视图：接口与 SuperLU 因子一致（``.solve``）。

    ⚠ CHOLMOD 的数值因子是**单例**（一次 ``factorize`` 覆盖上一次），而 ``em_step``
    里 frozen 的求解穿插在伴随解之间 ⇒ view 必须能「发现因子被换掉、按矩阵身份重
    分解」（与 ``_PardisoFactorView`` 同款语义；重分解只是数值步，符号仍复用）。
    """

    def __init__(self, ctx, A):
        self._ctx = ctx
        self._A = A

    def solve(self, b):
        if self._ctx._last_A is not self._A:
            self._ctx.factorize(self._A)
        return self._ctx.solve(b)

    def close(self) -> None:
        pass


#: 进程级单例会话（backend, mtype）→ SolverSession。pardiso 的 MKL 实例/运行时
#: 只初始化一次，跨 fun 调用复用（2026-09-24：每次构造虽已绕过递归 glob 降到
#: 0.1ms，但仍是新句柄 —— 单例把「加载」收敛为一次）。
_SESSIONS: dict = {}


def get_session(solver=None, mtype: int = 11):
    """进程级单例会话：``(backend, mtype)`` 维缓存，只构造一次。

    ⚠ 会话**非线程安全**（pardiso 的 MKL 句柄单线程）：并行（``n_jobs>1``）
    必须 ``as_session(..., shared=False)`` 每任务独立，不能共享本单例。
    """
    backend = _resolve_backend(solver)
    eff = 2 if (backend == PARDISO and mtype == 2) else 11
    key = (backend, eff)
    s = _SESSIONS.get(key)
    if s is None:
        s = _SESSIONS[key] = SolverSession(backend, eff)
    return s


def as_session(solver, mtype: int = 11, shared: bool = True):
    """把 ``None`` | 后端名 | ``SolverSession`` 归一成 ``(session, owned)``。

    ``owned=True`` ⇒ 本调用方构造了会话，负责 ``close()``；
    ``owned=False`` ⇒ 外部/单例会话，**不要** ``close()``。

    ``shared=True``（默认）⇒ 进程级单例（``get_session``），MKL 只加载一次、
    跨调用复用；``shared=False`` ⇒ 每调用独立构造（并行 ``n_jobs>1`` 时每个
    任务一个，会话非线程安全）。``mtype`` 只在 pardiso 后端生效（2 = 对称）。
    """
    if isinstance(solver, SolverSession):
        return solver, False
    if shared:
        return get_session(solver, mtype), False
    return SolverSession(solver, mtype), True


# =====================================================================
# Pardiso 后端
# =====================================================================

class _PardisoFactor:
    """MKL Pardiso ``mtype=11`` 因子的适配器：接口与 SuperLU 因子一致。

    构造即 ``factorize``（与 ``splu`` 的语义一致）；``.solve(rhs)`` 回代。
    ⚠ pypardiso 无析构 ⇒ 必须显式 ``.close()``（调 ``free_memory``）释放 MKL
    因子内存；``__del__`` 兜底（时机不可靠，别依赖它）。构造失败时自动释放。
    """

    def __init__(self, A):
        pypardiso = _load_pypardiso()  # 惰性 + env 引导

        self._A = A
        self._solver = pypardiso.PyPardisoSolver(mtype=11)
        try:
            self._solver.factorize(A)
        except BaseException:
            self._release()
            raise

    def solve(self, b):
        return self._solver.solve(self._A, b)

    def close(self) -> None:
        self._release()

    def _release(self) -> None:
        solver = getattr(self, "_solver", None)
        if solver is not None:
            try:
                solver.free_memory()
            except Exception:  # noqa: BLE001
                pass
            self._solver = None

    def __del__(self):  # noqa: D105
        try:
            self._release()
        except Exception:  # noqa: BLE001
            pass


def _pardiso_solve(A, b):
    f = _PardisoFactor(A)
    try:
        return f.solve(b)
    finally:
        f.close()
