"""CHOLMOD（SuiteSparse 稀疏 Cholesky）的 ctypes 绑定 —— 让 Python 3.9 也能用。

为什么是 ctypes（2026-09-30 实测）
--------------------------------
- ``scikit-sparse 0.5.0`` 要求 **Python >= 3.10**，而本项目是 3.9.13 ⇒ 装不进项目环境；
- 但 ``cholmod.dll`` 是**纯 C 库、与 Python 版本无关** —— 实测在 3.9 里
  ``ctypes.CDLL`` 加载并跑通 ``analyze`` / ``factorize`` / ``solve``。

收益（真实 J_free：4815²、nnz 34397、消元后对称，BLAS 单线程 / AC）
----------------------------------------------------------------
| 路径 | 时间 |
|---|---|
| SuperLU ``splu``（生产） | 14.32 ms |
| CHOLMOD ``analyze``（符号） | 2.12 ms（**每个非零模式只需一次**） |
| CHOLMOD ``factorize``（数值） | 2.05 ms（复用符号）⇒ **6.99×** |
| CHOLMOD ``solve`` | 0.19 ms |

正确性：残差 3.5e-14、与 SuperLU 两解相对差 5.0e-14。

★ **符号复用的依据**：``cholmod_l_factorize`` 的官方语义是「用 ``cholmod_l_analyze``
得到的因子，对**相同非零模式**的另一个矩阵再次分解」（``cholmod.h`` 原文：
"The analysis can be re-used simply by calling this routine a second time with
another matrix. A must have the same nonzero pattern..."）。Newton 每轮 ``J`` 的
结构不变，正好命中 ⇒ 每轮只付数值分解。

★ **2026-09-29：符号缓存改成「按签名多槽位」**（原来是单槽位，一个 ``self._L``）。
实测：真实 Jacobian 的稀疏模式**逐时间点不同**（11 个时间点 = 11 个唯一模式），
而 MMA 每步都要把所有时间点走一遍 ⇒ 单槽位会让**每步每个时间点都重做 analyze**
（1 步 189 次 ``factorize`` 里 15~20 次 ``analyze``，每次 ~5 ms）。多槽位后第 2 步起
analyze 降到 0。**实测 5/5 轮一致：1369.4 → 1284.7 ms/步（1.066×）**；3 步 MMA 的解
``np.array_equal``（analyze 只定排列，模式相同 ⇒ 排列相同 ⇒ 数值分解完全一样）。

⚠ **``close()`` 里不能二次 free 当前因子**：缓存里**已经包含** ``_L``，
释放完缓存再 ``free_factor(self._L)`` 是 double free，会崩 CHOLMOD 堆
（实测子进程退出码 ``0xC0000374`` = STATUS_HEAP_CORRUPTION）。
回归测试 ``tests/kernels/test_cholmod_backend.py::test_symbolic_cache_is_keyed_by_pattern``。

⚠ **多线程是负优化**（实测）：``OMP_NUM_THREADS=8`` 让 ``factorize`` 从 2.23 ms
**涨到 5.75 ms**（2.6× 慢）—— 本问题规模（4815²/34k）太小，线程开销盖过 Level-3 BLAS
的收益。**保持单线程。**

⚠ 三个必须知道的点
------------------
1. **DLL 目录必须进进程 PATH**：CHOLMOD 的 BLAS 依赖（``libblas`` → ``mkl_rt.3``）
   是**延迟加载**的，``os.add_dll_directory`` 覆盖不到它；不设 PATH 会在
   ``factorize`` 时崩 ``0xc06d007f``（DELAYLOAD_FAILURE；REPL 里表现为 exit 127 零输出）。
2. **副作用（重要）**：把 CHOLMOD 的 DLL 目录放进 PATH 会让**同一进程**的
   scipy / SuperLU 也改用 MKL（实测 ``splu`` 14.3 → 48 ms）。所以**只在显式选择
   ``solver_backend='cholmod'`` 时加载**；默认（superlu）路径不加载、不改 PATH。
3. **必须钉住 MKL 的串行 threading layer**（``load_lib`` 里 ``setdefault``）：MKL
   的线程层会加载**第二份** ``libiomp5md.dll``，而 torch 在 Windows 上自带一份同名的
   ⇒ 两份 Intel OpenMP 同进程 ⇒ **堆损坏、在 ``cholmod_l_factorize`` 里 abort**
   （实测：`tests/kernels tests/autograd` 必崩，改串行层后 671 passed）。CHOLMOD
   本来就钉死单线程（多线程是负优化，见上），**串行层零性能代价**。

⚠ 数值口径
----------
CHOLMOD 在 ``stype != 0`` 时**只读对应三角**（``cholmod.h``：「entries not in the
correct triangular part are ignored」）。本项目的 ``J_free`` 消元后对称到
``max|J-Jᵀ| ~5e-10``，取三角的影响实测两解相对差 ~5e-14 ⇒ 与 SuperLU **不逐位
一致**，属**显式后端选择**（与 ``pardiso`` 同类），golden 需单独标定。

⚠ 性能口径：传**完整 CSC** + ``stype=1``（不 ``triu``）—— ``triu`` 要 0.61 ms，
而 CHOLMOD 自己会忽略下三角 ⇒ 每轮附加成本只剩 ``tocsc`` 0.09 + ``sort`` 0.03 ms。
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import scipy.sparse as sp

__all__ = ["CholmodContext", "CholmodNotPositiveDefinite", "CHOLMOD_AVAILABLE",
           "default_dll_dir", "load_lib"]

# ---------------- cholmod.h 常量（SuiteSparse 5.3.1） ----------------
CHOLMOD_PATTERN, CHOLMOD_REAL, CHOLMOD_COMPLEX, CHOLMOD_ZOMPLEX = 0, 1, 2, 3
CHOLMOD_DOUBLE, CHOLMOD_SINGLE = 0, 4
CHOLMOD_INT, CHOLMOD_LONG = 0, 2
CHOLMOD_A, CHOLMOD_LDLt, CHOLMOD_LD, CHOLMOD_DLt = 0, 1, 2, 3

#: 符号因子缓存上限（按签名）。11 个时间点各一个模式 + 伴随/冻结路径的余量。
_MAX_SYMBOLIC = 32

#: cholmod_common 的字节数（不透明缓冲；SuiteSparse 5.x 实测 ~1.5 KB，给 16 KB 富余）
_COMMON_BYTES = 16384


class CholmodSparse(ctypes.Structure):
    """照抄 ``cholmod.h`` 的 ``cholmod_sparse_struct``（字段顺序/类型必须一致）。"""

    _fields_ = [
        ("nrow", ctypes.c_size_t), ("ncol", ctypes.c_size_t), ("nzmax", ctypes.c_size_t),
        ("p", ctypes.c_void_p), ("i", ctypes.c_void_p), ("nz", ctypes.c_void_p),
        ("x", ctypes.c_void_p), ("z", ctypes.c_void_p),
        ("stype", ctypes.c_int), ("itype", ctypes.c_int), ("xtype", ctypes.c_int),
        ("dtype", ctypes.c_int), ("sorted", ctypes.c_int), ("packed", ctypes.c_int),
    ]


class CholmodDense(ctypes.Structure):
    """照抄 ``cholmod.h`` 的 ``cholmod_dense_struct``。"""

    _fields_ = [
        ("nrow", ctypes.c_size_t), ("ncol", ctypes.c_size_t),
        ("nzmax", ctypes.c_size_t), ("d", ctypes.c_size_t),
        ("x", ctypes.c_void_p), ("z", ctypes.c_void_p),
        ("xtype", ctypes.c_int), ("dtype", ctypes.c_int),
    ]


class CholmodNotPositiveDefinite(RuntimeError):
    """矩阵非正定 / 奇异。

    ⚠ CHOLMOD 的判据**不是** ``cholmod_l_factorize`` 的返回值（它对非正定通常仍返回
    成功），而是因子头部的 ``minor``：``cholmod.h`` 原文「If the factorization failed
    because of numerical issues ... then L->minor is the column at which it failed.
    L->minor = n means the factorization was successful.」⇒ 必须读 ``minor``。
    """


class _FactorHead(ctypes.Structure):
    """``cholmod_factor`` 的前两个字段（只需 ``n`` 与 ``minor``）。"""

    _fields_ = [("n", ctypes.c_size_t), ("minor", ctypes.c_size_t)]


class _CommonHead(ctypes.Structure):
    """``cholmod_common`` 的前若干字段（只为设 ``supernodal``；偏移照抄 cholmod.h）。

    ⚠ 实测（2026-09-30，真实 J_free）**显式设 ``supernodal=CHOLMOD_SUPERNODAL``**：
    ① 更快（2.22 vs simplicial 3.79 ms，1.70×）；② 且 ``minor`` 能正确报非正定
    （simplicial 下实测 ``minor`` 恒为 n、**检测不到**，与通常预期相反）。
    默认 ``CHOLMOD_AUTO`` 在中小矩阵上会退化成 simplicial ⇒ 显式钉住 supernodal。
    """

    _fields_ = [
        ("dbound", ctypes.c_double), ("grow0", ctypes.c_double),
        ("grow1", ctypes.c_double), ("grow2", ctypes.c_size_t),
        ("maxrank", ctypes.c_size_t), ("supernodal_switch", ctypes.c_double),
        ("supernodal", ctypes.c_int),
    ]


def default_dll_dir() -> str:
    """CHOLMOD DLL 目录：``CHOLMOD_DLL_DIR`` 环境变量，默认 D:\\cholmod 环境。"""
    return os.environ.get("CHOLMOD_DLL_DIR", r"D:\cholmod\env\Library\bin")


def _dll_path() -> str:
    return os.path.join(default_dll_dir(), "cholmod.dll")


def CHOLMOD_AVAILABLE() -> bool:  # noqa: N802  （保持类常量式命名）
    """DLL 是否存在（**加载**失败只在真正使用时才报错，见 :func:`load_lib`）。"""
    return os.path.exists(_dll_path())


_LIB = None


def load_lib(dll_dir=None):
    """加载 ``cholmod.dll`` 并声明函数签名（进程内缓存一次）。

    ⚠ 会把 ``dll_dir`` **前置进进程 PATH**（延迟加载的 MKL 依赖需要）—— 见模块 docstring。
    """
    global _LIB
    if _LIB is not None:
        return _LIB
    dll_dir = dll_dir or default_dll_dir()
    # ★ 钉住 MKL 的**串行** threading layer（必须在 CDLL 之前，MKL 初始化时读它）。
    #   原因：MKL 的线程层会拖进 ``libiomp5md.dll``，而 **torch 在 Windows 上自带
    #   一份同名的 iomp** ⇒ 两份 Intel OpenMP 常驻同一进程 ⇒ 堆损坏、在
    #   ``cholmod_l_factorize``（MKL BLAS 内部）直接 abort。实测（2026-10-13）：
    #   ``pytest tests/kernels tests/autograd`` 不加必崩（17 个用例即死），
    #   加串行层后 671 passed / 1 xfailed。CHOLMOD 本就钉死单线程（线程是负优化，
    #   见模块 docstring）⇒ **串行层零性能代价**。
    #   ``setdefault``：用户显式指定（或 MKL 已在别处初始化）时不覆盖。
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    # ★ 延迟加载的 BLAS 依赖走传统搜索顺序 ⇒ 必须进 PATH（add_dll_directory 无效）
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)
    lib = ctypes.CDLL(os.path.join(dll_dir, "cholmod.dll"))

    lib.cholmod_l_start.argtypes = [ctypes.c_void_p]
    lib.cholmod_l_start.restype = None
    lib.cholmod_l_finish.argtypes = [ctypes.c_void_p]
    lib.cholmod_l_finish.restype = ctypes.c_int
    lib.cholmod_l_analyze.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cholmod_l_analyze.restype = ctypes.c_void_p
    lib.cholmod_l_factorize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.cholmod_l_factorize.restype = ctypes.c_int
    lib.cholmod_l_solve.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p]
    lib.cholmod_l_solve.restype = ctypes.c_void_p
    lib.cholmod_l_free_factor.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cholmod_l_free_factor.restype = ctypes.c_int
    lib.cholmod_l_free_dense.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cholmod_l_free_dense.restype = ctypes.c_int
    _LIB = lib
    return lib


def _pack_sparse(A_csc):
    """把对称 CSC 打包成 ``cholmod_sparse``（``stype=1``＝上三角语义，完整 CSC 可传）。"""
    A = A_csc.tocsc()
    if not A.has_sorted_indices:
        A = A.copy()
        A.sort_indices()
    n = A.shape[0]
    p = np.ascontiguousarray(A.indptr, dtype=np.int64)
    i = np.ascontiguousarray(A.indices, dtype=np.int64)
    x = np.ascontiguousarray(A.data, dtype=np.float64)
    S = CholmodSparse()
    S.nrow, S.ncol, S.nzmax = n, n, x.size
    S.p = p.ctypes.data_as(ctypes.c_void_p)
    S.i = i.ctypes.data_as(ctypes.c_void_p)
    S.nz = None
    S.x = x.ctypes.data_as(ctypes.c_void_p)
    S.z = None
    S.stype = 1
    S.itype = CHOLMOD_LONG
    S.xtype = CHOLMOD_REAL
    S.dtype = CHOLMOD_DOUBLE
    S.sorted = 1
    S.packed = 1
    return S, (p, i, x)      # 保持 buffer 存活


def _pack_dense(vec):
    v = np.ascontiguousarray(vec, dtype=np.float64)
    D = CholmodDense()
    D.nrow, D.ncol, D.nzmax, D.d = v.size, 1, v.size, v.size
    D.x = v.ctypes.data_as(ctypes.c_void_p)
    D.z = None
    D.xtype = CHOLMOD_REAL
    D.dtype = CHOLMOD_DOUBLE
    return D, v


class CholmodContext:
    """一个 ``cholmod_common`` + **符号因子缓存**（结构不变则只做数值分解）。

    接口与 ``solvers.SolverSession`` 对齐（``factorize`` / ``solve`` / ``close``），
    但**因子是单例的** —— ``factorize`` 会覆盖上一次的数值因子（与 pardiso 同款；
    需要并存的因子请用 ``_CholmodFactorView``，它在 ``solve`` 前按矩阵身份重分解）。
    """

    def __init__(self, dll_dir=None):
        self._lib = load_lib(dll_dir)
        self._common = ctypes.create_string_buffer(_COMMON_BYTES)
        self._lib.cholmod_l_start(self._common)
        # 显式钉住 supernodal：更快，且 minor 能报非正定（见 _CommonHead docstring）
        _CommonHead.from_buffer(self._common).supernodal = 2
        self._L = None
        self._sig = None          # 当前因子的结构签名
        self._last_A = None       # 最近一次 factorize 的矩阵（身份比较用）
        self._keep = None
        self._sym_cache = {}      # 签名 -> (符号因子指针, 保活缓冲)
        self.n_analyze = 0        # 诊断：真正做了几次符号分析
        self.n_sym_hit = 0        # 诊断：几次命中符号缓存

    # ---------------------------------------------------------------- 内部
    def _free_factor(self, L) -> None:
        if L:
            lib_L = ctypes.c_void_p(L)
            self._lib.cholmod_l_free_factor(ctypes.byref(lib_L), self._common)

    def _reanalyze(self, A_csc, sig):
        """新模式：做一次符号分析并**按签名缓存**（不是覆盖上一个）。"""
        S, keep = _pack_sparse(A_csc)
        L = self._lib.cholmod_l_analyze(ctypes.byref(S), self._common)
        if not L:
            raise RuntimeError("cholmod_l_analyze 返回 NULL")
        if len(self._sym_cache) >= _MAX_SYMBOLIC:
            # 真 FIFO（dict 保序）：淘汰**最旧**的一项。绝不淘汰当前因子 ——
            # 那样 `solve` 会用到已释放的指针。
            for _old in list(self._sym_cache):
                if _old != self._sig:
                    _oL, _ok = self._sym_cache.pop(_old)
                    self._free_factor(_oL)
                    break
        self._sym_cache[sig] = (L, keep)
        self.n_analyze += 1
        self._L = L
        self._keep = keep
        self._sig = sig

    # ---------------------------------------------------------------- 公开
    def factorize(self, A_csc) -> None:
        """数值分解（结构签名不变时**复用符号分析**）。

        ★ **符号按签名缓存多个**（不只一个槽位）。实测：真实 Jacobian 的稀疏模式
        **逐时间点不同**（11 个时间点 = 11 个唯一模式），而 MMA 每步都会把所有时间点
        走一遍 ⇒ 单槽位缓存会让**每步每个时间点都重做一次 analyze**
        （实测 1 步 189 次 factorize 里 12 次 analyze，× ~3.15 ms）。
        多槽位后第 2 步起 analyze 降到 ~0（逐位一致：analyze 只定排列，
        模式相同 ⇒ 排列相同 ⇒ 数值分解完全一样）。
        """
        A = A_csc.tocsc()
        sig = (A.shape, A.indptr.tobytes(), A.indices.tobytes())
        if sig != self._sig:
            ent = self._sym_cache.get(sig)
            if ent is None:
                self._reanalyze(A, sig)         # 新模式：analyze + 入缓存
            else:
                _L, _keep = ent                         # 命中：**0 次 analyze**
                self._L, self._keep, self._sig = _L, _keep, sig
                self.n_sym_hit += 1
        S, keep = _pack_sparse(A)
        ok = self._lib.cholmod_l_factorize(ctypes.byref(S), self._L, self._common)
        if not ok:
            raise CholmodNotPositiveDefinite(
                f"cholmod_l_factorize 返回 0（shape={A.shape}）")
        # ★ 非正定的真正判据：因子头部的 minor != n（见 CholmodNotPositiveDefinite）
        minor = ctypes.cast(self._L, ctypes.POINTER(_FactorHead)).contents.minor
        if minor != A.shape[0]:
            raise CholmodNotPositiveDefinite(
                f"矩阵非正定：在第 {minor} / {A.shape[0]} 列失败")
        self._keep = keep
        self._last_A = A_csc

    def solve(self, b) -> np.ndarray:
        """解当前因子（须先 ``factorize``）。"""
        B, keepB = _pack_dense(np.asarray(b))
        X = self._lib.cholmod_l_solve(CHOLMOD_A, self._L, ctypes.byref(B), self._common)
        if not X:
            raise RuntimeError("cholmod_l_solve 返回 NULL")
        Xd = ctypes.cast(X, ctypes.POINTER(CholmodDense)).contents
        nrow = Xd.nrow
        out = np.array(np.ctypeslib.as_array(
            ctypes.cast(Xd.x, ctypes.POINTER(ctypes.c_double)), shape=(nrow,)))
        # 释放 CHOLMOD 分配的 dense（x 的 buffer 是它自己的）
        px = ctypes.c_void_p(X)
        self._lib.cholmod_l_free_dense(ctypes.byref(px), self._common)
        return out

    def close(self) -> None:
        # ⚠ 缓存里**已经包含当前因子**（`_L` 永远是某个缓存项）⇒ 释放完缓存后
        #   只能把 `_L` 置空，**不能再 free 一次**（double free 会把 CHOLMOD 堆搞坏，
        #   实测表现为子进程 0xC0000374 崩溃）。
        for _sig, (L, _keep) in list(getattr(self, "_sym_cache", {}).items()):
            self._free_factor(L)
        if getattr(self, "_sym_cache", None) is not None:
            self._sym_cache.clear()
        self._L = None
        self._sig = None
        self._keep = None
        if getattr(self, "_common", None) is not None:
            try:
                self._lib.cholmod_l_finish(self._common)
            except Exception:  # noqa: BLE001
                pass
            self._common = None

    def __del__(self):  # noqa: D105
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
