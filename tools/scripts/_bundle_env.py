# -*- coding: utf-8 -*-
"""_bundle_env.py —— 提交包内的**运行环境适配层**。

只做一件事：把 ``RunConfig()`` 的默认值调整到**本机实际能跑**的配置。
它**不改变任何数值算法**，只选择后端：

| 条件 | 处理 |
|---|---|
| 有 CHOLMOD（``CHOLMOD_DLL_DIR`` 指向可用 DLL） | 保持 ``cholmod``（最快） |
| 没有 CHOLMOD | 回落 ``superlu``（仍与 golden 逐位一致） |
| 没有 numba | ``bh_lut_n=0``（B-H 改走 PCHIP 精确求值） |
| ``n_jobs`` 超过物理核数 | 压到 ``os.cpu_count()`` |

⚠ 为什么包内默认不带 CHOLMOD
------------------------------
``cholmod.dll`` 依赖 ``libblas.dll`` → ``mkl_rt.3.dll``，而 MKL 是
**x86-64 专用**、且运行时按 CPU 指令集（SSE/AVX2/AVX-512）dispatch。
把它打进包会让体积从 15 MB 涨到 180 MB，且换平台就失效。
⇒ 依赖写进 ``requirements.txt``，包内默认走 superlu；性能数字用开发机日志证明。

用法::

    from _bundle_env import make_config
    cfg = make_config()            # 自动选择后端
    cfg = make_config(prefer="superlu")   # 强制
"""
from __future__ import annotations

import os

__all__ = ["make_config", "apply_backend", "describe"]


def _cholmod_ok() -> bool:
    try:
        from motorfem.cholmod_ctypes import CHOLMOD_AVAILABLE
        return bool(CHOLMOD_AVAILABLE())
    except Exception:
        return False


def _numba_ok() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def apply_backend(cfg, prefer: str | None = None) -> str:
    """就地调整 ``cfg`` 的后端字段，返回实际采用的求解后端名。

    Parameters
    ----------
    cfg : RunConfig
        会被就地修改（只改 ``solver`` / ``material`` / ``runtime`` 三个子配置）。
    prefer : ``"auto"`` | ``"cholmod"`` | ``"superlu"`` | None
        None 时读环境变量 ``MOTORFEM_SOLVER``，再缺省为 ``"auto"``。
    """
    if prefer is None:
        prefer = os.environ.get("MOTORFEM_SOLVER", "auto").strip().lower()

    if prefer == "auto":
        prefer = "cholmod" if _cholmod_ok() else "superlu"

    if prefer == "cholmod":
        if not _cholmod_ok():
            print("[bundle_env] 请求 cholmod 但 DLL 不可用，回落 superlu")
            prefer = "superlu"
        else:
            cfg.solver.solver_backend = "cholmod"
    else:
        cfg.solver.solver_backend = "superlu"

    # superlu 路径要关掉 cholmod 专属的两次归约融合（它差浮点末位，非逐位）
    if prefer == "superlu":
        cfg.solver.csr_fuse = False

    if not _numba_ok():
        # ⚠ RunConfig() 默认 bh_lut_n=65536，缺 numba 时 approx/bh_lut 会 ImportError
        if getattr(cfg.material, "bh_lut_n", 0):
            cfg.material.bh_lut_n = 0
            print("[bundle_env] 无 numba，B-H 查表关闭（改走 PCHIP，结果更精确）")

    n = os.cpu_count() or 1
    if getattr(cfg.runtime, "n_jobs", None) and cfg.runtime.n_jobs > n:
        cfg.runtime.n_jobs = n
    return prefer


def make_config(prefer: str | None = None):
    """构造一个适配当前机器的 ``RunConfig``。"""
    from motorfem.config import RunConfig
    cfg = RunConfig()
    apply_backend(cfg, prefer=prefer)
    return cfg


def describe(cfg) -> str:
    """一行环境摘要，写进日志便于复核。"""
    import platform
    return (
        f"backend={cfg.solver.solver_backend} "
        f"csr_fuse={cfg.solver.csr_fuse} "
        f"n_jobs={cfg.runtime.n_jobs} "
        f"bh_lut_n={cfg.material.bh_lut_n} "
        f"cpus={os.cpu_count()} "
        f"{platform.system()}-{platform.machine()}"
    )
