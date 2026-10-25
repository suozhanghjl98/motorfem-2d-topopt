"""tests/conftest.py —— 位级一致的环境（golden 比对的前提）

golden 是在 ``OMP/OPENBLAS/MKL_NUM_THREADS=1`` 下采集的（``tools/golden_capture.py``
顶部）。凡与 golden 做 ``np.array_equal`` 的测试都必须跑在同样的单线程环境里
（多线程 BLAS 改变归约顺序 ⇒ 末位不同 ⇒ 假失败）。

本文件在 pytest 收集测试模块**之前**导入，且先于测试模块里的 ``import numpy``。
⚠ 若 pytest 插件（如 hypothesis）已先行导入 numpy，此处设置将不生效 —— 届时应把
golden 对比测试改成「子进程跑 ``tools/verify_golden.py`` 断言退出码」，而非在本
进程内重算。
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"


# =============================================================================
# 基线目录选择：golden（legacy 冻结基线） vs golden2（motorfem 当前口径）
# =============================================================================
#
# ``golden/`` 是 **legacy** 的位级基线，永远冻结、**不可改写**。
# 当 motorfem 有意偏离 legacy（例如 D13 补真空项、D14 掩膜修正）时，
# 那一层的结果不能再与 ``golden/`` 逐位一致 —— 于是另采 ``golden2/``
# 作为「当前 motorfem 口径」的基线，**两套并存**：
#
#   MOTORFEM_BASELINE=golden   （默认）走 legacy 基线 —— L1/L2 仍应逐位通过；
#   MOTORFEM_BASELINE=golden2           走 motorfem 基线 —— L3/L4 走新口径。
#
# 测试里一律用 ``baseline_path("<文件名>")`` 取路径，**不要**再写死
# ``_REPO_ROOT / "golden" / ...``。
import pathlib as _pathlib


def baseline_dir() -> _pathlib.Path:
    """当前基线目录（由 ``MOTORFEM_BASELINE`` 决定，默认 ``golden``）。"""
    name = os.environ.get("MOTORFEM_BASELINE", "golden")
    if name not in ("golden", "golden2"):
        raise ValueError(
            f"MOTORFEM_BASELINE={name!r} 无效，只能是 'golden' 或 'golden2'")
    return _pathlib.Path(__file__).resolve().parents[1] / name


def baseline_path(filename: str) -> _pathlib.Path:
    """基线目录下的某个文件路径（不检查存在性）。"""
    return baseline_dir() / filename


import pytest as _pytest


@_pytest.fixture(scope="session")
def baseline() -> str:
    """当前基线目录名（``golden`` / ``golden2``），由 ``MOTORFEM_BASELINE`` 决定。"""
    return baseline_dir().name


@_pytest.fixture(scope="session")
def baseline_file():
    """返回 ``name -> Path`` 的取文件函数（见 ``baseline_path``）。"""
    return baseline_path
