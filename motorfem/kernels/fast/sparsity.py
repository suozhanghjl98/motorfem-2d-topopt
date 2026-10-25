"""``fast`` 层的 CSR 快速装配：预计算 scipy ``sort_indices`` 的排列并复用。

背景与结论（2026-09-30 实测，第 10 步末真设计）
---------------------------------------------
``coo_matrix((data, (row, col))).tocsr()`` 每轮 ~1.12 ms（89681 条 → 34579 槽），
内部 = ① ``coo_tocsr``（按行计数排序）② ``sum_duplicates``（``sort_indices``
行内按列插入排序 + 压缩合并）。**实测成本大头是 ② 的 ``sort_indices``（0.78 ms）**，
不是结构分配，所以「只预计算结构」收益很小（1.2×）。

★ **正确且有效的做法**：预计算 **scipy ``sort_indices`` 产生的排列本身**并复用 ——
排序/合并**仍由 scipy 的 ``csr_sum_duplicates`` 做**（顺序左结合累加），只把
「怎么排」预计算掉。实测 **1.12 → 0.26 ms（4.3×）**，与 ``coo.tocsr()`` **逐位一致**
（``np.array_equal(data/indices/indptr)``，多组随机数据 max|Δ| = 0.0）。

三步（全部只依赖 (row, col, n)，与数值、ρ、MMA 步无关）：
1. ``order = np.argsort(row, kind="stable")`` —— 行分桶并保持桶内输入顺序
   （等价 ``coo_tocsr`` 的计数排序）；
2. 用**哑元 data**（``0..nnz-1``）跑一次 scipy 的 ``sort_indices``，从它重排后的
   data 读出排列 ``perm``，并记住排序后的 ``indices`` —— **这正是 scipy 自己的
   行内排序顺序**，所以后续 ``csr_sum_duplicates`` 的累加顺序与 ``coo.tocsr()``
   完全一致；
3. 合并成一次 gather：``gather = order[perm]``。每次组装只需
   ``data[gather]`` + ``csr_matrix(...)`` + ``sum_duplicates()``。

⚠ **为什么不能用 ``np.lexsort`` / ``bincount`` / ``reduceat`` 自己排+合并**：
实测三条自研路径**都差浮点末位**（max|Δ| 1.9e-9）—— scipy 的 ``_csr_sort_indices``
是**行内插入排序**（与「(row,col) 全局稳定排序」并不等价：lexsort 版实测差
3.6e-15），而 ``np.add.reduceat`` 用的是 pairwise 求和、``bincount`` 的顺序也与
它不同。⇒ **只能复用 scipy 自己的排序排列**，才能既快又逐位一致。

⚠ ``indices`` / ``indptr`` 每次必须 **copy**：``sum_duplicates`` 会**原地**压缩
indices/indptr（``csr_sum_duplicates`` 写回同一数组），复用预计算数组会在第二次
调用时被破坏（实测踩过）。

⚠ 缓存按 ``(id(row), id(col))`` 存，用**引用**校验 id 复用（与 ``em_worker`` 的
消元缓存同款纪律）。

------------------------------------------------------------------------------
★ 可选：CSR 装配的「两次归约融合」（``set_csr_fusion`` / ``cfg.solver.csr_fuse``，默认关）
------------------------------------------------------------------------------
默认路径 = 两次 ``build``（各一次 gather + ``sum_duplicates``）+ 逐元素相加；
融合路径 = 先在 COO 顺序上把两个 data 相加，再 gather/归约**各一次**。实测：

| 方案 | ms | indptr | indices | data |
|---|---:|---|---|---|
| 默认 | 0.400 | — | — | — |
| 融合 | **0.220**（1.82×） | ✓ | ✓ | ✗ ``max|Δ| = 1.86e-9`` |

⇒ **整步 −4.3%**（3/3 轮交替 A/B：−3.0% / −3.5% / −6.4%），但 ``data`` 差浮点末位
（尺度 8.8e6 ⇒ 相对 2e-16）。**默认关** —— 开启后本层不再与 legacy 逐位一致，
golden 比对必须保持关闭。

⚠ 融合还要求两个 ``gather`` 排列相同（``sa.key == sb.key`` 只保证**组装后**的 CSR 结构
相同，**不保证** COO 顺序相同）。生产实测为 True，但**必须显式校验**（``_same_gather``）
—— 否则是**静默算错**，不是末位差。

⚠ 开关是**进程级**（由 ``FieldSystem.__init__`` 从 config 设置、每次构造都重置）。
选进程级而非显式穿参：穿参要改 ``fun`` → ``em_segment`` → ``em_step`` →
``solve_field_vectorized`` → ``field_error_vectorized(_frozen)`` 共十几处签名，
为一个默认关闭的开关不值得。若将来同进程要跑多个不同配置的 FieldSystem，再改成随对象传。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp_sparse

__all__ = [
    "CSRSparsity",
    "get_sparsity",
    "assemble_csr",
    "assemble_csr_sum",
    "DirichletProjection",
    "get_dirichlet_projection",
    "project_dirichlet",
    "set_csr_fusion",
    "csr_fusion_enabled",
    "clear_sparsity_cache",
]


class CSRSparsity:
    """预计算的 ``coo(row, col) → csr`` 组装排列（只依赖模式，与数值无关）。

    Parameters
    ----------
    row, col : (nnz,) 整数数组
        COO 的行/列索引（照抄内核 8 的位置矩阵族）。
    n : int
        方阵阶数。

    Methods
    -------
    ``build(data)``
        用给定数值组装 CSR。返回值与 ``coo_matrix((data, (row, col))).tocsr()``
        **逐位一致**（排序/合并走 scipy 同一份代码、同一种顺序）。
    """

    __slots__ = ("row", "col", "n", "gather", "indices", "indptr", "key")

    def __init__(self, row, col, n: int):
        row = np.asarray(row)
        col = np.asarray(col)
        self.row = row
        self.col = col
        self.n = int(n)

        # ① 行分桶（等价 coo_tocsr：行分桶 + 桶内保持输入顺序）
        order = np.argsort(row, kind="stable")
        indptr = np.searchsorted(row[order], np.arange(self.n + 1)).astype(np.int32)
        col_bucket = col[order].astype(np.int32)

        # ② 用哑元 data 跑一次 scipy 的 sort_indices，读出它产生的排列
        probe = sp_sparse.csr_matrix(
            (np.arange(row.size, dtype=np.float64), col_bucket, indptr),
            shape=(self.n, self.n),
        )
        probe.sort_indices()
        perm = probe.data.astype(np.intp)         # 排序后位置 → 桶序位置
        self.indices = probe.indices.copy()       # scipy 排序后的列
        self.indptr = indptr
        # ③ 合并成一次 gather：data[order][perm] = data[order[perm]]
        self.gather = order[perm]
        # 结构签名（构造时算一次，供 assemble_csr_sum 判断两结构是否可共享）
        self.key = (self.n, self.indices.tobytes(), self.indptr.tobytes())

    def build(self, data) -> sp_sparse.csr_matrix:
        """组装 CSR（与 ``coo.tocsr()`` 逐位一致）。

        ⚠ ``indices`` / ``indptr`` 必须拷贝：``sum_duplicates`` 会原地压缩它们。
        """
        x = sp_sparse.csr_matrix(
            (data[self.gather], self.indices.copy(), self.indptr.copy()),
            shape=(self.n, self.n),
        )
        x.sum_duplicates()      # scipy 的压缩 + 顺序累加（逐位一致的来源）
        return x


#: 结构缓存：``(id(row), id(col)) → (row, col, CSRSparsity)``（引用校验防 id 复用）
_CACHE: dict = {}


def get_sparsity(row, col, n: int) -> CSRSparsity:
    """取/建 ``(row, col)`` 的预计算组装排列（跨 MMA 步复用）。"""
    key = (id(row), id(col))
    ent = _CACHE.get(key)
    if ent is not None and ent[0] is row and ent[1] is col:
        return ent[2]
    s = CSRSparsity(row, col, n)
    _CACHE[key] = (row, col, s)
    return s


def assemble_csr(row, col, data, n: int) -> sp_sparse.csr_matrix:
    """``coo_matrix((data, (row, col))).tocsr()`` 的预计算等价版（逐位一致，~4×）。"""
    return get_sparsity(row, col, n).build(data)


def assemble_csr_sum(row_a, col_a, data_a, row_b, col_b, data_b, n: int):
    """``assemble_csr(a) + assemble_csr(b)`` —— 同结构时走**逐元素相加**快速路径。

    ★ 实测（2026-09-30）：本项目的 ``K`` 与 ``Jabobi2`` 稀疏模式**逐位相同**
    （``K0_row/col`` 与 ``Jabobi2_row/col`` 内容一致；csr 化后 ``indices``/``indptr``
    逐位相同）。此时 scipy 的 ``csr + csr`` 等价于**逐元素 ``data`` 相加**：
    实测 ``max|Δ| = 0.0``，且 **0.095 → 0.008 ms（12×）**。

    ⚠ 结构不同时**自动回退**到通用的「各自 csr + 相加」（仍逐位正确）——
    不会静默出错。

    ★ **可选：两次归约融合成一次**（``set_csr_fusion(True)``，默认关）。
    默认路径是「各自 ``build``（各一次 gather + ``sum_duplicates``）+ 逐元素相加」；
    融合路径改成「先在 COO 顺序上把两个 data 相加，再 gather **一次**、归约**一次**」。
    实测 **0.400 → 0.220 ms（1.82×）**，但**累加顺序改变** ⇒ ``data`` 差浮点末位：

        indptr / indices 逐位相同；data  max|Δ| = 1.863e-09（尺度 8.8e6 ⇒ 相对 2e-16）

    这正是本模块 docstring 里「自研合并必差末位 1.9e-9」那条的老问题 —— 换了个写法
    撞同一堵墙。所以**默认关**，只在明确接受该末位差时打开。
    """
    sa = get_sparsity(row_a, col_a, n)
    sb = get_sparsity(row_b, col_b, n)
    if _FUSE and _same_gather(sa, sb):
        # 融合路径：一次 gather + 一次 sum_duplicates（累加顺序与默认路径不同）
        x = sp_sparse.csr_matrix(((data_a + data_b)[sa.gather],
                                  sa.indices.copy(), sa.indptr.copy()),
                                 shape=(n, n))
        x.sum_duplicates()
        return x
    a = sa.build(data_a)
    b = sb.build(data_b)
    if sa is sb or sa.key == sb.key:
        # 同结构：a/b 的 indices/indptr 逐位相同 ⇒ 逐元素相加 = csr + csr
        return sp_sparse.csr_matrix((a.data + b.data, a.indices, a.indptr),
                                    shape=(n, n))
    return a + b


def clear_sparsity_cache() -> None:
    """清空结构缓存（测试用）。"""
    _CACHE.clear()
    _DP_CACHE.clear()
    _FUSE_OK.clear()


# =============================================================================
# CSR 装配的「两次归约融合」开关
# =============================================================================
#
# **进程级**。由 ``FieldSystem.__init__`` 从 ``cfg.solver.csr_fuse`` 设置 ——
# 每次构造 FieldSystem 都会重置，所以测试之间不会互相污染。
#
# ⚠ 进程级是权衡后的选择：显式穿参要改 ``fun`` → ``em_segment`` → ``em_step`` →
#   ``solve_field_vectorized`` → ``field_error_vectorized`` / ``field_error_vectorized_frozen``
#   共十几处签名（含 frozen 的 4 个内部调用点）。为一个默认关闭的开关动这么多签名面，
#   风险大于收益。生产只有**一个** FieldSystem，进程级可接受；
#   若将来同进程要跑多个不同配置的 FieldSystem，再改成随对象传。
#
# ⚠ 它只影响 ``data`` 的**浮点末位**（相对 2e-16），不改稀疏模式。
#   golden 比对必须保持 False。

_FUSE: bool = False

#: ``(id(sa), id(sb)) -> (sa, sb, gather 是否相同)``（引用校验防 id 复用）
_FUSE_OK: dict = {}


def set_csr_fusion(enabled: bool) -> None:
    """开/关「两次归约融合」。**默认关**（逐位一致）。见 ``assemble_csr_sum``。"""
    global _FUSE
    _FUSE = bool(enabled)


def csr_fusion_enabled() -> bool:
    """当前是否开启融合（诊断 / 测试用）。"""
    return _FUSE


def _same_gather(sa, sb) -> bool:
    """两个 ``CSRSparsity`` 的 ``gather`` 排列是否相同。

    ``sa.key == sb.key`` 只保证**组装后**的 CSR 结构相同，**不保证** ``gather`` 相同
    （两份不同的 COO 顺序可能给出同一个 CSR）。而融合路径要求 ``data_a``/``data_b``
    与 ``sa.gather`` 同一套 COO 顺序 —— 否则是**静默算错**，不是末位差。
    故单独校验，并按 ``(sa, sb)`` 记忆化（``np.array_equal`` 在 89681 长度上是
    O(nnz) 的，不能每轮做）。
    """
    key = (id(sa), id(sb))
    ent = _FUSE_OK.get(key)
    if ent is not None and ent[0] is sa and ent[1] is sb:
        return ent[2]
    ok = (sa is sb) or (sa.key == sb.key and np.array_equal(sa.gather, sb.gather))
    _FUSE_OK[key] = (sa, sb, ok)
    return ok


# =============================================================================
# Dirichlet 投影的预计算版
# =============================================================================
#
# ``(J_driven @ J) @ J_col_mask + J_mat_driven`` 是 Newton 内层 ``N1`` 里**最大的一句**
# （``tools/probe_step_profile.py`` 表 F：0.48 ms/次、占 N1 的 15%、占一步的 6%）。
# 三个算子都是**几何常量**（在 ``kv`` 里，随 ρ 与 MMA 步不变），所以整个操作只有
# 「模式」和「槽位」是变量 ⇒ 可以完全预计算。
#
# 两条实测事实决定实现（探针 ``tools/probe_dirichlet_projection.py``，全部实测）：
#
# 1. ``J_driven`` / ``J_col_mask`` **只存 1、不存 0**（``J_driven`` 4786 个 1 / 331 个 0；
#    ``J_col_mask`` 5087 / 30）⇒ ``csr @ csr`` 让被置零的整行/整列**结构性消失**：
#    ``J`` 34579 → ``Jd@J`` 34429 → ``@Jcm`` 34368 → ``+Jmd`` 35237。
#    **所以「行/列置零」这一步根本不用做** —— 输出模式里没有那些槽位。
# 2. matmul **与** ``csr + csr`` **都会 prune 精确零**（真实 ``J_bulk.data`` 里恰有
#    2 个 0.0，每轮都在）⇒ **输出模式随数值变**，「跑一次真身捕获模式」是错的
#    （实测随机数据下差 2 个槽位）。故模式**自己按 ``row*n+col`` 排序去重算**，
#    prune 放到每次调用末尾统一做。
#
# ⚠ **prune 不能省**：``tests/kernels/test_newton.py`` 对 Jacobi 的 ``data/indices/indptr``
#   做 ``np.array_equal``，``tools/verify_golden.py`` 更是 L1~L4 全部 ``np.array_equal``。
#   留一个显式零就会失配。代价：一次 ``u == 0`` 扫描 + 罕见时的压缩。
#
# 实测（``tools/probe_dirichlet_projection.py``，AC / BLAS 单线程，5117²/nnz 34579）：
#   ``(Jd@J)@Jcm + Jmd`` 0.37~0.43 ms  →  预计算版 0.18~0.21 ms  ⇒ **2.1~2.35×**
#   逐位一致：5/5 随机样本的 indptr/indices/data 全等，下游 TᵀJT 消元也全等。
#   真实 Newton 一轮 15/15 次调用都落进 prune 分支（因为那 2 个零恒在）。


class DirichletProjection:
    """``(J_driven @ J) @ J_col_mask + J_mat_driven`` 的预计算等价实现。

    Parameters
    ----------
    j_driven, j_col_mask : csr
        **纯对角 0/1**（只存 1）。几何常量。
    j_mat_driven : csr
        受约束行上的约束系数。几何常量（**含非对角项**：869 个非零里 840 个不在
        ``J_bulk`` 的模式里 ⇒ 输出模式必须与它求并）。

    Notes
    -----
    构造是懒的：第一次 ``apply`` 才按输入模式算并集模式（一次 O(nnz log nnz) 的
    ``union1d``）。之后每次 ``apply`` 只做定长数值向量上的操作。
    输入模式变化时**自动重建**（不会静默用错映射）。
    """

    __slots__ = ("_jd", "_jcm", "_jmd", "_in_key", "_in_indptr", "_in_indices",
                 "_out_indptr", "_out_indices", "_out_rows",
                 "_gather", "_keep", "_slot_jmd", "_jmd_data", "_nnz_out", "_n")

    def __init__(self, j_driven, j_col_mask, j_mat_driven):
        self._jd = j_driven.tocsr() if sp_sparse.issparse(j_driven) else j_driven
        self._jcm = j_col_mask.tocsr() if sp_sparse.issparse(j_col_mask) else j_col_mask
        self._jmd = j_mat_driven.tocsr() if sp_sparse.issparse(j_mat_driven) else j_mat_driven
        self._n = int(self._jd.shape[0])
        self._in_key = None
        self._in_indptr = None
        self._in_indices = None

    # ---- 真身（回退与测试用） ----
    @staticmethod
    def reference(J, j_driven, j_col_mask, j_mat_driven):
        """原式。逐位比对与模式不符时的回退都走它。"""
        return (j_driven @ J) @ j_col_mask + j_mat_driven

    def _pattern_matches(self, J) -> bool:
        return (self._in_key is not None
                and J.shape == self._in_key
                and np.array_equal(J.indptr, self._in_indptr)
                and np.array_equal(J.indices, self._in_indices))

    def _capture(self, J):
        """按 ``(J_driven, J_col_mask, J_mat_driven)`` 与当前输入模式，算并集模式。"""
        jd_diag = np.asarray(self._jd.diagonal()).ravel()
        jcm_diag = np.asarray(self._jcm.diagonal()).ravel()

        rows_in = np.repeat(np.arange(self._n), np.diff(J.indptr))
        cols_in = J.indices
        surv = (jd_diag[rows_in] != 0.0) & (jcm_diag[cols_in] != 0.0)
        key_in = rows_in[surv].astype(np.int64) * self._n + cols_in[surv]
        mr, mc = self._jmd.nonzero()
        key_md = mr.astype(np.int64) * self._n + mc

        key_all = np.union1d(key_in, key_md)        # 排序 + 去重 ⇒ CSR 规范序
        out_rows, out_cols = np.divmod(key_all, self._n)
        self._out_rows = out_rows
        self._out_indices = out_cols.astype(J.indices.dtype)
        self._out_indptr = np.zeros(self._n + 1, dtype=J.indptr.dtype)
        np.cumsum(np.bincount(out_rows, minlength=self._n), out=self._out_indptr[1:])
        self._nnz_out = int(key_all.size)

        self._keep = surv
        self._gather = np.zeros(J.nnz, dtype=np.intp)
        self._gather[surv] = np.searchsorted(key_all, key_in)
        self._slot_jmd = np.searchsorted(key_all, key_md)
        self._jmd_data = np.asarray(self._jmd[mr, mc]).ravel().astype(np.float64)

        self._in_key = J.shape
        self._in_indptr = J.indptr.copy()
        self._in_indices = J.indices.copy()

    def apply(self, J):
        """与 ``reference`` **逐位一致**（含 prune 精确零）。"""
        if not self._pattern_matches(J):
            self._capture(J)

        u = np.zeros(self._nnz_out, dtype=np.float64)
        u[self._gather[self._keep]] = J.data[self._keep]
        u[self._slot_jmd] += self._jmd_data          # ← J_mat_driven 定长化后的直接求和

        zero = (u == 0.0)
        if not zero.any():
            return sp_sparse.csr_matrix((u, self._out_indices.copy(),
                                         self._out_indptr.copy()),
                                        shape=(self._n, self._n))
        # 罕见路径：与 scipy 同款 prune。indptr 走 O(n) 差分（被删槽位极少）。
        nz = np.flatnonzero(zero)
        rem = np.bincount(self._out_rows[nz], minlength=self._n)
        indptr = self._out_indptr - np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(rem))
        ).astype(self._out_indptr.dtype)
        keep_out = ~zero
        return sp_sparse.csr_matrix((u[keep_out], self._out_indices[keep_out],
                                     indptr), shape=(self._n, self._n))


#: Dirichlet 投影缓存：``(id(jd), id(jcm), id(jmd)) → (…, projection)``（引用校验防 id 复用）
_DP_CACHE: dict = {}


def get_dirichlet_projection(j_driven, j_col_mask, j_mat_driven) -> DirichletProjection:
    """取/建三个算子对应的预计算投影（算子随 ``kv`` 逐时间点不同 ⇒ 按身份缓存）。"""
    key = (id(j_driven), id(j_col_mask), id(j_mat_driven))
    ent = _DP_CACHE.get(key)
    if (ent is not None and ent[0] is j_driven
            and ent[1] is j_col_mask and ent[2] is j_mat_driven):
        return ent[3]
    proj = DirichletProjection(j_driven, j_col_mask, j_mat_driven)
    _DP_CACHE[key] = (j_driven, j_col_mask, j_mat_driven, proj)
    return proj


def project_dirichlet(J, j_driven, j_col_mask, j_mat_driven):
    """``(j_driven @ J) @ j_col_mask + j_mat_driven``（预计算版，**逐位一致**）。"""
    return get_dirichlet_projection(j_driven, j_col_mask, j_mat_driven).apply(J)
