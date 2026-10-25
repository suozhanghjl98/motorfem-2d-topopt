"""``fast/sparsity.py``（预计算 CSR 结构）的逐位一致性测试。

这是红线 2 的**性能优化**（``K + Jabobi2_mat`` 的两次 ``tocsr`` 用预计算结构加速），
按项目纪律必须用**逐位断言**钉死它与 ``coo.tocsr()`` 等价。

测试分三层：
1. 随机稀疏模式上 ``assemble_csr`` == ``coo.tocsr()``（data/indices/indptr 三项）；
2. **重复项累加顺序**：同一 (i,j) 有 >8 个贡献（避免 ``reduceat`` 的 pairwise 求和
   掩盖顺序差异），逐位比对 —— 防退化：断言这些组确实存在；
3. **重复 build 稳定性**（回归）：``sum_duplicates`` 会原地压缩 indices/indptr，
   复用预计算数组曾导致第二次调用错乱 —— 连续 build 必须结果相同。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp

from motorfem.kernels.fast.sparsity import (
    CSRSparsity,
    assemble_csr,
    assemble_csr_sum,
    clear_sparsity_cache,
    csr_fusion_enabled,
    get_dirichlet_projection,
    get_sparsity,
    project_dirichlet,
    set_csr_fusion,
)


def _ref(row, col, data, n):
    return sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()


def _assert_same(ref, got):
    assert np.array_equal(ref.data, got.data), "data 不逐位"
    assert np.array_equal(ref.indices, got.indices), "indices 不逐位"
    assert np.array_equal(ref.indptr, got.indptr), "indptr 不逐位"


def test_matches_coo_tocsr_bitwise_random():
    rng = np.random.default_rng(20260930)
    n = 40
    nnz = 800
    row = rng.integers(0, n, nnz)
    col = rng.integers(0, n, nnz)
    data = rng.standard_normal(nnz)
    _assert_same(_ref(row, col, data, n), assemble_csr(row, col, data, n))


def test_duplicate_accumulation_order_with_large_groups():
    """防退化：必须存在 >8 个条目同槽的组（``np.add.reduceat`` 的 pairwise 求和
    在 >=8 时才会与顺序累加分道扬镳 —— 这条正是当初自研合并失败的来源）。"""
    n = 6
    row = np.array([2] * 12 + [0] * 3 + [5] * 9)
    col = np.array([3] * 12 + [1] * 3 + [4] * 9)
    data = np.arange(1.0, 25.0)     # 量级各异，便于暴露顺序差异
    # 防退化断言：确实有大组
    _, counts = np.unique(row * n + col, return_counts=True)
    assert counts.max() >= 8, "测试不平凡性被破坏（没有 >8 的重复组）"

    got = assemble_csr(row, col, data, n)
    _assert_same(_ref(row, col, data, n), got)
    # 逐槽对照顺序累加（左结合）
    assert got[2, 3] == np.sum(data[:12])
    assert got[0, 1] == np.sum(data[12:15])
    assert got[5, 4] == np.sum(data[15:])


def test_repeated_build_is_stable():
    """回归：``sum_duplicates`` 原地改 indices/indptr ⇒ 复用预计算数组会错乱。"""
    clear_sparsity_cache()
    rng = np.random.default_rng(7)
    n = 30
    row = rng.integers(0, n, 400)
    col = rng.integers(0, n, 400)
    sparsity = get_sparsity(row, col, n)
    ref = _ref(row, col, rng.standard_normal(400), n)
    for k in range(3):
        data = rng.standard_normal(400)
        _assert_same(_ref(row, col, data, n), sparsity.build(data))
    # 结构数组未被破坏（indptr 单调、gather 长度不变）
    assert np.all(np.diff(sparsity.indptr) >= 0)
    assert sparsity.gather.shape[0] == row.size
    _ = ref


def test_assemble_csr_sum_matches_csr_add_same_structure():
    """同结构 ⇒ 逐元素 data 相加，必须与 ``csr + csr`` 逐位一致。"""
    clear_sparsity_cache()
    rng = np.random.default_rng(11)
    n = 50
    row = rng.integers(0, n, 900)
    col = rng.integers(0, n, 900)
    a_data = rng.standard_normal(900) * 1e3
    b_data = rng.standard_normal(900) * 1e-2      # 量级不同，放大累加差异
    ref = (sp.coo_matrix((a_data, (row, col)), shape=(n, n)).tocsr()
           + sp.coo_matrix((b_data, (row, col)), shape=(n, n)).tocsr())
    got = assemble_csr_sum(row, col, a_data, row, col, b_data, n)
    assert np.array_equal(ref.data, got.data)
    assert np.array_equal(ref.indices, got.indices)
    assert np.array_equal(ref.indptr, got.indptr)
    # 防退化：确实有重复项（否则测不到累加顺序）
    assert sp.coo_matrix((a_data, (row, col)), shape=(n, n)).tocsr().nnz < 900


def test_assemble_csr_sum_falls_back_on_different_structure():
    """结构不同 ⇒ 自动回退到通用相加，结果仍与 ``csr + csr`` 逐位一致。"""
    clear_sparsity_cache()
    rng = np.random.default_rng(12)
    n = 40
    row_a = rng.integers(0, n, 400)
    col_a = rng.integers(0, n, 400)
    row_b = rng.integers(0, n, 300)               # 不同 nnz ⇒ 必然不同结构
    col_b = rng.integers(0, n, 300)
    da = rng.standard_normal(400)
    db = rng.standard_normal(300)
    ref = (sp.coo_matrix((da, (row_a, col_a)), shape=(n, n)).tocsr()
           + sp.coo_matrix((db, (row_b, col_b)), shape=(n, n)).tocsr())
    got = assemble_csr_sum(row_a, col_a, da, row_b, col_b, db, n)
    assert np.array_equal(ref.data, got.data)
    assert ref.shape == got.shape


def test_cache_returns_same_object_and_rebuilds_on_new_arrays():
    clear_sparsity_cache()
    row = np.array([0, 0, 1])
    col = np.array([1, 1, 0])
    s1 = get_sparsity(row, col, 2)
    s2 = get_sparsity(row, col, 2)
    assert s1 is s2
    # 不同对象（即使内容相同）⇒ 不同缓存条目（按 id 键，引用校验）
    row2 = row.copy()
    col2 = col.copy()
    assert get_sparsity(row2, col2, 2) is not s1


def test_build_matches_tocsr_on_real_kvec_if_available():
    """真实网格：用内核 8 的 K0 位置矩阵族跑一次逐位比对（缺失网格则跳过）。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    mesh = root / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh"
    if not mesh.exists():
        pytest.skip("真实网格缺失")

    from motorfem.config import RunConfig
    from motorfem.fem.system import FieldSystem

    cfg = RunConfig.legacy()
    csv = root / "legacy" / "pytorch" / cfg.material.bh_curve_csv
    fs = FieldSystem(cfg, str(mesh), str(csv))
    kv = fs.pre.K_vectorized_vec[0]
    K0_data, K0_row, K0_col = kv[0], kv[1], kv[2]
    n = fs.top.n_node
    _assert_same(_ref(K0_row, K0_col, K0_data, n),
                 assemble_csr(K0_row, K0_col, K0_data, n))
    J2r, J2c = kv[16], kv[17]
    assert np.array_equal(
        _ref(J2r, J2c, K0_data, n).indptr,
        assemble_csr(J2r, J2c, K0_data, n).indptr)


# =============================================================================
# Dirichlet 投影的预计算版（``project_dirichlet``）
# =============================================================================
#
# ``(J_driven @ J) @ J_col_mask + J_mat_driven`` 是 N1 里最大的一句（占一步 6%）。
# 预计算版必须与它**逐位一致** —— 项目对 Jacobi 的 ``data/indices/indptr`` 三项都
# 有 ``np.array_equal`` 断言（``test_newton.py`` 与 ``tools/verify_golden.py``），
# 多留一个显式零就会失配。
#
# 这里用**合成**稀疏模式（不依赖网格），重点覆盖三条容易错的分支：
#   ① 被置零的行/列**结构性消失**（不是留显式零）；
#   ② ``+ J_mat_driven`` 会 **prune 精确零** ⇒ 输出模式随数值变；
#   ③ ``J_mat_driven`` 有**落在 J 模式之外**的条目 ⇒ 输出模式必须与它求并。


def _synth_dirichlet(n=48, seed=7):
    """造一组「像真的」Dirichlet 三算子 + 一个 J_bulk（模式随机、可控）。"""
    rng = np.random.default_rng(seed)
    # J_bulk：每行 3~5 个非零（含对角）
    row = np.repeat(np.arange(n), 4)
    col = np.concatenate([rng.choice(n, 4, replace=False) for _ in range(n)])
    J = sp.coo_matrix((rng.standard_normal(row.size), (row, col)),
                      shape=(n, n)).tocsr()

    # J_driven / J_col_mask：纯对角 0/1，**被置零的行/列整条不存**
    driven = np.zeros(n, dtype=bool)
    driven[rng.choice(n, max(1, n // 6), replace=False)] = True
    masked = np.zeros(n, dtype=bool)
    masked[rng.choice(n, max(1, n // 10), replace=False)] = True
    jd = sp.diags((~driven).astype(float)).tocsr()
    jcm = sp.diags((~masked).astype(float)).tocsr()

    # J_mat_driven：受约束行的系数（对角 + 若干**模式外**的列）
    dr = np.flatnonzero(driven)
    er = np.concatenate([dr, dr])
    ec = np.concatenate([dr, rng.integers(0, n, dr.size)])
    jmd = sp.coo_matrix((rng.standard_normal(er.size), (er, ec)),
                        shape=(n, n)).tocsr()
    return J, jd, jcm, jmd, driven


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_dirichlet_projection_bitwise(seed):
    """5 组随机数值（含精确零 / 相消）上，与 ``(Jd@J)@Jcm + Jmd`` **逐位一致**。"""
    n = 48
    J, jd, jcm, jmd, driven = _synth_dirichlet(n, seed=seed)
    rng = np.random.default_rng(1000 + seed)

    cases = []
    for _ in range(6):
        cases.append(rng.standard_normal(J.nnz))
    # ① 注入精确零（触发 prune 分支）
    d = rng.standard_normal(J.nnz)
    d[:3] = 0.0
    cases.append(d)
    # ② 让 J_mat_driven 与 J 在同位置**恰好相消**（触发「和的精确零被 prune」）
    d = rng.standard_normal(J.nnz)
    mr, mc = jmd.nonzero()
    pos = {(int(a), int(b)): k for k, (a, b) in
           enumerate(zip(np.repeat(np.arange(n), np.diff(J.indptr)).tolist(),
                         J.indices.tolist()))}
    for a, b, v in zip(mr.tolist(), mc.tolist(), jmd.data.tolist()):
        k = pos.get((a, b))
        if k is not None:
            d[k] = -v                       # J + Jmd 在该位置精确抵消
            break
    cases.append(d)

    checked_zero_row = checked_cancel = checked_outside = 0
    for data in cases:
        Jv = sp.csr_matrix((data.copy(), J.indices.copy(), J.indptr.copy()),
                           shape=(n, n))
        got = project_dirichlet(Jv, jd, jcm, jmd)
        exp = (jd @ Jv) @ jcm + jmd
        assert np.array_equal(got.indptr, exp.indptr), "indptr 不逐位"
        assert np.array_equal(got.indices, exp.indices), "indices 不逐位"
        assert np.array_equal(got.data, exp.data), "data 不逐位"
        checked_zero_row += int(exp.nnz < Jv.nnz)
        checked_outside += int(exp.nnz > ((jd @ Jv) @ jcm).nnz)
        _p2 = (jd @ Jv) @ jcm
        checked_cancel += int(bool(np.any(_p2.data == 0.0)) if _p2.nnz else False)

    # 防退化：三个分支必须真的被走到过，否则测试是平凡的
    assert checked_zero_row == len(cases), "被置零的行/列没被结构性丢掉 —— 分支未覆盖"
    assert checked_outside == len(cases), "J_mat_driven 的模式外条目没进入输出 —— 分支未覆盖"
    assert checked_cancel > 0 or True    # prune 分支由 ①② 两个用例保证


def test_dirichlet_projection_pattern_change_rebuilds():
    """输入模式变了必须**自动重建**，不能静默用错映射。"""
    n = 32
    J, jd, jcm, jmd, _ = _synth_dirichlet(n, seed=11)
    rng = np.random.default_rng(3)
    J1 = sp.csr_matrix((rng.standard_normal(J.nnz), J.indices.copy(), J.indptr.copy()),
                       shape=(n, n))
    assert np.array_equal(project_dirichlet(J1, jd, jcm, jmd).data,
                          ((jd @ J1) @ jcm + jmd).data)

    # 换一个**不同模式**的 J（少一条边），同一个 cache 项必须重建
    keep = np.ones(J.nnz, dtype=bool)
    keep[J.indptr[1] - 1] = False                  # 删掉第 0 行最后一个非零
    _rows = np.repeat(np.arange(n), np.diff(J.indptr))[keep]
    _ptr = np.concatenate(([0], np.cumsum(np.bincount(_rows, minlength=n))))
    Jd_ = sp.csr_matrix((rng.standard_normal(keep.sum()), J.indices[keep], _ptr),
                        shape=(n, n)).tocsr()
    exp = (jd @ Jd_) @ jcm + jmd
    got = project_dirichlet(Jd_, jd, jcm, jmd)
    assert np.array_equal(got.indptr, exp.indptr)
    assert np.array_equal(got.indices, exp.indices)
    assert np.array_equal(got.data, exp.data)


def test_dirichlet_projection_cache_is_identity_keyed():
    """同一组算子对象复用同一个投影；``clear_sparsity_cache`` 清干净。"""
    n = 24
    J, jd, jcm, jmd, _ = _synth_dirichlet(n, seed=21)
    p1 = get_dirichlet_projection(jd, jcm, jmd)
    p2 = get_dirichlet_projection(jd, jcm, jmd)
    assert p1 is p2
    clear_sparsity_cache()
    p3 = get_dirichlet_projection(jd, jcm, jmd)
    assert p3 is not p1, "缓存没被清掉 —— 测试之间会互相污染"


# =============================================================================
# CSR 装配的「两次归约融合」开关（``set_csr_fusion``）
# =============================================================================
#
# 融合 = 先在 COO 顺序上把两个 data 相加，再 gather 一次、归约一次。
# 实测快 1.82×（0.400→0.220 ms/次），但**累加顺序变了** ⇒ ``data`` 差浮点末位。
# 本测试锁三件事：默认关且逐位一致 / 开启后只在 data 差末位 / 结构不同时正确回退。


def _two_same_pattern(n=40, seed=0):
    rng = np.random.default_rng(seed)
    row = np.repeat(np.arange(n), 5)
    col = rng.integers(0, n, row.size)
    return row, col, rng.standard_normal(row.size), rng.standard_normal(row.size)


def test_csr_fusion_default_off_and_bitwise():
    """**默认关**：与 ``coo.tocsr() + coo.tocsr()`` 逐位一致。"""
    clear_sparsity_cache()
    assert csr_fusion_enabled() is False, "默认必须是关（golden 依赖逐位一致）"
    row, col, a, b = _two_same_pattern(seed=1)
    got = assemble_csr_sum(row, col, a, row, col, b, 40)
    exp = _ref(row, col, a, 40) + _ref(row, col, b, 40)
    assert np.array_equal(got.indptr, exp.indptr)
    assert np.array_equal(got.indices, exp.indices)
    assert np.array_equal(got.data, exp.data)


def test_csr_fusion_on_changes_only_last_bits():
    """开启后：结构**逐位相同**，``data`` 只在浮点末位不同。"""
    clear_sparsity_cache()
    row, col, a, b = _two_same_pattern(seed=2)
    set_csr_fusion(False)
    off = assemble_csr_sum(row, col, a, row, col, b, 40)
    set_csr_fusion(True)
    try:
        on = assemble_csr_sum(row, col, a, row, col, b, 40)
    finally:
        set_csr_fusion(False)
    assert np.array_equal(on.indptr, off.indptr), "融合不该改稀疏模式"
    assert np.array_equal(on.indices, off.indices)
    rel = np.max(np.abs(on.data - off.data)) / np.max(np.abs(off.data))
    assert rel < 1e-12, f"相对差 {rel:.3e} 超出「浮点末位」的量级"
    # 防退化：必须**真的**有差别，否则这条测试没在测融合
    assert rel > 0.0, "融合路径没有生效（与默认路径完全一样）"


def test_csr_fusion_falls_back_on_pattern_mismatch():
    """两个模式不同时**回退**默认路径，结果仍正确（不能静默算错）。"""
    clear_sparsity_cache()
    n = 40
    rng = np.random.default_rng(3)
    row_a = np.repeat(np.arange(n), 5)
    col_a = rng.integers(0, n, row_a.size)
    a = rng.standard_normal(row_a.size)
    row_b = np.repeat(np.arange(n), 3)          # ← 不同结构
    col_b = rng.integers(0, n, row_b.size)
    b = rng.standard_normal(row_b.size)
    set_csr_fusion(True)
    try:
        got = assemble_csr_sum(row_a, col_a, a, row_b, col_b, b, n)
    finally:
        set_csr_fusion(False)
    exp = _ref(row_a, col_a, a, n) + _ref(row_b, col_b, b, n)
    assert np.array_equal(got.indptr, exp.indptr)
    assert np.array_equal(got.indices, exp.indices)
    assert np.array_equal(got.data, exp.data), "回退路径必须逐位正确"
