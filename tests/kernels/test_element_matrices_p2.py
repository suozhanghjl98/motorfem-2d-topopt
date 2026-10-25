"""P2 单元矩阵测试（路线 A：内核 4 的 P2 路径，`element_matrices_p2`）。

测试三级来源（本模块目前只有**来源 A 解析性质**；来源 C ref-vs-fast 待
fast 层实现后补，来源 B legacy 无 P2 对应）:

  来源 A 解析性质
      —— K0 行和恒零（常值零空间）、对称半正定；
      —— R0 顶点 0 / 边中点 Δ/3 / 和 = Δ（一致载荷）；
      —— 中心 Bxcoef/Bycoef 对线性场精确恢复 ∂A/∂x、−∂A/∂y（legacy 约定）；
      —— P2 能量 ½AᵀKA 对**线性场精确**（P2 含 P1）、对**二次场精确**
         （P2 含二次插值，机器精度）。

运行::

    python -m pytest tests/kernels/test_element_matrices_p2.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from motorfem.kernels.reference.element_matrices_p2 import (
    electromagnetic_elements_p2,
)

# ---------------------------------------------------------------------------
# 网格辅助
# ---------------------------------------------------------------------------


def make_single_triangle():
    """单个 P2 三角形：(0,0),(1,0),(0,1) 逆时针 + 三边中点。"""
    node = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],   # 顶点 i,j,m
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],   # 边中点 ij, jm, mi
    ])
    tri6 = np.array([[0, 1, 2, 3, 4, 5]])
    return node, tri6


def make_rectangle_p2(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
    """矩形域结构化网格 + P2 边中点（确定性，共享边唯一化）。

    每个矩形 2 个三角形（下三角 + 上三角），逆时针。
    返回 ``(node, tri6)``，``tri6`` 每行 = [i,j,m, k(ij), l(jm), n(mi)]。
    """
    xs, ys = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    node = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris += [[a, b, c], [a, c, d]]

    mid2id = {}
    tri6 = np.zeros((len(tris), 6), dtype=int)
    for e, (i, j, m) in enumerate(tris):
        mids = []
        for (a, b) in ((i, j), (j, m), (m, i)):
            key = tuple(sorted((int(a), int(b))))
            if key not in mid2id:
                mid2id[key] = node.shape[0] + len(mid2id)
            mids.append(mid2id[key])
        tri6[e] = [i, j, m, *mids]
    node = np.vstack([node, np.array([
        (node[i] + node[j]) / 2 for (i, j), _ in sorted(mid2id.items(), key=lambda t: t[1])
    ])])
    return node, tri6


def _expand21(K0e):
    K = np.zeros((6, 6)); idx = 0
    for a in range(6):
        for b in range(a, 6):
            K[a, b] = K[b, a] = K0e[idx]; idx += 1
    return K


def _energy(em, A, e=None):
    """P2 能量 ½AᵀKA。e=None 时用全部单元装配的稀疏 K。"""
    from scipy.sparse import coo_matrix
    if e is None:
        rows, cols, vals = [], [], []
        for ee in range(em.n_elem):
            Ke = _expand21(em.K0[ee])
            for a in range(6):
                for b in range(6):
                    rows.append(int(em.tri6[ee, a])); cols.append(int(em.tri6[ee, b]))
                    vals.append(Ke[a, b])
        K = coo_matrix((vals, (rows, cols)), shape=(A.shape[0], A.shape[0])).tocsr()
        return 0.5 * float(A @ (K @ A))
    Ke = _expand21(em.K0[e])
    return 0.5 * float(A[em.tri6[e]] @ (Ke @ A[em.tri6[e]]))


# ---------------------------------------------------------------------------
# 来源 A 解析性质
# ---------------------------------------------------------------------------


def test_k0_rowsum_zero_constant_nullspace():
    """常值 A 下能量为零 ⇒ K 的行和（每列和）恒零。"""
    node, tri6 = make_single_triangle()
    em = electromagnetic_elements_p2(node, tri6)
    K = _expand21(em.K0[0])
    one = np.ones(6)
    assert abs(0.5 * one @ (K @ one)) < 1e-12
    assert np.abs(K @ one).max() < 1e-12


def test_k0_symmetric_psd():
    node, tri6 = make_rectangle_p2(2, 2)
    em = electromagnetic_elements_p2(node, tri6)
    for e in range(em.n_elem):
        K = _expand21(em.K0[e])
        assert np.abs(K - K.T).max() < 1e-14
        assert np.linalg.eigvalsh(K).min() > -1e-12


def test_r0_consistent_load():
    """一致载荷：顶点 ∫N=0、边中点 ∫N=Δ/3、ΣR0=Δ。"""
    node, tri6 = make_rectangle_p2(2, 2)
    em = electromagnetic_elements_p2(node, tri6)
    for e in range(em.n_elem):
        D = em.area[e]
        R0 = em.R0[e]
        assert np.abs(R0[:3]).max() < 1e-12          # 顶点 = 0
        assert np.abs(R0[3:] - D / 3).max() < 1e-12   # 边中点 = Δ/3
        assert abs(R0.sum() - D) < 1e-12              # 总和 = Δ


def test_center_B_linear_exact():
    """中心 B 对线性场精确（legacy 约定：Bx=∂A/∂x、By=−∂A/∂y）。"""
    node, tri6 = make_single_triangle()
    em = electromagnetic_elements_p2(node, tri6)
    alpha, beta = 2.0, -3.0
    A = alpha * node[:, 0] + beta * node[:, 1]
    Bx = float(em.Bxcoef[0] @ A[em.tri6[0]])
    By = float(em.Bycoef[0] @ A[em.tri6[0]])
    assert abs(Bx - alpha) < 1e-12
    assert abs(By - (-beta)) < 1e-12


def test_energy_linear_exact():
    """P2 能量对线性场精确：½AᵀKA = ½(α²+β²)Δ。"""
    node, tri6 = make_single_triangle()
    em = electromagnetic_elements_p2(node, tri6)
    alpha, beta = 2.0, -3.0
    A = alpha * node[:, 0] + beta * node[:, 1]
    D = em.area[0]
    E = _energy(em, A, 0)
    assert abs(E - 0.5 * (alpha ** 2 + beta ** 2) * D) < 1e-12


def test_energy_quadratic_exact_any_mesh():
    """P2 能量对二次场精确（机器精度，任意网格）—— 同精度收益验证。"""
    # 解析：A=x²+y² ⇒ ∇A=(2x,2y)，∫½|∇A|²dV = 2∫(x²+y²)dV = 2(1/3+1/3) = 4/3
    exact = 4.0 / 3.0
    for nx, ny in [(1, 1), (2, 2), (4, 4)]:
        node, tri6 = make_rectangle_p2(nx, ny)
        em = electromagnetic_elements_p2(node, tri6)
        A = node[:, 0] ** 2 + node[:, 1] ** 2
        E = _energy(em, A)
        assert abs(E - exact) < 1e-10, f"nx={nx}: E={E} exact={exact}"
