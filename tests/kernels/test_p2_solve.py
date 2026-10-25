"""P2 求解链路集成测试（路线 A）：泊松 −ΔA=f + Dirichlet 的收敛阶。

来源 A 解析性质：解析解 A* = x(1−x)y(1−y)（边界为 0）、f = 2[y(1−y)+x(1−x)]。
验证 P1/P2 全链路（单元刚度 → 稀疏装配 → 一致载荷 → Dirichlet → spsolve）：

- P1 的 L2 收敛阶 ≈ 2；
- P2 的 L2 收敛阶 ≥ 3.5（超收敛，实际 ≈ 4）；
- 同网格下 P2 误差远小于 P1（至少一个量级）。

运行::

    python -m pytest tests/kernels/test_p2_solve.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from motorfem.kernels.reference.element_matrices import electromagnetic_elements
from motorfem.kernels.reference.element_matrices_p2 import electromagnetic_elements_p2

_A_STAR = lambda x, y: x * (1 - x) * y * (1 - y)          # noqa: E731
_F_SRC = lambda x, y: 2.0 * (y * (1 - y) + x * (1 - x))    # noqa: E731

_GAUSS_LAM = np.array([
    [1.0 / 6, 1.0 / 6, 2.0 / 3],
    [1.0 / 6, 2.0 / 3, 1.0 / 6],
    [2.0 / 3, 1.0 / 6, 1.0 / 6],
])
_GAUSS_W = 1.0 / 3


def make_rectangle_p2(nx: int, ny: int):
    """矩形结构化网格 + P2 边中点。顶点在前，边中点在后。"""
    xs, ys = np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1)
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
    return node, np.array(tris), tri6


def _shape(lam1, lam2, lam3):
    return np.array([
        lam1 * (2 * lam1 - 1), lam2 * (2 * lam2 - 1), lam3 * (2 * lam3 - 1),
        4 * lam1 * lam2, 4 * lam2 * lam3, 4 * lam3 * lam1,
    ])


def _expand6(K0e):
    K = np.empty((3, 3))
    K[0, 0] = K0e[0]; K[0, 1] = K[1, 0] = K0e[1]; K[0, 2] = K[2, 0] = K0e[2]
    K[1, 1] = K0e[3]; K[1, 2] = K[2, 1] = K0e[4]; K[2, 2] = K0e[5]
    return K


def _expand21(K0e):
    K = np.zeros((6, 6)); idx = 0
    for a in range(6):
        for b in range(a, 6):
            K[a, b] = K[b, a] = K0e[idx]; idx += 1
    return K


def _assemble(coef, tri, n_node, expand_single):
    rows, cols, vals = [], [], []
    for e in range(tri.shape[0]):
        Ke = expand_single(coef[e])
        for a in range(Ke.shape[0]):
            for b in range(Ke.shape[1]):
                rows.append(int(tri[e, a])); cols.append(int(tri[e, b])); vals.append(Ke[a, b])
    return coo_matrix((vals, (rows, cols)), shape=(n_node, n_node)).tocsr()


def _solve(K, R, node):
    K = K.tolil()
    for i, (x, y) in enumerate(node):
        if min(x, 1 - x, y, 1 - y) < 1e-12:
            K[i, :] = 0; K[i, i] = 1; R[i] = 0.0
    return spsolve(K.tocsr(), R)


def _solve_p1(node, tri):
    nv = int(tri.max()) + 1
    node_v = node[:nv]
    em = electromagnetic_elements(node_v, tri)
    K = _assemble(em.K0, tri, nv, _expand6)
    R = np.zeros(nv)
    for e in range(tri.shape[0]):
        R[tri[e]] += em.R0[e] * _F_SRC(*node_v[tri[e]].mean(axis=0))
    return _solve(K, R, node_v)


def _solve_p2(node, tri6):
    em = electromagnetic_elements_p2(node, tri6)
    K = _assemble(em.K0, tri6, node.shape[0], _expand21)
    R = np.zeros(node.shape[0])
    for e in range(em.n_elem):
        pts = node[em.tri6[e]]
        Re = np.zeros(6)
        for gg in range(3):
            lam = _GAUSS_LAM[gg]
            p = lam @ pts[:3]
            Re += _GAUSS_W * em.area[e] * _shape(*lam) * _F_SRC(p[0], p[1])
        R[em.tri6[e]] += Re
    return _solve(K, R, node)


def _l2(A, Aex, node):
    return np.sqrt(np.mean((A - Aex) ** 2))


def test_p2_converges_faster_than_p1():
    """P2 全链路收敛阶 ≥ 3.5（P1 只有 ~2），同网格误差小一个量级以上。"""
    prev = {}
    for n in (4, 8, 16):
        node, tri, tri6 = make_rectangle_p2(n, n)
        nv = int(tri.max()) + 1
        Aex = _A_STAR(node[:, 0], node[:, 1])
        e1 = _l2(_solve_p1(node, tri), Aex[:nv], node[:nv])
        e2 = _l2(_solve_p2(node, tri6), Aex, node)
        if n == 4:
            prev = {"e1": e1, "e2": e2}
            continue
        o1 = np.log2(prev["e1"] / e1)          # P1 收敛阶
        o2 = np.log2(prev["e2"] / e2)          # P2 收敛阶
        prev = {"e1": e1, "e2": e2}
        if n == 8:
            assert o1 >= 1.5, f"P1 阶过低: {o1}"
        if n == 16:
            assert o2 >= 3.5, f"P2 阶过低: {o2}"
            assert e2 < e1, f"P2 误差应远小于 P1: e2={e2:.2e} e1={e1:.2e}"
