"""P2 单元变换矩阵测试（路线 A：`assembly_p2.transformation_matrix_p2`）。

来源 A 解析性质，覆盖三分支：

- 普通单元：T = I₆，index = 6 节点；
- 从属单元：T = diag(signs)（反周期符号逐节点）；
- 转子气隙单元（分支 1）：
  · 全部 6 槽位统一走 brackets 插值（−K_rb 系数）—— 含接口圆边中点（Design B，
    2026-10-02：中点走自己的插值约束，不是端点平均，见 assembly_p2 模块 docstring）；
  · 不在 brackets 的槽位（气隙内部/跨接口边中点、非接口顶点）= 独立自由度。

运行::

    python -m pytest tests/kernels/test_assembly_p2.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from motorfem.kernels.reference.assembly import TopologyContext
from motorfem.kernels.reference.assembly_p2 import transformation_matrix_p2


class _FakeRotating:
    def __init__(self, brackets, matrix):
        self.brackets = brackets
        self.matrix = matrix      # (N,N) numpy 数组，支持 [node, bracket] 索引


def _ctx(tri, element_rotor_gap=frozenset(), element_slave=frozenset(),
         master_nodes=frozenset(), virtual_tri=None):
    return TopologyContext(
        tri=np.asarray(tri, dtype=int),
        virtual_tri=virtual_tri or {},
        element_rotor_gap=frozenset(element_rotor_gap),
        element_slave=frozenset(element_slave),
        master_nodes=frozenset(master_nodes),
        a_eq=False,
        n_node=20,
    )


def test_ordinary_unit_is_identity():
    """普通单元：T = I₆，index = 6 节点。"""
    ctx = _ctx([[0, 1, 2, 3, 4, 5]])
    tr = transformation_matrix_p2(0, ctx, _FakeRotating({}, np.zeros((20, 20))))
    assert np.array_equal(tr.matrix, np.eye(6))
    assert tr.index == (0, 1, 2, 3, 4, 5)


def test_slave_unit_diagonal_signs():
    """从属单元：T = diag(signs)，master 节点反周期符号 −1。"""
    # master 节点(主动边界) → +1、非 master → −1（反周期 a_eq=False）
    ctx = _ctx([[0, 1, 2, 3, 4, 5]], element_slave=frozenset({0}),
               master_nodes=frozenset({0, 3}),
               virtual_tri={0: (0, 1, 2, 3, 4, 5)})
    tr = transformation_matrix_p2(0, ctx, _FakeRotating({}, np.zeros((20, 20))))
    expected = np.diag([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
    assert np.allclose(tr.matrix, expected)
    assert tr.index == (0, 1, 2, 3, 4, 5)


def test_band_vertex_bracket_and_midedges():
    """气隙单元：顶点走 brackets 插值；边中点按端点在不在 brackets 处理。

    单元 [0,1,2, 3,4,5]，顶点 0 是动节点（brackets 到 4,5，系数 kp,kn=0.3,0.7）：
    - 顶点 0 行：−0.3@up(4)、−0.7@low(5)；
    - 顶点 1、2：独立；
    - 中点 3（边 0-1，一端动）：独立 @节点3；
    - 中点 4（边 1-2，都不动）：独立 @节点4；
    - 中点 5（边 2-0，一端动）：独立 @节点5。
    """
    brackets = {0: (4, 5)}
    matrix = np.zeros((20, 20))
    matrix[0, 4] = 0.3      # kp
    matrix[0, 5] = 0.7      # kn
    rot = _FakeRotating(brackets, matrix)
    ctx = _ctx([[0, 1, 2, 3, 4, 5]], element_rotor_gap=frozenset({0}))
    tr = transformation_matrix_p2(0, ctx, rot)

    col = {n: i for i, n in enumerate(tr.index)}
    assert tr.matrix[0, col[4]] == pytest.approx(-0.3)
    assert tr.matrix[0, col[5]] == pytest.approx(-0.7)
    assert tr.matrix[1, col[1]] == pytest.approx(1.0)   # 顶点 1 独立
    assert tr.matrix[2, col[2]] == pytest.approx(1.0)   # 顶点 2 独立
    assert tr.matrix[3, col[3]] == pytest.approx(1.0)   # 中点 3 独立
    assert tr.matrix[4, col[4]] == pytest.approx(1.0)   # 中点 4 独立
    assert tr.matrix[5, col[5]] == pytest.approx(1.0)   # 中点 5 独立
    # 列集应为 {1,2,3,4,5}（顶点0 被 brackets 消掉）
    assert set(tr.index) == {1, 2, 3, 4, 5}


def test_band_interface_edge_midpoint_own_bracket():
    """接口圆边（两端点都动网格）中点 = 自己的 bracket 行（Design B）。

    顶点 0、1 是动节点（0→(4,5)、1→(6,7)）；中点 3（边 0-1）**也是**动节点，
    有自己的 bracket 行 3→(8,9, kp=0.5,kn=0.5)。T 的中点行 = 它自己的插值行，
    中点 3 本身从活跃自由度里消去（不是端点插值平均）。
    """
    brackets = {0: (4, 5), 1: (6, 7), 3: (8, 9)}
    matrix = np.zeros((20, 20))
    matrix[0, 4], matrix[0, 5] = 0.3, 0.7
    matrix[1, 6], matrix[1, 7] = 0.6, 0.4
    matrix[3, 8], matrix[3, 9] = 0.5, 0.5
    rot = _FakeRotating(brackets, matrix)
    ctx = _ctx([[0, 1, 2, 3, 4, 5]], element_rotor_gap=frozenset({0}))
    tr = transformation_matrix_p2(0, ctx, rot)
    col = {n: i for i, n in enumerate(tr.index)}
    assert tr.matrix[3, col[8]] == pytest.approx(-0.5)
    assert tr.matrix[3, col[9]] == pytest.approx(-0.5)
    assert 3 not in tr.index            # 中点 3 被自己的插值行消去
    assert tr.matrix[3, col[4]] == pytest.approx(0.0)   # 不再混入端点 0 的 bracket


# ---------------------------------------------------------------------------
# element_operators_p2（装配 18 项）
# ---------------------------------------------------------------------------


def _expand21(K0e):
    K = np.zeros((6, 6)); idx = 0
    for a in range(6):
        for b in range(a, 6):
            K[a, b] = K[b, a] = K0e[idx]; idx += 1
    return K


def test_element_operators_ordinary_unit():
    """普通单元：k0_data = K0 的逐项遍历、bxcoef/bycoef_mat 行、掩码。"""
    from motorfem.kernels.reference.assembly_p2 import element_operators_p2
    from motorfem.kernels.reference.element_matrices_p2 import electromagnetic_elements_p2
    node = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
    ])
    tri6 = np.array([[0, 1, 2, 3, 4, 5]])
    em = electromagnetic_elements_p2(node, tri6)
    ctx = _ctx(tri6)
    ops = element_operators_p2(
        ctx, _FakeRotating({}, np.zeros((6, 6))), n_node=6,
        k0elem=em.K0, bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        element_iron=(), element_topo_opti=(), use_nonlinear_iron=True)

    K0 = _expand21(em.K0[0])
    expected_k0 = np.array([K0[a, b] for a in range(6) for b in range(6)])
    assert np.allclose(ops.k0_data, expected_k0)
    assert ops.k0_row.tolist() == [i for i in range(6) for _ in range(6)]
    assert ops.k0_col.tolist() == [0, 1, 2, 3, 4, 5] * 6

    assert np.allclose(ops.bxcoef_mat[0, :].toarray().ravel(), em.Bxcoef[0])
    assert np.allclose(ops.bycoef_mat[0, :].toarray().ravel(), em.Bycoef[0])

    assert ops.is_vaccum[0] == 1.0          # 非铁 → 真空
    assert ops.n_k0 == 36
    assert ops.n_row_g0i == 6


# ---------------------------------------------------------------------------
# get_R_vector_p2（源项装配）
# ---------------------------------------------------------------------------


def test_get_R_p2_ordinary_element_matches_r0_times_j():
    """普通单元：R = Tᵀ R0 J = R0·J（T = I₆）—— 顶点 0、边中点 Δ/3。"""
    from motorfem.kernels.reference.assembly_p2 import get_R_vector_p2
    from motorfem.kernels.reference.element_matrices_p2 import electromagnetic_elements_p2
    node = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
    ])
    tri6 = np.array([[0, 1, 2, 3, 4, 5]])
    em = electromagnetic_elements_p2(node, tri6)
    ctx = _ctx(tri6)
    J = 3.0
    R = get_R_vector_p2(
        ctx, _FakeRotating({}, np.zeros((6, 6))), n_node=6,
        r0elem=em.R0, current_of_element={0: J})
    # 面积 Δ = 1/2 ⇒ R0 边中点 = Δ/3 = 1/6
    assert np.allclose(R[:3], 0.0)
    assert np.allclose(R[3:], em.R0[0, 3:] * J)
    assert np.isclose(R.sum(), em.area[0] * J)   # ΣR0 = Δ 守恒


def test_get_R_p2_moving_nodes_get_nothing():
    """含插值行的单元：源项只落到括号节点，动节点（顶点 *和* 中点）收不到。

    中点 R0 = Δ/3 > 0：它的源被 Tᵀ 拉到自己的括号节点上；动顶点 R0 = 0 且被消去。
    """
    from motorfem.kernels.reference.assembly_p2 import get_R_vector_p2
    # 单三角形，面积 Δ = 1/2 ⇒ R0[3:] = 1/6
    node = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
    ])
    tri6 = np.array([[0, 1, 2, 3, 4, 5]])
    r0 = np.zeros((1, 6))
    r0[0, 3:] = 1.0 / 6.0          # 顶点 0、边中点 Δ/3
    brackets = {0: (6, 7), 3: (8, 9)}   # 动顶点 0、动中点 3
    matrix = np.zeros((20, 20))
    matrix[0, 6], matrix[0, 7] = 0.4, 0.6      # 顶点 0 的 bracket
    matrix[3, 8], matrix[3, 9] = 0.7, 0.3      # 中点 3 的 bracket
    rot = _FakeRotating(brackets, matrix)
    ctx = _ctx(tri6, element_rotor_gap=frozenset({0}))
    J = 3.0
    R = get_R_vector_p2(
        ctx, rot, n_node=20, r0elem=r0, current_of_element={0: J})

    assert R[0] == 0.0             # 动顶点：R0=0 且被消去
    assert R[3] == 0.0             # 动中点：源被拉到括号节点
    # 中点 3 的 R0 = 1/6 拉到它的括号：T[3, 8]·R0[3]·J = −0.7·(1/6)·3
    assert np.isclose(R[8], -0.7 * (1.0 / 6.0) * J)
    assert np.isclose(R[9], -0.3 * (1.0 / 6.0) * J)
    # 顶点 0 的 bracket 拉回恒 0（R0[0] = 0）
    assert R[6] == 0.0 and R[7] == 0.0
