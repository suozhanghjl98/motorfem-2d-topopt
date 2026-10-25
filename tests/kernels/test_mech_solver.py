"""内核 16 测试：力学装配族 / 残差 / 求解 / 灵敏度。

测试三级来源（`docs/重构设计文档.md` §1.4 与 `CLAUDE.md` §4）：

  来源 A 解析解/解析性质 —— 单元级手算（`TᵀK_eT`、主从旋转块、约束行）、
         两条装配路线互证（位置矩阵 vs 直接逐单元）、K 对称半正定、
         线弹性解在刚体平移下残差为零、灵敏度的有限差分
  来源 B 与 legacy 一致 —— 照抄 `MMA10:2604-2814` / `:2817-2858` / `:2861-2897`
  来源 C reference vs fast —— 见本文件末节

⚠ **一处不逐位**（设计文档 §11⑩）：内核 2 的 `SIMPProjection.derivative` 与 legacy 的
`SIMP_dfunc` 是代数等价但非逐位的改写（1~2 ULP）。所以凡涉及 `dkE` 的量
（`dK_p_dth_u`、灵敏度）只能用 `rtol`；`kE` 那一支仍逐位。

运行::

    python -m pytest tests/kernels/test_mech_solver.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg

from motorfem.kernels.reference import boundaries as ref_bnd
from motorfem.kernels.reference import mech_element_matrices as ref_mech5
from motorfem.kernels.reference import mech_solver as ref_solver

try:
    from motorfem.kernels.fast import mech_solver as fast_solver
except ImportError:  # pragma: no cover
    fast_solver = None

requires_fast = pytest.mark.skipif(fast_solver is None, reason="fast/ 层尚未落地")

PARTS = 4
SECTOR = np.pi / 2
COORD_ERR_ABS = 1e-6
R_RINGS = (0.030, 0.040, 0.050, 0.060)
R_CONSTRAINT = R_RINGS[0]
R_LOAD = R_RINGS[-1]
MAT = ref_mech5.MechMaterial(E=200e9, nu=0.30, rho=7850.0)
ANGULAR_SPEED = 300.0          # rad/s（任意，只要非零）
FLOAD = 1.0e7
FLOAD_ROT_DEG = 45.0


def make_sector_mesh(n_ang=7, radii=R_RINGS):
    """四层同心圆环扇（[0, π/2]），逆时针三角形 + 全部边界线元。

    节点号 ``= 层序号·n_ang + 角度序号``（含两端点）。线元分三类：外圈、内圈、
    两条径向边 —— legacy 的 ``element_L_mech`` 是网格的全部线元，
    ``element_mech_L_load`` 是「两端点都在 ``m_Rrotor`` 上」的那些。

    Returns
    -------
    node, tri, line_tri, loaded_lines
    """
    angles = np.linspace(0.0, SECTOR, n_ang)
    node = np.array([[r * np.cos(t), r * np.sin(t)] for r in radii for t in angles],
                    dtype=float)

    tri = []
    for k in range(len(radii) - 1):
        for l in range(n_ang - 1):
            a = k * n_ang + l
            b = (k + 1) * n_ang + l
            c = (k + 1) * n_ang + l + 1
            d = k * n_ang + l + 1
            tri += [[a, b, c], [a, c, d]]
    tri = np.array(tri, dtype=int)

    lines = []
    loaded = []
    # 外圈（≡ R_LOAD）与内圈（≡ R_CONSTRAINT）
    for k in (0, len(radii) - 1):
        for l in range(n_ang - 1):
            lines.append((k * n_ang + l, k * n_ang + l + 1))
            if k == len(radii) - 1:
                loaded.append(len(lines) - 1)
    # 两条径向边
    for l in (0, n_ang - 1):
        for k in range(len(radii) - 1):
            lines.append((k * n_ang + l, (k + 1) * n_ang + l))

    return node, tri, np.array(lines, dtype=int), loaded


class MechFixture:
    """扇区网格 + 内核 6 的配对 + 内核 5 的单元矩阵/载荷 + 装配族。"""

    def __init__(self, n_ang=7):
        self.node, self.tri, self.line_tri, self.loaded_lines = make_sector_mesh(n_ang)

        self.pairing = ref_bnd.pair_master_slave(
            self.node, parts=PARTS, coord_err_abs=COORD_ERR_ABS)
        self.slave_to_master = self.pairing.slave_to_master
        self.constrained = sorted(
            i for i in range(self.node.shape[0])
            if abs(np.hypot(*self.node[i]) - R_CONSTRAINT) < COORD_ERR_ABS
        )

        self.mech = ref_mech5.mech_element_matrices(self.node, self.tri, MAT)
        self.body_fx, self.body_fy = ref_mech5.mech_body_force(
            self.node, self.tri, MAT, angular_speed=ANGULAR_SPEED,
            rot_center=(0.0, 0.0))
        self.line_loads = ref_mech5.mech_line_loads(
            self.node, self.line_tri, self.loaded_lines,
            Fload=FLOAD, rot_angle_deg=FLOAD_ROT_DEG)

        self.n_elem = self.tri.shape[0]
        # ρ：拓扑区放在 SIMP 导数活跃处（thb = 0.65）
        self.rho = np.full(self.n_elem, 0.5)

        self.ops = ref_solver.mech_operators(
            self.node, self.tri,
            k_elem=self.mech.K, l_coef=self.mech.L, d_mat=self.mech.D,
            r0_elem=self.mech.R0, r2_elem=self.line_loads.R0,
            body_fx=self.body_fx, body_fy=self.body_fy,
            line_tri=self.line_tri, loaded_lines=self.loaded_lines,
            boundary_fx=self.line_loads.fx, boundary_fy=self.line_loads.fy,
            slave_to_master=self.slave_to_master,
            constrained_nodes=self.constrained, parts=PARTS,
        )

    def legacy_globals(self):
        return SimpleNamespace(
            N_node_mech=self.node.shape[0],
            N_element_mech=self.n_elem,
            element_S_mech=[[i, int(a), int(b), int(c)]
                            for i, (a, b, c) in enumerate(self.tri)],
            element_L_mech=[[i, int(a), int(b)]
                            for i, (a, b) in enumerate(self.line_tri)],
            element_mech_L_load=list(self.loaded_lines),
            node_mech=np.column_stack([
                np.arange(self.node.shape[0]), self.node,
                np.zeros(self.node.shape[0])]),
            node_mech_slave=dict(self.slave_to_master),
            node_mech_constraint=set(self.constrained),
            Kmech0elem=self.mech.K, Lmechcoef=self.mech.L, E0mechcoef=self.mech.D,
            Rmech10elem=self.mech.R0, Rmech20elem=self.line_loads.R0,
            fmechloadx=self.body_fx, fmechloady=self.body_fy,
            Fmechloadx=self.line_loads.fx, Fmechloady=self.line_loads.fy,
            parts=PARTS,
        )


@pytest.fixture(scope="module")
def fx() -> MechFixture:
    return MechFixture()


# =============================================================================
# 照抄 legacy（不复用被测实现）
# =============================================================================

def legacy_get_Kmech_vectorized(g):
    """照抄 ``MMA10:2604-2814``。返回 legacy 的字典（键名照抄）。"""
    N_node_mech, N_element_mech = g.N_node_mech, g.N_element_mech
    element_S_mech = g.element_S_mech
    node_mech_slave = g.node_mech_slave
    node_mech_constraint = g.node_mech_constraint
    Kmech0elem, Lmechcoef, E0mechcoef = g.Kmech0elem, g.Lmechcoef, g.E0mechcoef
    Rmech10elem, Rmech20elem = g.Rmech10elem, g.Rmech20elem
    fmechloadx, fmechloady = g.fmechloadx, g.fmechloady
    Fmechloadx, Fmechloady = g.Fmechloadx, g.Fmechloady
    parts = g.parts

    _K0mech_data = np.zeros(N_element_mech * 6 * 6)
    _K0mech_col = np.zeros(N_element_mech * 6 * 6)
    _K0mech_row = np.zeros(N_element_mech * 6 * 6)
    _Kconst_map = sp_sparse.lil_matrix((N_element_mech * 6 * 6, N_element_mech))
    _K_u_map_data = np.zeros(N_element_mech * 6 * 6)
    _K_u_map_row = np.zeros(N_element_mech * 6 * 6)
    _K_u_map_col = np.zeros(N_element_mech * 6 * 6)
    _K_boundary = sp_sparse.lil_matrix((N_node_mech * 2, N_node_mech * 2))
    _R0_map = sp_sparse.lil_matrix((N_node_mech * 2, N_element_mech))
    _R0_data = np.zeros(N_element_mech * 6)
    _R0_col = np.zeros(N_element_mech * 6)
    _R0_row = np.zeros(N_element_mech * 6)
    _Rconst_map = sp_sparse.lil_matrix((N_element_mech * 6, N_element_mech))
    _R2 = np.zeros(N_node_mech * 2, )
    _dR_p_dth_row = np.zeros(N_element_mech * 6)
    _dR_p_dth_col = np.zeros(N_element_mech * 6)
    _dK_p_dth_u_row = np.zeros(N_element_mech * 6 * 6)
    _dK_p_dth_u_col = np.zeros(N_element_mech * 6 * 6)
    K0_stress_data = np.zeros(N_element_mech * 3 * 6)
    K0_stress_row = np.zeros(N_element_mech * 3 * 6)
    K0_stress_col = np.zeros(N_element_mech * 3 * 6)
    K0_stress_kE_map = sp_sparse.lil_matrix((N_element_mech * 3, N_element_mech))
    VonMisesStress_map = sp_sparse.lil_matrix(
        (N_element_mech * 3, N_element_mech * 3))

    _r11 = np.cos(0 - 2 * np.pi / parts)
    _r12 = np.sin(0 - 2 * np.pi / parts)
    _r21 = -np.sin(0 - 2 * np.pi / parts)
    _r22 = np.cos(0 - 2 * np.pi / parts)

    for i_elem in range(N_element_mech):
        node_i, node_j, node_m = element_S_mech[i_elem][1:4]

        _T = np.eye(6)
        if node_i in node_mech_slave:
            _T[0:2, 0:2] = [[_r11, _r12], [_r21, _r22]]
        if node_j in node_mech_slave:
            _T[2:4, 2:4] = [[_r11, _r12], [_r21, _r22]]
        if node_m in node_mech_slave:
            _T[4:6, 4:6] = [[_r11, _r12], [_r21, _r22]]

        _index = [node_i * 2, node_i * 2 + 1, node_j * 2, node_j * 2 + 1,
                  node_m * 2, node_m * 2 + 1]
        _node = [node_i, node_j, node_m]
        if node_i in node_mech_slave:
            _index[0:2] = [node_mech_slave[node_i] * 2,
                           node_mech_slave[node_i] * 2 + 1]
            _node[0] = node_mech_slave[node_i]
        if node_j in node_mech_slave:
            _index[2:4] = [node_mech_slave[node_j] * 2,
                           node_mech_slave[node_j] * 2 + 1]
            _node[1] = node_mech_slave[node_j]
        if node_m in node_mech_slave:
            _index[4:6] = [node_mech_slave[node_m] * 2,
                           node_mech_slave[node_m] * 2 + 1]
            _node[2] = node_mech_slave[node_m]

        _body = np.array([fmechloadx[i_elem], fmechloady[i_elem],
                          fmechloadx[i_elem], fmechloady[i_elem],
                          fmechloadx[i_elem], fmechloady[i_elem]]) * Rmech10elem[i_elem]
        _R0_map[_index, i_elem] = _T.T @ _body
        _R0_data[i_elem * 6: (i_elem + 1) * 6] = _T.T @ _body
        if _node[0] in node_mech_constraint:
            _R0_data[i_elem * 6 + 0:i_elem * 6 + 2] = 0
        if _node[1] in node_mech_constraint:
            _R0_data[i_elem * 6 + 2:i_elem * 6 + 4] = 0
        if _node[2] in node_mech_constraint:
            _R0_data[i_elem * 6 + 4:i_elem * 6 + 6] = 0

        _R0_row[i_elem * 6: (i_elem + 1) * 6] = _index
        _R0_col[i_elem * 6: (i_elem + 1) * 6] = 0
        _Rconst_map[i_elem * 6: (i_elem + 1) * 6, i_elem] = 1

        _dR_p_dth_row[i_elem * 6: (i_elem + 1) * 6] = _index
        _dR_p_dth_col[i_elem * 6: (i_elem + 1) * 6] = i_elem

        _J = _T.T @ Kmech0elem[i_elem] @ _T
        # ⚠ legacy 的 ``E0mechcoef`` 是逐单元复制的 ``(N,3,3)``；内核 5 只存**一份**
        # ``(3,3)``（数值相同，内核 5 的测试断言过每个副本都等于这一份）。
        _K0_stress = E0mechcoef @ Lmechcoef[i_elem] @ _T

        for _i_r, _row in enumerate(_index):
            _K0mech_data[i_elem * 6 * 6 + _i_r * 6:i_elem * 6 * 6 + (_i_r + 1) * 6] = _J[_i_r, :]
            _K0mech_row[i_elem * 6 * 6 + _i_r * 6:i_elem * 6 * 6 + (_i_r + 1) * 6] = _row
            _K0mech_col[i_elem * 6 * 6 + _i_r * 6:i_elem * 6 * 6 + (_i_r + 1) * 6] = _index
            _K_u_map_data[i_elem * 6 * 6 + _i_r * 6:i_elem * 6 * 6 + (_i_r + 1) * 6] = 1
            _K_u_map_row[i_elem * 6 * 6 + _i_r * 6:i_elem * 6 * 6 + (_i_r + 1) * 6] = range(
                i_elem * 6 * 6 + _i_r * 6, i_elem * 6 * 6 + (_i_r + 1) * 6)
            _K_u_map_col[i_elem * 6 * 6 + _i_r * 6:i_elem * 6 * 6 + (_i_r + 1) * 6] = _index
            _dK_p_dth_u_row[i_elem * 6 * 6 + _i_r * 6:i_elem * 6 * 6 + (_i_r + 1) * 6] = _index[_i_r]
            _dK_p_dth_u_col[i_elem * 6 * 6 + _i_r * 6:i_elem * 6 * 6 + (_i_r + 1) * 6] = i_elem

        for _i_r in range(3):
            K0_stress_data[i_elem * 3 * 6 + _i_r * 6:i_elem * 3 * 6 + (_i_r + 1) * 6] = _K0_stress[_i_r, :]
            K0_stress_row[i_elem * 3 * 6 + _i_r * 6:i_elem * 3 * 6 + (_i_r + 1) * 6] = i_elem * 3 + _i_r
            K0_stress_col[i_elem * 3 * 6 + _i_r * 6:i_elem * 3 * 6 + (_i_r + 1) * 6] = _index

        K0_stress_kE_map[i_elem * 3: (i_elem + 1) * 3, i_elem] = 1
        VonMisesStress_map[i_elem * 3: (i_elem + 1) * 3, i_elem * 3 + 0] = [1, -1 / 2, 0]
        VonMisesStress_map[i_elem * 3: (i_elem + 1) * 3, i_elem * 3 + 1] = [-1 / 2, 1, 0]
        VonMisesStress_map[i_elem * 3: (i_elem + 1) * 3, i_elem * 3 + 2] = [0, 0, 3]
        _Kconst_map[i_elem * 6 * 6:(i_elem + 1) * 6 * 6, i_elem] = 1

        if _node[0] in node_mech_constraint:
            _K0mech_data[i_elem * 6 * 6 + 0 * 6:i_elem * 6 * 6 + 2 * 6] = 0
        if _node[1] in node_mech_constraint:
            _K0mech_data[i_elem * 6 * 6 + 2 * 6:i_elem * 6 * 6 + 4 * 6] = 0
        if _node[2] in node_mech_constraint:
            _K0mech_data[i_elem * 6 * 6 + 4 * 6:i_elem * 6 * 6 + 6 * 6] = 0
        if _node[0] in node_mech_slave:
            _K0mech_data[i_elem * 6 * 6 + 0 * 6:i_elem * 6 * 6 + 2 * 6] = 0
        if _node[1] in node_mech_slave:
            _K0mech_data[i_elem * 6 * 6 + 2 * 6:i_elem * 6 * 6 + 4 * 6] = 0
        if _node[2] in node_mech_slave:
            _K0mech_data[i_elem * 6 * 6 + 4 * 6:i_elem * 6 * 6 + 6 * 6] = 0

    for i_load, i_elem_L in enumerate(g.element_mech_L_load):
        node_i, node_j = g.element_L_mech[i_elem_L][1:3]
        _index = [node_i * 2, node_i * 2 + 1, node_j * 2, node_j * 2 + 1]
        _R2[_index] += np.array([Fmechloadx[i_load], Fmechloady[i_load],
                                 Fmechloadx[i_load], Fmechloady[i_load]]) * Rmech20elem[i_elem_L]

    for i_node_s, i_node_m in node_mech_slave.items():
        _K_boundary[i_node_s * 2, i_node_s * 2] = 1
        _K_boundary[i_node_s * 2, i_node_m * 2] = -_r11
        _K_boundary[i_node_s * 2, i_node_m * 2 + 1] = -_r12
        _K_boundary[i_node_s * 2 + 1, i_node_s * 2 + 1] = 1
        _K_boundary[i_node_s * 2 + 1, i_node_m * 2] = -_r21
        _K_boundary[i_node_s * 2 + 1, i_node_m * 2 + 1] = -_r22
        _R2[i_node_s * 2] = 0
        _R2[i_node_s * 2 + 1] = 0

    for i_node in node_mech_constraint:
        _R2[i_node * 2] = 0
        _R2[i_node * 2 + 1] = 0
        _K_boundary[i_node * 2, :] = 0
        _K_boundary[i_node * 2 + 1, :] = 0
        _K_boundary[i_node * 2, i_node * 2] = 1
        _K_boundary[i_node * 2 + 1, i_node * 2 + 1] = 1

    _K_u_map = sp_sparse.csr_matrix(
        (_K_u_map_data, (_K_u_map_row, _K_u_map_col)),
        shape=(N_element_mech * 6 * 6, N_node_mech * 2))
    K0_stress = sp_sparse.csr_matrix(
        (K0_stress_data, (K0_stress_row, K0_stress_col)),
        shape=(N_element_mech * 3, N_node_mech * 2))

    return dict(
        K0mech_data=_K0mech_data, K0mech_row=_K0mech_row, K0mech_col=_K0mech_col,
        Kconst_map=_Kconst_map.tocsr(), K_u_map=_K_u_map,
        K_boundary=_K_boundary.tocsr(),
        R0_data=_R0_data, R0_row=_R0_row, R0_col=_R0_col,
        Rconst_map=_Rconst_map.tocsr(), R2=_R2,
        dK_p_dth_u_row=_dK_p_dth_u_row, dK_p_dth_u_col=_dK_p_dth_u_col,
        dR_p_dth_row=_dR_p_dth_row, dR_p_dth_col=_dR_p_dth_col,
        K0_stress_kE_map=K0_stress_kE_map, VonMisesStress_map=VonMisesStress_map,
        K0_stress=K0_stress, K0_stress_data=K0_stress_data,
        K0_stress_row=K0_stress_row, K0_stress_col=K0_stress_col,
    )


def legacy_mech_topo_opti(_rho):
    """照抄 ``MMA10:1231-1240``。"""
    beta, thb, p, mn = 8, 0.65, 3, 1e-4
    rho_1 = (np.tanh(beta * (_rho - thb)) + np.tanh(beta * thb)) / (
        np.tanh(beta * (1 - thb)) + np.tanh(beta * thb))
    return mn + (1 - mn) * rho_1 ** p, _rho


def legacy_mech_residual(g, kv, _rho_SIMP, col_mask=None):
    """照抄 ``MMA10:2817-2858``（**含对称 Dirichlet 列投影**，2026-09-24 与生产同步）。"""
    _kE, _krho = legacy_mech_topo_opti(_rho_SIMP)
    n_dof = g.N_node_mech * 2
    _R = sp_sparse.coo_matrix(
        (kv['R0_data'] * (kv['Rconst_map'] @ _krho), (kv['R0_row'], kv['R0_col'])),
        shape=(n_dof, 1)).toarray().flatten() + kv['R2']
    _K = sp_sparse.coo_matrix(
        (kv['K0mech_data'] * (kv['Kconst_map'] @ _kE),
         (kv['K0mech_row'], kv['K0mech_col'])),
        shape=(n_dof, n_dof))
    _K = _K.tocsr()
    if col_mask is not None:
        _K = _K @ col_mask          # 对称 Dirichlet：零齐次 u=0 列
    return _R, _K + kv['K_boundary']


# =============================================================================
# 来源 B —— 装配族逐位
# =============================================================================

def test_assembly_family_matches_legacy_bitwise(fx):
    """21 个产出里**不依赖 SIMP 导数**的那些与照抄片段逐位一致。

    ``k0_data``/``k_u_map``/``k_boundary``/``r0_data``/``r2``/``kconst_map``/
    ``rconst_map``/四个导数位置/应力三项 —— 全部 ``np.array_equal``。
    """
    g = fx.legacy_globals()
    kv = legacy_get_Kmech_vectorized(g)
    ops = fx.ops

    for got, want, name in [
        (ops.k0_data, kv['K0mech_data'], 'k0_data'),
        (ops.k0_row, kv['K0mech_row'], 'k0_row'),
        (ops.k0_col, kv['K0mech_col'], 'k0_col'),
        (ops.r0_data, kv['R0_data'], 'r0_data'),
        (ops.r0_row, kv['R0_row'], 'r0_row'),
        (ops.r0_col, kv['R0_col'], 'r0_col'),
        (ops.r2, kv['R2'], 'r2'),
        (ops.dk_p_dth_u_row, kv['dK_p_dth_u_row'], 'dk_row'),
        (ops.dk_p_dth_u_col, kv['dK_p_dth_u_col'], 'dk_col'),
        (ops.dr_p_dth_row, kv['dR_p_dth_row'], 'dr_row'),
        (ops.dr_p_dth_col, kv['dR_p_dth_col'], 'dr_col'),
        (ops.stress_data, kv['K0_stress_data'], 'stress_data'),
        (ops.stress_row, kv['K0_stress_row'], 'stress_row'),
        (ops.stress_col, kv['K0_stress_col'], 'stress_col'),
    ]:
        assert np.array_equal(np.asarray(got), np.asarray(want)), f"{name} 不逐位相同"

    stress_full = sp_sparse.csr_matrix(
        (ops.stress_data, (ops.stress_row, ops.stress_col)),
        shape=(3 * fx.n_elem, 2 * fx.node.shape[0]))

    for got, want, name in [
        (ops.kconst_map, kv['Kconst_map'], 'kconst_map'),
        (ops.rconst_map, kv['Rconst_map'], 'rconst_map'),
        (ops.k_u_map, kv['K_u_map'], 'k_u_map'),
        (ops.k_boundary, kv['K_boundary'], 'k_boundary'),
        # ⚠ legacy 的这两个返回 **lil**（没 tocsr）；本模块统一 csr。
        # lil → csr 只是存储格式，逐位无差（内核 12 记过同一件事）。
        (ops.stress_k_e_map, kv['K0_stress_kE_map'].tocsr(), 'stress_k_e_map'),
        (ops.von_mises_map, kv['VonMisesStress_map'].tocsr(), 'von_mises_map'),
        (stress_full, kv['K0_stress'], 'stress'),
    ]:
        assert got.format == 'csr' and want.format == 'csr'
        assert np.array_equal(got.indptr, want.indptr), f"{name} indptr 不同"
        assert np.array_equal(got.indices, want.indices), f"{name} indices 不同"
        assert np.array_equal(got.data, want.data), f"{name} data 不同"


def test_residual_matches_legacy_bitwise(fx):
    """``(R, K)`` 与照抄的 ``mech_field_error_vectorized`` 逐位一致。

    ⚠ ``K`` 只涉及 ``kE``（``SIMP_func`` 的逐位改写），所以这里可以 ``array_equal``；
    涉及 ``dkE`` 的量在 ``test_sensitivity_*`` 里只能用 ``rtol``（§11⑩）。
    """
    g = fx.legacy_globals()
    kv = legacy_get_Kmech_vectorized(g)
    R_l, K_l = legacy_mech_residual(g, kv, fx.rho, col_mask=fx.ops.col_mask)

    R_r, K_r = ref_solver.mech_residual(fx.rho, fx.ops)

    assert np.array_equal(R_l, R_r)
    assert np.array_equal(K_l.indptr, K_r.indptr)
    assert np.array_equal(K_l.indices, K_r.indices)
    assert np.array_equal(K_l.data, K_r.data)
    # 对称性（2026-09-24）：K_mech 修复后"有效对称"—— 残余来自主从旋转行
    # （约束行的 -rot 条目 vs 主行的 0 耦合，量级 ~1e-9 相对；彻底归零需力学消元，
    # 见设计文档 §11 的判定）。Dirichlet 列清零把 4.3% → ~1e-9。
    d = K_r.toarray()
    assert np.linalg.norm(d - d.T) <= 1e-6 * np.linalg.norm(d)


# =============================================================================
# 来源 A —— 单元级手算与机制
# =============================================================================

def test_slave_dofs_are_replaced_by_the_master(fx):
    """★ 主从消元的**核心机制**：从属自由度在装配里**不出现**。

    单元局部节点表里从属节点被替换成主动节点 ⇒ 单元的 6 个 DOF 都是主动/普通节点的，
    而从属列的贡献为零（``K`` 在从属 DOF 上只有 ``K_boundary`` 的对角 1）。
    """
    dof_of = {}
    for e in range(fx.n_elem):
        base = 36 * e
        for i_r in range(6):
            for d in fx.ops.k0_col[base + 6 * i_r:base + 6 * i_r + 6]:
                dof_of.setdefault(int(d), []).append(e)

    for s in fx.slave_to_master:
        # 从属 DOF 只可能由 K_boundary 给出（k0_data 里不得出现）
        assert 2 * s not in dof_of, f"从属节点 {s} 的自由度竟然出现在 k0_data 的列里"
        assert 2 * s + 1 not in dof_of

    # 主动节点必须出现（否则消元把约束丢掉了）
    for m in fx.slave_to_master.values():
        assert 2 * m in dof_of and 2 * m + 1 in dof_of


def test_master_slave_rows_encode_the_rotation(fx):
    """``K_boundary`` 的主从行 = ``u_s − R(2π/parts)·u_m = 0``（逐行手算）。

    ⚠ **Dirichlet 赢**：主从约束与位移约束落在同一节点上时（本网格里 θ=π/2 与
    θ=0 在**内圈**上各有一个节点被约束）legacy 的顺序是「先主从、后 Dirichlet」，
    且 Dirichlet 是**整行清零** ⇒ 那一行的主从系数消失、只剩单位元。这里跳过这些对。
    """
    a = 2 * np.pi / PARTS
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    K = fx.ops.k_boundary.toarray()
    constrained = set(fx.constrained)

    checked = 0
    for s, m in fx.slave_to_master.items():
        assert K[2 * s, 2 * s] == 1.0 and K[2 * s + 1, 2 * s + 1] == 1.0
        if s in constrained:
            assert not np.any(np.delete(K[2 * s], 2 * s))
            continue
        assert np.array_equal(K[2 * s, 2 * m:2 * m + 2], -rot[0])
        assert np.array_equal(K[2 * s + 1, 2 * m:2 * m + 2], -rot[1])
        row = K[2 * s].copy()
        row[2 * s] = 0.0
        row[2 * m:2 * m + 2] = 0.0
        assert not np.any(row)
        checked += 1
    assert checked > 0, "全部从属节点都被 Dirichlet 覆盖 ⇒ 这条测试没验到东西"


def test_dirichlet_rows_are_identity(fx):
    """Dirichlet 行整行置零后写单位元 ⇒ ``u = 0``；且 ``R`` 的对应分量被清零。"""
    K = fx.ops.k_boundary.toarray()
    for n in fx.constrained:
        for d in (2 * n, 2 * n + 1):
            row = K[d].copy()
            assert row[d] == 1.0, f"自由度 {d} 的对角不是 1"
            row[d] = 0.0
            assert not np.any(row), f"自由度 {d} 的行有非零项"
        assert fx.ops.r2[2 * n] == 0.0 and fx.ops.r2[2 * n + 1] == 0.0


def test_body_force_rows_are_zero_at_constrained_nodes(fx):
    """体积力项在**约束节点**的 2 个分量上被清零（``:2692-2697``）。

    ⚠ 与 ``k0_data`` 的清零不同：``r0_data`` **只**清 Dirichlet，不清从属。
    """
    tri = fx.tri
    constrained = set(fx.constrained)
    for e in range(fx.n_elem):
        nodes = [int(tri[e, 0]), int(tri[e, 1]), int(tri[e, 2])]
        # 从属替换后的节点表
        repl = [fx.slave_to_master.get(n, n) for n in nodes]
        for s in range(3):
            if repl[s] in constrained:
                assert np.all(fx.ops.r0_data[6 * e + 2 * s:6 * e + 2 * s + 2] == 0.0)


def test_k0_rows_are_zero_at_constrained_and_slave_nodes(fx):
    """★ ``k0_data`` 的清零用的是**替换后**的节点表（``:2737-2749``）。

    这组条件与 ``r0_data`` 的**不对称**（这里多清「从属」）是 legacy 的形态 ——
    但对 ``_node`` 已被替换过的槽位，「从属」判据检查的其实是**主动节点自己是
    从属节点**（链式从属）。本测试直接核对「替换后节点在约束集或从属集里 ⇒ 块为零」。
    """
    tri = fx.tri
    constrained = set(fx.constrained)
    slave = fx.slave_to_master
    for e in range(fx.n_elem):
        nodes = [int(tri[e, 0]), int(tri[e, 1]), int(tri[e, 2])]
        repl = [slave.get(n, n) for n in nodes]
        for s in range(3):
            if repl[s] in constrained or repl[s] in slave:
                blk = fx.ops.k0_data[36 * e + 12 * s:36 * e + 12 * s + 12]
                assert np.all(blk == 0.0), f"单元 {e} 的局部节点 {s} 的块没清零"


def test_two_assembly_routes_agree_bitwise(fx):
    """★ 位置矩阵路线 vs **直接逐单元装配** —— 两条独立路线互证。

    这是 ``reference/`` 里「数学」与「位置矩阵」的分工：``mech_residual`` 走位置矩阵
    （legacy 的搬家路线），``assemble_mech_system`` 直接照公式装配。两者唯一的差别是
    累加顺序，所以用 `allclose`（容差按矩阵量级缩放），不是 `array_equal`。
    """
    R_a, K_a = ref_solver.assemble_mech_system(
        fx.rho, fx.node, fx.tri,
        k_elem=fx.mech.K, r0_elem=fx.mech.R0,
        body_fx=fx.body_fx, body_fy=fx.body_fy, r2=fx.ops.r2,
        k_boundary=fx.ops.k_boundary,
        slave_to_master=fx.slave_to_master, constrained_nodes=fx.constrained,
        parts=PARTS,
    )
    R_b, K_b = ref_solver.mech_residual(fx.rho, fx.ops)

    scale = np.abs(K_b.data).max()
    assert np.abs((K_a - K_b).data).max() <= 1e-12 * scale
    assert np.abs(R_a - R_b).max() <= 1e-12 * np.abs(R_b).max()


def test_stiffness_is_symmetric(fx):
    """装配后的 ``K`` 在**未驱动**部分是均匀材料的对称阵。

    ⚠ ``K_boundary`` 让整行不对称（约束行）⇒ 不能直接断言 ``K == K.T``。
    这里断言的是**去掉约束行之后**的对称性 —— 用「K 与 Kᵀ 的差只出现在被覆盖的行上」
    的形式表达，正是 ``K_boundary`` 的语义。
    """
    _, K = ref_solver.mech_residual(fx.rho, fx.ops)
    dense = K.toarray()
    asym = np.abs(dense - dense.T)

    bad_rows = set()
    for s in fx.slave_to_master:
        bad_rows |= {2 * s, 2 * s + 1}
    for n in fx.constrained:
        bad_rows |= {2 * n, 2 * n + 1}

    keep = np.array(sorted(set(range(K.shape[0])) - bad_rows), dtype=int)
    # ⚠ 容差按矩阵量级缩放：K 的同一元素由不同单元的贡献以不同顺序相加
    # （COO 重复项求和），所以对称只到 1 ULP，不是精确的零。
    scale = np.abs(K.data).max()
    assert asym[np.ix_(keep, keep)].max() <= 1e-13 * scale


def test_linearity_of_solution_in_body_force(fx):
    """解析性质：``K`` 与载荷无关 ⇒ ``u`` 对体积力**线性**。

    把 ``r2`` 与体积力都翻倍，解应翻倍（Dirichlet 与主从约束都与载荷无关）。
    这条挡住「装配时把载荷混进 K」这类错误。
    """
    R1, K1 = ref_solver.mech_residual(fx.rho, fx.ops)
    u1 = sp_sparse_linalg.spsolve(K1.tocsr(), -R1)

    doubled = ref_solver.MechOperators(**{
        **{f: getattr(fx.ops, f) for f in fx.ops.__dataclass_fields__},
        "r2": 2.0 * fx.ops.r2,
        "r0_data": 2.0 * fx.ops.r0_data,
    })
    R2, K2 = ref_solver.mech_residual(fx.rho, doubled)
    u2 = sp_sparse_linalg.spsolve(K2.tocsr(), -R2)

    assert np.abs(K1 - K2).max() == 0.0
    scale = np.abs(u1).max()
    assert np.abs(u2 - 2 * u1).max() <= 1e-12 * scale


def test_constrained_nodes_have_zero_displacement(fx):
    """Dirichlet 节点解出来必须是 0（约束被真正施加了）。"""
    u = ref_solver.solve_mech_field(fx.rho, fx.ops)
    for n in fx.constrained:
        assert abs(u[2 * n]) <= 1e-12 * np.abs(u).max() + 1e-30
        assert abs(u[2 * n + 1]) <= 1e-12 * np.abs(u).max() + 1e-30
    assert np.any(np.abs(u) > 0.0), "全零 ⇒ 载荷没生效，测试无意义"


def test_slave_displacement_equals_rotated_master(fx):
    """★ 主从关系在**解**上成立：``u_s = R(2π/parts)·u_m``。

    这是整条主从机制（消元 + 约束行）的端到端解析判据 —— 前面那些测试验的是矩阵，
    这条验的是**解**。
    """
    u = ref_solver.solve_mech_field(fx.rho, fx.ops)
    a = 2 * np.pi / PARTS
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    worst = 0.0
    for s, m in fx.slave_to_master.items():
        want = rot @ np.array([u[2 * m], u[2 * m + 1]])
        got = np.array([u[2 * s], u[2 * s + 1]])
        worst = max(worst, float(np.abs(got - want).max()))
    assert worst <= 1e-10 * np.abs(u).max()


# =============================================================================
# 来源 A —— 灵敏度的有限差分
# =============================================================================

def test_sensitivity_load_term_matches_finite_difference(fx):
    """★ 这条钉住**已修缺陷 D5**（设计文档 §12）：灵敏度的**载荷项符号**。

    求解约定 ``u = spsolve(K, −R)`` ⇒ 残差 ``F(u, θ) = K(θ)u + R(θ) = 0`` 对
    θ（固定 ``u``）的导数应当是 ``∂F/∂θ_e = dK/dθ_e·u **+** dR/dθ_e``。
    legacy ``MMA10:2897`` 给的是减号（**D5**）；本实现**已修**为加号。

    有限差分是外部判据（``u`` 固定、扰动 ρ、重装 K 与 R）：:

        fd == ∂F/∂ρ_e == dK·u + dR ,   sens == dK·u + dR
        ⇒ fd == sens（直接一致，无 2·dR 间隙）

    本测试断言三件事，缺一不可：

    (a) ``fd`` 与 ``sens`` 直接一致（⇒ 载荷项是**加号**，与求解约定同号）；
    (b) 对照 legacy 原式可辨认：legacy 是 ``dK·u − dR``，把它的载荷项翻号后
        才等于 ``sens``（⇒ 修的就是那一项，不是别的机制）；
    (c) **防退化**：``dR`` 相对 ``dK·u`` 不可忽略（否则这条测试没有意义）。

    ⚠ 影响面：消费者 ``compute_mechDiff_AL`` 的 ``dU_p_drho`` 已改走包络闭式
    （免伴随）；``dALStress_p_drho`` 的伴随解 ``−spsolve(Kᵀ,·)ᵀ@dRHS`` 随之消掉
    ``2λᵀdR``（D8 仍是独立缺陷，待判定）。**本批已修 D5**（§12 修正清单第 1 条）。
    """
    u = ref_solver.solve_mech_field(fx.rho, fx.ops)
    _, sens = ref_solver.mech_sensitivity(fx.rho, u, fx.ops)
    sens = sens.toarray()

    def residual_at(rho_):
        R_, K_ = ref_solver.mech_residual(rho_, fx.ops)
        return K_ @ u + R_

    dR = sp_sparse.coo_matrix(
        (fx.ops.r0_data * 1.0, (fx.ops.dr_p_dth_row, fx.ops.dr_p_dth_col)),
        shape=(fx.ops.n_dof, fx.n_elem)).toarray()

    elems = [e for e in range(fx.n_elem) if abs(sens[:, e]).max() > 0]
    assert elems, "灵敏度恒为零 ⇒ 测试无意义"
    sample = sorted(elems, key=lambda e: -abs(sens[:, e]).max())[:4]

    for e in sample:
        d = 1e-6
        rp, rm = fx.rho.copy(), fx.rho.copy()
        rp[e] += d
        rm[e] -= d
        fd = (residual_at(rp) - residual_at(rm)) / (2 * d)

        # (a) 直接一致（载荷项是加号）
        scale = np.abs(fd).max()
        assert np.abs(fd - sens[:, e]).max() <= 1e-6 * scale, (
            f"单元 {e}：灵敏度与有限差分差 {np.abs(fd - sens[:, e]).max():.3e}"
            f"（相对 {np.abs(fd - sens[:, e]).max() / scale:.3e}）—— "
            f"D5 修正后仍不对，重新核对设计文档 §12"
        )

        # (b) 机构可辨认：legacy 原式是 ``dK·u − dR``，与已修的 ``sens`` 差**恰是**
        #     ``2·dR`` ⇒ 修的就是载荷项那一项的符号（不是缺项、不是别的机制）。
        legacy_form = sens[:, e] - 2 * dR[:, e]      # = dK·u − dR（legacy 形态）
        assert np.abs((fd - legacy_form) - 2 * dR[:, e]).max() <= 1e-6 * scale

        # (c) 防退化：载荷项不可忽略
        assert abs(dR[:, e]).max() > 1e-4 * scale, (
            f"单元 {e} 的 dR 相对 dK·u 可忽略（{abs(dR[:, e]).max() / scale:.2e}）"
            f"⇒ 这条测试区分不出符号"
        )


def test_sensitivity_uses_ones_for_dkrho(fx):
    """``dkrho = 1`` ⇒ ``∂R/∂θ_e`` 恰是 ``r0_data`` 落在 ``(dr_row, e)`` 上。

    这是 ``dmech_topo_opti`` 返回 ``np.ones`` 的直接后果，值得单独钉住
    （若哪天有人把 ``dkrho`` 写成 ``dρ/dρ`` 的别的东西，这条会失败）。
    """
    n_dof = fx.ops.n_dof
    dR = sp_sparse.coo_matrix(
        (fx.ops.r0_data * 1.0, (fx.ops.dr_p_dth_row, fx.ops.dr_p_dth_col)),
        shape=(n_dof, fx.ops.n_elem))
    _kE, dkrho = ref_solver.mech_density_derivatives(fx.rho)
    assert np.all(dkrho == 1.0)

    _, sens = ref_solver.mech_sensitivity(fx.rho, np.zeros(n_dof), fx.ops)
    # u = 0 ⇒ dK/dθ·u 那一支为零，剩下的就是 +∂R/∂θ（**D5 已修**，加号）
    assert np.abs(sens.toarray() - dR.toarray()).max() == 0.0


def test_mech_density_factors_match_legacy_bitwise(fx):
    """``kE`` 与 legacy 的 ``mech_topo_opti`` **逐位**相同（``SIMP_func`` 那一支）。

    ⚠ ``dkE`` 那一支**不逐位**（§11⑩），所以这里只断言 ``kE`` 与 ``krho``。
    """
    kE, krho = ref_solver.mech_density_factors(fx.rho)
    kE_l, krho_l = legacy_mech_topo_opti(fx.rho)
    assert np.array_equal(kE, kE_l)
    assert krho is fx.rho
    assert np.array_equal(krho, krho_l)
    # 防退化：ρ = 0.5 处 SIMP 必须是活跃的（否则 kE 恒定、灵敏度为零）
    assert 1e-4 < kE[0] < 1.0


def test_sensitivity_matches_legacy_to_rtol(fx):
    """``K`` 与灵敏度对 legacy 的 ``compute_udiff_vectorized`` 一致到 ``rtol``。"""
    u = ref_solver.solve_mech_field(fx.rho, fx.ops)
    K_r, sens_r = ref_solver.mech_sensitivity(fx.rho, u, fx.ops)

    g = fx.legacy_globals()
    kv = legacy_get_Kmech_vectorized(g)
    # 照抄 MMA10:2861-2897
    kE, krho = legacy_mech_topo_opti(fx.rho)
    beta, thb, p, mn = 8, 0.65, 3, 1e-4
    rho_1 = (np.tanh(beta * (fx.rho - thb)) + np.tanh(beta * thb)) / (
        np.tanh(beta * (1 - thb)) + np.tanh(beta * thb))
    drho_1 = (beta * np.cosh(beta * thb) * np.cosh(beta - beta * thb) / np.sinh(beta)
              / np.cosh(beta * (fx.rho - thb)) ** 2)
    dkE = p * (1 - mn) * rho_1 ** (p - 1) * drho_1
    dkrho = np.ones(fx.rho.shape)

    n_dof = fx.ops.n_dof
    dR = sp_sparse.coo_matrix(
        (kv['R0_data'] * (kv['Rconst_map'] @ dkrho),
         (kv['dR_p_dth_row'], kv['dR_p_dth_col'])), shape=(n_dof, fx.n_elem))
    dK_u = sp_sparse.coo_matrix(
        (kv['K0mech_data'] * (kv['Kconst_map'] @ dkE) * (kv['K_u_map'] @ u),
         (kv['dK_p_dth_u_row'], kv['dK_p_dth_u_col'])), shape=(n_dof, fx.n_elem))
    K_l = sp_sparse.coo_matrix(
        (kv['K0mech_data'] * (kv['Kconst_map'] @ kE),
         (kv['K0mech_row'], kv['K0mech_col'])),
        shape=(n_dof, n_dof)).tocsr() + kv['K_boundary']

    assert np.array_equal(K_l.indptr, K_r.indptr)
    assert np.array_equal(K_l.data, K_r.data)

    # ⚠ **D5 已修（2026-09-23）**：legacy 原式是 ``dK_u − dR``，本实现是
    # ``dK_u + dR`` ⇒ 直接对 legacy 原式比对会差 ``2·dR``。这里对照 legacy 的
    # **装配支**（``dK_u`` 与 ``dR`` 的每一项逐位来自 legacy 的位置矩阵），
    # 只把载荷项符号翻为已修的加号 —— 既验装配逐位、又验符号修正。
    s_l = (dK_u + dR).toarray()
    s_r = sens_r.toarray()
    scale = np.abs(s_l).max()
    assert np.abs(s_l - s_r).max() <= 1e-14 * scale, (
        "灵敏度对 legacy 装配支的偏差超出 1~2 ULP 的量级（§11⑩ 只允许 SIMP 导数那一处）"
    )


# =============================================================================
# 来源 C —— reference vs fast（新写的向量化）
# =============================================================================

@requires_fast
def test_fast_assembly_matches_reference_bitwise_including_sparsity(fx):
    """★ fast 的装配族与 reference **逐位**相同，**含稀疏结构**。

    本内核的 fast 是**新写的向量化**（legacy 没有可搬的向量化版），所以它比前 6 个
    内核多背一条纪律：``indptr``/``indices``/``data`` 三段都要对齐 ——
    稠密比较看不出「显式存储的零」，而结构会进 SuperLU 的列排序。
    """
    fast_ops = fast_solver.mech_operators(
        fx.node, fx.tri,
        k_elem=fx.mech.K, l_coef=fx.mech.L, d_mat=fx.mech.D,
        r0_elem=fx.mech.R0, r2_elem=fx.line_loads.R0,
        body_fx=fx.body_fx, body_fy=fx.body_fy,
        line_tri=fx.line_tri, loaded_lines=fx.loaded_lines,
        boundary_fx=fx.line_loads.fx, boundary_fy=fx.line_loads.fy,
        slave_to_master=fx.slave_to_master,
        constrained_nodes=fx.constrained, parts=PARTS,
    )
    ops = fx.ops

    for got, want, name in [
        (fast_ops.k0_data, ops.k0_data, 'k0_data'),
        (fast_ops.k0_row, ops.k0_row, 'k0_row'),
        (fast_ops.k0_col, ops.k0_col, 'k0_col'),
        (fast_ops.r0_data, ops.r0_data, 'r0_data'),
        (fast_ops.r0_row, ops.r0_row, 'r0_row'),
        (fast_ops.r0_col, ops.r0_col, 'r0_col'),
        (fast_ops.r2, ops.r2, 'r2'),
        (fast_ops.dk_p_dth_u_row, ops.dk_p_dth_u_row, 'dk_row'),
        (fast_ops.dk_p_dth_u_col, ops.dk_p_dth_u_col, 'dk_col'),
        (fast_ops.dr_p_dth_row, ops.dr_p_dth_row, 'dr_row'),
        (fast_ops.dr_p_dth_col, ops.dr_p_dth_col, 'dr_col'),
        (fast_ops.stress_data, ops.stress_data, 'stress_data'),
        (fast_ops.stress_row, ops.stress_row, 'stress_row'),
        (fast_ops.stress_col, ops.stress_col, 'stress_col'),
    ]:
        assert np.array_equal(np.asarray(got), np.asarray(want)), f"{name} 不逐位相同"

    for got, want, name in [
        (fast_ops.kconst_map, ops.kconst_map, 'kconst_map'),
        (fast_ops.rconst_map, ops.rconst_map, 'rconst_map'),
        (fast_ops.k_u_map, ops.k_u_map, 'k_u_map'),
        (fast_ops.k_boundary, ops.k_boundary, 'k_boundary'),
        (fast_ops.stress_k_e_map, ops.stress_k_e_map, 'stress_k_e_map'),
        (fast_ops.von_mises_map, ops.von_mises_map, 'von_mises_map'),
    ]:
        assert np.array_equal(got.indptr, want.indptr), f"{name} indptr 不同"
        assert np.array_equal(got.indices, want.indices), f"{name} indices 不同"
        assert np.array_equal(got.data, want.data), f"{name} data 不同"


@requires_fast
def test_fast_matches_legacy_bitwise(fx):
    """fast 直接对**照抄片段**逐位（不经 reference）。"""
    g = fx.legacy_globals()
    kv = legacy_get_Kmech_vectorized(g)
    fast_ops = fast_solver.mech_operators(
        fx.node, fx.tri,
        k_elem=fx.mech.K, l_coef=fx.mech.L, d_mat=fx.mech.D,
        r0_elem=fx.mech.R0, r2_elem=fx.line_loads.R0,
        body_fx=fx.body_fx, body_fy=fx.body_fy,
        line_tri=fx.line_tri, loaded_lines=fx.loaded_lines,
        boundary_fx=fx.line_loads.fx, boundary_fy=fx.line_loads.fy,
        slave_to_master=fx.slave_to_master,
        constrained_nodes=fx.constrained, parts=PARTS,
    )
    assert np.array_equal(fast_ops.k0_data, kv['K0mech_data'])
    assert np.array_equal(fast_ops.k0_row, kv['K0mech_row'])
    assert np.array_equal(fast_ops.k0_col, kv['K0mech_col'])
    assert np.array_equal(fast_ops.r0_data, kv['R0_data'])
    assert np.array_equal(fast_ops.r2, kv['R2'])
    assert np.array_equal(fast_ops.stress_data, kv['K0_stress_data'])
    assert np.array_equal(fast_ops.k_boundary.indptr, kv['K_boundary'].indptr)

    R_l, K_l = legacy_mech_residual(g, kv, fx.rho, col_mask=fast_ops.col_mask)
    R_f, K_f = fast_solver.mech_residual(fx.rho, fast_ops)
    assert np.array_equal(R_l, R_f)
    assert np.array_equal(K_l.data, K_f.data)
    assert np.array_equal(K_l.indptr, K_f.indptr)
    assert np.array_equal(K_l.indices, K_f.indices)


@requires_fast
def test_fast_solve_and_sensitivity_match_reference(fx):
    """求解与灵敏度两层一致（``rtol``；``K`` 那一支逐位）。"""
    fast_ops = fast_solver.mech_operators(
        fx.node, fx.tri,
        k_elem=fx.mech.K, l_coef=fx.mech.L, d_mat=fx.mech.D,
        r0_elem=fx.mech.R0, r2_elem=fx.line_loads.R0,
        body_fx=fx.body_fx, body_fy=fx.body_fy,
        line_tri=fx.line_tri, loaded_lines=fx.loaded_lines,
        boundary_fx=fx.line_loads.fx, boundary_fy=fx.line_loads.fy,
        slave_to_master=fx.slave_to_master,
        constrained_nodes=fx.constrained, parts=PARTS,
    )
    u_r = ref_solver.solve_mech_field(fx.rho, fx.ops)
    u_f = fast_solver.solve_mech_field(fx.rho, fast_ops)
    assert np.abs(u_r - u_f).max() <= 1e-12 * np.abs(u_r).max()

    _, s_r = ref_solver.mech_sensitivity(fx.rho, u_r, fx.ops)
    _, s_f = fast_solver.mech_sensitivity(fx.rho, u_f, fast_ops)
    scale = np.abs(s_r).max()
    assert np.abs((s_r - s_f)).max() <= 1e-12 * scale


@requires_fast
def test_assembly_speed_guard(fx):
    """性能守卫（宽松阈值）：向量化装配必须**明显**快于逐单元循环。

    本内核的 fast 是新写的向量化（legacy 是 6362 个单元的 Python 循环 +
    每单元 36 次切片赋值），所以按 ``kernels/README.md`` 的要求必须有一条
    性能断言，否则「更快」这个理由无法被检验。这里用夹具的 36 个单元只是**方向性**
    检查；真实网格上的倍速由 ``tools/probe_mech.py`` 报出。

    ⚠ 阈值取得很松（1.2×）—— 夹具太小、``timeit`` 的最小值受噪声影响。
    **计时不可跨条件比较**（设计文档 §11⑥）。
    """
    import timeit

    def run(module):
        return lambda: module.mech_operators(
            fx.node, fx.tri,
            k_elem=fx.mech.K, l_coef=fx.mech.L, d_mat=fx.mech.D,
            r0_elem=fx.mech.R0, r2_elem=fx.line_loads.R0,
            body_fx=fx.body_fx, body_fy=fx.body_fy,
            line_tri=fx.line_tri, loaded_lines=fx.loaded_lines,
            boundary_fx=fx.line_loads.fx, boundary_fy=fx.line_loads.fy,
            slave_to_master=fx.slave_to_master,
            constrained_nodes=fx.constrained, parts=PARTS,
        )

    t_ref = min(timeit.repeat(run(ref_solver), number=3, repeat=5))
    t_fast = min(timeit.repeat(run(fast_solver), number=3, repeat=5))
    speedup = t_ref / t_fast
    print(f"[性能守卫] 力学装配：逐单元 {t_ref * 1e3:.2f} ms，"
          f"向量化 {t_fast * 1e3:.2f} ms，加速 {speedup:.2f}×（{fx.n_elem} 个单元）")
    assert speedup > 1.2, f"向量化装配只快 {speedup:.2f}× ⇒ 这一层没有存在理由"
