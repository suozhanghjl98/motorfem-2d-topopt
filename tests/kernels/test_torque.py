"""内核 12/13/14 测试：Arkkio 系数与导数、虚位移系数与导数。

测试三级来源（`docs/重构设计文档.md` §1.4 与 `CLAUDE.md` §4）：

  来源 A 解析解/解析性质
        内核 12 —— 单单元闭式（``r̄``/``θ̄``/``K_e`` 手算）、系数矩阵**逐位对称**、
                   二次型只看见对称部分、只有「气隙 ∩ 含动节点」的单元参与
        内核 13 —— **有限差分**：扰动 γ_e 重解场、比转矩；伴随恒等式
        内核 14 —— 同上 + v2 变换矩阵的 ``dT`` 与内核 7 的 ``d_matrix`` 一致
  来源 B 与 legacy 一致 —— 照抄 ``MMA10:666-701`` 等片段到本文件，**逐位**比对
  来源 C reference vs fast —— 见本文件末节

⚠ **F4 在本批是「原始案例」**：``get_Arkkio_coef`` 遍历的是 ``element_rotor_gap``（一个
``set``）并在 ``lil`` 上逐项 ``+=`` ⇒ 加数顺序由 set 迭代顺序决定。所以
「``gap_elements`` 必须是真正的 ``set``」是硬约束，本文件有专门的守卫测试。

运行::

    python -m pytest tests/kernels/test_torque.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse as sp_sparse

from motorfem.kernels.reference import assembly as ref_asm
from motorfem.kernels.reference import boundaries as ref_bnd
from motorfem.kernels.reference import element_matrices as ref_em
from motorfem.kernels.reference import moving_mesh as ref_mm
from motorfem.kernels.reference import newton as ref_nt
from motorfem.kernels.reference import torque as ref_tq
from motorfem.kernels.reference.assembly import TopologyContext
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection

try:
    from motorfem.kernels.fast import torque as fast_tq
except ImportError:  # pragma: no cover
    fast_tq = None

requires_fast = pytest.mark.skipif(fast_tq is None, reason="fast/ 层尚未落地")

SECTOR = np.pi / 2
COORD_ERR_ABS = 1e-6
R_RINGS = (0.030, 0.040, 0.050, 0.060)
PARTS = 4
MU0 = 4e-7 * np.pi
#: legacy 硬编码在 ``MMA10:693`` 的 ``0.25e-3``
GAP_M = 0.25e-3


def make_donut_mesh(n_ang=7, radii=R_RINGS):
    """四层同心圆环扇。节点号 ``= 层序号·n_ang + 角度序号``，含两端点。"""
    angles = np.linspace(0.0, SECTOR, n_ang)
    xy = np.array(
        [[r * np.cos(t), r * np.sin(t)] for r in radii for t in angles], dtype=float
    )
    element_S = []
    for k in range(len(radii) - 1):
        for l in range(n_ang - 1):
            a = k * n_ang + l
            b = (k + 1) * n_ang + l
            c = (k + 1) * n_ang + l + 1
            d = k * n_ang + l + 1
            element_S.append([len(element_S), a, b, c, k])
            element_S.append([len(element_S), a, c, d, k])
    element_rotor_gap = set([e[0] for e in element_S if e[4] == 1])
    element_stator = set([e[0] for e in element_S if e[4] == 2])
    element_rotor = set([e[0] for e in element_S if e[4] in (0, 1)])
    return xy, element_S, element_rotor_gap, element_stator, element_rotor


def interface_nodes_legacy(xy, r_interface=0.050, coord_err_abs=COORD_ERR_ABS):
    """照抄 ``MMA10:339-340``：半径**平方差**判据（返回 set，顺序与 legacy 同源）。"""
    return set(
        i for i in range(xy.shape[0])
        if abs((xy[i, 0] ** 2 + xy[i, 1] ** 2) - r_interface ** 2) < coord_err_abs
    )


class TorqueFixture:
    """同心圆环扇 + 内核 6/7/8 产出 + 一个解好的场。

    **为什么夹具要解一次场**：内核 12/13/14 的系数只依赖几何，但**导数**与**转矩值**
    要一个真实的 ``A``。用 ``A=0`` 会让所有二次型恒为 0、所有导数被病态伴随放大 ——
    测试会平凡通过。所以夹具先跑一次 Newton（用真实铁芯 + 真实电流）。
    """

    def __init__(self, n_ang=7, a_eq=False, r_dirichlet=0.060):
        self.xy0, element_S, self.element_rotor_gap, self.element_stator, \
            self.element_rotor = make_donut_mesh(n_ang=n_ang)
        self.a_eq = a_eq

        pairing0 = ref_bnd.pair_master_slave(self.xy0, parts=PARTS,
                                            coord_err_abs=COORD_ERR_ABS)
        self.layer = ref_mm.insert_moving_mesh_layer(
            self.xy0, np.array([e[1:4] for e in element_S]),
            tuple(interface_nodes_legacy(self.xy0)), self.element_rotor_gap, pairing0,
            element_stator=self.element_stator, element_rotor=self.element_rotor,
        )
        self.virtual = ref_bnd.slave_virtual_elements(self.layer.tri, self.layer.pairing)
        self.n_node, self.n_elem = self.layer.n_node, self.layer.tri.shape[0]
        self.xy = self.layer.xy

        self.ctx = TopologyContext.from_boundaries(
            self.layer.tri, self.virtual, self.layer.pairing,
            self.element_rotor_gap, a_eq=a_eq, n_node=self.n_node,
        )
        self.rot = ref_mm.rotating_boundary(
            self.layer, parts=PARTS, pos=0.31, a_eq=a_eq)
        self.rot_d = ref_mm.rotating_boundary_with_derivative(
            self.layer, parts=PARTS, pos=0.31, a_eq=a_eq)
        self.em = ref_em.electromagnetic_elements(self.layer.xy, self.layer.tri)

        # ---- 驱动行：Dirichlet + 主从 + 动网格插值行 ----
        self.dirichlet = np.array(sorted(
            i for i in range(self.n_node)
            if abs(np.hypot(*self.xy[i]) - r_dirichlet) < COORD_ERR_ABS
        ), dtype=int)
        self.row_replacements = {
            int(n): np.asarray(self.rot.matrix[n, :].todense()).ravel()
            for n in self.layer.rotor_nodes
        }
        self.driving = ref_bnd.driving_rows(
            self.n_node, self.dirichlet, self.layer.pairing.slave_to_master, a_eq,
            self.row_replacements,
        )

        # ---- 材料与密度：铁芯做实、其余留空 ----
        # 用 ``from_csv``（走 ``extend_bh_table`` 的对称化 + 尾段外推），与生产路径一致
        from motorfem.kernels.reference.bh_curve import BHCurve
        self.material = ref_nt.FieldMaterial(
            curve=BHCurve.from_csv(self._csv()),
            simp=SIMPProjection(SIMPParams()),
        )
        self.element_iron = set(self.element_rotor) | set(self.element_stator)
        self.element_topo = set(e for e in self.element_rotor if e < 12)
        all_e = np.arange(self.n_elem)
        is_iron = np.isin(all_e, np.array(sorted(self.element_iron), dtype=int))
        is_topo = np.isin(all_e, np.array(sorted(self.element_topo), dtype=int))
        # ⚠ ρ 取 0.5（= SIMP 阈值）而不是 1.0：ρ=1 处 tanh 已饱和、``SIMP_dfunc ≈ 0``，
        # 于是 ``dγ/dρ ≈ 0``、``dT/dρ`` 恒为 0，所有导数测试都会平凡通过。
        self.rho = np.where(is_iron, 0.5, 0.0)
        masks = ref_nt.MaterialMasks(
            is_vacuum=(~is_iron).astype(float),
            is_simp=is_topo.astype(float),
            is_non_simp=(is_iron & ~is_topo).astype(float),
            is_linear_iron=np.zeros(self.n_elem),
        )
        self.field = ref_nt.field_geometry(
            self.ctx, self.rot, n_node=self.n_node, k0elem=self.em.K0,
            bxcoef=self.em.Bxcoef, bycoef=self.em.Bycoef, aelem=self.em.area,
            masks=masks,
        )
        # 内核 8 的 18 项装包 —— 导数（内核 13/14）的 fast 路径与 22 元组要用它
        self.ops = ref_asm.element_operators(
            self.ctx, self.rot, n_node=self.n_node, k0elem=self.em.K0,
            bxcoef=self.em.Bxcoef, bycoef=self.em.Bycoef,
            element_iron=self.element_iron, element_topo_opti=self.element_topo,
            use_nonlinear_iron=True,
        )

        # ---- 源项：定子侧若干单元的电流 ----
        self.R = np.zeros(self.n_node)
        for e in sorted(self.element_stator):
            self.R[list(self.field.transforms[e].index)] += 5.0e7 * self.em.R0[e]

        self.A, self.aux, self.Jacobi, self.history = ref_nt.newton_solve(
            None, self.rho, self.field, self.material, self.R, self.driving,
            solve_abs_err=1e-3, max_iter=100, iter_relaxation=20,
            relaxation_ratio=0.5, output=False,
        )

    @staticmethod
    def _csv():
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for cand in (root / "磁障设计/pytorch/50CS470.csv",
                     root / "legacy/pytorch/50CS470.csv"):
            if cand.exists():
                return cand
        raise FileNotFoundError("找不到 50CS470.csv")

    # ---- 内核 12 的输入束 ----
    def arkkio_geometry(self, gap_elements=None):
        return ref_tq.ArkkioGeometry(
            tri=self.layer.tri, xy=self.xy, area=self.em.area,
            bxcoef=self.em.Bxcoef, bycoef=self.em.Bycoef,
            gap_elements=(self.element_rotor_gap if gap_elements is None
                          else gap_elements),
            rotor_nodes=self.layer.rotor_nodes,
            parts=PARTS, mu0=MU0, gap=GAP_M,
        )

    def legacy_globals(self):
        """legacy 片段要的模块级全局量替身（``element_S`` 是**插入后**的 5 列格式）。"""
        element_S = [[i, int(a), int(b), int(c), 0]
                     for i, (a, b, c) in enumerate(self.layer.tri)]
        node = np.column_stack([
            np.arange(self.n_node), self.xy[:, 0], self.xy[:, 1],
            np.zeros(self.n_node),
        ])
        element_S_slave = {
            i: [i, *[int(n) for n in self.virtual.virtual_tri[i]], 0]
            for i in self.virtual.virtual_tri
        }
        return SimpleNamespace(
            N_node=self.n_node, N_element=self.n_elem,
            element_S=element_S, element_S_slave=element_S_slave, node=node,
            rot_d=self.rot_d,
            Bxcoef=self.em.Bxcoef, Bycoef=self.em.Bycoef, Aelem=self.em.area,
            mu0=MU0, parts=PARTS,
            element_rotor_gap=self.element_rotor_gap,
            node_gap_rotor={int(n): int(n) for n in self.layer.rotor_nodes},
            node_rotating_boundary=self.rot_d.brackets,
            node_master=self.layer.pairing.master_to_slave,
            EM_MS_boundary_A_eq=self.a_eq,
        )


@pytest.fixture(scope="module")
def fx() -> TorqueFixture:
    return TorqueFixture()


# =============================================================================
# 给内核 13/14 的导数校验用的「按给定 μ 重解」入口
# =============================================================================

def ref_asm_as_legacy_tuple(fx):
    """内核 8 的「22 元组兼容装置」—— 让 legacy 的解包可以原样保留。"""
    return ref_asm.as_legacy_tuple(fx.ops, fx.driving)

def solve_with_mu(fx, mu, A0=None):
    """给定逐单元磁导率 ``μ``，重解一次。

    ``dgamma = 0`` ⇒ 残差对 ``A`` **线性**（``K(μ)`` 与 ``A`` 无关），所以 Newton 从任意
    起点都在两步内精确收敛。这让我们可以在**不动材料模型**的前提下扰动单元素 μ，
    从而用有限差分独立检验伴随导数。
    """
    if A0 is None:
        A0 = fx.A

    def step(A):
        B = ref_nt.flux_density(A, fx.field)
        F, J = ref_nt.field_system(
            A, fx.field, mu, np.zeros(fx.n_elem), B, fx.R, fx.driving)
        return F, None, J

    A, _, J, _ = ref_nt.newton_iterations(
        step, A0, n_node=fx.n_node, max_iter=10, solve_abs_err=1e-14,
        iter_relaxation=20, relaxation_ratio=0.5, output=False)
    return A


def legacy_transformation_mat_v2(g, i_elem):
    """照抄 ``MMA10:1776-1869``。"""
    element_rotor_gap = g.element_rotor_gap
    element_S = g.element_S
    element_S_slave = g.element_S_slave
    node_master = g.node_master
    EM_MS_boundary_A_eq = g.EM_MS_boundary_A_eq
    N_node = g.N_node
    K_rotating_boundary = g.rot_d.matrix
    node_rotating_boundary = g.node_rotating_boundary
    dKrb_p_dth = g.rot_d.d_matrix

    if i_elem in element_rotor_gap:
        if i_elem not in element_S_slave:
            node_i, node_j, node_m = element_S[i_elem][1:4]
            _ki = 1
            _kj = 1
            _km = 1
        else:
            node_i, node_j, node_m = element_S_slave[i_elem][1:4]
            _ki = 1 if node_i in node_master else (1 if EM_MS_boundary_A_eq else -1)
            _kj = 1 if node_j in node_master else (1 if EM_MS_boundary_A_eq else -1)
            _km = 1 if node_m in node_master else (1 if EM_MS_boundary_A_eq else -1)

        _T_temp = np.zeros((3, N_node))
        _dT_temp = np.zeros((3, N_node))
        _index = set()
        for s, node, sign in ((0, node_i, _ki), (1, node_j, _kj), (2, node_m, _km)):
            if node in node_rotating_boundary:
                _node_1, _node_2 = node_rotating_boundary[node]
                _T_temp[s, _node_1] += -K_rotating_boundary[node, _node_1] * sign
                _T_temp[s, _node_2] += -K_rotating_boundary[node, _node_2] * sign
                _dT_temp[s, _node_1] += -dKrb_p_dth[node, _node_1] * sign
                _dT_temp[s, _node_2] += -dKrb_p_dth[node, _node_2] * sign
                _index.add(_node_1)
                _index.add(_node_2)
            else:
                _T_temp[s, node] += sign
                _index.add(node)

        _index = list(_index)
        _T = _T_temp[:, _index]
        _dT = _dT_temp[:, _index]
    elif i_elem in element_S_slave:
        node_i, node_j, node_m = element_S_slave[i_elem][1:4]
        _ki = 1 if node_i in node_master else (1 if EM_MS_boundary_A_eq else -1)
        _kj = 1 if node_j in node_master else (1 if EM_MS_boundary_A_eq else -1)
        _km = 1 if node_m in node_master else (1 if EM_MS_boundary_A_eq else -1)
        _T = np.diag([_ki, _kj, _km])
        _dT = np.diag([0, 0, 0])
        _index = [node_i, node_j, node_m]
    else:
        node_i, node_j, node_m = element_S[i_elem][1:4]
        _T = np.eye(3)
        _dT = np.eye(3) * 0
        _index = [node_i, node_j, node_m]

    return _T, _index, _dT


def legacy_get_VD_coef(g):
    """照抄 ``MMA10:707-736``。"""
    N_node = g.N_node
    N_element = g.N_element
    Bxcoef, Bycoef, Aelem = g.Bxcoef, g.Bycoef, g.Aelem
    mu0, parts = g.mu0, g.parts

    _dKe_p_dth = sp_sparse.lil_matrix((N_node, N_node))

    for i_elem in range(N_element):
        if i_elem not in g.element_rotor_gap:
            continue

        _T, _index, _dT = legacy_transformation_mat_v2(g, i_elem)

        _Bxcoef = Bxcoef[i_elem] @ _T
        _Bycoef = Bycoef[i_elem] @ _T
        _dBxcoef = Bxcoef[i_elem] @ _dT
        _dBycoef = Bycoef[i_elem] @ _dT

        _dKe = 2 * (_Bxcoef.reshape((-1, 1)) @ _dBxcoef.reshape((1, -1))
                    + _Bycoef.reshape((-1, 1)) @ _dBycoef.reshape((1, -1))
                    ) / (2 * mu0) * Aelem[i_elem]

        for i, node_i in enumerate(_index):
            for j, node_j in enumerate(_index):
                _dKe_p_dth[node_i, node_j] += _dKe[i, j]

    _dKe_p_dth *= -parts
    return _dKe_p_dth


def legacy_DTorque_vectorized(g, _A_node, _Jacobi, elements_to_compute,
                              K_vectorized, Torque_coef):
    """照抄 ``MMA10:2260-2277``（= ``:2280-2297``，两处函数体逐字相同）。"""
    N_node = g.N_node
    N_element = g.N_element
    (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
     Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
     g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
     dK_p_dgamma_A_row, dK_p_dgamma_A_col,
     J_driven, J_mat_driven, J_col_mask) = K_vectorized

    g0i_vec = g0i_mat @ _A_node

    dK_p_dgamma_A_vectorized = sp_sparse.coo_matrix(
        (g0i_vec, (dK_p_dgamma_A_row, dK_p_dgamma_A_col)),
        shape=(N_element, N_node))

    _result = -2 * dK_p_dgamma_A_vectorized @ sp_sparse.linalg.spsolve(
        _Jacobi.T, Torque_coef @ _A_node)

    dT_p_dgamma_vectorized = np.zeros(N_element)
    dT_p_dgamma_vectorized[list(elements_to_compute)] = _result[list(elements_to_compute)]

    return dT_p_dgamma_vectorized


# =============================================================================
# 来源 A —— 内核 13：伴随导数 vs 有限差分
# =============================================================================

def torque_at(fx, rho_, coef):
    """在给定 ``ρ`` 上重解一次并返回 ``T = Aᵀ(M A)``（**全非线性**，与生产同路径）。"""
    A, _, _, _ = ref_nt.newton_solve(
        None, rho_, fx.field, fx.material, fx.R, fx.driving,
        solve_abs_err=1e-3, max_iter=100, iter_relaxation=20, relaxation_ratio=0.5,
        output=False,
    )
    return ref_tq.torque_quadratic(A, coef)


def test_arkkio_derivative_matches_finite_difference(fx):
    """扰动**设计变量 ρ**、全非线性重解、比转矩 —— 独立于实现的**外部**判据。

    验的是生产契约 ``dT/dρ = dT/dγ · dγ/dρ``：伴随给 ``dT/dγ``，
    ``dγ/dρ`` 走内核 3 的 ``nonlinear_material_dnu_drho``。

    ⚠ ``ρ`` 取 0.5 而不是 1.0：ρ=1 处 tanh 已饱和、``SIMP_dfunc ≈ 0``，
    ``dT/dρ`` 会恒为 0，测试平凡通过（本文件踩过这个坑）。

    实测相对差 **6.5e-06 ~ 3.8e-05**（9 轮非线性 Newton 下 FD 的分辨率上限）；
    阈值取 1e-3 留 25 倍余量。真实网格上的对照见 ``tools/probe_torque.py``
    （实测 1.000~1.003，同样量级）。
    """
    geom = fx.arkkio_geometry()
    coef = ref_tq.arkkio_coefficient(geom)
    dgdr = ref_tq.torque_derivative_wrt_rho(
        ref_tq.adjoint_torque_derivative(
            fx.A, fx.field, fx.Jacobi, coef, sorted(fx.element_topo)),
        np.asarray(fx.aux.B), fx.rho, fx.material.curve, fx.material.simp,
    )
    elems = sorted(fx.element_topo)
    cand = [e for e in elems if abs(dgdr[e]) > 1e-3 * np.abs(dgdr[elems]).max()]
    assert cand, "没有单元的 dT/dρ 非零 —— 测试无意义（检查 ρ 是否落在 SIMP 饱和区）"

    worst = 0.0
    for e in cand[:5]:
        dr = 1e-4
        rp, rm = fx.rho.copy(), fx.rho.copy()
        rp[e] += dr
        rm[e] -= dr
        fd = (torque_at(fx, rp, coef) - torque_at(fx, rm, coef)) / (2 * dr)
        adj = dgdr[e]
        scale = max(abs(fd), abs(adj))
        worst = max(worst, abs(fd - adj) / scale if scale else 0.0)

    assert worst < 1e-3, f"伴随导数与有限差分最大相对差 {worst:.3e}"


def test_driven_rows_are_absent_from_the_topo_region_here(fx):
    """★ 这条测试钉住 **D2**（设计文档 §12）**当前是潜伏的**这一前提。

    legacy 的伴随把 ``∂F/∂γ`` 用在**驱动行**上，而驱动行（Dirichlet / 主从 / 动网格
    插值）与 ``γ`` **无关** ⇒ 那里应为 0。若拓扑区单元的活跃自由度里含驱动节点，
    那条非零分量会被 ``J⁻¹`` 放大（驱动行的 ``J`` 元素是 O(1) 约束系数，而自由行是
    O(1e6)）—— 实测在「Dirichlet 紧邻拓扑区」的夹具上偏差达 **1e6~1e7 倍且符号相反**。

    **当前生产网格上不会发作**：实测 6362 个拓扑区单元里，活跃自由度含驱动节点的
    **0 个**（Dirichlet 在 r=72.5 mm 的定子外径，从属节点已被虚拟替换成主动节点）。
    这条测试把这个前提**变成可执行的检查** —— 哪天它失败，就说明 D2 从潜伏变成活跃。
    """
    drv = np.asarray(fx.driving.is_driven) > 0
    bad = [e for e in sorted(fx.element_topo)
           if any(drv[n] for n in fx.field.transforms[e].index)]
    assert not bad, (
        f"拓扑区单元 {bad[:5]} 的活跃自由度里出现了驱动节点 ⇒ D2 已从潜伏变为活跃，"
        f"伴随导数会偏 1e6 倍以上。见 docs/重构设计文档.md §12 的 D2。"
    )


# =============================================================================
# 来源 B —— 内核 13：与 legacy 逐位
# =============================================================================

@requires_fast
def test_fast_matches_legacy_adjoint_derivative_bitwise(fx):
    """``fast`` 与照抄的 ``DTorque_Arkkio_vectorized`` **逐位**一致。"""
    g = fx.legacy_globals()
    kv = ref_asm_as_legacy_tuple(fx)
    coef = ref_tq.arkkio_coefficient(fx.arkkio_geometry())
    elems = sorted(fx.element_topo)

    legacy = legacy_DTorque_vectorized(g, fx.A, fx.Jacobi, elems, kv, coef)
    fast = fast_tq.adjoint_torque_derivative(
        fx.A, kv, fx.Jacobi, coef, elems,
        n_node=fx.n_node, n_element=fx.n_elem,
    )

    assert np.array_equal(legacy, fast)


@requires_fast
def test_fast_and_reference_agree_on_the_adjoint(fx):
    """两层在同一个系数矩阵上给同一个答案（``rtol=1e-12``）。"""
    kv = ref_asm_as_legacy_tuple(fx)
    coef = ref_tq.arkkio_coefficient(fx.arkkio_geometry())
    elems = sorted(fx.element_topo)

    ref = ref_tq.adjoint_torque_derivative(fx.A, fx.field, fx.Jacobi, coef, elems)
    fast = fast_tq.adjoint_torque_derivative(
        fx.A, kv, fx.Jacobi, coef, elems,
        n_node=fx.n_node, n_element=fx.n_elem,
    )
    scale = max(np.abs(ref).max(), np.abs(fast).max())
    assert np.abs(ref - fast).max() <= 1e-12 * scale


@requires_fast
def test_factor_path_matches_spsolve_bitwise(fx):
    """阶段 8 分解复用：``factor=splu(J.T)`` 与默认 ``spsolve(J.T,·)`` **逐位**一致。

    ⚠ 前提：``J.T`` 是 **CSC** —— splu 复用只对 CSC 矩阵成立（spsolve 对 CSR 走
    SuperLU 的 CSR 路径，splu 强制转 CSC，两者不逐位）。EM 伴随解恰好是 ``J.T``。
    """
    from scipy.sparse import linalg as sp_linalg

    kv = ref_asm_as_legacy_tuple(fx)
    coef = ref_tq.arkkio_coefficient(fx.arkkio_geometry())
    elems = sorted(fx.element_topo)

    # 默认路径（spsolve）与 factor 路径（splu 因子 + 逐右端回代）
    fast_default = fast_tq.adjoint_torque_derivative(
        fx.A, kv, fx.Jacobi, coef, elems,
        n_node=fx.n_node, n_element=fx.n_elem,
    )
    factor = sp_linalg.splu(fx.Jacobi.T)
    fast_reuse = fast_tq.adjoint_torque_derivative(
        fx.A, kv, fx.Jacobi, coef, elems,
        n_node=fx.n_node, n_element=fx.n_elem, factor=factor,
    )

    assert np.array_equal(fast_default, fast_reuse)


def test_derivative_is_zeroed_outside_the_topo_region(fx):
    """``elements_to_compute`` 之外必须**精确为 0**（设计变量只在拓扑区）。"""
    coef = ref_tq.arkkio_coefficient(fx.arkkio_geometry())
    elems = sorted(fx.element_topo)
    d = ref_tq.adjoint_torque_derivative(fx.A, fx.field, fx.Jacobi, coef, elems)

    outside = [e for e in range(fx.n_elem) if e not in set(elems)]
    assert np.all(d[outside] == 0.0)
    assert np.any(d[list(elems)] != 0.0), "拓扑区里全是 0 ⇒ 测试无意义"


def test_chain_rule_helper_reuses_the_material_derivative(fx):
    """``dT/dρ`` 的公共层与 legacy 的三处逐字写法给出同一结果。"""
    from motorfem.kernels.reference.material import (
        nonlinear_material_dnu_drho as legacy_chain,
    )
    B = np.asarray(fx.aux.B)
    dT_dgamma = np.arange(fx.n_elem, dtype=float) + 0.5

    got = ref_tq.torque_derivative_wrt_rho(
        dT_dgamma, B, fx.rho, fx.material.curve, fx.material.simp)
    want = dT_dgamma * legacy_chain(B, fx.rho, fx.material.curve, fx.material.simp)
    assert np.array_equal(got, want)


# =============================================================================
# 照抄 legacy（不复用被测实现）
# =============================================================================

def legacy_get_Arkkio_coef(g):
    """照抄 ``MMA10:666-701``。``g`` 提供 ``N_node``/``element_S``/``node`` 等全局量替身。"""
    N_node = g.N_node
    element_S = g.element_S
    node = g.node
    Bxcoef, Bycoef, Aelem = g.Bxcoef, g.Bycoef, g.Aelem
    mu0, parts = g.mu0, g.parts

    _Torque_Arkkio_coef = sp_sparse.lil_matrix((N_node, N_node))

    for i_elem in g.element_rotor_gap:
        _node_gap = g.node_gap_rotor
        node_i, node_j, node_m = element_S[i_elem][1:-1]
        if not (node_i in _node_gap or node_j in _node_gap or node_m in _node_gap):
            continue

        _th_i = np.arctan2(node[node_i, 2], node[node_i, 1])
        _th_j = np.arctan2(node[node_j, 2], node[node_j, 1])
        _th_m = np.arctan2(node[node_m, 2], node[node_m, 1])
        _th_0 = (_th_i + _th_j + _th_m) / 3

        _r0 = np.sqrt(
            (node[node_i, 1] + node[node_j, 1] + node[node_m, 1]) ** 2 / 9
            + (node[node_i, 2] + node[node_j, 2] + node[node_m, 2]) ** 2 / 9
        )

        _K_Arkkio = _r0 * Aelem[i_elem] * (Bxcoef[i_elem] * np.cos(_th_0) - Bycoef[i_elem] * np.sin(_th_0)).reshape(
            (-1, 1)) @ (Bxcoef[i_elem] * np.sin(_th_0) + Bycoef[i_elem] * np.cos(_th_0)).reshape((1, -1))

        for _ki, node_index1 in enumerate([node_i, node_j, node_m]):
            for _kj, node_index2 in enumerate([node_i, node_j, node_m]):
                _Torque_Arkkio_coef[node_index1, node_index2] += _K_Arkkio[_ki, _kj]

    _Torque_Arkkio_coef *= 1 / mu0 * parts / (0.25e-3 / 4)

    _Torque_Arkkio_coef = (_Torque_Arkkio_coef + _Torque_Arkkio_coef.T) / 2

    return _Torque_Arkkio_coef


# =============================================================================
# 来源 A —— 内核 12
# =============================================================================

def test_single_element_closed_form(fx):
    """单单元闭式：``K^(e) = r̄·Δ·scale·k_Br ⊗ k_Bθ``，对称化后即系数矩阵。

    只把**一个**单元放进 ``gap_elements``，于是系数矩阵就是那一个单元的贡献
    （它含动节点、不会被跳过）。手算 ``r̄``/``θ̄``/``k_Br``/``k_Bθ`` 后逐位对照
    公式，正面验证内核 12 的全部数学。
    """
    gap = sorted(fx.element_rotor_gap)
    rot_nodes = set(int(n) for n in fx.layer.rotor_nodes)
    e = next(i for i in gap if any(int(n) in rot_nodes for n in fx.layer.tri[i]))

    geom = fx.arkkio_geometry(gap_elements={e})
    M = ref_tq.arkkio_coefficient(geom).toarray()

    nodes = [int(n) for n in fx.layer.tri[e]]
    x, y = fx.xy[nodes, 0], fx.xy[nodes, 1]
    r_bar = np.sqrt((x.sum() ** 2) / 9 + (y.sum() ** 2) / 9)
    th_bar = (np.arctan2(y[0], x[0]) + np.arctan2(y[1], x[1])
              + np.arctan2(y[2], x[2])) / 3
    k_br = fx.em.Bxcoef[e] * np.cos(th_bar) - fx.em.Bycoef[e] * np.sin(th_bar)
    k_bt = fx.em.Bxcoef[e] * np.sin(th_bar) + fx.em.Bycoef[e] * np.cos(th_bar)
    scale = 1 / MU0 * PARTS / (GAP_M / PARTS)
    block = (r_bar * fx.em.area[e] * scale) * np.outer(k_br, k_bt)
    block = (block + block.T) / 2

    expected = np.zeros_like(M)
    expected[np.ix_(nodes, nodes)] = block
    assert np.allclose(M, expected, rtol=1e-13, atol=0)


def test_coefficient_is_symmetric_bitwise(fx):
    """``(M + Mᵀ)/2`` 在 IEEE 下**精确对称**（加法可交换 ⇒ ``M_ij+M_ji`` 两侧相同）。

    所以这条用 ``np.array_equal``，不是 ``allclose`` —— 它同时守住
    「对称化没有被漏掉」与「没有引入不对称的浮点运算」。
    """
    M = ref_tq.arkkio_coefficient(fx.arkkio_geometry())
    assert np.array_equal(M.toarray(), M.toarray().T)


def test_quadratic_form_sees_only_the_symmetric_part(fx):
    """二次型 ``AᵀMA`` 只看见 ``M`` 的对称部分 —— 这是内核 14 那个疑点的前提。"""
    M = ref_tq.arkkio_coefficient(fx.arkkio_geometry())
    rng = np.random.default_rng(3)
    A = rng.normal(size=fx.n_node)

    # Arkkio 的 M 本来就对称，所以先人为造一个不对称的矩阵来验「只看对称部分」这条性质
    antisym = sp_sparse.random(fx.n_node, fx.n_node, density=0.2, random_state=5)
    antisym = antisym - antisym.T
    M_asym = (M + antisym).tocsr()

    assert ref_tq.torque_quadratic(A, M_asym) == pytest.approx(
        ref_tq.torque_quadratic(A, M), rel=1e-14)


def test_only_gap_elements_with_rotor_nodes_contribute(fx):
    """只有「气隙单元 ∩ 局部含动节点」参与：两类排除各造一个空集验一次。"""
    full = ref_tq.arkkio_coefficient(fx.arkkio_geometry())

    # 把 gap_elements 换成一个「与气隙不交」的集合 ⇒ 必须全零
    empty = ref_tq.arkkio_coefficient(
        fx.arkkio_geometry(gap_elements=set()))
    assert empty.nnz == 0

    # 只保留**不含**动节点的气隙单元 ⇒ 也必须全零
    rot_nodes = set(int(n) for n in fx.layer.rotor_nodes)
    no_rotor = {
        e for e in fx.element_rotor_gap
        if not any(int(n) in rot_nodes for n in fx.layer.tri[e])
    }
    assert ref_tq.arkkio_coefficient(
        fx.arkkio_geometry(gap_elements=no_rotor)).nnz == 0

    # 剩下那些单元合起来必须等于全量
    used = fx.element_rotor_gap - no_rotor
    part = ref_tq.arkkio_coefficient(fx.arkkio_geometry(gap_elements=used))
    assert np.allclose(part.toarray(), full.toarray(), rtol=1e-12, atol=0)


def test_torque_value_is_a_quadratic_form(fx):
    """转矩值 = ``Aᵀ(M A)``，且与逐单元求和一致（防「装配漏项」）。"""
    geom = fx.arkkio_geometry()
    M = ref_tq.arkkio_coefficient(geom)
    T = ref_tq.arkkio_torque(fx.A, geom)
    assert T == pytest.approx(float(fx.A @ (M @ fx.A)), rel=1e-15)
    assert np.isfinite(T) and T != 0.0


# =============================================================================
# 来源 B —— 与 legacy 逐位
# =============================================================================

def test_fast_matches_legacy_arkkio_bitwise(fx):
    """``fast`` 与 legacy 照抄片段**逐位**一致（含 ``set`` 迭代顺序与 ``lil`` 累加）。"""
    g = fx.legacy_globals()
    legacy = legacy_get_Arkkio_coef(g)

    fast = fast_tq.arkkio_coefficient(fx.arkkio_geometry())

    assert fast.format == "csr"
    assert np.array_equal(legacy.data, fast.data)
    assert np.array_equal(legacy.indices, fast.indices)
    assert np.array_equal(legacy.indptr, fast.indptr)


def test_frozenset_is_rejected(fx):
    """``gap_elements`` 传 ``frozenset`` 必须显式报错。

    实测 ``frozenset`` 与 ``set`` 的迭代顺序不保证一致（500 组随机整数集合里 176 组
    不同），而这里的累加顺序参与末位 —— 静默接受它等于埋一颗 F4 地雷。
    """
    geom = ref_tq.ArkkioGeometry(
        tri=fx.layer.tri, xy=fx.xy, area=fx.em.area,
        bxcoef=fx.em.Bxcoef, bycoef=fx.em.Bycoef,
        gap_elements=frozenset(fx.element_rotor_gap),
        rotor_nodes=fx.layer.rotor_nodes, parts=PARTS, mu0=MU0, gap=GAP_M,
    )
    with pytest.raises(TypeError, match="frozenset"):
        fast_tq.arkkio_coefficient(geom)


# =============================================================================
# 内核 14 —— 虚位移法
# =============================================================================

def test_v2_transform_reuses_the_kernel8_matrix_bitwise(fx):
    """``transformation_matrix_v2`` 的 ``T`` 与内核 8 的 ``transformation_matrix``
    **逐位**相同 —— 两者是同一份数学，v2 只多一个 ``dT``。

    这条同时守住「v2 是照内核 8 重写的」这件事：若哪天有人在 v2 里改了列序或符号，
    这里会立刻失败，而不是等到 ``Bxcoef@T`` 差末位。
    """
    for e in (0, 1, 5, 8, 12):
        tr8 = ref_asm.transformation_matrix(e, fx.ctx, fx.rot)
        tr2 = ref_tq.transformation_matrix_v2(e, fx.ctx, fx.rot_d)
        assert tr2.index == tr8.index, f"单元 {e} 的列序不同"
        assert np.array_equal(tr2.matrix, tr8.matrix), f"单元 {e} 的 T 不逐位相同"


def test_v2_derivative_is_zero_off_the_rotor_gap_branch(fx):
    """``dT`` 只在转子气隙单元上非零（其余分支的 ``T`` 与转子位置无关）。"""
    zero_elems = [
        e for e in range(fx.n_elem)
        if e not in fx.element_rotor_gap
    ]
    for e in zero_elems[:10]:
        tr = ref_tq.transformation_matrix_v2(e, fx.ctx, fx.rot_d)
        assert np.all(tr.d_matrix == 0.0)

    gap_with_dT = [
        e for e in sorted(fx.element_rotor_gap)
        if np.abs(ref_tq.transformation_matrix_v2(e, fx.ctx, fx.rot_d).d_matrix).max() > 0
    ]
    assert gap_with_dT, "没有任何气隙单元的 dT 非零 ⇒ 测试无意义"


@requires_fast
def test_fast_matches_legacy_v2_transform_bitwise(fx):
    """``fast`` 的 ``T``/``dT`` 与照抄片段**逐位**一致（含 F 连续的布局）。"""
    g = fx.legacy_globals()
    for e in (0, 1, 5, 8, 12):
        T_l, idx_l, dT_l = legacy_transformation_mat_v2(g, e)
        tr = fast_tq.transformation_matrix_v2(e, fx.ctx, fx.rot_d, n_node=fx.n_node)
        assert list(tr.index) == list(idx_l)
        assert np.array_equal(tr.matrix, T_l)
        assert np.array_equal(tr.d_matrix, dT_l)


@requires_fast
def test_fast_matches_legacy_vd_coefficient_bitwise(fx):
    """``fast`` 的虚位移系数与照抄 ``get_VD_coef`` 的片段**逐位**一致。

    ⚠ **D3 已修（2026-09-23）**：legacy 返回**不对称**的 ``M``，本层返回对称化后的
    ``M_sym = (M+Mᵀ)/2`` —— 所以这里比的是 ``legacy 对称化版``（``(M+Mᵀ)/2`` 再乘
    ``−parts``）。"fast == legacy 对称化版" 是 D3 修正的落地形态。
    """
    g = fx.legacy_globals()
    legacy = legacy_get_VD_coef(g).tocsr()      # legacy 返回 lil（:735 的 *= 不换类型）
    fast = fast_tq.vd_coefficient(
        fx.ctx, fx.rot_d, n_node=fx.n_node, n_element=fx.n_elem,
        bxcoef=fx.em.Bxcoef, bycoef=fx.em.Bycoef, area=fx.em.area,
        parts=PARTS, mu0=MU0,
    )
    # legacy_get_VD_coef 已含 ``*= −parts``，对称化 ``(M+Mᵀ)/2`` 与缩放可交换
    legacy_sym = ((legacy + legacy.T) / 2)
    assert fast.format == "csr"
    assert np.array_equal(legacy_sym.data, fast.data)
    assert np.array_equal(legacy_sym.indices, fast.indices)
    assert np.array_equal(legacy_sym.indptr, fast.indptr)


def test_vd_coefficient_is_symmetric_and_the_value_unchanged(fx):
    """★ ``M_VD`` 现为**对称**（D3 修正，2026-09-23），且值相对 legacy 不变。

    修前的形态是 ``M`` 相对不对称度 ~1.08（两个**不同**秩 1 项之和，一般不对称）、
    二次型只看对称部分 ⇒ 值不受影响、但伴随导数错（D3）。修后对称化，
    这条把三件事一起钉住：
    (a) ``M`` 确实**对称**（asym ≈ 0）；
    (b) ``AᵀMA`` 的值与 legacy（不对称 ``M``）的二次型值**逐位不变** ——
        这是「D3 = 纯梯度修正、无需重标定」的依据；
    (c) 防退化：``M`` 的对称部分确实非零（不是全零矩阵）。
    """
    M = fast_tq.vd_coefficient(
        fx.ctx, fx.rot_d, n_node=fx.n_node, n_element=fx.n_elem,
        bxcoef=fx.em.Bxcoef, bycoef=fx.em.Bycoef, area=fx.em.area,
        parts=PARTS, mu0=MU0,
    )
    dense = M.toarray()
    asym = np.abs(dense - dense.T).max()
    assert asym <= 1e-12 * np.abs(dense).max(), (
        f"M 竟然不对称（asym {asym:.3e}）—— D3 的对称化没落地？")

    # (b) 值不变：legacy 的不对称 M 与对称化版给出相同的二次型值（数学恒等，
    #    浮点 ≤1 ULP，见另一个测试的同款注释）
    g = fx.legacy_globals()
    legacy = legacy_get_VD_coef(g).tocsr()
    rng = np.random.default_rng(17)
    v = rng.normal(size=fx.n_node)
    assert ref_tq.torque_quadratic(v, M) == pytest.approx(
        ref_tq.torque_quadratic(v, legacy), rel=1e-15)
    assert np.abs(v @ (M @ v)) > 0.0
    # (c) 防退化：M 的对称部分非零
    assert np.abs(dense).max() > 0.0


@requires_fast
def test_fast_and_reference_agree_on_vd(fx):
    """两层在虚位移系数上一致（``rtol=1e-12``；容器不同：字典 vs ``lil``）。"""
    ref = ref_tq.vd_coefficient(
        fx.ctx, fx.rot_d, n_node=fx.n_node,
        bxcoef=fx.em.Bxcoef, bycoef=fx.em.Bycoef, area=fx.em.area,
        parts=PARTS, mu0=MU0,
    )
    fast = fast_tq.vd_coefficient(
        fx.ctx, fx.rot_d, n_node=fx.n_node, n_element=fx.n_elem,
        bxcoef=fx.em.Bxcoef, bycoef=fx.em.Bycoef, area=fx.em.area,
        parts=PARTS, mu0=MU0,
    )
    assert np.array_equal(ref.indptr, fast.indptr)
    assert np.array_equal(ref.indices, fast.indices)
    scale = np.abs(ref.data).max()
    assert np.abs(ref.data - fast.data).max() <= 1e-12 * scale


def test_vd_adjoint_uses_the_symmetrisation_that_arkkio_has(fx):
    """★ 这条钉住**D3 修正**（设计文档 §12，2026-09-23）：虚位移的伴随导数用对称的 ``M``。

    修前的缺陷：``_dKe = 2(Bx⊗dBx + By⊗dBy)/(2μ0)·Δ``（两个**不同**秩 1 项之和，
    相对不对称度实测 1.08）未对称化 ⇒ 伴随导数 ``dT/dγ_e = 2(∂A)ᵀ M_sym A``
    **必须用 ``M_sym``** 而 legacy 用了 ``M`` ⇒ 生产 ``dTVD_p_drho`` 偏（真实网格
    中位 31%、10.3% 符号翻转）。修后 ``vd_coefficient`` 返回对称矩阵。

    本条断言三件事：
    (a) **决定性**：生产伴随导数 == 有限差分（扰动 ρ、全非线性重解，FD 分辨率内）；
    (b) 值逐位不变：``T_VD`` 用对称化 M 与 legacy 不对称 M 的二次型值**逐位相同**
        （这是「纯梯度修正、无需重标定」的依据）；
    (c) 防退化：导数非零（ρ 在 SIMP 活跃区）。

    修前形态（夹具 1.73 倍、真实网格 min −253.8/中位 0.813/max +3104.8、
    10.3% 符号翻转）已写入 §12，不再核对。
    """
    M = fast_tq.vd_coefficient(
        fx.ctx, fx.rot_d, n_node=fx.n_node, n_element=fx.n_elem,
        bxcoef=fx.em.Bxcoef, bycoef=fx.em.Bycoef, area=fx.em.area,
        parts=PARTS, mu0=MU0,
    )
    elems = sorted(fx.element_topo)
    worst = 0.0
    for e in elems[:5]:
        dr = 1e-4
        rp, rm = fx.rho.copy(), fx.rho.copy()
        rp[e] += dr
        rm[e] -= dr
        fd = (torque_at(fx, rp, M) - torque_at(fx, rm, M)) / (2 * dr)

        adj = ref_tq.torque_derivative_wrt_rho(
            ref_tq.adjoint_torque_derivative(
                fx.A, fx.field, fx.Jacobi, M, elems),
            np.asarray(fx.aux.B), fx.rho, fx.material.curve, fx.material.simp,
        )[e]
        assert adj != 0.0, "导数恒为 0 ⇒ 测试无意义（检查 ρ 是否在 SIMP 活跃区）"
        worst = max(worst, abs(fd - adj) / abs(fd))

    # (a) 生产（对称 M）== FD
    assert worst < 1e-3, f"D3 修后生产导数仍与 FD 差 {worst:.3e}"

    # (b) 值不变（legacy 不对称 M 的二次型 == 对称化 M 的二次型；数学上恒等，
    #    浮点上差 ≤1 ULP —— 不对称部分的贡献在中间向量里舍入后不精确相消）
    g = fx.legacy_globals()
    legacy = legacy_get_VD_coef(g).tocsr()
    rng = np.random.default_rng(17)
    v = rng.normal(size=fx.n_node)
    assert ref_tq.torque_quadratic(v, M) == pytest.approx(
        ref_tq.torque_quadratic(v, legacy), rel=1e-15)


def test_arkkio_and_vd_are_the_only_difference_here(fx):
    """对照：**同一套**伴随脚手架在 Arkkio 上通过（它的 ``M`` 是对称的）。

    这条与上一条成对出现，把「差别来自 ``M`` 的不对称性」这一点钉死 ——
    否则无法排除「伴随实现本身有问题」。
    """
    M = ref_tq.arkkio_coefficient(fx.arkkio_geometry())
    assert np.array_equal(M.toarray(), M.toarray().T), "Arkkio 的 M 竟然不对称"
    elems = sorted(fx.element_topo)
    dgdr = ref_tq.torque_derivative_wrt_rho(
        ref_tq.adjoint_torque_derivative(fx.A, fx.field, fx.Jacobi, M, elems),
        np.asarray(fx.aux.B), fx.rho, fx.material.curve, fx.material.simp,
    )
    worst = 0.0
    for e in elems[:5]:
        dr = 1e-4
        rp, rm = fx.rho.copy(), fx.rho.copy()
        rp[e] += dr
        rm[e] -= dr
        fd = (torque_at(fx, rp, M) - torque_at(fx, rm, M)) / (2 * dr)
        scale = max(abs(fd), abs(dgdr[e]))
        worst = max(worst, abs(fd - dgdr[e]) / scale if scale else 0.0)
    assert worst < 1e-3, f"Arkkio 的伴随导数与 FD 差 {worst:.3e} —— 那就不只是 M 的问题"


# =============================================================================
# 来源 C —— reference vs fast
# =============================================================================

@requires_fast
def test_ref_fast_parity_arkkio(fx):
    """``rtol=1e-12``：两层的差别只来自**单元遍历顺序**（``range`` vs ``set``）。

    这不是「实现差异」而是「累加顺序差异」—— F4 的原始形态。故稀疏模式相同、
    数值在末位内一致。
    """
    M_ref = ref_tq.arkkio_coefficient(fx.arkkio_geometry())
    M_fast = fast_tq.arkkio_coefficient(fx.arkkio_geometry())

    assert np.array_equal(M_ref.indptr, M_fast.indptr)
    assert np.array_equal(M_ref.indices, M_fast.indices)
    scale = np.abs(M_ref.data).max()
    assert np.abs(M_ref.data - M_fast.data).max() <= 1e-12 * scale
