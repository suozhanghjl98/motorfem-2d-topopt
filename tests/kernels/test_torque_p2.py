"""P2 虚位移转矩测试（路线 A：`torque.vd_coefficient_p2`）。

来源 A 解析性质：

- 系数对称（D3 口径）：
  `T = AᵀMA` 只看见对称部分，`(M + Mᵀ)/2` 是伴随导数需要的形态；
- **虚功与共能导数一致**（最强判据）：`T_VD_p2 = AᵀMA` 与
  `dWc/dθ = (Wc(pos+ε) − Wc(pos−ε))/(2ε)` 一致 —— 独立于实现的外部判据，
  Wc 由逐高斯点共能（`B·H − Wm`，真空 `B²/2μ0`）直接计算，不复用 M。

注意（与 P1 同源）：`K(pos)` 在括号节点交换/奇扇区翻转处是分段光滑的，
FD 必须取在同一段内 —— 本测试在 pos=0 求值（规则网格上非切换点），ε 远小于
接口角间距。

运行::

    python -m pytest tests/kernels/test_torque_p2.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from motorfem.kernels.reference import assembly as ref_asm
from motorfem.kernels.reference import assembly_p2 as ref_asm_p2
from motorfem.kernels.reference import boundaries as ref_bnd
from motorfem.kernels.reference.element_matrices_p2 import electromagnetic_elements_p2
from motorfem.kernels.reference import moving_mesh as ref_mm
from motorfem.kernels.reference import newton as ref_nt
from motorfem.kernels.reference import newton_p2 as ref_nt_p2
from motorfem.kernels.reference import torque as ref_tq
from motorfem.kernels.reference.bh_curve import BHCurve
from motorfem.kernels.reference.simp import SIMPParams, SIMPProjection

SECTOR = np.pi / 2
R_RINGS = (0.030, 0.040, 0.050, 0.060)
MU0 = 4e-7 * np.pi


def _ring_p2(n_ang=7):
    """6 节点环扇网格（同 test_moving_mesh.make_interface_mesh_p2，内联避免跨测试导入）。"""
    angles = np.linspace(0.0, SECTOR, n_ang)
    n_ring = len(R_RINGS)
    xy = [[r * np.cos(t), r * np.sin(t)] for r in R_RINGS for t in angles]
    tri3, regs = [], []
    for k in range(n_ring - 1):
        for l in range(n_ang - 1):
            a = k * n_ang + l
            b = (k + 1) * n_ang + l
            c = (k + 1) * n_ang + l + 1
            d = k * n_ang + l + 1
            tri3.append((a, b, c)); regs.append(k)
            tri3.append((a, c, d)); regs.append(k)
    n_vertex = len(xy)
    edge_to_mid = {}

    def _mid_of(va, vb):
        key = (min(int(va), int(vb)), max(int(va), int(vb)))
        if key not in edge_to_mid:
            m = n_vertex + len(edge_to_mid)
            edge_to_mid[key] = m
            xa, ya = xy[va]; xb, yb = xy[vb]
            xy.append(((xa + xb) / 2.0, (ya + yb) / 2.0))
        return edge_to_mid[key]

    element_S = []
    for (a, b, c), reg in zip(tri3, regs):
        element_S.append([len(element_S), a, b, c,
                          _mid_of(a, b), _mid_of(b, c), _mid_of(c, a), reg])
    node = np.column_stack(
        [np.arange(len(xy), dtype=float), np.asarray(xy), np.zeros(len(xy))])
    e_gap = set(_e[0] for _e in element_S if _e[7] == 1)
    e_stator = set(_e[0] for _e in element_S if _e[7] == 2)
    e_rotor = set(_e[0] for _e in element_S if _e[7] in (0, 1))
    tri6 = np.array([_e[1:7] for _e in element_S], dtype=int)
    return node, tri6, e_gap, e_stator, e_rotor


def _build(pos, n_ang=7):
    """合成 P2 环网格在转子位置 pos 的完整求解输入（真空材料 + 定子侧电流）。"""
    node, tri6, e_gap, e_stator, e_rotor = _ring_p2(n_ang)
    xy = node[:, 1:3]
    r_iface = R_RINGS[2]

    pairing0 = ref_bnd.pair_master_slave(xy, parts=4)
    ngs = set(int(n[0]) for n in node
              if abs((n[1] ** 2 + n[2] ** 2) - r_iface ** 2) < 1e-6)
    layer = ref_mm.insert_moving_mesh_layer(
        xy, tri6, tuple(ngs), e_gap, pairing0,
        element_stator=e_stator, element_rotor=e_rotor)
    virtual = ref_bnd.slave_virtual_elements(layer.tri, layer.pairing)
    ctx = ref_asm.TopologyContext.from_boundaries(
        layer.tri, virtual, layer.pairing, e_gap, a_eq=False, n_node=layer.n_node)
    em = electromagnetic_elements_p2(layer.xy, layer.tri)
    rot = ref_mm.rotating_boundary(layer, parts=4, pos=pos, a_eq=False)
    rot_d = ref_mm.rotating_boundary_with_derivative(
        layer, parts=4, pos=pos, a_eq=False)
    ops = ref_asm_p2.element_operators_p2(
        ctx, rot, n_node=layer.n_node, k0elem=em.K0, bxcoef=em.Bxcoef,
        bycoef=em.Bycoef, element_iron=(), element_topo_opti=(), use_nonlinear_iron=True)
    # 定子侧（region 2）电流：A 相正半带
    r_outer = R_RINGS[3]
    dirichlet = np.array(sorted(
        int(n[0]) for n in node
        if abs((n[1] ** 2 + n[2] ** 2) - r_outer ** 2) < 1e-6), dtype=int)
    row_rep = {int(n): np.asarray(rot.matrix[n, :].todense()).ravel()
               for n in layer.rotor_nodes}
    driving = ref_bnd.driving_rows(
        layer.n_node, dirichlet, layer.pairing.slave_to_master, False, row_rep)
    masks = ref_nt.MaterialMasks.from_operators(ops)
    field = ref_nt_p2.field_geometry_p2(
        ctx, rot, n_node=layer.n_node, k0elem=em.K0, bxcoef=em.Bxcoef,
        bycoef=em.Bycoef, bxcoef_g=em.Bxcoef_g, bycoef_g=em.Bycoef_g,
        gauss_w=em.gauss_w, aelem=em.area, masks=masks)
    rho = np.zeros(len(tri6))
    J = 1e6
    R = ref_asm_p2.get_R_vector_p2(
        ctx, rot, n_node=layer.n_node, r0elem=em.R0,
        current_of_element={e: J for e in e_stator})
    material = ref_nt.FieldMaterial(
        curve=BHCurve(np.array([0.0, 1e3, 1e6]), 4e-7 * np.pi * np.array([0.0, 1e3, 1e6])),
        simp=SIMPProjection(SIMPParams()))
    return dict(layer=layer, ctx=ctx, em=em, rot=rot, rot_d=rot_d, ops=ops,
                field=field, masks=masks, rho=rho, R=R, driving=driving)


def _full_K(env):
    """全线性刚度（真空）：K = Σ_e (1/μ0)·TᵀK0_e·T，散到 (n_node, n_node)。"""
    from scipy.sparse import coo_matrix
    field = env["field"]
    rows, cols, vals = [], [], []
    for e in range(field.n_elem):
        tr = field.transforms[e]
        index = list(tr.index)
        k0a = tr.matrix.T @ (field.k0[e] @ tr.matrix)
        for a in range(len(index)):
            for b in range(len(index)):
                rows.append(index[a])
                cols.append(index[b])
                vals.append((1 / MU0) * k0a[a, b])
    return coo_matrix((vals, (rows, cols)),
                      shape=(field.n_node, field.n_node)).tocsr()


def _solve(env):
    material = ref_nt.FieldMaterial(
        curve=BHCurve(np.array([0.0, 1e3, 1e6]), 4e-7 * np.pi * np.array([0.0, 1e3, 1e6])),
        simp=SIMPProjection(SIMPParams()))
    A, aux, J, hist = ref_nt_p2.newton_solve_p2(
        None, env["rho"], env["field"], material, env["R"], env["driving"],
        solve_abs_err=1e-6, max_iter=20, iter_relaxation=20,
        relaxation_ratio=0.5, output=False)
    return A, aux


def test_vd_p2_coefficient_is_symmetric():
    """`(M + Mᵀ)/2` 后系数对称（D3 口径，伴随导数需要）。"""
    env = _build(pos=0.0)
    M = ref_tq.vd_coefficient_p2(
        env["ctx"], env["rot_d"], n_node=env["layer"].n_node,
        bxcoef_g=env["em"].Bxcoef_g, bycoef_g=env["em"].Bycoef_g,
        gauss_w=env["em"].gauss_w, area=env["em"].area, parts=4)
    Md = M.toarray()
    assert Md.shape[0] == env["layer"].n_node
    assert np.allclose(Md, Md.T, atol=1e-15)
    assert M.nnz > 0, "气隙 band 应有非零贡献"


def test_vd_p2_matches_stiffness_fd():
    """最强判据：`vd_coefficient_p2` = `∂K/∂θ`（有限差分，独立于实现）。

    直接对照定义：`AᵀMA = −parts · ½Aᵀ(K(θ+ε)−K(θ−ε))/(2ε)A` —— K 由全装配
    （Σ(1/μ0)TᵀK0T）重算，不复用 M。选这个而不是共能 FD：共能 FD 有
    −parts/扇区因子纠缠，且 pos=0（节点角）的尖点会污染 FD（实测真实电机
    T_VD/fd_U = 2.11 —— 正是「小夹具倍数不能当通用常数」的教训）。

    ⚠ pos 必须取**非节点角**：P2 完整节点角网格是每 7.5°（顶点 15° + 边中点嵌中），
    pos 不能是 7.5° 的倍数（否则有动节点 φ 恰在定子节点角上，数值 ΔT 出现 ~5e4
    跳变）。取 3.75°，φ 恒在定子节点之间。
    """
    eps = 1e-5
    pos0 = np.pi / 48
    env0 = _build(pos=pos0)
    A0, _ = _solve(env0)
    M = ref_tq.vd_coefficient_p2(
        env0["ctx"], env0["rot_d"], n_node=env0["layer"].n_node,
        bxcoef_g=env0["em"].Bxcoef_g, bycoef_g=env0["em"].Bycoef_g,
        gauss_w=env0["em"].gauss_w, area=env0["em"].area, parts=4)

    env_p = _build(pos=pos0 + eps)
    env_m = _build(pos=pos0 - eps)
    dK = (_full_K(env_p) - _full_K(env_m)) / (2 * eps)
    kfd = 0.5 * A0 @ (dK @ A0)

    got = A0 @ (M @ A0)
    expect = -4 * kfd
    rel = abs(got - expect) / max(abs(expect), 1e-12)
    assert rel < 1e-6, f"T_VD={got:.10e}  −parts·K-FD={expect:.10e}  rel={rel:.2e}"
