"""阶段 4A：golden L1 的组装链（``FieldTopology``）。

把阶段 1 的数值内核 + 阶段 2 的网格契约组装成 legacy 模块体的「预组装段」
（``legacy/pytorch/nonlinear_solver_energy_MMA10.py``，下称 MMA10，行号以此为
准）：网格与区域（``:177-340``）、力学子网格（``:346-436``）、动网格插层
（``:442-491``）、单元矩阵（``:500-545``）、Arkkio 系数（``:700``）、初始密度
（``:1174-1179``）、滤波器（``:1661-1665``）。产出与 golden L1 逐位可比的状态，
并为后续子阶段（L2 预组装、L3/L4 求解、L4 力学）提供全部中间对象。

------------------------------------------------------------------------------
⚠ 三条 F4 红线（本模块的生死线）
------------------------------------------------------------------------------

1. 区域集合直接用 ``RegionMap.from_tags`` 的产物（构造式逐字照抄 legacy，
   ``motorfem/mesh/mesh.py`` 已保证），**禁止 ``sorted()``**。
2. 力学子网格的 ``element_rotation`` 是 ``dict(enumerate(set))`` —— 用 set 的
   迭代顺序定编号（``MMA10:346-348``）。探针（``tools/probe_mech.py``）用
   ``sorted`` 是为了**自身**可复现，FieldSystem 要对齐 golden 就绝不能照抄探针。
3. ``list(element_topo_opti_m)`` 的行列顺序必须来自 set —— 它决定滤波器
   （``MMA10:1264``）与一切 ``dT/dρ`` 的单元顺序。

------------------------------------------------------------------------------
力学子网格为什么在这里
------------------------------------------------------------------------------

力学子网格（转子铁芯重编号）的数据结构全部记录在 golden L1
（``element_S_mech`` / ``element_mech2raw`` / ``element_L_mech`` /
``element_mech_L_load`` / ``element_mech_slave`` …），所以它的构造归入本模块；
L2 的 ``Kmech_vectorized`` 与 L4 的力学求解都消费它。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..config import RunConfig
from ..kernels.fast.filter import compute_filter
from ..kernels.fast.simp import TableSIMPProjection
from ..kernels.fast.torque import arkkio_coefficient, arkbe_coefficient
from ..kernels.reference import assembly as asm
from ..kernels.reference import boundaries as bnd
from ..kernels.reference import moving_mesh as mm
from ..kernels.reference.element_matrices import electromagnetic_elements
from ..kernels.reference.simp import SIMPParams
from ..kernels.reference.torque import ArkkioGeometry
from ..mesh import DEFAULT_TAG_TABLE, MotorMesh, RegionMap

from ..winding.layout import SEC_TO_NAME, build_phase_element_sets, star_of_slots
from .init_density import filter_inverse_rho, initial_rho_simp
from . import mesh_sanity as msanity

__all__ = ["FieldTopology", "build_field_topology", "verify_winding_layout"]

#: legacy ``MMA10:338`` 硬编码的定子侧气隙半径（stator_gap 从 45 mm 开始）。
#: ⚠ 已由 ``cfg.geometry.stator_inner_radius_m`` 取代（默认值同 45 mm ⇒ SynRM
#: 逐位不变）；本常量仅存 legacy 参考值。
STATOR_GAP_RADIUS_M = 45.0e-3

#: legacy ``get_Arkkio_coef`` 硬编码的气隙宽度（``MMA10:693`` 的 ``0.25e-3``）。
#: ★ **它仍是有效默认值**：``build_field_topology`` 里当
#: ``cfg.geometry.arkkio_gap_m is None``（legacy/36 槽）时就用它。**不要**改成
#: ``stator_inner_radius_m − rotor_outer_radius_m`` —— 那个差在浮点下 =
#: 0.0002500000000000002 ≠ 0.25e-3，会让 4 个 fem golden 测试失败（2026-10-12
#: 实测）。6s4p 通过 ``geometry.arkkio_gap_m=0.5e-3`` 显式覆盖。
LEGACY_ARKKIO_GAP_M = 0.25e-3


def _rot_x(x: float, y: float, angle: float) -> float:
    """绕原点旋转的 x 分量。照抄 ``MMA10:177-178``。"""
    return x * np.cos(angle) - y * np.sin(angle)


def _rot_y(x: float, y: float, angle: float) -> float:
    """绕原点旋转的 y 分量。照抄 ``MMA10:181-182``。"""
    return x * np.sin(angle) + y * np.cos(angle)


def _mirror(x: float, y: float, angle: float) -> Tuple[float, float]:
    """关于角平分线 ``angle/2`` 的镜像。照抄 ``MMA10:189-190``。

    ``_mirror(x, y, θ) = (rot(−θ), rot_y(x, −y, θ))`` —— 注意 y 取负号，这是
    legacy 的约定（不是普通旋转），``:296`` 的对称单元配对靠它。
    """
    return _rot_x(x, y, -angle), _rot_y(x, -y, angle)


@dataclass
class FieldTopology:
    """golden L1 的全部状态（与 golden 逐位可比）+ 后续子阶段的内核对象。

    所有字段都指向前一个 ``build_field_topology`` 的产出；字段顺序即构造顺序。
    """

    cfg: RunConfig

    # ---- 原始网格（MMA10:227-232 读入 + mm→m）----
    node_raw: np.ndarray          # (4881, 4) [节点号, x, y, z]
    element_S_raw: List[list]     # (9599, 5) [单元id, n1, n2, n3, tag]
    element_L_raw: List[list]     # (1374, 4) [线元id, n1, n2, tag]
    element_P_raw: List[list]     # (187, 3)  [点元id, n1, tag]
    tags: np.ndarray              # (9599,) 每行 tag（= element_S_raw[:, 4]）

    # ---- 区域（RegionMap，构造式照抄 legacy ⇒ F4 序）----
    regions: RegionMap
    element_rotor_core_s: Dict[int, int]   # 对称合并后的 dict（MMA10:302）
    element_topo_opti_m: Set[int]          # = element_rotor_core_m（set）
    element_topo_opti_s: Dict[int, int]    # = element_rotor_core_s（dict）
    element_topo_opti: Set[int]            # set().union(m, s.keys())（MMA10:945）

    # ---- 电磁拓扑（动网格插层之后）----
    pairing0: bnd.PeriodicBoundary    # raw 主从配对（插层前）
    layer: mm.MovingMeshLayer         # 插层后的 xy/tri/pairing/rotor_nodes
    virtual: bnd.SlaveVirtualElements
    ctx: asm.TopologyContext
    em: Any                           # ElectromagneticElements（K0/Bxcoef/Bycoef/area）
    element_slave: Set[int]           # raw 表 + raw 配对（MMA10:323-325）
    element_S_slave: Dict[int, list]  # 插层更新后的虚拟单元表（键序 = element_slave）

    # ---- 插层后的表（golden L1 的 node / element_S）----
    node_post: np.ndarray             # (5117, 4)
    element_S_post: np.ndarray        # (9599, 5) —— 与 node_post 配套

    # ---- 三个 CSR（golden L1）----
    filter_SIMP: Any                  # (3181, 3181) csr
    filter_SIMP_grad: Any
    torque_arkkio_coef: Any           # (5117, 5117) csr
    # ★ 2026-10-14：体边积分版 Maxwell 应力转矩系数（T_ArkBE = AᵀQ_arkbe A）。
    #   生产默认的**纹波口径**（见 `config.ObjectiveConfig.ripple_source`）。
    #   与 `torque_arkkio_coef` 同样是**几何常数**（θ 无关、ρ 无关）⇒ 预组装一次。
    torque_arkbe_coef: Any            # (5117, 5117) csr

    # ---- 初始密度（golden L2 的 rho_SIMP_initial）----
    rho_SIMP_initial: np.ndarray      # (9599,)
    simp_inv: TableSIMPProjection

    # ---- 力学子网格（raw 编号体系，golden L1）----
    element_rotation: Dict[int, int]  # 单元id → 力学单元号（enumerate(set) 序）
    node_rotation: Set[int]
    node_mech: np.ndarray             # (3292, 4)，行序 = list(node_rotation) 的 set 序
    node_raw2mech: Dict[int, int]
    node_mech2raw: Dict[int, int]
    element_S_mech: List[list]        # (6362, 5)
    element_mech2raw: Dict[int, int]
    element_raw2mech: Dict[int, int]
    element_L_mech: List[list]        # (259, 4)
    element_L_mech2raw: Dict[int, int]
    element_L_raw2mech: Dict[int, int]
    element_mech_L_load: List[int]    # 118 条
    node_mech_constraint: Set[int]
    mech_pairing: bnd.PeriodicBoundary
    element_mech_slave: Set[int]
    element_S_mech_slave: Dict[int, list]

    # ---- 派生便捷量 ----
    n_node: int = 0                    # = node_post.shape[0]（5117）
    n_element: int = 0                 # = element_S_post.shape[0]（9599）
    n_node_mech: int = 0
    n_element_mech: int = 0

    # ---- 两级网格（2026-10-20，可选）----
    #: 细设计单元 → 粗设计变量索引（长度 = len(element_topo_opti_m)）。
    #: None = 单级（默认，行为逐位不变）；非 None = “细 FEM / 粗设计变量”。
    design_parent: Optional[np.ndarray] = None
    #: 设计变量个数（= 粗设计单元数）。None 时等于 len(element_topo_opti_m)。
    n_design: Optional[int] = None

    # ---------------------------------------------------------------- 快照
    def L1_snapshot(self) -> Dict[str, np.ndarray]:
        """golden L1 命名下的快照（键名与 ``golden/L1_regions.npz`` 一致）。

        存储语义照抄 ``tools/golden_capture.py`` 的 ``_capture_L1_L2``：
        set → ``__sorted``/``__raw_order`` 双份；dict → ``__keys``/``__values``；
        list of sets → ``__raw_order_{i:02d}__sorted``/``__raw``；列表 → ``__raw_order``。
        """
        out: Dict[str, np.ndarray] = {}
        out["node"] = np.asarray(self.node_post)
        out["element_P"] = np.asarray(self.element_P_raw)
        out["element_L"] = np.asarray(self.element_L_raw)
        out["element_S"] = np.asarray(self.element_S_post)

        for name, M in (
            ("filter_SIMP", self.filter_SIMP),
            ("filter_SIMP_grad", self.filter_SIMP_grad),
            ("Torque_Arkkio_coef", self.torque_arkkio_coef),
            # ⚠ 不把 `Torque_ArkBE_coef` 放进 `_arrays`：那是 golden L1 的快照，
            #   多一个 key 会被 `verify_golden` 报"多出"（147/0 的计数会变）。
        ):
            c = M.tocoo()
            out[f"{name}__data"] = np.asarray(c.data)
            out[f"{name}__row"] = np.asarray(c.row)
            out[f"{name}__col"] = np.asarray(c.col)
            out[f"{name}__shape"] = np.asarray(c.shape, dtype=np.int64)

        for name, value in self._legacy_globals().items():
            if isinstance(value, set):
                out[f"{name}__sorted"] = np.asarray(sorted(value))
                out[f"{name}__raw_order"] = np.asarray(list(value))
            elif isinstance(value, (list, tuple)):
                if value and all(isinstance(v, set) for v in value):
                    # element_stator_coil：list of sets（F4 双份）
                    for i, s in enumerate(value):
                        out[f"{name}__raw_order_{i:02d}__sorted"] = np.asarray(sorted(s))
                        out[f"{name}__raw_order_{i:02d}__raw"] = np.asarray(list(s))
                    continue
                # 同构表（element_S_mech / element_L_mech / element_mech_L_load …）
                out[f"{name}__raw_order"] = np.asarray(value)
            elif isinstance(value, dict):
                out[f"{name}__keys"] = np.asarray(list(value.keys()))
                out[f"{name}__values"] = np.asarray(list(value.values()))
            else:
                raise TypeError(f"{name}: 无法快照的类型 {type(value)}")

        # entity_* 列表（tag 表，legacy :246-252）
        for name, v in self._entity_lists().items():
            out[f"{name}__raw_order"] = np.asarray(v)
        return out

    def L2_snapshot_partial(self) -> Dict[str, np.ndarray]:
        """golden L2 里与 ρ 无关、本子阶段已能提供的部分。"""
        out: Dict[str, np.ndarray] = {}
        out["rho_SIMP_initial"] = np.asarray(self.rho_SIMP_initial)
        out["element_topo_opti_m__raw_order"] = np.asarray(list(self.element_topo_opti_m))
        out["element_topo_opti_s__keys"] = np.asarray(list(self.element_topo_opti_s.keys()))
        out["element_topo_opti_s__values"] = np.asarray(list(self.element_topo_opti_s.values()))
        return out

    # ---------------------------------------------------------------- 内部
    def _legacy_globals(self) -> Dict[str, Any]:
        """与 legacy capture 时 ``MMA10`` 模块全局同构（name → set/dict/list）。

        名字与 ``tools/golden_capture.py`` 的过滤条件（``element_*``/``entity_*``）
        对齐；``element_rotor_core_s`` 已是合并后的 dict，``element_topo_opti_s``
        归 L2 不在此列。
        """
        r = self.regions
        return {
            "element_rotor_shaft": r.region("element_rotor_shaft"),
            "element_rotor_core_m": r.region("element_rotor_core_m"),
            "element_rotor_core_s": self.element_rotor_core_s,
            "element_rotor_gap": r.region("element_rotor_gap"),
            "element_stator_gap": r.region("element_stator_gap"),
            "element_stator_core": r.region("element_stator_core"),
            "element_rotor_core": r.region("element_rotor_core"),
            "element_stator_coil": r.region("element_stator_coil"),
            "element_iron": r.region("element_iron"),
            "element_phaseAp": r.region("element_phaseAp"),
            "element_phaseAn": r.region("element_phaseAn"),
            "element_phaseBp": r.region("element_phaseBp"),
            "element_phaseBn": r.region("element_phaseBn"),
            "element_phaseCp": r.region("element_phaseCp"),
            "element_phaseCn": r.region("element_phaseCn"),
            "element_current": r.region("element_current"),
            "element_stator": r.region("element_stator"),
            "element_rotor": r.region("element_rotor"),
            "element_rotation": self.element_rotation,
            "element_S_mech": self.element_S_mech,
            "element_L_mech": self.element_L_mech,
            "element_mech2raw": self.element_mech2raw,
            "element_raw2mech": self.element_raw2mech,
            "element_L_mech2raw": self.element_L_mech2raw,
            "element_L_raw2mech": self.element_L_raw2mech,
            "element_mech_L_load": self.element_mech_L_load,
            "element_mech_slave": self.element_mech_slave,
            "element_S_mech_slave": self.element_S_mech_slave,
            "element_slave": self.element_slave,
            "element_S_slave": self.element_S_slave,
            "element_topo_opti_m": self.element_topo_opti_m,
            "element_topo_opti": self.element_topo_opti,
        }

    def _entity_lists(self) -> Dict[str, list]:
        """entity_*（legacy :246-252 的列表）。从 tag 表取，与 MotorMesh 一致。"""
        t = DEFAULT_TAG_TABLE
        return {
            "entity_shaft": list(t["element_rotor_shaft"]),
            "entity_rotor_m": list(t["element_rotor_core_m"]),
            "entity_rotor_s": list(t["element_rotor_core_s"]),
            "entity_rotor_gap": list(t["element_rotor_gap"]),
            "entity_stator_gap": list(t["element_stator_gap"]),
            "entity_stator": list(t["element_stator_core"]),
            "entity_coil": list(t["element_stator_coil"]),
        }


def verify_winding_layout(top: "FieldTopology") -> None:
    """断言生成器推导的相带单元集合 == RegionMap 的 ``element_phase*``（阶段 5）。

    用 star-of-slots 生成器（``ref_angle='edge'``，已验证逐位复现 legacy 表）从
    ``(Z, p, parts)`` 推导 6 个相带的单元集合，与 ``FieldTopology.regions`` 里
    tag 表构造的 ``element_phase{Ap,An,Bp,Bn,Cp,Cn}`` 逐位对比。不一致即 ``raise``
    —— 这是「换 Z/p/线圈几何后相带错位」的自动哨兵（设计文档 §1 的
    「换极槽必须改代码」问题）。

    ⚠ ``FieldSystem.__init__`` 默认调用（成本 ~1 ms）；对账已由
    ``tests/winding/test_layout.py`` 逐位钉住。
    """
    cfg = top.cfg
    w = cfg.electrical.winding
    lay = star_of_slots(
        cfg.geometry.Z, cfg.geometry.pole_pairs, 3,
        parts=cfg.geometry.parts, layers=2,
        inner_ref=w.inner_ref, outer_ref=w.outer_ref, outer_shift=w.outer_shift,
    )
    generated = build_phase_element_sets(lay, top.regions.region("element_stator_coil"))
    for name in SEC_TO_NAME[:6]:
        got = generated[name]
        exp = top.regions.region(f"element_phase{name}")
        if got != exp:
            # 诊断：先报集合差，再提示可能的原因
            only_got = sorted(got - exp)[:8]
            only_exp = sorted(exp - got)[:8]
            raise AssertionError(
                f"生成器推导的 element_phase{name} 与 tag 表不一致：\n"
                f"  生成器独有 {len(only_got)} 个（如 {only_got}）\n"
                f"  tag 表独有 {len(only_exp)} 个（如 {only_exp}）\n"
                f"  可能：Z/p/parts 与网格不符、线圈实体顺序变化、或生成器参数"
                f"（ref_angle）需要调整"
            )


def build_field_topology(
    config: RunConfig,
    mesh: MotorMesh,
    *,
    seed: int = 1002,
) -> FieldTopology:
    """组装 golden L1 的全部状态。构造顺序照抄 legacy 模块体。

    Parameters
    ----------
    config : RunConfig
        生产默认值即 golden 的配置（``config/config.py`` 已锚定）。
    mesh : MotorMesh
        已读入的 ``model_mirror4.msh``（节点已 mm→m，区域已按 tag 表划分）。
    seed : int
        ``np.random.seed`` 的值（初始密度的 RNG 重放，生产 1002）。

    Returns
    -------
    FieldTopology

    Notes
    -----
    本函数**有副作用**：会 ``np.random.seed(seed)`` 并消耗固定数量的随机数
    （初始密度）。这是 legacy 生产行为的一部分（``MMA10:42``），不是 import 级
    副作用 —— 调用一次等价于 legacy 跑一次初始化。
    """
    cfg = config
    geo = cfg.geometry
    node_raw = np.asarray(mesh.node)
    element_S_raw = mesh.element_S
    element_L_raw = mesh.element_L
    element_P_raw = mesh.element_P
    n_element = mesh.N_element
    tags = np.array([e[4] for e in element_S_raw], dtype=np.int64)
    xy = node_raw[:, 1:3]
    tri = np.array([e[1:4] for e in element_S_raw], dtype=np.int64)
    regions = mesh.regions
    err = geo.coord_err_abs
    parts = geo.parts

    # =================================================================
    # 1. element_rotor_core_s 对称合并（MMA10:292-302）
    # =================================================================
    element_rotor_core_m = regions.region("element_rotor_core_m")
    element_rotor_core_s = regions.region("element_rotor_core_s")
    _elem_x = np.array([float(np.mean(node_raw[element_S_raw[i][1:4], 1]))
                        for i in range(n_element)])
    _elem_y = np.array([float(np.mean(node_raw[element_S_raw[i][1:4], 2]))
                        for i in range(n_element)])
    mirror_ang = 2 * np.pi / parts

    # ⚠ 迭代 + 原地 remove 的模式（legacy :295-301）：内层循环命中即 remove 并
    # break，外层下一个 i_elem_m 时对**缩小后**的集合新建迭代器 —— CPython 下
    # 不会抛 RuntimeError，因为 remove 之后立刻 break。set 的迭代顺序决定
    # 命中的是哪个从属单元（多点容差内取首个），必须保留原式。
    _s_set = set(element_rotor_core_s)  # 副本：不动 RegionMap 的原集合
    _temp_rotor_core_s: Dict[int, int] = {}
    for i_elem_m in element_rotor_core_m:
        x_s, y_s = _mirror(_elem_x[i_elem_m], _elem_y[i_elem_m], mirror_ang)
        for i_elem_s in _s_set:
            if (_elem_x[i_elem_s] - x_s) ** 2 + (_elem_y[i_elem_s] - y_s) ** 2 < err ** 2:
                _temp_rotor_core_s[i_elem_s] = i_elem_m
                _s_set.remove(i_elem_s)
                break
    element_rotor_core_s_dict = _temp_rotor_core_s

    # ---- ★ 6s4p：半极设计域（设计域 = core_m = 0..45°）----
    # 上面 legacy 循环用 mirror_ang = 2π/parts 只映射了 q4（0..45° 关于 90° 镜像）。
    # 6s4p 的 core_s 还含 q2(45..90°)、q3(90..135°)，需归约到 core_m：
    #   q2 关于 45° 轴镜像、q3 绕原点旋转 −90°。
    # 逐单元试两种变换取更近者，命中容差 err 即写入 —— 设计域因此缩为**半个极**，
    # 与 36 槽 legacy 同构（用户 2026-10-12：设计域扩大到一个极会让对称性变差、
    # 优化难收敛）。legacy 路径（开关 False）一字未动。
    if cfg.geometry.rotor_design_half_pole:
        _m_list = list(element_rotor_core_m)
        _mx = np.array([_elem_x[_m] for _m in _m_list])
        _my = np.array([_elem_y[_m] for _m in _m_list])
        for i_elem_s in list(_s_set):
            _xs, _ys = _elem_x[i_elem_s], _elem_y[i_elem_s]
            _cand = []
            for _x2, _y2 in (
                _mirror(_xs, _ys, np.pi / 2),                       # q2：关于 45° 镜像
                (_rot_x(_xs, _ys, -np.pi / 2),                      # q3：旋转 −90°
                 _rot_y(_xs, _ys, -np.pi / 2)),
            ):
                _d2 = (_mx - _x2) ** 2 + (_my - _y2) ** 2
                _k = int(np.argmin(_d2))
                _cand.append((_d2[_k], _m_list[_k]))
            _best_d2, _best_m = min(_cand, key=lambda _t: _t[0])
            if _best_d2 < err ** 2:
                element_rotor_core_s_dict[i_elem_s] = _best_m

    # =================================================================
    # 2. 力学子网格（MMA10:346-436；★ 用 legacy 的 set 迭代序，不用 sorted）
    # =================================================================
    element_rotation = dict(
        (i_elem, i_elem_mech)
        for (i_elem_mech, i_elem) in enumerate(regions.region("element_rotor_core"))
    )
    node_rotation: Set[int] = set()
    for i_elem in element_rotation:
        node_rotation.add(element_S_raw[i_elem][1])
        node_rotation.add(element_S_raw[i_elem][2])
        node_rotation.add(element_S_raw[i_elem][3])

    node_mech = node_raw[list(node_rotation)]
    node_raw2mech = dict((int(node_mech[_i, 0]), _i) for _i in range(node_mech.shape[0]))
    node_mech2raw = dict((_i, int(node_mech[_i, 0])) for _i in range(node_mech.shape[0]))
    # legacy :359 把第 0 列就地改成 arange —— 这里不改原表，直接按行序解释
    #（行序 == list(node_rotation) 的 set 序 == 新编号序）。

    element_S_mech: List[list] = []
    element_mech2raw: Dict[int, int] = {}
    element_raw2mech: Dict[int, int] = {}
    for i_elem, i_elem_mech in element_rotation.items():
        element_S_mech.append([
            i_elem_mech,
            node_raw2mech[element_S_raw[i_elem][1]],
            node_raw2mech[element_S_raw[i_elem][2]],
            node_raw2mech[element_S_raw[i_elem][3]],
            element_S_raw[i_elem][4],
        ])
        element_mech2raw[i_elem_mech] = i_elem
        element_raw2mech[i_elem] = i_elem_mech

    # element_L_mech：element_L 里两端点都在 node_rotation 的行（MMA10:376-383）
    element_L_mech: List[list] = []
    element_L_mech2raw: Dict[int, int] = {}
    element_L_raw2mech: Dict[int, int] = {}
    for i_elem_mech, i_elem in enumerate([
        _elem_L[0] for _elem_L in element_L_raw
        if _elem_L[1] in node_rotation and _elem_L[2] in node_rotation
    ]):
        element_L_mech.append([
            i_elem_mech,
            node_raw2mech[element_L_raw[i_elem][1]],
            node_raw2mech[element_L_raw[i_elem][2]],
            element_L_raw[i_elem][3],
        ])
        element_L_mech2raw[i_elem_mech] = i_elem
        element_L_raw2mech[i_elem] = i_elem_mech

    n_node_mech = node_mech.shape[0]
    n_element_mech = len(element_S_mech)

    # 应力边界：两端点都在 m_Rrotor 上的线元（MMA10:389-395）
    element_mech_L_load: List[int] = []
    for elem_L in element_L_mech:
        i_elem_L, node_i, node_j = elem_L[0:3]
        if (abs(node_mech[node_i, 1] ** 2 + node_mech[node_i, 2] ** 2 - geo.rotor_outer_radius_m ** 2) <= err
                and abs(node_mech[node_j, 1] ** 2 + node_mech[node_j, 2] ** 2
                        - geo.rotor_outer_radius_m ** 2) <= err):
            element_mech_L_load.append(i_elem_L)

    # 位移边界：节点在 m_Rirotor 圆上（MMA10:398-400）
    node_mech_constraint: Set[int] = set(
        int(_i) for _i in range(n_node_mech)
        if abs((node_mech[_i, 1] ** 2 + node_mech[_i, 2] ** 2) - geo.shaft_radius_m ** 2) < err
    )

    # 力学主从配对（内核 6 通用配对，逐字同构于 :402-419）
    mech_pairing = bnd.pair_master_slave(
        node_mech[:, 1:3], parts=parts, coord_err_abs=err,
    )
    # 从属单元（:421-426）与虚拟表（:427-436）
    element_mech_slave: Set[int] = set(
        element_S_mech[i_elem][0] for i_elem in range(n_element_mech)
        if (element_S_mech[i_elem][1] in mech_pairing.slave_to_master
            or element_S_mech[i_elem][2] in mech_pairing.slave_to_master
            or element_S_mech[i_elem][3] in mech_pairing.slave_to_master)
    )
    element_S_mech_slave: Dict[int, list] = {}
    for i_elem in element_mech_slave:
        element_S_mech_slave[i_elem] = [
            element_S_mech[i_elem][0],
            mech_pairing.slave_to_master[element_S_mech[i_elem][1]]
            if element_S_mech[i_elem][1] in mech_pairing.slave_to_master else element_S_mech[i_elem][1],
            mech_pairing.slave_to_master[element_S_mech[i_elem][2]]
            if element_S_mech[i_elem][2] in mech_pairing.slave_to_master else element_S_mech[i_elem][2],
            mech_pairing.slave_to_master[element_S_mech[i_elem][3]]
            if element_S_mech[i_elem][3] in mech_pairing.slave_to_master else element_S_mech[i_elem][3],
            element_S_mech[i_elem][4],
        ]

    # =================================================================
    # 3. 电磁拓扑：主从配对 → 从属单元 → 动网格插层 → 单元矩阵
    # =================================================================
    pairing0 = bnd.pair_master_slave(xy, parts=parts, coord_err_abs=err)

    # element_slave：raw 表 + raw 配对（MMA10:323-325，插层之前构造、不再更新）
    node_slave0 = pairing0.slave_to_master
    element_slave: Set[int] = set(
        _elem[0] for _elem in element_S_raw
        if _elem[1] in node_slave0 or _elem[2] in node_slave0 or _elem[3] in node_slave0
    )

    # 气隙中线（MMA10:338-340）
    m_R_gap = (cfg.geometry.stator_inner_radius_m + geo.rotor_outer_radius_m) / 2
    node_gap_stator: Set[int] = set(
        int(_node[0]) for _node in node_raw
        if abs((_node[1] ** 2 + _node[2] ** 2) - m_R_gap ** 2) < err
    )

    # ★ 护栏（2026-10-14）：`insert_moving_mesh_layer` 给 `interface_nodes` 里
    #   **每个**节点复制一份，但**只有落在 `element_rotor_gap` 单元里的才被重连**。
    #   半径判据 `|x²+y² − m_R_gap²| < err` 在**细网格**上会**多收**不在气隙单元层里的
    #   节点（实测：`gapall0.02_order2` 多收 14114 个、`gapall0.02_order1` 多收 4 个），
    #   它们的副本就成了**零连通节点** ⇒ 组装后出现**零对角行** ⇒
    #   `CholmodNotPositiveDefinite` / `Matrix is exactly singular`。
    #   ⇒ 裁成「**真正是 `element_rotor_gap` 单元顶点**」的节点。
    #   **对正常网格是 no-op**（接口圆上的节点本来就都是气隙单元顶点），
    #   且 `prune_interface_nodes` **顺序保持** ⇒ 新节点编号 `N_node + i` 不变 ⇒
    #   **golden 逐位不受影响**（已验证 147/0）。
    #   详见 `docs/转矩口径等价链核对与动网格插值_2026-10-13.md` §4.6。
    _elm_rotor_gap = regions.region("element_rotor_gap")
    node_gap_kept, _node_gap_dropped = msanity.prune_interface_nodes(
        node_gap_stator, tri, _elm_rotor_gap)
    if _node_gap_dropped:
        print(f"[mesh_sanity] 气隙中线判据多收 {len(_node_gap_dropped)} 个不在气隙单元层"
              f"的节点（{sorted(_node_gap_dropped)[:8]}…），已剔除"
              f" —— 否则其副本零连通 ⇒ 矩阵奇异")

    layer = mm.insert_moving_mesh_layer(
        xy, tri, node_gap_kept, _elm_rotor_gap, pairing0,
        element_stator=regions.region("element_stator"),
        element_rotor=regions.region("element_rotor"),
    )
    virtual = bnd.slave_virtual_elements(layer.tri, layer.pairing)
    ctx = asm.TopologyContext.from_boundaries(
        layer.tri, virtual, layer.pairing, regions.region("element_rotor_gap"),
        a_eq=cfg.solver.EM_MS_boundary_A_eq, n_node=layer.n_node,
    )
    em = electromagnetic_elements(layer.xy, layer.tri)

    # 插层后的表（golden 的 node / element_S；MMA10:448-463）
    gap_rows = np.array(
        [[new, node_raw[orig, 1], node_raw[orig, 2], 0.0]
         for new, orig in layer.rotor_to_stator.items()],
        dtype=float,
    )
    node_post = np.concatenate([node_raw, gap_rows], axis=0)
    element_S_post = np.array(
        [[i, layer.tri[i, 0], layer.tri[i, 1], layer.tri[i, 2], tags[i]]
         for i in range(n_element)],
        dtype=np.int64,
    )

    # element_S_slave 最终值（虚拟节点号，键序 = element_slave 的 set 序）
    element_S_slave: Dict[int, list] = {}
    for e in element_slave:
        n1, n2, n3 = virtual.virtual_tri[e]
        element_S_slave[e] = [e, n1, n2, n3, int(tags[e])]

    # =================================================================
    # 4. 拓扑优化区域（MMA10:943-945）＋ 滤波器（:1661-1665）＋ Arkkio（:700）
    # =================================================================
    element_topo_opti_m = element_rotor_core_m          # set（F4：list() 即行序）
    element_topo_opti_s = element_rotor_core_s_dict
    element_topo_opti = set().union(element_topo_opti_m, element_topo_opti_s.keys())

    filter_SIMP = compute_filter(
        node_post, element_S_post, element_topo_opti_m,
        cfg.filter.filter_radius_m, extend=True, p_filter=cfg.filter.filter_p_exp,
        parts=parts,
    )
    filter_SIMP_grad = compute_filter(
        node_post, element_S_post, element_topo_opti_m,
        cfg.filter.filter_grad_radius_m, extend=True, p_filter=cfg.filter.filter_grad_p_exp,
        parts=parts,
    )
    torque_arkkio_coef = arkkio_coefficient(ArkkioGeometry(
        tri=layer.tri, xy=layer.xy, area=em.area,
        bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        gap_elements=regions.region("element_rotor_gap"),
        rotor_nodes=layer.rotor_nodes,
        parts=parts, mu0=cfg.material.mu0,
        gap=(LEGACY_ARKKIO_GAP_M if cfg.geometry.arkkio_gap_m is None
             else cfg.geometry.arkkio_gap_m),
        parts_exponent=cfg.geometry.arkkio_parts_exponent,
    ))
    # ★ 2026-10-14：体边积分版 Maxwell 应力转矩系数（新，无 legacy 对应物）。
    #   `T_ArkBE = AᵀQ_arkbe A`，与 `torque_arkkio_coef` 同构 ⇒ 可直接复用
    #   `torque_quadratic`（值）与 `adjoint_torque_derivative`（伴导）。
    #   几何常数（θ/ρ 无关）⇒ 预组装一次。
    #   为什么生产要用它：`T_Arkkio` 的 h 标定在细网格上偏 **45%**，而 `T_ArkBE`
    #   是**无线积分**、网格稳健（细网格贴 ANSYS 0.69%、与 T_VD 互吻 0.34%）。
    #   详见 `docs/转矩口径等价链核对与动网格插值_2026-10-13.md` §4.4。
    torque_arkbe_coef = arkbe_coefficient(ArkkioGeometry(
        tri=layer.tri, xy=layer.xy, area=em.area,
        bxcoef=em.Bxcoef, bycoef=em.Bycoef,
        gap_elements=regions.region("element_rotor_gap"),
        rotor_nodes=layer.rotor_nodes,
        parts=parts, mu0=cfg.material.mu0,
        gap=(LEGACY_ARKKIO_GAP_M if cfg.geometry.arkkio_gap_m is None
             else cfg.geometry.arkkio_gap_m),
        parts_exponent=cfg.geometry.arkkio_parts_exponent,
    ))

    # =================================================================
    # 5. 初始密度（MMA10:42 + :1174-1179 随机空洞，:1672-1686 滤波器前像）
    # =================================================================
    simp_inv = TableSIMPProjection(SIMPParams())
    rho_SIMP_initial = initial_rho_simp(
        node_post, element_S_post, element_topo_opti_m,
        n_element=n_element, seed=seed,
        rho_init_radius=cfg.topology.rho_init_radius_m,
        rho_init_distance=cfg.topology.rho_init_distance_m,
        m_Rrotor=geo.rotor_outer_radius_m,
        m_Rirotor=geo.shaft_radius_m,
        parts=parts,
        simp_inv=simp_inv.invert,
        min_density=SIMPParams().min_density,
    )
    # 滤波器前像：rho_SIMP[topo_m] = res.x（MMA10:1672-1686，L-BFGS-B）。
    # ★ golden 的 rho_SIMP_initial 是这一步之后的密度 —— 漏掉这一步
    #   `verify_golden --layer L2` 会差 ~0.7（见 init_density 模块 docstring 更正）。
    rho_SIMP_initial = filter_inverse_rho(
        filter_SIMP, rho_SIMP_initial, element_topo_opti_m,
        min_density=SIMPParams().min_density,
    )

    return FieldTopology(
        cfg=cfg,
        node_raw=node_raw,
        element_S_raw=element_S_raw,
        element_L_raw=element_L_raw,
        element_P_raw=element_P_raw,
        tags=tags,
        regions=regions,
        element_rotor_core_s=element_rotor_core_s_dict,
        element_topo_opti_m=element_topo_opti_m,
        element_topo_opti_s=element_topo_opti_s,
        element_topo_opti=element_topo_opti,
        pairing0=pairing0,
        layer=layer,
        virtual=virtual,
        ctx=ctx,
        em=em,
        element_slave=element_slave,
        element_S_slave=element_S_slave,
        node_post=node_post,
        element_S_post=element_S_post,
        filter_SIMP=filter_SIMP,
        filter_SIMP_grad=filter_SIMP_grad,
        torque_arkkio_coef=torque_arkkio_coef,
        torque_arkbe_coef=torque_arkbe_coef,
        rho_SIMP_initial=rho_SIMP_initial,
        simp_inv=simp_inv,
        element_rotation=element_rotation,
        node_rotation=node_rotation,
        node_mech=node_mech,
        node_raw2mech=node_raw2mech,
        node_mech2raw=node_mech2raw,
        element_S_mech=element_S_mech,
        element_mech2raw=element_mech2raw,
        element_raw2mech=element_raw2mech,
        element_L_mech=element_L_mech,
        element_L_mech2raw=element_L_mech2raw,
        element_L_raw2mech=element_L_raw2mech,
        element_mech_L_load=element_mech_L_load,
        node_mech_constraint=node_mech_constraint,
        mech_pairing=mech_pairing,
        element_mech_slave=element_mech_slave,
        element_S_mech_slave=element_S_mech_slave,
        n_node=node_post.shape[0],
        n_element=n_element,
        n_node_mech=n_node_mech,
        n_element_mech=n_element_mech,
    )
