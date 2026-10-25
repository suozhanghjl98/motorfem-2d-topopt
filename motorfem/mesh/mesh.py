"""motorfem.mesh —— 阶段 2：网格与区域数据契约（MotorMesh + RegionMap）

把 legacy 的「读 .msh + 区域划分」从模块级脚本里提取成可独立构造的对象
（设计文档 §阶段 2 末段）：

* ``gmsh_reader``：从 ``legacy/pytorch/my_gmsh/gmsh_reader.py`` **保真搬移**
  （逐行照抄，唯一改动是 docstring）。读取语义必须与 legacy 完全相同
  （``index_begins_with_zero=True``、``unique_index_for_any_element=False``，
  见 ``MMA10:227-229``），否则区域编号对不上。
* ``MotorMesh``：读网格 + mm→m 单位换算（``node[:, 1:] *= 1e-3``，同 MMA10:230）
  + 按 tag 表划分区域，暴露 ``node`` / ``element_{S,L,P,V}`` / ``region(name)``。

**⚠ 与 golden L1 的差异是设计使然**：golden 的 ``node``/``element_S`` 是 MMA10
**运行时**视图 —— 模块体在 ``MMA10:455-456`` 给气隙转子侧**追加 236 个滑移层节点**
并把 470 个 tag5 气隙单元的节点索引重映射（``MMA10:463``，动网格，内核 7 处理）。
``MotorMesh`` 提供**原始网格契约**（reader 输出 + mm 换算，4881 节点）：滑移层是
求解器运行时行为，不属于网格数据契约。验证：``MotorMesh.node`` == golden ``node[:4881]``
（扩充节点 append 在后），``element_S`` 的单元 idx 列不变。
* ``RegionMap``：**旧名 → set 的冻结映射**（设计文档阶段 3 的过渡支点，
  ``element_rotor_shaft``、``element_phaseAp`` … 全部保留 legacy 原名）。
  ``from_tags`` 由 tag 表重建 tag 驱动的区域，构造表达式**逐字照抄** legacy
  （``set([_elem[0] for _elem in element_S if _elem[4] in entity])``），保证
  set 的迭代顺序与 legacy 逐位一致（F4，见设计文档 §2）；``from_legacy``
  从命名空间快照直接拿集合，是搬家期间的权威来源。

**不覆盖的部分**（需要节点坐标/排序，见 ``MMA10:288-375``，留给求解器层）：
``node_Dirichlet`` / ``node_master`` / ``node_slave`` / ``element_slave`` /
``node_gap_stator`` / 力学子网格（``element_rotation``、``element_S_mech`` 等）——
它们依赖坐标计算与 dict 迭代顺序（F4 敏感），tag 表重建不保证逐位一致。

------------------------------------------------------------------------------
阶段 3 增量（区域命名契约，见 ``docs/重构设计文档.md`` §阶段 3）
------------------------------------------------------------------------------

* ``RegionMap.from_physical_groups``：**过渡支点** —— 由物理组（.msh 的
  ``$PhysicalNames`` + ``$Entities`` 解析，``motorfem/mesh/physical.py``）+ 角色
  映射还原旧名映射。实现方式是「解析物理组 → 派生 tag 表 → 委托 ``from_tags``」，
  保证与 tag 表构造**位级相同**（F4：构造表达式一字未动）。
* ``RegionMap.check_partition``：三道闸之三（区域覆盖性）。
* ``MotorMesh(regions="auto")``：网格携带 ``$PhysicalNames`` 时自动走物理组路径，
  否则回退 tag 表 —— 冻结生产网格（无 ``$PhysicalNames``）行为不变。
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .gates import require_regions, verify_fingerprint
from .physical import read_physical_groups
from .physical_groups import SYNRM_ROLE_TO_GROUP, SYNRM_SLOT_GROUP_NAMES
from ..winding.layout import build_phase_element_sets

__all__ = ["gmsh_reader", "MotorMesh", "RegionMap", "DEFAULT_TAG_TABLE"]


# ---------------------------------------------------------------- gmsh_reader
def gmsh_reader(filename: str, index_begins_with_zero: bool = True,
                unique_index_for_any_element: bool = True):
    """读取 gmsh 2.x/4.x 文本格式的 .msh，返回 (node, element_P, element_L,
    element_S, element_V)。

    从 legacy 保真搬移（``legacy/pytorch/my_gmsh/gmsh_reader.py``，2024-06-12）。
    **语义约定**：``index_begins_with_zero=True`` 把节点/单元编号减 1（0 基）；
    ``unique_index_for_any_element=False`` 时 P/L/S/V 各自连续编号（MMA10 用它）；
    True 时全局唯一编号。会剔除未用节点并保证三角形正序（Δ>0）。
    注意 legacy 对 'e' 记号：``data.split(' ')`` 依赖 gmsh 用单个空格分隔。
    """
    node_flag = 0
    element_flag = 0

    node = None
    element_P = None
    element_L = None
    element_S = None
    element_V = None

    fid = open(filename, "r")

    # 连续读取直到$Nodes标签
    while True:
        data = fid.readline()
        if not data:
            break
        if "$Nodes" in data:
            break

    if "$Nodes" in data and not node_flag:
        data = fid.readline()
        data = data.split(" ")
        numEntityBlocks = int(data[0])
        numNodes = int(data[1])
        minNodeTag = int(data[2])
        maxNodeTag = int(data[3])

        node = np.zeros((numNodes, 4))
        index_node = 0

        for iEntityBlocks in range(numEntityBlocks):
            data = fid.readline()
            data = data.split(" ")
            entityDim = int(data[0])
            entityTag = int(data[1])
            parametric = int(data[2])
            numNodesInBlock = int(data[3])

            for iNodesInBlock in range(numNodesInBlock):
                data = fid.readline()
                _index_node = int(data)
                if index_begins_with_zero:
                    _index_node -= 1
                node[index_node + iNodesInBlock, 0] = _index_node

            for iNodesInBlock in range(numNodesInBlock):
                data = fid.readline()
                data = data.split(" ")
                _x = float(data[0])
                _y = float(data[1])
                _z = float(data[2])
                node[index_node + iNodesInBlock, 1] = _x
                node[index_node + iNodesInBlock, 2] = _y
                node[index_node + iNodesInBlock, 3] = _z

            index_node += numNodesInBlock

        node_flag = 1
        data = fid.readline()  # 下一行应该为$EndNodes
        if "$EndNodes" not in data:
            print("$EndNodes NOT FOUND!!")

    # 连续读取直到$Elements标签
    while True:
        data = fid.readline()
        if not data:
            break
        if "$Elements" in data:
            break

    if "$Elements" in data and not element_flag:
        data = fid.readline()
        data = data.split(" ")
        numEntityBlocks = int(data[0])
        numElements = int(data[1])
        minElementTag = int(data[2])
        maxElementTag = int(data[3])

        element_P = []
        element_L = []
        element_S = []
        element_V = []

        for iEntityBlocks in range(numEntityBlocks):
            data = fid.readline()
            data = data.split(" ")
            entityDim = int(data[0])
            entityTag = int(data[1])
            elementType = int(data[2])
            numElementsInBlock = int(data[3])

            for iElementsInBlock in range(numElementsInBlock):
                data = fid.readline()
                data = data.split(" ")
                if entityDim == 0:
                    if not index_begins_with_zero:
                        _element = [int(data[0]), int(data[1]), entityTag]
                    else:
                        _element = [int(data[0]) - 1, int(data[1]) - 1, entityTag]
                    element_P.append(_element)
                elif entityDim == 1:
                    if not index_begins_with_zero:
                        _element = [int(data[0]), int(data[1]), int(data[2]), entityTag]
                    else:
                        _element = [int(data[0]) - 1, int(data[1]) - 1, int(data[2]) - 1,
                                    entityTag]
                    element_L.append(_element)
                elif entityDim == 2:
                    if not index_begins_with_zero:
                        _element = [int(data[0]), int(data[1]), int(data[2]), int(data[3]),
                                    entityTag]
                    else:
                        _element = [int(data[0]) - 1, int(data[1]) - 1, int(data[2]) - 1,
                                    int(data[3]) - 1, entityTag]
                    element_S.append(_element)
                elif entityDim == 3:
                    if not index_begins_with_zero:
                        _element = [int(data[0]), int(data[1]), int(data[2]), int(data[3]),
                                    int(data[4]), entityTag]
                    else:
                        _element = [
                            int(data[0]) - 1, int(data[1]) - 1, int(data[2]) - 1,
                            int(data[3]) - 1, int(data[4]) - 1, entityTag,
                        ]
                        element_V.append(_element)

    element_flag = 1
    data = fid.readline()
    if "$EndElements" not in data:
        print("$EndElements NOT FOUND!!")

    fid.close()

    if not unique_index_for_any_element:
        # 令每种类型的单元分别进行编号
        for i_P in range(len(element_P)):
            element_P[i_P][0] = i_P if index_begins_with_zero else (i_P + 1)
        for i_L in range(len(element_L)):
            element_L[i_L][0] = i_L if index_begins_with_zero else (i_L + 1)
        for i_S in range(len(element_S)):
            element_S[i_S][0] = i_S if index_begins_with_zero else (i_S + 1)
        for i_V in range(len(element_V)):
            element_V[i_V][0] = i_V if index_begins_with_zero else (i_V + 1)

    # 去除不含于单元中的节点
    node_in_elem = set()
    for _elem in element_L:
        node_in_elem.update(_elem[1:-1])
    for _elem in element_S:
        node_in_elem.update(_elem[1:-1])
    for _elem in element_V:
        node_in_elem.update(_elem[1:-1])

    if len(node_in_elem) != len(node):
        new_node = dict((j, i) for (i, j) in enumerate(sorted(list(node_in_elem))))

        node = node[list(node_in_elem), :]
        node[:, 0] = np.arange(len(node_in_elem))

        for _elem in element_P:
            if _elem[1] not in node_in_elem:
                del _elem
        for _elem in element_L:
            for _i in range(1, 3):
                _elem[_i] = new_node[_elem[_i]]
        for _elem in element_S:
            for _i in range(1, 4):
                _elem[_i] = new_node[_elem[_i]]
        for _elem in element_V:
            for _i in range(1, 5):
                _elem[_i] = new_node[_elem[_i]]

    # 确保三角形单元的正序
    for _elem in element_S:
        node_i, node_j, node_m = _elem[1:4]
        xi, yi = node[node_i, 1:3]
        xj, yj = node[node_j, 1:3]
        xm, ym = node[node_m, 1:3]
        bi = yj - ym
        bj = ym - yi
        ci = -xj + xm
        cj = -xm + xi

        Delta = 1 / 2 * (bi * cj - bj * ci)

        if Delta < 0:
            _elem[1:4] = [node_j, node_i, node_m]

    return node, element_P, element_L, element_S, element_V


# ---------------------------------------------------------------- tag 表
#: model_mirror4 的 entity tag 表（照抄 MMA10:246-252）。tag 语义由
#: legacy_msh/model_mirror4.model.json 的 dim2 直方图确认（阶段 0.3）。
DEFAULT_TAG_TABLE: Dict[str, Tuple[int, ...]] = {
    "element_rotor_shaft": (1,),
    "element_rotor_core_m": (2,),
    "element_rotor_core_s": (3,),
    "element_rotor_gap": (4, 5),
    "element_stator_gap": (6, 7),
    "element_stator_core": (8,),
    "element_stator_coil": tuple(range(9, 9 + 18)),
}


# ---------------------------------------------------------------- RegionMap
class RegionMap:
    """旧名 → 单元集合的冻结映射（设计文档阶段 3 的 ResolvedRegions）。

    名字与 legacy 完全一致（``element_rotor_shaft``、``element_phaseAp`` …），
    使「搬家期间由 tag 表构造、阶段 3 后由物理组 + 绕组布局构造，而上层数值代码
    一行没动」成为可能。
    """

    def __init__(self, regions: Dict[str, Any],
                 primitives: Tuple[Tuple[str, Set[int]], ...] = ()):
        self._regions = dict(regions)
        #: primitive 区域（两两不相交、并集 == 全部单元；覆盖性检查用）。
        #: from_tags：tag 表键（6）+ 每槽（18）；from_physical_groups：同（委托 from_tags）。
        self._primitives = primitives

    def region(self, name: str):
        """按 legacy 名取区域（set / dict / list of sets 原样返回）。未知名 KeyError。"""
        return self._regions[name]

    def names(self) -> Tuple[str, ...]:
        return tuple(self._regions.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._regions

    # -------------------------------------------------- 构造：tag 表重建
    @classmethod
    def from_tags(cls, element_S, tag_table: Dict[str, Tuple[int, ...]] = None,
                  *, layout=None) -> "RegionMap":
        """由 tag 表 + ``element_S``（每行 ``[idx, n1, n2, n3, tag]``）重建
        tag 驱动的区域集合。

        ⚠ **构造表达式逐字照抄 legacy**（``MMA10:255-283``）：set 的迭代顺序由
        构造路径决定（F4），换写法会让同一集合的 ``list(set)`` 顺序不同、浮点末位
        不同。这里每个 ``set([_elem[0] for _elem in element_S if _elem[4] in tags])``
        都是 legacy 原式的直接搬移。

        Parameters
        ----------
        layout : WindingLayout, 可选
            ``None``（默认）= 相带由 legacy 的 **18 槽硬编码表**合成（
            ``MMA10:270-277``，SynRM golden 路径，逐位不变）；
            给出 ``WindingLayout`` 时，相带改由 star-of-slots 生成器
            （``build_phase_element_sets``）推导 —— 这是**换极槽/换电机**的参数化
            入口（如 6s4p：``star_of_slots(6, 2, 3, parts=2, inner_ref='start',
            outer_ref='center', outer_shift=0)``）。此时 ``element_stator_coil``
            长度须 == ``layout.n_coils_per_part``。
        """
        tags = tag_table or DEFAULT_TAG_TABLE
        regions: Dict[str, Any] = {}

        def _by_tags(name: str, entity: Iterable[int]) -> Set[int]:
            return set([_elem[0] for _elem in element_S if _elem[4] in entity])

        # 基础（tag 驱动）
        element_rotor_shaft = _by_tags("shaft", tags["element_rotor_shaft"])
        element_rotor_core_m = _by_tags("core_m", tags["element_rotor_core_m"])
        element_rotor_core_s = _by_tags("core_s", tags["element_rotor_core_s"])
        element_rotor_gap = _by_tags("rotor_gap", tags["element_rotor_gap"])
        element_stator_gap = _by_tags("stator_gap", tags["element_stator_gap"])
        element_stator_core = _by_tags("stator_core", tags["element_stator_core"])

        # element_stator_coil：每个槽一个 set（list of sets，照抄 MMA10:264-267）
        element_stator_coil = []
        for _entity in tags["element_stator_coil"]:
            element_stator_coil.append(set(_elem[0] for _elem in element_S if _elem[4] == _entity))

        # 派生态（union 链，照抄 MMA10:262-283）
        element_rotor_core = set().union(element_rotor_core_m, element_rotor_core_s)
        element_iron = set().union(element_rotor_core, element_stator_core)
        if layout is None:
            # ⚠ legacy 18 槽硬编码相带合成（MMA10:270-277）—— 表达式与原书写顺序
            # 一字未动（仅整体缩进一层），SynRM golden 逐位不变。
            element_phaseAp = set().union(element_stator_coil[0], element_stator_coil[1],
                                          element_stator_coil[2])
            element_phaseAn = set().union(element_stator_coil[15], element_stator_coil[16],
                                          element_stator_coil[17])
            element_phaseBp = set().union(element_stator_coil[9], element_stator_coil[10],
                                          element_stator_coil[11], element_stator_coil[12],
                                          element_stator_coil[13], element_stator_coil[14])
            element_phaseBn = set()
            element_phaseCp = set()
            element_phaseCn = set().union(element_stator_coil[3], element_stator_coil[4],
                                          element_stator_coil[5], element_stator_coil[6],
                                          element_stator_coil[7], element_stator_coil[8])
        else:
            # 参数化：相带由 star-of-slots 生成器推导（换极槽/换电机的唯一入口）。
            _ps = build_phase_element_sets(layout, element_stator_coil)
            element_phaseAp = _ps["Ap"]
            element_phaseAn = _ps["An"]
            element_phaseBp = _ps["Bp"]
            element_phaseBn = _ps["Bn"]
            element_phaseCp = _ps["Cp"]
            element_phaseCn = _ps["Cn"]
        element_current = set().union(element_phaseAp, element_phaseAn, element_phaseBp,
                                      element_phaseBn, element_phaseCp, element_phaseCn)
        element_stator = set().union(*element_stator_coil, element_stator_core, element_stator_gap)
        element_rotor = set().union(element_rotor_core, element_rotor_shaft, element_rotor_gap)

        # 覆盖性检查用的 primitive 区域（引用上面构造的同一批 set，不加不减、
        # 不改构造表达式 —— 只记录引用）。6 个 tag 键 + 18 个槽 = 24 个。
        primitives = (
            ("element_rotor_shaft", element_rotor_shaft),
            ("element_rotor_core_m", element_rotor_core_m),
            ("element_rotor_core_s", element_rotor_core_s),
            ("element_rotor_gap", element_rotor_gap),
            ("element_stator_gap", element_stator_gap),
            ("element_stator_core", element_stator_core),
        ) + tuple(
            (f"winding.slot{k}", s) for k, s in enumerate(element_stator_coil)
        )

        return cls({
            "element_rotor_shaft": element_rotor_shaft,
            "element_rotor_core_m": element_rotor_core_m,
            "element_rotor_core_s": element_rotor_core_s,
            "element_rotor_gap": element_rotor_gap,
            "element_stator_gap": element_stator_gap,
            "element_stator_core": element_stator_core,
            "element_stator_coil": element_stator_coil,
            "element_rotor_core": element_rotor_core,
            "element_iron": element_iron,
            "element_phaseAp": element_phaseAp,
            "element_phaseAn": element_phaseAn,
            "element_phaseBp": element_phaseBp,
            "element_phaseBn": element_phaseBn,
            "element_phaseCp": element_phaseCp,
            "element_phaseCn": element_phaseCn,
            "element_current": element_current,
            "element_stator": element_stator,
            "element_rotor": element_rotor,
        }, primitives=primitives)

    # -------------------------------------------------- 构造：物理组（阶段 3）
    @classmethod
    def from_physical_groups(
        cls,
        element_S,
        group_to_entity_tags: Dict[str, Tuple[int, ...]],
        *,
        role_map: Dict[str, str],
        slot_group_names: Tuple[str, ...],
        layout=None,
    ) -> "RegionMap":
        """由物理组还原旧名映射（设计文档阶段 3 的**过渡支点**）。

        Parameters
        ----------
        element_S :
            面单元表（每行 ``[idx, n1, n2, n3, entity_tag]``）。
        group_to_entity_tags :
            dim-2 物理组名 → 其 elementary entity tag 集合（由
            ``motorfem/mesh/physical.py`` 的 ``read_physical_groups`` 从 .msh 的
            ``$Entities`` 解析）。未入任何物理组的 elementary 面不在任何组里，
            覆盖性检查会抓到它们。
        role_map :
            legacy 区域键 → 物理组名（``SYNRM_ROLE_TO_GROUP``；换电机传对应表）。
        slot_group_names :
            槽组名（按全局槽序），``element_stator_coil`` 的 list 顺序由它决定。

        Notes
        -----
        **实现方式**：解析物理组 → 派生一张 legacy tag 表 → **委托 ``from_tags``**。
        ``from_tags`` 的构造表达式一字未动 ⇒ 与 tag 表构造位级相同（F4）。
        区域的事实来源因此变成物理组：换网格后 elementary tag 重排，只要物理组名
        还在，旧名映射就仍然正确 —— 这就是「断掉 tag 恰好对齐」。
        「+ 绕组布局」由阶段 5 的 ``verify_winding_layout`` 哨兵承担（生成器推导
        相带 vs 区域表逐位对账），不在本构造函数里重复实现。
        """
        missing = [name for name in role_map.values() if name not in group_to_entity_tags]
        if missing:
            raise KeyError(
                f"物理组缺少 legacy 角色所需的名: {missing}；网格实际拥有的面组: "
                f"{sorted(group_to_entity_tags)}"
            )
        missing_slots = [n for n in slot_group_names if n not in group_to_entity_tags]
        if missing_slots:
            raise KeyError(
                f"物理组缺少槽组: {missing_slots}；网格实际拥有的面组: "
                f"{sorted(group_to_entity_tags)}"
            )
        # role_map 必须覆盖全部 6 个非槽 legacy 键（from_tags 的构造依赖它们），
        # 槽数必须 == 18（from_tags 的相带合成硬编码 element_stator_coil[0..17]）。
        # 缺了直接报清晰错误，而不是让 from_tags 深处抛 KeyError/IndexError。
        missing_keys = [k for k in DEFAULT_TAG_TABLE if k != "element_stator_coil"
                        and k not in role_map]
        if missing_keys:
            raise KeyError(
                f"role_map 缺 legacy 键: {missing_keys}（需覆盖 "
                f"{sorted(k for k in DEFAULT_TAG_TABLE if k != 'element_stator_coil')}）"
            )
        n_slots = len(DEFAULT_TAG_TABLE["element_stator_coil"])
        if layout is None:
            if len(slot_group_names) != n_slots:
                raise ValueError(
                    f"槽组数 {len(slot_group_names)} != {n_slots}（from_tags 的相带合成"
                    f"按 legacy 表写死 18 槽，见设计文档 §12 登记；参数化请传 layout）"
                )
        else:
            if len(slot_group_names) != layout.n_coils_per_part:
                raise ValueError(
                    f"槽组数 {len(slot_group_names)} != 生成器每扇区线圈数 "
                    f"{layout.n_coils_per_part}（Z={layout.Z} parts={layout.parts} "
                    f"layers={layout.layers}）"
                )
        tag_table = {role: group_to_entity_tags[name] for role, name in role_map.items()}
        # 槽表：按槽序展平每个槽组的实体 tag（SynRM 每槽恰 1 个实体 ⇒ 18 个 tag）。
        # ⚠ 每槽 >1 实体时 from_tags 的逐实体结构会失配 —— 现阶段几何没有这种情况，
        #   若出现需先扩展 from_tags（登记，见设计文档 §12）。
        tag_table["element_stator_coil"] = tuple(
            tag for name in slot_group_names for tag in group_to_entity_tags[name]
        )
        return cls.from_tags(element_S, tag_table, layout=layout)

    # -------------------------------------------------- 覆盖性检查（闸③）
    def check_partition(self, n_element: int) -> None:
        """区域覆盖性检查（三道闸之三）：primitive 区域两两不相交、并集 == 全部单元。

        primitive 区域 = 构成映射的不可再分单元集合（tag 路径：6 个 tag 键 + 18 个
        槽；物理组路径：同，委托 from_tags）。断言：

        * 两两不相交：任何单元至多属于一个 primitive 区域；
        * 并集 == ``range(n_element)``：每个单元都属于某个 primitive 区域。

        这条检查在 legacy 里做不到，正是因为物理组不存在 —— 它是 F1（改几何就
        静默算错）的自动探测器：多一个没进表的 tag / 少一组都会立刻报错。
        """
        union: Set[int] = set()
        for name, s in self._primitives:
            overlap = union & s
            if overlap:
                sample = sorted(overlap)[:8]
                raise ValueError(
                    f"区域覆盖性检查失败：{name} 与既有 primitive 交集 {len(overlap)} 个"
                    f"（如 {sample}）—— 区域不互斥，网格/物理组表可能不匹配。"
                )
            union |= s
        expected = set(range(n_element))
        if union != expected:
            missing = sorted(expected - union)
            extra = sorted(union - expected)
            raise ValueError(
                f"区域覆盖性检查失败：primitive 并集 {len(union)} != 全部单元 "
                f"{n_element}。未覆盖 {len(missing)} 个（如 {missing[:8]}）、"
                f"多出 {len(extra)} 个（如 {extra[:8]}）。"
            )

    # -------------------------------------------------- 构造：legacy 快照
    @classmethod
    def from_legacy(cls, globals_dict: Dict[str, Any]) -> "RegionMap":
        """从 legacy 模块命名空间快照直接拿 ``element_*`` 全局（过渡支点）。

        这是权威来源：legacy 运行时构造的集合（含对称单元合并、dict 迭代顺序等
        无法由 tag 表重建的部分）原样冻结。数值层搬家期间用这个，阶段 3 之后
        换 ``from_tags`` + 物理组。
        """
        regions = {}
        for name, value in globals_dict.items():
            if name.startswith("element_") and isinstance(value, (set, dict, list)):
                regions[name] = value
        return cls(regions)


# ---------------------------------------------------------------- MotorMesh
class MotorMesh:
    """gmsh 网格 + 区域映射。

    读取语义与 legacy 完全一致（``MMA10:227-230``）：
    ``unique_index_for_any_element=False`` + ``node[:, 1:] *= 1e-3``（mm→m）。

    Parameters
    ----------
    msh_path :
        .msh 路径（gmsh 4.1 格式，如 ``legacy_msh/model_mirror4.msh``）。
    tag_table :
        仅 ``regions="tags"`` 或 auto 回退路径使用；默认 ``DEFAULT_TAG_TABLE``
        （model_mirror4），换网格时传自定义 tag 表。
    regions :
        ``"auto"``（默认）：网格携带 ``$PhysicalNames`` → 由**物理组**划分区域
        （阶段 3），否则回退 tag 表（阶段 2）。``"tags"``：强制 tag 表路径。
    role_map / slot_group_names :
        物理组路径的 SynRM 角色映射（默认 ``SYNRM_ROLE_TO_GROUP`` /
        ``SYNRM_SLOT_GROUP_NAMES``；换电机/换模型时传对应表）。给 ``layout``
        且未显式传 ``slot_group_names`` 时，后者按 ``winding.slot{k}``
        （k = 0..``layout.n_coils_per_part``-1）推导。
    layout : WindingLayout, 可选
        相带参数化入口（换极槽/换电机）。``None``（默认）= legacy 18 槽硬编码
        相带（golden 路径，逐位不变）；给出时相带由 star-of-slots 推导。
    check_partition :
        默认 True：构造后跑覆盖性检查（三道闸之三）。
    expect_fingerprint :
        可选 .msh 期望 sha256（三道闸之二，resume 校验用）。
    """

    def __init__(self, msh_path, tag_table: Dict[str, Tuple[int, ...]] = None,
                 *,
                 regions: str = "auto",
                 role_map: Optional[Dict[str, str]] = None,
                 slot_group_names: Optional[Tuple[str, ...]] = None,
                 layout=None,
                 check_partition: bool = True,
                 expect_fingerprint: Optional[str] = None):
        self.msh_path = Path(msh_path)
        if expect_fingerprint is not None:
            verify_fingerprint(self.msh_path, expect_fingerprint,
                               model_name=self.msh_path.name)
        self._node, self._element_P, self._element_L, self._element_S, self._element_V = \
            gmsh_reader(filename=str(self.msh_path),
                        index_begins_with_zero=True,
                        unique_index_for_any_element=False)
        # mm → m（同 MMA10:230；坐标列 1:3）
        self._node[:, 1:] *= 1e-3
        self._physical_groups = None   # 网格携带物理组时的解析结果（None = 无）
        if regions == "auto":
            pg = read_physical_groups(str(self.msh_path))
            if pg.names:
                self._physical_groups = pg
                if slot_group_names is None:
                    if layout is not None:
                        slot_group_names = tuple(
                            f"winding.slot{k}"
                            for k in range(layout.n_coils_per_part))
                    else:
                        slot_group_names = SYNRM_SLOT_GROUP_NAMES
                self.regions = RegionMap.from_physical_groups(
                    self._element_S,
                    pg.face_groups(),
                    role_map=role_map if role_map is not None else SYNRM_ROLE_TO_GROUP,
                    slot_group_names=slot_group_names,
                    layout=layout,
                )
            else:
                self.regions = RegionMap.from_tags(self._element_S, tag_table,
                                                   layout=layout)
        elif regions == "tags":
            self.regions = RegionMap.from_tags(self._element_S, tag_table,
                                               layout=layout)
        else:
            raise ValueError(f"regions={regions!r}（应为 'auto' 或 'tags'）")
        if check_partition:
            self.regions.check_partition(len(self._element_S))

    # ---- 网格量 ----
    @property
    def node(self) -> np.ndarray:
        """(N_node, 4)：列 0=节点号，1:3=坐标（米），3=z（二维为 0）。"""
        return self._node

    @property
    def element_S(self) -> List[List[int]]:
        """面单元表：每行 [idx, n1, n2, n3, entity_tag]（idx 0 基、文件顺序）。"""
        return self._element_S

    @property
    def element_L(self) -> List[List[int]]:
        return self._element_L

    @property
    def element_P(self) -> List[List[int]]:
        return self._element_P

    @property
    def N_node(self) -> int:
        return self._node.shape[0]

    @property
    def N_element(self) -> int:
        return len(self._element_S)

    # ---- 区域 ----
    def region(self, name: str):
        """按 legacy 名取区域集合（委托 RegionMap）。"""
        return self.regions.region(name)

    def __repr__(self) -> str:
        return (f"MotorMesh({self.msh_path.name}: {self.N_node} nodes, "
                f"{self.N_element} elements, {len(self.regions.names())} regions)")
