"""tests/mesh/test_physical.py —— 阶段 3：区域命名契约（物理组 + 三道闸）

验证核心（对应设计文档 §阶段 3）：

1. **v4.1 解析器**（``$PhysicalNames`` + ``$Entities``）在真实 gmsh 物理组网格
   夹具（``tests/mesh/fixtures/tiny_physical.msh``，由同目录 ``make_tiny_physical.py``
   生成）上读出名与成员 —— 冻结网格（无 ``$PhysicalNames``）读为空。
2. **过渡支点硬门槛**：``RegionMap.from_physical_groups``（用常量表
   ``SYNRM_FACE_GROUPS`` 合成的成员）在冻结网格上与 ``from_tags`` **全部旧名
   元素一致 + F4 迭代顺序逐位相同**。
3. **三道闸**：``require_regions`` 缺名 raise 并列出实际组；``verify_fingerprint``
   指纹不符 raise；``RegionMap.check_partition`` 通过冻结网格、对「多出未入表 tag」
   的篡改网格 raise。
4. **常量表一致性**：``SYNRM_FACE_GROUPS`` 的 elementary 成员与
   ``DEFAULT_TAG_TABLE`` 逐值一致（防两边漂移）。

⚠ golden L1 的 ``*__raw_order`` 是 **MMA10 运行时**视图（滑移层 236 节点已追加），
本测试只用冻结网格原始契约（同 ``test_mesh.py``）。
"""
from pathlib import Path

import numpy as np
import pytest

from motorfem.mesh import (
    DEFAULT_TAG_TABLE,
    MotorMesh,
    RegionMap,
    read_physical_groups,
    require_regions,
    verify_fingerprint,
)
from motorfem.mesh.physical_groups import (
    SYNRM_FACE_GROUPS,
    SYNRM_ROLE_TO_GROUP,
    SYNRM_SLOT_GROUP_NAMES,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests" / "mesh" / "fixtures" / "tiny_physical.msh"
_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


# ---------------------------------------------------------------- 解析器
def test_reader_parses_real_physical_group_file():
    """v4.1 $PhysicalNames + $Entities 解析：真实 gmsh 物理组网格的名与成员。"""
    pg = read_physical_groups(_FIXTURE)
    assert pg.names == {
        (1, 101): "bnd.outer",
        (2, 10): "region.alpha",
        (2, 20): "region.beta",
    }
    # dim-2 面：elementary 面 1 → alpha、面 2 → beta
    assert pg.entities[2] == {1: (10,), 2: (20,)}
    assert pg.face_groups() == {"region.alpha": (1,), "region.beta": (2,)}
    # dim-1 边界：四条外边线都属于 bnd.outer
    assert pg.entities[1] == {1: (101,), 2: (101,), 3: (101,), 4: (101,)}
    assert pg.group_names(1) == ("bnd.outer",)


def test_reader_returns_empty_for_frozen_mesh():
    """冻结网格没有 $PhysicalNames ⇒ 无物理组名（调用方回退 tag 表路径）。

    冻结网格有 ``$Entities`` 但没有物理组归属 —— 各维的成员映射都为空。
    """
    pg = read_physical_groups(_mesh_path())
    assert pg.names == {}
    assert all(not members for members in pg.entities.values())
    assert pg.face_groups() == {}


# ---------------------------------------------------------------- 过渡支点
def _synrm_group_table() -> dict:
    """由常量表合成「组名 → elementary tag」表（生成器按此接线）。"""
    return {g.name: g.entity_tags for g in SYNRM_FACE_GROUPS}


def test_from_physical_groups_equals_from_tags_bitwise():
    """过渡支点硬门槛：物理组构造 == tag 表构造（元素一致 + F4 迭代顺序逐位）。"""
    mesh = MotorMesh(_mesh_path())
    rm = RegionMap.from_physical_groups(
        mesh.element_S, _synrm_group_table(),
        role_map=SYNRM_ROLE_TO_GROUP, slot_group_names=SYNRM_SLOT_GROUP_NAMES,
    )
    assert rm.names() == mesh.regions.names()
    for name in rm.names():
        a, b = mesh.regions.region(name), rm.region(name)
        if isinstance(a, list):       # element_stator_coil：list of sets，逐槽比迭代序
            assert [list(x) for x in a] == [list(x) for x in b], f"{name}: 槽迭代顺序不同"
        else:
            assert list(a) == list(b), f"{name}: list(set) 迭代顺序不同（F4 风险）"


def test_from_physical_groups_missing_role_raises():
    """缺一个 legacy 角色所需的面组名 ⇒ raise 并列出实际面组。"""
    mesh = MotorMesh(_mesh_path())
    table = dict(_synrm_group_table())
    del table["stator.gap"]
    with pytest.raises(KeyError, match="stator.gap"):
        RegionMap.from_physical_groups(
            mesh.element_S, table,
            role_map=SYNRM_ROLE_TO_GROUP, slot_group_names=SYNRM_SLOT_GROUP_NAMES,
        )


def test_from_physical_groups_incomplete_role_map_raises():
    """role_map 缺 legacy 键 ⇒ 明确报错（而不是 from_tags 深处 KeyError）。"""
    mesh = MotorMesh(_mesh_path())
    with pytest.raises(KeyError, match="role_map 缺 legacy 键"):
        RegionMap.from_physical_groups(
            mesh.element_S, _synrm_group_table(),
            role_map={"element_rotor_core_m": "rotor.core.main"},
            slot_group_names=SYNRM_SLOT_GROUP_NAMES,
        )


def test_from_physical_groups_wrong_slot_count_raises():
    """槽组数 != 18 ⇒ 明确报错（from_tags 的相带合成按 legacy 表写死 18 槽）。"""
    mesh = MotorMesh(_mesh_path())
    with pytest.raises(ValueError, match="槽组数"):
        RegionMap.from_physical_groups(
            mesh.element_S, _synrm_group_table(),
            role_map=SYNRM_ROLE_TO_GROUP, slot_group_names=("winding.slot0",),
        )


# ---------------------------------------------------------------- 常量表
def test_synrm_face_groups_match_default_tag_table():
    """SYNRM_FACE_GROUPS 的 elementary 成员 == DEFAULT_TAG_TABLE 的 tag 分组。"""
    role_to_key = {v: k for k, v in SYNRM_ROLE_TO_GROUP.items()}
    for g in SYNRM_FACE_GROUPS:
        if g.name in role_to_key:
            assert g.entity_tags == DEFAULT_TAG_TABLE[role_to_key[g.name]], g.name
        elif g.name.startswith("winding.slot"):
            k = int(g.name.rsplit("slot", 1)[1])
            assert g.entity_tags == (9 + k,), g.name
    # 面组物理组 tag：rotor.gap 取 min(4,5)=4、stator.gap 取 min(6,7)=6，
    # 5/7 空缺 —— 物理组 tag 只需显式、稳定（设计决策），不必占满 1..26。
    tags = {g.tag for g in SYNRM_FACE_GROUPS}
    assert tags == set(range(1, 27)) - {5, 7}
    assert len(SYNRM_FACE_GROUPS) == 24  # 6 核心 + 18 槽


# ---------------------------------------------------------------- 三道闸
def test_require_regions_missing_raises():
    """闸①：驱动声明缺一个必需区域 ⇒ raise 并列出实际拥有的组。"""
    mesh = MotorMesh(_mesh_path())
    with pytest.raises(KeyError) as ei:
        require_regions(mesh.regions,
                        ["element_rotor_shaft", "element_no_such"], who="test-driver")
    msg = str(ei.value)
    assert "element_no_such" in msg
    assert "element_rotor_shaft" in msg          # 列出实际拥有的组
    # 全部满足时静默通过
    require_regions(mesh.regions, ["element_rotor_shaft", "element_phaseAp"], who="t")


def test_verify_fingerprint_match_and_mismatch():
    """闸②：.msh 实算 sha256 与期望不符 ⇒ raise；相符静默。"""
    import hashlib

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    path = _mesh_path()
    verify_fingerprint(path, sha(path))          # 相符：静默
    with pytest.raises(ValueError, match="网格指纹不匹配"):
        verify_fingerprint(path, "0" * 64)


def test_check_partition_passes_on_frozen_mesh():
    """闸③：冻结网格的 primitive 区域两两不相交、并集 == 全部单元。"""
    mesh = MotorMesh(_mesh_path())
    mesh.regions.check_partition(mesh.N_element)   # 不抛即过
    # 数值自洽：各 primitive 单元数之和 == 9599
    total = sum(len(s) for _, s in mesh.regions._primitives)
    assert total == 9599


def test_check_partition_detects_uncovered_elements():
    """闸③：多出一个没进 tag 表的 elementary tag ⇒ 覆盖性检查 raise（F1 探测器）。"""
    # 3 个单元：tag 1、1、99（99 不在 DEFAULT_TAG_TABLE）
    element_S = [[0, 0, 1, 2, 1], [1, 0, 1, 2, 1], [2, 0, 1, 2, 99]]
    rm = RegionMap.from_tags(element_S)
    with pytest.raises(ValueError, match="未覆盖"):
        rm.check_partition(3)


def test_check_partition_detects_overlap():
    """闸③：构造一个重叠的 RegionMap ⇒ 覆盖性检查 raise 不互斥。"""
    rm = RegionMap({"a": {0, 1}, "b": {1, 2}}, primitives=(("a", {0, 1}), ("b", {1, 2})))
    with pytest.raises(ValueError, match="不互斥"):
        rm.check_partition(3)


# ---------------------------------------------------------------- MotorMesh
def test_motormesh_auto_falls_back_to_tags_without_physical_names():
    """冻结网格无 $PhysicalNames ⇒ auto 走 from_tags，区域与显式 tags 路径一致。"""
    mesh_auto = MotorMesh(_mesh_path())
    mesh_tags = MotorMesh(_mesh_path(), regions="tags")
    assert mesh_auto.regions.names() == mesh_tags.regions.names()
    for name in mesh_auto.regions.names():
        assert list(mesh_auto.regions.region(name)) == list(mesh_tags.regions.region(name))


def test_motormesh_auto_uses_physical_groups_when_present():
    """带 $PhysicalNames 的网格 ⇒ auto 走物理组路径（缺 SynRM 角色名 ⇒ 报缺名）。"""
    # 夹具不是 SynRM 电机：默认 SynRM 角色映射必然缺名 —— 这正是闸①该报的错
    with pytest.raises(KeyError, match="物理组缺少 legacy 角色所需的名"):
        MotorMesh(_FIXTURE)
    # 强制 tag 表路径不受物理组影响
    m = MotorMesh(_FIXTURE, regions="tags")
    assert m.N_element == 2
    assert m._physical_groups is None
    # 显式物理组信息可从 MotorMesh 解析到（供调用方查边界组等）
    pg = read_physical_groups(_FIXTURE)
    assert pg.group_names(1) == ("bnd.outer",)


def test_motormesh_expect_fingerprint():
    """MotorMesh 可选 expect_fingerprint：不符即 raise（resume 校验入口）。"""
    import hashlib

    path = _mesh_path()
    good = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    MotorMesh(path, expect_fingerprint=good)          # 相符：静默
    with pytest.raises(ValueError, match="网格指纹不匹配"):
        MotorMesh(path, expect_fingerprint="0" * 64)


# ---------------------------------------------------------------- sidecar
def test_freeze_msh_sidecar_has_physical_group_names():
    """freeze_msh sidecar 增补 physical_group_names 键（冻结网格 ⇒ 空列表）。"""
    sidecar = _REPO_ROOT / "legacy_msh" / "model_mirror4.model.json"
    if not sidecar.exists():
        pytest.skip("legacy_msh/model_mirror4.model.json 不存在")
    import json

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "physical_group_names" in data
    assert data["physical_group_names"] == []       # 冻结网格无 $PhysicalNames
