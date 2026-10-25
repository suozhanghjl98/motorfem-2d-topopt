"""tests/mesh/test_mesh.py —— 阶段 2 网格与区域数据契约（motorfem/mesh）

验证核心：MotorMesh/RegionMap 与 legacy（golden L1 / freeze_msh sidecar）逐位一致。

**语义澄清（重要）**：golden L1 的 ``node``/``element_S`` 是 **MMA10 运行时**视图
——模块体在 MMA10:455-456 给气隙转子侧**追加了 236 个滑移层节点**
（``node = np.concatenate([node, _node_gap_rotor])``）并把 470 个 tag5 气隙单元的
节点索引重映射（MMA10:463）。MotorMesh 提供的是**原始网格契约**（reader 输出 +
mm 换算，4881 节点），滑移层是求解器运行时行为（内核 7 / 阶段 4）。因此：

* ``MotorMesh.node`` == golden ``node[:4881]``（扩充节点 append 在后，前段完整保留）
* ``element_S`` 的**单元 idx 列**不变；470 行节点索引差异精确落在 tag5（滑移层）

**F4 强验证**：RegionMap 区域集合的 ``list(set)`` 迭代顺序与 golden 的
``*__raw_order`` 逐位相同 —— from_tags 构造表达式逐字照抄 legacy 的直接证明。

**特殊区域**：``element_rotor_core_s`` 在 MMA10:300 被对称合并成 **dict**（本网格
无镜像重合单元，remove 未触发 ⇒ dict keys == tag3 全集且迭代顺序 == 原 set 顺序）；
``element_S_mech`` 等力学子网格表是 2D，不在 tag 驱动重建范围内。
"""
from pathlib import Path

import numpy as np
import pytest

from motorfem.mesh import DEFAULT_TAG_TABLE, MotorMesh, RegionMap

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: 候选网格路径：阶段 0 冻结在 legacy_msh/（旧 checkout 时用 legacy/pytorch 的）
_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


def _golden():
    path = _REPO_ROOT / "golden" / "L1_regions.npz"
    if not path.exists():
        pytest.skip("golden/L1_regions.npz 不存在 —— 基线冻结尚未入库")
    return np.load(path)


def _make_mesh() -> MotorMesh:
    return MotorMesh(_mesh_path())


# ---------------------------------------------------------------- 网格
def test_motormesh_matches_golden_original_grid():
    """原始网格契约：node == golden 前 4881 行（扩充的滑移层节点 append 在后）。"""
    mesh = _make_mesh()
    g = _golden()
    assert mesh.N_node == 4881
    assert g["node"].shape[0] == 5117  # 4881 + 236 滑移层节点
    assert np.array_equal(mesh.node, g["node"][:4881]), "原始 node 与 golden 前段不一致"


def test_element_S_idx_column_matches_golden():
    """element_S：单元 idx 列与 golden 一致；470 行节点索引差异精确落在 tag5。"""
    mesh = _make_mesh()
    g = _golden()
    s_raw, s_gold = np.asarray(mesh.element_S), g["element_S"]
    assert len(s_raw) == 9599
    assert np.array_equal(s_raw[:, 0], s_gold[:, 0]), "单元 idx 列不一致"
    diff = (s_raw != s_gold).any(axis=1)
    assert diff.sum() == 470
    assert set(s_gold[diff, 4]) == {5}, "节点索引差异应全部是 tag5（滑移层气隙单元）"


def test_motormesh_matches_freeze_msh_sidecar():
    """原始网格规模与 freeze_msh sidecar 对照（4881 节点 / 9599 面单元 / 26 tags）。"""
    mesh = _make_mesh()
    sidecar = _REPO_ROOT / "legacy_msh" / "model_mirror4.model.json"
    if not sidecar.exists():
        pytest.skip("legacy_msh/model_mirror4.model.json 不存在")
    import json
    stats = json.loads(sidecar.read_text(encoding="utf-8"))["mesh_stats"]
    assert mesh.N_node == stats["num_nodes"] == 4881
    assert mesh.N_element == stats["num_element_S"] == 9599
    assert len({e[4] for e in mesh.element_S}) == 26


def test_motormesh_coordinates_in_meters():
    """mm→m 换算：外径 72.5 mm 的节点应出现在 0.0725 m 处。"""
    mesh = _make_mesh()
    r2 = mesh.node[:, 1] ** 2 + mesh.node[:, 2] ** 2
    assert np.isclose(np.max(r2), 72.5e-3 ** 2, rtol=1e-9)


# ---------------------------------------------------------------- 区域
def test_region_sets_match_golden_iteration_order():
    """F4 强验证：区域集合的 list(set) 迭代顺序与 golden 的 *__raw_order 逐位相同。

    from_tags 的构造表达式逐字照抄 legacy（MMA10:255-283），因此同一集合的
    迭代顺序与 legacy 完全一致 —— 「搬家不引入 F4 风险」的直接证据。
    element_rotor_core_s 特殊：legacy 运行时是 dict（MMA10:300 对称合并），
    本网格 remove 未触发 ⇒ 迭代顺序与原 set 相同，用 golden 的 __keys 对照。
    """
    mesh = _make_mesh()
    g = _golden()
    for name in ("element_rotor_shaft", "element_rotor_core_m", "element_rotor_gap",
                 "element_stator_gap", "element_stator_core", "element_rotor_core",
                 "element_iron", "element_current", "element_phaseAp", "element_phaseAn",
                 "element_phaseBp", "element_phaseCn", "element_stator", "element_rotor"):
        region = mesh.region(name)
        assert set(region) == set(g[f"{name}__sorted"]), f"{name}: 内容不一致"
        assert np.array_equal(np.asarray(list(region)), g[f"{name}__raw_order"]), \
            f"{name}: list(set) 迭代顺序与 legacy 不一致（F4 风险）"

    # element_rotor_core_s：tag3 全集 3181，迭代顺序 == golden dict 的插入序（__keys）
    core_s = mesh.region("element_rotor_core_s")
    assert len(core_s) == len(g["element_rotor_core_s__keys"]) == 3181
    assert np.array_equal(np.asarray(list(core_s)), g["element_rotor_core_s__keys"])


def test_coil_regions_match_golden():
    """element_stator_coil（list of sets）的每个槽与 golden 一致。"""
    mesh = _make_mesh()
    g = _golden()
    coil = mesh.region("element_stator_coil")
    assert isinstance(coil, list) and len(coil) == 18
    for i, s in enumerate(coil):
        assert set(s) == set(g[f"element_stator_coil__raw_order_{i:02d}__sorted"])


def test_phase_belts_cover_current_exactly():
    """六个相带互斥且穷尽 element_current（=144，内核 8 探针的断言）。"""
    mesh = _make_mesh()
    belts = [mesh.region(f"element_phase{p}") for p in ("Ap", "An", "Bp", "Bn", "Cp", "Cn")]
    current = mesh.region("element_current")
    assert len(current) == 144
    for i in range(len(belts)):
        for j in range(i + 1, len(belts)):
            assert not (belts[i] & belts[j]), f"相带 {i}/{j} 有交集"
    assert set().union(*belts) == current


def test_region_unknown_name_raises():
    mesh = _make_mesh()
    with pytest.raises(KeyError):
        mesh.region("element_no_such_region")
    assert "element_rotor_shaft" in mesh.regions


def test_regionmap_from_legacy_transitional():
    """过渡支点：from_legacy 从命名空间快照拿 element_*，与 golden 一致。

    只对 1D 数组重建 set；2D 表（element_S_mech/element_L_mech 等力学子网格）
    不参与 —— 它们是区域表不是集合。
    """
    g = _golden()
    snap = {}
    for k in g.files:
        if k.startswith("element_") and k.endswith("__raw_order") and g[k].ndim == 1:
            snap[k[: -len("__raw_order")]] = set(g[k].tolist())
    # dict 区域（element_rotor_core_s）也进快照
    if "element_rotor_core_s" not in snap:
        snap["element_rotor_core_s"] = set(g["element_rotor_core_s__keys"].tolist())
    rm = RegionMap.from_legacy(snap)
    assert set(rm.region("element_rotor_shaft")) == set(g["element_rotor_shaft__sorted"])
    assert set(rm.region("element_rotor_core_s")) == set(g["element_rotor_core_s__keys"])
    assert "element_stator_coil" not in rm.names()  # list-of-sets 不在 1D __raw_order 快照里


def test_tag_table_matches_sidecar():
    """DEFAULT_TAG_TABLE 覆盖的 tag 集 == freeze_msh sidecar 的 dim2 直方图 tag 集。"""
    sidecar = _REPO_ROOT / "legacy_msh" / "model_mirror4.model.json"
    if not sidecar.exists():
        pytest.skip("legacy_msh/model_mirror4.model.json 不存在")
    import json
    h2 = json.loads(sidecar.read_text(encoding="utf-8"))["entity_tag_histograms"]["dim2"]
    covered = set()
    for ents in DEFAULT_TAG_TABLE.values():
        covered.update(ents)
    assert covered == {int(k) for k in h2}
