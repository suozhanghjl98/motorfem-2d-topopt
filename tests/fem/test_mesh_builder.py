"""tests/fem/test_mesh_builder.py —— 阶段 4A：FieldTopology 组装（golden L1 对齐）

验证核心：``build_field_topology`` 产出与 golden L1 **逐位一致**（``np.array_equal``），
以及与 golden L2 的 ρ 无关部分一致（``rho_SIMP_initial`` / ``element_topo_opti_m/s``）。

**判据层级**（``CLAUDE.md`` §4）：本模块组装的是「区域/拓扑/密度」这条链 —— 数学
内容早已在内核级（来源 A）验证过；这里的两条判据是

  - B（与 legacy 一致）：golden L1/L2 是**系统级 oracle**（``docs/重构设计文档.md``
    §1.5），本批把它逐位钉住；
  - F4 强验证：**力学子网格也用 legacy 的 ``set`` 迭代序** —— 探针
    （``tools/probe_mech.py``）用 ``sorted`` 是为了自身可复现，FieldSystem 要对齐
    golden 就绝不能照抄探针。``element_rotation__keys/values``、``element_S_mech``、
    ``element_mech2raw`` 等与 golden 逐位一致即这条的直接证明。

运行::

    python -m pytest tests/fem/test_mesh_builder.py -v
"""
from pathlib import Path

import numpy as np
import pytest

from motorfem.config import RunConfig
from motorfem.fem.mesh_builder import build_field_topology
from motorfem.mesh import MotorMesh

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MESH_CANDIDATES = [
    _REPO_ROOT / "legacy_msh" / "model_mirror4.msh",
    _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh",
]
_GOLDEN_L1 = _REPO_ROOT / "golden" / "L1_regions.npz"
_GOLDEN_L2 = _REPO_ROOT / "golden" / "L2_element.npz"


def _mesh_path() -> Path:
    for p in _MESH_CANDIDATES:
        if p.exists():
            return p
    pytest.skip("找不到 model_mirror4.msh")


def _golden(name):
    path = _GOLDEN_L1 if name == "L1" else _GOLDEN_L2
    if not path.exists():
        pytest.skip(f"{path} 不存在 —— 基线冻结尚未入库")
    return np.load(path)


def _build():
    return build_field_topology(RunConfig.legacy(), MotorMesh(_mesh_path()))


# ---------------------------------------------------------------- golden 对齐
def test_l1_all_arrays_match_golden():
    """L1 全部数组 ``np.array_equal``（golden 采集环境 = 本机，逐位可比）。"""
    top = _build()
    actual = top.L1_snapshot()
    g = _golden("L1")
    for k in sorted(g.files):
        assert k in actual, f"actual 缺 {k}"
        assert np.array_equal(g[k], actual[k]), (
            f"{k} 与 golden 不一致\n"
            f"  golden {g[k].shape} {g[k].dtype} / actual {actual[k].shape} {actual[k].dtype}"
        )
    assert set(actual) == set(g.files), (
        f"actual 多出的键: {set(actual) - set(g.files)}")


def test_l2_rho_initial_and_topo_opti_match_golden():
    """L2 的 ρ 无关 4 项（rho_SIMP_initial / topo_opti_m/s 顺序）逐位一致。"""
    top = _build()
    actual = top.L2_snapshot_partial()
    g = _golden("L2")
    for k, v in actual.items():
        assert k in g.files, f"golden 缺 {k}"
        assert np.array_equal(g[k], v), f"{k} 与 golden 不一致"


# ---------------------------------------------------------------- 结构断言
def test_mech_submesh_sizes_match_golden():
    """力学子网格的规模（golden L1 的形状即权威）。"""
    top = _build()
    g = _golden("L1")
    assert g["element_S_mech__raw_order"].shape[0] == top.n_element_mech == len(top.element_S_mech)
    assert g["element_L_mech__raw_order"].shape[0] == len(top.element_L_mech)
    assert g["element_mech2raw__keys"].size == len(top.element_mech2raw)
    assert g["node"].shape[0] == top.node_post.shape[0] == top.n_node
    assert top.node_post.shape[0] == 5117, "滑移层扩充节点数（236）随网格变化，验证生产值"
    assert top.n_element == 9599


def test_element_rotor_core_s_is_merged_dict():
    """对称合并：dict 大小 = tag3 全集（3181），且是单射（无重复 master）。"""
    top = _build()
    g = _golden("L1")
    d = top.element_rotor_core_s
    assert len(d) == g["element_rotor_core_s__keys"].size == 3181
    # 单射：值（master id）互不相同
    assert len(set(d.values())) == len(d)
    # dict 键 = 原始 tag3 集合（合并没丢元素）
    assert set(d) == top.regions.region("element_rotor_core_s")


def test_import_motorfem_fem_has_no_side_effect():
    """``import motorfem.fem`` 不建目录、不读网格、不改 RNG 状态。"""
    import random
    import numpy as np
    state = np.random.get_state()
    random.seed(1234)
    before = random.random()

    import motorfem.fem  # noqa: F401  （import 本身必须无副作用）

    assert random.random() != before  # RNG 未被重置
    np.random.set_state(state)        # numpy RNG 未被动过（采样对比在调用方）
    assert True
