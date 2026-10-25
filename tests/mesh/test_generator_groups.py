"""tests/mesh/test_generator_groups.py —— 阶段 3：生成器物理组块的静态钉住

``gmsh_motor_rotor_mirror.py`` / ``gmsh_4p6s_PMSM.py`` 是**自包含模块级脚本**
（import 即跑 gmsh + fltk 弹窗），不能 import motorfem ⇒ 它们的物理组块用
**字面 dict** 复刻 solver 侧常量表。这两条测试钉住两边 name/(dim, tag) 不漂移。

elementary 成员（字面 dict 的第三个元素）是实体表达式（``[surface_rotor1]`` /
``line_slots``），只做存在性检查、不做机械校验 —— 它们的运行时正确性等真重生成
网格后核对（设计文档 §阶段 3 的偏差登记）。

⚠ 不跑 gmsh：只 ``ast.parse`` 源文件 + 字符串匹配，无 GUI、无副作用。
"""
from pathlib import Path

import pytest

from motorfem.mesh.physical_groups import (
    PMSM_BOUNDARY_GROUPS,
    PMSM_FACE_GROUPS,
    SYNRM_BOUNDARY_GROUPS,
    SYNRM_FACE_GROUPS,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _norm(src: str) -> str:
    """去掉所有空白，让字面 dict 的子串匹配对缩进不敏感。"""
    return "".join(src.split())


def test_synrm_generator_parses_and_matches_constant_tables():
    """SynRM 生成器：语法合法 + _FACE_GROUPS/_BND_GROUPS 与常量表逐项一致。

    生成器脚本是自包含的（不能 import motorfem），物理组块里的字面 dict 是
    solver 侧常量表的复刻 —— 这张测试钉住两边不漂移。elementary 成员（第三个
    元素）是实体表达式，只做存在性检查，不做机械校验（等真重生成后核对）。
    """
    src_path = _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "gmsh_motor_rotor_mirror.py"
    if not src_path.exists():
        pytest.skip("legacy/pytorch/my_gmsh/gmsh_motor_rotor_mirror.py 不存在")
    src = src_path.read_text(encoding="utf-8")
    import ast

    ast.parse(src)  # 语法合法（我的改动没把脚本改坏）
    n = _norm(src)
    for g in SYNRM_FACE_GROUPS:
        if g.name.startswith("winding.slot"):
            # 源文件是 f-string：{k} 与 `9 + k` 都以字面形式出现
            assert 'f"winding.slot{k}":(2,9+k,[' in n, f"生成器缺 {g.name}"
        else:
            assert f'"{g.name}":(2,{g.tag},[' in n, f"生成器缺 {g.name}"
    for g in SYNRM_BOUNDARY_GROUPS:
        # 第三元可能是 [显式列表]（周期主从等）或裸列表变量（line_slots），
        # 只钉 name/(dim, tag)，elementary 成员等真重生成后核对。
        assert f'"{g.name}":(1,{g.tag},' in n, f"生成器缺 {g.name}"


def test_pmsm_generator_parses_and_matches_constant_tables():
    """PMSM 生成器：同款静态钉住（winding.layer{k} / rotor.pm.{k} 用 f-string 展开）。"""
    src_path = _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "gmsh_4p6s_PMSM.py"
    if not src_path.exists():
        pytest.skip("legacy/pytorch/my_gmsh/gmsh_4p6s_PMSM.py 不存在")
    src = src_path.read_text(encoding="utf-8")
    import ast

    ast.parse(src)
    n = _norm(src)
    for g in PMSM_FACE_GROUPS:
        if g.name.startswith("winding.layer"):
            k = int(g.name.rsplit("layer", 1)[1])
            assert g.tag == 14 + k, g.name
            assert 'f"winding.layer{k}":(2,14+k,[' in n, f"生成器缺 {g.name}"
        elif g.name.startswith("rotor.pm."):
            k = int(g.name.rsplit(".", 1)[1])
            assert g.tag == 20 + k, g.name
            assert 'f"rotor.pm.{k}":(2,20+k,[' in n, f"生成器缺 {g.name}"
        else:
            assert f'"{g.name}":(2,{g.tag},[' in n, f"生成器缺 {g.name}"
    for g in PMSM_BOUNDARY_GROUPS:
        assert f'"{g.name}":(1,{g.tag},' in n, f"生成器缺 {g.name}"
