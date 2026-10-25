"""motorfem.mesh.physical —— 阶段 3：v4.1 .msh 的物理组解析（$PhysicalNames + $Entities）

冻结网格（``legacy_msh/``，gmsh 4.1）**没有** ``$PhysicalNames``；本解析器是
**新增能力**：生成器改造后重生成的网格会携带物理组名，求解器从这里读
「elementary entity → 物理组」而不是靠硬编码的 tag 表（阶段 3 断掉「tag 恰好
对齐」）。

v4.1 格式（已用 gmsh 4.13.1 真实输出逐行核对，ASCII）：:

    $PhysicalNames
    numPhysicalNames
    dim physicalTag "name"
    ...
    $EndPhysicalNames

    $Entities
    numPoints numCurves numSurfaces numVolumes
    <dim0> tag x y z [numPhys physTags...]              # 点：min==max 折叠成单组坐标
    <dim1> tag minX..maxZ [numPhys physTags...] numBoundingPoints pointTags...
    <dim2> tag minX..maxZ [numPhys physTags...] numCurvesInLoop curveTags...
    <dim3> tag minX..maxZ [numPhys physTags...] numTunneledEntities tags...
    $EndEntities

关键：``$Elements`` 块仍按 **elementary entity** 分组（``gmsh_reader`` 读的
``entityTag`` 列不变），物理组归属只写在 ``$Entities``。所以
``element → 物理组`` 的解析路径是：元素块头 elementary tag → ``$Entities`` 查该
实体的物理组 tag → ``$PhysicalNames`` 查名。

缺段（冻结网格）返回空映射，调用方据此回退 tag 表路径。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

__all__ = ["PhysicalGroups", "read_physical_groups"]


@dataclass(frozen=True)
class PhysicalGroups:
    """一个 .msh 的物理组信息（解析自 $PhysicalNames + $Entities）。

    Attributes
    ----------
    names : dict[(dim, physical_tag) → name]
        ``$PhysicalNames`` 的完整内容。
    entities : dict[dim → {elementary_entity_tag → (physical_tags, ...)}]
        ``$Entities`` 中带物理组的实体（无物理组的实体不出现）。
    """

    names: Dict[Tuple[int, int], str] = field(default_factory=dict)
    entities: Dict[int, Dict[int, Tuple[int, ...]]] = field(default_factory=dict)

    def group_names(self, dim: int) -> Tuple[str, ...]:
        """该维度下所有物理组名（按名字典序，确定性）。"""
        return tuple(
            sorted(name for (d, _t), name in self.names.items() if d == dim)
        )

    def face_groups(self) -> Dict[str, Tuple[int, ...]]:
        """dim-2 面组：组名 → 其 elementary entity tag 集合（RegionMap 用它还原旧名）。

        未入任何面组的 elementary 面不在任何组里 —— 覆盖性检查会抓到它们。
        """
        out: Dict[str, List[int]] = {}
        for entity_tag, phys_tags in self.entities.get(2, {}).items():
            for ptag in phys_tags:
                name = self.names.get((2, ptag))
                if name is None:
                    continue  # $Entities 引用了 $PhysicalNames 没有的 tag：忽略并交由调用方
                out.setdefault(name, []).append(entity_tag)
        return {name: tuple(sorted(tags)) for name, tags in out.items()}


def read_physical_groups(msh_path) -> PhysicalGroups:
    """解析 .msh 的 ``$PhysicalNames`` / ``$Entities`` 段。

    段缺失（如冻结网格无 ``$PhysicalNames``）返回空映射，不报错。
    ``$PhysicalNames`` 存在但 ``$Entities`` 缺失（或反之）同样容忍。
    """
    with open(msh_path, "r", encoding="utf-8") as f:
        text = f.read()
    names = _parse_physical_names(text)
    entities = _parse_entities(text)
    return PhysicalGroups(names=names, entities=entities)


# ---------------------------------------------------------------- 实现
def _sections(text: str) -> Dict[str, List[str]]:
    """把 .msh 切成段落：``$NAME`` … ``$EndNAME`` → 行列表（不含头尾标记）。"""
    out: Dict[str, List[str]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("$") and not line.startswith("$End"):
            tag = line[1:]
            j = i + 1
            body = []
            while j < len(lines) and lines[j].strip() != f"$End{tag}":
                body.append(lines[j])
                j += 1
            out[tag] = body
            i = j + 1
        else:
            i += 1
    return out


def _parse_physical_names(text: str) -> Dict[Tuple[int, int], str]:
    body = _sections(text).get("PhysicalNames")
    if not body:
        return {}
    names: Dict[Tuple[int, int], str] = {}
    for line in body[1:]:  # 首行是计数
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        dim, tag = int(parts[0]), int(parts[1])
        name = parts[2].strip().strip('"')
        names[(dim, tag)] = name
    return names


def _parse_entities(text: str) -> Dict[int, Dict[int, Tuple[int, ...]]]:
    body = _sections(text).get("Entities")
    if not body:
        return {}
    # 首行：numPoints numCurves numSurfaces numVolumes（顺序即 dim 0..3）
    counts = [int(x) for x in body[0].split()]
    entities: Dict[int, Dict[int, Tuple[int, ...]]] = {}
    line_no = 1
    for dim, n_ent in enumerate(counts):
        m: Dict[int, Tuple[int, ...]] = {}
        # 点线面的字段起点不同（点：tag + 折叠坐标 + numPhys；线/面/体：tag + 6 个 bbox 数）
        idx_phys = 4 if dim == 0 else 7
        for _ in range(n_ent):
            fields = body[line_no].split()
            line_no += 1
            if not fields:
                continue
            entity_tag = int(fields[0])
            num_phys = int(fields[idx_phys])
            if num_phys > 0:
                m[entity_tag] = tuple(
                    int(x) for x in fields[idx_phys + 1: idx_phys + 1 + num_phys]
                )
        entities[dim] = m
    return entities
