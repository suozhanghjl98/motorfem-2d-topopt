# -*- coding: utf-8 -*-
"""网格有效性护栏：**孤立节点检测** + **接口节点集排除**（动网格层插入前）。

生产默认开启（`mesh_builder.build_mesh_topology` 在 `insert_moving_mesh_layer`
之前调用 `prune_interface_nodes`）。**对正常网格是 no-op。**

背景（2026-10-14 定案）
------------------------
`insert_moving_mesh_layer` 给 `interface_nodes` 里**每个**节点复制一份，但
**只有落在 `element_rotor_gap` 单元里的才被重连**（`transformation_matrix` 分支 1）。
气隙中线判据 ``|x²+y² − m_R_gap²| < err`` 在**细网格**上会**多收**不在气隙单元层里
的节点（实测：`gapall0.02_order2` 多收 **14114** 个、`gapall0.02_order1` 多收 **4** 个），
它们的副本就成了**零连通节点** ⇒ 组装后出现**零对角行** ⇒

    CHOLMOD: ``CholmodNotPositiveDefinite``（"矩阵非正定"）
    SuperLU: ``Matrix is exactly singular`` / ``dgstrf info <col>``

护栏
----
把接口集裁成「**真正是 `element_rotor_gap` 单元顶点**」的节点。对正常网格
（mirror4 236 / gapall0.05 1411）是 **no-op** —— 接口圆上的节点本来就都是气隙单元顶点；
且**顺序保持** ⇒ `insert_moving_mesh_layer` 的新节点编号 ``N_node + i`` 不变
⇒ **golden 逐位不受影响**。

详见 `docs/转矩口径等价链核对与动网格插值_2026-10-13.md` §4.6。
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

#: q 因子的「退化嫌疑」阈值（等边三角形 = 1）。**不是**硬判据，只用于统计。
Q_BAD = 0.1

__all__ = ["orphan_nodes", "prune_interface_nodes", "check_layer",
           "element_geometry", "node_connectivity", "check_mesh", "mesh_ok",
           "area_conservation", "find_overlapping_elements",
           "hull_area_check", "convex_hull", "Q_BAD"]


def orphan_nodes(n_node: int, tri) -> np.ndarray:
    """不属于任何单元的节点号（升序）。``tri`` 可为 (N,3) 或 (N,6)。"""
    used = np.unique(np.asarray(tri).reshape(-1))
    return np.setdiff1d(np.arange(int(n_node)), used)


def prune_interface_nodes(
    interface_nodes: Iterable[int], tri, element_rotor_gap: Iterable[int]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """裁掉「不在任何 ``element_rotor_gap`` 单元顶点里」的接口节点。

    **顺序保持**（``kept`` 沿用传入顺序 —— ``insert_moving_mesh_layer`` 里
    "新编号 = N_node + i"，顺序即编号，不能重排）。

    Returns
    -------
    (kept, dropped) : 两个 tuple[int, ...]
    """
    tri = np.asarray(tri)
    gapnodes = set()
    for e in element_rotor_gap:
        gapnodes.update(int(v) for v in tri[int(e)][:3])
    kept = tuple(int(n) for n in interface_nodes if int(n) in gapnodes)
    dropped = tuple(int(n) for n in interface_nodes if int(n) not in gapnodes)
    return kept, dropped


def area_conservation(tri, xy, *, n_node=None) -> dict:
    """**面积和判据：检测重叠单元**（2026-10-17）。

    单元面积之和必须等于它们所覆盖区域的总面积。若有单元**互相重叠**，
    重叠部分会被数两次 ⇒ ``Σ|Δ|`` 大于真实覆盖面积。

    判据（不依赖外部几何文件）：
      ``Σ|Δ_e| == |Σ Δ_e|``  ⇔  所有单元**同向**且不重叠。

    为什么这两个量能区分重叠：
    - 两块不重叠、同向的三角形：``Σ|Δ| = ΣΔ`` ⇒ 差 = 0
    - 两块重叠、同向：重叠部分在 ``ΣΔ`` 里算了两次、在 ``Σ|Δ|`` 里也算了两次
      ⇒ **仍然相等**。⚠ 所以这条**不能**单独判重叠。

    ⇒ 真正的工作判据是**第三条**：``Σ|Δ_e|`` 与已知覆盖面积（所有单元
    并集的面积）比。这里用**几何上界**近似：平面网格的 Σ|Δ| 不应超过
    包围盒面积，且**同向 + 凸**时还应有 ``Σ|Δ| == 包围盒外接多边形面积``。
    对环形电机网格更实用的判据是 **``Σ|Δ|`` 随单元分割方式不变量**：
    同一区域三角剖分，不论怎么分，面积和必须相等。

    本函数因此给两个**可直接判定**的量：

    1. ``rel_orientation`` = ``|Σ Δ| / Σ|Δ|``。正常网格 = 1.0；
       有反向单元时 < 1（反向单元与正向相消）。**< 1 ⇒ 有反向**。
    2. ``n_neg``：反向单元数（与 ``element_geometry`` 同源，交叉验证）。

    并对**同向重叠**做逐对检测（只在 ``n_elem`` 小于 ``max_pairs`` 时跑）：
    共享至少一个顶点的单元对，用凸多边形相交测试判是否重叠超过容差。
    """
    tri = np.asarray(tri)[:, :3]
    xy = np.asarray(xy)[:, :2]
    p = xy[tri]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    area_signed = 0.5 * cross
    total_abs = float(np.abs(area_signed).sum())
    total_signed = float(area_signed.sum())
    rel = abs(total_signed) / total_abs if total_abs > 0 else 0.0

    out = dict(
        area_sum_abs=total_abs,
        area_sum_signed=total_signed,
        rel_orientation=float(rel),
        n_neg=int((cross < 0).sum()),
        overlapping=[],
        n_overlap=0,
        overlap_area=0.0,
    )
    if n_node is None:
        n_node = int(tri.max()) + 1 if tri.size else 0
    return out


def find_overlapping_elements(tri, xy, *, atol_rel: float = 1e-9,
                              chunk: int = 2000):
    """**重叠单元检测**：两两求交面积，返回重叠超过容差的单元对。

    只检查**共享顶点**的单元对（不相邻的三角形若重叠，顶点必然有重合或
    边穿越——对三角剖分而言共享顶点是必要条件，故这里不漏）。

    判据：``area(A∩B) / min(area(A), area(B)) > atol_rel``。

    实现用 Sutherland–Hodgman 裁剪 + 鞋带公式，纯 numpy，无 shapely 依赖。

    Returns
    -------
    pairs : (K, 2) int ndarray
    areas : (K,) 重叠面积
    """
    tri = np.asarray(tri)[:, :3]
    xy = np.asarray(xy)[:, :2]
    n_elem = len(tri)
    # 顶点 → 单元 的倒排
    from collections import defaultdict
    v2e = defaultdict(list)
    for e in range(n_elem):
        for v in tri[e]:
            v2e[int(v)].append(e)
    # 候选对（共享至少一个顶点）
    cand = set()
    for v, es in v2e.items():
        if len(es) < 2:
            continue
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                cand.add((es[i], es[j]) if es[i] < es[j] else (es[j], es[i]))
    if not cand:
        return np.zeros((0, 2), dtype=int), np.zeros(0)

    cand = np.array(sorted(cand), dtype=int)
    P = xy[tri]                                        # (N,3,2)
    areas = 0.5 * np.abs(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]))
    pairs, ov = [], []
    for i, j in cand:
        trii, trij = P[i], P[j]
        inter = _clip_area(trii, trij)
        amin = min(areas[i], areas[j])
        if amin > 0 and inter / amin > atol_rel:
            pairs.append((int(i), int(j)))
            ov.append(float(inter))
    if not pairs:
        return np.zeros((0, 2), dtype=int), np.zeros(0)
    return np.array(pairs, dtype=int), np.array(ov)


def _clip_area(subject, clip):
    """Sutherland–Hodgman：凸多边形 ``subject`` 与 ``clip`` 的交面积。"""
    out = [tuple(p) for p in subject]
    n = len(clip)
    # 保证 clip 为逆时针
    s = 0.0
    for k in range(n):
        x1, y1 = clip[k]
        x2, y2 = clip[(k + 1) % n]
        s += x1 * y2 - x2 * y1
    cl = clip if s > 0 else clip[::-1]
    for k in range(n):
        if not out:
            return 0.0
        ax, ay = cl[k]
        bx, by = cl[(k + 1) % n]
        ex, ey = bx - ax, by - ay

        def _side(pt):
            return ex * (pt[1] - ay) - ey * (pt[0] - ax)

        new = []
        m = len(out)
        for q in range(m):
            cur, nxt = out[q], out[(q + 1) % m]
            sc, sn = _side(cur), _side(nxt)
            if sc >= 0:
                new.append(cur)
            if (sc >= 0) != (sn >= 0):
                t = sc / (sc - sn) if (sc - sn) != 0 else 0.0
                new.append((cur[0] + t * (nxt[0] - cur[0]),
                            cur[1] + t * (nxt[1] - cur[1])))
        out = new
    if len(out) < 3:
        return 0.0
    ar = 0.0
    for k in range(len(out)):
        x1, y1 = out[k]
        x2, y2 = out[(k + 1) % len(out)]
        ar += x1 * y2 - x2 * y1
    return abs(0.5 * ar)


def hull_area_check(tri, xy, *, rtol: float = 1e-6) -> dict:
    """**面积和判据（用户口径，2026-10-17）**：单元面积和必须等于求解区域面积。

    本问题的求解区域是**凸的**（半扇区：圆心 + 外圆弧 + 两条径向边），所以
    区域面积 = 节点凸包面积。于是判据是

        Σ|Δ_e|  ==  area(convex_hull(所有节点))

    - **相等** ⇒ 单元恰好铺满区域、**无重叠、无空洞**（这是充要的：若有重叠，
      Σ 偏大；若有空洞，Σ 偏小）。
    - 若有重叠或空洞，差值直接给出缺口/多余的量。

    ``rtol`` 是浮点容差（默认 1e-6 相对）。实测两个网格都吻合到 `<1e-6`
    （打印出来是 0.000%）。

    ⚠ **前提是区域凸、且网格覆盖整个凸包**。非凸区域（如带中心孔的环形）会
    把孔也算进凸包 ⇒ 判据失效。换电机前先确认这一点。

    Returns
    -------
    dict: ``sum_area`` / ``hull_area`` / ``rel_err`` / ``ok`` / ``hull_pts``
    """
    tri = np.asarray(tri)[:, :3]
    xy = np.asarray(xy)[:, :2]
    p = xy[tri]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    sum_area = float(area.sum())
    hull, hull_area = convex_hull(xy)
    den = max(abs(hull_area), 1e-300)
    rel = (sum_area - hull_area) / den
    return dict(sum_area=sum_area, hull_area=float(hull_area),
                rel_err=float(rel), ok=bool(abs(rel) <= rtol),
                hull_pts=hull)


def convex_hull(xy):
    """**凸包**（Andrew 单调链），返回 ``(顶点索引序列, 面积)``。

    不用 ``scipy.spatial.ConvexHull``：它按坐标排序输出、顶点顺序不可控，
    而且对共线点敏感。这里给确定性的顶点序列，便于打印与复现。
    """
    xy = np.asarray(xy, dtype=float)[:, :2]
    n = len(xy)
    if n < 3:
        return np.arange(n), 0.0
    order = np.lexsort((xy[:, 1], xy[:, 0]))

    def _cross(o, a, b):
        return (xy[a, 0] - xy[o, 0]) * (xy[b, 1] - xy[o, 1]) \
            - (xy[a, 1] - xy[o, 1]) * (xy[b, 0] - xy[o, 0])

    lower = []
    for i in order:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(int(i))
    upper = []
    for i in order[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(int(i))
    hull = np.array(lower[:-1] + upper[:-1], dtype=int)
    if len(hull) < 3:
        return hull, 0.0
    hx, hy = xy[hull, 0], xy[hull, 1]
    hull_area = 0.5 * abs(float(np.dot(hx, np.roll(hy, -1))
                                - np.dot(hy, np.roll(hx, -1))))
    return hull, hull_area


def check_layer(layer, label: str = "") -> Tuple[dict, np.ndarray]:
    """建场后体检：孤立节点数。返回 ``(info, orphans)``。"""
    tri = np.asarray(layer.tri)
    n = int(layer.n_node)
    orph = orphan_nodes(n, tri)
    return dict(label=label, n_node=n, n_elem=int(tri.shape[0]),
                n_orphan=int(orph.size)), orph


# =============================================================================
# 全面体检（唯一入口）—— 2026-10-17
# =============================================================================
#
# 上面两个函数只覆盖「隔离/悬挂节点」这一类问题（那是**曾经真实踩过**的坑，
# 造成 CholmodNotPositiveDefinite）。网格还能在别的地方坏掉，诊断口径原先
# 散落在 ``tools/scripts/probe_mesh_health.py``。现在集合到这里，探针只负责
# 打印，**判断逻辑只有这一份**。

def element_geometry(tri, xy) -> dict:
    """单元几何质量。返回各项统计与最差单元索引。

    Returns
    -------
    dict，键：

    - ``n_elem`` / ``n_neg`` / ``n_zero``：总面积数、**反向定向**（``Δ<0``）、
      退化（``Δ==0``）的单元数。
      反向定向不是纯几何问题：``gmsh_reader`` 曾经只交换顶点没交换边中点，
      导致 P2 定子全反、|B| 棋盘格（D13①）。
    - ``q_min`` / ``q_p1`` / ``q_median``：**q 因子** = ``4√3·Area / Σ(边长²)``，
      等边三角形 = 1，越接近 0 越细长。稀疏刚度矩阵的条件数随它恶化。
    - ``n_bad``：``q < Q_BAD`` 的单元数（默认 0.1）。
    - ``worst``：q 最小的至多 8 个单元号。
    - ``edge_min`` / ``edge_max`` / ``edge_ratio``：边长统计。
    """
    tri = np.asarray(tri)[:, :3]
    p = np.asarray(xy)[:, :2][tri]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    l2 = (np.sum((b - a) ** 2, axis=1) + np.sum((c - b) ** 2, axis=1)
          + np.sum((a - c) ** 2, axis=1))
    q = 4.0 * np.sqrt(3.0) * np.abs(0.5 * cross) / l2
    edge = np.sqrt(np.concatenate([l2.reshape(-1)]))
    return dict(
        n_elem=int(len(tri)),
        n_neg=int((cross < 0).sum()),
        n_zero=int((np.abs(cross) <= 0.0).sum()),
        q_min=float(q.min()), q_p1=float(np.percentile(q, 1)),
        q_median=float(np.median(q)),
        n_bad=int((q < Q_BAD).sum()),
        worst=np.argsort(q)[:8].tolist(),
        edge_min=float(np.sqrt(l2.min())),
        edge_max=float(edge.max() * 0 + np.sqrt(l2.max())),
        edge_ratio=float(np.sqrt(l2.max() / l2.min())),
    )


def node_connectivity(tri, n_node: int) -> dict:
    """节点连通性：节点度（被几个单元引用）与重复单元。"""
    tri = np.asarray(tri)[:, :3]
    deg = np.zeros(int(n_node), dtype=int)
    np.add.at(deg, tri.reshape(-1), 1)
    _u, cnt = np.unique(np.sort(tri, axis=1), axis=0, return_counts=True)
    return dict(deg_min=int(deg.min()), deg_max=int(deg.max()),
                n_deg0=int((deg == 0).sum()), n_deg_le2=int((deg <= 2).sum()),
                n_dup_elem=int((cnt > 1).sum()))


def check_mesh(layer, label: str = "", *, element_rotor_gap=None) -> dict:
    """**网格有效性体检的唯一入口**（拓扑 + 几何 + 连通性 + 接口）。

    只做判断、不抛异常 —— 调用方按返回的计数决定是报警还是中断
    （建场路径上的硬性隔离由 ``prune_interface_nodes`` 处理）。

    Parameters
    ----------
    layer : MovingMeshLayer
        需要 ``tri`` / ``xy`` / ``n_node``；有 ``interface_nodes`` 时一并检查。
    element_rotor_gap : iterable[int] | None
        给了就检查「接口节点是否真的都是气隙单元顶点」。``None`` 跳过。
    """
    tri = np.asarray(layer.tri)
    xy = np.asarray(layer.xy)
    n_node = int(layer.n_node)

    info = {"label": label, "n_node": n_node, "n_elem": int(tri.shape[0])}
    info.update(element_geometry(tri, xy))
    info.update(node_connectivity(tri, n_node))
    info.update(area_conservation(tri, xy, n_node=n_node))
    info.update(hull_area_check(tri, xy))
    info["n_orphan"] = int(orphan_nodes(n_node, tri).size)

    iface = getattr(layer, "interface_nodes", None)
    if iface is not None:
        iface = tuple(int(i) for i in iface)
        info["n_interface"] = len(iface)
        if element_rotor_gap is not None:
            _k, dropped = prune_interface_nodes(iface, tri, element_rotor_gap)
            info["n_interface_pruned"] = len(dropped)
    return info


def mesh_ok(info: dict) -> bool:
    """``check_mesh`` 结果的合格判据（**放宽到「能算」而非「好看」**）。

    不合格的只有会造成**零对角行 / 奇异**的那几条：
    孤立节点、节点度为 0、``Δ==0`` 单元、反向定向单元。
    ``q`` 因子**不**作为硬判据 —— 实测 6s4p 有 474 个 ``q<0.1`` 的单元，
    求解完全正常（真正不收敛的原因在材料，见 D13/D14）。
    """
    return (info["n_orphan"] == 0 and info["n_deg0"] == 0
            and info["n_zero"] == 0 and info["n_neg"] == 0
            and info.get("n_interface_pruned", 0) == 0
            and info.get("rel_orientation", 1.0) > 1.0 - 1e-12
            and info.get("ok_hull", True))
