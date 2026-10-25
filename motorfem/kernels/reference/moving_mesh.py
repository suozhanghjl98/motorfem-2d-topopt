"""动网格滑移层与旋转边界权重矩阵（内核 7）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）

    动网格节点层的插入   ``:442-491``
    ``rotating_boundary``      ``:738-806``
    ``rotating_boundary_v2``   ``:809-903``

------------------------------------------------------------------------------
一、这是什么：三样东西
------------------------------------------------------------------------------

本内核是唯一一个把**几何拓扑**与**线性代数**混在一起的内核，它产出三样东西：

1. **``MovingMeshLayer`` 动网格节点层**（几何 + 拓扑）
   转子和定子的三角形单元原本共享气隙中线上的那层节点。插入动网格层后，**在同一
   位置复制**一层新节点，把转子气隙侧单元的节点号换成新节点号 —— 于是定转子单元
   不再有公共节点，转子的每一个节点都可以绕原点自由旋转而不改变任何单元的形状
   （``同步磁阻电机磁障结构分析.md``「旋转动网格的插值方法」约 :2863-2865）。

   复制出来的新节点是**转子侧**的：它们随转子转，但**坐标不移动** —— 旋转是通过
   下面的插值约束行表达的，不是通过改坐标。

2. **``RotatingBoundary`` 旋转边界权重矩阵**（线性代数）
   转子动节点的自由度不独立，它由「转过去之后落点两侧的两个**定子侧接口节点**」
   线性插值决定。本内核把它写成**整行替换**形式的驱动行系数
   （装置本身是内核 6 的 ``driving_rows``，消费点 ``MMA10:2104-2112``）：
   ``F_mat_driven[i, :] = K_rotating_boundary[i, :]``。
   同时返回 ``brackets``（legacy 的 ``node_rotating_boundary``）：
   ``{动节点: (上侧节点, 下侧节点)}``，供内核 8/14 构造单元变换矩阵。

3. **``RotatingBoundaryWithDerivative.d_matrix`` 对转子位置的解析导数**
   ``∂K_rotating_boundary/∂θ``（legacy 的 ``_dKrb_p_dth``）。虚位移法转矩要它
   （内核 14；``get_VD_coef`` ``MMA10:707-736``，末尾还有一处 ``*= -parts`` 的
   θ_m → θ_e 链式因子，属消费方）。

------------------------------------------------------------------------------
二、数学：插值约束行
------------------------------------------------------------------------------

记 ``w = 2π/parts`` 为扇区角（真实模型 ``parts = 4`` → 90°）。每个转子动节点 ``i``
在转动 ``pos`` 之后位于角度

    θ_i = arctan2(y_i, x_i) + pos

其中 ``(x_i, y_i)`` 是它的**原始**坐标（插入时从定子侧节点复制而来）。把它折回一个
扇区：``φ = mod(θ_i, w)``。设接口节点（气隙中线圆上的节点）的角度升序表为
``θ_(0) < θ_(1) < …``，则 ``φ`` 落在某两个相邻接口节点之间：

    kp = (φ − θ_low)/(θ_up − θ_low)，  kn = 1 − kp

``kp`` 是「离下侧节点的相对距离」，所以 ``φ = kp·θ_up + kn·θ_low``。约束行要求

    A_i = s · (kp · A_up + kn · A_low)

其中 ``A_up/A_low`` 是那两个接口节点上的势值。``s = ±1`` 是**奇扇区的反周期符号**：
模型只算一个扇区，θ 落在第 1、3、5… 扇区（``floor(θ/w)`` 为奇数）时，真实场是基准
扇区场的**取负**（``parts = 4``、``2p = 4`` ⇒ 一个扇区 = 180° 电角度 ⇒ 反周期）。
默认 ``a_eq = False``（legacy 的 ``EM_MS_boundary_A_eq``，注释「周期边界 True
反周期边界 False」）即此情形；``a_eq = True`` 时 ``s ≡ +1``。

另外，若括号节点正好是**从属节点**（θ = 2π/parts 射线上的节点，``node_slave`` 的键），
它没有独立自由度：``A_slave = σ·A_master``，``σ = −1``（反周期）或 ``+1``（周期）。
把这个关系代入，行里就只出现**主动节点**，系数乘上 ``σ``。

于是 legacy 里那套看着杂乱的分支（``-kp if _other_region else kp`` 与
``kp if _other_region else -kp`` 的不对称写法）**其实是同一个式子**：

.. code-block:: text

    s    = −1 if (not a_eq and floor(θ_i/w) mod 2 == 1) else +1
    σ_n  = −1 if (not a_eq and n ∈ node_slave) else +1        （a_eq 时恒 +1）
    idx(n) = node_slave[n] if n ∈ node_slave else n

    K_rb[i, i]        = 1
    K_rb[i, idx(up)]  = −s·σ_up·kp
    K_rb[i, idx(low)] = −s·σ_low·kn

逐分支核对过：``a_eq`` (2) × ``other_region`` (2) × 括号节点是否从属 (2) = **8 种
组合全部与 legacy 一致**。乘 ±1 在 IEEE 下是精确的（只是改符号位），所以这个统一式
与 legacy 的分支写法**逐位相同**。

**理论文档没有记录这套符号逻辑。** ``同步磁阻电机磁障结构分析.md`` 的
「旋转动网格的插值方法」一节只有 :2861-2904（:2906 即下一个标题），内容只有两层
节点的构造、``A_gap,r = A_gap,s1·k1 + A_gap,s2·k2``（:2871-2873）、变换矩阵的稀疏
结构（:2875-2902）与「两列相同节点编号必须合并」（:2903）。``_other_region`` 这个
奇扇区符号、``node_slave`` 替换时**为什么不带负号**、以及 ``∂k/∂θ`` 全部**没有
书面记录**（全仓库 10 份 .md 对 ``_other_region``/``_neg_region`` 零命中）。本模块的
这一段就是它们的**首次书面记录**。

------------------------------------------------------------------------------
三、实测事实（真实网格 ``my_gmsh/model_mirror4.msh``）
------------------------------------------------------------------------------

用仓库里纯 Python 的 ``my_gmsh/gmsh_reader.py`` 解析（不依赖 gmsh 库；注意该网格
坐标是**毫米**，MMA10:232 有 ``node[:, 1:] *= 1e-3``）：

.. code-block:: text

    N_node = 4881   N_element = 9599   模型域 = 1/4 扇区 [0°, 90°]，关于 45° 镜像
    主从配对: 主动 66 = 从属 66
    接口圆半径 = m_R_gap = (45.0 + 44.75)/2 mm = 44.875 mm
    接口节点 = 236 个，角度恰好覆盖 [0°, 90°] 含两端点，最小角间距 0.383° = 90/235
    接口圆上无重角  ⇒ 角度排序的平局顺序不影响结果
    接口圆上的从属节点 = 恰好 1 个（θ = 90°），其主动节点是圆上 θ = 0° 的节点
    插入层后 N_node: 4881 → 5117（新增 236 个，编号区间 [4881, 5117)）

**代价与调用频率**（2026-09-22 实测；真网格 N_node = 5117、动节点 236 个）：

.. code-block:: text

    legacy rotating_boundary                     14.2 ms      ← 照抄 :738-806 真跑
    legacy rotating_boundary_v2                  18.3 ms      ← 照抄 :809-903 真跑
    reference（dict→COO，仍逐节点循环）            9.6 / 10.2 ms
    fast（向量化）                                 1.4 /  1.9 ms   ⇒ 相对 legacy 9.6×
    空 lil_matrix((5117,5117)) + tocsr 一个        2.15 ms
    纯逐节点循环（236 次，不含任何稀疏装配）          5.95 ms

即 legacy 的成本**一半以上与「236 个动节点」无关**：``(N_node, N_node)`` 的 lil 容器本身
就要 2.15 ms/个（``rotating_boundary_v2`` 建两个），再逐项赋值、再 ``tocsr``。

**调用频率是这一代才变高的。** ``cProfileReport.txt``（2024-06-26，旧代
``nonlinear_solver_period2.py``）里 ``rotating_boundary`` 只有 **11 次调用、0.02 s**
—— 占那次 150 s profile 的 0.013%，所以当时判断「不值得向量化」是对的。MMA10 把它搬进
了 MMA 迭代体（``:3509-3510``，生产路径 ``obj_func_parallel_AL_v2.__call__``）：每个迭代
对 11 个采样点各调一次 ⇒ ``max_step=3000`` 时 **33,000 次**，主线程串行约 10 分钟
（向量化后约 1 分钟）。

**但真正的修法是缓存，不是向量化**：``rotor_position = rotor_position0 +
100π·t/pole_pairs`` 里两个量都在 ``__init__`` 定死（``:3447``、``:3417``），优化循环中
**不变** ⇒ 这 33,000 次调用的输入逐位相同、结果也逐位相同，全是重复计算；而且 :3510 取
的是带导数的 ``_v2``，那份导数在 worker 里**没被用到**（worker 用的是 :3481 预计算的
``Torque_VD_coef_vec``）。两处都属「改结构」而非「重构」，登记在
``docs/重构设计文档.md`` §6-4 的缓存项里，本内核不动。

三条由此得来的**前提条件**（本实现显式检查，legacy 只会 assert 失败或 IndexError）：

1. 接口节点角必须**覆盖 [0, w] 的右端点**。否则 ``φ`` 找不到上侧括号节点，
   legacy 的 ``while True`` 循环会直接 IndexError。
2. 接口节点角必须**覆盖左端点**（θ_min ≤ φ）。否则 legacy 的
   ``assert index_stator > 0``（:759）失败。
3. 插入动网格层后，``node_slave`` / ``node_master`` 必须**重绑**
   （``MMA10:465-479``），即内核 6 的 ``PeriodicBoundary.rebind``。本内核的
   ``insert_moving_mesh_layer`` 把这一步做掉并放进 ``MovingMeshLayer.pairing``，
   以免调用方漏掉 —— 漏掉的后果是**静默**的：θ=90° 那个新节点不再被认作从属节点，
   该行的符号会错，而解仍然算得出来。

------------------------------------------------------------------------------
四、与 legacy 的差异（逐条，均不改数值行为）
------------------------------------------------------------------------------

===============================  ==============================================
本实现                            legacy
===============================  ==============================================
``K_rb``/``dKrb`` 用 ``dict`` 去重后一次性装成 COO → CSR
                                 ``lil_matrix`` 逐项 ``=`` 赋值
``brackets`` 是普通 ``dict``      ``_node_rotating_boundary`` 是普通 dict（同）
越界/覆盖性检查抛 ``ValueError``（含节点号与角度）  ``assert`` / IndexError
接口节点按调用方给的顺序持有（``MovingMeshLayer.interface_nodes`` 是元组）
                                 ``node_gap_stator`` 是 **set**，顺序即迭代顺序
===============================  ==============================================

**为什么 ``dict`` → COO 与 lil 逐项赋值逐位相同**：本内核的每个稀疏位置**至多被写
一次**，且写入的都是「常数 ±1 乘一个权重」，没有浮点累加，所以「谁先谁后」不参与
结果。（实测：``dict`` 去重 → COO → CSR 与 ``lil`` 逐项赋值 → CSR 逐位相同。）
但有两条 ``lil`` 的语义必须显式复现，否则结构与 legacy 不同：

1. **后写覆盖**。``coo_matrix.tocsr()`` 会把同一 ``(i, j)`` 的重复项**相加**，而
   legacy 的 ``lil[i, j] = v`` 是**覆盖**。真实网格上不会出现重复
   （``idx(up) ≠ idx(low)``，见 ``brackets``），但退化几何（接口只有两个节点）会 ——
   测试 ``test_degenerate_interface_duplicate_write_is_last_wins`` 把它锁住。
2. **恰好为零的元被丢弃**。``lil[i, j] = 0.0``（含 ``-0.0``）**不留下存储元**
   （实测 scipy 1.12；``1e-300`` 则保留）。``pos = 0`` 时 ``kp = 0``，这条差异真的
   会出现。数值上「多存一个零」完全无害（加 ``0.0`` 是精确的），但**稀疏结构**
   会进全局矩阵的三元组、进而进 SuperLU 的列排序，能在**末位**上偏离 legacy。
   所以本实现显式过滤零元，并有 ``test_sparse_structure_matches_legacy`` 用
   ``indptr``/``indices`` 逐位钉住（稠密比较看不出显式零）。

------------------------------------------------------------------------------
五、红线（F4）在本内核的传导路径 —— 与内核 6 不同的结论
------------------------------------------------------------------------------

``insert_moving_mesh_layer`` 里新节点的编号是 ``N_node + i``，``i`` 来自
``enumerate(接口节点集合)`` ⇒ **编号依赖 ``set`` 的迭代顺序**（``CLAUDE.md`` §1 的
F4 红线）。所以：``set`` 构造表达式与 ``enumerate`` 的书写顺序逐字符保留，
调用方必须传入**同一个 set**（构造方式相同）才能得到相同编号。

但**本内核的 F4 后果比内核 6 窄**：本内核产出的所有稀疏元都只写一次、无浮点累加，
所以 set 顺序**不改变任何一个数值**；它只通过「新节点编号 → 全局矩阵的行列序 →
SuperLU 的主元顺序」影响**末位**。内核 6 那边 ``TᵀKT`` 的累加顺序会直接进结果，
这里不会。

另注：``for i_elem in element_rotor_gap`` 是 **set 迭代**，只决定赋值顺序（无累加），
但红线第 4 条仍要求保留写法 —— 不得改成 ``sorted()`` 或 ``range()``。

------------------------------------------------------------------------------
六、与相邻内核的接口
------------------------------------------------------------------------------

- **内核 6**：``PeriodicBoundary.rebind``（重绑）与 ``driving_rows`` 的
  ``row_replacements`` 入口（整行替换）都是为本内核预留的。
  ``element_S_slave`` **不需要**单独重绑：用重编号后的 ``tri`` 与
  ``layer.pairing`` 重新调用 ``slave_virtual_elements`` 即可（内核 6 的
  docstring 第六节第 3 条）；legacy 的增量更新（``MMA10:471-479``）与之等价，
  测试 ``test_virtual_elements_incremental_equals_recompute`` 把它固化成事实。
- **内核 8**：消费 ``matrix`` 与 ``brackets``（``MMA10:1693-1743``）。
- **内核 14**：消费 ``d_matrix``（``MMA10:1776-1835``）。

------------------------------------------------------------------------------
七、P2（6 节点）推广 —— 2026-10-02，路线 A（规划 `docs/重构设计文档.md` §13.3）
------------------------------------------------------------------------------

P2 网格的三角形是 6 节点（3 顶点 + 3 边中点，gmsh 约定列序
``[v0, v1, v2, m01, m12, m20]``）。本内核的 P2 推广只有**插入层**需要显式代码
（6 列连接表 + 接口节点表补全）；**旋转边界的求值逻辑一行都不用改** —— 它本来就
逐动节点用 ``xy``/角度表通用求值，只要 ``layer.interface_nodes`` 含中点、角度表
自动变密，``layer.rotor_nodes`` 含中点、自动给中点建插值行。

1. **接口节点表 = 顶点 + 接口圆边中点**。气隙 band 跨接口圆的边（两端点都在
   接口圆上）其中点在弦中点（半径 ``r·cos(Δθ/2) < r``，**不**过半径判据 ——
   细网格上误差 ~1e-7 m 会误过 ``coord_err_abs=1e-6``）。所以 ``interface_nodes``
   由调用方传**顶点集**（半径判据），插入层从连接表推导出接口圆边中点补进去：
   「任一单元的边两端点都在顶点集 ⇒ 该边中点进接口集」。传入集里若已含中点
   （细网格半径判据的误判），按角色分类去重。

2. **中点动节点走自己的 bracket 行（不是端点平均）**。对每个转子动节点（顶点
   *和* 边中点）建一条插值行：``φ = mod(atan2(y_i,x_i) + pos, w)``，在**含中点**
   的角度表里 bracket 搜索、``A_i = s·(kp·A_up + kn·A_low)``。三个理由（与
   ``assembly_p2`` 的配套约定一致，见该模块 docstring）：

   - ``pos = 0`` 时中点动节点与定子中点角重合 ⇒ 行退化为恒等
     （``A_中点 = A_定子中点``）—— 静网格一致性，与 P1 顶点同款；端点平均
     会给 ``(A_a + A_b)/2``，在 pos=0 就引入 O(h²) 虚假跳变。
   - 转子界面场沿边保持 P2 二次（中点值独立），不把界面压回 P1 一阶。
   - 转子动节点是**纯输出** DOF（被 ``T`` 消去、驱动行喂值），但转矩/磁密
     消费方读 ``A[动节点]`` ⇒ 驱动行必须与 ``T`` 的消元一致。``assembly_p2``
     的 ``transformation_matrix_p2`` 因此与 moving_mesh 同步改为「中点在
     brackets → 自己的 bracket 行」，两端点是配套的一对。

3. **brackets 因此含中点动节点**：``{动节点: (上, 下)}``，上/下可为定子顶点或
   定子中点（后者是独立 DOF）。P1 消费方（``transformation_matrix`` 只查 3 个
   顶点槽位）不受影响 —— P1 路径逐位不变。

4. **新节点编号**（P2）：顶点保持调用方传入顺序，中点按角度稳定排序追加。
   P2 无 legacy 逐位约束，编号顺序只求确定（角度排序不改变任何数值）。

本节的三个解析性质判据（pos=0 恒等、精确场不动点、权重=角度比）在
``tests/kernels/test_moving_mesh.py`` 的 P2 节全部重验，且对顶点与中点同时成立。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp_sparse

from .boundaries import PeriodicBoundary

__all__ = [
    "MovingMeshLayer",
    "RotatingBoundary",
    "RotatingBoundaryWithDerivative",
    "nodes_of_elements",
    "insert_moving_mesh_layer",
    "rotating_boundary",
    "rotating_boundary_with_derivative",
]


# =============================================================================
# 数据结构
# =============================================================================

@dataclass(frozen=True)
class MovingMeshLayer:
    """插入动网格节点层之后的几何与拓扑。

    Attributes
    ----------
    xy : (N_node + N_gap, 2) ndarray
        追加了新节点层之后的节点坐标。**新节点的坐标与原节点逐位相同**
        （legacy ``MMA10:450-452`` 直接复制 x/y）。行号即节点号。
    tri : (N_elem, 3) 或 (N_elem, 6) ndarray
        改写后的连接表：``element_rotor_gap`` 里落在接口圆上的节点号已换成新节点号。
        P2（6 列）下「落在接口圆上」含**接口圆边中点**（模块 docstring 第七节）。
    stator_to_rotor : dict[int, int]
        接口圆上的**原**节点号 → 新节点号。对应 legacy 翻转后的
        ``node_gap_stator``（``MMA10:447``），也是 ``PeriodicBoundary.rebind``
        要的 ``node_remap``。
    rotor_to_stator : dict[int, int]
        新节点号 → 原节点号。对应 legacy 的 ``node_gap_rotor``（``MMA10:446``）。
    interface_nodes : tuple[int, ...]
        接口圆上的原节点号，**按传入集合的迭代顺序**。``rotating_boundary`` 用它
        取角度表；因为排序是稳定的，这个顺序只在「两个接口节点角度完全相同」时
        才影响结果（真实网格上无重角）。
    rotor_nodes : tuple[int, ...]
        新节点号，顺序与 ``interface_nodes`` 相同（即 legacy 的
        ``node_gap_rotor`` 迭代顺序）。
    pairing : PeriodicBoundary
        **已重绑**的主从配对（``MMA10:465-479``）。见模块 docstring 第三节第 3 条：
        不重绑是静默错误。
    node_stator, node_rotor : frozenset[int] 或 None
        ``element_stator`` / ``element_rotor`` 里各单元节点的并集
        （``MMA10:482-491``）。调用方给了这两个区域集合才有值；消费方是绘图代码
        （``MMA10:3352-3357``：判断某个节点是否要跟着转子转）。
        **二者交集为空**正是动网格层的意义 —— 定转子单元不再共享节点。
    """

    xy: np.ndarray
    tri: np.ndarray
    stator_to_rotor: Dict[int, int]
    rotor_to_stator: Dict[int, int]
    interface_nodes: Tuple[int, ...]
    rotor_nodes: Tuple[int, ...]
    pairing: PeriodicBoundary
    node_stator: Optional[frozenset] = None
    node_rotor: Optional[frozenset] = None

    @property
    def n_node(self) -> int:
        return int(self.xy.shape[0])


@dataclass(frozen=True)
class RotatingBoundary:
    """旋转边界的插值约束行。

    Attributes
    ----------
    matrix : (N_node, N_node) csr_matrix
        ``K_rotating_boundary``（``MMA10:740``）。**只对动节点那一行有非零元**：
        对角 ``1``，加最多两个接口节点列。整行替换进驱动行装置。
    brackets : dict[int, tuple[int, ...]]
        ``{动节点: 该约束行参与的所有列}``，节点号都已做从属→主动替换
        （legacy 的 ``_node_rotating_boundary``，``MMA10:797-802``）。

        ⚠ **长度随插值阶数变**：``interp='linear'``（默认，legacy）下是
        ``(上侧节点, 下侧节点)`` **两个**；``interp='cubic'`` 下是三次插值窗口的
        **四个**节点。消费方必须**遍历**它，不得按 2 元组解包
        （见 ``assembly.transformation_matrix`` 分支 1、``torque.transformation_matrix_v2``）。
    """

    matrix: object
    brackets: Dict[int, Tuple[int, ...]]


@dataclass(frozen=True)
class RotatingBoundaryWithDerivative(RotatingBoundary):
    """``RotatingBoundary`` 加上对转子位置 θ 的解析导数。

    Attributes
    ----------
    d_matrix : (N_node, N_node) csr_matrix
        ``dK_rotating_boundary/dθ``（legacy 的 ``_dKrb_p_dth``，``MMA10:812``）。
        **对角恒为 0**（对角元 ``1`` 与 θ 无关）。
    """

    d_matrix: object = None


# =============================================================================
# 节点集合辅助
# =============================================================================

def nodes_of_elements(tri: np.ndarray, elements: Iterable[int]) -> frozenset:
    """给定单元集合，返回其全部节点的并集（对应 ``MMA10:482-491``）。

    插入顺序与 legacy 一致（逐单元、按 ``tri`` 的列顺序 ``add``），因此即使
    调用方对结果做迭代，顺序也与 legacy 相同。P2 网格（6 列）下会含边中点。
    """
    tri = np.asarray(tri)
    if tri.ndim != 2 or tri.shape[1] not in (3, 6):
        raise ValueError(
            f"tri 必须是 (N_elem, 3) 或 (N_elem, 6) 的连接表，实得形状 {tri.shape}。"
            f"legacy 的 element_S 是 5 列 [单元号, n1, n2, n3, 区域号]"
        )
    nodes = set()
    for i_elem in elements:
        for k in range(tri.shape[1]):
            nodes.add(int(tri[i_elem, k]))
    return frozenset(nodes)


# =============================================================================
# 动网格节点层的插入
# =============================================================================

def insert_moving_mesh_layer(
    xy: np.ndarray,
    tri: np.ndarray,
    interface_nodes: Sequence[int],
    element_rotor_gap: Iterable[int],
    pairing: PeriodicBoundary,
    element_stator: Optional[Iterable[int]] = None,
    element_rotor: Optional[Iterable[int]] = None,
) -> MovingMeshLayer:
    """在气隙中线上复制一层节点，把转子气隙侧单元的节点改指到新层。

    对应 ``MMA10:442-491``（``if True:`` 块）。

    Parameters
    ----------
    xy : (N_node, 2) ndarray
        节点坐标。**行号必须等于节点号**（legacy 直接以节点号索引 ``node`` 表）。
        ⚠ legacy 的表是 4 列 ``[节点号, x, y, z]``（坐标单位 m，``MMA10:232`` 已
        从 mm 换算），本实现只取 ``(x, y)``：第 4 列在 2D 下恒为 0，且全文件零消费。
    tri : (N_elem, 3) 或 (N_elem, 6) ndarray
        三角形连接表（``element_S`` 的 ``[n1, n2, n3]``；P2 6 列按 gmsh 约定
        ``[v0, v1, v2, m01, m12, m20]``）。
    interface_nodes : 序列 int
        P1：气隙中线圆上的节点集合（legacy 的 ``node_gap_stator``，
        ``MMA10:339-340``：``|x² + y² − m_R_gap²| < _coord_err_abs`` —— 注意是
        **半径平方差**）。**顺序即新节点编号**（``新编号 = N_node + i``，``i`` 是
        这里的下标），所以传入顺序必须与 legacy 的 set 迭代顺序一致，见模块
        docstring 第五节。
        **P2：传顶点集**（同一半径判据）。接口圆边中点由本函数从连接表推导补全
        （细网格上弦中点约偏出 1e-7 m，半径判据会误收进传入集，按角色分类去重）
        —— 模块 docstring 第七节。P2 的新节点编号 = 顶点（传入顺序）+ 中点
        （按角度排序），无 legacy 逐位约束。
    element_rotor_gap : 可迭代 int
        转子气隙单元集合（legacy 的 ``element_rotor_gap``）。
    pairing : PeriodicBoundary
        插入**之前**的主从配对。返回的 ``MovingMeshLayer.pairing`` 是重绑后的
        （``MMA10:465-479``）。
    element_stator, element_rotor : 可迭代 int, 可选
        定子侧 / 转子侧单元集合。给了就顺带算出 ``node_stator`` / ``node_rotor``
        （``MMA10:482-491``）。

    Returns
    -------
    MovingMeshLayer

    Notes
    -----
    **新节点是从原节点复制出来的**（坐标逐位相同、编号为 ``N_node + i``），
    而且**转子气隙单元的全部槽位都会被检查**（P1 三个、P2 六个）：只要节点号在
    接口集合里就换掉。单元里位于接口圆**之外**的节点（例如转子铁芯外表面上的
    节点）保持原号 —— 这也意味着转子的非接口节点**不**随转子转动，转子的
    「运动」完全由插值行表达。

    几何前提：``element_rotor_gap`` 里的单元必须**恰好**在接口圆这一侧有节点；
    若某个气隙单元的节点里混了非接口的定子节点，它不会被替换，于是定转子仍共享
    该节点 —— legacy 同样如此（只按接口集合成员测试）。
    """
    xy = np.asarray(xy, dtype=float)
    tri = np.asarray(tri).copy()
    n_cols = tri.shape[1]
    if tri.ndim != 2 or n_cols not in (3, 6):
        raise ValueError(
            f"tri 必须是 (N_elem, 3) 或 (N_elem, 6) 的连接表，实得形状 {tri.shape}"
        )

    n_node = xy.shape[0]
    ordered_interface = tuple(int(n) for n in interface_nodes)

    if n_cols == 6:
        # ---- P2（6 节点）：接口节点表补全 = 顶点 + 接口圆边中点 ----
        # 顶点保持调用方传入顺序；中点（含细网格上半径判据误收进传入集里的）
        # 按角度稳定排序追加。推导与角色判定都只靠连接表（模块 docstring 第七节）。
        midpoint_of: Dict[int, Tuple[int, int]] = {}
        for i_elem in range(tri.shape[0]):
            for k in range(3):
                m = int(tri[i_elem, 3 + k])
                if m not in midpoint_of:
                    midpoint_of[m] = (
                        int(tri[i_elem, k]), int(tri[i_elem, (k + 1) % 3]),
                    )
        vertex_ordered = [n for n in ordered_interface if n not in midpoint_of]
        vertex_set = set(vertex_ordered)
        midpoints = {n for n in ordered_interface if n in midpoint_of}
        for i_elem in range(tri.shape[0]):
            for k in range(3):
                va = int(tri[i_elem, k])
                vb = int(tri[i_elem, (k + 1) % 3])
                if va in vertex_set and vb in vertex_set:
                    midpoints.add(int(tri[i_elem, 3 + k]))
        ordered_interface = tuple(vertex_ordered) + tuple(
            sorted(
                midpoints,
                key=lambda m: (np.arctan2(xy[m, 1], xy[m, 0]), m),
            )
        )

    # legacy MMA10:446-447：先建 新→旧，再翻转成 旧→新
    rotor_to_stator = dict((i + n_node, j) for (i, j) in enumerate(ordered_interface))
    stator_to_rotor = dict((j, i) for (i, j) in rotor_to_stator.items())

    # legacy MMA10:448-455：复制坐标追加到节点表末尾。行号 = 新节点号。
    new_rows = np.array(
        [xy[i_node] for i_node in ordered_interface], dtype=float
    ).reshape(len(ordered_interface), 2)
    xy_new = np.concatenate([xy, new_rows], axis=0)

    # legacy MMA10:459-463：把 element_rotor_gap 里的接口节点换成新节点。
    # P1 是 3 个槽位（``(0, 1, 2)``）；P2 全 6 槽位都检查（顶点 + 边中点）。
    for i_elem in element_rotor_gap:
        for k in range(n_cols):
            i_node = int(tri[i_elem, k])
            if i_node in stator_to_rotor:
                tri[i_elem, k] = stator_to_rotor[i_node]

    # legacy MMA10:465-479：主从表就地重绑（内核 6 的纯函数版）
    pairing = pairing.rebind(stator_to_rotor)

    # legacy MMA10:482-491：记录定转子节点
    node_stator = (
        nodes_of_elements(tri, element_stator)
        if element_stator is not None
        else None
    )
    node_rotor = (
        nodes_of_elements(tri, element_rotor) if element_rotor is not None else None
    )

    return MovingMeshLayer(
        xy=xy_new,
        tri=tri,
        stator_to_rotor=stator_to_rotor,
        rotor_to_stator=rotor_to_stator,
        interface_nodes=ordered_interface,
        rotor_nodes=tuple(rotor_to_stator),
        pairing=pairing,
        node_stator=node_stator,
        node_rotor=node_rotor,
    )


# =============================================================================
# 旋转边界：插值行（以及它对 θ 的导数）
# =============================================================================

def _interface_angle_table(xy: np.ndarray, interface_nodes: Sequence[int]):
    """接口节点的 ``[(θ, 节点号), …]``，按角度**稳定**升序。

    对应 ``MMA10:743-745``。稳定排序保证「角度完全相同」时保持传入顺序 —— 与
    legacy 的 ``sorted(key=…)``（Python 的 sort 是稳定的）一致。
    """
    table = [
        (np.arctan2(xy[i_node, 1], xy[i_node, 0]), int(i_node))
        for i_node in interface_nodes
    ]
    table.sort(key=lambda item: item[0])
    return table


def _bracket(table, phi, i_rotor, angle_range):
    """在角度表里取 ``φ`` 的上下括号节点下标。对应 ``MMA10:753-759``。

    legacy 是 ``while True: if table[index][0] > φ: break; index += 1``，即
    「第一个角度**严格大于** φ 的下标」。两个前提都以可读错误显式给出：
    ``index == 0``（θ_min > φ，legacy 的 ``assert index_stator > 0`` 失败）
    与 ``index == len(table)``（没有任何角度 > φ，legacy 会 IndexError）。
    """
    index = 0
    n = len(table)
    while index < n and table[index][0] <= phi:
        index += 1
    if index == 0:
        raise ValueError(
            f"动节点 {i_rotor} 的折叠角 φ = {phi!r} 小于接口节点的最小角度 "
            f"{table[0][0]!r}；接口节点角必须覆盖 [0, {angle_range!r}] 的左端点"
            f"（legacy 此处 assert index_stator > 0 失败）"
        )
    if index == n:
        raise ValueError(
            f"动节点 {i_rotor} 的折叠角 φ = {phi!r} 超出接口节点的最大角度 "
            f"{table[-1][0]!r}；接口节点角必须覆盖右端点"
            f"（legacy 此处会 IndexError）"
        )
    return index


def _rotating_boundary_terms(xy, layer, parts, pos, a_eq, with_derivative,
                             interp: str = "linear"):
    """逐动节点算出插值行的三元组与括号节点。返回 ``(k, dk, brackets)``。

    这是 ``rotating_boundary`` 与 ``rotating_boundary_with_derivative`` 的**唯一**
    核心：legacy 把同一套符号逻辑在 ``:738-806`` 与 ``:809-903`` 里**抄了两遍**
    （正是本项目要治的病），本实现只写一遍，K 的取值与两处 legacy 都逐位相同。

    对每个动节点（模块 docstring 第二节的记号）：

    .. code-block:: text

        θ = arctan2(y, x) + pos
        φ = mod(θ, w)                             w = 2π/parts
        s = −1 if (not a_eq and mod(θ // w, 2) > 0.5) else +1
        kp = (φ − θ_low)/(θ_up − θ_low),  kn = 1 − kp
        dkp = 1/(θ_up − θ_low),           dkn = −dkp      （∂φ/∂pos = 1）

        σ_n  = −1 if (not a_eq and n ∈ node_slave) else +1
        idx  = 从属 → 主动的替换
        K_rb[i, i]        = 1                       dK_rb[i, i]        = 0
        K_rb[i, idx(up)]  = −s·σ_up·kp              dK_rb[i, idx(up)]  = −s·σ_up·dkp
        K_rb[i, idx(low)] = −s·σ_low·kn             dK_rb[i, idx(low)] = −s·σ_low·dkn
    """
    xy = np.asarray(xy, dtype=float)
    n_node = xy.shape[0]
    sector = 2 * np.pi / parts

    table = _interface_angle_table(xy, layer.interface_nodes)
    slave_to_master = layer.pairing.slave_to_master
    n_iface = len(table)

    # ---- 显式检测：cubic 分支需要至少 4 个环绕窗口节点、且相邻角间距不得为 0 ----
    # **不假设接口等距**（间距由网格决定）；非均匀由 `_hermite_weights` 直接处理。
    # 这里只把「无法做三次插值」的真正退化情形**大声**报出来。
    hw = None
    n_per = n_iface
    if interp == "cubic":
        hw = _cyclic_widths(table, sector)
        n_per = int(hw.size)
        if n_per < 4:
            raise ValueError(
                f"interp='cubic' 需要接口环绕周期 ≥ 4 个节点（当前 {n_per}）—— "
                f"三次插值的窗口是 4 个相邻节点。（注意：接口节点角覆盖 "
                f"[0, w] **含两端点** ⇒ 末节点与首节点是同一物理点，环绕周期 = 节点数 − 1。）"
                f"请改用 interp='linear'。"
            )
        if not np.all(hw > 0.0):
            bad = int(np.argmin(hw))
            raise ValueError(
                f"interp='cubic'：接口圆上存在**重复角度**（hw[{bad}] = {hw[bad]!r} ≤ 0），"
                f"切线公式要除以相邻间距 ⇒ 无法三次插值。请查网格或改用 interp='linear'。"
            )

    # 常数 ±1 乘权重：IEEE 下精确（只改符号位），故与 legacy 的分支写法逐位相同
    sigma = 1.0 if a_eq else -1.0

    k_entries: Dict[Tuple[int, int], float] = {}
    dk_entries: Dict[Tuple[int, int], float] = {}
    brackets: Dict[int, Tuple[int, ...]] = {}

    for i_rotor in layer.rotor_nodes:
        theta = np.arctan2(xy[i_rotor, 1], xy[i_rotor, 0]) + pos
        phi = np.mod(theta, sector)
        other_region = np.mod(theta // sector, 2) > 0.5

        index = _bracket(table, phi, i_rotor, sector)
        theta_up, i_up = table[index]
        theta_low, i_low = table[index - 1]

        kp = (phi - theta_low) / (theta_up - theta_low)
        kn = 1.0 - kp

        # 奇扇区的反周期符号：周期边界（a_eq）下恒 +1
        s = -1.0 if (not a_eq and other_region) else 1.0

        k_entries[(i_rotor, i_rotor)] = 1.0

        if interp == "linear":
            # ---- 现状（legacy）：落点两侧两个接口节点的分段线性插值 ----
            # 从属 → 主动的替换：系数上乘 σ（A_slave = σ·A_master）
            sigma_up = sigma if i_up in slave_to_master else 1.0
            sigma_low = sigma if i_low in slave_to_master else 1.0
            idx_up = slave_to_master.get(i_up, i_up)
            idx_low = slave_to_master.get(i_low, i_low)

            k_entries[(i_rotor, idx_up)] = -s * sigma_up * kp
            k_entries[(i_rotor, idx_low)] = -s * sigma_low * kn
            brackets[i_rotor] = (idx_up, idx_low)

            if with_derivative:
                dkp = 1.0 / (theta_up - theta_low)
                dkn = -dkp
                dk_entries[(i_rotor, idx_up)] = -s * sigma_up * dkp
                dk_entries[(i_rotor, idx_low)] = -s * sigma_low * dkn
            continue

        if interp != "cubic":
            raise ValueError(f"未知 interp={interp!r}（可选 'linear' / 'cubic'）")

        # ---- 新方案：接口圆上的**三次 Hermite（Catmull-Rom 型）**插值，C¹ ----
        # 窗口节点（表下标）为 index−2 .. index+1，局部参数 t = kp ∈ [0,1) 落在
        # [table[index−1], table[index]] 区间内。
        # ★ **均匀与非均匀统一处理**：切线用三点加权公式，均匀时逐项退化为经典
        #   Catmull-Rom（见 `_hermite_weights` 的文档）。**不得假设边界等距** ——
        #   接口间距由网格决定，非均匀时要能算对（检测见 `interface_spacing_stats`）。
        # 周期环绕：序列 F_k = σ^{⌊k/N⌋}·A_{k mod N}（σ 为周期/反周期符号），
        # 于是绕缝处也是光滑的（A_slave = σ·A_master）。
        w4, dw4 = _hermite_weights(
            kp,
            float(hw[(index - 2) % n_per]),
            float(hw[(index - 1) % n_per]),
            float(hw[index % n_per]),
        )
        cols = []
        for j, k_tab in enumerate((index - 2, index - 1, index, index + 1)):
            m = k_tab % n_per
            wrap = k_tab // n_per             # Python 地板除：−1//P = −1
            node_j = table[m][1]
            coef = (-s) * (sigma ** wrap)
            if node_j in slave_to_master:      # 从属 → 主动（系数上乘 σ）
                coef *= sigma
                col = slave_to_master[node_j]
            else:
                col = node_j
            k_entries[(i_rotor, col)] = (
                k_entries.get((i_rotor, col), 0.0) + coef * w4[j])
            cols.append(int(col))
            if with_derivative:
                dk_entries[(i_rotor, col)] = (
                    dk_entries.get((i_rotor, col), 0.0) + coef * dw4[j])
        brackets[i_rotor] = tuple(cols)

    return (k_entries, dk_entries, brackets), n_node


def _interface_period(table, sector: float) -> int:
    """接口节点表的**环绕周期**（用多少个节点走完一圈）。

    接口节点角覆盖 ``[0, w]`` **含两端点**，而 ``w`` 正是扇区角 ⇒ **末节点与首节点
    是同一个物理点**（相差恰好一个扇区角）。因此周期是 ``N − 1``，不是 ``N``——
    否则绕缝那一跨的宽度为 0（退化），切线公式会除以 0。

    实测（``model_mirror4.msh``）：236 个接口节点、角度 0°…90° 等距
    ``90/235``，末节点在 θ=90°、与首节点（θ=0°）相差正好 ``w`` ⇒ 周期 235。

    ⚠ **不假设等距**：本函数只判定「端点是否重合」，间距由 :func:`_cyclic_widths`
    如实返回。
    """
    ang = np.array([item[0] for item in table], dtype=float)
    n = ang.size
    if n >= 2 and abs((ang[-1] - ang[0]) - sector) <= 1e-9 * abs(sector):
        return n - 1
    return n


def _cyclic_widths(table, sector: float) -> np.ndarray:
    """接口圆的**环绕**相邻角间距，长度 = :func:`_interface_period`。

    ``hw[k] = φ_(k+1) − φ_k``（``k = 0 … P−2``），最后一个是绕缝那一跨
    ``(φ_0 + w) − φ_(P−1)``。**不假设等距** —— 返回的就是实际间距。
    """
    ang = np.array([item[0] for item in table], dtype=float)
    p = _interface_period(table, sector)
    hw = np.empty(p, dtype=float)
    hw[: p - 1] = ang[1:p] - ang[: p - 1]
    hw[p - 1] = (ang[0] + sector) - ang[p - 1]
    return hw


def interface_spacing_stats(table, sector: float) -> Dict[str, object]:
    """**显式检测**接口圆上角间距是否均匀。返回统计量，供调用方报告/断言。

    Returns
    -------
    dict
        ``{"n": 节点数, "h_min"/"h_max"/"h_mean": 环绕间距, "is_uniform": bool,
        "max_rel_dev": max|h−h̄|/h̄}``。``is_uniform`` 用 ``max_rel_dev <= 1e-9`` 判定。
    """
    hw = _cyclic_widths(table, sector)
    h_mean = float(hw.mean())
    dev = float(np.abs(hw - h_mean).max() / h_mean) if h_mean > 0 else float("inf")
    return {
        "n": int(hw.size),
        "h_min": float(hw.min()),
        "h_max": float(hw.max()),
        "h_mean": h_mean,
        "is_uniform": dev <= 1e-9,
        "max_rel_dev": dev,
    }


def _hermite_weights(t: float, hm: float, h0: float, hp: float):
    """**非均匀**三次 Hermite 在局部参数 ``t ∈ [0,1)`` 处对窗口 ``(A₋₁,A₀,A₁,A₂)`` 的权重。

    区间是 ``[A₀, A₁]``（``t = 0`` 落在 ``A₀``、``t = 1`` 落在 ``A₁``），
    ``hm/h0/hp`` 分别是前一跨、本跨、后一跨的角间距。

    切线取**三点加权公式**（Catmull-Rom 的非均匀推广）::

        m₀ = ( h₀·(A₀−A₋₁)/hm + hm·(A₁−A₀)/h₀ ) / (hm + h₀)
        m₁ = ( hp·(A₁−A₀)/h₀ + h₀·(A₂−A₁)/hp ) / (h₀ + hp)

    于是 ``A(t) = h00·A₀ + h10·h₀·m₀ + h01·A₁ + h11·h₀·m₁``，其中
    ``h00 = 2t³−3t²+1``、``h10 = t³−2t²+t``、``h01 = −2t³+3t²``、``h11 = t³−t²``。

    **两条性质**（本项目依赖）：
    1. **均匀时退化为经典 Catmull-Rom**：``hm=h₀=hp=h`` ⇒ 权重恰为
       ``½[−t³+2t²−t, 3t³−5t²+2, −3t³+4t²+t, t³−t²]``（逐项可验）。
    2. **C¹ 与均匀性无关**：切线 ``m_k`` 只由该节点两侧的间距与值决定、**与区间无关**，
       故跨节点导数连续 —— 这正是相对现状（分段线性、节点处导数跳变）的改进所在。

    返回 ``(w, dw)``，``dw`` 是**对 φ** 的导数（已含 ``1/h₀`` 的链式因子）。
    """
    t2 = t * t
    t3 = t2 * t
    h00, h10, h01, h11 = 2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t, \
        -2.0 * t3 + 3.0 * t2, t3 - t2
    d00, d10, d01, d11 = 6.0 * t2 - 6.0 * t, 3.0 * t2 - 4.0 * t + 1.0, \
        -6.0 * t2 + 6.0 * t, 3.0 * t2 - 2.0 * t

    s_m = 1.0 / (hm + h0)      # m₀ 的分母
    s_p = 1.0 / (h0 + hp)      # m₁ 的分母

    # 各窗口节点在 A(t) 里的系数（m₀ 的 A₋₁ 项系数是 −h₀/hm，A₀ 项是 +h₀/hm…
    # 展开后按下标归并；下面直接写归并结果）。
    w = (h10 * h0 * (-h0 / hm) * s_m,
         h00 + h10 * h0 * (h0 / hm - hm / h0) * s_m - h11 * hp * s_p,
         h01 + h10 * h0 * (hm / h0) * s_m + h11 * h0 * (hp / h0 - h0 / hp) * s_p,
         h11 * h0 * (h0 / hp) * s_p)
    dw = ((d10 * h0 * (-h0 / hm) * s_m) / h0,
          (d00 + d10 * h0 * (h0 / hm - hm / h0) * s_m - d11 * hp * s_p) / h0,
          (d01 + d10 * h0 * (hm / h0) * s_m + d11 * h0 * (hp / h0 - h0 / hp) * s_p) / h0,
          (d11 * h0 * (h0 / hp) * s_p) / h0)
    return w, dw


def _entries_to_csr(entries: Mapping[Tuple[int, int], float], n_node: int):
    """``dict`` → COO → CSR。两处必须复现 legacy 的 ``lil`` 语义：

    1. **后写覆盖**：``dict`` 的赋值语义就是 legacy ``lil`` 的 ``=``（同一位置重复写
       只剩最后一个值）。这里因此**没有** ``coo.tocsr()`` 把重复项相加的问题
       （见模块 docstring 第四节）。
    2. **恰好为零的元被丢弃**：``lil[i, j] = 0.0`` **不会**留下存储元（实测
       scipy 1.12：写 ``0.0`` 或 ``-0.0`` 都把该位置从行里删掉；``1e-300`` 则保留）。
       ``pos = 0`` 时 ``kp = 0``，这条差异会真的出现，所以必须显式过滤：
       否则数值完全相同，但**稀疏结构**多出一个存储零 —— 而结构会进全局矩阵的
       CSR 三元组，进而进 SuperLU 的列排序，能在**末位**上偏离 legacy。
       测试 ``test_sparse_structure_matches_legacy`` 用的是 ``indptr``/``indices``
       逐位比对，就是为了钉住这一条（稠密比较看不出显式零）。
    """
    if not entries:
        return sp_sparse.csr_matrix((n_node, n_node))
    keys = [k for k in entries if entries[k] != 0.0]
    if not keys:
        return sp_sparse.csr_matrix((n_node, n_node))
    rows = np.array([k[0] for k in keys], dtype=np.int64)
    cols = np.array([k[1] for k in keys], dtype=np.int64)
    vals = np.array([entries[k] for k in keys], dtype=float)
    return sp_sparse.coo_matrix((vals, (rows, cols)), shape=(n_node, n_node)).tocsr()


def rotating_boundary(
    layer: MovingMeshLayer, parts: int, pos: float, a_eq: bool,
    interp: str = "linear",
) -> RotatingBoundary:
    """旋转边界的插值约束行 ``K_rotating_boundary``。对应 ``MMA10:738-806``。

    Parameters
    ----------
    layer : MovingMeshLayer
        ``insert_moving_mesh_layer`` 的返回值。用其中的 ``xy``（含新节点层）、
        ``interface_nodes``、``rotor_nodes`` 与**已重绑**的 ``pairing``。
    parts : int
        旋转对称扇区数（真实模型 4）。扇区角 ``w = 2π/parts``。
    pos : float
        转子位置（弧度）。legacy 传的是 ``rotor_position``
        （``MMA10:3135-3136``：``rotor_position0 + 100π·t/pole_pairs``），
        可以取任意实数 —— ``mod`` 与 ``//`` 会把它折回扇区。
    a_eq : bool
        周期（``True``，``A_slave = +A_master``）还是反周期（``False``，取负）。
        legacy 的 ``EM_MS_boundary_A_eq``，默认 ``False``。

    Returns
    -------
    RotatingBoundary

    Notes
    -----
    行语义 ``A_i − s·(kp·A_up + kn·A_low) = 0``：动节点的值是两侧接口节点值的线性
    插值（乘上奇扇区的反周期符号）。整行替换由内核 6 的
    ``driving_rows(..., row_replacements=...)`` 完成（``MMA10:2104-2112``）。

    ``pos = 0`` 时每个动节点与自己下方的接口节点角完全重合 ⇒ ``kp = 0, kn = 1``，
    行退化为「``A_i`` 等于那个接口节点（或其主动节点，带 σ 符号）」—— 即**静网格
    下插值是恒等**，不会引入任何虚假跳变。

    **P2**：本函数对 P2 零改动 —— ``layer.rotor_nodes`` 含中点、``layer.interface_nodes``
    含中点（角度表自动变密）时，中点动节点也逐节点走同一套求值（模块 docstring
    第七节第 2 条）。``brackets`` 因而含中点条目，供 ``transformation_matrix_p2``
    （``assembly_p2``，与 moving_mesh 配套的 Design B）逐槽位查用。
    """
    (k_entries, _, brackets), n_node = _rotating_boundary_terms(
        layer.xy, layer, parts, pos, a_eq, with_derivative=False, interp=interp
    )
    return RotatingBoundary(
        matrix=_entries_to_csr(k_entries, n_node), brackets=brackets
    )


def rotating_boundary_with_derivative(
    layer: MovingMeshLayer, parts: int, pos: float, a_eq: bool,
    interp: str = "linear",
) -> RotatingBoundaryWithDerivative:
    """``rotating_boundary`` 加上 ``∂K_rotating_boundary/∂θ``。对应 ``MMA10:809-903``。

    导数只来自插值权重：``∂φ/∂pos = 1`` ⇒ ``∂kp/∂pos = 1/(θ_up − θ_low)``、
    ``∂kn/∂pos = −∂kp/∂pos``。**对角元恒为 0**（``K_rb[i, i] = 1`` 与 ``pos`` 无关）。

    Notes
    -----
    这里返回的是**对 ``pos``（即 ``rotor_position``）**的导数。消费方还要乘链式
    因子：``get_VD_coef`` 末尾的 ``*= -parts``（``MMA10:735``）来自「扇区角 →
    机械角」的关系，属内核 14。

    在「括号节点发生交换」与「``other_region`` 翻转」的位置上，``K_rb(pos)``
    是**分段光滑**的：导数在那里不连续，中心差分校验必须取在同一段内
    （测试 ``test_derivative_matches_central_difference`` 说明这一点）。
    """
    (k_entries, dk_entries, brackets), n_node = _rotating_boundary_terms(
        layer.xy, layer, parts, pos, a_eq, with_derivative=True, interp=interp
    )
    return RotatingBoundaryWithDerivative(
        matrix=_entries_to_csr(k_entries, n_node),
        brackets=brackets,
        d_matrix=_entries_to_csr(dk_entries, n_node),
    )
