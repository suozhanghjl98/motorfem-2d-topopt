"""主从周期边界、从属虚拟单元与驱动行矩阵（内核 6）。

来源：``legacy/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）

    主从节点配对      ``:304-321``（电磁侧）、``:402-419``（力学侧）
    从属虚拟单元      ``:323-335``（电磁侧）、``:421-436``（力学侧，见下方警告）
    驱动行矩阵        ``:2073-2102``（全在 ``get_K_vectorized`` 内）

------------------------------------------------------------------------------
一、这是什么：三类对象
------------------------------------------------------------------------------

本内核是唯一一个把**几何拓扑**与**线性代数**混在一起的内核，它产出三样东西：

1. **``PeriodicBoundary`` 主从配对**（纯几何）
   在 ``parts`` 重旋转对称模型里只建一个扇区。扇区两条边界上的自由度不独立：

   - **主动边界**：θ=0 射线（判据 ``|y| < err < |x|``；注意 θ=π 的反向射线
     也满足同一判据，真实网格里两条射线上的节点都会入选）。
   - **从属边界**：绕原点转 −2π/parts 后落在 θ=0 上的节点，即 θ=±2π/parts。

   配对就是把每个从属节点配到「转 +2π/parts 后与它重合」的主动节点。

   ⚠ 电磁侧与力学侧**是同一个算法**（MMA10:402-419 与 304-321 逐字同构，
   只是作用在压缩后的力学节点表上），所以本内核只写一份，两侧共用。

2. **``SlaveVirtualElements`` 从属虚拟单元**（几何 + 拓扑）
   从属节点所在的单元，把它的从属节点换成对应的主动节点，得到「虚拟单元」。
   这样从属节点那一行虽然会被约束行覆盖，单元的能量信息仍不丢失。

3. **``DrivingRows`` 驱动行矩阵**（线性代数）
   把残差/Jacobi 的指定行**整行替换**成边界条件行。MMA10 里每个残差求值都走：

   .. code-block:: python

       F      = F * (1 - F_is_driven) + (F_mat_driven @ A) * F_is_driven  # 2244
       Jacobi = J_driven @ Jacobi + J_mat_driven                        # 2255

------------------------------------------------------------------------------
二、数学表述（驱动行装置）
------------------------------------------------------------------------------

有限元方程记为 ``K A = R``（等价残差 ``F(A) = K A − R = 0``，MMA10:2243）。
边界条件**不是**用消行消列或罚函数施加的，而是**整行替换**：对每个受约束的
节点 ``i``，把第 ``i`` 行换成该约束的线性表达式。具体三类：

.. code-block:: text

    齐次 Dirichlet          第 i 行 ← A_i                        （= 0）
    主从周期/反周期         第 s 行 ← A_s + σ·A_m                （= 0）
    动网格插值（内核 7）     第 i 行 ← A_i − Σ_k w_k·A_k          （= 0）

其中 ``σ = −1`` 为周期边界（``A_slave = +A_master``）、``σ = +1`` 为反周期
边界（``A_slave = −A_master``）。详见「三、``a_eq`` 的语义」。

若把「被替换的行集合」记为 ``d``（``is_driven``，取值 0/1），则

.. code-block:: text

    F ← (1 − d) ⊙ F + d ⊙ (M A)          M = 受约束行的约束系数矩阵（mat_driven）
    J ← D_a J + M_J                      D_a = diag(1 − d)（j_driven）

``(1−d) ⊙ F`` 把自由行保留、``d ⊙ F`` 把受约束行上原有的值乘 0 丢掉 —— 合起来
就是「自由行照旧、受约束行换成约束表达式」。

**为什么一步 Newton 就能精确满足约束**：Newton 步满足 ``J δ = −F``。
对受约束行 ``i``，``J`` 的第 ``i`` 行恰是约束行本身（``D_a`` 把该行清零、
``M_J`` 提供之），所以

.. code-block:: text

    δ_i = −(该行的约束表达式)

对主从行即 ``δ_s = −(A_s + σ A_m)``。若当前 ``A_s + σ A_m = c``，则更新后
``A_s + σ A_m = 0`` —— **精度只受该行浮点求和的舍入限制**，与 Newton 的
非线性收敛无关（约束本身是线性的）。这条性质是内核级测试的主要解析判据。

**覆盖顺序**（legacy 顺序，必须保留）：Dirichlet 行先写，主从行后写。
这不是假想情况：角点 ``(0, ±m_R)`` 既在外圆上（Dirichlet）又在从属射线上
（从属），必然同时命中；主从分支会把该行清空重写。真实网格里 θ=0/360 上的
节点就同时属于两类。

------------------------------------------------------------------------------
三、``a_eq`` 的语义（极易读反，特别说明）
------------------------------------------------------------------------------

legacy 的开关叫 ``EM_MS_boundary_A_eq``（MMA10:91，注释「周期边界True
反周期边界False」）。它的意思是「**A 相等**」：

.. code-block:: text

    a_eq = True    →  A_slave = +A_master   周期边界
    a_eq = False   →  A_slave = −A_master   反周期边界   ← MMA10 的默认值

它同时决定两处的符号，两处必须一致：

- 约束行符号：``mat_driven[s, m] = (−1 if a_eq else +1)``
- 虚拟单元符号：``virtual_element_signs`` 给出 ``+1``（主动节点）或
  ``(1 if a_eq else −1)``（其余节点）

周期情形（``a_eq=True``）下符号恒为 ``+1``，变换矩阵是单位阵，一切平凡。

------------------------------------------------------------------------------
四、★ 待核对：legacy 的变换矩阵符号（当前模型**不触发**）★
------------------------------------------------------------------------------

理论文档 ``磁障设计/同步磁阻电机磁障结构分析.md``「主从边界的处理问题」
（约 2782-2846 行）给出把从属自由度消去后的**正确**变换：

.. code-block:: text

    A^(e) = [T^(e)] A_master^(e)          T 把主动值映到单元局部值
    p_s = −1  if 节点 s 在主动边界上       ← 理论文档约 :2844
          +1  otherwise

    即 T_theory = diag(−1 于主动槽位, +1 其余)

而 legacy（MMA10:1761 的从属分支；1706 与 1789 同理）用的是

.. code-block:: python

    _ki = 1 if node_i in node_master else (1 if EM_MS_boundary_A_eq else -1)

即 **``T_legacy = −T_theory``**。注意 ``node_master`` 此时已是
**master→slave 的反向表**（MMA10:321 覆盖而成），所以 ``node_i in node_master``
的含义是「节点 i 是某个从属节点的主动节点」。

**这个等式有三个限定，缺一条就不成立**：

1. **只在反周期分支（``a_eq=False``，默认）成立。** ``a_eq=True``（周期）时
   legacy 给恒 ``+1``（单位阵），而理论那条 ``p_s = −1`` 公式本身声明了「假设
   主从边界为反周期边界」，此时不适用——两边都是单位阵，不存在取负关系。
2. **需要「主动射线上的每个节点都配对成功」。** legacy 的成员测试是
   ``node_i in node_master``，那是**配对成功的**主动节点的 dict；理论的判据是
   几何的（「在主动边界上」）。二者相等 ⟺ 每个主动射线节点都配了从属节点。
3. 因此「逐槽位符号恰好互补」只在上述两条同时成立时才是「整体差一个全局负号」。
   若某个虚拟单元槽位是「**在主动射线上但没配上**」的节点，legacy 给 ``−1``、
   理论也给 ``−1``（**同号**）——此时既不是整体取负，``TᵀKT`` 的不变性也**一并
   失效**（不再是「差一个符号」，而是真的算错）。

真实网格上主动射线节点 66 个全部配对（66 = 66），限定 2 满足；但**跨两个扇区的
网格就会破坏它**——见 ``PeriodicBoundary`` 的 Notes，以及测试
``test_legacy_equals_minus_theory_requires_all_masters_paired`` 里造出的反例。

**逐项核对反周期分支（``a_eq=False``，默认值）下这个全局负号的影响**：

=============================  ==========  =========================================
量                              受影响？    原因 / 位置
=============================  ==========  =========================================
``Tᵀ K T``（单元刚度）           **不变**    两次负号在二次型里抵消（MMA10:2021）
``g gᵀ``（饱和项）、``g0i`` 族    **不变**    均由 ``TᵀKT`` 派生
``Bxcoef @ T``（Bx, By）         变号        MMA10:2000-2001；但**全部消费点都是
                                            偶次组合或取模**，见下方注①
Arkkio 转矩系数                  **无关**    用 ``element_S`` 真实节点 + raw
                                            ``Bxcoef``，不经过 T（MMA10:671、686）
VD 转矩                         **不变**    非气隙从属单元走从属分支
                                            ``_dT = np.diag([0,0,0])``（MMA10:1860）
                                            → 贡献恒为 0；气隙从属单元见下方注②
``_T.T @ R0elem``（**源项**）     **变号**    MMA10:2203
=============================  ==========  =========================================

注① ``Bxcoef @ T`` 变号后的**全部**下游消费点（逐条核实过，都是偶次或取模）：

.. code-block:: text

    MMA10:2229-2231   B = sqrt(Bx² + By²)                      取模
    MMA10:2347-2348   B = sqrt(Bx² + By²)（冻结磁导率解）        取模
    MMA10:2427-2430   返回 (_Bx, _By)，下游 Ldd/Lqq/Ldq 于
                      :2948-2950 为 Bx²、By²、Bx·Bx 形式         偶次
    MMA10:2360-2365   Bxa²+Bya²−Bxb²−Byb² 与 Bxa·Bxb+Bya·Byb   偶次

所以变号在当前代码里观测不到。**但这是「全部消费者恰好都是偶次」的巧合，不是
结构性保证**：将来若新增一个**奇次**消费者（例如带符号的磁链方向、B 矢量图导出、
逐单元的 Bx 平均值），它就会静默出错。新增消费者时请回到这张表核对。

注② **VD 转矩那一行取决于单元的归属**：``get_VD_coef`` 只遍历
``element_rotor_gap``（MMA10:712-713），而变换矩阵的分支判据是
``if i_elem in element_rotor_gap`` **优先**于 ``elif i_elem in element_S_slave``
（MMA10:1694、1777）。所以「既是从属单元、又在转子气隙里」的单元走的是**气隙
分支**，``_dT`` 不为零、贡献也不为零；它之所以仍不变，靠的是 ``_T`` 与 ``_dT``
**同时**翻号使外积不变（MMA10:718-724），而不是「贡献为零」。只有**非气隙**的
从属单元才走 ``_dT = diag(0,0,0)`` 那条路径（MMA10:1860）。真实网格上同时命中
两条分支的单元有 4 个。

即**唯一**可能真正影响解的是源项：若某个从属单元落在载流区，它向主动节点注入的
电流符号会被翻转（刚度项符号正确、源项符号相反 → 同一个单元内部不一致）。

**已在真实网格上判定：本模型不触发。** 用 ``my_gmsh/gmsh_reader.py`` 直接读
``my_gmsh/model_mirror4.msh``（纯 Python 解析器，不依赖 gmsh 库）：

.. code-block:: text

    N_node=4881  N_element=9599   主动=66  从属=66  从属单元=133
    |element_current| = 144                     ← 非空，检验有效
    element_slave ∩ element_current = 0         ← 关键
    含从属节点的载流单元 = 0
    线圈单元中有节点距 θ=90° 射线 < 1e-4 m 的个数 = 0

线圈几何上避开了对称射线（每 90° 有 9 个槽、槽边界正好落在射线上），所以源项的
符号翻转不会被激活。**这是「潜伏」而非「无害」**：换 ``Z`` / ``parts`` / 线圈
几何使线圈跨过对称射线就会发作。

**本内核的处理**：保真保留 legacy 的 ``T`` 写法（本阶段铁律是不改数值行为），
把符号关系固化成测试，并提供可复现的探针 ``tools/probe_slave_current.py``
（重跑它即可判断潜伏项是否被激活）。

------------------------------------------------------------------------------
五、力学侧的范围修正
------------------------------------------------------------------------------

MMA10 力学侧（``:402-436``）复制了同一套配对与虚拟单元代码，但
``node_mech_master`` 转成 dict 后**零消费者**，``element_mech_slave`` /
``element_S_mech_slave`` 也**零消费者** —— 都是死代码。

力学侧真正生效的是 ``node_mech_slave`` 加一个 2×2 旋转矩阵（MMA10:2653-2656
定义 ``_r11.._r22``，2763-2773 写约束行），约束形如

.. code-block:: text

    u_slave = R(+2π/parts) · u_master        （笛卡尔分量，不是极坐标分量）

**结论**：本内核只提供**通用配对**（力学可直接调用 ``pair_master_slave``），
**不**搬运力学的虚拟单元死代码。力学主从的旋转耦合属于内核 16。

------------------------------------------------------------------------------
六、与内核 7 的接口
------------------------------------------------------------------------------

1. ``driving_rows`` 的 ``row_replacements`` 参数：legacy 在构造驱动行的同一段
   循环里还写了动网格插值行（MMA10:2104-2112，用 ``K_rotating_boundary`` 整行
   替换）。那属于内核 7。本内核留出这个入口，内核 7 直接接上即可，无需改动本
   内核的书写顺序。

2. ``PeriodicBoundary.rebind``：插进动网格节点层后，气隙单元里的定子侧节点号被
   换成新造的转子侧节点号（MMA10:458-463），原有三张表随之失效，
   MMA10:465-479 就地重绑。本内核提供纯函数 ``rebind(node_remap)``，节点重编号
   本身由内核 7 负责。

3. 虚拟单元表**不需要**单独的 rebind：用重编号后的 ``tri`` 与 ``rebind`` 后的
   pairing 重新调用 ``slave_virtual_elements`` 即可。这与 legacy「只重算
   ``element_rotor_gap ∩ 旧键``」等价 —— 新键只出现在被改写的转子气隙单元里，
   其余单元的连接表与从属节点集合都没变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp_sparse

__all__ = [
    "PeriodicBoundary",
    "SlaveVirtualElements",
    "DrivingRows",
    "pair_master_slave",
    "slave_virtual_elements",
    "virtual_element_signs",
    "driving_rows",
    "driving_row_residual",
    "driving_row_jacobian",
]


# =============================================================================
# 几何辅助：与 legacy 逐字一致
# =============================================================================

def _rot_x(x, y, angle):
    """绕原点旋转 angle 后的 x 分量。对应 MMA10:177-178。"""
    return x * np.cos(angle) - y * np.sin(angle)


def _rot_y(x, y, angle):
    """绕原点旋转 angle 后的 y 分量。对应 MMA10:181-182。"""
    return x * np.sin(angle) + y * np.cos(angle)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass(frozen=True)
class PeriodicBoundary:
    """扇区周期主从边界的节点配对。

    Attributes
    ----------
    slave_to_master : dict[int, int]
        从属节点 → 主动节点。对应 legacy 的 ``node_slave``（MMA10:307-317）。
    master_to_slave : dict[int, int]
        主动节点 → 从属节点。对应 legacy 的 ``node_master``（MMA10:321 把原来的
        list 覆盖成的反向表）。

    Notes
    -----
    **``master_to_slave`` 的键是「有从属节点的主动节点」，不是「主动射线上的
    所有节点」。** 二者在真实网格上恰好相等（66 = 66），但没有理由总相等：
    例如网格跨两个扇区时，θ=π 射线上的主动候选就无从属节点与之配对。
    ``virtual_element_signs`` 的成员测试用的正是这张表的**键**（因为
    ``node_i in node_master`` 对 dict 是键测试），所以这个区别有实际后果。

    legacy 里 ``node_master`` 有一个「先 list 后 dict」的双态（MMA10:305 建
    list、:321 覆盖为 dict），list 态只作为配对的遍历源。本数据结构把两者分开：
    遍历源是 ``pair_master_slave`` 的内部局部变量，不再对外暴露。
    """

    slave_to_master: Dict[int, int]
    master_to_slave: Dict[int, int]

    @property
    def master_nodes(self) -> frozenset:
        """有从属节点的主动节点集合（= ``master_to_slave`` 的键集合）。"""
        return frozenset(self.master_to_slave)

    def rebind(self, node_remap: Mapping[int, int]) -> "PeriodicBoundary":
        """按节点重编号映射刷新配对表。对应 MMA10:465-479。

        插入动网格层后，气隙单元里的定子侧节点号被换成了新造的转子侧节点号
        （MMA10:458-463），于是原来的从属/主动节点号在新单元表里不再出现。
        这里为每个「自己落在 ``node_remap`` 里的从属节点」补一对新条目：
        新从属 = 它的像，新主动 = 它主动节点的像。

        Parameters
        ----------
        node_remap : Mapping[int, int]
            旧节点号 → 新节点号。对应 legacy 的 ``node_gap_stator``（一个 dict，
            MMA10:447 处由 set 翻转为「原定子节点号 → 新转子节点号」）。

        Returns
        -------
        PeriodicBoundary
            新表 = 旧表 **加上** 新条目（legacy 是就地新增，不删旧键）。旧键指向
            已不再被单元引用的定子侧节点，保真保留。

        Notes
        -----
        legacy 的循环写作 ``for ... in node_gap_rotor.items(): if 定子节点 in
        node_slave``，即遍历所有转子动节点再筛。这里等价地遍历
        ``slave_to_master``，得到的**键值对集合相同**，只是新键的插入顺序不同
        （不影响数值）。

        legacy 只检查从属节点是否在映射里，不检查其主动节点 —— 若主动节点不在
        映射中会直接 ``KeyError``。这里显式检查并给出可读的报错。
        """
        new_slave_to_master = dict(self.slave_to_master)
        new_master_to_slave = dict(self.master_to_slave)

        for i_node_s, i_node_m in self.slave_to_master.items():
            if i_node_s not in node_remap:
                continue
            if i_node_m not in node_remap:
                raise ValueError(
                    f"节点 {i_node_s} 要重绑到 {node_remap[i_node_s]}，"
                    f"但它的主动节点 {i_node_m} 不在 node_remap 里"
                    f"（legacy 此处会抛 KeyError）"
                )
            new_s = node_remap[i_node_s]
            new_m = node_remap[i_node_m]
            new_slave_to_master[new_s] = new_m
            new_master_to_slave[new_m] = new_s

        return PeriodicBoundary(new_slave_to_master, new_master_to_slave)


@dataclass(frozen=True)
class SlaveVirtualElements:
    """含从属节点的单元，及其「把从属节点换成主动节点」后的虚拟连接表。

    Attributes
    ----------
    elements : frozenset[int]
        至少含一个从属节点的单元编号。对应 legacy 的 ``element_slave``。
    virtual_tri : dict[int, tuple[int, ...]]
        单元编号 → 虚拟单元的节点号（P1 三个、P2 六个）。对应 legacy 的
        ``element_S_slave[i][1:4]``。

    Notes
    -----
    legacy 的 ``element_S_slave`` 是 5 元组 ``[单元号, n1, n2, n3, 区域号]``，
    但**消费者只读 ``[1:4]``**（节点号）；``[0]`` 与 ``[4]`` 在全文件里零命中
    （唯一的读取者是变换矩阵函数 MMA10:1701/1755/1784/1849）。所以这里只保留
    节点号，单元号已由 dict 的键表达。

    **理论文档的硬性要求**（``同步磁阻电机磁障结构分析.md`` 约 2842 行）：
    「注意，**虚拟单元的节点只能为非从属边界上的真实节点**」。
    这是本内核最有力的外部判据 —— 若替换写错（例如把一个从属节点又映射到另一个
    从属节点），虚拟单元会退化或残留从属节点，此处即违反。
    """

    elements: frozenset
    virtual_tri: Dict[int, Tuple[int, int, int]]


@dataclass(frozen=True)
class DrivingRows:
    """驱动行装置的四个量 + 对称 Dirichlet 的列投影。字段名与 legacy 一一对应。

    Attributes
    ----------
    is_driven : (N_node,) ndarray
        ``F_is_driven``（MMA10:2074）。受约束节点的示性向量（0/1）。
    mat_driven : (N_node, N_node) csr_matrix
        ``F_mat_driven``（MMA10:2075）。受约束行上是约束表达式的系数。
    j_driven : (N_node, N_node) csr_matrix
        ``J_driven``（MMA10:2078）。**对角** 0/1 矩阵，受约束行为 0。
    j_mat_driven : (N_node, N_node) csr_matrix
        ``J_mat_driven``（MMA10:2079）。受约束行上是约束表达式的系数。
    col_mask : (N_node, N_node) csr_matrix
        **对称 Dirichlet 列投影（2026-09-24 新增）**：单位阵把**齐次 Dirichlet
        列**清零（对角保留）。``J_sym = (J_driven @ J_bulk) @ col_mask +
        J_mat_driven``。只零 Dirichlet 列 —— 反周期主从与动网格插值行本就对称
        （配对行互为转置，单行隔离实测 0 不对称贡献）。解不变（齐次 A_i=0 的列
        只乘 A_i=0；真实网格 max|ΔA|=4e-15，探针 ``probe_sym_dirichlet.py``）。

    Notes
    -----
    legacy 返回的 ``J_driven`` 是 ``lil_matrix``（``sp.eye(N).tolil()``，
    MMA10:2078 未 ``tocsr``），另两个是 ``csr``。本实现统一返回 ``csr``。

    这是**唯一**一处与 legacy 容器类型不同的地方，且**逐位无害**：
    ``J_driven`` 是对角矩阵，``J_driven @ J`` 的每个结果项
    ``Σ_k J_driven[i,k]·J[k,j]`` 只有一个非零乘积，没有浮点累加，因此存储格式与
    稀疏求和顺序都不参与结果。
    """

    is_driven: np.ndarray
    mat_driven: object
    j_driven: object
    j_mat_driven: object
    col_mask: object


# =============================================================================
# 主从配对
# =============================================================================

def pair_master_slave(
    xy: np.ndarray, parts: int, coord_err_abs: float = 1e-6
) -> PeriodicBoundary:
    """把从属边界（θ=±2π/parts）上的节点配到主动边界（θ=0）上的节点。

    对应 MMA10:304-321（电磁侧）与 ``:402-419``（力学侧，逐字同构）。

    Parameters
    ----------
    xy : (N_node, 2) ndarray
        节点坐标。**行号必须等于节点号** —— legacy 直接以节点号索引 ``node``
        表（``node[i_node_m][1]``），本函数沿用该约定。
    parts : int
        旋转对称的扇区数（真实模型 ``parts = 4``）。
    coord_err_abs : float
        坐标判等容差，对应 legacy 的 ``_coord_err_abs``（MMA10:87，实参 1e-6）。

    Returns
    -------
    PeriodicBoundary

    Notes
    -----
    判据（★ 2026-10-12 方向限定修正，见下）：

    .. code-block:: text

        主动候选:  |y| < err  且  x > err          ← θ=0 **正**射线
        从属候选:  |rot_y(x,y,−2π/parts)| < err
                   且 rot_x(x,y,−2π/parts) > err   ← θ=+2π/parts 射线

    两者都要求「在射线上」**且**「不在原点上」。

    ★ **parts=2 修正（2026-10-12，用户报「主从边界检测需改，扇区角度范围变了」）**：
    legacy 原判据用 ``err < |x|``（取绝对值），对 θ=0 与 θ=π **两条**射线同时成立。
    parts=4（90° 扇区）时 θ=π 不在域内 ⇒ 无害；parts=2（180° 扇区）时域的两条边
    恰是 θ=0 与 θ=π，**互为对方的候选** ⇒ 配出双向映射（θ=0↔θ=π 互相配对），
    ``element_slave`` 覆盖两侧、边界条件错乱。改法：方向限定为 ``x > err`` /
    ``rot_x(...) > err``（只取 +2π/parts 侧那条射线）。在 legacy 网格上（域内
    无 x<0 的射线节点）与旧判据**逐位等价** ⇒ SynRM golden 不变。

    配对时对每个从属节点遍历主动候选，**取第一个满足
    ``rotate(主动候选, +2π/parts) == 从属节点``（分量各自在容差内）者，随即
    ``break``**。取「列表顺序中的第一个」而非「最近的一个」是 legacy 的语义，
    向量化实现必须保留（见 ``fast/boundaries.py``）。在几何精确的网格上匹配是
    唯一的，两种取法一致；但索引顺序不能改。

    未配上的从属节点在 legacy 是 ``assert i_node_s in node_slave``（MMA10:319）。
    这里改为抛 ``ValueError``（条件相同，只是错误类型更明确）。
    """
    xy = np.asarray(xy, dtype=float)
    n_node = xy.shape[0]
    angle = 2 * np.pi / parts

    # 主动候选：θ=0 正射线（|y| < err 且 x > err；方向限定见 docstring ★）。
    node_master = [
        i for i in range(n_node)
        if np.abs(xy[i, 1]) < coord_err_abs < xy[i, 0]
    ]

    # 从属候选：转 −2π/parts 后落在 θ=0 **正**射线上
    node_slave_candidates = [
        i for i in range(n_node)
        if np.abs(_rot_y(xy[i, 0], xy[i, 1], -angle))
        < coord_err_abs
        < _rot_x(xy[i, 0], xy[i, 1], -angle)
    ]

    slave_to_master: Dict[int, int] = {}
    for i_node_s in node_slave_candidates:
        for i_node_m in node_master:
            if (
                np.abs(_rot_x(xy[i_node_m, 0], xy[i_node_m, 1], angle) - xy[i_node_s, 0])
                < coord_err_abs
                and np.abs(_rot_y(xy[i_node_m, 0], xy[i_node_m, 1], angle) - xy[i_node_s, 1])
                < coord_err_abs
            ):
                slave_to_master[i_node_s] = i_node_m
                break
        if i_node_s not in slave_to_master:
            raise ValueError(
                f"从属节点 {i_node_s}（坐标 {xy[i_node_s]}）在 {len(node_master)} 个"
                f"主动候选里找不到转 +2π/{parts} 后与之重合的节点"
                f"（容差 {coord_err_abs:g}）；legacy 此处是 assert 失败"
            )

    # legacy MMA10:321：node_master = dict((v, k) for (k, v) in node_slave.items())
    master_to_slave = dict((v, k) for (k, v) in slave_to_master.items())

    return PeriodicBoundary(slave_to_master, master_to_slave)


# =============================================================================
# 从属虚拟单元
# =============================================================================

def slave_virtual_elements(
    tri: np.ndarray, pairing: PeriodicBoundary
) -> SlaveVirtualElements:
    """构造从属虚拟单元表。

    对应 MMA10:323-335（电磁侧）与 ``:421-436``（力学侧，逐字同构；力学那份在
    MMA10 里是死代码，见模块 docstring 第五节）。

    Parameters
    ----------
    tri : (N_elem, 3) 或 (N_elem, 6) ndarray
        三角形连接表（节点号）。P2（6 列，gmsh 约定 ``[v0, v1, v2, m01, m12, m20]``）
        下从属判定与替换覆盖**全部 6 个槽位** —— 落在从属射线（θ=90°）上的边中点
        同样是从属节点（``pair_master_slave`` 按坐标判定，与列角色无关）。
    pairing : PeriodicBoundary

    Returns
    -------
    SlaveVirtualElements

    Notes
    -----
    两步，与 legacy 一致：

    1. ``elements`` = **全部槽位**里**至少有一个**在 ``slave_to_master`` 的键里的单元。
    2. 对这些单元，把每个从属节点号换成它的主动节点号。

    第 2 步只替换从属节点；非从属节点原样保留，因此虚拟单元的节点可能重复（例如
    两个从属节点映射到同一个主动节点，或一个单元的主动节点本来就是它的另一个
    节点）。legacy 不做去重检查，本函数同样不做 —— 但测试会断言在合法网格上
    不出现退化。

    P1（3 列）行为逐位不变：槽位循环 ``range(n_cols)`` 对 3 列与原来的
    ``(0, 1, 2)`` 是同一个迭代。
    """
    tri = np.asarray(tri)
    n_cols = tri.shape[1]
    if tri.ndim != 2 or n_cols not in (3, 6):
        raise ValueError(
            f"tri 必须是 (N_elem, 3) 或 (N_elem, 6) 的连接表，实得形状 {tri.shape}。"
            f"注意 legacy 的 element_S 是 5 列 [单元号, n1, n2, n3, 区域号]，"
            f"传它进来会让两个实现一起错（来源 C 的 ref/fast 对比抓不到这种错）"
        )
    slave_to_master = pairing.slave_to_master

    element_set = set()
    for e in range(tri.shape[0]):
        for k in range(n_cols):
            if tri[e, k] in slave_to_master:
                element_set.add(e)
                break

    virtual_tri: Dict[int, Tuple[int, ...]] = {}
    for e in element_set:
        virtual_tri[e] = tuple(
            slave_to_master[n] if n in slave_to_master else n
            for n in (int(tri[e, k]) for k in range(n_cols))
        )

    return SlaveVirtualElements(frozenset(element_set), virtual_tri)


# =============================================================================
# 虚拟单元的符号向量
# =============================================================================

def virtual_element_signs(
    virtual_nodes: Sequence[int],
    master_nodes: Iterable[int],
    a_eq: bool,
) -> np.ndarray:
    """虚拟单元的节点符号 ``p``（legacy 的 ``_ki/_kj/_km``）。

    对应 MMA10:1706-1708 与 ``:1761-1763``（v2 版在 ``:1789-1791``、``:1855-1857``）。

    Parameters
    ----------
    virtual_nodes : (…, 3) int 序列
        虚拟单元的节点号（``slave_virtual_elements`` 的输出）。
    master_nodes : 可迭代
        **有从属节点的主动节点集合**（``PeriodicBoundary.master_nodes``）。
        legacy 写 ``node_i in node_master``，而 ``node_master`` 是 dict，
        因此这是**键**测试。
    a_eq : bool
        见模块 docstring 第三节。

    Returns
    -------
    (…, 3) ndarray
        每个槽位取 ``+1``（节点在主动边界上）或 ``(1 if a_eq else −1)``。
        周期情形（``a_eq=True``）下恒为 ``+1``。

    Notes
    -----
    ★ **定义域：只能用于从属单元（虚拟单元）。** ★

    legacy 的**非从属**分支是**硬编码 ``+1``**，根本不调用这个符号公式：

    .. code-block:: python

        else:                                  # MMA10:1767-1771、1863 附近
            _T = np.eye(3)                     # 恒为单位阵

    所以对一个**非从属**单元误用本函数会得到**混合**符号模式。例如真实网格上的
    转子气隙单元 6610（``tri = [4, 7, 442]``）既在 ``element_rotor_gap`` 里、
    又含一个**配对成功的主动节点**（节点 4），本函数给 ``(1, 1, −1)``，而 legacy
    给 ``(1, 1, 1)``。

    **这不是「差一个符号」而是真错**：``TᵀKT`` 只在整体取负时才不变，混合符号会
    改变它，于是该单元的插值行有 2/3 行被反号，与内核 7 写的约束行
    ``F_mat[i, :] = K_rotating_boundary[i, :]`` 自相矛盾。

    真实网格上实测的爆炸半径（``model_mirror4.msh``，逐单元统计）：

    .. code-block:: text

        在 element_rotor_gap(823 个) 里误用        4 个   ← 内核 7 的实际风险面
        全网格（9599 个）里误用                  134 个
        其中符号模式为「混合」的                  134 个  ← 全部都会真改 TᵀKT
        其中恰好整体同号（K 仍不变）的              0 个

    即**没有一个误用情形是「碰巧无害」的**。

    正确用法是让调用方**按单元分别处理**：非从属单元直接取 ``+1``，从属单元才
    调用本函数。测试 ``test_signs_must_not_be_used_on_non_slave_elements``
    把这个区别固化成可执行事实。

    本函数与理论文档的 ``p_s`` 的关系（``T_legacy = −T_theory``，以及它成立所需
    的三条限定）、影响范围与「当前模型不触发」的实测结论，见模块 docstring
    第四节。本函数保真复刻 legacy。
    """
    master_set = set(master_nodes)
    arr = np.asarray(virtual_nodes)
    out = np.empty(arr.shape, dtype=float)
    flat_in = arr.reshape(-1, arr.shape[-1])
    flat_out = out.reshape(-1, arr.shape[-1])
    for r in range(flat_in.shape[0]):
        for k in range(flat_in.shape[1]):
            node = flat_in[r, k]
            flat_out[r, k] = 1.0 if node in master_set else (1.0 if a_eq else -1.0)
    return out


# =============================================================================
# 驱动行
# =============================================================================

def driving_rows(
    n_node: int,
    dirichlet: Iterable[int],
    slave_to_master: Mapping[int, int],
    a_eq: bool,
    row_replacements: Optional[Mapping[int, np.ndarray]] = None,
) -> DrivingRows:
    """构造驱动行装置的四个量。

    对应 MMA10:2073-2112（其中 ``:2104-2112`` 的动网格行通过
    ``row_replacements`` 由内核 7 注入）。

    Parameters
    ----------
    n_node : int
        节点数。
    dirichlet : 可迭代 int
        齐次 Dirichlet 节点集合（legacy 的 ``node_Dirichlet``，MMA10:288）。
    slave_to_master : Mapping[int, int]
        从属 → 主动。
    a_eq : bool
        周期（``True``，``A_slave = +A_master``）还是反周期（``False``）。
    row_replacements : Mapping[int, 整行], 可选
        额外「整行替换」条目：键为行号（节点号），值为该行的系数向量。用于内核 7
        的动网格插值行（legacy ``F_mat_driven[i, :] = K_rotating_boundary[i, :]``）。

    Returns
    -------
    DrivingRows

    Notes
    -----
    **写入顺序与 legacy 一致**：Dirichlet → 主从 → ``row_replacements``。
    主从分支会**先清空该行**（legacy ``F_mat_driven[i_node_s, :] = 0``，
    MMA10:2092），后写的分支覆盖先写的。``j_mat_driven`` 的主从分支在 legacy 里
    **没有**清空动作（MMA10:2099-2100 那两行是注释掉的）；但在同样的写入顺序下,
    Dirichlet 行的唯一非零元恰是 ``[i, i] = 1``、与主从分支要写的值相同，两种
    写法结果一致。本实现照 legacy 保真区分二者。

    主从行的系数：``σ = (−1 if a_eq else +1)``，即行内容为 ``A_s + σ·A_m``。

    额外整行替换条目会**同时**清空 ``f_entries`` 与 ``j_entries`` 的该行 ——
    legacy 对动网格行做的是 ``lil[i, :] = 整行``（清空后写入），两个矩阵都是。
    """
    n_node = int(n_node)
    sigma = -1.0 if a_eq else 1.0

    # (row, col) -> 值。用 dict 累加再一次性装成 COO —— 比 lil 逐项赋值快，
    # 且与 legacy 完全一致（值只有 0/±1 与注入行的系数，无浮点累加）。
    f_entries: Dict[Tuple[int, int], float] = {}
    j_entries: Dict[Tuple[int, int], float] = {}
    driven_rows = []

    # --- 齐次 Dirichlet 行（MMA10:2082-2087）---
    for i_node in dirichlet:
        f_entries[(i_node, i_node)] = 1.0
        j_entries[(i_node, i_node)] = 1.0
        driven_rows.append(i_node)

    # --- 主从行（MMA10:2090-2102）---
    for i_node_s, i_node_m in slave_to_master.items():
        # legacy: F_mat_driven[i_node_s, :] = 0
        for key in [k for k in f_entries if k[0] == i_node_s]:
            del f_entries[key]
        f_entries[(i_node_s, i_node_s)] = 1.0
        f_entries[(i_node_s, i_node_m)] = sigma
        # legacy 对 J_mat_driven 不清空，直接写这两项
        j_entries[(i_node_s, i_node_s)] = 1.0
        j_entries[(i_node_s, i_node_m)] = sigma
        driven_rows.append(i_node_s)

    # --- 额外整行替换（内核 7 的动网格插值行，MMA10:2104-2112）---
    for i_row, row_vector in (row_replacements or {}).items():
        for key in [k for k in f_entries if k[0] == i_row]:
            del f_entries[key]
        for key in [k for k in j_entries if k[0] == i_row]:
            del j_entries[key]
        row_vector = np.asarray(row_vector, dtype=float).ravel()
        if row_vector.shape[0] != n_node:
            raise ValueError(
                f"row_replacements 中第 {i_row} 行的长度是 {row_vector.shape[0]}，"
                f"应为 n_node = {n_node}"
            )
        for i_col in np.nonzero(row_vector)[0]:
            f_entries[(i_row, int(i_col))] = float(row_vector[i_col])
            j_entries[(i_row, int(i_col))] = float(row_vector[i_col])
        driven_rows.append(i_row)

    def _to_csr(entries):
        if not entries:
            return sp_sparse.csr_matrix((n_node, n_node))
        keys = list(entries.keys())
        rows = np.array([k[0] for k in keys], dtype=np.int64)
        cols = np.array([k[1] for k in keys], dtype=np.int64)
        vals = np.array([entries[k] for k in keys], dtype=float)
        return sp_sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n_node, n_node)
        ).tocsr()

    mat_driven = _to_csr(f_entries)
    j_mat_driven = _to_csr(j_entries)

    # J_driven：对角 0/1，受约束行取 0（legacy: sp.eye(N).tolil() 后逐项置 0）
    is_driven = np.zeros(n_node)
    diagonal = np.ones(n_node)
    for i_node in driven_rows:
        is_driven[i_node] = 1.0
        diagonal[i_node] = 0.0
    j_driven = sp_sparse.diags(diagonal, format="csr")

    # 对称 Dirichlet（2026-09-24）：齐次 A_i=0 ⇒ 行换 e_i 之外还要**零列 i**。
    # 只零 Dirichlet 列 —— 反周期主从 / 动网格插值行本就对称（实测 0 贡献）。
    # ``J_sym = (J_driven @ J_bulk) @ col_mask + J_mat_driven``；解不变（4e-15）。
    col_mask = sp_sparse.identity(n_node).tolil()
    for i_node in dirichlet:
        col_mask[:, i_node] = 0.0
    col_mask = col_mask.tocsr()

    return DrivingRows(
        is_driven=is_driven,
        mat_driven=mat_driven,
        j_driven=j_driven,
        j_mat_driven=j_mat_driven,
        col_mask=col_mask,
    )


def driving_row_residual(F, A, rows: DrivingRows):
    """把残差 ``F`` 的受约束行换成约束表达式。对应 MMA10:2244。

    ``F ← F * (1 - is_driven) + (mat_driven @ A) * is_driven``
    """
    return F * (1 - rows.is_driven) + (rows.mat_driven @ A) * rows.is_driven


def driving_row_jacobian(J, rows: DrivingRows):
    """把 Jacobian ``J`` 的受约束行换成约束系数行。对应 MMA10:2255。

    ``J ← (j_driven @ J) @ col_mask + j_mat_driven``。

    ⚠ **对称 Dirichlet（2026-09-24）**：在行替换之外再零**齐次 Dirichlet 列**
    （``col_mask``）—— J 由此**严格对称**（0.000000），解不变（4e-15）。见
    ``DrivingRows.col_mask`` 与设计文档 §11 的判定。
    """
    return (rows.j_driven @ J) @ rows.col_mask + rows.j_mat_driven
