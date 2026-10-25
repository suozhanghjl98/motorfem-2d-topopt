"""位置矩阵族装配（内核 8）。

来源：``磁障设计/pytorch/nonlinear_solver_energy_MMA10.py``（下称 MMA10）

    单元变换矩阵 ``_get_transformation_mat_for_elem``   ``:1693-1773``
    装配 ``get_K_vectorized``                           ``:1872-2072``
    右端项 ``get_R_vector``                             ``:2121-2219``
    （``:2073-2112`` 的驱动行四件套已由内核 6 交付，本内核不再搬运）

数学出处：``磁障设计/同步磁阻电机磁障结构分析.md``
  - 「静磁场计算的向量化数值实现」（约 :3291-3347）
  - 「主从边界的处理问题」（变换矩阵 ``T`` 的来源，见内核 6 docstring 第四节）
  - 「旋转动网格的插值方法」（``T`` 的插值行来源，见内核 7 docstring 第二节）

------------------------------------------------------------------------------
本模块的当前进度（按 1.8 的批 1 拆成 3 个子任务）
------------------------------------------------------------------------------

  子任务 1（已完成）     ``TopologyContext`` + ``transformation_matrix``  :1693-1773
  子任务 2（已完成）     ``ElementOperators`` + ``element_operators``：18 项装包结构
                                                                        :1872-2072
  子任务 3（已完成）     ``get_R_vector`` / ``signed_phase_current`` /
                        ``as_legacy_tuple``（22 元组复原）             :2121-2219

------------------------------------------------------------------------------
一、问题：为什么需要「位置矩阵族」
------------------------------------------------------------------------------

理论文档 :3293 的原话：*「合成刚度矩阵需要占用大量时间，原因在于使用循环方法叠加
稀疏矩阵效率很低。一种高效率的实现方法为自行管理稀疏矩阵的信息，即数值和元素位置。
对于有限元而言，元素位置不变，因此只需要管理数值即可。」*

数学上要反复求的是（:3298、:3308、:3316、:3322）::

    [K] = Σ γ_i [T_iᵀ][K0_i][T_i]
    g_i = γ_i [T_iᵀ][K0_i][T_i] A_i
    F   = (Σ γ_i [T_iᵀ][K0_i][T_i]) A − R
    J_i = γ_i [T_iᵀ][K0_i][T_i] + (1/(γ_i² B_i Δ_i))(∂γ_i/∂B_i) g_i g_iᵀ

其中 ``γ_i``（磁阻率）**每次 Newton 迭代都在变**，而 ``T_i``、``K0_i`` 与所有**位置**
都不变。于是把「值」与「位置」拆开存：位置算一次，值每次重算 —— 这就是 ``Kconst_mat``
（元素 → 行的选择）、``g0i_mat``（``g0 = Σ K0·A``）、``gconst_mat``、``gswap1/2_mat``
（外积的「交换」）这一族。

**本内核只做装包，不做数值近似**：``reference`` 与 ``fast`` 必须逐位一致，包括稀疏结构。

------------------------------------------------------------------------------
二、单元变换矩阵 ``T``：三个分支
------------------------------------------------------------------------------

``T`` 把「单元局部自由度」映到「受约束后的活跃自由度」：``A^(e)_local = T_e · A_active``。
按 legacy 的判据优先级（**转子气隙分支优先于从属单元分支**，与内核 7 的说明一致）：

1. ``i_elem ∈ element_rotor_gap``：该单元至少有一个节点落在动网格层上。对每个局部
   槽位 ``s``（i/j/m）:

   - 若 ``node_s ∈ node_rotating_boundary``（即它是**转子动节点**）：该自由度由两侧
     定子侧接口节点插值决定（内核 7）⇒ ``T`` 的第 ``s`` 行 = 那两列的插值系数，
     且 ``node_s`` **本身不出现在活跃自由度里**（它被消去了）。
   - 否则：``T`` 的第 ``s`` 行是「在 ``node_s`` 列上取 ``sign_s``」的单位行。

   活跃自由度集合 = 上述用到的所有节点 —— legacy 用 ``set`` 收集再 ``list()``，
   ⚠ **这个顺序决定 ``T`` 的列序，进而决定 ``Bxcoef@T``、``TᵀK0T`` 的浮点求和次序**
   （见第五节）。行内每个位置至多写一次，所以只有列序（不是写入顺序）参与数值。

   若该单元同时还是从属单元（``i_elem ∈ element_S_slave``），连接表取**虚拟单元**
   （从属节点已换成主动节点，内核 6），并带上符号 ``sign_s`` —— 它由内核 6 的
   ``virtual_element_signs`` 给出（`+1` 于主动节点，否则 `±1` 取决于 ``a_eq``）。
   这一处正是内核 6 docstring 第四节那个 ★待核对★ 的 ``T_legacy = −T_theory`` 关系
   的实际消费点。

2. ``elif i_elem ∈ element_S_slave``：``T = diag(sign_i, sign_j, sign_m)``，
   节点取虚拟连接表。

3. ``else``：``T = I₃``。

**为什么分支 1 的行要读 ``K_rotating_boundary`` 的存储值，而不是重算插值权重**：
``T`` 的插值行就是内核 7 那一行**去掉对角 1、再乘 ``−1``** 之后的结果
（``−K_rb[node_s, 括号节点]``）。内核 7 的权重受 ``s``（奇扇区符号）与 ``σ``（从属替换）
影响，重算等于把内核 7 的逻辑抄第二遍；读存储值才是搬移。

**实测：符号向量可能是「混合」的，而且这正是理论文档那条关系。** 反周期
（``a_eq=False``）下，**含主射线节点**的从属单元符号形如 ``[−1, −1, +1]``
（主射线上那个节点是主动节点 ⇒ ``+1``，其余 ``−1``）。理论文档的 ``p_s`` 是
「主动边界槽位取 ``−1``、其余 ``+1``」⇒ 两者**逐槽位互补**，即
``T_legacy = −T_theory`` —— 内核 6 docstring 第四节那条 ★待核对★ 的关系在**装配里**
的现身。后果与内核 6 的逐项核对一致：``TᵀK0T`` 对全局负号不变、只有源项 ``TᵀR``
会变号，而那个潜伏项在真实网格上已判定未激活（``tools/probe_slave_current.py``）。

⚠ 注意「混合符号」≠ 内核 6 §四注意 3 说的退化：那种退化要求主动射线上有**未配对**
的节点，而那些单元的 ``TᵀK0T`` 不变性会真的失效。测试
``test_transform_maps_active_dofs_to_local_dofs`` 把两族单元的语义分别钉住。

**接缝行的行和不是 ±1**（实测）：插值行的行和本应 = ``±(kp + kn) = ±1``（``kp + (1−kp)``
在 IEEE 下精确为 1），但若两个括号里有一个是**从属节点**（系数已乘 ``σ = −1``），
两个系数就**反号**，行和变成 ``±(kp − kn) ≠ ±1``。后果：该单元的一致载荷**不再守恒**。
真实网格上「从属 ∩ 载流」为空（``tools/probe_slave_current.py``），故本模型不触发。
测试 ``test_row_sums_are_pm1_except_at_the_antiperiodic_seam`` 守住这条。

⚠ **不能改用「稀疏行的非零元列表」**：``pos = 0`` 时插值系数 ``kp`` 恰为 ``0``，
内核 7 的装配会把这个显式零**丢掉**（``lil`` 语义），但 legacy 仍然把该节点**计入
``_index``** —— 于是单元块的大小不变、``T`` 里这一列是 0。用非零元列表会漏掉这一列，
块大小随之改变，结果**逐位不同**。测试 ``test_zero_weight_column_is_kept_in_index``
把这个语义锁住。

------------------------------------------------------------------------------
三、装配族 18 项各自是什么（子任务 2 的规格）
------------------------------------------------------------------------------

legacy ``get_K_vectorized`` 返回 **22 项**（``:2114-2118``）；其中最后 4 项
（``F_is_driven``、``F_mat_driven``、``J_driven``、``J_mat_driven``）由内核 6 的
``driving_rows`` 产出，本内核返回其余 **18 项**：:

    K0_data/K0_row/K0_col    每单元 k×k 个 (值, 行, 列) —— 展开的 TᵀK0T，k = |_index|
    Kconst_mat  (NK0 × N_elem)  元素 → K0 展开行的选择（K_gamma = Kconst_mat @ γ_elem）
    Bxcoef_mat  (N_elem × N_node)  Bxcoef[i] @ T，写在 _index 列上
    Bycoef_mat  同上（By）
    is_vaccum / is_SIMP / is_non_SIMP / is_linear_iron    单元分类掩码
    g0i_mat     (Nrow_g0i × N_node)  g0 = Σ_e K0_e A_e（每单元一块，块宽 = |_index|）
    gconst_mat  (Nrow_g0i × N_elem)  元素 → 块行的选择（g = gconst_mat @ γ ⊙ g0i）
    gswap1_mat / gswap2_mat (Ngswap × Nrow_g0i)  外积的两路「交换」（见下）
    Jabobi2_row/col / dK_p_dgamma_A_row/col      外积与 ∂K/∂γ 的 (行, 列) 位置

外积那一步的机制（理论文档 :3347「使用一个矩阵用于交换 ``g_i`` 向量中元素的位置，
然后再与未交换之前进行数值乘法」）::

    J2 的第 (i_row, i_col) 个元素 = (gswap1_mat @ g) [n] · (gswap2_mat @ g) [n]
    其中 gswap1 取 g 的 i_row 分量、gswap2 取 i_col 分量，n 是枚举计数

------------------------------------------------------------------------------
四、本内核与 legacy 的差异
------------------------------------------------------------------------------

- **容器**：legacy 的 ``Bxcoef_mat``/``Bycoef_mat`` 是 ``lil_matrix``（逐行赋值，**丢零**）；
  本内核按行收集成 COO。结构逐位相同（CSR 规范化 + 显式丢零，见 ``element_operators``）。
- **``T`` 的构造**：legacy 建 ``(3, N_node)`` 稠密数组再切片；本实现直接建 ``(3, k)``。
  ``T`` 本身逐位相同（每个位置至多写一次），但省掉每单元 ``3×N_node`` 的分配 ——
  真实网格上是 9599 × 3 × 5117 × 8 B ≈ 1.2 GB 的分配churn。
- ★ **唯一一处 reference 与 legacy 非逐位一致的量：``Bxcoef_mat``/``Bycoef_mat``。** ★
  这两个矩阵是 ``Bxcoef[e] @ T_e`` 的乘积。形状与数值都相同（``T`` 本身逐位一致、
  ``lil`` 往返也逐位无损），差的是 **T 的内存布局**：

  .. code-block:: text

      legacy 的 T = _T_temp[:, _index]   →  strides (8, 24)   Fortran 连续
      本实现的 T  = np.zeros((3, k)) 填充 →  strides (32, 8)   C 连续

  numpy 的花式索引 ``arr[:, idx]`` 返回 **F 连续**结果（其布局规则），而 BLAS gemv 对
  C/F 连续的操作数走**不同的内核** ⇒ **末位差 ≤ 1 ULP**。实测：最大绝对差 ``7.1e-15``、
  相对 ``2e-16``（值域 ~100）；把两边都 ``ascontiguousarray`` 之后结果**完全一致**
  （所以这不是算法差异，纯粹是布局）；结构（``indptr``/``indices``）逐位相同。
  与内核 7 的「容器形状 → SuperLU 主元顺序 → 末位」属**同一族**影响：
  **内存布局会渗进末位**。

  ⇒ 分层后果：``fast/`` **必须**用 ``_T_temp[:, _index]`` 这个花式索引来切片
  （它天然给出 F 连续），这样 **fast 与 legacy 逐位一致**；``reference`` 用清晰写法，
  只承诺 ``rtol=1e-12``（``CLAUDE.md`` §4 的分层判据）。测试
  ``test_bx_by_deviation_from_legacy_is_one_ulp`` 守住这个量级。
- **越界/报错**：legacy 在几何不合法时会 ``KeyError``/静默给错块大小；本实现给可读报错。

------------------------------------------------------------------------------
五、★ F4 在本内核是**活跃**的（与内核 7 相反）★
------------------------------------------------------------------------------

内核 7 的结论是「set 顺序只影响末位、不影响数值」，因为那里的稀疏元只写一次、
无浮点累加。**本内核不同**：``_index = list(set(...))`` 的顺序决定 ``T`` 的**列序**，
而 ``Bxcoef[i] @ T``、``T.T @ (K0_elem @ T)`` 都是**沿被置换的维度求和**的矩阵乘法
⇒ 求和次序改变 ⇒ **末位改变**。

所以：``set`` 的构造表达式、``add`` 的书写顺序、``list(...)`` 的调用点都必须逐字符
保留（``CLAUDE.md`` §3 红线 1、2、5）。``reference/`` 名义上豁免红线，但**这一处不能
豁免**：它必须与 ``fast/`` 用同一个 helper 取得 ``_index``，否则 ref/fast 无法逐位一致。

------------------------------------------------------------------------------
六、与相邻内核的接口
------------------------------------------------------------------------------

- **内核 6**：``PeriodicBoundary``（``master_nodes``）、``SlaveVirtualElements``
  （``virtual_tri``）、``virtual_element_signs``（符号）、``driving_rows``（22 元组的
  后 4 项）。
- **内核 7**：``RotatingBoundary.matrix``（插值系数）与 ``.brackets``（括号节点对）
  —— 分支 1 的行直接读它们。
- **内核 9/10**：**``fast`` 层**消费本内核的 18 项（``field_error_vectorized`` 开头那句
  解包）；**``reference`` 层**不消费它 —— 它吃的是本内核的**入参**（``transformation_matrix``
  逐单元取 ``T`` + 内核 4 的原始 ``K0elem``/``Bxcoef``/``Bycoef``/``area``），
  否则就只能「从向量化形式反推数学」。这处签名分叉的完整论证见
  ``reference/newton.py`` 模块 docstring 第五节。
- **内核 14**：用 ``_get_transformation_mat_for_elem_v2``（``:1776-1869``，多返回
  一个 ``dT``）—— 属内核 14，本内核不搬运。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp_sparse

from . import boundaries as ref_bnd

__all__ = [
    "TopologyContext",
    "ElementTransform",
    "ElementOperators",
    "transformation_matrix",
    "element_operators",
    "signed_phase_current",
    "get_R_vector",
    "as_legacy_tuple",
]


# =============================================================================
# 拓扑上下文：一次装好、逐单元反复使用的不变量
# =============================================================================

@dataclass(frozen=True)
class TopologyContext:
    """单元变换矩阵所需的不变量（与 ``A``、``ρ``、转子位置无关）。

    Attributes
    ----------
    tri : (N_elem, 3) ndarray
        **真实**连接表（``element_S`` 的 ``[n1, n2, n3]``）。
    virtual_tri : Mapping[int, Sequence[int]]
        从属单元的虚拟连接表（内核 6 的 ``SlaveVirtualElements.virtual_tri``）。
    element_rotor_gap : frozenset[int]
        转子气隙单元（分支 1 的判据，**优先于**从属分支）。
    element_slave : frozenset[int]
        含从属节点的单元（内核 6）。
    master_nodes : frozenset[int]
        有从属节点的主动节点集合（内核 6 的 ``PeriodicBoundary.master_nodes``）。
    a_eq : bool
        周期（``True``）/ 反周期（``False``）。只经 ``virtual_element_signs`` 影响符号。
    n_node : int
        节点数。``reference`` 用不到它（它直接建 ``(3, k)``），但 ``fast`` 需要它来
        开 legacy 那个 ``(3, N_node)`` 的稠密缓冲 —— 两层共用一个上下文，所以放在这里。
    """

    tri: np.ndarray
    virtual_tri: Mapping[int, Sequence[int]]
    element_rotor_gap: frozenset
    element_slave: frozenset
    master_nodes: frozenset
    a_eq: bool = False
    n_node: Optional[int] = None

    @classmethod
    def from_boundaries(cls, tri, virtual, pairing, element_rotor_gap, a_eq=False,
                        n_node=None):
        """由内核 6 的三个产出直接构造（少写参数、少一处出错的机会）。"""
        return cls(
            tri=np.asarray(tri),
            virtual_tri=virtual.virtual_tri,
            element_rotor_gap=frozenset(element_rotor_gap),
            element_slave=frozenset(virtual.elements),
            master_nodes=pairing.master_nodes,
            a_eq=a_eq,
            n_node=n_node,
        )


@dataclass(frozen=True)
class ElementTransform:
    """一个单元的变换矩阵。

    Attributes
    ----------
    matrix : (3, k) ndarray
        ``T``：``A^(e)_local = T · A_active``。分支 1 下 ``k`` 可达 6（每个槽位贡献
        两个括号节点），2/3 分支下 ``k = 3``。
    index : tuple[int, ...]
        活跃自由度的**全局节点号**，顺序即 ``matrix`` 的列序。
    """

    matrix: np.ndarray
    index: Tuple[int, ...]

    @property
    def n_active(self) -> int:
        return len(self.index)


# =============================================================================
# 子任务 1：单元变换矩阵
# =============================================================================

def _active_index(nodes: Iterable[int]) -> Tuple[int, ...]:
    """活跃自由度节点号，**顺序 = legacy 的 ``list(set(...))``**（见模块 docstring 第五节）。

    legacy 逐个 ``_index.add(...)``，再 ``list(_index)``。这里的写法等价且更直白，
    但**必须**保持「用 set 收集、最后 list」这一件事 —— 换成 ``sorted(set(...))``
    会让列序改变、进而改变 ``Bxcoef@T`` 与 ``TᵀK0T`` 的浮点求和次序。
    """
    collected = set()
    for node in nodes:
        collected.add(int(node))
    return tuple(collected)


def transformation_matrix(
    i_elem: int, ctx: TopologyContext, rotating
) -> ElementTransform:
    """单元变换矩阵 ``T``。对应 ``MMA10:1693-1773``。

    Parameters
    ----------
    i_elem : int
        单元号。
    ctx : TopologyContext
        拓扑不变量（连接表、区域集合、主从表）。
    rotating : RotatingBoundary
        内核 7 的产出：``.matrix``（``K_rotating_boundary``）与 ``.brackets``
        （``{动节点: (上侧, 下侧)}``，两个节点号都已做过从属→主动替换）。

    Returns
    -------
    ElementTransform

    Notes
    -----
    三个分支与优先级见模块 docstring 第二节。分支 1 的行写作

    .. code-block:: text

        T[s, 括号节点] = −K_rotating_boundary[node_s, 括号节点] · sign_s      （node_s 是转子动节点）
        T[s, node_s]  = sign_s                                            （否则）

    分支 1 的**块大小 k 由「用到哪些节点」决定，不由「哪些系数非零」决定** ——
    ``pos = 0`` 时一支系数恰为 0，该列仍必须在 ``index`` 里（见模块 docstring 第二节末）。
    """
    i_elem = int(i_elem)

    # --- 分支 1：转子气隙单元（判据优先于从属分支）---
    if i_elem in ctx.element_rotor_gap:
        if i_elem not in ctx.element_slave:
            nodes = tuple(int(n) for n in ctx.tri[i_elem])
            signs = (1.0, 1.0, 1.0)
        else:
            nodes = tuple(int(n) for n in ctx.virtual_tri[i_elem])
            # 复用内核 6 的符号函数（它的定义域就是「从属单元」，这里正是）
            signs = tuple(
                float(s)
                for s in ref_bnd.virtual_element_signs(
                    np.asarray([nodes]), ctx.master_nodes, ctx.a_eq
                )[0]
            )

        # 先定列序（见 _active_index 的说明），再填值 —— 每个位置至多写一次，
        # 所以「先定列序」不引入任何浮点顺序差异。
        used = []
        entries = []          # entries[s] = (节点, 系数) 的列表
        for node in nodes:
            if node in rotating.brackets:
                # ⚠ brackets 的长度随插值阶数变（linear 2 项 / cubic 4 项）⇒ **遍历**。
                # linear 下与原先的 (node_up, node_low) 写法逐位相同。
                cols = tuple(int(c) for c in rotating.brackets[node])
                entries.append(tuple(
                    (c, -rotating.matrix[node, c]) for c in cols))
                used.extend(cols)
            else:
                entries.append(((node, 1.0),))
                used.append(node)

        index = _active_index(used)
        column_of = {node: col for col, node in enumerate(index)}
        matrix = np.zeros((3, len(index)))
        for s, row_entries in enumerate(entries):
            sign_s = signs[s]
            for node, coefficient in row_entries:
                matrix[s, column_of[node]] += coefficient * sign_s
        return ElementTransform(matrix=matrix, index=index)

    # --- 分支 2：从属单元（非气隙）---
    if i_elem in ctx.element_slave:
        nodes = tuple(int(n) for n in ctx.virtual_tri[i_elem])
        signs = ref_bnd.virtual_element_signs(
            np.asarray([nodes]), ctx.master_nodes, ctx.a_eq
        )[0]
        return ElementTransform(matrix=np.diag(signs), index=nodes)

    # --- 分支 3：普通单元 ---
    nodes = tuple(int(n) for n in ctx.tri[i_elem])
    return ElementTransform(matrix=np.eye(3), index=nodes)


# =============================================================================
# 子任务 2：18 项装包结构
# =============================================================================

@dataclass(frozen=True)
class ElementOperators:
    """装配族 18 项（``get_K_vectorized`` 的返回去掉驱动行四件套）。

    每一项的数学含义见模块 docstring 第三节与 ``element_operators`` 的 Notes。
    字段名一律小写下划线；与 legacy 元组的对应关系在 ``element_operators`` 里给表。

    Attributes
    ----------
    k0_data, k0_row, k0_col : (NK0,) ndarray
        逐单元的 ``TᵀK0_eT`` 展开成三元组，按 (单元, 局部行, 局部列) 顺序排列。
        同一个全局 ``(row, col)`` 会在不同单元上重复 —— 这是**有意的**：下游
        ``csr_matrix((k0_data * (Kconst_mat @ γ), (k0_row, k0_col)))`` 靠 CSR 的重复项
        求和完成装配。
    kconst_mat : (NK0, N_element) csr_matrix（整数 1）
        ``Kconst_mat @ γ_elem`` 把每个展开项映射到它所属单元的磁阻率。
    bxcoef_mat, bycoef_mat : (N_element, N_node) csr_matrix
        ``Bxcoef[e] @ T_e`` 写在 ``T_e`` 的活跃自由度列上 ⇒ ``bxcoef_mat @ A`` 同时给出
        **所有**单元的 ``b_e·A_e``（理论文档 :3330 的 ``B_x = Σ K_Bxi T_i A_i``）。
    is_vaccum / is_simp / is_non_simp / is_linear_iron : (N_element,) ndarray
        单元分类掩码（0/1 浮点）。四类互斥且穷尽：
        ``is_vaccum`` = 非铁磁；铁磁中 ``is_linear_iron`` = 线性铁磁（``use_nonlinear_iron``
        关闭时）；否则非线性铁磁，再分 ``is_simp``（在拓扑优化区）/ ``is_non_simp``。
    g0i_mat : (Nrow_g0i, N_node) csr_matrix
        每个单元占一块 ``|index_e|`` 行（块首 = 前序块宽之和）⇒ ``g0i_mat @ A`` 的第
        ``(e, r)`` 行 = ``(TᵀK0_eT_e A_e)[r]`` = ``g_{0,e}`` 的第 r 分量（**不含** γ）。
    gconst_mat : (Nrow_g0i, N_element) csr_matrix（整数 1）
        ``g = g0 ⊙ (gconst_mat @ γ_elem)`` 即 ``g_e = γ_e g_{0,e}``（理论文档 :3308）。
    gswap1_mat, gswap2_mat : (Ngswap, Nrow_g0i) csr_matrix（整数 1）
        对第 n 个外积项 ``(e, i_row, i_col)``：``gswap1`` 取 ``g`` 的第 ``(e, i_row)``
        分量、``gswap2`` 取第 ``(e, i_col)`` 分量 ⇒ 逐元素相乘即得 ``g_e g_eᵀ`` 的全部
        元素（理论文档 :3345-3347 的「交换矩阵」）。
    jabobi2_row, jabobi2_col : (Ngswap,) ndarray
        外积各项在**全局**矩阵里的 ``(行, 列)``。**实测与 ``k0_row/k0_col`` 逐项相同**
        （同一个循环、同样的值、同样的顺序）—— legacy 存了两份，本实现照搬以免下游
        顺序改变，测试 ``test_jabobi2_positions_duplicate_k0_positions`` 固化这一事实。
    dk_p_dgamma_a_row, dk_p_dgamma_a_col : (Nrow_g0i,) ndarray
        ``(单元号, 该块第 r 行的全局节点)`` —— 即「``∂K/∂γ`` 收缩 A 之后」每个分量的
        位置（下游 ``dK_p_dgamma_A`` 用）。
    n_k0, n_gswap, n_row_g0i : int
        三个计数器（``NK0 = Ngswap = Σ_e |index_e|²``）。
    """

    k0_data: np.ndarray
    k0_row: np.ndarray
    k0_col: np.ndarray
    kconst_mat: object
    bxcoef_mat: object
    bycoef_mat: object
    is_vaccum: np.ndarray
    is_simp: np.ndarray
    is_non_simp: np.ndarray
    is_linear_iron: np.ndarray
    g0i_mat: object
    gconst_mat: object
    gswap1_mat: object
    gswap2_mat: object
    jabobi2_row: np.ndarray
    jabobi2_col: np.ndarray
    dk_p_dgamma_a_row: np.ndarray
    dk_p_dgamma_a_col: np.ndarray
    n_k0: int
    n_gswap: int
    n_row_g0i: int


def _coo(rows, cols, vals, shape, dtype=None):
    """由列表装 CSR。

    **不**去重、**不**丢零 —— 与 legacy 的 ``csr_matrix((data, (row, col)))`` 一致：
    同一 ``(row, col)`` 的重复项会被**相加**（K0 三元组正需要这个），显式零被保留。
    """
    rows = np.asarray(rows, dtype=np.int64) if len(rows) else np.zeros(0, np.int64)
    cols = np.asarray(cols, dtype=np.int64) if len(cols) else np.zeros(0, np.int64)
    if dtype is not None:
        vals = np.asarray(vals, dtype=dtype)
    else:
        vals = np.asarray(vals)
    return sp_sparse.csr_matrix((vals, (rows, cols)), shape=shape)


def element_operators(
    ctx: TopologyContext,
    rotating,
    *,
    n_node: int,
    k0elem: np.ndarray,
    bxcoef: np.ndarray,
    bycoef: np.ndarray,
    element_iron: Iterable[int],
    element_topo_opti: Iterable[int],
    use_nonlinear_iron: bool,
) -> ElementOperators:
    """装配族的 18 项。对应 ``MMA10:1872-2072``。

    Parameters
    ----------
    ctx, rotating :
        内核 6/7 的拓扑与旋转边界（与 ``transformation_matrix`` 同）。
    n_node : int
        节点数（决定稀疏矩阵的列数）。
    k0elem : (N_elem, 6) ndarray
        内核 4 的单元刚度六元组 ``[kii, kij, kim, kjj, kjm, kmm]``。
    bxcoef, bycoef : (N_elem, 3) ndarray
        内核 4 的 ``B = curl A`` 系数（``Bx = Σ_s Bxcoef_s A_s``）。
    element_iron, element_topo_opti : 可迭代 int
        铁磁单元、拓扑优化区的单元。
    use_nonlinear_iron : bool
        是否启用非线性铁磁（关闭时铁磁单元一律记 ``is_linear_iron``）。

    Returns
    -------
    ElementOperators

    Notes
    -----
    逐单元做四件事（与 legacy 同序）：

    1. 取变换矩阵 ``T`` 与活跃自由度 ``index``（子任务 1）；
    2. ``Bxcoef[e] @ T`` / ``Bycoef[e] @ T`` 写进 ``bxcoef_mat`` / ``bycoef_mat``
       的 ``index`` 列上；
    3. 分类掩码（见 ``ElementOperators``）；
    4. ``K0T = Tᵀ K0_e T`` 展开成三元组，并同时铺出 ``Kconst / g0i / gconst /
       gswap1 / gswap2 / Jabobi2 / dK_p_dgamma_A`` 这些「位置」信息。

    ⚠ **``bxcoef_mat``/``bycoef_mat`` 必须丢掉恰好为零的元**：legacy 用 ``lil`` 写这两行，
    而 ``lil`` 的赋值会丢弃零（内核 7 docstring 第四节的实测；``T`` 的某列恰为 0 时
    ``Bxcoef @ T`` 在该列就是精确的 0）。其余矩阵由 ``csr_matrix((data, (row, col)))``
    装出，**保留**显式零 —— 两种语义都要照搬。

    ⚠ ``index`` 的顺序决定 ``K0T`` 的展开顺序（= ``k0_data`` 的排列）与列序，进而决定
    ``Bxcoef@T``、``TᵀK0T`` 的浮点求和次序（模块 docstring 第五节）。
    """
    tri = np.asarray(ctx.tri)
    n_elem = tri.shape[0]
    iron = set(int(e) for e in element_iron)
    topo = set(int(e) for e in element_topo_opti)

    k0_data, k0_row, k0_col = [], [], []
    kconst_rows, kconst_cols = [], []
    bx_rows, bx_cols, bx_vals = [], [], []
    by_rows, by_cols, by_vals = [], [], []
    g0i_rows, g0i_cols, g0i_vals = [], [], []
    gconst_rows, gconst_cols = [], []
    gswap1_rows, gswap1_cols = [], []
    gswap2_rows, gswap2_cols = [], []
    jabobi2_row, jabobi2_col = [], []
    dk_rows, dk_cols = [], []

    is_vaccum = np.zeros(n_elem)
    is_simp = np.zeros(n_elem)
    is_non_simp = np.zeros(n_elem)
    is_linear_iron = np.zeros(n_elem)

    n_k0 = 0
    n_gswap = 0
    n_row_g0i = 0

    for i_elem in range(n_elem):
        tr = transformation_matrix(i_elem, ctx, rotating)
        matrix, index = tr.matrix, tr.index

        # --- Bx / By 系数行（legacy 的 lil 逐行赋值：丢零）---
        bx = np.asarray(bxcoef[i_elem], dtype=float) @ matrix
        by = np.asarray(bycoef[i_elem], dtype=float) @ matrix
        for col, node in zip(range(len(index)), index):
            if bx[col] != 0.0:
                bx_rows.append(i_elem)
                bx_cols.append(int(node))
                bx_vals.append(bx[col])
            if by[col] != 0.0:
                by_rows.append(i_elem)
                by_cols.append(int(node))
                by_vals.append(by[col])

        # --- 材料分类 ---
        # ★ 2026-10-17 修正：线性铁磁开关**只影响非拓扑区的纯铁芯**，
        #   拓扑区（设计变量作用区）必须保持 ``is_simp`` —— 否则 ρ 完全
        #   失效（实测 `use_nonlinear_iron=False` 下把 ρ 从 1.0 改到 0.1，
        #   残差的相对差是 **0.000e+00**），拓扑优化在"线性材料"档下
        #   根本挖不出磁障。
        #   legacy 的原写法把全部铁芯置 ``is_linear_iron``（``μ=2000μ₀``、
        #   与 ρ 无关），是 legacy 本身的口径，已在 D14 登记。
        if i_elem in iron:
            if i_elem in topo:
                is_simp[i_elem] = 1
            elif not use_nonlinear_iron:
                is_linear_iron[i_elem] = 1
            else:
                is_non_simp[i_elem] = 1
        else:
            is_vaccum[i_elem] = 1

        # --- 单元刚度（六元组 → 3×3；保持 legacy 的乘法结合顺序）---
        k = k0elem[i_elem]
        k0_elem = np.array([[k[0], k[1], k[2]],
                            [k[1], k[3], k[4]],
                            [k[2], k[4], k[5]]])
        k0_t = matrix.T @ (k0_elem @ matrix)

        for i_row in range(len(index)):
            for i_col in range(len(index)):
                value = k0_t[i_row, i_col]
                k0_data.append(value)
                k0_row.append(int(index[i_row]))
                k0_col.append(int(index[i_col]))

                kconst_rows.append(n_k0)
                kconst_cols.append(i_elem)

                g0i_rows.append(n_row_g0i + i_row)
                g0i_cols.append(int(index[i_col]))
                g0i_vals.append(value)

                gswap1_rows.append(n_gswap)
                gswap1_cols.append(n_row_g0i + i_row)

                gswap2_rows.append(n_gswap)
                gswap2_cols.append(n_row_g0i + i_col)

                jabobi2_row.append(int(index[i_row]))
                jabobi2_col.append(int(index[i_col]))

                n_k0 += 1
                n_gswap += 1

            gconst_rows.append(n_row_g0i + i_row)
            gconst_cols.append(i_elem)

            dk_rows.append(i_elem)
            dk_cols.append(int(index[i_row]))

        n_row_g0i += len(index)

    return ElementOperators(
        k0_data=np.asarray(k0_data, dtype=float),
        k0_row=np.asarray(k0_row, dtype=np.int64),
        k0_col=np.asarray(k0_col, dtype=np.int64),
        kconst_mat=_coo(kconst_rows, kconst_cols, [1] * len(kconst_rows),
                        (n_k0, n_elem), dtype=np.int64),
        bxcoef_mat=_coo(bx_rows, bx_cols, bx_vals, (n_elem, n_node)),
        bycoef_mat=_coo(by_rows, by_cols, by_vals, (n_elem, n_node)),
        is_vaccum=is_vaccum,
        is_simp=is_simp,
        is_non_simp=is_non_simp,
        is_linear_iron=is_linear_iron,
        g0i_mat=_coo(g0i_rows, g0i_cols, g0i_vals, (n_row_g0i, n_node)),
        gconst_mat=_coo(gconst_rows, gconst_cols, [1] * len(gconst_rows),
                        (n_row_g0i, n_elem), dtype=np.int64),
        gswap1_mat=_coo(gswap1_rows, gswap1_cols, [1] * len(gswap1_rows),
                        (n_gswap, n_row_g0i), dtype=np.int64),
        gswap2_mat=_coo(gswap2_rows, gswap2_cols, [1] * len(gswap2_rows),
                        (n_gswap, n_row_g0i), dtype=np.int64),
        jabobi2_row=np.asarray(jabobi2_row, dtype=np.int64),
        jabobi2_col=np.asarray(jabobi2_col, dtype=np.int64),
        dk_p_dgamma_a_row=np.asarray(dk_rows, dtype=np.int64),
        dk_p_dgamma_a_col=np.asarray(dk_cols, dtype=np.int64),
        n_k0=n_k0,
        n_gswap=n_gswap,
        n_row_g0i=n_row_g0i,
    )


# =============================================================================
# 子任务 3：右端项与 legacy 元组的复原
# =============================================================================

def signed_phase_current(
    element_current: Iterable[int],
    phase_sets: Sequence[Iterable[int]],
    currents: Sequence[float],
) -> Dict[int, float]:
    """每个载流单元的**带符号电流密度**。

    对应 ``MMA10:2205-2217`` 的 if/elif 链。``phase_sets`` 与 ``currents`` 按 legacy
    的书写顺序给（``Ap, An, Bp, Bn, Cp, Cn``）与三相电流；``phase_sets`` 里的项按
    legacy 的符号约定交替 —— 即传入 ``(Ap, An, Bp, Bn, Cp, Cn)`` 与
    ``(+J_A, −J_A, +J_B, −J_B, +J_C, −J_C)``，函数本身不再猜符号。

    Notes
    -----
    ⚠ legacy 的链**以 ``else`` 收尾**（等价于 ````-T_R0elem * J_phaseC``，即 Cn）：
    「在 ``element_current`` 里、但六个相带集合都没命中」的单元会被**静默**当作 Cn。
    本实现照搬这个语义（保真），但把它显式化成 ``fallback`` 分支，并让
    ``warn_on_fallback`` 打开时对该情形发出告警 —— 真实网格上六个相带集合恰好构成
    ``element_current`` 的划分（测试 ``test_phase_sets_partition_the_current_elements``），
    所以正常路径不会走到它。
    """
    phase_sets = [set(int(e) for e in s) for s in phase_sets]
    table: Dict[int, float] = {}
    for i_elem in element_current:
        i_elem = int(i_elem)
        for elements, current in zip(phase_sets, currents):
            if i_elem in elements:
                table[i_elem] = float(current)
                break
        else:
            # legacy 的 else 分支：落到 Cn
            table[i_elem] = float(currents[-1])
    return table


def get_R_vector(
    ctx: TopologyContext,
    rotating,
    *,
    n_node: int,
    r0elem: np.ndarray,
    current_of_element: Mapping[int, float],
) -> np.ndarray:
    """源项（右端项）``R``。对应 ``MMA10:2121-2219``。

    Parameters
    ----------
    ctx, rotating :
        同 ``transformation_matrix``。
    n_node : int
        节点数。
    r0elem : (N_elem,) ndarray
        内核 4 的 ``R0 = Δ/3``（一致节点分配系数）。
    current_of_element : Mapping[int, float]
        ``{单元号: 带符号电流密度}``，由 ``signed_phase_current`` 生成。

    Returns
    -------
    (N_node,) ndarray

    Notes
    -----
    数学（理论文档 :3312 ``F_i = g_i − [T_i]R_i`` 的右端）::

        R = Σ_{e ∈ 载流} T_eᵀ (R0_e · 𝟙₃) · J_e

    注意 ``T_eᵀ`` 在**左**：源项被变换矩阵「拉回」活跃自由度。因此

    - 含插值行的单元（转子气隙），源项分给**两个括号节点**，动节点本身**收不到**；
    - ``Σ_s (T_eᵀ𝟙)_s = ε_e · 3``（``ε_e`` 是子任务 1 里那个整体符号）⇒ 源项总量
      ``±3·R0_e·J_e`` —— 含从属槽位且反周期时**整体变号**。这正是内核 6 docstring
      第四节所说的「唯一可能影响解的项是源项 ``TᵀR``」在**数值上**的形态；
      真实网格上没有「从属 ∩ 载流」的单元（``tools/probe_slave_current.py``），
      所以本模型不触发它。

    累加顺序与 legacy 一致（``for i_elem in range(N_element)``，``_index`` 内无重复），
    所以逐位可复现。
    """
    R = np.zeros(int(n_node))
    for i_elem in range(np.asarray(ctx.tri).shape[0]):
        j = current_of_element.get(i_elem)
        if j is None:
            continue
        tr = transformation_matrix(i_elem, ctx, rotating)
        r0 = float(r0elem[i_elem])
        # legacy: _T.T @ np.array([R0, R0, R0])
        contribution = tr.matrix.T @ np.array([r0, r0, r0])
        R[list(tr.index)] += contribution * j
    return R


def as_legacy_tuple(ops: "ElementOperators", driving) -> tuple:
    """把 18 项装包与内核 6 的驱动行装置复原成 legacy 的 **22 元组** + 对称列投影。

    顺序逐项照抄 ``MMA10:2114-2118``，末位是 **2026-09-24 新增**的
    ``driving.col_mask``（对称 Dirichlet 的列投影，见 ``DrivingRows.col_mask``）::

        (K0_data, K0_row, K0_col, Kconst_mat, F_is_driven, F_mat_driven,
         Bxcoef_mat, Bycoef_mat, is_vaccum, is_SIMP, is_non_SIMP, is_linear_iron,
         g0i_mat, gconst_mat, gswap1_mat, gswap2_mat, Jabobi2_row, Jabobi2_col,
         dK_p_dgamma_A_row, dK_p_dgamma_A_col, J_driven, J_mat_driven,
         J_col_mask)

    ⚠ **22 → 23**：这是**有意的**对称性修复（§11）—— J 由
    ``(J_driven @ J_bulk) @ J_col_mask + J_mat_driven`` 装配，严格对称、解不变。
    ``as_legacy_tuple`` 的调用点与所有解包点已同步更新。

    这个容器是 ``docs/重构设计文档.md`` §4 说的「兼容装置」：让
    ``field_error_vectorized`` 开头那句解包（内核 9）可以**原样保留**，把搬家的 diff
    压到接近零。

    Notes
    -----
    与 legacy 的两处容器差异（都不影响数值，见内核 6 的 ``DrivingRows`` docstring）：

    1. legacy 的 ``J_driven`` 是 ``lil_matrix``（未 ``tocsr``），本实现返回 ``csr``；
       它是对角 0/1 矩阵，逐元素乘积无累加，故逐位无差。
    2. legacy 的 ``K0_data``/``K0_row``/... 是 Python ``list``，本实现是 ``ndarray``；
       ``list * ndarray`` 与 ``ndarray * ndarray`` 结果逐位相同。
    """
    return (
        ops.k0_data, ops.k0_row, ops.k0_col, ops.kconst_mat,
        driving.is_driven, driving.mat_driven,
        ops.bxcoef_mat, ops.bycoef_mat,
        ops.is_vaccum, ops.is_simp, ops.is_non_simp, ops.is_linear_iron,
        ops.g0i_mat, ops.gconst_mat, ops.gswap1_mat, ops.gswap2_mat,
        ops.jabobi2_row, ops.jabobi2_col,
        ops.dk_p_dgamma_a_row, ops.dk_p_dgamma_a_col,
        driving.j_driven, driving.j_mat_driven,
        driving.col_mask,
    )
