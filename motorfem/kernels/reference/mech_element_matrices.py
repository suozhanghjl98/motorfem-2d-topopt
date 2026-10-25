"""力学单元矩阵与载荷（内核 5）。

来源：``legacy/pytorch/nonlinear_solver_energy_MMA10.py:550-613``（矩阵）
      与 ``:619-659``（载荷）

------------------------------------------------------------------------------
数学表述
------------------------------------------------------------------------------

一阶三角形常应变单元（CST），**平面应力**（plane stress）假定，
每个节点 2 个自由度（u, v），单元共 6 个自由度。

1) 应变-位移矩阵 ``L``（FE 文献里通常叫 ``B``），形状 ``(3, 6)``
   自由度顺序为 ``(u_i, v_i, u_j, v_j, u_m, v_m)``::

       εx  = ∂u/∂x      → L[0, 0]=b_i/(2Δ), L[0, 2]=b_j/(2Δ), L[0, 4]=b_m/(2Δ)
       εy  = ∂v/∂y      → L[1, 1]=c_i/(2Δ), L[1, 3]=c_j/(2Δ), L[1, 5]=c_m/(2Δ)
       γxy = ∂u/∂y+∂v/∂x→ L[2, 0]=c_i/(2Δ), L[2, 1]=b_i/(2Δ), 同理其余两组

2) 弹性矩阵 ``D``（平面应力），形状 ``(3, 3)``
   ::

       D[0,0] = D[1,1] = E/(1−ν²)
       D[0,1] = D[1,0] = E·ν/(1−ν²)
       D[2,2] = E/(2(1+ν))

   ⚠ **平面应力**，不是平面应变。平面应变应为
   ``E(1−ν)/((1+ν)(1−2ν))`` 形式。

   **出处**：``磁障设计/弹性力学与有限元.md`` 的「平行平面场中的方程」
   一节（约第 291-300 行）明确给出了两类问题的区分，原文：
   「第一类问题为平面应力问题，认为在垂直于参考面上的边界为零应力边界，
   如薄板上的应力问题。该类问题会产生 ε_zz」。本模块采用的是这一类。

   legacy 把 D **逐单元复制**存储为 ``E0mechcoef(N,3,3)``，而 D 只依赖材料、
   与单元无关。本模块只存一份 ``D(3,3)``——数值完全相同，只是不再冗余复制。
   测试会断言 legacy 的每个副本都等于本模块的这一份。

3) 单元刚度
   ::

       K_e = Δ · Lᵀ · D · L        ∈ R^{6×6}

4) 体积力右端向量
   ::

       Rmech10elem[e] = −Δ/2

   即 ``MechElements.R0``。⚠⚠ **这个系数需要核对**，见下方专节。

5) 边界线元右端向量
   ::

       Rmech20elem[l] = −L_l/2      L_l 为线元长度

   即 ``mech_line_loads(...).R0``。

6) 载荷
   ::

       fmechload[e]  = ρ·ω²·(centroid_e − rot_center)              离心体积力
       Fmechload[l]  = rot45( (x_i+x_j)/2 · Fload/r̄ · L_l,
                              (y_i+y_j)/2 · Fload/r̄ · L_l ) / ΣL  边界面力

   其中 ``r̄`` 是该线元两端点坐标之和的模的一半，``rot45`` 是绕原点转 45°，
   ``ΣL`` 是加载边界总长（归一化）。分别见 ``mech_body_force`` 与
   ``mech_line_loads``。

   ⚠ **这两支是批 5 补上的**：本模块的第一版只交付了单元矩阵（``K/L/D/R0``），
   第 4~6 条的公式写了却没有实现，而内核 16 的装配族要把它们当输入读。
   在 2026-09-22 补全之前，仓库里**没有任何**这三个载荷的可调用实现。

------------------------------------------------------------------------------
关于 Rmech10elem = −Δ/2 的来源与疑点（★ 需要你确认）
------------------------------------------------------------------------------

**该系数在理论文档里是有推导的**，不是凭空来的。
``磁障设计/弹性力学与有限元.md``「有限元求解方法 › 三角形单元线性形函数
的情况」一节（约第 443-475 行）给出::

    Π̄₁⁽ᵉ⁾ = (Δ/2)(σ̄xx·ε̄xx + σ̄yy·ε̄yy + τ̄xy·γ̄xy) − (Δ/2)(f_x·ū_x + f_y·ū_y)

    R₁ = −(Δ/2)·[f_x, f_y, f_x, f_y, f_x, f_y]ᵀ

    Π̄₁⁽ᵉ⁾ = ½·uᵀ(Δ·[L]ᵀ[E][L])u + R₁ᵀ·u

代码的 ``Rmech10elem = −Δ/2`` 与之**逐字一致**。

**但读推导过程，这一项疑似有一处沿用错误。**

文档第 385 行定义 ``ū_x = N_i u_{xi}``（求和约定），即 ``ū_x`` 是**形函数
插值出的场量**。而 ``R₁ = −(Δ/2)·f·[1,1,1]`` 等价于把 ``ū_x`` 取成
``Σ_i u_{xi}``——也就是**把形函数当成 1、而不是对它们积分**。

变分上正确的一致载荷应当对形函数积分::

    三角形:  ∫ₑ N_i dΩ = Δ/3      ⇒  每节点 F_i = f·Δ/3，总和 f·Δ
    线单元:  ∫ₑ N_i dl = L/2      ⇒  每节点 F_i = f·L/2，总和 f·L

**关键旁证**：同一份文档对**线单元**给出的边界载荷是

    R₂ = −(L/2)·[F_x, F_y, F_x, F_y]ᵀ      （均匀分布的情形，约第 517 行）

这与 ``∫N_i dl = L/2`` **完全吻合，是对的**（代码的
``Rmech20elem = −L/2`` 也对上）。文档对线元同时给出了点载荷的
``−(L/6)(2F_i + F_j)`` 形式，同样是教科书标准式。

**所以文档在 1D 上是对的、在 2D 上给了 1/2**——很像是把线元的 ``L/2``
直接类比过来，而没有重新对三角形的形函数积分。

**影响量级**：每节点 ``−Δ/2`` vs 正确的 ``−Δ/3``，即 **1.5 倍**。
符号可被残差约定吸收（``Ku + R = 0`` ⇒ ``R = −F``），**但幅值 1.5 倍
不能被任何约定吸收**。若确有误，机械应力会被系统性放大，进而影响增广
拉格朗日应力约束的标定（``max_stress_constraint``、λ/μ 的初值与上限）。

**本模块的处理**：保真保留 ``−Δ/2``（阶段 1 的目标是与 legacy 位级一致），
并在测试 ``test_R0_factor_is_documented`` 中把实际倍率 ``−1.5`` 固化下来。

**★★ 已确认为笔误（作者 2026-09-22 确认），待修正 ★★**

修正计划（不在阶段 1 做，因为阶段 1 的目标是与 legacy 位级一致）：
  1. 修正代码：``Rmech10elem`` 由 ``−Δ/2`` 改为 ``−Δ/3``
  2. 同步修正理论文档 ``磁障设计/弹性力学与有限元.md`` 约 446 与 468 行
     的 ``(Δ/2)`` 与 ``R₁`` 两式（那里同源）
  3. **重新标定应力约束**：``max_stress_constraint``（legacy 50e7）、
     增广拉格朗日的 ``lambda0`` / ``mu0`` / ``mu_max``，以及
     ``weight_density`` 等权重——因为载荷缩小 2/3 后全场应力会同比例变化
  4. 修正测试：``test_R0_factor_is_documented`` 会失败，届时更新其期望倍率

**修正前必须先冻结基线**，否则无法量化"修正前后"的差异。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MechMaterial",
    "MechElements",
    "MechLineLoads",
    "mech_element_matrices",
    "mech_body_force",
    "mech_line_loads",
    "MECH_LOAD_NOTE",
    "R0ELEM_FACTOR",
    "R2ELEM_FACTOR",
]

#: ``Rmech10elem = −Δ/2`` 的标量因子（legacy 原值，来源未记录）
R0ELEM_FACTOR = -1.0 / 2.0

#: ``Rmech20elem = −L/2`` 的标量因子。**这个是对的** —— 与线元的
#: ``∫N_i dl = L/2`` 一致（见 ``MECH_LOAD_NOTE`` 的旁证段落）。
R2ELEM_FACTOR = -1.0 / 2.0

#: 关于该因子的说明，供其它模块引用
MECH_LOAD_NOTE = (
    "Rmech10elem = -Δ/2：每节点施加 f·(-Δ/2)，总和 -1.5·f·Δ，"
    "而三角形一致体积力载荷应为每节点 f·Δ/3（总和 f·Δ）。"
    "符号可能被残差约定吸收，但 1.5 倍幅值不能。已保真保留，待核对。"
)


@dataclass(frozen=True)
class MechMaterial:
    """平面应力线弹性材料常数。

    Attributes
    ----------
    E : float 杨氏模量，Pa
    nu : float 泊松比
    rho : float 密度，kg/m³（离心体积力用）
    """

    E: float = 200e9
    nu: float = 0.30
    rho: float = 7850.0


@dataclass(frozen=True)
class MechElements:
    """力学单元级预计算量。

    Attributes
    ----------
    L : (N, 3, 6) 应变-位移矩阵（每个单元不同）
    D : (3, 3) 平面应力弹性矩阵（**只存一份**，与单元无关）
    K : (N, 6, 6) 单元刚度 ``Δ·LᵀDL``
    R0 : (N,) 体积力右端系数 ``−Δ/2``（见 ``MECH_LOAD_NOTE``）
    area : (N,) 单元面积 Δ
    tri : (N, 3) 连接表
    """
    L: np.ndarray
    D: np.ndarray
    K: np.ndarray
    R0: np.ndarray
    area: np.ndarray
    tri: np.ndarray

    @property
    def n_elem(self) -> int:
        return self.area.shape[0]


def mech_element_matrices(
    node: np.ndarray, tri: np.ndarray, material: MechMaterial = MechMaterial()
) -> MechElements:
    """计算力学单元矩阵。

    Parameters
    ----------
    node : (N_node, 2) 节点坐标（力学子网格）
    tri : (N_elem, 3) 三角形连接表，**必须逆时针**
    material : MechMaterial

    Returns
    -------
    MechElements

    Notes
    -----
    逐单元循环，与 legacy ``MMA10:557-603`` 一一对应。
    Δ 的定向检查在此处显式做（力学网格与电磁网格是两套网格，不能假定
    前者已被电磁侧检查过）。
    """
    n_elem = tri.shape[0]
    E, nu = material.E, material.nu

    x = node[tri, 0]
    y = node[tri, 1]
    b = np.roll(y, -1, axis=1) - np.roll(y, -2, axis=1)
    c = -np.roll(x, -1, axis=1) + np.roll(x, -2, axis=1)
    delta = (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]) / 2.0

    if np.any(delta <= 0.0):
        raise ValueError(
            f"力学网格有 {int(np.sum(delta <= 0.0))} 个单元面积非正（须逆时针编号）"
        )

    # --- 弹性矩阵 D（平面应力）—— 只算一份 ---
    D = np.zeros((3, 3))
    D[0, 0] = E / (1 - nu ** 2)
    D[0, 1] = E * nu / (1 - nu ** 2)
    D[1, 0] = E * nu / (1 - nu ** 2)
    D[1, 1] = E / (1 - nu ** 2)
    D[2, 2] = E / 2 / (1 + nu)

    L = np.zeros((n_elem, 3, 6))
    K = np.zeros((n_elem, 6, 6))
    R0 = np.zeros(n_elem)

    for e in range(n_elem):
        bi, bj, bm = b[e]
        ci, cj, cm = c[e]
        Dlt = delta[e]

        # 应变-位移矩阵：保持 legacy 的 1/(2*Delta) 写法
        inv2d = 1 / (2 * Dlt)
        L[e, 0, 0] = inv2d * bi
        L[e, 0, 2] = inv2d * bj
        L[e, 0, 4] = inv2d * bm

        L[e, 1, 1] = inv2d * ci
        L[e, 1, 3] = inv2d * cj
        L[e, 1, 5] = inv2d * cm

        L[e, 2, 0] = inv2d * ci
        L[e, 2, 1] = inv2d * bi
        L[e, 2, 2] = inv2d * cj
        L[e, 2, 3] = inv2d * bj
        L[e, 2, 4] = inv2d * cm
        L[e, 2, 5] = inv2d * bm

        K[e] = Dlt * L[e].T @ D @ L[e]

        # ⚠ 见 MECH_LOAD_NOTE
        R0[e] = -Dlt / 2

    return MechElements(L=L, D=D, K=K, R0=R0, area=delta, tri=tri)


# =============================================================================
# 载荷：离心体积力与边界面力（legacy :605-660）
# =============================================================================
#
# ⚠ **本节的来历**：内核 5 的模块 docstring 第 4~6 条**一直**写着这三个载荷的公式，
# 但第一版只交付了单元矩阵（``K/L/D/R0``），载荷没落地 —— 而内核 16 的装配族
# （``get_Kmech_vectorized``）把它们当输入读。批 5 开工时发现这个缺口，在此补齐。
# 它们属于内核 5 而不是内核 16：与 ``Rmech10elem`` 同族，是**单元/几何级**的量，
# 不含任何装配或求解语义。

@dataclass(frozen=True)
class MechLineLoads:
    """边界（线元）载荷的产出。

    Attributes
    ----------
    R0 : (N_line,) ndarray
        ``Rmech20elem`` = ``−L_l/2``。**只在加载线元上非零** —— legacy 先把整条
        数组置零、再对 ``element_mech_L_load`` 逐个赋值（``:605-613``）。
    fx, fy : (N_load,) ndarray
        逐**加载**线元的等效节点力分量，**已按加载边界总长归一化**（``:660-661``）。
    boundary_length : float
        加载边界总长（归一化用的分母，legacy 的 ``_len_boundary``）。

    Notes
    -----
    ``N_line`` 是**全部**线元数（含未加载的），``N_load`` 是加载线元数 ——
    两者不同，这是 legacy 的形状（``Rmech20elem`` 与 ``Fmechloadx`` 长度不等）。
    """

    R0: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    boundary_length: float


def mech_body_force(
    node: np.ndarray,
    tri: np.ndarray,
    material: MechMaterial = MechMaterial(),
    *,
    angular_speed: float,
    rot_center=(0.0, 0.0),
) -> tuple:
    """逐单元离心体积力 ``fmechload``。对应 ``MMA10:620-632``。

    逐单元（形心为 ``(x̄, ȳ)``，旋转中心为 ``(x_c, y_c)``）::

        f_x = ρ·ω²·(x̄ − x_c)        f_y = ρ·ω²·(ȳ − y_c)

    Parameters
    ----------
    node : (N_node, 2) ndarray
    tri : (N_elem, 3) ndarray
    material : MechMaterial
        只用它的 ``rho``。
    angular_speed : float
        角速度 ω（rad/s）。legacy 由 ``rot_speed_rpm / 30 * np.pi`` 得到（``:94``），
        即 ``2π·rpm/60`` ✓ 正确。本函数只收 ω，换算留在调用方（它是 config 的事）。
    rot_center : (float, float)
        旋转中心。legacy 是 ``rot_center_x = rot_center_y = 0``（``:96-97``）。

    Returns
    -------
    (fx, fy) : 两个 ``(N_elem,)`` ndarray

    Notes
    -----
    ⚠ 形心写成 ``xi/3 + xj/3 + xm/3`` —— **照抄 legacy 的三项和**，
    不是 ``(xi+xj+xm)/3``（结合顺序不同 ⇒ 末位可能不同）。
    """
    fx = np.empty(tri.shape[0])
    fy = np.empty(tri.shape[0])
    m = material.rho
    xc, yc = rot_center

    for e in range(tri.shape[0]):
        ni, nj, nm = int(tri[e, 0]), int(tri[e, 1]), int(tri[e, 2])
        xi, yi = node[ni]
        xj, yj = node[nj]
        xm, ym = node[nm]
        fx[e] = m * angular_speed ** 2 * (xi / 3 + xj / 3 + xm / 3 - xc)
        fy[e] = m * angular_speed ** 2 * (yi / 3 + yj / 3 + ym / 3 - yc)

    return fx, fy


def mech_line_loads(
    node: np.ndarray,
    line_tri: np.ndarray,
    loaded_lines,
    *,
    Fload: float,
    rot_angle_deg: float = 45.0,
) -> MechLineLoads:
    """边（线元）载荷。对应 ``MMA10:605-613``（系数）与 ``:639-661``（力）。

    Parameters
    ----------
    node : (N_node, 2) ndarray
    line_tri : (N_line, 2) ndarray
        全部线元的端点节点号（legacy 的 ``element_L_mech`` 去掉编号列）。
    loaded_lines : 可迭代 int
        **加载**线元的行号（legacy 的 ``element_mech_L_load``，是 ``line_tri`` 的索引）。
    Fload : float
        面力幅值。⚠ legacy 生产配置里 ``Fload = 0``（``MMA10:99``）—— 这一项
        在真实运行中恒为零，但它是 config 旋钮，照搬。
    rot_angle_deg : float
        力的旋转角（度）。legacy 硬编码 ``45 * np.pi / 180``（``:648-649``）。

    Returns
    -------
    MechLineLoads

    Notes
    -----
    两条量共用同一个循环的两半：

    1. **系数** ``R0[i] = −L_i/2``（``:611-613``）。这是线元一致载荷系数，
       与 ``∫N dl = L/2`` 吻合 —— ``MECH_LOAD_NOTE`` 用来佐证 2D 那一处是笔误。
    2. **面力**（``:642-661``）::

           p = (x_i+x_j)/2 · Fload / r̄ · L_l        （同理 p_y）
           (f_x, f_y) = rot(p_x, p_y, 45°)
           最后整体除以加载边界总长

       其中 ``r̄ = |(x_i+x_j, y_i+y_j)| / 2`` —— **两端点坐标之和的模的一半**，
       不是形心到原点的距离。

    ``loaded_lines`` 为空时不做归一化除法（legacy 会 ``/0``，但被除的是空数组，
    结果同为「空数组」）。
    """
    line_tri = np.asarray(line_tri, dtype=int)
    loaded = [int(i) for i in loaded_lines]
    rot_angle = rot_angle_deg * np.pi / 180

    R0 = np.zeros(line_tri.shape[0])
    fx = np.zeros(len(loaded))
    fy = np.zeros(len(loaded))
    boundary_length = 0.0

    for i_load, i_line in enumerate(loaded):
        ni, nj = int(line_tri[i_line, 0]), int(line_tri[i_line, 1])
        xi, yi = node[ni]
        xj, yj = node[nj]

        length = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        R0[i_line] = R2ELEM_FACTOR * length

        r_bar = np.sqrt((xi + xj) ** 2 + (yi + yj) ** 2) / 2
        # legacy: _rot((x_i+x_j)/2 * Fload / _r * _l, (y_i+y_j)/2 * Fload / _r * _l, 45°)
        px = (xi + xj) / 2 * Fload / r_bar * length
        py = (yi + yj) / 2 * Fload / r_bar * length
        fx[i_load] = px * np.cos(rot_angle) - py * np.sin(rot_angle)
        fy[i_load] = px * np.sin(rot_angle) + py * np.cos(rot_angle)

        boundary_length += length

    if loaded:
        fx = fx / boundary_length
        fy = fy / boundary_length

    return MechLineLoads(
        R0=R0, fx=fx, fy=fy, boundary_length=float(boundary_length))
