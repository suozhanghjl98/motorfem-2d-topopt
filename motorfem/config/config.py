"""motorfem.config —— 阶段 2：参数对象（14 个 dataclass 聚合为 RunConfig）

把 legacy ``MMA10:71-170`` 的 ~65 个配置全局量按语义分成 14 个 dataclass
（设计文档 §阶段 2）。legacy 是「模块级全局量 + 调用点随手传字面量」，无法序列化、
无法参数化、混用 mm/m 靠 ``node[:,1:] *= 1e-3`` 兜底 —— 参数对象解决这三点：

* **字段名带单位后缀**（``_m`` / ``_pa`` / ``_rpm``）：SI 单位写死，杜绝混用。
* **JSON 可序列化**：字段只用 int/float/bool/str/tuple/list/None 或嵌套 dataclass，
  **不放 ndarray**。``to_json/from_json`` 支持往返。
* **``from_legacy(globals_dict)``**：从 MMA10 模块命名空间快照构造。每个字段的
  ``field(metadata={"legacy": 原名})`` 记录映射，从 legacy 取值时不靠猜。

**字段默认值 = legacy 生产值**：``RunConfig()`` 的默认构造即生产配置
（与 ``golden/manifest.json`` 的 module_scalars 一致，测试锚定防漂移）；
``from_legacy`` 按 metadata 从快照覆盖。这使默认值自文档化且可跑。

**折并**：``use_type{1,2}_stress`` / ``use_type{1..4}_rel_stress`` 六布尔折成
``StressType`` / ``RelStressType`` 两个枚举（见 ``enums.py``），是阶段 2 唯一
改结构的参数改动。

**不动的量**：纯运行时推导量（``N_node``/``N_element``/``area_current`` 等）
放进对应组，由 ``from_legacy`` 从快照取；MMA 算法参数（``MMA_dgrad`` 等）不在
配置块里，来自 ``MMA_optimize`` 签名默认值与 ``MMA10:4292-4295`` 调用处的
字面量，``MMAConfig`` 直接按生产生效值固化并注明来源。

⚠ 不用 ``from __future__ import annotations``：反射（``dataclasses.is_dataclass``）
依赖运行时注解是真类，PEP 563 的字符串化注解会破坏它。
"""
import dataclasses
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .enums import (RelStressType, StressType, collapse_rel_stress_type,
                    collapse_stress_type)

__all__ = [
    "GeometryConfig", "MaterialConfig", "WindingSpec", "ElectricalConfig",
    "DiscretizationConfig", "TimeConfig", "TopologyConfig", "FilterConfig",
    "ObjectiveConfig", "ALConfig", "StressConfig", "MechConfig",
    "SolverConfig", "MMAConfig", "OutputConfig", "RuntimeConfig", "RunConfig",
]


def _fl(legacy: str, default: Any) -> Any:
    """字段工厂：legacy 原名（from_legacy 按此覆盖）+ 生产默认值。"""
    return field(default=default, metadata={"legacy": legacy})


@dataclass
class GeometryConfig:
    """几何：外径/转子外径/轴径（SI 米）、极槽、周期份数。"""

    pole_pairs: int = _fl("pole_pairs", 2)            # 极对数
    Z: int = _fl("Z", 36)                             # 槽数
    parts: int = _fl("parts", 4)                      # 周期份数
    stator_outer_radius_m: float = _fl("m_R", 72.5e-3)    # 定子外径
    rotor_outer_radius_m: float = _fl("m_Rrotor", 44.75e-3)  # 转子外径
    shaft_radius_m: float = _fl("m_Rirotor", 15.0e-3)     # 轴径
    coord_err_abs: float = _fl("_coord_err_abs", 1e-6)    # 区域识别容差
    # ⚠ 不参与 legacy（无 ``_fl``）：legacy 里定子内径是**硬编码常量 45mm**。
    #   本字段是它的显式化 —— 纯默认值 ⇒ from_legacy 跳过、legacy 值不变。
    #   用途：**动网格层半径** ``m_R_gap = (stator_inner_radius_m +
    #   rotor_outer_radius_m)/2``（legacy 44.875mm、6s4p 44.75mm）。
    #   ⚠ 它**不**派生 Arkkio 气隙 —— 那是独立的 ``arkkio_gap_m``（见下），
    #   因为 ``stator_inner − rotor_outer`` 在浮点上有末位差。
    stator_inner_radius_m: float = 45.0e-3                # 定子内径（= 气隙外侧）
    # Arkkio 转矩气隙宽度（米）。None = 用 legacy 硬编码常量 0.25e-3（逐位不变）；
    # 6s4p 显式设 0.5e-3。⚠ **不要**用 ``stator_inner − rotor_outer`` 派生：
    # 45e-3−44.75e-3 在浮点下 = 0.0002500000000000002 ≠ 0.25e-3（末位差），
    # 会经 ``1/gap`` 破坏 ``Torque_Arkkio_coef`` 的 golden 逐位一致
    # （2026-10-12 实测 4 个 fem golden 测试因此失败）。
    arkkio_gap_m: Optional[float] = None
    # ★ 2026-10-12：Arkkio 系数 `parts²/gap` 里那个多余 `parts` 的开关。
    #   1 = legacy 原式（默认，36 槽 golden 逐位不变）；0 = 理论自然式 parts/gap
    #   （6s4p 用）。去掉多余 parts 后，同极对数下「槽数 → 转矩」的影响降到 ~20%
    #   量级（只余气隙磁密² 的差），见 kernels/reference/torque.py §二 与
    #   tools/_probe_strut.py 的对照。
    arkkio_parts_exponent: int = 1
    # ★ 6s4p（2026-10-12）：转子设计域 = 半个极（0..45°）。
    #   True 时 build_field_topology 把 core_s 中 q2(45..90°)/q3(90..135°) 也归约到
    #   core_m（关于 45° 镜像 / 旋转 −90°），设计域缩为半个极 —— 与 36 槽 legacy 的
    #   「element_topo_opti_m = 半个极」同构，改善优化对称性/收敛。
    #   False（默认）= legacy：设计域 = core_m（36 槽天然就是半个极）。
    rotor_design_half_pole: bool = False


@dataclass
class MaterialConfig:
    """材料：磁常数、B-H 曲线、力学材料参数。"""

    mu0: float = _fl("mu0", 4 * math.pi * 1e-7)  # 真空磁导率
    #: **B-H 曲线文件**（可替换：放一个新 CSV，改这里即可）。
    #: 解析规则见 `resolve_bh_csv`：含 `:`（盘符）或以 `/` 开头 ⇒ 当绝对路径；
    #: 否则按 `legacy/pytorch/<值>` 解析。
    #: ⚠ 换曲线 = **换铁芯材料**（物理上不同的问题），`use_nonlinear_iron` 开关
    #:   与本项无关（那个开关是把铁芯换成**常量 μ**，会关掉 ρ 依赖，见 §7 说明）。
    bh_curve_csv: str = "50CS470.csv"                # B-H 曲线文件（无 legacy 全局名）
    # B-H 曲线的**查表求值**表点数（approx 层，见 kernels/approx/bh_lut.py）。
    #   0  = 关闭 → scipy PCHIP，legacy 路径，参与 golden 位级对齐；
    #   >0 = 开启 → uniform-x 表 + 精确仿射尾支，**放弃位级一致**。
    # 实测（2026-09-29，9599 单元）：field_error 2.44→1.74 ms（1.40×），
    # 整轮 Newton 250→242 ms（1.034×）；Newton 轮数不变，逐点 μ 相对差 2e-6。
    # ⚠ 开启后：① `tools/verify_golden.py` 会失配（它已显式钉死 PCHIP，故需放开才看得到）；
    #   ② **autograd oracle 层直接报错**（`autograd/field.py` 读 `curve._H_of_B` 重建
    #   精确 PCHIP，不能用近似替身）。实测：默认打开会挂 8 个测试。
    # ⚠ 故意**不**用 `_fl`：本字段在 legacy 里没有对应物，走 `_fl` 会让
    #   `from_legacy` 去 legacy 命名空间里找它并 KeyError（见 `_from_legacy_fields`）。
    #   纯默认值 ⇒ `from_legacy` 跳过它、保留 0（= legacy 行为），语义正好。
    bh_lut_n: int = 65536
    mech_E_pa: float = _fl("mech_E", 200e9)          # 杨氏模量
    mech_nu: float = _fl("mech_nu", 0.30)            # 泊松比
    mech_rho_kg_m3: float = _fl("mech_rho", 7850)    # 密度


@dataclass
class WindingSpec:
    """绕组：极槽 + 电流 + 载流区面积 + star-of-slots 生成器参数。

    阶段 5 把相带表从手写（``MMA10:270-277``）升级为生成器推导
    （``motorfem/winding/layout.py``）。``inner_ref/outer_ref/outer_shift`` 是
    生成器的两套 legacy 表复现参数（SynRM 生产默认；PMSM 用 start/center/0）。
    默认值与 golden 锚定（``verify_winding_layout`` 断言生成器推导 == tag 表）。
    """

    Z: int = _fl("Z", 36)
    pole_pairs: int = _fl("pole_pairs", 2)
    parts: int = _fl("parts", 4)
    current_amp: float = _fl("current_amp", 5)       # 相电流幅值（A）
    N_conductor: int = _fl("N_conductor", 20)        # 每槽导体数
    # ⚠ 每极每相载流区面积：legacy 初值 0 是占位，模块体运行后被几何重算
    #（golden 快照 = 9.9e-05）。默认值取生产运行后的真实值。
    area_current_m2: float = _fl("area_current", 9.9e-05)
    # star-of-slots 生成器参数（阶段 5）：槽参考位置与外层槽号偏移
    inner_ref: str = "edge"          # SynRM：内层 = 槽结束边相带
    outer_ref: str = "edge"          # SynRM：外层 = 全局槽 k+1 的结束边相带
    outer_shift: int = 1             # SynRM：双层 1 槽错位
    # ⚠ 纯默认值（无 legacy 对应物 ⇒ from_legacy 跳过、保留 False）。
    #   False = 相带由 legacy 的 18 槽硬编码表合成（golden 逐位不变）；
    #   True  = 相带改由 star-of-slots（Z, pole_pairs, parts, inner/outer_ref,
    #           outer_shift）推导 —— 换极槽/换电机的唯一开关（6s4p 用它，
    #           参数 ``inner_ref='start', outer_ref='center', outer_shift=0``）。
    phase_from_generator: bool = False


@dataclass
class ElectricalConfig:
    """电气（绕组）。"""

    winding: WindingSpec = field(default_factory=WindingSpec)


@dataclass
class DiscretizationConfig:
    """离散化：节点/单元规模（运行时由网格推导，from_legacy 从快照取）。"""

    N_node: int = _fl("N_node", 0)
    N_element: int = _fl("N_element", 0)
    N_element_mech: int = _fl("N_element_mech", 0)


@dataclass
class TimeConfig:
    """时间：仿真时长与采样。"""

    time_simulation_s: float = _fl("time_simulation", 10e-3 / 9)
    N_point_simulation: int = _fl("N_point_simulation", 11)
    N_point_sampled: int = _fl("N_point_sampled", 13)
    use_random_sample_method: bool = _fl("use_random_sample_method", False)
    # ⚠ 转子扫描采样设置（2026-10-11 新增，legacy 无对应物 ⇒ 不用 _fl）。
    #   默认 uniform = legacy 均匀时间点（golden 逐位不变）；random/table 改变
    #   采样 ⇒ **放弃位级一致**（新功能路径）。验证集（test-train 分离）：
    #   use_validation_set 时额外生成 n_validation_points 个与训练**不重叠**的
    #   时间点，objective 优化只用训练点，验证点由 FieldSystem.evaluate(
    #   subset="val") 求值（评估真实脉动，防过拟合训练点）。
    sampling_mode: str = field(default="uniform", metadata={
        "doc": "转子扫描采样：uniform（legacy 均匀，golden）/ random（随机种子）/ "
               "table（显式转子位置度表）"})
    random_seed: int = field(default=0, metadata={
        "doc": "sampling_mode=random 的随机种子（np.random.RandomState）"})
    positions_deg: Optional[List[float]] = field(default=None, metadata={
        "doc": "sampling_mode=table 的显式转子扫描位置表（度）。t 由位置反解，"
               "电流相位自动对齐；None=未设置"})
    use_validation_set: bool = field(default=False, metadata={
        "doc": "生成验证集（不参与优化，只用于评估）"})
    n_validation_points: int = field(default=6, metadata={
        "doc": "验证集时间点数。uniform 模式在时间轴均匀取；random 模式用 "
               "validation_seed 随机取，均避开训练点"})
    validation_seed: int = field(default=1, metadata={
        "doc": "验证集随机种子（uniform 模式也用它给半周期偏移）"})
    # ★ 2026-10-12：转子初始位置（度）。legacy 硬编码 22.5°（``:3447``）。
    #   纯默认值（不用 ``_fl``）⇒ from_legacy 跳过、legacy 逐位不变。
    #   设 0 可让 th=0 时转子**不转**，便于画「定转子对齐」的磁场图。
    rotor_position0_deg: float = 22.5
    # ★ 2026-10-12：电流 A 相相位偏移（度，电角）。legacy 里 ``phaseA_position =
    #   100πt``（无偏移），配合 rotor_position0=22.5° 给出**功角 −pole_pairs×22.5°**。
    #   改 ``rotor_position0_deg`` 时须同时给本偏移 ``pole_pairs×(θ0_new−22.5°)``，
    #   才能**保持功角不变**（否则转矩不可比）。纯默认 0 ⇒ legacy 逐位不变。
    phaseA_offset_deg: float = 0.0


@dataclass
class TopologyConfig:
    """拓扑优化：SIMP 投影、密度初值、密度约束形式。"""

    p_SIMP: float = _fl("p_SIMP", 3)
    min_SIMP: float = _fl("min_SIMP", 1e-6)
    SIMP_beta: float = _fl("SIMP_beta", 8)           # tanh 投影陡度
    SIMP_thb: float = _fl("SIMP_thb", 0.5)
    SIMP_linear_factor: float = _fl("SIMP_linear_factor", 0.0)
    max_density: float = _fl("max_density", 0.65)    # 密度上限
    rho_init_radius_m: float = _fl("rho_init_radius", 1.25e-3)     # 初始空洞半径
    rho_init_distance_m: float = _fl("rho_init_distance", 4.0e-3)  # 初始空洞间距
    use_poly_constraint: bool = _fl("use_poly_constraint", True)   # 密度约束多项式形式


@dataclass
class FilterConfig:
    """密度/梯度滤波。"""

    if_filter_rho: bool = _fl("if_filter_rho", True)
    if_filter_grad: bool = _fl("if_filter_grad", True)
    filter_radius_m: float = _fl("filter_radius", 2e-3)
    filter_p_exp: float = _fl("filter_p_exp", 2)
    filter_grad_radius_m: float = _fl("filter_grad_radius", 0.4e-3)
    filter_grad_p_exp: float = _fl("filter_grad_p_exp", 0.5)


@dataclass
class ObjectiveConfig:
    """目标：转矩/波动/密度的权重与 p-范数。"""

    mean_torque_obj: float = _fl("mean_torque_obj", 90)
    max_torque_ripple: float = _fl("max_torque_ripple", 5)
    p_exp_torque_ripple: float = _fl("p_exp_torque_ripple", 10)

    # ★ 2026-10-14：**平均转矩 / 纹波分别用哪个口径**（本轮转矩验证的落地）。
    #
    # 背景（详见 `docs/转矩口径等价链核对与动网格插值_2026-10-13.md` §0/§4.4）：
    #   三个口径在"细网格 + cubic"下已验证（181 点 vs ANSYS）：
    #     T_flux  —— 平均转矩最准（与 ANSYS 差 **0.06%**），但**脉动全丢**
    #                （dq 投影只保基波：脉动峰峰 0.55 vs ANSYS 9.39，且反相，§4.5）；
    #     T_ArkBE —— 网格稳健、**脉动准**（波形相关 0.996），与 T_VD 互吻 0.34%；
    #     T_Arkkio（legacy 生产）—— 有 h 标定因子，**在细网格上偏 45%**（§4.9）。
    #   ⇒ **平均用 flux、纹波用 ArkBE**；legacy 口径经 `RunConfig.legacy()` 回落。
    mean_torque_source: str = field(default="flux", metadata={
        "doc": "平均转矩口径：'flux'（默认，最准）| 'arkkio'（legacy，逐位）"})
    ripple_source: str = field(default="arkbe", metadata={
        "doc": "纹波口径（con1）：'arkbe'（默认，网格稳健）| 'arkkio'（legacy，逐位）"
               "| 'vd'（用 con4 那套 T_VD）"})
    # ⚠ 脉动范数 p 连续化（continuation）：先低 p 平滑、后高 p 精化。
    #   背景：p=10 的 `|T−T_obj|^p` 高度非凸（梯度 ∝|T−T_obj|^9，argmax 时间点
    #   切换 → MMA 一阶凸逼近失稳 → 密度锯齿），3000 步仍不收敛（见
    #   `docs/全量优化记录_2026-09-29.md` §3/§6）。legacy 早版（MMA3/4/5）用过
    #   p=3/4，升到 10 是后来的事 —— 低 p 才是常态。
    #   连续化 = 从 `p_exp_start` 线性爬升到 `p_exp_torque_ripple`（`p_exp_ramp_steps`
    #   步内），之后固定。先低 p 让 MMA 找到结构，再高 p 收紧脉动约束。
    # ⚠ 三个字段 legacy 无对应物 ⇒ **不用 _fl**（否则 from_legacy 会 KeyError，
    #   参照 `bh_lut_n` 的处理）。默认关 = p 恒为 `p_exp_torque_ripple`，行为逐位
    #   不变（golden L4 锚定不受影响）。
    p_exp_continuation: bool = field(default=False, metadata={
        "doc": "脉动范数 p 连续化开关（False=固定 p_exp_torque_ripple，保 golden）"})
    p_exp_start: float = field(default=3.0, metadata={
        "doc": "连续化起始 p（低 → 平滑；legacy MMA4/MMA5 生产值就是 3）"})
    p_exp_ramp_steps: int = field(default=300, metadata={
        "doc": "连续化爬升步数：p 在 [p_exp_start, p_exp_torque_ripple] 线性爬升，"
               "之后固定"})
    # ⚠ p 连续化配套的**移动极限联动**（2026-10-10 探针结论）：p 爬到高值时
    #   约束梯度 ∝|T−T_obj|^(p−1) 对设计极敏感，固定的 MMA_move 会让 MMA
    #   overshoot → con1 反弹。p 每爬升 `p_exp_move_step`，MMA_move ×
    #   `p_exp_move_factor`（<1 = 收紧）。只影响 p_exp_continuation=True 的
    #   路径；默认关时 `_current_move` 恒返回 MMA_move，golden 逐位不变。
    p_exp_move_linkage: bool = field(default=False, metadata={
        "doc": "p 连续化时是否同步收紧 MMA 移动极限（默认关 = cont 行为；"
               "开 = contm 行为）。见 p_exp_move_step/factor"})
    p_exp_move_step: float = field(default=1.0, metadata={
        "doc": "p 每爬升多少，移动极限收窄一档（默认 1 = 每升 1 个指数收窄）"})
    p_exp_move_factor: float = field(default=0.7, metadata={
        "doc": "移动极限每档的收窄系数（<1=收紧；0.7 起，p 从 3→10 共收窄 ~0.7^7≈0.08"})
    torque_ripple_computed_around_obj_torque: bool = _fl(
        "torque_ripple_computed_around_obj_torque", True)
    weight_torque: float = _fl("weight_torque", 100)
    weight_torque_ripple: float = _fl("weight_torque_ripple", 10)
    weight_density: float = _fl("weight_density", 1)
    Tflux_ratio: float = _fl("Tflux_ratio", 0.5)
    use_random_mean_torque: bool = _fl("use_random_mean_torque", False)
    # ⚠ 目标/约束使能与角色分配（2026-10-10 新增，legacy 无对应物 ⇒ 不用 _fl）。
    #   6 个参与项（2 机械目标 + 4 电磁约束）中，`objective_role` 指定哪项进
    #   f[0] 作目标（minimize），其余启用项进约束（≤0），禁用项不进 f。
    #   - `enable_uelastic`/`enable_stress` **都关**时跳过整个力学段
    #     （`mech_worker` 不调用，obj 机械项恒 0，AL 冻结）—— 省一次力学求解
    #     + 2 个伴随 spsolve，纯电磁约束下的可行设计搜索。
    #   - `objective_role` 默认 `"uelastic+stress"` = legacy 组合目标（golden 逐位
    #     不变）；可选 `"uelastic"/"stress"/"con1"/"con2"/"con3"/"con4"` 单选。
    #     选 con3（转矩）作目标 = 最小化 (1−T/T_obj)·w = 最大化转矩。
    #   f 行数 = 1 目标 + 参与约束数（默认 5 行 = golden；禁用后行数可变）。
    objective_role: str = field(default="uelastic+stress", metadata={
        "doc": "目标选择：uelastic+stress（legacy 组合）/ uelastic / stress / "
               "con1 / con2 / con3 / con4"})
    enable_uelastic: bool = field(default=True, metadata={
        "doc": "应变能参与（作目标或约束）；与 enable_stress 都关 ⇒ 跳过力学段"})
    enable_stress: bool = field(default=True, metadata={
        "doc": "应力 AL 罚参与；与 enable_uelastic 都关 ⇒ 跳过力学段"})
    enable_con1: bool = field(default=True, metadata={
        "doc": "Arkkio 转矩波动约束参与"})
    enable_con2: bool = field(default=True, metadata={
        "doc": "平均密度约束参与"})
    enable_con3: bool = field(default=True, metadata={
        "doc": "平均转矩约束参与"})
    enable_con4: bool = field(default=True, metadata={
        "doc": "VD 转矩波动约束参与"})
    # ⚠ 约束逐步激活调度（2026-10-12 新增，legacy 无对应物 ⇒ 不用 _fl）。
    #   背景：con3 目标 + con1（绕固定 T_obj 的 p=10 脉振）在 x0 处梯度反平行
    #   死锁（cos≈−0.999，见设计文档 §11）。若先只保留密度约束让转矩升起来
    #   （此时 con1 梯度与目标不再反平行，cos→+0.19 @T≈87），再激活脉振约束，
    #   可绕开死锁。默认关 = 所有启用约束立即参与，行为与 legacy 逐位一致。
    constraint_schedule: bool = field(default=False, metadata={
        "doc": "约束逐步激活开关（False=启用项立即全部参与，保 golden）"})
    con1_activate_step: int = field(default=0, metadata={
        "doc": "con1（Arkkio 脉振）激活步数：前 N 步不参与（只保留 con2 密度 + "
               "目标），第 N 步起参与。0=立即参与"})
    con4_activate_step: int = field(default=0, metadata={
        "doc": "con4（VD 脉振）激活步数：同 con1_activate_step。0=立即参与"})
    con3_activate_step: int = field(default=0, metadata={
        "doc": "con3（平均转矩 ≥ T_obj）激活步数：前 N 步不参与。⚠ 仅当 con3 作"
               "约束时有意义（objective_role='con3' 时目标是 con3，必须始终参与，"
               "调度自动不覆盖）。0=立即参与"})
    stress_activate_step: int = field(default=0, metadata={
        "doc": "应力 AL 罚项（目标的组成 2）激活步数：前 N 步目标 = 纯弹性应变能"
               "（uelastic），第 N 步起 + 应力罚项。用于 6 个量逐个进场的完整调度"
               "（uelastic 恒参与作目标基座）。0=立即 = legacy 组合目标，golden 不变"})
    uelastic_activate_step: int = field(default=0, metadata={
        "doc": "应变能作**约束**的激活步数。⚠ 只对**非组合目标**的角色（如 "
               "objective_role='con3'）有效 —— 组合目标（'uelastic+stress'）下 "
               "uelastic 恒作目标基座、本字段无效（见 objective.py 的组合目标分支，"
               "2026-10-12 review）。0=立即"})
    objective_role_phase2: Optional[str] = field(default=None, metadata={
        "doc": "目标角色切换（2026-10-12，用户方案：先转矩再机械最后脉动）："
               "``objective_role_switch_step`` 起目标从 ``objective_role``（如 con3 "
               "最大化转矩）切到本字段（如 'uelastic+stress' 组合目标，转矩变 ≥90 约束"
               "保持）。None = 不切换，golden 不变"})
    objective_role_switch_step: int = field(default=0, metadata={
        "doc": "目标角色切换步数：前 N 步用 objective_role，第 N 步起用 "
               "objective_role_phase2"})
    # ⚠ 脉振目标连续化（2026-10-12，配合 constraint_schedule）：con1 激活瞬间
    #   若转矩已远超 T_obj，脉振违规巨大（5 点训练集实测 con1=288），MMA 会
    #   过冲到转矩极低（T≈18）回不来。激活后 max_torque_ripple 从
    #   `ripple_target_start` 线性收到 `max_torque_ripple`，让 MMA 温和把转矩
    #   拉回可行区再收紧。默认 0 步 = 立即用 max_torque_ripple，golden 不变。
    ripple_ramp_steps: int = field(default=0, metadata={
        "doc": "con1 激活后脉振上限从 ripple_target_start 线性收到 "
               "max_torque_ripple 的步数。0=立即用 max_torque_ripple（默认）"})
    ripple_target_start: float = field(default=50.0, metadata={
        "doc": "con1 激活瞬间的脉振上限起点（宽，先放行再收紧）"})


@dataclass
class ALConfig:
    """增广拉格朗日罚因子调度（内核 19 提取的 legacy 实际调度，不改）。"""

    mu0: float = _fl("AL_stress_mu0", 0.01)
    mu_max: float = _fl("AL_stress_mu_max", 1e3)
    mu_update_step: int = _fl("AL_stress_mu_update_step", 50)


@dataclass
class StressConfig:
    """应力约束：上限、相对应力形式、两个互斥布尔折并的枚举、AL 子配置。"""

    max_stress_constraint_pa: float = _fl("max_stress_constraint", 50e7)
    use_rel_stress_constraint: bool = _fl("use_rel_stress_constraint", True)
    rel_stress_p_exp: float = _fl("rel_stress_p_exp", 1 / 2)
    max_rel_stress_constraint_pa: float = _fl("max_rel_stress_constraint", 50e7 * 0.8)
    # ⚠ 折并字段（无 legacy 原名，from_legacy 从 enum_from 的布尔折并）
    stress_type: StressType = field(
        default=StressType.TYPE2,
        metadata={"enum_from": ("use_type1_stress", "use_type2_stress")},
    )
    rel_stress_type: RelStressType = field(
        default=RelStressType.TYPE3,
        metadata={"enum_from": ("use_type1_rel_stress", "use_type2_rel_stress",
                                "use_type3_rel_stress", "use_type4_rel_stress")},
    )
    al: ALConfig = field(default_factory=ALConfig)
    #: 应力平滑半径（米）。0 = 关闭（逐位走原路径，golden 不变）。
    #: >0 时用固定物理半径的滤波把逐单元 σ̄ 平滑后再进 AL 约束 —— 峰值 von Mises
    #: 在尖角处奇异、随网格加密发散（实测 82→187→303 MPa），平滑后与网格弱相关
    #: （2026-10-20）。
    stress_filter_radius_m: float = 0.0


@dataclass
class MechConfig:
    """力学：转速、离心体积力、力学 SIMP。"""

    rot_speed_rpm: int = _fl("rot_speed_rpm", 10000)
    rot_center_x_m: float = _fl("rot_center_x", 0)
    rot_center_y_m: float = _fl("rot_center_y", 0)
    Fload_n: float = _fl("Fload", 0)               # 外载荷
    p_SIMP_mech: float = _fl("p_SIMP_mech", 3)
    min_SIMP_mech: float = _fl("min_SIMP_mech", 1e-4)
    SIMP_beta_mech: float = _fl("SIMP_beta_mech", 8)
    SIMP_thb_mech: float = _fl("SIMP_thb_mech", 0.65)


@dataclass
class SolverConfig:
    """求解：非线性开关、主从边界类型、并行度、性能钩子。"""

    use_nonlinear_iron: bool = _fl("use_nonlinear_iron", True)
    #: 线性铁磁的相对磁导率。⚠ **只在 ``use_nonlinear_iron=False`` 时生效**
    #: （两个是配对的：``newton.py`` 里 ``is_linear_iron`` 只在该开关关掉时置 1，
    #: 而 ``mur_linear`` 只在 ``is_linear_iron`` 那一支参与 μ 合成）。
    #: 生产默认 2000 = legacy 硬编码值。**降低它**（如 20~100）可做"软化"预优化：
    #: 铁磁与空气的 μ 差变小 ⇒ 磁场对拓扑的敏感性降低 ⇒ 优化问题更良态。
    #:
    #: ⚠ **用普通 ``field`` 而非 ``_fl``**：legacy 里这个 2000 是**硬编码字面量**
    #: （``newton.py`` 的 ``2000 * mu0``），不是模块级变量 ⇒ ``from_legacy``
    #: 找不到 ``g['mur_linear']`` 会 ``KeyError``（2026-10-16 实测：加 ``_fl``
    #: 让 10 个 config 测试全挂）。同 ``sampling_mode`` 的处境。
    mur_linear: float = field(default=2000.0, metadata={
        "doc": "线性铁磁相对磁导率（只在 use_nonlinear_iron=False 时生效）"})
    EM_MS_boundary_A_eq: bool = _fl("EM_MS_boundary_A_eq", False)  # 周期(True)/反周期(False)
    job_n_jobs: int = _fl("job_n_jobs", 14)         # joblib 并行度
    # 动网格（滑移层）边界插值的阶数。**无 legacy 对应物** ⇒ 普通 field。
    #
    # ★ 2026-10-14：**默认已改为 'cubic'**（生产）。理由（实测，barrier1 vs ANSYS，
    #   181 点全曲线；详见 `docs/转矩口径等价链核对与动网格插值_2026-10-13.md` §4.8）：
    #   - `K_rb(θ)` 在 'linear' 下是 φ 的**分段线性**函数（段长 = 接口角间距 h、
    #     斜率 1/h）⇒ 段间**导数跳变**（C⁰，非 C¹）。接口**协调**（转子副本与定子
    #     原始节点同角）⇒ 所有节点在 h 的整数倍处**同时重合**，断点集体跳变。
    #   - 后果：凡"对 θ 求导"的转矩口径（虚位移 `T_VD`、能量导数）在断点附近
    #     **无良定义的值**：粗网格 181 点里 **63 个点** T_VD 与 T_ArkBE 差 8~20；
    #     pos=13° 处 T_VD 与 ANSYS 差 **43%**（ε 扫描在 −17.5↔−37.4 间翻覆）。
    #   - 换 'cubic'（周期三次 Hermite，C¹：越过 stencil 交换点**导数连续**）后：
    #     同一点 T_VD 差 ANSYS **0.5%**（43%→0.5%）；181 点 mean|Δ| **18.7%→3.19%**；
    #     `T_VD − T_ArkBE` **18.6%→3.5%**。细网格（接口 1411）上三者互吻 **0.34%**。
    #   - **⇒ 本字段默认 'cubic' = 已验证的"对 θ 求导"口径的必要条件之一**
    #     （另一个是接口分辨率 ≥1411）。
    #
    # 'linear' = legacy 现状（分段线性，逐位保 golden）。
    #   ⚠ 只有 `RunConfig.legacy()` / golden 比对才用它 —— **必须显式回落**。
    # 'cubic' = 新方案：接口圆上**非均匀周期三次 Hermite**（4 点，C¹）。
    #   **非逐位**（改数值行为）。
    #   详见 `docs/协调气隙带_几何虚功转矩方案_2026-10-13.md`。
    moving_mesh_interp: str = field(default="cubic", metadata={
        "doc": "动网格插值阶数：'cubic'（默认，C¹，已验证）| 'linear'（legacy，逐位）"})
    # ★ 2026-10-14 记录（**结论：四个伴随都正确，不需要改**）：曾以为生产
    #   `flux_torque_derivative` 错（1e6 / 1431 / 2.33 等），并实现过一条补项 ——
    #   全部被**更严格的 FD** 推翻。用「全链 FD + `solve_abs_err=1e-12`」重测：
    #     ε=1e-3: Arkkio 7.4e-4 / flux 4.7e-4 / VD 4.5e-4 / ArkBE 2.9e-3
    #     ε=1e-4:        3.0e-4 /      2.7e-4 /    2.8e-4 /   1.4e-3
    #   ⇒ 随 ε 减小而减小 ⇒ 真误差 ~1e-4 ⇒ **`T_flux` 的伴随与另三个同水平**。
    #   历史上的"错"是三种**口径陷阱**（详见 `em_worker.py` 同处注释与
    #   `tools/probe_flux_chain_fd.py` 的 docstring）：随机 ρ / 固定-B 口径 /
    #   容差噪声 ∝1/ε。⇒ **无开关、无改动**。
    # ⚠ 性能钩子（阶段 7：只留开关，不实现）—— 无 legacy 对应物，普通 field。
    # 真开启前先做阶段 8 的分解复用实现 + 基准采集，否则开关无效但无害。
    spsolve_reuse_symbolic: bool = field(default=False, metadata={
        "doc": "收敛点 splu 一次、伴随解复用同一分解（阶段 8，F5③）"})
    # 线性求解器后端（`motorfem/solvers.py`）。'superlu' = 生产现状（逐位保 golden）；
    # 'pardiso' = MKL Pardiso mtype=11（非对称 LU）。⚠ 矩阵主从/驱动行 O(1) 绝对
    # 不对称 ⇒ 禁止对称模式（mtype=2 会解错系统，见 solvers.py 与 probe_pardiso.py）。
    # CSR 装配的「两次归约融合成一次」开关（见 fast/sparsity.py 的 set_csr_fusion）。
    # False（默认）= 逐位一致路径：两次 sum_duplicates + 逐元素相加。
    # True = 一次归约，**快 ~1.8×**，但累加顺序改变 ⇒ `data` 差浮点末位
    #   （实测 max|Δ| = 1.86e-9，尺度 8.8e6 ⇒ 相对 2e-16）。
    # ⚠ 开启后 `fast/` 不再与 legacy 逐位一致 ⇒ golden 比对须保持 False
    #   （`tools/verify_golden.py` 用的是默认配置）。
    csr_fuse: bool = True
    solver_backend: str = field(default="cholmod", metadata={
        "doc": "线性求解后端：'cholmod'（★生产默认：稀疏 Cholesky，复用符号 6.99×、"
               "10 步 MMA −69%；非逐位）| 'superlu'（legacy，逐位保 golden）| "
               "'pardiso'（MKL Pardiso，mtype=2 时更快）。★ 2026-10-13：默认从 "
               "superlu 改为 cholmod —— 旧的逐位路径保留在 RunConfig.legacy()"})
    # ⚠ mtype=2（实对称正定 LDLᵀ/Cholesky）要求矩阵**真对称** —— 主从+动网格
    # 插值消元（`motorfem/elimination.py`，2026-09-24）后 J/K_mech 对称才可用；
    # 对称模式内部自动 ``sp.triu``（Pardiso 按「只存上三角」解释 CSR）。
    pardiso_mtype: int = field(default=11, metadata={
        "doc": "Pardiso 矩阵类型：11 = 非对称 LU（默认）| 2 = 实对称正定（消元后可用，更快）"})


@dataclass
class MMAConfig:
    """MMA 算法参数。

    ⚠ 不在配置块里：来源是 ``MMA_optimize`` 签名默认值（``MMA.py:10-20``）与
    ``MMA10:4292-4295`` 调用处的字面量。这里固化的是生产生效值。
    """

    max_step: int = 3000          # MMA10:4294 字面量
    MMA_move: float = 0.1         # MMA10:4294 字面量
    f_rel_change: float = 1e-6    # MMA10:4295 字面量
    x_rel_change: float = 1e-6    # MMA10:4295 字面量
    MMA_dgrad: float = 1e-3       # MMA.py 签名默认
    MMA_rra0: float = 1e-5        # MMA.py 签名默认
    MMA_albefa: float = 0.1       # MMA.py 签名默认
    MMA_asyinit: float = 0.5      # MMA.py 签名默认
    MMA_asydecr: float = 0.7      # MMA.py 签名默认
    MMA_asyincr: float = 1.2      # MMA.py 签名默认


@dataclass
class OutputConfig:
    """输出：是否存图/数据、画图尺寸。"""

    save_numpy_data: bool = _fl("save_numpy_data", True)
    show_convergence_curve: bool = _fl("show_convergence_curve", True)
    save_figure_per_epoch: int = _fl("save_figure_per_epoch", 10)
    figsize_model: Tuple[float, float] = _fl("figsize_model", (10, 9))
    figsize_curve: Tuple[float, float] = _fl("figsize_curve", (6, 5))


@dataclass
class RuntimeConfig:
    """运行时：输出目录（时间戳）与运行时性能开关—— 仅运行时填充/覆盖。"""

    img_path: str = _fl("img_path", "")
    now: str = _fl("now", "")
    # ⚠ 性能钩子（阶段 7 留开关、阶段 8 的 8c 接线）。
    # omp_num_threads=None = 不动线程数（沿用环境），设整数则在求解前设
    # OMP/OPENBLAS/MKL_NUM_THREADS；use_gpu 预留 autograd/求解的 GPU 开关。
    omp_num_threads: int = field(default=None, metadata={
        "doc": "BLAS 线程数（None = 沿用环境；阶段 8 前无效）"})
    # n_jobs：EM 时间点并行度（阶段 8 的 8c 接线）。None = 串行（golden 对齐路径，
    # 默认行为不变）；设整数 >1 = ``em_segment`` 用 ``ThreadPoolExecutor`` 并行
    # 11 个时间点。⚠ 并行时**自动**设 OMP/OPENBLAS/MKL_NUM_THREADS=1（任务并行
    # 维度下不给单次求解开 BLAS 线程，避免超订 —— 设计文档 §阶段 8 的并行项）。
    n_jobs: int = field(default=8, metadata={
        "doc": "EM 时间点并行度（None=串行=golden 对齐；>1=ThreadPool 任务并行）"})
    use_gpu: bool = field(default=False, metadata={
        "doc": "GPU 求解开关（阶段 6 探针证明批量下才有收益，阶段 8 实现）"})
    # 梯度滤波模式（设计文档 §11① 的「修 vs 保留」判定开关）。
    # "legacy" = ``filter_SIMP_grad`` 前向滤波（MMA10:4042 原样，默认保 golden）；
    # "adjoint" = ``filter_SIMP.T`` 伴随一致（f 的真梯度，判定实验用）。
    gradient_filter: str = field(default="legacy", metadata={
        "doc": "梯度滤波模式：legacy=filter_SIMP_grad 前向（保 golden）/ "
               "adjoint=filter_SIMP.T 伴随一致（§11① 判定）"})
    optimizer: str = field(default="mma", metadata={
        "doc": "优化器：mma（legacy 逐位，默认）/ slsqp（scipy 序列二次规划）/ "
               "adam（Adam+增广拉格朗日罚，支持随机子采样）/ gcmma（mmapy 全局"
               "收敛 MM）。见 fem/optimizers.py"})


# ---------------------------------------------------------------- RunConfig
@dataclass
class RunConfig:
    """全部配置的聚合根。JSON 可序列化；``from_legacy`` 从 legacy 命名空间构造。"""

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    electrical: ElectricalConfig = field(default_factory=ElectricalConfig)
    discretization: DiscretizationConfig = field(default_factory=DiscretizationConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    mech: MechConfig = field(default_factory=MechConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    mma: MMAConfig = field(default_factory=MMAConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ------------------------------------------------------------ 构造
    @classmethod
    def legacy(cls) -> "RunConfig":
        """**旧方案**（性能优化前的逐位路径）。

        ★ 2026-10-13：``RunConfig()`` 的默认已切成**生产优化**（cholmod 稀疏
        Cholesky + 时间点并行 + CSR 归约融合 + B-H 查表），它有加速但**不再与
        legacy 逐位一致**。本方法恢复「优化前」的配置：

        ============  =========================  ==================
        字段            生产默认（快）               legacy（逐位）
        ============  =========================  ==================
        solver_backend  ``cholmod``                  ``superlu``
        csr_fuse        ``True``                     ``False``
        n_jobs          ``8``                        ``None``（串行）
        bh_lut_n        ``65536``                    ``0``（PCHIP）
        moving_mesh_interp ``cubic``（C¹）           ``linear``（legacy）
        ============  =========================  ==================

        **凡与 golden 做 ``np.array_equal`` 的测试、``tools/verify_golden.py``、
        ``tests/autograd/*``（查表与 oracle 结构互斥）都必须用它。**
        """
        c = cls()
        c.solver.solver_backend = "superlu"
        c.solver.csr_fuse = False
        c.runtime.n_jobs = None
        c.material.bh_lut_n = 0
        # ★ 动网格插值必须显式回落 'linear'：`SolverConfig.moving_mesh_interp`
        #   的生产默认已改为 'cubic'（C¹，见该字段注释），而 golden 是在
        #   'linear' 下生成的（分段线性 ⇒ 换 cubic 会改变所有数值）。
        #   **这是本方法必须显式设置它、不能靠默认值继承的原因。**
        c.solver.moving_mesh_interp = "linear"
        # ★ 转矩口径也必须显式回落：生产默认已改为「平均用 flux、纹波用 ArkBE」
        #   （见 `ObjectiveConfig.mean_torque_source` / `ripple_source` 的注释），
        #   而 golden L4 是在 legacy 口径（两者都 'arkkio'）下生成的。
        c.objective.mean_torque_source = "arkkio"
        c.objective.ripple_source = "arkkio"
        return c

    @classmethod
    def from_legacy(cls, g: Dict[str, Any]) -> "RunConfig":
        """从 MMA10 模块命名空间快照构造（``vars(nonlinear_solver_energy_MMA10)``）。

        每个字段按 ``metadata["legacy"]`` 从 ``g`` 取原名；枚举字段按
        ``metadata["enum_from"]`` 里的布尔名折并；嵌套 dataclass 递归。
        """
        return cls(**_from_legacy_fields(cls, g))

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> Dict[str, Any]:
        """dataclass → 纯 dict（枚举转 value，tuple 转 list —— 可 JSON 化）。"""
        return _enum_to_value(dataclasses.asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        """``to_dict`` 的逆：dict → dataclass（含嵌套/tuple/枚举还原）。"""
        return cls(**_coerce_dict(cls, d))

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------- 内部实现
def _from_legacy_fields(cls, g: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {}
    for f in fields(cls):
        if dataclasses.is_dataclass(f.type):  # 嵌套 dataclass：递归并**实例化**
            kwargs[f.name] = f.type(**_from_legacy_fields(f.type, g))
            continue
        enum_from = f.metadata.get("enum_from")
        if enum_from:  # 枚举：从互斥布尔折并
            flags = [bool(g[n]) for n in enum_from]
            if len(flags) == 2:
                kwargs[f.name] = collapse_stress_type(*flags)
            else:
                kwargs[f.name] = collapse_rel_stress_type(*flags)
            continue
        lname = f.metadata.get("legacy")
        if lname is not None:
            kwargs[f.name] = g[lname]
    return kwargs


def _enum_to_value(obj: Any) -> Any:
    """递归把枚举成员换成 value（asdict 后枚举仍在，JSON 不接受枚举）。"""
    if isinstance(obj, dict):
        return {k: _enum_to_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_enum_to_value(v) for v in obj]
    if isinstance(obj, (StressType, RelStressType)):
        return obj.value
    return obj


def _coerce_dict(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    """dict → dataclass 字段值：嵌套递归、tuple/list 还原、枚举还原。"""
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        if dataclasses.is_dataclass(f.type):
            kwargs[f.name] = f.type(**_coerce_dict(f.type, val))
        elif f.type is tuple or getattr(f.type, "__origin__", None) is tuple:
            kwargs[f.name] = tuple(val)
        elif isinstance(f.type, type) and issubclass(f.type, (StressType, RelStressType)):
            kwargs[f.name] = f.type(val)
        else:
            kwargs[f.name] = val
    return kwargs


# ---------------------------------------------------------------- 路径解析
_REPO_ROOT_FOR_PATHS = Path(__file__).resolve().parents[2]


def resolve_bh_csv(cfg_or_value) -> str:
    """B-H 曲线路径解析（**唯一口径**，全仓共用）。

    规则（照抄 `optim_gui/server.py::_resolve_csv`，2026-10-16 提升到核心层）：
    - 含 ``:``（Windows 盘符）或以 ``/`` 开头 ⇒ **绝对路径，原样返回**；
    - 否则按 ``<仓库根>/legacy/pytorch/<值>`` 解析。

    用途：**替换 B-H 曲线**（= 换铁芯材料）。把新 CSV 放到任意位置，
    把 ``cfg.material.bh_curve_csv`` 设为绝对路径即可，不必再往
    ``legacy/pytorch/`` 里塞。

    ⚠ 之前各脚本各自拼 ``_REPO_ROOT / "legacy" / "pytorch" / cfg.material.bh_curve_csv``,
    无法用绝对路径。统一走本函数。
    """
    csv = cfg_or_value
    if hasattr(cfg_or_value, "material"):
        csv = cfg_or_value.material.bh_curve_csv
    csv = str(csv)
    if ":" in csv or csv.startswith("/") or csv.startswith("\\\\"):
        return csv
    return str(_REPO_ROOT_FOR_PATHS / "legacy" / "pytorch" / csv)
