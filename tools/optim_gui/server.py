#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/optim_gui/server.py —— 拓扑优化网页 GUI（stdlib，零依赖）

在浏览器里看 MMA 优化过程：转子几何随 ρ 实时填色、目标/约束/相对变化轨迹、
逐段耗时；支持 **开始 / 暂停 / 单步 / 继续 / 停止 / 保存检查点 / 加载续跑**。

架构
----
- ``ThreadingHTTPServer``（默认 127.0.0.1:8765）提供 REST 接口 + 静态页；
- MMA 在**后台线程**跑（``FieldSystem.optimize`` + ``mma_driver`` 的
  ``control``/``initial_state`` 钩子），**步间**协作式暂停/停止（一步
  ~1.5~6s，暂停最迟在当前步结束后生效）；
- 每步回调 ``on_step`` 把标量轨迹 + 设计区 ρ 快照打进内存，浏览器轮询
  ``/api/state`` 取走；几何（节点/三角/区域色）只在构建/加载时传一次。

用法::

    PYTHONIOENCODING=utf-8 python -u tools/optim_gui/server.py --port 8765

然后浏览器打开 http://127.0.0.1:8765 。

⚠ 服务启动即把 BLAS 钉到单线程（与 probe_mma_full 同纪律）；计时受电源状态
影响（``docs/项目状态.md`` §11⑥）。golden 位级对齐不是本 GUI 的职责
（那是 ``tools/verify_golden.py``）；默认配置 + BLAS 单线程下轨迹与生产一致。

⚠ 本文件 **不是** 内核/驱动层代码：它只做「计时包装 + 进程内存快照」，数值
路径一字不动（同一套包装手法见 ``tools/probe_mma_full.py``）。
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

#: ⚠ **计算栈按需加载**。只读监视（``--read-only``）的卖点之一就是「网页
#: 端的进程不认识优化器」；若在模块顶部无条件 import，光是想看个进度就得
#: 把 numpy/scipy/CHOLMOD 全套拉起来，解耦也就名存实亡了。
#: 下面四个名字**只**在非只读模式（或只读要建几何时）由 ``_load_compute_stack``
#: 填充；只读路径若不碰几何，永远不会执行到这里。
RunConfig = None            # type: ignore[assignment]
density_from_x = None       # type: ignore[assignment]
FieldSystem = None          # type: ignore[assignment]
obj_mod = None              # type: ignore[assignment]


def _load_compute_stack() -> None:
    """**唯一** import 计算栈的地方。幂等。"""
    global RunConfig, density_from_x, FieldSystem, obj_mod
    if FieldSystem is not None:
        return
    from motorfem.config import RunConfig as _RunConfig
    from motorfem.fem.em_worker import density_from_x as _density_from_x
    from motorfem.fem.system import FieldSystem as _FieldSystem
    from motorfem.fem import objective as _obj_mod
    RunConfig, density_from_x = _RunConfig, _density_from_x
    FieldSystem, obj_mod = _FieldSystem, _obj_mod

_DEFAULT_MESH = _REPO_ROOT / "legacy" / "pytorch" / "my_gmsh" / "model_mirror4.msh"
_DEFAULT_CSV = _REPO_ROOT / "legacy" / "pytorch" / "50CS470.csv"
_SAVES_DIR = _REPO_ROOT / "tools" / "_optim_gui_saves"
_STATIC_DIR = Path(__file__).parent / "static"
_FORMAT_VERSION = 1

#: 只读模式的默认约束名（生产端可用日志里的 ``LABELS`` 行覆盖）。
_DEFAULT_LABELS = {"con1": "con1", "con2": "con2", "con3": "con3", "con4": "con4"}

#: 配置表单 schema：浏览器按它渲染输入框，POST /api/start 时按 path 收集。
#: 只暴露**会改变优化行为**的选项；其余（几何常量、输出开关等）走高级 JSON。
#: 每个字段的 ``hint`` 是鼠标悬浮（title tooltip）提示：含义 + 生产默认值/建议值。
CONFIG_SCHEMA: List[Dict[str, Any]] = [
    {"group": "优化器", "fields": [
        {"path": ["runtime", "optimizer"], "type": "select", "label": "优化器",
         "options": [
             {"value": "mma", "label": "MMA（默认，legacy 逐位）",
              "tip": "自研 MM 渐近线驱动，golden 钉死。生产默认。"},
             {"value": "slsqp", "label": "SLSQP（scipy 序列二次规划）",
              "tip": "QP 子问题 + BFGS 近 Hessian，处理非线性约束方式与 MM 不同。"},
             {"value": "adam", "label": "Adam + AL 罚函数",
              "tip": "无约束一阶 + 增广拉格朗日罚，支持随机转子位置子采样（SGD 效应）。"},
             {"value": "gcmma", "label": "GCMMA（mmapy 全局收敛 MM）",
              "tip": "保守近似内层迭代，每外层步 2~3 次求值。"},
         ],
         "hint": "优化器可替换（2026-10-11）。mma = legacy 逐位（golden 契约）；\n"
                 "slsqp / adam / gcmma 是新对比路径（可能不收敛，用于平行实验）。\n"
                 "adam 配合下方『采样设置』的随机采样做 SGD 效果最佳。"},
        {"path": ["time", "sampling_mode"], "type": "select", "label": "转子扫描采样",
         "options": [
             {"value": "uniform", "label": "均匀（legacy，保 golden）",
              "tip": "时间轴均匀 N 点，legacy 默认，逐位一致。"},
             {"value": "random", "label": "随机（种子可设）",
              "tip": "在仿真时长内随机取 N 个时间点，放弃位级一致。"},
             {"value": "table", "label": "位置表（下栏填度数）",
              "tip": "用下方『位置表（度）』显式指定转子扫描位置。"},
         ],
         "hint": "转子扫描采样模式（2026-10-11）。uniform = legacy 均匀（保 golden）；\n"
                 "random = 随机时间点；table = 显式位置度表（见下栏）。\n"
                 "非均匀/随机采样 = 任务 3 的优化样本设置。"},
        {"path": ["time", "random_seed"], "type": "int", "label": "随机采样种子",
         "hint": "sampling_mode=random 的随机种子（np.random.RandomState）。\n"
                 "固定种子可复现。"},
        {"path": ["time", "positions_deg"], "type": "text",
         "label": "位置表（度，逗号分隔）",
         "hint": "sampling_mode=table 时用：显式转子扫描位置表，如 0,5,10,15…\n"
                 "（机械角度，度）。t 由位置反解，电流相位自动对齐。"},
        {"path": ["time", "use_validation_set"], "type": "bool", "label": "生成验证集",
         "hint": "额外生成验证转子位置，不参与优化（训练-验证分离，防过拟合\n"
                 "训练采样点）。优化结束后用验证点评估真实转矩/脉动。"},
        {"path": ["time", "n_validation_points"], "type": "int", "label": "验证集点数",
         "hint": "验证转子位置数。uniform 模式在时间轴均匀取（与训练不重叠）；\n"
                 "random/table 模式用下面的种子随机取。"},
        {"path": ["time", "validation_seed"], "type": "int", "label": "验证集种子",
         "hint": "验证集随机种子（random/table 采样模式时用）。"},
    ]},
    {"group": "算法 MMA", "fields": [
        {"path": ["mma", "max_step"], "type": "int", "label": "总步数（3000=生产默认）",
         "hint": "MMA 总迭代步数。生产默认 3000（superlu 约 1~2 小时）。\n"
                 "快速看效果：10~100；完整优化：3000。达到此值自动结束。"},
        {"path": ["mma", "MMA_move"], "type": "float", "label": "MMA_move 单步移动上限",
         "hint": "单步移动上限（0~1）：每次迭代 x 相对上下界的最大移动比例。\n"
                 "0.1=生产（保守）；调大到 0.5 收敛更快但可能震荡。"},
        {"path": ["mma", "MMA_dgrad"], "type": "float", "label": "MMA_dgrad",
         "hint": "渐近线阻尼系数，取值 [0,1)。梯度正则强度，生产 1e-3，一般不动。"},
        {"path": ["mma", "MMA_rra0"], "type": "float", "label": "MMA_rra0",
         "hint": "渐近线除零保护正则，(0,1)，生产 1e-5，一般不动。"},
        {"path": ["mma", "MMA_albefa"], "type": "float", "label": "MMA_albefa",
         "hint": "子问题可行域 α/β 扩展系数，(0,1)，生产 0.1。\n"
                 "越大边界离当前点越远，一般不动。"},
        {"path": ["mma", "MMA_asyinit"], "type": "float", "label": "MMA_asyinit",
         "hint": "渐近线初始距离比例，(0,1)，生产 0.5。决定前两步的渐近线位置。"},
        {"path": ["mma", "MMA_asydecr"], "type": "float", "label": "MMA_asydecr",
         "hint": "渐近线收缩因子，(0,1)，生产 0.7。步间振荡时收缩渐近线。"},
        {"path": ["mma", "MMA_asyincr"], "type": "float", "label": "MMA_asyincr",
         "hint": "渐近线扩张因子，必须 >1，生产 1.2。步间平稳时扩张渐近线。"},
        {"path": ["mma", "f_rel_change"], "type": "float", "label": "f 收敛判据（-1=关）",
         "hint": "收敛判据：|Δf|/|f| 小于该值且 x 判据满足且约束全部可行 → 提前结束。\n"
                 "设 -1 关闭（跑满 max_step）；生产 1e-6。"},
        {"path": ["mma", "x_rel_change"], "type": "float", "label": "x 收敛判据（-1=关）",
         "hint": "收敛判据：||Δx||/||x|| 小于该值（配合 f_rel_change 一起判断）。\n"
                 "设 -1 关闭；生产 1e-6。"},
    ]},
    {"group": "性能优化", "collapsible": True, "preset": {
        "label": "性能预设（一键应用，详细数值见下方折叠菜单）",
        "options": [
            {"value": "prod", "label": "① 逐位一致（golden 口径 = RunConfig.legacy()）",
             "hint": "superlu 后端 + 串行 + 关闭所有近似优化。★ 2026-10-13 起，这**不再**\n"
                     "是 RunConfig() 的默认（默认已切为下面的「生产默认」）—— 凡与 golden\n"
                     "做 np.array_equal 的验证请选它（等价 RunConfig.legacy()）。",
             "values": {"solver.solver_backend": "superlu", "runtime.n_jobs": None,
                        "solver.csr_fuse": False, "material.bh_lut_n": 0,
                        "runtime.gradient_filter": "legacy", "solver.pardiso_mtype": 11}},
            {"value": "fast", "label": "② 并行加速（superlu，逐位一致）",
             "hint": "superlu + n_jobs=8：实测整步快 ~2.56×，轨迹仍逐位一致。",
             "values": {"solver.solver_backend": "superlu", "runtime.n_jobs": 8,
                        "solver.csr_fuse": False, "material.bh_lut_n": 0,
                        "runtime.gradient_filter": "legacy", "solver.pardiso_mtype": 11}},
            {"value": "fastest", "label": "③ 生产默认（cholmod+并行+融合+查表，最快）",
             "hint": "★ 这就是 RunConfig() 的默认值（2026-10-13 起）：cholmod + n_jobs=8\n"
                     "+ csr_fuse + B-H 查表，整步 ~3.5×。放弃位级一致，且与 autograd\n"
                     "oracle 层冲突（要跑 golden / autograd 请选 ①）。",
             "values": {"solver.solver_backend": "cholmod", "runtime.n_jobs": 8,
                        "solver.csr_fuse": True, "material.bh_lut_n": 65536,
                        "runtime.gradient_filter": "legacy", "solver.pardiso_mtype": 11}},
        ],
    }, "fields": [
        {"path": ["solver", "solver_backend"], "type": "select", "label": "线性求解后端",
         "options": ["superlu", "pardiso", "cholmod"],
         "hint": "线性求解后端。\n"
                 "superlu = scipy 默认（逐位保 golden）；\n"
                 "pardiso = MKL 非对称 LU（实测快 ~2×，非逐位）；\n"
                 "cholmod = SuiteSparse 稀疏 Cholesky（消元后 J/K 对称正定可用，快 ~7×，\n"
                 "            非逐位，需 CHOLMOD_DLL_DIR 环境变量）。\n"
                 "★ 2026-10-13 起默认 cholmod（要与 golden 逐位请选 superlu，或预设①）。"},
        {"path": ["solver", "pardiso_mtype"], "type": "select", "label": "Pardiso 矩阵类型",
         "options": ["11", "2"],
         "hint": "Pardiso 矩阵类型。11 = 非对称 LU（默认）；2 = 实对称正定\n"
                 "（主从消元后 J/K_mech 真对称才可用，更快）。\n"
                 "仅 solver_backend=pardiso 时生效。"},
        {"path": ["runtime", "n_jobs"], "type": "int_none", "label": "EM 时间点并行 n_jobs（空=串行）",
         "hint": "EM 段 11 个时间点的并行度。留空 = 串行（golden 对齐路径）。\n"
                 "设 8 实测快 ~2.56×（自动把 BLAS 钉单线程，轨迹逐位一致）。\n"
                 "数字越大内存占用越高；建议 8。"},
        {"path": ["runtime", "gradient_filter"], "type": "select", "label": "梯度滤波模式",
         "options": ["legacy", "adjoint"],
         "hint": "梯度滤波模式。legacy = 前向滤波（保 golden，默认）；\n"
                 "adjoint = 伴随一致真梯度（§11① 判定实验用，300 步实证反而不如 legacy）。\n默认 legacy。"},
        {"path": ["solver", "csr_fuse"], "type": "bool", "label": "CSR 融合（快 ~1.8×，差末位）",
         "hint": "CSR 组装两次归约融合成一次：快 ~1.8×，但累加顺序改变、\n"
                 "差浮点末位（~1.9e-9）。要逐位一致（verify_golden）必须关。\n"
                 "★ 2026-10-13 起默认开（生产优化）。"},
        {"path": ["material", "bh_lut_n"], "type": "int", "label": "B-H 查表点数（0=关闭）",
         "hint": "B-H 曲线查表求值点数。0 = 关闭（scipy PCHIP，逐位一致，默认）；\n"
                 ">0 = uniform-x 查表（整步快 ~1.1×，但放弃位级一致，\n"
                 "且与 autograd oracle 层冲突）。建议 0。"},
    ]},
    {"group": "目标与约束", "fields": [
        {"path": ["objective", "mean_torque_obj"], "type": "float", "label": "目标平均转矩 (N·m)",
         "hint": "目标平均转矩 (N·m)，生产 90。\n"
                 "约束 con3 = (1 − 平均转矩/该值)·权重，要求平均转矩 ≥ 该值。"},
        {"path": ["objective", "max_torque_ripple"], "type": "float", "label": "转矩波动上限",
         "hint": "转矩波动上限 (N·m)，生产 5。\n"
                 "con1（Arkkio 波动）/ con4（VD 波动）要求波动 ≤ 该值。"},
        {"path": ["objective", "p_exp_torque_ripple"], "type": "float", "label": "波动 p-范数",
         "hint": "转矩波动的 p-范数指数，生产 10。\n"
                 "越大越逼近最大波动值，越小越接近平均波动。\n"
                 "p=10 高度非凸（梯度 ∝|T−T_obj|^9，argmax 切换）是 MMA 密度锯齿的\n"
                 "主因；legacy MMA4/MMA5 用 p=3。开『p 连续化』可让 p 从低值爬升。"},
        {"path": ["objective", "p_exp_continuation"], "type": "bool", "label": "p 连续化开关",
         "hint": "脉动范数 p 从 p_exp_start 线性爬升到 p_exp_torque_ripple。\n"
                 "先低 p 平滑收敛、再高 p 收紧脉动约束，治 MMA 密度锯齿。\n"
                 "默认关（p 恒为 p_exp_torque_ripple，保 golden 逐位一致）。"},
        {"path": ["objective", "p_exp_start"], "type": "float", "label": "p 连续化起始值",
         "hint": "连续化起始 p，默认 3（legacy MMA4/MMA5 的生产值）。\n"
                 "低 p → 约束平滑、梯度连续，MMA 一阶模型拟合好。"},
        {"path": ["objective", "p_exp_ramp_steps"], "type": "int", "label": "p 连续化爬升步数",
         "hint": "p 从起始值爬到终值的 MMA 步数，默认 300。\n"
                 "之后固定为 p_exp_torque_ripple。太短 → 没等低 p 收敛就变陡；\n"
                 "太长 → 高 p 阶段步数不足。3000 步总量建议 500~1000。"},
        {"path": ["objective", "p_exp_move_step"], "type": "float", "label": "move 联动档距",
         "hint": "p 每爬升多少，移动极限收窄一档（默认 1 = 每升 1 个指数收窄）。\n"
                 "高 p 时约束梯度极陡，MMA 需要收紧步长防 overshoot。"},
        {"path": ["objective", "p_exp_move_factor"], "type": "float", "label": "move 联动收窄系数",
         "hint": "移动极限每档的收窄系数（<1=收紧，默认 0.7）。\n"
                 "p 从 3 爬到 10 共 7 档 ⇒ move × 0.7^7 ≈ 0.08，\n"
                 "即从默认 0.1 收到 ~0.008。越小越保守。"},
        {"path": ["objective", "weight_torque"], "type": "float", "label": "转矩约束权重",
         "hint": "转矩约束权重，生产 100。放大 con3 在 MMA 子问题里的影响。"},
        {"path": ["objective", "weight_torque_ripple"], "type": "float", "label": "波动约束权重",
         "hint": "波动约束权重，生产 10。放大 con1/con4 的影响。"},
        {"path": ["objective", "weight_density"], "type": "float", "label": "密度约束权重",
         "hint": "密度（体积）约束权重，生产 1。放大 con2 的影响。"},
        {"path": ["objective", "Tflux_ratio"], "type": "float", "label": "Tflux_ratio",
         "hint": "磁链法转矩混合比，生产 0.5。⚠ 当前目标计算未使用（legacy 保留字段），\n"
                 "改动不影响本轮优化。"},
        {"path": ["objective", "objective_role"], "type": "select", "label": "优化目标选择",
         "options": [
             {"value": "uelastic+stress", "label": "应变能 + 应力（默认，保 golden）",
              "tip": "f[0] = g_Uelastic + g_stress，legacy 组合目标"},
             {"value": "uelastic", "label": "弹性应变能（最小化）",
              "tip": "f[0] = g_Uelastic = ½uᵀKu"},
             {"value": "stress", "label": "应力 AL 罚（最小化）",
              "tip": "f[0] = g_stress = Σ P_e/N"},
             {"value": "con1", "label": "Arkkio 转矩波动（最小化）",
              "tip": "f[0] = g_con1 = (Arkkio 波动/上限−1)·权重"},
             {"value": "con2", "label": "平均密度（最小化）",
              "tip": "f[0] = g_con2 = 密度多项式约束"},
             {"value": "con3", "label": "平均转矩（最大化）",
              "tip": "f[0] = g_con3 = (1−平均转矩/目标)·权重，最小化=转矩推高"},
             {"value": "con4", "label": "VD 转矩波动（最小化）",
              "tip": "f[0] = g_con4 = (VD 波动/上限−1)·权重"},
         ],
         "hint": "哪项进 f[0] 作目标（最小化）。默认『应变能+应力』= legacy 组合目标\n"
                 "（保 golden）。选『平均转矩』作目标 = 最小化 (1−T/T_obj)·w = 最大化转矩。\n"
                 "其余启用项自动变约束。悬停每个选项可见内部变量名。"},
        {"path": ["objective", "enable_uelastic"], "type": "bool", "label": "应变能参与",
         "hint": "弹性应变能参与（作目标或约束）。\n"
                 "与『应力参与』都关 ⇒ 跳过力学段求解（省一次力学 spsolve + 2 伴随），\n"
                 "obj 机械项恒 0，AL 冻结。"},
        {"path": ["objective", "enable_stress"], "type": "bool", "label": "应力参与",
         "hint": "应力 AL 罚项参与（作目标或约束）。\n"
                 "与『应变能参与』都关 ⇒ 跳过力学段求解。"},
        {"path": ["objective", "enable_con1"], "type": "bool", "label": "Arkkio 脉动约束",
         "hint": "Arkkio 转矩波动约束（con1）参与。关掉 = 不要求 Arkkio 波动达标。"},
        {"path": ["objective", "enable_con2"], "type": "bool", "label": "密度约束",
         "hint": "平均密度约束（con2）参与。关掉 = 不限制材料用量。"},
        {"path": ["objective", "enable_con3"], "type": "bool", "label": "转矩约束",
         "hint": "平均转矩约束（con3）参与。关掉 = 不要求平均转矩 ≥ 目标。"},
        {"path": ["objective", "enable_con4"], "type": "bool", "label": "VD 脉动约束",
         "hint": "VD 转矩波动约束（con4）参与。关掉 = 不要求 VD 波动达标。"},
    ]},
    {"group": "拓扑 / 滤波", "fields": [
        {"path": ["topology", "p_SIMP"], "type": "float", "label": "SIMP 惩罚指数 p",
         "hint": "SIMP 惩罚指数 p，生产 3。越大中间密度惩罚越强、结果越接近 0/1，\n"
                 "但梯度越陡、收敛越难。"},
        {"path": ["topology", "SIMP_beta"], "type": "float", "label": "SIMP tanh 陡度 β",
         "hint": "SIMP tanh 投影陡度 β，生产 8。越大密度投影越接近阶跃（0/1 化）。"},
        {"path": ["topology", "SIMP_thb"], "type": "float", "label": "SIMP 阈值",
         "hint": "SIMP 投影阈值，生产 0.5。投影的翻转中点。"},
        {"path": ["topology", "max_density"], "type": "float", "label": "密度上限",
         "hint": "平均密度上限，生产 0.65。con2 = 平均密度/该值 − 1，是体积约束。"},
        {"path": ["topology", "min_SIMP"], "type": "float", "label": "最小密度",
         "hint": "最小密度，生产 1e-6。避免材料模型除零/奇异。"},
        {"path": ["filter", "if_filter_rho"], "type": "bool", "label": "密度滤波",
         "hint": "密度滤波开关，生产开。抑制棋盘格伪影，长度尺度由 filter_radius_m 决定。"},
        {"path": ["filter", "filter_radius_m"], "type": "float", "label": "密度滤波半径 (m)",
         "hint": "密度滤波半径 (m)，生产 2e-3。决定拓扑的最小特征尺寸。"},
        {"path": ["filter", "if_filter_grad"], "type": "bool", "label": "梯度滤波",
         "hint": "梯度滤波开关，生产开。对目标/约束梯度再滤波一次。"},
        {"path": ["filter", "filter_grad_radius_m"], "type": "float", "label": "梯度滤波半径 (m)",
         "hint": "梯度滤波半径 (m)，生产 0.4e-3。梯度滤波的长度尺度。"},
    ]},
    {"group": "力学 / 应力", "fields": [
        {"path": ["mech", "rot_speed_rpm"], "type": "int", "label": "转速 (rpm)",
         "hint": "转速 (rpm)，生产 10000。离心体积力与转速²成正比，决定机械应力水平。"},
        {"path": ["mech", "p_SIMP_mech"], "type": "float", "label": "力学 SIMP p",
         "hint": "力学子网格的 SIMP 惩罚指数，生产 3。"},
        {"path": ["mech", "SIMP_beta_mech"], "type": "float", "label": "力学 SIMP β",
         "hint": "力学 SIMP tanh 陡度，生产 8。"},
        {"path": ["mech", "SIMP_thb_mech"], "type": "float", "label": "力学 SIMP 阈值",
         "hint": "力学 SIMP 投影阈值，生产 0.65。"},
        {"path": ["stress", "max_stress_constraint_pa"], "type": "float", "label": "应力上限 (Pa)",
         "hint": "应力约束上限 (Pa)，生产 50e7 = 500 MPa。\n"
                 "应力约束以 AL 罚项进入目标（objective = 应变能 + 罚项）。"},
        {"path": ["stress", "max_rel_stress_constraint_pa"], "type": "float", "label": "相对应力上限 (Pa)",
         "hint": "相对应力约束上限 (Pa)，生产 40e7 = 400 MPa。\n"
                 "仅在 use_rel_stress_constraint 打开时生效。"},
        {"path": ["stress", "use_rel_stress_constraint"], "type": "bool", "label": "用相对应力约束",
         "hint": "用相对应力约束形式，生产开。关掉则改用绝对应力形式。"},
        {"path": ["stress", "rel_stress_p_exp"], "type": "float", "label": "相对应力 p",
         "hint": "相对应力的 p-范数指数，生产 0.5。"},
    ]},
]

_BASE_COLORS = {0: "rgb(140,140,148)", 1: "rgb(230,184,107)", 2: "rgb(255,255,255)"}


def _coerce_stage(v: Any) -> Any:
    """NPZ 的 ``stage`` 字段：legacy 写 int，``run_auto_schedule`` 写字符串（"S0"）。

    ⚠ 直接 ``int(z["stage"])`` 会让整个快照读取失败（页面静默失去 ρ）。
    """
    a = np.asarray(v)
    x = a.ravel()[0] if a.size else None
    try:
        return int(x)
    except (TypeError, ValueError):
        return str(x)


def _empty_history() -> Dict[str, list]:
    return {
        "step": [], "obj": [], "cons": [],
        "fun_ms": [], "em_ms": [], "mech_ms": [], "step_ms": [],
        "dx_rel": [],
    }


def _ac_power() -> str:
    """AC/电池：Win32_Battery.BatteryStatus（2 = 接电）。失败返回 '未知'。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance -ClassName Win32_Battery).BatteryStatus"],
            capture_output=True, text=True, timeout=5)
        v = out.stdout.strip()
        return "AC（BatteryStatus=2）" if v == "2" else f"未知状态({v})"
    except Exception as e:  # noqa: BLE001
        return f"未知({type(e).__name__})"


def _build_geometry(top) -> Dict[str, Any]:
    """转子截面的静态几何 + 区域色：只依赖网格，与 ρ 无关（传一次）。

    颜色复刻 ``tools/plot_mma_short.plot_geometry`` 的口径：
    ``color_id`` = 0 灰（铁芯/默认）| 1 tan（线圈）| 2 白（气隙）| 3 设计区
    （设计区每帧用 viridis(ρ) 上色，其他用 base 色）。
    """
    xy = np.asarray(top.layer.xy, dtype=float)
    tri = np.asarray(top.layer.tri, dtype=int)
    n_elem = tri.shape[0]
    topo = set(top.element_topo_opti_m) | set(top.element_topo_opti_s.keys())

    color_id = np.zeros(n_elem, dtype=int)            # 默认灰
    coil = set().union(*top.regions.region("element_stator_coil"))
    for e in coil - topo:
        color_id[e] = 1
    for e in top.regions.region("element_rotor_gap"):
        color_id[e] = 2
    for e in topo:
        color_id[e] = 3

    design_idx = np.array(list(top.element_topo_opti_m)
                          + list(top.element_topo_opti_s.keys()), dtype=int)
    return {
        "xy": xy.tolist(),
        "tri": tri.tolist(),
        "color_id": color_id.tolist(),
        "design_idx": design_idx.tolist(),
        "n_elem": int(n_elem),
        "n_node": int(xy.shape[0]),
        "n_topo": int(len(topo)),
        "base_colors": _BASE_COLORS,
    }


class LogManager:
    """**只读**优化监视器：全部数据来自「事件日志 + NPZ 快照」两件产物。

    契约见 ``docs/GUI只读监控产物契约.md``。本类**不 import ``FieldSystem``**
    （模块顶部那行 import 在只读模式下也不会被执行，见 ``_load_compute_stack``），
    不创建任何优化对象，没有任何 POST 入口 —— 计算程序怎么跑、由谁跑，
    与本类无关；唯一耦合是 §2/§3 的行格式与 key 名。

    ⚠ 三条硬纪律（都是踩过的坑）：
    - **日志存在 ≠ 生产端还活着**：靠 ``RUNNING`` 里的 pid 查进程，
      查不到就报 ``unknown``，绝不判成功；
    - **NPZ 读失败就清空 ρ**：不沿用上一帧旧密度冒充当前状态；
    - **解析失败显式报错**：不把半截数据当真相（早期 ``monitor.py`` 的毛病）。
    """

    def __init__(self, log: Path, npz: Path, mesh: str, csv: str):
        self.lock = threading.RLock()
        self.log = Path(log)
        self.npz = Path(npz)
        self.mesh_path = mesh
        self.csv_path = csv
        #: 只读模式下后端信息来自 NPZ 的 config_json，不能建 FieldSystem
        self.cfg: Optional[dict] = None
        self.geometry: Optional[dict] = None
        self.geometry_version = -1
        self.rho: Optional[list] = None
        self.snapshot: Optional[dict] = None
        self.snapshot_error: Optional[str] = None
        self._snapshot_key: Optional[tuple] = None
        self._design_idx: Optional[np.ndarray] = None
        self._topology = None
        self._mesh_key: Optional[tuple] = None
        self.rows: List[dict] = []
        self.labels: Dict[str, str] = dict(_DEFAULT_LABELS)
        self.target: Optional[int] = None
        self.status = "waiting"
        self.msg = "等待运行产物（日志尚未出现）"
        self.error: Optional[str] = None
        self.run_id = 1
        #: 已消费的**完整行**数（增量读日志用，见 ``refresh``）
        self._n_lines = 0
        self._pid: Optional[int] = None
        self._terminal = False
        self._labels_version = 0

    # ------------------------------------------------------------ 日志
    def refresh(self) -> None:
        """增量重读日志；只消费尚未处理过的完整行。

        ⚠ **必须增量**：``refresh()`` 每次轮询都会调用，整文件重扫会把每一行
        重复 append 进 ``self.rows``（实测第 2 次轮询就把轨迹翻倍）。
        ``self._n_lines`` 记录已消费的行数，只有新增行才解析。
        """
        with self.lock:
            if not self.log.exists():
                if self.status != "waiting":
                    self.status, self.msg = "waiting", "等待运行产物（日志尚未出现）"
                else:
                    self.msg = "等待运行产物（日志尚未出现）"
                self.error = None
                return
            try:
                text = self.log.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                self._fail(f"日志读取失败：{exc}")
                return
            # 只消费完整行；最后一行若没有换行符说明生产端还在写它。
            lines = [ln for ln in text.splitlines(keepends=True) if ln.endswith("\n")]
            if len(lines) < self._n_lines:
                # 日志被截断/换新（生产端重启）—— 从头重来，不留旧轨迹。
                self._n_lines = 0
                self.rows = []
            for line in lines[self._n_lines:]:
                self._consume(line)
                self._n_lines += 1
            if self.error is None and not self._terminal:
                self._check_alive()

    def _consume(self, line: str) -> None:
        """解析一行。任何解析失败 → 显式 error，不清空已有行（但页面会标注）。"""
        head, _, rest = line.rstrip("\n").partition(" ")
        try:
            if head == "RUNNING":
                kv = _parse_kv(rest)
                self._pid = int(kv["pid"]) if kv.get("pid") else None
                if kv.get("target"):
                    self.target = int(float(kv["target"]))
                self._terminal = False
                self._set_status("running", "运行中（外部进程）")
            elif head == "LABELS":
                kv = _parse_kv(rest)
                for k, v in kv.items():
                    if k in self.labels:
                        self.labels[k] = str(v)
                self._labels_version += 1
            elif head == "EVAL":
                self.rows.append(self._parse_eval(_parse_kv(rest)))
            elif head == "CHECKPOINT":
                pass                       # 快照由 NPZ 的 mtime 驱动，此处仅标记
            elif head == "COMPLETE":
                self._terminal = True
                self._set_status("done", "完成（COMPLETE）")
            elif head == "FAILED":
                self._terminal = True
                self._fail("运行失败：" + rest.strip())
            elif head == "STOP:":
                self._terminal = True
                self._set_status("stopped", "已停止：" + rest.strip())
            else:
                # 认不出的行一律忽略：生产端加日志不得让消费者崩。
                return
        except Exception as exc:  # noqa: BLE001
            self._fail(f"{head} 行解析失败：{exc}（原文：{line.strip()[:160]}）")

    def _parse_eval(self, kv: Dict[str, str]) -> dict:
        """``EVAL`` 行 → 一行的轨迹数据。缺列填 None（页面显示 —）。"""
        def num(key: str) -> Optional[float]:
            v = kv.get(key)
            if v is None or v in ("-", ""):
                return None
            f = float(v)
            return f if np.isfinite(f) else None

        stage = kv.get("stage")
        n = kv.get("n")
        obj = num("obj")
        if stage is None or n is None or obj is None:
            raise ValueError("EVAL 缺 stage/n/obj")
        cons = [num(f"con{i}") for i in (1, 2, 3, 4)]
        return {"stage": int(float(stage)), "n": int(float(n)), "obj": obj,
                "seconds": num("seconds"), "cons": cons}

    def _check_alive(self) -> None:
        """日志在、但没终态标记时，确认生产端进程是否还活着。

        只认 Windows 的 ``GetExitCodeProcess == 259``（STILL_ACTIVE）；
        查不到 pid 或查询失败 → ``unknown``，**绝不乐观假设还在跑**。
        """
        if self._pid is None:
            self._set_status("unknown", "日志已出现，但无 RUNNING pid，无法确认进程状态")
            return
        try:
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_ulong)]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x1000, False, int(self._pid))
            code = ctypes.c_ulong()
            ok = bool(handle) and bool(
                kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
            alive = ok and code.value == 259
            if handle:
                kernel.CloseHandle(handle)
            if alive:
                self._set_status("running", "运行中（外部进程）")
            else:
                self._set_status(
                    "unknown",
                    "进程已退出或不可查询，未见 COMPLETE/FAILED/STOP；不判定成功")
        except Exception as exc:  # noqa: BLE001
            self._set_status("unknown", f"进程状态查询失败：{exc}")

    def _set_status(self, s: str, msg: str) -> None:
        self.status, self.msg = s, msg
        if s != "error":
            self.error = None

    def _fail(self, msg: str) -> None:
        self.status, self.msg, self.error = "error", msg, msg

    # ------------------------------------------------------------ NPZ 快照
    def _npz_candidates(self) -> List[Path]:
        """``--npz`` 既可以是具体文件，也可以是 ``<前缀>``（探测 ``_stage*``）。"""
        if self.npz.suffix == ".npz":
            return [self.npz] if self.npz.exists() else []
        out = list(self.npz.parent.glob(self.npz.name + "_stage*.npz"))
        out += list(self.npz.parent.glob(self.npz.name + ".npz"))
        return sorted(out, key=lambda p: p.stat().st_mtime)

    def update_snapshot(self) -> None:
        """后台线程：轮询 NPZ，ρ 变了才重画。不 import 计算栈。"""
        while True:
            try:
                cands = self._npz_candidates()
                if cands:
                    path = cands[-1]
                    key = (str(path), path.stat().st_mtime_ns)
                    if key != self._snapshot_key:
                        self._load_snapshot(path, key)
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.snapshot_error = f"快照读取失败（不显示旧密度）：{exc}"
                    self.rho = None
            time.sleep(2)

    def _load_snapshot(self, path: Path, key: tuple) -> None:
        with np.load(path, allow_pickle=False) as z:
            if "x_calls" not in z.files:
                raise ValueError(f"{path.name} 缺 x_calls")
            x = np.asarray(z["x_calls"])
            n = int(x.shape[0])
            last = x[-1].copy() if n else None
            cfg_json = str(z["config_json"]) if "config_json" in z.files else None
            # ``run_auto_schedule`` 把 stage 写成字符串（"S0"），legacy runner 写 int。
            # 不能直接 ``int(z["stage"])``：那会让整个快照读取失败、页面不显示密度。
            stage = _coerce_stage(z["stage"]) if "stage" in z.files else None
            mesh = str(z["mesh_path"]) if "mesh_path" in z.files else self.mesh_path
            csv = str(z["csv_path"]) if "csv_path" in z.files else self.csv_path
            if "labels" in z.files:
                lab = [str(s) for s in np.asarray(z["labels"]).tolist()]
        if last is None:
            raise ValueError(f"{path.name} 的 x_calls 为空")
        # 几何只依赖网格：网格换了才重建（此时才需要计算栈）。
        mk = (str(mesh), str(csv))
        if self._design_idx is None or mk != self._mesh_key:
            top = _load_topology(mesh, csv)
            geometry = _build_geometry(top)
            self._topology = top
            self._mesh_key = mk
            with self.lock:
                self.geometry = geometry
                self._design_idx = np.asarray(geometry["design_idx"], dtype=int)
                self.geometry_version += 1
        _load_compute_stack()          # density_from_x 由计算栈提供
        rho = density_from_x(last, self._topology)
        if not np.isfinite(rho).all():
            raise ValueError("快照密度含 NaN/Inf")
        with self.lock:
            self.rho = rho[self._design_idx].tolist()
            self.snapshot = {"stage": stage, "path": path.name, "evaluation": n}
            self._snapshot_key = key
            self.snapshot_error = None
            if cfg_json:
                try:
                    self.cfg = json.loads(cfg_json)
                except Exception:  # noqa: BLE001
                    self.cfg = None

    # ------------------------------------------------------------ 对外 payload
    def state_payload(self) -> dict:
        self.refresh()
        with self.lock:
            row = self.rows[-1] if self.rows else {}
            snap = self.snapshot
            note = (self.snapshot_error or
                    "等待首个 NPZ 快照" if not snap else
                    f"快照：阶段 {snap['stage']} / 第 {snap['evaluation']} 次评价"
                    "；按生产端节奏落盘，非实时密度")
            cons = row.get("cons")
            return {
                "read_only": True,
                "status": self.status,
                "msg": "只读监控 · " + self.msg,
                "error": self.error,
                "power": "",
                "step": row.get("n", 0),
                "target": self.target,
                "stage": row.get("stage"),
                "obj": row.get("obj"),
                "cons": cons,
                "dx_rel": None,
                "times": {"fun_ms": (row["seconds"] * 1e3
                                     if row.get("seconds") is not None else None)},
                "al_mu": None, "al_n_call": None,
                "elapsed_s": None, "eta_s": None,
                "history_len": len(self.rows),
                "geometry_version": self.geometry_version,
                "rho": None if self.error else self.rho,
                "run_id": self.run_id,
                "cfg": self.cfg,
                "schema": [],
                "mesh_path": self.mesh_path,
                "csv_path": self.csv_path,
                "saves": [],
                "n_topo": (self.geometry or {}).get("n_topo"),
                "labels": dict(self.labels),
                "labels_version": self._labels_version,
                "snapshot": snap,
                "snapshot_note": note,
            }

    def history_payload(self, start: int) -> dict:
        with self.lock:
            n_all = len(self.rows)
            start = max(0, min(int(start), n_all))
            rows = self.rows[start:]
            m = len(rows)
            return {
                "start": start, "n": n_all,
                "step": [r["n"] for r in rows],
                "obj": [r["obj"] for r in rows],
                "cons": [r["cons"] for r in rows],
                "fun_ms": [(r["seconds"] * 1e3 if r["seconds"] is not None else None)
                           for r in rows],
                "em_ms": [None] * m, "mech_ms": [None] * m,
                "step_ms": [None] * m, "dx_rel": [None] * m,
            }

    def list_saves(self) -> List[str]:
        return []


def _parse_kv(rest: str) -> Dict[str, str]:
    """``key=value key=value`` → dict。值里不含空格（契约 §2）。"""
    out: Dict[str, str] = {}
    for tok in rest.split():
        k, sep, v = tok.partition("=")
        if sep and k:
            out[k] = v
    return out


_TOPOLOGY_CACHE: Dict[tuple, Any] = {}


def _load_topology(mesh: str, csv: str):
    """**唯一**在只读模式下会拉起计算栈的地方（且只为几何，不为求解）。

    只在网格换过时调一次；缓存避免重复构建。
    """
    key = (str(mesh), str(csv))
    if key not in _TOPOLOGY_CACHE:
        _load_compute_stack()
        fs = FieldSystem(RunConfig(), str(mesh), str(csv))
        _TOPOLOGY_CACHE[key] = fs.top
    return _TOPOLOGY_CACHE[key]


def _readonly_index_html(text: str) -> str:
    """把可写控制台的 index.html 改写成只读监视版。

    只做两件事（都在服务端完成，避免前端分叉）：
    1. 给所有表单控件加 ``disabled``（**首帧之前**就禁，不给可点击窗口）；
    2. 标题/分区名改成监视语义，避免误以为这里能启动优化。
    """
    import re
    text = re.sub(r"<(button|input|select|textarea)\b", r"<\1 disabled", text)
    text = text.replace("SynRM 拓扑优化（MMA + 网页控制台）",
                        "SynRM 拓扑优化 · 只读监视")
    text = text.replace("<title>SynRM 拓扑优化 GUI</title>",
                        "<title>SynRM 拓扑优化 · 只读监视</title>")
    text = text.replace(">运行控制<", ">运行信息（只读，控制请用命令行）<")
    text = text.replace(">保存 / 加载<", ">结果产物（只读）<")
    text = text.replace(">配置选项", ">配置（来自产物，只读）")
    return text


class MmaControl:
    """``mma_driver.MMA_optimize`` 的 control 鸭子类型（暂停/单步/停止/每步回调）。

    锁纪律：永远 **control 锁 → manager 锁** 的顺序（on_step 里先取 control
    锁再进 manager），外部操作一律走 control 方法，避免与 on_step 死锁。
    """

    def __init__(self, mgr: "OptimizerManager"):
        self.mgr = mgr
        self._lock = threading.Lock()
        self._stop = False
        self._pause = False
        self._single_shot = False

    # ---- mma_driver 回调 ----
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop

    def wait_if_paused(self) -> None:
        while True:
            with self._lock:
                if self._stop or not self._pause:
                    return
            time.sleep(0.05)

    def on_step(self, i_step: int, state: dict) -> None:
        with self._lock:
            if self._single_shot:
                self._single_shot = False
                self._pause = True
                self.mgr.set_status("paused", f"已暂停（第 {i_step} 步，单步）")
        self.mgr.on_optim_step(i_step, state)

    # ---- 外部操作（HTTP handler 调用）----
    def pause(self) -> None:
        with self._lock:
            self._pause = True
        self.mgr.set_status("paused", "暂停中（完成当前步后生效）")

    def resume(self) -> None:
        with self._lock:
            self._pause = False
        if not self.stop_requested():
            self.mgr.set_status("running", "运行中")

    def step_once(self) -> None:
        with self._lock:
            self._single_shot = True
            self._pause = False
        if not self.stop_requested():
            self.mgr.set_status("running", "单步模式：跑一步后暂停")

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._pause = False
        self.mgr.set_status("stopping", "停止请求已发出（当前步结束后）")


class OptimizerManager:
    """优化任务的所有者：FieldSystem + 后台线程 + 历史/几何/检查点。"""

    def __init__(self, saves_dir: Path):
        self.lock = threading.RLock()
        self.saves_dir = Path(saves_dir)
        self.saves_dir.mkdir(parents=True, exist_ok=True)
        self.fs: Optional[FieldSystem] = None
        self.cfg: RunConfig = RunConfig()   # 空闲也用默认配置，页面配置表单始终可编辑
        self.mesh_path: Optional[str] = None
        self.csv_path: Optional[str] = None
        self.control: Optional[MmaControl] = None
        self.thread: Optional[threading.Thread] = None
        self.status = "idle"
        self.msg = "空闲"
        self.error: Optional[str] = None
        self.step_count = 0
        self.steps_target = 0
        self.last_mma_state: Optional[dict] = None
        self._pending: Optional[dict] = None
        self._latest_times: dict = {}
        self._latest_rho: Optional[list] = None
        self.history = _empty_history()
        self.geometry: Optional[dict] = None
        self.geometry_version = 0
        self._design_idx = np.empty(0, dtype=int)
        self.resume_state: Optional[dict] = None
        self._resume_fs_state: Optional[dict] = None
        self._build_key: Optional[tuple] = None
        self.run_id = 0                # 每次全新开始/加载自增，浏览器据此重置本地历史
        self._start_t = 0.0
        self._last_step_t = 0.0
        self._em_acc: List[float] = []
        self._mech_acc: List[float] = []
        self._prev_x: Optional[np.ndarray] = None
        self._orig_warper = None
        self._orig_em = None
        self._orig_mech = None

    # ------------------------------------------------------------ 状态
    def set_status(self, s: str, msg: Optional[str] = None) -> None:
        with self.lock:
            self.status = s
            if msg is not None:
                self.msg = msg

    # ------------------------------------------------------------ 启动/运行
    def start(self, body: dict) -> Tuple[bool, str]:
        _load_compute_stack()
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False, "优化已在运行/构建中，先停止再启动"
            cfg_dict = body.get("config") or (
                self.cfg.to_dict() if self.cfg else {})
            if body.get("steps") is not None:
                cfg_dict = dict(cfg_dict)
                cfg_dict.setdefault("mma", {})["max_step"] = int(body["steps"])
            # positions_deg 是 text 字段 ⇒ 前端给字符串，解析成 List[float]
            _pd = cfg_dict.get("time", {}).get("positions_deg")
            if isinstance(_pd, str):
                cfg_dict["time"]["positions_deg"] = [
                    float(x) for x in _pd.replace(";", ",").split(",")
                    if x.strip()]
            cfg = RunConfig.from_dict(cfg_dict)
            mesh = _norm_path(body.get("mesh_path") or _DEFAULT_MESH)
            csv = _norm_path(body.get("csv_path") or _resolve_csv(cfg))
            if not Path(mesh).exists():
                return False, f"网格不存在：{mesh}"
            if not Path(csv).exists():
                return False, f"B-H 曲线不存在：{csv}"
            resume_state = self.resume_state
            self.resume_state = None
            self.cfg = cfg
            self.mesh_path = mesh
            self.csv_path = csv
            self.steps_target = cfg.mma.max_step
            self.run_id += 1
            if resume_state is None:                 # 全新开始：清空历史
                self.history = _empty_history()
                self.step_count = 0
                self.last_mma_state = None
                self.geometry = None
            self.error = None
            self.thread = threading.Thread(
                target=self._run, args=(cfg, mesh, csv, resume_state),
                daemon=True)
            self.thread.start()
            return True, "已启动"

    def _run(self, cfg: RunConfig, mesh: str, csv: str,
             resume_state: Optional[dict]) -> None:
        try:
            self.set_status("building", "构建 FieldSystem…")
            key = (cfg.to_json(), mesh, csv)
            if resume_state is not None and (
                    self.fs is None or self._build_key != key):
                raise RuntimeError(
                    "检查点续跑要求配置/网格与保存时一致；如要改配置请先停止后从头开始")
            if self.fs is None or self._build_key != key:
                self.fs = FieldSystem(cfg, mesh, csv)
                self._build_key = key
            if self._resume_fs_state is not None:
                self.fs.restore_checkpoint_state(self._resume_fs_state)
                self._resume_fs_state = None
            self.geometry = _build_geometry(self.fs.top)
            self.geometry_version += 1
            self._design_idx = np.asarray(self.geometry["design_idx"], dtype=int)
            self._em_acc = []
            self._mech_acc = []
            self._prev_x = None
            self._pending = None
            self._latest_times = {}
            self._latest_rho = None
            self._install_wrappers(self.fs)
            ctl = MmaControl(self)
            self.control = ctl
            self._start_t = time.time()
            self._last_step_t = self._start_t
            self.step_count = resume_state["i_step"] if resume_state else 0
            self.set_status("running", f"运行中（目标 {self.steps_target} 步，"
                                       f"优化器 {cfg.runtime.optimizer}）")
            self.fs.optimize(max_step=self.steps_target, control=ctl,
                             initial_state=resume_state,
                             optimizer=cfg.runtime.optimizer)
            if ctl.stop_requested():
                self.set_status("stopped", f"已停止（第 {self.step_count} 步）")
            else:
                self.set_status("done", f"完成（{self.step_count} 步）")
        except Exception as exc:  # noqa: BLE001
            self.error = traceback.format_exc()
            self.set_status("error", f"出错：{exc}")
        finally:
            self._uninstall_wrappers()
            with self.lock:
                self.thread = None
                self.control = None

    def _install_wrappers(self, fs: FieldSystem) -> None:
        """计时包装（数值路径不动，手法同 probe_mma_full）。"""
        self._orig_em = obj_mod.em_segment
        self._orig_mech = obj_mod.mech_worker
        self._orig_warper = fs.warper

        def em_w(*a, **k):
            t0 = time.perf_counter()
            r = self._orig_em(*a, **k)
            self._em_acc.append(time.perf_counter() - t0)
            return r

        def mech_w(*a, **k):
            t0 = time.perf_counter()
            r = self._orig_mech(*a, **k)
            self._mech_acc.append(time.perf_counter() - t0)
            return r

        obj_mod.em_segment = em_w
        obj_mod.mech_worker = mech_w

        def warper_w(x):
            t0 = time.perf_counter()
            res = self._orig_warper(x)
            fun_ms = (time.perf_counter() - t0) * 1e3
            xa = np.asarray(x, dtype=float)
            rho = density_from_x(xa, fs.top)
            self._pending = {
                "fun_ms": fun_ms,
                "em_ms": self._em_acc[-1] * 1e3 if self._em_acc else float("nan"),
                "mech_ms": self._mech_acc[-1] * 1e3 if self._mech_acc else float("nan"),
                "rho": rho[self._design_idx].tolist(),
                "x": xa,
            }
            return res

        fs.warper = warper_w

    def _uninstall_wrappers(self) -> None:
        try:
            obj_mod.em_segment = self._orig_em
            obj_mod.mech_worker = self._orig_mech
            if self.fs is not None and self._orig_warper is not None:
                self.fs.warper = self._orig_warper
        except Exception:  # noqa: BLE001
            pass
        self._orig_em = self._orig_mech = self._orig_warper = None

    # ------------------------------------------------------------ 每步回调
    def on_optim_step(self, i_step: int, state: dict) -> None:
        with self.lock:
            now = time.time()
            self.step_count = int(i_step)
            self.last_mma_state = state
            f = np.asarray(state["f_val"])
            p = self._pending or {}
            step_ms = (now - self._last_step_t) * 1e3
            self.history["step"].append(int(i_step))
            self.history["obj"].append(float(f[0]))
            self.history["cons"].append([float(v) for v in f[1:]])
            self.history["fun_ms"].append(float(p.get("fun_ms", float("nan"))))
            self.history["em_ms"].append(float(p.get("em_ms", float("nan"))))
            self.history["mech_ms"].append(float(p.get("mech_ms", float("nan"))))
            self.history["step_ms"].append(step_ms)
            x = np.asarray(state["x"])
            if self._prev_x is not None:
                dx = np.linalg.norm(x - self._prev_x) / (np.linalg.norm(x) + 1e-20)
            else:
                dx = 0.0
            self._prev_x = np.copy(x)
            self.history["dx_rel"].append(float(dx))
            self._latest_times = {k: v for k, v in p.items()
                                  if k not in ("rho", "x")}
            self._latest_times["step_ms"] = step_ms
            self._latest_rho = p.get("rho")
            self._pending = None
            self._last_step_t = now

    # ------------------------------------------------------------ 控制
    def pause(self) -> Tuple[bool, str]:
        ctl = self.control
        if ctl is None:
            return False, "没有运行中的优化"
        ctl.pause()
        return True, "已暂停（完成当前步后生效）"

    def resume(self) -> Tuple[bool, str]:
        ctl = self.control
        if ctl is None:
            return False, "没有运行中的优化"
        ctl.resume()
        return True, "已继续"

    def step_once(self) -> Tuple[bool, str]:
        ctl = self.control
        if ctl is None:
            return False, "没有运行中的优化"
        ctl.step_once()
        return True, "单步：跑一步后暂停"

    def stop(self) -> Tuple[bool, str]:
        ctl = self.control
        if ctl is None:
            return False, "没有运行中的优化"
        ctl.stop()
        return True, "停止请求已发出"

    # ------------------------------------------------------------ 保存/加载
    def save(self, name: str) -> Tuple[bool, str]:
        with self.lock:
            if self.fs is None or self.last_mma_state is None:
                return False, "还没有可保存的优化状态（至少先跑一步）"
            data = {
                "format_version": _FORMAT_VERSION,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": self.status,
                "step": int(self.step_count),
                "mesh_path": self.mesh_path,
                "csv_path": self.csv_path,
                "cfg": self.cfg.to_dict(),
                "mma": _state_to_jsonable(self.last_mma_state),
                "fs": self.fs.checkpoint_state(),
                "history": self.history,
            }
            safe = (name.strip() or f"step{self.step_count}_{int(time.time())}")
            p = self.saves_dir / f"{safe}.json"
            p.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                         encoding="utf-8")
            return True, str(p)

    def load(self, name: str) -> Tuple[bool, str]:
        _load_compute_stack()
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False, "优化正在运行，先停止再加载"
            p = self.saves_dir / f"{name.strip()}.json"
            if not p.exists():
                return False, f"没有检查点：{name}"
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("format_version") != _FORMAT_VERSION:
                return False, f"不支持的检查点版本：{data.get('format_version')}"
            self._resume_fs_state = data["fs"]
            self.resume_state = data["mma"]
            self.cfg = RunConfig.from_dict(data["cfg"])
            self.mesh_path = _norm_path(data["mesh_path"])
            self.csv_path = _norm_path(data.get("csv_path") or _resolve_csv(self.cfg))
            self.steps_target = self.cfg.mma.max_step
            self.step_count = int(data["step"])
            self.history = data["history"]
            self.last_mma_state = None
            self.geometry = None
            self.fs = None
            self._build_key = None
            self.run_id += 1
            self.error = None
            self.set_status("loading", f"加载 {name}，重建 FieldSystem…")
            self.thread = threading.Thread(
                target=self._rebuild_after_load, args=(name,), daemon=True)
            self.thread.start()
            return True, f"正在加载 {name}"

    def _rebuild_after_load(self, name: str) -> None:
        try:
            self.set_status("loading", f"加载 {name}，重建 FieldSystem…")
            fs = FieldSystem(self.cfg, self.mesh_path, self.csv_path)
            fs.restore_checkpoint_state(self._resume_fs_state)
            self._resume_fs_state = None
            with self.lock:
                self.fs = fs
                self._build_key = (self.cfg.to_json(), self.mesh_path,
                                   self.csv_path)
                self.geometry = _build_geometry(fs.top)
                self.geometry_version += 1
                self._design_idx = np.asarray(self.geometry["design_idx"],
                                              dtype=int)
                self.set_status("paused",
                                f"已加载 {name}（第 {self.step_count} 步），"
                                "点『开始/继续』续跑")
        except Exception as exc:  # noqa: BLE001
            self.error = traceback.format_exc()
            self.set_status("error", f"加载失败：{exc}")
        finally:
            with self.lock:
                self.thread = None

    def list_saves(self) -> List[str]:
        files = list(self.saves_dir.glob("*.json"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files]

    # ------------------------------------------------------------ 对外 payload
    def state_payload(self) -> dict:
        with self.lock:
            cfg = self.cfg.to_dict() if self.cfg else None
            hist = self.history
            elapsed = (time.time() - self._start_t) if self._start_t else 0.0
            eta = None
            if self.step_count and self.steps_target > self.step_count:
                avg = elapsed / max(self.step_count, 1)
                eta = (self.steps_target - self.step_count) * avg
            return {
                "status": self.status,
                "msg": self.msg,
                "error": self.error,
                "power": _ac_power(),
                "step": int(self.step_count),
                "target": int(self.steps_target),
                "obj": hist["obj"][-1] if hist["obj"] else None,
                "cons": hist["cons"][-1] if hist["cons"] else None,
                "dx_rel": hist["dx_rel"][-1] if hist["dx_rel"] else None,
                "times": dict(self._latest_times),
                "al_mu": float(self.fs.al_state.al_mu) if self.fs else None,
                "al_n_call": int(self.fs.al_state.al_n_call) if self.fs else None,
                "elapsed_s": elapsed,
                "eta_s": eta,
                "history_len": len(hist["step"]),
                "geometry_version": self.geometry_version,
                "rho": self._latest_rho,
                "run_id": self.run_id,
                "cfg": cfg,
                "schema": CONFIG_SCHEMA,
                "mesh_path": self.mesh_path,
                "csv_path": self.csv_path,
                "saves": self.list_saves(),
            }

    def history_payload(self, start: int) -> dict:
        with self.lock:
            h = self.history
            n = len(h["step"])
            start = max(0, min(int(start), n))
            return {
                "start": start,
                "n": n,
                "step": h["step"][start:],
                "obj": h["obj"][start:],
                "cons": h["cons"][start:],
                "fun_ms": h["fun_ms"][start:],
                "em_ms": h["em_ms"][start:],
                "mech_ms": h["mech_ms"][start:],
                "step_ms": h["step_ms"][start:],
                "dx_rel": h["dx_rel"][start:],
            }


def _resolve_csv(cfg: RunConfig) -> str:
    csv = cfg.material.bh_curve_csv
    if ":" in csv or csv.startswith("/"):
        return csv
    return str(_REPO_ROOT / "legacy" / "pytorch" / csv)


def _norm_path(p: str) -> str:
    """统一成绝对路径（相对路径按仓库根解析），避免 build_key 比较失配。"""
    return str(Path(p).resolve())


def _state_to_jsonable(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif k == "f_val":
            out[k] = list(np.asarray(v).tolist())
        else:
            out[k] = v
    return out


def browse_native(title: str, initialdir: str) -> dict:
    """「浏览」按钮：启动独立进程弹**系统原生**文件选择对话框。

    Web 无法直接拿本地绝对路径（``<input type=file>`` 只给 fakepath），
    所以由 server 用 ``subprocess`` 拉起 ``file_dialog.py``（tkinter 系统
    对话框，Windows 原生文件选择器），把用户选中的绝对路径打回前端。
    用户取消 → ``{ok:True, cancelled:True}``；选中 → ``{ok:True, path}``。

    ⚠ 本端点是**阻塞**的：等待用户在系统对话框里做完选择才返回。期间
    ``ThreadingHTTPServer`` 的其他请求（/api/state 轮询）不受影响。
    """
    script = Path(__file__).parent / "file_dialog.py"
    try:
        cmd = [sys.executable, "-u", str(script),
               "--title", title, "--dir", initialdir]
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
        out = proc.stdout.decode("utf-8", "replace").strip()
        if out:
            return {"ok": True, "path": out}
        return {"ok": True, "cancelled": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"系统文件对话框启动失败：{exc}"}


# ---------------------------------------------------------------- HTTP
def _json_safe(obj):
    """递归把 NaN/Inf 换成 None —— JSON 标准不接受 NaN，浏览器 JSON.parse 会拒收
    整份响应（实测 mech_ms 在力学段被跳过的步是 NaN，页面轮询 23 次全挂）。"""
    if isinstance(obj, (float, np.floating)):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    mgr: OptimizerManager = None
    #: ``True`` 时所有控制端点 403，页面按钮在首帧前就禁用（见 ``_serve_static``）
    read_only: bool = False
    server_version = "optim-gui/1.0"

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _send(self, code: int, obj, ctype: str = "application/json; charset=utf-8") -> None:
        body = json.dumps(_json_safe(obj), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            # 客户端在中途断开（刷新/关页/取消 fetch）—— 正常，不必报 traceback。
            pass

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _query(self, key: str, default: str = "") -> str:
        q = self.path.split("?", 1)[1] if "?" in self.path else ""
        for pair in q.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                if k == key:
                    return v
        return default

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            name = path[len("/static/"):]
            ctype = {".js": "application/javascript; charset=utf-8",
                     ".css": "text/css; charset=utf-8",
                     ".html": "text/html; charset=utf-8"}.get(
                         Path(name).suffix, "application/octet-stream")
            self._serve_static(name, ctype)
        elif path == "/api/state":
            self._send(200, self.mgr.state_payload())
        elif path == "/api/history":
            start = self._query("start", "0")
            try:
                start_i = int(start)
            except ValueError:
                start_i = 0
            self._send(200, self.mgr.history_payload(start_i))
        elif path == "/api/geometry":
            with self.mgr.lock:
                self._send(200, {"version": self.mgr.geometry_version,
                                 "geometry": self.mgr.geometry})
        elif path == "/api/saves":
            self._send(200, {"saves": self.mgr.list_saves()})
        else:
            self._send(404, {"ok": False, "error": f"未知路径 {path}"})

    def _serve_static(self, name: str, ctype: str) -> None:
        p = _STATIC_DIR / name
        if not p.exists():
            self._send(404, {"ok": False, "error": f"静态文件不存在：{name}"})
            return
        if self.read_only and name == "index.html":
            # 只读模式：在 HTML 里就把控件禁掉、并把标题改成监视语义。
            # 必须在首帧之前生效 —— 靠 JS 事后 disable 会有一个可点击的窗口。
            body = _readonly_index_html(p.read_text(encoding="utf-8")).encode("utf-8")
        else:
            body = p.read_bytes()
        try:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            # 客户端在中途断开 —— 正常，不必报 traceback（同 _send）。
            pass

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if self.read_only:
            self._send(403, {"ok": False,
                             "error": "只读监视模式：所有控制端点均已禁用"})
            return
        _load_compute_stack()
        body = self._read_json()
        mgr = self.mgr
        if path == "/api/start":
            ok, msg = mgr.start(body)
            self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        elif path == "/api/pause":
            ok, msg = mgr.pause()
            self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        elif path == "/api/resume":
            ok, msg = mgr.resume()
            self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        elif path == "/api/step":
            ok, msg = mgr.step_once()
            self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        elif path == "/api/stop":
            ok, msg = mgr.stop()
            self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        elif path == "/api/browse-native":
            title = body.get("title") or "选择文件"
            initialdir = body.get("dir") or ""
            self._send(200, browse_native(title, initialdir))
        elif path == "/api/save":
            ok, msg = mgr.save(body.get("name") or "")
            self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        elif path == "/api/load":
            ok, msg = mgr.load(body.get("name") or "")
            self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        else:
            self._send(404, {"ok": False, "error": f"未知路径 {path}"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--saves-dir", type=str, default=str(_SAVES_DIR))
    ap.add_argument("--read-only", action="store_true",
                    help="只读监视模式：数据全部来自外部运行的日志 + NPZ 产物，"
                         "不 import 计算栈、不创建优化、所有 POST 返回 403。")
    ap.add_argument("--log", type=str, default="",
                    help="只读模式的事件日志路径（契约见 "
                         "docs/GUI只读监控产物契约.md）。")
    ap.add_argument("--npz", type=str, default="",
                    help="只读模式的结果快照：具体 .npz 文件，或 <前缀>"
                         "（按 _stage<N>.npz 探测，取 mtime 最新）。")
    ap.add_argument("--mesh", type=str, default=str(_DEFAULT_MESH),
                    help="只读模式下 NPZ 未携带 mesh_path 时用的网格（仅供几何）。")
    ap.add_argument("--csv", type=str, default=str(_DEFAULT_CSV),
                    help="同上，B-H 曲线。")
    args = ap.parse_args()

    if args.read_only:
        if not args.log or not args.npz:
            ap.error("--read-only 需要同时给出 --log 与 --npz")
        mgr = LogManager(Path(args.log), Path(args.npz),
                         _norm_path(args.mesh), _norm_path(args.csv))
        Handler.read_only = True
        threading.Thread(target=mgr.update_snapshot, daemon=True).start()
        print(f"SynRM 拓扑优化 GUI（只读监视）：http://{args.host}:{args.port}")
        print(f"日志：{args.log}")
        print(f"快照：{args.npz}")
        print("所有控制端点已禁用；启动/暂停/停止请在命令行进行。")
    else:
        _load_compute_stack()
        mgr = OptimizerManager(Path(args.saves_dir))
        Handler.read_only = False
        print(f"SynRM 拓扑优化 GUI：http://{args.host}:{args.port}")
        print(f"默认网格：{_DEFAULT_MESH}")
        print(f"检查点目录：{args.saves_dir}")
        print("BLAS 已钉单线程（OMP/OPENBLAS/MKL=1）；计时受电源状态影响")

    Handler.mgr = mgr
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("按 Ctrl-C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
