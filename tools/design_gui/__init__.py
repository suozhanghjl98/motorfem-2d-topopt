# -*- coding: utf-8 -*-
"""tools/design_gui —— 手绘转子初始结构的网页 GUI。

模块划分：
- ``core``   ：纯逻辑（拓扑 bundle、几何 payload、滤波缓存、NPZ 导入导出）
- ``jobs``   ：优化子进程管理（命令拼装、日志增量解析、生命周期）
- ``server`` ：stdlib HTTP 服务 + 静态页

⚠ 本包不修改任何数值内核；导出的 ``x0`` 走既有
``tools/scripts/run_auto_schedule.py --x0-npz`` 路径。
"""

__version__ = "1.0.0"
