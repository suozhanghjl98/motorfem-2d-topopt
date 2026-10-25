"""motorfem.fem —— 阶段 4：求解器链组装（FieldSystem）。

阶段 1 的 19 个数值内核（``motorfem/kernels/``）+ 阶段 2 的参数对象
（``motorfem/config/``）与网格契约（``motorfem/mesh/``）都已就位。本包把它们
组装成一条**可调用**的求解器链，替代 legacy ``nonlinear_solver_energy_MMA10.py``
的模块级脚本（``obj_func_parallel_AL_v2`` + ``optimizer_func_warper_AL_v2`` +
``MMA_optimize``）。

验收（``docs/重构设计文档.md`` §1.7 第 2 条）：新代码跑同样输入，golden 四层
L1~L4 **全部** ``np.array_equal``。

分层：
    mesh_builder.py    L1 组装：网格 / 区域 / CSR / 初始密度（本文件所在批次）
    init_density.py    初始密度的 RNG 重放（legacy 驱动层初始化，非内核）
    preassembly.py     L2 预组装（下一批）
    em_worker.py       L3 单步电磁（下一批）
    mech_worker.py     L4 力学（下一批）
    objective.py       目标聚合（下一批）
    system.py          FieldSystem orchestrator（下一批）
    mma_driver.py      MMA_optimize 驱动移植（下一批）
"""
