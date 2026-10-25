"""motorfem.geometry —— 阶段 5F：几何参数化。

把散落在多个 gmsh 生成脚本里的几何工具与设计公式收敛成一份：
- ``base.py``：磁障参数 → 几何节点（``barrier_test.py`` 提取）+ 旋转/镜像
- ``periodic.py``：周期边界仿射矩阵（统一两套 gmsh 的符号约定）
- ``design.py``：``design.m`` 的等磁路宽尺寸估算公式（参数预填层）

**注意**：不 import gmsh —— 生成脚本会弹窗阻塞（CLAUDE.md 有记录），
这些是纯几何/公式函数，可独立测试。
"""
