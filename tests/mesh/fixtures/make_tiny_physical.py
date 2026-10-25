#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests/mesh/fixtures/make_tiny_physical.py —— 阶段 3 夹具生成器

生成 ``tiny_physical.msh``：一个单位正方形拆成 2 个三角形，带 2 个面物理组
（region.alpha / region.beta）和 1 个边界物理组（bnd.outer）。

用途：解析器（$PhysicalNames + $Entities）的端到端测试夹具 —— 冻结网格没有
$PhysicalNames，而这里有一个真实 gmsh 生成的物理组网格。确定性：gmsh 4.13.1
固定版本，重跑得到逐字节相同的文件（提交时自带本脚本以便复核）。

用法：
    python tests/mesh/fixtures/make_tiny_physical.py
"""
import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.model.add("tiny_physical")

p1 = gmsh.model.geo.addPoint(0, 0, 0, 10.0)
p2 = gmsh.model.geo.addPoint(1, 0, 0, 10.0)
p3 = gmsh.model.geo.addPoint(1, 1, 0, 10.0)
p4 = gmsh.model.geo.addPoint(0, 1, 0, 10.0)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)
l5 = gmsh.model.geo.addLine(p1, p3)

c1 = gmsh.model.geo.addCurveLoop([l1, l2, -l5])   # 三角形 p1-p2-p3
c2 = gmsh.model.geo.addCurveLoop([l5, l3, l4])    # 三角形 p1-p3-p4
s1 = gmsh.model.geo.addPlaneSurface([c1])
s2 = gmsh.model.geo.addPlaneSurface([c2])
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(2, [s1], tag=10)
gmsh.model.setPhysicalName(2, 10, "region.alpha")
gmsh.model.addPhysicalGroup(2, [s2], tag=20)
gmsh.model.setPhysicalName(2, 20, "region.beta")
gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], tag=101)
gmsh.model.setPhysicalName(1, 101, "bnd.outer")

gmsh.option.setNumber("Mesh.MeshSizeMax", 10.0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 10.0)   # 整个正方形只出 2 个三角形
gmsh.model.mesh.generate(2)
gmsh.write("tiny_physical.msh")
print("OK")
gmsh.finalize()
