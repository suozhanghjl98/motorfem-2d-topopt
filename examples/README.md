# examples/ —— 示例输入

| 文件 | 说明 |
|---|---|
| `hand_design.npz` | 一次真实手绘的转子密度场（6s4p，可编辑主域 1207 单元，滤波半径 0.9 mm），可直接作为优化起点 |

## 这份 NPZ 从哪来

2026-10-25 在 `tools/design_gui` 网页上手工绘制并导出，随后被 `run_auto_schedule.py`
用作 `--x0-npz` 起点。它与本仓库的 `legacy/pytorch/my_gmsh/model_6s4p.msh`
同源（`mesh_sha256` 一致），因此按元素顺序直接对应，省去换算。

关键字段：

| key | 形状 | 含义 |
|---|---|---|
| `rho_master` | (1207,) | 手绘目标密度（优化真实看到的是 `W @ x0`） |
| `x0` | (1207,) | 给 `run_auto_schedule.py --x0-key x0` 的设计变量 |
| `rho_eff_master` | (1207,) | `W @ x0`，即优化实际起点密度 |
| `master_cells` / `design_master` | — | 主/从属映射，导入时按元素 id 重排 |
| `filter_radius_m` | 标量 | 生成时的滤波半径（本文件 0.9 mm） |
| `mean_density_master` | 标量 | 面积加权平均密度（本文件 ≈0.822） |

## 自己画一份

```bash
python -u tools/design_gui/server.py --port 8766
# 浏览器打开 http://127.0.0.1:8766，画完点“导出 NPZ”
```

导出文件落在 `tools/_out/design_gui/designs/`。把它的路径交给优化：

```bash
python -u tools/scripts/run_auto_schedule.py --motor 6s4p \
  --x0-npz tools/_out/design_gui/designs/<时间戳>_<名称>.npz --x0-key x0 \
  --filter-radius-mm 0.9 --max-density 0.85 \
  --stages "S0:T=40,R=80" --max-eval-per-stage 10 --n-min 10 \
  --max-budget-shrinks 0 --max-relax-back 0 --wall-seconds 1800 \
  --out tools/_out/demo/hand_run
```

`--filter-radius-mm` 必须与导出时一致，`--max-density` 要高于手绘的平均密度，
否则优化第一步就会先把材料削掉、抹平手绘结构。
