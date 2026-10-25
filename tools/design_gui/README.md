# tools/design_gui —— 手绘转子初始结构 GUI

用鼠标在真实网格上画转子密度场（ρ），再以它为**全自由度（6s4p：1207 单元）**
优化起点跑 `run_auto_schedule.py`。动机：低维极坐标傅里叶基（K·N_r）虽然在投影
时能保留复杂结构，但在低维基上继续优化只能在局部权衡转矩/应力，生成不出想要的
复杂拓扑；把拓扑交给用户、把细节交给优化器。

## 启动

```bash
# 仓库根目录
PYTHONIOENCODING=utf-8 python -u tools/design_gui/server.py --port 8766
# 浏览器打开 http://127.0.0.1:8766
```

Windows 可直接双击 `start_server.cmd`（默认 8766）。

- 几何在后台构建（`build_field_topology`，约 8~15 s）；页面显示“构建中”，完成后自动加载。
- 服务只绑定 `127.0.0.1`；POST 需要页面注入的 token（`--no-token` 仅本机调试用）。
- 服务端**不做 FEM 求解**；优化在独立子进程里跑，不会与 GUI 争用 CHOLMOD。

## 绘制语义（重要）

- 6s4p 的模型是 **0..180° 半平面**，`parts=2`。
- **可编辑主域 = `top.element_topo_opti_m`（1207 个单元，约 0..45° 半极）**；
  其余 45..180° 是镜像从属区（3621 个单元），物理上直接复制主单元密度。
- 页面显示全部 4828 个设计单元；**在任意镜像副本上画也会写回主单元**
  （映射由服务端 `design_master` 给出，不做几何反变换）。
- 默认初始为全实心（ρ=1）；用“挖空/橡皮”画磁障。

工具：画笔、橡皮、直线、矩形、圆、圆环扇区、多边形套索；撤销/重做（120 步）；
滚轮缩放、空格/中键平移；悬停显示单元/半径/角度/密度。

## 滤波半径（关键）

优化路径的物理密度是 `ρ_phys = filter_SIMP @ x`（**滤波器总是应用**）。本 GUI 的
起点 `x0` 默认就是手绘 ρ（`x_mode=direct`），所以：

| 滤波半径 | 行为 |
|---|---|
| 2.0 mm（生产默认） | 手绘细节被明显平滑（二值条纹实测 RMSE≈0.08） |
| 0.5 / 1.0 mm | 折中 |
| ≈0.198 mm（本网格恒等半径） | `W = I`，手绘 ρ 1:1 生效；但棋盘格/收敛风险更高 |

- 恒等半径由服务端按“主单元质心最近邻距离的一半”计算，**不要写死**。
- 半径 **0 非法**（`compute_filter` 的 `dist < radius` 会让行归一化为空）。
- “反滤波求前像”是高级选项：`filter_inverse_rho` 不能恢复小于滤波尺度的细节
  （实测 direct 与 preimage 的 RMSE 基本相同），默认关闭。

启动优化时，GUI **总是显式传** `--filter-radius-mm`，保证子进程的滤波与预览一致。

## 密度约束提示

`con2` 的平均密度上限默认 0.65（面积加权）。手绘“实心 + 磁障”常会超过它，
优化会先大幅削材料、抹掉手绘拓扑。页面实时显示平均密度并在超限时警告；
可在“优化”面板调 `密度上限`（透传 `--max-density`）。

## 导出 NPZ schema

保存到 `tools/_out/design_gui/designs/<时间戳>_<名称>.npz`（原子写）：

| key | 说明 |
|---|---|
| `rho_master` (1207,) | 手绘目标密度 |
| `rho_full` (23453,) | 主值 + 从属复制，其余 0 |
| `x0` (1207,) | 给 `run_auto_schedule --x0-key x0` 的设计变量 |
| `rho_eff_master` / `rho_eff_full` | `W @ x0`（优化实际起点） |
| `master_cells` / `slave_cells` / `design_cells` / `design_master` | 主/从属映射（导入按元素 id 重映射） |
| `filter_radius_m` / `filter_p_exp` / `x_mode` | 生成口径 |
| `rmse_master` / `mean_density_master` | 手绘值 vs 实际起点、面积加权平均密度 |
| `config_hint` (JSON 字符串) | `motor/x0_key/filter_radius_mm/ladder/x_mode` |
| `mesh_path` / `mesh_sha256` / `csv_path` / `created_utc` | 溯源 |

命令行等价物：

```bash
PYTHONIOENCODING=utf-8 python -u tools/scripts/run_auto_schedule.py \
  --motor 6s4p \
  --x0-npz tools/_out/design_gui/designs/<file>.npz --x0-key x0 \
  --stages "S0:T=50,R=80" \
  --filter-radius-mm 0.5 \
  --out tools/_out/design_gui/runs/<name>/run
```

## 导入已有结果

“导入 NPZ…”（原生文件对话框）后选择键。换算规则：

- 长度 = `n_element`（23453）：全场密度，取 `master_cells`；从属不一致只警告；
- 长度 = `n_master`（1207）：
  - 键以 `x` 开头（`x_best/x_selected/x_calls/x0`）⇒ 按**源滤波半径**做 `W@x` 得到密度；
    源半径优先取 NPZ 的 `config_json.filter.filter_radius_m`，其次 `filter_radius_m`；
  - 其他键（含 `rho`）⇒ 直接当密度；
- 长度 = `n_design`（4828）：取前 1207（现有 GUI 的显示顺序）；
- 2D 数组（如 `x_calls`）：默认最后一行（`行=-1`），可指定；
- 带 `master_cells` 的 NPZ 会按元素 id 重排到当前顺序；集合不一致直接拒绝。

**精确续跑**：导入 x 后未编辑且半径与源半径一致时，导出/启动直接复用原 `x0`，
避免 `W(W@x)` 二次平滑。

## 优化子进程

点“保存并启动优化”后：

- 阶段链预设：`S0:T=50,R=80`（默认）/ `S0:T=1`（纯基准）/ `night` /
  `stress`（应力轴）/ `rip`（脉振轴）/ 自定义文本；
- 先按当前画布导出 NPZ，再启动 `tools/scripts/run_auto_schedule.py`；
- 参数：`--x0-npz/--x0-key x0`、`--stages`（或 `--ladder night`）、
  `--max-eval-per-stage`、`--mma-move`、`--n-train/--n-eval`、
  `--filter-radius-mm`、`--max-density`、`--strict`；
- 子进程 stdout/stderr 写 `tools/_out/design_gui/runs/<ts>_<name>/run.console.log`；
  事件日志 `run.log` 的格式由生产端与监视端共用，可同时用只读监视 GUI：
  ```bash
  python -u tools/optim_gui/server.py --read-only --port 8765 \
      --log <run>.log --npz <run 前缀>
  ```
- 同一时刻只允许一个 job；停止按钮先 CTRL_BREAK，5 s 后 `taskkill /T /F` 兜底。
- 进程退出但日志无 `COMPLETE/FAILED/STOP` ⇒ 页面显示“状态未知”，不判成功。

## 测试

```bash
PYTHONPATH=. python -m pytest tests/optimization/test_design_gui.py -q
```

覆盖几何/主从映射、滤波/预览、NPZ 导出/导入换算、`run_auto_schedule` 半径开关
默认行为不变、命令拼装与日志解析。

## 已知限制

- 首次打开需等几何构建（约 8~15 s）；切换网格会重新构建。
- 优化从手绘起点**自由演化**，不保证保留手绘拓扑（未实现区域锁定）。
- 手绘细节受网格分辨率（设计域单元等效边长中位数约 0.78 mm）与滤波半径限制。
- GUI 退出不会终止已启动的优化子进程。
