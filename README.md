# 二维电机有限元与拓扑优化 · 可运行仓库

二维静磁有限元 + 线弹性 + SIMP/MMA 拓扑优化，针对 **6 槽 4 极（6s4p）同步磁阻电机**。
本仓库是从 4311 行单文件程序重构出来的可运行版本，保留与原始程序逐位一致的数值路径，
并把「跑优化」「手绘初始结构」「网页 GUI」三条链路整理成可直接执行的命令。

> 课程大作业 · 模式一「借助 AI 开展科研工作」 · 黄俊力 · 清华大学电机工程与应用电子技术系

---

## 1. 三条链路（本仓库能做什么）

| 链路 | 入口 | 产物 |
|---|---|---|
| **跑一次完整优化** | `tools/scripts/run_auto_schedule.py` | `tools/_out/<out>.log`、`_summary.json`、`_stage_<档>.npz` |
| **手绘转子初始结构** | `tools/design_gui/server.py`（网页） | `tools/_out/design_gui/designs/*.npz`（可作优化起点） |
| **网页监视优化进程** | `tools/optim_gui/server.py --read-only` | 浏览器实时曲线与设计云图 |

30 秒自检用 `tools/scripts/demo_smoke.py`（只建场、评价、2 步 MMA）；环境体检用 `verify_env.py`。

---

## 2. 环境

开发机实测：Python 3.9.13 / numpy 1.26.2 / scipy 1.12.0 / numba 0.60.0 / Windows AMD64。

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
python verify_env.py                   # 期望全部 [OK]
```

`verify_env.py` 依次检查依赖、求解后端、golden 基线、FieldSystem 构造、单次评价。
CHOLMOD 缺失时自动回落 superlu（结果与 golden 逐位一致，慢约 2.7×），这一项允许显示 SKIP。

> 所有命令都在**仓库根目录**执行。Windows 上 `PYTHONPATH` 分隔符是 `;`，
> 本文命令用 `set PYTHONPATH=.` / `export PYTHONPATH=.` 两种写法。

---

## 3. 链路一：跑一次完整的优化

```bash
set PYTHONPATH=.                      # PowerShell: $env:PYTHONPATH="."
set PYTHONIOENCODING=utf-8
python -u tools/scripts/run_auto_schedule.py --motor 6s4p ^
  --x0-npz examples/hand_design.npz --x0-key x0 ^
  --filter-radius-mm 0.9 --max-density 0.85 ^
  --stages "S0:T=40,R=80" --max-eval-per-stage 10 --n-min 10 ^
  --max-budget-shrinks 0 --max-relax-back 0 --wall-seconds 1800 ^
  --out tools/_out/demo/hand_run
```

这条命令做的是「以手绘结构为起点、目标平均转矩 40 N·m / 脉振 ≤80 的单档自动加严优化」。
每步一次全量评价（`n_train=11` 个时间点），单步 20–55 s 视机器负载而定，10 步总耗时约几分钟到十几分钟。

实测输出（2026-10-25，本机 6 个无关进程同时在跑）：

```
  [   1] ρ=1.091 T̄= 43.645 R= 24.81/80 σ=305.99MPa g=[0.0, -6.8989, -0.033, -6.9231] 55.5s
  ...
  [  10] ρ=1.219 T̄= 48.757 R= 29.90/80 σ=272.78MPa g=[0.0, -6.2619, -0.0492, -6.2995] 54.0s
--- S0: advance (evals=10) feasible=True T̄=48.757 R=29.90 σ=272.78MPa ρ̄=0.815
    validate(6pt) {"T_flux": 48.5551, "T_ArkBE": 51.4312, "ripple_arkbe": 19.9206, ...}
COMPLETE tag=run evals=10 580s
```

即：手绘起点 T̄=43.645 N·m，10 步后 48.757 N·m（+11.7%），验证集独立复算 48.555 N·m，
平均密度 0.815 仍在 `--max-density 0.85` 之内，判据判 `advance`。

成功判据（三条同时成立）：

1. 控制台末行 `COMPLETE tag=run evals=<N> <秒>s`；
2. `tools/_out/demo/hand_run_summary.json` 的 `stages` 里出现 `advance` 或 `hit_budget`，`n_evals > 0`；
3. `tools/_out/demo/hand_run_stage_<档>.npz` 里 `x_best` 形状 `(1207,)`，`metrics_json` 可解析。

把 `--stages` 换成 `--ladder night` 即为通宵调度（多档、3.5 h 预算），那是产出论文数字的口径。

---

## 4. 链路二：手绘转子初始结构

```bash
python -u tools/design_gui/server.py --port 8766
# 浏览器打开 http://127.0.0.1:8766
```

- 可编辑主域是 `0..45°` 的 1207 个单元，`45..180°` 的 3621 个镜像单元自动跟随；
  在镜像区落笔也写回主单元。
- 工具：画笔 / 橡皮 / 直线 / 矩形 / 圆 / 圆环扇区 / 多边形套索，撤销 120 步，滚轮缩放。
- 页面实时显示平均密度，超过 `con2` 上限（默认 0.65）会警告；可在「优化」面板调高。
- 「导出 NPZ」落到 `tools/_out/design_gui/designs/`，随后可直接：
  ```bash
  python -u tools/scripts/run_auto_schedule.py --motor 6s4p ^
    --x0-npz tools/_out/design_gui/designs/<文件>.npz --x0-key x0 ^
    --filter-radius-mm 0.9 --max-density 0.85 --stages "S0:T=40,R=80" ^
    --max-eval-per-stage 10 --n-min 10 --max-budget-shrinks 0 --max-relax-back 0 ^
    --wall-seconds 1800 --out tools/_out/demo/hand_run
  ```
- 点「保存并启动优化」等价于上面这条命令，参数在页面里填。

`examples/hand_design.npz` 是一份已画好的示例，字段与导出格式完全一致，
`mesh_sha256` 与本仓库网格相同，可直接当起点用。

不开浏览器也能验证 GUI 三个接口是否正常：

```bash
python -u tools/scripts/verify_gui.py --url http://127.0.0.1:8766
```

---

## 5. 链路三：网页监视优化进程

```bash
python -u tools/optim_gui/server.py --read-only --port 8765 ^
  --log tools/_out/demo/hand_run.log --npz tools/_out/demo/hand_run
# 浏览器打开 http://127.0.0.1:8765
```

只读模式只解析日志与 NPZ，写入动作全部关闭，可与优化进程同时运行。

---

## 6. 30 秒自检与测试

```bash
set PYTHONPATH=.
python -u tools/scripts/demo_smoke.py --start bars        # 建场 → 评价 → 2 步 MMA
python -m pytest tests/kernels tests/config -q            # 核心内核回归
python -m pytest tests -q                                 # 全量（含 GUI / 优化器）
```

`demo_smoke.py` 的判据是：`T_flux` 为有限值、MMA 步后设计变量维度不变。
默认 `--start bars` 是平行磁障几何起点；`--start init` 是实心转子，磁阻转矩本来≈0 N·m。

逐位基线（与原始 4311 行程序 `np.array_equal` 对齐）：

```bash
python tools/scripts/verify_golden.py --layer L1
python tools/scripts/verify_golden.py --layer L3 --inject-x
```

---

## 7. 目录结构

```
仓库/
├─ motorfem/                 核心包：config / mesh / fem / kernels / autograd / winding
│  └─ kernels/               reference（权威数学）/ fast（生产路径，逐位一致）/ approx（显式近似）
├─ tools/
│  ├─ design_gui/            手绘转子初始结构 GUI（stdlib HTTP + 原生 Canvas，零前端依赖）
│  ├─ optim_gui/             只读监视 GUI
│  └─ scripts/               run_auto_schedule.py / demo_smoke.py / verify_gui.py / verify_golden.py …
├─ tests/                    内核、FEM、优化器、GUI 的回归测试
├─ golden/                   L1/L3/L4 逐位基线
├─ legacy/pytorch/           网格 model_6s4p.msh 与 B-H 曲线 50CS470.csv（运行时只读）
├─ examples/                 示例手绘 NPZ
├─ verify_env.py             环境体检
├─ requirements.txt
├─ start_design_gui.cmd      Windows 一键启动手绘 GUI
└─ start_monitor_gui.cmd     Windows 一键启动只读监视
```

`tools/_out/` 是运行产物目录（日志、NPZ、导出设计），已在 `.gitignore` 中排除。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: numba` | `bh_lut_n=65536` 需要 numba；缺省时 `_bundle_env.py` 自动关查表改走 PCHIP |
| `init_opti_elem_random_hole: 超过最大尝试次数` | 正常：6s4p 主设计域小，随机挖孔填不满；不影响结果 |
| 建 FieldSystem 要 1–2 分钟 | 23453 单元的 Python 循环，正常 |
| 手绘细节被抹平 | 滤波半径太大；改 `--filter-radius-mm 0.5` 或导出时选更小的半径 |
| 优化第一步就削材料 | `--max-density` 低于手绘平均密度；调高到 0.85 以上 |
| 中文乱码 | `set PYTHONIOENCODING=utf-8` |
| 想更快 | 装 CHOLMOD（见 `requirements.txt` 末尾），设 `CHOLMOD_DLL_DIR` |

---

## 9. 来源与许可

- **代码**：原始 4311 行程序（`legacy/pytorch/nonlinear_solver_energy_MMA10.py`）由作者 2024 年独立编写；
  `motorfem/` 为 2026 年重构与拓展，AI（Claude Code）辅助实现，逐位一致性由 `golden/` 基线把关。
- **数据**：网格由 gmsh 生成，B-H 曲线 `50CS470.csv` 取自硅钢片手册；
  ANSYS Maxwell 结果只出现在对账记录里，计算过程用本仓库自己的求解器完成。
- **第三方**：SciPy、NumPy、numba、matplotlib、shapely；SuiteSparse CHOLMOD 与 MKL 为可选后端。
- 本仓库的范围是 2D 静磁、线弹性与 SIMP/MMA 三条主线；3D、热耦合、转子动力学在范围之外。
