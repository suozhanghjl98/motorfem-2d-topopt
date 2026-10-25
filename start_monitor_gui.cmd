@echo off
REM 只读监视 GUI —— 浏览器打开 http://127.0.0.1:8765
REM 用法：先跑优化，再把 --log / --npz 指到那次运行的产物
setlocal
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
python -u tools\optim_gui\server.py --read-only --port 8765 ^
  --log tools\_out\demo\hand_run.log --npz tools\_out\demo\hand_run
