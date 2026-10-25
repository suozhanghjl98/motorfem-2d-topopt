@echo off
REM 手绘转子初始结构 GUI —— 浏览器打开 http://127.0.0.1:8766
setlocal
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
python -u tools\design_gui\server.py --host 127.0.0.1 --port 8766
