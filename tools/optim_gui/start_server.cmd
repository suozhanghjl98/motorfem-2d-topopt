@echo off
rem ============================================================
rem  Start the SynRM topology-optimization web GUI server.
rem  Usage:  start_server.cmd [port]     (default port 8765)
rem  Then open http://127.0.0.1:<port> in a browser.
rem  Stop with Ctrl-C in this window.
rem ============================================================
setlocal

set "REPO=%~dp0..\.."
cd /d "%REPO%"

set PORT=8765
if not "%1"=="" set PORT=%1

title SynRM Optim GUI Server (port %PORT%)
echo.
echo  Starting SynRM optimization GUI server...
echo    URL     : http://127.0.0.1:%PORT%
echo    Mesh    : legacy\pytorch\my_gmsh\model_mirror4.msh
echo    Ctrl-C  to stop
echo.

set PYTHONIOENCODING=utf-8
python -u tools\optim_gui\server.py --port %PORT%

echo.
echo  Server exited.
pause
