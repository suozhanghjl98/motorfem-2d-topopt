@echo off
rem ============================================================
rem  Design GUI for drawing rotor initial structures.
rem  Usage:  start_server.cmd [port]     (default port 8766)
rem  Then open http://127.0.0.1:<port> in a browser.
rem  Stop with Ctrl-C (does NOT kill an already started optimizer).
rem ============================================================
setlocal

pushd "%~dp0..\.."

set "PORT=8766"
if not "%~1"=="" set "PORT=%~1"

title Design GUI (port %PORT%)
set "PYTHONIOENCODING=utf-8"
echo.
echo  Starting design GUI on http://127.0.0.1:%PORT%
echo  Geometry builds in background (~8-15 s).
echo  Ctrl-C to stop.
echo.

python -u tools\design_gui\server.py --port %PORT%

popd
echo.
echo  Server exited.
pause
