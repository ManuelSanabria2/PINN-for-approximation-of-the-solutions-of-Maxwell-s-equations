@echo off
setlocal

cd /d "%~dp0pinn-maxwell"

where py >nul 2>nul
if %errorlevel%==0 (
  start "" "http://localhost:8000/"
  py -m uvicorn server:app --host 127.0.0.1 --port 8000
  if errorlevel 1 pause
  goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
  start "" "http://localhost:8000/"
  python -m uvicorn server:app --host 127.0.0.1 --port 8000
  if errorlevel 1 pause
  goto :end
)

echo No se encontro Python en PATH.
echo Instala Python o abre una terminal donde funcionen py o python.
pause

:end
endlocal
