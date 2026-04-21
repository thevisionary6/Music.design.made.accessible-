@echo off
REM MDMA environment installer (Windows).
REM
REM Delegates to scripts\setup_env.py so the real logic stays in
REM one place. Any extra args are forwarded — e.g.:
REM   scripts\setup_env.bat --profile full --with-dev

setlocal

set "PYTHON_BIN=%PYTHON%"
if "%PYTHON_BIN%"=="" set "PYTHON_BIN=python"

where %PYTHON_BIN% >nul 2>&1
if errorlevel 1 (
    echo error: %PYTHON_BIN% not on PATH. Set PYTHON=C:\Path\To\python.exe or install Python 3.9+.
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
"%PYTHON_BIN%" "%SCRIPT_DIR%setup_env.py" %*
exit /b %errorlevel%
