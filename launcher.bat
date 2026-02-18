@echo off
title MDMA - Music Design Made Accessible
echo ============================================
echo   MDMA - Music Design Made Accessible
echo   Vision-optional audio workstation
echo ============================================
echo.

:: ── Locate Python ────────────────────────────────────────────────
:: Try py launcher first (standard on modern Windows installs),
:: then fall back to python on PATH.

where py >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=py -3
    goto :found_python
)

where python >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=python
    goto :found_python
)

echo [ERROR] Python 3 was not found on this system.
echo.
echo Install Python 3.9 or later from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found_python
echo [OK] Found Python: %PYTHON%

:: ── Install dependencies if needed ───────────────────────────────
echo.
echo Checking dependencies...
%PYTHON% -c "import numpy, scipy, soundfile, sounddevice" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing core dependencies...
    %PYTHON% -m pip install -r "%~dp0requirements.txt" --quiet
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        echo Try running manually:  pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed.
) else (
    echo [OK] All core dependencies present.
)

:: ── Install wxPython for GUI if needed ───────────────────────────
%PYTHON% -c "import wx" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Installing wxPython for the GUI...
    %PYTHON% -m pip install wxPython --quiet
    if %errorlevel% neq 0 (
        echo [WARNING] Could not install wxPython. GUI may not be available.
        echo The launcher will fall back to the best available interface.
    ) else (
        echo [OK] wxPython installed.
    )
)

:: ── Launch MDMA GUI ──────────────────────────────────────────────
echo.
echo Launching MDMA...
echo.
%PYTHON% "%~dp0run_mdma.py" --gui
if %errorlevel% neq 0 (
    echo.
    echo MDMA exited with an error. Check the output above.
    pause
    exit /b %errorlevel%
)
