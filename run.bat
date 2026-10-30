@echo off
rem Double-click this file to start FaceTracker.
rem The first time, it sets everything up (this takes a minute or two).

rem Work from the folder this file is in.
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python is not installed. Get it from https://www.python.org/downloads/
    pause
    exit /b
)

if not exist ".venv" (
    echo Setting up FaceTracker for the first time...
    py -m venv .venv
)

echo Checking that everything is installed...
.venv\Scripts\python -m pip install --quiet -r requirements.txt

echo.
echo Starting FaceTracker. Close this window to stop it.
.venv\Scripts\python app.py
pause
