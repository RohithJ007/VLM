@echo off
echo Starting PC Client for Remote VLM Inspection...
echo.

call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Error: Could not activate virtual environment.
    echo Please run setup_venv.bat first.
    pause
    exit /b
)

set PYTHONPATH=%PYTHONPATH%;%CD%
python pc_client/pc_client/web_app/app.py

pause