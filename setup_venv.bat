@echo off
REM Setup script for machine defect detection system
REM Run this script to create virtual environment and install dependencies

echo ========================================
echo Machine Defect Detection - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo [4/4] Installing dependencies (this may take several minutes)...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some dependencies
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Keep your internet connected and run: python download_model.py
echo 2. This will download the Qwen2-VL-2B model (one-time, ~4GB)
echo 3. After download, you can work completely offline
echo 4. Run the system: python main.py
echo.
pause
