@echo off
REM Quick Start - Machine Defect Detection System
REM Run this after setup_venv.bat

echo ========================================
echo Machine Defect Detection - Quick Start
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Checking system...
echo.

REM Test camera
echo [1/3] Testing camera...
python camera.py
if errorlevel 1 (
    echo ERROR: Camera test failed
    pause
    exit /b 1
)

echo.
echo [2/3] Checking if model is downloaded...
if not exist "models\" (
    echo.
    echo ========================================
    echo MODEL NOT FOUND
    echo ========================================
    echo.
    echo The AI model needs to be downloaded first.
    echo This is a ONE-TIME step that requires internet.
    echo.
    echo Download size: ~4-5 GB
    echo Time: 10-30 minutes
    echo.
    set /p download="Download now? (y/n): "
    if /i "%download%"=="y" (
        echo.
        echo Starting download...
        python download_model.py
        if errorlevel 1 (
            echo.
            echo Download failed. Please check your internet connection.
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo Please run: python download_model.py
        echo when you have internet connection.
        pause
        exit /b 0
    )
)

echo.
echo [3/3] Starting application...
echo.
python main.py

pause
