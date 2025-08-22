@echo off
REM SeismoGuard Backend Startup Script for Windows
REM This script starts the backend server with proper environment setup

echo.
echo ================================================================
echo                    SeismoGuard Backend Server
echo ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app\api.py" (
    echo Error: Please run this script from the backend directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import flask, obspy, numpy, pandas, scipy, sklearn, tensorflow" >nul 2>&1
if errorlevel 1 (
    echo Installing/updating dependencies...
    pip install -r requirements.txt
)

REM Create necessary directories
if not exist "models" mkdir models
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
if not exist "cache" mkdir cache
if not exist "data" mkdir data

REM Run quick health check
echo Running health check...
python -c "from app.api import app; print('Backend modules loaded successfully')" >nul 2>&1
if errorlevel 1 (
    echo Error: Backend modules failed to load
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ================================================================
echo                        Starting Server
echo ================================================================
echo.
echo Server will be available at: http://127.0.0.1:5000
echo Frontend integration ready
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python run_server.py

echo.
echo Server stopped.
pause
