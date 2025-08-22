#!/bin/bash
# SeismoGuard Backend Startup Script for Unix/Linux/macOS
# This script starts the backend server with proper environment setup

set -e  # Exit on any error

echo ""
echo "================================================================"
echo "                    SeismoGuard Backend Server"
echo "================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app/api.py" ]; then
    echo "Error: Please run this script from the backend directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python -c "import flask, obspy, numpy, pandas, scipy, sklearn, tensorflow" &> /dev/null; then
    echo "Installing/updating dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p models uploads outputs cache data

# Make scripts executable
chmod +x run_server.py
chmod +x test_backend.py

# Run quick health check
echo "Running health check..."
if ! python -c "from app.api import app; print('Backend modules loaded successfully')" &> /dev/null; then
    echo "Error: Backend modules failed to load"
    echo "Please check the error messages above"
    exit 1
fi

echo ""
echo "================================================================"
echo "                        Starting Server"
echo "================================================================"
echo ""
echo "Server will be available at: http://127.0.0.1:5000"
echo "Frontend integration ready"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python run_server.py

echo ""
echo "Server stopped."
