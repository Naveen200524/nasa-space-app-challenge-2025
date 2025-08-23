@echo off
REM SeismoGuard: Start Backend and Frontend (Windows)
setlocal enabledelayedexpansion

set FRONTEND_PORT=8080
set BACKEND_HOST=127.0.0.1
set BACKEND_PORT=5000

powershell -ExecutionPolicy Bypass -File "%~dp0start_all.ps1" -FrontendPort %FRONTEND_PORT% -BackendHost %BACKEND_HOST% -BackendPort %BACKEND_PORT%

endlocal
