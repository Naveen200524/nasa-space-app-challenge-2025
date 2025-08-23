# SeismoGuard: Start Backend and Frontend (Windows PowerShell)
param(
    [int]$FrontendPort = 8080,
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 5000
)

$ErrorActionPreference = "Stop"

Write-Host "=== SeismoGuard: Starting stack ===" -ForegroundColor Cyan

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# 1) Backend: ensure venv and deps
$backendDir = Join-Path $repoRoot 'backend'
$venvPy = Join-Path $backendDir 'venv\Scripts\python.exe'
if (!(Test-Path $venvPy)) {
    Write-Host "Creating backend virtual environment..." -ForegroundColor Yellow
    Push-Location $backendDir
    python -m venv venv
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r requirements.txt
    Pop-Location
} else {
    Write-Host "Backend venv found: $venvPy" -ForegroundColor Green
}

# 2) Start backend server in a background PowerShell
Write-Host "Starting backend on http://$($BackendHost):$($BackendPort) ..." -ForegroundColor Cyan
$backendCmd = "`"$venvPy`" `"$backendDir\run_server.py`" --host 0.0.0.0 --port $BackendPort"
$backendPsArgs = "-NoExit -Command `"cd `'$backendDir`'; $backendCmd`""
Start-Process -FilePath "powershell.exe" -ArgumentList $backendPsArgs -WindowStyle Minimized

# 3) Start static frontend server from repo root
Write-Host "Starting static frontend on http://127.0.0.1:$FrontendPort ..." -ForegroundColor Cyan
$frontendPsArgs = "-NoExit -Command `"cd `'$repoRoot`'; python -m http.server $FrontendPort`""
Start-Process -FilePath "powershell.exe" -ArgumentList $frontendPsArgs -WindowStyle Minimized

# 4) Probe backend health
Start-Sleep -Seconds 2
try {
    $health = Invoke-RestMethod -Uri "http://$($BackendHost):$($BackendPort)/health" -TimeoutSec 10
    Write-Host "Backend health: $($health.status)" -ForegroundColor Green
} catch {
    Write-Warning "Backend health check failed. It may still be starting."
}

# 5) Open browser to frontend
Start-Process "http://127.0.0.1:$FrontendPort"

Write-Host "All set. Use Ctrl+C in the opened terminals to stop servers." -ForegroundColor Green
