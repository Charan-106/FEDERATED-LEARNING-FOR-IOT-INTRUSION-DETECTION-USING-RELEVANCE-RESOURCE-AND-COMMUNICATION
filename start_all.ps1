# ARC-FL PowerShell Startup Script
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "ARC-FL: Federated Learning Platform Service Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. PostgreSQL 18 (Port 5433)
Write-Host "[1/4] Verifying PostgreSQL 18 on port 5433..." -ForegroundColor Yellow

$portActive = $false
try {
    $conn = Get-NetTCPConnection -LocalPort 5433 -State Listen -ErrorAction SilentlyContinue
    if ($conn) { $portActive = $true }
} catch {}

if ($portActive) {
    Write-Host "PostgreSQL is already active and listening on port 5433." -ForegroundColor Green
} else {
    # Check if a Windows service is managing PostgreSQL
    $pgService = Get-Service | Where-Object { $_.Name -like "*postgres*" -or $_.DisplayName -like "*PostgreSQL*" } | Where-Object { $_.Status -eq "Running" } | Select-Object -First 1
    if ($pgService) {
        Write-Host "Detected running PostgreSQL service: $($pgService.DisplayName)" -ForegroundColor Cyan
    }

    # Locate PostgreSQL 18 binary
    $pg18Exe = "C:\Program Files\PostgreSQL\18\bin\postgres.exe"
    if (-not (Test-Path $pg18Exe)) {
        $found = Get-ChildItem "C:\Program Files\PostgreSQL" -Filter "postgres.exe" -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match "\\18\\bin" } | Select-Object -First 1
        if ($found) { $pg18Exe = $found.FullName }
    }

    if (-not (Test-Path $pg18Exe)) {
        Write-Host "ERROR: PostgreSQL 18 executable not found at $pg18Exe" -ForegroundColor Red
        exit 1
    }

    $pgData = Join-Path $PSScriptRoot "pgdata18"
    if (-not (Test-Path $pgData)) {
        Write-Host "ERROR: PostgreSQL 18 data directory not found at $pgData" -ForegroundColor Red
        exit 1
    }

    # Clean up stale postmaster.pid if postgres process isn't running
    $pidFile = Join-Path $pgData "postmaster.pid"
    if (Test-Path $pidFile) {
        $pidLines = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($pidLines -and $pidLines.Count -gt 0) {
            $oldPid = [int]$pidLines[0]
            $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if (-not $proc) {
                Write-Host "Removing stale postmaster.pid (PID: $oldPid)..." -ForegroundColor Gray
                Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Write-Host "Starting PostgreSQL 18 cluster on port 5433..." -ForegroundColor Yellow
    $pgCtl = Join-Path (Split-Path $pg18Exe -Parent) "pg_ctl.exe"
    $pgLog = Join-Path $PSScriptRoot "pg.log"
    if (Test-Path $pgCtl) {
        & $pgCtl -D $pgData -l $pgLog start
    } else {
        Start-Process $pg18Exe -ArgumentList "-D `"$pgData`"" -WindowStyle Hidden
    }

    # Poll until port 5433 is listening (up to 10 seconds)
    $waited = 0
    while ($waited -lt 10) {
        Start-Sleep -Seconds 1
        $waited++
        $conn = Get-NetTCPConnection -LocalPort 5433 -State Listen -ErrorAction SilentlyContinue
        if ($conn) {
            Write-Host "PostgreSQL 18 is active on port 5433 (waited ${waited}s)." -ForegroundColor Green
            $portActive = $true
            break
        }
    }

    if (-not $portActive) {
        Write-Host "WARNING: PostgreSQL port 5433 not verified within 10s, continuing startup..." -ForegroundColor Yellow
    }
}

# 2. Python ML Service
Write-Host "[2/4] Starting Python ML Service on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$PSScriptRoot\ml-service`"; python run.py"
Start-Sleep -Seconds 2

# 3. Node.js Backend
Write-Host "[3/4] Starting Node.js Backend on port 5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$PSScriptRoot\backend`"; node app.js"
Start-Sleep -Seconds 2

# 4. React Frontend
Write-Host "[4/4] Starting React Frontend on port 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$PSScriptRoot\frontend`"; npm run dev -- --host 127.0.0.1 --port 5173"
Start-Sleep -Seconds 3

Write-Host "`nAll 4 services are running!" -ForegroundColor Green
Write-Host "👉 Open: http://localhost:5173" -ForegroundColor Cyan
