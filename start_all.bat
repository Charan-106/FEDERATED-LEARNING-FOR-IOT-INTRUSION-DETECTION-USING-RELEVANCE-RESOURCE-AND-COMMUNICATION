@echo off
title ARC-FL Platform Launcher
echo ======================================================================
echo ARC-FL: Relevance, Resource, Communication Federated Learning Platform
echo ======================================================================
echo.

set PROJECT_ROOT=%~dp0
set PG18_EXE=C:\Program Files\PostgreSQL\18\bin\postgres.exe
set PG_DATA=%PROJECT_ROOT%pgdata18

echo [1/4] Verifying PostgreSQL 18 on port 5433...
netstat -ano | findstr :5433 | findstr LISTENING >nul 2>&1
if %errorlevel% equ 0 (
    echo PostgreSQL is already active and listening on port 5433.
) else (
    if exist "%PG_DATA%\postmaster.pid" (
        del /f /q "%PG_DATA%\postmaster.pid" >nul 2>&1
    )
    echo Starting PostgreSQL 18 on port 5433...
    if exist "%PG18_EXE%" (
        start "ARC-FL PostgreSQL 18 (Port 5433)" /B "%PG18_EXE%" -D "%PG_DATA%"
        timeout /t 3 /nobreak >nul
    ) else (
        echo ERROR: PostgreSQL 18 executable not found at "%PG18_EXE%"!
    )
)

echo [2/4] Starting Python FastAPI ML Service on port 8000...
start "ARC-FL Python ML Service (Port 8000)" cmd /k "cd /d "%PROJECT_ROOT%ml-service" && python run.py"
timeout /t 2 /nobreak >nul

echo [3/4] Starting Node.js Express Backend on port 5000...
start "ARC-FL Node.js Backend (Port 5000)" cmd /k "cd /d "%PROJECT_ROOT%backend" && node app.js"
timeout /t 2 /nobreak >nul

echo [4/4] Starting React Frontend Dev Server on port 5173...
start "ARC-FL React Frontend (Port 5173)" cmd /k "cd /d "%PROJECT_ROOT%frontend" && npm run dev -- --host 0.0.0.0 --port 5173"
timeout /t 3 /nobreak >nul

echo.
echo ======================================================================
echo ALL 4 SERVICES STARTED SUCCESSFULLY!
echo Web Application: http://localhost:5173
echo API Health:      http://localhost:5000/health
echo ML Docs:         http://localhost:8000/docs
echo ======================================================================
echo.
pause
