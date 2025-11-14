@echo off
echo ========================================
echo TechGropse Voice AI Assistant
echo ========================================
echo.

REM Check if Redis is running
echo [1/4] Checking Redis...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo [!] Redis is not running!
    echo Please start Redis first:
    echo    - WSL: wsl sudo service redis-server start
    echo    - Windows: redis-server
    echo.
    pause
    exit /b 1
)
echo [✓] Redis is running

REM Check if .env exists
echo [2/4] Checking configuration...
if not exist .env (
    echo [!] .env file not found!
    echo Please create .env with your OpenAI API key
    echo.
    pause
    exit /b 1
)
echo [✓] Configuration found

REM Start Backend in new window
echo [3/4] Starting Backend Server...
start "Backend - Socket.IO Server" cmd /k "py socketio_server.py"
timeout /t 3 /nobreak >nul

REM Start Frontend in new window
echo [4/4] Starting Frontend...
start "Frontend - Next.js" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo ✓ All services started!
echo ========================================
echo.
echo Backend:  Running on port 8889
echo Frontend: http://localhost:3000
echo.
echo Press any key to open browser...
pause >nul

REM Open browser
start http://localhost:3000

echo.
echo Services are running in separate windows.
echo Close those windows to stop the servers.
echo.
pause

