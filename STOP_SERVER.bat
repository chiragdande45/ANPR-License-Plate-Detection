@echo off
REM Stop ANPR Web Server
REM This script stops the Flask server running on port 5000

echo.
echo ================================================
echo  🛑 Stopping ANPR Web Server
echo ================================================
echo.

REM Check if anything is running on port 5000
netstat -ano | findstr ":5000" > nul

if errorlevel 1 (
    echo [!] Server is not running
    echo [!] Port 5000 is free
) else (
    echo [*] Found process on port 5000
    echo [*] Stopping server...
    
    REM Kill the process on port 5000
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000"') do (
        taskkill /PID %%a /F
    )
    
    echo [✓] Server stopped successfully!
    echo [✓] Port 5000 is now free
)

echo.
pause
