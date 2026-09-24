@echo off
REM Start ANPR Web Server
REM This script starts the Flask server for the ANPR model

cd /d "c:\Projects\ANPR-Project_Code_File\ANPR"

echo.
echo ================================================
echo  🚀 ANPR License Plate Detector - Web Server
echo ================================================
echo.

python app.py

pause
