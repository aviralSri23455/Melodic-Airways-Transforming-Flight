@echo off
echo ========================================
echo Fixing Database Schema and Running ETL
echo ========================================

echo.
echo Step 1: Fixing database schema...
mysql -u root -p aero_melody < ..\..\sql\fix_etl_issues.sql

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to fix database schema
    pause
    exit /b 1
)

echo.
echo Step 2: Running ETL script...
python ..\etl_openflights.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ETL script failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo ETL completed successfully!
echo ========================================
pause
