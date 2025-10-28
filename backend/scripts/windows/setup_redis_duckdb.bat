@echo off
echo ============================================================
echo Aero Melody - Redis Cloud and DuckDB Setup
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "..\..\venv\" (
    echo Creating virtual environment...
    python -m venv ..\..\venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call ..\..\venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r ..\..\requirements.txt
echo.

REM Create data directory for DuckDB
echo Creating data directory for DuckDB...
if not exist "..\..\data\" mkdir ..\..\data
echo.

REM Create MIDI output directory
echo Creating MIDI output directory...
if not exist "..\..\midi_output\" mkdir ..\..\midi_output
echo.

REM Test Redis Cloud connection
echo ============================================================
echo Testing Redis Cloud Connection...
echo ============================================================
python ..\..\test_redis_cloud.py
echo.

REM Check if test passed
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Setup completed successfully!
    echo ============================================================
    echo.
    echo Next steps:
    echo 1. Review .env file for configuration
    echo 2. Start the backend server: python main.py
    echo 3. Access API docs at: http://localhost:8000/docs
    echo 4. Test analytics endpoints at: http://localhost:8000/api/v1/analytics/
    echo.
) else (
    echo.
    echo ============================================================
    echo Setup completed with warnings
    echo ============================================================
    echo.
    echo Redis Cloud connection test failed.
    echo Please check your REDIS_URL in .env file.
    echo.
    echo You can still run the backend, but caching will be disabled.
    echo.
)

pause
