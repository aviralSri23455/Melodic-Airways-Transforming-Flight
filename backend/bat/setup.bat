@echo off
REM Aero Melody Backend Setup Script for Windows
REM Complete setup with Redis Cloud and DuckDB

echo ============================================================
echo Aero Melody - Complete Backend Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Navigate to backend directory
cd ..

REM Create virtual environment
if not exist "venv\" (
    echo [1/7] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/7] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [2/7] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [3/7] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [4/7] Installing dependencies (Redis, DuckDB, FastAPI, etc.)...
pip install -r requirements.txt
echo.

REM Create directories for DuckDB and outputs
echo [5/7] Creating directories...
if not exist "data\" mkdir data
if not exist "midi_output\" mkdir midi_output
if not exist "logs\" mkdir logs
if not exist "uploads\" mkdir uploads
echo     - data/ (DuckDB database)
echo     - midi_output/ (Generated MIDI files)
echo     - logs/ (Application logs)
echo     - uploads/ (File uploads)
echo.

REM Create .env file
echo [6/7] Setting up configuration...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo     Created .env from template
    ) else (
        echo     WARNING: No .env.example found
    )
) else (
    echo     .env already exists
)
echo.

REM Test Redis connection
echo [7/7] Testing Redis Cloud connection...
python -c "import redis, os; from dotenv import load_dotenv; load_dotenv(); r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379')); r.ping(); print('     [OK] Redis connected')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo     [WARNING] Redis connection failed - check REDIS_URL in .env
    echo     Backend will work but caching will be disabled
) else (
    echo     [OK] Redis Cloud connected successfully
)
echo.

REM Initialize DuckDB
echo Initializing DuckDB database...
if exist "scripts\init_duckdb.py" (
    python scripts\init_duckdb.py
    if %ERRORLEVEL% EQU 0 (
        echo     [OK] DuckDB initialized
    ) else (
        echo     [WARNING] DuckDB initialization had issues
    )
) else (
    echo     [SKIP] init_duckdb.py not found
)
echo.

echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Your backend is ready with:
echo   - Python virtual environment
echo   - Redis Cloud connection
echo   - DuckDB database
echo   - All dependencies installed
echo.
echo Next steps:
echo   1. Edit .env with your Redis URL and other settings
echo   2. Run: venv\Scripts\activate
echo   3. Run: uvicorn main:app --reload
echo   4. Visit: http://localhost:8000/docs
echo.
echo To load flight data:
echo   python scripts\etl_openflights.py
echo.
pause
