@echo off
REM Aero Melody Development Startup Script
REM Starts both backend and frontend for development

echo 🚀 Starting Aero Melody Development Environment
echo ================================================

REM Check if we're in the right directory
if not exist "backend\main.py" (
    echo ❌ Error: Please run this script from the aero-melody-main directory
    pause
    exit /b 1
)
if not exist "package.json" (
    echo ❌ Error: Please run this script from the aero-melody-main directory
    pause
    exit /b 1
)

echo ✅ Starting backend server...
cd backend

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ❌ Virtual environment not found. Please run setup.bat first
    pause
    exit /b 1
)

REM Start backend in background
start "Aero Melody Backend" cmd /k "python main.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

echo ✅ Starting frontend server...
cd ..

REM Start frontend
start "Aero Melody Frontend" cmd /k "npm run dev"

echo.
echo 🎉 Both servers are starting!
echo.
echo 📍 Access points:
echo - Frontend: http://localhost:8080
echo - Backend API: http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo.
echo 💡 The frontend will automatically connect to the backend via proxy
echo 🛑 Close the command windows to stop the servers
echo.
echo ✈️🎵 Happy coding! Your flight-to-music converter is ready...

pause
