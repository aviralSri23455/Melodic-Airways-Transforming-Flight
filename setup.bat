@echo off
REM Aero Melody Setup Script for Windows
REM Run this script to quickly set up the complete system

echo 🚀 Aero Melody Complete Setup Script
echo ====================================

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

echo ✅ In correct directory

REM Backend Setup
echo.
echo 🔧 Setting up backend...
cd backend

REM Create virtual environment
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Python dependencies
echo 📥 Installing Python dependencies...
pip install -r requirements.txt

REM Test backend components
echo 🧪 Testing backend components...
python test_backend.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Backend test failed. Please check the errors above.
    pause
    exit /b 1
)

echo ✅ Backend setup completed successfully

REM Frontend Setup
echo.
echo 🎨 Setting up frontend...
cd ..

REM Install Node.js dependencies
echo 📥 Installing Node.js dependencies...
npm install

echo ✅ Frontend setup completed successfully

REM Database Setup Instructions
echo.
echo 🗄️  Database Setup Instructions:
echo 1. Start MariaDB: cd backend ^&^& docker-compose up -d mariadb
echo 2. Load OpenFlights data: python scripts/etl_openflights.py
echo 3. Start backend server: python main.py
echo 4. Start frontend: npm run dev

echo.
echo 📚 Documentation:
echo - API Guide: backend/FRONTEND_API_GUIDE.md
echo - Interactive API Docs: http://localhost:8000/docs
echo - Project README: README.md

echo.
echo 🎉 Setup completed! Ready for frontend development.
echo.
echo Next steps:
echo 1. Start MariaDB: docker-compose up -d mariadb (in backend directory)
echo 2. Load data: python scripts/etl_openflights.py (in backend directory)
echo 3. Start backend: python main.py (in backend directory)
echo 4. Start frontend: npm run dev (in root directory)
echo.
echo ✈️🎵 Happy coding! Flight routes to music await...

pause
