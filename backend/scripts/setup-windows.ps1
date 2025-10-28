# Aero Melody Backend Setup Script for Windows
# This script helps you set up the development environment on Windows

param(
    [switch]$SkipDocker,
    [switch]$SkipPython,
    [switch]$SkipData,
    [switch]$Help
)

Write-Host "🎵 Aero Melody Backend Setup for Windows" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if ($Help) {
    Write-Host "Usage: .\setup-windows.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -SkipDocker    Skip Docker installation" -ForegroundColor Gray
    Write-Host "  -SkipPython    Skip Python installation" -ForegroundColor Gray
    Write-Host "  -SkipData      Skip data import" -ForegroundColor Gray
    Write-Host "  -Help          Show this help message" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\setup-windows.ps1              # Full setup" -ForegroundColor Gray
    Write-Host "  .\setup-windows.ps1 -SkipDocker  # Skip Docker parts" -ForegroundColor Gray
    exit
}

# Function to check if command exists
function Test-CommandExists {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to install with winget
function Install-WithWinget {
    param($PackageId, $Name)

    Write-Host "Installing $Name..." -ForegroundColor Cyan
    try {
        winget install --id $PackageId --accept-package-agreements --accept-source-agreements
        Write-Host "✅ $Name installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install $Name. You may need to install it manually." -ForegroundColor Red
    }
}

# Function to install with chocolatey
function Install-WithChocolatey {
    param($PackageName, $Name)

    Write-Host "Installing $Name..." -ForegroundColor Cyan
    try {
        choco install $PackageName -y
        Write-Host "✅ $Name installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install $Name. You may need to install it manually." -ForegroundColor Red
    }
}

Write-Host "🔍 Checking prerequisites..." -ForegroundColor Cyan

# Check if winget is available
if (-not (Test-CommandExists "winget")) {
    Write-Host "📦 Installing Winget (Windows Package Manager)..." -ForegroundColor Cyan
    # Download and install winget from GitHub releases or Microsoft Store
    Write-Host "Please install Winget from Microsoft Store or download from GitHub" -ForegroundColor Yellow
    Write-Host "Alternatively, you can install packages manually" -ForegroundColor Yellow
}

# Check if chocolatey is available
if (-not (Test-CommandExists "choco")) {
    Write-Host "🍫 Installing Chocolatey (Package Manager)..." -ForegroundColor Cyan
    try {
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    }
    catch {
        Write-Host "❌ Failed to install Chocolatey. You can install it manually from https://chocolatey.org/" -ForegroundColor Red
    }
}

# Install Python if not present and not skipped
if ((-not $SkipPython) -and (-not (Test-CommandExists "python"))) {
    Write-Host ""
    Write-Host "🐍 Installing Python..." -ForegroundColor Cyan
    Install-WithWinget "Python.Python.3.11" "Python 3.11"
}

# Verify Python installation
if (Test-CommandExists "python") {
    $pythonVersion = & python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
}
else {
    Write-Host "❌ Python not found. Please install Python 3.8+ manually from https://python.org" -ForegroundColor Red
    exit 1
}

# Install pip if not present
if (-not (Test-CommandExists "pip")) {
    Write-Host "📦 Installing pip..." -ForegroundColor Cyan
    & python -m ensurepip --upgrade
}

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Cyan
& pip install --upgrade pip

# Install Docker if not present and not skipped
if ((-not $SkipDocker) -and (-not (Test-CommandExists "docker"))) {
    Write-Host ""
    Write-Host "🐳 Installing Docker Desktop..." -ForegroundColor Cyan
    Install-WithWinget "Docker.DockerDesktop" "Docker Desktop"
}

# Verify Docker installation
if (Test-CommandExists "docker") {
    Write-Host "✅ Docker found" -ForegroundColor Green
}
else {
    Write-Host "❌ Docker not found. Please install Docker Desktop manually from https://docker.com/desktop" -ForegroundColor Red
}

# Install MariaDB if not using Docker
if ($SkipDocker) {
    Write-Host ""
    Write-Host "🗄️ Installing MariaDB..." -ForegroundColor Cyan
    Install-WithChocolatey "mariadb" "MariaDB"
}

# Install Git if not present
if (-not (Test-CommandExists "git")) {
    Write-Host ""
    Write-Host "📋 Installing Git..." -ForegroundColor Cyan
    Install-WithWinget "Git.Git" "Git"
}

Write-Host ""
Write-Host "🔧 Setting up the project..." -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ requirements.txt not found. Please run this script from the backend directory." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "📦 Creating Python virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    & python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}
else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment and install dependencies
Write-Host "🔄 Activating virtual environment and installing dependencies..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
& pip install -r requirements.txt

Write-Host ""
Write-Host "📊 Setting up the database..." -ForegroundColor Cyan

# Start MariaDB if using Docker
if ((-not $SkipDocker) -and (Test-CommandExists "docker")) {
    Write-Host "🐳 Starting MariaDB with Docker..." -ForegroundColor Cyan
    & docker-compose up -d mariadb

    # Wait for MariaDB to be ready
    Write-Host "⏳ Waiting for MariaDB to be ready..." -ForegroundColor Cyan
    $retryCount = 0
    do {
        Start-Sleep -Seconds 5
        $retryCount++
        try {
            $result = & docker exec aero-melody-mariadb mysqladmin ping -h localhost 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ MariaDB is ready!" -ForegroundColor Green
                break
            }
        }
        catch {
            Write-Host "⏳ Still waiting for MariaDB... ($retryCount/12)" -ForegroundColor Yellow
        }
    } while ($retryCount -lt 12)

    if ($retryCount -ge 12) {
        Write-Host "❌ MariaDB failed to start. Please check Docker and try again." -ForegroundColor Red
        exit 1
    }
}

# Import data if not skipped
if (-not $SkipData) {
    Write-Host "📥 Importing OpenFlights data..." -ForegroundColor Cyan
    try {
        & python scripts/etl_openflights.py
        Write-Host "✅ Data import completed" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Data import failed. You can run it manually later with: python scripts/etl_openflights.py" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🚀 Starting the development server..." -ForegroundColor Cyan

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "📝 Creating .env file from template..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env" -ErrorAction SilentlyContinue
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "✅ .env file created. Please edit it with your settings!" -ForegroundColor Green
    }
    else {
        Write-Host "📝 Creating basic .env file..." -ForegroundColor Cyan
        @"
# Database configuration
DATABASE_URL=mysql://user:password@localhost:3306/aero_melody

# Redis configuration (optional)
REDIS_URL=redis://localhost:6379/0

# JWT configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production

# CORS configuration (add your frontend URLs)
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Other settings
DEBUG=true
LOG_LEVEL=INFO
"@ | Out-File -FilePath ".env" -Encoding UTF8
        Write-Host "✅ Basic .env file created. Please edit it with your settings!" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🎉 Setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit your .env file with the correct database credentials" -ForegroundColor White
Write-Host "2. Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload" -ForegroundColor White
Write-Host "3. Visit: http://localhost:8000/docs to try the API!" -ForegroundColor White
Write-Host ""
Write-Host "For help, run: .\setup-windows.ps1 -Help" -ForegroundColor Gray

# Deactivate virtual environment
deactivate
