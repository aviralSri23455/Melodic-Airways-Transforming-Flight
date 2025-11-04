#!/bin/bash
# Aero Melody Backend Setup Script
# Complete setup with Redis Cloud and DuckDB
# Works on Linux and macOS

set -e

echo "============================================================"
echo "🎵 Aero Melody - Complete Backend Setup"
echo "============================================================"
echo ""

# Navigate to backend directory
cd ..

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

echo "[OK] Python found: $(python3 --version)"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/7] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/7] Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[2/7] Activating virtual environment..."
source venv/bin/activate
echo ""

# Upgrade pip
echo "[3/7] Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "[4/7] Installing dependencies (Redis, DuckDB, FastAPI, etc.)..."
pip install -r requirements.txt
echo ""

# Create necessary directories
echo "[5/7] Creating directories..."
mkdir -p data midi_output logs uploads
echo "    - data/ (DuckDB database)"
echo "    - midi_output/ (Generated MIDI files)"
echo "    - logs/ (Application logs)"
echo "    - uploads/ (File uploads)"
echo ""

# Create .env if it doesn't exist
echo "[6/7] Setting up configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "    Created .env from template"
    else
        echo "    WARNING: No .env.example found"
    fi
else
    echo "    .env already exists"
fi
echo ""

# Test Redis connection
echo "[7/7] Testing Redis Cloud connection..."
python3 -c "import redis, os; from dotenv import load_dotenv; load_dotenv(); r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379')); r.ping(); print('    [OK] Redis connected')" 2>/dev/null || echo "    [WARNING] Redis connection failed - check REDIS_URL in .env"
echo ""

# Initialize DuckDB
echo "Initializing DuckDB database..."
if [ -f "scripts/init_duckdb.py" ]; then
    python3 scripts/init_duckdb.py && echo "    [OK] DuckDB initialized" || echo "    [WARNING] DuckDB initialization had issues"
else
    echo "    [SKIP] init_duckdb.py not found"
fi
echo ""

echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "Your backend is ready with:"
echo "  - Python virtual environment"
echo "  - Redis Cloud connection"
echo "  - DuckDB database"
echo "  - All dependencies installed"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Redis URL and other settings"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: uvicorn main:app --reload"
echo "  4. Visit: http://localhost:8000/docs"
echo ""
echo "To load flight data:"
echo "  python3 scripts/etl_openflights.py"
echo ""
