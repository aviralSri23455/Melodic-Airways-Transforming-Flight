#!/bin/bash
# Aero Melody Setup Script for Frontend Team
# Run this script to quickly set up the complete system

echo "🚀 Aero Melody Complete Setup Script"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "backend/main.py" ] || [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the aero-melody-main directory"
    exit 1
fi

echo "✅ In correct directory"

# Backend Setup
echo ""
echo "🔧 Setting up backend..."
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
fi

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Test backend components
echo "🧪 Testing backend components..."
python test_backend.py

if [ $? -ne 0 ]; then
    echo "❌ Backend test failed. Please check the errors above."
    exit 1
fi

echo "✅ Backend setup completed successfully"

# Frontend Setup
echo ""
echo "🎨 Setting up frontend..."
cd ..

# Install Node.js dependencies
echo "📥 Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup completed successfully"

# Database Setup
echo ""
echo "🗄️  Database Setup Instructions:"
echo "1. Start MariaDB: cd backend && docker-compose up -d mariadb"
echo "2. Load OpenFlights data: python scripts/etl_openflights.py"
echo "3. Start backend server: python main.py"
echo "4. Start frontend: npm run dev"

echo ""
echo "📚 Documentation:"
echo "- API Guide: backend/FRONTEND_API_GUIDE.md"
echo "- Interactive API Docs: http://localhost:8000/docs"
echo "- Project README: README.md"

echo ""
echo "🎉 Setup completed! Ready for frontend development."
echo ""
echo "Next steps:"
echo "1. Start MariaDB: docker-compose up -d mariadb (in backend directory)"
echo "2. Load data: python scripts/etl_openflights.py (in backend directory)"
echo "3. Start backend: python main.py (in backend directory)"
echo "4. Start frontend: npm run dev (in root directory)"
echo ""
echo "✈️🎵 Happy coding! Flight routes to music await..."
