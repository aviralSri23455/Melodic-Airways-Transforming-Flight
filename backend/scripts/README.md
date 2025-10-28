# Scripts Documentation

This directory contains all setup and utility scripts for the Aero Melody backend.

## 📁 Available Scripts

### Setup Scripts
- **`setup_database.py`** - Database initialization and schema setup
- **`setup-windows.ps1`** - Windows-specific environment setup
- **`start.sh`** - Linux/Mac startup script

### Data Management
- **`etl_openflights.py`** - Import OpenFlights dataset into database
- **`download_data.py`** - Download and prepare OpenFlights data

### Windows Batch Scripts
- **`windows/setup_redis_duckdb.bat`** - Complete Windows setup with Redis and DuckDB
- **`windows/run_etl_with_fix.bat`** - Database schema fixes and ETL execution

### Verification & Testing
- **`verify_integration.py`** - Comprehensive system verification script

## 🚀 Quick Start

### Database Setup
```bash
# Initialize database and create tables
python scripts/setup_database.py

# Import OpenFlights data (67,000+ routes)
python scripts/etl_openflights.py
```

### System Verification
```bash
# Verify all components work together
python scripts/verify_integration.py
```

### Development Environment
```bash
# Windows setup
powershell -ExecutionPolicy Bypass -File scripts/setup-windows.ps1

# Linux/Mac startup
chmod +x scripts/start.sh
./scripts/start.sh
```

### Windows Batch Scripts
```cmd
# Complete Windows setup (run from backend directory)
scripts\windows\setup_redis_duckdb.bat

# Database fixes and ETL (run from backend directory)
scripts\windows\run_etl_with_fix.bat
```

## 📋 Script Details

### setup_database.py
Creates all database tables and indexes required for the application.

**Usage:**
```bash
python scripts/setup_database.py
```

**Features:**
- Creates MariaDB tables with proper indexes
- Sets up JSON columns for vector storage
- Configures full-text search
- Validates database connection

### etl_openflights.py
Imports the complete OpenFlights dataset into the database.

**Usage:**
```bash
python scripts/etl_openflights.py
```

**Data Source:**
- 7,000+ airports with coordinates
- 67,000+ flight routes
- Distance calculations using Haversine formula

### verify_integration.py
Comprehensive verification script that tests all system components.

**Usage:**
```bash
python scripts/verify_integration.py
```

**Tests:**
- ✅ Database models and relationships
- ✅ JSON vector storage and similarity
- ✅ API routes and endpoints
- ✅ MariaDB configuration
- ✅ Real-time features verification
- ✅ No paid extensions required

**Output:**
- Detailed verification report
- System readiness confirmation
- Setup instructions for frontend team

### setup-windows.ps1
Windows-specific setup script for development environment.

**Features:**
- Environment configuration
- Path setup
- Windows-specific optimizations

### start.sh
Linux/Mac startup script for development environment.

**Features:**
- Environment activation
- Service startup
- Development server launch

### download_data.py
Helper script for downloading and preparing OpenFlights data.

### Windows Batch Scripts

#### setup_redis_duckdb.bat
Complete Windows setup script that handles the entire development environment setup.

**Usage:**
```cmd
scripts\windows\setup_redis_duckdb.bat
```

**Features:**
- Creates Python virtual environment
- Upgrades pip to latest version
- Installs all dependencies from requirements.txt
- Creates necessary directories (data/, midi_output/)
- Tests Redis Cloud connection
- Provides setup completion status

#### run_etl_with_fix.bat
Database maintenance script that fixes schema issues and runs the ETL process.

**Usage:**
```cmd
scripts\windows\run_etl_with_fix.bat
```

**Features:**
- Applies database schema fixes from SQL files
- Runs the OpenFlights ETL import process
- Error handling with detailed messages
- Step-by-step progress reporting

## 🔧 Configuration

All scripts use configuration from:
- Environment variables (`.env` file)
- Database connection settings
- Redis configuration

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database is running
   python scripts/verify_integration.py
   ```

2. **Import Errors**
   ```bash
   # Verify all dependencies installed
   pip install -r requirements.txt
   ```

3. **Permission Issues**
   ```bash
   # Make scripts executable (Linux/Mac)
   chmod +x scripts/*.sh
   ```

4. **Batch Script Path Issues**
   ```cmd
   # Run from project root directory
   scripts\windows\setup_redis_duckdb.bat
   ```

## 📚 Related Documentation

- **Main Documentation**: See [../docs/README.md](../docs/README.md)
- **Database Setup**: See [../docs/database/README.md](../docs/database/README.md)
- **API Reference**: See [../docs/api/README.md](../docs/api/README.md)
- **Testing**: See [../tests/README.md](../tests/README.md)
