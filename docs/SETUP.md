# Aero Melody Backend - Setup Guide

Complete setup instructions for the Aero Melody backend.

---

## Prerequisites

- Python 3.9+
- MariaDB 10.5+
- Redis (Cloud or Local)
- Git

---

## Installation Steps

### 1. Install Dependencies

```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Login to MariaDB
mysql -u root -p

# Create database
CREATE DATABASE aero_melody CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'aero_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON aero_melody.* TO 'aero_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 3. Run Migrations

Apply schema migrations in order:

```bash
# Step 1: Base tables
mysql -u root -p aero_melody < sql/create_tables.sql

# Step 2: Enhanced schema
mysql -u root -p aero_melody < sql/enhanced_schema_migration.sql

# Step 3: Community features
mysql -u root -p aero_melody < sql/community_features_tables.sql
```

### 4. Configure Environment

Create `.env` file in `backend/` directory:

```bash
# Database
DATABASE_URL=mysql://aero_user:your_password@localhost:3306/aero_melody

# Redis Cloud
REDIS_HOST=your-redis-host.redis-cloud.com
REDIS_PORT=14696
REDIS_PASSWORD=your-redis-password
REDIS_USERNAME=default
REDIS_CACHE_TTL=3600

# Security
JWT_SECRET_KEY=your-secret-key-here
BACKEND_CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Optional
DUCKDB_PATH=./data/analytics.duckdb
DUCKDB_MEMORY_LIMIT=2GB
```

### 5. Import OpenFlights Data

```bash
python scripts/etl_openflights.py
```

This imports:
- 3,000+ airports
- 67,000+ flight routes
- Calculates distances and durations

**Time**: 2-5 minutes

### 6. Setup Vector Embeddings (Optional)

```bash
# Windows
setup_vector_embeddings.bat

# Linux/Mac
python scripts/generate_route_embeddings.py
```

This generates:
- 128D embeddings for all routes
- FAISS index for fast similarity search
- Complexity metrics

**Time**: 5-10 minutes

---

## Start the Server

```bash
# Method 1: Using Python
python main.py

# Method 2: Using Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Server runs at**: http://localhost:8000  
**API Docs**: http://localhost:8000/docs

---

## Verify Installation

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Test Database
```bash
python tests/test_db.py
python tests/test_skysql_connection.py
```

### 3. Test Redis
```bash
python tests/test_redis_cloud.py
```

Expected output:
```
✅ PING successful!
✅ SET successful!
✅ GET successful!
🎉 All Redis Cloud tests passed!
```

### 4. Test Vector Embeddings
```bash
python test_vector_embeddings.py
```

### 5. Generate Test Music
```bash
curl "http://localhost:8000/api/v1/demo/complete-demo?origin=JFK&destination=LAX"
```

### 6. Run Test Suite
```bash
pytest tests/test_new_features.py -v
```

---

## Verify Database Tables

```bash
mysql -u root -p aero_melody -e "SHOW TABLES;"
```

Expected: 19 tables including:
- airports
- routes
- music_compositions
- users
- user_datasets
- collaboration_sessions
- 
- contests

---

## Troubleshooting

### Module Not Found
```bash
pip install -r requirements.txt
```

### Database Connection Failed
1. Check `.env` file has correct DATABASE_URL
2. Verify MariaDB is running: `sudo systemctl status mariadb`
3. Test connection: `mysql -u aero_user -p aero_melody`

### Redis Connection Failed
1. Test connection: `python tests/test_redis_cloud.py`
2. Verify REDIS_HOST, REDIS_PORT, REDIS_PASSWORD in `.env`
3. Check firewall allows connections to Redis Cloud

### DuckDB Issues
```bash
mkdir data
```
DuckDB will auto-create on first use (optional component).

### CORS Errors
Update `BACKEND_CORS_ORIGINS` in `.env`:
```bash
BACKEND_CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://localhost:8080
```

---

## What Gets Installed

### Core Services
- `music_generator.py` - MIDI generation with PyTorch
- `vector_service.py` - Similarity search
- `dataset_manager.py` - User dataset management
- `websocket_manager.py` - Real-time collaboration
- `activity_service.py` - Activity tracking
- `cache.py` - Redis caching

### Extended Services
- `graph_pathfinder.py` - NetworkX + Dijkstra
- `genre_composer.py` - AI genre composition
- `community_service.py` - Forums, contests
- `duckdb_analytics.py` - Analytics engine

---

## Next Steps

1. **Read API Documentation**: Visit http://localhost:8000/docs
2. **Test Endpoints**: Use Swagger UI for interactive testing
3. **Frontend Integration**: See [API_GUIDE.md](./API_GUIDE.md)
4. **Explore Features**: Try music generation, vector search, education endpoints

---

## Production Deployment

### Docker Setup
```bash
docker-compose up -d
```

### Environment Checklist
- [ ] Update DATABASE_URL with production credentials
- [ ] Set strong JWT_SECRET_KEY
- [ ] Configure SSL/TLS
- [ ] Set up monitoring and logging
- [ ] Configure backups
- [ ] Update BACKEND_CORS_ORIGINS
- [ ] Test all endpoints

---

## Support

For issues:
1. Check this setup guide
2. Review [API_GUIDE.md](./API_GUIDE.md)
3. Test with http://localhost:8000/docs
4. Check logs in terminal
