# Aero Melody Backend - Complete Documentation

**Transform Flight Routes into Musical Compositions using AI**

A comprehensive FastAPI backend that converts aviation data into unique MIDI music files using PyTorch, NetworkX, and Mido.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MariaDB 10.6+
- Redis (optional, for caching)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Import OpenFlights data
python scripts/etl_openflights.py

# Start server
python main.py
```

### Access API
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📖 Table of Contents

1. [Technology Stack](#technology-stack)
2. [Architecture](#architecture)
3. [Database Setup](#database-setup)
4. [API Endpoints](#api-endpoints)
5. [Music Generation](#music-generation)
6. [Real-Time Features](#real-time-features)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## 🛠 Technology Stack

### Core Framework
- **FastAPI** - Async web framework with automatic OpenAPI docs
- **Python 3.8+** - Primary language with async/await support
- **SQLAlchemy** - Async ORM for database operations

### AI & Music Generation
- **PyTorch** - Neural networks for route embeddings
- **NetworkX** - Graph algorithms (Dijkstra pathfinding)
- **Mido** - MIDI file generation

### Databases
- **MariaDB** - Primary database (FREE version)
- **Redis** - Caching and real-time pub/sub
- **DuckDB** - Analytics database

### Real-Time
- **WebSocket** - Live collaboration
- **Redis Pub/Sub** - Real-time messaging
- **AsyncIO** - Non-blocking I/O

---

## 🏗 Architecture

### Data Flow
```
OpenFlights Dataset (GitHub)
    ↓
MariaDB (3K+ airports, 67K+ routes)
    ↓
FastAPI Backend
├─ NetworkX → Dijkstra pathfinding
├─ PyTorch → Route embeddings
├─ Mido → MIDI generation
└─ Redis → Caching
    ↓
MIDI Files + Analytics
```

### Project Structure
```
backend/
├── app/
│   ├── api/          # API routes
│   ├── core/         # Configuration
│   ├── db/           # Database
│   ├── models/       # Data models
│   └── services/     # Business logic
├── scripts/          # Setup scripts
├── tests/            # Test suites
├── docs/             # Documentation
└── main.py           # Entry point
```

---

## 🗄 Database Setup

### MariaDB Configuration

**Tables:**
- `airports` - 3,000+ airports with coordinates
- `routes` - 67,000+ flight routes
- `music_compositions` - Generated music metadata
- `users` - Authentication
- `user_datasets` - Personal collections
- `collaboration_sessions` - Real-time sessions

### Setup Commands
```bash
# Using Docker
docker-compose up -d mariadb

# Or local installation
mysql -u root -p
CREATE DATABASE aero_melody;
CREATE USER 'aero_user'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON aero_melody.* TO 'aero_user'@'localhost';
```

### Import OpenFlights Data
```bash
python scripts/etl_openflights.py
```

**Data Source**: https://github.com/MariaDB/openflights

---

## 🌐 API Endpoints

### Authentication
```http
POST /api/v1/auth/register  # Create user
POST /api/v1/auth/login     # Get JWT token
GET  /api/v1/auth/me        # Current user info
```

### Music Generation
```http
POST /api/v1/generate-midi  # Generate music from route
GET  /api/v1/download/{id}  # Download MIDI file
GET  /api/v1/analytics/{id} # Get composition analytics
```

### Route Information
```http
GET /api/v1/routes          # List all routes
GET /api/v1/similar         # Find similar routes
GET /api/v1/recent          # Recent compositions
```

### Airport Search
```http
GET /api/v1/airports/search?query=New York
GET /api/v1/airports/JFK
```

### Example Request
```bash
curl -X POST "http://localhost:8000/api/v1/generate-midi" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "origin_code": "JFK",
    "destination_code": "LAX",
    "music_style": "classical",
    "tempo": 120,
    "duration_minutes": 3
  }'
```

---

## 🎵 Music Generation

### Workflow
```
1. Fetch route from database
2. Build NetworkX graph (all routes)
3. Find optimal path (Dijkstra)
4. Generate PyTorch embeddings
5. Map route features → music parameters
6. Generate MIDI file (Mido)
7. Save to database
8. Return MIDI + analytics
```

### Music Parameters
- **Tempo**: Based on distance (60-200 BPM)
- **Pitch**: Based on direction (eastward = ascending)
- **Harmony**: Based on route complexity
- **Scale**: major, minor, pentatonic, etc.
- **Key**: C, D, E, F, G, A, B

### Supported Styles
- Classical
- Jazz
- Electronic
- Ambient
- Rock

---

## ⚡ Real-Time Features

### WebSocket Collaboration
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/collaborate/1/1');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'state_update',
    state: { tempo: 120, notes: [...] }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

### Redis Caching
- Route-to-music mappings
- Vector embeddings
- User sessions
- API responses

### DuckDB Analytics
- Route complexity statistics
- Genre distribution
- Performance metrics
- Similarity search

---

## 🧪 Testing

### Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test
pytest tests/test_routes.py
```

### Test Files
- `test_db.py` - Database tests
- `test_skysql_connection.py` - Connection tests
- `test_new_features.py` - Feature tests

### Interactive Testing
Visit http://localhost:8000/docs for built-in API testing

---

## 🚀 Deployment

### Docker Setup
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables
```env
DATABASE_URL=mysql://user:password@localhost/aero_melody
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=your-secret-key
BACKEND_CORS_ORIGINS=http://localhost:3000
```

### Production Checklist
- [ ] Update environment variables
- [ ] Configure SSL/TLS
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Enable logging
- [ ] Test all endpoints

---

## 🔧 Troubleshooting

### Database Connection Failed
```bash
# Check MariaDB status
sudo systemctl status mariadb

# Verify credentials in .env
cat .env | grep DATABASE_URL

# Test connection
python -c "from app.db.database import engine; print('Connected!')"
```

### Import Errors
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### CORS Issues
```bash
# Update BACKEND_CORS_ORIGINS in .env
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### MIDI Generation Fails
```bash
# Check file permissions
chmod 755 midi_output/

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

---

## 📊 Database Schema

### Core Tables

**airports**
- id, name, city, country
- iata_code, icao_code
- latitude, longitude, altitude

**routes**
- origin_airport_id, destination_airport_id
- distance_km, duration_min

**music_compositions**
- route_id, user_id
- tempo, pitch, harmony
- midi_path, complexity_score

**users**
- username, email, hashed_password
- role, is_active

---

## 🎯 Features

### ✅ Implemented
- OpenFlights data integration
- Dijkstra pathfinding (NetworkX)
- PyTorch embeddings
- MIDI generation (Mido)
- User authentication (JWT)
- Real-time collaboration (WebSocket)
- Vector similarity search
- Dataset management
- Collection organization
- Activity feeds
- Redis caching
- DuckDB analytics

### 🔄 In Progress
- Advanced vector operations
- Enhanced collaboration features
- Community features
- Performance optimization

---

## 📚 Additional Documentation

### API Documentation
- **Complete API Reference**: `docs/api/API_DOCUMENTATION.md`
- **Testing Guide**: `docs/api/API_TESTING_GUIDE.md`

### Database Documentation
- **Setup Guide**: `docs/database/DATABASE_SETUP.md`
- **Data Import**: `docs/database/DATA_IMPORT_GUIDE.md`

### Implementation Details
- **Technical Guide**: `docs/implementation/IMPLEMENTATION_GUIDE.md`
- **Feature Status**: `docs/implementation/IMPLEMENTATION_STATUS.md`
- **Quick Start**: `docs/project/QUICK_START.md`

### Scripts
- **Setup Scripts**: `scripts/README.md`
- **Testing**: `tests/README.md`

---

## 🤝 Support

### Getting Help
1. Check API documentation at `/docs`
2. Review database schema in `sql/` directory
3. Check service implementations in `app/services/`
4. Review test suite in `tests/`

### Common Issues
- Database connection problems
- Import errors
- CORS configuration
- MIDI generation failures

---

## 📝 Version

**Backend Version**: 1.0.0  
**Last Updated**: October 27, 2025  
**Status**: Production Ready

---

## 🔗 External Resources

- **OpenFlights Data**: https://github.com/MariaDB/openflights
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **NetworkX Docs**: https://networkx.org/
- **PyTorch Docs**: https://pytorch.org/
- **Mido Docs**: https://mido.readthedocs.io/

---

**Made with ❤️ for music and flight lovers!**


---

## 🔧 Fixes & Configuration Guides

### Database Configuration

#### DuckDB + MariaDB Setup

The system uses both databases with graceful fallback:

**Normal Operation:**
- MariaDB (Primary) → Stores flight/music data
- DuckDB (Analytics) → Logs analytics for insights
- Redis (Cache) → Caches music generations

**Fallback Mode (DuckDB Unavailable):**
- MariaDB (Primary) → Stores flight/music data
- DuckDB (Analytics) → Fallback service (no-op)
- Redis (Cache) → Caches music generations

#### DuckDB Configuration

Environment variables in `.env`:
```bash
DUCKDB_PATH=./data/analytics.duckdb
DUCKDB_MEMORY_LIMIT=2GB
DUCKDB_THREADS=4
```

#### Troubleshooting DuckDB

If logs show "Using fallback DuckDB analytics service":

```bash
# Create data directory
mkdir data

# Check if duckdb is installed
pip list | grep duckdb

# Install if missing
pip install duckdb

# Fix permissions (Windows)
icacls data /grant Users:F

# Restart backend
python main.py
```

---

### Redis Caching Configuration

#### Redis Cloud Credentials

Configure in `.env`:
```bash
REDIS_HOST=redis-14696.crce219.us-east-1-4.ec2.redns.redis-cloud.com
REDIS_PORT=14696
REDIS_PASSWORD=your_password
REDIS_USERNAME=default
REDIS_CACHE_TTL=3600
```

#### Cache Key Structure

All keys use `aero:` prefix:
- `aero:music:{ORIGIN}:{DESTINATION}` - Music by route
- `aero:composition:{COMPOSITION_ID}` - Music by ID
- `aero:latest:music` - Most recent generation

#### Testing Redis

```bash
# Test connection
python test_redis_cloud.py

# Test caching endpoint
curl "http://localhost:8000/api/v1/redis/test/save-music?origin=DEL&destination=LHR"

# Check cache stats
curl "http://localhost:8000/api/v1/redis/cache/stats"

# Check storage info
curl "http://localhost:8000/api/v1/redis/storage/info"
```

#### Viewing Data in Redis Insight

1. Open Redis Insight: https://ri.redis.io/13680228/browser
2. Look for keys starting with `aero:`
3. Click any key to view cached JSON data

#### Cached Data Structure

```json
{
  "composition_id": 1730000000,
  "origin": "DEL",
  "destination": "LHR",
  "tempo": 120,
  "duration_seconds": 30,
  "note_count": 60,
  "notes": [
    {"note": 60, "velocity": 80, "time": 0, "duration": 480}
  ],
  "key": "C",
  "scale": "ambient",
  "midi_file": "route_DEL_LHR_1730000000.mid",
  "generated_at": "2025-10-27T12:00:00.000000"
}
```

---

### MIDI Generation Fixes

#### Duration Calculation

The backend correctly calculates duration based on route distance:
```python
duration_seconds = distance_km / 500  # ~500 km per second of music
```

#### Note Timing

Notes are spread across the calculated duration (not all at time=0):
```python
for i in range(int(duration_seconds * 2)):  # 2 notes per second
    note_time = int(i * ticks_per_beat / 2)
    track.append(Message('note_on', note=note, velocity=velocity, time=note_time))
```

#### Tempo Settings
- **Tempo**: 120 BPM
- **Ticks per beat**: 480
- **Notes per second**: 2
- **Note spacing**: 480 ticks (0.5 seconds apart)

#### Response Structure

Duration is available at top-level for easy access:
```json
{
  "composition": {
    "composition_id": 1698423421,
    "duration_seconds": 11.08,
    "note_count": 57,
    "tempo": 120,
    "key": "C",
    "scale": "ambient"
  }
}
```

---

### WebSocket & Real-time Updates

#### Testing WebSocket Connection

```javascript
// Browser console test
const ws = new WebSocket('ws://localhost:8000/api/v1/demo/redis-subscriber');
ws.onopen = () => console.log('WebSocket opened');
ws.onmessage = (e) => console.log('Message:', JSON.parse(e.data));
ws.onerror = (e) => console.error('WebSocket error:', e);
ws.onclose = (e) => console.log('WebSocket closed:', e.code, e.reason);
```

#### Common WebSocket Issues

**Issue**: Connection closes prematurely  
**Fix**: Backend implements keep-alive and proper error handling

**Issue**: Redis pub/sub not working  
**Fix**: Use async Redis client with proper subscription handling

#### Testing Real-time Functionality

```bash
# Method 1: Browser test
# Open realtime_test.html in browser

# Method 2: Python test
python test_realtime_functionality.py
```

---

### CORS Configuration

Frontend origins allowed in `config.py`:
```python
BACKEND_CORS_ORIGINS = "http://localhost:8080,http://localhost:3000,https://yourdomain.com"
```

After changing CORS settings, restart the backend.

---

### Storage Limits

#### Redis Cloud (Free Tier)
- **Limit**: 30MB
- **TTL**: 1 hour per key
- **Cleanup**: Automatic via TTL, manual via `/redis/storage/cleanup`

#### DuckDB
- **Location**: `./data/analytics.duckdb`
- **Memory**: 2GB (configurable)
- **Purpose**: Analytics only (optional)

---

### Performance Tips

1. **Use Redis caching** - Reduces database load for repeated routes
2. **Monitor storage** - Check `/redis/storage/info` regularly
3. **Clean up old data** - Use `/redis/storage/cleanup` when needed
4. **DuckDB is optional** - System works fine without it
5. **Use connection pooling** - Already configured for MariaDB

---

### What Gets Logged

#### DuckDB Analytics (when available)
- Route complexity and distance
- Music generation parameters
- Performance metrics
- Genre distribution

#### Application Logs
- API requests and responses
- Database operations
- Redis cache hits/misses
- WebSocket connections
- Error traces

---

## ✅ System Configuration Summary

The backend is configured with:
- ✅ MariaDB for primary data storage
- ✅ Redis Cloud for caching (30MB free tier)
- ✅ DuckDB for analytics (optional, graceful fallback)
- ✅ WebSocket for real-time updates
- ✅ CORS configured for local development
- ✅ Proper MIDI timing and duration calculation
- ✅ Readable cache keys for easy debugging

All systems work independently with proper error handling and fallback mechanisms.
