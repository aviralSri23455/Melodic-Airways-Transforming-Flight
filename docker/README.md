# Aero Melody - Docker Setup

Complete Docker setup for running Aero Melody with all services.

## What is Aero Melody?

**Aero Melody** transforms flight routes into unique musical compositions using AI. It analyzes route characteristics (distance, direction, complexity) and generates MIDI music in various genres.

### Key Features
- 🎵 AI-powered music generation from 67,000+ flight routes
- 🌍 3,000+ airports worldwide
- 🎼 8 music genres (classical, jazz, electronic, ambient, etc.)
- 🔍 Vector similarity search for route recommendations
- 🤝 Real-time collaboration via WebSocket
- 📊 Analytics with DuckDB
- ⚡ Redis caching for performance

---

## Quick Start

### 1. Setup Environment

```bash
# Navigate to docker folder
cd docker

# Copy environment file
cp .env.example .env

# (Optional) Edit .env to customize passwords
```

### 2. Start All Services

```bash
# Start all containers
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 3. Initialize Database

```bash
# Wait for MariaDB to be ready (about 30 seconds)
docker-compose logs mariadb | grep "ready for connections"

# Import OpenFlights data (3,000+ airports, 67,000+ routes)
docker-compose exec backend python scripts/etl_openflights.py
```

### 4. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Aero Melody Stack                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Frontend (React + Vite) - Port 5173                         │
│  └─ Interactive UI for route selection and music playback   │
│                                                               │
│  Backend (FastAPI + Python) - Port 8000                      │
│  └─ REST API + WebSocket                                     │
│  └─ AI Music Generation (PyTorch)                            │
│  └─ Vector Similarity Search                                 │
│  └─ Real-time Collaboration                                  │
│                                                               │
│  MariaDB Database - Port 3306                                │
│  └─ 19 tables: airports, routes, compositions, etc.         │
│                                                               │
│  Redis Cache - Port 6379                                     │
│  └─ Caching + Pub/Sub for real-time features                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Services

### MariaDB (Port 3306)
- Stores all application data
- Auto-runs SQL migrations on startup
- Persistent volume for data

### Redis (Port 6379)
- Caching layer for performance
- Pub/Sub for real-time updates
- Session management

### Backend (Port 8000)
- FastAPI REST API
- PyTorch AI music generation
- Vector similarity search
- WebSocket for collaboration
- DuckDB analytics

### Frontend (Port 5173)
- React + TypeScript
- Vite for fast development
- Hot module replacement (HMR)

---

## Common Commands

### Service Management

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart a service
docker-compose restart backend

# View logs
docker-compose logs -f backend

# Check status
docker-compose ps
```

### Database Operations

```bash
# Access MariaDB (replace with your password from .env)
docker-compose exec mariadb mysql -u aero_user -p Melodic-Airways-Transforming-Flight

# Check tables
docker-compose exec mariadb mysql -u aero_user -p melody_aero -e "SHOW TABLES;"

# Count airports
docker-compose exec mariadb mysql -u aero_user -p melody_aero -e "SELECT COUNT(*) FROM airports;"

# Backup database
docker-compose exec mariadb mysqldump -u aero_user -p melody_aero > backup.sql
```

### Backend Operations

```bash
# Access backend shell
docker-compose exec backend bash

# Run ETL script
docker-compose exec backend python scripts/etl_openflights.py

# Run tests
docker-compose exec backend pytest tests/ -v

# Install new package
docker-compose exec backend pip install <package>
```

### Redis Operations

```bash
# Access Redis CLI
docker-compose exec redis redis-cli

# Check cache keys
docker-compose exec redis redis-cli KEYS '*'

# Monitor Redis
docker-compose exec redis redis-cli MONITOR
```

---

## Development

### Hot Reload

Both services support hot reload:
- **Backend**: Edit files in `../backend/` - auto-reloads via uvicorn
- **Frontend**: Edit files in `../src/` - auto-reloads via Vite HMR

### Adding Dependencies

```bash
# Backend (Python)
docker-compose exec backend pip install <package>
docker-compose exec backend pip freeze > ../backend/requirements.txt

# Frontend (Node)
docker-compose exec frontend npm install <package>
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs <service-name>

# Rebuild
docker-compose build --no-cache <service-name>
docker-compose up -d
```

### Database Connection Failed

```bash
# Check MariaDB health
docker-compose ps mariadb

# Restart MariaDB
docker-compose restart mariadb

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Port Already in Use

Edit `docker-compose.yml` and change port mapping:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead of 8000
```

### Out of Memory

Increase Docker Desktop memory:
- Settings > Resources > Memory > 4GB+

Or reduce DuckDB memory in `.env`:
```
DUCKDB_MEMORY_LIMIT=1GB
```

---

## API Endpoints

### Music Generation
- `GET /api/v1/demo/complete-demo?origin=JFK&destination=LAX` - Quick demo
- `POST /api/v1/compositions/generate` - Custom generation

### Airports & Routes
- `GET /api/v1/airports/search?query=New York` - Search airports
- `GET /api/v1/routes?limit=100` - List routes

### Vector Similarity
- `GET /api/v1/vectors/similar-routes?origin=JFK&destination=LAX` - Find similar routes
- `GET /api/v1/vectors/routes-by-genre?genre=ambient` - Routes by genre

### Real-time Collaboration
- `WS /api/v1/ws/collaborate/{session_id}/{user_id}` - WebSocket connection

Full documentation: http://localhost:8000/docs

---

## Data Flow

```
1. User selects route (JFK → LAX)
   ↓
2. Backend fetches route data from MariaDB
   ↓
3. AI analyzes route characteristics
   ↓
4. PyTorch generates MIDI composition
   ↓
5. Composition saved to database
   ↓
6. Vector embeddings created for similarity
   ↓
7. Frontend plays music
```

---

## Technology Stack

**Backend:**
- FastAPI (Python web framework)
- PyTorch (AI music generation)
- SQLAlchemy (ORM)
- Redis (caching)
- DuckDB (analytics)
- NetworkX (graph algorithms)

**Frontend:**
- React (UI framework)
- TypeScript (type safety)
- Vite (build tool)

**Database:**
- MariaDB 10.11 (primary database)
- Redis 7 (cache + pub/sub)
- DuckDB (analytics)

---

## Production Deployment

### Security Checklist
- [ ] Change all passwords in `.env`
- [ ] Set strong `JWT_SECRET_KEY`
- [ ] Use Redis Cloud (not local Redis)
- [ ] Configure SSL/TLS
- [ ] Update CORS origins
- [ ] Enable database backups
- [ ] Set up monitoring
- [ ] Use production WSGI server (Gunicorn)

### Production Tips
1. Use managed database (AWS RDS, DigitalOcean)
2. Use Redis Cloud for caching
3. Set up reverse proxy (Nginx)
4. Enable HTTPS
5. Configure log aggregation
6. Set up health monitoring

---

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Review [../backend/docs/SETUP.md](../backend/docs/SETUP.md)
3. Check [../backend/docs/API_GUIDE.md](../backend/docs/API_GUIDE.md)
4. Test API: http://localhost:8000/docs

---

## Clean Up

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```
