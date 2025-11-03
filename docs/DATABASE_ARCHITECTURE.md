# Database Architecture - Aero Melody

## 🗄️ Database Stack Overview

Your project uses a **multi-database architecture** for optimal performance:

### 1. MariaDB (Primary Database)
**Connection**: `mysql+asyncmy://root:***@localhost:3306/melody_aero`

**Purpose**: Main relational database for structured data

**Tables**:
- `airports` - OpenFlights airport data (IATA codes, coordinates, etc.)
- `routes` - Flight routes between airports
- `music_compositions` - Generated music metadata
- `travel_logs` - User-created travel journeys
- `users` - User accounts (if authentication enabled)

**Key Features**:
- Async support via `asyncmy` driver
- Connection pooling (20 connections, 30 max overflow)
- JSON support for embeddings and metadata
- Full-text search capabilities

### 2. Redis Cloud (Caching & Real-time)
**Connection**: `redis://default:***@redis-14696.crce219.us-east-1-4.ec2.redns.redis-cloud.com:14696`

**Purpose**: High-speed caching and real-time features

**Plan**: 30MB Redis Cloud (optimized settings)

**Usage**:
- Cache airport search results (30 min TTL)
- Cache route calculations
- Real-time music generation updates
- Session management (2 hour TTL)
- WebSocket message broadcasting

**Optimizations**:
- Reduced cache TTL (30 min vs 1 hour)
- Limited connections (10 max)
- Efficient key expiration

### 3. DuckDB (Analytics Engine)
**Path**: `./data/analytics.duckdb`

**Purpose**: Fast analytical queries and data processing

**Configuration**:
- Memory limit: 2GB
- Threads: 4
- Embedded database (no server needed)

**Usage**:
- Route analytics and statistics
- Music pattern analysis
- Performance metrics
- Aggregated reporting
- Data export for visualization

---

## 📊 Data Flow

### Airport Search (Travel Logs)
```
User types "JFK"
    ↓
Frontend → Backend API
    ↓
MariaDB Query (airports table)
    ↓
Results cached in Redis (30 min)
    ↓
Return suggestions to frontend
```

### Music Generation
```
User selects route
    ↓
Check Redis cache
    ↓
If not cached:
  - Query MariaDB for route data
  - Generate music with PyTorch
  - Store composition in MariaDB
  - Cache result in Redis
    ↓
Log analytics to DuckDB
    ↓
Return music to user
```

### Travel Log Creation
```
User creates travel log
    ↓
Validate waypoints (MariaDB airports)
    ↓
Store in MariaDB (travel_logs table)
    ↓
Invalidate related Redis cache
    ↓
Log event to DuckDB
```

---

## 🏗️ Schema Details

### MariaDB Tables

#### airports
```sql
CREATE TABLE airports (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(255),
    country VARCHAR(255) NOT NULL,
    iata_code VARCHAR(3) UNIQUE,  -- e.g., "JFK"
    icao_code VARCHAR(4) UNIQUE,  -- e.g., "KJFK"
    latitude DECIMAL(12,10) NOT NULL,
    longitude DECIMAL(13,10) NOT NULL,
    altitude INT,
    timezone VARCHAR(50),
    dst VARCHAR(10),
    tz_database_time_zone VARCHAR(100),
    type VARCHAR(50),
    source VARCHAR(50),
    INDEX idx_iata (iata_code),
    INDEX idx_city (city),
    INDEX idx_country (country)
);
```

#### travel_logs
```sql
CREATE TABLE travel_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    waypoints JSON NOT NULL,  -- Array of {airport_code, timestamp, notes}
    travel_date DATETIME,
    tags JSON,  -- Array of tag strings
    is_public BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user (user_id),
    INDEX idx_public (is_public),
    INDEX idx_date (travel_date)
);
```

#### routes
```sql
CREATE TABLE routes (
    id INT PRIMARY KEY AUTO_INCREMENT,
    origin_airport_id INT NOT NULL,
    destination_airport_id INT NOT NULL,
    distance_km DECIMAL(10,2),
    duration_min INT,
    route_embedding JSON,  -- Vector for similarity search
    FOREIGN KEY (origin_airport_id) REFERENCES airports(id),
    FOREIGN KEY (destination_airport_id) REFERENCES airports(id),
    INDEX idx_origin (origin_airport_id),
    INDEX idx_destination (destination_airport_id)
);
```

#### music_compositions
```sql
CREATE TABLE music_compositions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    route_id INT,
    user_id INT,
    tempo INT DEFAULT 120,
    scale VARCHAR(50),
    key_signature VARCHAR(10),
    midi_data JSON,  -- MIDI note data
    metadata JSON,  -- Additional composition info
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (route_id) REFERENCES routes(id),
    INDEX idx_route (route_id),
    INDEX idx_user (user_id)
);
```

---

## 🔧 Configuration

### MariaDB Connection Pool
```python
DATABASE_POOL_SIZE = 20          # Base connections
DATABASE_MAX_OVERFLOW = 30       # Additional connections
DATABASE_POOL_TIMEOUT = 30       # Connection timeout (seconds)
DATABASE_POOL_RECYCLE = 3600     # Recycle connections after 1 hour
```

### Redis Optimization (30MB Plan)
```python
REDIS_CACHE_TTL = 1800           # 30 minutes (reduced for memory)
REDIS_SESSION_TTL = 7200         # 2 hours (reduced for memory)
REDIS_MAX_CONNECTIONS = 10       # Limited connections
```

### DuckDB Settings
```python
DUCKDB_MEMORY_LIMIT = "2GB"      # Memory limit
DUCKDB_THREADS = 4               # Parallel processing
```

---

## 🚀 Performance Optimizations

### 1. Caching Strategy
- **Airport searches**: Cached 30 min (frequently accessed)
- **Route calculations**: Cached 30 min (computationally expensive)
- **Music compositions**: Cached 30 min (large data)

### 2. Database Indexes
- IATA code index for fast airport lookups
- City/country indexes for autocomplete
- User ID indexes for travel logs
- Route indexes for music generation

### 3. Connection Pooling
- Reuse database connections
- Async operations for non-blocking I/O
- Automatic connection recycling

### 4. Query Optimization
- Use `LIMIT` for autocomplete (max 10 results)
- Case-insensitive search with `ILIKE`
- JSON queries for flexible data structures

---

## 📈 Scalability

### Current Setup (Development)
- MariaDB: Local instance
- Redis: Cloud (30MB)
- DuckDB: Embedded file

### Production Recommendations
1. **MariaDB**: Move to cloud (AWS RDS, DigitalOcean, etc.)
2. **Redis**: Upgrade plan if needed (monitor memory usage)
3. **DuckDB**: Consider separate analytics server
4. **Add Read Replicas**: For high-traffic scenarios
5. **Implement Sharding**: If data grows beyond single server

---

## 🔐 Security

### Current State
- Database credentials in `.env` file
- Redis Cloud with authentication
- No SQL injection (using SQLAlchemy ORM)
- Parameterized queries

### Production Checklist
- [ ] Use environment variables (not .env in production)
- [ ] Enable SSL/TLS for database connections
- [ ] Implement connection encryption
- [ ] Regular backups (MariaDB + DuckDB)
- [ ] Monitor Redis memory usage
- [ ] Set up database user permissions (not root)

---

## 📊 Monitoring

### Key Metrics to Track
1. **MariaDB**:
   - Connection pool usage
   - Query execution time
   - Slow query log
   - Table sizes

2. **Redis**:
   - Memory usage (critical for 30MB plan)
   - Cache hit rate
   - Eviction rate
   - Connection count

3. **DuckDB**:
   - Query performance
   - File size growth
   - Memory usage

---

## 🛠️ Maintenance

### Regular Tasks
- **Daily**: Monitor Redis memory (30MB limit)
- **Weekly**: Check MariaDB slow queries
- **Monthly**: Optimize DuckDB file (VACUUM)
- **Quarterly**: Review and archive old data

### Backup Strategy
```bash
# MariaDB backup
mysqldump -u root -p melody_aero > backup_$(date +%Y%m%d).sql

# DuckDB backup
cp ./data/analytics.duckdb ./backups/analytics_$(date +%Y%m%d).duckdb
```

---

## ✅ Summary

**Your Stack**:
- ✅ MariaDB - Primary database (airports, routes, travel logs)
- ✅ Redis Cloud - Caching (30MB optimized)
- ✅ DuckDB - Analytics (embedded)

**NOT using**:
- ❌ PostgreSQL
- ❌ MongoDB
- ❌ Cassandra

This is a well-architected multi-database system that leverages the strengths of each database type!
