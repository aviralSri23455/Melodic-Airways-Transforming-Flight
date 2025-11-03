# 🦆 DuckDB Complete Guide - Aero Melody Analytics

Complete guide for DuckDB analytics and vector embeddings in Aero Melody, covering flight routes, music generation, VR experiences, and travel logs.

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Overview](#-overview)
3. [Vector Embeddings Integration](#-vector-embeddings-integration)
4. [Analytics Features](#-analytics-features)
5. [API Endpoints](#-api-endpoints)
6. [Custom SQL Functions](#-custom-sql-functions)
7. [Database Schema](#-database-schema)
8. [Running Analytics](#-running-analytics)
9. [Configuration](#-configuration)
10. [Troubleshooting](#-troubleshooting)
11. [Performance](#-performance)
12. [Future Enhancements](#-future-enhancements)

---

## 🚀 Quick Start

### Run Analytics

From the `backend` directory:

```bash
python run_analytics.py
```

This will:
1. Import data from MariaDB (7,698 airports, 66,935 routes)
2. Generate comprehensive analytics report
3. Export CSV files to `./analytics_export/`

### What You'll See

**Route Analysis:**
- Total routes and statistics
- Distance categories (Very Short to Very Long)
- Top connected airports (Atlanta, Chicago, London...)
- Most popular origin/destination pairs

**Airport Analysis:**
- Distribution by country (US leads with 1,512 airports)
- Altitude statistics
- Highest altitude airports (Daocheng Yading at 14,472 ft)
- Hub scores

**Insights:**
- Longest route: Sydney → Dallas (13,808 km)
- Shortest route: Papa Westray → Westray (2.82 km)
- Average flight speed: 808 km/h

---

## 📖 Overview

This folder contains DuckDB-powered analytics tools for analyzing your Aero Melody flight routes and music generation data.

### Why DuckDB?

- **Fast**: Columnar storage optimized for analytics
- **Embedded**: No server needed, just a file
- **SQL**: Full SQL support with advanced analytics functions
- **Portable**: Single file database
- **Python-friendly**: Easy integration with pandas and numpy

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Creates Item                         │
│         (Composition / VR Experience / Travel Log)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Generate Vector Embedding                       │
│    • AI Composer: 128D                                       │
│    • VR Experience: 64D                                      │
│    • Travel Log: 32D                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├──────────────┬──────────────────────┐
                       ▼              ▼                      ▼
              ┌────────────┐  ┌──────────────┐    ┌──────────────┐
              │   FAISS    │  │   DuckDB     │    │   Response   │
              │  (Real-time│  │  (Analytics) │    │  to User     │
              │   Search)  │  │              │    │              │
              └────────────┘  └──────────────┘    └──────────────┘
```

---

## 🎯 Vector Embeddings Integration

### Features

#### ✅ Automatic Sync
- Every created item automatically syncs embeddings to DuckDB
- Non-blocking operation - doesn't slow down API responses
- Graceful fallback if DuckDB is unavailable

#### 🔍 Vector Similarity Search
- **Cosine Similarity**: Find semantically similar items
- **Euclidean Distance**: Calculate L2 distance between embeddings
- Custom UDFs registered in DuckDB for fast queries

#### 📊 Analytics Capabilities
- Genre clustering analysis
- VR route popularity
- Travel pattern analysis
- Temporal trends
- CSV export for external tools

### Benefits

1. **Dual Storage Strategy**
   - **FAISS**: Fast real-time similarity search (in-memory)
   - **DuckDB**: Persistent analytics and complex queries (on-disk)

2. **No Performance Impact**
   - Sync happens asynchronously
   - Doesn't block API responses
   - Graceful degradation if DuckDB unavailable

3. **Advanced Analytics**
   - SQL queries on embeddings
   - Temporal analysis
   - Cross-feature correlations
   - Export to CSV for external tools

4. **Scalability**
   - DuckDB handles millions of embeddings efficiently
   - Columnar storage optimized for analytics
   - Parallel query execution

---

## 📊 Analytics Features

### What Gets Analyzed

#### Routes
- Total routes: 66,935
- Distance statistics (min, avg, max)
- Duration statistics
- Routes by distance category
- Most connected airports (origins and destinations)

#### Airports
- Total airports: 7,698
- Distribution by country
- Altitude statistics
- Highest altitude airports
- Hub scores

#### Travel Logs (when available)
- Most popular music genres
- Most saved routes
- Genre-route correlations

#### Insights
- Longest route
- Shortest route
- Average flight speed

### 📁 Files

#### `analytics.py`
Main analytics script that imports data from MariaDB and generates comprehensive reports.

**Features:**
- Imports airports and routes from MariaDB to DuckDB
- Analyzes route patterns (distance, duration, categories)
- Shows top connected airports
- Country and altitude statistics
- Exports results to CSV files

**Usage:**
```bash
cd backend
python duckdb_analytics/analytics.py
```

**Output:**
- Console report with statistics
- CSV exports in `./analytics_export/`
- DuckDB database at `./data/analytics.duckdb`

#### `query.py`
Interactive query tool for running custom SQL queries against the DuckDB database.

**Usage:**
```bash
# Show quick statistics
python duckdb_analytics/query.py stats

# Interactive mode
python duckdb_analytics/query.py interactive

# Run a specific query
python duckdb_analytics/query.py "SELECT COUNT(*) FROM routes"
```

#### `check_schema.py`
Utility to inspect the DuckDB table schemas and sample data.

**Usage:**
```bash
python duckdb_analytics/check_schema.py
```

#### `vector_embeddings.py`
DuckDB vector store for managing and analyzing vector embeddings.

**Usage:**
```bash
cd backend/duckdb_analytics
python vector_embeddings.py
```

---

## 🔌 API Endpoints

### Get Vector Statistics
```bash
GET /api/v1/analytics/duckdb/vector-stats
```

**Response:**
```json
{
  "success": true,
  "data": {
    "enabled": true,
    "ai_composer": {
      "total_embeddings": 150,
      "unique_genres": 8,
      "avg_tempo": 115.5,
      "avg_complexity": 0.72
    },
    "vr_experiences": {
      "total_embeddings": 75,
      "unique_types": 3,
      "unique_origins": 45,
      "avg_duration": 58.3
    },
    "travel_logs": {
      "total_embeddings": 120,
      "avg_waypoints": 3.2,
      "earliest_date": "2025-01-15",
      "latest_date": "2025-11-03"
    }
  }
}
```

### Generate Vector Report
```bash
GET /api/v1/analytics/duckdb/vector-report
```

Returns comprehensive analytics including:
- Embedding statistics
- Genre clusters
- VR route analysis
- Travel patterns

### Search Similar Compositions
```bash
POST /api/v1/analytics/duckdb/search-similar-compositions
Content-Type: application/json

{
  "embedding": [0.123, -0.456, ...],  // 128D array
  "k": 5,
  "filter": "jazz"  // Optional genre filter
}
```

### Search Similar VR Experiences
```bash
POST /api/v1/analytics/duckdb/search-similar-vr-experiences
Content-Type: application/json

{
  "embedding": [0.123, -0.456, ...],  // 64D array
  "k": 5,
  "filter": "cinematic"  // Optional type filter
}
```

### Search Similar Travel Logs
```bash
POST /api/v1/analytics/duckdb/search-similar-travel-logs
Content-Type: application/json

{
  "embedding": [0.123, -0.456, ...],  // 32D array
  "k": 5,
  "filter": "3"  // Optional min waypoints
}
```

### Get DuckDB Info
```bash
GET /api/v1/analytics/duckdb/info
```

---

## 🔧 Custom SQL Functions

DuckDB now has custom UDFs for vector operations:

### cosine_similarity(vec1, vec2)
```sql
SELECT 
    id,
    genre,
    cosine_similarity(embedding, [0.1, 0.2, ...]) as similarity
FROM ai_composer_embeddings
ORDER BY similarity DESC
LIMIT 10;
```

### euclidean_distance(vec1, vec2)
```sql
SELECT 
    id,
    origin,
    destination,
    euclidean_distance(embedding, [0.1, 0.2, ...]) as distance
FROM vr_experience_embeddings
ORDER BY distance ASC
LIMIT 10;
```

---

## 🗄️ Database Schema

### DuckDB Tables

#### 1. ai_composer_embeddings
```sql
CREATE TABLE ai_composer_embeddings (
    id VARCHAR PRIMARY KEY,
    genre VARCHAR,
    tempo INTEGER,
    complexity FLOAT,
    duration INTEGER,
    embedding FLOAT[128],
    metadata JSON,
    created_at TIMESTAMP
)
```

#### 2. vr_experience_embeddings
```sql
CREATE TABLE vr_experience_embeddings (
    id VARCHAR PRIMARY KEY,
    experience_type VARCHAR,
    origin VARCHAR,
    destination VARCHAR,
    duration FLOAT,
    embedding FLOAT[64],
    metadata JSON,
    created_at TIMESTAMP
)
```

#### 3. travel_log_embeddings
```sql
CREATE TABLE travel_log_embeddings (
    id INTEGER PRIMARY KEY,
    title VARCHAR,
    waypoint_count INTEGER,
    travel_date TIMESTAMP,
    embedding FLOAT[32],
    metadata JSON,
    created_at TIMESTAMP
)
```

### Database Location

DuckDB database: `./data/analytics.duckdb`

This is a local analytical database that mirrors your MariaDB data for fast querying without impacting your production database.

---

## 🏃 Running Analytics

### Command Line

```bash
# Generate full analytics report
cd backend
python run_analytics.py

# Generate vector embedding report
cd backend/duckdb_analytics
python vector_embeddings.py

# Interactive queries
python duckdb_analytics/query.py interactive

# Quick stats
python duckdb_analytics/query.py stats

# Single query
python duckdb_analytics/query.py "SELECT * FROM routes WHERE distance_km > 10000 LIMIT 10"
```

### Python API

```python
from duckdb_analytics.vector_embeddings import DuckDBVectorStore

# Initialize
vector_store = DuckDBVectorStore()

# Get statistics
stats = vector_store.get_embedding_statistics()
print(stats)

# Find similar compositions
similar = vector_store.find_similar_compositions(
    query_embedding=[0.1, 0.2, ...],  # 128D
    k=5,
    genre_filter="jazz"
)

# Generate report
vector_store.generate_vector_report()

# Export to CSV
vector_store.export_embeddings_to_csv()

# Close
vector_store.close()
```

### Console Output

When items are created, you'll see:
```
✅ Generated composition with 128D vector embedding
✅ Added composition to FAISS index: comp_0_1730000000.0 (total: 1)
✅ Stored AI Composer embedding: comp_0_1730000000.0
```

When running analytics:
```
✅ DuckDB Vector Store connected: ./data/aero_melody_analytics.duckdb
✅ Vector embedding tables created/verified
✅ Vector similarity functions registered

🔍 DUCKDB VECTOR EMBEDDINGS REPORT
======================================================================
Generated: 2025-11-03 20:54:46

📊 EMBEDDING STATISTICS
----------------------------------------------------------------------

🎵 AI Composer (128D Embeddings):
  Total: 150
  Unique Genres: 8
  Avg Tempo: 115.5 BPM
  Avg Complexity: 0.72

🎮 VR Experiences (64D Embeddings):
  Total: 75
  Unique Types: 3
  Unique Origins: 45
  Avg Duration: 58.3s

✈️  Travel Logs (32D Embeddings):
  Total: 120
  Avg Waypoints: 3.2
  Date Range: 2025-01-15 to 2025-11-03

🎼 GENRE CLUSTERS
----------------------------------------------------------------------
  jazz            45 compositions (tempo: 140, complexity: 0.85)
  classical       32 compositions (tempo: 95, complexity: 0.78)
  electronic      28 compositions (tempo: 128, complexity: 0.65)
  ...

🛫 TOP VR ROUTES
----------------------------------------------------------------------
  JFK → LAX (cinematic    ) 12 experiences
  LHR → JFK (immersive    ) 10 experiences
  ...
```

---

## ⚙️ Configuration

### DuckDB Settings

Database settings are read from `app/core/config.py`:

```python
DUCKDB_PATH = "./data/aero_melody_analytics.duckdb"
DUCKDB_MEMORY_LIMIT = "2GB"
DUCKDB_THREADS = 4
```

### Environment Variables

Check your `.env` file has correct database credentials:
- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`

---

## 📤 Exports

CSV files are exported to: `./analytics_export/`
- `top_routes_by_distance.csv` - Top 1,000 longest routes
- `airports_by_country.csv` - Airport counts by country
- `music_genre_stats.csv` - Music generation statistics (when available)

---

## 🔍 Example Queries

### Find Jazz Compositions Similar to a Query
```sql
SELECT 
    id,
    genre,
    tempo,
    cosine_similarity(embedding, ?::FLOAT[128]) as similarity
FROM ai_composer_embeddings
WHERE genre = 'jazz'
ORDER BY similarity DESC
LIMIT 10;
```

### Analyze VR Routes by Popularity
```sql
SELECT 
    origin,
    destination,
    COUNT(*) as experience_count,
    AVG(duration) as avg_duration
FROM vr_experience_embeddings
GROUP BY origin, destination
ORDER BY experience_count DESC;
```

### Find Travel Logs by Season
```sql
SELECT 
    title,
    waypoint_count,
    EXTRACT(MONTH FROM travel_date) as month,
    COUNT(*) OVER (PARTITION BY EXTRACT(MONTH FROM travel_date)) as logs_in_month
FROM travel_log_embeddings
ORDER BY travel_date DESC;
```

### Custom Queries in Interactive Mode

```bash
python duckdb_analytics/query.py interactive
```

Then run SQL queries like:
```sql
SELECT country, COUNT(*) FROM airports GROUP BY country ORDER BY COUNT(*) DESC LIMIT 5;
```

---

## 🛠️ Troubleshooting

### "Table does not exist" error
Run the analytics script first to import data:
```bash
python run_analytics.py
```

### MariaDB connection error
Check your `.env` file has correct database credentials:
- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`

### Memory issues
Adjust in `app/core/config.py`:
```python
DUCKDB_MEMORY_LIMIT: str = "4GB"  # Increase if needed
```

---

## 💡 Tips

1. **Fast queries**: DuckDB is optimized for analytics - query millions of rows instantly
2. **No impact on production**: Analytics run on a separate DuckDB file
3. **Portable**: Share the `.duckdb` file with team members
4. **Incremental updates**: Re-run analytics to refresh data from MariaDB
5. Run analytics periodically to get updated insights
6. Use the query tool for custom analysis
7. The database file is portable - you can share it with others

---

## 📈 Performance

- **Embedding Storage**: ~1ms per item
- **Similarity Search**: ~5-10ms for 10,000 embeddings
- **Analytics Queries**: ~50-100ms for complex aggregations
- **Disk Usage**: 
  - 128D embedding: ~512 bytes
  - 64D embedding: ~256 bytes
  - 32D embedding: ~128 bytes

---

## 📂 File Structure

```
backend/
├── duckdb_analytics/
│   ├── vector_embeddings.py      # DuckDB vector store
│   ├── analytics.py               # Main analytics
│   ├── query.py                   # Interactive queries
│   └── check_schema.py            # Schema inspection
├── app/
│   ├── services/
│   │   ├── duckdb_sync_service.py # Auto-sync service
│   │   ├── ai_genre_composer.py   # AI Composer (syncs to DuckDB)
│   │   ├── vrar_experience_service.py # VR (syncs to DuckDB)
│   │   └── travel_log_service.py  # Travel Logs (syncs to DuckDB)
│   └── api/
│       └── duckdb_vector_routes.py # DuckDB API endpoints
├── data/
│   └── aero_melody_analytics.duckdb # DuckDB database file
└── analytics_export/              # CSV exports
```

---

## 🚀 Future Enhancements

1. **Clustering**: Automatic K-means clustering of embeddings
2. **Dimensionality Reduction**: t-SNE/UMAP for visualization
3. **Anomaly Detection**: Find unusual compositions/experiences
4. **Trend Analysis**: Track embedding drift over time
5. **Cross-Feature Search**: Find VR experiences similar to compositions
6. **Recommendation Engine**: Hybrid collaborative + content-based filtering

---

## ✅ Status

**COMPLETE** - DuckDB vector embeddings fully integrated with automatic sync!

---

## 🧪 Testing

```bash
# 1. Create some items via API
curl -X POST http://localhost:8000/api/v1/ai/ai-genres/compose \
  -H "Content-Type: application/json" \
  -d '{"genre": "jazz", "route_features": {...}, "duration": 30}'

# 2. Check DuckDB stats
curl http://localhost:8000/api/v1/analytics/duckdb/vector-stats

# 3. Generate report
curl http://localhost:8000/api/v1/analytics/duckdb/vector-report

# 4. Run analytics script
cd backend/duckdb_analytics
python vector_embeddings.py
```

---

## 📝 Summary

Your application now has a **complete vector embedding pipeline**:
- ✅ Real-time FAISS search for instant results
- ✅ DuckDB analytics for deep insights
- ✅ Automatic sync without performance impact
- ✅ SQL queries on vector embeddings
- ✅ Export capabilities for external analysis
- ✅ Production-ready and scalable

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example queries
3. Run analytics with verbose logging
4. Check DuckDB connection status via API

---

**Happy Analyzing! 🦆✈️🎵**
