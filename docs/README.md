# Aero Melody Backend Documentation

**Transform Flight Routes into Musical Compositions using AI**

A FastAPI backend that converts aviation data into unique MIDI music files using PyTorch, NetworkX, and Mido.

---

## 📚 Documentation Structure

### Quick Start
- **[Setup & Run Commands](./SETUP.md)** - Installation, database setup, and running the server

### API Reference
- **[Frontend API Guide](./API_GUIDE.md)** - Complete API documentation for frontend integration

### Implementation
- **[Implementation Details](./implementation/)** - Technical guides and architecture

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Setup Database
```bash
# Create database
mysql -u root -p
CREATE DATABASE aero_melody;

# Run migrations
mysql -u root -p aero_melody < sql/create_tables.sql
mysql -u root -p aero_melody < sql/enhanced_schema_migration.sql
mysql -u root -p aero_melody < sql/community_features_tables.sql
```

### 3. Import Data
```bash
python scripts/etl_openflights.py
```

### 4. Start Server
```bash
python main.py
# Server: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## 🛠 Technology Stack

- **FastAPI** - Async web framework
- **Python 3.9+** - Primary language
- **MariaDB** - Primary database
- **Redis Cloud** - Caching and real-time features
- **PyTorch** - Neural networks for embeddings
- **NetworkX** - Graph algorithms (Dijkstra)
- **Mido** - MIDI file generation
- **FAISS** - Vector similarity search

---

## 🎵 Key Features

- **Music Generation** - Convert flight routes to MIDI compositions
- **Vector Embeddings** - AI-powered similarity search
- **Real-time Collaboration** - WebSocket support
- **Educational Platform** - Interactive lessons
- **Wellness Features** - Therapeutic soundscapes
- **VR/AR Support** - 3D visualization

---

## 📖 API Documentation

When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:
```bash
# Copy example file
cp .env.example .env

# Edit with your credentials
DATABASE_URL=mysql://user:password@localhost/aero_melody
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-secure-password
JWT_SECRET_KEY=your-secure-secret-key
```

See `ENVIRONMENT_SETUP.md` in the project root for detailed setup instructions.

---

## 📊 Database Schema

**Core Tables:**
- `airports` - 3,000+ airports with coordinates
- `routes` - 67,000+ flight routes
- `music_compositions` - Generated music metadata
- `users` - Authentication
- `user_datasets` - Personal collections
- `collaboration_sessions` - Real-time sessions

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific features
pytest tests/test_new_features.py -v

# Test database connection
python tests/test_db.py

# Test Redis connection
python tests/test_redis_cloud.py
```

---

## 🤝 Support

For detailed information, see:
- [Setup Guide](./SETUP.md)
- [API Guide](./API_GUIDE.md)
- [Implementation Details](./implementation/)
