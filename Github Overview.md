# 🎵 Aero Melody - Flight Routes to Musical Compositions

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![React](https://img.shields.io/badge/react-18.3-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)

**Transform global flight routes into beautiful musical compositions using AI and data visualization**

[Quick Start](#-quick-start) • [Features](#-features) • [API Docs](#-api-documentation) • [Tech Stack](#-technology-stack)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Backend Setup](#-backend-setup)
- [API Documentation](#-api-documentation)
- [Technology Stack](#-technology-stack)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

**Aero Melody** transforms flight routes into unique musical compositions using **3,000+ airports** and **67,000+ routes** from OpenFlights. Every route generates deterministic music with dynamic tempo (70-140 BPM), multiple scales, and real-time AI embeddings for similarity search.

### Why This Matters

**Data Sonification for Learning & Creativity**

Aero Melody bridges the gap between abstract data and human perception by transforming geographic information into sound. This approach:

- **Makes Data Tangible**: Geographic distances, coordinates, and route complexity become audible patterns, making abstract concepts concrete and memorable
- **Enhances Education**: Students learn geography, graph theory, and music theory simultaneously through multi-sensory experiences that improve retention
- **Enables Creative Exploration**: Artists and musicians can discover new compositional ideas by exploring the musical patterns hidden in global flight networks
- **Supports Wellness**: Long-distance routes generate calming ambient soundscapes, offering a unique approach to therapeutic audio
- **Demonstrates AI Applications**: Shows practical use of neural networks, vector embeddings, and similarity search in a creative, accessible context
- **Inspires Innovation**: Proves that any structured dataset can be transformed into meaningful artistic expression, opening doors for other data sonification projects

Whether you're an educator teaching complex concepts, a developer exploring AI/ML applications, a musician seeking inspiration, or simply curious about the intersection of data and art, Aero Melody offers a unique lens to experience our connected world.

### Key Highlights

- **Deterministic Music Generation**: Same route = same composition
- **6 Musical Scales**: Major, Minor, Pentatonic, Blues, Dorian, Phrygian
- **Multi-track Harmony**: Melody, harmony, and bass
- **Real-time Embeddings**: AI-powered similarity search
- **Educational Platform**: Interactive lessons on geography and music theory
- **Wellness Features**: Therapeutic soundscapes
- **VR/AR Support**: 3D globe with spatial audio
- **Redis Cloud Caching**: Sub-millisecond performance
- **FAISS Vector Search**: Fast similarity matching

---

## ✨ Features

### 🎼 Music Generation
- AI-powered composition with PyTorch embeddings
- Dynamic tempo based on flight distance
- MIDI export and real-time playback
- Genre-specific composition (8 genres)

### 🗺️ Visualization & Analytics
- Global route maps with Mapbox GL
- Real-time route tracking
- Airport search across 3,000+ airports
- Route complexity and performance metrics
- DuckDB analytics for insights

### 🎓 Educational Platform
- Interactive lessons (Geography, Graph Theory, Music Theory)
- Real-time quizzes with visual feedback
- Interactive lab for experimentation
- Learning insights dashboard

### 🧘 Wellness Features
- Calming soundscapes (3 themes)
- Adjustable calm level (0-100)
- Binaural frequency support
- Serene route recommendations

### 🥽 VR/AR Experience
- 3D globe visualization
- Animated flight paths
- WebXR support (Oculus, HTC Vive, Valve Index)
- Orbit controls for navigation

### ⚡ Performance & Security
- Redis Cloud caching (30-min TTL)
- FAISS vector similarity search (~1ms)
- JWT authentication
- Rate limiting (1000 req/min)
- CORS protection
- **Input Validation**: Pydantic schemas for all endpoints

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** - [Download](https://nodejs.org/)
- **MariaDB 10.5+** - [Download](https://mariadb.org/download/)
- **Redis** - Cloud (recommended) or [Local](https://redis.io/download)
- **Git** - [Download](https://git-scm.com/downloads)

### Database Setup

Before installing the application, set up your MariaDB database:

```bash
# Login to MariaDB as root
mysql -u root -p

# Create database
CREATE DATABASE aero_melody CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# Create user with permissions
CREATE USER 'aero_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON aero_melody.* TO 'aero_user'@'localhost';
FLUSH PRIVILEGES;

# Verify database creation
SHOW DATABASES;
USE aero_melody;

# Exit
EXIT;
```

**Note**: Replace `your_secure_password` with a strong password. You'll use these credentials in your `.env` file.

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aero-melody.git
cd aero-melody
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Copy .env.example to .env and edit with your credentials
cp .env.example .env

# Load OpenFlights data (3,000 airports + 67,000 routes)
python scripts/etl_openflights.py

# Setup vector embeddings (optional but recommended)
setup_vector_embeddings.bat  # Windows
# or
python scripts/generate_route_embeddings.py  # Manual

# Start development server
python main.py
```

**Backend runs at**: `http://localhost:8000`  
**API Documentation**: `http://localhost:8000/docs`

#### 3. Frontend Setup

```bash
# Navigate to project root
cd ..

# Install dependencies
npm install

# Configure environment variables
echo "VITE_API_BASE_URL=http://localhost:8000/api/v1" > .env.local

# Start development server
npm run dev
```

**Frontend runs at**: `http://localhost:5173`

### Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Generate music for JFK to LAX route
curl "http://localhost:8000/api/v1/demo/complete-demo?origin=JFK&destination=LAX"

# Search airports
curl "http://localhost:8000/api/v1/airports/search?query=New%20York&limit=5"

# Find similar routes (requires vector embeddings)
curl "http://localhost:8000/api/v1/vectors/similar-routes?origin=JFK&destination=LAX&limit=10"
```

### ✅ Verify Vector Embeddings After Git Clone

After cloning the repository and setting up the backend, run this test to verify vector embeddings are working:

```bash
cd backend
python test_vector_embeddings.py
```

**This test will check:**
1. ✅ Database schema (vector columns exist)
2. ✅ Embeddings generated (coverage percentage)
3. ✅ Complexity metrics calculated
4. ✅ Sample embedding data
5. ✅ FAISS index (if available)

**Expected Output:**
```
============================================================
🔍 TESTING VECTOR EMBEDDINGS
============================================================

✓ Test 1: Checking database schema...
   ✅ All vector columns exist: ['route_embedding', 'melodic_complexity', 'harmonic_complexity', 'rhythmic_complexity']

✓ Test 2: Checking if embeddings are generated...
   Total routes: 67663
   Routes with embeddings: 67663
   Coverage: 100.0%
   ✅ Embeddings are generated!

✓ Test 3: Checking complexity metrics...
   Average melodic complexity: 0.450
   Average harmonic complexity: 0.320
   Average rhythmic complexity: 0.180
   ✅ Complexity metrics are calculated!

✓ Test 4: Checking sample embedding...
   Sample route: JFK → LAX
   Distance: 3974.0 km
   Embedding dimension: 128D
   Melodic: 0.199, Harmonic: 0.039, Rhythmic: 0.000
   ✅ Sample embedding looks good!

✓ Test 5: Checking FAISS index...
   FAISS index found: 67663 vectors
   ✅ FAISS index is ready!

============================================================
✅ VECTOR EMBEDDINGS ARE WORKING!
============================================================

📝 Summary:
   • Database schema: ✅ Ready
   • Embeddings generated: ✅ 100.0% coverage
   • Complexity metrics: ✅ Calculated
   • Sample data: ✅ Valid

🎵 You can now use vector similarity search!
   Try: GET /api/v1/vectors/similar-routes?origin=JFK&destination=LAX
```

**If embeddings are not set up:**
The test will show you exactly what's missing and provide commands to fix it:
- Missing database columns → Run SQL migration
- No embeddings → Run generation script
- No FAISS index → Optional but recommended

---

## 🧬 Vector Embeddings - AI-Powered Music Similarity

### Overview

Vector embeddings enable **semantic similarity search** for routes based on **real-time music characteristics** extracted during playback. Using PyTorch neural networks and FAISS indexing, the system can find musically similar routes in ~1ms.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OpenFlights Dataset                              │
│                  3,000+ Airports | 67,000+ Routes                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Feature Extraction (16D)                          │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐     │
│  │ Geographic   │ Route        │ Musical      │ Semantic     │     │
│  │ Features     │ Chars        │ Mapping      │ Features     │     │
│  │ (8D)         │ (4D)         │ (4D)         │ (4D)         │     │
│  └──────────────┴──────────────┴──────────────┴──────────────┘     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PyTorch Neural Network                             │
│                                                                       │
│  Input Layer (16D)                                                   │
│         ↓                                                            │
│  Hidden Layer 1 (64D) + ReLU + BatchNorm + Dropout(0.2)            │
│         ↓                                                            │
│  Hidden Layer 2 (128D) + ReLU + BatchNorm + Dropout(0.2)           │
│         ↓                                                            │
│  Output Layer (128D) + Tanh Normalization                           │
│                                                                       │
│                   128-Dimensional Embeddings                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Storage & Indexing                              │
│  ┌──────────────────┬──────────────────┬──────────────────┐        │
│  │   MariaDB        │   FAISS Index    │   DuckDB         │        │
│  │   (Metadata)     │   (Vectors)      │   (Analytics)    │        │
│  │                  │                  │                  │        │
│  │ • Route info     │ • 128D vectors   │ • Statistics     │        │
│  │ • Complexity     │ • Fast search    │ • Aggregations   │        │
│  │ • JSON storage   │ • ~1ms queries   │ • Metrics        │        │
│  │ • Real-time      │ • 35MB index     │ • Cache          │        │
│  └──────────────────┴──────────────────┴──────────────────┘        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API Endpoints                                   │
│  • /vectors/similar-routes     - Find similar routes                │
│  • /vectors/routes-by-genre    - Genre-based discovery              │
│  • /vectors/route/{id}/complexity - Complexity metrics              │
│  • /vectors/statistics         - System statistics                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Similarity Search

```
User Query (JFK → LAX)
        │
        ▼
┌───────────────────────┐
│ Get Route Coordinates │
│ • Origin: 40.64°N     │
│ • Dest: 33.94°N       │
│ • Distance: 3,974 km  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Extract Features (16D)│
│ • Geographic (8D)     │
│ • Route chars (4D)    │
│ • Musical map (4D)    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ PyTorch Encoder       │
│ 16D → 64D → 128D      │
│ Generate Embedding    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ FAISS Similarity      │
│ Search (~1ms)         │
│ Find Top K Neighbors  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Fetch Metadata        │
│ from MariaDB          │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Return Results (JSON) │
│ • JFK → SFO (0.95)    │
│ • JFK → SEA (0.92)    │
│ • EWR → LAX (0.90)    │
└───────────────────────┘
```

### Features

#### 🎯 Semantic Similarity Search
Find routes that are musically similar based on:
- **Geographic characteristics** - Distance, direction, coordinates
- **Route properties** - Number of stops, airlines, popularity
- **Musical features** - Tempo, pitch, harmony, rhythm patterns
- **Complexity metrics** - Harmonic, rhythmic, melodic complexity

#### 🎵 Genre-Based Discovery
Match routes to musical genres with AI-powered classification:

| Genre | Characteristics | Example Routes |
|-------|----------------|----------------|
| **Classical** | Complex, formal, multiple stops | JFK → LHR → CDG → FCO |
| **Jazz** | Improvisational, varied, unpredictable | JFK → MIA → PTY → BOG |
| **Electronic** | Repetitive, rhythmic, medium-distance | JFK → ORD → DEN → LAX |
| **Ambient** | Long, calm, minimal stops, transoceanic | JFK → NRT (10,850 km) |
| **Pop** | Popular, straightforward, direct | JFK → LAX (3,974 km) |

#### 📊 Complexity Metrics
Calculate three types of complexity for any route:

```
Harmonic Complexity = |dest_lat - origin_lat| / 180
  → Measures latitude change (north-south movement)
  → Range: 0.0 (no change) to 1.0 (pole to pole)

Rhythmic Complexity = stops / 5
  → Measures number of stops/connections
  → Range: 0.0 (direct) to 1.0+ (5+ stops)

Melodic Complexity = distance_km / 20000
  → Measures route distance
  → Range: 0.0 (short) to 1.0 (20,000+ km)

Overall Complexity = (harmonic × 0.3) + (rhythmic × 0.3) + (melodic × 0.4)
  → Weighted average of all three metrics
```

#### ⚡ Performance
- **Search Time**: ~1ms per query (FAISS IndexFlatL2)
- **Throughput**: ~1,000 queries/second
- **Memory**: ~35MB for 67,000 route index
- **Accuracy**: 100% (exact search, no approximation)
- **Generation**: 5-10 minutes for all routes

### Quick Setup

```bash
cd backend
setup_vector_embeddings.bat
```

This automated script will:
1. ✅ Check PyTorch and FAISS installation
2. ✅ Add database columns for embeddings
3. ✅ Generate 128D vectors for all 67,000 routes
4. ✅ Build FAISS index for fast search
5. ✅ Test similarity search functionality

**Time**: 5-10 minutes | **One-time setup**

### API Endpoints

#### Find Similar Routes
```bash
curl "http://localhost:8000/api/v1/vectors/similar-routes?origin=JFK&destination=LAX&limit=10"
```

**Response**:
```json
[
  {
    "route_id": 12345,
    "origin_code": "JFK",
    "dest_code": "SFO",
    "distance_km": 4139.0,
    "similarity_score": 0.95
  },
  {
    "route_id": 12346,
    "origin_code": "JFK",
    "dest_code": "SEA",
    "distance_km": 3876.0,
    "similarity_score": 0.92
  }
]
```

#### Find Routes by Genre
```bash
# Calm, long-distance routes
curl "http://localhost:8000/api/v1/vectors/routes-by-genre?genre=ambient&limit=20"

# Complex, formal routes
curl "http://localhost:8000/api/v1/vectors/routes-by-genre?genre=classical&limit=20"

# Improvisational, varied routes
curl "http://localhost:8000/api/v1/vectors/routes-by-genre?genre=jazz&limit=20"
```

#### Get Route Complexity
```bash
curl "http://localhost:8000/api/v1/vectors/route/12345/complexity"
```

**Response**:
```json
{
  "harmonic_complexity": 0.75,
  "rhythmic_complexity": 0.60,
  "melodic_complexity": 0.82,
  "overall_complexity": 0.72
}
```

#### Get Statistics
```bash
curl "http://localhost:8000/api/v1/vectors/statistics"
```

**Response**:
```json
{
  "total_routes": 67663,
  "routes_with_embeddings": 67663,
  "embedding_coverage": 100.0,
  "avg_melodic_complexity": 0.45,
  "faiss_index_size": 67663,
  "embedding_dimension": 128
}
```

### How It Works

#### Step 1: Feature Extraction (16 Dimensions)

```python
# Geographic Features (8D)
origin_lat_norm = (origin_lat + 90) / 180  # Normalize to [0, 1]
origin_lon_norm = (origin_lon + 180) / 360
dest_lat_norm = (dest_lat + 90) / 180
dest_lon_norm = (dest_lon + 180) / 360
lat_diff = abs(dest_lat - origin_lat) / 180
lon_diff = abs(dest_lon - origin_lon) / 360
distance_norm = distance_km / 20000  # Normalize to [0, 1]
bearing = calculate_bearing(origin, dest) / 360

# Route Characteristics (4D)
stops_norm = stops / 5
airline_norm = num_airlines / 10
avg_lat = (origin_lat + dest_lat) / 2 / 90
avg_lon = (origin_lon + dest_lon) / 2 / 180

# Musical Mapping (4D)
tempo_feature = 1 - distance_norm  # Shorter = faster tempo
pitch_feature = avg_lat  # Latitude affects pitch
harmony_feature = stops_norm  # More stops = more harmony
rhythm_feature = airline_norm  # More airlines = varied rhythm
```

#### Step 2: Neural Network Encoding (16D → 128D)

```python
class RouteEmbeddingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 128)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.tanh(self.fc3(x))  # Normalize to [-1, 1]
        return x
```

#### Step 3: FAISS Indexing

```python
import faiss

# Create FAISS index (exact search)
dimension = 128
index = faiss.IndexFlatL2(dimension)

# Add all route embeddings
embeddings = np.array([route.embedding for route in routes])
index.add(embeddings)  # 67,000 vectors

# Search for similar routes (~1ms)
query_embedding = generate_embedding(origin, destination)
distances, indices = index.search(query_embedding, k=10)
```

#### Step 4: Similarity Calculation

```python
# Cosine similarity
similarity = 1 - (distance / 2)  # Convert L2 distance to similarity

# Results sorted by similarity score
results = [
    {"route_id": idx, "similarity": sim}
    for idx, sim in zip(indices[0], similarities)
]
```

### Use Cases

#### 🎓 Education
```python
# Find complex routes for teaching graph theory
complex_routes = await service.find_routes_by_genre("classical", limit=10)
# Use these routes to demonstrate Dijkstra's algorithm with sound
```

#### 🧘 Wellness
```python
# Find calm routes for therapeutic soundscapes
calm_routes = await service.find_routes_by_genre("ambient", limit=20)
# Generate relaxing music from transoceanic flights
```

#### 🎵 Entertainment
```python
# Find similar routes for playlist generation
similar = await service.find_similar_routes("JFK", "LAX", limit=10)
# Create "routes that sound like your trip"
```

#### 📊 Analytics
```python
# Analyze melodic complexity patterns
complexity = await service.calculate_melodic_complexity(route_id)
# Identify most complex routes for advanced compositions
```

### Architecture Benefits

- ✅ **Fast similarity search** (~1ms per query)
- ✅ **Scalable to millions of routes** (current: 67,000)
- ✅ **Integrates with existing infrastructure** (MariaDB, Redis, DuckDB)
- ✅ **Supports multiple use cases** (education, wellness, entertainment, analytics)
- ✅ **Real-time updates** with Redis pub/sub
- ✅ **Cached results** with Redis (1-hour TTL)
- ✅ **Analytics with DuckDB** for aggregations and metrics
- ✅ **100% accuracy** with exact search (IndexFlatL2)
- ✅ **Low memory footprint** (~35MB for 67,000 vectors)
- ✅ **Easy to extend** for new features and use cases

### Documentation

Comprehensive documentation available in `backend/docs/`:

- **📘 Quick Start** (`VECTOR_QUICK_START.md`) - 5-minute setup guide
- **📗 Complete Guide** (`VECTOR_EMBEDDING_GUIDE.md`) - Full documentation
- **📙 Commands** (`VECTOR_COMMANDS.md`) - Command reference
- **📕 Architecture** (`VECTOR_ARCHITECTURE.md`) - System design
- **📔 Index** (`VECTOR_INDEX.md`) - Documentation navigation

### Technical Specifications

| Component | Technology | Details |
|-----------|-----------|---------|
| **Neural Network** | PyTorch | 3-layer encoder (16→64→128→128) |
| **Vector Index** | FAISS | IndexFlatL2 (exact search) |
| **Database** | MariaDB | JSON storage for embeddings |
| **Analytics** | DuckDB | Fast aggregations and metrics |
| **Cache** | Redis | 1-hour TTL for search results |
| **Dimension** | 128D | Optimal balance of accuracy/speed |
| **Coverage** | 100% | All 67,000 routes embedded |

---

## 🎵 Music Generation

### How It Works

Every flight route generates **completely unique** music based on geographic and distance characteristics.

### Musical Scale Selection (6 Scales)

| Scale | Mood | Selection Criteria |
|-------|------|-------------------|
| **Major** | Bright, Happy | Short routes, small lat/lon range |
| **Minor** | Melancholic | Long north-south journeys (lat > 90°) |
| **Pentatonic** | Asian-inspired | Long east-west journeys (lon > 120°) |
| **Blues** | Soulful | Medium routes with moderate complexity |
| **Dorian** | Jazz-influenced | Very long haul routes (> 8000km) |
| **Phrygian** | Spanish, Exotic | Routes with unique geographic patterns |

### Dynamic Tempo (Based on Distance)

| Distance | Tempo Range | Feel |
|----------|-------------|------|
| **Very Long Haul** (>8000km) | 70-90 BPM | Slow, ambient |
| **Long Haul** (>5000km) | 80-100 BPM | Moderate |
| **Medium** (1000-5000km) | 100-120 BPM | Standard |
| **Short Haul** (<1000km) | 120-140 BPM | Fast, energetic |

### Three-Track Harmony

1. **Melody Track** (Channel 0) - Main melodic line based on latitude
2. **Harmony Track** (Channel 1) - Thirds above melody, plays every 4th note
3. **Bass Track** (Channel 2) - Root notes one octave down, plays every 8th note

### Route-to-Music Mapping

```python
# Latitude → Scale Degree
note_index = int((latitude + 90) / 180 * len(scale)) % len(scale)

# Longitude → Octave Shifts
octave_shift = int((longitude + 180) / 360 * 2) - 1

# Progress → Velocity (volume increases during flight)
velocity = 60 + int(progress * 40)  # 60 → 100

# Distance → Duration
duration = min(30, max(10, distance / 500))  # 10-30 seconds
```

---

## 📚 Educational Platform

### Features

#### 🗺️ Geography Through Sound
- Learn world geography by hearing musical representations of flight routes
- Distance-to-pitch correlation
- Direction-based melody generation
- Interactive map-based learning

#### 🕸️ Graph Theory Visualization
- Dijkstra's algorithm sonification
- Network connectivity through harmony
- Shortest path musical representation
- Algorithm step-by-step audio visualization

#### 🎵 Music Theory Lessons
- Interactive scale and mode exploration
- Tempo and rhythm understanding
- Harmony and chord progression learning
- Beginner to advanced difficulty levels

### Interactive Quizzes

- **Multiple choice questions** with visual feedback
- **Select answer** → Highlights in blue
- **Check answer** → Shows correct (green) or incorrect (red)
- **Explanations** provided for all answers
- **Quiz locks** after checking to prevent changes

### Interactive Lab

- **Real-time route generation** with backend integration
- **Select origin and destination** from 8 major airports
- **Generate music** and see results instantly
- **Learning insights** explain how data becomes music
- **Experiment suggestions** for hands-on learning

### Access

Navigate to `/education` or click "Education" in the navigation bar

---

## 🧘 Wellness & Therapeutic Features

### Calming Soundscapes

Generate therapeutic music from serene flight routes with three themes:

#### Ocean Breeze
- Gentle wave-like melodies
- Coastal route recommendations (LAX → HNL, MIA → CUN)
- Calming tempo (60-70 BPM)

#### Mountain Serenity
- Peaceful ambient soundscapes
- Mountain route recommendations (DEN → SLC, GVA → INN)
- Meditative tempo (50-60 BPM)

#### Night Flight
- Soothing overnight compositions
- Long-haul route recommendations (JFK → LHR, LAX → NRT)
- Deep relaxation with binaural frequencies (45-55 BPM)

### Customization

- Adjustable calm level (0-100)
- Duration control (1-30 minutes)
- Binaural beat integration for deep relaxation

### Access

Navigate to `/wellness` or click "Wellness" in the navigation bar

---

## 🥽 VR/AR Immersive Experience

### Features

#### Interactive 3D Globe
- Real Earth representation with transparent blue sphere
- Rotating animation during playback
- Mouse controls for 360° viewing

#### Airport Markers
- 8 major airports with color-coded markers
- Animated rotating spheres
- Airport code labels
- Real geographic coordinates

#### Animated Flight Paths
- Curved trajectory between airports
- Real-time progress indicator
- Animated plane model following the path
- Trail effect showing completed journey

#### Playback Controls
- Play/Pause functionality
- Reset to beginning
- Adjustable speed (0.5x to 3x)
- Progress bar with percentage

#### WebXR VR Support
- Detects VR headset capability
- "Enter VR" button for immersive mode
- Compatible with Oculus Quest, HTC Vive, Valve Index

### How to Use

1. Navigate to `/vr-ar`
2. Select origin and destination airports
3. Click "Play" to start animation
4. Use mouse to rotate, zoom, and pan
5. Click "Enter VR" for immersive mode (requires VR headset)

### Available Airports

| Code | Name | Location | Color |
|------|------|----------|-------|
| JFK | New York JFK | USA | Blue |
| CDG | Paris CDG | France | Purple |
| LHR | London Heathrow | UK | Pink |
| NRT | Tokyo Narita | Japan | Orange |
| DXB | Dubai | UAE | Green |
| SYD | Sydney | Australia | Cyan |
| LAX | Los Angeles | USA | Red |
| SIN | Singapore | Singapore | Purple |

### Access

Navigate to `/vr-ar` or click "VR/AR" in the navigation bar

---

## 📚 API Documentation

### Base URL

```
Development: http://localhost:8000/api/v1
Production: https://your-domain.com/api/v1
```

### Core Endpoints

#### Music Generation

```bash
# Generate music (complete demo)
GET /api/v1/demo/complete-demo?origin=JFK&destination=LAX

# Generate with custom parameters
POST /api/v1/compositions/generate
Body: {
  "origin_code": "JFK",
  "destination_code": "LAX",
  "music_style": "jazz",
  "tempo": 120
}
```

#### Vector Embeddings

```bash
# Find similar routes
GET /api/v1/vectors/similar-routes?origin=JFK&destination=LAX&limit=10

# Find routes by genre
GET /api/v1/vectors/routes-by-genre?genre=ambient&limit=20

# Get route complexity
GET /api/v1/vectors/route/{route_id}/complexity

# Get statistics
GET /api/v1/vectors/statistics
```

#### Education

```bash
# Get available lessons
GET /api/v1/education/lessons

# Start a lesson
POST /api/v1/education/lessons/{id}/start

# Get graph visualization
GET /api/v1/education/graph-visualization/{origin}/{destination}
```

#### Wellness

```bash
# Generate calming soundscape
POST /api/v1/wellness/generate-wellness
Body: {
  "theme": "ocean",
  "calm_level": 80,
  "duration_minutes": 5
}

# Get wellness themes
GET /api/v1/wellness/wellness-themes
```

#### VR/AR

```bash
# Create VR session
POST /api/v1/vr-ar/create-session
Body: {
  "origin": "JFK",
  "destination": "CDG",
  "enable_spatial_audio": true,
  "quality": "high"
}

# Get supported airports
GET /api/v1/vr-ar/supported-airports

# Get VR capabilities
GET /api/v1/vr-ar/vr-capabilities
```

### Interactive Documentation

Visit these URLs when the backend is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

---

## 🛠️ Technology Stack

### Frontend

| Technology | Purpose | Version |
|-----------|---------|---------|
| **React** | UI Framework | 18.3.1 |
| **TypeScript** | Type Safety | 5.8.3 |
| **Vite** | Build Tool | 5.4.19 |
| **Tailwind CSS** | Styling | 3.4.17 |
| **shadcn/ui** | Component Library | Latest |
| **React Query** | Data Fetching | 5.83.0 |
| **React Router** | Navigation | 6.30.1 |
| **Mapbox GL** | Map Visualization | Via CDN |
| **Three.js** | 3D Graphics | Latest |
| **Framer Motion** | Animations | 12.23.24 |

### Backend

| Technology | Purpose | Version |
|-----------|---------|---------|
| **FastAPI** | Web Framework | Latest |
| **Python** | Language | 3.9+ |
| **SQLAlchemy** | ORM | Latest (async) |
| **Pydantic** | Validation | Latest |
| **PyTorch** | ML/AI | Latest |
| **FAISS** | Vector Search | faiss-cpu |
| **NetworkX** | Graph Algorithms | Latest |
| **Mido** | MIDI Generation | Latest |

### Databases & Storage

| Technology | Purpose | Details |
|-----------|---------|---------|
| **MariaDB** | Primary Database | 10.5+, Async with asyncmy |
| **Redis Cloud** | Caching & Pub/Sub | 30MB plan, 30min TTL |
| **DuckDB** | Analytics | File-based, in-memory |

---

## 📁 Project Structure

```
aero-melody/
│
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── api/                      # API Routes
│   │   │   ├── routes.py            # Core endpoints
│   │   │   ├── vector_routes.py     # Vector embeddings
│   │   │   ├── education_routes.py  # Education platform
│   │   │   ├── wellness_routes.py   # Wellness features
│   │   │   ├── vr_ar_routes.py      # VR/AR endpoints
│   │   │   └── ...
│   │   │
│   │   ├── core/                     # Core Configuration
│   │   │   ├── config.py            # Settings
│   │   │   └── security.py          # JWT auth
│   │   │
│   │   ├── db/                       # Database
│   │   │   ├── database.py          # SQLAlchemy setup
│   │   │   └── models.py            # Database models
│   │   │
│   │   ├── models/                   # Pydantic Schemas
│   │   │   └── ...
│   │   │
│   │   └── services/                 # Business Logic
│   │       ├── music_generator.py   # MIDI generation
│   │       ├── route_embedding_service.py  # Vector embeddings
│   │       ├── realtime_vector_sync.py     # Real-time sync
│   │       └── ...
│   │
│   ├── scripts/                      # Utility Scripts
│   │   ├── etl_openflights.py       # Load OpenFlights data
│   │   ├── generate_route_embeddings.py  # Generate embeddings
│   │   └── ...
│   │
│   ├── sql/                          # SQL Scripts
│   │   └── add_vector_embeddings.sql
│   │
│   ├── docs/                         # Documentation
│   │   ├── VECTOR_QUICK_START.md
│   │   ├── VECTOR_EMBEDDING_GUIDE.md
│   │   ├── VECTOR_COMMANDS.md
│   │   ├── VECTOR_ARCHITECTURE.md
│   │   └── VECTOR_INDEX.md
│   │
│   ├── main.py                       # Application entry point
│   ├── requirements.txt              # Python dependencies
│   └── .env                          # Environment variables
│
├── src/                              # React Frontend
│   ├── components/                   # React Components
│   │   ├── ui/                      # shadcn/ui components
│   │   ├── Hero.tsx
│   │   ├── RouteSelector.tsx
│   │   ├── MusicPlayer.tsx
│   │   └── ...
│   │
│   ├── pages/                        # Route Pages
│   │   ├── Index.tsx                # Home page
│   │   ├── Education.tsx            # Education platform
│   │   ├── Wellness.tsx             # Wellness features
│   │   ├── VrAr.tsx                 # VR/AR experience
│   │   └── ...
│   │
│   ├── App.tsx                       # App component
│   └── main.tsx                      # Entry point
│
├── public/                           # Static Assets
├── .env                              # Frontend environment variables
├── package.json                      # Node dependencies
├── vite.config.ts                    # Vite configuration
├── tailwind.config.ts                # Tailwind configuration
└── README.md                         # This file
```

---

## 🧪 Testing

### Quick Test Checklist

#### Navigation
- [ ] All tabs clickable
- [ ] Active tab highlighted
- [ ] URLs update correctly
- [ ] Back button works

#### Home Page
- [ ] Route selection works
- [ ] Music generation works
- [ ] Audio playback works
- [ ] Analytics display
- [ ] Map visualization

#### Wellness
- [ ] Theme selection works
- [ ] Calm level slider works
- [ ] Generation works
- [ ] Playback works
- [ ] All 3 themes tested

#### Education
- [ ] Lessons display
- [ ] Quizzes interactive
- [ ] "Try Interactive Lab" works
- [ ] Lab generates music
- [ ] Tabs work

#### VR/AR
- [ ] Globe renders
- [ ] Route selection works
- [ ] Animation plays
- [ ] Controls work
- [ ] VR button shows (if supported)

#### Backend APIs
- [ ] /generate works
- [ ] /wellness/* works
- [ ] /education/* works
- [ ] /vectors/* works
- [ ] Swagger UI accessible

### Testing Commands

```bash
# Health check
curl http://localhost:8000/health

# Test music generation
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{"origin":"JFK","destination":"CDG","music_style":"major","tempo":120}'

# Test wellness
curl -X POST "http://localhost:8000/api/v1/wellness/generate-wellness" \
  -H "Content-Type: application/json" \
  -d '{"theme":"ocean","calm_level":70,"duration_minutes":5}'

# Test education
curl "http://localhost:8000/api/v1/education/lessons"

# Test vectors
curl "http://localhost:8000/api/v1/vectors/statistics"
```

### Performance Testing

- **Load Time**: < 2 seconds for initial load
- **API Calls**: < 500ms response time
- **Memory**: No continuous growth
- **Search**: ~1ms for vector similarity

---

## � Teroubleshooting

### Common Issues

#### Database Connection Errors

**Problem**: `Can't connect to MySQL server` or `Access denied for user`

**Solutions**:
```bash
# Check MariaDB is running
# Windows:
net start MariaDB

# Linux/Mac:
sudo systemctl status mariadb

# Verify credentials in backend/.env
DATABASE_URL=mysql+asyncmy://aero_user:your_password@localhost:3306/aero_melody

# Test connection manually
mysql -u aero_user -p aero_melody
```

#### Redis Connection Issues

**Problem**: `Error connecting to Redis` or `Connection refused`

**Solutions**:
```bash
# For Redis Cloud: Verify URL format in .env
REDIS_URL=redis://default:password@host:port

# For local Redis, start the service:
# Windows:
redis-server

# Linux/Mac:
sudo systemctl start redis
```

#### Vector Embeddings Not Working

**Problem**: `No embeddings found` or `FAISS index missing`

**Solutions**:
```bash
cd backend

# Run the setup script
setup_vector_embeddings.bat  # Windows
# or
python scripts/generate_route_embeddings.py  # Manual

# Verify with test
python test_vector_embeddings.py
```

#### Port Already in Use

**Problem**: `Address already in use` on port 8000 or 5173

**Solutions**:
```bash
# Find and kill process using the port
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9

# Or change the port in your config
# Backend: uvicorn app.main:app --port 8001
# Frontend: npm run dev -- --port 5174
```

#### Module Not Found Errors

**Problem**: `ModuleNotFoundError: No module named 'X'`

**Solutions**:
```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Frontend Build Errors

**Problem**: `Cannot find module` or TypeScript errors

**Solutions**:
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf .vite

# Rebuild
npm run build
```

#### MIDI Playback Issues

**Problem**: No sound or audio errors in browser

**Solutions**:
- Check browser console for errors
- Ensure browser supports Web Audio API (Chrome, Firefox, Edge recommended)
- Verify audio is not muted in browser/system
- Try a different browser
- Check that MIDI data is being generated (inspect network tab)

#### Slow Performance

**Problem**: API responses are slow or UI is laggy

**Solutions**:
```bash
# Check Redis is working (should cache results)
redis-cli ping  # Should return PONG

# Verify database indexes exist
mysql -u aero_user -p aero_melody
SHOW INDEX FROM routes;

# Monitor backend logs for slow queries
# Check DuckDB analytics cache

# For vector search, ensure FAISS index is built
ls backend/faiss_index.bin  # Should exist
```

#### Environment Variables Not Loading

**Problem**: `KeyError` or missing configuration values

**Solutions**:
```bash
# Verify .env file exists and has correct format
# Backend: backend/.env
# Frontend: .env.local (at project root)

# Check for typos in variable names
# Ensure no spaces around = sign
# Example: DATABASE_URL=value (not DATABASE_URL = value)

# Restart servers after changing .env files
```

### Getting Help

If you encounter issues not covered here:

1. Check the [API Documentation](http://localhost:8000/docs) for endpoint details
2. Review backend logs for error messages
3. Check browser console for frontend errors
4. Verify all prerequisites are installed and running
5. Ensure all environment variables are set correctly
6. Try the test commands in the [Testing](#-testing) section
7. Open an issue on GitHub with error logs and steps to reproduce

---

## 🚢 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Deployment

#### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
npm run build
# Serve the dist/ folder with your web server
```

### Environment Variables

#### Backend (.env)

```env
DATABASE_URL=mysql+asyncmy://user:password@localhost:3306/aero_melody
REDIS_URL=redis://default:password@host:port
SECRET_KEY=your-secret-key
MAPBOX_TOKEN=your-mapbox-token
```

#### Frontend (.env.local)

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_MAPBOX_TOKEN=your-mapbox-token
```

---

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or suggesting ideas, your help is appreciated.

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/aero-melody.git
   cd aero-melody
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up development environment** (see [Quick Start](#-quick-start))

### Development Guidelines

#### Code Style

**Python (Backend)**
- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use docstrings for classes and functions
- Format code with `black`:
  ```bash
  pip install black
  black backend/app
  ```

**TypeScript/React (Frontend)**
- Follow [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use functional components with hooks
- Use TypeScript for type safety
- Format code with Prettier:
  ```bash
  npm run format
  ```

#### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add wellness theme customization
fix: resolve MIDI playback issue in Safari
docs: update vector embeddings guide
style: format code with black
refactor: simplify route generation logic
test: add unit tests for music generator
chore: update dependencies
```

#### Testing Requirements

**Before submitting a PR, ensure:**

1. **Backend tests pass**:
   ```bash
   cd backend
   pytest tests/
   ```

2. **Frontend builds without errors**:
   ```bash
   npm run build
   ```

3. **Linting passes**:
   ```bash
   # Backend
   flake8 backend/app
   
   # Frontend
   npm run lint
   ```

4. **Manual testing checklist**:
   - [ ] Feature works as expected
   - [ ] No console errors
   - [ ] No breaking changes to existing features
   - [ ] API endpoints return correct responses
   - [ ] UI is responsive on mobile/desktop

#### Adding New Features

**For new API endpoints:**
1. Add route in `backend/app/api/`
2. Create Pydantic schemas in `backend/app/models/`
3. Add business logic in `backend/app/services/`
4. Update API documentation (docstrings)
5. Add tests in `backend/tests/`

**For new UI components:**
1. Create component in `src/components/`
2. Use TypeScript interfaces for props
3. Follow existing component patterns
4. Ensure accessibility (ARIA labels, keyboard navigation)
5. Test on multiple screen sizes

**For new features:**
1. Discuss in GitHub Issues first (for major changes)
2. Update documentation (README, backend/docs/)
3. Add examples and usage instructions
4. Consider backward compatibility

#### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass** locally
4. **Update CHANGELOG.md** with your changes
5. **Submit PR** with clear description:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Manual testing completed
   
   ## Screenshots (if applicable)
   ```

6. **Respond to review feedback** promptly
7. **Squash commits** if requested before merge

### Areas for Contribution

**Good First Issues:**
- Documentation improvements
- UI/UX enhancements
- Bug fixes
- Test coverage improvements
- Performance optimizations

**Feature Ideas:**
- Additional musical scales and modes
- More wellness themes
- Enhanced VR/AR features
- Mobile app development
- Additional data sources beyond OpenFlights
- Social features (share compositions)
- Export to different audio formats
- Real-time collaboration features

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow
- Follow the [Contributor Covenant](https://www.contributor-covenant.org/)

### Questions?

- Open a [GitHub Discussion](https://github.com/yourusername/aero-melody/discussions)
- Check existing [Issues](https://github.com/yourusername/aero-melody/issues)
- Review [Documentation](backend/docs/)

Thank you for contributing to Aero Melody! 🎵✈️

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **OpenFlights** - For the comprehensive aviation dataset
- **FastAPI** - For the excellent Python web framework
- **React** - For the powerful UI library
- **PyTorch** - For AI/ML capabilities
- **FAISS** - For fast similarity search
- **shadcn/ui** - For beautiful UI components

---

## 📞 Support

- **Documentation**: See the `backend/docs/` folder
- **API Docs**: http://localhost:8000/docs
- **Issues**: GitHub Issues
- **Email**: support@aeromelody.com

---

## 🎉 Get Started

```bash
# Clone the repository
git clone https://github.com/yourusername/aero-melody.git
cd aero-melody

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/etl_openflights.py
setup_vector_embeddings.bat  # Optional but recommended
python main.py

# Setup frontend (in new terminal)
cd ..
npm install
npm run dev
```

Visit http://localhost:5173 and start creating music from flight routes! 🎵✈️

---

**Built with ❤️ by the Aero Melody Team**
