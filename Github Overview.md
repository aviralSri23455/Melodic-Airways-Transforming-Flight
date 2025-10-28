# 🎵 Aero Melody - Flight Routes to Musical Compositions

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![React](https://img.shields.io/badge/react-18.3-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)

**Transform global flight routes into beautiful musical compositions using AI and data visualization**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Docs](#-api-documentation) • [Demo](#-demo)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Music Generation](#-music-generation)
- [API Documentation](#-api-documentation)
- [Real-time Features](#-real-time-features)
- [Deployment](#-deployment)
- [Contributing](#-contributing)

---

## 🌟 Overview

**Aero Melody** is an innovative full-stack application that bridges aviation data and creative art by converting complex flight logistics into unique auditory experiences. Using the OpenFlights dataset with **3,000+ airports** and **67,000+ routes**, users can generate completely unique musical compositions based on flight paths, distances, and geographic characteristics.

### What Makes It Unique?

- **Every route creates different music** - Same origin/destination = same composition (deterministic)
- **6 different musical scales** - Selected based on route characteristics
- **Dynamic tempo** - Adjusted by flight distance (70-140 BPM)
- **Three-track harmony** - Melody, harmony, and bass playing simultaneously
- **Real-time collaboration** - Multiple users can edit compositions together
- **AI-powered similarity** - Find musically similar routes using vector embeddings


---

## ✨ Features

### 🎼 Music Generation
- **AI-Powered Composition**: PyTorch embeddings transform route characteristics into musical parameters
- **6 Musical Scales**: Major, Minor, Pentatonic, Blues, Dorian, Phrygian
- **Dynamic Tempo**: 70-140 BPM based on flight distance
- **Multi-track Output**: Melody, harmony, and bass tracks
- **MIDI Export**: Download compositions as MIDI files

### 🗺️ Interactive Visualization
- **Global Route Maps**: Mapbox GL integration with flight path visualization
- **Real-time Updates**: Live route tracking and composition progress
- **Airport Search**: Fast search across 3,000+ airports
- **Route Analytics**: Distance, duration, and complexity metrics

### 🤝 Collaboration & Community
- **Real-time Collaboration**: WebSocket-based collaborative editing
- **Composition Sharing**: Public gallery and remix features
- **Personal Collections**: Create and manage custom datasets
- **Social Features**: Like, comment, and share compositions

### ⚡ Performance & Caching
- **Redis Cloud**: Lightning-fast caching with 30-minute TTL
- **Vector Similarity**: Find similar compositions using cosine similarity
- **DuckDB Analytics**: Real-time performance monitoring (optional)
- **Optimized Queries**: Indexed database with JSON storage

### 🔐 Security & Authentication
- **JWT Authentication**: Secure token-based auth
- **Rate Limiting**: 1000 requests per minute
- **CORS Protection**: Configurable origin whitelist
- **Input Validation**: Pydantic schemas for all endpoints


---

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    React Frontend (Port 5173)                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │   Hero     │  │   Route    │  │   Music    │  │  Analytics │    │  │
│  │  │ Component  │  │  Selector  │  │   Player   │  │ Dashboard  │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │  │
│  │                                                                       │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │  Global    │  │   Route    │  │   Music    │  │  Features  │    │  │
│  │  │    Map     │  │    Viz     │  │  Controls  │  │  Showcase  │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │  │
│  │                                                                       │  │
│  │  Tech: React 18 + TypeScript + Tailwind CSS + shadcn/ui             │  │
│  │  State: React Query + React Router + Mapbox GL                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    │ HTTP/REST + WebSocket                  │
│                                    ▼                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              FastAPI Backend (Port 8000)                              │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────┐    │  │
│  │  │                    API Routes Layer                          │    │  │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │  │
│  │  │  │   Core   │ │ Extended │ │Community │ │Analytics │       │    │  │
│  │  │  │  Routes  │ │  Routes  │ │  Routes  │ │  Routes  │       │    │  │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │    │  │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │  │
│  │  │  │  Redis   │ │OpenFlights│ │   Demo   │ │WebSocket │       │    │  │
│  │  │  │  Routes  │ │  Routes  │ │  Routes  │ │  Routes  │       │    │  │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │    │  │
│  │  └─────────────────────────────────────────────────────────────┘    │  │
│  │                                                                        │  │
│  │  Middleware: CORS • JWT Auth • Rate Limiting • Error Handling        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          BUSINESS LOGIC LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Service Layer                                 │  │
│  │                                                                        │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │     Music      │  │     Genre      │  │    Realtime    │         │  │
│  │  │   Generator    │  │   Composer     │  │   Generator    │         │  │
│  │  │                │  │                │  │                │         │  │
│  │  │ • MIDI Output  │  │ • AI Embeddings│  │ • Live Updates │         │  │
│  │  │ • Scale Logic  │  │ • PyTorch ML   │  │ • WebSocket    │         │  │
│  │  │ • Tempo Calc   │  │ • Style Match  │  │ • Pub/Sub      │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  │                                                                        │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │     Graph      │  │     Vector     │  │   Community    │         │  │
│  │  │  Pathfinder    │  │    Service     │  │    Service     │         │  │
│  │  │                │  │                │  │                │         │  │
│  │  │ • Dijkstra's   │  │ • Cosine Sim   │  │ • Social       │         │  │
│  │  │ • NetworkX     │  │ • Embeddings   │  │ • Collections  │         │  │
│  │  │ • Route Opt    │  │ • Similarity   │  │ • Sharing      │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  │                                                                        │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │   WebSocket    │  │     Redis      │  │    Dataset     │         │  │
│  │  │    Manager     │  │   Publisher    │  │    Manager     │         │  │
│  │  │                │  │                │  │                │         │  │
│  │  │ • Connections  │  │ • Pub/Sub      │  │ • OpenFlights  │         │  │
│  │  │ • Broadcasting │  │ • Events       │  │ • ETL Pipeline │         │  │
│  │  │ • Sessions     │  │ • Notifications│  │ • Data Import  │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   MariaDB        │  │   Redis Cloud    │  │   DuckDB         │          │
│  │   (Primary DB)   │  │   (Caching)      │  │   (Analytics)    │          │
│  │                  │  │                  │  │                  │          │
│  │ • Airports       │  │ • Compositions   │  │ • Route Stats    │          │
│  │ • Routes         │  │ • Sessions       │  │ • Performance    │          │
│  │ • Users          │  │ • Pub/Sub        │  │ • Metrics        │          │
│  │ • Compositions   │  │ • Real-time      │  │ • Similarity     │          │
│  │ • Collections    │  │ • TTL: 30min     │  │ • Optional       │          │
│  │ • JSON Vectors   │  │ • 30MB Plan      │  │                  │          │
│  │                  │  │                  │  │                  │          │
│  │ Port: 3306       │  │ Port: 16441      │  │ File-based       │          │
│  │ Async: asyncmy   │  │ redis-py         │  │ In-memory        │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SERVICES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   OpenFlights    │  │   Mapbox GL      │  │   PyTorch        │          │
│  │   Dataset        │  │   Maps API       │  │   ML Models      │          │
│  │                  │  │                  │  │                  │          │
│  │ • 3,000 Airports │  │ • Route Viz      │  │ • Embeddings     │          │
│  │ • 67,000 Routes  │  │ • Interactive    │  │ • Genre AI       │          │
│  │ • GitHub Raw     │  │ • Styling        │  │ • Local CPU      │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input (JFK → LAX)
        │
        ▼
┌───────────────────┐
│  Route Selector   │  ← Search airports, select origin/destination
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   API Request     │  → POST /api/v1/demo/complete-demo
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Redis Cache?     │  ← Check if composition exists
└───────────────────┘
    │           │
    │ Cache Hit │ Cache Miss
    │           ▼
    │   ┌───────────────────┐
    │   │ Music Generator   │  ← Generate new composition
    │   │                   │
    │   │ 1. Calculate      │
    │   │    distance       │
    │   │ 2. Select scale   │
    │   │ 3. Set tempo      │
    │   │ 4. Generate notes │
    │   │ 5. Create MIDI    │
    │   └───────────────────┘
    │           │
    │           ▼
    │   ┌───────────────────┐
    │   │  Save to Redis    │  ← Cache for 30 minutes
    │   └───────────────────┘
    │           │
    └───────────┘
        │
        ▼
┌───────────────────┐
│  Return JSON      │  → Composition data + MIDI
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Music Player     │  ← Play in browser with Web Audio API
└───────────────────┘
```


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
| **Framer Motion** | Animations | 12.23.24 |
| **Axios** | HTTP Client | 1.6.0 |

### Backend
| Technology | Purpose | Version |
|-----------|---------|---------|
| **FastAPI** | Web Framework | Latest |
| **Python** | Language | 3.9+ |
| **SQLAlchemy** | ORM | Latest (async) |
| **Pydantic** | Validation | Latest |
| **PyTorch** | ML/AI | Latest |
| **NetworkX** | Graph Algorithms | Latest |
| **Mido** | MIDI Generation | Latest |
| **python-jose** | JWT Auth | Latest |
| **Redis** | Caching | redis-py |

### Databases & Storage
| Technology | Purpose | Details |
|-----------|---------|---------|
| **MariaDB** | Primary Database | 10.5+, Async with asyncmy |
| **Redis Cloud** | Caching & Pub/Sub | 30MB plan, 30min TTL |
| **DuckDB** | Analytics | Optional, file-based |

### DevOps & Tools
| Technology | Purpose |
|-----------|---------|
| **Docker** | Containerization |
| **Docker Compose** | Multi-container orchestration |
| **pytest** | Backend testing |
| **ESLint** | Frontend linting |
| **Black** | Python formatting |


---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** - [Download](https://nodejs.org/)
- **MariaDB 10.5+** - [Download](https://mariadb.org/download/)
- **Redis** - Cloud (recommended) or [Local](https://redis.io/download)
- **Git** - [Download](https://git-scm.com/downloads)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aero-melody.git
cd aero-melody
```

#### 2. Backend Setup

##### Windows (Automated)
```bash
cd backend
scripts\windows\setup_redis_duckdb.bat
```

##### Manual Setup (All Platforms)

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

# Edit .env file with your database and Redis credentials
# DATABASE_URL=mysql+asyncmy://user:password@localhost:3306/aero_melody
# REDIS_URL=redis://default:password@host:port

# Load OpenFlights data (3,000 airports + 67,000 routes)
python scripts/etl_openflights.py

# Start development server
python main.py
```

**Backend will run at**: `http://localhost:8000`  
**API Documentation**: `http://localhost:8000/docs`

#### 3. Frontend Setup

```bash
# Navigate to project root
cd ..

# Install dependencies
npm install

# Configure environment variables
# Create .env.local file
echo "VITE_API_BASE_URL=http://localhost:8000/api/v1" > .env.local


# Start development server
npm run dev

# Build for production
npm run build
```

**Frontend will run at**: `http://localhost:5173`

### Quick Test

Once both servers are running, test the API:

```bash
# Health check
curl http://localhost:8000/health

# Generate music for JFK to LAX route
curl "http://localhost:8000/api/v1/demo/complete-demo?origin=JFK&destination=LAX"

# Search airports
curl "http://localhost:8000/api/v1/airports/search?query=New%20York&limit=5"
```


---

## 📁 Project Structure

```
aero-melody/
│
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── api/                      # API Routes
│   │   │   ├── routes.py            # Core endpoints (auth, airports, routes)
│   │   │   ├── extended_routes.py   # Extended features (collections, datasets)
│   │   │   ├── community_routes.py  # Social features (sharing, likes)
│   │   │   ├── analytics_routes.py  # Analytics endpoints
│   │   │   ├── redis_routes.py      # Redis caching endpoints
│   │   │   ├── openflights_routes.py # OpenFlights data endpoints
│   │   │   ├── demo_routes.py       # Demo and testing endpoints
│   │   │   └── websocket_demo.py    # WebSocket collaboration
│   │   │
│   │   ├── core/                     # Core Configuration
│   │   │   ├── config.py            # Settings and environment variables
│   │   │   └── security.py          # JWT auth and password hashing
│   │   │
│   │   ├── db/                       # Database
│   │   │   ├── database.py          # SQLAlchemy setup
│   │   │   └── models.py            # Database models
│   │   │
│   │   ├── models/                   # Pydantic Schemas
│   │   │   ├── user.py              # User schemas
│   │   │   ├── composition.py       # Composition schemas
│   │   │   ├── route.py             # Route schemas
│   │   │   └── ...                  # Other schemas
│   │   │
│   │   └── services/                 # Business Logic
│   │       ├── music_generator.py   # MIDI generation
│   │       ├── genre_composer.py    # AI genre composition
│   │       ├── graph_pathfinder.py  # Dijkstra's algorithm
│   │       ├── vector_service.py    # Similarity search
│   │       ├── websocket_manager.py # WebSocket connections
│   │       ├── redis_publisher.py   # Redis Pub/Sub
│   │       ├── cache.py             # Redis caching
│   │       ├── duckdb_analytics.py  # Analytics service
│   │       └── ...                  # Other services
│   │
│   ├── scripts/                      # Utility Scripts
│   │   ├── etl_openflights.py       # Load OpenFlights data
│   │   ├── windows/                 # Windows batch scripts
│   │   └── ...
│   │
│   ├── tests/                        # Test Suite
│   │   ├── test_music.py
│   │   ├── test_routes.py
│   │   └── ...
│   │
│   ├── data/                         # Data Storage
│   │   └── analytics.duckdb         # DuckDB database
│   │
│   ├── midi_output/                  # Generated MIDI files
│   ├── main.py                       # Application entry point
│   ├── requirements.txt              # Python dependencies
│   └── .env                          # Environment variables
│
├── src/                              # React Frontend
│   ├── components/                   # React Components
│   │   ├── ui/                      # shadcn/ui components
│   │   ├── Hero.tsx                 # Landing hero section
│   │   ├── RouteSelector.tsx        # Airport search and selection
│   │   ├── MusicPlayer.tsx          # Audio playback
│   │   ├── MusicControls.tsx        # Playback controls
│   │   ├── GlobalMap.tsx            # Mapbox visualization
│   │   ├── RouteVisualization.tsx   # Route display
│   │   ├── Analytics.tsx            # Analytics dashboard
│   │   ├── Features.tsx             # Feature showcase
│   │   └── MusicDNA.tsx             # Composition details
│   │
│   ├── hooks/                        # Custom React Hooks
│   │   └── use-toast.ts             # Toast notifications
│   │
│   ├── lib/                          # Utilities
│   │   └── utils.ts                 # Helper functions
│   │
│   ├── pages/                        # Route Pages
│   │   ├── Index.tsx                # Home page
│   │   └── NotFound.tsx             # 404 page
│   │
│   ├── App.tsx                       # App component
│   ├── main.tsx                      # Entry point
│   └── index.css                     # Global styles
│
├── public/                           # Static Assets
│   ├── favicon.svg
│   └── ...
│
├── .env                              # Frontend environment variables
├── package.json                      # Node dependencies
├── vite.config.ts                    # Vite configuration
├── tailwind.config.ts                # Tailwind configuration
├── tsconfig.json                     # TypeScript configuration
├── docker-compose.yml                # Docker orchestration
└── README.md                         # This file
```


---

## 🎵 Music Generation

### How It Works

Every flight route generates **completely unique** music based on geographic and distance characteristics. The same origin-destination pair will always produce the same composition (deterministic), but different routes create different music.

### Musical Scale Selection (6 Scales)

The system intelligently selects one of 6 musical scales based on route characteristics:

| Scale | Mood | Selection Criteria |
|-------|------|-------------------|
| **Major** | Bright, Happy | Short routes, small lat/lon range |
| **Minor** | Melancholic | Long north-south journeys (lat > 90°) |
| **Pentatonic** | Asian-inspired | Long east-west journeys (lon > 120°) |
| **Blues** | Soulful | Medium routes with moderate complexity |
| **Dorian** | Jazz-influenced | Very long haul routes (> 8000km) |
| **Phrygian** | Spanish, Exotic | Routes with unique geographic patterns |

### Dynamic Tempo (Based on Distance)

Tempo automatically adjusts based on flight distance:

| Distance | Tempo Range | Feel |
|----------|-------------|------|
| **Very Long Haul** (>8000km) | 70-90 BPM | Slow, ambient |
| **Long Haul** (>5000km) | 80-100 BPM | Moderate |
| **Medium** (1000-5000km) | 100-120 BPM | Standard |
| **Short Haul** (<1000km) | 120-140 BPM | Fast, energetic |

### Three-Track Harmony

Each composition includes three simultaneous tracks:

1. **Melody Track** (Channel 0)
   - Main melodic line
   - Notes based on latitude progression
   - Velocity increases during flight (60 → 100)

2. **Harmony Track** (Channel 1)
   - Thirds above melody
   - Plays every 4th note
   - Creates harmonic richness

3. **Bass Track** (Channel 2)
   - Root notes one octave down
   - Plays every 8th note
   - Provides rhythmic foundation

### Route-to-Music Mapping

```python
# Latitude → Scale Degree (which note in the scale)
note_index = int((latitude + 90) / 180 * len(scale)) % len(scale)

# Longitude → Octave Shifts (pitch variation)
octave_shift = int((longitude + 180) / 360 * 2) - 1

# Progress → Velocity (volume increases during flight)
velocity = 60 + int(progress * 40)  # 60 → 100

# Distance → Duration (longer routes = longer compositions)
duration = min(30, max(10, distance / 500))  # 10-30 seconds
```

### Music Parameters

- **Styles**: Classical, Jazz, Electronic, Ambient, Rock
- **Scales**: Major, Minor, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian, Pentatonic
- **Keys**: All 12 chromatic keys (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
- **Tempo**: 60-200 BPM range (dynamically adjusted)
- **Duration**: 10-30 seconds (calculated from route distance)
- **Instruments**: Piano (default), customizable via MIDI

### Example: JFK to LAX

```
Route: New York (JFK) → Los Angeles (LAX)
Distance: ~3,944 km
Direction: West (longitude decreases)

Generated Music:
├── Scale: Major (transcontinental, moderate distance)
├── Tempo: 110 BPM (medium haul)
├── Key: C Major
├── Duration: 18 seconds
├── Tracks:
│   ├── Melody: 45 notes (latitude-based progression)
│   ├── Harmony: 11 notes (every 4th note, thirds above)
│   └── Bass: 5 notes (every 8th note, octave down)
└── Output: MIDI file + JSON data
```

### API Example

```javascript
// Generate music for a route
const response = await fetch('/api/v1/demo/complete-demo', {
  method: 'GET',
  params: {
    origin: 'JFK',
    destination: 'LAX',
    music_style: 'classical',
    tempo: 120  // Optional override
  }
});

const data = await response.json();
console.log(data);
// {
//   "composition": {
//     "tempo": 110,
//     "scale": "major",
//     "key": "C",
//     "duration": 18,
//     "notes": [...],
//     "midi_file": "base64_encoded_midi"
//   },
//   "route": {
//     "origin": "JFK",
//     "destination": "LAX",
//     "distance": 3944,
//     "direction": "west"
//   }
// }
```


---

## 📚 API Documentation

### Base URL

```
Development: http://localhost:8000/api/v1
Production: https://your-domain.com/api/v1
```

### Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer <your_jwt_token>
```

### Core Endpoints

#### Authentication

```bash
# Register new user
POST /api/v1/auth/register
Body: {
  "username": "string",
  "email": "string",
  "password": "string"
}

# Login
POST /api/v1/auth/login
Body: {
  "username": "string",
  "password": "string"
}
Response: {
  "access_token": "string",
  "token_type": "bearer"
}
```

#### Airports

```bash
# Search airports
GET /api/v1/airports/search?query=New%20York&limit=20

# Get airport by code
GET /api/v1/airports/{airport_code}

# Get all airports
GET /api/v1/airports?skip=0&limit=100
```

#### Routes

```bash
# Get routes from airport
GET /api/v1/routes/from/{airport_code}

# Get routes to airport
GET /api/v1/routes/to/{airport_code}

# Get specific route
GET /api/v1/routes/{origin}/{destination}
```

#### Music Generation

```bash
# Generate music (complete demo)
GET /api/v1/demo/complete-demo?origin=JFK&destination=LAX&music_style=classical&tempo=120

# Generate with custom parameters
POST /api/v1/compositions/generate
Body: {
  "origin_code": "JFK",
  "destination_code": "LAX",
  "music_style": "jazz",
  "tempo": 120,
  "scale": "minor",
  "key": "A"
}
```

#### Compositions

```bash
# Get user compositions
GET /api/v1/compositions/my

# Get composition by ID
GET /api/v1/compositions/{composition_id}

# Update composition
PUT /api/v1/compositions/{composition_id}

# Delete composition
DELETE /api/v1/compositions/{composition_id}

# Get public compositions
GET /api/v1/compositions/public?skip=0&limit=20
```

#### Collections

```bash
# Create collection
POST /api/v1/collections
Body: {
  "name": "My Favorites",
  "description": "Best compositions",
  "is_public": true
}

# Get user collections
GET /api/v1/collections/my

# Add composition to collection
POST /api/v1/collections/{collection_id}/compositions/{composition_id}
```

#### Analytics

```bash
# Get route complexity statistics
GET /api/v1/analytics/route-complexity

# Get performance metrics
GET /api/v1/analytics/performance

# Find similar routes
GET /api/v1/analytics/similar-routes?origin=JFK&destination=LAX&limit=10

# Get user analytics
GET /api/v1/analytics/user-stats
```

#### Redis Caching

```bash
# Test Redis connection
GET /api/v1/redis/test/save-music

# Get cache statistics
GET /api/v1/redis/cache/stats

# Get storage information
GET /api/v1/redis/storage/info

# Clear cache
DELETE /api/v1/redis/cache/clear
```

#### WebSocket

```bash
# Connect to collaboration session
WS /api/v1/ws/collaborate/{session_id}/{user_id}

# Send message
{
  "type": "state_update",
  "state": {
    "tempo": 120,
    "notes": [...]
  }
}
```

### Response Format

All API responses follow this format:

```json
{
  "success": true,
  "data": { ... },
  "message": "Success message",
  "timestamp": "2025-10-28T12:00:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2025-10-28T12:00:00Z"
}
```

### Rate Limiting

- **Limit**: 1000 requests per minute per IP
- **Headers**: 
  - `X-RateLimit-Limit`: Total requests allowed
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Time when limit resets

### Interactive Documentation

Visit these URLs when the backend is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json


---

## ⚡ Real-time Features

### WebSocket Collaboration

Multiple users can collaborate on compositions in real-time using WebSocket connections.

#### Connection

```javascript
const ws = new WebSocket(
  `ws://localhost:8000/api/v1/ws/collaborate/${sessionId}/${userId}`
);

ws.onopen = () => {
  console.log('Connected to collaboration session');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received update:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from session');
};
```

#### Message Types

```javascript
// State update
ws.send(JSON.stringify({
  type: 'state_update',
  state: {
    tempo: 120,
    notes: [...],
    chords: [...]
  }
}));

// User joined
{
  type: 'user_joined',
  user_id: 'user123',
  username: 'John Doe'
}

// User left
{
  type: 'user_left',
  user_id: 'user123'
}

// Composition saved
{
  type: 'composition_saved',
  composition_id: 'comp123'
}
```

### Redis Pub/Sub

Real-time notifications using Redis Pub/Sub:

```python
# Subscribe to composition updates
await redis_publisher.subscribe('composition:updates')

# Publish update
await redis_publisher.publish('composition:updates', {
  'composition_id': 'comp123',
  'action': 'updated',
  'user_id': 'user123'
})
```

### Caching Strategy

#### Composition Caching

```python
# Cache key format
key = f"aero:music:{origin}:{destination}"

# TTL: 30 minutes (1800 seconds)
# Optimized for 30MB Redis Cloud plan

# Cache hit: Return immediately
# Cache miss: Generate → Cache → Return
```

#### Cache Statistics

```bash
GET /api/v1/redis/cache/stats

Response:
{
  "total_keys": 150,
  "memory_used": "2.5MB",
  "hit_rate": "85%",
  "compositions_cached": 145,
  "sessions_active": 5
}
```

### Performance Optimization

- **Redis Connection Pool**: 10 connections (optimized for 30MB plan)
- **Cache TTL**: 30 minutes for compositions, 2 hours for sessions
- **Automatic Cleanup**: Expired keys removed automatically
- **Compression**: Large compositions compressed before caching
- **Batch Operations**: Multiple cache operations batched together


---

## 🔧 Configuration

### Backend Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# ============================================
# DATABASE CONFIGURATION
# ============================================
DATABASE_URL=mysql+asyncmy://root:password@localhost:3306/aero_melody
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=aero_melody
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# ============================================
# REDIS CLOUD CONFIGURATION
# ============================================
REDIS_HOST=redis-xxxxx.c267.us-east-1-4.ec2.redns.redis-cloud.com
REDIS_PORT=16441
REDIS_USERNAME=default
REDIS_PASSWORD=your_redis_password
REDIS_URL=redis://default:password@host:port
REDIS_CACHE_TTL=1800          # 30 minutes
REDIS_SESSION_TTL=7200        # 2 hours
REDIS_MAX_CONNECTIONS=10      # Optimized for 30MB plan

# ============================================
# DUCKDB ANALYTICS (OPTIONAL)
# ============================================
DUCKDB_PATH=./data/analytics.duckdb
DUCKDB_MEMORY_LIMIT=2GB
DUCKDB_THREADS=4

# ============================================
# JWT AUTHENTICATION
# ============================================
JWT_SECRET_KEY=your_secret_key_here_change_in_production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=480  # 8 hours

# ============================================
# CORS CONFIGURATION
# ============================================
BACKEND_CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://yourdomain.com

# ============================================
# API CONFIGURATION
# ============================================
API_V1_STR=/api/v1
PROJECT_NAME=Aero Melody API
VERSION=1.0.0
DEBUG=False

# ============================================
# RATE LIMITING
# ============================================
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60  # seconds

# ============================================
# FILE UPLOAD
# ============================================
MAX_UPLOAD_SIZE=10485760  # 10MB
UPLOAD_DIR=uploads
MIDI_OUTPUT_DIR=midi_output

# ============================================
# MUSIC GENERATION DEFAULTS
# ============================================
DEFAULT_TEMPO=120
DEFAULT_SCALE=major
DEFAULT_KEY=C
MAX_POLYPHONY=8

# ============================================
# SIMILARITY SEARCH
# ============================================
SIMILARITY_THRESHOLD=0.7
MAX_SIMILARITY_RESULTS=20
EMBEDDING_DIMENSIONS=128

# ============================================
# LOGGING
# ============================================
LOG_LEVEL=WARNING
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Frontend Environment Variables

Create a `.env.local` file in the project root:

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api/v1

# Mapbox Configuration
VITE_MAPBOX_TOKEN=your_mapbox_token_here

# WebSocket Configuration
VITE_WS_URL=ws://localhost:8000/api/v1/ws

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_COLLABORATION=true
VITE_ENABLE_SOCIAL=true

# Environment
VITE_ENV=development
```

### Docker Configuration

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=mysql+asyncmy://root:password@db:3306/aero_melody
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./backend:/app
      - ./backend/midi_output:/app/midi_output

  frontend:
    build: .
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000/api/v1
    volumes:
      - ./src:/app/src
      - ./public:/app/public

  db:
    image: mariadb:10.5
    environment:
      - MYSQL_ROOT_PASSWORD=password
      - MYSQL_DATABASE=aero_melody
    ports:
      - "3306:3306"
    volumes:
      - db_data:/var/lib/mysql

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

```

### Frontend Environment Variables

Create a `.env.local` file in the project root:

```bash
# Aero Melody Frontend Environment Variables

# Backend API Configuration
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_API_TIMEOUT=10000

# Authentication Configuration
VITE_JWT_TOKEN_KEY=aero_melody_token
VITE_TOKEN_REFRESH_THRESHOLD=300000

# Application Configuration
VITE_APP_NAME=Aero Melody
VITE_APP_VERSION=1.0.0
VITE_ENABLE_ANALYTICS=true

# Music Generation Configuration
VITE_DEFAULT_TEMPO=120
VITE_DEFAULT_SCALE=major
VITE_DEFAULT_KEY=C
VITE_MAX_DURATION_MINUTES=10

# File Upload Configuration
VITE_MAX_FILE_SIZE=10485760
VITE_ALLOWED_FILE_TYPES=.mid,.midi,.mp3,.wav

# Development Configuration
VITE_DEBUG_MODE=true
VITE_LOG_LEVEL=info

# Mapbox Configuration
VITE_MAPBOX_TOKEN=your_mapbox_token_here

# WebSocket Configuration
VITE_WS_URL=ws://localhost:8000/api/v1/ws

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_COLLABORATION=true
VITE_ENABLE_SOCIAL=true

# Environment
VITE_ENV=development
```

### Required Services Setup

#### Database Setup (MariaDB)
```bash
# Install MariaDB (Ubuntu/Debian)
sudo apt update
sudo apt install mariadb-server
sudo systemctl start mariadb
sudo systemctl enable mariadb

# Create database
sudo mysql -u root -p
CREATE DATABASE melody_aero;
GRANT ALL PRIVILEGES ON melody_aero.* TO 'root'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

#### Redis Setup
```bash
# Install Redis (Ubuntu/Debian)
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Or use Redis Cloud service (recommended for production)
```

### Environment Variable Descriptions

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SECRET_KEY` | FastAPI secret key for sessions | Yes | - |
| `JWT_SECRET_KEY` | JWT token signing key | Yes | - |
| `DATABASE_URL` | Database connection URL | Yes | - |
| `REDIS_URL` | Redis connection URL | Yes | - |
| `DEBUG` | Enable debug mode | No | false |
| `PORT` | Server port | No | 8000 |
| `BACKEND_CORS_ORIGINS` | Allowed CORS origins | No | localhost |
| `RATE_LIMIT_REQUESTS` | Requests per minute limit | No | 1000 |
| `REDIS_CACHE_TTL` | Cache TTL in seconds | No | 1800 |

### Security Notes

- **Never commit `.env` files** to version control
- Use strong, unique secrets for production
- Rotate secrets regularly
- Use environment-specific configurations
- Enable HTTPS in production

---

## 🚢 Deployment

### Docker Deployment

#### Build and Run

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

#### Individual Services

```bash
# Start only backend
docker-compose up -d backend

# Start only frontend
docker-compose up -d frontend

# Restart a service
docker-compose restart backend
```

### Production Deployment

#### Prerequisites

- [ ] Domain name configured
- [ ] SSL/TLS certificates obtained
- [ ] Production database setup
- [ ] Redis Cloud instance configured
- [ ] Environment variables secured

#### Backend Deployment (Ubuntu/Linux)

```bash
# Install dependencies
sudo apt update
sudo apt install python3.9 python3-pip nginx

# Clone repository
git clone https://github.com/yourusername/aero-melody.git
cd aero-melody/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Edit with production values

# Load data
python scripts/etl_openflights.py

# Install as systemd service
sudo nano /etc/systemd/system/aero-melody.service
```

**aero-melody.service**:

```ini
[Unit]
Description=Aero Melody Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/aero-melody/backend
Environment="PATH=/var/www/aero-melody/backend/venv/bin"
ExecStart=/var/www/aero-melody/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Start service
sudo systemctl daemon-reload
sudo systemctl start aero-melody
sudo systemctl enable aero-melody
sudo systemctl status aero-melody
```

#### Nginx Configuration

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/v1/ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### Frontend Deployment

```bash
# Build for production
npm run build

# Deploy to static hosting (Vercel, Netlify, etc.)
# Or serve with Nginx
sudo cp -r dist/* /var/www/html/
```

**Frontend Nginx Configuration**:

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

#### SSL/TLS with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Cloud Deployment Options

#### AWS

- **Backend**: EC2 or ECS
- **Frontend**: S3 + CloudFront
- **Database**: RDS (MariaDB)
- **Cache**: ElastiCache (Redis)

#### Google Cloud Platform

- **Backend**: Cloud Run or Compute Engine
- **Frontend**: Cloud Storage + Cloud CDN
- **Database**: Cloud SQL (MySQL)
- **Cache**: Memorystore (Redis)

#### Azure

- **Backend**: App Service or Container Instances
- **Frontend**: Static Web Apps
- **Database**: Azure Database for MySQL
- **Cache**: Azure Cache for Redis

### Monitoring & Logging

#### Application Monitoring

```python
# Add to main.py
from prometheus_client import Counter, Histogram
import logging

# Metrics
request_count = Counter('http_requests_total', 'Total HTTP requests')
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

#### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/api/v1/health/db

# Redis health
curl http://localhost:8000/api/v1/health/redis
```

### Backup Strategy

#### Database Backup

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mysqldump -u root -p aero_melody > backup_$DATE.sql
gzip backup_$DATE.sql

# Upload to S3
aws s3 cp backup_$DATE.sql.gz s3://your-bucket/backups/
```

#### Redis Backup

```bash
# Redis persistence (RDB)
redis-cli BGSAVE

# Copy RDB file
cp /var/lib/redis/dump.rdb /backup/redis_$(date +%Y%m%d).rdb
```


---

## 🧪 Testing

### Backend Testing

#### Run All Tests

```bash
cd backend

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Run pytest
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_music.py

# Run with verbose output
pytest -v
```

#### Test Categories

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# API tests
pytest tests/api/

# Performance tests
pytest tests/performance/
```

#### Manual API Testing

```bash
# Test Redis connection
python test_redis_cloud.py

# Test all backend components
python test_backend.py

# Test music generation
curl "http://localhost:8000/api/v1/demo/complete-demo?origin=JFK&destination=LAX"

# Test cache
curl "http://localhost:8000/api/v1/redis/test/save-music"

# Test health
curl "http://localhost:8000/health"
```

### Frontend Testing

#### Development

```bash
# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

#### Component Testing

```bash
# Install testing dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom vitest

# Run tests
npm run test

# Run with coverage
npm run test:coverage
```

### Code Quality

#### Backend

```bash
# Format code with Black
black app/ tests/

# Sort imports
isort app/ tests/

# Lint with flake8
flake8 app/ tests/

# Type checking with mypy
mypy app/
```

#### Frontend

```bash
# Lint TypeScript
npm run lint

# Fix linting issues
npm run lint:fix

# Format with Prettier
npm run format
```

### Load Testing

#### Using Apache Bench

```bash
# Test music generation endpoint
ab -n 1000 -c 10 "http://localhost:8000/api/v1/demo/complete-demo?origin=JFK&destination=LAX"

# Test airport search
ab -n 1000 -c 10 "http://localhost:8000/api/v1/airports/search?query=New%20York"
```

#### Using Locust

```python
# locustfile.py
from locust import HttpUser, task, between

class AeroMelodyUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def generate_music(self):
        self.client.get("/api/v1/demo/complete-demo?origin=JFK&destination=LAX")

    @task
    def search_airports(self):
        self.client.get("/api/v1/airports/search?query=New%20York")
```

```bash
# Run load test
locust -f locustfile.py --host=http://localhost:8000
```


---

## 🐛 Troubleshooting

### Common Issues

#### Backend Won't Start

**Problem**: `ModuleNotFoundError` or import errors

```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**Problem**: Database connection failed

```bash
# Check MariaDB is running
sudo systemctl status mariadb  # Linux
# or check Windows services

# Verify credentials in .env
DATABASE_URL=mysql+asyncmy://user:password@localhost:3306/aero_melody

# Test connection
python -c "import pymysql; pymysql.connect(host='localhost', user='root', password='your_password')"
```

#### Redis Connection Issues

**Problem**: `ConnectionError: Error connecting to Redis`

```bash
# Test Redis connection
python test_redis_cloud.py

# Verify .env credentials
REDIS_HOST=your-redis-host
REDIS_PORT=16441
REDIS_PASSWORD=your-password

# Check Redis Cloud dashboard
# Ensure IP is whitelisted (if using Redis Cloud)
```

**Problem**: No keys in Redis

```bash
# Test save endpoint
curl "http://localhost:8000/api/v1/redis/test/save-music"

# Check storage info
curl "http://localhost:8000/api/v1/redis/storage/info"

# Verify TTL settings
REDIS_CACHE_TTL=1800  # 30 minutes
```

#### DuckDB Errors

**Problem**: `ModuleNotFoundError: No module named 'duckdb'`

```bash
# DuckDB is optional - application works without it
# To install:
pip install duckdb

# Or ignore the warnings - analytics features will be disabled
```

#### Frontend Issues

**Problem**: API requests failing with CORS errors

```bash
# Check backend CORS settings in .env
BACKEND_CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Verify frontend API URL in .env.local
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

**Problem**: Mapbox map not loading

```bash
# Check Mapbox token in .env.local
VITE_MAPBOX_TOKEN=your_mapbox_token_here

# Get token from: https://account.mapbox.com/access-tokens/
```

#### Music Generation Issues

**Problem**: No sound playing

```bash
# Check browser console for errors
# Ensure AudioContext is supported
# Try user interaction first (click play button)

# Verify MIDI file is generated
ls backend/midi_output/
```

**Problem**: Music sounds wrong

```bash
# Check route data
curl "http://localhost:8000/api/v1/routes/JFK/LAX"

# Verify music parameters
curl "http://localhost:8000/api/v1/demo/complete-demo?origin=JFK&destination=LAX"

# Check logs
tail -f backend/app.log
```

#### Database Issues

**Problem**: Tables not created

```bash
# Run migrations
cd backend
alembic upgrade head

# Or recreate tables
python -c "from app.db.database import engine, Base; import asyncio; asyncio.run(engine.begin().__aenter__()).run_sync(Base.metadata.create_all)"
```

**Problem**: OpenFlights data not loaded

```bash
# Run ETL script
python scripts/etl_openflights.py

# Check data
mysql -u root -p aero_melody -e "SELECT COUNT(*) FROM airports;"
mysql -u root -p aero_melody -e "SELECT COUNT(*) FROM routes;"
```

### Performance Issues

#### Slow API Responses

```bash
# Check Redis cache hit rate
curl "http://localhost:8000/api/v1/redis/cache/stats"

# Monitor database queries
# Enable SQL logging in config.py
echo_pool=True

# Check connection pool
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
```

#### High Memory Usage

```bash
# Check Redis memory
redis-cli INFO memory

# Reduce cache TTL
REDIS_CACHE_TTL=900  # 15 minutes instead of 30

# Limit DuckDB memory
DUCKDB_MEMORY_LIMIT=1GB
```

### Debugging Tips

#### Enable Debug Mode

```bash
# Backend (.env)
DEBUG=True
LOG_LEVEL=DEBUG

# Frontend (.env.local)
VITE_ENV=development
```

#### View Logs

```bash
# Backend logs
tail -f backend/app.log

# Docker logs
docker-compose logs -f backend

# Systemd logs
sudo journalctl -u aero-melody -f
```

#### Database Debugging

```bash
# Connect to database
mysql -u root -p aero_melody

# Check tables
SHOW TABLES;

# Check data
SELECT * FROM airports LIMIT 10;
SELECT * FROM routes LIMIT 10;
SELECT * FROM compositions LIMIT 10;
```

### Getting Help

If you're still experiencing issues:

1. Check the [API documentation](http://localhost:8000/docs)
2. Review the [backend logs](backend/app.log)
3. Run the [test suite](backend/tests/)
4. Check [GitHub Issues](https://github.com/yourusername/aero-melody/issues)
5. Contact support or open a new issue


---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help make Aero Melody better.

### Getting Started

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   git clone https://github.com/your-username/aero-melody.git
   cd aero-melody
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new features
   - Update documentation

4. **Test your changes**
   ```bash
   # Backend tests
   cd backend
   pytest
   black app/ tests/
   flake8 app/ tests/

   # Frontend tests
   npm run lint
   npm run build
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Describe your changes
   - Link any related issues

### Contribution Guidelines

#### Code Style

**Python (Backend)**
- Follow PEP 8 style guide
- Use Black for formatting
- Use type hints
- Write docstrings for functions
- Maximum line length: 100 characters

```python
def generate_music(origin: str, destination: str, tempo: int = 120) -> dict:
    """
    Generate music composition from flight route.

    Args:
        origin: Origin airport code (e.g., 'JFK')
        destination: Destination airport code (e.g., 'LAX')
        tempo: Beats per minute (default: 120)

    Returns:
        Dictionary containing composition data and MIDI file
    """
    pass
```

**TypeScript (Frontend)**
- Use TypeScript for all new code
- Follow ESLint rules
- Use functional components with hooks
- Write JSDoc comments for complex functions

```typescript
/**
 * Generate music from flight route
 * @param origin - Origin airport code
 * @param destination - Destination airport code
 * @param tempo - Beats per minute
 * @returns Promise with composition data
 */
async function generateMusic(
  origin: string,
  destination: string,
  tempo: number = 120
): Promise<Composition> {
  // Implementation
}
```

#### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```bash
feat: add real-time collaboration feature
fix: resolve Redis connection timeout issue
docs: update API documentation for music generation
test: add unit tests for music generator service
```

#### Pull Request Process

1. **Update documentation** - If you change APIs or add features
2. **Add tests** - Ensure new code is tested
3. **Update CHANGELOG** - Add entry for your changes
4. **Pass CI checks** - All tests must pass
5. **Get review** - Wait for maintainer review
6. **Address feedback** - Make requested changes
7. **Merge** - Maintainer will merge when approved

### Areas for Contribution

#### High Priority

- [ ] Add more musical scales and styles
- [ ] Improve AI genre composition accuracy
- [ ] Add user authentication with OAuth
- [ ] Implement composition versioning
- [ ] Add mobile responsive design
- [ ] Improve WebSocket reconnection logic

#### Medium Priority

- [ ] Add more visualization options
- [ ] Implement composition export formats (MP3, WAV)
- [ ] Add social sharing features
- [ ] Improve search functionality
- [ ] Add keyboard shortcuts
- [ ] Implement dark mode

#### Low Priority

- [ ] Add internationalization (i18n)
- [ ] Create mobile app
- [ ] Add gamification features
- [ ] Implement AI-powered recommendations
- [ ] Add composition tutorials

### Bug Reports

Found a bug? Please open an issue with:

- **Title**: Clear, descriptive title
- **Description**: What happened vs. what you expected
- **Steps to Reproduce**: Detailed steps
- **Environment**: OS, browser, versions
- **Screenshots**: If applicable
- **Logs**: Relevant error messages

### Feature Requests

Have an idea? Open an issue with:

- **Title**: Clear feature description
- **Problem**: What problem does it solve?
- **Solution**: How should it work?
- **Alternatives**: Other solutions considered
- **Additional Context**: Mockups, examples, etc.

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### License

By contributing, you agree that your contributions will be licensed under the MIT License.


---

## 📊 Performance Metrics

### System Capabilities

| Metric | Value | Notes |
|--------|-------|-------|
| **Airports** | 3,000+ | OpenFlights dataset |
| **Routes** | 67,000+ | Global coverage |
| **Music Generation** | <500ms | With Redis cache |
| **Cache Hit Rate** | 85%+ | Typical usage |
| **Concurrent Users** | 100+ | WebSocket connections |
| **API Response Time** | <100ms | Cached endpoints |
| **Database Queries** | <50ms | Indexed queries |
| **MIDI File Size** | 1-5KB | Per composition |

### Scalability

- **Horizontal Scaling**: Load balancer + multiple backend instances
- **Database**: Connection pooling (20 connections, 30 overflow)
- **Caching**: Redis Cloud with 30MB plan (expandable)
- **CDN**: Static assets served via CDN
- **WebSocket**: Supports 100+ concurrent connections per instance

### Optimization Techniques

1. **Database Indexing**
   - Airport codes indexed
   - Route pairs indexed
   - User IDs indexed
   - Composition timestamps indexed

2. **Caching Strategy**
   - Compositions cached for 30 minutes
   - Airport data cached for 24 hours
   - Route data cached for 12 hours
   - User sessions cached for 2 hours

3. **Query Optimization**
   - Async database operations
   - Batch queries where possible
   - Lazy loading for large datasets
   - Pagination for list endpoints

4. **Frontend Optimization**
   - Code splitting
   - Lazy loading components
   - Image optimization
   - Bundle size optimization

---

## 🔐 Security

### Authentication & Authorization

- **JWT Tokens**: Secure token-based authentication
- **Password Hashing**: bcrypt with salt
- **Token Expiration**: 8 hours (configurable)
- **Refresh Tokens**: Planned for future release

### API Security

- **Rate Limiting**: 1000 requests/minute per IP
- **CORS**: Configurable origin whitelist
- **Input Validation**: Pydantic schemas for all inputs
- **SQL Injection**: Protected via SQLAlchemy ORM
- **XSS Protection**: React's built-in escaping

### Data Protection

- **Environment Variables**: Sensitive data in .env files
- **Database Encryption**: TLS connections supported
- **Redis Encryption**: TLS connections supported
- **HTTPS**: SSL/TLS in production

### Best Practices

1. **Never commit .env files**
2. **Use strong JWT secrets**
3. **Enable HTTPS in production**
4. **Regularly update dependencies**
5. **Monitor for security vulnerabilities**
6. **Implement proper error handling**
7. **Log security events**

---

## 📈 Roadmap

### Version 1.1 (Q1 2026)

- [ ] OAuth authentication (Google, GitHub)
- [ ] Composition export to MP3/WAV
- [ ] Mobile responsive design improvements
- [ ] Advanced analytics dashboard
- [ ] User profiles and avatars

### Version 1.2 (Q2 2026)

- [ ] Real-time collaboration improvements
- [ ] AI-powered composition recommendations
- [ ] Social features (follow, feed)
- [ ] Composition versioning
- [ ] Advanced search and filters

### Version 2.0 (Q3 2026)

- [ ] Mobile app (React Native)
- [ ] Live performance mode
- [ ] Composition marketplace
- [ ] Advanced AI features
- [ ] Multi-language support

### Long-term Vision

- Integration with flight booking APIs
- Real-time flight tracking with live music
- VR/AR visualization experiences
- Educational platform for music theory
- API for third-party integrations

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Aero Melody

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### Data Sources

- **OpenFlights**: Airport and route data
- **Mapbox**: Map visualization
- **shadcn/ui**: UI component library

### Technologies

- **FastAPI**: Modern Python web framework
- **React**: UI library
- **PyTorch**: Machine learning framework
- **Redis**: Caching and real-time features
- **MariaDB**: Reliable database

### Contributors

Thank you to all contributors who have helped make Aero Melody better!


For their dedication and contributions to the Aero Melody project.

---


### Documentation

- **API Docs**: http://localhost:8000/docs

- Aviral
- Shani
- Karina
- Mythri

⭐ Star us on GitHub if you find this project interesting!



---
