# 🎵 Melodic Airways – Transforming  - Flight Routes to Musical Compositions

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![React](https://img.shields.io/badge/react-18.3-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-red.svg)
![VR/AR](https://img.shields.io/badge/VR%2FAR-Ready-purple.svg)

**Transform global flight routes into beautiful musical compositions using AI, 3D visualization, and immersive experiences**

[Features](#-features) • [Quick Start](#-quick-start) • [New Features](#-new-advanced-features) • [API Docs](#-api-documentation) • [3D VR](#-vr-ar-experiences)

</div>

---


## 🎥 Video Demos

**Watch Melodic Airways in Action:**

- **Part 1 - Core Features & Music Generation**: [Watch on Loom](https://www.loom.com/share/3258c1e904644252947adfdf76b01f08)
- **Part 2 - AI Composer & VR Experience**: [Watch on Loom](https://www.loom.com/share/52b7f23b0dc64ed090fc552e876a4104)



## � Recento Updates (Nov 2025)

**Project Cleanup & Optimization:**
- ✅ Consolidated setup scripts into `backend/bat/` folder
- ✅ Removed redundant Windows batch files and shell scripts
- ✅ Cleaned up old MySQL/MariaDB scripts (now using DuckDB)
- ✅ Removed unnecessary lockfiles (bun.lockb) and test configs
- ✅ Streamlined project structure for better maintainability
- ✅ Updated setup process: Single `setup.bat` for Windows, `setup.sh` for Linux/Mac

**Setup is now simpler:**
```bash
cd backend/bat
setup.bat  # Windows - includes Redis + DuckDB setup
```

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [🚀 Advanced Features](#-advanced-features)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Music Generation](#-music-generation)
- [VR/AR Experiences](#-vr-ar-experiences)
- [Vector Embeddings & Analytics](#-vector-embeddings--analytics)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🌟 Overview

**Melodic Airways** is a revolutionary full-stack application that transforms aviation data into immersive musical and visual experiences. Using real OpenFlights data with **3,000+ airports** and **67,000+ routes**, the platform generates unique musical compositions, 3D flight visualizations, and VR/AR experiences.

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

### What Makes It Revolutionary?

- **🎵 AI-Powered Music Generation** - PyTorch neural networks create genre-specific compositions (70-105 seconds)
- **🥽 Immersive VR/AR Experiences** - 3D flight paths with spatial audio and multiple visual styles
- **📍 Personal Travel Logs** - Multi-waypoint journeys converted to musical stories
- **🤖 8 AI Music Genres** - Classical, Jazz, Electronic, Ambient, Rock, World, Cinematic, Lofi
- **🎨 Unique Visual Experiences** - Every generation creates different visuals, music, and animations
- **📊 Vector Embeddings Analytics** - 7 types of embeddings for similarity search and recommendations
- **⚡ Real-time Collaboration** - WebSocket-based collaborative editing
- **🌍 Real OpenFlights Data** - Authentic airport coordinates and flight routes


---

## ✨ Features

### 🎼 Music Generation
- **AI-Powered Composition**: PyTorch embeddings transform route characteristics into musical parameters
- **Musical Scales**: Major, Minor, Pentatonic, Blues, Dorian, Phrygian
- **Dynamic Tempo**: 70-140 BPM based on flight distance
- **Multi-track Output**: Melody, harmony, and bass tracks
- **MIDI Export**: Download compositions as MIDI files

### 🗺️ Interactive Visualization

- **Real-time Updates**: Live route tracking and composition progress
- **Airport Search**: Fast search across 3,000+ airports
- **Route Analytics**: Distance, duration, and complexity metrics

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

## 🚀 Advanced Features

### 1. 📍 Personal Travel Logs - User-Generated Datasets

Create and manage personal travel logs with multiple waypoints, then convert entire journeys into multi-segment musical compositions.

**Key Features:**
- ✅ Multi-waypoint journey tracking with timestamps
- ✅ Tag-based organization and filtering
- ✅ Convert entire travel logs to music


**Example Usage:**
```typescript
// Create a travel log
const travelLog = {
  title: "European Adventure 2025",
  description: "My summer trip across Europe",
  waypoints: [
    { airport_code: "JFK", timestamp: "2025-06-01T10:00:00", notes: "Departure" },
    { airport_code: "LHR", timestamp: "2025-06-01T22:00:00", notes: "London layover" },
    { airport_code: "CDG", timestamp: "2025-06-02T08:00:00", notes: "Paris" },
    { airport_code: "FCO", timestamp: "2025-06-05T14:00:00", notes: "Rome" }
  ],
  tags: ["vacation", "europe", "summer"]
};
```

**Access:** http://localhost:5173/travel-logs

### 2. 🤖 AI Genre Composer - PyTorch Neural Networks

Advanced AI models that generate genre-specific music using PyTorch with neural network embeddings and LSTM pattern generation.

**AI Models:**
- **GenreEmbeddingModel**: 3-layer feedforward (10→64→32 dimensions)
- **MusicPatternGenerator**: 2-layer LSTM (32→128→12 chromatic notes)

**8 AI Genres:**
| Genre | Tempo | Characteristics | Duration |
|-------|-------|----------------|----------|
| **Classical** | 60-120 BPM | Elegant, structured | 70-105s |
| **Jazz** | 100-180 BPM | Syncopated, improvisational | 70-105s |
| **Electronic** | 120-140 BPM | Modern, synthesized | 70-105s |
| **Ambient** | 60-90 BPM | Atmospheric, flowing | 70-105s |
| **Rock** | 110-140 BPM | Energetic, driving | 70-105s |
| **World** | 80-130 BPM | Diverse cultural influences | 70-105s |
| **Cinematic** | 70-110 BPM | Dramatic, emotional | 70-105s |
| **Lofi** | 70-95 BPM | Relaxed, nostalgic | 70-105s |

**Features:**
- ✅ AI-powered genre recommendations based on route characteristics
- ✅ Genre blending with adjustable ratios
- ✅ Real-time composition generation (70-105 second pieces)
- ✅ Musical scales: Major, Minor, Dorian, Pentatonic, Blues, and more
- ✅ Variable note durations and dynamic velocities

**Access:** http://localhost:5173/ai-composer

### 3. 🥽 VR/AR Immersive Experiences - 3D Flight Visualization

Complete VR/AR experience generation with 3D flight paths, camera animations, spatial audio, and multi-platform export.

**Visual Styles (Randomly Selected):**
1. **Default** - Classic blue theme with clean aesthetics
2. **Neon** - Cyberpunk cyan/green glow effects  
3. **Aurora** - Purple/pink multi-colored lighting
4. **Cosmic** - Orange theme with 5,000 animated stars

**Dynamic Elements:**
- ✅ Animated 3D plane with bobbing motion
- ✅ Particle trail system following the plane
- ✅ Progress markers every 10 points along the path
- ✅ Earth glow effect and auto-rotating camera
- ✅ Variable animation speed (0.8x - 1.2x) for uniqueness
- ✅ Synchronized AI music playback

**Camera Modes:**
- **Follow**: Camera follows behind aircraft
- **Orbit**: Camera orbits around flight path  
- **Cinematic**: Dynamic angles and movements

**Supported Platforms:**
- **WebXR** - Browser-based VR (Three.js, A-Frame, Babylon.js)
- **Oculus Quest** - Native VR headsets
- **HTC Vive** - PC VR systems
- **Apple ARKit** - iOS AR experiences
- **Google ARCore** - Android AR experiences
- **Unity** - Game engine integration

**Access:** http://localhost:5173/vr-experience

### 4. 📊 Vector Embeddings & Analytics - 7 Embedding Types

Comprehensive vector embeddings system with automatic syncing to DuckDB for analytics and similarity search.

**7 Embedding Types:**
| Endpoint | Dimensions | Purpose | Auto-Sync |
|----------|------------|---------|-----------|
| **Home Routes** | 96D | Main flight route music generation | ✅ |
| **AI Composer** | 128D | AI-generated music compositions | ✅ |
| **Wellness** | 48D | Therapeutic music (ocean, mountain, night) | ✅ |
| **Education** | 64D | Learning lessons (geography, graph theory) | ✅ |
| **AR/VR** | 80D | Immersive VR/AR flight experiences | ✅ |
| **VR Experience** | 64D | Legacy VR experiences | ✅ |
| **Travel Logs** | 32D | User travel log waypoints | ✅ |

**Analytics Features:**
- ✅ Automatic similarity search and recommendations
- ✅ Real-time performance monitoring
- ✅ CSV export for external analysis
- ✅ Comprehensive statistics and reports
- ✅ DuckDB columnar storage for efficiency

**Test All Embeddings:**
```bash
cd backend
python test_all_vector_embeddings.py
```

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
| **Three.js** | 3D Graphics | Latest |
| **@react-three/fiber** | React Three.js | Latest |
| **@react-three/drei** | Three.js Helpers | Latest |

### Backend
| Technology | Purpose | Version |
|-----------|---------|---------|
| **FastAPI** | Web Framework | Latest |
| **Python** | Language | 3.9+ |
| **SQLAlchemy** | ORM | Latest (async) |
| **Pydantic** | Validation | Latest |
| **PyTorch** | ML/AI Neural Networks | Latest |
| **NetworkX** | Graph Algorithms | Latest |
| **Mido** | MIDI Generation | Latest |
| **python-jose** | JWT Auth | Latest |
| **Redis** | Caching & Pub/Sub | redis-py |
| **DuckDB** | Analytics & Vector Storage | Latest |
| **NumPy** | Numerical Computing | Latest |

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
git clone https://github.com/aviralSri23455/Melodic-Airways-Transforming-Flight.git
cd Melodic-Airways-Transforming-Flight
```

#### 2. Backend Setup

##### Automated Setup

**Windows:**
```bash
cd backend\bat
setup.bat
```

**Linux/Mac:**
```bash
cd backend/bat
chmod +x setup.sh
./setup.sh
```

##### Manual Setup

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
# DATABASE_URL=mysql+asyncmy://user:password@localhost:3306/your_database_name
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
echo "VITE_API_BASE_URL=http://localhost:8000/api/v1"  .env.local


# Start development server
npm run dev

# Build for production
npm run build
```

**Frontend will run at**: `http://localhost:5173`

### Quick Test

Once both servers are running, test the system:

```bash
# Health check
curl http://localhost:8000/health

# Generate music for JFK to LAX route
curl "http://localhost:8000/api/v1/demo/complete-demo?origin=JFK&destination=LAX"

# Test AI Composer
curl http://localhost:8000/api/v1/ai/ai-genres/available

# Test VR Experience
curl "http://localhost:8000/api/v1/vr/vr-experiences/demo?origin=JFK&destination=LAX"

# Test Vector Embeddings
cd backend && python test_all_vector_embeddings.py
```

### Access New Features

- **🏠 Home**: http://localhost:5173/ (Original music generation)
- **📍 Travel Logs**: http://localhost:5173/travel-logs (Personal journey tracking)
- **🤖 AI Composer**: http://localhost:5173/ai-composer (PyTorch music generation)
- **🥽 VR Experience**: http://localhost:5173/vr-experience (3D immersive visualization)
- **📚 Education**: http://localhost:5173/education (Learning modules)
- **📊 API Docs**: http://localhost:8000/docs (Interactive API documentation)


---

## 📁 Project Structure

```
Melodic Airways/
│
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── api/                      # API Routes
│   │   │   ├── routes.py            # Core endpoints (auth, airports, routes)
│   │   │   ├── extended_routes.py   # Extended features (collections, datasets)
│   │   │   ├── analytics_routes.py  # Analytics endpoints
│   │   │   ├── analytics_showcase_routes.py # Analytics showcase
│   │   │   ├── redis_routes.py      # Redis caching endpoints
│   │   │   ├── openflights_routes.py # OpenFlights data endpoints
│   │   │   ├── demo_routes.py       # Demo and testing endpoints
│   │   │   ├── websocket_demo.py    # WebSocket collaboration
│   │   │   ├── travel_log_routes.py # Travel log endpoints
│   │   │   ├── ai_genre_routes.py   # AI composer endpoints
│   │   │   ├── vrar_routes.py       # VR/AR endpoints
│   │   │   ├── vrar_experience_routes.py # VR/AR experience
│   │   │   ├── education_routes.py  # Education endpoints
│   │   │   ├── wellness_routes.py   # Wellness endpoints
│   │   │   ├── premium_routes.py    # Premium features
│   │   │   ├── vector_routes.py     # Vector search endpoints
│   │   │   ├── duckdb_vector_routes.py # DuckDB vector operations
│   │   │   └── metrics_routes.py    # Performance metrics
│   │   │
│   │   ├── core/                     # Core Configuration
│   │   │   ├── config.py            # Settings and environment variables
│   │   │   └── security.py          # JWT auth and password hashing
│   │   │
│   │   ├── db/                       # Database
│   │   │   └── database.py          # SQLAlchemy setup
│   │   │
│   │   ├── models/                   # Database Models & Schemas
│   │   │   ├── models.py            # SQLAlchemy models
│   │   │   └── schemas.py           # Pydantic schemas
│   │   │
│   │   ├── middleware/               # Middleware
│   │   │   └── throughput_monitor.py # Performance monitoring
│   │   │
│   │   └── services/                 # Business Logic
│   │       ├── music_generator.py   # MIDI generation
│   │       ├── ai_genre_composer.py # PyTorch AI models
│   │       ├── genre_composer.py    # Genre composition
│   │       ├── travel_log_service.py # Travel log management
│   │       ├── vrar_experience_service.py # VR/AR generation
│   │       ├── vector_sync_helper.py # Vector embeddings sync
│   │       ├── realtime_vector_sync.py # Real-time vector sync
│   │       ├── graph_pathfinder.py  # Dijkstra's algorithm
│   │       ├── vector_service.py    # Similarity search
│   │       ├── route_embedding_service.py # Route embeddings
│   │       ├── route_embedding_service_duckdb.py # DuckDB embeddings
│   │       ├── websocket_manager.py # WebSocket connections
│   │       ├── redis_publisher.py   # Redis Pub/Sub
│   │       ├── redis_session_manager.py # Redis sessions
│   │       ├── cache.py             # Redis caching
│   │       ├── openflights_cache.py # OpenFlights caching
│   │       ├── duckdb_analytics.py  # Analytics service
│   │       ├── duckdb_sync_service.py # DuckDB sync
│   │       ├── faiss_duckdb_service.py # FAISS vector search
│   │       ├── dataset_manager.py   # Dataset management
│   │       ├── activity_service.py  # User activity tracking
│   │       └── realtime_generator.py # Real-time generation
│   │
│   ├── bat/                          # Setup Scripts
│   │   ├── setup.bat                # Windows setup (Redis + DuckDB)
│   │   └── setup.sh                 # Linux/Mac setup (Redis + DuckDB)
│   │
│   ├── scripts/                      # Utility Scripts
│   │   ├── etl_openflights.py       # Load OpenFlights data
│   │   ├── download_data.py         # Download datasets
│   │   ├── generate_route_embeddings_duckdb.py # DuckDB embeddings
│   │   ├── add_vector_columns_duckdb.py # Add vector columns
│   │   ├── init_duckdb.py           # Initialize DuckDB
│   │   └── redis_cleanup.py         # Redis cleanup
│   │
│   ├── sql/                          # SQL Scripts
│   │   ├── create_tables.sql        # Main schema
│   │   ├── add_travel_logs_table.sql # Travel logs migration
│   │   ├── add_vector_embeddings.sql # Vector embeddings
│   │   ├── enhanced_schema_migration.sql # Schema updates
│   │   └── ...                      # Other SQL scripts
│   │
│   ├── tests/                        # Test Suite
│   │   ├── test_db.py               # Database tests
│   │   ├── test_populate_openflights_auto.py # OpenFlights tests
│   │   ├── test_redis_cloud.py      # Redis tests
│   │   ├── test_skysql_connection.py # SkySQL tests
│   │   └── verify_openflights_data.py # Data verification
│   │
│   ├── duckdb_analytics/             # Analytics System
│   │   ├── analytics.py             # Analytics engine
│   │   ├── query.py                 # Query interface
│   │   ├── run_analytics.py         # Analytics runner
│   │   └── vector_embeddings.py     # Vector storage and analysis
│   │
│   ├── performance/                  # Performance Testing
│   │   ├── measure_throughput.py    # Throughput measurement
│   │   └── measure_throughput_optimized.py # Optimized tests
│   │
│   ├── main.py                       # Application entry point
│   ├── requirements.txt              # Python dependencies
│   ├── .env.example                  # Environment template
│   └── Dockerfile                    # Docker configuration
│
├── src/                              # React Frontend
│   ├── components/                   # React Components
│   │   ├── ui/                      # shadcn/ui components
│   │   ├── Hero.tsx                 # Landing hero section
│   │   ├── RouteSelector.tsx        # Airport search and selection
│   │   ├── AirportAutocomplete.tsx  # Airport autocomplete
│   │   ├── MusicPlayer.tsx          # Audio playback
│   │   ├── MusicControls.tsx        # Playback controls
│   │   ├── GlobalMap.tsx            # Mapbox visualization
│   │   ├── RouteVisualization.tsx   # Route display
│   │   ├── Analytics.tsx            # Analytics dashboard
│   │   ├── Features.tsx             # Feature showcase
│   │   ├── MusicDNA.tsx             # Composition details
│   │   └── Navigation.tsx           # Navigation menu
│   │
│   ├── hooks/                        # Custom React Hooks
│   │   ├── use-toast.ts             # Toast notifications
│   │   ├── use-mobile.tsx           # Mobile detection
│   │   └── useRouteVisualization.ts # Route visualization
│   │
│   ├── lib/                          # Utilities
│   │   ├── utils.ts                 # Helper functions
│   │   ├── audioPlayer.ts           # Audio player utilities
│   │   ├── midiExport.ts            # MIDI export
│   │   ├── api/                     # API clients
│   │   └── visualization/           # Visualization utilities
│   │
│   ├── pages/                        # Route Pages
│   │   ├── Index.tsx                # Home page
│   │   ├── TravelLogs.tsx           # Travel log management
│   │   ├── AIGenreComposer.tsx      # AI composition interface
│   │   ├── VRExperience.tsx         # VR experience with 3D viz
│   │   ├── VRAR.tsx                 # VR/AR page
│   │   ├── Education.tsx            # Learning modules
│   │   ├── Wellness.tsx             # Wellness features
│   │   ├── Premium.tsx              # Premium features
│   │   └── NotFound.tsx             # 404 page
│   │
│   ├── assets/                       # Static Assets
│   │   └── hero-globe.jpg           # Hero image
│   │
│   ├── App.tsx                       # App component
│   ├── main.tsx                      # Entry point
│   └── index.css                     # Global styles
│
├── docs/                             # Documentation
│   └── DUCKDB.md                    # DuckDB documentation
│
├── docker/                           # Docker files
│
├── public/                           # Public Assets
│
├── .env.example                      # Frontend environment template
├── package.json                      # Node dependencies
├── package-lock.json                 # npm lockfile
├── vite.config.ts                    # Vite configuration
├── tailwind.config.ts                # Tailwind configuration
├── tsconfig.json                     # TypeScript configuration
└── Github Overview.md                # This file
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

**Traditional Music Generation:**
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

**AI-Generated Music (New):**
```
Route: JFK → LAX (AI Analysis)
Distance: 3,983 km | Lat Range: 6.7° | Lon Range: 44.6° | Direction: W

AI Recommendation: Cinematic (confidence: 0.89)
Generated Composition:
├── Genre: Cinematic
├── Scale: Lydian
├── Tempo: 90 BPM
├── Duration: 79.75 seconds
├── Notes: 162 properly timed notes
├── Velocity Range: 50-110 (dramatic crescendo)
└── Output: Real-time playable composition + MIDI
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

## 🥽 VR/AR Experiences

### Immersive 3D Flight Visualization

Every VR experience is completely unique with randomized visual styles, animation speeds, and AI-generated music synchronized to the flight path.

### How It Works

1. **Route Selection**: Enter origin and destination airport codes
2. **AI Music Generation**: System analyzes route and generates appropriate music
3. **3D Path Creation**: Calculates great circle route with 200 3D points
4. **Visual Style Selection**: Randomly selects one of 4 visual themes
5. **Auto-Start**: Animation and music begin automatically after 500ms

### Visual Styles

#### 1. Default Theme
- Blue earth wireframe (#1e40af)
- Cyan flight path (#00aaff)
- Red animated plane
- Clean, professional aesthetic

#### 2. Neon Theme  
- Dark teal earth (#001a1a)
- Bright cyan path with glow effects
- Green plane with emission
- Cyberpunk aesthetic with low ambient lighting

#### 3. Aurora Theme
- Purple-tinted earth (#1a001a)
- Magenta flight path (#ff00ff)
- Pink plane with glow
- Multiple colored point lights (purple, cyan)
- Dreamy, ethereal atmosphere

#### 4. Cosmic Theme
- Dark orange earth (#1a0a00)
- Yellow/orange flight path
- Orange plane
- 5,000 animated background stars
- Space exploration vibe

### Dynamic Elements

- **Animated Plane**: 3D cone model with subtle bobbing motion
- **Particle Trail**: Dynamic particle system following the plane
- **Progress Markers**: Glowing spheres every 10 points along the path
- **Earth Glow**: Ambient glow effect around the planet
- **Auto-Rotation**: Camera orbits automatically during playback
- **Variable Speed**: Each experience has randomized animation speed (0.8x - 1.2x)

### Camera Modes

#### Follow Mode
- Camera follows behind the aircraft
- Stable and predictable movement
- Good for first-time users
- Maintains consistent viewing angle

#### Orbit Mode
- Camera orbits around the flight path
- Provides 360° views of the route
- Dynamic perspective changes
- Best for showcasing the complete route

#### Cinematic Mode
- Dynamic camera angles and movements
- Varying field of view and camera roll
- Dramatic presentation style
- Perfect for presentations and demos

### Experience Types

#### Immersive
- Full VR experience optimized for headsets
- Spatial audio positioning
- Interactive elements and hotspots
- Best for: VR headsets (Oculus, HTC Vive)

#### Cinematic
- Movie-like camera movements
- Dynamic angles and dramatic effects
- Optimized for viewing and presentations
- Best for: Demos, presentations, social sharing

#### Educational
- Interactive hotspots with information
- Learning-focused overlays
- Geographic and aviation education
- Best for: Schools, training programs

### Multi-Platform Export

The system generates experiences compatible with:

- **WebXR**: Browser-based VR using Three.js, A-Frame, Babylon.js
- **Oculus Quest**: Native VR headset experiences
- **HTC Vive**: PC VR system integration
- **Apple ARKit**: iOS AR experiences
- **Google ARCore**: Android AR experiences  
- **Unity**: Game engine integration for custom development
- **Unreal Engine**: High-fidelity experience development

### API Integration

```bash
# Create complete VR experience
POST /api/v1/vr/vr-experiences/create
{
  "origin_code": "JFK",
  "destination_code": "LAX", 
  "experience_type": "cinematic",
  "camera_mode": "orbit"
}

# Get 3D flight path
GET /api/v1/vr/vr-experiences/flight-path?origin_code=JFK&destination_code=LAX&num_points=200

# Export for Unity
GET /api/v1/vr/vr-experiences/export/unity/{experience_id}

# Export for WebXR
GET /api/v1/vr/vr-experiences/export/webxr/{experience_id}
```

### Performance

- **3D Rendering**: Smooth 60fps with Three.js
- **Path Generation**: ~50-100ms for 200 points
- **Music Integration**: Synchronized playback with Web Audio API
- **Memory Usage**: Optimized for browser performance
- **Mobile Support**: Responsive design for tablets and phones

---

## 📊 Vector Embeddings & Analytics

### Comprehensive Analytics System

The platform automatically generates and stores vector embeddings for all user interactions, enabling powerful analytics and recommendation systems.

### 7 Embedding Types

#### 1. Home Routes (96D)
**Source**: Main music generation endpoint
**Features**: Origin/destination, distance, music style, tempo, note count, duration
```python
vector_sync.sync_home_route(
    origin="JFK", destination="LAX", distance_km=3983,
    music_style="major", tempo=120, note_count=150, duration=180
)
```

#### 2. AI Composer (128D)  
**Source**: PyTorch AI music generation
**Features**: Genre, tempo, complexity, duration, neural network outputs
```python
vector_sync.sync_ai_composer(
    composition_id="ai_jazz_123", genre="jazz", 
    tempo=140, complexity=0.8, duration=180
)
```

#### 3. Wellness (48D)
**Source**: Therapeutic music generation  
**Features**: Theme (ocean/mountain/night), calm level, binaural frequency
```python
vector_sync.sync_wellness_composition(
    theme="ocean", calm_level=50, duration=300,
    note_count=120, binaural_frequency=4.0
)
```

#### 4. Education (64D)
**Source**: Learning modules
**Features**: Lesson type, difficulty, topic, interaction count
```python
vector_sync.sync_education_lesson(
    lesson_type="geography", difficulty="beginner",
    topic="Distance and pitch correlation", interaction_count=5
)
```

#### 5. AR/VR (80D)
**Source**: VR/AR experience generation
**Features**: Session type, waypoints, spatial audio, quality, duration
```python
vector_sync.sync_arvr_session(
    session_type="vr_flight", origin="JFK", destination="LAX",
    waypoint_count=100, spatial_audio=True, quality="high"
)
```

#### 6. VR Experience (64D)
**Source**: Legacy VR experiences
**Features**: Experience metadata and performance metrics

#### 7. Travel Logs (32D)
**Source**: User travel log creation
**Features**: Title, waypoint count, travel date, user metadata
```python
vector_sync.sync_travel_log(
    log_id=1, title="European Adventure",
    waypoint_count=15, travel_date=datetime(2024, 6, 15)
)
```

### Analytics Capabilities

#### Similarity Search
```python
# Find similar routes
similar_routes = vector_store.find_similar_home_routes(query_embedding, k=5)

# Find similar wellness sessions  
similar_wellness = vector_store.find_similar_wellness(user_embedding, k=5)

# Find similar VR experiences
similar_vr = vector_store.find_similar_arvr(user_vr_embedding, k=5)
```

#### Statistics & Reports
```bash
# Generate comprehensive report
cd backend/duckdb_analytics
python vector_embeddings.py

# Test all embedding types
cd backend  
python test_all_vector_embeddings.py
```

#### Data Export
```python
# Export all embeddings to CSV
vector_store.export_embeddings_to_csv("./exports")
# Creates: home_route_embeddings.csv, ai_composer_embeddings.csv, etc.
```

### Use Cases

1. **Recommendation System**: Find similar routes based on user preferences
2. **Personalization**: Recommend music genres based on past compositions  
3. **Educational Pathways**: Suggest next lessons based on completed ones
4. **VR Matching**: Match users with similar VR experience preferences
5. **Analytics**: Track user behavior patterns and system performance

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

#### Travel Logs

```bash
# Create travel log
POST /api/v1/user/travel-logs
Body: {
  "title": "My Trip",
  "description": "Amazing journey",
  "waypoints": [
    {"airport_code": "JFK", "timestamp": "2025-01-15T10:00:00"},
    {"airport_code": "LAX", "timestamp": "2025-01-15T18:00:00"}
  ],
  "tags": ["vacation", "usa"],
  "travel_date": "2025-01-15"
}

# Get my travel logs
GET /api/v1/user/travel-logs/my?skip=0&limit=20

# Get public travel logs
GET /api/v1/user/travel-logs/public

# Convert to music
POST /api/v1/user/travel-logs/{id}/convert-to-music?music_style=ambient

# Share publicly
PATCH /api/v1/user/travel-logs/{id}/share
```

#### AI Genre Composer

```bash
# Get available genres
GET /api/v1/ai/ai-genres/available

# Generate AI composition
POST /api/v1/ai/ai-genres/compose
Body: {
  "genre": "jazz",
  "route_features": {
    "distance": 5000,
    "latitude_range": 40,
    "longitude_range": 80,
    "direction": "E"
  },
  "duration": 30
}

# Get AI recommendations
POST /api/v1/ai/ai-genres/recommendations
Body: {
  "distance": 5000,
  "latitude_range": 30,
  "longitude_range": 50,
  "direction": "W"
}

# Blend genres
POST /api/v1/ai/ai-genres/blend
Body: {
  "primary_genre": "classical",
  "secondary_genre": "electronic",
  "blend_ratio": 0.3
}

# Quick demo
GET /api/v1/ai/ai-genres/demo/{genre}?distance=5000&duration=30

# Model information
GET /api/v1/ai/ai-genres/model-info
```

#### VR/AR Experiences

```bash
# Create VR experience
POST /api/v1/vr/vr-experiences/create
Body: {
  "origin_code": "JFK",
  "destination_code": "LAX",
  "experience_type": "cinematic",
  "camera_mode": "orbit",
  "music_composition": {
    "tempo": 120,
    "duration": 60
  }
}

# Get 3D flight path
GET /api/v1/vr/vr-experiences/flight-path?origin_code=JFK&destination_code=LAX&num_points=200

# Generate camera animation
POST /api/v1/vr/vr-experiences/camera-animation
Body: {
  "flight_path": [...],
  "camera_mode": "orbit",
  "duration": 60
}

# Generate spatial audio
POST /api/v1/vr/vr-experiences/spatial-audio
Body: {
  "flight_path": [...],
  "music_composition": {...}
}

# Export for Unity
GET /api/v1/vr/vr-experiences/export/unity/{experience_id}

# Export for WebXR
GET /api/v1/vr/vr-experiences/export/webxr/{experience_id}

# Quick demo
GET /api/v1/vr/vr-experiences/demo?origin=JFK&destination=LAX

# VR system info
GET /api/v1/vr/vr-experiences/info
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
DATABASE_URL=mysql+asyncmy://your_user:your_password@localhost:3306/your_database_name
DB_HOST=localhost
DB_PORT=6379
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_NAME=aero_melody_db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# ============================================
# REDIS CLOUD CONFIGURATION
# ============================================
REDIS_HOST=your-redis-host.redis-cloud.com
REDIS_PORT=6379
REDIS_USERNAME=default
REDIS_PASSWORD=your_redis_cloud_password
REDIS_URL=redis://default:your_password@your-redis-host.redis-cloud.com:16441
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
      - DATABASE_URL=mysql+asyncmy://root:your_password@db:3306/your_database_name
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
      - MYSQL_ROOT_PASSWORD=your_secure_password
      - MYSQL_DATABASE=aero_melody_db
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
VITE_JWT_TOKEN_KEY=melody_aero_token
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
git clone https://github.com/yourusername/Melodic-Airways-Transforming-Flight.git
cd Melodic-Airways-Transforming-Flight/backend

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
sudo nano /etc/systemd/system/Melodic-Airways-Transforming-Flight.service
```

**Melodic-Airways-Transforming-Flight.service**:

```ini
[Unit]
Description=Aero Melody Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/Melodic-Airways-Transforming-Flight/backend
Environment="PATH=/var/www/Melodic-Airways-Transforming-Flight/backend/venv/bin"
ExecStart=/var/www/Melodic-Airways-Transforming-Flight/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Start service
sudo systemctl daemon-reload
sudo systemctl start Melodic-Airways-Transforming-Flight
sudo systemctl enable Melodic-Airways-Transforming-Flight
sudo systemctl status Melodic-Airways-Transforming-Flight
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
mysqldump -u root -p melody_aero > backup_$DATE.sql
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

#### Travel Logs
- [ ] Create new travel log
- [ ] Add waypoints
- [ ] Convert to music
- [ ] Share publicly
- [ ] Tag filtering

#### AI Composer
- [ ] Genre selection works
- [ ] AI recommendations display
- [ ] Composition generation works
- [ ] Genre blending works
- [ ] Playback works

#### VR Experience
- [ ] 3D visualization loads
- [ ] Route selection works
- [ ] Animation plays automatically
- [ ] Music synchronizes
- [ ] Controls work (play/pause)
- [ ] Visual styles randomize

#### Education
- [ ] Lessons display
- [ ] Quizzes interactive
- [ ] "Try Interactive Lab" works
- [ ] Lab generates music
- [ ] Tabs work

#### Wellness
- [ ] Theme selection works
- [ ] Calm level slider works
- [ ] Generation works
- [ ] Playback works
- [ ] All 3 themes tested

#### Backend APIs
- [ ] /health endpoint responds
- [ ] /api/v1/demo/complete-demo works
- [ ] /api/v1/ai/ai-genres/* works
- [ ] /api/v1/vr/vr-experiences/* works
- [ ] /api/v1/user/travel-logs/* works
- [ ] Swagger UI accessible at /docs

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

# Test AI Composer
curl http://localhost:8000/api/v1/ai/ai-genres/available
curl -X POST http://localhost:8000/api/v1/ai/ai-genres/compose \
  -H "Content-Type: application/json" \
  -d '{"genre":"jazz","route_features":{"distance":5000},"duration":30}'

# Test VR Experience
curl "http://localhost:8000/api/v1/vr/vr-experiences/demo?origin=JFK&destination=LAX"
curl "http://localhost:8000/api/v1/vr/vr-experiences/flight-path?origin_code=JFK&destination_code=LAX"

# Test Travel Logs (public endpoint)
curl http://localhost:8000/api/v1/user/travel-logs/public

# Test Vector Embeddings
cd backend
python test_all_vector_embeddings.py
python test_duckdb_sync.py

# Test cache
curl "http://localhost:8000/api/v1/redis/test/save-music"

# Test health
curl "http://localhost:8000/health"
```

#### New Features Testing

```bash
# Test all new features
cd backend

# Test performance
python tests/test_throughput.py

# Test vector embeddings system
python test_all_vector_embeddings.py

# Generate analytics report
cd duckdb_analytics
python vector_embeddings.py
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
DATABASE_URL=mysql+asyncmy://user:password@localhost:3306/melody_aero

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

#### AI Composer Issues

**Problem**: "Composition too short" warnings

```bash
# This was fixed - AI now generates 70-105 second compositions
# Test the fix:
curl -X POST http://localhost:8000/api/v1/ai/ai-genres/compose \
  -H "Content-Type: application/json" \
  -d '{"genre":"jazz","duration":30}'

# Should return composition with proper duration
```

**Problem**: PyTorch not found

```bash
# Install PyTorch
pip install torch torchvision torchaudio

# Verify installation
python -c "import torch; print(torch.__version__)"
```

#### VR Experience Issues

**Problem**: 3D visualization not loading

```bash
# Check Three.js dependencies
npm list @react-three/fiber @react-three/drei three

# Reinstall if needed
npm install @react-three/fiber @react-three/drei three

# Check browser WebGL support
# Visit: https://get.webgl.org/
```

**Problem**: Animation not starting automatically

```bash
# Check browser console for errors
# Ensure route data is loaded first
# Try manual play button if auto-start fails
```

#### Travel Logs Issues

**Problem**: "travel_logs table not found"

```bash
# Run database migration
mysql -u root -p melody_aero < backend/migrations/add_travel_logs_table_safe.sql

# Or create manually:
mysql -u root -p Melodic-Airways-Transforming-Flight -e "
CREATE TABLE IF NOT EXISTS travel_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    waypoints JSON NOT NULL,
    travel_date DATETIME,
    tags JSON,
    is_public BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);"
```

#### Vector Embeddings Issues

**Problem**: No embeddings showing up

```bash
# Test vector sync
cd backend
python test_all_vector_embeddings.py

# Check DuckDB file
ls -lh data/analytics.duckdb

# Generate report
cd duckdb_analytics
python vector_embeddings.py
```

**Problem**: DuckDB errors

```bash
# DuckDB is optional - application works without it
# To install:
pip install duckdb

# Or ignore warnings - analytics features will be disabled
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
mysql -u root -p melody_aero -e "SELECT COUNT(*) FROM airports;"
mysql -u root -p melody_aero -e "SELECT COUNT(*) FROM routes;"
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
sudo journalctl -u Melodic-Airways-Transforming-Flight -f
```

#### Database Debugging

```bash
# Connect to database
mysql -u root -p melody_aero

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
4. Check [GitHub Issues](https://github.com/yourusername/Melodic-Airways-Transforming-Flight/issues)
5. Contact support or open a new issue


---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help make Aero Melody better.

### Getting Started

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   git clone https://github.com/your-username/Melodic-Airways-Transforming-Flight.git
   cd Melodic-Airways-Transforming-Flight
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

- [ ] Mobile app (React Native /flutter)
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
- **Redis Cloud**: Caching and real-time features
- **MariaDB/MySQL**: Primary database
- **DuckDB**: Analytics and vector embeddings

---

## 👨‍💻 Built With ❤️

**Created by Aviral Shani & KarinaMythri**

*Transforming flight data into musical experiences*

### Contributors

Thank you to all contributors who have helped make ! Melodic Airways 


For their dedication and contributions to the Melodic Airways .

---


### Documentation

- **API Docs**: http://localhost:8000/docs

- Aviral
- Shani
- Karina
- Mythri

⭐ Star us on GitHub if you find this project interesting!



---


## 📞 Support

- **Documentation**: See the `backend/docs/` folder for detailed guides
- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **ReDoc**: http://localhost:8000/redoc (Alternative API documentation)
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas

### Quick Links

- [Vector Embeddings Guide](backend/docs/VECTOR_EMBEDDING_GUIDE.md)

---

## 🎉 Get Started Now

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Melodic-Airways-Transforming-Flight.git
cd Melodic-Airways-Transforming-Flight

# 2. Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # Edit with your credentials
python scripts/etl_openflights.py
python main.py

# 3. Setup frontend (in new terminal)
cd ..
npm install
cp .env.example .env  # Edit with your API URL
npm run dev

# 4. Optional: Setup vector embeddings for similarity search
cd backend
python scripts/generate_route_embeddings_duckdb.py
```

**Visit http://localhost:5173 and start creating music from flight routes!** 🎵✈️

### First Steps

1. **Try the Home Page**: Generate music from JFK to LAX
2. **Explore Travel Logs**: Create a multi-waypoint journey
3. **Test AI Composer**: Let AI generate genre-specific music
4. **Experience VR**: Watch 3D flight paths with synchronized music
5. **Learn**: Try the Education modules
6. **Relax**: Generate wellness soundscapes

---

# Screen Shot 

![Alt text](./Screen%20Shot/Duck%20DB%20SS-1.png)
![Alt text](./Screen%20Shot/Duck%20DB%20SS-2.png)
![Alt text](./Screen%20Shot/Duck%20DB%20SS-3.png)
![Alt text](./Screen%20Shot/front%20end%20ss-1.png)
![Alt text](./Screen%20Shot/front%20end%20ss-1.png)
![Alt text](./Screen%20Shot/front%20end%20ss-2.png)
![Alt text](./Screen%20Shot/front%20end%20ss-3.png)
![Alt text](./Screen%20Shot/front%20end%20ss-4.png)
![Alt text](./Screen%20Shot/VECTOR%20EMEDING.png)



**Built with  by 
Aviral 
Shani 
Karina
Mythri
**

⭐ Star us on GitHub if you find this project interesting!

