# Aero Melody API Documentation for Frontend Team

## Overview
Aero Melody is a comprehensive system that transforms global flight routes into musical compositions. The backend provides a complete REST API with real-time collaboration features, vector similarity search, and music generation capabilities.

## Tech Stack
- **Backend**: FastAPI (Python) with async support
- **Database**: MariaDB (FREE version - no paid extensions required)
- **Authentication**: JWT tokens with secure session management
- **Real-time**: WebSocket support for live collaboration
- **Music Generation**: PyTorch embeddings + Mido for MIDI files
- **Pathfinding**: NetworkX with Dijkstra's algorithm
- **Analytics**: DuckDB for similarity analysis and performance metrics

## Getting Started

### 1. Environment Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Database Setup
```bash
# Start MariaDB with Docker (FREE version)
docker-compose up -d mariadb

# Load OpenFlights data
python scripts/etl_openflights.py
```

### 3. Start Backend Server
```bash
python main.py
# Server runs on http://localhost:8000
# API docs: http://localhost:8000/docs
```

## API Endpoints

### Authentication

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "string",
  "email": "user@example.com",
  "password": "string"
}
```

#### Login User
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

Response:
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Get Current User
```http
GET /api/v1/auth/me
Authorization: Bearer <token>
```

### Airport Management

#### Search Airports
```http
GET /api/v1/airports/search?query=New York&country=USA&limit=20
Authorization: Bearer <token>
```

#### Get Airport by IATA Code
```http
GET /api/v1/airports/JFK
```

### Music Generation

#### Generate Music from Route
```http
POST /api/v1/generate-midi
Authorization: Bearer <token>
Content-Type: application/json

{
  "origin_code": "JFK",
  "destination_code": "LAX",
  "music_style": "classical",
  "scale": "major",
  "key": "C",
  "tempo": 120,
  "duration_minutes": 3
}
```

Response:
```json
{
  "composition_id": 1,
  "route_id": 1,
  "midi_file_url": "/api/download/1",
  "analytics": {
    "melodic_complexity": 0.75,
    "harmonic_richness": 0.8,
    "tempo_variation": 0.1,
    "pitch_range": 24,
    "note_density": 2.5
  },
  "message": "Music composition generated successfully"
}
```

### Composition Management

#### Get User Compositions
```http
GET /api/v1/compositions?limit=50&offset=0
Authorization: Bearer <token>
```

#### Get Specific Composition
```http
GET /api/v1/compositions/1
Authorization: Bearer <token>
```

#### Get Public Compositions
```http
GET /api/v1/public/compositions?genre=classical&limit=20
# No auth required for public compositions
```

#### Delete Composition
```http
DELETE /api/v1/compositions/1
Authorization: Bearer <token>
```

### Route Analysis

#### Get All Routes
```http
GET /api/v1/routes?limit=100&offset=0
Authorization: Bearer <token>
```

#### Find Similar Routes
```http
GET /api/v1/similar?origin_code=JFK&destination_code=LAX&limit=5
Authorization: Bearer <token>
```

#### Get Route Analytics
```http
GET /api/v1/analytics/1
Authorization: Bearer <token>
```

### Vector Similarity Search

#### Search Similar Compositions
```http
POST /api/v1/search/similar
Authorization: Bearer <token>
Content-Type: application/json

{
  "composition_id": 1,
  "limit": 10,
  "genre_filter": "jazz"
}
```

#### Search by Vector Parameters
```http
POST /api/v1/search/by-vector
Authorization: Bearer <token>
Content-Type: application/json

{
  "tempo": 120,
  "pitch": 60.5,
  "harmony": 0.8,
  "complexity": 0.7,
  "genre": "classical",
  "limit": 10
}
```

### Dataset Management

#### Create Dataset
```http
POST /api/v1/datasets
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "My Travel Routes",
  "route_data": {
    "routes": [
      {
        "origin": "JFK",
        "destination": "LAX",
        "distance": 2475
      }
    ]
  },
  "metadata": {
    "description": "Personal flight routes"
  }
}
```

#### Get User Datasets
```http
GET /api/v1/datasets
Authorization: Bearer <token>
```

### Collection Management

#### Create Collection
```http
POST /api/v1/collections
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "My Favorites",
  "description": "Favorite compositions",
  "tags": ["classical", "travel"]
}
```

#### Add Composition to Collection
```http
POST /api/v1/collections/1/compositions/1
Authorization: Bearer <token>
```

### Real-time Collaboration

#### Create Collaboration Session
```http
POST /api/v1/collaborations/sessions
Authorization: Bearer <token>
Content-Type: application/json

{
  "composition_id": 1
}
```

#### WebSocket Connection
```javascript
// Connect to collaboration session
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/collaborate/1/1');

// Send state updates
ws.send(JSON.stringify({
  type: 'state_update',
  state: {
    tempo: 120,
    notes: [...]
  }
}));
```

### Activity Feed

#### Get User Activities
```http
GET /api/v1/activities?limit=50&activity_type=composition_created
Authorization: Bearer <token>
```

#### Get Recent Activities
```http
GET /api/v1/activities/recent?minutes=60&limit=100
Authorization: Bearer <token>
```

## Response Models

### AirportSearchResponse
```json
{
  "id": 1,
  "name": "John F Kennedy International Airport",
  "city": "New York",
  "country": "United States",
  "iata_code": "JFK",
  "latitude": 40.6413,
  "longitude": -73.7781
}
```

### CompositionInfo
```json
{
  "id": 1,
  "route": {
    "id": 1,
    "origin_airport": {...},
    "destination_airport": {...},
    "distance_km": 2475.0,
    "duration_min": 185
  },
  "tempo": 120,
  "pitch": 60.5,
  "harmony": 0.8,
  "midi_path": "/path/to/file.mid",
  "complexity_score": 0.75,
  "harmonic_richness": 0.8,
  "duration_seconds": 180,
  "unique_notes": 45,
  "musical_key": "C",
  "scale": "major",
  "created_at": "2025-01-01T00:00:00"
}
```

## Error Handling

All endpoints return standard HTTP status codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `500`: Internal Server Error

Error response format:
```json
{
  "detail": "Error description"
}
```

## Frontend Integration Examples

### React Hook for API Calls
```javascript
// useAuth.js
import { useState } from 'react';

export const useAuth = () => {
  const [token, setToken] = useState(localStorage.getItem('token'));

  const login = async (username, password) => {
    const response = await fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const data = await response.json();
    if (response.ok) {
      setToken(data.access_token);
      localStorage.setItem('token', data.access_token);
    }
    return data;
  };

  return { token, login };
};
```

### Generate Music Component
```javascript
// GenerateMusic.js
import { useState } from 'react';

export const GenerateMusic = () => {
  const [loading, setLoading] = useState(false);

  const generateMusic = async (origin, destination) => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/generate-midi', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          origin_code: origin,
          destination_code: destination,
          music_style: 'classical',
          scale: 'major',
          key: 'C',
          tempo: 120,
          duration_minutes: 3
        })
      });
      const data = await response.json();
      return data;
    } finally {
      setLoading(false);
    }
  };

  return { generateMusic, loading };
};
```

## Database Schema

The system uses MariaDB with the following key tables:
- `airports`: OpenFlights airport data (3000+ airports)
- `routes`: Flight routes with distances and durations (67000+ routes)
- `music_compositions`: Generated music metadata
- `users`: User accounts and authentication
- `user_datasets`: Personal route collections
- `collaboration_sessions`: Real-time collaboration sessions

## Real-time Features

- **WebSocket collaboration**: Multiple users can edit compositions simultaneously
- **Activity feeds**: Real-time updates on user activities
- **Live composition sharing**: Share compositions with other users in real-time

## Performance Considerations

- **Caching**: Redis integration for frequently accessed data
- **Vector search**: Cosine similarity for music recommendations
- **Batch processing**: Efficient handling of large datasets
- **Connection pooling**: Optimized database connections

## Deployment

The system is containerized with Docker and includes:
- MariaDB with standard features
- Redis for caching
- WebSocket support for real-time features
- SSL/TLS configuration ready for production

## Support

For questions or issues:
1. Check the API documentation at `/docs`
2. Review the database schema in `sql/` directory
3. Check the service implementations in `app/services/`
4. Review the comprehensive test suite in `tests/`
