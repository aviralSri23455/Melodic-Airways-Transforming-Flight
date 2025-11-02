# Aero Melody - API Guide for Frontend Integration

Complete API reference for frontend developers integrating with the Aero Melody backend.

---

## Base URL

```
Development: http://localhost:8000/api/v1
Production: https://your-domain.com/api/v1
```

---



## Airport Management

### Search Airports
```http
GET /api/v1/airports/search?query=New York&country=USA&limit=20
```

**Response:**
```json
[
  {
    "id": 1,
    "name": "John F Kennedy International Airport",
    "city": "New York",
    "country": "United States",
    "iata_code": "JFK",
    "latitude": 40.6413,
    "longitude": -73.7781
  }
]
```

### Get Airport by Code
```http
GET /api/v1/airports/JFK
```

---

## Music Generation

### Generate Music (Quick Demo)
```http
GET /api/v1/demo/complete-demo?origin=JFK&destination=LAX
```

**Response:**
```json
{
  "composition": {
    "composition_id": 1730000000,
    "origin": "JFK",
    "destination": "LAX",
    "tempo": 120,
    "duration_seconds": 30,
    "note_count": 60,
    "key": "C",
    "scale": "major",
    "midi_file": "route_JFK_LAX_1730000000.mid"
  },
  "analytics": {
    "melodic_complexity": 0.75,
    "harmonic_richness": 0.8,
    "tempo_variation": 0.1
  }
}
```

### Generate with Custom Parameters
```http
POST /api/v1/compositions/generate
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

---

## Vector Similarity Search

### Find Similar Routes
```http
GET /api/v1/vectors/similar-routes?origin=JFK&destination=LAX&limit=10
```

**Response:**
```json
[
  {
    "route_id": 12345,
    "origin_code": "JFK",
    "dest_code": "SFO",
    "distance_km": 4139.0,
    "similarity_score": 0.95
  }
]
```

### Find Routes by Genre
```http
GET /api/v1/vectors/routes-by-genre?genre=ambient&limit=20
```

**Supported Genres:**
- `classical` - Complex, formal routes
- `jazz` - Improvisational, varied routes
- `electronic` - Repetitive, rhythmic routes
- `ambient` - Long, calm, transoceanic routes
- `pop` - Popular, direct routes

### Get Route Complexity
```http
GET /api/v1/vectors/route/{route_id}/complexity
```

**Response:**
```json
{
  "harmonic_complexity": 0.75,
  "rhythmic_complexity": 0.60,
  "melodic_complexity": 0.82,
  "overall_complexity": 0.72
}
```

### Get Statistics
```http
GET /api/v1/vectors/statistics
```

---

## Composition Management

### Get User Compositions
```http
GET /api/v1/compositions?limit=50&offset=0
```

### Get Specific Composition
```http
GET /api/v1/compositions/1
```

### Get Public Compositions
```http
GET /api/v1/public/compositions?genre=classical&limit=20
```

### Delete Composition
```http
DELETE /api/v1/compositions/1
```

---

## Route Analysis

### Get All Routes
```http
GET /api/v1/routes?limit=100&offset=0
```

### Find Similar Routes
```http
GET /api/v1/similar?origin_code=JFK&destination_code=LAX&limit=5
```

### Get Route Analytics
```http
GET /api/v1/analytics/1
```

---

## Dataset Management

### Create Dataset
```http
POST /api/v1/datasets
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

### Get User Datasets
```http
GET /api/v1/datasets
```

---

## Collection Management

### Create Collection
```http
POST /api/v1/collections
Content-Type: application/json

{
  "name": "My Favorites",
  "description": "Favorite compositions",
  "tags": ["classical", "travel"]
}
```

### Add Composition to Collection
```http
POST /api/v1/collections/1/compositions/1
```

---

## Real-time Collaboration

### Create Collaboration Session
```http
POST /api/v1/collaborations/sessions
Content-Type: application/json

{
  "composition_id": 1
}
```

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/collaborate/1/1');

// Send state updates
ws.send(JSON.stringify({
  type: 'state_update',
  state: {
    tempo: 120,
    notes: [...]
  }
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

---

## Education

### Get Available Lessons
```http
GET /api/v1/education/lessons
```

### Start a Lesson
```http
POST /api/v1/education/lessons/{id}/start
```

### Get Graph Visualization
```http
GET /api/v1/education/graph-visualization/{origin}/{destination}
```

---

## Wellness

### Generate Calming Soundscape
```http
POST /api/v1/wellness/generate-wellness
Content-Type: application/json

{
  "theme": "ocean",
  "calm_level": 80,
  "duration_minutes": 5
}
```

**Themes:**
- `ocean` - Gentle wave-like melodies
- `mountain` - Peaceful ambient soundscapes
- `night` - Soothing overnight compositions

### Get Wellness Themes
```http
GET /api/v1/wellness/wellness-themes
```

---

## VR/AR

### Create VR Session
```http
POST /api/v1/vr-ar/create-session
Content-Type: application/json

{
  "origin": "JFK",
  "destination": "CDG",
  "enable_spatial_audio": true,
  "quality": "high"
}
```

### Get Supported Airports
```http
GET /api/v1/vr-ar/supported-airports
```

### Get VR Capabilities
```http
GET /api/v1/vr-ar/vr-capabilities
```

---

## Activity Feed

### Get User Activities
```http
GET /api/v1/activities?limit=50&activity_type=composition_created
```

### Get Recent Activities
```http
GET /api/v1/activities/recent?minutes=60&limit=100
```

---

## Error Handling

All endpoints return standard HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 500 | Internal Server Error |

**Error Response Format:**
```json
{
  "detail": "Error description"
}
```

---

## Frontend Integration Examples

### Generate Music Component
```javascript
import { useState } from 'react';

export const GenerateMusic = () => {
  const [loading, setLoading] = useState(false);

  const generateMusic = async (origin, destination) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/demo/complete-demo?origin=${origin}&destination=${destination}`);
      const data = await response.json();
      return data;
    } finally {
      setLoading(false);
    }
  };

  return { generateMusic, loading };
};
```



### WebSocket Hook
```javascript
import { useEffect, useState } from 'react';

export const useCollaboration = (sessionId, userId) => {
  const [ws, setWs] = useState(null);
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const websocket = new WebSocket(
      `ws://localhost:8000/api/v1/ws/collaborate/${sessionId}/${userId}`
    );

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessages(prev => [...prev, data]);
    };

    setWs(websocket);

    return () => websocket.close();
  }, [sessionId, userId]);

  const sendUpdate = (state) => {
    if (ws) {
      ws.send(JSON.stringify({
        type: 'state_update',
        state
      }));
    }
  };

  return { messages, sendUpdate };
};
```

---

## Interactive Documentation

When the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

---

## Performance Considerations

- **Caching**: Redis integration for frequently accessed data
- **Vector Search**: FAISS for fast similarity matching (~1ms)
- **Batch Processing**: Efficient handling of large datasets
- **Connection Pooling**: Optimized database connections

---

## Support

For questions or issues:
1. Check the API documentation at `/docs`
2. Review the [Setup Guide](./SETUP.md)
3. Test endpoints with Swagger UI
4. Check error responses for details
