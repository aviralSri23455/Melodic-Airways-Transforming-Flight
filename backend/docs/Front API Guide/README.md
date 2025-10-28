# Frontend API Integration Guide

This directory contains comprehensive guides for frontend developers integrating with the Aero Melody backend.

## 📄 Available Documents

### [FRONTEND_API_GUIDE.md](FRONTEND_API_GUIDE.md)
**Complete Frontend Integration Guide**
- All API endpoints with examples
- Request/response formats
- Authentication workflow
- Error handling
- React/Vue/TypeScript code examples
- WebSocket integration
- Real-time features

## 🚀 Quick Start for Frontend Developers

1. **Read the Frontend API Guide**: [FRONTEND_API_GUIDE.md](FRONTEND_API_GUIDE.md)
2. **Test APIs**: Visit http://localhost:8000/docs when server is running
3. **Authentication**: See login/register examples in the guide
4. **Music Generation**: Follow the workflow examples
5. **Real-time Features**: Implement WebSocket connections

## 🎯 Key Integration Points

### Authentication
- JWT-based authentication
- Token management
- User registration and login

### Music Generation
- Route selection
- Music parameter configuration
- MIDI file download

### Real-time Collaboration
- WebSocket connections
- Live composition editing
- Participant tracking

### Data Management
- Dataset creation
- Collection organization
- Activity feeds

## 🔗 Related Documentation

- **Backend Overview**: See main [README.md](../README.md)
- **API Reference**: See [../api/README.md](../api/README.md)
- **Database Schema**: See [../database/README.md](../database/README.md)

## 📊 Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: MariaDB (FREE version)
- **Real-time**: WebSocket + Redis
- **Music**: PyTorch + NetworkX + Mido
- **Auth**: JWT tokens

## 🎵 Music Generation Workflow

```
Frontend Request
    ↓
POST /api/v1/generate-midi
    ↓
Backend Processing
├─ NetworkX (Dijkstra pathfinding)
├─ PyTorch (embeddings)
├─ Mido (MIDI generation)
└─ Database (storage)
    ↓
MIDI File + Analytics
    ↓
Frontend Download/Play
```

## 💡 Example Integration

```javascript
// Simple music generation example
const generateMusic = async (origin, destination) => {
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
      tempo: 120,
      duration_minutes: 3
    })
  });
  return await response.json();
};
```

## 🤝 Support

For questions or issues:
1. Check [FRONTEND_API_GUIDE.md](FRONTEND_API_GUIDE.md)
2. Test at http://localhost:8000/docs
3. Review code examples in the guide
4. Check error handling section
