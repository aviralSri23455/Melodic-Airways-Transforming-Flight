"""
VR/AR immersive experience routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import math

logger = logging.getLogger(__name__)

router = APIRouter()


class FlightPath3D(BaseModel):
    origin: str
    destination: str
    waypoints: List[Dict[str, float]]
    duration_seconds: float
    distance_km: float


class VRSessionRequest(BaseModel):
    origin: str
    destination: str
    enable_spatial_audio: bool = True
    quality: str = "high"  # low, medium, high, ultra


class VRSessionResponse(BaseModel):
    session_id: str
    flight_path: FlightPath3D
    audio_zones: List[Dict]
    recommended_duration: float
    vr_settings: Dict


class SpatialAudioZone(BaseModel):
    position: Dict[str, float]
    sound_type: str  # ambient, engine, wind, music
    volume: float
    frequency: float


def calculate_3d_waypoints(origin_lat: float, origin_lng: float, 
                          dest_lat: float, dest_lng: float, 
                          num_points: int = 50) -> List[Dict[str, float]]:
    """Calculate 3D waypoints for flight path on a sphere"""
    waypoints = []
    
    for i in range(num_points + 1):
        t = i / num_points
        
        # Interpolate latitude and longitude
        lat = origin_lat + (dest_lat - origin_lat) * t
        lng = origin_lng + (dest_lng - origin_lng) * t
        
        # Convert to 3D coordinates (sphere with radius 5)
        radius = 5.0
        phi = (90 - lat) * (math.pi / 180)
        theta = (lng + 180) * (math.pi / 180)
        
        x = -(radius * math.sin(phi) * math.cos(theta))
        z = radius * math.sin(phi) * math.sin(theta)
        y = radius * math.cos(phi)
        
        # Add altitude curve (higher in the middle of journey)
        altitude_boost = math.sin(t * math.pi) * 2.0
        
        waypoints.append({
            "x": x,
            "y": y + altitude_boost,
            "z": z,
            "progress": t,
            "lat": lat,
            "lng": lng,
        })
    
    return waypoints


def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate great circle distance between two points"""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lng / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


@router.post("/create-session", response_model=VRSessionResponse)
async def create_vr_session(request: VRSessionRequest):
    """Create a VR/AR session for immersive flight experience"""
    try:
        # Airport coordinates (in production, fetch from database)
        airports = {
            "JFK": {"lat": 40.6413, "lng": -73.7781, "name": "New York JFK"},
            "CDG": {"lat": 49.0097, "lng": 2.5479, "name": "Paris CDG"},
            "LHR": {"lat": 51.4700, "lng": -0.4543, "name": "London Heathrow"},
            "NRT": {"lat": 35.7720, "lng": 140.3929, "name": "Tokyo Narita"},
            "DXB": {"lat": 25.2532, "lng": 55.3657, "name": "Dubai"},
            "SYD": {"lat": -33.9399, "lng": 151.1753, "name": "Sydney"},
            "LAX": {"lat": 33.9416, "lng": -118.4085, "name": "Los Angeles"},
            "SIN": {"lat": 1.3644, "lng": 103.9915, "name": "Singapore"},
        }
        
        if request.origin not in airports or request.destination not in airports:
            raise HTTPException(status_code=400, detail="Invalid airport code")
        
        origin = airports[request.origin]
        destination = airports[request.destination]
        
        # Calculate distance
        distance = calculate_distance(
            origin["lat"], origin["lng"],
            destination["lat"], destination["lng"]
        )
        
        # Calculate 3D waypoints
        waypoints = calculate_3d_waypoints(
            origin["lat"], origin["lng"],
            destination["lat"], destination["lng"],
            num_points=100
        )
        
        # Calculate duration (roughly 1 minute per 1000 km for visualization)
        duration = (distance / 1000) * 60
        
        # Create flight path
        flight_path = FlightPath3D(
            origin=request.origin,
            destination=request.destination,
            waypoints=waypoints,
            duration_seconds=duration,
            distance_km=distance,
        )
        
        # Create spatial audio zones
        audio_zones = []
        if request.enable_spatial_audio:
            # Add audio zones at key points
            for i, waypoint in enumerate(waypoints[::20]):  # Every 20th waypoint
                audio_zones.append({
                    "id": f"zone_{i}",
                    "position": {
                        "x": waypoint["x"],
                        "y": waypoint["y"],
                        "z": waypoint["z"],
                    },
                    "sound_type": "ambient" if i % 2 == 0 else "music",
                    "volume": 0.5 + (waypoint["progress"] * 0.3),
                    "frequency": 440 + (i * 50),  # Musical frequency
                })
        
        # VR settings based on quality
        quality_settings = {
            "low": {"resolution": "1280x720", "fps": 30, "antialiasing": False},
            "medium": {"resolution": "1920x1080", "fps": 60, "antialiasing": True},
            "high": {"resolution": "2560x1440", "fps": 90, "antialiasing": True},
            "ultra": {"resolution": "3840x2160", "fps": 120, "antialiasing": True},
        }
        
        vr_settings = quality_settings.get(request.quality, quality_settings["high"])
        vr_settings.update({
            "spatial_audio": request.enable_spatial_audio,
            "hand_tracking": True,
            "room_scale": True,
        })
        
        session_id = f"vr_{request.origin}_{request.destination}_{int(distance)}"
        
        # ✅ Sync AR/VR session to DuckDB using vector sync helper
        try:
            from app.services.vector_sync_helper import get_vector_sync_helper
            
            vector_sync = get_vector_sync_helper()
            vector_sync.sync_arvr_session(
                session_type="vr_flight",
                origin=request.origin,
                destination=request.destination,
                waypoint_count=len(waypoints),
                spatial_audio=request.enable_spatial_audio,
                quality=request.quality,
                duration=duration,
                metadata={
                    "distance_km": distance,
                    "audio_zones": len(audio_zones),
                    "resolution": vr_settings.get("resolution"),
                    "fps": vr_settings.get("fps")
                }
            )
            logger.info(f"✅ Synced AR/VR session to DuckDB: {request.origin} → {request.destination}")
        except Exception as e:
            logger.warning(f"Could not sync AR/VR session to DuckDB: {e}")
        
        return VRSessionResponse(
            session_id=session_id,
            flight_path=flight_path,
            audio_zones=audio_zones,
            recommended_duration=duration,
            vr_settings=vr_settings,
        )
        
    except Exception as e:
        logger.error(f"Error creating VR session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-airports")
async def get_supported_airports():
    """Get list of airports with 3D coordinates"""
    return {
        "airports": [
            {"code": "JFK", "name": "New York JFK", "lat": 40.6413, "lng": -73.7781, "color": "#3b82f6"},
            {"code": "CDG", "name": "Paris CDG", "lat": 49.0097, "lng": 2.5479, "color": "#8b5cf6"},
            {"code": "LHR", "name": "London Heathrow", "lat": 51.4700, "lng": -0.4543, "color": "#ec4899"},
            {"code": "NRT", "name": "Tokyo Narita", "lat": 35.7720, "lng": 140.3929, "color": "#f59e0b"},
            {"code": "DXB", "name": "Dubai", "lat": 25.2532, "lng": 55.3657, "color": "#10b981"},
            {"code": "SYD", "name": "Sydney", "lat": -33.9399, "lng": 151.1753, "color": "#06b6d4"},
            {"code": "LAX", "name": "Los Angeles", "lat": 33.9416, "lng": -118.4085, "color": "#f43f5e"},
            {"code": "SIN", "name": "Singapore", "lat": 1.3644, "lng": 103.9915, "color": "#a855f7"},
        ]
    }


@router.get("/vr-capabilities")
async def get_vr_capabilities():
    """Get VR/AR capabilities and requirements"""
    return {
        "webxr_support": True,
        "features": [
            {
                "name": "Immersive VR Mode",
                "description": "Full VR headset support with WebXR",
                "supported": True,
            },
            {
                "name": "Hand Tracking",
                "description": "Natural hand gestures for interaction",
                "supported": True,
            },
            {
                "name": "Spatial Audio",
                "description": "3D positional audio for immersive experience",
                "supported": True,
            },
            {
                "name": "Room Scale",
                "description": "Physical movement tracking",
                "supported": True,
            },
            {
                "name": "AR Overlay",
                "description": "Augmented reality flight paths",
                "supported": False,
                "coming_soon": True,
            },
        ],
        "requirements": {
            "browser": "Chrome 79+, Firefox 70+, Edge 79+",
            "vr_headset": "Oculus Quest, HTC Vive, Valve Index, or compatible",
            "minimum_specs": {
                "gpu": "GTX 1060 or equivalent",
                "ram": "8GB",
                "cpu": "Intel i5 or equivalent",
            },
        },
    }


class SpatialAudioRequest(BaseModel):
    origin: str
    destination: str
    audio_type: str = "ambient"


@router.post("/spatial-audio/generate")
async def generate_spatial_audio(request: SpatialAudioRequest):
    """Generate spatial audio configuration for flight path"""
    try:
        # Calculate audio zones along the path
        audio_config = {
            "origin": request.origin,
            "destination": request.destination,
            "audio_type": request.audio_type,
            "zones": [
                {
                    "position": "start",
                    "sounds": ["takeoff", "engine_startup", "ambient_airport"],
                    "volume": 0.8,
                },
                {
                    "position": "cruise",
                    "sounds": ["wind", "engine_cruise", "music_layer"],
                    "volume": 0.6,
                },
                {
                    "position": "end",
                    "sounds": ["landing", "engine_shutdown", "ambient_airport"],
                    "volume": 0.8,
                },
            ],
            "binaural": True,
            "reverb": "large_hall",
        }
        
        return audio_config
        
    except Exception as e:
        logger.error(f"Error generating spatial audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get VR session status and metrics"""
    return {
        "session_id": session_id,
        "status": "active",
        "elapsed_time": 45.5,
        "progress": 0.35,
        "current_position": {
            "lat": 45.2,
            "lng": -10.5,
            "altitude": 35000,
        },
        "performance": {
            "fps": 90,
            "latency_ms": 12,
            "quality": "high",
        },
    }
