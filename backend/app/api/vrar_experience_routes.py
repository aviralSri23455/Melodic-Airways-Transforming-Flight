"""
VR/AR Experience API Routes - Immersive fly-through experiences
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

from app.services.vrar_experience_service import VRARExperienceService
from app.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.models import Airport


router = APIRouter()
vrar_service = VRARExperienceService()


class VRExperienceRequest(BaseModel):
    origin_code: str
    destination_code: str
    experience_type: str = "immersive"  # immersive, cinematic, educational
    camera_mode: str = "follow"  # follow, orbit, cinematic
    duration: Optional[float] = None
    music_composition: Optional[dict] = None


@router.post("/vr-experiences/create", response_model=dict)
async def create_vr_experience(
    request: VRExperienceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a complete VR/AR experience for a flight route
    
    Example request:
    {
        "origin_code": "JFK",
        "destination_code": "LAX",
        "experience_type": "cinematic",
        "camera_mode": "orbit",
        "music_composition": {
            "tempo": 120,
            "duration": 60,
            "segments": [...]
        }
    }
    """
    try:
        # Get airport coordinates
        origin_result = await db.execute(
            select(Airport).where(Airport.iata_code == request.origin_code)
        )
        origin_airport = origin_result.scalar_one_or_none()
        
        dest_result = await db.execute(
            select(Airport).where(Airport.iata_code == request.destination_code)
        )
        dest_airport = dest_result.scalar_one_or_none()
        
        if not origin_airport or not dest_airport:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both airports not found"
            )
        
        # Default music composition if not provided
        music_comp = request.music_composition or {
            "tempo": 120,
            "duration": 60,
            "scale": "major",
            "segments": []
        }
        
        # Create VR experience
        experience = vrar_service.create_vr_experience(
            origin_code=request.origin_code,
            destination_code=request.destination_code,
            origin_coords=(float(origin_airport.latitude), float(origin_airport.longitude)),
            destination_coords=(float(dest_airport.latitude), float(dest_airport.longitude)),
            music_composition=music_comp,
            experience_type=request.experience_type
        )
        
        # ✅ Sync VR experience to DuckDB (64D VR embeddings)
        try:
            from app.services.duckdb_sync_service import duckdb_sync
            import numpy as np
            import logging
            
            logger = logging.getLogger(__name__)
            
            # Generate 64D embedding for VR experience
            embedding_64d = np.random.randn(64).tolist()  # In production, use real embedding model
            
            vr_data = {
                "vector_embedding": embedding_64d,
                "experience_id": experience.get("experience_id", f"vr_{request.origin_code}_{request.destination_code}"),
                "type": request.experience_type,
                "route": {
                    "origin": request.origin_code,
                    "destination": request.destination_code
                },
                "duration": experience.get("duration", 60.0),
                "vr_ready": True,
                "ar_ready": True,
                "platforms": experience.get("platforms", ["WebXR", "Oculus", "HTC Vive"])
            }
            
            duckdb_sync.sync_vr_experience_embedding(vr_data)
            logger.info(f"✅ Synced VR experience to DuckDB: {request.origin_code} → {request.destination_code}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not sync VR experience to DuckDB: {e}")
        
        return {
            "success": True,
            "data": experience,
            "message": "VR/AR experience created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create VR experience: {str(e)}"
        )


@router.get("/vr-experiences/flight-path", response_model=dict)
async def get_flight_path_3d(
    origin_code: str = Query(..., description="Origin airport code"),
    destination_code: str = Query(..., description="Destination airport code"),
    num_points: int = Query(100, description="Number of path points"),
    flight_altitude: float = Query(10.0, description="Cruise altitude in km"),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate 3D flight path coordinates for visualization
    
    Returns coordinates suitable for Three.js, Unity, or other 3D engines
    """
    try:
        # Get airport coordinates
        origin_result = await db.execute(
            select(Airport).where(Airport.iata_code == origin_code)
        )
        origin_airport = origin_result.scalar_one_or_none()
        
        dest_result = await db.execute(
            select(Airport).where(Airport.iata_code == destination_code)
        )
        dest_airport = dest_result.scalar_one_or_none()
        
        if not origin_airport or not dest_airport:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both airports not found"
            )
        
        # Generate 3D path
        flight_path = vrar_service.generate_3d_flight_path(
            origin_coords=(float(origin_airport.latitude), float(origin_airport.longitude), 0.0),
            destination_coords=(float(dest_airport.latitude), float(dest_airport.longitude), 0.0),
            num_points=num_points,
            flight_altitude=flight_altitude
        )
        
        return {
            "success": True,
            "data": {
                "origin": origin_code,
                "destination": destination_code,
                "path_points": flight_path,
                "total_points": len(flight_path)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate flight path: {str(e)}"
        )


@router.post("/vr-experiences/camera-animation", response_model=dict)
async def generate_camera_animation(
    flight_path: list,
    duration: float = Query(60.0, description="Animation duration in seconds"),
    camera_mode: str = Query("follow", description="Camera mode: follow, orbit, cinematic")
):
    """
    Generate camera animation keyframes for a given flight path
    
    Accepts a flight path and returns camera animation data
    """
    try:
        animation = vrar_service.generate_camera_animation(
            flight_path=flight_path,
            duration=duration,
            camera_mode=camera_mode
        )
        
        return {
            "success": True,
            "data": animation
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate camera animation: {str(e)}"
        )


class SpatialAudioRequest(BaseModel):
    flight_path: list
    music_segments: list


@router.post("/vr-experiences/spatial-audio", response_model=dict)
async def generate_spatial_audio(request: SpatialAudioRequest):
    """
    Generate spatial audio positioning for immersive experience
    
    Maps music segments to 3D positions along the flight path
    """
    try:
        audio_zones = vrar_service.generate_spatial_audio_zones(
            flight_path=request.flight_path,
            music_segments=request.music_segments
        )
        
        return {
            "success": True,
            "data": {
                "audio_zones": audio_zones,
                "total_zones": len(audio_zones)
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate spatial audio: {str(e)}"
        )


@router.get("/vr-experiences/export/unity/{experience_id}", response_model=dict)
async def export_for_unity(experience_id: str):
    """
    Export VR experience in Unity-compatible format
    
    Returns JSON formatted for Unity import
    """
    try:
        # In production, fetch experience from database
        # For now, return format specification
        return {
            "success": True,
            "message": "Unity export format",
            "data": {
                "format": "Unity JSON",
                "coordinate_system": "Left-handed Y-up",
                "note": "Use the create endpoint first, then export the experience_id"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export for Unity: {str(e)}"
        )


@router.get("/vr-experiences/export/webxr/{experience_id}", response_model=dict)
async def export_for_webxr(experience_id: str):
    """
    Export VR experience in WebXR-compatible format
    
    Returns JSON formatted for WebXR/Three.js
    """
    try:
        return {
            "success": True,
            "message": "WebXR export format",
            "data": {
                "format": "WebXR JSON",
                "coordinate_system": "Right-handed Y-up",
                "frameworks": ["Three.js", "A-Frame", "Babylon.js"],
                "note": "Use the create endpoint first, then export the experience_id"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export for WebXR: {str(e)}"
        )


@router.get("/vr-experiences/demo", response_model=dict)
async def demo_vr_experience(
    origin: str = Query("JFK", description="Origin airport code"),
    destination: str = Query("LAX", description="Destination airport code"),
    db: AsyncSession = Depends(get_db)
):
    """
    Quick demo of VR experience generation
    
    Example: /vr-experiences/demo?origin=JFK&destination=LAX
    """
    try:
        # Get airport coordinates
        origin_result = await db.execute(
            select(Airport).where(Airport.iata_code == origin)
        )
        origin_airport = origin_result.scalar_one_or_none()
        
        dest_result = await db.execute(
            select(Airport).where(Airport.iata_code == destination)
        )
        dest_airport = dest_result.scalar_one_or_none()
        
        if not origin_airport or not dest_airport:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both airports not found"
            )
        
        # Create simplified VR experience
        experience = vrar_service.create_vr_experience(
            origin_code=origin,
            destination_code=destination,
            origin_coords=(float(origin_airport.latitude), float(origin_airport.longitude)),
            destination_coords=(float(dest_airport.latitude), float(dest_airport.longitude)),
            music_composition={"tempo": 120, "duration": 30, "segments": []},
            experience_type="immersive"
        )
        
        # Return simplified version for demo
        return {
            "success": True,
            "data": {
                "experience_id": experience["experience_id"],
                "route": experience["route"],
                "path_points_count": len(experience["flight_path"]),
                "camera_keyframes_count": len(experience["camera_animation"]["keyframes"]),
                "audio_zones_count": len(experience["spatial_audio"]),
                "duration": experience["duration"],
                "platforms": experience["platforms"],
                "sample_path_point": experience["flight_path"][0] if experience["flight_path"] else None,
                "sample_camera_keyframe": experience["camera_animation"]["keyframes"][0] if experience["camera_animation"]["keyframes"] else None
            },
            "demo": True,
            "message": "VR experience demo generated. Use /vr-experiences/create for full experience."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate demo: {str(e)}"
        )


@router.post("/vr-experiences/similar", response_model=dict)
async def find_similar_vr_experiences(experience: dict):
    """
    Find similar VR experiences using vector similarity search
    
    Example request:
    {
        "type": "cinematic",
        "route": {
            "origin": "JFK",
            "destination": "LAX"
        },
        "duration": 60
    }
    """
    try:
        similar = vrar_service.find_similar_experiences(
            query_experience=experience,
            k=5
        )
        
        return {
            "success": True,
            "data": {
                "similar_experiences": similar,
                "total_indexed": vrar_service.faiss_index.ntotal,
                "search_method": "FAISS vector similarity (L2 distance)"
            },
            "message": f"Found {len(similar)} similar VR experiences"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar experiences: {str(e)}"
        )


@router.get("/vr-experiences/index-stats", response_model=dict)
async def get_vr_index_statistics():
    """Get statistics about the VR experience FAISS index"""
    try:
        return {
            "success": True,
            "data": {
                "total_experiences": vrar_service.faiss_index.ntotal,
                "embedding_dimension": vrar_service.embedding_dim,
                "index_type": "FAISS IndexFlatL2",
                "metadata_count": len(vrar_service.experience_metadata),
                "recent_experiences": vrar_service.experience_metadata[-5:] if vrar_service.experience_metadata else []
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get VR index stats: {str(e)}"
        )


@router.get("/vr-experiences/info", response_model=dict)
async def get_vr_info():
    """Get information about VR/AR capabilities"""
    import faiss
    
    return {
        "success": True,
        "data": {
            "features": [
                "3D flight path generation with great circle routes",
                "Dynamic camera animations (follow, orbit, cinematic)",
                "Spatial audio positioning",
                "Interactive hotspots",
                "Environmental effects (clouds, lighting, weather)",
                "Multi-platform export (Unity, WebXR, Unreal)",
                "✨ Vector similarity search (FAISS)",
                "✨ 64D experience embeddings",
                "✨ Semantic experience recommendations"
            ],
            "supported_platforms": [
                "WebXR (Three.js, A-Frame, Babylon.js)",
                "Oculus Quest",
                "HTC Vive",
                "Apple ARKit",
                "Google ARCore",
                "Unity",
                "Unreal Engine"
            ],
            "experience_types": [
                "immersive - Full VR experience with spatial audio",
                "cinematic - Movie-like camera movements",
                "educational - Interactive learning with hotspots"
            ],
            "camera_modes": [
                "follow - Camera follows behind aircraft",
                "orbit - Camera orbits around flight path",
                "cinematic - Dynamic angles and movements"
            ],
            "vector_search": {
                "enabled": True,
                "faiss_version": faiss.__version__,
                "embedding_dim": 64,
                "indexed_experiences": vrar_service.faiss_index.ntotal
            }
        }
    }
