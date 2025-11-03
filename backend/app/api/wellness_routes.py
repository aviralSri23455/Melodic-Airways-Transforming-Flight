"""
Wellness and therapeutic music generation routes
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.db.database import get_db
from app.services.music_generator import MusicGenerator

logger = logging.getLogger(__name__)

router = APIRouter()
music_generator = MusicGenerator()


class WellnessRequest(BaseModel):
    theme: str  # ocean, mountain, night
    calm_level: int  # 0-100
    route: Optional[str] = None
    duration_minutes: Optional[int] = 5


class WellnessResponse(BaseModel):
    composition_id: str
    theme: str
    calm_level: int
    duration: float
    notes: List[dict]
    binaural_frequency: Optional[float] = None


@router.post("/generate-wellness", response_model=WellnessResponse)
async def generate_wellness_composition(
    request: WellnessRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate therapeutic music based on wellness parameters using real music generator"""
    try:
        # Theme-specific parameters
        theme_params = {
            "ocean": {
                "tempo": 60 + (request.calm_level // 10),
                "scale": "pentatonic",
                "style": "major",
            },
            "mountain": {
                "tempo": 50 + (request.calm_level // 10),
                "scale": "minor",
                "style": "minor",
            },
            "night": {
                "tempo": 45 + (request.calm_level // 10),
                "scale": "dorian",
                "style": "minor",
            }
        }
        
        params = theme_params.get(request.theme, theme_params["ocean"])
        
        # Use default wellness routes if not specified
        default_routes = {
            "ocean": ("LAX", "HNL"),  # Los Angeles to Honolulu
            "mountain": ("DEN", "SLC"),  # Denver to Salt Lake City
            "night": ("JFK", "LHR"),  # New York to London
        }
        
        origin_code, dest_code = default_routes.get(request.theme, ("JFK", "LAX"))
        
        # Generate music using the actual music generator
        from app.models.schemas import MusicStyle, ScaleType
        
        # Map theme to music style and scale
        style_map = {
            "ocean": (MusicStyle.AMBIENT, ScaleType.PENTATONIC),
            "mountain": (MusicStyle.AMBIENT, ScaleType.MINOR),
            "night": (MusicStyle.AMBIENT, ScaleType.DORIAN),
        }
        
        music_style, scale_type = style_map.get(request.theme, (MusicStyle.AMBIENT, ScaleType.PENTATONIC))
        
        # Generate composition using the real music generator service
        from app.services.music_generator import get_music_generation_service
        
        music_service = get_music_generation_service()
        composition, midi_path, analytics = await music_service.generate_music_for_route(
            db=db,
            origin_code=origin_code,
            destination_code=dest_code,
            music_style=music_style,
            scale=scale_type,
            key="C",
            tempo=params["tempo"],
            duration_minutes=request.duration_minutes or 5
        )
        
        # Extract notes from the generated composition
        # Generate melody for the response
        music_params = music_service.generator.map_route_to_music_params_multi_segment(
            [{"distance_km": 1000, "direction_angle": 90, "origin_iata": origin_code, "destination_iata": dest_code}],
            1000,
            music_style,
            scale_type,
            "C",
            params["tempo"]
        )
        melody = music_service.generator.generate_melody(music_params, request.duration_minutes or 5)
        
        # Convert MIDI messages to note dictionaries
        notes = []
        for msg in melody:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append({
                    "note": msg.note,
                    "velocity": msg.velocity,
                    "time": msg.time,
                    "duration": 480  # Default quarter note duration
                })
        
        # Add binaural frequency for deep relaxation
        binaural_freq = None
        if request.theme == "night":
            binaural_freq = 4.0  # Theta waves for deep relaxation
        
        duration = request.duration_minutes * 60 if request.duration_minutes else 300
        
        # ✅ Generate real-time vector embeddings from wellness music
        try:
            from app.services.realtime_vector_sync import get_realtime_vector_sync
            
            # Extract real music features
            pitches = [n["note"] for n in notes if "note" in n]
            pitch_range = max(pitches) - min(pitches) if pitches else 24
            rhythm_density = len(notes) / max(1, duration)
            
            music_features = {
                "tempo": params["tempo"],
                "note_count": len(notes),
                "duration_seconds": duration,
                "harmony_complexity": 0.6,  # Wellness music is moderately complex
                "pitch_range": pitch_range,
                "rhythm_density": rhythm_density
            }
            
            vector_sync = get_realtime_vector_sync()
            vector_sync.sync_route_embedding(
                origin=origin_code,
                destination=dest_code,
                distance_km=analytics.get("distance_km", 1000) if analytics else 1000,
                complexity_score=0.5,
                intermediate_stops=0,
                music_features=music_features  # Pass REAL wellness music data
            )
            logger.info(f"🎵 Synced wellness music embedding: {origin_code} → {dest_code}")
        except Exception as e:
            logger.warning(f"Could not sync wellness music embedding: {e}")
        
        # ✅ Sync to DuckDB for analytics using vector sync helper
        try:
            from app.services.vector_sync_helper import get_vector_sync_helper
            
            vector_sync = get_vector_sync_helper()
            vector_sync.sync_wellness_composition(
                theme=request.theme,
                calm_level=request.calm_level,
                duration=int(duration),
                note_count=len(notes),
                binaural_frequency=binaural_freq,
                metadata={
                    "origin": origin_code,
                    "destination": dest_code,
                    "tempo": params["tempo"],
                    "scale": params.get("scale", "pentatonic")
                }
            )
            logger.info(f"✅ Synced wellness composition to DuckDB: {request.theme}")
        except Exception as e:
            logger.warning(f"Could not sync wellness to DuckDB: {e}")
        
        return WellnessResponse(
            composition_id=f"wellness_{request.theme}_{request.calm_level}_{origin_code}_{dest_code}",
            theme=request.theme,
            calm_level=request.calm_level,
            duration=duration,
            notes=notes,
            binaural_frequency=binaural_freq,
        )
        
    except Exception as e:
        logger.error(f"Error generating wellness composition: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wellness-themes")
async def get_wellness_themes():
    """Get available wellness themes"""
    return {
        "themes": [
            {
                "id": "ocean",
                "name": "Ocean Breeze",
                "description": "Calming coastal routes with gentle wave-like melodies",
                "recommended_routes": ["LAX → HNL", "MIA → CUN"],
            },
            {
                "id": "mountain",
                "name": "Mountain Serenity",
                "description": "Peaceful mountain routes with ambient soundscapes",
                "recommended_routes": ["DEN → SLC", "GVA → INN"],
            },
            {
                "id": "night",
                "name": "Night Flight",
                "description": "Soothing overnight routes for relaxation",
                "recommended_routes": ["JFK → LHR", "LAX → NRT"],
            },
        ]
    }
