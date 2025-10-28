"""
Redis Integration Routes
Endpoints for session management, live collaboration, and cache monitoring
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
import json

from app.db.database import get_db
from app.services.redis_session_manager import get_session_manager
from app.services.redis_publisher import get_publisher
from app.services.cache import get_cache
from app.services.duckdb_analytics import get_analytics
import numpy as np
from app.services.faiss_duckdb_service import get_faiss_duckdb_service
from app.core.config import settings

logger = logging.getLogger(__name__)

# Pydantic models for request bodies
class MusicSearchRequest(BaseModel):
    query_vector: List[float]
    limit: int = 10
    genre_filter: Optional[str] = None

router = APIRouter()

# Debug: Print all routes being registered
print(f"DEBUG: Creating Redis router with prefix: {settings.API_V1_STR}/redis")
print("DEBUG: Registering Redis endpoints...")

# Simple test endpoint
@router.get("/test")
async def test_redis():
    """Simple test endpoint to verify Redis router is working"""
    print("DEBUG: Redis test endpoint called")
    return {
        "status": "success",
        "message": "Redis router is working!",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


@router.post("/test/save-music")
async def test_save_music_to_redis(
    origin: str = Query("DEL", description="Origin airport code"),
    destination: str = Query("LHR", description="Destination airport code")
):
    """Test endpoint to save music data to Redis and verify it appears in Redis Insight"""
    try:
        cache = get_cache()
        
        # Create test music data
        import time
        test_music_data = {
            "composition_id": int(time.time()),
            "origin": origin,
            "destination": destination,
            "tempo": 120,
            "duration_seconds": 30,
            "note_count": 60,
            "notes": [
                {"note": 60, "velocity": 80, "time": 0, "duration": 480},
                {"note": 62, "velocity": 80, "time": 480, "duration": 480},
                {"note": 64, "velocity": 80, "time": 960, "duration": 480}
            ],
            "generated_at": __import__('datetime').datetime.utcnow().isoformat(),
            "test": True
        }
        
        # Save to Redis using the cache service
        success = cache.set_route_music(origin, destination, test_music_data)
        
        # Also save with a simple readable key for easy viewing in Redis Insight
        if cache.redis_client:
            simple_key = f"aero:test:music:{origin}:{destination}"
            cache.redis_client.setex(
                simple_key,
                3600,  # 1 hour TTL
                json.dumps(test_music_data, default=str)
            )
            
            # Save another key with composition details
            composition_key = f"aero:composition:{test_music_data['composition_id']}"
            cache.redis_client.setex(
                composition_key,
                3600,
                json.dumps(test_music_data, default=str)
            )
        
        # Verify it was saved
        retrieved = cache.get_route_music(origin, destination)
        
        # Get all keys to show what's in Redis
        all_keys = []
        if cache.redis_client:
            all_keys = cache.redis_client.keys("aero:*")
        
        return {
            "status": "success",
            "message": f"Test music data saved to Redis for {origin} → {destination}",
            "saved": success,
            "data_saved": test_music_data,
            "data_retrieved": retrieved,
            "redis_keys_created": [
                f"aero:music:{origin}:{destination}",
                f"aero:test:music:{origin}:{destination}",
                f"aero:composition:{test_music_data['composition_id']}"
            ],
            "all_aero_keys_in_redis": all_keys[:20],  # Show first 20 keys
            "total_keys": len(all_keys),
            "redis_insight_tip": "Check Redis Insight for keys starting with 'aero:'"
        }
        
    except Exception as e:
        logger.error(f"Error testing Redis save: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SESSION MANAGEMENT ====================

@router.post("/sessions/create")
async def create_session(
    origin: str = Query(..., description="Origin airport code"),
    destination: str = Query(..., description="Destination airport code"),
    session_type: str = Query("generation", description="Type of session")
):
    """Create a new live collaboration session"""
    try:
        session_manager = get_session_manager()
        session_id = session_manager.create_session_with_storage_check(
            user_id="public",
            origin=origin,
            destination=destination,
            session_type=session_type
        )

        if not session_id:
            raise HTTPException(status_code=507, detail="Redis storage limit reached. Please try again later.")

        session = session_manager.get_session(session_id)
        return {
            "status": "success",
            "session_id": session_id,
            "session": session
        }

    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/stats")
async def get_session_stats():
    """Get session statistics"""
    try:
        session_manager = get_session_manager()
        stats = session_manager.get_session_stats()

        return {
            "status": "success",
            "session_stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str
):
    """Get session details"""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "status": "success",
            "session": session
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}/music")
async def update_session_music(
    session_id: str,
    tempo: int = Query(120, ge=40, le=300),
    scale: str = Query("major"),
    key: str = Query("C")
):
    """Update music parameters in a session"""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        success = session_manager.update_session_music(
            session_id=session_id,
            tempo=tempo,
            scale=scale,
            key=key
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update session")

        updated_session = session_manager.get_session(session_id)
        return {
            "status": "success",
            "message": "Music parameters updated",
            "session": updated_session
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session music: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/edits")
async def add_session_edit(
    session_id: str,
    edit_type: str,
    edit_data: Dict[str, Any]
):
    """Add an edit to session history"""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        success = session_manager.add_session_edit(
            session_id=session_id,
            user_id="public",
            edit_type=edit_type,
            edit_data=edit_data
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add edit")

        return {
            "status": "success",
            "message": "Edit added to session"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding session edit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/participants/{user_id}")
async def add_session_participant(
    session_id: str,
    user_id: str
):
    """Add participant to session"""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        success = session_manager.add_participant(session_id, user_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add participant")

        updated_session = session_manager.get_session(session_id)
        return {
            "status": "success",
            "message": "Participant added",
            "session": updated_session
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding participant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/close")
async def close_session(
    session_id: str
):
    """Close a session"""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        success = session_manager.close_session(session_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to close session")

        return {
            "status": "success",
            "message": "Session closed"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/user/active")
async def get_user_sessions():
    """Get all active sessions for public user"""
    try:
        session_manager = get_session_manager()
        sessions = session_manager.get_user_sessions("public")

        return {
            "status": "success",
            "count": len(sessions),
            "sessions": sessions
        }

    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/active/all")
async def get_all_active_sessions():
    """Get all active sessions (admin only)"""
    try:
        session_manager = get_session_manager()
        sessions = session_manager.get_active_sessions()

        return {
            "status": "success",
            "count": len(sessions),
            "sessions": sessions
        }

    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CACHE MONITORING ====================

@router.get("/cache/stats")
async def get_cache_stats():
    """Get Redis cache statistics"""
    try:
        cache_service = get_cache()
        stats = cache_service.get_cache_stats()

        return {
            "status": "success",
            "cache_stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = None
):
    """Clear cache (optionally by pattern)"""
    try:
        cache_service = get_cache()

        if pattern:
            cleared = cache_service.clear_pattern(pattern)
            return {
                "status": "success",
                "message": f"Cleared {cleared} keys matching pattern '{pattern}'"
            }
        else:
            cache_service.clear_all_cache()
            return {
                "status": "success",
                "message": "Cleared all cache"
            }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/info")
async def get_storage_info():
    """Get Redis storage usage information"""
    try:
        cache_service = get_cache()
        storage_info = cache_service.get_storage_info()

        return {
            "status": "success",
            "storage_info": storage_info
        }

    except Exception as e:
        logger.error(f"Error getting storage info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/storage/cleanup")
async def cleanup_storage():
    """Clean up old sessions and expired cache"""
    try:
        session_manager = get_session_manager()
        cache_service = get_cache()

        # Get current storage info
        storage_before = cache_service.get_storage_info()

        # Clean up old sessions (older than 7 days)
        cleanup_count = 0
        active_sessions = session_manager.get_active_sessions()

        for session in active_sessions:
            # Check if session is old (more than 7 days)
            created_at = session.get('created_at')
            if created_at:
                from datetime import datetime
                try:
                    created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if (datetime.now() - created).days > 7:
                        session_manager.close_session(session['session_id'])
                        cleanup_count += 1
                except:
                    pass  # Skip malformed dates

        # Clear old cache patterns
        cache_service.clear_pattern("music:*")  # Clear music cache older than TTL
        cache_service.clear_pattern("airport:*")
        cache_service.clear_pattern("route:*")

        # Get storage after cleanup
        storage_after = cache_service.get_storage_info()

        return {
            "status": "success",
            "message": f"Cleaned up {cleanup_count} old sessions",
            "storage_before": storage_before,
            "storage_after": storage_after
        }

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== LIVE DATA ENDPOINTS ====================

@router.get("/live/routes/cached")
async def get_cached_routes(
    limit: int = Query(10, ge=1, le=100)
):
    """Get recently cached routes"""
    try:
        cache_service = get_cache()

        # Get all cached route keys
        route_keys = cache_service.redis_client.keys("music:*") if cache_service.redis_client else []

        cached_routes = []
        for key in route_keys[:limit]:
            route_data = cache_service.redis_client.get(key) if cache_service.redis_client else None
            if route_data:
                import json
                cached_routes.append(json.loads(route_data))

        return {
            "status": "success",
            "count": len(cached_routes),
            "cached_routes": cached_routes
        }

    except Exception as e:
        logger.error(f"Error getting cached routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live/sessions/active")
async def get_live_active_sessions():
    """Get live active sessions with real-time data"""
    try:
        session_manager = get_session_manager()
        sessions = session_manager.get_active_sessions()

        return {
            "status": "success",
            "count": len(sessions),
            "active_sessions": sessions,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting live sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REAL-TIME VECTOR SEARCH ====================

@router.post("/vectors/search/music")
async def search_similar_music_realtime(
    request: MusicSearchRequest
):
    """Search for similar music using FAISS + DuckDB in real-time"""
    try:
        faiss_service = get_faiss_duckdb_service()
        query_array = np.array(request.query_vector)

        # Perform vector search
        results = faiss_service.search_similar_music(
            query_vector=query_array,
            limit=request.limit,
            genre_filter=request.genre_filter
        )

        # Publish search results to Redis for real-time updates (without user context)
        publisher = get_publisher()
        publisher.publish_vector_search_results(
            search_id=f"search_public_{int(__import__('time').time())}",
            user_id="public",
            query_vector=request.query_vector,
            results=results,
            search_type="music"
        )

        return {
            "status": "success",
            "query_vector_length": len(request.query_vector),
            "results_count": len(results),
            "results": results,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in real-time music search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/search/routes")
async def search_similar_routes_realtime(
    origin: str = Query(..., description="Origin airport code"),
    destination: str = Query(..., description="Destination airport code"),
    limit: int = Query(10, ge=1, le=50)
):
    """Search for similar routes using FAISS + DuckDB in real-time"""
    try:
        faiss_service = get_faiss_duckdb_service()

        # Perform route search
        results = faiss_service.search_similar_routes(
            origin=origin,
            destination=destination,
            limit=limit
        )

        # Publish search results to Redis for real-time updates (without user context)
        publisher = get_publisher()
        publisher.publish_vector_search_results(
            search_id=f"route_search_public_{int(__import__('time').time())}",
            user_id="public",
            query_vector=[hash(origin) % 1000, hash(destination) % 1000],  # Simple embedding
            results=results,
            search_type="routes"
        )

        return {
            "status": "success",
            "origin": origin,
            "destination": destination,
            "results_count": len(results),
            "results": results,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in real-time route search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/store")
async def store_music_vector_realtime(
    composition_id: int,
    route_id: int,
    origin: str,
    destination: str,
    genre: str,
    tempo: int,
    pitch: float,
    harmony: float,
    complexity: float,
    vector: List[float],
    metadata: Optional[Dict[str, Any]] = None
):
    """Store music vector with real-time sync"""
    try:
        faiss_service = get_faiss_duckdb_service()
        vector_array = np.array(vector)

        # Store vector
        success = faiss_service.store_music_vector(
            composition_id=composition_id,
            route_id=route_id,
            origin=origin,
            destination=destination,
            genre=genre,
            tempo=tempo,
            pitch=pitch,
            harmony=harmony,
            complexity=complexity,
            vector=vector_array,
            metadata=metadata
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to store vector")

        # Publish real-time update
        publisher = get_publisher()
        publisher.publish_route_music_sync(
            route_id=str(route_id),
            origin=origin,
            destination=destination,
            music_params={
                "composition_id": composition_id,
                "genre": genre,
                "tempo": tempo,
                "pitch": pitch,
                "harmony": harmony,
                "complexity": complexity,
                "vector_length": len(vector)
            }
        )

        # Also publish to music generation progress channel
        publisher.publish_music_update_real_time(
            session_id=f"composition_{composition_id}",
            user_id="public",
            update_type="vector_stored",
            music_data={
                "composition_id": composition_id,
                "vector_stored": True,
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }
        )

        return {
            "status": "success",
            "message": "Vector stored successfully",
            "composition_id": composition_id,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing music vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectors/stats")
async def get_vector_stats():
    """Get FAISS + DuckDB vector statistics"""
    try:
        faiss_service = get_faiss_duckdb_service()
        stats = faiss_service.get_statistics()

        return {
            "status": "success",
            "vector_stats": stats,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting vector stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REAL-TIME GENERATION PROGRESS ====================

@router.post("/generation/progress")
async def publish_generation_progress(
    generation_id: str,
    progress: float = Query(0.0, ge=0.0, le=1.0),
    status: str = "processing",
    current_step: Optional[str] = None
):
    """Publish music generation progress in real-time"""
    try:
        publisher = get_publisher()
        subscribers = publisher.publish_generation_progress(
            generation_id=generation_id,
            user_id="public",
            progress=progress,
            status=status,
            current_step=current_step
        )

        return {
            "status": "success",
            "generation_id": generation_id,
            "progress": progress,
            "status": status,
            "current_step": current_step,
            "subscribers_notified": subscribers,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error publishing generation progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generation/complete")
async def publish_generation_complete(
    generation_id: str,
    composition_id: int,
    music_data: Dict[str, Any]
):
    """Publish music generation completion"""
    try:
        publisher = get_publisher()

        # Publish completion to generation progress channel
        publisher.publish_generation_progress(
            generation_id=generation_id,
            user_id="public",
            progress=1.0,
            status="completed",
            current_step="Generation complete"
        )

        # Publish music generation event
        publisher.publish_music_generated(
            route_id=str(composition_id),
            user_id="public",
            music_data=music_data
        )

        # Publish real-time update to session
        publisher.publish_music_update_real_time(
            session_id=f"composition_{composition_id}",
            user_id="public",
            update_type="generation_completed",
            music_data={
                "generation_id": generation_id,
                "composition_id": composition_id,
                "completed_at": __import__('datetime').datetime.utcnow().isoformat(),
                **music_data
            }
        )

        return {
            "status": "success",
            "message": "Generation completed and published",
            "generation_id": generation_id,
            "composition_id": composition_id,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error publishing generation completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REAL-TIME SYSTEM MONITORING ====================

@router.post("/system/status")
async def publish_system_status(
    status_type: str,
    status_data: Dict[str, Any]
):
    """Publish system status updates"""
    try:
        publisher = get_publisher()
        subscribers = publisher.publish_system_status(
            status_type=status_type,
            status_data=status_data
        )

        return {
            "status": "success",
            "status_type": status_type,
            "subscribers_notified": subscribers,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error publishing system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health")
async def get_system_health():
    """Get real-time system health status"""
    try:
        publisher = get_publisher()
        cache_service = get_cache()
        faiss_service = get_faiss_duckdb_service()

        # Get Redis info
        redis_info = {}
        if cache_service.redis_client:
            try:
                redis_info = cache_service.redis_client.info()
            except:
                redis_info = {"error": "Failed to get Redis info"}

        # Get FAISS stats
        faiss_stats = faiss_service.get_statistics()

        # Get DuckDB analytics
        duckdb_stats = get_analytics().get_route_complexity_stats()

        health_data = {
            "redis_connected": cache_service.redis_client is not None,
            "redis_info": redis_info,
            "faiss_vectors": faiss_stats.get("total_vectors", 0),
            "duckdb_routes": duckdb_stats.get("total_routes", 0),
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

        # Publish health status
        publisher.publish_system_status(
            status_type="health_check",
            status_data=health_data
        )

        return {
            "status": "healthy",
            "health_data": health_data
        }

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
