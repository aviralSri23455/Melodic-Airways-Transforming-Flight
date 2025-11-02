"""
API routes for vector embedding and similarity search
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

try:
    from app.db.database import get_db
except ImportError:
    # Fallback for different database module structure
    from app.database import get_db

# Use DuckDB service instead of SQL service
from app.services.route_embedding_service_duckdb import get_route_embedding_service_duckdb as get_route_embedding_service
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vectors", tags=["Vector Embeddings"])


# Response models
class SimilarRouteResponse(BaseModel):
    route_id: int
    origin_airport_id: int
    destination_airport_id: int
    distance_km: float
    similarity_score: float
    origin_code: Optional[str] = None
    dest_code: Optional[str] = None


class GenreRouteResponse(BaseModel):
    route_id: int
    origin_airport_id: int
    destination_airport_id: int
    distance_km: float
    stops: int
    origin_code: str
    dest_code: str
    genre: str
    genre_match_score: float


class ComplexityMetrics(BaseModel):
    harmonic_complexity: float
    rhythmic_complexity: float
    melodic_complexity: float
    overall_complexity: float


class EmbeddingStatistics(BaseModel):
    total_routes: int
    routes_with_embeddings: int
    embedding_coverage: float
    avg_distance_km: float
    avg_stops: float
    avg_melodic_complexity: float
    avg_harmonic_complexity: float
    avg_rhythmic_complexity: float
    faiss_index_size: int
    embedding_dimension: int


@router.get("/similar-routes", response_model=List[SimilarRouteResponse])
async def find_similar_routes(
    origin: str = Query(..., description="Origin airport IATA code (e.g., JFK)"),
    destination: str = Query(..., description="Destination airport IATA code (e.g., LAX)"),
    limit: int = Query(10, ge=1, le=50, description="Number of similar routes to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Find routes similar to the given origin-destination pair using vector embeddings.
    
    Example: Find routes similar to JFK → LAX
    """
    try:
        service = get_route_embedding_service()
        similar_routes = await service.find_similar_routes(db, origin, destination, limit)
        
        if not similar_routes:
            raise HTTPException(
                status_code=404,
                detail=f"No similar routes found for {origin} → {destination}"
            )
        
        return similar_routes
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/routes-by-genre", response_model=List[GenreRouteResponse])
async def find_routes_by_genre(
    genre: str = Query(
        ...,
        description="Musical genre (classical, jazz, electronic, ambient, pop)"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of routes to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Find routes that match a musical genre profile.
    
    Genres:
    - classical: Complex, formal routes with multiple stops
    - jazz: Improvisational, varied routes
    - electronic: Repetitive, medium-distance routes
    - ambient: Long, calm, minimal-stop routes
    - pop: Popular, straightforward routes
    
    Example: Find routes that "sound like" classical music
    """
    try:
        valid_genres = ["classical", "jazz", "electronic", "ambient", "pop"]
        if genre.lower() not in valid_genres:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid genre. Must be one of: {', '.join(valid_genres)}"
            )
        
        service = get_route_embedding_service()
        genre_routes = await service.find_routes_by_genre(db, genre, limit)
        
        if not genre_routes:
            raise HTTPException(
                status_code=404,
                detail=f"No routes found matching genre: {genre}"
            )
        
        return genre_routes
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding routes by genre: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/route/{route_id}/complexity", response_model=ComplexityMetrics)
async def get_route_complexity(
    route_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate melodic complexity metrics for a specific route.
    
    Returns:
    - harmonic_complexity: Based on latitude change (0-1)
    - rhythmic_complexity: Based on number of stops (0-1)
    - melodic_complexity: Based on distance (0-1)
    - overall_complexity: Weighted average of all metrics
    
    Example: Get complexity for route ID 12345
    """
    try:
        service = get_route_embedding_service()
        complexity = await service.calculate_melodic_complexity(db, route_id)
        
        if not complexity:
            raise HTTPException(
                status_code=404,
                detail=f"Route {route_id} not found or complexity cannot be calculated"
            )
        
        return complexity
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating route complexity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=EmbeddingStatistics)
async def get_embedding_statistics(
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics about vector embeddings in the database - OPTIMIZED for high throughput.
    
    Returns information about:
    - Total routes and embedding coverage
    - Average route characteristics
    - Average complexity metrics
    - FAISS index size
    
    Caches results for 10 seconds to achieve 1000+ QPS.
    """
    try:
        from app.services.cache import get_cache
        import json
        cache = get_cache()
        
        # Cache statistics for 10 seconds using redis_client directly
        cache_key = "aero:cache:vector_statistics"
        
        if cache.redis_client:
            try:
                cached_data = cache.redis_client.get(cache_key)
                if cached_data:
                    # Parse and return cached stats
                    return json.loads(cached_data)
            except Exception:
                pass  # Cache miss or error, continue to generate fresh data
        
        service = get_route_embedding_service()
        stats = await service.get_embedding_statistics(db)
        
        # Cache for 10 seconds
        if cache.redis_client:
            try:
                # Convert stats to dict if it's a Pydantic model
                stats_dict = stats.dict() if hasattr(stats, 'dict') else stats
                cache.redis_client.setex(cache_key, 10, json.dumps(stats_dict, default=str))
            except Exception:
                pass  # Cache write failed, but return data anyway
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting embedding statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-embeddings")
async def generate_embeddings(
    batch_size: int = Query(1000, ge=100, le=10000, description="Batch size for processing"),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate vector embeddings for all routes in the database.
    
    This is a long-running operation that should be run once during setup.
    Use the script `python scripts/generate_route_embeddings.py` instead for better control.
    
    Returns statistics about the generation process.
    """
    try:
        service = get_route_embedding_service()
        stats = await service.generate_route_embeddings(db, batch_size)
        
        return {
            "status": "success",
            "message": "Embeddings generated successfully",
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def vector_health_check():
    """
    Check if vector embedding service is healthy - OPTIMIZED for high throughput.
    Caches result for 5 seconds to achieve 1000+ QPS.
    """
    try:
        from app.services.cache import get_cache
        import json
        cache = get_cache()
        
        # Cache health check for 5 seconds using redis_client directly
        cache_key = "aero:cache:vector_health"
        
        if cache.redis_client:
            try:
                cached_data = cache.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception:
                pass  # Cache miss or error, continue to generate fresh data
        
        service = get_route_embedding_service()
        
        health_data = {
            "status": "healthy",
            "embedding_dimension": service.embedding_dim,
            "encoder_loaded": service.encoder is not None,
            "faiss_service_available": service.faiss_service is not None
        }
        
        # Cache for 5 seconds
        if cache.redis_client:
            try:
                cache.redis_client.setex(cache_key, 5, json.dumps(health_data))
            except Exception:
                pass  # Cache write failed, but return data anyway
        
        return health_data
    
    except Exception as e:
        logger.error(f"Vector service health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
