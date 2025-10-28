"""
Analytics API routes for DuckDB analytics and performance metrics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
import logging

from app.services.duckdb_analytics import get_analytics
from app.services.cache import get_cache

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/analytics/route-complexity")
async def get_route_complexity_stats() -> Dict[str, Any]:
    """
    Get statistical summary of route complexity
    
    Returns:
        Dictionary with complexity statistics including:
        - total_routes: Total number of routes analyzed
        - avg_complexity: Average complexity score
        - min_complexity: Minimum complexity score
        - max_complexity: Maximum complexity score
        - std_complexity: Standard deviation of complexity
        - avg_distance: Average route distance
        - avg_stops: Average intermediate stops
    """
    try:
        analytics = get_analytics()
        stats = analytics.get_route_complexity_stats()
        
        if not stats:
            return {
                "message": "No route analytics data available yet",
                "total_routes": 0
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting route complexity stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/genre-distribution")
async def get_genre_distribution() -> Dict[str, int]:
    """
    Get distribution of music genres generated
    
    Returns:
        Dictionary mapping genre names to count
    """
    try:
        analytics = get_analytics()
        distribution = analytics.get_genre_distribution()
        
        if not distribution:
            return {"message": "No genre data available yet"}
        
        return distribution
        
    except Exception as e:
        logger.error(f"Error getting genre distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/performance")
async def get_performance_metrics(
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results")
) -> List[Dict[str, Any]]:
    """
    Get performance metrics for operations
    
    Args:
        operation_type: Optional filter by operation type
        limit: Maximum number of results (1-1000)
    
    Returns:
        List of performance metrics with:
        - operation_type: Type of operation
        - avg_time_ms: Average execution time in milliseconds
        - min_time_ms: Minimum execution time
        - max_time_ms: Maximum execution time
        - total_operations: Total number of operations
        - successful: Number of successful operations
        - failed: Number of failed operations
        - success_rate: Success rate percentage
    """
    try:
        analytics = get_analytics()
        metrics = analytics.get_performance_metrics(operation_type, limit)
        
        if not metrics:
            return []
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/similar-routes")
async def get_similar_routes(
    origin: str = Query(..., description="Origin airport IATA code"),
    destination: str = Query(..., description="Destination airport IATA code"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    min_similarity: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
) -> List[Dict[str, Any]]:
    """
    Find similar routes based on analytics data
    
    Args:
        origin: Origin airport IATA code
        destination: Destination airport IATA code
        limit: Maximum number of results (1-50)
        min_similarity: Minimum similarity threshold (0.0-1.0)
    
    Returns:
        List of similar routes with:
        - origin: Route origin
        - destination: Route destination
        - similarity_score: Similarity score (0-1)
        - distance_km: Route distance
        - complexity_score: Route complexity
    """
    try:
        analytics = get_analytics()
        similar_routes = analytics.get_similar_routes(
            origin.upper(),
            destination.upper(),
            limit,
            min_similarity
        )
        
        if not similar_routes:
            return []
        
        return similar_routes
        
    except Exception as e:
        logger.error(f"Error finding similar routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/cache-stats")
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get Redis cache statistics
    
    Returns:
        Dictionary with cache statistics:
        - connected: Whether Redis is connected
        - memory_used: Memory used by cache
        - hit_rate: Cache hit rate
        - keys_count: Number of keys in cache
    """
    try:
        cache = get_cache()
        stats = cache.get_cache_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/cache/clear")
async def clear_cache() -> Dict[str, str]:
    """
    Clear all cache (use with caution)
    
    Returns:
        Success message
    """
    try:
        cache = get_cache()
        success = cache.clear_all_cache()
        
        if success:
            return {"message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/analytics/cache/route")
async def invalidate_route_cache(
    origin: str = Query(..., description="Origin airport IATA code"),
    destination: str = Query(..., description="Destination airport IATA code")
) -> Dict[str, str]:
    """
    Invalidate cache for a specific route
    
    Args:
        origin: Origin airport IATA code
        destination: Destination airport IATA code
    
    Returns:
        Success message
    """
    try:
        cache = get_cache()
        success = cache.invalidate_route_cache(origin.upper(), destination.upper())
        
        if success:
            return {"message": f"Cache invalidated for route {origin} -> {destination}"}
        else:
            return {"message": "No cache found for this route"}
        
    except Exception as e:
        logger.error(f"Error invalidating route cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
