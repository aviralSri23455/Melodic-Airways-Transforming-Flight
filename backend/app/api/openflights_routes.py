"""
OpenFlights Data Routes - Populate Redis with Real-Time Flight Data
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any
import logging

from app.db.database import get_db
from app.models.models import Airport, Route, User
from app.core.security import get_current_active_user
from app.services.openflights_cache import get_openflights_cache

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== AIRPORT DATA ENDPOINTS ====================

@router.post("/openflights/cache/airports")
async def cache_airports_to_redis(
    limit: int = Query(100, ge=1, le=1000, description="Number of airports to cache"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch airports from MariaDB and cache them in Redis
    This populates Redis with real OpenFlights airport data
    """
    try:
        cache_service = get_openflights_cache()
        
        # Fetch airports from MariaDB
        result = await db.execute(
            select(Airport).limit(limit)
        )
        airports = result.scalars().all()
        
        if not airports:
            return {
                "status": "warning",
                "message": "No airports found in database",
                "cached_count": 0
            }
        
        # Convert to dict and cache in Redis
        airports_data = []
        for airport in airports:
            airport_dict = {
                "id": airport.id,
                "name": airport.name,
                "city": airport.city,
                "country": airport.country,
                "iata_code": airport.iata_code,
                "icao_code": airport.icao_code,
                "latitude": float(airport.latitude) if airport.latitude else None,
                "longitude": float(airport.longitude) if airport.longitude else None,
                "altitude": airport.altitude,
                "timezone": airport.timezone
            }
            airports_data.append(airport_dict)
        
        # Cache in Redis
        cached_count = cache_service.cache_multiple_airports(airports_data, ttl=3600)
        
        return {
            "status": "success",
            "message": f"Cached {cached_count} airports in Redis",
            "cached_count": cached_count,
            "total_fetched": len(airports),
            "sample_airports": [a["iata_code"] for a in airports_data[:10]]
        }
    
    except Exception as e:
        logger.error(f"Error caching airports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/openflights/airports/{airport_code}")
async def get_airport_with_cache(
    airport_code: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get airport data - checks Redis cache first, then MariaDB
    Automatically caches the result in Redis
    """
    try:
        cache_service = get_openflights_cache()
        
        # Check Redis cache first
        cached_airport = cache_service.get_cached_airport(airport_code)
        if cached_airport:
            return {
                "status": "success",
                "source": "redis_cache",
                "airport": cached_airport
            }
        
        # Not in cache, fetch from MariaDB
        result = await db.execute(
            select(Airport).where(Airport.iata_code == airport_code)
        )
        airport = result.scalar_one_or_none()
        
        if not airport:
            raise HTTPException(status_code=404, detail=f"Airport {airport_code} not found")
        
        # Convert to dict
        airport_dict = {
            "id": airport.id,
            "name": airport.name,
            "city": airport.city,
            "country": airport.country,
            "iata_code": airport.iata_code,
            "icao_code": airport.icao_code,
            "latitude": float(airport.latitude) if airport.latitude else None,
            "longitude": float(airport.longitude) if airport.longitude else None,
            "altitude": airport.altitude,
            "timezone": airport.timezone
        }
        
        # Cache in Redis for next time
        cache_service.cache_airport(airport_code, airport_dict, ttl=3600)
        
        return {
            "status": "success",
            "source": "mariadb",
            "airport": airport_dict,
            "message": "Cached in Redis for future requests"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting airport: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ROUTE DATA ENDPOINTS ====================

@router.post("/openflights/cache/routes")
async def cache_routes_to_redis(
    limit: int = Query(100, ge=1, le=1000, description="Number of routes to cache"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch routes from MariaDB and cache them in Redis
    This populates Redis with real OpenFlights route data
    """
    try:
        cache_service = get_openflights_cache()
        
        # Fetch routes from MariaDB
        result = await db.execute(
            select(Route).limit(limit)
        )
        routes = result.scalars().all()
        
        if not routes:
            return {
                "status": "warning",
                "message": "No routes found in database",
                "cached_count": 0
            }
        
        # Cache each route in Redis
        cached_count = 0
        sample_routes = []
        
        for route in routes:
            # Get origin and destination airports
            origin_result = await db.execute(
                select(Airport).where(Airport.id == route.source_airport_id)
            )
            origin_airport = origin_result.scalar_one_or_none()
            
            dest_result = await db.execute(
                select(Airport).where(Airport.id == route.destination_airport_id)
            )
            dest_airport = dest_result.scalar_one_or_none()
            
            if origin_airport and dest_airport:
                route_dict = {
                    "id": route.id,
                    "origin_code": origin_airport.iata_code,
                    "origin_name": origin_airport.name,
                    "origin_city": origin_airport.city,
                    "destination_code": dest_airport.iata_code,
                    "destination_name": dest_airport.name,
                    "destination_city": dest_airport.city,
                    "airline": route.airline,
                    "distance_km": route.distance_km,
                    "stops": route.stops
                }
                
                if cache_service.cache_route(
                    origin_airport.iata_code,
                    dest_airport.iata_code,
                    route_dict,
                    ttl=1800
                ):
                    cached_count += 1
                    if len(sample_routes) < 10:
                        sample_routes.append(f"{origin_airport.iata_code}-{dest_airport.iata_code}")
        
        return {
            "status": "success",
            "message": f"Cached {cached_count} routes in Redis",
            "cached_count": cached_count,
            "total_fetched": len(routes),
            "sample_routes": sample_routes
        }
    
    except Exception as e:
        logger.error(f"Error caching routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/openflights/routes/{origin}/{destination}")
async def get_route_with_cache(
    origin: str,
    destination: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get route data - checks Redis cache first, then MariaDB
    Automatically caches the result and publishes real-time event
    """
    try:
        cache_service = get_openflights_cache()
        
        # Check Redis cache first
        cached_route = cache_service.get_cached_route(origin, destination)
        if cached_route:
            # Publish real-time lookup event
            cache_service.publish_route_lookup(origin, destination, str(current_user.id))
            
            return {
                "status": "success",
                "source": "redis_cache",
                "route": cached_route
            }
        
        # Not in cache, fetch from MariaDB
        # Get origin airport
        origin_result = await db.execute(
            select(Airport).where(Airport.iata_code == origin)
        )
        origin_airport = origin_result.scalar_one_or_none()
        
        # Get destination airport
        dest_result = await db.execute(
            select(Airport).where(Airport.iata_code == destination)
        )
        dest_airport = dest_result.scalar_one_or_none()
        
        if not origin_airport or not dest_airport:
            raise HTTPException(status_code=404, detail="Airport not found")
        
        # Find route
        route_result = await db.execute(
            select(Route).where(
                Route.source_airport_id == origin_airport.id,
                Route.destination_airport_id == dest_airport.id
            )
        )
        route = route_result.scalar_one_or_none()
        
        if not route:
            raise HTTPException(status_code=404, detail=f"Route {origin}-{destination} not found")
        
        # Convert to dict
        route_dict = {
            "id": route.id,
            "origin_code": origin_airport.iata_code,
            "origin_name": origin_airport.name,
            "origin_city": origin_airport.city,
            "destination_code": dest_airport.iata_code,
            "destination_name": dest_airport.name,
            "destination_city": dest_airport.city,
            "airline": route.airline,
            "distance_km": route.distance_km,
            "stops": route.stops
        }
        
        # Cache in Redis for next time
        cache_service.cache_route(origin, destination, route_dict, ttl=1800)
        
        # Publish real-time lookup event
        cache_service.publish_route_lookup(origin, destination, str(current_user.id))
        
        return {
            "status": "success",
            "source": "mariadb",
            "route": route_dict,
            "message": "Cached in Redis and published to subscribers"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== BULK POPULATION ====================

@router.post("/openflights/populate/all")
async def populate_all_openflights_data(
    airports_limit: int = Query(500, ge=1, le=3000),
    routes_limit: int = Query(500, ge=1, le=5000),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    🚀 BULK POPULATE: Cache both airports and routes in Redis
    This is the MAIN endpoint to populate Redis with OpenFlights data
    """
    try:
        cache_service = get_openflights_cache()
        
        # 1. Cache Airports
        airport_result = await db.execute(
            select(Airport).limit(airports_limit)
        )
        airports = airport_result.scalars().all()
        
        airports_data = []
        for airport in airports:
            airport_dict = {
                "id": airport.id,
                "name": airport.name,
                "city": airport.city,
                "country": airport.country,
                "iata_code": airport.iata_code,
                "icao_code": airport.icao_code,
                "latitude": float(airport.latitude) if airport.latitude else None,
                "longitude": float(airport.longitude) if airport.longitude else None,
                "altitude": airport.altitude,
                "timezone": airport.timezone
            }
            airports_data.append(airport_dict)
        
        airports_cached = cache_service.cache_multiple_airports(airports_data, ttl=3600)
        
        # 2. Cache Routes
        route_result = await db.execute(
            select(Route).limit(routes_limit)
        )
        routes = route_result.scalars().all()
        
        routes_cached = 0
        for route in routes:
            origin_result = await db.execute(
                select(Airport).where(Airport.id == route.source_airport_id)
            )
            origin_airport = origin_result.scalar_one_or_none()
            
            dest_result = await db.execute(
                select(Airport).where(Airport.id == route.destination_airport_id)
            )
            dest_airport = dest_result.scalar_one_or_none()
            
            if origin_airport and dest_airport:
                route_dict = {
                    "id": route.id,
                    "origin_code": origin_airport.iata_code,
                    "origin_name": origin_airport.name,
                    "destination_code": dest_airport.iata_code,
                    "destination_name": dest_airport.name,
                    "airline": route.airline,
                    "distance_km": route.distance_km,
                    "stops": route.stops
                }
                
                if cache_service.cache_route(
                    origin_airport.iata_code,
                    dest_airport.iata_code,
                    route_dict,
                    ttl=1800
                ):
                    routes_cached += 1
        
        # Get statistics
        stats = cache_service.get_cache_statistics()
        
        return {
            "status": "success",
            "message": "✅ Redis populated with OpenFlights data!",
            "airports_cached": airports_cached,
            "routes_cached": routes_cached,
            "cache_stats": stats,
            "next_steps": [
                "1. Refresh RedisInsight (https://ri.redis.io/13667761/browser)",
                "2. You should see keys like: airport:JFK, route:JFK:LAX",
                "3. Try GET /openflights/airports/JFK to see cached data",
                "4. Try GET /openflights/routes/JFK/LAX to see cached routes"
            ]
        }
    
    except Exception as e:
        logger.error(f"Error populating OpenFlights data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STATISTICS ====================

@router.get("/openflights/cache/stats")
async def get_openflights_cache_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Get comprehensive OpenFlights cache statistics"""
    try:
        cache_service = get_openflights_cache()
        stats = cache_service.get_cache_statistics()
        
        return {
            "status": "success",
            "cache_stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/openflights/cache/clear")
async def clear_openflights_cache(
    current_user: User = Depends(get_current_active_user)
):
    """Clear all OpenFlights cached data from Redis"""
    try:
        cache_service = get_openflights_cache()
        success = cache_service.clear_all_openflights_cache()
        
        if success:
            return {
                "status": "success",
                "message": "All OpenFlights cache cleared"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
