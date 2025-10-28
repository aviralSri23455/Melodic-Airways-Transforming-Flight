"""
OpenFlights Data Caching Service
Automatically caches airport and route data from MariaDB into Redis
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import redis
from redis.exceptions import ConnectionError, TimeoutError

from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenFlightsCacheService:
    """Service to cache OpenFlights data (airports, routes) in Redis"""

    def __init__(self):
        """Initialize Redis connection for OpenFlights caching"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            self.redis_client.ping()
            logger.info("OpenFlightsCacheService: Connected successfully")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"OpenFlightsCacheService: Failed to connect to Redis: {e}")
            self.redis_client = None

    # ==================== AIRPORT CACHING ====================

    def cache_airport(self, airport_code: str, airport_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """
        Cache airport data in Redis
        
        Args:
            airport_code: IATA code (e.g., "JFK")
            airport_data: Airport information dict
            ttl: Time to live in seconds (default: 1 hour)
        
        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        try:
            key = f"airport:{airport_code}"
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(airport_data, default=str)
            )
            
            # Add to airport index
            self.redis_client.sadd("airports:cached", airport_code)
            self.redis_client.expire("airports:cached", ttl)
            
            logger.info(f"Cached airport: {airport_code}")
            return True

        except Exception as e:
            logger.error(f"Error caching airport {airport_code}: {e}")
            return False

    def get_cached_airport(self, airport_code: str) -> Optional[Dict[str, Any]]:
        """Get cached airport data"""
        if not self.redis_client:
            return None

        try:
            key = f"airport:{airport_code}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cached airport {airport_code}: {e}")
            return None

    def cache_multiple_airports(self, airports: List[Dict[str, Any]], ttl: int = 3600) -> int:
        """
        Cache multiple airports at once
        
        Returns:
            Number of airports cached
        """
        if not self.redis_client:
            return 0

        cached_count = 0
        for airport in airports:
            airport_code = airport.get("iata_code") or airport.get("code")
            if airport_code:
                if self.cache_airport(airport_code, airport, ttl):
                    cached_count += 1

        logger.info(f"Cached {cached_count} airports")
        return cached_count

    # ==================== ROUTE CACHING ====================

    def cache_route(
        self,
        origin: str,
        destination: str,
        route_data: Dict[str, Any],
        ttl: int = 1800
    ) -> bool:
        """
        Cache route data in Redis
        
        Args:
            origin: Origin airport code
            destination: Destination airport code
            route_data: Route information dict
            ttl: Time to live in seconds (default: 30 minutes)
        
        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        try:
            key = f"route:{origin}:{destination}"
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(route_data, default=str)
            )
            
            # Add to routes index
            route_pair = f"{origin}-{destination}"
            self.redis_client.sadd("routes:cached", route_pair)
            self.redis_client.expire("routes:cached", ttl)
            
            # Track popular routes
            self.redis_client.zincrby("routes:popular", 1, route_pair)
            
            logger.info(f"Cached route: {origin} -> {destination}")
            return True

        except Exception as e:
            logger.error(f"Error caching route {origin}-{destination}: {e}")
            return False

    def get_cached_route(self, origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """Get cached route data"""
        if not self.redis_client:
            return None

        try:
            key = f"route:{origin}:{destination}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cached route {origin}-{destination}: {e}")
            return None

    # ==================== MUSIC GENERATION CACHING ====================

    def cache_music_generation(
        self,
        origin: str,
        destination: str,
        music_params: Dict[str, Any],
        composition_data: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """
        Cache music generation result
        
        Args:
            origin: Origin airport code
            destination: Destination airport code
            music_params: Music parameters (tempo, scale, key, etc.)
            composition_data: Generated composition data
            ttl: Time to live in seconds (default: 1 hour)
        
        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        try:
            # Create cache key based on route and music params
            params_hash = f"{music_params.get('tempo', 120)}_{music_params.get('scale', 'major')}_{music_params.get('key', 'C')}"
            key = f"music:{origin}:{destination}:{params_hash}"
            
            cache_data = {
                "origin": origin,
                "destination": destination,
                "music_params": music_params,
                "composition": composition_data,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(cache_data, default=str)
            )
            
            # Add to music cache index
            self.redis_client.sadd("music:cached", f"{origin}-{destination}")
            self.redis_client.expire("music:cached", ttl)
            
            logger.info(f"Cached music generation: {origin} -> {destination}")
            return True

        except Exception as e:
            logger.error(f"Error caching music generation: {e}")
            return False

    def get_cached_music(
        self,
        origin: str,
        destination: str,
        music_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached music generation"""
        if not self.redis_client:
            return None

        try:
            params_hash = f"{music_params.get('tempo', 120)}_{music_params.get('scale', 'major')}_{music_params.get('key', 'C')}"
            key = f"music:{origin}:{destination}:{params_hash}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cached music: {e}")
            return None

    # ==================== EMBEDDING CACHING ====================

    def cache_embedding(
        self,
        route_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        ttl: int = 7200
    ) -> bool:
        """
        Cache route embedding vector
        
        Args:
            route_id: Route identifier
            embedding: Vector embedding
            metadata: Additional metadata
            ttl: Time to live in seconds (default: 2 hours)
        
        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        try:
            key = f"embedding:{route_id}"
            cache_data = {
                "route_id": route_id,
                "embedding": embedding,
                "metadata": metadata,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(cache_data, default=str)
            )
            
            # Add to embeddings index
            self.redis_client.sadd("embeddings:cached", route_id)
            self.redis_client.expire("embeddings:cached", ttl)
            
            logger.info(f"Cached embedding for route: {route_id}")
            return True

        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
            return False

    def get_cached_embedding(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Get cached embedding"""
        if not self.redis_client:
            return None

        try:
            key = f"embedding:{route_id}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cached embedding: {e}")
            return None

    # ==================== REAL-TIME EVENTS ====================

    def publish_route_lookup(self, origin: str, destination: str, user_id: str) -> int:
        """Publish route lookup event via Pub/Sub"""
        if not self.redis_client:
            return 0

        try:
            channel = "routes:lookup"
            message = {
                "event": "route_lookup",
                "origin": origin,
                "destination": destination,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )
            
            logger.info(f"Published route lookup event to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing route lookup: {e}")
            return 0

    def publish_music_generated(
        self,
        origin: str,
        destination: str,
        composition_id: int,
        user_id: str
    ) -> int:
        """Publish music generation event via Pub/Sub"""
        if not self.redis_client:
            return 0

        try:
            channel = "music:generated"
            message = {
                "event": "music_generated",
                "origin": origin,
                "destination": destination,
                "composition_id": composition_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )
            
            logger.info(f"Published music generation event to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing music generation: {e}")
            return 0

    # ==================== STATISTICS ====================

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self.redis_client:
            return {}

        try:
            stats = {
                "airports_cached": self.redis_client.scard("airports:cached") or 0,
                "routes_cached": self.redis_client.scard("routes:cached") or 0,
                "music_cached": self.redis_client.scard("music:cached") or 0,
                "embeddings_cached": self.redis_client.scard("embeddings:cached") or 0,
                "popular_routes": [],
                "memory_used": self.redis_client.info().get("used_memory_human", "unknown"),
                "total_keys": self.redis_client.dbsize()
            }
            
            # Get top 10 popular routes
            popular = self.redis_client.zrevrange("routes:popular", 0, 9, withscores=True)
            stats["popular_routes"] = [
                {"route": route, "lookups": int(score)}
                for route, score in popular
            ]
            
            return stats

        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}

    def clear_all_openflights_cache(self) -> bool:
        """Clear all OpenFlights cached data"""
        if not self.redis_client:
            return False

        try:
            patterns = [
                "airport:*",
                "route:*",
                "music:*",
                "embedding:*",
                "airports:cached",
                "routes:cached",
                "music:cached",
                "embeddings:cached",
                "routes:popular"
            ]
            
            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            
            logger.info("Cleared all OpenFlights cache")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


# Global instance
openflights_cache = OpenFlightsCacheService()


def get_openflights_cache() -> OpenFlightsCacheService:
    """Get the global OpenFlights cache service instance"""
    return openflights_cache
