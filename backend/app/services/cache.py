"""
Redis caching service for Aero Melody Backend
Provides caching for route-to-MIDI mappings and other frequently accessed data
"""

import json
import logging
from typing import Any, Dict, Optional, Union
import redis
from redis.exceptions import ConnectionError, TimeoutError
import hashlib
import pickle

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache service for storing route-to-MIDI mappings and other data"""

    def __init__(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def _get_cache_key(self, route_key: str, prefix: str = "route") -> str:
        """Generate a consistent cache key for a route"""
        # Use readable keys instead of hashes for easier debugging in Redis Insight
        return f"aero:{prefix}:{route_key}"

    def get_route_music(self, origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """
        Get cached music data for a route

        Args:
            origin: Origin airport code
            destination: Destination airport code

        Returns:
            Cached music data or None if not found
        """
        if not self.redis_client:
            return None

        route_key = f"{origin}:{destination}"
        cache_key = self._get_cache_key(route_key, "music")

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for route {route_key}")
                return json.loads(cached_data)
            else:
                logger.info(f"Cache miss for route {route_key}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving cached music data: {e}")
            return None

    def set_route_music(self, origin: str, destination: str, music_data: Dict[str, Any]) -> bool:
        """
        Cache music data for a route

        Args:
            origin: Origin airport code
            destination: Destination airport code
            music_data: Music metadata to cache

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False

        route_key = f"{origin}:{destination}"
        cache_key = self._get_cache_key(route_key, "music")

        try:
            # Store with TTL
            serialized_data = json.dumps(music_data, default=str)
            success = self.redis_client.setex(
                cache_key,
                settings.REDIS_CACHE_TTL if hasattr(settings, 'REDIS_CACHE_TTL') else 1800,  # 30 minutes default
                serialized_data
            )
            if success:
                logger.info(f"Cached music data for route {route_key}")
            return success
        except Exception as e:
            logger.error(f"Error caching music data: {e}")
            return False

    def get_route_embedding(self, origin: str, destination: str) -> Optional[Dict[str, Any]]:
        """
        Get cached embedding data for a route

        Args:
            origin: Origin airport code
            destination: Destination airport code

        Returns:
            Cached embedding data or None if not found
        """
        if not self.redis_client:
            return None

        route_key = f"{origin}:{destination}"
        cache_key = self._get_cache_key(route_key, "embedding")

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for embedding {route_key}")
                return pickle.loads(cached_data.encode())
            else:
                logger.info(f"Cache miss for embedding {route_key}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving cached embedding: {e}")
            return None

    def set_route_embedding(self, origin: str, destination: str, embedding_data: Dict[str, Any]) -> bool:
        """
        Cache embedding data for a route

        Args:
            origin: Origin airport code
            destination: Destination airport code
            embedding_data: Embedding data to cache

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False

        route_key = f"{origin}:{destination}"
        cache_key = self._get_cache_key(route_key, "embedding")

        try:
            # Store with TTL (embeddings can be cached longer since they don't change)
            serialized_data = pickle.dumps(embedding_data)
            success = self.redis_client.setex(
                cache_key,
                settings.REDIS_CACHE_TTL * 4 if hasattr(settings, 'REDIS_CACHE_TTL') else 7200,  # 2 hours for embeddings
                serialized_data.decode('latin1')
            )
            if success:
                logger.info(f"Cached embedding for route {route_key}")
            return success
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
            return False

    def invalidate_route_cache(self, origin: str, destination: str) -> bool:
        """
        Invalidate cache for a specific route

        Args:
            origin: Origin airport code
            destination: Destination airport code

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False

        route_key = f"{origin}:{destination}"

        try:
            # Delete both music and embedding cache for this route
            music_key = self._get_cache_key(route_key, "music")
            embedding_key = self._get_cache_key(route_key, "embedding")

            deleted_count = self.redis_client.delete(music_key, embedding_key)
            logger.info(f"Invalidated cache for route {route_key} ({deleted_count} keys)")
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Error invalidating route cache: {e}")
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get cache statistics with storage usage

        Returns:
            Dictionary with cache statistics
        """
        if not self.redis_client:
            return {"error": "Redis not connected"}

        try:
            info = self.redis_client.info()
            memory_used = info.get("used_memory", 0)
            memory_used_mb = memory_used / (1024 * 1024)  # Convert to MB
            memory_limit_mb = 30  # Free plan limit
            
            return {
                "connected": True,
                "memory_used_bytes": memory_used,
                "memory_used_mb": round(memory_used_mb, 2),
                "memory_limit_mb": memory_limit_mb,
                "memory_usage_percent": round((memory_used_mb / memory_limit_mb) * 100, 1),
                "storage_status": "CRITICAL" if memory_used_mb > 25 else "WARNING" if memory_used_mb > 20 else "OK",
                "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)),
                "keys_count": info.get("db0", {}).get("keys", 0)
            }
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {"error": str(e)}

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        return self.get_storage_info()

    def clear_all_cache(self) -> bool:
        """
        Clear all cache (use with caution)

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            self.redis_client.flushdb()
            logger.warning("Cleared all cache")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


class CacheService:
    """Generic cache service for various data types"""

    def __init__(self, redis_client=None):
        self.redis_client = None
        try:
            self.redis_client = redis_client or redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            # Test connection
            self.redis_client.ping()
            logger.info("CacheService: Redis connected successfully")
        except (ConnectionError, TimeoutError, Exception) as e:
            logger.warning(f"CacheService: Redis connection failed ({e}). Cache operations will be disabled.")
            self.redis_client = None

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.delete(key) > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False

    async def get_json(self, key: str) -> Optional[Dict]:
        """Get JSON value from cache"""
        if not self.redis_client:
            return None
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get_json error: {e}")
            return None

    async def set_json(self, key: str, value: Dict, ttl: int = 3600) -> bool:
        """Set JSON value in cache"""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"Cache set_json error: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        if not self.redis_client:
            return 0
        try:
            return self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error: {e}")
            return 0

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.redis_client:
            return 0
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear_pattern error: {e}")
            return 0


# Global cache instance
cache = None


def get_cache():
    """Get the global cache instance"""
    global cache
    if cache is None:
        cache = RedisCache()
    return cache
