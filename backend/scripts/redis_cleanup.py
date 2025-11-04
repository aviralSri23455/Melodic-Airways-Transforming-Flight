#!/usr/bin/env python3
"""
Redis Storage Cleanup and Optimization Script
For Aero Melody Backend - 30MB Free Plan Management

This script helps manage Redis storage to stay within the 30MB free plan limit.
Run this periodically to clean up old data and optimize storage usage.

Usage:
    python scripts/redis_cleanup.py --dry-run  # See what would be cleaned
    python scripts/redis_cleanup.py            # Actually clean up
    python scripts/redis_cleanup.py --analyze  # Analyze storage patterns
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import redis
from redis.exceptions import ConnectionError, TimeoutError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedisStorageManager:
    """Manages Redis storage for the 30MB free plan"""

    def __init__(self, redis_url: str):
        """Initialize Redis connection"""
        self.redis_url = redis_url
        self.redis_client = None
        self.connect()

    def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=10
            )
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def get_storage_info(self) -> Dict[str, Any]:
        """Get comprehensive storage information"""
        if not self.redis_client:
            return {"error": "Redis not connected"}

        try:
            info = self.redis_client.info()
            memory_used = info.get("used_memory", 0)
            memory_used_mb = memory_used / (1024 * 1024)
            memory_limit_mb = 30

            # Get key distribution
            key_patterns = {
                "sessions": len(self.redis_client.keys("session:*")),
                "active_sessions": self.redis_client.scard("active_sessions") if self.redis_client.exists("active_sessions") else 0,
                "user_sessions": len(self.redis_client.keys("user_sessions:*")),
                "music_cache": len(self.redis_client.keys("music:*")),
                "airport_cache": len(self.redis_client.keys("airport:*")),
                "route_cache": len(self.redis_client.keys("route:*")),
                "embedding_cache": len(self.redis_client.keys("embedding:*")),
                "total_keys": info.get("db0", {}).get("keys", 0)
            }

            return {
                "connected": True,
                "memory_used_bytes": memory_used,
                "memory_used_mb": round(memory_used_mb, 2),
                "memory_limit_mb": memory_limit_mb,
                "memory_usage_percent": round((memory_used_mb / memory_limit_mb) * 100, 1),
                "storage_status": "CRITICAL" if memory_used_mb > 25 else "WARNING" if memory_used_mb > 20 else "OK",
                "key_distribution": key_patterns,
                "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands": info.get("total_commands_processed", 0)
            }

        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {"error": str(e)}

    def analyze_key_sizes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze the size of different key types"""
        if not self.redis_client:
            return {}

        analysis = {
            "sessions": [],
            "cache": [],
            "other": []
        }

        try:
            # Get all keys
            all_keys = self.redis_client.keys("*")
            total_size = 0

            for key in all_keys[:100]:  # Sample first 100 keys
                key_type = self.redis_client.type(key)
                key_ttl = self.redis_client.ttl(key)

                if key.startswith("session:"):
                    category = "sessions"
                elif any(key.startswith(prefix) for prefix in ["music:", "airport:", "route:", "embedding:"]):
                    category = "cache"
                else:
                    category = "other"

                # Get approximate size
                if key_type == "string":
                    size = len(self.redis_client.get(key) or "")
                elif key_type == "set":
                    size = len(self.redis_client.smembers(key))
                elif key_type == "zset":
                    size = self.redis_client.zcard(key)
                else:
                    size = 0

                analysis[category].append({
                    "key": key,
                    "type": key_type,
                    "size": size,
                    "ttl": key_ttl,
                    "category": category
                })

                total_size += size

            # Sort by size
            for category in analysis:
                analysis[category].sort(key=lambda x: x["size"], reverse=True)

            logger.info(f"Analyzed {len(all_keys)} keys, estimated total size: {total_size} bytes")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing keys: {e}")
            return {}

    def cleanup_old_sessions(self, max_age_hours: int = 2) -> int:
        """Clean up sessions older than specified hours"""
        if not self.redis_client:
            return 0

        try:
            cleanup_count = 0
            active_sessions = self.redis_client.smembers("active_sessions")

            for session_id in active_sessions:
                session_key = f"session:{session_id}"
                session_data = self.redis_client.get(session_key)

                if session_data:
                    try:
                        session = json.loads(session_data)
                        created_at = session.get('created_at')

                        if created_at:
                            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            age_hours = (datetime.now() - created).total_seconds() / 3600

                            if age_hours > max_age_hours:
                                # Close old session
                                self.redis_client.srem("active_sessions", session_id)
                                user_id = session.get('user_id')
                                if user_id:
                                    user_sessions_key = f"user_sessions:{user_id}"
                                    self.redis_client.srem(user_sessions_key, session_id)

                                cleanup_count += 1
                                logger.info(f"Cleaned up old session: {session_id} (age: {age_hours:.1f}h)")

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid session data for {session_id}, removing")
                        self.redis_client.delete(session_key)
                        cleanup_count += 1

            logger.info(f"Cleaned up {cleanup_count} old sessions")
            return cleanup_count

        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0

    def cleanup_empty_sets(self) -> int:
        """Clean up empty user session sets"""
        if not self.redis_client:
            return 0

        try:
            cleanup_count = 0
            user_sessions_keys = self.redis_client.keys("user_sessions:*")

            for key in user_sessions_keys:
                if self.redis_client.scard(key) == 0:
                    self.redis_client.delete(key)
                    cleanup_count += 1

            logger.info(f"Cleaned up {cleanup_count} empty user session sets")
            return cleanup_count

        except Exception as e:
            logger.error(f"Error cleaning up empty sets: {e}")
            return 0

    def optimize_cache_ttl(self) -> int:
        """Optimize TTL for cache entries to save space"""
        if not self.redis_client:
            return 0

        try:
            optimized_count = 0

            # Reduce TTL for old music cache (from 1h to 30min if not accessed recently)
            music_keys = self.redis_client.keys("music:*")
            for key in music_keys:
                # If key has been idle for a while, reduce its TTL
                if self.redis_client.object("idletime", key) and self.redis_client.object("idletime", key) > 1800:  # 30 minutes
                    self.redis_client.expire(key, 1800)  # 30 minutes
                    optimized_count += 1

            logger.info(f"Optimized TTL for {optimized_count} cache entries")
            return optimized_count

        except Exception as e:
            logger.error(f"Error optimizing cache TTL: {e}")
            return 0

    def run_cleanup(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run comprehensive cleanup"""
        logger.info(f"Starting Redis cleanup {'(DRY RUN)' if dry_run else '(LIVE)'}")

        # Get initial state
        initial_info = self.get_storage_info()

        cleanup_summary = {
            "initial_storage": initial_info,
            "actions_taken": [],
            "sessions_cleaned": 0,
            "sets_cleaned": 0,
            "cache_optimized": 0,
            "space_freed_mb": 0
        }

        if dry_run:
            logger.info("DRY RUN - No actual changes will be made")
            # Just analyze what would be cleaned
            analysis = self.analyze_key_sizes()
            logger.info(f"Would clean up {len(analysis.get('sessions', []))} old sessions")
            logger.info(f"Would clean up {len(analysis.get('cache', []))} cache entries")
        else:
            # Actually perform cleanup
            sessions_cleaned = self.cleanup_old_sessions(max_age_hours=2)
            sets_cleaned = self.cleanup_empty_sets()
            cache_optimized = self.optimize_cache_ttl()

            cleanup_summary["sessions_cleaned"] = sessions_cleaned
            cleanup_summary["sets_cleaned"] = sets_cleaned
            cleanup_summary["cache_optimized"] = cache_optimized

        # Get final state
        final_info = self.get_storage_info()
        cleanup_summary["final_storage"] = final_info

        # Calculate space freed
        if not dry_run:
            initial_mb = initial_info.get("memory_used_mb", 0)
            final_mb = final_info.get("memory_used_mb", 0)
            cleanup_summary["space_freed_mb"] = round(initial_mb - final_mb, 2)

        logger.info(f"Cleanup completed. Space freed: {cleanup_summary['space_freed_mb']} MB")
        return cleanup_summary

def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description="Redis Storage Cleanup for Aero Melody")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without doing it")
    parser.add_argument("--analyze", action="store_true", help="Analyze storage usage without cleanup")
    parser.add_argument("--redis-url", default=None, help="Redis URL (default: from env)")
    parser.add_argument("--max-age", type=int, default=2, help="Max session age in hours (default: 2)")

    args = parser.parse_args()

    # Get Redis URL from environment or command line
    redis_url = args.redis_url or os.getenv("REDIS_URL", "redis://default:your_password@localhost:6379")

    manager = RedisStorageManager(redis_url)

    if not manager.redis_client:
        logger.error("Cannot connect to Redis. Exiting.")
        return 1

    # Show current storage info
    storage_info = manager.get_storage_info()
    logger.info(f"Current storage usage: {storage_info.get('memory_used_mb', 'unknown')}MB / {storage_info.get('memory_limit_mb', 30)}MB")
    logger.info(f"Status: {storage_info.get('storage_status', 'UNKNOWN')}")

    if args.analyze:
        # Just analyze
        logger.info("=== STORAGE ANALYSIS ===")
        analysis = manager.analyze_key_sizes()

        for category, keys in analysis.items():
            if keys:
                logger.info(f"{category.upper()}: {len(keys)} keys")
                if len(keys) > 0:
                    largest = keys[0]
                    logger.info(f"  Largest {category} key: {largest['key']} ({largest['size']} bytes)")

    else:
        # Run cleanup
        cleanup_summary = manager.run_cleanup(dry_run=args.dry_run)

        logger.info("=== CLEANUP SUMMARY ===")
        logger.info(f"Sessions cleaned: {cleanup_summary['sessions_cleaned']}")
        logger.info(f"Sets cleaned: {cleanup_summary['sets_cleaned']}")
        logger.info(f"Cache optimized: {cleanup_summary['cache_optimized']}")
        logger.info(f"Space freed: {cleanup_summary['space_freed_mb']} MB")

        final_info = cleanup_summary['final_storage']
        logger.info(f"Final storage: {final_info.get('memory_used_mb', 'unknown')}MB ({final_info.get('memory_usage_percent', 0)}%)")

        if final_info.get('memory_usage_percent', 0) > 90:
            logger.warning("⚠️  Storage usage is still very high! Consider upgrading your Redis plan.")
        elif final_info.get('memory_usage_percent', 0) > 80:
            logger.warning("⚠️  Storage usage is high. Monitor closely.")
        else:
            logger.info("✅ Storage usage is now at a safe level.")

    return 0

if __name__ == "__main__":
    exit(main())
