"""
Redis Publisher for Real-time Updates
Handles Pub/Sub for live flight-to-music generation updates
"""

import json
import logging
from typing import Any, Dict, Optional, Callable, List
from datetime import datetime
import redis
from redis.exceptions import ConnectionError, TimeoutError
import threading

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisPublisher:
    """Publisher for real-time updates via Redis Pub/Sub"""

    def __init__(self):
        """Initialize Redis publisher"""
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
            logger.info("RedisPublisher: Connected successfully")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"RedisPublisher: Failed to connect to Redis: {e}")
            self.redis_client = None

    def publish_route_cached(
        self,
        origin: str,
        destination: str,
        cache_data: Dict[str, Any]
    ) -> int:
        """Publish route cache event"""
        if not self.redis_client:
            return 0

        try:
            channel = "routes:cached"
            message = {
                "event": "route_cached",
                "origin": origin,
                "destination": destination,
                "timestamp": datetime.utcnow().isoformat(),
                "data": cache_data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published route cache event to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing route cache: {e}")
            return 0

    def publish_music_generated(
        self,
        route_id: str,
        user_id: str,
        music_data: Dict[str, Any]
    ) -> int:
        """Publish music generation event"""
        if not self.redis_client:
            return 0

        try:
            channel = "music:generated"
            message = {
                "event": "music_generated",
                "route_id": route_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": music_data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published music generation event to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing music generated: {e}")
            return 0

    def publish_collaboration_update(
        self,
        session_id: str,
        update_type: str,
        update_data: Dict[str, Any]
    ) -> int:
        """Publish collaboration session update"""
        if not self.redis_client:
            return 0

        try:
            channel = f"collab:session:{session_id}"
            message = {
                "event": update_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": update_data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published collaboration update to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing collaboration update: {e}")
            return 0

    def publish_airport_lookup(
        self,
        airport_code: str,
        lookup_data: Dict[str, Any]
    ) -> int:
        """Publish airport lookup event (for caching)"""
        if not self.redis_client:
            return 0

        try:
            channel = "airports:lookup"
            message = {
                "event": "airport_lookup",
                "airport_code": airport_code,
                "timestamp": datetime.utcnow().isoformat(),
                "data": lookup_data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published airport lookup to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing airport lookup: {e}")
            return 0

    def publish_embedding_cached(
        self,
        route_id: str,
        embedding_data: Dict[str, Any]
    ) -> int:
        """Publish embedding cache event"""
        if not self.redis_client:
            return 0

        try:
            channel = "embeddings:cached"
            message = {
                "event": "embedding_cached",
                "route_id": route_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": embedding_data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published embedding cache event to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing embedding cache: {e}")
            return 0

    def publish_music_update_real_time(
        self,
        session_id: str,
        user_id: str,
        update_type: str,
        music_data: Dict[str, Any]
    ) -> int:
        """Publish real-time music generation update"""
        if not self.redis_client:
            return 0

        try:
            channel = f"music:realtime:{session_id}"
            message = {
                "event": "music_update",
                "update_type": update_type,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": music_data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published real-time music update to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing real-time music update: {e}")
            return 0

    def publish_vector_search_results(
        self,
        search_id: str,
        user_id: str,
        query_vector: List[float],
        results: List[Dict[str, Any]],
        search_type: str = "music"
    ) -> int:
        """Publish vector search results in real-time"""
        if not self.redis_client:
            return 0

        try:
            channel = f"vector:search:{user_id}"
            message = {
                "event": "vector_search_results",
                "search_id": search_id,
                "search_type": search_type,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "query_vector": query_vector,
                "results": results,
                "result_count": len(results)
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published vector search results to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing vector search results: {e}")
            return 0

    def publish_collaborative_edit(
        self,
        session_id: str,
        user_id: str,
        edit_type: str,
        edit_data: Dict[str, Any],
        target_users: List[str] = None
    ) -> int:
        """Publish collaborative editing update"""
        if not self.redis_client:
            return 0

        try:
            # Publish to session channel
            session_channel = f"collab:session:{session_id}"
            message = {
                "event": "collaborative_edit",
                "edit_type": edit_type,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": edit_data,
                "target_users": target_users or []
            }

            subscribers = self.redis_client.publish(
                session_channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published collaborative edit to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing collaborative edit: {e}")
            return 0

    def publish_route_music_sync(
        self,
        route_id: str,
        origin: str,
        destination: str,
        music_params: Dict[str, Any]
    ) -> int:
        """Publish route-music synchronization update"""
        if not self.redis_client:
            return 0

        try:
            channel = "route:music:sync"
            message = {
                "event": "route_music_sync",
                "route_id": route_id,
                "origin": origin,
                "destination": destination,
                "timestamp": datetime.utcnow().isoformat(),
                "music_params": music_params
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published route-music sync to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing route-music sync: {e}")
            return 0

    def publish_generation_progress(
        self,
        generation_id: str,
        user_id: str,
        progress: float,
        status: str,
        current_step: str = None
    ) -> int:
        """Publish music generation progress updates"""
        if not self.redis_client:
            return 0

        try:
            channel = f"generation:progress:{user_id}"
            message = {
                "event": "generation_progress",
                "generation_id": generation_id,
                "user_id": user_id,
                "progress": progress,
                "status": status,
                "current_step": current_step,
                "timestamp": datetime.utcnow().isoformat()
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published generation progress to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing generation progress: {e}")
            return 0

    def publish_system_status(
        self,
        status_type: str,
        status_data: Dict[str, Any]
    ) -> int:
        """Publish system status updates (performance, health, etc.)"""
        if not self.redis_client:
            return 0

        try:
            channel = "system:status"
            message = {
                "event": "system_status",
                "status_type": status_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": status_data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published system status to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing system status: {e}")
            return 0

    def publish_generic(
        self,
        channel: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> int:
        """Publish generic event to channel"""
        if not self.redis_client:
            return 0

        try:
            message = {
                "event": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published {event_type} to {subscribers} subscribers on {channel}")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing generic event: {e}")
            return 0


class RedisSubscriber:
    """Subscriber for real-time updates via Redis Pub/Sub"""

    def __init__(self):
        """Initialize Redis subscriber"""
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
            self.pubsub = self.redis_client.pubsub()
            self.handlers: Dict[str, List[Callable]] = {}
            logger.info("RedisSubscriber: Connected successfully")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"RedisSubscriber: Failed to connect to Redis: {e}")
            self.redis_client = None
            self.pubsub = None

    def subscribe(self, channel: str, handler: Callable) -> bool:
        """Subscribe to a channel with a handler"""
        if not self.pubsub:
            return False

        try:
            if channel not in self.handlers:
                self.handlers[channel] = []
                self.pubsub.subscribe(channel)

            self.handlers[channel].append(handler)
            logger.info(f"Subscribed to channel: {channel}")
            return True

        except Exception as e:
            logger.error(f"Error subscribing to channel: {e}")
            return False

    def unsubscribe(self, channel: str) -> bool:
        """Unsubscribe from a channel"""
        if not self.pubsub:
            return False

        try:
            self.pubsub.unsubscribe(channel)
            if channel in self.handlers:
                del self.handlers[channel]

            logger.info(f"Unsubscribed from channel: {channel}")
            return True

        except Exception as e:
            logger.error(f"Error unsubscribing from channel: {e}")
            return False

    def listen(self) -> None:
        """Listen for messages (blocking)"""
        if not self.pubsub:
            return

        try:
            for message in self.pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    data = json.loads(message["data"])

                    if channel in self.handlers:
                        for handler in self.handlers[channel]:
                            try:
                                handler(data)
                            except Exception as e:
                                logger.error(f"Error in message handler: {e}")

        except Exception as e:
            logger.error(f"Error listening for messages: {e}")

    def listen_async(self) -> None:
        """Listen for messages in a background thread"""
        if not self.pubsub:
            return

        thread = threading.Thread(target=self.listen, daemon=True)
        thread.start()
        logger.info("Started async message listener")


# Global publisher and subscriber instances
publisher = None
subscriber = None


class FallbackRedisPublisher:
    """Fallback publisher when Redis connection fails"""

    def __init__(self):
        logger.warning("Using fallback Redis publisher - real-time features disabled")

    def publish_route_cached(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_music_generated(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_collaboration_update(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_airport_lookup(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_embedding_cached(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_music_update_real_time(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_vector_search_results(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_collaborative_edit(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_route_music_sync(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_generation_progress(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_system_status(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0

    def publish_generic(self, *args, **kwargs):
        """Fallback - return 0 subscribers"""
        return 0


class FallbackRedisSubscriber:
    """Fallback subscriber when Redis connection fails"""

    def __init__(self):
        logger.warning("Using fallback Redis subscriber")

    def subscribe(self, *args, **kwargs):
        """Fallback - return False"""
        return False

    def unsubscribe(self, *args, **kwargs):
        """Fallback - return False"""
        return False

    def listen(self):
        """Fallback - do nothing"""
        pass

    def listen_async(self):
        """Fallback - do nothing"""
        pass


def get_publisher():
    """Get the global publisher instance"""
    global publisher
    if publisher is None:
        try:
            publisher = RedisPublisher()
        except Exception as e:
            logger.error(f"Failed to initialize Redis publisher: {e}")
            # Use fallback service instead of breaking the router
            publisher = FallbackRedisPublisher()
    return publisher


def get_subscriber():
    """Get the global subscriber instance"""
    global subscriber
    if subscriber is None:
        try:
            subscriber = RedisSubscriber()
        except Exception as e:
            logger.error(f"Failed to initialize Redis subscriber: {e}")
            # Use fallback service instead of breaking the router
            subscriber = FallbackRedisSubscriber()
    return subscriber
