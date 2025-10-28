"""
Redis Session Manager for Live Collaboration
Manages real-time flight-to-music generation sessions with live updates
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
import redis
from redis.exceptions import ConnectionError, TimeoutError
import uuid

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisSessionManager:
    """Manages live collaboration sessions in Redis"""

    def __init__(self):
        """Initialize Redis connection for session management"""
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
            logger.info("RedisSessionManager: Connected successfully")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"RedisSessionManager: Failed to connect to Redis: {e}")
            self.redis_client = None

    def create_session_with_storage_check(
        self,
        user_id: str,
        origin: str,
        destination: str,
        session_type: str = "generation"
    ) -> Optional[str]:
        """
        Create session with storage limit check

        Args:
            user_id: User ID initiating the session
            origin: Origin airport code
            destination: Destination airport code
            session_type: Type of session (generation, collaboration, remix)

        Returns:
            Session ID or None if failed
        """
        if not self.redis_client:
            return None

        # Check storage usage before creating
        try:
            info = self.redis_client.info()
            memory_used = info.get("used_memory", 0) / (1024 * 1024)  # MB

            # If over 25MB, try cleanup first
            if memory_used > 25:
                self._cleanup_old_sessions()
                info = self.redis_client.info()
                memory_used = info.get("used_memory", 0) / (1024 * 1024)

                # If still over limit, reject
                if memory_used > 28:
                    logger.warning(f"Redis storage limit approaching. Used: {memory_used:.1f}MB")
                    return None
        except Exception as e:
            logger.error(f"Error checking storage: {e}")

        # Create session normally
        return self.create_session_basic(user_id, origin, destination, session_type)

    def create_session_basic(
        self,
        user_id: str,
        origin: str,
        destination: str,
        session_type: str = "generation"
    ) -> Optional[str]:
        """
        Create a basic session without storage checks

        Args:
            user_id: User ID initiating the session
            origin: Origin airport code
            destination: Destination airport code
            session_type: Type of session (generation, collaboration, remix)

        Returns:
            Session ID or None if failed
        """
        if not self.redis_client:
            return None

        try:
            session_id = str(uuid.uuid4())
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "origin": origin,
                "destination": destination,
                "session_type": session_type,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active",
                "participants": [user_id],
                "tempo": settings.DEFAULT_TEMPO,
                "scale": settings.DEFAULT_SCALE,
                "key": settings.DEFAULT_KEY,
                "edits": [],
                "midi_data": None
            }

            session_key = f"session:{session_id}"
            ttl = settings.REDIS_SESSION_TTL if hasattr(settings, 'REDIS_SESSION_TTL') else 7200  # 2 hours default

            self.redis_client.setex(
                session_key,
                ttl,
                json.dumps(session_data, default=str)
            )

            # Add to active sessions set
            self.redis_client.sadd("active_sessions", session_id)
            self.redis_client.expire("active_sessions", ttl)

            # Add to user's sessions
            user_sessions_key = f"user_sessions:{user_id}"
            self.redis_client.sadd(user_sessions_key, session_id)
            self.redis_client.expire(user_sessions_key, ttl)

            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None

    def _cleanup_old_sessions(self) -> int:
        """Clean up old sessions to free space"""
        try:
            cleanup_count = 0
            active_sessions = self.get_active_sessions()

            for session in active_sessions:
                # Check if session is old (more than 2 hours)
                created_at = session.get('created_at')
                if created_at:
                    from datetime import datetime
                    try:
                        created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        # Close sessions older than 2 hours
                        if (datetime.now() - created).seconds > 7200:
                            self.close_session(session['session_id'])
                            cleanup_count += 1
                    except:
                        pass

            # Also clean up user session sets that might be empty
            user_sessions_keys = self.redis_client.keys("user_sessions:*")
            for key in user_sessions_keys:
                if self.redis_client.scard(key) == 0:
                    self.redis_client.delete(key)
                    cleanup_count += 1

            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old sessions/sets")
            return cleanup_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if not self.redis_client:
            return None

        try:
            session_key = f"session:{session_id}"
            session_data = self.redis_client.get(session_key)
            return json.loads(session_data) if session_data else None
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None

    def update_session_music(
        self,
        session_id: str,
        tempo: int,
        scale: str,
        key: str,
        midi_data: Optional[str] = None
    ) -> bool:
        """Update session music parameters"""
        if not self.redis_client:
            return False

        try:
            session = self.get_session(session_id)
            if not session:
                return False

            session["tempo"] = tempo
            session["scale"] = scale
            session["key"] = key
            if midi_data:
                session["midi_data"] = midi_data

            session_key = f"session:{session_id}"
            ttl = settings.REDIS_SESSION_TTL if hasattr(settings, 'REDIS_SESSION_TTL') else 7200  # 2 hours default
            self.redis_client.setex(
                session_key,
                ttl,
                json.dumps(session, default=str)
            )

            # Publish update event
            self.publish_session_update(session_id, "music_updated", session)

            logger.info(f"Updated music for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating session music: {e}")
            return False

    def add_session_edit(
        self,
        session_id: str,
        user_id: str,
        edit_type: str,
        edit_data: Dict[str, Any]
    ) -> bool:
        """Add an edit to session history"""
        if not self.redis_client:
            return False

        try:
            session = self.get_session(session_id)
            if not session:
                return False

            edit = {
                "user_id": user_id,
                "type": edit_type,
                "data": edit_data,
                "timestamp": datetime.utcnow().isoformat()
            }

            session["edits"].append(edit)

            session_key = f"session:{session_id}"
            ttl = settings.REDIS_SESSION_TTL if hasattr(settings, 'REDIS_SESSION_TTL') else 7200  # 2 hours default
            self.redis_client.setex(
                session_key,
                ttl,
                json.dumps(session, default=str)
            )

            # Publish edit event
            self.publish_session_update(session_id, "edit_made", edit)

            logger.info(f"Added edit to session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding session edit: {e}")
            return False

    def add_participant(self, session_id: str, user_id: str) -> bool:
        """Add participant to session"""
        if not self.redis_client:
            return False

        try:
            session = self.get_session(session_id)
            if not session:
                return False

            if user_id not in session["participants"]:
                session["participants"].append(user_id)

                session_key = f"session:{session_id}"
                ttl = settings.REDIS_SESSION_TTL if hasattr(settings, 'REDIS_SESSION_TTL') else 7200  # 2 hours default
                self.redis_client.setex(
                    session_key,
                    ttl,
                    json.dumps(session, default=str)
                )

                # Publish participant joined event
                self.publish_session_update(
                    session_id,
                    "participant_joined",
                    {"user_id": user_id, "participants": session["participants"]}
                )

                logger.info(f"Added participant {user_id} to session {session_id}")

            return True

        except Exception as e:
            logger.error(f"Error adding participant: {e}")
            return False

    def close_session(self, session_id: str) -> bool:
        """Close a session"""
        if not self.redis_client:
            return False

        try:
            session = self.get_session(session_id)
            if not session:
                return False

            session["status"] = "closed"
            session["closed_at"] = datetime.utcnow().isoformat()

            session_key = f"session:{session_id}"
            ttl = settings.REDIS_SESSION_TTL if hasattr(settings, 'REDIS_SESSION_TTL') else 7200  # 2 hours default
            self.redis_client.setex(
                session_key,
                ttl,
                json.dumps(session, default=str)
            )

            # Remove from active sessions
            self.redis_client.srem("active_sessions", session_id)

            logger.info(f"Closed session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error closing session: {e}")
            return False

    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        if not self.redis_client:
            return []

        try:
            user_sessions_key = f"user_sessions:{user_id}"
            session_ids = self.redis_client.smembers(user_sessions_key)

            sessions = []
            for session_id in session_ids:
                session = self.get_session(session_id)
                if session:
                    sessions.append(session)

            return sessions

        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions"""
        if not self.redis_client:
            return []

        try:
            session_ids = self.redis_client.smembers("active_sessions")
            sessions = []

            for session_id in session_ids:
                session = self.get_session(session_id)
                if session and session.get("status") == "active":
                    sessions.append(session)

            return sessions

        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []

    def publish_session_update(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> int:
        """Publish session update via Pub/Sub"""
        if not self.redis_client:
            return 0

        try:
            channel = f"session:{session_id}"
            message = {
                "event": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }

            subscribers = self.redis_client.publish(
                channel,
                json.dumps(message, default=str)
            )

            logger.info(f"Published {event_type} to {subscribers} subscribers")
            return subscribers

        except Exception as e:
            logger.error(f"Error publishing session update: {e}")
            return 0

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        if not self.redis_client:
            return {}

        try:
            active_sessions = self.redis_client.scard("active_sessions")
            info = self.redis_client.info()

            return {
                "active_sessions": active_sessions,
                "memory_used": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands": info.get("total_commands_processed", 0)
            }

        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}


# Global session manager instance
session_manager = None


def get_session_manager():
    """Get the global session manager instance"""
    global session_manager
    if session_manager is None:
        session_manager = RedisSessionManager()
    return session_manager
