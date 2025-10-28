"""
WebSocket manager for real-time collaboration
"""

from app.services.redis_publisher import get_publisher, get_subscriber

from typing import Dict, List, Set, Optional
from fastapi import WebSocket
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and lifecycle"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.user_sessions: Dict[str, str] = {}  # websocket_id -> session_id
        self.connection_metadata: Dict[str, Dict] = {}  # websocket_id -> metadata

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: int,
        username: str
    ) -> str:
        """Register a new WebSocket connection"""
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = []

        connection_id = f"{user_id}_{datetime.utcnow().timestamp()}"
        self.active_connections[session_id].append(websocket)
        self.user_sessions[connection_id] = session_id
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "username": username,
            "connected_at": datetime.utcnow().isoformat(),
            "websocket": websocket
        }

        logger.info(f"User {username} connected to session {session_id}")
        return connection_id

    async def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.user_sessions:
            session_id = self.user_sessions[connection_id]
            metadata = self.connection_metadata.get(connection_id, {})

            if session_id in self.active_connections:
                websocket = metadata.get("websocket")
                if websocket in self.active_connections[session_id]:
                    self.active_connections[session_id].remove(websocket)

                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]

            del self.user_sessions[connection_id]
            del self.connection_metadata[connection_id]

            logger.info(f"User disconnected from session {session_id}")

    async def broadcast_to_room(self, session_id: str, message: dict):
        """Broadcast message to all users in a session"""
        if session_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to session {session_id}: {e}")
                    disconnected.append(websocket)

            # Clean up disconnected clients
            for ws in disconnected:
                if ws in self.active_connections[session_id]:
                    self.active_connections[session_id].remove(ws)

    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send message to specific user"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    def get_session_participants(self, session_id: str) -> List[Dict]:
        """Get list of participants in a session"""
        participants = []
        if session_id in self.active_connections:
            for connection_id, metadata in self.connection_metadata.items():
                if self.user_sessions.get(connection_id) == session_id:
                    participants.append({
                        "user_id": metadata["user_id"],
                        "username": metadata["username"],
                        "connected_at": metadata["connected_at"]
                    })

        return participants

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_connections.keys())

    async def broadcast_to_room_with_redis(
        self,
        session_id: str,
        message: dict,
        exclude_user_id: Optional[int] = None
    ):
        """Broadcast message via WebSocket and publish to Redis"""
        # Broadcast via WebSocket
        await self.broadcast_to_room(session_id, message)

        # Publish to Redis for cross-server sync
        publisher = get_publisher()
        publisher.publish_collaborative_edit(
            session_id=session_id,
            user_id="system",
            edit_type=message.get("type", "unknown"),
            edit_data=message,
            target_users=[]  # Broadcast to all
        )

    async def send_user_specific_update(
        self,
        user_id: int,
        message: dict,
        session_id: Optional[str] = None
    ):
        """Send update to specific user via WebSocket and Redis"""
        # Find user's WebSocket connection
        user_websocket = None
        user_connection_id = None

        for connection_id, metadata in self.connection_metadata.items():
            if metadata.get("user_id") == user_id:
                if session_id is None or self.user_sessions.get(connection_id) == session_id:
                    user_websocket = metadata.get("websocket")
                    user_connection_id = connection_id
                    break

        if user_websocket:
            await self.send_personal_message(user_websocket, message)

        # Also publish to user's personal Redis channel
        publisher = get_publisher()
        publisher.publish_music_update_real_time(
            session_id=f"user_{user_id}",
            user_id=str(user_id),
            update_type=message.get("type", "personal_update"),
            music_data=message
        )


class RoomManager:
    """Manages collaboration rooms and participant tracking"""

    def __init__(self):
        self.rooms: Dict[str, Dict] = {}  # session_id -> room_data

    def create_room(
        self,
        session_id: str,
        creator_id: int,
        composition_id: Optional[int] = None
    ) -> Dict:
        """Create a new collaboration room"""
        room = {
            "session_id": session_id,
            "creator_id": creator_id,
            "composition_id": composition_id,
            "participants": set(),
            "state": {},
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        self.rooms[session_id] = room
        logger.info(f"Room {session_id} created by user {creator_id}")
        return room

    def add_participant(self, session_id: str, user_id: int, username: str):
        """Add participant to room"""
        if session_id in self.rooms:
            self.rooms[session_id]["participants"].add(user_id)
            self.rooms[session_id]["last_activity"] = datetime.utcnow().isoformat()
            logger.info(f"User {username} ({user_id}) added to room {session_id}")

    def remove_participant(self, session_id: str, user_id: int):
        """Remove participant from room"""
        if session_id in self.rooms:
            self.rooms[session_id]["participants"].discard(user_id)
            self.rooms[session_id]["last_activity"] = datetime.utcnow().isoformat()

    def get_room(self, session_id: str) -> Optional[Dict]:
        """Get room information"""
        return self.rooms.get(session_id)

    def update_room_state(self, session_id: str, state_update: Dict):
        """Update room state"""
        if session_id in self.rooms:
            self.rooms[session_id]["state"].update(state_update)
            self.rooms[session_id]["last_activity"] = datetime.utcnow().isoformat()

    def close_room(self, session_id: str):
        """Close and remove a room"""
        if session_id in self.rooms:
            del self.rooms[session_id]
            logger.info(f"Room {session_id} closed")

    def get_active_rooms(self) -> List[Dict]:
        """Get list of active rooms"""
        return [
            {
                "session_id": room["session_id"],
                "creator_id": room["creator_id"],
                "participants_count": len(room["participants"]),
                "created_at": room["created_at"]
            }
            for room in self.rooms.values()
        ]


class StateSync:
    """Synchronizes composition state across clients"""

    def __init__(self):
        self.session_states: Dict[str, Dict] = {}

    def initialize_state(self, session_id: str, initial_state: Dict):
        """Initialize state for a session"""
        self.session_states[session_id] = {
            "version": 0,
            "data": initial_state,
            "history": [initial_state.copy()],
            "last_updated": datetime.utcnow().isoformat()
        }

    def update_state(self, session_id: str, updates: Dict) -> Dict:
        """Update session state with conflict resolution"""
        if session_id not in self.session_states:
            self.initialize_state(session_id, updates)
            return self.session_states[session_id]

        state = self.session_states[session_id]
        state["version"] += 1
        state["data"].update(updates)
        state["last_updated"] = datetime.utcnow().isoformat()
        state["history"].append(state["data"].copy())

        # Keep only last 100 history entries
        if len(state["history"]) > 100:
            state["history"] = state["history"][-100:]

        return state

    def get_state(self, session_id: str) -> Optional[Dict]:
        """Get current state for a session"""
        return self.session_states.get(session_id)

    def resolve_conflict(
        self,
        session_id: str,
        client_version: int,
        server_version: int,
        client_update: Dict
    ) -> Dict:
        """Resolve conflicts using operational transformation"""
        if session_id not in self.session_states:
            return {"status": "error", "message": "Session not found"}

        state = self.session_states[session_id]

        if client_version == server_version:
            # No conflict
            return self.update_state(session_id, client_update)
        else:
            # Conflict detected - server version takes precedence
            return {
                "status": "conflict",
                "server_version": state["version"],
                "server_state": state["data"],
                "message": "Version mismatch - server state is authoritative"
            }

    def clear_state(self, session_id: str):
        """Clear state for a session"""
        if session_id in self.session_states:
            del self.session_states[session_id]


class WebSocketManager:
    """Main WebSocket manager combining all components"""

    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.room_manager = RoomManager()
        self.state_sync = StateSync()

    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: int,
        username: str
    ) -> str:
        """Handle new WebSocket connection"""
        connection_id = await self.connection_manager.connect(
            websocket, session_id, user_id, username
        )

        # Create room if it doesn't exist
        room = self.room_manager.get_room(session_id)
        if room is None:
            self.room_manager.create_room(session_id, user_id, None)

        # Add participant to room
        self.room_manager.add_participant(session_id, user_id, username)

        # Notify others about new participant
        await self.connection_manager.broadcast_to_room(
            session_id,
            {
                "type": "user_joined",
                "user_id": user_id,
                "username": username,
                "participants": self.room_manager.get_room(session_id)["participants"]
            }
        )

        return connection_id

    async def handle_disconnection(self, connection_id: str, session_id: str):
        """Handle WebSocket disconnection"""
        metadata = self.connection_manager.connection_metadata.get(connection_id, {})
        user_id = metadata.get("user_id")
        username = metadata.get("username")

        await self.connection_manager.disconnect(connection_id)
        self.room_manager.remove_participant(session_id, user_id)

        # Notify others about participant leaving
        await self.connection_manager.broadcast_to_room(
            session_id,
            {
                "type": "user_left",
                "user_id": user_id,
                "username": username
            }
        )

    async def broadcast_state_update(self, session_id: str, updates: Dict):
        """Broadcast state update to all participants"""
        updated_state = self.state_sync.update_state(session_id, updates)
        await self.connection_manager.broadcast_to_room(
            session_id,
            {
                "type": "state_update",
                "state": updated_state["data"],
                "version": updated_state["version"]
            }
        )

    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a session"""
        room = self.room_manager.get_room(session_id)
        state = self.state_sync.get_state(session_id)

        return {
            "session_id": session_id,
            "room": room,
            "state": state,
            "participants": self.connection_manager.get_session_participants(session_id)
        }

    async def broadcast_activity(self, activity_data: Dict):
        """Broadcast activity update to all connected clients"""
        message = {
            "type": "activity_update",
            "data": activity_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Broadcast to all active sessions
        disconnected = []
        for session_id, websockets in self.connection_manager.active_connections.items():
            for websocket in websockets:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting activity to session {session_id}: {e}")
                    disconnected.append((session_id, websocket))

    async def broadcast_state_update_with_redis(self, session_id: str, updates: Dict):
        """Broadcast state update via WebSocket and Redis"""
        updated_state = self.state_sync.update_state(session_id, updates)

        # Broadcast via WebSocket
        await self.connection_manager.broadcast_to_room(
            session_id,
            {
                "type": "state_update",
                "state": updated_state["data"],
                "version": updated_state["version"]
            }
        )

        # Publish to Redis for cross-server sync
        publisher = get_publisher()
        publisher.publish_collaborative_edit(
            session_id=session_id,
            user_id="system",
            edit_type="state_update",
            edit_data={
                "type": "state_update",
                "state": updated_state["data"],
                "version": updated_state["version"]
            }
        )

    async def handle_real_time_music_update(
        self,
        session_id: str,
        user_id: int,
        update_type: str,
        music_data: Dict
    ):
        """Handle real-time music updates with Redis sync"""
        # Update room state
        self.room_manager.update_room_state(session_id, {
            "last_music_update": datetime.utcnow().isoformat(),
            "last_update_user": user_id,
            "update_type": update_type
        })

        # Broadcast to all participants via WebSocket and Redis
        await self.connection_manager.broadcast_to_room_with_redis(
            session_id,
            {
                "type": "music_update",
                "update_type": update_type,
                "user_id": user_id,
                "data": music_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def handle_vector_search_update(
        self,
        user_id: int,
        search_results: List[Dict],
        search_type: str = "music"
    ):
        """Handle vector search results update"""
        # Send to specific user via WebSocket and Redis
        await self.connection_manager.send_user_specific_update(
            user_id,
            {
                "type": "vector_search_results",
                "search_type": search_type,
                "results": search_results,
                "result_count": len(search_results),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def handle_generation_progress_update(
        self,
        user_id: int,
        generation_id: str,
        progress: float,
        status: str,
        current_step: Optional[str] = None
    ):
        """Handle generation progress updates"""
        await self.connection_manager.send_user_specific_update(
            user_id,
            {
                "type": "generation_progress",
                "generation_id": generation_id,
                "progress": progress,
                "status": status,
                "current_step": current_step,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Also publish to Redis for potential cross-server notifications
        publisher = get_publisher()
        publisher.publish_generation_progress(
            generation_id=generation_id,
            user_id=str(user_id),
            progress=progress,
            status=status,
            current_step=current_step
        )

    def get_enhanced_session_info(self, session_id: str) -> Dict:
        """Get enhanced session information including Redis sync status"""
        basic_info = self.get_session_info(session_id)
        room = self.room_manager.get_room(session_id)

        # Add Redis sync information
        publisher = get_publisher()
        redis_connected = publisher.redis_client is not None if publisher else False

        return {
            **basic_info,
            "redis_sync_enabled": redis_connected,
            "room_state": room.get("state", {}) if room else {},
            "last_activity": room.get("last_activity") if room else None,
            "participants_count": len(basic_info.get("participants", []))
        }
