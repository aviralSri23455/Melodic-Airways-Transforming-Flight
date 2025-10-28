"""
WebSocket Demo for Redis Pub/Sub Testing
Allows real-time testing of Redis Pub/Sub functionality
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
from typing import List
try:
    import redis.asyncio as redis
except ImportError:
    import redis
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.redis_client = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Initialize Redis connection if not exists
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                # Skip ping test to avoid async/sync issues - connection will be tested when used
                logger.info("Redis connection established for WebSocket")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def subscribe_to_redis(self):
        """Subscribe to Redis channels and broadcast to WebSocket clients"""
        if not self.redis_client:
            return
            
        try:
            pubsub = self.redis_client.pubsub()
            
            # Subscribe to demo channels
            channels = [
                "music:generated",
                "demo:test", 
                "system:status",
                "vector:search:public",
                "generation:progress:public"
            ]
            
            if hasattr(pubsub, 'subscribe') and asyncio.iscoroutinefunction(pubsub.subscribe):
                await pubsub.subscribe(*channels)
            else:
                pubsub.subscribe(*channels)
            
            logger.info("Subscribed to Redis channels via WebSocket")
            
            # Handle both async and sync Redis clients
            if hasattr(pubsub, 'listen') and asyncio.iscoroutinefunction(pubsub.listen):
                async for message in pubsub.listen():
                    await self._handle_redis_message(message)
            else:
                # For sync Redis, we need to poll
                while len(self.active_connections) > 0:
                    try:
                        # Check if get_message is async or sync
                        if asyncio.iscoroutinefunction(pubsub.get_message):
                            message = await pubsub.get_message(timeout=1.0)
                        else:
                            message = pubsub.get_message(timeout=1.0)
                        if message:
                            await self._handle_redis_message(message)
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Error polling Redis: {e}")
                        break
                    
        except Exception as e:
            logger.error(f"Redis subscription error: {e}")
    
    async def _handle_redis_message(self, message):
        """Handle Redis message and broadcast to WebSocket clients"""
        if message and message.get("type") == "message":
            channel = message.get("channel", "unknown")
            data = message.get("data", "")
            
            # Format message for WebSocket clients
            ws_message = {
                "type": "redis_message",
                "channel": channel,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "subscriber_count": len(self.active_connections)
            }
            
            await self.broadcast(json.dumps(ws_message))
            logger.info(f"Broadcasted Redis message from {channel} to {len(self.active_connections)} WebSocket clients")

manager = ConnectionManager()

@router.websocket("/redis-subscriber")
async def websocket_redis_subscriber(websocket: WebSocket):
    """
    🔌 WebSocket Redis Subscriber
    
    Connect to this WebSocket to receive real-time Redis Pub/Sub messages.
    This demonstrates how frontend applications can subscribe to Redis channels.
    
    Usage:
    - Connect to: ws://localhost:8000/api/v1/demo/redis-subscriber
    - Send any message to keep connection alive
    - Receive Redis Pub/Sub messages in real-time
    """
    await manager.connect(websocket)
    
    # Send welcome message
    welcome_msg = {
        "type": "connection_established",
        "message": "Connected to Redis Pub/Sub via WebSocket",
        "subscribed_channels": [
            "music:generated",
            "demo:test", 
            "system:status",
            "vector:search:public",
            "generation:progress:public"
        ],
        "instructions": [
            "Now run: GET /api/v1/demo/complete-demo?origin=DEL&destination=LHR",
            "Or run: GET /api/v1/demo/test-redis-subscriber",
            "You'll see Redis messages appear here in real-time!"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await manager.send_personal_message(json.dumps(welcome_msg), websocket)
    
    # Start Redis subscription in background
    if len(manager.active_connections) == 1:  # First connection starts the subscriber
        asyncio.create_task(manager.subscribe_to_redis())
    
    try:
        # Give more time for connection to fully establish
        await asyncio.sleep(0.5)
        
        while True:
            # Keep connection alive and handle client messages
            try:
                # Check if WebSocket is still connected before receiving
                if websocket.client_state.name != 'CONNECTED':
                    break
                    
                # Use timeout to prevent hanging
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Echo back client message
                response = {
                    "type": "echo",
                    "your_message": data,
                    "active_subscribers": len(manager.active_connections),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await manager.send_personal_message(json.dumps(response), websocket)
                
            except asyncio.TimeoutError:
                # Send keepalive message
                keepalive = {
                    "type": "keepalive",
                    "active_subscribers": len(manager.active_connections),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await manager.send_personal_message(json.dumps(keepalive), websocket)
                
            except Exception as e:
                # Handle any WebSocket-specific errors
                logger.info(f"WebSocket client disconnected: {e}")
                break
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(manager.active_connections)}")
    except Exception as e:
        logger.info(f"WebSocket connection ended: {e}")
        manager.disconnect(websocket)


@router.get("/websocket-info")
async def get_websocket_info():
    """
    📡 WebSocket Redis Subscriber Information
    
    Instructions for testing Redis Pub/Sub with WebSocket
    """
    return {
        "websocket_endpoint": "ws://localhost:8000/api/v1/demo/redis-subscriber",
        "purpose": "Real-time Redis Pub/Sub message subscriber for testing",
        "how_to_test": [
            "1. Connect to the WebSocket endpoint above",
            "2. Run: GET /api/v1/demo/complete-demo?origin=DEL&destination=LHR", 
            "3. Run: GET /api/v1/demo/test-redis-subscriber",
            "4. Watch Redis messages appear in real-time via WebSocket"
        ],
        "subscribed_channels": [
            "music:generated - New music compositions",
            "demo:test - Test messages",
            "system:status - System health updates", 
            "vector:search:public - Vector search results",
            "generation:progress:public - Music generation progress"
        ],
        "javascript_example": '''
const ws = new WebSocket('ws://localhost:8000/api/v1/demo/redis-subscriber');

ws.onopen = function() {
    console.log('Connected to Redis Pub/Sub via WebSocket');
    ws.send('Hello from browser!');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Redis message received:', message);
};
        ''',
        "python_example": '''
import asyncio
import websockets
import json

async def subscribe_to_redis():
    uri = "ws://localhost:8000/api/v1/demo/redis-subscriber"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello from Python!")
        
        async for message in websocket:
            data = json.loads(message)
            print(f"Redis message: {data}")

asyncio.run(subscribe_to_redis())
        ''',
        "current_status": {
            "active_websocket_connections": len(manager.active_connections),
            "redis_connected": manager.redis_client is not None
        }
    }


@router.websocket("/simple-websocket-test")
async def simple_websocket_test(websocket: WebSocket):
    """
    🔌 Simple WebSocket Test - Minimal Implementation
    
    A basic WebSocket endpoint for testing connectivity without Redis complexity.
    Use this for frontend testing when you don't need Redis Pub/Sub.
    """
    try:
        await websocket.accept()
        
        # Send welcome message
        welcome = {
            "type": "connection_established",
            "message": "Simple WebSocket connection successful",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "ready"
        }
        await websocket.send_text(json.dumps(welcome))
        
        # Simple echo loop
        message_count = 0
        while True:
            try:
                # Wait for client message with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_count += 1
                
                # Echo back with counter
                response = {
                    "type": "echo",
                    "message_count": message_count,
                    "your_message": data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(response))
                
            except asyncio.TimeoutError:
                # Send keepalive
                keepalive = {
                    "type": "keepalive",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(keepalive))
                
            except Exception as e:
                logger.info(f"WebSocket client disconnected: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.get("/simple-websocket-info")
async def get_simple_websocket_info():
    """
    📡 Simple WebSocket Information
    
    Information about the simple WebSocket test endpoint
    """
    return {
        "websocket_endpoint": "ws://localhost:8000/api/v1/demo/simple-websocket-test",
        "purpose": "Simple WebSocket connectivity test without Redis complexity",
        "features": [
            "✅ Basic connection test",
            "✅ Echo functionality", 
            "✅ Message counter",
            "✅ Keepalive messages",
            "✅ Graceful disconnection"
        ],
        "javascript_example": '''
const ws = new WebSocket('ws://localhost:8000/api/v1/demo/simple-websocket-test');

ws.onopen = function() {
    console.log('Connected to simple WebSocket');
    ws.send('Hello from browser!');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};

ws.onclose = function() {
    console.log('WebSocket connection closed');
};
        ''',
        "testing_steps": [
            "1. Connect to the WebSocket endpoint above",
            "2. Send any text message",
            "3. Receive echo response with message counter",
            "4. Connection stays alive with keepalive messages"
        ]
    }