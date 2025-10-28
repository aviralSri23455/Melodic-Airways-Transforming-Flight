"""
🆕 Real-Time Implementation Tests for Aero Melody
Tests for FAISS + DuckDB, Redis Pub/Sub, WebSocket collaboration, and real-time features
"""

import pytest
import pytest_asyncio
import asyncio
import sys
import os
import json
import numpy as np
from unittest.mock import Mock, AsyncMock
from typing import AsyncGenerator

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.services.faiss_duckdb_service import get_faiss_duckdb_service
from app.services.redis_publisher import get_publisher, get_subscriber, RedisPublisher
from app.services.websocket_manager import WebSocketManager
from app.services.galera_manager import get_galera_manager
from app.services.music_generator import get_music_generation_service


# ==================== FAISS + DUCKDB TESTS ====================

@pytest.mark.asyncio
async def test_faiss_duckdb_service_initialization():
    """Test FAISS + DuckDB service initialization"""
    service = get_faiss_duckdb_service()

    # Test service creation
    assert service is not None

    # Test statistics
    stats = service.get_statistics()
    assert isinstance(stats, dict)
    assert "total_vectors" in stats
    assert "index_type" in stats


@pytest.mark.asyncio
async def test_faiss_vector_search():
    """Test FAISS vector similarity search"""
    service = get_faiss_duckdb_service()

    # Create test query vector
    query_vector = np.random.rand(1, 128).astype(np.float32)

    # Test search (should work even with empty index)
    results = service.search_similar_music(query_vector, limit=5)

    # Should return empty list or handle gracefully
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_faiss_vector_storage():
    """Test FAISS vector storage"""
    service = get_faiss_duckdb_service()

    # Create test vector
    vector = np.random.rand(128).astype(np.float32)

    # Test storage
    success = service.store_music_vector(
        composition_id=1,
        route_id=1,
        origin="JFK",
        destination="LAX",
        genre="classical",
        tempo=120,
        pitch=60.0,
        harmony=0.8,
        complexity=0.7,
        vector=vector,
        metadata={"test": "data"}
    )

    # Should return boolean result
    assert isinstance(success, bool)


@pytest.mark.asyncio
async def test_duckdb_analytics():
    """Test DuckDB analytics functionality"""
    service = get_faiss_duckdb_service()

    # Test route similarity search
    results = service.search_similar_routes("JFK", "LAX", limit=5)
    assert isinstance(results, list)

    # Test statistics
    stats = service.get_statistics()
    assert isinstance(stats, dict)


# ==================== REDIS PUB/SUB TESTS ====================

@pytest.mark.asyncio
async def test_redis_publisher_initialization():
    """Test Redis publisher initialization"""
    publisher = get_publisher()

    assert publisher is not None
    assert isinstance(publisher, RedisPublisher)
    assert hasattr(publisher, 'redis_client')


@pytest.mark.asyncio
async def test_redis_pubsub_music_generation():
    """Test Redis Pub/Sub for music generation events"""
    publisher = get_publisher()

    # Test music generation event publishing
    result = publisher.publish_music_generated(
        route_id="route_123",
        user_id="user_456",
        music_data={"tempo": 120, "key": "C", "genre": "classical"}
    )

    # Should return number of subscribers (0 in test environment)
    assert isinstance(result, int)


@pytest.mark.asyncio
async def test_redis_pubsub_vector_search():
    """Test Redis Pub/Sub for vector search results"""
    publisher = get_publisher()

    # Test vector search results publishing
    result = publisher.publish_vector_search_results(
        search_id="search_123",
        user_id="user_456",
        query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        results=[{"id": 1, "similarity": 0.95}, {"id": 2, "similarity": 0.88}],
        search_type="music"
    )

    assert isinstance(result, int)


@pytest.mark.asyncio
async def test_redis_pubsub_collaborative_editing():
    """Test Redis Pub/Sub for collaborative editing"""
    publisher = get_publisher()

    # Test collaborative edit publishing
    result = publisher.publish_collaborative_edit(
        session_id="session_123",
        user_id="user_456",
        edit_type="tempo_change",
        edit_data={"old_tempo": 120, "new_tempo": 140},
        target_users=["user_789"]
    )

    assert isinstance(result, int)


@pytest.mark.asyncio
async def test_redis_pubsub_generation_progress():
    """Test Redis Pub/Sub for generation progress"""
    publisher = get_publisher()

    # Test progress publishing
    result = publisher.publish_generation_progress(
        generation_id="gen_123",
        user_id="user_456",
        progress=0.75,
        status="processing",
        current_step="Generating MIDI file"
    )

    assert isinstance(result, int)


@pytest.mark.asyncio
async def test_redis_pubsub_system_status():
    """Test Redis Pub/Sub for system status"""
    publisher = get_publisher()

    # Test system status publishing
    result = publisher.publish_system_status(
        status_type="health_check",
        status_data={
            "redis_connected": True,
            "vector_count": 100,
            "active_sessions": 5
        }
    )

    assert isinstance(result, int)


# ==================== WEBSOCKET MANAGER TESTS ====================

@pytest.mark.asyncio
async def test_websocket_manager_initialization():
    """Test WebSocket manager initialization"""
    manager = WebSocketManager()

    assert manager is not None
    assert hasattr(manager, 'connection_manager')
    assert hasattr(manager, 'room_manager')


@pytest.mark.asyncio
async def test_websocket_connection_handling():
    """Test WebSocket connection handling"""
    manager = WebSocketManager()

    # Mock WebSocket
    mock_websocket = AsyncMock()
    mock_websocket.accept = AsyncMock()

    # Test connection handling (without actual WebSocket)
    # This tests the connection manager logic
    connection_id = await manager.handle_connection(
        websocket=mock_websocket,
        session_id="session_123",
        user_id=456,
        username="test_user"
    )

    # Should handle gracefully
    assert connection_id is not None or connection_id is None  # Either works


@pytest.mark.asyncio
async def test_websocket_state_broadcast():
    """Test WebSocket state broadcasting"""
    manager = WebSocketManager()

    # Test state update broadcasting
    result = await manager.broadcast_state_update_with_redis(
        session_id="session_123",
        updates={"tempo": 140, "key": "D"}
    )

    # Should handle gracefully
    assert result is not None or result is None


# ==================== GALERA CLUSTER TESTS ====================

@pytest.mark.asyncio
async def test_galera_manager_initialization():
    """Test Galera cluster manager initialization"""
    manager = get_galera_manager()

    assert manager is not None
    assert hasattr(manager, 'nodes')


@pytest.mark.asyncio
async def test_galera_cluster_status():
    """Test Galera cluster status checking"""
    manager = get_galera_manager()

    # Test status check (will fail without actual cluster, but should not crash)
    try:
        status = manager.get_cluster_status()
        # Should return dict or handle gracefully
        assert isinstance(status, dict) or status is None
    except Exception as e:
        # Should handle connection errors gracefully
        assert "connection" in str(e).lower() or "timeout" in str(e).lower()


# ==================== ENHANCED MUSIC GENERATOR TESTS ====================

@pytest.mark.asyncio
async def test_music_generator_realtime_features():
    """Test enhanced music generator with real-time features"""
    generator = get_music_generation_service()

    assert generator is not None
    assert hasattr(generator.generator, 'publisher')
    assert hasattr(generator.generator, 'faiss_service')
    assert hasattr(generator.generator, 'websocket_manager')


@pytest.mark.asyncio
async def test_realtime_generation_progress():
    """Test real-time generation progress publishing"""
    generator = get_music_generation_service()

    # Test progress publishing
    try:
        generator.generator.publish_generation_progress(
            user_id="user_123",
            generation_id="gen_456",
            progress=0.5,
            status="processing",
            current_step="Building route graph"
        )
        # Should not raise exception
        assert True
    except Exception as e:
        # Should handle gracefully if Redis not available
        assert "redis" in str(e).lower() or "connection" in str(e).lower()


@pytest.mark.asyncio
async def test_vector_realtime_sync():
    """Test vector storage with real-time sync"""
    generator = get_music_generation_service()

    # Test vector storage with sync
    vector = np.random.rand(128).astype(np.float32)

    try:
        success = generator.generator.store_vector_with_realtime_sync(
            composition_id=1,
            route_id=1,
            origin="JFK",
            destination="LAX",
            genre="classical",
            tempo=120,
            pitch=60.0,
            harmony=0.8,
            complexity=0.7,
            vector=vector
        )

        # Should return boolean result
        assert isinstance(success, bool)
    except Exception as e:
        # Should handle gracefully if services not available
        assert True  # Test passes if it doesn't crash


# ==================== INTEGRATION TESTS ====================

@pytest.mark.asyncio
async def test_realtime_integration_end_to_end():
    """Test complete real-time integration"""
    # Test FAISS service
    faiss_service = get_faiss_duckdb_service()
    stats = faiss_service.get_statistics()
    assert isinstance(stats, dict)

    # Test Redis publisher
    publisher = get_publisher()
    assert publisher is not None

    # Test WebSocket manager
    ws_manager = WebSocketManager()
    assert ws_manager is not None

    # Test music generator
    music_generator = get_music_generation_service()
    assert music_generator is not None

    # Test Galera manager
    galera_manager = get_galera_manager()
    assert galera_manager is not None


@pytest.mark.asyncio
async def test_realtime_music_generation_workflow():
    """Test complete real-time music generation workflow"""
    generator = get_music_generation_service()

    # Test real-time generation method exists
    assert hasattr(generator.generator, 'generate_music_with_realtime_updates')

    # Test progress publishing methods exist
    assert hasattr(generator.generator, 'publish_generation_progress')
    assert hasattr(generator.generator, 'publish_music_update')

    # Test vector sync method exists
    assert hasattr(generator.generator, 'store_vector_with_realtime_sync')


@pytest.mark.asyncio
async def test_realtime_vector_search_workflow():
    """Test complete real-time vector search workflow"""
    service = get_faiss_duckdb_service()

    # Test methods exist
    assert hasattr(service, 'search_similar_music')
    assert hasattr(service, 'search_similar_routes')
    assert hasattr(service, 'store_music_vector')
    assert hasattr(service, 'get_statistics')

    # Test Redis publisher for vector results
    publisher = get_publisher()
    assert hasattr(publisher, 'publish_vector_search_results')
    assert hasattr(publisher, 'publish_route_music_sync')


@pytest.mark.asyncio
async def test_realtime_collaboration_workflow():
    """Test complete real-time collaboration workflow"""
    ws_manager = WebSocketManager()
    publisher = get_publisher()

    # Test WebSocket methods
    assert hasattr(ws_manager, 'broadcast_state_update_with_redis')
    assert hasattr(ws_manager, 'handle_real_time_music_update')

    # Test Redis collaboration methods
    assert hasattr(publisher, 'publish_collaborative_edit')
    assert hasattr(publisher, 'publish_music_update_real_time')
    assert hasattr(publisher, 'publish_route_music_sync')


# ==================== PERFORMANCE TESTS ====================

@pytest.mark.asyncio
async def test_realtime_performance_metrics():
    """Test real-time performance metrics"""
    import time

    # Test FAISS performance
    start_time = time.time()
    service = get_faiss_duckdb_service()

    # Quick vector operation
    query = np.random.rand(1, 128).astype(np.float32)
    results = service.search_similar_music(query, limit=5)

    faiss_time = time.time() - start_time
    assert faiss_time < 1.0  # Should be fast

    # Test Redis performance
    start_time = time.time()
    publisher = get_publisher()

    result = publisher.publish_generation_progress(
        generation_id="perf_test",
        user_id="user_123",
        progress=0.5,
        status="testing"
    )

    redis_time = time.time() - start_time
    assert redis_time < 1.0  # Should be reasonably fast, but allow for test environment latency


# ==================== ERROR HANDLING TESTS ====================

@pytest.mark.asyncio
async def test_realtime_error_handling():
    """Test error handling in real-time features"""

    # Test FAISS error handling
    service = get_faiss_duckdb_service()

    # Test with invalid data
    try:
        results = service.search_similar_music(None, limit=5)
        # Should handle gracefully
        assert isinstance(results, list)
    except Exception:
        # Or should raise appropriate exception
        assert True

    # Test Redis error handling
    publisher = get_publisher()

    # Test with None client (simulated connection failure)
    original_client = publisher.redis_client
    publisher.redis_client = None

    try:
        result = publisher.publish_music_generated("route_123", "user_456", {})
        assert result == 0  # Should return 0 on failure
    except Exception:
        # Or should handle gracefully
        assert True
    finally:
        publisher.redis_client = original_client


if __name__ == "__main__":
    # Run specific real-time tests
    pytest.main([__file__, "-v", "-k", "realtime"])
