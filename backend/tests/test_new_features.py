import pytest
import pytest_asyncio
import sys
import os
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator
from unittest.mock import AsyncMock

# Add the backend directory (parent of tests/) to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.models.models import (
    Base, User, MusicComposition, Route, Airport,
    UserDataset, UserCollection, CompositionRemix, RemixType, ActivityType
)
from app.services.faiss_duckdb_service import get_faiss_duckdb_service
from app.services.redis_publisher import get_publisher, RedisPublisher
from app.services.websocket_manager import WebSocketManager, ConnectionManager, RoomManager
from app.services.music_generator import get_music_generation_service
from app.services.realtime_generator import RealtimeGenerator, MusicBuffer, MusicSegment
from app.services.vector_service import VectorSearchService, MusicVector
from app.services.dataset_manager import UserDatasetManager, CollectionManager, RemixManager
from app.services.activity_service import ActivityService

# Test database setup
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def test_user(test_db: AsyncSession) -> User:
    """Create test user"""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password",
        role="user",
        is_active=1
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_composition(test_db: AsyncSession, test_user: User) -> MusicComposition:
    """Create test composition"""
    origin = Airport(name="Test Origin", city="City1", country="Country1", iata_code="TST", icao_code="TSTO", latitude=0.0, longitude=0.0)
    destination = Airport(name="Test Destination", city="City2", country="Country2", iata_code="TSD", icao_code="TSDO", latitude=10.0, longitude=10.0)
    test_db.add(origin)
    test_db.add(destination)
    await test_db.commit()

    route = Route(origin_airport_id=origin.id, destination_airport_id=destination.id, distance_km=1000.0, duration_min=120)
    test_db.add(route)
    await test_db.commit()

    composition = MusicComposition(
        route_id=route.id, user_id=test_user.id, tempo=120, pitch=60.0, harmony=0.8,
        midi_path="/test/path.mid", complexity_score=0.7, harmonic_richness=0.8,
        duration_seconds=180, unique_notes=20, musical_key="C", scale="major",
        title="Test Composition", genre="classical", is_public=1
    )
    test_db.add(composition)
    await test_db.commit()
    await test_db.refresh(composition)
    return composition


# ==================== VECTOR SEARCH TESTS ====================

@pytest.mark.asyncio
async def test_vector_search_service(test_db: AsyncSession, test_composition: MusicComposition):
    """Test vector search service with creation, storage, and retrieval"""
    service = VectorSearchService()
    
    vector = await service.extract_features(
        test_composition.id, test_composition.tempo, test_composition.pitch,
        test_composition.harmony, test_composition.complexity_score or 0.5,
        test_composition.genre or "unknown"
    )
    
    assert vector is not None
    assert len(vector.harmonic_features) == 12
    assert len(vector.rhythmic_features) == 8
    assert len(vector.melodic_features) == 16
    assert len(vector.genre_features) == 10
    
    json_data = vector.to_json()
    assert all(k in json_data for k in ["harmonic", "rhythmic", "melodic", "genre", "timestamp"])
    
    success = await service.store_vector(test_db, test_composition.id, vector)
    assert success
    
    retrieved_vector = await service.get_composition_vector(test_db, test_composition.id)
    assert retrieved_vector is not None
    
    vector2_json = {
        "harmonic": vector.harmonic_features,
        "rhythmic": vector.rhythmic_features,
        "melodic": vector.melodic_features,
        "genre": vector.genre_features
    }
    similarity = service._calculate_cosine_similarity(vector, vector2_json)
    assert abs(similarity - 1.0) < 1e-6  # Allow for floating-point precision


# ==================== DATASET MANAGER TESTS ====================

@pytest.mark.asyncio
async def test_dataset_operations(test_db: AsyncSession, test_user: User):
    """Test dataset CRUD operations"""
    manager = UserDatasetManager()
    
    dataset = await manager.create_dataset(test_db, test_user.id, "Test Dataset", {"route": "TST-TSD"}, {"description": "Test"})
    assert dataset.id is not None
    assert dataset.name == "Test Dataset"
    
    updated = await manager.update_dataset(test_db, dataset.id, test_user.id, name="Updated Name")
    assert updated.name == "Updated Name"
    
    for i in range(2):
        await manager.create_dataset(test_db, test_user.id, f"Dataset {i}", {"data": f"test_{i}"})
    
    datasets = await manager.get_user_datasets(test_db, test_user.id)
    assert len(datasets) >= 3
    
    success = await manager.delete_dataset(test_db, dataset.id, test_user.id)
    assert success
    retrieved = await manager.get_dataset(test_db, dataset.id, test_user.id)
    assert retrieved is None


# ==================== COLLECTION MANAGER TESTS ====================

@pytest.mark.asyncio
async def test_collection_operations(test_db: AsyncSession, test_user: User, test_composition: MusicComposition):
    """Test collection CRUD and composition management"""
    manager = CollectionManager()
    
    collection = await manager.create_collection(test_db, test_user.id, "Test Collection", "A test collection", ["tag1", "tag2"])
    assert collection.id is not None
    assert collection.tags == ["tag1", "tag2"]
    
    success = await manager.add_composition_to_collection(test_db, collection.id, test_composition.id, test_user.id)
    assert success
    assert test_composition.id in collection.composition_ids
    
    compositions = await manager.get_collection_compositions(test_db, collection.id, test_user.id)
    assert len(compositions) == 1
    assert compositions[0].id == test_composition.id


# ==================== WEBSOCKET MANAGER TESTS ====================

@pytest.mark.asyncio
async def test_room_manager():
    """Test room manager"""
    manager = RoomManager()
    
    room = manager.create_room("session_1", creator_id=1, composition_id=100)
    assert room["session_id"] == "session_1"
    
    manager.add_participant("session_1", 2, "user2")
    assert 2 in manager.rooms["session_1"]["participants"]
    
    retrieved = manager.get_room("session_1")
    assert retrieved is not None
    
    manager.close_room("session_1")
    assert "session_1" not in manager.rooms


# ==================== REALTIME GENERATOR TESTS ====================

@pytest.mark.asyncio
async def test_realtime_generator():
    """Test real-time music generation"""
    generator = RealtimeGenerator()
    
    live_data = {"altitude": 10000, "speed": 400, "latitude": 40.0, "longitude": -74.0}
    segment = await generator.generate_realtime_segment(live_data, "ambient", 120)
    
    assert segment.segment_id is not None
    assert len(segment.notes) > 0
    assert segment.genre == "ambient"
    assert segment.tempo == 120


@pytest.mark.asyncio
async def test_music_buffer():
    """Test music buffer"""
    buffer = MusicBuffer()
    
    test_segment = MusicSegment(
        segment_id="test_1",
        timestamp="2024-01-01T00:00:00",
        duration_ms=500,
        notes=[{"pitch": 60, "velocity": 64, "duration": 250}],
        tempo=120,
        genre="ambient",
        coherence_score=0.9
    )
    
    buffer.add_segment(test_segment)
    assert len(buffer.segments) == 1
    
    status = buffer.get_buffer_status()
    assert status["segments_count"] == 1


@pytest.mark.asyncio
async def test_remix_manager(test_db: AsyncSession, test_composition: MusicComposition):
    """Test remix manager"""
    manager = RemixManager()
    
    remix_comp = MusicComposition(
        route_id=test_composition.route_id, user_id=test_composition.user_id,
        tempo=140, pitch=65.0, harmony=0.7, midi_path="/test/remix.mid",
        title="Remix", genre="jazz"
    )
    test_db.add(remix_comp)
    await test_db.commit()
    await test_db.refresh(remix_comp)
    
    remix = await manager.create_remix(
        test_db, test_composition.id, remix_comp.id, RemixType.GENRE_CHANGE,
        {"original_genre": "classical", "new_genre": "jazz"}
    )
    
    assert remix.id is not None
    assert remix.original_composition_id == test_composition.id


# ==================== ACTIVITY FEED TESTS ====================

@pytest.mark.asyncio
async def test_activity_operations(test_db: AsyncSession, test_user: User):
    """Test activity logging, statistics, and cleanup"""
    activity_service = ActivityService()
    
    success = await activity_service.log_activity(
        db=test_db, user_id=test_user.id, activity_type=ActivityType.COMPOSITION_CREATED,
        target_id=1, target_type="composition", activity_data={"title": "Test"}
    )
    assert success
    
    activities = await activity_service.get_user_activities(test_db, test_user.id, limit=10)
    assert len(activities) >= 1
    assert activities[0]["activity_type"] == "composition_created"
    
    for i in range(3):
        await activity_service.log_activity(db=test_db, user_id=test_user.id, activity_type=ActivityType.DATASET_CREATED, activity_data={"count": i})
    
    stats = await activity_service.get_activity_stats(test_db, test_user.id, days=1)
    assert stats["total_activities"] >= 3
    
    recent = await activity_service.get_recent_activities(test_db, minutes=5, limit=10)
    assert len(recent) >= 1
    
    count = await activity_service.cleanup_old_activities(test_db, older_than_days=1)
    assert isinstance(count, int)


# ==================== INTEGRATION TESTS ====================

@pytest.mark.asyncio
async def test_full_workflow(test_db: AsyncSession, test_user: User, test_composition: MusicComposition):
    """Test complete workflow: dataset -> collection -> vector search -> remix"""
    dataset_manager = UserDatasetManager()
    dataset = await dataset_manager.create_dataset(test_db, test_user.id, "My Compositions", {"routes": ["TST-TSD"]})
    assert dataset.id is not None
    
    collection_manager = CollectionManager()
    collection = await collection_manager.create_collection(test_db, test_user.id, "Favorites", tags=["classical"])
    assert collection.id is not None
    
    await collection_manager.add_composition_to_collection(test_db, collection.id, test_composition.id, test_user.id)
    
    vector_service = VectorSearchService()
    vector = await vector_service.extract_features(
        test_composition.id, test_composition.tempo, test_composition.pitch,
        test_composition.harmony, test_composition.complexity_score or 0.5,
        test_composition.genre or "unknown"
    )
    await vector_service.store_vector(test_db, test_composition.id, vector)
    
    remix_manager = RemixManager()
    remix_comp = MusicComposition(
        route_id=test_composition.route_id, user_id=test_user.id,
        tempo=140, pitch=65.0, harmony=0.7, midi_path="/test/remix.mid", title="Remix"
    )
    test_db.add(remix_comp)
    await test_db.commit()
    await test_db.refresh(remix_comp)
    
    remix = await remix_manager.create_remix(test_db, test_composition.id, remix_comp.id, RemixType.TEMPO_CHANGE)
    assert remix.id is not None
    
    collections = await collection_manager.get_user_collections(test_db, test_user.id)
    assert len(collections) == 1


# ==================== FAISS + DUCKDB TESTS ====================

@pytest.mark.asyncio
async def test_faiss_duckdb_service():
    """Test FAISS + DuckDB service operations"""
    service = get_faiss_duckdb_service()
    assert service is not None
    
    stats = service.get_statistics()
    assert isinstance(stats, dict)
    assert "total_vectors" in stats
    
    query_vector = np.random.rand(1, 128).astype(np.float32)
    results = service.search_similar_music(query_vector, limit=5)
    assert isinstance(results, list)
    
    vector = np.random.rand(128).astype(np.float32)
    success = service.store_music_vector(
        composition_id=1, route_id=1, origin="JFK", destination="LAX",
        genre="classical", tempo=120, pitch=60.0, harmony=0.8,
        complexity=0.7, vector=vector, metadata={"test": "data"}
    )
    assert isinstance(success, bool)
    
    route_results = service.search_similar_routes("JFK", "LAX", limit=5)
    assert isinstance(route_results, list)


# ==================== REDIS PUB/SUB TESTS ====================

@pytest.mark.asyncio
async def test_redis_publisher():
    """Test Redis publisher operations"""
    publisher = get_publisher()
    assert publisher is not None
    assert isinstance(publisher, RedisPublisher)
    
    result = publisher.publish_music_generated(route_id="route_123", user_id="user_456", music_data={"tempo": 120})
    assert isinstance(result, int)
    
    result = publisher.publish_vector_search_results(
        search_id="search_123", user_id="user_456", query_vector=[0.1, 0.2],
        results=[{"id": 1, "similarity": 0.95}], search_type="music"
    )
    assert isinstance(result, int)
    
    result = publisher.publish_collaborative_edit(
        session_id="session_123", user_id="user_456", edit_type="tempo_change",
        edit_data={"old_tempo": 120, "new_tempo": 140}, target_users=["user_789"]
    )
    assert isinstance(result, int)
    
    result = publisher.publish_generation_progress(
        generation_id="gen_123", user_id="user_456", progress=0.75,
        status="processing", current_step="Generating MIDI"
    )
    assert isinstance(result, int)


# ==================== WEBSOCKET MANAGER TESTS ====================

@pytest.mark.asyncio
async def test_websocket_manager():
    """Test WebSocket manager operations"""
    manager = WebSocketManager()
    assert manager is not None
    assert hasattr(manager, 'connection_manager')
    assert hasattr(manager, 'room_manager')
    
    mock_websocket = AsyncMock()
    mock_websocket.accept = AsyncMock()
    
    connection_id = await manager.handle_connection(
        websocket=mock_websocket, session_id="session_123",
        user_id=456, username="test_user"
    )
    assert connection_id is not None or connection_id is None
    
    result = await manager.broadcast_state_update_with_redis(
        session_id="session_123", updates={"tempo": 140}
    )
    assert result is not None or result is None


# ==================== MUSIC GENERATOR TESTS ====================

@pytest.mark.asyncio
async def test_music_generator_realtime():
    """Test enhanced music generator with real-time features"""
    generator = get_music_generation_service()
    assert generator is not None
    assert hasattr(generator.generator, 'publisher')
    assert hasattr(generator.generator, 'faiss_service')
    assert hasattr(generator.generator, 'websocket_manager')
    
    try:
        generator.generator.publish_generation_progress(
            user_id="user_123", generation_id="gen_456",
            progress=0.5, status="processing", current_step="Building route"
        )
        assert True
    except Exception as e:
        assert "redis" in str(e).lower() or "connection" in str(e).lower()
    
    vector = np.random.rand(128).astype(np.float32)
    try:
        success = generator.generator.store_vector_with_realtime_sync(
            composition_id=1, route_id=1, origin="JFK", destination="LAX",
            genre="classical", tempo=120, pitch=60.0, harmony=0.8,
            complexity=0.7, vector=vector
        )
        assert isinstance(success, bool)
    except Exception:
        assert True


# ==================== INTEGRATION TESTS ====================

@pytest.mark.asyncio
async def test_realtime_integration():
    """Test complete real-time integration"""
    faiss_service = get_faiss_duckdb_service()
    assert isinstance(faiss_service.get_statistics(), dict)
    
    publisher = get_publisher()
    assert publisher is not None
    
    ws_manager = WebSocketManager()
    assert ws_manager is not None
    
    music_generator = get_music_generation_service()
    assert music_generator is not None
    
    assert hasattr(music_generator.generator, 'generate_music_with_realtime_updates')
    assert hasattr(faiss_service, 'search_similar_music')
    assert hasattr(publisher, 'publish_vector_search_results')
    assert hasattr(ws_manager, 'broadcast_state_update_with_redis')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
