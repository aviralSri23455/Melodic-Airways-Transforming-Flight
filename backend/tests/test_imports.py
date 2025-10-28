#!/usr/bin/env python3
"""
Test script to verify that all services can be imported and used correctly
"""

import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

def test_imports():
    """Test that all services can be imported"""
    try:
        # Test vector services
        from app.services.vector_service import MusicVector, VectorSearchService
        print("Vector services imported successfully")

        # Test dataset managers
        from app.services.dataset_manager import UserDatasetManager, CollectionManager, RemixManager
        print("Dataset managers imported successfully")

        # Test activity service
        from app.services.activity_service import ActivityService
        print("Activity service imported successfully")

        # Test real-time services
        from app.services.faiss_duckdb_service import get_faiss_duckdb_service
        from app.services.redis_publisher import get_publisher
        from app.services.websocket_manager import WebSocketManager
        from app.services.galera_manager import get_galera_manager
        from app.services.music_generator import get_music_generation_service
        print("Real-time services imported successfully")

        return True, MusicVector  # Return MusicVector too

    except ImportError as e:
        print(f"Import error: {e}")
        return False, None

def test_basic_functionality(music_vector_class):
    """Test basic functionality of services"""
    if music_vector_class is None:
        return False

    try:
        # Test MusicVector creation
        vector = music_vector_class(
            harmonic_features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.5, 0.3],
            rhythmic_features=[0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.1, 0.2],
            melodic_features=[0.1] * 16,
            genre_features=[0.5] * 10
        )
        print("MusicVector creation works")

        # Test vector from composition data
        vector2 = music_vector_class.from_composition_data(
            tempo=120,
            pitch=60.0,
            harmony=0.8,
            complexity=0.7,
            genre="classical"
        )
        print("MusicVector.from_composition_data works")

        return True

    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing service imports and basic functionality...")

    imports_ok, music_vector = test_imports()
    if imports_ok:
        functionality_ok = test_basic_functionality(music_vector)

        if functionality_ok:
            print("All tests passed! Services are working correctly.")
            sys.exit(0)
        else:
            print("Basic functionality tests failed.")
            sys.exit(1)
    else:
        print("Import tests failed.")
        sys.exit(1)
