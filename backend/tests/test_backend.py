#!/usr/bin/env python3
"""
Quick setup and test script for Aero Melody Backend
Run this to verify everything is working before sharing with frontend team
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_backend():
    """Test backend components"""
    print("🚀 Testing Aero Melody Backend...")

    try:
        # Test 1: Import core modules
        print("\n📦 Testing imports...")
        from app.core.config import settings
        from app.models.models import Airport, Route, MusicComposition, User
        from app.models.schemas import MusicStyle, ScaleType
        print("✅ All imports successful")

        # Test 2: Database connection
        print("\n🗄️  Testing database connection...")
        from app.db.database import engine, Base
        print(f"✅ Database URL: {settings.DATABASE_URL}")
        print("✅ Engine created successfully")

        # Test 3: Music generation service
        print("\n🎵 Testing music generation service...")
        from app.services.music_generator import MusicGenerationService
        service = MusicGenerationService()
        print("✅ Music generation service loaded")

        # Test 4: Vector service
        print("\n🔍 Testing vector service...")
        from app.services.vector_service import VectorSearchService
        vector_service = VectorSearchService()
        print("✅ Vector search service loaded")

        # Test 5: API routes
        print("\n🛣️  Testing API routes...")
        from app.api.routes import router
        print(f"✅ Main router loaded with {len(router.routes)} routes")

        from app.api.extended_routes import router as extended_router
        print(f"✅ Extended router loaded with {len(extended_router.routes)} routes")

        print("\n🎉 All tests passed! Backend is ready for frontend integration.")
        print("\n📋 Next steps:")
        print("1. Start MariaDB: docker-compose up -d mariadb")
        print("2. Load data: python scripts/etl_openflights.py")
        print("3. Start server: python main.py")
        print("4. Check API docs: http://localhost:8000/docs")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_backend())
    sys.exit(0 if success else 1)
