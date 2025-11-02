"""
Test script for vector similarity search
Tests the route embedding functionality
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.services.route_embedding_service import get_route_embedding_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_similarity_search():
    """Test similarity search functionality"""
    
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    try:
        async with async_session() as session:
            service = get_route_embedding_service()
            
            # Test 1: Find similar routes
            logger.info("\n" + "=" * 60)
            logger.info("TEST 1: Finding routes similar to JFK → LAX")
            logger.info("=" * 60)
            
            similar = await service.find_similar_routes(session, "JFK", "LAX", limit=5)
            
            if similar:
                logger.info(f"Found {len(similar)} similar routes:")
                for i, route in enumerate(similar, 1):
                    logger.info(f"  {i}. Route {route.get('route_id')}: "
                              f"{route.get('origin_code', 'N/A')} → {route.get('dest_code', 'N/A')} "
                              f"(similarity: {route.get('similarity_score', 0):.3f})")
            else:
                logger.warning("No similar routes found")
            
            # Test 2: Find routes by genre
            logger.info("\n" + "=" * 60)
            logger.info("TEST 2: Finding routes that sound like 'ambient' music")
            logger.info("=" * 60)
            
            ambient_routes = await service.find_routes_by_genre(session, "ambient", limit=5)
            
            if ambient_routes:
                logger.info(f"Found {len(ambient_routes)} ambient-style routes:")
                for i, route in enumerate(ambient_routes, 1):
                    logger.info(f"  {i}. {route['origin_code']} → {route['dest_code']} "
                              f"({route['distance_km']:.0f} km, {route['stops']} stops)")
            else:
                logger.warning("No ambient routes found")
            
            # Test 3: Calculate complexity
            logger.info("\n" + "=" * 60)
            logger.info("TEST 3: Calculating melodic complexity")
            logger.info("=" * 60)
            
            if similar and len(similar) > 0:
                route_id = similar[0].get('route_id')
                complexity = await service.calculate_melodic_complexity(session, route_id)
                
                if complexity:
                    logger.info(f"Complexity metrics for route {route_id}:")
                    logger.info(f"  Harmonic: {complexity['harmonic_complexity']:.3f}")
                    logger.info(f"  Rhythmic: {complexity['rhythmic_complexity']:.3f}")
                    logger.info(f"  Melodic: {complexity['melodic_complexity']:.3f}")
                    logger.info(f"  Overall: {complexity['overall_complexity']:.3f}")
            
            # Test 4: Get statistics
            logger.info("\n" + "=" * 60)
            logger.info("TEST 4: Embedding statistics")
            logger.info("=" * 60)
            
            stats = await service.get_embedding_statistics(session)
            
            logger.info(f"Total routes: {stats['total_routes']}")
            logger.info(f"Routes with embeddings: {stats['routes_with_embeddings']}")
            logger.info(f"Coverage: {stats['embedding_coverage']}%")
            logger.info(f"FAISS index size: {stats['faiss_index_size']} vectors")
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
            logger.info("\nYou can now:")
            logger.info("1. Start the backend: python main.py")
            logger.info("2. Try the API endpoints:")
            logger.info("   GET /api/v1/vectors/similar-routes?origin=JFK&destination=LAX")
            logger.info("   GET /api/v1/vectors/routes-by-genre?genre=ambient")
            logger.info("   GET /api/v1/vectors/statistics")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_similarity_search())
