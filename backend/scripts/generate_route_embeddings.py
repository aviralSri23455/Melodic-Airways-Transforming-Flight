"""
Generate vector embeddings for all routes in the OpenFlights dataset
Run this script to populate route_embedding column with 128D vectors
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.services.route_embedding_service import get_route_embedding_service
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def generate_embeddings():
    """Generate embeddings for all routes"""
    logger.info("Starting route embedding generation...")
    
    # Create database engine
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True
    )
    
    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    try:
        async with async_session() as session:
            # Get embedding service
            embedding_service = get_route_embedding_service()
            
            logger.info("Generating embeddings for all routes...")
            stats = await embedding_service.generate_route_embeddings(session)
            
            logger.info("=" * 60)
            logger.info("EMBEDDING GENERATION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total routes processed: {stats['total']}")
            logger.info(f"Embeddings generated: {stats['generated']}")
            logger.info(f"Failed: {stats['failed']}")
            logger.info(f"Success rate: {stats['generated'] / stats['total'] * 100:.2f}%")
            logger.info("=" * 60)
            
            # Get statistics
            logger.info("\nFetching embedding statistics...")
            embedding_stats = await embedding_service.get_embedding_statistics(session)
            
            logger.info("\nEMBEDDING STATISTICS:")
            logger.info(f"  Total routes: {embedding_stats['total_routes']}")
            logger.info(f"  Routes with embeddings: {embedding_stats['routes_with_embeddings']}")
            logger.info(f"  Coverage: {embedding_stats['embedding_coverage']}%")
            logger.info(f"  Avg distance: {embedding_stats['avg_distance_km']} km")
            logger.info(f"  Avg stops: {embedding_stats['avg_stops']}")
            logger.info(f"  FAISS index size: {embedding_stats['faiss_index_size']} vectors")
            logger.info(f"  Embedding dimension: {embedding_stats['embedding_dimension']}D")
            
            if embedding_stats.get('avg_melodic_complexity'):
                logger.info("\nCOMPLEXITY METRICS:")
                logger.info(f"  Avg melodic complexity: {embedding_stats['avg_melodic_complexity']}")
                logger.info(f"  Avg harmonic complexity: {embedding_stats['avg_harmonic_complexity']}")
                logger.info(f"  Avg rhythmic complexity: {embedding_stats['avg_rhythmic_complexity']}")
            
            logger.info("\n✅ Embedding generation completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Test similarity search: python scripts/test_similarity_search.py")
            logger.info("2. Start the backend: python main.py")
            logger.info("3. Try the API: GET /api/v1/routes/similar?origin=JFK&destination=LAX")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise
    
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(generate_embeddings())
