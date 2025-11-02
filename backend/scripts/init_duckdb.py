#!/usr/bin/env python3
"""
Initialize DuckDB Analytics Database
This script creates the DuckDB database and tables if they don't exist.
Safe to run multiple times - will not overwrite existing data.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.duckdb_analytics import DuckDBAnalytics
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def init_duckdb():
    """Initialize DuckDB database and tables"""
    try:
        logger.info("Initializing DuckDB analytics database...")
        
        # Create analytics instance - this will auto-create the database
        analytics = DuckDBAnalytics()
        
        # Verify tables were created
        stats = analytics.get_route_complexity_stats()
        
        logger.info("✅ DuckDB database initialized successfully!")
        logger.info(f"   Database location: {analytics.db_path}")
        logger.info(f"   Total routes: {stats.get('total_routes', 0)}")
        
        analytics.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize DuckDB: {e}")
        return False


if __name__ == "__main__":
    success = init_duckdb()
    sys.exit(0 if success else 1)
