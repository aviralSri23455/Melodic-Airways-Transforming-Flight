#!/usr/bin/env python3
"""Test database connection"""

import asyncio
import sys
import os

# Add the backend directory (parent of tests/) to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.core.config import settings

async def test_connection():
    """Test database connection"""
    print(f"Testing connection to: {settings.DATABASE_URL}")

    engine = create_async_engine(settings.DATABASE_URL)

    try:
        async with engine.begin() as conn:
            # Try to execute a simple query using text()
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"Connection successful! Test result: {row}")
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())
