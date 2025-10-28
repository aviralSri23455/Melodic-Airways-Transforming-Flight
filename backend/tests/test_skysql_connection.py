#!/usr/bin/env python3
"""
Test Local MariaDB connection using SQLAlchemy with asyncmy
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Load environment variables from .env
load_dotenv()

# Get credentials from environment variables or use DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ Error: DATABASE_URL environment variable not set")
    print("Please check your .env file")
    sys.exit(1)

# Convert to asyncmy dialect for async support
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+asyncmy://", 1)

async def test_connection():
    """Test database connection using SQLAlchemy asyncmy"""
    print("🔧 Testing Local MariaDB connection...")
    print(f"📍 Connection URL: {DATABASE_URL}")

    # Create async engine
    engine = create_async_engine(DATABASE_URL)

    try:
        async with engine.begin() as conn:
            # Test basic connection
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"\n✅ Connection successful! Test result: {row}")

            # Show available databases
            result = await conn.execute(text("SHOW DATABASES;"))
            databases = result.fetchall()
            print(f"\n📦 Available databases ({len(databases)}):")
            for db in databases:
                print(f"   - {db[0]}")

            # Test the target database
            db_name = DATABASE_URL.split('/')[-1]  # Extract database name from URL
            await conn.execute(text(f"USE {db_name}"))
            result = await conn.execute(text("SHOW TABLES;"))
            tables = result.fetchall()

            if tables:
                print(f"\n📋 Tables in '{db_name}' ({len(tables)}):")
                for table in tables:
                    print(f"   - {table[0]}")
            else:
                print(f"\n📋 No tables found in '{db_name}'")

            print("\n🎉 Connection test completed successfully!")
            return True

    except Exception as e:
        print(f"\n❌ Connection error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   - Ensure MariaDB server is running locally")
        print("   - Check username, password, and database name")
        print("   - Verify port (default 3306)")
        print("   - For local installations, SSL might need to be disabled")
        return False

    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())
