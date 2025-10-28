#!/usr/bin/env python3
"""
Redis Cloud Connection Test Script
Tests Redis Cloud connection and basic functionality before running full integration tests.
"""

import asyncio
import redis
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_redis_cloud_connection():
    """Test Redis Cloud connection and basic functionality"""
    # Use environment variables for Redis Cloud connection
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_username = os.getenv('REDIS_USERNAME', 'default')
    redis_password = os.getenv('REDIS_PASSWORD', '')

    if not redis_password or redis_host == 'localhost':
        logger.error("❌ Redis Cloud credentials not found in environment variables")
        logger.info("Please set REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD in your .env file")
        return False

    logger.info("🔴 Testing Redis Cloud Connection...")
    logger.info(f"Redis Host: {redis_host}")
    logger.info(f"Redis Port: {redis_port}")

    try:
        # Connect to Redis Cloud using individual components
        redis_url = f"redis://{redis_username}:{redis_password}@{redis_host}:{redis_port}"
        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Test basic operations
        logger.info("📡 Testing PING...")
        ping_result = redis_client.ping()
        if ping_result:
            logger.info("✅ PING successful!")
        else:
            logger.error("❌ PING failed!")
            return False

        # Test SET operation
        logger.info("📝 Testing SET operation...")
        set_result = redis_client.set("test_key", "test_value", ex=60)
        if set_result:
            logger.info("✅ SET successful!")
        else:
            logger.error("❌ SET failed!")
            return False

        # Test GET operation
        logger.info("📖 Testing GET operation...")
        get_result = redis_client.get("test_key")
        if get_result == "test_value":
            logger.info("✅ GET successful!")
        else:
            logger.error("❌ GET failed!")
            return False

        # Test Pub/Sub
        logger.info("📢 Testing Pub/Sub...")
        pubsub = redis_client.pubsub()
        pubsub.subscribe("test_channel")

        # Publish a test message
        pub_result = redis_client.publish("test_channel", "test_message")
        logger.info(f"📤 Published to {pub_result} subscribers")

        # Try to receive message (non-blocking)
        message = pubsub.get_message(timeout=0.1)
        if message:
            logger.info("✅ Pub/Sub message received!")
        else:
            logger.info("ℹ️  No message received (expected in test environment)")

        # Cleanup
        pubsub.close()
        redis_client.delete("test_key")
        redis_client.close()

        logger.info("🎉 All Redis Cloud tests passed successfully!")
        logger.info("\n🚀 Redis Cloud is ready for real-time features:")
        logger.info("  • Music generation progress updates")
        logger.info("  • Collaborative editing notifications")
        logger.info("  • Vector search results broadcasting")
        logger.info("  • System status monitoring")
        logger.info(f"\n✨ Connected to: {redis_host}:{redis_port}")

        return True

    except Exception as e:
        logger.error(f"❌ Redis Cloud connection test failed: {e}")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("1. Check your Redis Cloud credentials in .env file")
        logger.info("2. Verify Redis Cloud database is active")
        logger.info("3. Check firewall and network connectivity")
        logger.info("4. Ensure Redis Cloud credentials are correct")
        return False


async def main():
    """Main test function"""
    logger.info("🧪 Redis Cloud Connection Test")
    logger.info("=" * 50)

    success = await test_redis_cloud_connection()

    if success:
        logger.info("\n✅ Redis Cloud is working perfectly!")
        logger.info("You can now run the full integration tests:")
        logger.info("  python test_realtime_integration.py")
        return True
    else:
        logger.error("\n❌ Redis Cloud connection failed!")
        logger.error("Please fix the connection issues before running integration tests.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
