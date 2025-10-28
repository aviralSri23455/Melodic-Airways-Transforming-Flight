#!/usr/bin/env python3
"""
Real-time Functionality Test
Tests Redis, WebSocket, and DuckDB integration for real-time features
"""

import asyncio
import json
import time
import websockets
import redis
from datetime import datetime
import duckdb
import requests

# Configuration
BACKEND_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/api/v1/demo/redis-subscriber"
REDIS_URL = "redis://default:zcUJQD3G4uebZD0Ve5hz6J171zwohat2@redis-16441.c267.us-east-1-4.ec2.redns.redis-cloud.com:16441"
DUCKDB_PATH = "data/analytics.duckdb"

class RealtimeTester:
    def __init__(self):
        self.redis_client = None
        self.websocket_messages = []
        self.test_results = {}
        
    async def test_redis_connection(self):
        """Test Redis connection and pub/sub functionality"""
        print("🔴 Testing Redis Connection...")
        
        try:
            # Test Redis connection
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            
            # Test basic operations
            test_key = f"test:realtime:{int(time.time())}"
            test_value = json.dumps({
                "test": "redis_connection",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Redis is working!"
            })
            
            # Set test value
            self.redis_client.set(test_key, test_value)
            retrieved = self.redis_client.get(test_key)
            
            if retrieved == test_value:
                print("✅ Redis basic operations: PASS")
                self.test_results["redis_basic"] = True
            else:
                print("❌ Redis basic operations: FAIL")
                self.test_results["redis_basic"] = False
            
            # Test pub/sub
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("test:realtime:channel")
            
            # Publish test message
            test_message = json.dumps({
                "type": "test_pubsub",
                "timestamp": datetime.utcnow().isoformat(),
                "data": "Pub/Sub test message"
            })
            
            self.redis_client.publish("test:realtime:channel", test_message)
            
            # Wait for message
            message = pubsub.get_message(timeout=2)
            if message and message['type'] == 'message':
                print("✅ Redis pub/sub: PASS")
                self.test_results["redis_pubsub"] = True
            else:
                print("❌ Redis pub/sub: FAIL")
                self.test_results["redis_pubsub"] = False
                
            pubsub.close()
            
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.test_results["redis_connection"] = False
            
    async def test_websocket_connection(self):
        """Test WebSocket connection and message handling"""
        print("🔌 Testing WebSocket Connection...")
        
        try:
            async with websockets.connect(WEBSOCKET_URL) as websocket:
                print("✅ WebSocket connected successfully")
                
                # Wait for welcome message
                welcome_message = await asyncio.wait_for(websocket.recv(), timeout=5)
                welcome_data = json.loads(welcome_message)
                
                if welcome_data.get("type") == "connection_established":
                    print("✅ WebSocket welcome message: PASS")
                    self.test_results["websocket_welcome"] = True
                else:
                    print("❌ WebSocket welcome message: FAIL")
                    self.test_results["websocket_welcome"] = False
                
                # Send test message
                test_message = json.dumps({
                    "type": "test_message",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": "WebSocket test from Python"
                })
                
                await websocket.send(test_message)
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                response_data = json.loads(response)
                
                if response_data.get("type") == "echo":
                    print("✅ WebSocket message echo: PASS")
                    self.test_results["websocket_echo"] = True
                else:
                    print("❌ WebSocket message echo: FAIL")
                    self.test_results["websocket_echo"] = False
                    
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            self.test_results["websocket_connection"] = False
            
    def test_duckdb_connection(self):
        """Test DuckDB connection and basic operations"""
        print("🦆 Testing DuckDB Connection...")
        
        try:
            # Test DuckDB connection
            conn = duckdb.connect(DUCKDB_PATH)
            
            # Test basic query
            result = conn.execute("SELECT 1 as test_column").fetchall()
            
            if result and result[0][0] == 1:
                print("✅ DuckDB basic query: PASS")
                self.test_results["duckdb_basic"] = True
            else:
                print("❌ DuckDB basic query: FAIL")
                self.test_results["duckdb_basic"] = False
            
            # Test if analytics tables exist
            tables_query = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'main' 
            AND table_name IN ('flight_routes', 'analytics_summary')
            """
            
            tables = conn.execute(tables_query).fetchall()
            table_names = [row[0] for row in tables]
            
            if 'flight_routes' in table_names:
                print("✅ DuckDB flight_routes table: PASS")
                self.test_results["duckdb_tables"] = True
            else:
                print("⚠️ DuckDB tables not found (may need initial data load)")
                self.test_results["duckdb_tables"] = False
                
            conn.close()
            
        except Exception as e:
            print(f"❌ DuckDB connection failed: {e}")
            self.test_results["duckdb_connection"] = False
            
    async def test_backend_api(self):
        """Test backend API endpoints"""
        print("🌐 Testing Backend API...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            
            if response.status_code == 200:
                print("✅ Backend health endpoint: PASS")
                self.test_results["backend_health"] = True
            else:
                print("❌ Backend health endpoint: FAIL")
                self.test_results["backend_health"] = False
            
            # Test demo endpoint (triggers Redis pub/sub)
            response = requests.get(
                f"{BACKEND_URL}/api/v1/demo/test-redis-subscriber", 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("redis_test") == "success":
                    print("✅ Backend Redis integration: PASS")
                    self.test_results["backend_redis"] = True
                else:
                    print("❌ Backend Redis integration: FAIL")
                    self.test_results["backend_redis"] = False
            else:
                print("❌ Backend demo endpoint: FAIL")
                self.test_results["backend_demo"] = False
                
        except Exception as e:
            print(f"❌ Backend API test failed: {e}")
            self.test_results["backend_api"] = False
            
    async def test_realtime_workflow(self):
        """Test complete real-time workflow"""
        print("🔄 Testing Complete Real-time Workflow...")
        
        try:
            # Start WebSocket listener
            async with websockets.connect(WEBSOCKET_URL) as websocket:
                # Wait for welcome
                await asyncio.wait_for(websocket.recv(), timeout=5)
                
                # Trigger backend action that publishes to Redis
                response = requests.get(
                    f"{BACKEND_URL}/api/v1/demo/complete-demo?origin=JFK&destination=LHR",
                    timeout=15
                )
                
                if response.status_code == 200:
                    # Wait for WebSocket message from Redis
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=10)
                        message_data = json.loads(message)
                        
                        if message_data.get("type") in ["music_generated", "generation_progress"]:
                            print("✅ Real-time workflow: PASS")
                            self.test_results["realtime_workflow"] = True
                        else:
                            print("⚠️ Real-time workflow: PARTIAL (connected but no Redis message)")
                            self.test_results["realtime_workflow"] = False
                            
                    except asyncio.TimeoutError:
                        print("❌ Real-time workflow: FAIL (no WebSocket message received)")
                        self.test_results["realtime_workflow"] = False
                else:
                    print("❌ Real-time workflow: FAIL (backend request failed)")
                    self.test_results["realtime_workflow"] = False
                    
        except Exception as e:
            print(f"❌ Real-time workflow test failed: {e}")
            self.test_results["realtime_workflow"] = False
            
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("🎯 REAL-TIME FUNCTIONALITY TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:25} : {status}")
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Real-time functionality is working correctly.")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️ MOST TESTS PASSED. Real-time functionality is mostly working.")
        else:
            print("❌ MANY TESTS FAILED. Real-time functionality needs attention.")
            
        print("="*60)
        
        # Provide specific recommendations
        print("\n📋 RECOMMENDATIONS:")
        
        if not self.test_results.get("redis_connection", True):
            print("- Check Redis URL and network connectivity")
            print("- Verify Redis Cloud credentials")
            
        if not self.test_results.get("websocket_connection", True):
            print("- Check if backend server is running on localhost:8000")
            print("- Verify WebSocket endpoint configuration")
            
        if not self.test_results.get("duckdb_connection", True):
            print("- Check DuckDB file permissions")
            print("- Verify data/analytics.duckdb file exists")
            
        if not self.test_results.get("realtime_workflow", True):
            print("- Check Redis pub/sub integration")
            print("- Verify backend WebSocket message broadcasting")

async def main():
    """Run all real-time functionality tests"""
    print("🚀 Starting Real-time Functionality Tests...")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"WebSocket URL: {WEBSOCKET_URL}")
    print(f"Redis URL: {REDIS_URL}")
    print("-" * 60)
    
    tester = RealtimeTester()
    
    # Run all tests
    await tester.test_redis_connection()
    await tester.test_websocket_connection()
    tester.test_duckdb_connection()
    await tester.test_backend_api()
    await tester.test_realtime_workflow()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
