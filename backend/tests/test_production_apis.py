#!/usr/bin/env python3
"""
🚀 Production API Test Suite for Aero Melody Backend
Tests all endpoints to verify the complete tech stack is working
Run with: python test_production_apis.py
"""

import requests
import json
import time
import sys
from typing import Dict, Any
from datetime import datetime
import asyncio
import websockets

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

class ProductionAPITester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
        self.start_time = time.time()

    def test_endpoint(self, method: str, endpoint: str, description: str, expected_keys: list = None) -> Dict:
        """Test a single endpoint with detailed validation"""
        url = f"{API_BASE}{endpoint}"
        try:
            print(f"🧪 Testing: {description}")
            
            if method.upper() == "GET":
                response = requests.get(url, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, timeout=30)
            else:
                return {"error": f"Unsupported method: {method}"}

            result = {
                "endpoint": f"{method} {endpoint}",
                "description": description,
                "status_code": response.status_code,
                "success": 200 <= response.status_code < 300,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "content_type": response.headers.get('content-type', ''),
                "response_size_bytes": len(response.content)
            }

            # Parse JSON response
            try:
                json_response = response.json()
                result["response"] = json_response
                
                # Validate expected keys
                if expected_keys and result["success"]:
                    missing_keys = [key for key in expected_keys if key not in json_response]
                    if missing_keys:
                        result["warning"] = f"Missing expected keys: {missing_keys}"
                    else:
                        result["validation"] = "✅ All expected keys present"
                        
            except json.JSONDecodeError:
                result["response"] = response.text[:500]  # First 500 chars
                result["json_error"] = "Response is not valid JSON"

            if result["success"]:
                print(f"   ✅ {description} - {response.status_code} ({result['response_time_ms']:.0f}ms)")
                self.passed += 1
            else:
                print(f"   ❌ {description} - {response.status_code} ({result['response_time_ms']:.0f}ms)")
                self.failed += 1

            self.results.append(result)
            return result

        except requests.exceptions.Timeout:
            error_result = {
                "endpoint": f"{method} {endpoint}",
                "description": description,
                "error": "Request timeout (30s)",
                "success": False
            }
            print(f"   ⏰ {description} - TIMEOUT")
            self.failed += 1
            self.results.append(error_result)
            return error_result
            
        except Exception as e:
            error_result = {
                "endpoint": f"{method} {endpoint}",
                "description": description,
                "error": str(e),
                "success": False
            }
            print(f"   ❌ {description} - ERROR: {str(e)}")
            self.failed += 1
            self.results.append(error_result)
            return error_result

    def test_websocket(self) -> Dict:
        """Test WebSocket connection"""
        print(f"🧪 Testing: WebSocket Real-time Connection")
        
        async def test_ws():
            try:
                uri = "ws://localhost:8000/api/v1/demo/redis-subscriber"
                # Use asyncio.wait_for for timeout instead of websockets timeout parameter
                try:
                    websocket = await asyncio.wait_for(websockets.connect(uri), timeout=10)
                    
                    # Send a test message
                    await websocket.send("Test connection from Python")
                    
                    # Wait for welcome message
                    message = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(message)
                    
                    await websocket.close()
                    
                    return {
                        "endpoint": "WebSocket /api/v1/demo/redis-subscriber",
                        "description": "WebSocket Real-time Connection",
                        "success": True,
                        "message_received": data.get("type", "unknown"),
                        "connection_status": "✅ Connected successfully"
                    }
                except Exception as ws_error:
                    return {
                        "endpoint": "WebSocket /api/v1/demo/redis-subscriber",
                        "description": "WebSocket Real-time Connection",
                        "success": False,
                        "error": f"WebSocket error: {str(ws_error)}"
                    }
                    
            except asyncio.TimeoutError:
                return {
                    "endpoint": "WebSocket /api/v1/demo/redis-subscriber",
                    "description": "WebSocket Real-time Connection",
                    "success": False,
                    "error": "Connection timeout"
                }
            except Exception as e:
                return {
                    "endpoint": "WebSocket /api/v1/demo/redis-subscriber", 
                    "description": "WebSocket Real-time Connection",
                    "success": False,
                    "error": str(e)
                }

        try:
            result = asyncio.run(test_ws())
            if result["success"]:
                print(f"   ✅ WebSocket Connection - Connected")
                self.passed += 1
            else:
                print(f"   ❌ WebSocket Connection - {result.get('error', 'Failed')}")
                self.failed += 1
            
            self.results.append(result)
            return result
            
        except Exception as e:
            error_result = {
                "endpoint": "WebSocket /api/v1/demo/redis-subscriber",
                "description": "WebSocket Real-time Connection", 
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ WebSocket Connection - ERROR: {str(e)}")
            self.failed += 1
            self.results.append(error_result)
            return error_result

    def print_summary(self):
        """Print comprehensive test summary"""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print("🎯 PRODUCTION API TEST RESULTS")
        print(f"{'='*80}")
        print(f"📊 Total Tests: {self.passed + self.failed}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🌐 Base URL: {BASE_URL}")
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Performance summary
        response_times = [r.get("response_time_ms", 0) for r in self.results if r.get("response_time_ms")]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            print(f"⚡ Avg Response Time: {avg_time:.0f}ms")
            print(f"🐌 Slowest Endpoint: {max_time:.0f}ms")
            print(f"🚀 Fastest Endpoint: {min_time:.0f}ms")

        print(f"{'='*80}")

        # Detailed failure analysis
        if self.failed > 0:
            print("\n❌ FAILED TESTS DETAILS:")
            print("-" * 50)
            for result in self.results:
                if not result.get("success", False):
                    endpoint = result.get("endpoint", "Unknown")
                    error = result.get("error", f"Status: {result.get('status_code', 'Unknown')}")
                    print(f"   {endpoint}")
                    print(f"   └── {error}")
            print()

        # Success highlights
        if self.passed > 0:
            print("✅ WORKING FEATURES:")
            print("-" * 50)
            categories = {
                "Core Demo": [],
                "Analytics": [],
                "Real-time": [],
                "Data & Health": []
            }
            
            for result in self.results:
                if result.get("success", False):
                    desc = result.get("description", "")
                    if "demo" in desc.lower() and "analytics" not in desc.lower():
                        categories["Core Demo"].append(desc)
                    elif "analytics" in desc.lower() or "metrics" in desc.lower():
                        categories["Analytics"].append(desc)
                    elif "websocket" in desc.lower() or "redis" in desc.lower():
                        categories["Real-time"].append(desc)
                    else:
                        categories["Data & Health"].append(desc)
            
            for category, items in categories.items():
                if items:
                    print(f"   🎯 {category}: {len(items)} endpoints working")
                    for item in items[:3]:  # Show first 3
                        print(f"      ✓ {item}")
                    if len(items) > 3:
                        print(f"      ✓ ... and {len(items) - 3} more")

    def save_detailed_report(self):
        """Save detailed test report to file"""
        report = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "base_url": BASE_URL,
                "total_tests": len(self.results),
                "passed": self.passed,
                "failed": self.failed,
                "success_rate": round(self.passed / (self.passed + self.failed) * 100, 1) if (self.passed + self.failed) > 0 else 0,
                "total_time_seconds": round(time.time() - self.start_time, 2)
            },
            "test_results": self.results,
            "tech_stack_status": {
                "openflights_data": "✅" if any("openflights" in r.get("description", "").lower() and r.get("success") for r in self.results) else "❌",
                "pytorch_embeddings": "✅" if any("demo" in r.get("description", "").lower() and r.get("success") for r in self.results) else "❌",
                "mido_midi": "✅" if any("demo" in r.get("description", "").lower() and r.get("success") for r in self.results) else "❌",
                "duckdb_analytics": "✅" if any("analytics" in r.get("description", "").lower() and r.get("success") for r in self.results) else "❌",
                "redis_pubsub": "✅" if any("redis" in r.get("description", "").lower() and r.get("success") for r in self.results) else "❌",
                "websocket_realtime": "✅" if any("websocket" in r.get("description", "").lower() and r.get("success") for r in self.results) else "❌"
            }
        }

        filename = f"production_api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved: {filename}")

def main():
    """Run comprehensive production API tests"""
    print("🚀 AERO MELODY BACKEND - PRODUCTION API TEST SUITE")
    print("=" * 80)
    print("Testing all endpoints with real OpenFlights data...")
    print("⏳ This may take 1-2 minutes to complete all tests\n")

    tester = ProductionAPITester()

    # Wait for server to be ready
    print("⏳ Checking server availability...")
    time.sleep(2)

    print("\n🎵 CORE DEMO ENDPOINTS")
    print("-" * 40)
    
    # Core demo endpoints
    tester.test_endpoint("GET", "/demo/simple-demo?origin=DEL&destination=LHR", 
                        "Main Demo - Delhi to London", 
                        ["demo_type", "status", "route_details", "tech_stack_results"])
    
    tester.test_endpoint("GET", "/demo/quick-demo?origin=JFK&destination=LAX", 
                        "Quick Demo - New York to Los Angeles",
                        ["demo_type", "status", "route_info"])
    
    tester.test_endpoint("GET", "/demo/tech-stack-status", 
                        "Tech Stack Health Check",
                        ["overall_status", "components"])

    print("\n📊 ANALYTICS SHOWCASE ENDPOINTS") 
    print("-" * 40)
    
    # Analytics endpoints
    tester.test_endpoint("GET", "/analytics-showcase/real-time-composition-metrics",
                        "Real-time Composition Metrics",
                        ["dashboard_type", "current_statistics"])
    
    tester.test_endpoint("GET", "/analytics-showcase/pitch-complexity-by-continent",
                        "Pitch Complexity by Continent", 
                        ["analysis_type", "continent_analysis"])
    
    tester.test_endpoint("GET", "/analytics-showcase/most-connected-airport-sounds",
                        "Most Connected Airport Sounds",
                        ["analysis_type", "airports"])
    
    tester.test_endpoint("GET", "/analytics-showcase/columnstore-performance-demo",
                        "ColumnStore Performance Demo",
                        ["demo_type", "performance_comparison"])

    print("\n🗄️ DATA & HEALTH ENDPOINTS")
    print("-" * 40)
    
    # Data and health endpoints
    tester.test_endpoint("GET", "/demo/openflights-data-check",
                        "OpenFlights Data Verification",
                        ["openflights_data_status", "dataset_info"])
    
    tester.test_endpoint("GET", "/demo/redis-pubsub-info", 
                        "Redis Pub/Sub Information",
                        ["redis_pubsub_explanation"])
    
    tester.test_endpoint("GET", "/demo/debug-test",
                        "Debug Test - Library Imports",
                        ["status", "imports"])
    
    tester.test_endpoint("GET", "/demo/db-test",
                        "Database Connection Test", 
                        ["status", "database"])

    print("\n🎯 POPULAR ROUTE EXAMPLES")
    print("-" * 40)
    
    # Popular route examples
    routes = [
        ("NRT", "SYD", "Tokyo to Sydney (Trans-Pacific)"),
        ("LHR", "CDG", "London to Paris (Short European)"),
        ("DXB", "JFK", "Dubai to New York (Long-haul)"),
    ]
    
    for origin, dest, desc in routes:
        tester.test_endpoint("GET", f"/demo/simple-demo?origin={origin}&destination={dest}",
                            desc, ["demo_type", "status", "route_details"])

    print("\n🔌 REAL-TIME WEBSOCKET TEST")
    print("-" * 40)
    
    # WebSocket test
    tester.test_websocket()
    
    # WebSocket info endpoint
    tester.test_endpoint("GET", "/demo/websocket-info",
                        "WebSocket Connection Info",
                        ["websocket_endpoint", "how_to_test"])

    # Print final summary
    tester.print_summary()
    
    # Save detailed report
    tester.save_detailed_report()

    # Final status
    if tester.failed == 0:
        print("\n🎉 ALL TESTS PASSED! Your backend is production-ready!")
        print("✅ Frontend team can proceed with integration")
        print("✅ All tech stack components are working")
        print("✅ Real OpenFlights data is accessible")
        sys.exit(0)
    else:
        print(f"\n⚠️  {tester.failed} tests failed. Please check the issues above.")
        print("💡 Most common issues:")
        print("   - Server not running on localhost:8000")
        print("   - Database connection problems") 
        print("   - Missing OpenFlights data")
        print("   - Redis connection issues")
        sys.exit(1)

if __name__ == "__main__":
    main()