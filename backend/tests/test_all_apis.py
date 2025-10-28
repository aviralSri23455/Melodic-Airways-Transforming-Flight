#!/usr/bin/env python3
"""
Comprehensive API Test Script for Aero Melody Backend
Tests all endpoints without authentication requirements
Run with: python test_all_apis.py
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

class APITester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def test_endpoint(self, method: str, endpoint: str, data: Dict = None, description: str = "") -> Dict:
        """Test a single endpoint"""
        url = f"{API_BASE}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=10)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, timeout=10)
            else:
                return {"error": f"Unsupported method: {method}"}

            result = {
                "endpoint": f"{method} {endpoint}",
                "description": description,
                "status_code": response.status_code,
                "success": 200 <= response.status_code < 300,
                "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }

            if result["success"]:
                print(f"✅ {result['endpoint']} - {description}")
                self.passed += 1
            else:
                print(f"❌ {result['endpoint']} - {description} (Status: {result['status_code']})")
                self.failed += 1

            self.results.append(result)
            return result

        except Exception as e:
            error_result = {
                "endpoint": f"{method} {endpoint}",
                "description": description,
                "error": str(e),
                "success": False
            }
            print(f"❌ {error_result['endpoint']} - {description} (Error: {str(e)})")
            self.failed += 1
            self.results.append(error_result)
            return error_result

    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
        print(f"{'='*60}")

        if self.failed > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result.get("success", False) and "error" not in result:
                    print(f"   {result['endpoint']} - Status: {result['status_code']}")

def main():
    """Run all API tests"""
    print("🚀 Aero Melody Backend API Test (No Auth Required)")
    print("=" * 60)

    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(3)

    tester = APITester()

    # Test basic health
    tester.test_endpoint("GET", "/health", description="Basic health check")

    # Test Redis endpoints (no auth required)
    tester.test_endpoint("GET", "/redis/vectors/stats", description="Vector database statistics")
    tester.test_endpoint("GET", "/redis/cache/stats", description="Redis cache statistics")
    tester.test_endpoint("GET", "/redis/system/health", description="System health status")

    # Test vector search (no auth required)
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tester.test_endpoint(
        "POST",
        "/redis/vectors/search/music",
        {"query_vector": test_vector, "limit": 5},
        "Music vector similarity search"
    )

    tester.test_endpoint(
        "POST",
        "/redis/vectors/search/routes?origin=JFK&destination=LAX&limit=5",
        None,
        "Route similarity search"
    )

    # Test session management (no auth required)
    tester.test_endpoint(
        "POST",
        "/redis/sessions/create?origin=JFK&destination=LAX&session_type=generation",
        None,
        "Create collaboration session"
    )

    # Test storage and monitoring (no auth required)
    tester.test_endpoint("GET", "/redis/storage/info", description="Redis storage information")
    tester.test_endpoint("GET", "/redis/sessions/stats", description="Session statistics")
    tester.test_endpoint("GET", "/redis/live/sessions/active", description="Live active sessions")

    # Test public API endpoints (should work with /api/v1 prefix)
    tester.test_endpoint("GET", "/airports/search?query=New+York", description="Search airports")
    tester.test_endpoint("GET", "/airports/JFK", description="Get airport by IATA code")
    tester.test_endpoint("GET", "/public/compositions", description="Get public compositions")

    # Test routes (should work with /api/v1 prefix for basic data)
    tester.test_endpoint("GET", "/routes?limit=5", description="Get flight routes")

    # Test new demo endpoints
    tester.test_endpoint("GET", "/demo/tech-stack-status", description="Tech stack health check")
    tester.test_endpoint("GET", "/demo/demo-examples", description="Demo examples and use cases")
    
    # Test analytics showcase endpoints
    tester.test_endpoint("GET", "/analytics-showcase/pitch-complexity-by-continent", description="Pitch complexity by continent")
    tester.test_endpoint("GET", "/analytics-showcase/most-connected-airport-sounds", description="Most connected airport sounds")
    tester.test_endpoint("GET", "/analytics-showcase/real-time-composition-metrics", description="Real-time composition metrics")

    # Print summary
    tester.print_summary()

    # Save detailed results to file
    with open("api_test_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(tester.results),
            "passed": tester.passed,
            "failed": tester.failed,
            "success_rate": round(tester.passed / (tester.passed + tester.failed) * 100, 1) if (tester.passed + tester.failed) > 0 else 0,
            "results": tester.results
        }, f, indent=2)

    print("💾 Detailed results saved to api_test_results.json")

    # Exit with appropriate code
    if tester.failed > 0:
        print("⚠️  Some tests failed. Check the output above.")
        sys.exit(1)
    else:
        print("🎉 All tests passed! Backend is ready for frontend integration.")
        sys.exit(0)

if __name__ == "__main__":
    main()
