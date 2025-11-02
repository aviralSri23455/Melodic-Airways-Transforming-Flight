"""
Throughput Performance Test for Aero Melody API
Measures queries per second (QPS) for various endpoints
"""

import asyncio
import httpx
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime
import json


class ThroughputTester:
    """Test API throughput and performance"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def test_endpoint(
        self,
        endpoint: str,
        num_requests: int = 1000,
        concurrent: int = 50,
        method: str = "GET",
        payload: Dict = None
    ) -> Dict[str, Any]:
        """
        Test a single endpoint's throughput
        
        Args:
            endpoint: API endpoint to test
            num_requests: Total number of requests to make
            concurrent: Number of concurrent requests
            method: HTTP method (GET, POST, etc.)
            payload: Request payload for POST requests
        
        Returns:
            Performance metrics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Testing: {method} {endpoint}")
        print(f"Total requests: {num_requests}")
        print(f"Concurrent requests: {concurrent}")
        print(f"{'='*60}")
        
        url = f"{self.base_url}{endpoint}"
        response_times = []
        errors = 0
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            
            # Create batches of concurrent requests
            for batch_start in range(0, num_requests, concurrent):
                batch_size = min(concurrent, num_requests - batch_start)
                tasks = []
                
                for _ in range(batch_size):
                    if method == "GET":
                        task = self._make_get_request(client, url, response_times)
                    elif method == "POST":
                        task = self._make_post_request(client, url, payload, response_times)
                    tasks.append(task)
                
                # Execute batch
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count errors
                for result in results:
                    if isinstance(result, Exception):
                        errors += 1
                
                # Progress indicator
                completed = batch_start + batch_size
                progress = (completed / num_requests) * 100
                print(f"Progress: {completed}/{num_requests} ({progress:.1f}%)", end="\r")
            
            total_time = time.time() - start_time
        
        # Calculate metrics
        successful_requests = len(response_times)
        qps = successful_requests / total_time if total_time > 0 else 0
        
        metrics = {
            "endpoint": endpoint,
            "method": method,
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": errors,
            "total_time_seconds": round(total_time, 2),
            "queries_per_second": round(qps, 2),
            "avg_response_time_ms": round(statistics.mean(response_times), 2) if response_times else 0,
            "min_response_time_ms": round(min(response_times), 2) if response_times else 0,
            "max_response_time_ms": round(max(response_times), 2) if response_times else 0,
            "median_response_time_ms": round(statistics.median(response_times), 2) if response_times else 0,
            "p95_response_time_ms": round(self._percentile(response_times, 95), 2) if response_times else 0,
            "p99_response_time_ms": round(self._percentile(response_times, 99), 2) if response_times else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(metrics)
        self._print_metrics(metrics)
        
        return metrics
    
    async def _make_get_request(self, client: httpx.AsyncClient, url: str, response_times: List[float]):
        """Make a GET request and record response time"""
        try:
            start = time.time()
            response = await client.get(url)
            elapsed_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                response_times.append(elapsed_ms)
            
            return response.status_code
        except Exception as e:
            return e
    
    async def _make_post_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: Dict,
        response_times: List[float]
    ):
        """Make a POST request and record response time"""
        try:
            start = time.time()
            response = await client.post(url, json=payload)
            elapsed_ms = (time.time() - start) * 1000
            
            if response.status_code in [200, 201]:
                response_times.append(elapsed_ms)
            
            return response.status_code
        except Exception as e:
            return e
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a readable format"""
        print(f"\n\n{'='*60}")
        print(f"RESULTS: {metrics['method']} {metrics['endpoint']}")
        print(f"{'='*60}")
        print(f"✅ Successful: {metrics['successful_requests']}/{metrics['total_requests']}")
        print(f"❌ Failed: {metrics['failed_requests']}")
        print(f"⏱️  Total Time: {metrics['total_time_seconds']}s")
        print(f"🚀 Throughput: {metrics['queries_per_second']} queries/sec")
        print(f"\nResponse Times:")
        print(f"  Average: {metrics['avg_response_time_ms']}ms")
        print(f"  Median:  {metrics['median_response_time_ms']}ms")
        print(f"  Min:     {metrics['min_response_time_ms']}ms")
        print(f"  Max:     {metrics['max_response_time_ms']}ms")
        print(f"  P95:     {metrics['p95_response_time_ms']}ms")
        print(f"  P99:     {metrics['p99_response_time_ms']}ms")
        print(f"{'='*60}\n")
    
    def save_results(self, filename: str = "throughput_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Results saved to {filename}")
    
    def print_summary(self):
        """Print summary of all tests"""
        if not self.results:
            print("No test results available")
            return
        
        print(f"\n\n{'='*60}")
        print("THROUGHPUT TEST SUMMARY")
        print(f"{'='*60}")
        print(f"{'Endpoint':<40} {'QPS':<10} {'Avg (ms)':<10}")
        print(f"{'-'*60}")
        
        for result in self.results:
            endpoint = result['endpoint'][:37] + "..." if len(result['endpoint']) > 40 else result['endpoint']
            print(f"{endpoint:<40} {result['queries_per_second']:<10} {result['avg_response_time_ms']:<10}")
        
        print(f"{'='*60}\n")


async def run_comprehensive_test():
    """Run comprehensive throughput tests on various endpoints"""
    tester = ThroughputTester()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         AERO MELODY API THROUGHPUT PERFORMANCE TEST          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Test 1: Health check (lightweight endpoint)
    await tester.test_endpoint(
        endpoint="/health",
        num_requests=1000,
        concurrent=50
    )
    
    # Test 2: Vector statistics (database query)
    await tester.test_endpoint(
        endpoint="/api/v1/vectors/statistics",
        num_requests=500,
        concurrent=25
    )
    
    # Test 3: Redis cache info
    await tester.test_endpoint(
        endpoint="/api/v1/redis/storage-info",
        num_requests=1000,
        concurrent=50
    )
    
    # Test 4: Analytics performance metrics
    await tester.test_endpoint(
        endpoint="/api/v1/analytics/performance-metrics",
        num_requests=500,
        concurrent=25
    )
    
    # Test 5: Vector health check
    await tester.test_endpoint(
        endpoint="/api/v1/vectors/health",
        num_requests=1000,
        concurrent=50
    )
    
    # Print summary
    tester.print_summary()
    
    # Save results
    tester.save_results("backend/tests/throughput_results.json")
    
    # Check if we meet the 1000 QPS target
    print("\n🎯 TARGET ANALYSIS:")
    print(f"{'='*60}")
    target_qps = 1000
    
    for result in tester.results:
        qps = result['queries_per_second']
        status = "✅ PASS" if qps >= target_qps else "⚠️  BELOW TARGET"
        print(f"{result['endpoint'][:40]:<40} {status}")
        print(f"  Achieved: {qps} QPS (Target: {target_qps} QPS)")
    
    print(f"{'='*60}\n")


async def quick_test():
    """Quick test with fewer requests"""
    tester = ThroughputTester()
    
    print("\n🚀 Quick Throughput Test (100 requests)\n")
    
    await tester.test_endpoint(
        endpoint="/health",
        num_requests=100,
        concurrent=10
    )
    
    tester.print_summary()


if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║              AERO MELODY THROUGHPUT TESTER                   ║
║                                                              ║
║  This script measures API throughput (queries per second)    ║
║  Make sure the backend is running on http://localhost:8000   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if backend is running
    async def check_backend():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/health", timeout=5.0)
                if response.status_code == 200:
                    print("✅ Backend is running\n")
                    return True
        except Exception as e:
            print(f"❌ Backend is not running: {e}")
            print("Please start the backend with: python backend/main.py")
            return False
    
    if not asyncio.run(check_backend()):
        sys.exit(1)
    
    # Run tests
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        asyncio.run(quick_test())
    else:
        asyncio.run(run_comprehensive_test())
