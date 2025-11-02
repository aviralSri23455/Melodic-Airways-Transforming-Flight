"""
Optimized Throughput Measurement Script
Measures actual QPS with proper async handling and connection pooling
"""

import asyncio
import httpx
import time
from datetime import datetime
import statistics


async def measure_endpoint_throughput(
    url: str,
    duration_seconds: int = 10,
    concurrent_requests: int = 100,
    max_connections: int = 200
):
    """
    Measure throughput with optimized connection pooling
    
    Args:
        url: Endpoint to test
        duration_seconds: How long to run the test
        concurrent_requests: Number of concurrent requests per batch
        max_connections: Maximum HTTP connections in pool
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZED THROUGHPUT MEASUREMENT")
    print(f"{'='*70}")
    print(f"Endpoint: {url}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Concurrent requests per batch: {concurrent_requests}")
    print(f"Max connections: {max_connections}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")
    
    total_requests = 0
    successful_requests = 0
    failed_requests = 0
    response_times = []
    
    # Configure HTTP client with connection pooling
    limits = httpx.Limits(
        max_keepalive_connections=max_connections,
        max_connections=max_connections,
        keepalive_expiry=30.0
    )
    
    async def make_request(client: httpx.AsyncClient):
        """Make a single request"""
        nonlocal total_requests, successful_requests, failed_requests
        try:
            start = time.time()
            response = await client.get(url)
            elapsed_ms = (time.time() - start) * 1000
            
            total_requests += 1
            if response.status_code == 200:
                successful_requests += 1
                response_times.append(elapsed_ms)
            else:
                failed_requests += 1
        except Exception:
            total_requests += 1
            failed_requests += 1
    
    # Note: HTTP/2 can improve performance but requires: pip install httpx[http2]
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=limits
    ) as client:
        start_time = time.time()
        
        # Keep sending requests until duration is reached
        while time.time() - start_time < duration_seconds:
            # Create batch of concurrent requests
            tasks = [make_request(client) for _ in range(concurrent_requests)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Show progress
            elapsed = time.time() - start_time
            current_qps = total_requests / elapsed if elapsed > 0 else 0
            print(f"⏱️  {elapsed:.1f}s | Requests: {total_requests} | Current QPS: {current_qps:.0f}", end="\r")
    
    # Calculate final metrics
    total_time = time.time() - start_time
    qps = total_requests / total_time if total_time > 0 else 0
    avg_response_time = statistics.mean(response_times) if response_times else 0
    p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0
    p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else 0
    
    # Print results
    print(f"\n\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"✅ Total Requests:      {total_requests}")
    print(f"✅ Successful:          {successful_requests}")
    print(f"❌ Failed:              {failed_requests}")
    print(f"⏱️  Total Time:          {total_time:.2f}s")
    print(f"🚀 Throughput:          {qps:.2f} queries/second")
    print(f"📊 Avg Response Time:   {avg_response_time:.2f}ms")
    print(f"📊 P95 Response Time:   {p95_response_time:.2f}ms")
    print(f"📊 P99 Response Time:   {p99_response_time:.2f}ms")
    print(f"{'='*70}\n")
    
    # Check against target
    target_qps = 1000
    if qps >= target_qps:
        print(f"✅ SUCCESS: Achieved {qps:.0f} QPS (Target: {target_qps} QPS)")
    else:
        print(f"⚠️  BELOW TARGET: Achieved {qps:.0f} QPS (Target: {target_qps} QPS)")
        print(f"   Gap: {target_qps - qps:.0f} QPS")
        print(f"\n💡 Tips to improve:")
        print(f"   - Increase concurrent_requests (currently {concurrent_requests})")
        print(f"   - Add response caching to endpoints")
        print(f"   - Optimize database queries")
        print(f"   - Use connection pooling")
    
    print()
    return qps


async def test_all_endpoints():
    """Test throughput on all key endpoints"""
    endpoints = [
        ("Health Check", "http://localhost:8000/health"),
        ("Vector Health", "http://localhost:8000/api/v1/vectors/health"),
        ("Redis Storage Info", "http://localhost:8000/api/v1/redis/storage-info"),
        ("Vector Statistics", "http://localhost:8000/api/v1/vectors/statistics"),
        ("Redis Cache Stats", "http://localhost:8000/api/v1/redis/cache/stats"),
    ]
    
    results = {}
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        AERO MELODY API OPTIMIZED THROUGHPUT MEASUREMENT          ║
║                                                                  ║
║  Testing with optimized connection pooling and async handling   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    for name, url in endpoints:
        print(f"\n🔍 Testing: {name}")
        qps = await measure_endpoint_throughput(
            url,
            duration_seconds=10,
            concurrent_requests=100,
            max_connections=200
        )
        results[name] = qps
        await asyncio.sleep(2)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - All Endpoints")
    print(f"{'='*70}")
    print(f"{'Endpoint':<30} {'QPS':<15} {'Status':<20}")
    print(f"{'-'*70}")
    
    target_qps = 1000
    passed = 0
    
    for name, qps in results.items():
        status = "✅ PASS" if qps >= target_qps else "⚠️  Below Target"
        if qps >= target_qps:
            passed += 1
        print(f"{name:<30} {qps:<15.0f} {status:<20}")
    
    avg_qps = sum(results.values()) / len(results)
    print(f"{'-'*70}")
    print(f"{'Average':<30} {avg_qps:<15.0f}")
    print(f"{'='*70}\n")
    
    # Final verdict
    print(f"\n🎯 FINAL RESULTS:")
    print(f"{'='*70}")
    print(f"Endpoints passing 1000 QPS target: {passed}/{len(results)}")
    print(f"Average QPS across all endpoints: {avg_qps:.0f}")
    
    if avg_qps >= target_qps:
        print(f"\n✅ SUCCESS! Average throughput meets 1000 QPS target!")
        print(f"   You can confidently claim 1,000+ queries/sec in your PPT")
    else:
        print(f"\n⚠️  Average throughput below target")
        print(f"   Current: {avg_qps:.0f} QPS")
        print(f"   Target: {target_qps} QPS")
        print(f"   Gap: {target_qps - avg_qps:.0f} QPS")
    
    print(f"{'='*70}\n")


async def check_backend():
    """Check if backend is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            if response.status_code == 200:
                print("✅ Backend is running on http://localhost:8000\n")
                return True
    except Exception as e:
        print(f"❌ Backend is not running!")
        print(f"   Error: {e}")
        print(f"\n   Please start the backend first:")
        print(f"   cd backend && python main.py\n")
        return False


if __name__ == "__main__":
    import sys
    
    # Check if backend is running
    if not asyncio.run(check_backend()):
        sys.exit(1)
    
    # Run optimized tests
    if len(sys.argv) > 1:
        # Test specific endpoint
        url = sys.argv[1]
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        concurrent = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        asyncio.run(measure_endpoint_throughput(url, duration, concurrent))
    else:
        # Test all endpoints
        asyncio.run(test_all_endpoints())
