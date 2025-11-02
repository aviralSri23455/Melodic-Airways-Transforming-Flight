"""
Simple Throughput Measurement Script
Run this while your backend is running to measure actual QPS
"""

import asyncio
import httpx
import time
from datetime import datetime


async def measure_throughput(
    url: str = "http://localhost:8000/health",
    duration_seconds: int = 10,
    concurrent_requests: int = 50
):
    """
    Measure throughput by sending continuous requests for a duration
    
    Args:
        url: Endpoint to test
        duration_seconds: How long to run the test
        concurrent_requests: Number of concurrent requests
    """
    print(f"\n{'='*70}")
    print(f"THROUGHPUT MEASUREMENT")
    print(f"{'='*70}")
    print(f"Endpoint: {url}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Concurrent requests: {concurrent_requests}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")
    
    total_requests = 0
    successful_requests = 0
    failed_requests = 0
    response_times = []
    
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
    
    async with httpx.AsyncClient(timeout=30.0) as client:
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
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
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
    print(f"{'='*70}\n")
    
    # Check against target
    target_qps = 1000
    if qps >= target_qps:
        print(f"✅ SUCCESS: Achieved {qps:.0f} QPS (Target: {target_qps} QPS)")
    else:
        print(f"⚠️  BELOW TARGET: Achieved {qps:.0f} QPS (Target: {target_qps} QPS)")
        print(f"   Gap: {target_qps - qps:.0f} QPS")
    
    print()
    return qps


async def test_multiple_endpoints():
    """Test throughput on multiple endpoints"""
    endpoints = [
        ("Health Check", "http://localhost:8000/health"),
        ("Vector Health", "http://localhost:8000/api/v1/vectors/health"),
        ("Redis Info", "http://localhost:8000/api/v1/redis/storage-info"),
        ("Vector Stats", "http://localhost:8000/api/v1/vectors/statistics"),
    ]
    
    results = {}
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           AERO MELODY API THROUGHPUT MEASUREMENT                 ║
║                                                                  ║
║  Testing multiple endpoints to measure queries per second (QPS)  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    for name, url in endpoints:
        print(f"\n🔍 Testing: {name}")
        qps = await measure_throughput(url, duration_seconds=5, concurrent_requests=50)
        results[name] = qps
        await asyncio.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - All Endpoints")
    print(f"{'='*70}")
    print(f"{'Endpoint':<30} {'QPS':<15} {'Status':<20}")
    print(f"{'-'*70}")
    
    for name, qps in results.items():
        status = "✅ PASS" if qps >= 1000 else "⚠️  Below Target"
        print(f"{name:<30} {qps:<15.0f} {status:<20}")
    
    avg_qps = sum(results.values()) / len(results)
    print(f"{'-'*70}")
    print(f"{'Average':<30} {avg_qps:<15.0f}")
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
    
    # Run tests
    if len(sys.argv) > 1:
        # Test specific endpoint
        url = sys.argv[1]
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        asyncio.run(measure_throughput(url, duration_seconds=duration))
    else:
        # Test all endpoints
        asyncio.run(test_multiple_endpoints())
