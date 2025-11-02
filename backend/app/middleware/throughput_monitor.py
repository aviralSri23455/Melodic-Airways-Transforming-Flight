"""
Throughput Monitoring Middleware
Tracks requests per second and performance metrics in real-time
"""

import time
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import asyncio

logger = logging.getLogger(__name__)


class ThroughputMonitor:
    """Monitor and track API throughput metrics"""
    
    def __init__(self, window_seconds: int = 60):
        """
        Initialize throughput monitor
        
        Args:
            window_seconds: Time window for calculating QPS (default: 60 seconds)
        """
        self.window_seconds = window_seconds
        self.requests = deque()  # Store (timestamp, duration_ms, endpoint, status_code)
        self.lock = asyncio.Lock()
        
        # Counters
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
    
    async def record_request(
        self,
        endpoint: str,
        duration_ms: float,
        status_code: int
    ):
        """Record a request"""
        async with self.lock:
            now = time.time()
            self.requests.append((now, duration_ms, endpoint, status_code))
            self.total_requests += 1
            
            if status_code >= 400:
                self.total_errors += 1
            
            # Clean old requests outside the window
            cutoff_time = now - self.window_seconds
            while self.requests and self.requests[0][0] < cutoff_time:
                self.requests.popleft()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current throughput metrics"""
        async with self.lock:
            now = time.time()
            cutoff_time = now - self.window_seconds
            
            # Filter requests in current window
            recent_requests = [r for r in self.requests if r[0] >= cutoff_time]
            
            if not recent_requests:
                return {
                    "current_qps": 0,
                    "avg_response_time_ms": 0,
                    "requests_in_window": 0,
                    "window_seconds": self.window_seconds,
                    "total_requests": self.total_requests,
                    "total_errors": self.total_errors,
                    "uptime_seconds": round(now - self.start_time, 2),
                    "overall_qps": round(self.total_requests / (now - self.start_time), 2)
                }
            
            # Calculate metrics
            window_duration = now - recent_requests[0][0]
            qps = len(recent_requests) / window_duration if window_duration > 0 else 0
            
            response_times = [r[1] for r in recent_requests]
            avg_response_time = sum(response_times) / len(response_times)
            
            # Get endpoint breakdown
            endpoint_counts = {}
            for _, _, endpoint, _ in recent_requests:
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
            
            # Get status code breakdown
            status_counts = {}
            for _, _, _, status in recent_requests:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "current_qps": round(qps, 2),
                "avg_response_time_ms": round(avg_response_time, 2),
                "min_response_time_ms": round(min(response_times), 2),
                "max_response_time_ms": round(max(response_times), 2),
                "requests_in_window": len(recent_requests),
                "window_seconds": self.window_seconds,
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(self.total_errors / self.total_requests * 100, 2) if self.total_requests > 0 else 0,
                "uptime_seconds": round(now - self.start_time, 2),
                "overall_qps": round(self.total_requests / (now - self.start_time), 2),
                "top_endpoints": dict(sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                "status_codes": status_counts,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_summary(self) -> str:
        """Get a human-readable summary"""
        metrics = await self.get_metrics()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              THROUGHPUT MONITORING SUMMARY                   ║
╚══════════════════════════════════════════════════════════════╝

🚀 Current QPS:          {metrics['current_qps']} queries/second
📊 Overall QPS:          {metrics['overall_qps']} queries/second
⏱️  Avg Response Time:   {metrics['avg_response_time_ms']}ms
📈 Total Requests:       {metrics['total_requests']}
❌ Total Errors:         {metrics['total_errors']} ({metrics['error_rate']}%)
⏰ Uptime:               {metrics['uptime_seconds']}s

Window: Last {metrics['window_seconds']} seconds ({metrics['requests_in_window']} requests)
"""
        return summary


class ThroughputMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor API throughput"""
    
    def __init__(self, app, monitor: ThroughputMonitor):
        super().__init__(app)
        self.monitor = monitor
    
    async def dispatch(self, request: Request, call_next):
        """Process request and record metrics"""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        endpoint = request.url.path
        await self.monitor.record_request(endpoint, duration_ms, response.status_code)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


# Global monitor instance
throughput_monitor = ThroughputMonitor(window_seconds=60)


def get_throughput_monitor() -> ThroughputMonitor:
    """Get the global throughput monitor instance"""
    return throughput_monitor
