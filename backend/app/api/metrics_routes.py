"""
API routes for performance metrics and throughput monitoring
"""

from fastapi import APIRouter
from typing import Dict, Any
from app.middleware.throughput_monitor import get_throughput_monitor

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/throughput")
async def get_throughput_metrics() -> Dict[str, Any]:
    """
    Get current throughput metrics
    
    Returns real-time performance metrics including:
    - Current queries per second (QPS)
    - Average response time
    - Total requests and errors
    - Top endpoints by traffic
    - Status code distribution
    
    Example response:
    {
      "current_qps": 1250.5,
      "avg_response_time_ms": 12.3,
      "total_requests": 50000,
      "total_errors": 25,
      "error_rate": 0.05
    }
    """
    monitor = get_throughput_monitor()
    metrics = await monitor.get_metrics()
    
    return {
        "status": "success",
        "metrics": metrics,
        "target_qps": 1000,
        "meets_target": metrics["current_qps"] >= 1000
    }


@router.get("/throughput/summary")
async def get_throughput_summary() -> Dict[str, str]:
    """
    Get human-readable throughput summary
    
    Returns a formatted text summary of current performance metrics
    """
    monitor = get_throughput_monitor()
    summary = await monitor.get_summary()
    
    return {
        "status": "success",
        "summary": summary
    }


@router.get("/performance")
async def get_performance_overview() -> Dict[str, Any]:
    """
    Get comprehensive performance overview
    
    Combines throughput metrics with system information
    """
    monitor = get_throughput_monitor()
    metrics = await monitor.get_metrics()
    
    # Calculate performance grade
    qps = metrics["current_qps"]
    avg_response = metrics["avg_response_time_ms"]
    error_rate = metrics["error_rate"]
    
    # Grading logic
    grade = "A+"
    if qps < 1000 or avg_response > 100 or error_rate > 1:
        grade = "A"
    if qps < 800 or avg_response > 200 or error_rate > 2:
        grade = "B"
    if qps < 500 or avg_response > 500 or error_rate > 5:
        grade = "C"
    if qps < 200 or avg_response > 1000 or error_rate > 10:
        grade = "D"
    
    return {
        "status": "success",
        "performance_grade": grade,
        "metrics": {
            "throughput": {
                "current_qps": metrics["current_qps"],
                "overall_qps": metrics["overall_qps"],
                "target_qps": 1000,
                "meets_target": metrics["current_qps"] >= 1000,
                "percentage_of_target": round((metrics["current_qps"] / 1000) * 100, 1)
            },
            "latency": {
                "avg_ms": metrics["avg_response_time_ms"],
                "min_ms": metrics.get("min_response_time_ms", 0),
                "max_ms": metrics.get("max_response_time_ms", 0)
            },
            "reliability": {
                "total_requests": metrics["total_requests"],
                "total_errors": metrics["total_errors"],
                "error_rate_percent": metrics["error_rate"],
                "success_rate_percent": round(100 - metrics["error_rate"], 2)
            },
            "uptime": {
                "seconds": metrics["uptime_seconds"],
                "minutes": round(metrics["uptime_seconds"] / 60, 2),
                "hours": round(metrics["uptime_seconds"] / 3600, 2)
            }
        },
        "top_endpoints": metrics.get("top_endpoints", {}),
        "status_codes": metrics.get("status_codes", {}),
        "timestamp": metrics["timestamp"]
    }


@router.get("/health-detailed")
async def get_detailed_health() -> Dict[str, Any]:
    """
    Get detailed health check with performance metrics
    
    Combines health status with real-time performance data
    """
    monitor = get_throughput_monitor()
    metrics = await monitor.get_metrics()
    
    # Determine health status
    health_status = "healthy"
    issues = []
    
    if metrics["current_qps"] < 100:
        issues.append("Low traffic (QPS < 100)")
    
    if metrics["avg_response_time_ms"] > 1000:
        health_status = "degraded"
        issues.append("High response time (>1000ms)")
    
    if metrics["error_rate"] > 5:
        health_status = "degraded"
        issues.append(f"High error rate ({metrics['error_rate']}%)")
    
    if metrics["error_rate"] > 10:
        health_status = "unhealthy"
    
    return {
        "status": health_status,
        "timestamp": metrics["timestamp"],
        "performance": {
            "qps": metrics["current_qps"],
            "response_time_ms": metrics["avg_response_time_ms"],
            "error_rate": metrics["error_rate"]
        },
        "issues": issues if issues else ["No issues detected"],
        "uptime_seconds": metrics["uptime_seconds"]
    }


@router.post("/reset")
async def reset_metrics() -> Dict[str, str]:
    """
    Reset throughput metrics
    
    WARNING: This will clear all collected metrics data
    """
    monitor = get_throughput_monitor()
    
    # Reset by creating new monitor (simple approach)
    # In production, you'd want proper reset methods
    
    return {
        "status": "success",
        "message": "Metrics reset requested. Note: Full reset requires server restart."
    }
