"""
Aero Melody Backend - Flight Route to Music Generation API
Comprehensive FastAPI backend for converting flight routes into musical compositions
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from sqlalchemy import text

from app.api.routes import router as api_router
from app.api.extended_routes import router as extended_router
from app.api.analytics_routes import router as analytics_router
from app.api.redis_routes import router as redis_router
from app.api.openflights_routes import router as openflights_router
from app.api.demo_routes import router as demo_router
from app.api.analytics_showcase_routes import router as analytics_showcase_router
from app.api.websocket_demo import router as websocket_demo_router
from app.api.wellness_routes import router as wellness_router
from app.api.premium_routes import router as premium_router
from app.api.education_routes import router as education_router
from app.api.vrar_routes import router as vrar_router
from app.api.vector_routes import router as vector_router
from app.api.metrics_routes import router as metrics_router
from app.core.config import settings
from app.db.database import engine, Base, get_db, async_session
from app.middleware.throughput_monitor import ThroughputMiddleware, get_throughput_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events"""
    logger.info("Starting Aero Melody Backend...")

    # Try to create database tables (only if database is configured)
    try:
        # Check if we have proper database configuration
        if "your_password_here" not in settings.DATABASE_URL and "your_secret_key_here" not in settings.SECRET_KEY:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ Database tables created successfully")
        else:
            logger.warning("⚠️  Database not configured - skipping table creation")
            logger.warning("Please configure your DATABASE_URL in the .env file")
    except Exception as e:
        logger.warning(f"⚠️  Database connection failed: {e}")
        logger.warning("Application will continue without database functionality")
        logger.warning("Please check your database configuration in the .env file")

    yield
    logger.info("Shutting down Aero Melody Backend...")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add throughput monitoring middleware
app.add_middleware(ThroughputMiddleware, monitor=get_throughput_monitor())

# Include API routes
try:
    app.include_router(api_router, prefix=settings.API_V1_STR, tags=["Core"])
    logger.info("✅ api_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include api_router: {e}")

try:
    app.include_router(extended_router, prefix=settings.API_V1_STR, tags=["Extended"])
    logger.info("✅ extended_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include extended_router: {e}")

try:
    app.include_router(analytics_router, prefix=settings.API_V1_STR, tags=["Analytics"])
    logger.info("✅ analytics_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include analytics_router: {e}")

try:
    app.include_router(redis_router, prefix=f"{settings.API_V1_STR}/redis", tags=["Redis"])
    logger.info("✅ redis_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include redis_router: {e}")
    # Print more details about the error
    import traceback
    logger.error(f"Redis router error details: {traceback.format_exc()}")
    # Continue without Redis router to prevent app startup failure
    logger.warning("Continuing without Redis router")

try:
    app.include_router(openflights_router, prefix=settings.API_V1_STR, tags=["OpenFlights"])
    logger.info("✅ openflights_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include openflights_router: {e}")

try:
    app.include_router(demo_router, prefix=f"{settings.API_V1_STR}/demo", tags=["Demo"])
    logger.info("✅ demo_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include demo_router: {e}")

try:
    app.include_router(analytics_showcase_router, prefix=f"{settings.API_V1_STR}/analytics-showcase", tags=["Analytics Showcase"])
    logger.info("✅ analytics_showcase_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include analytics_showcase_router: {e}")

try:
    app.include_router(websocket_demo_router, prefix=f"{settings.API_V1_STR}/demo", tags=["WebSocket Demo"])
    logger.info("✅ websocket_demo_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include websocket_demo_router: {e}")

try:
    app.include_router(wellness_router, prefix=f"{settings.API_V1_STR}/wellness", tags=["Wellness"])
    logger.info("✅ wellness_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include wellness_router: {e}")

try:
    app.include_router(premium_router, prefix=f"{settings.API_V1_STR}/premium", tags=["Premium"])
    logger.info("✅ premium_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include premium_router: {e}")

try:
    app.include_router(education_router, prefix=f"{settings.API_V1_STR}/education", tags=["Education"])
    logger.info("✅ education_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include education_router: {e}")

try:
    app.include_router(vrar_router, prefix=f"{settings.API_V1_STR}/vr-ar", tags=["VR/AR"])
    logger.info("✅ vrar_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include vrar_router: {e}")

try:
    app.include_router(vector_router, prefix=settings.API_V1_STR, tags=["Vector Embeddings"])
    logger.info("✅ vector_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include vector_router: {e}")

try:
    app.include_router(metrics_router, prefix=settings.API_V1_STR, tags=["Performance Metrics"])
    logger.info("✅ metrics_router included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include metrics_router: {e}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# OAuth callback placeholder endpoint
@app.get("/callback")
async def oauth_callback(code: str = None, error: str = None):
    """OAuth callback endpoint - Placeholder for future OAuth implementation"""
    if error:
        return {
            "error": error,
            "message": "OAuth authentication failed"
        }
    
    return {
        "message": "OAuth callback endpoint is not implemented yet",
        "note": "Please use /api/v1/auth/register and /api/v1/auth/login for authentication",
        "code_received": code is not None
    }
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Aero Melody API",
        "docs": "/docs",
        "version": settings.VERSION
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "service": "aero-melody-backend",
        "version": settings.VERSION,
        "database": "unknown",
        "redis": "unknown",
        "configuration": "partial"
    }

    # Check database connectivity
    try:
        if "your_password_here" not in settings.DATABASE_URL:
            async with async_session() as session:
                await session.execute(text("SELECT 1"))
            health_status["database"] = "connected"
        else:
            health_status["database"] = "not_configured"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Check Redis connectivity (if configured)
    try:
        if "your_redis_password_here" not in settings.REDIS_PASSWORD and settings.REDIS_HOST != "your_redis_host":
            # Redis check would go here if we had redis client
            health_status["redis"] = "configured"
        else:
            health_status["redis"] = "not_configured"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"

    # Check configuration completeness
    critical_placeholders = [
        "your_secret_key_here",
        "your_password_here",
        "your_jwt_secret_key_here",
        "your_redis_host"
    ]

    config_issues = []
    if settings.SECRET_KEY in critical_placeholders:
        config_issues.append("SECRET_KEY")
    if settings.DB_PASSWORD in critical_placeholders:
        config_issues.append("DB_PASSWORD")
    if settings.JWT_SECRET_KEY in critical_placeholders:
        config_issues.append("JWT_SECRET_KEY")
    if settings.REDIS_HOST in critical_placeholders:
        config_issues.append("REDIS_HOST")

    if not config_issues:
        health_status["configuration"] = "complete"
    else:
        health_status["configuration"] = f"incomplete: {', '.join(config_issues)}"
        if health_status["status"] == "healthy":
            health_status["status"] = "configuration_needed"

    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
