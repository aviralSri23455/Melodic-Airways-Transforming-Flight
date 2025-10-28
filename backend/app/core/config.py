"""
Configuration settings for Aero Melody Backend
"""

import os
import secrets
from typing import Any, List, Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Project info
    PROJECT_NAME: str = "Aero Melody API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Flight route to music generation API using OpenFlights dataset"

    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your_secret_key_here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # CORS settings
    BACKEND_CORS_ORIGINS: Union[str, List[str]] = "http://localhost:8080,http://localhost:3000,http://localhost:5173,http://localhost:4173,http://localhost:8000,http://127.0.0.1:8080,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:4173,http://127.0.0.1:8000,https://yourdomain.com,https://www.yourdomain.com"

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            # Split comma-separated string into list
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return []

    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = "your_password_here"
    DB_NAME: str = "melody_aero"
    DATABASE_URL: str = "mysql://root:your_password_here@localhost:3306/melody_aero"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 3600

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def convert_to_async_url(cls, v: str) -> str:
        """Convert mysql:// to mysql+asyncmy:// for async support"""
        if v.startswith("mysql://"):
            return v.replace("mysql://", "mysql+asyncmy://", 1)
        return v

    # Redis Cloud settings
    REDIS_HOST: str = "your_redis_host"
    REDIS_PORT: int = 16441
    REDIS_USERNAME: str = "default"
    REDIS_PASSWORD: str = "your_redis_password_here"
    REDIS_URL: str = "redis://default:your_redis_password_here@your_redis_host:your_redis_port"
    REDIS_CACHE_TTL: int = 1800  # 30 minutes default cache TTL (optimized for 30MB plan)
    REDIS_SESSION_TTL: int = 7200  # 2 hours default session TTL (optimized for 30MB plan)
    REDIS_MAX_CONNECTIONS: int = 10  # Reduced connections for 30MB plan

    @field_validator("REDIS_URL", mode="after")
    @classmethod
    def construct_redis_url(cls, v: str, values) -> str:
        """Construct Redis URL from individual components if available"""
        # If we have Redis Cloud components configured, use them
        host = values.data.get("REDIS_HOST", "localhost")
        port = values.data.get("REDIS_PORT", 6379)
        username = values.data.get("REDIS_USERNAME", "default")
        password = values.data.get("REDIS_PASSWORD", "")

        if (host != "localhost" or port != 6379 or password):
            if password:  # Redis Cloud with authentication
                return f"redis://{username}:{password}@{host}:{port}"
            else:  # Local Redis without authentication
                return f"redis://{host}:{port}"
        else:
            return v  # Return original URL if no components configured

    # DuckDB Analytics settings
    DUCKDB_PATH: str = "./data/analytics.duckdb"
    DUCKDB_MEMORY_LIMIT: str = "2GB"
    DUCKDB_THREADS: int = 4

    # JWT settings
    JWT_SECRET_KEY: str = "your_jwt_secret_key_here"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "uploads"
    MIDI_OUTPUT_DIR: str = "midi_output"

    # External API settings
    OPENFLIGHTS_BASE_URL: str = "https://raw.githubusercontent.com/MariaDB/openflights/master/data"

    # Music generation settings
    DEFAULT_TEMPO: int = 120
    DEFAULT_SCALE: str = "major"
    DEFAULT_KEY: str = "C"
    MAX_POLYPHONY: int = 8

    # JSON Similarity Search settings (using free MariaDB JSON features)
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_SIMILARITY_RESULTS: int = 20
    EMBEDDING_DIMENSIONS: int = 128

    # Logging
    LOG_LEVEL: str = "WARNING"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # SSL/TLS settings
    SSL_CERT_FILE: Optional[str] = None
    SSL_KEY_FILE: Optional[str] = None

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8"
    )


# Create global settings instance
settings = Settings()

# Basic validation for critical configuration
def validate_critical_settings():
    """Validate that critical settings are properly configured"""
    errors = []

    # Check for placeholder values that indicate missing configuration
    placeholder_indicators = [
        "your_secret_key_here",
        "your_password_here",
        "your_redis_host",
        "your_jwt_secret_key_here",
        "your_app_secret_key_here"
    ]

    critical_settings = {
        "SECRET_KEY": settings.SECRET_KEY,
        "DB_PASSWORD": settings.DB_PASSWORD,
        "JWT_SECRET_KEY": settings.JWT_SECRET_KEY,
        "REDIS_PASSWORD": settings.REDIS_PASSWORD,
    }

    for setting_name, setting_value in critical_settings.items():
        if setting_value in placeholder_indicators:
            errors.append(f"{setting_name} is not configured (contains placeholder value)")

    if errors:
        print("⚠️  Configuration Warning:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease configure your environment variables in the .env file.")
        print("See the README.md for configuration instructions.\n")

    return len(errors) == 0

# Validate settings on import
validate_critical_settings()
