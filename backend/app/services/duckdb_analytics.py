"""
DuckDB Analytics Service for Aero Melody Backend
Provides real-time analytics for route complexity, similarity analysis, and music generation metrics
"""

import duckdb
import logging
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class DuckDBAnalytics:
    """DuckDB analytics service for route and music analysis"""

    def __init__(self):
        """Initialize DuckDB connection and create analytics tables"""
        self.db_path = getattr(settings, 'DUCKDB_PATH', './data/analytics.duckdb')
        self.conn = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize DuckDB connection and create tables"""
        try:
            # Create data directory if it doesn't exist
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            # Connect to DuckDB
            self.conn = duckdb.connect(
                database=self.db_path,
                read_only=False
            )

            # Configure DuckDB settings with safe defaults
            try:
                memory_limit = getattr(settings, 'DUCKDB_MEMORY_LIMIT', '2GB')
                threads = getattr(settings, 'DUCKDB_THREADS', 4)
                self.conn.execute(f"SET memory_limit='{memory_limit}'")
                self.conn.execute(f"SET threads={threads}")
            except Exception as e:
                logger.warning(f"Could not set DuckDB configuration: {e}")

            # Create analytics tables
            self._create_tables()

            logger.info(f"DuckDB analytics initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")
            # Don't raise - allow fallback to work
            self.conn = None

    def _create_tables(self):
        """Create analytics tables for route and music data"""
        
        try:
            # Route analytics table - Using auto-increment without sequence
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS route_analytics (
                    id INTEGER PRIMARY KEY,
                    origin VARCHAR,
                    destination VARCHAR,
                    distance_km DOUBLE,
                    complexity_score DOUBLE,
                    path_length INTEGER,
                    intermediate_stops INTEGER,
                    route_embedding VARCHAR,
                    created_at TIMESTAMP
                )
            """)

            # Music generation analytics table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS music_analytics (
                    id INTEGER PRIMARY KEY,
                    route_id INTEGER,
                    tempo INTEGER,
                    key VARCHAR,
                    scale VARCHAR,
                    duration_seconds DOUBLE,
                    note_count INTEGER,
                    harmony_complexity DOUBLE,
                    genre VARCHAR,
                    embedding_vector VARCHAR,
                    music_vector VARCHAR,
                    created_at TIMESTAMP
                )
            """)

            # Route similarity cache table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS route_similarity (
                    id INTEGER PRIMARY KEY,
                    route1_origin VARCHAR,
                    route1_destination VARCHAR,
                    route2_origin VARCHAR,
                    route2_destination VARCHAR,
                    similarity_score DOUBLE,
                    distance_difference DOUBLE,
                    complexity_difference DOUBLE,
                    computed_at TIMESTAMP
                )
            """)

            # Performance metrics table
            self.conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS performance_metrics_seq START 1
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY DEFAULT nextval('performance_metrics_seq'),
                    operation_type VARCHAR,
                    execution_time_ms DOUBLE,
                    success BOOLEAN,
                    error_message VARCHAR,
                    metadata VARCHAR,
                    created_at TIMESTAMP
                )
            """)
            
            logger.info("DuckDB analytics tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating DuckDB tables: {e}")
            # Don't raise - allow fallback to work
            pass

        logger.info("DuckDB analytics tables created successfully")

    def log_route_analytics(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        complexity_score: float,
        path_length: int,
        intermediate_stops: int
    ) -> bool:
        """
        Log route analytics data
        
        Args:
            origin: Origin airport code
            destination: Destination airport code
            distance_km: Route distance in kilometers
            complexity_score: Computed complexity score
            path_length: Number of segments in path
            intermediate_stops: Number of intermediate stops
            
        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            logger.warning("DuckDB connection not available, skipping route analytics")
            return False
            
        try:
            # Generate ID manually - keep it within INT32 range
            import random
            record_id = int(time.time() % 1000000) + random.randint(1, 999)  # Keep under 2 billion
            
            self.conn.execute("""
                INSERT INTO route_analytics (
                    id, origin, destination, distance_km, complexity_score,
                    path_length, intermediate_stops, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record_id, origin, destination, distance_km, complexity_score,
                path_length, intermediate_stops, datetime.utcnow()
            ])
            
            logger.info(f"Logged route analytics for {origin} -> {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging route analytics: {e}")
            return False

    def log_music_analytics(
        self,
        route_id: int,
        tempo: int,
        key: str,
        scale: str,
        duration_seconds: float,
        note_count: int,
        harmony_complexity: float,
        genre: str,
        embedding_vector: List[float]
    ) -> bool:
        """
        Log music generation analytics
        
        Args:
            route_id: Associated route ID
            tempo: Music tempo (BPM)
            key: Musical key
            scale: Musical scale
            duration_seconds: Duration in seconds
            note_count: Total number of notes
            harmony_complexity: Harmony complexity score
            genre: Music genre
            embedding_vector: Vector embedding of the music
            
        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            logger.warning("DuckDB connection not available, skipping music analytics")
            return False
            
        try:
            # Generate ID manually - keep it within INT32 range
            import random
            record_id = int(time.time() % 1000000) + random.randint(1, 999)  # Keep under 2 billion
            
            # Validate and sanitize inputs to prevent NULL constraint violations
            route_id = route_id if route_id is not None else 0
            tempo = tempo if tempo is not None else 120
            key = key if key is not None else "C"
            scale = scale if scale is not None else "major"
            duration_seconds = float(duration_seconds) if duration_seconds is not None else 0.0
            note_count = note_count if note_count is not None else 0
            harmony_complexity = float(harmony_complexity) if harmony_complexity is not None else 0.0
            genre = genre if genre is not None else "unknown"
            embedding_vector = embedding_vector if embedding_vector is not None else []
            
            # Serialize embedding vector as JSON
            embedding_json = json.dumps(embedding_vector)
            
            self.conn.execute("""
                INSERT INTO music_analytics (
                    id, route_id, tempo, key, scale, duration_seconds,
                    note_count, harmony_complexity, genre, embedding_vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record_id, route_id, tempo, key, scale, duration_seconds,
                note_count, harmony_complexity, genre, embedding_json, datetime.utcnow()
            ])
            
            logger.info(f"Logged music analytics for route_id {route_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging music analytics: {e}")
            return False

    def compute_route_similarity(
        self,
        route1_origin: str,
        route1_dest: str,
        route2_origin: str,
        route2_dest: str
    ) -> Optional[float]:
        """
        Compute similarity between two routes based on historical data
        
        Args:
            route1_origin: First route origin
            route1_dest: First route destination
            route2_origin: Second route origin
            route2_dest: Second route destination
            
        Returns:
            Similarity score (0-1) or None if not enough data
        """
        try:
            # Check if similarity already computed
            result = self.conn.execute("""
                SELECT similarity_score FROM route_similarity
                WHERE (route1_origin = ? AND route1_destination = ?
                       AND route2_origin = ? AND route2_destination = ?)
                   OR (route1_origin = ? AND route1_destination = ?
                       AND route2_origin = ? AND route2_destination = ?)
                ORDER BY computed_at DESC
                LIMIT 1
            """, [
                route1_origin, route1_dest, route2_origin, route2_dest,
                route2_origin, route2_dest, route1_origin, route1_dest
            ]).fetchone()
            
            if result:
                return result[0]
            
            # Compute new similarity based on route characteristics
            route1_data = self.conn.execute("""
                SELECT distance_km, complexity_score
                FROM route_analytics
                WHERE origin = ? AND destination = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, [route1_origin, route1_dest]).fetchone()
            
            route2_data = self.conn.execute("""
                SELECT distance_km, complexity_score
                FROM route_analytics
                WHERE origin = ? AND destination = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, [route2_origin, route2_dest]).fetchone()
            
            if not route1_data or not route2_data:
                return None
            
            # Calculate similarity (inverse of normalized differences)
            dist_diff = abs(route1_data[0] - route2_data[0])
            complexity_diff = abs(route1_data[1] - route2_data[1])
            
            # Normalize and compute similarity (0-1 scale)
            max_dist = max(route1_data[0], route2_data[0], 1)
            max_complexity = max(route1_data[1], route2_data[1], 1)
            
            similarity = 1.0 - (
                (dist_diff / max_dist * 0.5) +
                (complexity_diff / max_complexity * 0.5)
            )
            
            # Cache the result
            self.conn.execute("""
                INSERT INTO route_similarity (
                    route1_origin, route1_destination,
                    route2_origin, route2_destination,
                    similarity_score, distance_difference,
                    complexity_difference, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                route1_origin, route1_dest, route2_origin, route2_dest,
                similarity, dist_diff, complexity_diff, datetime.utcnow()
            ])
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing route similarity: {e}")
            return None

    def get_route_complexity_stats(self) -> Dict[str, Any]:
        """
        Get statistical summary of route complexity
        
        Returns:
            Dictionary with complexity statistics
        """
        if not self.conn:
            return {"total_routes": 0}
            
        try:
            result = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_routes,
                    AVG(complexity_score) as avg_complexity,
                    MIN(complexity_score) as min_complexity,
                    MAX(complexity_score) as max_complexity,
                    STDDEV(complexity_score) as std_complexity,
                    AVG(distance_km) as avg_distance,
                    AVG(intermediate_stops) as avg_stops
                FROM route_analytics
            """).fetchone()
            
            if result:
                return {
                    "total_routes": result[0],
                    "avg_complexity": round(result[1], 2) if result[1] else 0,
                    "min_complexity": round(result[2], 2) if result[2] else 0,
                    "max_complexity": round(result[3], 2) if result[3] else 0,
                    "std_complexity": round(result[4], 2) if result[4] else 0,
                    "avg_distance": round(result[5], 2) if result[5] else 0,
                    "avg_stops": round(result[6], 2) if result[6] else 0
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting complexity stats: {e}")
            return {}

    def get_genre_distribution(self) -> Dict[str, int]:
        """
        Get distribution of music genres generated
        
        Returns:
            Dictionary mapping genre to count
        """
        if not self.conn:
            return {}
            
        try:
            results = self.conn.execute("""
                SELECT genre, COUNT(*) as count
                FROM music_analytics
                GROUP BY genre
                ORDER BY count DESC
            """).fetchall()
            
            return {row[0]: row[1] for row in results}
            
        except Exception as e:
            logger.error(f"Error getting genre distribution: {e}")
            return {}

    def get_performance_metrics(
        self,
        operation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get performance metrics for operations
        
        Args:
            operation_type: Filter by operation type (optional)
            limit: Maximum number of results
            
        Returns:
            List of performance metric records
        """
        try:
            query = """
                SELECT 
                    operation_type,
                    AVG(execution_time_ms) as avg_time,
                    MIN(execution_time_ms) as min_time,
                    MAX(execution_time_ms) as max_time,
                    COUNT(*) as total_operations,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed
                FROM performance_metrics
            """
            
            if operation_type:
                query += " WHERE operation_type = ?"
                results = self.conn.execute(query + " GROUP BY operation_type", [operation_type]).fetchall()
            else:
                results = self.conn.execute(query + " GROUP BY operation_type LIMIT ?", [limit]).fetchall()
            
            return [
                {
                    "operation_type": row[0],
                    "avg_time_ms": round(row[1], 2) if row[1] else 0,
                    "min_time_ms": round(row[2], 2) if row[2] else 0,
                    "max_time_ms": round(row[3], 2) if row[3] else 0,
                    "total_operations": row[4],
                    "successful": row[5],
                    "failed": row[6],
                    "success_rate": round((row[5] / row[4] * 100), 2) if row[4] > 0 else 0
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return []

    def log_performance_metric(
        self,
        operation_type: str,
        execution_time_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a performance metric
        
        Args:
            operation_type: Type of operation
            execution_time_ms: Execution time in milliseconds
            success: Whether operation succeeded
            error_message: Error message if failed
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert Decimal and other non-serializable types to float/str
            if metadata:
                from decimal import Decimal
                def convert_decimals(obj):
                    if isinstance(obj, Decimal):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_decimals(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_decimals(item) for item in obj]
                    return obj
                
                metadata = convert_decimals(metadata)
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            self.conn.execute("""
                INSERT INTO performance_metrics (
                    operation_type, execution_time_ms, success,
                    error_message, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, [
                operation_type, execution_time_ms, success,
                error_message, metadata_json, datetime.utcnow()
            ])
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging performance metric: {e}")
            return False

    def get_similar_routes(
        self,
        origin: str,
        destination: str,
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar routes based on analytics data
        
        Args:
            origin: Origin airport code
            destination: Destination airport code
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar routes with similarity scores
        """
        try:
            results = self.conn.execute("""
                SELECT 
                    rs.route2_origin,
                    rs.route2_destination,
                    rs.similarity_score,
                    ra.distance_km,
                    ra.complexity_score
                FROM route_similarity rs
                JOIN route_analytics ra 
                    ON rs.route2_origin = ra.origin 
                    AND rs.route2_destination = ra.destination
                WHERE rs.route1_origin = ? 
                    AND rs.route1_destination = ?
                    AND rs.similarity_score >= ?
                ORDER BY rs.similarity_score DESC
                LIMIT ?
            """, [origin, destination, min_similarity, limit]).fetchall()
            
            return [
                {
                    "origin": row[0],
                    "destination": row[1],
                    "similarity_score": round(row[2], 3),
                    "distance_km": round(row[3], 2),
                    "complexity_score": round(row[4], 2)
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error finding similar routes: {e}")
            return []

    def close(self):
        """Close DuckDB connection"""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")


# Global analytics instance
analytics = None


class FallbackDuckDBAnalytics:
    """Fallback service when DuckDB initialization fails"""

    def __init__(self):
        logger.warning("Using fallback DuckDB analytics service")

    def get_route_complexity_stats(self):
        """Fallback - return empty statistics"""
        return {"total_routes": 0}

    def log_route_analytics(self, *args, **kwargs):
        """Fallback - return False"""
        return False

    def log_music_analytics(self, *args, **kwargs):
        """Fallback - return False"""
        return False

    def compute_route_similarity(self, *args, **kwargs):
        """Fallback - return None"""
        return None

    def get_genre_distribution(self):
        """Fallback - return empty dict"""
        return {}

    def get_performance_metrics(self, *args, **kwargs):
        """Fallback - return empty list"""
        return []

    def log_performance_metric(self, *args, **kwargs):
        """Fallback - return False"""
        return False

    def get_similar_routes(self, *args, **kwargs):
        """Fallback - return empty list"""
        return []

    def close(self):
        """Fallback - do nothing"""
        pass


def get_analytics():
    """Get the global analytics instance"""
    global analytics
    if analytics is None:
        try:
            analytics = DuckDBAnalytics()
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB analytics: {e}")
            # Use fallback service instead of breaking the router
            analytics = FallbackDuckDBAnalytics()
    return analytics
