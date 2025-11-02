"""
Route Embedding Service for DuckDB Analytics
Generates vector embeddings for flight routes using DuckDB
"""

import duckdb
import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from app.core.config import settings
from app.services.duckdb_analytics import get_analytics

logger = logging.getLogger(__name__)


class RouteEncoder(nn.Module):
    """PyTorch neural network for encoding routes into vector embeddings"""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 128):
        super(RouteEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
    
    def forward(self, x):
        return self.encoder(x)


class RouteEmbeddingServiceDuckDB:
    """Service for generating and managing route vector embeddings with DuckDB"""
    
    def __init__(self):
        self.embedding_dim = 128
        self.encoder = RouteEncoder(input_dim=16, hidden_dim=64, output_dim=self.embedding_dim)
        self.encoder.eval()  # Set to evaluation mode
        self.faiss_service = None  # Not using FAISS with DuckDB
        
        # Reuse the analytics service connection to avoid conflicts
        self.analytics = get_analytics()
        
        # Genre characteristics for musical mapping
        self.genre_profiles = {
            "classical": {"complexity": 0.8, "min_distance": 5000, "max_stops": 3},
            "jazz": {"complexity": 0.9, "min_distance": 2000, "max_stops": 5},
            "electronic": {"complexity": 0.6, "min_distance": 1000, "max_stops": 2},
            "ambient": {"complexity": 0.3, "min_distance": 8000, "max_stops": 1},
            "pop": {"complexity": 0.5, "min_distance": 3000, "max_stops": 2}
        }
    
    def _get_connection(self):
        """Get DuckDB connection - reuse analytics connection"""
        if self.analytics and self.analytics.conn:
            return self.analytics.conn
        # Fallback: create new connection if analytics not available
        db_path = getattr(settings, 'DUCKDB_PATH', './data/analytics.duckdb')
        return duckdb.connect(database=db_path, read_only=False)
    
    def _extract_route_features(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        complexity_score: float = 0.5,
        stops: int = 0
    ) -> np.ndarray:
        """Extract features from route data"""
        # Create feature vector
        origin_hash = sum(ord(c) for c in origin) / 1000.0
        dest_hash = sum(ord(c) for c in destination) / 1000.0
        
        features = [
            origin_hash,
            dest_hash,
            distance_km / 10000.0,  # Normalize distance
            complexity_score,
            stops / 5.0,  # Normalize stops
            abs(origin_hash - dest_hash),  # Route variation
            (origin_hash + dest_hash) / 2,  # Average
            distance_km * complexity_score / 10000.0,  # Combined metric
            # Add more features to reach 16 dimensions
            np.sin(origin_hash * np.pi),
            np.cos(dest_hash * np.pi),
            np.sqrt(distance_km) / 100.0,
            np.log1p(distance_km) / 10.0,
            complexity_score ** 2,
            stops ** 0.5 if stops > 0 else 0,
            (distance_km / (stops + 1)) / 1000.0,  # Distance per segment
            1.0 if distance_km > 5000 else 0.5  # Long/short route indicator
        ]
        
        return np.array(features, dtype=np.float32)
    
    def generate_embedding(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        complexity_score: float = 0.5,
        stops: int = 0
    ) -> np.ndarray:
        """Generate embedding vector for a route"""
        features = self._extract_route_features(origin, destination, distance_km, complexity_score, stops)
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            embedding = self.encoder(features_tensor)
            return embedding.squeeze(0).numpy()
    
    async def find_similar_routes(
        self,
        db,  # AsyncSession (not used, we use DuckDB directly)
        origin: str,
        destination: str,
        limit: int = 10
    ) -> List[Dict]:
        """Find similar routes using DuckDB"""
        try:
            conn = self._get_connection()
            
            # Check if route_analytics table exists and has data
            tables = conn.execute("SHOW TABLES").fetchall()
            if not any('route_analytics' in str(table) for table in tables):
                logger.warning("route_analytics table not found")
                return []
            
            # Get all routes with embeddings
            routes = conn.execute("""
                SELECT id, origin, destination, distance_km, complexity_score, 
                       intermediate_stops, route_embedding
                FROM route_analytics
                WHERE route_embedding IS NOT NULL AND route_embedding != ''
                LIMIT 1000
            """).fetchall()
            
            # Don't close shared connection
            
            if not routes:
                logger.warning("No routes with embeddings found")
                return []
            
            # Generate embedding for query route
            query_embedding = self.generate_embedding(origin, destination, 0, 0.5, 0)
            
            # Calculate similarities
            similarities = []
            for route in routes:
                try:
                    route_id, r_origin, r_dest, r_distance, r_complexity, r_stops, r_embedding_str = route
                    
                    # Skip if same route
                    if r_origin == origin and r_dest == destination:
                        continue
                    
                    # Parse embedding
                    route_embedding = np.array(json.loads(r_embedding_str))
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, route_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(route_embedding) + 1e-8
                    )
                    
                    similarities.append({
                        "route_id": route_id,
                        "origin_airport_id": 0,  # Not available in DuckDB
                        "destination_airport_id": 0,
                        "distance_km": float(r_distance or 0),
                        "similarity_score": float(similarity),
                        "origin_code": r_origin,
                        "dest_code": r_dest
                    })
                except Exception as e:
                    logger.error(f"Error processing route {route[0]}: {e}")
                    continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:limit]
        
        except Exception as e:
            logger.error(f"Error finding similar routes: {e}")
            return []
    
    async def find_routes_by_genre(
        self,
        db,  # AsyncSession (not used)
        genre: str,
        limit: int = 20
    ) -> List[Dict]:
        """Find routes matching a musical genre profile"""
        try:
            if genre.lower() not in self.genre_profiles:
                return []
            
            profile = self.genre_profiles[genre.lower()]
            
            conn = self._get_connection()
            
            # Query routes matching genre characteristics
            query = f"""
                SELECT id, origin, destination, distance_km, complexity_score, 
                       intermediate_stops
                FROM route_analytics
                WHERE distance_km >= {profile['min_distance']}
                  AND intermediate_stops <= {profile['max_stops']}
                  AND complexity_score >= {profile['complexity'] - 0.2}
                  AND complexity_score <= {profile['complexity'] + 0.2}
                LIMIT {limit}
            """
            
            routes = conn.execute(query).fetchall()
            # Don't close shared connection
            
            result = []
            for route in routes:
                route_id, origin, dest, distance, complexity, stops = route
                result.append({
                    "route_id": route_id,
                    "origin_airport_id": 0,
                    "destination_airport_id": 0,
                    "distance_km": float(distance or 0),
                    "stops": int(stops or 0),
                    "origin_code": origin,
                    "dest_code": dest,
                    "genre": genre,
                    "genre_match_score": float(complexity or 0.5)
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error finding routes by genre: {e}")
            return []
    
    async def calculate_melodic_complexity(
        self,
        db,  # AsyncSession (not used)
        route_id: int
    ) -> Optional[Dict]:
        """Calculate melodic complexity for a route"""
        try:
            conn = self._get_connection()
            
            route = conn.execute("""
                SELECT distance_km, complexity_score, intermediate_stops
                FROM route_analytics
                WHERE id = ?
            """, [route_id]).fetchone()
            
            # Don't close shared connection
            
            if not route:
                return None
            
            distance, complexity, stops = route
            
            return {
                "harmonic_complexity": float(complexity or 0.5),
                "rhythmic_complexity": float(min(stops / 5.0, 1.0) if stops else 0.3),
                "melodic_complexity": float(min(distance / 10000.0, 1.0) if distance else 0.5),
                "overall_complexity": float((complexity or 0.5) * 0.5 + 
                                          min(stops / 5.0, 1.0) * 0.3 + 
                                          min(distance / 10000.0, 1.0) * 0.2)
            }
        
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            return None
    
    async def get_embedding_statistics(
        self,
        db  # AsyncSession (not used)
    ) -> Dict:
        """Get statistics about embeddings"""
        try:
            conn = self._get_connection()
            
            # Get total routes
            total = conn.execute("SELECT COUNT(*) FROM route_analytics").fetchone()[0]
            
            # Get routes with embeddings
            with_embeddings = conn.execute("""
                SELECT COUNT(*) FROM route_analytics 
                WHERE route_embedding IS NOT NULL AND route_embedding != ''
            """).fetchone()[0]
            
            # Get averages
            stats = conn.execute("""
                SELECT 
                    AVG(distance_km) as avg_distance,
                    AVG(intermediate_stops) as avg_stops,
                    AVG(complexity_score) as avg_complexity
                FROM route_analytics
                WHERE route_embedding IS NOT NULL
            """).fetchone()
            
            # Don't close shared connection
            
            return {
                "total_routes": total,
                "routes_with_embeddings": with_embeddings,
                "embedding_coverage": float(with_embeddings / total) if total > 0 else 0.0,
                "avg_distance_km": float(stats[0] or 0),
                "avg_stops": float(stats[1] or 0),
                "avg_melodic_complexity": float(stats[2] or 0.5),
                "avg_harmonic_complexity": float(stats[2] or 0.5),
                "avg_rhythmic_complexity": float(stats[1] / 5.0 if stats[1] else 0.3),
                "faiss_index_size": with_embeddings,
                "embedding_dimension": self.embedding_dim
            }
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "total_routes": 0,
                "routes_with_embeddings": 0,
                "embedding_coverage": 0.0,
                "avg_distance_km": 0.0,
                "avg_stops": 0.0,
                "avg_melodic_complexity": 0.5,
                "avg_harmonic_complexity": 0.5,
                "avg_rhythmic_complexity": 0.3,
                "faiss_index_size": 0,
                "embedding_dimension": self.embedding_dim
            }


# Global service instance
_route_embedding_service_duckdb = None

def get_route_embedding_service_duckdb() -> RouteEmbeddingServiceDuckDB:
    """Get the global DuckDB route embedding service instance"""
    global _route_embedding_service_duckdb
    if _route_embedding_service_duckdb is None:
        _route_embedding_service_duckdb = RouteEmbeddingServiceDuckDB()
    return _route_embedding_service_duckdb
