"""
Route Embedding Service for OpenFlights Dataset
Generates vector embeddings for flight routes using PyTorch
Enables semantic similarity search for routes
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text

from app.models.models import Route, Airport
from app.services.faiss_duckdb_service import get_faiss_duckdb_service

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


class RouteEmbeddingService:
    """Service for generating and managing route vector embeddings"""
    
    def __init__(self):
        self.embedding_dim = 128
        self.encoder = RouteEncoder(input_dim=16, hidden_dim=64, output_dim=self.embedding_dim)
        self.encoder.eval()  # Set to evaluation mode
        self.faiss_service = get_faiss_duckdb_service()
        
        # Genre characteristics for musical mapping
        self.genre_profiles = {
            "classical": {
                "complexity": 0.8,
                "tempo_range": (60, 120),
                "harmony": 0.9,
                "structure": "formal"
            },
            "jazz": {
                "complexity": 0.9,
                "tempo_range": (100, 180),
                "harmony": 0.7,
                "structure": "improvisational"
            },
            "electronic": {
                "complexity": 0.6,
                "tempo_range": (120, 140),
                "harmony": 0.5,
                "structure": "repetitive"
            },
            "ambient": {
                "complexity": 0.3,
                "tempo_range": (60, 90),
                "harmony": 0.8,
                "structure": "flowing"
            },
            "pop": {
                "complexity": 0.5,
                "tempo_range": (100, 130),
                "harmony": 0.6,
                "structure": "verse-chorus"
            }
        }
    
    def _extract_route_features(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        distance_km: float,
        stops: int = 0,
        airline_count: int = 1
    ) -> np.ndarray:
        """
        Extract numerical features from route data
        
        Returns 16-dimensional feature vector:
        - Geographic features (8D)
        - Route characteristics (4D)
        - Musical mapping (4D)
        """
        # Geographic features
        lat_diff = dest_lat - origin_lat
        lon_diff = dest_lon - origin_lon
        avg_lat = (origin_lat + dest_lat) / 2
        avg_lon = (origin_lon + dest_lon) / 2
        
        # Normalize coordinates to [-1, 1]
        origin_lat_norm = origin_lat / 90.0
        origin_lon_norm = origin_lon / 180.0
        dest_lat_norm = dest_lat / 90.0
        dest_lon_norm = dest_lon / 180.0
        
        # Route characteristics
        distance_norm = min(distance_km / 20000.0, 1.0)  # Normalize to max 20,000 km
        stops_norm = min(stops / 5.0, 1.0)  # Normalize to max 5 stops
        airline_norm = min(airline_count / 10.0, 1.0)  # Normalize to max 10 airlines
        
        # Calculate bearing (direction)
        bearing = np.arctan2(lon_diff, lat_diff) / np.pi  # Normalize to [-1, 1]
        
        # Musical mapping features
        # Tempo: based on distance (longer = slower)
        tempo_feature = 1.0 - distance_norm
        
        # Pitch: based on latitude (higher latitude = higher pitch)
        pitch_feature = avg_lat / 90.0
        
        # Harmony: based on route complexity (more stops = more complex)
        harmony_feature = stops_norm
        
        # Rhythm: based on airline frequency
        rhythm_feature = airline_norm
        
        features = np.array([
            origin_lat_norm,
            origin_lon_norm,
            dest_lat_norm,
            dest_lon_norm,
            lat_diff / 180.0,
            lon_diff / 360.0,
            distance_norm,
            bearing,
            stops_norm,
            airline_norm,
            avg_lat / 90.0,
            avg_lon / 180.0,
            tempo_feature,
            pitch_feature,
            harmony_feature,
            rhythm_feature
        ], dtype=np.float32)
        
        return features
    
    def generate_embedding(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        distance_km: float,
        stops: int = 0,
        airline_count: int = 1
    ) -> np.ndarray:
        """Generate 128D embedding for a route"""
        # Extract features
        features = self._extract_route_features(
            origin_lat, origin_lon, dest_lat, dest_lon,
            distance_km, stops, airline_count
        )
        
        # Convert to PyTorch tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.encoder(features_tensor)
        
        # Convert back to numpy
        embedding_np = embedding.squeeze(0).numpy()
        
        return embedding_np
    
    async def generate_route_embeddings(
        self,
        db: AsyncSession,
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        Generate embeddings for all routes in database
        
        Returns statistics about generated embeddings
        """
        try:
            # Get all routes with airport information
            query = """
            SELECT 
                r.id,
                r.origin_airport_id,
                r.destination_airport_id,
                r.distance_km,
                r.stops,
                ao.latitude as origin_lat,
                ao.longitude as origin_lon,
                ad.latitude as dest_lat,
                ad.longitude as dest_lon,
                COUNT(DISTINCT r.airline_id) as airline_count
            FROM routes r
            JOIN airports ao ON r.origin_airport_id = ao.id
            JOIN airports ad ON r.destination_airport_id = ad.id
            WHERE ao.latitude IS NOT NULL 
            AND ao.longitude IS NOT NULL
            AND ad.latitude IS NOT NULL 
            AND ad.longitude IS NOT NULL
            GROUP BY r.id, r.origin_airport_id, r.destination_airport_id, 
                     r.distance_km, r.stops, ao.latitude, ao.longitude, 
                     ad.latitude, ad.longitude
            """
            
            result = await db.execute(text(query))
            routes = result.fetchall()
            
            generated_count = 0
            failed_count = 0
            
            for route in routes:
                try:
                    route_id = route[0]
                    origin_airport_id = route[1]
                    dest_airport_id = route[2]
                    distance_km = route[3] or 0
                    stops = route[4] or 0
                    origin_lat = route[5]
                    origin_lon = route[6]
                    dest_lat = route[7]
                    dest_lon = route[8]
                    airline_count = route[9] or 1
                    
                    # Generate embedding
                    embedding = self.generate_embedding(
                        origin_lat, origin_lon,
                        dest_lat, dest_lon,
                        distance_km, stops, airline_count
                    )
                    
                    # Store in database
                    embedding_json = embedding.tolist()
                    await db.execute(
                        text("""
                        UPDATE routes 
                        SET route_embedding = :embedding
                        WHERE id = :route_id
                        """),
                        {"embedding": str(embedding_json), "route_id": route_id}
                    )
                    
                    # Store in FAISS index
                    self.faiss_service.store_music_vector(
                        composition_id=route_id,
                        route_id=route_id,
                        origin=str(origin_airport_id),
                        destination=str(dest_airport_id),
                        genre="route",
                        tempo=int(120 * (1 - min(distance_km / 20000, 1))),
                        pitch=float(origin_lat / 90.0),
                        harmony=float(stops / 5.0),
                        complexity=float(distance_km / 20000),
                        vector=embedding
                    )
                    
                    generated_count += 1
                    
                    if generated_count % 100 == 0:
                        await db.commit()
                        logger.info(f"Generated {generated_count} embeddings...")
                
                except Exception as e:
                    logger.error(f"Failed to generate embedding for route {route[0]}: {e}")
                    failed_count += 1
            
            await db.commit()
            
            logger.info(f"Generated {generated_count} embeddings, {failed_count} failed")
            
            return {
                "generated": generated_count,
                "failed": failed_count,
                "total": len(routes)
            }
        
        except Exception as e:
            logger.error(f"Error generating route embeddings: {e}")
            await db.rollback()
            raise
    
    async def find_similar_routes(
        self,
        db: AsyncSession,
        origin: str,
        destination: str,
        limit: int = 10
    ) -> List[Dict]:
        """Find routes similar to the given origin-destination pair"""
        try:
            # Get origin and destination airport coordinates
            query = """
            SELECT 
                ao.latitude as origin_lat,
                ao.longitude as origin_lon,
                ad.latitude as dest_lat,
                ad.longitude as dest_lon,
                r.distance_km,
                r.stops
            FROM routes r
            JOIN airports ao ON r.origin_airport_id = ao.id
            JOIN airports ad ON r.destination_airport_id = ad.id
            WHERE ao.iata_code = :origin AND ad.iata_code = :destination
            LIMIT 1
            """
            
            result = await db.execute(
                text(query),
                {"origin": origin, "destination": destination}
            )
            route_data = result.fetchone()
            
            if not route_data:
                return []
            
            # Generate embedding for query route
            query_embedding = self.generate_embedding(
                route_data[0], route_data[1],
                route_data[2], route_data[3],
                route_data[4] or 0,
                route_data[5] or 0
            )
            
            # Search for similar routes using FAISS
            similar = self.faiss_service.search_similar_music(
                query_embedding,
                limit=limit
            )
            
            return similar
        
        except Exception as e:
            logger.error(f"Error finding similar routes: {e}")
            return []
    
    async def find_routes_by_genre(
        self,
        db: AsyncSession,
        genre: str,
        limit: int = 20
    ) -> List[Dict]:
        """Find routes that match a musical genre profile"""
        try:
            genre_profile = self.genre_profiles.get(genre.lower())
            if not genre_profile:
                return []
            
            # Build query based on genre characteristics
            complexity = genre_profile["complexity"]
            tempo_min, tempo_max = genre_profile["tempo_range"]
            
            # Map genre characteristics to route features
            # High complexity = more stops, longer distance
            # Tempo = inverse of distance (faster = shorter routes)
            
            query = """
            SELECT 
                r.id,
                r.origin_airport_id,
                r.destination_airport_id,
                r.distance_km,
                r.stops,
                r.route_embedding,
                ao.iata_code as origin_code,
                ad.iata_code as dest_code
            FROM routes r
            JOIN airports ao ON r.origin_airport_id = ao.id
            JOIN airports ad ON r.destination_airport_id = ad.id
            WHERE r.route_embedding IS NOT NULL
            """
            
            # Add genre-specific filters
            if complexity > 0.7:
                query += " AND r.stops >= 1"
            if complexity < 0.4:
                query += " AND r.stops = 0"
            
            # Tempo-based distance filtering
            if tempo_min > 120:
                query += " AND r.distance_km < 5000"
            elif tempo_max < 100:
                query += " AND r.distance_km > 5000"
            
            query += f" LIMIT {limit}"
            
            result = await db.execute(text(query))
            routes = result.fetchall()
            
            return [
                {
                    "route_id": route[0],
                    "origin_airport_id": route[1],
                    "destination_airport_id": route[2],
                    "distance_km": route[3],
                    "stops": route[4],
                    "origin_code": route[6],
                    "dest_code": route[7],
                    "genre": genre,
                    "genre_match_score": 0.8  # Placeholder
                }
                for route in routes
            ]
        
        except Exception as e:
            logger.error(f"Error finding routes by genre: {e}")
            return []
    
    async def calculate_melodic_complexity(
        self,
        db: AsyncSession,
        route_id: int
    ) -> Dict[str, float]:
        """Calculate melodic complexity metrics for a route"""
        try:
            query = """
            SELECT 
                r.distance_km,
                r.stops,
                r.route_embedding,
                ao.latitude as origin_lat,
                ad.latitude as dest_lat
            FROM routes r
            JOIN airports ao ON r.origin_airport_id = ao.id
            JOIN airports ad ON r.destination_airport_id = ad.id
            WHERE r.id = :route_id
            """
            
            result = await db.execute(text(query), {"route_id": route_id})
            route = result.fetchone()
            
            if not route:
                return {}
            
            distance_km = route[0] or 0
            stops = route[1] or 0
            origin_lat = route[3] or 0
            dest_lat = route[4] or 0
            
            # Calculate complexity metrics
            # Harmonic complexity: based on latitude change
            lat_change = abs(dest_lat - origin_lat)
            harmonic_complexity = min(lat_change / 180.0, 1.0)
            
            # Rhythmic complexity: based on stops
            rhythmic_complexity = min(stops / 5.0, 1.0)
            
            # Melodic complexity: based on distance
            melodic_complexity = min(distance_km / 20000.0, 1.0)
            
            # Overall complexity: weighted average
            overall_complexity = (
                harmonic_complexity * 0.3 +
                rhythmic_complexity * 0.3 +
                melodic_complexity * 0.4
            )
            
            # Update database
            await db.execute(
                text("""
                UPDATE routes 
                SET 
                    melodic_complexity = :melodic,
                    harmonic_complexity = :harmonic,
                    rhythmic_complexity = :rhythmic
                WHERE id = :route_id
                """),
                {
                    "melodic": melodic_complexity,
                    "harmonic": harmonic_complexity,
                    "rhythmic": rhythmic_complexity,
                    "route_id": route_id
                }
            )
            await db.commit()
            
            return {
                "harmonic_complexity": round(harmonic_complexity, 3),
                "rhythmic_complexity": round(rhythmic_complexity, 3),
                "melodic_complexity": round(melodic_complexity, 3),
                "overall_complexity": round(overall_complexity, 3)
            }
        
        except Exception as e:
            logger.error(f"Error calculating melodic complexity: {e}")
            return {}
    
    async def get_embedding_statistics(self, db: AsyncSession) -> Dict:
        """Get statistics about route embeddings"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_routes,
                COUNT(route_embedding) as routes_with_embeddings,
                AVG(distance_km) as avg_distance,
                AVG(stops) as avg_stops,
                AVG(melodic_complexity) as avg_melodic_complexity,
                AVG(harmonic_complexity) as avg_harmonic_complexity,
                AVG(rhythmic_complexity) as avg_rhythmic_complexity
            FROM routes
            """
            
            result = await db.execute(text(query))
            stats = result.fetchone()
            
            faiss_stats = self.faiss_service.get_statistics()
            
            return {
                "total_routes": stats[0] or 0,
                "routes_with_embeddings": stats[1] or 0,
                "embedding_coverage": round((stats[1] or 0) / (stats[0] or 1) * 100, 2),
                "avg_distance_km": round(stats[2] or 0, 2),
                "avg_stops": round(stats[3] or 0, 2),
                "avg_melodic_complexity": round(stats[4] or 0, 3),
                "avg_harmonic_complexity": round(stats[5] or 0, 3),
                "avg_rhythmic_complexity": round(stats[6] or 0, 3),
                "faiss_index_size": faiss_stats.get("faiss_index_size", 0),
                "embedding_dimension": self.embedding_dim
            }
        
        except Exception as e:
            logger.error(f"Error getting embedding statistics: {e}")
            return {}


# Global service instance
_route_embedding_service = None

def get_route_embedding_service() -> RouteEmbeddingService:
    """Get the global route embedding service instance"""
    global _route_embedding_service
    if _route_embedding_service is None:
        _route_embedding_service = RouteEmbeddingService()
    return _route_embedding_service
