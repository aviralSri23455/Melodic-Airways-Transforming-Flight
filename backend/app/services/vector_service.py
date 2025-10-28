"""
Vector search service for music similarity using FREE MariaDB JSON features
Application-level real-time similarity search without paid extensions
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import math
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, text

from app.models.models import MusicComposition, Route
from app.services.cache import CacheService


@dataclass
class MusicVector:
    """Music vector representation with multiple feature dimensions"""
    harmonic_features: List[float]  # 12-dimensional chroma features
    rhythmic_features: List[float]  # 8-dimensional rhythm patterns
    melodic_features: List[float]   # 16-dimensional melodic contours
    genre_features: List[float]     # 10-dimensional genre characteristics
    timestamp: Optional[str] = None

    def to_json(self) -> dict:
        """Convert vector to JSON representation for MariaDB storage"""
        return {
            "harmonic": self.harmonic_features,
            "rhythmic": self.rhythmic_features,
            "melodic": self.melodic_features,
            "genre": self.genre_features,
            "timestamp": self.timestamp or datetime.utcnow().isoformat()
        }

    @staticmethod
    def from_json(json_data: dict) -> "MusicVector":
        """Create MusicVector from JSON data"""
        return MusicVector(
            harmonic_features=json_data.get("harmonic", []),
            rhythmic_features=json_data.get("rhythmic", []),
            melodic_features=json_data.get("melodic", []),
            genre_features=json_data.get("genre", []),
            timestamp=json_data.get("timestamp")
        )

    @staticmethod
    def from_composition_data(
        tempo: int,
        pitch: float,
        harmony: float,
        complexity: float,
        genre: str = "unknown"
    ) -> "MusicVector":
        """Create music vector from composition parameters"""
        # Generate feature vectors from composition data
        harmonic_features = [
            harmony, pitch / 127, complexity,
            (tempo / 240) * 0.8, harmony * 0.9, pitch * 0.7,
            complexity * 0.6, (tempo / 240) * 0.5, harmony * 0.4,
            pitch * 0.3, complexity * 0.2, (tempo / 240) * 0.1
        ]

        rhythmic_features = [
            (tempo / 240), (tempo / 240) * 0.8, (tempo / 240) * 0.6,
            (tempo / 240) * 0.4, (tempo / 240) * 0.2, complexity * 0.5,
            complexity * 0.3, complexity * 0.1
        ]

        melodic_features = [
            pitch / 127, pitch * 0.9, pitch * 0.8, pitch * 0.7,
            pitch * 0.6, pitch * 0.5, pitch * 0.4, pitch * 0.3,
            complexity, complexity * 0.8, complexity * 0.6, complexity * 0.4,
            harmony, harmony * 0.8, harmony * 0.6, harmony * 0.4
        ]

        genre_map = {
            "classical": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            "jazz": [0.6, 0.8, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0],
            "electronic": [0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
            "ambient": [0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "pop": [0.5, 0.7, 0.9, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0],
        }

        genre_features = genre_map.get(genre.lower(), [0.5] * 10)

        return MusicVector(
            harmonic_features=harmonic_features,
            rhythmic_features=rhythmic_features,
            melodic_features=melodic_features,
            genre_features=genre_features
        )


@dataclass
class SimilarityResult:
    """Result of similarity search"""
    composition_id: int
    title: str
    genre: str
    similarity_score: float
    distance: float


class VectorSearchService:
    """Service for vector-based music similarity search using FREE MariaDB JSON"""

    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache_service = cache_service
        self.cache_ttl = 3600  # 1 hour
    
    def _calculate_cosine_similarity(self, vector1: MusicVector, vector2_json: dict) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1: MusicVector object
            vector2_json: JSON representation of second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert vector2 from JSON to flat list
        vector2_flat = (
            vector2_json.get("harmonic", []) +
            vector2_json.get("rhythmic", []) +
            vector2_json.get("melodic", []) +
            vector2_json.get("genre", [])
        )
        
        # Convert vector1 to flat list
        vector1_flat = (
            vector1.harmonic_features +
            vector1.rhythmic_features +
            vector1.melodic_features +
            vector1.genre_features
        )
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector1_flat, vector2_flat))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vector1_flat))
        magnitude2 = math.sqrt(sum(b * b for b in vector2_flat))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

    async def extract_features(
        self,
        composition_id: int,
        tempo: int,
        pitch: float,
        harmony: float,
        complexity: float,
        genre: str = "unknown"
    ) -> MusicVector:
        """Extract music features and create vector representation"""
        vector = MusicVector.from_composition_data(tempo, pitch, harmony, complexity, genre)
        return vector

    async def store_vector(
        self,
        db: AsyncSession,
        composition_id: int,
        vector: MusicVector
    ) -> bool:
        """Store vector in database as JSON"""
        try:
            result = await db.execute(
                select(MusicComposition).where(MusicComposition.id == composition_id)
            )
            composition = result.scalar_one_or_none()

            if composition:
                composition.music_vector = vector.to_json()
                await db.commit()
                return True
            return False
        except Exception as e:
            await db.rollback()
            raise e

    async def find_similar(
        self,
        db: AsyncSession,
        vector: MusicVector,
        limit: int = 10,
        genre_filter: Optional[str] = None
    ) -> List[SimilarityResult]:
        """Find similar compositions using JSON-based similarity"""
        cache_key = f"similarity:{hash(str(vector.to_json()))}:{genre_filter}:{limit}"

        if self.cache_service:
            cached = await self.cache_service.get(cache_key)
            if cached:
                return [SimilarityResult(**item) for item in json.loads(cached)]

        try:
            # Use raw SQL for complex JSON similarity queries
            query = """
            SELECT
                mc.id,
                mc.title,
                mc.genre,
                mc.music_vector,
                (
                    (JSON_EXTRACT(mc.music_vector, '$.harmonic[0]') * %s +
                     JSON_EXTRACT(mc.music_vector, '$.harmonic[1]') * %s +
                     JSON_EXTRACT(mc.music_vector, '$.harmonic[2]') * %s +
                     JSON_EXTRACT(mc.music_vector, '$.rhythmic[0]') * %s +
                     JSON_EXTRACT(mc.music_vector, '$.rhythmic[1]') * %s +
                     JSON_EXTRACT(mc.music_vector, '$.melodic[0]') * %s +
                     JSON_EXTRACT(mc.music_vector, '$.melodic[1]') * %s) /
                    SQRT(
                        POWER(JSON_EXTRACT(mc.music_vector, '$.harmonic[0]'), 2) +
                        POWER(JSON_EXTRACT(mc.music_vector, '$.harmonic[1]'), 2) +
                        POWER(JSON_EXTRACT(mc.music_vector, '$.harmonic[2]'), 2) +
                        POWER(JSON_EXTRACT(mc.music_vector, '$.rhythmic[0]'), 2) +
                        POWER(JSON_EXTRACT(mc.music_vector, '$.rhythmic[1]'), 2) +
                        POWER(JSON_EXTRACT(mc.music_vector, '$.melodic[0]'), 2) +
                        POWER(JSON_EXTRACT(mc.music_vector, '$.melodic[1]'), 2) + 0.0001
                    ) /
                    SQRT(%s + %s + %s + %s + %s + %s + %s + %s + 0.0001)
                ) as similarity_score
            FROM music_compositions mc
            WHERE mc.music_vector IS NOT NULL
            AND mc.is_public = 1
            """

            params = [
                vector.harmonic_features[0], vector.harmonic_features[1], vector.harmonic_features[2],
                vector.rhythmic_features[0], vector.rhythmic_features[1],
                vector.melodic_features[0], vector.melodic_features[1],
                vector.harmonic_features[0], vector.harmonic_features[1], vector.harmonic_features[2],
                vector.rhythmic_features[0], vector.rhythmic_features[1],
                vector.melodic_features[0], vector.melodic_features[1]
            ]

            if genre_filter:
                query += " AND mc.genre = %s"
                params.append(genre_filter)

            query += " ORDER BY similarity_score DESC LIMIT %s"
            params.append(limit)

            result = await db.execute(text(query), params)
            rows = result.fetchall()

            results = []
            for row in rows:
                similarity_score = float(row.similarity_score) if row.similarity_score else 0.0
                results.append(SimilarityResult(
                    composition_id=row.id,
                    title=row.title or f"Composition {row.id}",
                    genre=row.genre or "unknown",
                    similarity_score=similarity_score,
                    distance=1 - similarity_score
                ))

            if self.cache_service:
                await self.cache_service.set(
                    cache_key,
                    json.dumps([asdict(r) for r in results]),
                    ttl=self.cache_ttl
                )

            return results
        except Exception as e:
            raise e

    async def batch_similarity_search(
        self,
        db: AsyncSession,
        vectors: List[MusicVector],
        limit: int = 10
    ) -> List[List[SimilarityResult]]:
        """Perform similarity search for multiple vectors"""
        results = []
        for vector in vectors:
            similar = await self.find_similar(db, vector, limit)
            results.append(similar)
        return results

    async def get_composition_vector(
        self,
        db: AsyncSession,
        composition_id: int
    ) -> Optional[MusicVector]:
        """Retrieve vector for a specific composition"""
        result = await db.execute(
            select(MusicComposition).where(MusicComposition.id == composition_id)
        )
        composition = result.scalar_one_or_none()

        if composition and composition.music_vector:
            return MusicVector.from_json(composition.music_vector)
        return None

    async def find_similar_routes(
        self,
        db: AsyncSession,
        route_embedding: List[float],
        limit: int = 10
    ) -> List[Dict]:
        """Find similar routes using JSON-based embedding similarity"""
        try:
            # Use raw SQL for route similarity
            query = """
            SELECT
                r.id,
                r.origin_airport_id,
                r.destination_airport_id,
                r.distance_km,
                r.route_embedding,
                (
                    JSON_EXTRACT(r.route_embedding, '$[0]') * %s +
                    JSON_EXTRACT(r.route_embedding, '$[1]') * %s +
                    JSON_EXTRACT(r.route_embedding, '$[2]') * %s +
                    JSON_EXTRACT(r.route_embedding, '$[3]') * %s
                ) /
                SQRT(
                    POWER(JSON_EXTRACT(r.route_embedding, '$[0]'), 2) +
                    POWER(JSON_EXTRACT(r.route_embedding, '$[1]'), 2) +
                    POWER(JSON_EXTRACT(r.route_embedding, '$[2]'), 2) +
                    POWER(JSON_EXTRACT(r.route_embedding, '$[3]'), 2) + 0.0001
                ) /
                SQRT(%s + %s + %s + %s + 0.0001) as similarity_score
            FROM routes r
            WHERE r.route_embedding IS NOT NULL
            ORDER BY similarity_score DESC
            LIMIT %s
            """

            params = [
                route_embedding[0], route_embedding[1], route_embedding[2], route_embedding[3],
                route_embedding[0], route_embedding[1], route_embedding[2], route_embedding[3],
                limit
            ]

            result = await db.execute(text(query), params)
            rows = result.fetchall()

            similar_routes = []
            for row in rows:
                similarity_score = float(row.similarity_score) if row.similarity_score else 0.0
                similar_routes.append({
                    "route_id": row.id,
                    "origin_airport_id": row.origin_airport_id,
                    "destination_airport_id": row.destination_airport_id,
                    "distance_km": row.distance_km,
                    "similarity_score": similarity_score
                })

            return similar_routes
        except Exception as e:
            # Fallback to simple distance-based similarity
            result = await db.execute(
                select(Route)
                .where(Route.distance_km.isnot(None))
                .order_by(func.abs(Route.distance_km - route_embedding[0] * 10000))
                .limit(limit)
            )
            routes = result.scalars().all()

            return [
                {
                    "route_id": route.id,
                    "origin_airport_id": route.origin_airport_id,
                    "destination_airport_id": route.destination_airport_id,
                    "distance_km": route.distance_km,
                    "similarity_score": 0.5  # Default similarity
                }
                for route in routes
            ]
