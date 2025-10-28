"""
FAISS + DuckDB Hybrid Vector Search Service
Combines FAISS for fast vector similarity search with DuckDB for metadata storage
"""

import faiss
import numpy as np
import duckdb
import logging
import json
import os
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import threading
from contextlib import contextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)


class FAISSVectorIndex:
    """FAISS vector index for music similarity search"""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.index = None
        self.id_to_metadata = {}
        self.metadata_to_id = {}
        self.next_id = 0
        self._lock = threading.Lock()

    def build_index(self, vectors: np.ndarray, metadatas: List[Dict] = None):
        """Build FAISS index from vectors"""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match expected {self.dimension}")

        # Create index (using IndexFlatL2 for exact search, can be replaced with IVF for large datasets)
        self.index = faiss.IndexFlatL2(self.dimension)

        # Add vectors to index
        self.index.add(vectors.astype(np.float32))

        # Store metadata mapping
        if metadatas:
            with self._lock:
                for i, metadata in enumerate(metadatas):
                    vector_id = self.next_id
                    self.id_to_metadata[vector_id] = metadata
                    self.metadata_to_id[metadata.get('id', f'vector_{vector_id}')] = vector_id
                    self.next_id += 1

        logger.info(f"Built FAISS index with {vectors.shape[0]} vectors")

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors"""
        if self.index is None:
            raise ValueError("Index not built yet")

        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_vector, k)

        return distances[0], indices[0]

    def add_vector(self, vector: np.ndarray, metadata: Dict):
        """Add single vector to index"""
        if self.index is None:
            # Initialize index if not exists
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_to_metadata = {}
            self.metadata_to_id = {}
            self.next_id = 0

        vector = vector.reshape(1, -1).astype(np.float32)
        self.index.add(vector)

        with self._lock:
            vector_id = self.next_id
            self.id_to_metadata[vector_id] = metadata
            self.metadata_to_id[metadata.get('id', f'vector_{vector_id}')] = vector_id
            self.next_id += 1

        logger.info(f"Added vector with ID {vector_id}")


class FAISSDuckDBService:
    """Hybrid service combining FAISS vector search with DuckDB metadata storage"""

    def __init__(self):
        self.db_path = settings.DUCKDB_PATH
        self.faiss_path = str(Path(self.db_path).parent / "vectors.faiss")
        self.metadata_path = str(Path(self.db_path).parent / "vector_metadata.json")
        self.conn = None
        self.vector_index = None
        self.dimension = 128  # Music feature vector dimension
        self._lock = threading.Lock()

        self._initialize_database()
        self._load_or_build_index()

    def _initialize_database(self):
        """Initialize DuckDB connection and create tables"""
        try:
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            self.conn = duckdb.connect(database=self.db_path, read_only=False)

            # Configure DuckDB
            self.conn.execute(f"SET memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
            self.conn.execute(f"SET threads={settings.DUCKDB_THREADS}")

            # Create vector metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS music_vectors (
                    id INTEGER PRIMARY KEY,
                    composition_id INTEGER,
                    route_id INTEGER,
                    origin VARCHAR,
                    destination VARCHAR,
                    genre VARCHAR,
                    tempo INTEGER,
                    pitch REAL,
                    harmony REAL,
                    complexity REAL,
                    vector BLOB,
                    metadata_json VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            # Create index on composition_id for fast lookups
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_composition_id
                ON music_vectors (composition_id)
            """)

            logger.info("FAISS + DuckDB service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS + DuckDB service: {e}")
            raise

    def _load_or_build_index(self):
        """Load existing FAISS index or build new one from database"""
        try:
            # Try to load existing index
            if os.path.exists(self.faiss_path) and os.path.exists(self.metadata_path):
                self._load_index()
            else:
                self._build_index_from_database()
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            self._build_index_from_database()

    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.vector_index = FAISSVectorIndex(self.dimension)

            # Load metadata mapping
            with open(self.metadata_path, 'r') as f:
                metadata_list = json.load(f)

            # Load vectors from database and rebuild FAISS index
            result = self.conn.execute("""
                SELECT id, vector, metadata_json
                FROM music_vectors
                ORDER BY id
            """).fetchall()

            if result:
                vectors = []
                metadatas = []

                for row in result:
                    vector_id, vector_blob, metadata_json = row
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    metadata = json.loads(metadata_json) if metadata_json else {}

                    vectors.append(vector)
                    metadatas.append(metadata)

                if vectors:
                    vectors_array = np.array(vectors)
                    self.vector_index.build_index(vectors_array, metadatas)

            logger.info(f"Loaded FAISS index with {len(result)} vectors")

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def _build_index_from_database(self):
        """Build FAISS index from database vectors"""
        try:
            self.vector_index = FAISSVectorIndex(self.dimension)

            # Get all vectors from database
            result = self.conn.execute("""
                SELECT id, vector, metadata_json
                FROM music_vectors
                ORDER BY id
            """).fetchall()

            if result:
                vectors = []
                metadatas = []

                for row in result:
                    vector_id, vector_blob, metadata_json = row
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    metadata = json.loads(metadata_json) if metadata_json else {}

                    vectors.append(vector)
                    metadatas.append(metadata)

                if vectors:
                    vectors_array = np.array(vectors)
                    self.vector_index.build_index(vectors_array, metadatas)

                    logger.info(f"Built FAISS index from database with {len(vectors)} vectors")

        except Exception as e:
            logger.error(f"Error building index from database: {e}")

    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.vector_index and self.vector_index.index:
                # Save FAISS index
                faiss.write_index(self.vector_index.index, self.faiss_path)

                # Save metadata mapping
                metadata_list = list(self.vector_index.id_to_metadata.values())
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata_list, f, default=str)

                logger.info("Saved FAISS index and metadata to disk")

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def store_music_vector(
        self,
        composition_id: int,
        route_id: int,
        origin: str,
        destination: str,
        genre: str,
        tempo: int,
        pitch: float,
        harmony: float,
        complexity: float,
        vector: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store music vector in FAISS + DuckDB"""
        try:
            # Store in DuckDB
            metadata_json = json.dumps(metadata) if metadata else None
            vector_blob = vector.astype(np.float32).tobytes()

            self.conn.execute("""
                INSERT INTO music_vectors (
                    composition_id, route_id, origin, destination, genre,
                    tempo, pitch, harmony, complexity, vector, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                composition_id, route_id, origin, destination, genre,
                tempo, pitch, harmony, complexity, vector_blob, metadata_json,
                datetime.utcnow(), datetime.utcnow()
            ])

            # Get the inserted ID
            result = self.conn.execute("SELECT last_insert_rowid()").fetchone()
            vector_db_id = result[0] if result else None

            # Add to FAISS index
            if self.vector_index:
                metadata = {
                    'id': str(vector_db_id),
                    'composition_id': composition_id,
                    'route_id': route_id,
                    'genre': genre,
                    'tempo': tempo,
                    'origin': origin,
                    'destination': destination
                }
                self.vector_index.add_vector(vector, metadata)

            # Save index to disk
            self._save_index()

            logger.info(f"Stored music vector for composition {composition_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing music vector: {e}")
            return False

    def search_similar_music(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        genre_filter: Optional[str] = None,
        origin_filter: Optional[str] = None,
        destination_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar music using FAISS + DuckDB filtering"""
        try:
            if not self.vector_index or not self.vector_index.index:
                return []

            # Search in FAISS
            distances, indices = self.vector_index.search(query_vector, limit * 2)  # Get more to filter

            # Get metadata for results
            results = []
            seen_ids = set()

            for distance, idx in zip(distances, indices):
                if idx == -1 or distance > 1.0:  # Skip invalid results and high distances
                    continue

                metadata = self.vector_index.id_to_metadata.get(idx)
                if not metadata:
                    continue

                # Apply filters
                if genre_filter and metadata.get('genre', '').lower() != genre_filter.lower():
                    continue
                if origin_filter and metadata.get('origin') != origin_filter:
                    continue
                if destination_filter and metadata.get('destination') != destination_filter:
                    continue

                # Avoid duplicates
                vector_id = metadata.get('id')
                if vector_id in seen_ids:
                    continue
                seen_ids.add(vector_id)

                # Get full metadata from database
                if vector_id:
                    db_result = self.conn.execute("""
                        SELECT * FROM music_vectors WHERE id = ?
                    """, [vector_id]).fetchone()

                    if db_result:
                        results.append({
                            'id': db_result[0],
                            'composition_id': db_result[1],
                            'route_id': db_result[2],
                            'origin': db_result[3],
                            'destination': db_result[4],
                            'genre': db_result[5],
                            'tempo': db_result[6],
                            'pitch': db_result[7],
                            'harmony': db_result[8],
                            'complexity': db_result[9],
                            'similarity_score': 1.0 - distance,  # Convert distance to similarity
                            'distance': distance,
                            'metadata': json.loads(db_result[11]) if db_result[11] else {}
                        })

                if len(results) >= limit:
                    break

            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)

            logger.info(f"Found {len(results)} similar music compositions")
            return results

        except Exception as e:
            logger.error(f"Error searching similar music: {e}")
            return []

    def search_similar_routes(
        self,
        origin: str,
        destination: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar routes using FAISS vector search"""
        try:
            # Get route vector from database (this would need to be implemented)
            # For now, create a simple embedding based on route characteristics
            route_embedding = self._create_route_embedding(origin, destination)

            # Search for similar routes
            return self.search_similar_music(
                route_embedding,
                limit=limit,
                origin_filter=origin,
                destination_filter=destination
            )

        except Exception as e:
            logger.error(f"Error searching similar routes: {e}")
            return []

    def _create_route_embedding(self, origin: str, destination: str) -> np.ndarray:
        """Create a simple route embedding (placeholder implementation)"""
        # This is a simplified implementation
        # In a real scenario, you'd use more sophisticated embeddings
        embedding = np.zeros(self.dimension)

        # Simple hash-based embedding
        origin_hash = hash(origin) % 1000
        dest_hash = hash(destination) % 1000

        embedding[0] = origin_hash / 1000.0
        embedding[1] = dest_hash / 1000.0
        embedding[2] = (origin_hash + dest_hash) / 2000.0

        return embedding

    def get_music_vector(self, composition_id: int) -> Optional[np.ndarray]:
        """Get music vector for a composition"""
        try:
            result = self.conn.execute("""
                SELECT vector FROM music_vectors WHERE composition_id = ?
            """, [composition_id]).fetchone()

            if result and result[0]:
                return np.frombuffer(result[0], dtype=np.float32)

            return None

        except Exception as e:
            logger.error(f"Error getting music vector: {e}")
            return None

    def update_music_vector(
        self,
        composition_id: int,
        vector: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Update music vector for a composition"""
        try:
            vector_blob = vector.astype(np.float32).tobytes()
            metadata_json = json.dumps(metadata) if metadata else None

            self.conn.execute("""
                UPDATE music_vectors
                SET vector = ?, metadata_json = ?, updated_at = ?
                WHERE composition_id = ?
            """, [vector_blob, metadata_json, datetime.utcnow(), composition_id])

            # Update in FAISS index (rebuild for simplicity)
            self._load_or_build_index()

            logger.info(f"Updated music vector for composition {composition_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating music vector: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        try:
            result = self.conn.execute("""
                SELECT
                    COUNT(*) as total_vectors,
                    COUNT(DISTINCT genre) as unique_genres,
                    COUNT(DISTINCT origin) as unique_origins,
                    COUNT(DISTINCT destination) as unique_destinations,
                    AVG(tempo) as avg_tempo,
                    AVG(complexity) as avg_complexity
                FROM music_vectors
            """).fetchone()

            if result:
                return {
                    'total_vectors': result[0],
                    'unique_genres': result[1],
                    'unique_origins': result[2],
                    'unique_destinations': result[3],
                    'avg_tempo': round(result[4], 2) if result[4] else 0,
                    'avg_complexity': round(result[5], 2) if result[5] else 0,
                    'faiss_index_size': self.vector_index.index.ntotal if self.vector_index and self.vector_index.index else 0,
                    'index_type': 'IndexFlatL2'
                }

            return {
                'total_vectors': 0,
                'unique_genres': 0,
                'unique_origins': 0,
                'unique_destinations': 0,
                'avg_tempo': 0,
                'avg_complexity': 0,
                'faiss_index_size': 0,
                'index_type': 'IndexFlatL2'
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'total_vectors': 0,
                'unique_genres': 0,
                'unique_origins': 0,
                'unique_destinations': 0,
                'avg_tempo': 0,
                'avg_complexity': 0,
                'faiss_index_size': 0,
                'index_type': 'IndexFlatL2'
            }


# Global service instance
faiss_duckdb_service = None


class FallbackFAISSDuckDBService:
    """Fallback service when FAISS initialization fails"""

    def __init__(self):
        logger.warning("Using fallback FAISS service - vector search disabled")

    def search_similar_music(self, query_vector, limit=10, **kwargs):
        """Fallback - return empty results"""
        return []

    def search_similar_routes(self, origin, destination, limit=10):
        """Fallback - return empty results"""
        return []

    def store_music_vector(self, *args, **kwargs):
        """Fallback - return False"""
        return False

    def get_statistics(self):
        """Fallback - return zero statistics"""
        return {
            'total_vectors': 0,
            'unique_genres': 0,
            'unique_origins': 0,
            'unique_destinations': 0,
            'avg_tempo': 0,
            'avg_complexity': 0,
            'faiss_index_size': 0,
            'index_type': 'fallback'
        }

def get_faiss_duckdb_service() -> Union[FAISSDuckDBService, FallbackFAISSDuckDBService]:
    """Get the global FAISS + DuckDB service instance"""
    global faiss_duckdb_service
    if faiss_duckdb_service is None:
        try:
            faiss_duckdb_service = FAISSDuckDBService()
        except Exception as e:
            logger.error(f"Failed to initialize FAISS DuckDB service: {e}")
            # Use fallback service instead of breaking the router
            faiss_duckdb_service = FallbackFAISSDuckDBService()
    return faiss_duckdb_service
