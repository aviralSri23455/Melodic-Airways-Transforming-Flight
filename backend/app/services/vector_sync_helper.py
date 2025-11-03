"""
Vector Sync Helper - Automatically sync data from all endpoints to DuckDB
Converts frontend data to vector embeddings and stores in DuckDB for analytics
"""

import numpy as np
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorSyncHelper:
    """Helper class to sync data from all endpoints to DuckDB vector store"""
    
    def __init__(self):
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize DuckDB vector store connection"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'duckdb_analytics'))
            from vector_embeddings import DuckDBVectorStore
            
            self.vector_store = DuckDBVectorStore()
            logger.info("✅ Vector sync helper initialized")
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")
            self.vector_store = None
    
    def _generate_embedding(self, features: Dict[str, Any], dimensions: int) -> List[float]:
        """
        Generate vector embedding from features
        In production, use a real embedding model (e.g., sentence-transformers, OpenAI embeddings)
        For now, use feature-based embedding generation
        """
        # Extract numeric features
        numeric_features = []
        for key, value in features.items():
            if isinstance(value, (int, float)):
                numeric_features.append(float(value))
            elif isinstance(value, bool):
                numeric_features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple hash-based encoding for strings
                numeric_features.append(float(hash(value) % 1000) / 1000.0)
        
        # Pad or truncate to desired dimensions
        if len(numeric_features) < dimensions:
            # Pad with random values
            padding = np.random.randn(dimensions - len(numeric_features)) * 0.1
            embedding = numeric_features + padding.tolist()
        else:
            # Truncate
            embedding = numeric_features[:dimensions]
        
        # Normalize to unit length
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        
        return embedding_array.tolist()
    
    # ==================== HOME ROUTE SYNC ====================
    
    def sync_home_route(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        music_style: str,
        tempo: int,
        note_count: int,
        duration: float,
        metadata: Dict[str, Any] = None
    ):
        """Sync home route data to DuckDB (96D embeddings)"""
        if not self.vector_store:
            return False
        
        try:
            route_id = f"home_{origin}_{destination}_{int(datetime.now().timestamp())}"
            
            # Generate 96D embedding from route features
            features = {
                "distance_km": distance_km,
                "tempo": tempo,
                "note_count": note_count,
                "duration": duration,
                "music_style": music_style,
                "origin_hash": hash(origin),
                "destination_hash": hash(destination),
            }
            
            embedding = self._generate_embedding(features, 96)
            
            self.vector_store.store_home_route_embedding(
                route_id=route_id,
                embedding=embedding,
                origin=origin,
                destination=destination,
                distance_km=distance_km,
                music_style=music_style,
                tempo=tempo,
                note_count=note_count,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Synced home route: {origin} → {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync home route: {e}")
            return False
    
    # ==================== WELLNESS SYNC ====================
    
    def sync_wellness_composition(
        self,
        theme: str,
        calm_level: int,
        duration: int,
        note_count: int,
        binaural_frequency: float = None,
        metadata: Dict[str, Any] = None
    ):
        """Sync wellness composition to DuckDB (48D embeddings)"""
        if not self.vector_store:
            return False
        
        try:
            wellness_id = f"wellness_{theme}_{calm_level}_{int(datetime.now().timestamp())}"
            
            # Generate 48D embedding from wellness features
            features = {
                "calm_level": calm_level,
                "duration": duration,
                "note_count": note_count,
                "binaural_frequency": binaural_frequency or 0,
                "theme": theme,
            }
            
            embedding = self._generate_embedding(features, 48)
            
            self.vector_store.store_wellness_embedding(
                wellness_id=wellness_id,
                embedding=embedding,
                theme=theme,
                calm_level=calm_level,
                duration=duration,
                binaural_frequency=binaural_frequency,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Synced wellness: {theme} (calm: {calm_level})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync wellness: {e}")
            return False
    
    # ==================== EDUCATION SYNC ====================
    
    def sync_education_lesson(
        self,
        lesson_type: str,
        difficulty: str,
        topic: str,
        interaction_count: int = 0,
        metadata: Dict[str, Any] = None
    ):
        """Sync education lesson to DuckDB (64D embeddings)"""
        if not self.vector_store:
            return False
        
        try:
            lesson_id = f"education_{lesson_type}_{difficulty}_{int(datetime.now().timestamp())}"
            
            # Generate 64D embedding from education features
            features = {
                "lesson_type": lesson_type,
                "difficulty": difficulty,
                "topic": topic,
                "interaction_count": interaction_count,
            }
            
            embedding = self._generate_embedding(features, 64)
            
            self.vector_store.store_education_embedding(
                lesson_id=lesson_id,
                embedding=embedding,
                lesson_type=lesson_type,
                difficulty=difficulty,
                topic=topic,
                interaction_count=interaction_count,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Synced education: {lesson_type} ({difficulty})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync education: {e}")
            return False
    
    # ==================== AR/VR SYNC ====================
    
    def sync_arvr_session(
        self,
        session_type: str,
        origin: str,
        destination: str,
        waypoint_count: int,
        spatial_audio: bool,
        quality: str,
        duration: float,
        metadata: Dict[str, Any] = None
    ):
        """Sync AR/VR session to DuckDB (80D embeddings)"""
        if not self.vector_store:
            return False
        
        try:
            session_id = f"arvr_{origin}_{destination}_{int(datetime.now().timestamp())}"
            
            # Generate 80D embedding from AR/VR features
            features = {
                "session_type": session_type,
                "origin": origin,
                "destination": destination,
                "waypoint_count": waypoint_count,
                "spatial_audio": spatial_audio,
                "quality": quality,
                "duration": duration,
            }
            
            embedding = self._generate_embedding(features, 80)
            
            self.vector_store.store_arvr_embedding(
                session_id=session_id,
                embedding=embedding,
                session_type=session_type,
                origin=origin,
                destination=destination,
                waypoint_count=waypoint_count,
                spatial_audio=spatial_audio,
                quality=quality,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Synced AR/VR: {origin} → {destination} ({quality})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync AR/VR: {e}")
            return False
    
    # ==================== AI COMPOSER SYNC ====================
    
    def sync_ai_composer(
        self,
        composition_id: str,
        genre: str,
        tempo: int,
        complexity: float,
        duration: int,
        metadata: Dict[str, Any] = None
    ):
        """Sync AI composer data to DuckDB (128D embeddings)"""
        if not self.vector_store:
            return False
        
        try:
            # Generate 128D embedding from composer features
            features = {
                "genre": genre,
                "tempo": tempo,
                "complexity": complexity,
                "duration": duration,
            }
            
            embedding = self._generate_embedding(features, 128)
            
            self.vector_store.store_ai_composer_embedding(
                composition_id=composition_id,
                embedding=embedding,
                genre=genre,
                tempo=tempo,
                complexity=complexity,
                duration=duration,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Synced AI composer: {genre} (tempo: {tempo})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync AI composer: {e}")
            return False
    
    # ==================== TRAVEL LOG SYNC ====================
    
    def sync_travel_log(
        self,
        log_id: int,
        title: str,
        waypoint_count: int,
        travel_date: datetime,
        metadata: Dict[str, Any] = None
    ):
        """Sync travel log to DuckDB (32D embeddings)"""
        if not self.vector_store:
            return False
        
        try:
            # Generate 32D embedding from travel log features
            features = {
                "waypoint_count": waypoint_count,
                "title": title,
                "travel_date": travel_date.timestamp() if travel_date else 0,
            }
            
            embedding = self._generate_embedding(features, 32)
            
            self.vector_store.store_travel_log_embedding(
                log_id=log_id,
                embedding=embedding,
                title=title,
                waypoint_count=waypoint_count,
                travel_date=travel_date,
                metadata=metadata or {}
            )
            
            logger.info(f"✅ Synced travel log: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync travel log: {e}")
            return False
    
    def close(self):
        """Close vector store connection"""
        if self.vector_store:
            self.vector_store.close()


# Singleton instance
_vector_sync_helper = None


def get_vector_sync_helper() -> VectorSyncHelper:
    """Get singleton vector sync helper instance"""
    global _vector_sync_helper
    if _vector_sync_helper is None:
        _vector_sync_helper = VectorSyncHelper()
    return _vector_sync_helper
