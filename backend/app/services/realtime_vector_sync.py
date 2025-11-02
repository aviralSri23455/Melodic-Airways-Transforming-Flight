"""
Real-time Vector Embedding Sync Service
Automatically generates and stores vector embeddings when music is created
"""

import logging
import json
from typing import Optional
from datetime import datetime

from app.services.route_embedding_service_duckdb import get_route_embedding_service_duckdb
from app.services.duckdb_analytics import DuckDBAnalytics

logger = logging.getLogger(__name__)


class RealtimeVectorSync:
    """Service to sync vector embeddings in real-time when music is generated"""
    
    def __init__(self):
        self.vector_service = get_route_embedding_service_duckdb()
        self.duckdb = DuckDBAnalytics()
    
    def sync_route_embedding(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        complexity_score: float = 0.5,
        intermediate_stops: int = 0,
        music_features: Optional[dict] = None
    ) -> bool:
        """
        Generate and store route embedding in real-time from ACTUAL MUSIC DATA
        
        Args:
            origin: Origin airport code
            destination: Destination airport code
            distance_km: Route distance in kilometers
            complexity_score: Route complexity (0-1)
            intermediate_stops: Number of stops
            music_features: REAL-TIME music features from playback (tempo, notes, harmony)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding from REAL MUSIC DATA if available
            if music_features:
                embedding = self._generate_music_based_embedding(
                    origin=origin,
                    destination=destination,
                    distance_km=distance_km,
                    music_features=music_features
                )
                logger.info(f"🎵 Generated embedding from REAL MUSIC DATA: {origin} → {destination}")
            else:
                # Fallback to route-based embedding
                embedding = self.vector_service.generate_embedding(
                    origin=origin,
                    destination=destination,
                    distance_km=distance_km,
                    complexity_score=complexity_score,
                    stops=intermediate_stops
                )
                logger.warning(f"⚠️ Generated embedding from route metadata (no music data): {origin} → {destination}")
            
            # Convert to JSON string
            embedding_json = json.dumps(embedding.tolist())
            
            # Store in DuckDB with embedding
            if self.duckdb.conn:
                import random
                import time
                record_id = int(time.time() % 1000000) + random.randint(1, 999)
                
                self.duckdb.conn.execute("""
                    INSERT INTO route_analytics (
                        id, origin, destination, distance_km, complexity_score,
                        path_length, intermediate_stops, route_embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    record_id, origin, destination, distance_km, complexity_score,
                    1, intermediate_stops, embedding_json, datetime.utcnow()
                ])
                
                logger.info(f"✅ Synced embedding for {origin} → {destination}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error syncing route embedding: {e}")
            return False
    
    def _generate_music_based_embedding(
        self,
        origin: str,
        destination: str,
        distance_km: float,
        music_features: dict
    ):
        """
        Generate embedding from ACTUAL MUSIC PLAYBACK DATA
        
        Args:
            music_features: Dict with keys:
                - tempo: BPM
                - note_count: Number of notes played
                - duration_seconds: Duration
                - harmony_complexity: Harmony score
                - pitch_range: Range of pitches used
                - rhythm_density: Notes per second
        """
        import numpy as np
        import torch
        
        # Extract real music features
        tempo = music_features.get('tempo', 120) / 200.0  # Normalize
        note_count = music_features.get('note_count', 100) / 500.0  # Normalize
        duration = music_features.get('duration_seconds', 60) / 300.0  # Normalize
        harmony = music_features.get('harmony_complexity', 0.5)
        pitch_range = music_features.get('pitch_range', 24) / 88.0  # Piano range
        rhythm_density = music_features.get('rhythm_density', 2.0) / 10.0  # Notes per second
        
        # Route context
        origin_hash = sum(ord(c) for c in origin) / 1000.0
        dest_hash = sum(ord(c) for c in destination) / 1000.0
        distance_norm = distance_km / 10000.0
        
        # Create 16-dimensional feature vector from REAL MUSIC
        features = np.array([
            tempo,  # Real tempo from music
            note_count,  # Real note count
            duration,  # Real duration
            harmony,  # Real harmony complexity
            pitch_range,  # Real pitch range
            rhythm_density,  # Real rhythm density
            tempo * harmony,  # Combined music feature
            note_count * rhythm_density,  # Note density
            origin_hash,  # Route context
            dest_hash,
            distance_norm,
            np.sin(tempo * np.pi),  # Temporal patterns
            np.cos(harmony * np.pi),  # Harmonic patterns
            pitch_range * harmony,  # Melodic complexity
            rhythm_density * tempo / 200.0,  # Rhythmic energy
            (tempo + harmony + pitch_range) / 3.0  # Overall musical complexity
        ], dtype=np.float32)
        
        # Use the encoder to generate 128-dimensional embedding
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            embedding = self.vector_service.encoder(features_tensor)
            return embedding.squeeze(0).numpy()
    
    def sync_music_vector(
        self,
        route_id: int,
        origin: str,
        destination: str,
        tempo: int,
        key: str,
        scale: str,
        duration_seconds: float,
        note_count: int,
        harmony_complexity: float,
        genre: str
    ) -> bool:
        """
        Generate and store music vector embedding in real-time
        
        Args:
            route_id: Associated route ID
            origin: Origin airport code
            destination: Destination airport code
            tempo: Music tempo (BPM)
            key: Musical key
            scale: Musical scale
            duration_seconds: Duration in seconds
            note_count: Total number of notes
            harmony_complexity: Harmony complexity score
            genre: Music genre
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate music embedding based on characteristics
            distance_km = 5000  # Default, will be updated if route exists
            
            # Create a simple music vector based on characteristics
            music_features = [
                tempo / 200.0,  # Normalized tempo
                harmony_complexity,
                note_count / 100.0,  # Normalized note count
                duration_seconds / 60.0,  # Normalized duration
            ]
            
            # Pad to match embedding dimension
            while len(music_features) < 128:
                music_features.append(0.0)
            
            music_vector_json = json.dumps(music_features[:128])
            
            # Store in DuckDB
            if self.duckdb.conn:
                import random
                import time
                record_id = int(time.time() % 1000000) + random.randint(1, 999)
                
                self.duckdb.conn.execute("""
                    INSERT INTO music_analytics (
                        id, route_id, tempo, key, scale, duration_seconds,
                        note_count, harmony_complexity, genre, embedding_vector, 
                        music_vector, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    record_id, route_id, tempo, key, scale, duration_seconds,
                    note_count, harmony_complexity, genre, music_vector_json,
                    music_vector_json, datetime.utcnow()
                ])
                
                logger.info(f"✅ Synced music vector for route {route_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error syncing music vector: {e}")
            return False


# Global instance
_realtime_sync = None

def get_realtime_vector_sync() -> RealtimeVectorSync:
    """Get the global real-time vector sync instance"""
    global _realtime_sync
    if _realtime_sync is None:
        _realtime_sync = RealtimeVectorSync()
    return _realtime_sync
