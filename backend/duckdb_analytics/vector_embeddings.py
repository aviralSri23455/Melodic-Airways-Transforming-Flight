"""
DuckDB Vector Embeddings Integration
Stores and queries vector embeddings from AI Composer, VR Experiences, and Travel Logs
"""

import duckdb
import numpy as np
import json
import os
import sys
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.core.config import settings


class DuckDBVectorStore:
    """
    Vector embedding storage and similarity search using DuckDB
    Stores embeddings from AI Composer (128D), VR Experiences (64D), and Travel Logs (32D)
    """
    
    def __init__(self, db_path: str = None):
        """Initialize DuckDB connection with vector support"""
        self.db_path = db_path or settings.DUCKDB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = duckdb.connect(self.db_path)
        self.conn.execute(f"SET memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
        self.conn.execute(f"SET threads={settings.DUCKDB_THREADS}")
        
        print(f"✅ DuckDB Vector Store connected: {self.db_path}")
        
        # Create vector embedding tables
        self._create_vector_tables()
        
        # Register custom similarity functions
        self._register_similarity_functions()
    
    def _create_vector_tables(self):
        """Create tables for storing vector embeddings"""
        
        # AI Composer embeddings (128D)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_composer_embeddings (
                id VARCHAR PRIMARY KEY,
                genre VARCHAR,
                tempo INTEGER,
                complexity FLOAT,
                duration INTEGER,
                embedding FLOAT[128],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # VR Experience embeddings (64D)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vr_experience_embeddings (
                id VARCHAR PRIMARY KEY,
                experience_type VARCHAR,
                origin VARCHAR,
                destination VARCHAR,
                duration FLOAT,
                embedding FLOAT[64],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Travel Log embeddings (32D)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS travel_log_embeddings (
                id INTEGER PRIMARY KEY,
                title VARCHAR,
                waypoint_count INTEGER,
                travel_date TIMESTAMP,
                embedding FLOAT[32],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ✅ NEW: Home/Route embeddings (96D) - Main flight routes
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS home_route_embeddings (
                id VARCHAR PRIMARY KEY,
                origin VARCHAR,
                destination VARCHAR,
                distance_km FLOAT,
                music_style VARCHAR,
                tempo INTEGER,
                note_count INTEGER,
                embedding FLOAT[96],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ✅ NEW: Wellness embeddings (48D) - Therapeutic music
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS wellness_embeddings (
                id VARCHAR PRIMARY KEY,
                theme VARCHAR,
                calm_level INTEGER,
                duration INTEGER,
                binaural_frequency FLOAT,
                embedding FLOAT[48],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ✅ NEW: Education embeddings (64D) - Learning content
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS education_embeddings (
                id VARCHAR PRIMARY KEY,
                lesson_type VARCHAR,
                difficulty VARCHAR,
                topic VARCHAR,
                interaction_count INTEGER,
                embedding FLOAT[64],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ✅ NEW: AR/VR embeddings (80D) - Immersive experiences
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS arvr_embeddings (
                id VARCHAR PRIMARY KEY,
                session_type VARCHAR,
                origin VARCHAR,
                destination VARCHAR,
                waypoint_count INTEGER,
                spatial_audio BOOLEAN,
                quality VARCHAR,
                embedding FLOAT[80],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("✅ Vector embedding tables created/verified (7 tables)")
    
    def _register_similarity_functions(self):
        """Register custom UDFs for vector similarity"""
        
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """Calculate cosine similarity between two vectors"""
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0
            
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        
        def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
            """Calculate Euclidean (L2) distance between two vectors"""
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return float('inf')
            
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            return float(np.linalg.norm(v1 - v2))
        
        # Register functions
        self.conn.create_function("cosine_similarity", cosine_similarity)
        self.conn.create_function("euclidean_distance", euclidean_distance)
        
        print("✅ Vector similarity functions registered")
    
    # ==================== AI COMPOSER EMBEDDINGS ====================
    
    def store_ai_composer_embedding(
        self,
        composition_id: str,
        embedding: List[float],
        genre: str,
        tempo: int,
        complexity: float,
        duration: int,
        metadata: Dict[str, Any] = None
    ):
        """Store AI Composer embedding in DuckDB"""
        
        if len(embedding) != 128:
            raise ValueError(f"AI Composer embedding must be 128D, got {len(embedding)}D")
        
        self.conn.execute("""
            INSERT OR REPLACE INTO ai_composer_embeddings 
            (id, genre, tempo, complexity, duration, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            composition_id,
            genre,
            tempo,
            complexity,
            duration,
            embedding,
            json.dumps(metadata or {})
        ])
        
        print(f"✅ Stored AI Composer embedding: {composition_id}")
    
    def find_similar_compositions(
        self,
        query_embedding: List[float],
        k: int = 5,
        genre_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Find similar compositions using vector similarity"""
        
        if len(query_embedding) != 128:
            raise ValueError(f"Query embedding must be 128D, got {len(query_embedding)}D")
        
        # Build query with optional genre filter
        where_clause = f"WHERE genre = '{genre_filter}'" if genre_filter else ""
        
        query = f"""
            SELECT 
                id,
                genre,
                tempo,
                complexity,
                duration,
                cosine_similarity(embedding, ?::FLOAT[128]) as similarity,
                euclidean_distance(embedding, ?::FLOAT[128]) as distance,
                metadata
            FROM ai_composer_embeddings
            {where_clause}
            ORDER BY similarity DESC
            LIMIT ?
        """
        
        results = self.conn.execute(query, [query_embedding, query_embedding, k]).fetchall()
        
        return [
            {
                "id": row[0],
                "genre": row[1],
                "tempo": row[2],
                "complexity": row[3],
                "duration": row[4],
                "similarity": float(row[5]),
                "distance": float(row[6]),
                "metadata": json.loads(row[7]) if row[7] else {}
            }
            for row in results
        ]
    
    # ==================== VR EXPERIENCE EMBEDDINGS ====================
    
    def store_vr_experience_embedding(
        self,
        experience_id: str,
        embedding: List[float],
        experience_type: str,
        origin: str,
        destination: str,
        duration: float,
        metadata: Dict[str, Any] = None
    ):
        """Store VR Experience embedding in DuckDB"""
        
        if len(embedding) != 64:
            raise ValueError(f"VR Experience embedding must be 64D, got {len(embedding)}D")
        
        self.conn.execute("""
            INSERT OR REPLACE INTO vr_experience_embeddings 
            (id, experience_type, origin, destination, duration, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            experience_id,
            experience_type,
            origin,
            destination,
            duration,
            embedding,
            json.dumps(metadata or {})
        ])
        
        print(f"✅ Stored VR Experience embedding: {experience_id}")
    
    def find_similar_vr_experiences(
        self,
        query_embedding: List[float],
        k: int = 5,
        experience_type_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Find similar VR experiences using vector similarity"""
        
        if len(query_embedding) != 64:
            raise ValueError(f"Query embedding must be 64D, got {len(query_embedding)}D")
        
        where_clause = f"WHERE experience_type = '{experience_type_filter}'" if experience_type_filter else ""
        
        query = f"""
            SELECT 
                id,
                experience_type,
                origin,
                destination,
                duration,
                cosine_similarity(embedding, ?::FLOAT[64]) as similarity,
                euclidean_distance(embedding, ?::FLOAT[64]) as distance,
                metadata
            FROM vr_experience_embeddings
            {where_clause}
            ORDER BY similarity DESC
            LIMIT ?
        """
        
        results = self.conn.execute(query, [query_embedding, query_embedding, k]).fetchall()
        
        return [
            {
                "id": row[0],
                "experience_type": row[1],
                "origin": row[2],
                "destination": row[3],
                "duration": row[4],
                "similarity": float(row[5]),
                "distance": float(row[6]),
                "metadata": json.loads(row[7]) if row[7] else {}
            }
            for row in results
        ]
    
    # ==================== TRAVEL LOG EMBEDDINGS ====================
    
    def store_travel_log_embedding(
        self,
        log_id: int,
        embedding: List[float],
        title: str,
        waypoint_count: int,
        travel_date: datetime,
        metadata: Dict[str, Any] = None
    ):
        """Store Travel Log embedding in DuckDB"""
        
        if len(embedding) != 32:
            raise ValueError(f"Travel Log embedding must be 32D, got {len(embedding)}D")
        
        self.conn.execute("""
            INSERT OR REPLACE INTO travel_log_embeddings 
            (id, title, waypoint_count, travel_date, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            log_id,
            title,
            waypoint_count,
            travel_date,
            embedding,
            json.dumps(metadata or {})
        ])
        
        print(f"✅ Stored Travel Log embedding: {log_id}")
    
    def find_similar_travel_logs(
        self,
        query_embedding: List[float],
        k: int = 5,
        min_waypoints: int = None
    ) -> List[Dict[str, Any]]:
        """Find similar travel logs using vector similarity"""
        
        if len(query_embedding) != 32:
            raise ValueError(f"Query embedding must be 32D, got {len(query_embedding)}D")
        
        where_clause = f"WHERE waypoint_count >= {min_waypoints}" if min_waypoints else ""
        
        query = f"""
            SELECT 
                id,
                title,
                waypoint_count,
                travel_date,
                cosine_similarity(embedding, ?::FLOAT[32]) as similarity,
                euclidean_distance(embedding, ?::FLOAT[32]) as distance,
                metadata
            FROM travel_log_embeddings
            {where_clause}
            ORDER BY similarity DESC
            LIMIT ?
        """
        
        results = self.conn.execute(query, [query_embedding, query_embedding, k]).fetchall()
        
        return [
            {
                "id": row[0],
                "title": row[1],
                "waypoint_count": row[2],
                "travel_date": row[3],
                "similarity": float(row[4]),
                "distance": float(row[5]),
                "metadata": json.loads(row[6]) if row[6] else {}
            }
            for row in results
        ]
    
    # ==================== NEW ENDPOINTS ====================
    
    def store_home_route_embedding(
        self,
        route_id: str,
        embedding: List[float],
        origin: str,
        destination: str,
        distance_km: float,
        music_style: str,
        tempo: int,
        note_count: int,
        metadata: Dict[str, Any] = None
    ):
        """Store Home/Route embedding in DuckDB (96D)"""
        
        if len(embedding) != 96:
            raise ValueError(f"Home route embedding must be 96D, got {len(embedding)}D")
        
        self.conn.execute("""
            INSERT OR REPLACE INTO home_route_embeddings 
            (id, origin, destination, distance_km, music_style, tempo, note_count, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            route_id,
            origin,
            destination,
            distance_km,
            music_style,
            tempo,
            note_count,
            embedding,
            json.dumps(metadata or {})
        ])
        
        print(f"✅ Stored Home Route embedding: {origin} → {destination}")
    
    def store_wellness_embedding(
        self,
        wellness_id: str,
        embedding: List[float],
        theme: str,
        calm_level: int,
        duration: int,
        binaural_frequency: float = None,
        metadata: Dict[str, Any] = None
    ):
        """Store Wellness embedding in DuckDB (48D)"""
        
        if len(embedding) != 48:
            raise ValueError(f"Wellness embedding must be 48D, got {len(embedding)}D")
        
        self.conn.execute("""
            INSERT OR REPLACE INTO wellness_embeddings 
            (id, theme, calm_level, duration, binaural_frequency, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            wellness_id,
            theme,
            calm_level,
            duration,
            binaural_frequency,
            embedding,
            json.dumps(metadata or {})
        ])
        
        print(f"✅ Stored Wellness embedding: {theme} (calm: {calm_level})")
    
    def store_education_embedding(
        self,
        lesson_id: str,
        embedding: List[float],
        lesson_type: str,
        difficulty: str,
        topic: str,
        interaction_count: int,
        metadata: Dict[str, Any] = None
    ):
        """Store Education embedding in DuckDB (64D)"""
        
        if len(embedding) != 64:
            raise ValueError(f"Education embedding must be 64D, got {len(embedding)}D")
        
        self.conn.execute("""
            INSERT OR REPLACE INTO education_embeddings 
            (id, lesson_type, difficulty, topic, interaction_count, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            lesson_id,
            lesson_type,
            difficulty,
            topic,
            interaction_count,
            embedding,
            json.dumps(metadata or {})
        ])
        
        print(f"✅ Stored Education embedding: {lesson_type} ({difficulty})")
    
    def store_arvr_embedding(
        self,
        session_id: str,
        embedding: List[float],
        session_type: str,
        origin: str,
        destination: str,
        waypoint_count: int,
        spatial_audio: bool,
        quality: str,
        metadata: Dict[str, Any] = None
    ):
        """Store AR/VR embedding in DuckDB (80D)"""
        
        if len(embedding) != 80:
            raise ValueError(f"AR/VR embedding must be 80D, got {len(embedding)}D")
        
        self.conn.execute("""
            INSERT OR REPLACE INTO arvr_embeddings 
            (id, session_type, origin, destination, waypoint_count, spatial_audio, quality, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            session_id,
            session_type,
            origin,
            destination,
            waypoint_count,
            spatial_audio,
            quality,
            embedding,
            json.dumps(metadata or {})
        ])
        
        print(f"✅ Stored AR/VR embedding: {origin} → {destination} ({quality})")
    
    # ==================== ANALYTICS ====================
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        
        stats = {}
        
        # AI Composer stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT genre) as unique_genres,
                AVG(tempo) as avg_tempo,
                AVG(complexity) as avg_complexity
            FROM ai_composer_embeddings
        """).fetchone()
        
        stats["ai_composer"] = {
            "total_embeddings": result[0],
            "unique_genres": result[1],
            "avg_tempo": float(result[2]) if result[2] else 0,
            "avg_complexity": float(result[3]) if result[3] else 0
        }
        
        # VR Experience stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT experience_type) as unique_types,
                COUNT(DISTINCT origin) as unique_origins,
                AVG(duration) as avg_duration
            FROM vr_experience_embeddings
        """).fetchone()
        
        stats["vr_experiences"] = {
            "total_embeddings": result[0],
            "unique_types": result[1],
            "unique_origins": result[2],
            "avg_duration": float(result[3]) if result[3] else 0
        }
        
        # Travel Log stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(waypoint_count) as avg_waypoints,
                MIN(travel_date) as earliest_date,
                MAX(travel_date) as latest_date
            FROM travel_log_embeddings
        """).fetchone()
        
        stats["travel_logs"] = {
            "total_embeddings": result[0],
            "avg_waypoints": float(result[1]) if result[1] else 0,
            "earliest_date": str(result[2]) if result[2] else None,
            "latest_date": str(result[3]) if result[3] else None
        }
        
        # ✅ NEW: Home Route stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT origin) as unique_origins,
                COUNT(DISTINCT destination) as unique_destinations,
                AVG(distance_km) as avg_distance,
                AVG(tempo) as avg_tempo,
                AVG(note_count) as avg_notes
            FROM home_route_embeddings
        """).fetchone()
        
        stats["home_routes"] = {
            "total_embeddings": result[0],
            "unique_origins": result[1],
            "unique_destinations": result[2],
            "avg_distance_km": float(result[3]) if result[3] else 0,
            "avg_tempo": float(result[4]) if result[4] else 0,
            "avg_note_count": float(result[5]) if result[5] else 0
        }
        
        # ✅ NEW: Wellness stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT theme) as unique_themes,
                AVG(calm_level) as avg_calm_level,
                AVG(duration) as avg_duration
            FROM wellness_embeddings
        """).fetchone()
        
        stats["wellness"] = {
            "total_embeddings": result[0],
            "unique_themes": result[1],
            "avg_calm_level": float(result[2]) if result[2] else 0,
            "avg_duration": float(result[3]) if result[3] else 0
        }
        
        # ✅ NEW: Education stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT lesson_type) as unique_lesson_types,
                COUNT(DISTINCT difficulty) as unique_difficulties,
                AVG(interaction_count) as avg_interactions
            FROM education_embeddings
        """).fetchone()
        
        stats["education"] = {
            "total_embeddings": result[0],
            "unique_lesson_types": result[1],
            "unique_difficulties": result[2],
            "avg_interactions": float(result[3]) if result[3] else 0
        }
        
        # ✅ NEW: AR/VR stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT session_type) as unique_session_types,
                COUNT(DISTINCT quality) as unique_qualities,
                AVG(waypoint_count) as avg_waypoints,
                SUM(CASE WHEN spatial_audio THEN 1 ELSE 0 END) as spatial_audio_count
            FROM arvr_embeddings
        """).fetchone()
        
        stats["arvr"] = {
            "total_embeddings": result[0],
            "unique_session_types": result[1],
            "unique_qualities": result[2],
            "avg_waypoints": float(result[3]) if result[3] else 0,
            "spatial_audio_sessions": result[4]
        }
        
        return stats
    
    def analyze_genre_clusters(self) -> List[Dict[str, Any]]:
        """Analyze AI Composer embeddings by genre"""
        
        results = self.conn.execute("""
            SELECT 
                genre,
                COUNT(*) as count,
                AVG(tempo) as avg_tempo,
                AVG(complexity) as avg_complexity,
                AVG(duration) as avg_duration
            FROM ai_composer_embeddings
            GROUP BY genre
            ORDER BY count DESC
        """).fetchall()
        
        return [
            {
                "genre": row[0],
                "count": row[1],
                "avg_tempo": float(row[2]),
                "avg_complexity": float(row[3]),
                "avg_duration": float(row[4])
            }
            for row in results
        ]
    
    def analyze_vr_routes(self) -> List[Dict[str, Any]]:
        """Analyze VR Experience embeddings by route"""
        
        results = self.conn.execute("""
            SELECT 
                origin,
                destination,
                COUNT(*) as count,
                AVG(duration) as avg_duration,
                experience_type
            FROM vr_experience_embeddings
            GROUP BY origin, destination, experience_type
            ORDER BY count DESC
            LIMIT 20
        """).fetchall()
        
        return [
            {
                "origin": row[0],
                "destination": row[1],
                "count": row[2],
                "avg_duration": float(row[3]),
                "experience_type": row[4]
            }
            for row in results
        ]
    
    def export_embeddings_to_csv(self, output_dir: str = "./vector_exports"):
        """Export embeddings to CSV for analysis"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export AI Composer embeddings
        self.conn.execute(f"""
            COPY (
                SELECT id, genre, tempo, complexity, duration, created_at
                FROM ai_composer_embeddings
                ORDER BY created_at DESC
            ) TO '{output_dir}/ai_composer_embeddings.csv' (HEADER, DELIMITER ',')
        """)
        print(f"✅ Exported: {output_dir}/ai_composer_embeddings.csv")
        
        # Export VR Experience embeddings
        self.conn.execute(f"""
            COPY (
                SELECT id, experience_type, origin, destination, duration, created_at
                FROM vr_experience_embeddings
                ORDER BY created_at DESC
            ) TO '{output_dir}/vr_experience_embeddings.csv' (HEADER, DELIMITER ',')
        """)
        print(f"✅ Exported: {output_dir}/vr_experience_embeddings.csv")
        
        # Export Travel Log embeddings
        self.conn.execute(f"""
            COPY (
                SELECT id, title, waypoint_count, travel_date, created_at
                FROM travel_log_embeddings
                ORDER BY created_at DESC
            ) TO '{output_dir}/travel_log_embeddings.csv' (HEADER, DELIMITER ',')
        """)
        print(f"✅ Exported: {output_dir}/travel_log_embeddings.csv")
        
        print(f"\n✅ All vector embeddings exported to: {output_dir}/")
    
    def generate_vector_report(self):
        """Generate comprehensive vector embedding report"""
        
        print("\n" + "=" * 70)
        print("🔍 DUCKDB VECTOR EMBEDDINGS REPORT")
        print("=" * 70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Statistics
        print("\n📊 EMBEDDING STATISTICS")
        print("-" * 70)
        stats = self.get_embedding_statistics()
        
        print(f"\n🏠 Home Routes (96D Embeddings):")
        print(f"  Total: {stats['home_routes']['total_embeddings']:,}")
        print(f"  Unique Origins: {stats['home_routes']['unique_origins']}")
        print(f"  Unique Destinations: {stats['home_routes']['unique_destinations']}")
        print(f"  Avg Distance: {stats['home_routes']['avg_distance_km']:.1f} km")
        print(f"  Avg Tempo: {stats['home_routes']['avg_tempo']:.1f} BPM")
        print(f"  Avg Notes: {stats['home_routes']['avg_note_count']:.1f}")
        
        print(f"\n🎵 AI Composer (128D Embeddings):")
        print(f"  Total: {stats['ai_composer']['total_embeddings']:,}")
        print(f"  Unique Genres: {stats['ai_composer']['unique_genres']}")
        print(f"  Avg Tempo: {stats['ai_composer']['avg_tempo']:.1f} BPM")
        print(f"  Avg Complexity: {stats['ai_composer']['avg_complexity']:.2f}")
        
        print(f"\n💆 Wellness (48D Embeddings):")
        print(f"  Total: {stats['wellness']['total_embeddings']:,}")
        print(f"  Unique Themes: {stats['wellness']['unique_themes']}")
        print(f"  Avg Calm Level: {stats['wellness']['avg_calm_level']:.1f}")
        print(f"  Avg Duration: {stats['wellness']['avg_duration']:.1f}s")
        
        print(f"\n📚 Education (64D Embeddings):")
        print(f"  Total: {stats['education']['total_embeddings']:,}")
        print(f"  Unique Lesson Types: {stats['education']['unique_lesson_types']}")
        print(f"  Unique Difficulties: {stats['education']['unique_difficulties']}")
        print(f"  Avg Interactions: {stats['education']['avg_interactions']:.1f}")
        
        print(f"\n🥽 AR/VR (80D Embeddings):")
        print(f"  Total: {stats['arvr']['total_embeddings']:,}")
        print(f"  Unique Session Types: {stats['arvr']['unique_session_types']}")
        print(f"  Unique Qualities: {stats['arvr']['unique_qualities']}")
        print(f"  Avg Waypoints: {stats['arvr']['avg_waypoints']:.1f}")
        print(f"  Spatial Audio Sessions: {stats['arvr']['spatial_audio_sessions']}")
        
        print(f"\n🎮 VR Experiences (64D Embeddings):")
        print(f"  Total: {stats['vr_experiences']['total_embeddings']:,}")
        print(f"  Unique Types: {stats['vr_experiences']['unique_types']}")
        print(f"  Unique Origins: {stats['vr_experiences']['unique_origins']}")
        print(f"  Avg Duration: {stats['vr_experiences']['avg_duration']:.1f}s")
        
        print(f"\n✈️  Travel Logs (32D Embeddings):")
        print(f"  Total: {stats['travel_logs']['total_embeddings']:,}")
        print(f"  Avg Waypoints: {stats['travel_logs']['avg_waypoints']:.1f}")
        if stats['travel_logs']['earliest_date']:
            print(f"  Date Range: {stats['travel_logs']['earliest_date']} to {stats['travel_logs']['latest_date']}")
        
        # Genre clusters
        if stats['ai_composer']['total_embeddings'] > 0:
            print("\n🎼 GENRE CLUSTERS")
            print("-" * 70)
            clusters = self.analyze_genre_clusters()
            for cluster in clusters[:10]:
                print(f"  {cluster['genre']:15s} {cluster['count']:4d} compositions "
                      f"(tempo: {cluster['avg_tempo']:.0f}, complexity: {cluster['avg_complexity']:.2f})")
        
        # VR routes
        if stats['vr_experiences']['total_embeddings'] > 0:
            print("\n🛫 TOP VR ROUTES")
            print("-" * 70)
            routes = self.analyze_vr_routes()
            for route in routes[:10]:
                print(f"  {route['origin']} → {route['destination']:4s} "
                      f"({route['experience_type']:12s}) {route['count']:3d} experiences")
        
        # ✅ NEW: Top Home Routes
        if stats['home_routes']['total_embeddings'] > 0:
            print("\n🏠 TOP HOME ROUTES")
            print("-" * 70)
            home_routes = self.conn.execute("""
                SELECT origin, destination, COUNT(*) as count, AVG(distance_km) as avg_distance
                FROM home_route_embeddings
                GROUP BY origin, destination
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()
            for route in home_routes:
                print(f"  {route[0]} → {route[1]:4s} {route[2]:3d} generations ({route[3]:.0f} km)")
        
        # ✅ NEW: Wellness Themes
        if stats['wellness']['total_embeddings'] > 0:
            print("\n💆 WELLNESS THEMES")
            print("-" * 70)
            wellness_themes = self.conn.execute("""
                SELECT theme, COUNT(*) as count, AVG(calm_level) as avg_calm
                FROM wellness_embeddings
                GROUP BY theme
                ORDER BY count DESC
            """).fetchall()
            for theme in wellness_themes:
                print(f"  {theme[0]:15s} {theme[1]:3d} sessions (calm: {theme[2]:.1f})")
        
        print("\n" + "=" * 70)
        print(f"✅ Vector Embedding Report Complete! (7 tables)")
        print("=" * 70)
    
    def close(self):
        """Close DuckDB connection"""
        self.conn.close()
        print("\n✅ DuckDB Vector Store closed")


def main():
    """Main execution - generate vector embedding report"""
    print("\n" + "=" * 70)
    print("🦆 DUCKDB VECTOR EMBEDDINGS")
    print("=" * 70)
    
    vector_store = DuckDBVectorStore()
    vector_store.generate_vector_report()
    vector_store.export_embeddings_to_csv()
    vector_store.close()


if __name__ == "__main__":
    main()
