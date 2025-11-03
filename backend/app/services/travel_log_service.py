"""
Travel Log Service - User-generated datasets for personal travel experiences
Allows users to create, manage, and convert their travel logs into musical compositions
Now with vector embeddings for similarity search
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import numpy as np
import faiss
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.models import TravelLog, User, MusicComposition
from app.services.music_generator import MusicGenerator


class TravelLogService:
    """Service for managing user travel logs and converting them to music with vector embeddings"""
    
    def __init__(self):
        self.music_generator = MusicGenerator()
        
        # Initialize FAISS index for travel log embeddings (32D)
        self.embedding_dim = 32
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.log_metadata = []
        
        print(f"✅ Travel Log Service initialized with FAISS v{faiss.__version__} vector search")
        print(f"🔍 Vector embeddings enabled for travel log similarity")
    
    async def create_travel_log(
        self,
        db: AsyncSession,
        user_id: int,
        title: str,
        description: Optional[str],
        waypoints: List[Dict[str, Any]],
        travel_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new travel log with multiple waypoints
        
        Args:
            db: Database session
            user_id: User ID
            title: Travel log title
            description: Optional description
            waypoints: List of waypoints with airport codes and timestamps
            travel_date: Date of travel
            tags: Optional tags for categorization
        
        Returns:
            Created travel log data
        """
        travel_log = TravelLog(
            user_id=user_id,
            title=title,
            description=description,
            waypoints=json.dumps(waypoints),
            travel_date=travel_date or datetime.utcnow(),
            tags=json.dumps(tags or []),
            is_public=False
        )
        
        db.add(travel_log)
        await db.commit()
        await db.refresh(travel_log)
        
        result = {
            "id": travel_log.id,
            "title": travel_log.title,
            "description": travel_log.description,
            "waypoints": json.loads(travel_log.waypoints),
            "travel_date": travel_log.travel_date.isoformat(),
            "tags": json.loads(travel_log.tags),
            "created_at": travel_log.created_at.isoformat()
        }
        
        # Generate and store vector embedding
        vector_embedding = self.generate_travel_log_embedding(result)
        result["vector_embedding"] = vector_embedding.tolist()
        
        # Add to FAISS index
        self.add_travel_log_to_index(result)
        
        # Sync to DuckDB for analytics (non-blocking)
        try:
            from app.services.duckdb_sync_service import duckdb_sync
            duckdb_sync.sync_travel_log_embedding(result)
        except Exception as e:
            pass  # Don't fail if DuckDB sync fails
        
        print(f"✅ Travel log created with 32D vector embedding")
        
        return result
    
    async def get_user_travel_logs(
        self,
        db: AsyncSession,
        user_id: int,
        skip: int = 0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all travel logs for a user"""
        result = await db.execute(
            select(TravelLog)
            .where(TravelLog.user_id == user_id)
            .order_by(TravelLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        logs = result.scalars().all()
        
        return [
            {
                "id": log.id,
                "title": log.title,
                "description": log.description,
                "waypoints": json.loads(log.waypoints),
                "travel_date": log.travel_date.isoformat(),
                "tags": json.loads(log.tags),
                "is_public": log.is_public,
                "created_at": log.created_at.isoformat()
            }
            for log in logs
        ]
    
    async def convert_travel_log_to_music(
        self,
        db: AsyncSession,
        travel_log_id: int,
        user_id: int,
        music_style: str = "ambient",
        tempo_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Convert a travel log into a musical composition
        Creates a multi-segment composition based on waypoints
        
        Args:
            db: Database session
            travel_log_id: Travel log ID
            user_id: User ID
            music_style: Musical style preference
            tempo_override: Optional tempo override
        
        Returns:
            Generated composition data with MIDI
        """
        # Get travel log
        result = await db.execute(
            select(TravelLog).where(
                and_(
                    TravelLog.id == travel_log_id,
                    TravelLog.user_id == user_id
                )
            )
        )
        travel_log = result.scalar_one_or_none()
        
        if not travel_log:
            raise ValueError("Travel log not found")
        
        waypoints = json.loads(travel_log.waypoints)
        
        # Generate music for each segment
        segments = []
        total_distance = 0
        
        for i in range(len(waypoints) - 1):
            origin = waypoints[i]
            destination = waypoints[i + 1]
            
            # Generate music for this segment
            segment_music = await self.music_generator.generate_route_music(
                origin_code=origin["airport_code"],
                destination_code=destination["airport_code"],
                music_style=music_style,
                tempo=tempo_override
            )
            
            segments.append({
                "segment": i + 1,
                "origin": origin["airport_code"],
                "destination": destination["airport_code"],
                "music": segment_music
            })
            
            total_distance += segment_music.get("distance", 0)
        
        # Combine segments into a single composition
        composition_data = {
            "travel_log_id": travel_log_id,
            "title": f"Musical Journey: {travel_log.title}",
            "segments": segments,
            "total_waypoints": len(waypoints),
            "total_distance": total_distance,
            "music_style": music_style,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return composition_data
    
    async def share_travel_log(
        self,
        db: AsyncSession,
        travel_log_id: int,
        user_id: int,
        is_public: bool = True
    ) -> Dict[str, Any]:
        """Make a travel log public or private"""
        result = await db.execute(
            select(TravelLog).where(
                and_(
                    TravelLog.id == travel_log_id,
                    TravelLog.user_id == user_id
                )
            )
        )
        travel_log = result.scalar_one_or_none()
        
        if not travel_log:
            raise ValueError("Travel log not found")
        
        travel_log.is_public = is_public
        await db.commit()
        
        return {
            "id": travel_log.id,
            "is_public": is_public,
            "message": f"Travel log is now {'public' if is_public else 'private'}"
        }
    
    async def delete_travel_log(
        self,
        db: AsyncSession,
        travel_log_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Delete a travel log"""
        result = await db.execute(
            select(TravelLog).where(
                and_(
                    TravelLog.id == travel_log_id,
                    TravelLog.user_id == user_id
                )
            )
        )
        travel_log = result.scalar_one_or_none()
        
        if not travel_log:
            raise ValueError("Travel log not found or you don't have permission to delete it")
        
        await db.delete(travel_log)
        await db.commit()
        
        return {
            "id": travel_log_id,
            "message": "Travel log deleted successfully"
        }
    
    async def get_public_travel_logs(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 20,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get public travel logs, optionally filtered by tags"""
        query = select(TravelLog).where(TravelLog.is_public == True)
        
        if tags:
            # Filter by tags (simplified - in production use proper JSON querying)
            query = query.where(TravelLog.tags.contains(json.dumps(tags[0])))
        
        query = query.order_by(TravelLog.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(query)
        logs = result.scalars().all()
        
        return [
            {
                "id": log.id,
                "title": log.title,
                "description": log.description,
                "waypoints": json.loads(log.waypoints),
                "travel_date": log.travel_date.isoformat(),
                "tags": json.loads(log.tags),
                "user_id": log.user_id,
                "created_at": log.created_at.isoformat()
            }
            for log in logs
        ]

    def generate_travel_log_embedding(
        self,
        travel_log: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate a 32D vector embedding for a travel log
        
        Args:
            travel_log: Travel log data with waypoints
        
        Returns:
            32D numpy array embedding
        """
        waypoints = travel_log.get("waypoints", [])
        
        if not waypoints or len(waypoints) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Waypoint count and complexity (4D)
        waypoint_features = [
            len(waypoints) / 10.0,  # Normalized waypoint count
            1.0 if len(waypoints) > 1 else 0.0,  # Multi-leg indicator
            1.0 if len(waypoints) > 3 else 0.0,  # Complex trip indicator
            len(waypoints) / 20.0  # Trip complexity
        ]
        
        # Geographic spread (8D)
        # Extract airport codes and calculate geographic diversity
        airport_codes = [wp.get("airport_code", "") for wp in waypoints if wp.get("airport_code")]
        unique_airports = len(set(airport_codes))
        
        # Calculate geographic diversity (simplified)
        geo_features = [
            unique_airports / 10.0,
            len(airport_codes) / 15.0,
            1.0 if unique_airports > 2 else 0.0,  # Multi-destination
            1.0 if len(waypoints) > len(set(airport_codes)) else 0.0  # Has revisits
        ]
        
        # Temporal features (4D)
        travel_date = travel_log.get("travel_date")
        if travel_date:
            try:
                if isinstance(travel_date, str):
                    date_obj = datetime.fromisoformat(travel_date.replace('Z', '+00:00'))
                else:
                    date_obj = travel_date
                
                temporal_features = [
                    date_obj.month / 12.0,  # Season
                    date_obj.weekday() / 7.0,  # Day of week
                    date_obj.year / 2030.0,  # Year (normalized)
                    1.0 if date_obj.weekday() >= 5 else 0.0  # Weekend indicator
                ]
            except:
                temporal_features = [0.5, 0.5, 0.5, 0.0]
        else:
            temporal_features = [0.5, 0.5, 0.5, 0.0]
        
        # Content features (8D)
        title = travel_log.get("title", "")
        description = travel_log.get("description", "")
        tags = travel_log.get("tags", [])
        
        content_features = [
            len(title) / 100.0,
            len(description) / 500.0 if description else 0.0,
            len(tags) / 10.0 if tags else 0.0,
            1.0 if description else 0.0,
            1.0 if tags else 0.0,
            1.0 if any(wp.get("notes") for wp in waypoints) else 0.0,  # Has notes
            sum(1 for wp in waypoints if wp.get("notes")) / len(waypoints) if waypoints else 0.0,
            1.0 if travel_log.get("is_public") else 0.0
        ]
        
        # Music composition features (8D) - if available
        music_comp = travel_log.get("music_composition")
        if music_comp:
            music_features = [
                music_comp.get("tempo", 120) / 200.0,
                music_comp.get("duration", 60) / 120.0,
                len(music_comp.get("segments", [])) / 10.0,
                1.0 if music_comp.get("genre") else 0.0
            ]
        else:
            music_features = [0.5, 0.5, 0.0, 0.0]
        
        # Combine all features
        all_features = (
            waypoint_features + geo_features + temporal_features +
            content_features + music_features
        )
        
        # Pad or truncate to exactly 32 dimensions
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        feature_count = min(len(all_features), self.embedding_dim)
        embedding[:feature_count] = all_features[:feature_count]
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def add_travel_log_to_index(
        self,
        travel_log: Dict[str, Any]
    ) -> str:
        """
        Add a travel log to the FAISS index
        
        Args:
            travel_log: Travel log data
        
        Returns:
            Travel log ID
        """
        # Generate embedding
        embedding = self.generate_travel_log_embedding(travel_log)
        
        # Add to FAISS index
        self.faiss_index.add(np.array([embedding]))
        
        # Store metadata
        log_id = travel_log.get("id", f"log_{len(self.log_metadata)}")
        metadata = {
            "id": log_id,
            "title": travel_log.get("title"),
            "waypoint_count": len(travel_log.get("waypoints", [])),
            "travel_date": travel_log.get("travel_date"),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.log_metadata.append(metadata)
        
        print(f"✅ Added travel log to FAISS index: {log_id} (total: {self.faiss_index.ntotal})")
        
        return str(log_id)
    
    def find_similar_travel_logs(
        self,
        query_log: Dict[str, Any],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar travel logs using vector similarity search
        
        Args:
            query_log: Travel log to find similar matches for
            k: Number of similar logs to return
        
        Returns:
            List of similar travel logs with similarity scores
        """
        if self.faiss_index.ntotal == 0:
            print("⚠️ Travel Log FAISS index is empty")
            return []
        
        # Generate embedding for query
        query_embedding = self.generate_travel_log_embedding(query_log)
        
        # Search FAISS index
        k = min(k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.log_metadata):
                similarity_score = 1.0 / (1.0 + distance)
                result = {
                    **self.log_metadata[idx],
                    "similarity_score": float(similarity_score),
                    "distance": float(distance),
                    "rank": i + 1
                }
                results.append(result)
        
        print(f"🔍 Found {len(results)} similar travel logs using vector search")
        
        return results
