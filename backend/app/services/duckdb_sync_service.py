"""
DuckDB Sync Service - Automatically sync vector embeddings to DuckDB
Runs in background to sync embeddings from FAISS to DuckDB for analytics
"""

import asyncio
from typing import Dict, Any
from datetime import datetime
import sys
import os

# Add duckdb_analytics to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'duckdb_analytics'))

try:
    from duckdb_analytics.vector_embeddings import DuckDBVectorStore
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("⚠️  DuckDB vector embeddings not available")


class DuckDBSyncService:
    """
    Service to sync vector embeddings from FAISS to DuckDB
    Enables analytics on embeddings without affecting real-time performance
    """
    
    def __init__(self):
        self.enabled = DUCKDB_AVAILABLE
        self.vector_store = None
        
        if self.enabled:
            try:
                self.vector_store = DuckDBVectorStore()
                print("✅ DuckDB Sync Service initialized")
            except Exception as e:
                print(f"⚠️  DuckDB Sync Service disabled: {e}")
                self.enabled = False
    
    def sync_ai_composer_embedding(
        self,
        composition: Dict[str, Any]
    ):
        """Sync AI Composer embedding to DuckDB"""
        if not self.enabled or not self.vector_store:
            return
        
        try:
            embedding = composition.get("vector_embedding", [])
            if not embedding or len(embedding) != 128:
                return
            
            self.vector_store.store_ai_composer_embedding(
                composition_id=composition.get("genre", "unknown") + "_" + str(int(datetime.now().timestamp())),
                embedding=embedding,
                genre=composition.get("genre", "unknown"),
                tempo=composition.get("tempo", 120),
                complexity=composition.get("complexity", 0.5),
                duration=composition.get("duration", 30),
                metadata={
                    "scale": composition.get("scale"),
                    "key": composition.get("key"),
                    "dynamics": composition.get("dynamics"),
                    "ai_generated": composition.get("ai_generated", True)
                }
            )
        except Exception as e:
            print(f"⚠️  Failed to sync AI Composer embedding to DuckDB: {e}")
    
    def sync_vr_experience_embedding(
        self,
        experience: Dict[str, Any]
    ):
        """Sync VR Experience embedding to DuckDB"""
        if not self.enabled or not self.vector_store:
            return
        
        try:
            embedding = experience.get("vector_embedding", [])
            if not embedding or len(embedding) != 64:
                return
            
            route = experience.get("route", {})
            
            self.vector_store.store_vr_experience_embedding(
                experience_id=experience.get("experience_id", "vr_" + str(int(datetime.now().timestamp()))),
                embedding=embedding,
                experience_type=experience.get("type", "immersive"),
                origin=route.get("origin", "UNKNOWN"),
                destination=route.get("destination", "UNKNOWN"),
                duration=experience.get("duration", 60.0),
                metadata={
                    "vr_ready": experience.get("vr_ready", True),
                    "ar_ready": experience.get("ar_ready", True),
                    "platforms": experience.get("platforms", [])
                }
            )
        except Exception as e:
            print(f"⚠️  Failed to sync VR Experience embedding to DuckDB: {e}")
    
    def sync_travel_log_embedding(
        self,
        travel_log: Dict[str, Any]
    ):
        """Sync Travel Log embedding to DuckDB"""
        if not self.enabled or not self.vector_store:
            return
        
        try:
            embedding = travel_log.get("vector_embedding", [])
            if not embedding or len(embedding) != 32:
                return
            
            # Parse travel date
            travel_date_str = travel_log.get("travel_date")
            if isinstance(travel_date_str, str):
                travel_date = datetime.fromisoformat(travel_date_str.replace('Z', '+00:00'))
            else:
                travel_date = travel_date_str or datetime.now()
            
            self.vector_store.store_travel_log_embedding(
                log_id=travel_log.get("id", 0),
                embedding=embedding,
                title=travel_log.get("title", "Untitled"),
                waypoint_count=len(travel_log.get("waypoints", [])),
                travel_date=travel_date,
                metadata={
                    "description": travel_log.get("description"),
                    "tags": travel_log.get("tags", []),
                    "is_public": travel_log.get("is_public", False)
                }
            )
        except Exception as e:
            print(f"⚠️  Failed to sync Travel Log embedding to DuckDB: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DuckDB embedding statistics"""
        if not self.enabled or not self.vector_store:
            return {"enabled": False}
        
        try:
            stats = self.vector_store.get_embedding_statistics()
            stats["enabled"] = True
            return stats
        except Exception as e:
            print(f"⚠️  Failed to get DuckDB statistics: {e}")
            return {"enabled": False, "error": str(e)}
    
    def close(self):
        """Close DuckDB connection"""
        if self.vector_store:
            self.vector_store.close()


# Global instance
duckdb_sync = DuckDBSyncService()
