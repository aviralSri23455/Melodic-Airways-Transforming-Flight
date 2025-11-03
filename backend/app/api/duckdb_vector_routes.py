"""
DuckDB Vector Embeddings API Routes
Query and analyze vector embeddings stored in DuckDB
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional
from pydantic import BaseModel

router = APIRouter()


@router.get("/duckdb/vector-stats", response_model=dict)
async def get_vector_statistics():
    """Get statistics about vector embeddings stored in DuckDB"""
    try:
        from app.services.duckdb_sync_service import duckdb_sync
        
        stats = duckdb_sync.get_statistics()
        
        return {
            "success": True,
            "data": stats,
            "message": "DuckDB vector statistics retrieved"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector statistics: {str(e)}"
        )


@router.get("/duckdb/vector-report", response_model=dict)
async def generate_vector_report():
    """Generate comprehensive vector embedding report from DuckDB"""
    try:
        from app.services.duckdb_sync_service import duckdb_sync
        
        if not duckdb_sync.enabled or not duckdb_sync.vector_store:
            return {
                "success": False,
                "message": "DuckDB vector store not available"
            }
        
        # Get statistics
        stats = duckdb_sync.vector_store.get_embedding_statistics()
        
        # Get genre clusters
        genre_clusters = []
        if stats["ai_composer"]["total_embeddings"] > 0:
            genre_clusters = duckdb_sync.vector_store.analyze_genre_clusters()
        
        # Get VR routes
        vr_routes = []
        if stats["vr_experiences"]["total_embeddings"] > 0:
            vr_routes = duckdb_sync.vector_store.analyze_vr_routes()
        
        return {
            "success": True,
            "data": {
                "statistics": stats,
                "genre_clusters": genre_clusters,
                "vr_routes": vr_routes
            },
            "message": "Vector embedding report generated"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate vector report: {str(e)}"
        )


class SimilaritySearchRequest(BaseModel):
    embedding: list
    k: int = 5
    filter: Optional[str] = None


@router.post("/duckdb/search-similar-compositions", response_model=dict)
async def search_similar_compositions(request: SimilaritySearchRequest):
    """Search for similar compositions in DuckDB using vector similarity"""
    try:
        from app.services.duckdb_sync_service import duckdb_sync
        
        if not duckdb_sync.enabled or not duckdb_sync.vector_store:
            return {
                "success": False,
                "message": "DuckDB vector store not available"
            }
        
        if len(request.embedding) != 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="AI Composer embedding must be 128D"
            )
        
        results = duckdb_sync.vector_store.find_similar_compositions(
            query_embedding=request.embedding,
            k=request.k,
            genre_filter=request.filter
        )
        
        return {
            "success": True,
            "data": {
                "similar_compositions": results,
                "search_method": "DuckDB cosine similarity",
                "embedding_dimension": 128
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search similar compositions: {str(e)}"
        )


@router.post("/duckdb/search-similar-vr-experiences", response_model=dict)
async def search_similar_vr_experiences(request: SimilaritySearchRequest):
    """Search for similar VR experiences in DuckDB using vector similarity"""
    try:
        from app.services.duckdb_sync_service import duckdb_sync
        
        if not duckdb_sync.enabled or not duckdb_sync.vector_store:
            return {
                "success": False,
                "message": "DuckDB vector store not available"
            }
        
        if len(request.embedding) != 64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="VR Experience embedding must be 64D"
            )
        
        results = duckdb_sync.vector_store.find_similar_vr_experiences(
            query_embedding=request.embedding,
            k=request.k,
            experience_type_filter=request.filter
        )
        
        return {
            "success": True,
            "data": {
                "similar_experiences": results,
                "search_method": "DuckDB cosine similarity",
                "embedding_dimension": 64
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search similar VR experiences: {str(e)}"
        )


@router.post("/duckdb/search-similar-travel-logs", response_model=dict)
async def search_similar_travel_logs(request: SimilaritySearchRequest):
    """Search for similar travel logs in DuckDB using vector similarity"""
    try:
        from app.services.duckdb_sync_service import duckdb_sync
        
        if not duckdb_sync.enabled or not duckdb_sync.vector_store:
            return {
                "success": False,
                "message": "DuckDB vector store not available"
            }
        
        if len(request.embedding) != 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Travel Log embedding must be 32D"
            )
        
        min_waypoints = int(request.filter) if request.filter and request.filter.isdigit() else None
        
        results = duckdb_sync.vector_store.find_similar_travel_logs(
            query_embedding=request.embedding,
            k=request.k,
            min_waypoints=min_waypoints
        )
        
        return {
            "success": True,
            "data": {
                "similar_logs": results,
                "search_method": "DuckDB cosine similarity",
                "embedding_dimension": 32
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search similar travel logs: {str(e)}"
        )


@router.get("/duckdb/info", response_model=dict)
async def get_duckdb_info():
    """Get information about DuckDB vector embedding integration"""
    try:
        from app.services.duckdb_sync_service import duckdb_sync
        
        return {
            "success": True,
            "data": {
                "enabled": duckdb_sync.enabled,
                "features": [
                    "Vector embedding storage in DuckDB",
                    "Cosine similarity search",
                    "Euclidean distance calculation",
                    "Genre clustering analysis",
                    "VR route analytics",
                    "Travel log pattern analysis",
                    "CSV export for external analysis"
                ],
                "embedding_types": {
                    "ai_composer": {
                        "dimension": 128,
                        "table": "ai_composer_embeddings"
                    },
                    "vr_experiences": {
                        "dimension": 64,
                        "table": "vr_experience_embeddings"
                    },
                    "travel_logs": {
                        "dimension": 32,
                        "table": "travel_log_embeddings"
                    }
                },
                "similarity_functions": [
                    "cosine_similarity(vec1, vec2)",
                    "euclidean_distance(vec1, vec2)"
                ]
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get DuckDB info: {str(e)}"
        )
