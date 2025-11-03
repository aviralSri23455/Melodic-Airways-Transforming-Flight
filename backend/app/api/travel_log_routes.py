"""
Travel Log API Routes - User-generated datasets for personal travel experiences
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.core.security import get_current_user
from app.models.models import User, Airport
from app.services.travel_log_service import TravelLogService
from pydantic import BaseModel


router = APIRouter()
travel_log_service = TravelLogService()


class WaypointCreate(BaseModel):
    airport_code: str
    timestamp: Optional[str] = None
    notes: Optional[str] = None


class TravelLogCreate(BaseModel):
    title: str
    description: Optional[str] = None
    waypoints: List[dict]
    travel_date: Optional[str] = None
    tags: Optional[List[str]] = None


class TravelLogResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    waypoints: List[dict]
    travel_date: str
    tags: List[str]
    is_public: bool
    created_at: str


@router.post("/travel-logs", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_travel_log(
    travel_log_data: TravelLogCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new travel log with multiple waypoints
    
    Example waypoints:
    [
        {"airport_code": "JFK", "timestamp": "2025-01-15T10:00:00", "notes": "Departure"},
        {"airport_code": "LHR", "timestamp": "2025-01-15T22:00:00", "notes": "Layover"},
        {"airport_code": "DXB", "timestamp": "2025-01-16T08:00:00", "notes": "Arrival"}
    ]
    """
    try:
        travel_date = None
        if travel_log_data.travel_date:
            travel_date = datetime.fromisoformat(travel_log_data.travel_date.replace('Z', '+00:00'))
        
        # Use demo user ID (1) for now - in production, use actual authentication
        result = await travel_log_service.create_travel_log(
            db=db,
            user_id=1,  # Demo user
            title=travel_log_data.title,
            description=travel_log_data.description,
            waypoints=travel_log_data.waypoints,
            travel_date=travel_date,
            tags=travel_log_data.tags
        )
        
        # ✅ Sync Travel Log to DuckDB (32D embeddings)
        try:
            from app.services.duckdb_sync_service import duckdb_sync
            import numpy as np
            import logging
            
            logger = logging.getLogger(__name__)
            
            # Generate 32D embedding for travel log
            embedding_32d = np.random.randn(32).tolist()  # In production, use real embedding model
            
            travel_log_data_dict = {
                "vector_embedding": embedding_32d,
                "id": result.get("id", 0),
                "title": travel_log_data.title,
                "waypoints": travel_log_data.waypoints,
                "travel_date": travel_date.isoformat() if travel_date else None,
                "description": travel_log_data.description,
                "tags": travel_log_data.tags,
                "is_public": False
            }
            
            duckdb_sync.sync_travel_log_embedding(travel_log_data_dict)
            logger.info(f"✅ Synced Travel Log to DuckDB: {travel_log_data.title}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not sync Travel Log to DuckDB: {e}")
        
        return {
            "success": True,
            "data": result,
            "message": "Travel log created successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create travel log: {str(e)}"
        )


@router.get("/travel-logs/my", response_model=dict)
async def get_my_travel_logs(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get all travel logs for the current user"""
    try:
        # Use demo user ID (1) for now - in production, use actual authentication
        logs = await travel_log_service.get_user_travel_logs(
            db=db,
            user_id=1,  # Demo user
            skip=skip,
            limit=limit
        )
        
        return {
            "success": True,
            "data": logs,
            "count": len(logs)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch travel logs: {str(e)}"
        )


@router.get("/travel-logs/public", response_model=dict)
async def get_public_travel_logs(
    skip: int = 0,
    limit: int = 20,
    tags: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get public travel logs, optionally filtered by tags"""
    try:
        tag_list = tags.split(',') if tags else None
        
        logs = await travel_log_service.get_public_travel_logs(
            db=db,
            skip=skip,
            limit=limit,
            tags=tag_list
        )
        
        return {
            "success": True,
            "data": logs,
            "count": len(logs)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch public travel logs: {str(e)}"
        )


@router.post("/travel-logs/{travel_log_id}/convert-to-music", response_model=dict)
async def convert_travel_log_to_music(
    travel_log_id: int,
    music_style: str = "ambient",
    tempo: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Convert a travel log into a musical composition
    Creates a multi-segment composition based on waypoints
    """
    try:
        # Use demo user ID (1) for now - in production, use actual authentication
        composition = await travel_log_service.convert_travel_log_to_music(
            db=db,
            travel_log_id=travel_log_id,
            user_id=1,  # Demo user
            music_style=music_style,
            tempo_override=tempo
        )
        
        return {
            "success": True,
            "data": composition,
            "message": "Travel log converted to music successfully"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to convert travel log: {str(e)}"
        )


@router.delete("/travel-logs/{travel_log_id}", response_model=dict)
async def delete_travel_log(
    travel_log_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a travel log"""
    try:
        # Use demo user ID (1) for now - in production, use actual authentication
        result = await travel_log_service.delete_travel_log(
            db=db,
            travel_log_id=travel_log_id,
            user_id=1  # Demo user
        )
        
        return {
            "success": True,
            "message": "Travel log deleted successfully"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete travel log: {str(e)}"
        )


@router.patch("/travel-logs/{travel_log_id}/share", response_model=dict)
async def share_travel_log(
    travel_log_id: int,
    is_public: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Make a travel log public or private"""
    try:
        # Use demo user ID (1) for now - in production, use actual authentication
        result = await travel_log_service.share_travel_log(
            db=db,
            travel_log_id=travel_log_id,
            user_id=1,  # Demo user
            is_public=is_public
        )
        
        return {
            "success": True,
            "data": result
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update travel log: {str(e)}"
        )


@router.get("/airports/search", response_model=dict)
async def search_airports(
    q: str = Query(..., min_length=1, description="Search query (airport code, name, or city)"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search for airports by code, name, or city
    Returns autocomplete suggestions for travel log waypoints
    
    Example: /airports/search?q=JFK
    Example: /airports/search?q=London
    """
    try:
        search_term = f"%{q.upper()}%"
        
        # Search in IATA code, airport name, and city
        query = select(Airport).where(
            or_(
                Airport.iata_code.ilike(search_term),
                Airport.name.ilike(search_term),
                Airport.city.ilike(search_term)
            )
        ).limit(limit)
        
        result = await db.execute(query)
        airports = result.scalars().all()
        
        # Format results for autocomplete
        suggestions = [
            {
                "code": airport.iata_code,
                "name": airport.name,
                "city": airport.city,
                "country": airport.country,
                "label": f"{airport.iata_code} - {airport.name} ({airport.city}, {airport.country})"
            }
            for airport in airports
            if airport.iata_code  # Only include airports with IATA codes
        ]
        
        return {
            "success": True,
            "data": suggestions,
            "count": len(suggestions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search airports: {str(e)}"
        )



@router.post("/travel-logs/similar", response_model=dict)
async def find_similar_travel_logs(travel_log: dict):
    """
    Find similar travel logs using vector similarity search
    
    Example request:
    {
        "title": "European Adventure",
        "waypoints": [
            {"airport_code": "JFK"},
            {"airport_code": "LHR"},
            {"airport_code": "CDG"}
        ],
        "travel_date": "2025-06-15"
    }
    """
    try:
        similar = travel_log_service.find_similar_travel_logs(
            query_log=travel_log,
            k=5
        )
        
        return {
            "success": True,
            "data": {
                "similar_logs": similar,
                "total_indexed": travel_log_service.faiss_index.ntotal,
                "search_method": "FAISS vector similarity (L2 distance)"
            },
            "message": f"Found {len(similar)} similar travel logs"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar travel logs: {str(e)}"
        )


@router.get("/travel-logs/index-stats", response_model=dict)
async def get_travel_log_index_statistics():
    """Get statistics about the travel log FAISS index"""
    try:
        import faiss
        
        return {
            "success": True,
            "data": {
                "total_logs": travel_log_service.faiss_index.ntotal,
                "embedding_dimension": travel_log_service.embedding_dim,
                "index_type": "FAISS IndexFlatL2",
                "metadata_count": len(travel_log_service.log_metadata),
                "recent_logs": travel_log_service.log_metadata[-5:] if travel_log_service.log_metadata else [],
                "faiss_version": faiss.__version__
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get travel log index stats: {str(e)}"
        )
