"""
Extended API routes for datasets, collections, vector search, and WebSocket collaboration
"""

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
import json
import logging

from app.db.database import get_db
from app.models.models import (
    User, MusicComposition, UserDataset, UserCollection,
    CollaborationSession, CompositionRemix, RemixType, ActivityType
)
from app.models.schemas import UserInfo
from app.core.security import get_current_active_user
from app.services.vector_service import VectorSearchService, MusicVector, SimilarityResult
from app.services.dataset_manager import UserDatasetManager, CollectionManager, RemixManager
from app.services.websocket_manager import WebSocketManager
from app.services.activity_service import ActivityService
from app.services.cache import CacheService

logger = logging.getLogger(__name__)

router = APIRouter()
vector_service = VectorSearchService()
dataset_manager = UserDatasetManager()
collection_manager = CollectionManager()
remix_manager = RemixManager()
websocket_manager = WebSocketManager()
cache_service = CacheService()
activity_service = ActivityService(websocket_manager)


# ==================== DATASET ENDPOINTS ====================

@router.post("/datasets")
async def create_dataset(
    name: str,
    route_data: dict,
    metadata: Optional[dict] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new user dataset"""
    try:
        dataset = await dataset_manager.create_dataset(
            db, current_user.id, name, route_data, metadata
        )
        return {
            "id": dataset.id,
            "name": dataset.name,
            "created_at": dataset.created_at.isoformat(),
            "message": "Dataset created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def get_user_datasets(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all datasets for current user"""
    try:
        datasets = await dataset_manager.get_user_datasets(
            db, current_user.id, limit, offset
        )
        return {
            "datasets": [
                {
                    "id": d.id,
                    "name": d.name,
                    "created_at": d.created_at.isoformat(),
                    "metadata": d.metadata
                }
                for d in datasets
            ],
            "total": len(datasets)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific dataset"""
    try:
        dataset = await dataset_manager.get_dataset(db, dataset_id, current_user.id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        return {
            "id": dataset.id,
            "name": dataset.name,
            "route_data": dataset.route_data,
            "metadata": dataset.metadata,
            "created_at": dataset.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/datasets/{dataset_id}")
async def update_dataset(
    dataset_id: int,
    name: Optional[str] = None,
    route_data: Optional[dict] = None,
    metadata: Optional[dict] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update dataset"""
    try:
        dataset = await dataset_manager.update_dataset(
            db, dataset_id, current_user.id, name, route_data, metadata
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        return {"message": "Dataset updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete dataset"""
    try:
        success = await dataset_manager.delete_dataset(db, dataset_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")

        return {"message": "Dataset deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}/compositions")
async def get_dataset_compositions(
    dataset_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get compositions in a dataset"""
    try:
        compositions = await dataset_manager.get_dataset_compositions(
            db, dataset_id, current_user.id, limit, offset
        )
        return {
            "compositions": [
                {
                    "id": c.id,
                    "title": c.title,
                    "genre": c.genre,
                    "tempo": c.tempo,
                    "created_at": c.created_at.isoformat()
                }
                for c in compositions
            ],
            "total": len(compositions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== COLLECTION ENDPOINTS ====================

@router.post("/collections")
async def create_collection(
    name: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new collection"""
    try:
        collection = await collection_manager.create_collection(
            db, current_user.id, name, description, tags
        )
        return {
            "id": collection.id,
            "name": collection.name,
            "created_at": collection.created_at.isoformat(),
            "message": "Collection created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def get_user_collections(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all collections for current user"""
    try:
        collections = await collection_manager.get_user_collections(
            db, current_user.id, limit, offset
        )
        return {
            "collections": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "tags": c.tags,
                    "composition_count": len(c.composition_ids) if c.composition_ids else 0,
                    "created_at": c.created_at.isoformat()
                }
                for c in collections
            ],
            "total": len(collections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{collection_id}/compositions/{composition_id}")
async def add_composition_to_collection(
    collection_id: int,
    composition_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Add composition to collection"""
    try:
        success = await collection_manager.add_composition_to_collection(
            db, collection_id, composition_id, current_user.id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Collection not found")

        return {"message": "Composition added to collection"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_id}/compositions/{composition_id}")
async def remove_composition_from_collection(
    collection_id: int,
    composition_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Remove composition from collection"""
    try:
        success = await collection_manager.remove_composition_from_collection(
            db, collection_id, composition_id, current_user.id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Collection not found")

        return {"message": "Composition removed from collection"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_id}/compositions")
async def get_collection_compositions(
    collection_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all compositions in a collection"""
    try:
        compositions = await collection_manager.get_collection_compositions(
            db, collection_id, current_user.id
        )
        return {
            "compositions": [
                {
                    "id": c.id,
                    "title": c.title,
                    "genre": c.genre,
                    "tempo": c.tempo,
                    "created_at": c.created_at.isoformat()
                }
                for c in compositions
            ],
            "total": len(compositions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_id}")
async def delete_collection(
    collection_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete collection"""
    try:
        success = await collection_manager.delete_collection(
            db, collection_id, current_user.id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Collection not found")

        return {"message": "Collection deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== VECTOR SEARCH ENDPOINTS ====================

@router.post("/search/similar")
async def search_similar_compositions(
    composition_id: int,
    limit: int = Query(10, ge=1, le=100),
    genre_filter: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Find similar compositions using vector search"""
    try:
        # Get the composition
        result = await db.execute(
            select(MusicComposition).where(MusicComposition.id == composition_id)
        )
        composition = result.scalar_one_or_none()

        if not composition:
            raise HTTPException(status_code=404, detail="Composition not found")

        # Get or create vector for composition
        vector = await vector_service.get_composition_vector(db, composition_id)
        if not vector:
            vector = await vector_service.extract_features(
                composition_id,
                composition.tempo,
                composition.pitch,
                composition.harmony,
                composition.complexity_score or 0.5,
                composition.genre or "unknown"
            )
            await vector_service.store_vector(db, composition_id, vector)

        # Find similar compositions
        similar = await vector_service.find_similar(
            db, vector, limit, genre_filter
        )

        return {
            "query_composition_id": composition_id,
            "similar_compositions": [
                {
                    "composition_id": s.composition_id,
                    "title": s.title,
                    "genre": s.genre,
                    "similarity_score": s.similarity_score,
                    "distance": s.distance
                }
                for s in similar
            ],
            "total": len(similar)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/by-vector")
async def search_by_vector(
    tempo: int,
    pitch: float,
    harmony: float,
    complexity: float,
    genre: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Search for similar compositions by vector parameters"""
    try:
        vector = await vector_service.extract_features(
            0, tempo, pitch, harmony, complexity, genre or "unknown"
        )

        similar = await vector_service.find_similar(db, vector, limit, genre)

        return {
            "query_vector": vector.to_json(),
            "similar_compositions": [
                {
                    "composition_id": s.composition_id,
                    "title": s.title,
                    "genre": s.genre,
                    "similarity_score": s.similarity_score
                }
                for s in similar
            ],
            "total": len(similar)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REMIX ENDPOINTS ====================

@router.post("/remixes")
async def create_remix(
    original_composition_id: int,
    remix_composition_id: int,
    remix_type: str,
    attribution_data: Optional[dict] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a remix relationship"""
    try:
        remix_enum = RemixType(remix_type)
        remix = await remix_manager.create_remix(
            db,
            original_composition_id,
            remix_composition_id,
            remix_enum,
            attribution_data
        )
        return {
            "id": remix.id,
            "original_id": remix.original_composition_id,
            "remix_id": remix.remix_composition_id,
            "type": remix.remix_type.value,
            "created_at": remix.created_at.isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid remix type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/remixes/{composition_id}/chain")
async def get_remix_chain(
    composition_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get remix chain for a composition"""
    try:
        chain = await remix_manager.get_remix_chain(db, composition_id)
        return chain
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/remixes/{composition_id}/history")
async def get_remix_history(
    composition_id: int,
    depth: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get full remix history"""
    try:
        history = await remix_manager.get_remix_history(db, composition_id, depth)
        return {
            "composition_id": composition_id,
            "history": history,
            "total_remixes": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ACTIVITY FEED ENDPOINTS ====================

@router.get("/activities")
async def get_user_activities(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    activity_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get activity feed for current user"""
    try:
        activity_enum = None
        if activity_type:
            try:
                activity_enum = ActivityType(activity_type)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid activity type")

        activities = await activity_service.get_user_activities(
            db, current_user.id, limit, offset, activity_enum
        )

        return {
            "activities": activities,
            "total": len(activities),
            "limit": limit,
            "offset": offset
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activities/recent")
async def get_recent_activities(
    minutes: int = Query(60, ge=1, le=1440),  # Max 24 hours
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get recent activities from all users (last N minutes)"""
    try:
        activities = await activity_service.get_recent_activities(db, minutes, limit)

        return {
            "activities": activities,
            "total": len(activities),
            "period_minutes": minutes,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activities/stats")
async def get_activity_stats(
    days: int = Query(7, ge=1, le=90),  # Max 90 days
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get activity statistics for current user"""
    try:
        stats = await activity_service.get_activity_stats(db, current_user.id, days)

        return {
            "user_id": current_user.id,
            "username": current_user.username,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/activities/track")
async def track_activity(
    activity_type: str,
    target_id: Optional[int] = None,
    target_type: Optional[str] = None,
    activity_data: Optional[dict] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Manually track an activity (for custom activities)"""
    try:
        activity_enum = ActivityType(activity_type)

        success = await activity_service.log_activity(
            db=db,
            user_id=current_user.id,
            activity_type=activity_enum,
            target_id=target_id,
            target_type=target_type,
            activity_data=activity_data
        )

        if success:
            return {"message": "Activity tracked successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to track activity")

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid activity type")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== COLLABORATION ENDPOINTS ====================

@router.post("/collaborations/sessions")
async def create_collaboration_session(
    composition_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new collaboration session"""
    try:
        session = CollaborationSession(
            creator_id=current_user.id,
            composition_id=composition_id,
            is_active=1,
            participants=[current_user.id]
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)

        return {
            "session_id": session.id,
            "creator_id": session.creator_id,
            "composition_id": session.composition_id,
            "created_at": session.created_at.isoformat(),
            "message": "Collaboration session created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collaborations/sessions/{session_id}")
async def get_collaboration_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get collaboration session info"""
    try:
        result = await db.execute(
            select(CollaborationSession).where(CollaborationSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session.id,
            "creator_id": session.creator_id,
            "composition_id": session.composition_id,
            "participants": session.participants or [],
            "is_active": bool(session.is_active),
            "created_at": session.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/collaborate/{session_id}/{user_id}")
async def websocket_collaborate(
    websocket: WebSocket,
    session_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """WebSocket endpoint for real-time collaboration"""
    try:
        # Get user info
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Handle connection
        connection_id = await websocket_manager.handle_connection(
            websocket, str(session_id), user_id, user.username
        )

        # Create room if doesn't exist
        if not websocket_manager.room_manager.get_room(str(session_id)):
            websocket_manager.room_manager.create_room(str(session_id), user_id)

        # Send initial state
        session_info = websocket_manager.get_session_info(str(session_id))
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "session_info": session_info
        })

        # Listen for messages
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "state_update":
                await websocket_manager.broadcast_state_update(
                    str(session_id),
                    data.get("state", {})
                )
            elif data.get("type") == "message":
                await websocket_manager.connection_manager.broadcast_to_room(
                    str(session_id),
                    {
                        "type": "message",
                        "user_id": user_id,
                        "username": user.username,
                        "content": data.get("content")
                    }
                )

    except WebSocketDisconnect:
        await websocket_manager.handle_disconnection(connection_id, str(session_id))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=status.WS_1011_SERVER_ERROR)
        except:
            pass
