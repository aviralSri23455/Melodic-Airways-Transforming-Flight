"""
API routes for community features: forums, contests, and social interactions
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime
import logging

from app.db.database import get_db
from app.models.models import User
from app.core.security import get_current_active_user
from app.services.community_service import CommunityManager
from app.services.graph_pathfinder import RoutePathfindingService
from app.services.genre_composer import GenreComposer

logger = logging.getLogger(__name__)

router = APIRouter()
community_manager = CommunityManager()
pathfinding_service = RoutePathfindingService()
genre_composer = GenreComposer()


# ==================== FORUM ENDPOINTS ====================

@router.post("/forum/threads")
async def create_forum_thread(
    title: str,
    content: str,
    category: str,
    tags: Optional[List[str]] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new forum thread"""
    try:
        result = await community_manager.forum_service.create_thread(
            db, current_user.id, title, content, category, tags
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forum/threads")
async def get_forum_threads(
    category: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Get forum threads"""
    try:
        threads = await community_manager.forum_service.get_threads(
            db, category, limit, offset
        )
        return {'threads': threads, 'total': len(threads)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forum/threads/{thread_id}/replies")
async def post_forum_reply(
    thread_id: int,
    content: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Post a reply to a forum thread"""
    try:
        result = await community_manager.forum_service.post_reply(
            db, thread_id, current_user.id, content
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CONTEST ENDPOINTS ====================

@router.post("/contests")
async def create_contest(
    title: str,
    description: str,
    start_date: datetime,
    end_date: datetime,
    rules: dict,
    prizes: Optional[List[dict]] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new composition contest"""
    try:
        result = await community_manager.contest_service.create_contest(
            db, current_user.id, title, description,
            start_date, end_date, rules, prizes
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contests/active")
async def get_active_contests(
    db: AsyncSession = Depends(get_db)
):
    """Get currently active contests"""
    try:
        contests = await community_manager.contest_service.get_active_contests(db)
        return {'contests': contests, 'total': len(contests)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contests/{contest_id}/submit")
async def submit_to_contest(
    contest_id: int,
    composition_id: int,
    description: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit a composition to a contest"""
    try:
        result = await community_manager.contest_service.submit_to_contest(
            db, contest_id, current_user.id, composition_id, description
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contests/{contest_id}/vote")
async def vote_for_submission(
    contest_id: int,
    composition_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Vote for a contest submission"""
    try:
        result = await community_manager.contest_service.vote_for_submission(
            db, contest_id, composition_id, current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contests/{contest_id}/leaderboard")
async def get_contest_leaderboard(
    contest_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get contest leaderboard"""
    try:
        leaderboard = await community_manager.contest_service.get_contest_leaderboard(
            db, contest_id
        )
        return {'leaderboard': leaderboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SOCIAL INTERACTION ENDPOINTS ====================

@router.post("/social/follow/{user_id}")
async def follow_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Follow another user"""
    try:
        result = await community_manager.social_service.follow_user(
            db, current_user.id, user_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social/like/{composition_id}")
async def like_composition(
    composition_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Like a composition"""
    try:
        result = await community_manager.social_service.like_composition(
            db, current_user.id, composition_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social/comment/{composition_id}")
async def comment_on_composition(
    composition_id: int,
    comment: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Comment on a composition"""
    try:
        result = await community_manager.social_service.comment_on_composition(
            db, current_user.id, composition_id, comment
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/social/comments/{composition_id}")
async def get_composition_comments(
    composition_id: int,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """Get comments for a composition"""
    try:
        comments = await community_manager.social_service.get_composition_comments(
            db, composition_id, limit
        )
        return {'comments': comments, 'total': len(comments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/social/trending")
async def get_trending_compositions(
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get trending compositions"""
    try:
        trending = await community_manager.social_service.get_trending_compositions(
            db, days, limit
        )
        return {'trending': trending, 'total': len(trending)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PATHFINDING ENDPOINTS ====================

@router.post("/pathfinding/optimal-route")
async def find_optimal_route(
    origin_code: str,
    destination_code: str,
    optimize_for: str = Query("distance", regex="^(distance|duration)$"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Find optimal route using Dijkstra's algorithm"""
    try:
        route = await pathfinding_service.find_optimal_route(
            db, origin_code, destination_code, optimize_for
        )
        if not route:
            raise HTTPException(status_code=404, detail="No route found")
        return route
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pathfinding/alternative-routes")
async def find_alternative_routes(
    origin_code: str,
    destination_code: str,
    max_stops: int = Query(3, ge=1, le=5),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Find alternative routes with multiple stops"""
    try:
        routes = await pathfinding_service.find_multi_stop_routes(
            db, origin_code, destination_code, max_stops
        )
        return {'routes': routes, 'total': len(routes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pathfinding/hubs")
async def get_hub_airports(
    top_n: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get major hub airports based on connectivity"""
    try:
        hubs = await pathfinding_service.discover_hub_airports(db, top_n)
        return {'hubs': hubs, 'total': len(hubs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pathfinding/nearby/{iata_code}")
async def find_nearby_airports(
    iata_code: str,
    max_distance_km: float = Query(500, ge=1, le=2000),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Find nearby alternative airports"""
    try:
        nearby = await pathfinding_service.find_nearby_alternatives(
            db, iata_code, max_distance_km
        )
        return {'nearby_airports': nearby, 'total': len(nearby)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pathfinding/network-stats")
async def get_network_statistics(
    current_user: User = Depends(get_current_active_user)
):
    """Get flight network statistics"""
    try:
        stats = pathfinding_service.get_network_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== GENRE COMPOSITION ENDPOINTS ====================

@router.get("/genres")
async def get_available_genres():
    """Get list of available music genres"""
    try:
        genres = genre_composer.get_available_genres()
        return {'genres': genres, 'total': len(genres)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/genres/{genre}")
async def get_genre_info(genre: str):
    """Get information about a specific genre"""
    try:
        info = genre_composer.get_genre_info(genre)
        if not info:
            raise HTTPException(status_code=404, detail="Genre not found")
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genres/compose")
async def generate_genre_composition(
    genre: str,
    route_features: List[float],
    duration_seconds: int = Query(180, ge=30, le=600),
    current_user: User = Depends(get_current_active_user)
):
    """Generate a genre-specific composition"""
    try:
        import numpy as np
        
        features_array = np.array(route_features, dtype=np.float32)
        composition = genre_composer.generate_genre_composition(
            genre, features_array, duration_seconds
        )
        return composition
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== COMMUNITY STATS ENDPOINT ====================

@router.get("/community/stats")
async def get_community_stats(
    db: AsyncSession = Depends(get_db)
):
    """Get overall community statistics"""
    try:
        stats = await community_manager.get_community_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
