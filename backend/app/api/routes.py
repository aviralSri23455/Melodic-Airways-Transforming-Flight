"""
Main API router for Aero Melody
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional, Dict
import os
import shutil

from app.db.database import get_db
from app.models.models import Airport, Route, MusicComposition, User
from app.models.schemas import (
    RouteGenerateRequest, RouteGenerateResponse, AirportSearchResponse,
    RouteInfo, CompositionInfo, AnalyticsResponse, SimilarRoutesResponse,
    UserCreate, UserLogin, Token, UserInfo, RecentCompositionsResponse
)
from app.services.vector_service import VectorSearchService
from app.services.music_generator import MusicGenerationService
from app.core.security import (
    authenticate_user, create_access_token, get_current_active_user
)

router = APIRouter()
music_service = MusicGenerationService()
vector_service = VectorSearchService()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": int(__import__('time').time()),
        "version": "1.0.0"
    }


@router.get("/routes", response_model=List[RouteInfo])
async def get_all_routes(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Get all flight routes with pagination"""
    result = await db.execute(
        select(Route)
        .limit(limit)
        .offset(offset)
    )
    routes = result.scalars().all()

    route_infos = []
    for route in routes:
        route_infos.append(await get_route_info(route, db))

    return route_infos


@router.post("/generate-midi", response_model=RouteGenerateResponse)
async def generate_midi(
    request: RouteGenerateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate unique MIDI for a flight route (no auth required)"""
    try:
        composition, midi_path, analytics = await music_service.generate_music_for_route(
            db, request.origin_code, request.destination_code,
            request.music_style, request.scale, request.key,
            request.tempo or 120, request.duration_minutes or 3
        )

        # ✅ Sync home route to DuckDB using vector sync helper
        try:
            from app.services.vector_sync_helper import get_vector_sync_helper
            import logging
            
            logger = logging.getLogger(__name__)
            vector_sync = get_vector_sync_helper()
            vector_sync.sync_home_route(
                origin=request.origin_code,
                destination=request.destination_code,
                distance_km=analytics.get("distance_km", 0),
                music_style=request.music_style.value if hasattr(request.music_style, 'value') else str(request.music_style),
                tempo=request.tempo or 120,
                note_count=analytics.get("note_count", 0),
                duration=(request.duration_minutes or 3) * 60,
                metadata={
                    "composition_id": composition.id,
                    "route_id": composition.route_id,
                    "scale": request.scale.value if hasattr(request.scale, 'value') else str(request.scale),
                    "key": request.key,
                    "complexity": analytics.get("complexity", 0),
                    "harmonic_richness": analytics.get("harmonic_richness", 0)
                }
            )
            logger.info(f"✅ Synced home route to DuckDB: {request.origin_code} → {request.destination_code}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not sync home route to DuckDB: {e}")

        return RouteGenerateResponse(
            composition_id=composition.id,
            route_id=composition.route_id,
            midi_file_url=f"/api/download/{composition.id}",
            analytics=analytics,
            message="Music composition generated successfully"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/analytics/{composition_id}", response_model=AnalyticsResponse)
async def get_analytics(
    composition_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Fetch melody complexity & stats for a composition"""
    result = await db.execute(
        select(MusicComposition).where(MusicComposition.id == composition_id)
    )
    composition = result.scalar_one_or_none()

    if not composition:
        raise HTTPException(status_code=404, detail="Composition not found")

    # Get route info for similar routes
    route_result = await db.execute(
        select(Route).where(Route.id == composition.route_id)
    )
    route = route_result.scalar_one_or_none()

    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # Find similar routes based on embeddings
    similar_routes = await find_similar_routes(db, route.id, limit=5)

    analytics = {
        'melodic_complexity': composition.complexity_score or 0,
        'harmonic_richness': composition.harmonic_richness or 0,
        'tempo_variation': 0,  # Could be calculated from composition data
        'pitch_range': 24,  # Default value
        'note_density': 0,  # Could be calculated from composition data
        'similar_routes': similar_routes
    }

    return AnalyticsResponse(composition_id=composition_id, **analytics)


@router.get("/similar", response_model=SimilarRoutesResponse)
async def find_similar_routes_endpoint(
    origin_code: str,
    destination_code: str,
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Find similar-sounding routes"""
    # Find the route
    origin_result = await db.execute(
        select(Airport).where(Airport.iata_code == origin_code.upper())
    )
    destination_result = await db.execute(
        select(Airport).where(Airport.iata_code == destination_code.upper())
    )

    origin = origin_result.scalar_one_or_none()
    destination = destination_result.scalar_one_or_none()

    if not origin or not destination:
        raise HTTPException(status_code=404, detail="Airports not found")

    route_result = await db.execute(
        select(Route).where(
            Route.origin_airport_id == origin.id,
            Route.destination_airport_id == destination.id
        )
    )
    route = route_result.scalar_one_or_none()

    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    similar_routes = await find_similar_routes(db, route.id, limit)

    return SimilarRoutesResponse(route_id=route.id, similar_routes=similar_routes)


@router.get("/download/{composition_id}")
async def download_midi(
    composition_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Download generated MIDI file"""
    result = await db.execute(
        select(MusicComposition).where(MusicComposition.id == composition_id)
    )
    composition = result.scalar_one_or_none()

    if not composition:
        raise HTTPException(status_code=404, detail="Composition not found")

    if not os.path.exists(composition.midi_path):
        raise HTTPException(status_code=404, detail="MIDI file not found")

    return {"file_path": composition.midi_path}


@router.get("/recent", response_model=RecentCompositionsResponse)
async def get_recent_compositions(
    limit: int = Query(5, ge=1, le=50),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Fetch 5 most recent compositions"""
    result = await db.execute(
        select(MusicComposition)
        .order_by(MusicComposition.created_at.desc())
        .limit(limit)
    )
    compositions = result.scalars().all()

    composition_infos = []
    for composition in compositions:
        route_result = await db.execute(
            select(Route).where(Route.id == composition.route_id)
        )
        route = route_result.scalar_one_or_none()

        if route:
            composition_infos.append(await get_composition_info(composition, route, db))

    return RecentCompositionsResponse(
        compositions=composition_infos,
        total_count=len(composition_infos)
    )


@router.post("/auth/register", response_model=UserInfo)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    # Check if user already exists
    result = await db.execute(
        select(User).where(
            (User.username == user_data.username) | (User.email == user_data.email)
        )
    )
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username or email already registered"
        )

    # Create new user
    from app.core.security import get_password_hash
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password)
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserInfo(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=bool(user.is_active),
        created_at=user.created_at
    )


@router.post("/auth/login", response_model=Token)
async def login_user(
    user_data: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Login user and return JWT token"""
    user = await authenticate_user(db, user_data.username, user_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, expires_in=3600)


@router.get("/auth/me", response_model=UserInfo)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserInfo(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=bool(current_user.is_active),
        created_at=current_user.created_at
    )


async def get_route_info(route: Route, db: AsyncSession) -> RouteInfo:
    """Helper function to get route information"""
    origin_result = await db.execute(
        select(Airport).where(Airport.id == route.origin_airport_id)
    )
    destination_result = await db.execute(
        select(Airport).where(Airport.id == route.destination_airport_id)
    )

    origin = origin_result.scalar_one_or_none()
    destination = destination_result.scalar_one_or_none()

    if not origin or not destination:
        raise HTTPException(status_code=404, detail="Airports not found")

    return RouteInfo(
        id=route.id,
        origin_airport=AirportSearchResponse(
            id=origin.id,
            name=origin.name,
            city=origin.city,
            country=origin.country,
            iata_code=origin.iata_code,
            latitude=float(origin.latitude),
            longitude=float(origin.longitude)
        ),
        destination_airport=AirportSearchResponse(
            id=destination.id,
            name=destination.name,
            city=destination.city,
            country=destination.country,
            iata_code=destination.iata_code,
            latitude=float(destination.latitude),
            longitude=float(destination.longitude)
        ),
        distance_km=route.distance_km,
        duration_min=route.duration_min
    )


async def get_composition_info(composition: MusicComposition, route: Route, db: AsyncSession) -> CompositionInfo:
    """Helper function to get composition information"""
    origin_result = await db.execute(
        select(Airport).where(Airport.id == route.origin_airport_id)
    )
    destination_result = await db.execute(
        select(Airport).where(Airport.id == route.destination_airport_id)
    )

    origin = origin_result.scalar_one_or_none()
    destination = destination_result.scalar_one_or_none()

    # Build composition info with helper fields
    comp_info = CompositionInfo(
        id=composition.id,
        route=RouteInfo(
            id=route.id,
            origin_airport=AirportSearchResponse(
                id=origin.id, name=origin.name, city=origin.city,
                country=origin.country, iata_code=origin.iata_code,
                latitude=float(origin.latitude), longitude=float(origin.longitude)
            ),
            destination_airport=AirportSearchResponse(
                id=destination.id, name=destination.name, city=destination.city,
                country=destination.country, iata_code=destination.iata_code,
                latitude=float(destination.latitude), longitude=float(destination.longitude)
            ),
            distance_km=route.distance_km,
            duration_min=route.duration_min
        ),
        tempo=composition.tempo,
        pitch=composition.pitch,
        harmony=composition.harmony,
        midi_path=composition.midi_path,
        complexity_score=composition.complexity_score,
        harmonic_richness=composition.harmonic_richness,
        duration_seconds=composition.duration_seconds,
        unique_notes=composition.unique_notes,
        musical_key=composition.musical_key,
        scale=composition.scale,
        created_at=composition.created_at
    )
    
    # Add helper fields for easier frontend access
    comp_dict = comp_info.model_dump()
    comp_dict['title'] = f"{origin.iata_code} → {destination.iata_code}"
    comp_dict['route_name'] = f"{origin.city} to {destination.city}"
    comp_dict['origin_code'] = origin.iata_code
    comp_dict['destination_code'] = destination.iata_code
    comp_dict['genre'] = composition.scale
    comp_dict['likes_count'] = 0  # TODO: Get from database
    comp_dict['play_count'] = 0  # TODO: Get from database
    
    # Load MIDI data for playback
    try:
        import os
        from mido import MidiFile
        
        if composition.midi_path and os.path.exists(composition.midi_path):
            mid = MidiFile(composition.midi_path)
            notes = []
            
            for track in mid.tracks:
                current_time = 0
                for msg in track:
                    current_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        notes.append({
                            'note': msg.note,
                            'velocity': msg.velocity,
                            'time': current_time,
                            'duration': 480  # Default duration
                        })
            
            comp_dict['midi_data'] = {
                'notes': notes,
                'tempo': composition.tempo or 120
            }
    except Exception as e:
        print(f"Error loading MIDI data: {e}")
        comp_dict['midi_data'] = None
    
    return comp_dict


@router.get("/airports/search", response_model=List[AirportSearchResponse])
async def search_airports(
    query: str,
    limit: int = Query(20, ge=1, le=100),
    country: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Search airports by name, city, or IATA code (only returns airports with IATA codes)"""
    try:
        # Base filter: only airports with IATA codes
        search_filter = (Airport.name.contains(query) | \
                        Airport.city.contains(query) | \
                        Airport.iata_code.contains(query.upper())) & \
                        Airport.iata_code.isnot(None)

        if country:
            search_filter = search_filter & Airport.country.contains(country)

        result = await db.execute(
            select(Airport)
            .where(search_filter)
            .order_by(Airport.name)
            .limit(limit)
        )
        airports = result.scalars().all()

        return [
            AirportSearchResponse(
                id=airport.id,
                name=airport.name,
                city=airport.city,
                country=airport.country,
                iata_code=airport.iata_code,
                latitude=float(airport.latitude),
                longitude=float(airport.longitude)
            )
            for airport in airports
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/airports/{iata_code}", response_model=AirportSearchResponse)
async def get_airport(
    iata_code: str,
    db: AsyncSession = Depends(get_db)
):
    """Get airport by IATA code"""
    try:
        result = await db.execute(
            select(Airport).where(Airport.iata_code == iata_code.upper())
        )
        airport = result.scalar_one_or_none()

        if not airport:
            raise HTTPException(status_code=404, detail="Airport not found")

        return AirportSearchResponse(
            id=airport.id,
            name=airport.name,
            city=airport.city,
            country=airport.country,
            iata_code=airport.iata_code,
            latitude=float(airport.latitude),
            longitude=float(airport.longitude)
        )
    except HTTPException:
        raise
@router.get("/compositions/{composition_id}", response_model=CompositionInfo)
async def get_composition(
    composition_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific composition (no auth required)"""
    try:
        result = await db.execute(
            select(MusicComposition).where(MusicComposition.id == composition_id)
        )
        composition = result.scalar_one_or_none()

        if not composition:
            raise HTTPException(status_code=404, detail="Composition not found")

        # Get route info
        route_result = await db.execute(
            select(Route).where(Route.id == composition.route_id)
        )
        route = route_result.scalar_one_or_none()

        if not route:
            raise HTTPException(status_code=404, detail="Route not found")

        return await get_composition_info(composition, route, db)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get composition: {str(e)}")


@router.get("/compositions", response_model=List[CompositionInfo])
async def get_user_compositions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Get all public compositions (no auth required)"""
    try:
        result = await db.execute(
            select(MusicComposition)
            .order_by(MusicComposition.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        compositions = result.scalars().all()

        composition_infos = []
        for composition in compositions:
            route_result = await db.execute(
                select(Route).where(Route.id == composition.route_id)
            )
            route = route_result.scalar_one_or_none()

            if route:
                composition_infos.append(await get_composition_info(composition, route, db))

        return composition_infos

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get compositions: {str(e)}")


@router.get("/public/compositions", response_model=List[CompositionInfo])
async def get_public_compositions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    genre: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get public compositions (no auth required)"""
    try:
        query = select(MusicComposition).where(MusicComposition.is_public == 1)

        if genre:
            query = query.where(MusicComposition.genre == genre)

        result = await db.execute(
            query.order_by(MusicComposition.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        compositions = result.scalars().all()

        composition_infos = []
        for composition in compositions:
            route_result = await db.execute(
                select(Route).where(Route.id == composition.route_id)
            )
            route = route_result.scalar_one_or_none()

            if route:
                composition_infos.append(await get_composition_info(composition, route, db))

        return composition_infos

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get public compositions: {str(e)}")


@router.delete("/compositions/{composition_id}")
async def delete_composition(
    composition_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a composition"""
    try:
        result = await db.execute(
            select(MusicComposition).where(MusicComposition.id == composition_id)
        )
        composition = result.scalar_one_or_none()

        if not composition:
            raise HTTPException(status_code=404, detail="Composition not found")

        if composition.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Delete the composition
        await db.delete(composition)
        await db.commit()

        return {"message": "Composition deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete composition: {str(e)}")


async def find_similar_routes(db: AsyncSession, route_id: int, limit: int = 5) -> List[Dict]:
    """Find similar routes based on JSON embeddings"""
    try:
        # Get the source route
        result = await db.execute(
            select(Route).where(Route.id == route_id)
        )
        source_route = result.scalar_one_or_none()

        if not source_route or not source_route.route_embedding:
            # Fallback to random routes
            result = await db.execute(
                select(Route.id).where(Route.id != route_id).order_by(db.func.random()).limit(limit)
            )
            similar_route_ids = result.scalars().all()
            return [{"route_id": rid, "similarity_score": 0.5} for rid in similar_route_ids]

        # Parse the embedding
        import json
        embedding = json.loads(source_route.route_embedding)

        # Use vector service to find similar routes
        similar_routes = await vector_service.find_similar_routes(db, embedding, limit)

        return [
            {
                "route_id": route["route_id"],
                "similarity_score": route["similarity_score"]
            }
            for route in similar_routes
        ]

    except Exception as e:
        # Fallback to random routes on error
        result = await db.execute(
            select(Route.id).where(Route.id != route_id).order_by(db.func.random()).limit(limit)
        )
        similar_route_ids = result.scalars().all()
        return [{"route_id": rid, "similarity_score": 0.5} for rid in similar_route_ids]
