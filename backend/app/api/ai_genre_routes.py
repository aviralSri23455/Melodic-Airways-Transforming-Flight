"""
AI Genre Composition API Routes - Advanced PyTorch-based genre composition
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from pydantic import BaseModel

from app.services.ai_genre_composer import AIGenreComposer


router = APIRouter()
ai_composer = AIGenreComposer()


class RouteFeatures(BaseModel):
    distance: float
    latitude_range: float
    longitude_range: float
    direction: str
    time_of_day: Optional[str] = None


class GenreCompositionRequest(BaseModel):
    genre: str
    route_features: dict
    duration: int = 30
    tempo: Optional[int] = None


class GenreBlendRequest(BaseModel):
    primary_genre: str
    secondary_genre: str
    blend_ratio: float = 0.5
    route_features: Optional[dict] = None


@router.get("/ai-genres/available", response_model=dict)
async def get_available_genres():
    """Get list of available AI genres with their characteristics"""
    return {
        "success": True,
        "data": {
            "genres": ai_composer.GENRES,
            "total_genres": len(ai_composer.GENRES)
        }
    }


@router.post("/ai-genres/compose", response_model=dict)
async def compose_with_ai_genre(request: GenreCompositionRequest):
    """
    Generate a genre-specific composition using AI models
    
    Example request:
    {
        "genre": "jazz",
        "route_features": {
            "distance": 5000,
            "latitude_range": 40,
            "longitude_range": 80,
            "direction": "E"
        },
        "duration": 30,
        "tempo": 120
    }
    """
    try:
        composition = ai_composer.generate_genre_composition(
            genre=request.genre,
            route_features=request.route_features,
            duration=request.duration,
            tempo=request.tempo
        )
        
        # ✅ Sync AI Composer composition to DuckDB (128D embeddings)
        try:
            from app.services.duckdb_sync_service import duckdb_sync
            import numpy as np
            import logging
            
            logger = logging.getLogger(__name__)
            
            # Generate 128D embedding for AI composition
            embedding_128d = np.random.randn(128).tolist()  # In production, use real embedding model
            
            composition_data = {
                "vector_embedding": embedding_128d,
                "genre": request.genre,
                "tempo": request.tempo,
                "complexity": composition.get("complexity", 0.7),
                "duration": request.duration,
                "scale": composition.get("scale", "major"),
                "key": composition.get("key", "C"),
                "dynamics": composition.get("dynamics", "moderate"),
                "ai_generated": True
            }
            
            duckdb_sync.sync_ai_composer_embedding(composition_data)
            logger.info(f"✅ Synced AI Composer to DuckDB: {request.genre}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not sync AI Composer to DuckDB: {e}")
        
        return {
            "success": True,
            "data": composition,
            "message": f"AI-generated {request.genre} composition created successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate AI composition: {str(e)}"
        )


@router.post("/ai-genres/recommendations", response_model=dict)
async def get_genre_recommendations(route_features: RouteFeatures):
    """
    Get AI-powered genre recommendations based on route characteristics
    
    Returns genres ranked by suitability for the given route
    """
    try:
        recommendations = ai_composer.get_genre_recommendations(
            route_features=route_features.dict()
        )
        
        return {
            "success": True,
            "data": {
                "recommendations": [
                    {"genre": genre, "confidence": float(conf)}
                    for genre, conf in recommendations
                ],
                "top_genre": recommendations[0][0] if recommendations else None
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@router.post("/ai-genres/blend", response_model=dict)
async def blend_genres(request: GenreBlendRequest):
    """
    Create a blended composition mixing two genres
    
    Example request:
    {
        "primary_genre": "classical",
        "secondary_genre": "electronic",
        "blend_ratio": 0.3,
        "route_features": {
            "distance": 5000,
            "latitude_range": 30,
            "longitude_range": 50,
            "direction": "W"
        }
    }
    """
    try:
        if request.blend_ratio < 0 or request.blend_ratio > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Blend ratio must be between 0 and 1"
            )
        
        composition = ai_composer.blend_genres(
            primary_genre=request.primary_genre,
            secondary_genre=request.secondary_genre,
            blend_ratio=request.blend_ratio,
            route_features=request.route_features
        )
        
        return {
            "success": True,
            "data": composition,
            "message": f"Blended {request.primary_genre} and {request.secondary_genre} composition created"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to blend genres: {str(e)}"
        )


@router.get("/ai-genres/demo/{genre}", response_model=dict)
async def demo_genre_composition(
    genre: str,
    distance: float = Query(5000, description="Flight distance in km"),
    latitude_range: float = Query(30, description="Latitude difference"),
    longitude_range: float = Query(50, description="Longitude difference"),
    direction: str = Query("E", description="Cardinal direction"),
    duration: int = Query(30, description="Duration in seconds")
):
    """
    Quick demo of AI genre composition with simple parameters
    
    Example: /ai-genres/demo/jazz?distance=5000&duration=30
    """
    try:
        route_features = {
            "distance": distance,
            "latitude_range": latitude_range,
            "longitude_range": longitude_range,
            "direction": direction
        }
        
        composition = ai_composer.generate_genre_composition(
            genre=genre,
            route_features=route_features,
            duration=duration
        )
        
        return {
            "success": True,
            "data": composition,
            "demo": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate demo: {str(e)}"
        )


@router.post("/ai-genres/similar", response_model=dict)
async def find_similar_compositions(composition: dict):
    """
    Find similar compositions using vector similarity search
    
    Example request:
    {
        "genre": "jazz",
        "tempo": 120,
        "complexity": 0.8,
        "note_sequence": [...]
    }
    """
    try:
        similar = ai_composer.find_similar_compositions(
            query_composition=composition,
            k=5
        )
        
        return {
            "success": True,
            "data": {
                "similar_compositions": similar,
                "total_indexed": ai_composer.faiss_index.ntotal,
                "search_method": "FAISS vector similarity (L2 distance)"
            },
            "message": f"Found {len(similar)} similar compositions"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar compositions: {str(e)}"
        )


@router.get("/ai-genres/index-stats", response_model=dict)
async def get_index_statistics():
    """Get statistics about the FAISS vector index"""
    try:
        return {
            "success": True,
            "data": {
                "total_compositions": ai_composer.faiss_index.ntotal,
                "embedding_dimension": ai_composer.embedding_dim,
                "index_type": "FAISS IndexFlatL2",
                "metadata_count": len(ai_composer.composition_metadata),
                "recent_compositions": ai_composer.composition_metadata[-5:] if ai_composer.composition_metadata else []
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index stats: {str(e)}"
        )


@router.get("/ai-genres/model-info", response_model=dict)
async def get_model_info():
    """Get information about the AI models being used"""
    import torch
    
    return {
        "success": True,
        "data": {
            "framework": "PyTorch",
            "device": str(ai_composer.device),
            "cuda_available": torch.cuda.is_available(),
            "models": {
                "embedding_model": {
                    "type": "GenreEmbeddingModel",
                    "input_dim": 10,
                    "hidden_dim": 64,
                    "output_dim": 32
                },
                "pattern_generator": {
                    "type": "MusicPatternGenerator (LSTM)",
                    "input_dim": 32,
                    "hidden_dim": 128,
                    "output_dim": 12
                },
                "vector_search": {
                    "type": "FAISS IndexFlatL2",
                    "embedding_dim": 128,
                    "indexed_compositions": ai_composer.faiss_index.ntotal
                }
            },
            "supported_genres": list(ai_composer.GENRES.keys()),
            "features": [
                "Genre-specific embeddings",
                "LSTM-based note generation",
                "Dynamic tempo and scale selection",
                "Genre blending",
                "Confidence-based recommendations",
                "✨ Vector similarity search (FAISS)",
                "✨ 128D composition embeddings",
                "✨ Semantic music recommendations"
            ]
        }
    }
