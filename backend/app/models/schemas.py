"""
Pydantic schemas for API request/response models
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class MusicStyle(str, Enum):
    """Available music styles"""
    CLASSICAL = "classical"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    AMBIENT = "ambient"
    ROCK = "rock"


class ScaleType(str, Enum):
    """Available musical scales"""
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    PENTATONIC = "pentatonic"


class RouteGenerateRequest(BaseModel):
    """Request model for generating music from flight route"""
    origin_code: str = Field(..., min_length=3, max_length=3, description="Origin airport IATA code")
    destination_code: str = Field(..., min_length=3, max_length=3, description="Destination airport IATA code")
    music_style: MusicStyle = Field(MusicStyle.CLASSICAL, description="Desired music style")
    scale: ScaleType = Field(ScaleType.MAJOR, description="Musical scale")
    key: str = Field("C", description="Musical key (C, D, E, F, G, A, B)")
    tempo: Optional[int] = Field(120, ge=60, le=200, description="Tempo in BPM")
    duration_minutes: Optional[int] = Field(3, ge=1, le=10, description="Duration in minutes")

    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        valid_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C#', 'D#', 'F#', 'G#', 'A#']
        if v.upper() not in valid_keys:
            raise ValueError(f'Key must be one of: {", ".join(valid_keys)}')
        return v.upper()


class RouteGenerateResponse(BaseModel):
    """Response model for generated music"""
    composition_id: int
    route_id: int
    midi_file_url: str
    analytics: dict
    message: str


class AirportSearchResponse(BaseModel):
    """Response model for airport search"""
    id: int
    name: str
    city: str
    country: str
    iata_code: Optional[str] = None
    latitude: float
    longitude: float


class RouteInfo(BaseModel):
    """Route information model"""
    id: int
    origin_airport: AirportSearchResponse
    destination_airport: AirportSearchResponse
    distance_km: Optional[float]
    duration_min: Optional[int]


class CompositionInfo(BaseModel):
    """Composition information model"""
    id: int
    route: RouteInfo
    tempo: int
    pitch: float
    harmony: float
    midi_path: str
    complexity_score: Optional[float]
    harmonic_richness: Optional[float]
    duration_seconds: Optional[int]
    unique_notes: Optional[int]
    musical_key: str
    scale: str
    created_at: datetime


class AnalyticsResponse(BaseModel):
    """Analytics response model"""
    composition_id: int
    melodic_complexity: float
    harmonic_richness: float
    tempo_variation: float
    pitch_range: float
    note_density: float
    similar_routes: List[int]


class SimilarRoutesResponse(BaseModel):
    """Similar routes response model"""
    route_id: int
    similar_routes: List[dict]  # List of route objects with similarity scores


class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserInfo(BaseModel):
    """User information response"""
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime


class RecentCompositionsResponse(BaseModel):
    """Recent compositions response"""
    compositions: List[CompositionInfo]
    total_count: int
