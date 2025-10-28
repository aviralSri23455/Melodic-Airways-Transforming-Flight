"""
Database models for Aero Melody application
"""

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, DECIMAL, JSON, Boolean, Enum, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from typing import List, Optional
import json
import enum

from app.db.database import Base


class Airport(Base):
    """Airport model based on OpenFlights dataset"""
    __tablename__ = "airports"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    city = Column(String(255))
    country = Column(String(255), nullable=False)
    iata_code = Column(String(3), unique=True, index=True)
    icao_code = Column(String(4), unique=True, index=True)
    latitude = Column(DECIMAL(12, 10), nullable=False)
    longitude = Column(DECIMAL(13, 10), nullable=False)
    altitude = Column(Integer)
    timezone = Column(String(50))
    dst = Column(String(10))
    tz_database_time_zone = Column(String(100))
    type = Column(String(50))
    source = Column(String(50))

    # Relationships
    routes_from = relationship("Route", back_populates="origin_airport", foreign_keys="Route.origin_airport_id")
    routes_to = relationship("Route", back_populates="destination_airport", foreign_keys="Route.destination_airport_id")


class Route(Base):
    """Flight route model"""
    __tablename__ = "routes"

    id = Column(Integer, primary_key=True, index=True)
    origin_airport_id = Column(Integer, ForeignKey("airports.id"), nullable=False)
    destination_airport_id = Column(Integer, ForeignKey("airports.id"), nullable=False)
    distance_km = Column(DECIMAL(10, 2))
    duration_min = Column(Integer)
    route_embedding = Column(JSON, nullable=True)  # JSON array for similarity search

    # Relationships
    origin_airport = relationship("Airport", back_populates="routes_from", foreign_keys=[origin_airport_id])
    destination_airport = relationship("Airport", back_populates="routes_to", foreign_keys=[destination_airport_id])
    compositions = relationship("MusicComposition", back_populates="route")


class MusicComposition(Base):
    """Generated music composition metadata with vector storage and collaboration support"""
    __tablename__ = "music_compositions"

    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("user_datasets.id"), nullable=True)
    tempo = Column(Integer, nullable=False)
    pitch = Column(Float, nullable=False)
    harmony = Column(Float, nullable=False)
    midi_path = Column(String(500), nullable=False)
    complexity_score = Column(Float)
    harmonic_richness = Column(Float)
    duration_seconds = Column(Integer)
    unique_notes = Column(Integer)
    musical_key = Column(String(2))
    scale = Column(String(20))
    title = Column(String(255), nullable=True)
    genre = Column(String(100), nullable=True)
    music_vector = Column(JSON, nullable=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    route = relationship("Route", back_populates="compositions")
    user = relationship("User", back_populates="compositions")
    dataset = relationship("UserDataset", back_populates="compositions")
    remixes_as_original = relationship("CompositionRemix", foreign_keys="CompositionRemix.original_composition_id", back_populates="original_composition")
    remixes_as_remix = relationship("CompositionRemix", foreign_keys="CompositionRemix.remix_composition_id", back_populates="remix_composition")
    collaboration_sessions = relationship("CollaborationSession", back_populates="composition")


class User(Base):
    """User model for authentication and collaboration"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user")  # admin, user, premium
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    compositions = relationship("MusicComposition", back_populates="user")
    datasets = relationship("UserDataset", back_populates="user")
    collections = relationship("UserCollection", back_populates="user")
    collaboration_sessions = relationship("CollaborationSession", back_populates="creator")
    activities = relationship("UserActivity", back_populates="user")


class RemixType(str, enum.Enum):
    """Enum for remix types"""
    VARIATION = "variation"
    GENRE_CHANGE = "genre_change"
    TEMPO_CHANGE = "tempo_change"
    FULL_REMIX = "full_remix"


class UserDataset(Base):
    """User personal collection of flight routes and compositions"""
    __tablename__ = "user_datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    route_data = Column(JSON, nullable=False)
    dataset_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="datasets")
    compositions = relationship("MusicComposition", back_populates="dataset")


class UserCollection(Base):
    """User collection for organizing compositions"""
    __tablename__ = "user_collections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    composition_ids = Column(JSON, nullable=True)  # Array of composition IDs
    tags = Column(JSON, nullable=True)  # Array of tags
    created_at = Column(DateTime, default=func.now())

    # Relationships
    user = relationship("User", back_populates="collections")


class CollaborationSession(Base):
    """Real-time collaboration session for multiple users"""
    __tablename__ = "collaboration_sessions"

    id = Column(Integer, primary_key=True, index=True)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    composition_id = Column(Integer, ForeignKey("music_compositions.id"), nullable=True)
    session_state = Column(JSON, nullable=True)
    participants = Column(JSON, nullable=True)  # Array of participant IDs
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True)

    # Relationships
    creator = relationship("User", back_populates="collaboration_sessions")
    composition = relationship("MusicComposition", back_populates="collaboration_sessions")


class CompositionRemix(Base):
    """Track remix relationships and attribution"""
    __tablename__ = "composition_remixes"

    id = Column(Integer, primary_key=True, index=True)
    original_composition_id = Column(Integer, ForeignKey("music_compositions.id"), nullable=False)
    remix_composition_id = Column(Integer, ForeignKey("music_compositions.id"), nullable=False)
    remix_type = Column(Enum(RemixType), nullable=False)
    attribution_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    original_composition = relationship("MusicComposition", foreign_keys="CompositionRemix.original_composition_id", back_populates="remixes_as_original")
    remix_composition = relationship("MusicComposition", foreign_keys="CompositionRemix.remix_composition_id", back_populates="remixes_as_remix")

class ActivityType(str, enum.Enum):
    """Enum for different types of user activities"""
    COMPOSITION_CREATED = "composition_created"
    COMPOSITION_UPDATED = "composition_updated"
    COMPOSITION_DELETED = "composition_deleted"
    DATASET_CREATED = "dataset_created"
    DATASET_UPDATED = "dataset_updated"
    DATASET_DELETED = "dataset_deleted"
    COLLECTION_CREATED = "collection_created"
    COLLECTION_UPDATED = "collection_updated"
    COLLECTION_DELETED = "collection_deleted"
    REMIX_CREATED = "remix_created"
    REMIX_UPDATED = "remix_updated"
    COLLABORATION_JOINED = "collaboration_joined"
    COLLABORATION_LEFT = "collaboration_left"
    COLLABORATION_UPDATED = "collaboration_updated"
    SEARCH_PERFORMED = "search_performed"
    PROFILE_UPDATED = "profile_updated"
    LOGIN = "login"
    LOGOUT = "logout"


class UserActivity(Base):
    """Track user activities for real-time activity feed"""
    __tablename__ = "user_activities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    activity_type = Column(Enum(ActivityType), nullable=False)
    target_id = Column(Integer, nullable=True)  # ID of the affected resource
    target_type = Column(String(50), nullable=True)  # Type of target resource
    activity_data = Column(JSON, nullable=True)  # Additional activity data
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    user = relationship("User", back_populates="activities")
