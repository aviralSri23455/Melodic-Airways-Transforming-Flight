"""
Music generation service for converting flight routes to MIDI compositions
"""

from datetime import datetime

import asyncio
import math
import random
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
from typing import Dict, List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import logging
import json
import time

from app.core.config import settings
from app.models.models import Airport, Route, MusicComposition
from app.models.schemas import MusicStyle, ScaleType
from app.services.duckdb_analytics import get_analytics
from app.services.cache import get_cache
from app.services.redis_publisher import get_publisher
from app.services.faiss_duckdb_service import get_faiss_duckdb_service
from app.services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class RouteEmbeddingModel(nn.Module):
    """Neural network model for route embeddings"""

    def __init__(self, input_dim=6, hidden_dim=256, output_dim=512):
        super(RouteEmbeddingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


class MusicGenerator:
    """Main music generation service"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = RouteEmbeddingModel().to(self.device)
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'phrygian': [0, 1, 3, 5, 7, 8, 10],
            'lydian': [0, 2, 4, 6, 7, 9, 11],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'aeolian': [0, 2, 3, 5, 7, 8, 10],
            'locrian': [0, 1, 3, 5, 6, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9]
        }
        self.publisher = get_publisher()
        self.faiss_service = get_faiss_duckdb_service()
        self.websocket_manager = WebSocketManager()

    def get_route_features(self, route: Route, origin: Airport, destination: Airport) -> np.ndarray:
        """Extract features from route for embedding"""
        # Normalize coordinates to 0-1 range
        lat_range = 90  # -90 to 90
        lon_range = 180  # -180 to 180

        features = [
            origin.latitude / lat_range,
            origin.longitude / lon_range,
            destination.latitude / lat_range,
            destination.longitude / lon_range,
            route.distance_km / 20000,  # Normalize by max possible distance
            route.duration_min / 1440 if route.duration_min else 0  # Normalize by max flight time
        ]

        return np.array(features, dtype=np.float32)

    def generate_route_embedding(self, route: Route, origin: Airport, destination: Airport) -> np.ndarray:
        """Generate embedding vector for route"""
        features = self.get_route_features(route, origin, destination)
        features_tensor = torch.tensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.embedding_model(features_tensor)

        return embedding.cpu().numpy().flatten()

    def map_route_to_music_params(self, route: Route, origin: Airport, destination: Airport,
                                 style: MusicStyle, scale: ScaleType, key: str, tempo: int) -> Dict:
        """Map route characteristics to music parameters"""

        # Calculate direction vector
        lat_diff = destination.latitude - origin.latitude
        lon_diff = destination.longitude - origin.longitude

        # Determine pitch based on direction (eastward = ascending)
        direction_factor = (lon_diff + 180) / 360  # 0 to 1
        base_pitch = 60 + (direction_factor * 24)  # C4 to C6 range

        # Map distance to tempo (longer flights = slower tempo)
        distance_factor = min(float(route.distance_km) / 10000, 1.0)  # Normalize
        adjusted_tempo = int(float(tempo) * (1 - distance_factor * 0.3))  # Slower for long flights

        # Map altitude difference to harmony complexity
        alt_diff = abs((destination.altitude or 0) - (origin.altitude or 0))
        harmony_factor = min(alt_diff / 3000, 1.0)  # Normalize by 3000m difference

        # Determine scale notes
        scale_notes = self.scales[scale.value]
        root_note = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                     'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}[key]

        # Generate chord progression based on route characteristics
        chords = self.generate_chord_progression(scale_notes, root_note, harmony_factor)

        return {
            'tempo': adjusted_tempo,
            'base_pitch': base_pitch,
            'harmony_factor': harmony_factor,
            'chords': chords,
            'scale': scale.value,
            'key': key,
            'style': style.value
        }

    def generate_chord_progression(self, scale_notes: List[int], root_note: int, complexity: float) -> List[List[int]]:
        """Generate chord progression based on harmony factor"""
        num_chords = max(4, int(complexity * 8))  # 4-8 chords based on complexity

        chords = []
        current_root = root_note

        for i in range(num_chords):
            # Create chord (root, third, fifth from scale)
            chord_notes = [
                (current_root + scale_notes[0]) % 12,
                (current_root + scale_notes[2]) % 12,
                (current_root + scale_notes[4]) % 12
            ]

            # Add some variation for complex routes
            if complexity > 0.5 and i % 2 == 1:
                chord_notes.append((current_root + scale_notes[1]) % 12)

            chords.append(chord_notes)

            # Move to next chord root (simple progression)
            current_root = (current_root + scale_notes[3 if i % 2 == 0 else 4]) % 12

        return chords

    def generate_melody(self, music_params: Dict, duration_minutes: int) -> List[Message]:
        """Generate melodic line based on music parameters"""
        melody = []
        ticks_per_beat = 480  # Standard MIDI resolution
        ticks_per_minute = ticks_per_beat * music_params['tempo']
        total_ticks = int((duration_minutes * 60 * ticks_per_minute) / music_params['tempo'])

        current_tick = 0
        note_duration = ticks_per_beat  # Quarter notes

        while current_tick < total_ticks:
            # Choose note from current chord
            chord = music_params['chords'][len(melody) % len(music_params['chords'])]
            note = random.choice(chord) + music_params['base_pitch']

            # Add some variation
            if random.random() > 0.7:
                note += random.choice([-12, -1, 1, 12])

            melody.append(Message('note_on', note=int(note), velocity=64, time=current_tick))
            melody.append(Message('note_off', note=int(note), velocity=64, time=current_tick + note_duration))

            current_tick += note_duration

        return melody

    def generate_midi_file(self, music_params: Dict, duration_minutes: int, filename: str) -> str:
        """Generate complete MIDI file"""
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        track.append(MetaMessage('set_tempo', tempo=bpm2tempo(music_params['tempo']), time=0))

        # Set time signature (4/4)
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # Generate melody
        melody = self.generate_melody(music_params, duration_minutes)

        # Add melody to track
        for msg in melody:
            track.append(msg)

        # Save file
        mid.save(filename)
        return filename

    def calculate_music_analytics(self, music_params: Dict, melody: List[Message]) -> Dict:
        """Calculate analytics for the generated music"""
        notes = []
        for msg in melody:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)

        if not notes:
            return {
                'melodic_complexity': 0.0,
                'harmonic_richness': 0.0,
                'tempo_variation': 0.0,
                'pitch_range': 0,
                'note_density': 0.0
            }

        # Calculate metrics
        pitch_range = max(notes) - min(notes)
        unique_notes = len(set(notes))
        note_density = len(notes) / (len(melody) / 2)  # Notes per second

        # Simple complexity measure based on note variety
        melodic_complexity = unique_notes / len(notes) if notes else 0

        # Harmonic richness based on chord variety
        harmonic_richness = len(set(tuple(chord) for chord in music_params['chords'])) / len(music_params['chords'])

        tempo_variation = abs(music_params['tempo'] - 120) / 120  # Variation from standard tempo

        return {
            'melodic_complexity': round(melodic_complexity, 3),
            'harmonic_richness': round(harmonic_richness, 3),
            'tempo_variation': round(tempo_variation, 3),
            'pitch_range': pitch_range,
            'note_density': round(note_density, 3)
        }


    async def build_airport_graph(self, db: AsyncSession) -> nx.Graph:
        """Build NetworkX graph from all airport routes for path finding"""
        # Get all routes from database
        route_result = await db.execute(select(Route))
        routes = route_result.scalars().all()

        # Get all airports for coordinate data
        airport_result = await db.execute(select(Airport))
        airports = airport_result.scalars().all()

        # Create airport lookup dictionary
        airport_lookup = {airport.id: airport for airport in airports}

        # Build NetworkX graph
        graph = nx.Graph()

        # Add nodes (airports)
        for airport in airports:
            graph.add_node(airport.id, iata=airport.iata_code, lat=airport.latitude, lon=airport.longitude)

        # Add edges (routes) with distance as weight
        for route in routes:
            if route.origin_airport_id in airport_lookup and route.destination_airport_id in airport_lookup:
                origin = airport_lookup[route.origin_airport_id]
                destination = airport_lookup[route.destination_airport_id]

                # Calculate great circle distance if not stored
                if route.distance_km is None or route.distance_km == 0:
                    distance = self.calculate_distance(origin.latitude, origin.longitude,
                                                    destination.latitude, destination.longitude)
                else:
                    distance = route.distance_km

                graph.add_edge(route.origin_airport_id, route.destination_airport_id,
                             weight=distance, route_id=route.id)

        return graph

    def find_optimal_path(self, graph: nx.Graph, origin_id: int, destination_id: int) -> List[Dict]:
        """Find optimal path using Dijkstra's algorithm"""
        try:
            # Use Dijkstra's algorithm to find shortest path
            shortest_path = nx.shortest_path(graph, source=origin_id, target=destination_id, weight='weight')

            # Get path details
            path_details = []
            total_distance = 0

            for i in range(len(shortest_path) - 1):
                origin_node = shortest_path[i]
                dest_node = shortest_path[i + 1]

                # Get edge data
                edge_data = graph.get_edge_data(origin_node, dest_node)

                # Get airport details
                origin_airport = graph.nodes[origin_node]
                dest_airport = graph.nodes[dest_node]

                # Calculate additional metrics
                segment_distance = edge_data['weight']
                total_distance += segment_distance

                # Calculate direction (bearing)
                direction = self.calculate_bearing(
                    origin_airport['lat'], origin_airport['lon'],
                    dest_airport['lat'], dest_airport['lon']
                )

                path_details.append({
                    'origin_id': origin_node,
                    'destination_id': dest_node,
                    'origin_iata': origin_airport['iata'],
                    'destination_iata': dest_airport['iata'],
                    'distance_km': segment_distance,
                    'direction_angle': direction,
                    'route_id': edge_data.get('route_id')
                })

            return path_details, total_distance

        except nx.NetworkXNoPath:
            raise ValueError("No path found between the specified airports")
        except nx.NodeNotFound:
            raise ValueError("One or both airports not found in route network")

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        # Convert decimal degrees to radians
        lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers

        return c * r

    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing angle between two points"""
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)

        # Calculate bearing
        x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)

        bearing = math.atan2(x, y)

        # Convert to degrees and normalize to 0-360
        bearing_deg = math.degrees(bearing)
        bearing_normalized = (bearing_deg + 360) % 360

        return bearing_normalized

    def map_route_to_music_params_multi_segment(self, path_segments: List[Dict], total_distance: float,
                                              style: MusicStyle, scale: ScaleType, key: str, tempo: int) -> Dict:
        """Map multi-segment route characteristics to music parameters"""
        if not path_segments:
            raise ValueError("No path segments provided")

        # Calculate average direction and distance factors
        total_segments = len(path_segments)

        # Calculate average direction (bearing) across all segments
        directions = [segment['direction_angle'] for segment in path_segments]
        avg_direction = sum(directions) / len(directions)

        # Direction factor affects pitch (eastward = ascending)
        direction_factor = (avg_direction / 360)  # 0 to 1
        base_pitch = 60 + (direction_factor * 24)  # C4 to C6 range

        # Distance factor affects tempo (longer total distance = slower tempo)
        distance_factor = min(float(total_distance) / 10000, 1.0)  # Normalize
        adjusted_tempo = int(float(tempo) * (1 - distance_factor * 0.3))  # Slower for long distances

        # Complexity factor based on number of segments and total distance
        complexity_factor = min((total_segments * 0.1) + (float(total_distance) / 10000 * 0.05), 1.0)

        # Determine scale notes
        scale_notes = self.scales[scale.value]
        root_note = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                     'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}[key]

        # Generate chord progression based on route characteristics
        chords = self.generate_chord_progression(scale_notes, root_note, complexity_factor)

        return {
            'tempo': adjusted_tempo,
            'base_pitch': base_pitch,
            'harmony_factor': complexity_factor,
            'chords': chords,
            'scale': scale.value,
            'key': key,
            'style': style.value,
            'total_segments': total_segments,
            'total_distance': total_distance,
            'complexity_factor': complexity_factor
        }

    def generate_route_embedding_multi_segment(self, path_segments: List[Dict]) -> np.ndarray:
        """Generate embedding vector for multi-segment route"""
        if not path_segments:
            return np.zeros(128)

        # Extract features from all segments
        features = []

        for segment in path_segments:
            # Distance, direction, and segment characteristics
            features.extend([
                segment['distance_km'] / 10000,  # Normalize distance
                segment['direction_angle'] / 360,  # Normalize direction
                len(segment['origin_iata']) / 10,  # Airport code length factor
                len(segment['destination_iata']) / 10
            ])

        # Add aggregate features
        total_distance = sum(float(segment['distance_km']) for segment in path_segments)
        avg_distance = total_distance / len(path_segments)

        features.extend([
            total_distance / 10000,  # Total distance factor
            avg_distance / 5000,     # Average segment distance
            len(path_segments) / 10  # Number of segments factor
        ])

        # Pad or truncate to 128 dimensions
        embedding = np.zeros(128)
        embedding[:len(features)] = features[:128]

        return embedding


    def publish_generation_progress(
        self,
        user_id: str,
        generation_id: str,
        progress: float,
        status: str,
        current_step: str = None
    ):
        """Publish music generation progress to Redis"""
        try:
            self.publisher.publish_generation_progress(
                generation_id=generation_id,
                user_id=user_id,
                progress=progress,
                status=status,
                current_step=current_step
            )
            logger.info(f"Published generation progress: {progress:.1%} - {status}")
        except Exception as e:
            logger.error(f"Error publishing generation progress: {e}")


    def publish_music_update(
        self,
        session_id: str,
        user_id: str,
        update_type: str,
        music_data: Dict
    ):
        """Publish real-time music update"""
        try:
            self.publisher.publish_music_update_real_time(
                session_id=session_id,
                user_id=user_id,
                update_type=update_type,
                music_data=music_data
            )
        except Exception as e:
            logger.error(f"Error publishing music update: {e}")


    def store_vector_with_realtime_sync(
        self,
        composition_id: int,
        route_id: int,
        origin: str,
        destination: str,
        genre: str,
        tempo: int,
        pitch: float,
        harmony: float,
        complexity: float,
        vector: np.ndarray
    ):
        """Store vector with real-time synchronization"""
        try:
            # Store in FAISS + DuckDB
            success = self.faiss_service.store_music_vector(
                composition_id=composition_id,
                route_id=route_id,
                origin=origin,
                destination=destination,
                genre=genre,
                tempo=tempo,
                pitch=pitch,
                harmony=harmony,
                complexity=complexity,
                vector=vector,
                metadata={
                    "stored_at": datetime.utcnow().isoformat(),
                    "vector_length": len(vector)
                }
            )

            if success:
                # Publish real-time update
                self.publisher.publish_route_music_sync(
                    route_id=str(route_id),
                    origin=origin,
                    destination=destination,
                    music_params={
                        "composition_id": composition_id,
                        "genre": genre,
                        "tempo": tempo,
                        "pitch": pitch,
                        "harmony": harmony,
                        "complexity": complexity,
                        "vector_stored": True
                    }
                )

                logger.info(f"Vector stored and synced for composition {composition_id}")
                return True
            else:
                logger.error(f"Failed to store vector for composition {composition_id}")
                return False

        except Exception as e:
            logger.error(f"Error storing vector with sync: {e}")
            return False


    async def generate_music_with_realtime_updates(
        self,
        db: AsyncSession,
        origin_code: str,
        destination_code: str,
        music_style: MusicStyle,
        scale: ScaleType,
        key: str,
        tempo: int,
        duration_minutes: int,
        user_id: str,
        session_id: str
    ) -> Tuple[MusicComposition, str, Dict]:
        """Generate music with real-time progress updates"""
        generation_id = f"gen_{user_id}_{int(asyncio.get_event_loop().time())}"

        try:
            # Step 1: Initialize generation
            self.publish_generation_progress(user_id, generation_id, 0.0, "starting", "Initializing generation")
            self.publish_music_update(session_id, user_id, "generation_started", {
                "generation_id": generation_id,
                "origin": origin_code,
                "destination": destination_code,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Step 2: Find airports (10%)
            self.publish_generation_progress(user_id, generation_id, 0.1, "processing", "Finding airports")

            origin_result = await db.execute(
                select(Airport).where(Airport.iata_code == origin_code.upper())
            )
            destination_result = await db.execute(
                select(Airport).where(Airport.iata_code == destination_code.upper())
            )

            origin = origin_result.scalar_one_or_none()
            destination = destination_result.scalar_one_or_none()

            if not origin or not destination:
                raise ValueError(f"Could not find airports with codes {origin_code} and {destination_code}")

            # Step 3: Build graph and find path (20-40%)
            self.publish_generation_progress(user_id, generation_id, 0.2, "processing", "Building route graph")
            graph = await self.build_airport_graph(db)

            self.publish_generation_progress(user_id, generation_id, 0.3, "processing", "Finding optimal path")
            path_segments, total_distance = self.find_optimal_path(graph, origin.id, destination.id)

            if not path_segments:
                raise ValueError(f"No route found between {origin_code} and {destination_code}")

            # Step 4: Generate music parameters (40-60%)
            self.publish_generation_progress(user_id, generation_id, 0.4, "processing", "Generating music parameters")

            music_params = self.map_route_to_music_params_multi_segment(
                path_segments, total_distance, music_style, scale, key, tempo
            )

            # Step 5: Generate MIDI file (60-80%)
            self.publish_generation_progress(user_id, generation_id, 0.6, "processing", "Generating MIDI file")

            midi_filename = f"composition_{generation_id}.mid"
            midi_path = f"{settings.MIDI_OUTPUT_DIR}/{midi_filename}"

            import os
            os.makedirs(settings.MIDI_OUTPUT_DIR, exist_ok=True)

            self.generate_midi_file(music_params, duration_minutes, midi_path)

            # Step 6: Calculate analytics (80-90%)
            self.publish_generation_progress(user_id, generation_id, 0.8, "processing", "Calculating analytics")

            melody = self.generate_melody(music_params, duration_minutes)
            analytics = self.calculate_music_analytics(music_params, melody)

            # Step 7: Store in database (90-95%)
            self.publish_generation_progress(user_id, generation_id, 0.9, "processing", "Storing composition")

            # Create route record
            first_segment = path_segments[0]
            route_result = await db.execute(
                select(Route).where(
                    Route.origin_airport_id == first_segment['origin_id'],
                    Route.destination_airport_id == first_segment['destination_id']
                )
            )
            route = route_result.scalar_one_or_none()

            if not route:
                distance = float(first_segment['distance_km'])
                duration = int(distance / 800 * 60) if distance > 0 else None

                route = Route(
                    origin_airport_id=first_segment['origin_id'],
                    destination_airport_id=first_segment['destination_id'],
                    distance_km=round(distance, 2),
                    duration_min=duration
                )
                db.add(route)
                await db.flush()

            # Create composition record
            composition = MusicComposition(
                route_id=route.id,
                tempo=music_params['tempo'],
                pitch=music_params['base_pitch'],
                harmony=music_params['harmony_factor'],
                midi_path=midi_path,
                complexity_score=analytics['melodic_complexity'],
                harmonic_richness=analytics['harmonic_richness'],
                duration_seconds=duration_minutes * 60,
                unique_notes=len(set(msg.note for msg in melody if msg.type == 'note_on')),
                musical_key=key,
                scale=scale.value
            )

            db.add(composition)
            await db.commit()
            await db.refresh(composition)

            # Step 8: Generate and store embedding (95-100%)
            self.publish_generation_progress(user_id, generation_id, 0.95, "processing", "Generating embeddings")

            embedding_vector = self.generate_route_embedding_multi_segment(path_segments)

            # Update route with embedding
            route_result = await db.execute(select(Route).where(Route.id == route.id))
            existing_route = route_result.scalar_one_or_none()

            if existing_route:
                existing_route.route_embedding = json.dumps(embedding_vector.tolist())
                await db.commit()

            # Store vector with real-time sync
            self.store_vector_with_realtime_sync(
                composition_id=composition.id,
                route_id=route.id,
                origin=origin_code,
                destination=destination_code,
                genre=music_style.value,
                tempo=music_params['tempo'],
                pitch=music_params['base_pitch'],
                harmony=music_params['harmony_factor'],
                complexity=analytics['melodic_complexity'],
                vector=embedding_vector
            )

            # Step 9: Finalize (100%)
            self.publish_generation_progress(user_id, generation_id, 1.0, "completed", "Generation complete")

            # Publish completion
            self.publish_music_update(session_id, user_id, "generation_completed", {
                "generation_id": generation_id,
                "composition_id": composition.id,
                "midi_path": midi_path,
                "tempo": music_params['tempo'],
                "key": key,
                "scale": scale.value,
                "complexity_score": analytics['melodic_complexity'],
                "duration_seconds": duration_minutes * 60,
                "completed_at": datetime.utcnow().isoformat()
            })

            logger.info(f"Real-time music generation completed for {origin_code} -> {destination_code}")

            return composition, midi_path, analytics

        except Exception as e:
            # Publish error
            self.publish_generation_progress(user_id, generation_id, 0.0, "error", f"Error: {str(e)}")
            self.publish_music_update(session_id, user_id, "generation_error", {
                "generation_id": generation_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.error(f"Real-time music generation failed: {e}")
            raise


class MusicGenerationService:
    """Service class for music generation operations"""

    def __init__(self):
        self.generator = MusicGenerator()
        self.analytics = get_analytics()
        self.cache = get_cache()

    async def generate_music_for_route(self, db: AsyncSession, origin_code: str, destination_code: str,
                                     music_style: MusicStyle, scale: ScaleType, key: str,
                                     tempo: int, duration_minutes: int) -> Tuple[MusicComposition, str, Dict]:
        """Generate music for a flight route using Dijkstra's algorithm"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached_music = self.cache.get_route_music(origin_code, destination_code)
            if cached_music:
                logger.info(f"Using cached music for {origin_code} -> {destination_code}")
                # Still log the cache hit as a performance metric
                self.analytics.log_performance_metric(
                    "music_generation_cache_hit",
                    (time.time() - start_time) * 1000,
                    True
                )

            # Find airports
            origin_result = await db.execute(
                select(Airport).where(Airport.iata_code == origin_code.upper())
            )
            destination_result = await db.execute(
                select(Airport).where(Airport.iata_code == destination_code.upper())
            )

            origin = origin_result.scalar_one_or_none()
            destination = destination_result.scalar_one_or_none()

            if not origin or not destination:
                raise ValueError(f"Could not find airports with codes {origin_code} and {destination_code}")

            # Build airport graph and find optimal path
            graph = await self.generator.build_airport_graph(db)
            path_segments, total_distance = self.generator.find_optimal_path(
                graph, origin.id, destination.id
            )

            if not path_segments:
                raise ValueError(f"No route found between {origin_code} and {destination_code}")
            
            # Log route analytics to DuckDB
            complexity_score = len(path_segments) * (float(total_distance) / 10000)
            self.analytics.log_route_analytics(
                origin=origin_code,
                destination=destination_code,
                distance_km=total_distance,
                complexity_score=complexity_score,
                path_length=len(path_segments),
                intermediate_stops=len(path_segments) - 1
            )

            # For simplicity, use the first segment as the main route for database storage
            # In a full implementation, you might want to store the entire path
            first_segment = path_segments[0]

            # Find or create route for the first segment
            route_result = await db.execute(
                select(Route).where(
                    Route.origin_airport_id == first_segment['origin_id'],
                    Route.destination_airport_id == first_segment['destination_id']
                ).limit(1)
            )
            route = route_result.scalar_one_or_none()

            if not route:
                # Calculate route metrics if not exists
                distance = float(first_segment['distance_km'])
                duration = int(distance / 800 * 60) if distance > 0 else None

                route = Route(
                    origin_airport_id=first_segment['origin_id'],
                    destination_airport_id=first_segment['destination_id'],
                    distance_km=round(distance, 2),
                    duration_min=duration
                )
                db.add(route)
                await db.flush()  # Get the route ID

            # Generate music parameters based on the full path
            music_params = self.generator.map_route_to_music_params_multi_segment(
                path_segments, total_distance, music_style, scale, key, tempo
            )

            # Generate MIDI file
            midi_filename = f"composition_{route.id}_{int(asyncio.get_event_loop().time())}.mid"
            midi_path = f"{settings.MIDI_OUTPUT_DIR}/{midi_filename}"

            # Ensure output directory exists
            import os
            os.makedirs(settings.MIDI_OUTPUT_DIR, exist_ok=True)

            self.generator.generate_midi_file(music_params, duration_minutes, midi_path)

            # Calculate analytics
            melody = self.generator.generate_melody(music_params, duration_minutes)
            analytics = self.generator.calculate_music_analytics(music_params, melody)

            # Create composition record
            composition = MusicComposition(
                route_id=route.id,
                tempo=music_params['tempo'],
                pitch=music_params['base_pitch'],
                harmony=music_params['harmony_factor'],
                midi_path=midi_path,
                complexity_score=analytics['melodic_complexity'],
                harmonic_richness=analytics['harmonic_richness'],
                duration_seconds=duration_minutes * 60,
                unique_notes=len(set(msg.note for msg in melody if msg.type == 'note_on')),
                musical_key=key,
                scale=scale.value
            )

            db.add(composition)
            await db.commit()
            await db.refresh(composition)

            # Generate route embedding for similarity search based on full path
            embedding_vector = self.generator.generate_route_embedding_multi_segment(path_segments)

            # Find or update route with embedding
            route_result = await db.execute(
                select(Route).where(Route.id == route.id)
            )
            existing_route = route_result.scalar_one_or_none()

            if existing_route:
                existing_route.route_embedding = json.dumps(embedding_vector.tolist())
                await db.commit()
            
            # Log music analytics to DuckDB
            self.analytics.log_music_analytics(
                route_id=route.id,
                tempo=music_params['tempo'],
                key=key,
                scale=scale.value,
                duration_seconds=duration_minutes * 60,
                note_count=composition.unique_notes,
                harmony_complexity=analytics['harmonic_richness'],
                genre=music_style.value,
                embedding_vector=embedding_vector.tolist()
            )
            
            # Cache the music data in Redis
            music_data = {
                "composition_id": composition.id,
                "tempo": music_params['tempo'],
                "key": key,
                "scale": scale.value,
                "midi_path": midi_path,
                "complexity_score": analytics['melodic_complexity'],
                "duration_seconds": duration_minutes * 60
            }
            self.cache.set_route_music(origin_code, destination_code, music_data)
            
            # Log performance metrics
            execution_time = (time.time() - start_time) * 1000
            self.analytics.log_performance_metric(
                "music_generation_complete",
                execution_time,
                True,
                metadata={
                    "origin": origin_code,
                    "destination": destination_code,
                    "segments": len(path_segments),
                    "distance_km": total_distance
                }
            )
            
            logger.info(f"Music generation completed in {execution_time:.2f}ms")

            return composition, midi_path, analytics
            
        except Exception as e:
            # Log error metrics
            execution_time = (time.time() - start_time) * 1000
            self.analytics.log_performance_metric(
                "music_generation_error",
                execution_time,
                False,
                error_message=str(e),
                metadata={
                    "origin": origin_code,
                    "destination": destination_code
                }
            )
            logger.error(f"Music generation failed: {e}")
            raise

    def generate_music_with_realtime_updates(
        self,
        origin_code: str,
        destination_code: str,
        user_id: str,
        style: str = "classical",
        tempo: int = 120
    ) -> Tuple[MusicComposition, str, Dict]:
        """Generate music with real-time progress updates"""
        # This is a wrapper around the main generation method with progress updates
        return self.generate_music_from_route(origin_code, destination_code, user_id, style, tempo)

    def publish_generation_progress(
        self,
        user_id: str,
        generation_id: str,
        progress: float,
        status: str,
        current_step: str = ""
    ):
        """Publish generation progress via Redis"""
        if self.publisher:
            self.publisher.publish_generation_progress(
                generation_id=generation_id,
                user_id=user_id,
                progress=progress,
                status=status,
                current_step=current_step
            )

    def publish_music_update(
        self,
        session_id: str,
        user_id: str,
        update_type: str,
        music_data: Dict
    ):
        """Publish music update via Redis"""
        if self.publisher:
            self.publisher.publish_music_update_real_time(
                session_id=session_id,
                user_id=user_id,
                update_type=update_type,
                music_data=music_data
            )

    def store_vector_with_realtime_sync(
        self,
        composition_id: int,
        route_id: int,
        origin: str,
        destination: str,
        genre: str,
        tempo: int,
        pitch: float,
        harmony: float,
        complexity: float,
        vector: np.ndarray
    ) -> bool:
        """Store vector with real-time synchronization"""
        try:
            # Import here to avoid circular imports
            from app.services.vector_service import MusicVector

            # Create music vector
            music_vector = MusicVector.from_composition_data(
                tempo=tempo,
                pitch=pitch,
                harmony=harmony,
                complexity=complexity,
                genre=genre
            )

            # Store in FAISS service
            if self.faiss_service:
                success = self.faiss_service.store_music_vector(
                    composition_id=composition_id,
                    route_id=route_id,
                    origin=origin,
                    destination=destination,
                    genre=genre,
                    tempo=tempo,
                    pitch=pitch,
                    harmony=harmony,
                    complexity=complexity,
                    vector=vector,
                    metadata={"music_vector": music_vector.to_json()}
                )
                return success

            return False

        except Exception as e:
            logger.error(f"Error storing vector with real-time sync: {e}")
            return False


# Global service instance
music_generator_service = None


def get_music_generation_service() -> MusicGenerationService:
    """Get the global music generation service instance"""
    global music_generator_service
    if music_generator_service is None:
        music_generator_service = MusicGenerationService()
    return music_generator_service
