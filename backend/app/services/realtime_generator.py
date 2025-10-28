"""
Real-time music generation service with streaming and buffering capabilities
"""

from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MusicSegment:
    """Represents a generated music segment"""
    segment_id: str
    timestamp: str
    duration_ms: int
    notes: List[Dict]  # List of {pitch, velocity, duration}
    tempo: int
    genre: str
    coherence_score: float


class RealtimeGenerator:
    """Generates music in real-time from live flight data"""

    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.segment_buffer: List[MusicSegment] = []
        self.is_generating = False
        self.current_state = {
            "last_pitch": 60,
            "last_velocity": 64,
            "last_tempo": 120
        }

    async def generate_realtime_segment(
        self,
        live_data: Dict,
        genre: str = "ambient",
        tempo: int = 120
    ) -> MusicSegment:
        """Generate a music segment from live flight data"""
        segment_id = f"seg_{datetime.utcnow().timestamp()}"

        # Extract parameters from live data
        altitude = live_data.get("altitude", 0)
        speed = live_data.get("speed", 0)
        latitude = live_data.get("latitude", 0)
        longitude = live_data.get("longitude", 0)

        # Generate notes based on flight parameters
        notes = self._generate_notes_from_flight_data(
            altitude, speed, latitude, longitude, genre
        )

        segment = MusicSegment(
            segment_id=segment_id,
            timestamp=datetime.utcnow().isoformat(),
            duration_ms=500,  # 500ms segments for real-time
            notes=notes,
            tempo=tempo,
            genre=genre,
            coherence_score=self._calculate_coherence(notes)
        )

        return segment

    async def stream_music_segments(
        self,
        live_data_stream: AsyncGenerator,
        genre: str = "ambient",
        tempo: int = 120
    ) -> AsyncGenerator[MusicSegment, None]:
        """Stream music segments from live data"""
        self.is_generating = True

        try:
            async for live_data in live_data_stream:
                segment = await self.generate_realtime_segment(
                    live_data, genre, tempo
                )
                self.segment_buffer.append(segment)

                # Maintain buffer size
                if len(self.segment_buffer) > self.buffer_size:
                    self.segment_buffer.pop(0)

                yield segment
        finally:
            self.is_generating = False

    def _generate_notes_from_flight_data(
        self,
        altitude: float,
        speed: float,
        latitude: float,
        longitude: float,
        genre: str
    ) -> List[Dict]:
        """Generate musical notes from flight parameters"""
        notes = []

        # Map altitude to pitch (0-40000 ft -> 36-96 MIDI notes)
        pitch = int(36 + (altitude / 40000) * 60)
        pitch = max(36, min(96, pitch))

        # Map speed to velocity (0-500 knots -> 30-127 velocity)
        velocity = int(30 + (speed / 500) * 97)
        velocity = max(30, min(127, velocity))

        # Generate 4 notes per segment
        for i in range(4):
            # Add variation based on position
            note_pitch = pitch + (i - 1) * 2
            note_velocity = velocity + (i % 2) * 10

            notes.append({
                "pitch": max(0, min(127, note_pitch)),
                "velocity": max(0, min(127, note_velocity)),
                "duration": 125,  # 125ms per note
                "offset": i * 125
            })

        # Apply genre-specific modifications
        notes = self._apply_genre_style(notes, genre)

        return notes

    def _apply_genre_style(self, notes: List[Dict], genre: str) -> List[Dict]:
        """Apply genre-specific modifications to notes"""
        if genre == "classical":
            # Classical: more structured, longer notes
            for note in notes:
                note["duration"] = 250
                note["velocity"] = min(127, note["velocity"] + 10)

        elif genre == "jazz":
            # Jazz: syncopated rhythm, varied velocities
            for i, note in enumerate(notes):
                if i % 2 == 0:
                    note["offset"] += 30
                note["velocity"] = note["velocity"] + (i % 3) * 15

        elif genre == "electronic":
            # Electronic: shorter notes, consistent velocity
            for note in notes:
                note["duration"] = 100
                note["velocity"] = 100

        elif genre == "ambient":
            # Ambient: longer sustain, lower velocity
            for note in notes:
                note["duration"] = 300
                note["velocity"] = max(30, note["velocity"] - 20)

        return notes

    def _calculate_coherence(self, notes: List[Dict]) -> float:
        """Calculate coherence score for generated notes"""
        if not notes:
            return 0.0

        # Check for smooth pitch transitions
        pitch_diffs = []
        for i in range(1, len(notes)):
            diff = abs(notes[i]["pitch"] - notes[i-1]["pitch"])
            pitch_diffs.append(diff)

        # Penalize large jumps
        avg_diff = sum(pitch_diffs) / len(pitch_diffs) if pitch_diffs else 0
        coherence = 1.0 - min(avg_diff / 12, 1.0)  # Normalize by octave

        # Check velocity consistency
        velocities = [n["velocity"] for n in notes]
        velocity_variance = sum((v - sum(velocities)/len(velocities))**2 for v in velocities) / len(velocities)
        velocity_score = 1.0 - min(velocity_variance / 1000, 1.0)

        return (coherence + velocity_score) / 2

    def get_buffered_segments(self) -> List[MusicSegment]:
        """Get all buffered segments"""
        return self.segment_buffer.copy()

    def clear_buffer(self):
        """Clear the segment buffer"""
        self.segment_buffer.clear()

    def stop_generation(self):
        """Stop real-time generation"""
        self.is_generating = False


class MusicBuffer:
    """Manages buffering for continuous music playback"""

    def __init__(self, buffer_duration_ms: int = 2000):
        self.buffer_duration_ms = buffer_duration_ms
        self.segments: List[MusicSegment] = []
        self.playback_position = 0
        self.is_playing = False

    def add_segment(self, segment: MusicSegment):
        """Add a segment to the buffer"""
        self.segments.append(segment)

    def get_playback_data(self, duration_ms: int = 500) -> Dict:
        """Get data for playback"""
        if not self.segments:
            return {"notes": [], "duration": 0}

        # Collect notes for playback duration
        playback_notes = []
        current_time = 0

        for segment in self.segments:
            if current_time >= duration_ms:
                break

            for note in segment.notes:
                if current_time + note["offset"] < duration_ms:
                    playback_notes.append(note)

            current_time += segment.duration_ms

        return {
            "notes": playback_notes,
            "duration": min(current_time, duration_ms),
            "segments_used": len(self.segments)
        }

    def is_buffer_healthy(self) -> bool:
        """Check if buffer has enough data"""
        total_duration = sum(s.duration_ms for s in self.segments)
        return total_duration >= self.buffer_duration_ms

    def get_buffer_status(self) -> Dict:
        """Get buffer status information"""
        total_duration = sum(s.duration_ms for s in self.segments)
        return {
            "segments_count": len(self.segments),
            "total_duration_ms": total_duration,
            "buffer_health": "healthy" if self.is_buffer_healthy() else "low",
            "buffer_percentage": min(100, (total_duration / self.buffer_duration_ms) * 100)
        }

    def clear(self):
        """Clear the buffer"""
        self.segments.clear()
        self.playback_position = 0


class StreamingCompositionEngine:
    """Combines real-time generation with buffering for streaming"""

    def __init__(self):
        self.generator = RealtimeGenerator()
        self.buffer = MusicBuffer()
        self.is_streaming = False

    async def start_streaming(
        self,
        live_data_stream: AsyncGenerator,
        genre: str = "ambient",
        tempo: int = 120
    ):
        """Start streaming music composition"""
        self.is_streaming = True

        try:
            async for segment in self.generator.stream_music_segments(
                live_data_stream, genre, tempo
            ):
                self.buffer.add_segment(segment)

                # Maintain buffer size
                if len(self.buffer.segments) > 20:
                    self.buffer.segments.pop(0)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.is_streaming = False

    def get_stream_status(self) -> Dict:
        """Get current streaming status"""
        return {
            "is_streaming": self.is_streaming,
            "buffer_status": self.buffer.get_buffer_status(),
            "generator_active": self.generator.is_generating
        }

    def stop_streaming(self):
        """Stop streaming"""
        self.is_streaming = False
        self.generator.stop_generation()
