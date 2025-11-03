"""
AI Genre Composer - Advanced PyTorch models for genre-based music composition
Uses neural networks to generate genre-specific musical patterns with vector embeddings
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import faiss
from datetime import datetime


class GenreEmbeddingModel(nn.Module):
    """Neural network for learning genre-specific musical embeddings"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 32):
        super(GenreEmbeddingModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        self.genre_decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8)  # 8 genre classes
        )
    
    def forward(self, x):
        embeddings = self.encoder(x)
        genre_logits = self.genre_decoder(embeddings)
        return embeddings, genre_logits


class MusicPatternGenerator(nn.Module):
    """LSTM-based model for generating musical note sequences"""
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, output_dim: int = 12):
        super(MusicPatternGenerator, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        probabilities = self.softmax(output)
        return probabilities, hidden


class AIGenreComposer:
    """
    Advanced AI composer using PyTorch for genre-based music generation
    Supports multiple genres with learned patterns and characteristics
    """
    
    GENRES = {
        "classical": {
            "tempo_range": (60, 120),
            "scales": ["major", "minor", "dorian"],
            "complexity": 0.8,
            "dynamics": "varied",
            "note_density": 0.8,
            "octave_range": 3,
            "base_note": 60,
            "rhythm_pattern": "flowing"
        },
        "jazz": {
            "tempo_range": (100, 180),
            "scales": ["dorian", "mixolydian", "blues"],
            "complexity": 0.9,
            "dynamics": "syncopated",
            "note_density": 1.2,
            "octave_range": 2,
            "base_note": 55,
            "rhythm_pattern": "swing"
        },
        "electronic": {
            "tempo_range": (120, 140),
            "scales": ["minor", "phrygian", "locrian"],
            "complexity": 0.7,
            "dynamics": "steady",
            "note_density": 1.5,
            "octave_range": 4,
            "base_note": 48,
            "rhythm_pattern": "pulse"
        },
        "ambient": {
            "tempo_range": (60, 90),
            "scales": ["lydian", "major", "pentatonic"],
            "complexity": 0.5,
            "dynamics": "smooth",
            "note_density": 0.3,
            "octave_range": 3,
            "base_note": 72,
            "rhythm_pattern": "floating"
        },
        "rock": {
            "tempo_range": (110, 140),
            "scales": ["minor", "blues", "pentatonic"],
            "complexity": 0.6,
            "dynamics": "driving",
            "note_density": 1.0,
            "octave_range": 2,
            "base_note": 52,
            "rhythm_pattern": "power"
        },
        "world": {
            "tempo_range": (80, 130),
            "scales": ["phrygian", "harmonic_minor", "pentatonic"],
            "complexity": 0.7,
            "dynamics": "ethnic",
            "note_density": 0.9,
            "octave_range": 2,
            "base_note": 57,
            "rhythm_pattern": "exotic"
        },
        "cinematic": {
            "tempo_range": (70, 110),
            "scales": ["minor", "dorian", "lydian"],
            "complexity": 0.85,
            "dynamics": "dramatic",
            "note_density": 0.6,
            "octave_range": 4,
            "base_note": 48,
            "rhythm_pattern": "epic"
        },
        "lofi": {
            "tempo_range": (70, 95),
            "scales": ["minor", "dorian", "pentatonic"],
            "complexity": 0.4,
            "dynamics": "relaxed",
            "note_density": 0.5,
            "octave_range": 2,
            "base_note": 60,
            "rhythm_pattern": "chill"
        }
    }
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.embedding_model = GenreEmbeddingModel().to(self.device)
        self.pattern_generator = MusicPatternGenerator().to(self.device)
        
        # Set to evaluation mode (in production, load pre-trained weights)
        self.embedding_model.eval()
        self.pattern_generator.eval()
        
        # Initialize with random weights (in production, load trained weights)
        self._initialize_weights()
        
        # Initialize FAISS index for vector similarity search (128D embeddings)
        self.embedding_dim = 128
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.composition_metadata = []  # Store metadata for indexed compositions
        
        print(f"✅ AI Composer initialized with FAISS v{faiss.__version__} vector search (device: {self.device})")
        print(f"🔍 Vector embeddings enabled for AI Composer, VR Experiences, and Travel Logs")
    
    def _initialize_weights(self):
        """Initialize model weights (placeholder for pre-trained weights)"""
        for model in [self.embedding_model, self.pattern_generator]:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LSTM):
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.zeros_(param)
    
    def extract_route_features(
        self,
        distance: float,
        latitude_range: float,
        longitude_range: float,
        direction: str,
        time_of_day: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract numerical features from route characteristics
        
        Args:
            distance: Flight distance in km
            latitude_range: Latitude difference
            longitude_range: Longitude difference
            direction: Cardinal direction (N, S, E, W, NE, etc.)
            time_of_day: Optional time of day (morning, afternoon, evening, night)
        
        Returns:
            Feature tensor for model input
        """
        # Normalize features
        features = [
            distance / 20000.0,  # Normalize by max earth distance
            latitude_range / 180.0,
            longitude_range / 360.0,
            1.0 if 'N' in direction else 0.0,
            1.0 if 'S' in direction else 0.0,
            1.0 if 'E' in direction else 0.0,
            1.0 if 'W' in direction else 0.0,
            np.sin(2 * np.pi * distance / 20000.0),  # Cyclical encoding
            np.cos(2 * np.pi * distance / 20000.0),
            0.5  # Placeholder for additional features
        ]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def generate_genre_composition(
        self,
        genre: str,
        route_features: Dict[str, any],
        duration: int = 30,
        tempo: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Generate a genre-specific composition using AI models
        
        Args:
            genre: Musical genre
            route_features: Route characteristics
            duration: Composition duration in seconds
            tempo: Optional tempo override
        
        Returns:
            Composition data with AI-generated patterns
        """
        if genre not in self.GENRES:
            genre = "ambient"  # Default fallback
        
        genre_config = self.GENRES[genre]
        
        # Use seed for route-influenced but still varied compositions
        # Add timestamp component to make each generation unique
        seed = route_features.get("seed", None)
        if seed is not None:
            # Mix seed with current time to get variation while maintaining route influence
            import time
            time_component = int(time.time() * 1000) % 10000  # Last 4 digits of milliseconds
            varied_seed = (seed + time_component) % 2147483647
            np.random.seed(varied_seed)
            print(f"Using varied seed: {varied_seed} (base: {seed}, time: {time_component})")
        
        # Extract features
        feature_tensor = self.extract_route_features(
            distance=route_features.get("distance", 5000),
            latitude_range=route_features.get("latitude_range", 30),
            longitude_range=route_features.get("longitude_range", 50),
            direction=route_features.get("direction", "E")
        )
        
        # Generate embeddings
        with torch.no_grad():
            embeddings, genre_logits = self.embedding_model(feature_tensor)
            genre_probs = torch.softmax(genre_logits, dim=-1)
        
        # Generate note sequence
        # Calculate sequence length based on genre complexity and desired duration
        # Use note density to determine how many notes we need
        note_density = genre_config.get("note_density", 1.0)
        base_notes_per_second = 4.0  # Increased base rate for better coverage
        notes_per_second = base_notes_per_second * note_density
        
        # Generate enough notes to fill the target duration with proper buffer
        # Ensure minimum duration of 30 seconds for VR experiences
        target_duration = max(duration, 30.0)
        sequence_length = max(int(target_duration * notes_per_second), 120)  # Minimum 120 notes
        
        print(f"Generating {sequence_length} notes for {target_duration}s {genre} composition (density: {note_density})")
        
        note_sequence = self._generate_note_sequence(
            embeddings,
            sequence_length,
            genre_config,
            target_duration=target_duration  # Pass target duration
        )
        
        # Verify composition length and extend if needed
        if note_sequence:
            actual_duration = note_sequence[-1]["time"] + note_sequence[-1]["duration"]
            print(f"Generated {len(note_sequence)} notes, {actual_duration:.2f}s duration for {genre} composition")
            
            # Ensure we meet the target duration - minimum 30 seconds for VR experience
            target_min_duration = max(duration, 30.0)
            if actual_duration < target_min_duration:
                print(f"Extending composition from {actual_duration:.2f}s to {target_min_duration}s")
                # Add additional notes or extend duration to reach target
                extension_needed = target_min_duration - actual_duration
                if extension_needed > 5.0:
                    # Add more notes if we need significant extension
                    additional_notes = self._generate_note_sequence(
                        embeddings,
                        max(20, int(extension_needed * 2)),  # Generate more notes
                        genre_config,
                        target_duration=extension_needed,
                        start_time=actual_duration
                    )
                    note_sequence.extend(additional_notes)
                    actual_duration = note_sequence[-1]["time"] + note_sequence[-1]["duration"]
                else:
                    # Just extend the last note for small gaps
                    note_sequence[-1]["duration"] += extension_needed
                    actual_duration = target_min_duration
        
        # Determine tempo
        if tempo is None:
            tempo_min, tempo_max = genre_config["tempo_range"]
            tempo = int(tempo_min + (tempo_max - tempo_min) * route_features.get("distance", 5000) / 10000)
            tempo = max(tempo_min, min(tempo_max, tempo))
        
        # Select scale based on genre with variation
        scale = np.random.choice(genre_config["scales"])
        
        # Reset random seed to avoid affecting other operations
        np.random.seed(None)
        
        composition = {
            "genre": genre,
            "tempo": tempo,
            "scale": scale,
            "key": self._select_key_from_embeddings(embeddings),
            "note_sequence": note_sequence,
            "embeddings": embeddings.cpu().numpy().tolist(),
            "genre_confidence": genre_probs.cpu().numpy().tolist(),
            "complexity": genre_config["complexity"],
            "dynamics": genre_config["dynamics"],
            "duration": duration,
            "ai_generated": True,
            "seed": seed
        }
        
        # Generate and store vector embedding for similarity search
        vector_embedding = self.generate_composition_embedding(composition)
        composition["vector_embedding"] = vector_embedding.tolist()
        
        # Automatically add to FAISS index for future similarity searches
        self.add_composition_to_index(composition)
        
        # Sync to DuckDB for analytics (non-blocking)
        try:
            from app.services.duckdb_sync_service import duckdb_sync
            duckdb_sync.sync_ai_composer_embedding(composition)
        except Exception as e:
            pass  # Don't fail if DuckDB sync fails
        
        print(f"✅ Generated composition with 128D vector embedding")
        
        return composition
    
    def _generate_note_sequence(
        self,
        embeddings: torch.Tensor,
        length: int,
        genre_config: Dict,
        target_duration: Optional[float] = None,
        start_time: float = 0.0
    ) -> List[Dict[str, any]]:
        """Generate a sequence of notes using the LSTM pattern generator"""
        notes = []
        
        # Ensure minimum length
        length = max(length, 60)
        
        # Prepare input for LSTM
        lstm_input = embeddings.unsqueeze(1).repeat(1, length, 1)
        
        with torch.no_grad():
            note_probs, _ = self.pattern_generator(lstm_input)
        
        # Scale definitions for different modes
        scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "phrygian": [0, 1, 3, 5, 7, 8, 10],
            "lydian": [0, 2, 4, 6, 7, 9, 11],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "pentatonic": [0, 2, 4, 7, 9],
            "blues": [0, 3, 5, 6, 7, 10],
            "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
            "locrian": [0, 1, 3, 5, 6, 8, 10]
        }
        
        # Get scale for this genre
        scale_name = genre_config["scales"][0] if genre_config["scales"] else "major"
        scale_intervals = scales.get(scale_name, scales["major"])
        
        # Genre-specific base note and octave range
        base_note = genre_config.get("base_note", 60)
        octave_range = genre_config.get("octave_range", 2)
        
        # Generate notes with proper timing
        current_time = start_time
        
        # Genre-specific note density affects timing
        note_density = genre_config.get("note_density", 1.0)
        rhythm_pattern = genre_config.get("rhythm_pattern", "standard")
        
        # Calculate time per note to reach target duration
        if target_duration and target_duration > 0:
            # Calculate average duration per note to fill target time
            # Use a more conservative approach to ensure we reach the target
            base_avg_duration = target_duration / (length * 0.8)  # Use 80% of notes for timing
            avg_duration_per_note = max(base_avg_duration, 0.1)  # Minimum 0.1s per note
        else:
            avg_duration_per_note = 0.25  # Default 0.25s per note
        
        for i in range(length):
            probs = note_probs[0, i].cpu().numpy()
            
            # Select note from scale
            scale_degree = np.random.choice(len(scale_intervals), p=probs[:len(scale_intervals)] / probs[:len(scale_intervals)].sum())
            octave_offset = np.random.choice(octave_range) * 12
            pitch = base_note + scale_intervals[scale_degree] + octave_offset
            
            # Vary note duration based on genre complexity and rhythm pattern
            complexity = genre_config["complexity"]
            
            # Rhythm pattern affects note timing
            if rhythm_pattern == "swing":
                # Jazz swing - alternate long/short
                swing_factor = 1.5 if i % 2 == 0 else 0.7
                duration = avg_duration_per_note * swing_factor
            elif rhythm_pattern == "pulse":
                # Electronic - steady pulse with occasional longer notes
                duration = avg_duration_per_note * (1.0 if i % 8 != 7 else 2.0)
            elif rhythm_pattern == "floating":
                # Ambient - very long, overlapping notes
                duration = avg_duration_per_note * (2.0 + np.random.random() * 3.0)
            elif rhythm_pattern == "power":
                # Rock - driving rhythm with emphasis
                duration = avg_duration_per_note * (1.0 if i % 4 != 0 else 1.5)
            elif rhythm_pattern == "exotic":
                # World - irregular patterns
                duration = avg_duration_per_note * (0.5 + np.random.random() * 1.5)
            elif rhythm_pattern == "epic":
                # Cinematic - building intensity
                intensity = i / length
                duration = avg_duration_per_note * (0.5 + intensity * 2.0)
            elif rhythm_pattern == "chill":
                # Lofi - relaxed, slightly irregular
                duration = avg_duration_per_note * (0.8 + np.random.random() * 0.6)
            else:
                # Classical/flowing - complexity-based variation
                if complexity > 0.7:
                    duration = avg_duration_per_note * (0.5 + np.random.random() * 1.0)
                elif complexity > 0.5:
                    duration = avg_duration_per_note * (0.7 + np.random.random() * 0.6)
                else:
                    duration = avg_duration_per_note * (0.8 + np.random.random() * 0.4)
            
            # Ensure minimum note duration
            duration = max(duration, 0.05)
            
            # Calculate velocity based on genre dynamics and rhythm pattern
            if genre_config["dynamics"] == "varied":  # Classical
                velocity = int(60 + 40 * np.sin(i / length * 2 * np.pi))
            elif genre_config["dynamics"] == "syncopated":  # Jazz
                # Jazz emphasis on off-beats
                if i % 8 in [1, 3, 6]:
                    velocity = 95  # Strong off-beats
                elif i % 8 in [0, 4]:
                    velocity = 75  # Weaker on-beats
                else:
                    velocity = 85
            elif genre_config["dynamics"] == "steady":  # Electronic
                # Electronic - consistent with occasional accents
                velocity = 85 if i % 16 == 0 else 80
            elif genre_config["dynamics"] == "dramatic":  # Cinematic
                # Cinematic - building intensity
                base_velocity = int(40 + 70 * (i / length))
                velocity = base_velocity + int(20 * np.sin(i / 8))  # Add drama waves
            elif genre_config["dynamics"] == "driving":  # Rock
                # Rock - strong downbeats
                if i % 4 == 0:
                    velocity = 100  # Strong downbeat
                elif i % 4 == 2:
                    velocity = 90   # Backbeat
                else:
                    velocity = 75
            elif genre_config["dynamics"] == "smooth":  # Ambient
                velocity = int(50 + 20 * np.sin(i / length * np.pi))  # Gentle waves
            elif genre_config["dynamics"] == "ethnic":  # World
                # World - irregular accents
                accent_pattern = [100, 70, 85, 60, 95, 75, 80, 65]
                velocity = accent_pattern[i % len(accent_pattern)]
            elif genre_config["dynamics"] == "relaxed":  # Lofi
                velocity = int(60 + 15 * np.sin(i / 12))  # Subtle variation
            else:
                velocity = 75
            
            # Add some velocity variation
            velocity = max(40, min(127, velocity + np.random.randint(-10, 10)))
            
            notes.append({
                "pitch": pitch,
                "velocity": velocity,
                "duration": duration,
                "time": current_time
            })
            
            # Advance time
            current_time += duration
        
        # Ensure minimum composition length
        if notes:
            final_time = notes[-1]["time"] + notes[-1]["duration"]
            min_duration = max(target_duration if target_duration else 30.0, 30.0)  # Ensure 30s minimum
            
            if final_time < min_duration:
                # Extend last note to reach minimum duration
                notes[-1]["duration"] = min_duration - notes[-1]["time"]
                print(f"Extended final note to reach {min_duration}s minimum duration")
        
        return notes
    
    def _select_key_from_embeddings(self, embeddings: torch.Tensor) -> str:
        """Select musical key based on embedding values"""
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        embedding_sum = embeddings.sum().item()
        key_index = int(abs(embedding_sum * 10)) % 12
        return keys[key_index]
    
    def generate_composition_embedding(
        self,
        composition: Dict[str, any]
    ) -> np.ndarray:
        """
        Generate a 128D vector embedding for a composition using musical features
        
        Args:
            composition: Composition data with note_sequence, tempo, genre, etc.
        
        Returns:
            128D numpy array embedding
        """
        # Extract musical features from composition
        note_sequence = composition.get("note_sequence", [])
        
        if not note_sequence:
            # Return zero embedding for empty compositions
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Feature extraction
        pitches = [note["pitch"] for note in note_sequence]
        velocities = [note["velocity"] for note in note_sequence]
        durations = [note["duration"] for note in note_sequence]
        
        # Statistical features (32 dimensions)
        pitch_features = [
            np.mean(pitches), np.std(pitches), np.min(pitches), np.max(pitches),
            np.median(pitches), np.percentile(pitches, 25), np.percentile(pitches, 75)
        ]
        
        velocity_features = [
            np.mean(velocities), np.std(velocities), np.min(velocities), np.max(velocities)
        ]
        
        duration_features = [
            np.mean(durations), np.std(durations), np.min(durations), np.max(durations)
        ]
        
        # Melodic contour features (16 dimensions)
        pitch_intervals = np.diff(pitches)
        contour_features = [
            np.mean(np.abs(pitch_intervals)) if len(pitch_intervals) > 0 else 0,
            np.std(pitch_intervals) if len(pitch_intervals) > 0 else 0,
            np.sum(pitch_intervals > 0) / len(pitch_intervals) if len(pitch_intervals) > 0 else 0,  # Ascending ratio
            np.sum(pitch_intervals < 0) / len(pitch_intervals) if len(pitch_intervals) > 0 else 0,  # Descending ratio
        ]
        
        # Rhythmic features (16 dimensions)
        duration_ratios = np.array(durations[1:]) / np.array(durations[:-1]) if len(durations) > 1 else [1.0]
        rhythm_features = [
            np.mean(duration_ratios), np.std(duration_ratios),
            composition.get("tempo", 120) / 200.0,  # Normalized tempo
            len(note_sequence) / 200.0  # Note density
        ]
        
        # Genre encoding (8 dimensions - one-hot for 8 genres)
        genre = composition.get("genre", "ambient")
        genre_names = list(self.GENRES.keys())
        genre_encoding = [1.0 if genre == g else 0.0 for g in genre_names]
        
        # Complexity features (8 dimensions)
        complexity_features = [
            composition.get("complexity", 0.5),
            composition.get("melodic_complexity", 0.5),
            composition.get("harmonic_complexity", 0.5),
            composition.get("rhythmic_complexity", 0.5)
        ]
        
        # Combine all features
        all_features = (
            pitch_features + velocity_features + duration_features +
            contour_features + rhythm_features + genre_encoding + complexity_features
        )
        
        # Pad or truncate to exactly 128 dimensions
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        feature_count = min(len(all_features), self.embedding_dim)
        embedding[:feature_count] = all_features[:feature_count]
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def add_composition_to_index(
        self,
        composition: Dict[str, any],
        composition_id: Optional[str] = None
    ) -> str:
        """
        Add a composition to the FAISS index for similarity search
        
        Args:
            composition: Composition data
            composition_id: Optional ID for the composition
        
        Returns:
            Composition ID
        """
        # Generate embedding
        embedding = self.generate_composition_embedding(composition)
        
        # Add to FAISS index
        self.faiss_index.add(np.array([embedding]))
        
        # Store metadata
        if composition_id is None:
            composition_id = f"comp_{len(self.composition_metadata)}_{datetime.now().timestamp()}"
        
        metadata = {
            "id": composition_id,
            "genre": composition.get("genre"),
            "tempo": composition.get("tempo"),
            "complexity": composition.get("complexity"),
            "duration": composition.get("duration"),
            "timestamp": datetime.now().isoformat()
        }
        self.composition_metadata.append(metadata)
        
        print(f"✅ Added composition to FAISS index: {composition_id} (total: {self.faiss_index.ntotal})")
        
        return composition_id
    
    def find_similar_compositions(
        self,
        query_composition: Dict[str, any],
        k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Find similar compositions using vector similarity search
        
        Args:
            query_composition: Composition to find similar matches for
            k: Number of similar compositions to return
        
        Returns:
            List of similar compositions with similarity scores
        """
        if self.faiss_index.ntotal == 0:
            print("⚠️ FAISS index is empty, no similar compositions found")
            return []
        
        # Generate embedding for query
        query_embedding = self.generate_composition_embedding(query_composition)
        
        # Search FAISS index
        k = min(k, self.faiss_index.ntotal)  # Don't search for more than available
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.composition_metadata):
                similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                result = {
                    **self.composition_metadata[idx],
                    "similarity_score": float(similarity_score),
                    "distance": float(distance),
                    "rank": i + 1
                }
                results.append(result)
        
        print(f"🔍 Found {len(results)} similar compositions using vector search")
        
        return results
    
    def get_genre_recommendations(
        self,
        route_features: Dict[str, any],
        use_vector_search: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get genre recommendations based on route characteristics
        Now enhanced with vector similarity search
        
        Args:
            route_features: Route characteristics
            use_vector_search: Whether to use vector search for recommendations
        
        Returns:
            List of (genre, confidence) tuples sorted by confidence
        """
        feature_tensor = self.extract_route_features(
            distance=route_features.get("distance", 5000),
            latitude_range=route_features.get("latitude_range", 30),
            longitude_range=route_features.get("longitude_range", 50),
            direction=route_features.get("direction", "E")
        )
        
        with torch.no_grad():
            embeddings, genre_logits = self.embedding_model(feature_tensor)
            genre_probs = torch.softmax(genre_logits, dim=-1).cpu().numpy()[0]
        
        # Base recommendations from neural network
        genre_names = list(self.GENRES.keys())
        recommendations = list(zip(genre_names, genre_probs))
        
        # Enhance with vector similarity search if enabled and index has data
        if use_vector_search and self.faiss_index.ntotal > 0:
            print("🔍 Using vector similarity search for enhanced recommendations")
            
            # Create a dummy composition with route features
            dummy_composition = {
                "genre": "ambient",
                "tempo": 100,
                "complexity": 0.5,
                "note_sequence": [],
                "route_features": route_features
            }
            
            # Find similar compositions
            similar = self.find_similar_compositions(dummy_composition, k=5)
            
            # Boost genres that appear in similar compositions
            if similar:
                genre_boost = {genre: 0.0 for genre in genre_names}
                for comp in similar:
                    comp_genre = comp.get("genre")
                    if comp_genre in genre_boost:
                        genre_boost[comp_genre] += comp["similarity_score"]
                
                # Normalize boosts
                max_boost = max(genre_boost.values()) if genre_boost else 1.0
                if max_boost > 0:
                    genre_boost = {g: b / max_boost * 0.3 for g, b in genre_boost.items()}  # 30% weight
                
                # Apply boosts to recommendations
                recommendations = [
                    (genre, min(1.0, conf * 0.7 + genre_boost.get(genre, 0.0)))
                    for genre, conf in recommendations
                ]
                
                print(f"✅ Enhanced recommendations with vector search (boosted: {list(genre_boost.keys())})")
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def blend_genres(
        self,
        primary_genre: str,
        secondary_genre: str,
        blend_ratio: float = 0.5,
        route_features: Dict[str, any] = None
    ) -> Dict[str, any]:
        """
        Create a blended composition mixing two genres
        
        Args:
            primary_genre: Primary genre
            secondary_genre: Secondary genre
            blend_ratio: Ratio of secondary genre (0.0 to 1.0)
            route_features: Route characteristics
        
        Returns:
            Blended composition data
        """
        if route_features is None:
            route_features = {"distance": 5000, "latitude_range": 30, "longitude_range": 50, "direction": "E"}
        
        # Generate compositions for both genres
        primary_comp = self.generate_genre_composition(primary_genre, route_features)
        secondary_comp = self.generate_genre_composition(secondary_genre, route_features)
        
        # Blend parameters
        blended_tempo = int(
            primary_comp["tempo"] * (1 - blend_ratio) +
            secondary_comp["tempo"] * blend_ratio
        )
        
        # Interleave note sequences
        primary_notes = primary_comp["note_sequence"]
        secondary_notes = secondary_comp["note_sequence"]
        
        blended_notes = []
        for i in range(max(len(primary_notes), len(secondary_notes))):
            if i < len(primary_notes) and (i % 2 == 0 or i >= len(secondary_notes)):
                blended_notes.append(primary_notes[i])
            elif i < len(secondary_notes):
                blended_notes.append(secondary_notes[i])
        
        # Blend complexity and dynamics
        primary_config = self.GENRES.get(primary_genre, self.GENRES["ambient"])
        secondary_config = self.GENRES.get(secondary_genre, self.GENRES["ambient"])
        
        blended_complexity = (
            primary_config["complexity"] * (1 - blend_ratio) +
            secondary_config["complexity"] * blend_ratio
        )
        
        # Choose dynamics based on blend ratio
        blended_dynamics = primary_config["dynamics"] if blend_ratio < 0.5 else secondary_config["dynamics"]
        
        return {
            "genre": f"{primary_genre}_{secondary_genre}_blend",
            "primary_genre": primary_genre,
            "secondary_genre": secondary_genre,
            "blend_ratio": blend_ratio,
            "tempo": blended_tempo,
            "scale": primary_comp["scale"],
            "key": primary_comp["key"],
            "note_sequence": blended_notes,
            "complexity": blended_complexity,
            "dynamics": blended_dynamics,
            "duration": primary_comp.get("duration", 30),
            "ai_generated": True,
            "blended": True
        }
