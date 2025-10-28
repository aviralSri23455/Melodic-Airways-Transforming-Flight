"""
Advanced genre-based music composition service
Implements genre-specific AI models for diverse musical styles
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenreProfile:
    """Musical characteristics for a specific genre"""
    name: str
    tempo_range: Tuple[int, int]
    scale_preferences: List[str]
    rhythm_pattern: str
    harmony_complexity: float
    note_density: float
    dynamics_range: Tuple[int, int]


class GenreStyleModel(nn.Module):
    """Neural network for genre-specific style generation"""

    def __init__(self, input_dim=10, hidden_dim=128, output_dim=64):
        super(GenreStyleModel, self).__init__()
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

    def forward(self, x):
        return self.encoder(x)


class GenreComposer:
    """Advanced genre-based composition engine"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.genre_models = {}
        self._initialize_genre_profiles()
        self._load_genre_models()

    def _initialize_genre_profiles(self):
        """Initialize genre-specific musical profiles"""
        self.genre_profiles = {
            'classical': GenreProfile(
                name='classical',
                tempo_range=(60, 120),
                scale_preferences=['major', 'minor', 'dorian'],
                rhythm_pattern='structured',
                harmony_complexity=0.8,
                note_density=0.6,
                dynamics_range=(40, 100)
            ),
            'jazz': GenreProfile(
                name='jazz',
                tempo_range=(100, 180),
                scale_preferences=['dorian', 'mixolydian', 'blues'],
                rhythm_pattern='syncopated',
                harmony_complexity=0.9,
                note_density=0.7,
                dynamics_range=(50, 110)
            ),
            'electronic': GenreProfile(
                name='electronic',
                tempo_range=(120, 140),
                scale_preferences=['minor', 'phrygian'],
                rhythm_pattern='steady',
                harmony_complexity=0.5,
                note_density=0.9,
                dynamics_range=(80, 120)
            ),
            'ambient': GenreProfile(
                name='ambient',
                tempo_range=(60, 90),
                scale_preferences=['pentatonic', 'lydian'],
                rhythm_pattern='flowing',
                harmony_complexity=0.6,
                note_density=0.3,
                dynamics_range=(30, 70)
            ),
            'rock': GenreProfile(
                name='rock',
                tempo_range=(110, 140),
                scale_preferences=['minor', 'pentatonic'],
                rhythm_pattern='driving',
                harmony_complexity=0.5,
                note_density=0.7,
                dynamics_range=(70, 120)
            ),
            'blues': GenreProfile(
                name='blues',
                tempo_range=(80, 120),
                scale_preferences=['blues', 'pentatonic'],
                rhythm_pattern='shuffle',
                harmony_complexity=0.6,
                note_density=0.5,
                dynamics_range=(50, 90)
            ),
            'world': GenreProfile(
                name='world',
                tempo_range=(90, 130),
                scale_preferences=['phrygian', 'locrian', 'exotic'],
                rhythm_pattern='ethnic',
                harmony_complexity=0.7,
                note_density=0.6,
                dynamics_range=(40, 100)
            ),
            'cinematic': GenreProfile(
                name='cinematic',
                tempo_range=(70, 110),
                scale_preferences=['minor', 'lydian', 'dorian'],
                rhythm_pattern='dramatic',
                harmony_complexity=0.85,
                note_density=0.5,
                dynamics_range=(20, 127)
            )
        }

    def _load_genre_models(self):
        """Load pre-trained genre-specific models"""
        for genre_name in self.genre_profiles.keys():
            model = GenreStyleModel().to(self.device)
            # In production, load pre-trained weights here
            # model.load_state_dict(torch.load(f'models/{genre_name}_model.pth'))
            self.genre_models[genre_name] = model

    def generate_genre_composition(
        self,
        genre: str,
        route_features: np.ndarray,
        duration_seconds: int = 180
    ) -> Dict:
        """
        Generate genre-specific composition
        
        Args:
            genre: Musical genre
            route_features: Route characteristics as numpy array
            duration_seconds: Composition duration
            
        Returns:
            Genre-specific composition parameters
        """
        if genre not in self.genre_profiles:
            logger.warning(f"Unknown genre: {genre}, using ambient")
            genre = 'ambient'

        profile = self.genre_profiles[genre]
        model = self.genre_models[genre]

        # Generate genre-specific features using AI model
        features_tensor = torch.tensor(route_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            style_embedding = model(features_tensor)

        style_vector = style_embedding.cpu().numpy().flatten()

        # Generate composition parameters based on genre profile
        composition = self._create_genre_composition(
            profile,
            style_vector,
            duration_seconds
        )

        return composition

    def _create_genre_composition(
        self,
        profile: GenreProfile,
        style_vector: np.ndarray,
        duration_seconds: int
    ) -> Dict:
        """Create composition based on genre profile and style vector"""
        
        # Determine tempo from genre range and style
        tempo_range = profile.tempo_range[1] - profile.tempo_range[0]
        tempo = int(profile.tempo_range[0] + (style_vector[0] + 1) / 2 * tempo_range)

        # Generate melody structure
        melody_structure = self._generate_melody_structure(
            profile,
            style_vector,
            duration_seconds
        )

        # Generate harmony progression
        harmony_progression = self._generate_harmony_progression(
            profile,
            style_vector,
            duration_seconds
        )

        # Generate rhythm pattern
        rhythm_pattern = self._generate_rhythm_pattern(
            profile,
            style_vector,
            duration_seconds
        )

        return {
            'genre': profile.name,
            'tempo': tempo,
            'melody_structure': melody_structure,
            'harmony_progression': harmony_progression,
            'rhythm_pattern': rhythm_pattern,
            'dynamics': self._generate_dynamics(profile, style_vector),
            'articulation': self._get_genre_articulation(profile),
            'duration_seconds': duration_seconds
        }

    def _generate_melody_structure(
        self,
        profile: GenreProfile,
        style_vector: np.ndarray,
        duration_seconds: int
    ) -> List[Dict]:
        """Generate genre-specific melody structure"""
        
        num_phrases = max(4, int(duration_seconds / 15))  # ~15 seconds per phrase
        melody = []

        for i in range(num_phrases):
            # Base pitch influenced by style vector
            base_pitch = 60 + int(style_vector[i % len(style_vector)] * 24)
            
            # Note density based on genre
            notes_per_phrase = int(profile.note_density * 32)  # Up to 32 notes per phrase
            
            phrase = {
                'phrase_id': i,
                'base_pitch': base_pitch,
                'num_notes': notes_per_phrase,
                'contour': self._get_melodic_contour(profile, style_vector, i),
                'scale': profile.scale_preferences[i % len(profile.scale_preferences)]
            }
            melody.append(phrase)

        return melody

    def _generate_harmony_progression(
        self,
        profile: GenreProfile,
        style_vector: np.ndarray,
        duration_seconds: int
    ) -> List[Dict]:
        """Generate genre-specific harmony progression"""
        
        num_chords = max(4, int(duration_seconds / 4))  # ~4 seconds per chord
        progression = []

        # Genre-specific chord progressions
        chord_templates = self._get_genre_chord_templates(profile.name)

        for i in range(num_chords):
            template_idx = i % len(chord_templates)
            chord = {
                'chord_id': i,
                'root': chord_templates[template_idx]['root'],
                'quality': chord_templates[template_idx]['quality'],
                'extensions': chord_templates[template_idx].get('extensions', []),
                'duration': 4,  # 4 beats
                'voicing': self._get_genre_voicing(profile)
            }
            progression.append(chord)

        return progression

    def _generate_rhythm_pattern(
        self,
        profile: GenreProfile,
        style_vector: np.ndarray,
        duration_seconds: int
    ) -> Dict:
        """Generate genre-specific rhythm pattern"""
        
        patterns = {
            'structured': [1, 0, 1, 0, 1, 0, 1, 0],  # Classical
            'syncopated': [1, 0, 1, 1, 0, 1, 0, 1],  # Jazz
            'steady': [1, 1, 1, 1, 1, 1, 1, 1],      # Electronic
            'flowing': [1, 0, 0, 1, 0, 0, 1, 0],     # Ambient
            'driving': [1, 1, 0, 1, 1, 0, 1, 0],     # Rock
            'shuffle': [1, 0, 1, 1, 0, 1, 0, 0],     # Blues
            'ethnic': [1, 0, 1, 0, 1, 1, 0, 1],      # World
            'dramatic': [1, 0, 0, 0, 1, 1, 0, 0]     # Cinematic
        }

        return {
            'pattern': patterns.get(profile.rhythm_pattern, patterns['flowing']),
            'time_signature': self._get_genre_time_signature(profile),
            'subdivision': 8,  # 8th notes
            'swing': 0.1 if profile.name in ['jazz', 'blues'] else 0.0
        }

    def _generate_dynamics(
        self,
        profile: GenreProfile,
        style_vector: np.ndarray
    ) -> List[int]:
        """Generate dynamic variations"""
        
        dynamics = []
        min_vel, max_vel = profile.dynamics_range
        
        for i in range(16):  # 16 dynamic points
            # Vary dynamics based on style vector
            variation = style_vector[i % len(style_vector)]
            velocity = int(min_vel + (variation + 1) / 2 * (max_vel - min_vel))
            dynamics.append(max(0, min(127, velocity)))

        return dynamics

    def _get_melodic_contour(
        self,
        profile: GenreProfile,
        style_vector: np.ndarray,
        phrase_idx: int
    ) -> str:
        """Determine melodic contour for phrase"""
        
        contours = ['ascending', 'descending', 'arch', 'valley', 'static']
        idx = int((style_vector[phrase_idx % len(style_vector)] + 1) / 2 * len(contours))
        return contours[min(idx, len(contours) - 1)]

    def _get_genre_articulation(self, profile: GenreProfile) -> str:
        """Get genre-specific articulation style"""
        
        articulations = {
            'classical': 'legato',
            'jazz': 'staccato',
            'electronic': 'marcato',
            'ambient': 'sostenuto',
            'rock': 'accented',
            'blues': 'bent',
            'world': 'ornamental',
            'cinematic': 'expressive'
        }
        return articulations.get(profile.name, 'normal')

    def _get_genre_chord_templates(self, genre: str) -> List[Dict]:
        """Get genre-specific chord progression templates"""
        
        templates = {
            'classical': [
                {'root': 0, 'quality': 'major'},
                {'root': 5, 'quality': 'major'},
                {'root': 7, 'quality': 'minor'},
                {'root': 0, 'quality': 'major'}
            ],
            'jazz': [
                {'root': 0, 'quality': 'maj7', 'extensions': [9]},
                {'root': 2, 'quality': 'min7'},
                {'root': 5, 'quality': 'dom7', 'extensions': [9, 13]},
                {'root': 0, 'quality': 'maj7'}
            ],
            'electronic': [
                {'root': 0, 'quality': 'minor'},
                {'root': 7, 'quality': 'major'},
                {'root': 10, 'quality': 'major'},
                {'root': 5, 'quality': 'major'}
            ],
            'ambient': [
                {'root': 0, 'quality': 'sus2'},
                {'root': 5, 'quality': 'sus4'},
                {'root': 7, 'quality': 'add9'},
                {'root': 0, 'quality': 'sus2'}
            ]
        }
        
        return templates.get(genre, templates['ambient'])

    def _get_genre_voicing(self, profile: GenreProfile) -> str:
        """Get genre-specific chord voicing"""
        
        voicings = {
            'classical': 'close',
            'jazz': 'drop2',
            'electronic': 'wide',
            'ambient': 'open',
            'rock': 'power',
            'blues': 'seventh',
            'world': 'modal',
            'cinematic': 'orchestral'
        }
        return voicings.get(profile.name, 'close')

    def _get_genre_time_signature(self, profile: GenreProfile) -> str:
        """Get genre-specific time signature"""
        
        signatures = {
            'classical': '4/4',
            'jazz': '4/4',
            'electronic': '4/4',
            'ambient': '4/4',
            'rock': '4/4',
            'blues': '12/8',
            'world': '7/8',
            'cinematic': '4/4'
        }
        return signatures.get(profile.name, '4/4')

    def get_available_genres(self) -> List[str]:
        """Get list of available genres"""
        return list(self.genre_profiles.keys())

    def get_genre_info(self, genre: str) -> Optional[Dict]:
        """Get information about a specific genre"""
        if genre not in self.genre_profiles:
            return None

        profile = self.genre_profiles[genre]
        return {
            'name': profile.name,
            'tempo_range': profile.tempo_range,
            'scales': profile.scale_preferences,
            'rhythm': profile.rhythm_pattern,
            'complexity': profile.harmony_complexity,
            'density': profile.note_density
        }
