"""
Tests for AI Genre Composer API endpoints and PyTorch models.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from main import app

client = TestClient(app)

class TestAIGenreComposerAPI:
    """Test cases for AI Genre Composer API endpoints."""

    def test_get_available_genres(self):
        """Test retrieving available AI genres."""
        response = client.get("/api/v1/ai/ai-genres/available")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "genres" in data["data"]
        
        # Check that expected genres are present
        genres = data["data"]["genres"]
        expected_genres = ["classical", "jazz", "electronic", "ambient", "rock", "world", "cinematic", "lofi"]
        
        for genre in expected_genres:
            assert any(g["name"] == genre for g in genres)

    def test_compose_with_valid_genre(self):
        """Test AI composition with valid genre and route features."""
        composition_request = {
            "genre": "jazz",
            "route_features": {
                "distance": 5000,
                "latitude_range": 40,
                "longitude_range": 80,
                "direction": "E"
            },
            "duration": 30
        }
        
        with patch('app.api.ai_genre_routes.AIGenreComposer') as mock_composer:
            mock_instance = Mock()
            mock_composer.return_value = mock_instance
            mock_instance.compose.return_value = {
                "genre": "jazz",
                "notes": [60, 64, 67, 70],
                "tempo": 120,
                "duration": 30,
                "confidence": 0.85,
                "model_info": {"embedding_dim": 32, "pattern_length": 12}
            }
            
            response = client.post(
                "/api/v1/ai/ai-genres/compose",
                json=composition_request
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "composition" in data["data"]
        assert data["data"]["composition"]["genre"] == "jazz"

    def test_compose_with_invalid_genre(self):
        """Test AI composition with invalid genre."""
        composition_request = {
            "genre": "invalid_genre",
            "route_features": {
                "distance": 5000,
                "latitude_range": 40,
                "longitude_range": 80,
                "direction": "E"
            },
            "duration": 30
        }
        
        response = client.post(
            "/api/v1/ai/ai-genres/compose",
            json=composition_request
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "invalid genre" in data["message"].lower()

    def test_get_genre_recommendations(self):
        """Test AI genre recommendations based on route features."""
        route_features = {
            "distance": 8000,
            "latitude_range": 60,
            "longitude_range": 120,
            "direction": "W"
        }
        
        with patch('app.api.ai_genre_routes.AIGenreComposer') as mock_composer:
            mock_instance = Mock()
            mock_composer.return_value = mock_instance
            mock_instance.get_genre_recommendations.return_value = {
                "top_genre": "cinematic",
                "confidence": 0.92,
                "recommendations": [
                    {"genre": "cinematic", "score": 0.92, "reason": "Long distance suggests epic journey"},
                    {"genre": "ambient", "score": 0.78, "reason": "Wide latitude range suggests peaceful flight"},
                    {"genre": "world", "score": 0.65, "reason": "Westward direction suggests exploration"}
                ]
            }
            
            response = client.post(
                "/api/v1/ai/ai-genres/recommendations",
                json=route_features
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "top_genre" in data["data"]
        assert "recommendations" in data["data"]

    def test_blend_genres(self):
        """Test blending two AI genres."""
        blend_request = {
            "primary_genre": "classical",
            "secondary_genre": "electronic",
            "blend_ratio": 0.3
        }
        
        with patch('app.api.ai_genre_routes.AIGenreComposer') as mock_composer:
            mock_instance = Mock()
            mock_composer.return_value = mock_instance
            mock_instance.blend_genres.return_value = {
                "primary_genre": "classical",
                "secondary_genre": "electronic",
                "blend_ratio": 0.3,
                "blended_composition": {
                    "notes": [60, 64, 67, 72],
                    "tempo": 100,
                    "characteristics": ["orchestral", "synthesized"]
                }
            }
            
            response = client.post(
                "/api/v1/ai/ai-genres/blend",
                json=blend_request
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "blended_composition" in data["data"]

    def test_get_demo_composition(self):
        """Test getting a demo composition for a specific genre."""
        response = client.get("/api/v1/ai/ai-genres/demo/jazz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "composition" in data["data"]

    def test_get_model_info(self):
        """Test retrieving AI model information."""
        response = client.get("/api/v1/ai/ai-genres/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "models" in data["data"]
        
        models = data["data"]["models"]
        assert "GenreEmbeddingModel" in models
        assert "MusicPatternGenerator" in models

class TestAIGenreComposerService:
    """Test cases for AI Genre Composer Service and PyTorch models."""

    @pytest.fixture
    def mock_composer(self):
        """Create a mock AI Genre Composer instance."""
        with patch('app.services.ai_genre_composer.AIGenreComposer') as mock:
            yield mock.return_value

    def test_genre_embedding_model_architecture(self):
        """Test GenreEmbeddingModel architecture."""
        from app.services.ai_genre_composer import GenreEmbeddingModel
        
        model = GenreEmbeddingModel(input_dim=10, embedding_dim=32)
        
        # Test model structure
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')
        assert hasattr(model, 'dropout')
        
        # Test forward pass
        test_input = torch.randn(1, 10)
        output = model(test_input)
        
        assert output.shape == (1, 32)
        assert not torch.isnan(output).any()

    def test_music_pattern_generator_architecture(self):
        """Test MusicPatternGenerator architecture."""
        from app.services.ai_genre_composer import MusicPatternGenerator
        
        model = MusicPatternGenerator(input_dim=32, hidden_dim=128, output_dim=12)
        
        # Test model structure
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'fc')
        assert hasattr(model, 'dropout')
        
        # Test forward pass
        test_input = torch.randn(1, 10, 32)  # (batch, seq_len, input_dim)
        output = model(test_input)
        
        assert output.shape == (1, 10, 12)  # (batch, seq_len, output_dim)
        assert not torch.isnan(output).any()

    def test_route_features_to_tensor(self):
        """Test conversion of route features to tensor."""
        from app.services.ai_genre_composer import AIGenreComposer
        
        composer = AIGenreComposer()
        
        route_features = {
            "distance": 5000,
            "latitude_range": 40,
            "longitude_range": 80,
            "direction": "E"
        }
        
        tensor = composer._route_features_to_tensor(route_features)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 10)  # Expected input dimension
        assert not torch.isnan(tensor).any()

    def test_genre_characteristics(self):
        """Test genre characteristics mapping."""
        from app.services.ai_genre_composer import AIGenreComposer
        
        composer = AIGenreComposer()
        
        # Test all supported genres
        genres = ["classical", "jazz", "electronic", "ambient", "rock", "world", "cinematic", "lofi"]
        
        for genre in genres:
            characteristics = composer._get_genre_characteristics(genre)
            
            assert "tempo_range" in characteristics
            assert "key_signature" in characteristics
            assert "time_signature" in characteristics
            assert "dynamics" in characteristics
            
            # Validate tempo range
            tempo_min, tempo_max = characteristics["tempo_range"]
            assert 60 <= tempo_min <= tempo_max <= 180

    def test_composition_generation(self):
        """Test complete composition generation process."""
        from app.services.ai_genre_composer import AIGenreComposer
        
        composer = AIGenreComposer()
        
        route_features = {
            "distance": 5000,
            "latitude_range": 40,
            "longitude_range": 80,
            "direction": "E"
        }
        
        with patch.object(composer, '_generate_embedding') as mock_embedding, \
             patch.object(composer, '_generate_pattern') as mock_pattern:
            
            mock_embedding.return_value = torch.randn(1, 32)
            mock_pattern.return_value = torch.randn(1, 16, 12)
            
            composition = composer.compose("jazz", route_features, duration=30)
            
            assert "genre" in composition
            assert "notes" in composition
            assert "tempo" in composition
            assert "duration" in composition
            assert "confidence" in composition
            assert composition["genre"] == "jazz"

    def test_genre_blending(self):
        """Test genre blending functionality."""
        from app.services.ai_genre_composer import AIGenreComposer
        
        composer = AIGenreComposer()
        
        with patch.object(composer, 'compose') as mock_compose:
            # Mock compositions for both genres
            mock_compose.side_effect = [
                {
                    "notes": [60, 64, 67],
                    "tempo": 80,
                    "dynamics": [0.5, 0.6, 0.7]
                },
                {
                    "notes": [62, 66, 69],
                    "tempo": 120,
                    "dynamics": [0.8, 0.9, 1.0]
                }
            ]
            
            blended = composer.blend_genres("classical", "electronic", 0.3)
            
            assert "primary_genre" in blended
            assert "secondary_genre" in blended
            assert "blend_ratio" in blended
            assert "blended_composition" in blended

    def test_model_device_handling(self):
        """Test model device handling (CPU/GPU)."""
        from app.services.ai_genre_composer import AIGenreComposer
        
        composer = AIGenreComposer()
        
        # Test that models are on correct device
        assert next(composer.embedding_model.parameters()).device.type in ['cpu', 'cuda']
        assert next(composer.pattern_model.parameters()).device.type in ['cpu', 'cuda']

    def test_error_handling(self):
        """Test error handling in AI composer."""
        from app.services.ai_genre_composer import AIGenreComposer
        
        composer = AIGenreComposer()
        
        # Test invalid genre
        with pytest.raises(ValueError):
            composer.compose("invalid_genre", {}, 30)
        
        # Test invalid route features
        with pytest.raises((ValueError, KeyError)):
            composer.compose("jazz", {}, 30)  # Missing required features

class TestModelPerformance:
    """Test cases for model performance and optimization."""

    def test_inference_speed(self):
        """Test model inference speed."""
        from app.services.ai_genre_composer import AIGenreComposer
        import time
        
        composer = AIGenreComposer()
        
        route_features = {
            "distance": 5000,
            "latitude_range": 40,
            "longitude_range": 80,
            "direction": "E"
        }
        
        # Warm up
        composer.compose("jazz", route_features, 30)
        
        # Time multiple inferences
        start_time = time.time()
        for _ in range(10):
            composer.compose("jazz", route_features, 30)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 1.0  # Should be under 1 second per composition

    def test_memory_usage(self):
        """Test model memory usage."""
        from app.services.ai_genre_composer import AIGenreComposer
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        composer = AIGenreComposer()
        
        # Generate multiple compositions
        route_features = {
            "distance": 5000,
            "latitude_range": 40,
            "longitude_range": 80,
            "direction": "E"
        }
        
        for _ in range(50):
            composer.compose("jazz", route_features, 30)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

if __name__ == "__main__":
    pytest.main([__file__])