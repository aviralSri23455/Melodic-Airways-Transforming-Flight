"""
Tests for VR/AR Experience API endpoints and 3D functionality.
Real integration tests using OpenFlights dataset (3,000+ airports, 67,000+ routes)
"""
import pytest
from fastapi.testclient import TestClient
import json
import math

from main import app

client = TestClient(app)

class TestVRARExperienceAPI:
    """Test cases for VR/AR Experience API endpoints using real OpenFlights data."""

    def test_create_vr_experience_success(self):
        """Test successful VR experience creation with real airports."""
        experience_request = {
            "origin_code": "JFK",
            "destination_code": "LAX",
            "experience_type": "cinematic",
            "camera_mode": "orbit",
            "music_composition": {
                "tempo": 120,
                "duration": 60,
                "notes": [60, 64, 67]
            }
        }
        
        response = client.post(
            "/api/v1/vr/vr-experiences/create",
            json=experience_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "experience_id" in data["data"]
        assert "flight_path" in data["data"]
        assert "camera_animation" in data["data"]
        assert "spatial_audio" in data["data"]
        assert "route" in data["data"]
        assert data["data"]["route"]["origin"] == "JFK"
        assert data["data"]["route"]["destination"] == "LAX"
        assert len(data["data"]["flight_path"]) == 200  # Default num_points

    def test_create_vr_experience_international(self):
        """Test VR experience with international route using real data."""
        experience_request = {
            "origin_code": "LHR",  # London Heathrow
            "destination_code": "NRT",  # Tokyo Narita
            "experience_type": "immersive",
            "camera_mode": "follow"
        }
        
        response = client.post(
            "/api/v1/vr/vr-experiences/create",
            json=experience_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["vr_ready"] is True
        assert data["data"]["ar_ready"] is True

    def test_create_vr_experience_invalid_airports(self):
        """Test VR experience creation with invalid airport codes."""
        experience_request = {
            "origin_code": "INVALID",
            "destination_code": "ALSO_INVALID",
            "experience_type": "cinematic",
            "camera_mode": "orbit"
        }
        
        response = client.post(
            "/api/v1/vr/vr-experiences/create",
            json=experience_request
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_get_flight_path_3d_domestic(self):
        """Test retrieving 3D flight path for domestic route."""
        response = client.get(
            "/api/v1/vr/vr-experiences/flight-path",
            params={
                "origin_code": "JFK",
                "destination_code": "LAX",
                "num_points": 200
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "path_points" in data["data"]
        assert "total_points" in data["data"]
        assert data["data"]["total_points"] == 200
        
        # Verify path point structure
        first_point = data["data"]["path_points"][0]
        assert "position" in first_point
        assert "geographic" in first_point
        assert "camera" in first_point
        assert "progress" in first_point

    def test_get_flight_path_3d_international(self):
        """Test 3D flight path for long international route."""
        response = client.get(
            "/api/v1/vr/vr-experiences/flight-path",
            params={
                "origin_code": "SYD",  # Sydney
                "destination_code": "DXB",  # Dubai
                "num_points": 100,
                "flight_altitude": 12.0
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["path_points"]) == 100

    def test_generate_camera_animation(self):
        """Test camera animation generation with real flight path."""
        # First get a real flight path
        path_response = client.get(
            "/api/v1/vr/vr-experiences/flight-path",
            params={
                "origin_code": "JFK",
                "destination_code": "LAX",
                "num_points": 10
            }
        )
        assert path_response.status_code == 200
        flight_path = path_response.json()["data"]["path_points"]
        
        response = client.post(
            "/api/v1/vr/vr-experiences/camera-animation?duration=60&camera_mode=cinematic",
            json=flight_path
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "keyframes" in data["data"]
        assert len(data["data"]["keyframes"]) == 10

    def test_generate_spatial_audio(self):
        """Test spatial audio generation with real flight path."""
        # First get a real flight path
        path_response = client.get(
            "/api/v1/vr/vr-experiences/flight-path",
            params={
                "origin_code": "JFK",
                "destination_code": "LAX",
                "num_points": 10
            }
        )
        assert path_response.status_code == 200
        flight_path = path_response.json()["data"]["path_points"]
        
        music_segments = [
            {"notes": [60, 64, 67], "tempo": 120},
            {"notes": [62, 65, 69], "tempo": 120}
        ]
        
        audio_request = {
            "flight_path": flight_path,
            "music_segments": music_segments
        }
        
        response = client.post(
            "/api/v1/vr/vr-experiences/spatial-audio",
            json=audio_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "audio_zones" in data["data"]
        assert len(data["data"]["audio_zones"]) == 2

    def test_export_unity_format(self):
        """Test exporting VR experience for Unity."""
        response = client.get("/api/v1/vr/vr-experiences/export/unity/exp_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "format" in data["data"]
        assert data["data"]["format"] == "Unity JSON"

    def test_export_webxr_format(self):
        """Test exporting VR experience for WebXR."""
        response = client.get("/api/v1/vr/vr-experiences/export/webxr/exp_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "format" in data["data"]
        assert data["data"]["format"] == "WebXR JSON"

    def test_get_demo_experience(self):
        """Test getting a demo VR experience with real data."""
        response = client.get(
            "/api/v1/vr/vr-experiences/demo",
            params={"origin": "JFK", "destination": "LAX"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "experience_id" in data["data"]
        assert data["demo"] is True
        assert "path_points_count" in data["data"]
        assert data["data"]["path_points_count"] > 0

    def test_vr_info_endpoint(self):
        """Test VR/AR info endpoint."""
        response = client.get("/api/v1/vr/vr-experiences/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "features" in data["data"]
        assert "supported_platforms" in data["data"]

class TestVRARExperienceService:
    """Test cases for VR/AR Experience Service functionality."""

    def test_3d_flight_path_generation(self):
        """Test 3D flight path generation."""
        from app.services.vrar_experience_service import VRARExperienceService
        
        service = VRARExperienceService()
        
        # JFK to LAX coordinates
        origin_coords = (40.6413, -73.7781, 0.0)
        destination_coords = (34.0522, -118.2437, 0.0)
        
        path_points = service.generate_3d_flight_path(origin_coords, destination_coords, 100)
        
        assert len(path_points) == 100
        assert "position" in path_points[0]
        assert "geographic" in path_points[0]
        assert "camera" in path_points[0]
        assert path_points[0]["progress"] == 0.0
        assert path_points[-1]["progress"] == 1.0

    def test_camera_animation_generation(self):
        """Test camera animation generation."""
        from app.services.vrar_experience_service import VRARExperienceService
        
        service = VRARExperienceService()
        
        # Create simple flight path
        flight_path = service.generate_3d_flight_path(
            (40.6413, -73.7781, 0.0),
            (34.0522, -118.2437, 0.0),
            10
        )
        
        # Test different camera modes
        for mode in ["follow", "orbit", "cinematic"]:
            animation = service.generate_camera_animation(flight_path, 60.0, mode)
            assert "keyframes" in animation
            assert "duration" in animation
            assert animation["camera_mode"] == mode
            assert len(animation["keyframes"]) > 0

    def test_spatial_audio_zones(self):
        """Test spatial audio zone generation."""
        from app.services.vrar_experience_service import VRARExperienceService
        
        service = VRARExperienceService()
        
        flight_path = service.generate_3d_flight_path(
            (40.6413, -73.7781, 0.0),
            (34.0522, -118.2437, 0.0),
            10
        )
        
        music_segments = [
            {"notes": [60, 64, 67], "tempo": 120},
            {"notes": [62, 65, 69], "tempo": 120}
        ]
        
        audio_zones = service.generate_spatial_audio_zones(flight_path, music_segments)
        
        assert len(audio_zones) > 0
        for zone in audio_zones:
            assert "position" in zone
            assert "volume" in zone
            assert "pan" in zone
            assert "reverb" in zone
            assert 0 <= zone["volume"] <= 1
            assert -1 <= zone["pan"] <= 1
            assert 0 <= zone["reverb"] <= 1

    def test_interactive_hotspots(self):
        """Test interactive hotspot generation."""
        from app.services.vrar_experience_service import VRARExperienceService
        
        service = VRARExperienceService()
        
        flight_path = service.generate_3d_flight_path(
            (40.6413, -73.7781, 0.0),
            (34.0522, -118.2437, 0.0),
            10
        )
        
        hotspots = service._generate_interactive_hotspots(flight_path, "JFK", "LAX")
        
        assert len(hotspots) >= 3  # Origin, midpoint, destination
        for hotspot in hotspots:
            assert "position" in hotspot
            assert "type" in hotspot
            assert "title" in hotspot

    def test_environment_effects(self):
        """Test environment effects generation."""
        from app.services.vrar_experience_service import VRARExperienceService
        
        service = VRARExperienceService()
        
        flight_path = service.generate_3d_flight_path(
            (40.6413, -73.7781, 0.0),
            (34.0522, -118.2437, 0.0),
            10
        )
        
        effects = service._generate_environment_effects(flight_path)
        
        assert "lighting" in effects
        assert "weather" in effects
        assert "sky" in effects
        
        # Validate lighting parameters
        lighting = effects["lighting"]
        assert "sun_position" in lighting
        assert "ambient_intensity" in lighting

    def test_export_formats(self):
        """Test different export format generation."""
        from app.services.vrar_experience_service import VRARExperienceService
        
        service = VRARExperienceService()
        
        experience_data = {
            "flight_path": [{"position": {"x": 0, "y": 0, "z": 10000}}],
            "camera_animation": {"keyframes": []},
            "spatial_audio": []
        }
        
        # Test Unity export
        unity_export = service.export_for_unity(experience_data)
        assert isinstance(unity_export, str)
        unity_json = json.loads(unity_export)
        assert "flightPath" in unity_json
        
        # Test WebXR export
        webxr_export = service.export_for_webxr(experience_data)
        assert isinstance(webxr_export, str)
        webxr_json = json.loads(webxr_export)
        assert "scene" in webxr_json

class TestVRARPerformance:
    """Test cases for VR/AR performance and optimization."""

    def test_path_generation_performance(self):
        """Test 3D path generation performance."""
        from app.services.vrar_experience_service import VRARExperienceService
        import time
        
        service = VRARExperienceService()
        
        origin_coords = (40.6413, -73.7781, 0.0)
        destination_coords = (34.0522, -118.2437, 0.0)
        
        start_time = time.time()
        path_points = service.generate_3d_flight_path(origin_coords, destination_coords, 1000)
        end_time = time.time()
        
        generation_time = end_time - start_time
        assert generation_time < 2.0  # Should generate 1000 points in under 2 seconds
        assert len(path_points) == 1000

    def test_concurrent_generation(self):
        """Test concurrent VR experience generation."""
        from app.services.vrar_experience_service import VRARExperienceService
        import threading
        
        service = VRARExperienceService()
        results = []
        
        def generate_experience():
            try:
                experience = service.create_vr_experience(
                    "JFK", "LAX",
                    (40.6413, -73.7781),
                    (34.0522, -118.2437),
                    {"tempo": 120, "duration": 30, "segments": []},
                    "immersive"
                )
                results.append(experience)
            except Exception as e:
                results.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=generate_experience)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)
            assert "experience_id" in result

if __name__ == "__main__":
    pytest.main([__file__])