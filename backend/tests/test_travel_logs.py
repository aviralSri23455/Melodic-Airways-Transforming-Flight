"""
Tests for Travel Logs API endpoints and functionality.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from datetime import datetime

from main import app

client = TestClient(app)

# Mock user token for authentication
MOCK_TOKEN = "test_token_123"
MOCK_USER_ID = 1

@pytest.fixture
def auth_headers():
    """Mock authentication headers."""
    return {"Authorization": f"Bearer {MOCK_TOKEN}"}

@pytest.fixture
def sample_travel_log():
    """Sample travel log data for testing."""
    return {
        "title": "Test European Trip",
        "description": "A wonderful journey across Europe",
        "waypoints": [
            {
                "airport_code": "JFK",
                "timestamp": "2025-06-01T10:00:00",
                "notes": "Departure from New York"
            },
            {
                "airport_code": "LHR",
                "timestamp": "2025-06-01T22:00:00",
                "notes": "London layover"
            },
            {
                "airport_code": "CDG",
                "timestamp": "2025-06-02T08:00:00",
                "notes": "Paris arrival"
            }
        ],
        "tags": ["vacation", "europe", "summer"],
        "travel_date": "2025-06-01",
        "is_public": False
    }

class TestTravelLogsAPI:
    """Test cases for Travel Logs API endpoints."""

    @patch('app.api.travel_log_routes.get_current_user')
    @patch('app.api.travel_log_routes.get_db')
    def test_create_travel_log_success(self, mock_db, mock_user, auth_headers, sample_travel_log):
        """Test successful travel log creation."""
        # Mock user and database
        mock_user.return_value = Mock(id=MOCK_USER_ID)
        mock_db_session = Mock()
        mock_db.return_value = mock_db_session
        
        # Mock the database operations
        mock_db_session.add = Mock()
        mock_db_session.commit = Mock()
        mock_db_session.refresh = Mock()
        
        # Mock the created travel log
        created_log = Mock()
        created_log.id = 1
        created_log.title = sample_travel_log["title"]
        created_log.user_id = MOCK_USER_ID
        created_log.waypoints = sample_travel_log["waypoints"]
        
        with patch('app.api.travel_log_routes.TravelLog', return_value=created_log):
            response = client.post(
                "/api/v1/user/travel-logs",
                json=sample_travel_log,
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "travel_log" in data["data"]

    def test_create_travel_log_no_auth(self, sample_travel_log):
        """Test travel log creation without authentication."""
        response = client.post(
            "/api/v1/user/travel-logs",
            json=sample_travel_log
        )
        
        assert response.status_code == 401

    def test_create_travel_log_invalid_data(self, auth_headers):
        """Test travel log creation with invalid data."""
        invalid_data = {
            "title": "",  # Empty title should fail validation
            "waypoints": []  # Empty waypoints should fail
        }
        
        with patch('app.api.travel_log_routes.get_current_user') as mock_user:
            mock_user.return_value = Mock(id=MOCK_USER_ID)
            
            response = client.post(
                "/api/v1/user/travel-logs",
                json=invalid_data,
                headers=auth_headers
            )
        
        assert response.status_code == 422  # Validation error

    @patch('app.api.travel_log_routes.get_current_user')
    @patch('app.api.travel_log_routes.get_db')
    def test_get_my_travel_logs(self, mock_db, mock_user, auth_headers):
        """Test retrieving user's travel logs."""
        # Mock user and database
        mock_user.return_value = Mock(id=MOCK_USER_ID)
        mock_db_session = Mock()
        mock_db.return_value = mock_db_session
        
        # Mock query results
        mock_logs = [
            Mock(id=1, title="Trip 1", user_id=MOCK_USER_ID),
            Mock(id=2, title="Trip 2", user_id=MOCK_USER_ID)
        ]
        mock_db_session.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = mock_logs
        
        response = client.get(
            "/api/v1/user/travel-logs/my",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "travel_logs" in data["data"]

    @patch('app.api.travel_log_routes.get_db')
    def test_get_public_travel_logs(self, mock_db):
        """Test retrieving public travel logs."""
        mock_db_session = Mock()
        mock_db.return_value = mock_db_session
        
        # Mock public logs
        mock_logs = [
            Mock(id=1, title="Public Trip 1", is_public=True),
            Mock(id=2, title="Public Trip 2", is_public=True)
        ]
        mock_db_session.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = mock_logs
        
        response = client.get("/api/v1/user/travel-logs/public")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch('app.api.travel_log_routes.get_current_user')
    @patch('app.api.travel_log_routes.get_db')
    def test_convert_travel_log_to_music(self, mock_db, mock_user, auth_headers):
        """Test converting travel log to music composition."""
        # Mock user and database
        mock_user.return_value = Mock(id=MOCK_USER_ID)
        mock_db_session = Mock()
        mock_db.return_value = mock_db_session
        
        # Mock travel log
        mock_log = Mock()
        mock_log.id = 1
        mock_log.user_id = MOCK_USER_ID
        mock_log.waypoints = [
            {"airport_code": "JFK", "timestamp": "2025-06-01T10:00:00"},
            {"airport_code": "LAX", "timestamp": "2025-06-01T18:00:00"}
        ]
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_log
        
        with patch('app.api.travel_log_routes.TravelLogService') as mock_service:
            mock_service.return_value.convert_to_music.return_value = {
                "segments": 2,
                "total_duration": 60,
                "composition": {"notes": [60, 64, 67]}
            }
            
            response = client.post(
                "/api/v1/user/travel-logs/1/convert-to-music?music_style=ambient",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "composition" in data["data"]

    @patch('app.api.travel_log_routes.get_current_user')
    @patch('app.api.travel_log_routes.get_db')
    def test_share_travel_log(self, mock_db, mock_user, auth_headers):
        """Test making travel log public/private."""
        # Mock user and database
        mock_user.return_value = Mock(id=MOCK_USER_ID)
        mock_db_session = Mock()
        mock_db.return_value = mock_db_session
        
        # Mock travel log
        mock_log = Mock()
        mock_log.id = 1
        mock_log.user_id = MOCK_USER_ID
        mock_log.is_public = False
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_log
        
        response = client.patch(
            "/api/v1/user/travel-logs/1/share",
            json={"is_public": True},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert mock_log.is_public is True

class TestTravelLogService:
    """Test cases for Travel Log Service functionality."""

    def test_waypoint_validation(self):
        """Test waypoint data validation."""
        from app.services.travel_log_service import TravelLogService
        
        service = TravelLogService()
        
        # Valid waypoint
        valid_waypoint = {
            "airport_code": "JFK",
            "timestamp": "2025-06-01T10:00:00",
            "notes": "Departure"
        }
        assert service._validate_waypoint(valid_waypoint) is True
        
        # Invalid waypoint (missing airport_code)
        invalid_waypoint = {
            "timestamp": "2025-06-01T10:00:00",
            "notes": "Departure"
        }
        assert service._validate_waypoint(invalid_waypoint) is False

    def test_multi_segment_composition(self):
        """Test multi-segment music composition generation."""
        from app.services.travel_log_service import TravelLogService
        
        service = TravelLogService()
        
        waypoints = [
            {"airport_code": "JFK", "timestamp": "2025-06-01T10:00:00"},
            {"airport_code": "LHR", "timestamp": "2025-06-01T22:00:00"},
            {"airport_code": "CDG", "timestamp": "2025-06-02T08:00:00"}
        ]
        
        with patch('app.services.travel_log_service.generate_music_for_route') as mock_generate:
            mock_generate.return_value = {"notes": [60, 64, 67], "duration": 30}
            
            composition = service.convert_to_music(waypoints, "ambient")
            
            assert composition["segments"] == 2  # JFK->LHR, LHR->CDG
            assert "total_duration" in composition
            assert "composition" in composition

if __name__ == "__main__":
    pytest.main([__file__])