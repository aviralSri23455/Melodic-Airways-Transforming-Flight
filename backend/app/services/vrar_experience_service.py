"""
VR/AR Experience Service - Immersive fly-through experiences
Generates 3D flight paths, camera animations, and spatial audio for VR/AR
Now with vector embeddings for similarity search
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from datetime import datetime
import faiss


class VRARExperienceService:
    """
    Service for creating VR/AR immersive flight experiences
    Generates 3D coordinates, camera paths, and spatial audio positioning
    Now includes vector embeddings for experience similarity search
    """
    
    EARTH_RADIUS = 6371.0  # km
    
    def __init__(self):
        self.experience_cache = {}
        
        # Initialize FAISS index for VR experience embeddings (64D)
        self.embedding_dim = 64
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.experience_metadata = []
        
        print(f"✅ VR/AR Service initialized with FAISS v{faiss.__version__} vector search")
        print(f"🔍 Vector embeddings enabled for VR experience similarity")
    
    def generate_3d_flight_path(
        self,
        origin_coords: Tuple[float, float, float],  # lat, lon, alt
        destination_coords: Tuple[float, float, float],
        num_points: int = 100,
        flight_altitude: float = 10.0  # km above surface
    ) -> List[Dict[str, float]]:
        """
        Generate 3D coordinates for a smooth flight path
        Uses great circle route with altitude profile
        
        Args:
            origin_coords: (latitude, longitude, altitude) of origin
            destination_coords: (latitude, longitude, altitude) of destination
            num_points: Number of interpolation points
            flight_altitude: Cruise altitude in km
        
        Returns:
            List of 3D coordinates with camera orientation
        """
        lat1, lon1, alt1 = origin_coords
        lat2, lon2, alt2 = destination_coords
        
        # Convert to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        path_points = []
        
        for i in range(num_points):
            fraction = i / (num_points - 1)
            
            # Great circle interpolation
            d = np.arccos(
                np.sin(lat1_rad) * np.sin(lat2_rad) +
                np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
            )
            
            if d == 0:
                lat, lon = lat1_rad, lon1_rad
            else:
                a = np.sin((1 - fraction) * d) / np.sin(d)
                b = np.sin(fraction * d) / np.sin(d)
                
                x = a * np.cos(lat1_rad) * np.cos(lon1_rad) + b * np.cos(lat2_rad) * np.cos(lon2_rad)
                y = a * np.cos(lat1_rad) * np.sin(lon1_rad) + b * np.cos(lat2_rad) * np.sin(lon2_rad)
                z = a * np.sin(lat1_rad) + b * np.sin(lat2_rad)
                
                lat = np.arctan2(z, np.sqrt(x**2 + y**2))
                lon = np.arctan2(y, x)
            
            # Altitude profile (parabolic - climb, cruise, descend)
            if fraction < 0.2:  # Climb
                altitude = alt1 + (flight_altitude - alt1) * (fraction / 0.2)
            elif fraction > 0.8:  # Descend
                altitude = flight_altitude - (flight_altitude - alt2) * ((fraction - 0.8) / 0.2)
            else:  # Cruise
                altitude = flight_altitude
            
            # Convert to Cartesian coordinates for 3D rendering
            r = self.EARTH_RADIUS + altitude
            x_cart = r * np.cos(lat) * np.cos(lon)
            y_cart = r * np.cos(lat) * np.sin(lon)
            z_cart = r * np.sin(lat)
            
            # Calculate camera orientation (look-at direction)
            if i < num_points - 1:
                next_fraction = (i + 1) / (num_points - 1)
                next_lat = lat1_rad + (lat2_rad - lat1_rad) * next_fraction
                next_lon = lon1_rad + (lon2_rad - lon1_rad) * next_fraction
                
                look_x = np.cos(next_lat) * np.cos(next_lon) - np.cos(lat) * np.cos(lon)
                look_y = np.cos(next_lat) * np.sin(next_lon) - np.cos(lat) * np.sin(lon)
                look_z = np.sin(next_lat) - np.sin(lat)
            else:
                look_x, look_y, look_z = 0, 0, -1
            
            path_points.append({
                "position": {
                    "x": float(x_cart),
                    "y": float(y_cart),
                    "z": float(z_cart)
                },
                "geographic": {
                    "latitude": float(np.degrees(lat)),
                    "longitude": float(np.degrees(lon)),
                    "altitude": float(altitude)
                },
                "camera": {
                    "lookAt": {
                        "x": float(look_x),
                        "y": float(look_y),
                        "z": float(look_z)
                    },
                    "up": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 1.0
                    }
                },
                "progress": float(fraction)
            })
        
        return path_points
    
    def generate_camera_animation(
        self,
        flight_path: List[Dict[str, float]],
        duration: float = 60.0,
        camera_mode: str = "follow"
    ) -> Dict[str, any]:
        """
        Generate camera animation keyframes for VR/AR
        
        Args:
            flight_path: 3D flight path coordinates
            duration: Animation duration in seconds
            camera_mode: Camera behavior (follow, orbit, cinematic)
        
        Returns:
            Camera animation data with keyframes
        """
        keyframes = []
        num_points = len(flight_path)
        
        for i, point in enumerate(flight_path):
            time = (i / (num_points - 1)) * duration
            
            if camera_mode == "follow":
                # Camera follows directly behind aircraft
                camera_offset = {"x": -50, "y": 0, "z": 20}
            elif camera_mode == "orbit":
                # Camera orbits around flight path
                angle = (i / num_points) * 2 * np.pi
                camera_offset = {
                    "x": 100 * np.cos(angle),
                    "y": 100 * np.sin(angle),
                    "z": 50
                }
            else:  # cinematic
                # Dynamic camera with varying angles
                camera_offset = {
                    "x": -80 + 30 * np.sin(i / num_points * np.pi),
                    "y": 40 * np.cos(i / num_points * 2 * np.pi),
                    "z": 30 + 20 * np.sin(i / num_points * 3 * np.pi)
                }
            
            keyframes.append({
                "time": float(time),
                "position": {
                    "x": point["position"]["x"] + camera_offset["x"],
                    "y": point["position"]["y"] + camera_offset["y"],
                    "z": point["position"]["z"] + camera_offset["z"]
                },
                "target": point["position"],
                "fov": 60.0 + 10.0 * np.sin(i / num_points * np.pi),  # Dynamic FOV
                "roll": 5.0 * np.sin(i / num_points * 4 * np.pi)  # Slight roll for realism
            })
        
        return {
            "duration": duration,
            "keyframes": keyframes,
            "camera_mode": camera_mode,
            "interpolation": "smooth",
            "fps": 60
        }
    
    def generate_spatial_audio_zones(
        self,
        flight_path: List[Dict[str, float]],
        music_segments: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Generate spatial audio positioning for immersive experience
        
        Args:
            flight_path: 3D flight path
            music_segments: Musical segments to spatialize
        
        Returns:
            Spatial audio configuration
        """
        audio_zones = []
        
        for i, segment in enumerate(music_segments):
            # Calculate position along flight path
            segment_start = int((i / len(music_segments)) * len(flight_path))
            segment_end = int(((i + 1) / len(music_segments)) * len(flight_path))
            
            # Get midpoint for audio source
            midpoint = flight_path[segment_start + (segment_end - segment_start) // 2]
            
            audio_zones.append({
                "segment_id": i,
                "position": midpoint["position"],
                "radius": 500.0,  # Audio influence radius in meters
                "volume": 1.0,
                "pan": self._calculate_stereo_pan(midpoint["geographic"]["longitude"]),
                "reverb": self._calculate_reverb(midpoint["geographic"]["altitude"]),
                "doppler": True,
                "music_data": segment
            })
        
        return audio_zones
    
    def _calculate_stereo_pan(self, longitude: float) -> float:
        """Calculate stereo pan based on longitude (-1 to 1)"""
        return np.clip((longitude / 180.0), -1.0, 1.0)
    
    def _calculate_reverb(self, altitude: float) -> float:
        """Calculate reverb amount based on altitude (0 to 1)"""
        return np.clip(altitude / 15.0, 0.0, 1.0)
    
    def create_vr_experience(
        self,
        origin_code: str,
        destination_code: str,
        origin_coords: Tuple[float, float],
        destination_coords: Tuple[float, float],
        music_composition: Dict[str, any],
        experience_type: str = "immersive"
    ) -> Dict[str, any]:
        """
        Create complete VR/AR experience package
        
        Args:
            origin_code: Origin airport code
            destination_code: Destination airport code
            origin_coords: Origin (lat, lon)
            destination_coords: Destination (lat, lon)
            music_composition: Musical composition data
            experience_type: Type of experience (immersive, cinematic, educational)
        
        Returns:
            Complete VR/AR experience data
        """
        # Generate 3D flight path
        flight_path = self.generate_3d_flight_path(
            origin_coords=(origin_coords[0], origin_coords[1], 0.0),
            destination_coords=(destination_coords[0], destination_coords[1], 0.0),
            num_points=200
        )
        
        # Generate camera animation
        camera_mode = "cinematic" if experience_type == "cinematic" else "follow"
        camera_animation = self.generate_camera_animation(
            flight_path,
            duration=music_composition.get("duration", 60.0),
            camera_mode=camera_mode
        )
        
        # Generate spatial audio
        music_segments = music_composition.get("segments", [music_composition])
        spatial_audio = self.generate_spatial_audio_zones(flight_path, music_segments)
        
        # Generate environment effects
        environment = self._generate_environment_effects(flight_path)
        
        # Generate interactive hotspots
        hotspots = self._generate_interactive_hotspots(flight_path, origin_code, destination_code)
        
        experience_data = {
            "experience_id": f"vr_{origin_code}_{destination_code}_{int(datetime.utcnow().timestamp())}",
            "type": experience_type,
            "route": {
                "origin": origin_code,
                "destination": destination_code,
                "origin_coords": origin_coords,
                "destination_coords": destination_coords
            },
            "flight_path": flight_path,
            "camera_animation": camera_animation,
            "spatial_audio": spatial_audio,
            "environment": environment,
            "interactive_hotspots": hotspots,
            "music_composition": music_composition,
            "duration": camera_animation["duration"],
            "vr_ready": True,
            "ar_ready": True,
            "platforms": ["WebXR", "Oculus", "HTC Vive", "ARKit", "ARCore"],
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Generate and store vector embedding for similarity search
        vector_embedding = self.generate_experience_embedding(experience_data)
        experience_data["vector_embedding"] = vector_embedding.tolist()
        
        # Add to FAISS index
        self.add_experience_to_index(experience_data)
        
        # Sync to DuckDB for analytics (non-blocking)
        try:
            from app.services.duckdb_sync_service import duckdb_sync
            duckdb_sync.sync_vr_experience_embedding(experience_data)
        except Exception as e:
            pass  # Don't fail if DuckDB sync fails
        
        print(f"✅ VR experience created with 64D vector embedding")
        
        return experience_data
    
    def _generate_environment_effects(
        self,
        flight_path: List[Dict[str, float]]
    ) -> Dict[str, any]:
        """Generate environmental effects (clouds, lighting, weather)"""
        return {
            "sky": {
                "type": "dynamic",
                "time_of_day": "sunset",
                "cloud_coverage": 0.3,
                "stars": True
            },
            "lighting": {
                "sun_position": {"x": 1000, "y": 500, "z": 2000},
                "ambient_intensity": 0.4,
                "directional_intensity": 0.8
            },
            "weather": {
                "conditions": "clear",
                "wind_speed": 20,
                "visibility": 50000
            },
            "particles": {
                "clouds": True,
                "contrails": True,
                "stars": True
            }
        }
    
    def _generate_interactive_hotspots(
        self,
        flight_path: List[Dict[str, float]],
        origin_code: str,
        destination_code: str
    ) -> List[Dict[str, any]]:
        """Generate interactive information hotspots along the route"""
        hotspots = []
        
        # Origin hotspot
        hotspots.append({
            "id": "origin",
            "position": flight_path[0]["position"],
            "type": "airport",
            "title": f"Departure: {origin_code}",
            "description": "Your journey begins here",
            "interactive": True,
            "icon": "airport_departure"
        })
        
        # Midpoint hotspot
        mid_idx = len(flight_path) // 2
        hotspots.append({
            "id": "midpoint",
            "position": flight_path[mid_idx]["position"],
            "type": "waypoint",
            "title": "Cruise Altitude",
            "description": "Halfway through your musical journey",
            "interactive": True,
            "icon": "flight"
        })
        
        # Destination hotspot
        hotspots.append({
            "id": "destination",
            "position": flight_path[-1]["position"],
            "type": "airport",
            "title": f"Arrival: {destination_code}",
            "description": "Your journey ends here",
            "interactive": True,
            "icon": "airport_arrival"
        })
        
        return hotspots
    
    def export_for_unity(self, experience_data: Dict[str, any]) -> str:
        """Export experience data in Unity-compatible format"""
        unity_data = {
            "flightPath": [
                {
                    "position": {"x": p["position"]["x"], "y": p["position"]["z"], "z": p["position"]["y"]},
                    "rotation": {"x": 0, "y": 0, "z": 0}
                }
                for p in experience_data["flight_path"]
            ],
            "cameraKeyframes": experience_data["camera_animation"]["keyframes"],
            "audioSources": experience_data["spatial_audio"]
        }
        return json.dumps(unity_data, indent=2)
    
    def export_for_webxr(self, experience_data: Dict[str, any]) -> str:
        """Export experience data in WebXR-compatible format"""
        webxr_data = {
            "scene": {
                "flightPath": experience_data["flight_path"],
                "camera": experience_data["camera_animation"],
                "audio": experience_data["spatial_audio"],
                "environment": experience_data["environment"]
            },
            "metadata": {
                "duration": experience_data["duration"],
                "route": experience_data["route"]
            }
        }
        return json.dumps(webxr_data, indent=2)

    def generate_experience_embedding(
        self,
        experience: Dict[str, any]
    ) -> np.ndarray:
        """
        Generate a 64D vector embedding for a VR experience
        
        Args:
            experience: VR experience data
        
        Returns:
            64D numpy array embedding
        """
        # Extract features from VR experience
        flight_path = experience.get("flight_path", [])
        
        if not flight_path:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Path geometry features (16D)
        positions = [p["position"] for p in flight_path]
        x_coords = [p["x"] for p in positions]
        y_coords = [p["y"] for p in positions]
        z_coords = [p["z"] for p in positions]
        
        geometry_features = [
            np.mean(x_coords), np.std(x_coords),
            np.mean(y_coords), np.std(y_coords),
            np.mean(z_coords), np.std(z_coords),
            np.max(z_coords) - np.min(z_coords),  # Altitude range
            len(flight_path) / 200.0  # Path complexity
        ]
        
        # Geographic features (8D)
        route = experience.get("route", {})
        origin_coords = route.get("origin_coords", (0, 0))
        dest_coords = route.get("destination_coords", (0, 0))
        
        lat_diff = abs(dest_coords[0] - origin_coords[0])
        lon_diff = abs(dest_coords[1] - origin_coords[1])
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        
        geographic_features = [
            origin_coords[0] / 90.0,  # Normalized latitude
            origin_coords[1] / 180.0,  # Normalized longitude
            dest_coords[0] / 90.0,
            dest_coords[1] / 180.0,
            lat_diff / 180.0,
            lon_diff / 360.0,
            distance / 200.0,
            1.0 if lat_diff > lon_diff else 0.0  # Direction indicator
        ]
        
        # Camera animation features (12D)
        camera = experience.get("camera_animation", {})
        keyframes = camera.get("keyframes", [])
        
        if keyframes:
            fovs = [k.get("fov", 60) for k in keyframes]
            rolls = [k.get("roll", 0) for k in keyframes]
            
            camera_features = [
                np.mean(fovs) / 90.0,
                np.std(fovs) / 30.0,
                np.mean(np.abs(rolls)) / 45.0,
                len(keyframes) / 200.0
            ]
        else:
            camera_features = [0.0] * 4
        
        # Music composition features (12D)
        music = experience.get("music_composition", {})
        music_features = [
            music.get("tempo", 120) / 200.0,
            music.get("duration", 60) / 120.0,
            len(music.get("segments", [])) / 10.0,
            1.0 if music.get("genre") else 0.0
        ]
        
        # Experience type encoding (4D)
        exp_type = experience.get("type", "immersive")
        type_encoding = [
            1.0 if exp_type == "immersive" else 0.0,
            1.0 if exp_type == "cinematic" else 0.0,
            1.0 if exp_type == "educational" else 0.0,
            experience.get("duration", 60) / 120.0
        ]
        
        # Spatial audio features (4D)
        spatial_audio = experience.get("spatial_audio", [])
        audio_features = [
            len(spatial_audio) / 20.0,
            1.0 if spatial_audio else 0.0,
            0.5,  # Placeholder
            0.5   # Placeholder
        ]
        
        # Environment features (8D)
        env = experience.get("environment", {})
        sky = env.get("sky", {})
        weather = env.get("weather", {})
        
        env_features = [
            sky.get("cloud_coverage", 0.3),
            1.0 if sky.get("stars") else 0.0,
            weather.get("wind_speed", 20) / 100.0,
            weather.get("visibility", 50000) / 100000.0
        ]
        
        # Combine all features
        all_features = (
            geometry_features + geographic_features + camera_features +
            music_features + type_encoding + audio_features + env_features
        )
        
        # Pad or truncate to exactly 64 dimensions
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        feature_count = min(len(all_features), self.embedding_dim)
        embedding[:feature_count] = all_features[:feature_count]
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def add_experience_to_index(
        self,
        experience: Dict[str, any]
    ) -> str:
        """
        Add a VR experience to the FAISS index
        
        Args:
            experience: VR experience data
        
        Returns:
            Experience ID
        """
        # Generate embedding
        embedding = self.generate_experience_embedding(experience)
        
        # Add to FAISS index
        self.faiss_index.add(np.array([embedding]))
        
        # Store metadata
        experience_id = experience.get("experience_id")
        metadata = {
            "id": experience_id,
            "type": experience.get("type"),
            "origin": experience.get("route", {}).get("origin"),
            "destination": experience.get("route", {}).get("destination"),
            "duration": experience.get("duration"),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.experience_metadata.append(metadata)
        
        print(f"✅ Added VR experience to FAISS index: {experience_id} (total: {self.faiss_index.ntotal})")
        
        return experience_id
    
    def find_similar_experiences(
        self,
        query_experience: Dict[str, any],
        k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Find similar VR experiences using vector similarity search
        
        Args:
            query_experience: Experience to find similar matches for
            k: Number of similar experiences to return
        
        Returns:
            List of similar experiences with similarity scores
        """
        if self.faiss_index.ntotal == 0:
            print("⚠️ VR FAISS index is empty")
            return []
        
        # Generate embedding for query
        query_embedding = self.generate_experience_embedding(query_experience)
        
        # Search FAISS index
        k = min(k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.experience_metadata):
                similarity_score = 1.0 / (1.0 + distance)
                result = {
                    **self.experience_metadata[idx],
                    "similarity_score": float(similarity_score),
                    "distance": float(distance),
                    "rank": i + 1
                }
                results.append(result)
        
        print(f"🔍 Found {len(results)} similar VR experiences using vector search")
        
        return results
