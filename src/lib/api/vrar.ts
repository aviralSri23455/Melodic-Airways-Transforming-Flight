import { apiClient, ApiResponse } from './client';

export interface VRSessionRequest {
  origin: string;
  destination: string;
  enable_spatial_audio?: boolean;
  quality?: 'low' | 'medium' | 'high' | 'ultra';
}

export interface Waypoint {
  x: number;
  y: number;
  z: number;
  progress: number;
  lat: number;
  lng: number;
}

export interface FlightPath3D {
  origin: string;
  destination: string;
  waypoints: Waypoint[];
  duration_seconds: number;
  distance_km: number;
}

export interface VRSessionResponse {
  session_id: string;
  flight_path: FlightPath3D;
  audio_zones: Array<{
    id: string;
    position: { x: number; y: number; z: number };
    sound_type: string;
    volume: number;
    frequency: number;
  }>;
  recommended_duration: number;
  vr_settings: {
    resolution: string;
    fps: number;
    antialiasing: boolean;
    spatial_audio: boolean;
    hand_tracking: boolean;
    room_scale: boolean;
  };
}

export const createVRSession = async (
  request: VRSessionRequest
): Promise<ApiResponse<VRSessionResponse>> => {
  return await apiClient.post<VRSessionResponse>('/vr-ar/create-session', request);
};

export const getSupportedAirports = async (): Promise<ApiResponse<any>> => {
  return await apiClient.get<any>('/vr-ar/supported-airports');
};

export const getVRCapabilities = async (): Promise<ApiResponse<any>> => {
  return await apiClient.get<any>('/vr-ar/vr-capabilities');
};

export const generateSpatialAudio = async (
  origin: string,
  destination: string,
  audioType: string = 'ambient'
): Promise<ApiResponse<any>> => {
  return await apiClient.post<any>('/vr-ar/spatial-audio/generate', {
    origin,
    destination,
    audio_type: audioType,
  });
};

export const getSessionStatus = async (sessionId: string): Promise<ApiResponse<any>> => {
  return await apiClient.get<any>(`/vr-ar/session/${sessionId}/status`);
};
