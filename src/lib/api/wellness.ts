import { apiClient, ApiResponse } from './client';

export interface WellnessRequest {
  theme: string;
  calm_level: number;
  route?: string;
  duration_minutes?: number;
}

export interface WellnessResponse {
  composition_id: string;
  theme: string;
  calm_level: number;
  duration: number;
  notes: Array<{
    note: number;
    velocity: number;
    time: number;
    duration: number;
  }>;
  binaural_frequency?: number;
}

export interface WellnessTheme {
  id: string;
  name: string;
  description: string;
  suggested_routes: string[];
}

export const generateWellnessComposition = async (
  request: WellnessRequest
): Promise<ApiResponse<WellnessResponse>> => {
  return await apiClient.post<WellnessResponse>(
    '/wellness/generate-wellness',
    request
  );
};

export const getWellnessThemes = async (): Promise<ApiResponse<WellnessTheme[]>> => {
  return await apiClient.get<WellnessTheme[]>('/wellness/wellness-themes');
};
