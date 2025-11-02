// Music Generation Service - Handles music generation API calls

import { apiClient, ApiResponse } from './client';

/**
 * Request parameters for music generation
 */
export interface MusicGenerationRequest {
  origin: string;          // IATA code (e.g., "JFK")
  destination: string;     // IATA code (e.g., "LAX")
  music_style: string;     // e.g., "ambient", "classical"
  tempo: number;           // BPM (60-200)
}

/**
 * Airport information
 */
export interface AirportInfo {
  code: string;
  name: string;
  city: string;
  country: string;
}

/**
 * Route computation step from backend
 */
export interface RouteComputationStep {
  status: string;
  path: string[];
  distance_km: number;
  intermediate_stops: number;
  origin_airport: AirportInfo;
  destination_airport: AirportInfo;
}

/**
 * MIDI generation step from backend
 */
export interface MidiGenerationStep {
  status: string;
  composition_id: number;
  duration_seconds: number;
  note_count: number;
  notes?: Array<{
    note: number;
    velocity: number;
    time: number;
    duration: number;
  }>;
}

/**
 * Complete response from backend demo endpoint
 */
export interface MusicGenerationResponse {
  message: string;
  demo_results: {
    demo_id: string;
    origin: string;
    destination: string;
    demo_status: string;
    steps: {
      route_computation: RouteComputationStep;
      vector_embedding: any;
      similar_routes: any;
      midi_generation: MidiGenerationStep;
      redis_broadcast: any;
      duckdb_analytics: any;
    };
    timing: Record<string, number>;
    tech_stack_used: string[];
  };
  tech_stack_showcase: any;
  demo_flow_summary: any;
}

/**
 * Simplified result for UI consumption
 */
export interface MusicGenerationResult {
  compositionId: number;
  route: {
    origin: AirportInfo;
    destination: AirportInfo;
    distance: number;
    path: string[];
  };
  music: {
    duration: number;
    noteCount: number;
    tempo: number;
    style: string;
    scale?: string;
    root_note?: number;
    tracks?: {
      melody: number;
      harmony: number;
      bass: number;
    };
    update_id?: string;
    notes?: Array<{
      note: number;
      velocity: number;
      time: number;
      duration: number;
      type?: string;
    }>;
  };
  analytics: {
    complexity: number;
    harmonic_richness: number;
  };
}


/**
 * Generate music from a flight route
 */
export async function generateMusic(
  request: MusicGenerationRequest
): Promise<ApiResponse<MusicGenerationResult>> {
  try {
    // Call the backend demo endpoint
    const response = await apiClient.get<MusicGenerationResponse>('/demo/complete-demo', {
      origin: request.origin,
      destination: request.destination,
      music_style: request.music_style,
      tempo: request.tempo,
    });

    // If there's an error, return it
    if (response.error) {
      return {
        data: null,
        error: response.error,
        status: response.status,
      };
    }

    // Transform the response to simplified format
    if (response.data) {
      console.log('Raw backend response:', response.data);
      const result = transformResponse(response.data, request);
      console.log('Transformed result:', result);
      return {
        data: result,
        error: null,
        status: response.status,
      };
    }

    // Unexpected case: no data and no error
    return {
      data: null,
      error: {
        message: 'No data received from server',
        code: 'NO_DATA',
      },
      status: response.status,
    };
  } catch (error) {
    return {
      data: null,
      error: {
        message: 'Failed to generate music',
        code: 'GENERATION_ERROR',
        details: error,
      },
      status: 500,
    };
  }
}


/**
 * Transform backend response to simplified UI format
 */
function transformResponse(
  response: MusicGenerationResponse,
  request: MusicGenerationRequest
): MusicGenerationResult {
  console.log('Transforming response:', response);
  
  // Handle the actual backend response structure
  // Backend wraps data in demo_results, frontend expects direct structure
  const actualData = response.demo_results || response;
  console.log('Actual demo data:', actualData);
  
  const routeStep = actualData.steps?.route_computation;
  const midiStep = actualData.steps?.midi_generation;

  console.log('Route step:', routeStep);
  console.log('MIDI step:', midiStep);

  // Check if we have the required data
  if (!routeStep || !midiStep) {
    console.error('Missing required data in response:', { routeStep, midiStep });
    throw new Error('Invalid response structure: missing route or MIDI data');
  }

  // Calculate real analytics from backend data
  const distance = routeStep.distance_km;
  const noteCount = midiStep.note_count;
  const duration = midiStep.duration_seconds;
  
  console.log('Extracted data:', { distance, noteCount, duration });
  
  // Calculate complexity based on route characteristics
  const complexity = Math.min(1.0, (distance / 10000) * 0.6 + (noteCount / 100) * 0.4);
  
  // Calculate harmonic richness based on music style and duration
  const harmonicRichness = Math.min(1.0, (duration / 30) * 0.7 + 0.3);

  console.log('Calculated analytics:', { complexity, harmonicRichness });

  // Extract additional music data from backend response
  const backendMusicData = actualData.steps?.midi_generation || {};
  
  return {
    compositionId: midiStep.composition_id,
    route: {
      origin: {
        code: routeStep.origin_airport.code,
        name: routeStep.origin_airport.name,
        city: routeStep.origin_airport.city,
        country: routeStep.origin_airport.country,
      },
      destination: {
        code: routeStep.destination_airport.code,
        name: routeStep.destination_airport.name,
        city: routeStep.destination_airport.city,
        country: routeStep.destination_airport.country,
      },
      distance: routeStep.distance_km,
      path: routeStep.path,
    },
    music: {
      duration: midiStep.duration_seconds,
      noteCount: midiStep.note_count,
      tempo: request.tempo,
      style: request.music_style,
      scale: backendMusicData.scale || request.music_style,
      root_note: backendMusicData.root_note,
      tracks: backendMusicData.tracks || { melody: 0, harmony: 0, bass: 0 },
      update_id: `${request.origin}_${request.destination}_${Date.now()}`,
      notes: midiStep.notes || [],  // Include actual notes from backend
    },
    analytics: {
      complexity: complexity,
      harmonic_richness: harmonicRichness,
    },
  };
}
