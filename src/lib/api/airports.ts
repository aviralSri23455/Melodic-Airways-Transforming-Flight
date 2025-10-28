// Airport Search Service - Handles airport search API calls

import { apiClient, ApiResponse } from './client';

/**
 * Request parameters for airport search
 */
export interface AirportSearchRequest {
  query: string;
  limit?: number;
  country?: string;
}

/**
 * Airport data model
 */
export interface Airport {
  id: number;
  name: string;
  city: string;
  country: string;
  iata_code: string;
  latitude: number;
  longitude: number;
}


// In-memory cache for search results
const searchCache = new Map<string, { data: Airport[]; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

/**
 * Search for airports by name, city, or IATA code
 */
export async function searchAirports(
  request: AirportSearchRequest
): Promise<ApiResponse<Airport[]>> {
  try {
    // Check cache first
    const cacheKey = JSON.stringify(request);
    const cached = searchCache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      return {
        data: cached.data,
        error: null,
        status: 200,
      };
    }

    // Make API request
    const response = await apiClient.get<Airport[]>('/airports/search', {
      query: request.query,
      limit: request.limit || 20,
      country: request.country,
    });

    // Cache successful results
    if (response.data) {
      searchCache.set(cacheKey, {
        data: response.data,
        timestamp: Date.now(),
      });
    }

    return response;
  } catch (error) {
    return {
      data: null,
      error: {
        message: 'Failed to search airports',
        code: 'SEARCH_ERROR',
        details: error,
      },
      status: 500,
    };
  }
}


/**
 * Debounce utility function
 */
function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => Promise<ReturnType<T>> {
  let timeoutId: NodeJS.Timeout | null = null;
  let pendingResolve: ((value: ReturnType<T>) => void) | null = null;

  return (...args: Parameters<T>): Promise<ReturnType<T>> => {
    return new Promise((resolve) => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      pendingResolve = resolve;

      timeoutId = setTimeout(async () => {
        const result = await func(...args);
        if (pendingResolve) {
          pendingResolve(result);
        }
        timeoutId = null;
        pendingResolve = null;
      }, delay);
    });
  };
}

/**
 * Debounced version of searchAirports (300ms delay)
 */
export const searchAirportsDebounced = debounce(searchAirports, 300);


/**
 * Get airport by IATA code
 */
export async function getAirportByCode(iataCode: string): Promise<ApiResponse<Airport>> {
  try {
    // Validate IATA code format (3 uppercase letters)
    const iataRegex = /^[A-Z]{3}$/;
    if (!iataRegex.test(iataCode)) {
      return {
        data: null,
        error: {
          message: 'Invalid IATA code format. Must be 3 uppercase letters.',
          code: 'INVALID_IATA_CODE',
        },
        status: 400,
      };
    }

    // Make API request
    const response = await apiClient.get<Airport>(`/airports/${iataCode}`);
    return response;
  } catch (error) {
    return {
      data: null,
      error: {
        message: 'Failed to get airport',
        code: 'GET_AIRPORT_ERROR',
        details: error,
      },
      status: 500,
    };
  }
}
