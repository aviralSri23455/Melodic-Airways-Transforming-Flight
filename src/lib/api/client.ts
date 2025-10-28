// API Client - Base HTTP client for backend communication

import axios, { AxiosInstance, AxiosError } from 'axios';

/**
 * Configuration for the API client
 */
export interface ApiClientConfig {
  baseURL: string;
  timeout: number;
}

/**
 * Structured API response wrapper
 */
export interface ApiResponse<T> {
  data: T | null;
  error: ApiError | null;
  status: number;
}

/**
 * Structured error response
 */
export interface ApiError {
  message: string;
  code: string;
  details?: any;
}


/**
 * API Client for making HTTP requests to the backend
 */
export class ApiClient {
  private axiosInstance: AxiosInstance;

  constructor(config: ApiClientConfig) {
    this.axiosInstance = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }


  /**
   * Make a GET request
   */
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<ApiResponse<T>> {
    try {
      const url = this.buildURL(endpoint, params);
      const response = await this.axiosInstance.get<T>(url);
      
      return {
        data: response.data,
        error: null,
        status: response.status,
      };
    } catch (error) {
      const apiError = this.handleError(error);
      return {
        data: null,
        error: apiError,
        status: (error as AxiosError).response?.status || 500,
      };
    }
  }


  /**
   * Make a POST request
   */
  async post<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    try {
      const response = await this.axiosInstance.post<T>(endpoint, data);
      
      return {
        data: response.data,
        error: null,
        status: response.status,
      };
    } catch (error) {
      const apiError = this.handleError(error);
      return {
        data: null,
        error: apiError,
        status: (error as AxiosError).response?.status || 500,
      };
    }
  }


  /**
   * Make a DELETE request
   */
  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    try {
      const response = await this.axiosInstance.delete<T>(endpoint);
      
      return {
        data: response.data,
        error: null,
        status: response.status,
      };
    } catch (error) {
      const apiError = this.handleError(error);
      return {
        data: null,
        error: apiError,
        status: (error as AxiosError).response?.status || 500,
      };
    }
  }


  /**
   * Handle errors and convert to ApiError
   */
  private handleError(error: any): ApiError {
    const debugMode = import.meta.env.VITE_DEBUG_MODE === 'true';
    
    // Log errors in debug mode
    if (debugMode) {
      console.error('API Error:', error);
    }

    // Network timeout errors
    if (error.code === 'ECONNABORTED') {
      return {
        message: 'Request timeout. Please try again.',
        code: 'TIMEOUT',
      };
    }

    // Network connection errors
    if (error.code === 'ERR_NETWORK') {
      return {
        message: 'Network error. Please check your connection.',
        code: 'NETWORK_ERROR',
      };
    }

    // HTTP errors
    if (error.response) {
      const status = error.response.status;
      const detail = error.response.data?.detail || 'An error occurred';

      return {
        message: detail,
        code: `HTTP_${status}`,
        details: error.response.data,
      };
    }

    // Unknown errors
    return {
      message: 'An unexpected error occurred',
      code: 'UNKNOWN_ERROR',
      details: error.message,
    };
  }


  /**
   * Build URL with query parameters
   */
  private buildURL(endpoint: string, params?: Record<string, any>): string {
    if (!params || Object.keys(params).length === 0) {
      return endpoint;
    }

    const queryString = Object.entries(params)
      .filter(([_, value]) => value !== undefined && value !== null)
      .map(([key, value]) => {
        if (Array.isArray(value)) {
          return value.map(v => `${encodeURIComponent(key)}=${encodeURIComponent(v)}`).join('&');
        }
        return `${encodeURIComponent(key)}=${encodeURIComponent(value)}`;
      })
      .join('&');

    return queryString ? `${endpoint}?${queryString}` : endpoint;
  }
}

/**
 * Create and export a default API client instance
 */
export const apiClient = new ApiClient({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  timeout: parseInt(import.meta.env.VITE_API_TIMEOUT || '10000'),
});
