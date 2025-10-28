/**
 * Visualization type definitions for custom flight route rendering
 */

export interface Point {
  x: number;
  y: number;
}

export interface LatLng {
  lat: number;
  lng: number;
}

export interface Viewport {
  centerLat: number;
  centerLng: number;
  zoom: number;
  width: number;
  height: number;
}

export interface ProjectionConfig {
  width: number;
  height: number;
  centerLat: number;
  centerLng: number;
  zoom: number;
}

export interface RenderConfig {
  canvas: HTMLCanvasElement;
  theme: 'light' | 'dark';
}

export interface RouteStyle {
  strokeColor: string;
  strokeWidth: number;
  dashArray?: number[];
}

export interface AirportStyle {
  fillColor: string;
  strokeColor: string;
  radius: number;
}

export interface InteractionConfig {
  canvas: HTMLCanvasElement;
  onViewportChange: (viewport: Viewport) => void;
  onAirportHover: (iataCode: string | null) => void;
  onAirportClick: (iataCode: string) => void;
}

export interface AirportCoordinates {
  iataCode: string;
  name: string;
  city: string;
  country: string;
  latitude: number;
  longitude: number;
}
