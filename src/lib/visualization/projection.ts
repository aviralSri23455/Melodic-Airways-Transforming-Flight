/**
 * Geographic projection utilities for converting lat/lng to screen coordinates
 * Uses Web Mercator projection (EPSG:3857)
 */

import { Point, LatLng, ProjectionConfig } from './types';

export class MercatorProjection {
  private config: ProjectionConfig;

  constructor(config: ProjectionConfig) {
    this.config = config;
  }

  /**
   * Convert latitude/longitude to screen x/y coordinates
   * @param latLng Geographic coordinates
   * @returns Screen coordinates
   */
  project(latLng: LatLng): Point {
    // Validate input coordinates
    if (!latLng || typeof latLng.lat !== 'number' || typeof latLng.lng !== 'number' || 
        !isFinite(latLng.lat) || !isFinite(latLng.lng)) {
      // Return center point for invalid coordinates
      return { x: this.config.width / 2, y: this.config.height / 2 };
    }

    // Clamp latitude to ±85° to avoid singularities at poles
    const lat = Math.max(-85, Math.min(85, latLng.lat));
    const lng = latLng.lng;

    const scale = 256 * Math.pow(2, this.config.zoom);

    // Convert to radians
    const latRad = (lat * Math.PI) / 180;
    const lngRad = (lng * Math.PI) / 180;

    // Mercator projection
    const x = ((lngRad + Math.PI) / (2 * Math.PI)) * scale;
    const y =
      ((Math.PI - Math.log(Math.tan(Math.PI / 4 + latRad / 2))) / (2 * Math.PI)) * scale;

    // Adjust for viewport center - use direct calculation to avoid recursion
    const centerScale = 256 * Math.pow(2, this.config.zoom);
    const centerLatRad = (this.config.centerLat * Math.PI) / 180;
    const centerLngRad = (this.config.centerLng * Math.PI) / 180;
    const centerX = ((centerLngRad + Math.PI) / (2 * Math.PI)) * centerScale;
    const centerY = ((Math.PI - Math.log(Math.tan(Math.PI / 4 + centerLatRad / 2))) / (2 * Math.PI)) * centerScale;

    return {
      x: x - centerX + this.config.width / 2,
      y: y - centerY + this.config.height / 2,
    };
  }

  /**
   * Convert screen x/y coordinates to latitude/longitude
   * @param point Screen coordinates
   * @returns Geographic coordinates
   */
  unproject(point: Point): LatLng {
    const scale = 256 * Math.pow(2, this.config.zoom);

    // Adjust for viewport center - use direct calculation to avoid recursion
    const centerScale = 256 * Math.pow(2, this.config.zoom);
    const centerLatRad = (this.config.centerLat * Math.PI) / 180;
    const centerLngRad = (this.config.centerLng * Math.PI) / 180;
    const centerX = ((centerLngRad + Math.PI) / (2 * Math.PI)) * centerScale;
    const centerY = ((Math.PI - Math.log(Math.tan(Math.PI / 4 + centerLatRad / 2))) / (2 * Math.PI)) * centerScale;

    const x = point.x + centerX - this.config.width / 2;
    const y = point.y + centerY - this.config.height / 2;

    const lngRad = (x / scale) * (2 * Math.PI) - Math.PI;
    const latRad = 2 * (Math.atan(Math.exp(Math.PI - (y / scale) * (2 * Math.PI))) - Math.PI / 4);

    return {
      lat: (latRad * 180) / Math.PI,
      lng: (lngRad * 180) / Math.PI,
    };
  }

  /**
   * Update projection configuration
   * @param config New configuration
   */
  updateConfig(config: Partial<ProjectionConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   * @returns Current projection configuration
   */
  getConfig(): ProjectionConfig {
    return { ...this.config };
  }

  /**
   * Get bounds of current viewport
   * @returns Viewport bounds (north, south, east, west)
   */
  getBounds(): { north: number; south: number; east: number; west: number } {
    const topLeft = this.unproject({ x: 0, y: 0 });
    const bottomRight = this.unproject({
      x: this.config.width,
      y: this.config.height,
    });

    return {
      north: topLeft.lat,
      south: bottomRight.lat,
      east: bottomRight.lng,
      west: topLeft.lng,
    };
  }

  /**
   * Calculate zoom level to fit all points in viewport
   * @param points Array of geographic coordinates
   * @returns Optimal zoom level
   */
  fitBounds(points: LatLng[]): number {
    if (points.length === 0) return 2;
    if (points.length === 1) return 10;

    // Find bounding box
    let minLat = points[0].lat;
    let maxLat = points[0].lat;
    let minLng = points[0].lng;
    let maxLng = points[0].lng;

    points.forEach((point) => {
      minLat = Math.min(minLat, point.lat);
      maxLat = Math.max(maxLat, point.lat);
      minLng = Math.min(minLng, point.lng);
      maxLng = Math.max(maxLng, point.lng);
    });

    // Handle coordinate wrapping for routes crossing ±180° longitude
    const lngSpan = maxLng - minLng;
    if (lngSpan > 180) {
      // Route crosses date line, adjust coordinates
      const adjustedPoints = points.map((p) => ({
        ...p,
        lng: p.lng < 0 ? p.lng + 360 : p.lng,
      }));

      minLng = adjustedPoints[0].lng;
      maxLng = adjustedPoints[0].lng;
      adjustedPoints.forEach((point) => {
        minLng = Math.min(minLng, point.lng);
        maxLng = Math.max(maxLng, point.lng);
      });
    }

    // Calculate center
    const centerLat = (minLat + maxLat) / 2;
    const centerLng = (minLng + maxLng) / 2;

    // Calculate zoom level to fit bounds
    const latDiff = maxLat - minLat;
    const lngDiff = maxLng - minLng;

    const latZoom = Math.log2(this.config.height / (latDiff * 256)) - 1;
    const lngZoom = Math.log2(this.config.width / (lngDiff * 256)) - 1;

    const zoom = Math.max(1, Math.min(20, Math.floor(Math.min(latZoom, lngZoom))));

    return zoom;
  }

  /**
   * Calculate center point for array of coordinates
   * @param points Array of geographic coordinates
   * @returns Center point
   */
  getCenter(points: LatLng[]): LatLng {
    if (points.length === 0) return { lat: 0, lng: 0 };
    if (points.length === 1) return points[0];

    let minLat = points[0].lat;
    let maxLat = points[0].lat;
    let minLng = points[0].lng;
    let maxLng = points[0].lng;

    points.forEach((point) => {
      minLat = Math.min(minLat, point.lat);
      maxLat = Math.max(maxLat, point.lat);
      minLng = Math.min(minLng, point.lng);
      maxLng = Math.max(maxLng, point.lng);
    });

    // Handle coordinate wrapping for routes crossing ±180° longitude
    const lngSpan = maxLng - minLng;
    if (lngSpan > 180) {
      const adjustedPoints = points.map((p) => ({
        ...p,
        lng: p.lng < 0 ? p.lng + 360 : p.lng,
      }));

      minLng = adjustedPoints[0].lng;
      maxLng = adjustedPoints[0].lng;
      adjustedPoints.forEach((point) => {
        minLng = Math.min(minLng, point.lng);
        maxLng = Math.max(maxLng, point.lng);
      });

      let centerLng = (minLng + maxLng) / 2;
      if (centerLng > 180) centerLng -= 360;

      return {
        lat: (minLat + maxLat) / 2,
        lng: centerLng,
      };
    }

    return {
      lat: (minLat + maxLat) / 2,
      lng: (minLng + maxLng) / 2,
    };
  }
}
