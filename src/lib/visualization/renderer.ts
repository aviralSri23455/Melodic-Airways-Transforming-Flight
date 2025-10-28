/**
 * Canvas rendering utilities for drawing flight routes and airports
 */

import { LatLng, RouteStyle, AirportStyle } from './types';
import { MercatorProjection } from './projection';

export const THEME_COLORS = {
  light: {
    background: '#f8f9fa',
    routeLine: '#3b82f6',
    routeGradient: ['#3b82f6', '#8b5cf6'],
    airportOrigin: '#10b981',
    airportDestination: '#ef4444',
    airportIntermediate: '#6b7280',
    airportStroke: '#ffffff',
    label: '#1f2937',
    labelBackground: 'rgba(255, 255, 255, 0.9)',
    grid: '#e5e7eb',
  },
  dark: {
    background: '#1f2937',
    routeLine: '#60a5fa',
    routeGradient: ['#60a5fa', '#a78bfa'],
    airportOrigin: '#34d399',
    airportDestination: '#f87171',
    airportIntermediate: '#9ca3af',
    airportStroke: '#374151',
    label: '#f9fafb',
    labelBackground: 'rgba(31, 41, 55, 0.9)',
    grid: '#374151',
  },
};

export class CanvasRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private projection: MercatorProjection;
  private theme: 'light' | 'dark';
  private dpr: number;

  constructor(
    canvas: HTMLCanvasElement,
    projection: MercatorProjection,
    theme: 'light' | 'dark' = 'light'
  ) {
    this.canvas = canvas;
    this.projection = projection;
    this.theme = theme;
    this.dpr = window.devicePixelRatio || 1;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get 2D context');
    }
    this.ctx = ctx;

    this.setupCanvas();
  }

  /**
   * Set up canvas with proper scaling for retina displays
   */
  private setupCanvas(): void {
    const config = this.projection.getConfig();
    this.canvas.width = config.width * this.dpr;
    this.canvas.height = config.height * this.dpr;
    this.canvas.style.width = `${config.width}px`;
    this.canvas.style.height = `${config.height}px`;
    this.ctx.scale(this.dpr, this.dpr);
  }

  /**
   * Clear the canvas
   */
  clear(): void {
    const config = this.projection.getConfig();
    this.ctx.fillStyle = THEME_COLORS[this.theme].background;
    this.ctx.fillRect(0, 0, config.width, config.height);
  }

  /**
   * Update theme
   * @param theme New theme
   */
  setTheme(theme: 'light' | 'dark'): void {
    this.theme = theme;
  }

  /**
   * Update projection
   * @param projection New projection
   */
  updateProjection(projection: MercatorProjection): void {
    this.projection = projection;
    this.setupCanvas();
  }

  /**
   * Draw a route path
   * @param path Array of geographic coordinates
   * @param style Route style
   */
  drawRoute(path: LatLng[], style: RouteStyle): void {
    if (path.length < 2) return;

    this.ctx.strokeStyle = style.strokeColor;
    this.ctx.lineWidth = style.strokeWidth;

    if (style.dashArray) {
      this.ctx.setLineDash(style.dashArray);
    } else {
      this.ctx.setLineDash([]);
    }

    this.ctx.beginPath();
    const startPoint = this.projection.project(path[0]);
    this.ctx.moveTo(startPoint.x, startPoint.y);

    for (let i = 1; i < path.length; i++) {
      const point = this.projection.project(path[i]);
      this.ctx.lineTo(point.x, point.y);
    }

    this.ctx.stroke();
  }

  /**
   * Draw a great circle arc between two points
   * @param start Start coordinate
   * @param end End coordinate
   * @param style Route style
   */
  drawGreatCircle(start: LatLng, end: LatLng, style: RouteStyle): void {
    const points = this.calculateGreatCircle(start, end, 100);
    this.drawRoute(points, style);
  }

  /**
   * Calculate great circle intermediate points
   * @param start Start coordinate
   * @param end End coordinate
   * @param segments Number of segments
   * @returns Array of intermediate points
   */
  private calculateGreatCircle(start: LatLng, end: LatLng, segments: number): LatLng[] {
    const points: LatLng[] = [];

    // Convert to radians
    const lat1 = (start.lat * Math.PI) / 180;
    const lng1 = (start.lng * Math.PI) / 180;
    const lat2 = (end.lat * Math.PI) / 180;
    const lng2 = (end.lng * Math.PI) / 180;

    // Calculate angular distance
    const d =
      2 *
      Math.asin(
        Math.sqrt(
          Math.pow(Math.sin((lat1 - lat2) / 2), 2) +
            Math.cos(lat1) * Math.cos(lat2) * Math.pow(Math.sin((lng1 - lng2) / 2), 2)
        )
      );

    // Handle case where points are very close
    if (d < 0.0001) {
      return [start, end];
    }

    // Generate intermediate points
    for (let i = 0; i <= segments; i++) {
      const f = i / segments;
      const a = Math.sin((1 - f) * d) / Math.sin(d);
      const b = Math.sin(f * d) / Math.sin(d);

      const x = a * Math.cos(lat1) * Math.cos(lng1) + b * Math.cos(lat2) * Math.cos(lng2);
      const y = a * Math.cos(lat1) * Math.sin(lng1) + b * Math.cos(lat2) * Math.sin(lng2);
      const z = a * Math.sin(lat1) + b * Math.sin(lat2);

      const lat = Math.atan2(z, Math.sqrt(x * x + y * y));
      const lng = Math.atan2(y, x);

      points.push({
        lat: (lat * 180) / Math.PI,
        lng: (lng * 180) / Math.PI,
      });
    }

    return points;
  }

  /**
   * Draw an airport marker
   * @param position Airport coordinates
   * @param style Airport style
   */
  drawAirport(position: LatLng, style: AirportStyle): void {
    const point = this.projection.project(position);

    this.ctx.beginPath();
    this.ctx.arc(point.x, point.y, style.radius, 0, 2 * Math.PI);
    this.ctx.fillStyle = style.fillColor;
    this.ctx.fill();
    this.ctx.strokeStyle = style.strokeColor;
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
  }

  /**
   * Draw an airport label
   * @param position Airport coordinates
   * @param text Label text (IATA code)
   */
  drawLabel(position: LatLng, text: string): void {
    const point = this.projection.project(position);
    const colors = THEME_COLORS[this.theme];

    this.ctx.font = '12px sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'bottom';

    // Measure text for background
    const metrics = this.ctx.measureText(text);
    const padding = 4;
    const bgWidth = metrics.width + padding * 2;
    const bgHeight = 16;

    // Draw semi-transparent background
    this.ctx.fillStyle = colors.labelBackground;
    this.ctx.fillRect(
      point.x - bgWidth / 2,
      point.y - bgHeight - 8,
      bgWidth,
      bgHeight
    );

    // Draw text
    this.ctx.fillStyle = colors.label;
    this.ctx.fillText(text, point.x, point.y - 8);
  }

  /**
   * Draw a tooltip for an airport
   * @param position Airport coordinates
   * @param name Airport name
   * @param city City name
   * @param country Country name
   */
  drawTooltip(
    position: LatLng,
    name: string,
    city: string,
    country: string
  ): void {
    const point = this.projection.project(position);
    const colors = THEME_COLORS[this.theme];

    this.ctx.font = '14px sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';

    const lines = [name, `${city}, ${country}`];
    const padding = 8;
    const lineHeight = 18;

    // Measure text for background
    let maxWidth = 0;
    lines.forEach((line) => {
      const metrics = this.ctx.measureText(line);
      maxWidth = Math.max(maxWidth, metrics.width);
    });

    const bgWidth = maxWidth + padding * 2;
    const bgHeight = lines.length * lineHeight + padding * 2;

    // Position tooltip above and to the right of airport
    const tooltipX = point.x + 15;
    const tooltipY = point.y - bgHeight - 15;

    // Draw background with border
    this.ctx.fillStyle = colors.labelBackground;
    this.ctx.fillRect(tooltipX, tooltipY, bgWidth, bgHeight);
    this.ctx.strokeStyle = colors.label;
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(tooltipX, tooltipY, bgWidth, bgHeight);

    // Draw text lines
    this.ctx.fillStyle = colors.label;
    lines.forEach((line, index) => {
      this.ctx.fillText(
        line,
        tooltipX + padding,
        tooltipY + padding + index * lineHeight
      );
    });
  }
}
