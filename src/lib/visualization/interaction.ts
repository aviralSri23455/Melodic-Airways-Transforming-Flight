/**
 * Interaction handlers for pan, zoom, hover, and click
 */

import { InteractionConfig, Viewport, AirportCoordinates } from './types';
import { MercatorProjection } from './projection';

export class InteractionHandler {
  private config: InteractionConfig;
  private projection: MercatorProjection;
  private airports: Map<string, AirportCoordinates>;
  private enabled: boolean = false;

  // Pan state
  private isDragging: boolean = false;
  private dragStart: { x: number; y: number } | null = null;
  private dragStartViewport: Viewport | null = null;

  // Hover state
  private hoverDebounceTimer: number | null = null;

  // Bound event handlers
  private boundHandleWheel: (e: WheelEvent) => void;
  private boundHandleMouseDown: (e: MouseEvent) => void;
  private boundHandleMouseMove: (e: MouseEvent) => void;
  private boundHandleMouseUp: (e: MouseEvent) => void;
  private boundHandleClick: (e: MouseEvent) => void;

  constructor(
    config: InteractionConfig,
    projection: MercatorProjection,
    airports: Map<string, AirportCoordinates>
  ) {
    this.config = config;
    this.projection = projection;
    this.airports = airports;

    // Bind event handlers
    this.boundHandleWheel = this.handleWheel.bind(this);
    this.boundHandleMouseDown = this.handleMouseDown.bind(this);
    this.boundHandleMouseMove = this.handleMouseMove.bind(this);
    this.boundHandleMouseUp = this.handleMouseUp.bind(this);
    this.boundHandleClick = this.handleClick.bind(this);
  }

  /**
   * Enable interactions
   */
  enable(): void {
    if (this.enabled) return;

    this.config.canvas.addEventListener('wheel', this.boundHandleWheel, { passive: false });
    this.config.canvas.addEventListener('mousedown', this.boundHandleMouseDown);
    this.config.canvas.addEventListener('mousemove', this.boundHandleMouseMove);
    this.config.canvas.addEventListener('mouseup', this.boundHandleMouseUp);
    this.config.canvas.addEventListener('click', this.boundHandleClick);
    this.config.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    this.enabled = true;
  }

  /**
   * Disable interactions
   */
  disable(): void {
    if (!this.enabled) return;

    this.config.canvas.removeEventListener('wheel', this.boundHandleWheel);
    this.config.canvas.removeEventListener('mousedown', this.boundHandleMouseDown);
    this.config.canvas.removeEventListener('mousemove', this.boundHandleMouseMove);
    this.config.canvas.removeEventListener('mouseup', this.boundHandleMouseUp);
    this.config.canvas.removeEventListener('click', this.boundHandleClick);

    this.enabled = false;
  }

  /**
   * Update projection
   */
  updateProjection(projection: MercatorProjection): void {
    this.projection = projection;
  }

  /**
   * Update airports
   */
  updateAirports(airports: Map<string, AirportCoordinates>): void {
    this.airports = airports;
  }

  private handleWheel(event: WheelEvent): void {
    event.preventDefault();

    const config = this.projection.getConfig();

    // Calculate zoom delta
    const delta = -event.deltaY / 500;
    const newZoom = Math.max(1, Math.min(20, config.zoom + delta));

    // Get mouse position relative to canvas
    const rect = this.config.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // Get world coordinates at mouse position before zoom
    const mouseWorld = this.projection.unproject({ x: mouseX, y: mouseY });

    // Update viewport
    const newViewport: Viewport = {
      ...config,
      zoom: newZoom,
      centerLat: mouseWorld.lat,
      centerLng: mouseWorld.lng,
    };

    this.projection.updateConfig(newViewport);
    this.config.onViewportChange(newViewport);
  }

  private handleMouseDown(event: MouseEvent): void {
    const rect = this.config.canvas.getBoundingClientRect();
    this.isDragging = true;
    this.dragStart = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    this.dragStartViewport = { ...this.projection.getConfig() };
    this.config.canvas.style.cursor = 'grabbing';
  }

  private handleMouseMove(event: MouseEvent): void {
    const rect = this.config.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    if (this.isDragging && this.dragStart && this.dragStartViewport) {
      // Handle panning
      const dx = mouseX - this.dragStart.x;
      const dy = mouseY - this.dragStart.y;

      // Convert pixel delta to world coordinates
      const scale = 256 * Math.pow(2, this.dragStartViewport.zoom);
      const dLng = (-dx / scale) * 360;
      const dLat = (dy / scale) * 360;

      const newViewport: Viewport = {
        ...this.dragStartViewport,
        centerLat: this.dragStartViewport.centerLat + dLat,
        centerLng: this.dragStartViewport.centerLng + dLng,
      };

      this.projection.updateConfig(newViewport);
      this.config.onViewportChange(newViewport);
    } else {
      // Handle hover detection (debounced)
      if (this.hoverDebounceTimer) {
        clearTimeout(this.hoverDebounceTimer);
      }

      this.hoverDebounceTimer = window.setTimeout(() => {
        const airportCode = this.findAirportAt(mouseX, mouseY);
        this.config.onAirportHover(airportCode);
        this.config.canvas.style.cursor = airportCode ? 'pointer' : 'grab';
      }, 50);
    }
  }

  private handleMouseUp(): void {
    this.isDragging = false;
    this.dragStart = null;
    this.dragStartViewport = null;
    this.config.canvas.style.cursor = 'grab';
  }

  private handleClick(event: MouseEvent): void {
    const rect = this.config.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const airportCode = this.findAirportAt(mouseX, mouseY);
    if (airportCode) {
      this.config.onAirportClick(airportCode);
    }
  }

  private findAirportAt(x: number, y: number): string | null {
    const detectionRadius = 20;
    let closestAirport: string | null = null;
    let closestDistance = detectionRadius;

    this.airports.forEach((airport, iataCode) => {
      const screenPos = this.projection.project({
        lat: airport.latitude,
        lng: airport.longitude,
      });

      const distance = Math.sqrt(
        Math.pow(screenPos.x - x, 2) + Math.pow(screenPos.y - y, 2)
      );

      if (distance < closestDistance) {
        closestAirport = iataCode;
        closestDistance = distance;
      }
    });

    return closestAirport;
  }
}
