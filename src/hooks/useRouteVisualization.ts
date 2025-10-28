/**
 * Custom hook for managing route visualization state and lifecycle
 */

import { useRef, useState, useEffect, useCallback } from 'react';
import { Viewport, AirportCoordinates } from '../lib/visualization/types';
import { MercatorProjection } from '../lib/visualization/projection';
import { CanvasRenderer } from '../lib/visualization/renderer';
import { InteractionHandler } from '../lib/visualization/interaction';

export interface UseRouteVisualizationOptions {
  route: {
    origin: string;
    destination: string;
    path: string[];
  };
  onAirportClick?: (iataCode: string) => void;
  theme?: 'light' | 'dark';
}

export interface UseRouteVisualizationReturn {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  airports: Map<string, AirportCoordinates>;
  viewport: Viewport;
  hoveredAirport: string | null;
  loading: boolean;
  error: string | null;
  resetView: () => void;
}

export function useRouteVisualization(
  options: UseRouteVisualizationOptions
): UseRouteVisualizationReturn {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [airports, setAirports] = useState<Map<string, AirportCoordinates>>(new Map());
  const [viewport, setViewport] = useState<Viewport>({
    centerLat: 0,
    centerLng: 0,
    zoom: 2,
    width: 800,
    height: 600,
  });
  const [hoveredAirport, setHoveredAirport] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Service instances
  const projectionRef = useRef<MercatorProjection | null>(null);
  const rendererRef = useRef<CanvasRenderer | null>(null);
  const interactionRef = useRef<InteractionHandler | null>(null);

  // Fetch airport data
  useEffect(() => {
    const abortController = new AbortController();
    
    async function fetchAirports() {
      setLoading(true);
      setError(null);

      try {
        const { getAirportByCode } = await import('../lib/api/airports');
        const airportMap = new Map<string, AirportCoordinates>();

        // Fetch all airports in the route
        const uniqueAirports = Array.from(new Set(options.route.path));
        
        await Promise.all(
          uniqueAirports.map(async (iataCode) => {
            try {
              const response = await getAirportByCode(iataCode);
              
              if (response.data && response.data.latitude && response.data.longitude) {
                // Validate coordinates before storing
                const lat = response.data.latitude;
                const lng = response.data.longitude;
                
                if (isFinite(lat) && isFinite(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
                  airportMap.set(iataCode, {
                    iataCode: response.data.iata_code,
                    name: response.data.name,
                    city: response.data.city,
                    country: response.data.country,
                    latitude: lat,
                    longitude: lng,
                  });
                } else {
                  console.warn(`Invalid coordinates for airport ${iataCode}:`, { lat, lng });
                }
              }
            } catch (err) {
              console.warn(`Failed to fetch airport ${iataCode}:`, err);
            }
          })
        );

        if (!abortController.signal.aborted) {
          setAirports(airportMap);
          setLoading(false);
        }
      } catch (err) {
        if (!abortController.signal.aborted) {
          setError('Failed to load airport data');
          setLoading(false);
        }
      }
    }

    fetchAirports();

    return () => {
      abortController.abort();
    };
  }, [options.route.path]);

  // Initialize visualization services
  useEffect(() => {
    if (!canvasRef.current || airports.size === 0) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    // Get airport coordinates
    const airportCoords = Array.from(airports.values()).map((a) => ({
      lat: a.latitude,
      lng: a.longitude,
    }));

    // Create projection
    const projection = new MercatorProjection({
      width: rect.width,
      height: rect.height,
      centerLat: 0,
      centerLng: 0,
      zoom: 2,
    });

    // Calculate initial viewport to fit route
    const center = projection.getCenter(airportCoords);
    const zoom = projection.fitBounds(airportCoords);

    const initialViewport: Viewport = {
      centerLat: center.lat,
      centerLng: center.lng,
      zoom,
      width: rect.width,
      height: rect.height,
    };

    projection.updateConfig(initialViewport);
    setViewport(initialViewport);

    // Create renderer
    const renderer = new CanvasRenderer(
      canvas,
      projection,
      options.theme || 'light'
    );

    // Create interaction handler
    const interaction = new InteractionHandler(
      {
        canvas,
        onViewportChange: (newViewport) => {
          setViewport(newViewport);
        },
        onAirportHover: (iataCode) => {
          setHoveredAirport(iataCode);
        },
        onAirportClick: (iataCode) => {
          options.onAirportClick?.(iataCode);
        },
      },
      projection,
      airports
    );

    interaction.enable();

    projectionRef.current = projection;
    rendererRef.current = renderer;
    interactionRef.current = interaction;

    return () => {
      interaction.disable();
    };
  }, [airports, options.onAirportClick, options.theme]);

  // Render loop
  useEffect(() => {
    if (!rendererRef.current || !projectionRef.current || airports.size === 0) return;

    const renderer = rendererRef.current;
    const projection = projectionRef.current;
    let animationFrameId: number;

    function render() {
      if (!renderer || !projection) return;

      // Clear canvas
      renderer.clear();

      // Draw routes
      const routePath = options.route.path;
      const { routeLine } = renderer.constructor.name.includes('Canvas') 
        ? (renderer as any).constructor.THEME_COLORS?.[options.theme || 'light'] || {}
        : {};

      for (let i = 0; i < routePath.length - 1; i++) {
        const startAirport = airports.get(routePath[i]);
        const endAirport = airports.get(routePath[i + 1]);

        if (startAirport && endAirport) {
          renderer.drawGreatCircle(
            { lat: startAirport.latitude, lng: startAirport.longitude },
            { lat: endAirport.latitude, lng: endAirport.longitude },
            {
              strokeColor: routeLine || '#3b82f6',
              strokeWidth: 2,
            }
          );
        }
      }

      // Draw airports
      routePath.forEach((iataCode, index) => {
        const airport = airports.get(iataCode);
        if (!airport) return;

        const isOrigin = index === 0;
        const isDestination = index === routePath.length - 1;
        const isEndpoint = isOrigin || isDestination;

        const colors = (renderer as any).constructor.THEME_COLORS?.[options.theme || 'light'] || {};

        renderer.drawAirport(
          { lat: airport.latitude, lng: airport.longitude },
          {
            fillColor: isOrigin
              ? colors.airportOrigin || '#10b981'
              : isDestination
              ? colors.airportDestination || '#ef4444'
              : colors.airportIntermediate || '#6b7280',
            strokeColor: colors.airportStroke || '#ffffff',
            radius: isEndpoint ? 8 : 5,
          }
        );

        // Draw label
        renderer.drawLabel(
          { lat: airport.latitude, lng: airport.longitude },
          airport.iataCode
        );
      });

      // Draw tooltip for hovered airport
      if (hoveredAirport) {
        const airport = airports.get(hoveredAirport);
        if (airport) {
          renderer.drawTooltip(
            { lat: airport.latitude, lng: airport.longitude },
            airport.name,
            airport.city,
            airport.country
          );
        }
      }
    }

    function animate() {
      render();
      animationFrameId = requestAnimationFrame(animate);
    }

    animate();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [airports, viewport, hoveredAirport, options.route.path, options.theme]);

  const resetView = useCallback(() => {
    if (!projectionRef.current || airports.size === 0) return;

    const airportCoords = Array.from(airports.values()).map((a) => ({
      lat: a.latitude,
      lng: a.longitude,
    }));

    const center = projectionRef.current.getCenter(airportCoords);
    const zoom = projectionRef.current.fitBounds(airportCoords);

    const newViewport: Viewport = {
      ...viewport,
      centerLat: center.lat,
      centerLng: center.lng,
      zoom,
    };

    projectionRef.current.updateConfig(newViewport);
    setViewport(newViewport);
  }, [airports, viewport]);

  return {
    canvasRef,
    airports,
    viewport,
    hoveredAirport,
    loading,
    error,
    resetView,
  };
}
