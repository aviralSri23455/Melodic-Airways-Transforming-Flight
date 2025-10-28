/**
 * RouteVisualization Component
 * Displays flight routes on a canvas-based map without external dependencies
 */

import React, { useEffect, useRef, useState } from 'react';
import { useRouteVisualization } from '../hooks/useRouteVisualization';

export interface RouteVisualizationProps {
  route: {
    origin: string;
    destination: string;
    path: string[];
  };
  onAirportClick?: (iataCode: string) => void;
  className?: string;
  theme?: 'light' | 'dark';
}

export const RouteVisualization: React.FC<RouteVisualizationProps> = ({
  route,
  onAirportClick,
  className = '',
  theme = 'light',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const {
    canvasRef,
    loading,
    error,
  } = useRouteVisualization({
    route,
    onAirportClick,
    theme,
  });

  // Handle responsive sizing with ResizeObserver
  useEffect(() => {
    if (!containerRef.current) return;

    let resizeTimer: number;

    const resizeObserver = new ResizeObserver((entries) => {
      // Debounce resize events to 100ms
      if (resizeTimer) {
        clearTimeout(resizeTimer);
      }

      resizeTimer = window.setTimeout(() => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          setDimensions({ width, height });
        }
      }, 100);
    });

    resizeObserver.observe(containerRef.current);

    return () => {
      if (resizeTimer) {
        clearTimeout(resizeTimer);
      }
      resizeObserver.disconnect();
    };
  }, []);

  if (loading) {
    return (
      <div ref={containerRef} className={`relative w-full h-full flex items-center justify-center ${className}`}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading route visualization...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div ref={containerRef} className={`relative w-full h-full flex items-center justify-center ${className}`}>
        <div className="text-center text-red-600 dark:text-red-400">
          <p className="text-lg font-semibold mb-2">Error loading visualization</p>
          <p className="text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={`relative w-full h-full ${className}`}>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ cursor: 'grab' }}
        width={dimensions.width}
        height={dimensions.height}
      />
    </div>
  );
};
