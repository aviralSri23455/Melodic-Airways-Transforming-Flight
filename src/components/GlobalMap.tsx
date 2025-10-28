import { Card } from "@/components/ui/card";
import { RouteVisualization } from "./RouteVisualization";
import { useTheme } from "next-themes";

interface GlobalMapProps {
  origin?: { code: string; lat: number; lng: number };
  destination?: { code: string; lat: number; lng: number };
  path?: string[];
}

const GlobalMap = ({ origin, destination, path }: GlobalMapProps) => {
  const { theme } = useTheme();

  // If no route data, show placeholder
  if (!origin || !destination) {
    return (
      <Card className="cockpit-panel p-8">
        <div className="text-center space-y-4">
          <p className="text-muted-foreground">
            Select origin and destination airports to view the route
          </p>
        </div>
      </Card>
    );
  }

  // Build route path
  const routePath = path || [origin.code, destination.code];

  return (
    <Card className="cockpit-panel overflow-hidden h-[600px]">
      <RouteVisualization
        route={{
          origin: origin.code,
          destination: destination.code,
          path: routePath,
        }}
        theme={theme === 'dark' ? 'dark' : 'light'}
        className="w-full h-full"
      />
    </Card>
  );
};

export default GlobalMap;
