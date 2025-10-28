import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Plane, MapPin, ArrowRight, Music } from "lucide-react";
import { searchAirportsDebounced, Airport as ApiAirport } from "@/lib/api/airports";

interface Airport {
  code: string;
  name: string;
  city: string;
  country: string;
}

const popularAirports: Airport[] = [
  { code: "JFK", name: "John F. Kennedy International", city: "New York", country: "USA" },
  { code: "CDG", name: "Charles de Gaulle", city: "Paris", country: "France" },
  { code: "LHR", name: "Heathrow", city: "London", country: "UK" },
  { code: "NRT", name: "Narita International", city: "Tokyo", country: "Japan" },
  { code: "DXB", name: "Dubai International", city: "Dubai", country: "UAE" },
  { code: "SYD", name: "Sydney Kingsford Smith", city: "Sydney", country: "Australia" },
];

interface RouteSelectorProps {
  onRouteSelect: (origin: Airport, destination: Airport) => void;
}

const RouteSelector = ({ onRouteSelect }: RouteSelectorProps) => {
  const [origin, setOrigin] = useState<Airport | null>(null);
  const [destination, setDestination] = useState<Airport | null>(null);
  const [searchOrigin, setSearchOrigin] = useState("");
  const [searchDest, setSearchDest] = useState("");
  const [originSuggestions, setOriginSuggestions] = useState<Airport[]>([]);
  const [destSuggestions, setDestSuggestions] = useState<Airport[]>([]);
  const [showOriginDropdown, setShowOriginDropdown] = useState(false);
  const [showDestDropdown, setShowDestDropdown] = useState(false);

  // Search airports from backend API
  useEffect(() => {
    const searchAirports = async () => {
      if (searchOrigin.length < 2) {
        setOriginSuggestions([]);
        return;
      }

      const response = await searchAirportsDebounced({
        query: searchOrigin,
        limit: 5,
      });

      if (response.data) {
        setOriginSuggestions(response.data.map(a => ({
          code: a.iata_code,
          name: a.name,
          city: a.city,
          country: a.country
        })));
        setShowOriginDropdown(true);
      } else if (response.error) {
        console.error('Airport search error:', response.error);
        setOriginSuggestions([]);
      }
    };

    searchAirports();
  }, [searchOrigin]);

  useEffect(() => {
    const searchAirports = async () => {
      if (searchDest.length < 2) {
        setDestSuggestions([]);
        return;
      }

      const response = await searchAirportsDebounced({
        query: searchDest,
        limit: 5,
      });

      if (response.data) {
        setDestSuggestions(response.data.map(a => ({
          code: a.iata_code,
          name: a.name,
          city: a.city,
          country: a.country
        })));
        setShowDestDropdown(true);
      } else if (response.error) {
        console.error('Airport search error:', response.error);
        setDestSuggestions([]);
      }
    };

    searchAirports();
  }, [searchDest]);

  const handleQuickSelect = (airport: Airport, type: "origin" | "destination") => {
    if (type === "origin") {
      setOrigin(airport);
      setSearchOrigin(airport.code);
    } else {
      setDestination(airport);
      setSearchDest(airport.code);
    }
  };

  const handleGenerateRoute = () => {
    if (origin && destination) {
      onRouteSelect(origin, destination);
    }
  };

  return (
    <Card className="cockpit-panel p-6">
      <div className="flex items-center gap-2 mb-6">
        <Plane className="w-5 h-5 text-primary" />
        <h3 className="text-xl font-semibold">Select Flight Route</h3>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Origin */}
        <div className="space-y-3 relative">
          <Label className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-primary" />
            Origin Airport
          </Label>
          <Input
            placeholder="Search airport code or city..."
            value={searchOrigin}
            onChange={(e) => {
              setSearchOrigin(e.target.value.toUpperCase());
              setOrigin(null);
            }}
            onFocus={() => originSuggestions.length > 0 && setShowOriginDropdown(true)}
            className="bg-background/50"
          />
          {showOriginDropdown && originSuggestions.length > 0 && (
            <div className="absolute z-50 w-full mt-1 bg-card border border-border rounded-md shadow-lg max-h-60 overflow-auto">
              {originSuggestions.map((airport) => (
                <div
                  key={airport.code}
                  className="px-3 py-2 hover:bg-accent cursor-pointer border-b border-border last:border-0"
                  onClick={() => {
                    setOrigin(airport);
                    setSearchOrigin(airport.code);
                    setShowOriginDropdown(false);
                  }}
                >
                  <div className="font-semibold text-sm">{airport.code} - {airport.name}</div>
                  <div className="text-xs text-muted-foreground">{airport.city}, {airport.country}</div>
                </div>
              ))}
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            {popularAirports.slice(0, 3).map((airport) => (
              <Badge
                key={airport.code}
                variant={origin?.code === airport.code ? "default" : "outline"}
                className="cursor-pointer hover:bg-primary/20 transition-colors"
                onClick={() => handleQuickSelect(airport, "origin")}
              >
                {airport.code}
              </Badge>
            ))}
          </div>
        </div>

        {/* Destination */}
        <div className="space-y-3 relative">
          <Label className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-accent" />
            Destination Airport
          </Label>
          <Input
            placeholder="Search airport code or city..."
            value={searchDest}
            onChange={(e) => {
              setSearchDest(e.target.value.toUpperCase());
              setDestination(null);
            }}
            onFocus={() => destSuggestions.length > 0 && setShowDestDropdown(true)}
            className="bg-background/50"
          />
          {showDestDropdown && destSuggestions.length > 0 && (
            <div className="absolute z-50 w-full mt-1 bg-card border border-border rounded-md shadow-lg max-h-60 overflow-auto">
              {destSuggestions.map((airport) => (
                <div
                  key={airport.code}
                  className="px-3 py-2 hover:bg-accent cursor-pointer border-b border-border last:border-0"
                  onClick={() => {
                    setDestination(airport);
                    setSearchDest(airport.code);
                    setShowDestDropdown(false);
                  }}
                >
                  <div className="font-semibold text-sm">{airport.code} - {airport.name}</div>
                  <div className="text-xs text-muted-foreground">{airport.city}, {airport.country}</div>
                </div>
              ))}
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            {popularAirports.slice(3, 6).map((airport) => (
              <Badge
                key={airport.code}
                variant={destination?.code === airport.code ? "default" : "outline"}
                className="cursor-pointer hover:bg-primary/20 transition-colors"
                onClick={() => handleQuickSelect(airport, "destination")}
              >
                {airport.code}
              </Badge>
            ))}
          </div>
        </div>
      </div>

      {/* Selected Route Display */}
      {origin && destination && (
        <div className="mb-6 p-4 rounded-lg bg-primary/5 border border-primary/20">
          <div className="flex items-center justify-center gap-3 text-sm">
            <span className="font-semibold">{origin.code}</span>
            <ArrowRight className="w-4 h-4 text-primary" />
            <span className="font-semibold">{destination.code}</span>
          </div>
          <p className="text-center text-xs text-muted-foreground mt-2">
            {origin.city}, {origin.country} → {destination.city}, {destination.country}
          </p>
        </div>
      )}

      <Button
        variant="hero"
        className="w-full"
        disabled={!origin || !destination}
        onClick={handleGenerateRoute}
      >
        <Music className="w-5 h-5" />
        Generate Musical Route
      </Button>
    </Card>
  );
};

export default RouteSelector;
