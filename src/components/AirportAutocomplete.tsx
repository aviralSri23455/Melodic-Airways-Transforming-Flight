import { useState, useEffect, useRef } from 'react';
import { Input } from '@/components/ui/input';
import { Command, CommandEmpty, CommandGroup, CommandItem, CommandList } from '@/components/ui/command';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Plane } from 'lucide-react';

interface Airport {
  code: string;
  name: string;
  city: string;
  country: string;
  label: string;
}

interface AirportAutocompleteProps {
  value: string;
  onChange: (code: string) => void;
  placeholder?: string;
  className?: string;
}

export default function AirportAutocomplete({ 
  value, 
  onChange, 
  placeholder = "Airport Code (e.g., JFK)",
  className = ""
}: AirportAutocompleteProps) {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState(value);
  const [airports, setAirports] = useState<Airport[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceTimer = useRef<NodeJS.Timeout>();

  useEffect(() => {
    setSearchQuery(value);
  }, [value]);

  const searchAirports = async (query: string) => {
    if (query.length < 2) {
      setAirports([]);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/user/airports/search?q=${encodeURIComponent(query)}&limit=10`
      );
      const data = await response.json();
      if (data.success) {
        setAirports(data.data);
      }
    } catch (error) {
      console.error('Failed to search airports:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setSearchQuery(query);
    onChange(query);

    // Debounce search
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(() => {
      searchAirports(query);
      setOpen(query.length >= 2);
    }, 300);
  };

  const handleSelect = (airport: Airport) => {
    setSearchQuery(airport.code);
    onChange(airport.code);
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <div className="relative">
          <Plane className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={handleInputChange}
            placeholder={placeholder}
            className={`pl-10 ${className}`}
            onFocus={() => {
              if (searchQuery.length >= 2) {
                setOpen(true);
              }
            }}
          />
        </div>
      </PopoverTrigger>
      <PopoverContent className="w-[400px] p-0" align="start">
        <Command>
          <CommandList>
            {loading && (
              <div className="p-4 text-sm text-muted-foreground text-center">
                Searching airports...
              </div>
            )}
            {!loading && airports.length === 0 && searchQuery.length >= 2 && (
              <CommandEmpty>No airports found.</CommandEmpty>
            )}
            {!loading && airports.length > 0 && (
              <CommandGroup>
                {airports.map((airport) => (
                  <CommandItem
                    key={airport.code}
                    value={airport.code}
                    onSelect={() => handleSelect(airport)}
                    className="cursor-pointer"
                  >
                    <div className="flex items-center gap-2">
                      <Plane className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="font-semibold">{airport.code}</div>
                        <div className="text-xs text-muted-foreground">
                          {airport.name} - {airport.city}, {airport.country}
                        </div>
                      </div>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
