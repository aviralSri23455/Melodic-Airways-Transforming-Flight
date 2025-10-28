import { useState, useEffect } from "react";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import RouteSelector from "@/components/RouteSelector";
import MusicControls, { MusicSettings } from "@/components/MusicControls";
import GlobalMap from "@/components/GlobalMap";
import MusicPlayer from "@/components/MusicPlayer";
import MusicDNA from "@/components/MusicDNA";
import Analytics from "@/components/Analytics";
import { useToast } from "@/hooks/use-toast";
import { audioPlayer } from "@/lib/audioPlayer";
import { generateMusic } from "@/lib/api/music";

interface Airport {
  code: string;
  name: string;
  city: string;
  country: string;
}

// Sample airport coordinates (in real implementation, fetch from database)
const airportCoords: Record<string, { lat: number; lng: number }> = {
  JFK: { lat: 40.6413, lng: -73.7781 },
  CDG: { lat: 49.0097, lng: 2.5479 },
  LHR: { lat: 51.4700, lng: -0.4543 },
  NRT: { lat: 35.7720, lng: 140.3929 },
  DXB: { lat: 25.2532, lng: 55.3657 },
  SYD: { lat: -33.9399, lng: 151.1753 },
};

const Index = () => {
  const { toast } = useToast();
  const [selectedOrigin, setSelectedOrigin] = useState<(Airport & { lat: number; lng: number }) | undefined>();
  const [selectedDestination, setSelectedDestination] = useState<(Airport & { lat: number; lng: number }) | undefined>();
  const [musicSettings, setMusicSettings] = useState<MusicSettings>({
    tempo: 120,
    key: "C",
    scale: "major",
    complexity: 50,
    harmonization: "triads",
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentComposition, setCurrentComposition] = useState<any>(null);
  const [analytics, setAnalytics] = useState<any>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      audioPlayer.stop();
    };
  }, []);

  const handleRouteSelect = async (origin: Airport, destination: Airport) => {
    const originCoords = airportCoords[origin.code];
    const destCoords = airportCoords[destination.code];

    if (!originCoords || !destCoords) {
      toast({
        title: "Airport Not Found",
        description: "Please use one of the suggested airports for this demo.",
        variant: "destructive",
      });
      return;
    }

    setSelectedOrigin({ ...origin, ...originCoords });
    setSelectedDestination({ ...destination, ...destCoords });
    setIsGenerating(true);

    toast({
      title: "Generating Composition",
      description: `Creating musical route from ${origin.code} to ${destination.code}...`,
    });

    try {
      // Call backend API to generate music
      const response = await generateMusic({
        origin: origin.code,
        destination: destination.code,
        music_style: musicSettings.scale || 'major',
        tempo: musicSettings.tempo,
      });

      // Debug: Log the actual response
      console.log('Backend response:', response);

      if (response.error) {
        console.error('Backend error:', response.error);
        throw new Error(response.error.message);
      }

      if (response.data) {
        // Transform backend response to match existing composition format
        const composition = {
          id: response.data.compositionId,
          midi_data: {
            notes: generateNotesFromBackendData(response.data),
          },
          // Add music DNA fields for visualization
          scale: response.data.music.scale || musicSettings.scale,
          tempo: response.data.music.tempo || musicSettings.tempo,
          note_count: response.data.music.noteCount,
          root_note: response.data.music.root_note || 60,
          tracks: response.data.music.tracks || { melody: 0, harmony: 0, bass: 0 },
          update_id: response.data.music.update_id || `${origin.code}_${destination.code}_${Date.now()}`,
        };

        setCurrentComposition(composition);
        setAnalytics({
          uniqueNotes: response.data.music.noteCount,
          duration: response.data.music.duration,
          complexity: response.data.analytics.complexity,
          harmonicRichness: response.data.analytics.harmonic_richness,
        });
        setIsGenerating(false);
        
        toast({
          title: "Composition Ready!",
          description: `Your musical flight route has been generated with ${response.data.music.noteCount} notes.`,
        });
      }
    } catch (error) {
      console.error('Error generating music:', error);
      toast({
        title: "Generation Failed",
        description: error instanceof Error ? error.message : "Failed to generate musical composition. Please try again.",
        variant: "destructive",
      });
      setIsGenerating(false);
    }
  };

  // Helper function to use actual backend notes
  const generateNotesFromBackendData = (data: any) => {
    // Check if backend provided actual notes
    if (data.music.notes && Array.isArray(data.music.notes) && data.music.notes.length > 0) {
      console.log(`Using ${data.music.notes.length} notes from backend`);
      return data.music.notes;
    }
    
    // Fallback: If backend didn't provide notes, generate client-side
    console.warn('Backend did not provide notes, generating client-side fallback');
    const notes = [];
    const noteCount = data.music.noteCount;
    const duration = data.music.duration;
    const tempo = data.music.tempo;
    const style = data.music.style;
    
    // Calculate musical parameters based on route
    const distance = data.route.distance;
    const originLat = data.route.origin.city; // Use city for variation
    
    // Base note varies by origin city (creates different "moods" for different routes)
    const baseNoteMap: Record<string, number> = {
      'New York': 60, // C4
      'London': 62,   // D4
      'Tokyo': 64,    // E4
      'Paris': 65,    // F4
      'Los Angeles': 67, // G4
      'Chicago': 69,  // A4
      'San Francisco': 71, // B4
    };
    
    const baseNote = baseNoteMap[originLat] || 60;
    
    // Generate notes with more musical variation
    for (let i = 0; i < noteCount; i++) {
      const progress = i / noteCount;
      const timeOffset = (i * duration) / noteCount;
      
      // Create melody that rises and falls based on route progress
      let noteOffset = 0;
      if (style === 'major') {
        // Major scale: happier, ascending for eastward travel
        noteOffset = Math.floor(Math.sin(progress * Math.PI * 2) * 7) + [0, 2, 4, 7, 9][i % 5];
      } else if (style === 'minor') {
        // Minor scale: more melancholic
        noteOffset = Math.floor(Math.cos(progress * Math.PI * 2) * 5) + [0, 2, 3, 7, 8][i % 5];
      } else {
        // Ambient: more atmospheric
        noteOffset = Math.floor(Math.sin(progress * Math.PI * 4) * 12);
      }
      
      notes.push({
        note: baseNote + noteOffset,
        velocity: 60 + Math.floor(Math.random() * 40), // Velocity variation
        time: timeOffset,
        duration: 0.3 + Math.random() * 0.4, // Duration variation
      });
    }
    
    // Add harmony notes for longer routes
    if (distance > 5000) {
      for (let i = 0; i < Math.floor(noteCount / 3); i++) {
        const harmonyNote = notes[i * 3];
        if (harmonyNote) {
          notes.push({
            note: harmonyNote.note - 12, // Lower octave harmony
            velocity: 40,
            time: harmonyNote.time,
            duration: harmonyNote.duration * 1.5,
          });
        }
      }
    }
    
    // Sort by time to ensure proper playback order
    notes.sort((a, b) => a.time - b.time);
    
    console.log(`Generated ${notes.length} notes for ${originLat} route`);
    return notes;
  };

  const handleSettingsChange = (settings: MusicSettings) => {
    setMusicSettings(settings);
  };

  const handlePlay = async () => {
    if (!currentComposition?.midi_data?.notes) {
      toast({
        title: "No Composition",
        description: "Please generate a composition first.",
        variant: "destructive",
      });
      return;
    }

    console.log('Play button clicked, isPlaying:', isPlaying);
    console.log('Composition notes:', currentComposition.midi_data.notes.length);

    if (isPlaying) {
      console.log('Stopping playback...');
      audioPlayer.stop();
      setIsPlaying(false);
    } else {
      // Try to initialize audio context on user interaction
      try {
        setIsPlaying(true);
        console.log('Starting playback with tempo:', musicSettings.tempo);
        
        await audioPlayer.playComposition(
          currentComposition.midi_data.notes,
          musicSettings.tempo,
          (progress) => {
            console.log('Playback progress:', progress.toFixed(1) + '%');
          }
        );
        console.log('Playback completed');
        setIsPlaying(false);
        
        toast({
          title: "Playback Complete",
          description: `Played ${currentComposition.midi_data.notes.length} notes`,
        });
        
      } catch (error) {
        console.error('Playback error:', error);
        setIsPlaying(false);
        
        toast({
          title: "Playback Error",
          description: error instanceof Error ? error.message : "Could not play audio. Please try clicking the play button again.",
          variant: "destructive",
        });
      }
    }
  };

  return (
    <div className="min-h-screen">
      <Hero />
      <Features />
      
      {/* Main Application Section */}
      <section id="app-section" className="py-24 px-4 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-background via-secondary/20 to-background" />
        
        <div className="container mx-auto max-w-7xl relative z-10">
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Create Your Musical Journey
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Select your route, customize parameters, and generate unique compositions
            </p>
          </div>

          {/* Interactive Map */}
          <div className="mb-8">
            <GlobalMap origin={selectedOrigin} destination={selectedDestination} />
          </div>

          {/* Control Panels Grid */}
          <div className="grid lg:grid-cols-2 gap-8 mb-8">
            <RouteSelector onRouteSelect={handleRouteSelect} />
            <MusicControls onSettingsChange={handleSettingsChange} />
          </div>

          {/* Player and Analytics */}
          <div className="grid lg:grid-cols-2 gap-8 mb-8">
            <MusicPlayer
              routeName={
                selectedOrigin && selectedDestination
                  ? `${selectedOrigin.code} → ${selectedDestination.code}`
                  : undefined
              }
              isGenerating={isGenerating}
              isPlaying={isPlaying}
              onPlay={handlePlay}
              canPlay={!!currentComposition?.midi_data?.notes?.length}
              duration={analytics?.duration ?? 0}
              composition={currentComposition}
            />
            <Analytics metrics={analytics} />
          </div>

          {/* Music DNA - Show Uniqueness */}
          <div className="max-w-2xl mx-auto">
            <MusicDNA composition={currentComposition} />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t border-border">
        <div className="container mx-auto text-center">
          <p className="text-muted-foreground">
            FlightSymphony - Transforming aviation data into musical experiences
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            Built with React, MariaDB, PyTorch, and passion for data-driven art
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
