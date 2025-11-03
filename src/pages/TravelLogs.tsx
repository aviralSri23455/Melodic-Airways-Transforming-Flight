import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Plus, MapPin, Music, Share2, Calendar, Trash2, Play, Pause, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import AirportAutocomplete from '@/components/AirportAutocomplete';
import { audioPlayer } from '@/lib/audioPlayer';

interface Waypoint {
  airport_code: string;
  timestamp?: string;
  notes?: string;
}

interface TravelLog {
  id: number;
  title: string;
  description: string;
  waypoints: Waypoint[];
  travel_date: string;
  tags: string[];
  is_public: boolean;
  created_at: string;
}

export default function TravelLogs() {
  const [travelLogs, setTravelLogs] = useState<TravelLog[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newLog, setNewLog] = useState({
    title: '',
    description: '',
    waypoints: [{ airport_code: '', timestamp: '', notes: '' }],
    tags: '',
    travel_date: ''
  });
  const [generatingMusic, setGeneratingMusic] = useState<number | null>(null);
  const [playingMusic, setPlayingMusic] = useState<number | null>(null);
  const [compositions, setCompositions] = useState<Record<number, any>>({});
  const { toast } = useToast();

  useEffect(() => {
    fetchTravelLogs();
  }, []);

  const fetchTravelLogs = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/user/travel-logs/my`);
      const data = await response.json();
      if (data.success) {
        setTravelLogs(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch travel logs:', error);
    }
  };

  const addWaypoint = () => {
    setNewLog({
      ...newLog,
      waypoints: [...newLog.waypoints, { airport_code: '', timestamp: '', notes: '' }]
    });
  };

  const updateWaypoint = (index: number, field: string, value: string) => {
    const updatedWaypoints = [...newLog.waypoints];
    updatedWaypoints[index] = { ...updatedWaypoints[index], [field]: value };
    setNewLog({ ...newLog, waypoints: updatedWaypoints });
  };

  const createTravelLog = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/user/travel-logs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          title: newLog.title,
          description: newLog.description,
          waypoints: newLog.waypoints.filter(w => w.airport_code),
          tags: newLog.tags.split(',').map(t => t.trim()).filter(t => t),
          travel_date: newLog.travel_date || new Date().toISOString()
        })
      });

      const data = await response.json();
      if (data.success) {
        toast({
          title: 'Travel Log Created',
          description: 'Your travel log has been saved successfully.'
        });
        setShowCreateForm(false);
        fetchTravelLogs();
        setNewLog({
          title: '',
          description: '',
          waypoints: [{ airport_code: '', timestamp: '', notes: '' }],
          tags: '',
          travel_date: ''
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to create travel log.',
        variant: 'destructive'
      });
    }
  };

  const deleteTravelLog = async (logId: number) => {
    if (!confirm('Are you sure you want to delete this travel log?')) {
      return;
    }

    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/user/travel-logs/${logId}`,
        {
          method: 'DELETE'
        }
      );

      const data = await response.json();
      if (data.success) {
        toast({
          title: 'Travel Log Deleted',
          description: 'Your travel log has been deleted successfully.'
        });
        fetchTravelLogs();
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to delete travel log.',
        variant: 'destructive'
      });
    }
  };

  const convertToMusic = async (log: TravelLog) => {
    if (compositions[log.id]) {
      // Already have composition, just play it
      playMusic(log.id);
      return;
    }

    setGeneratingMusic(log.id);
    try {
      // Calculate route features from waypoints
      const waypoints = log.waypoints;
      if (waypoints.length < 2) {
        toast({
          title: 'Error',
          description: 'Need at least 2 waypoints to generate music.',
          variant: 'destructive'
        });
        return;
      }

      // Use first and last waypoint for route calculation
      const origin = waypoints[0].airport_code;
      const destination = waypoints[waypoints.length - 1].airport_code;
      
      // Create a unique seed based on the route
      const routeSeed = `${origin}-${destination}-${log.id}`.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
      
      // Get AI genre recommendations
      const recResponse = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          distance: 5000 + (waypoints.length * 1000), // Vary by waypoint count
          latitude_range: 30 + (waypoints.length * 5),
          longitude_range: 50 + (waypoints.length * 10),
          direction: ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'][routeSeed % 8]
        })
      });
      
      const recData = await recResponse.json();
      
      // Select genre with time variation for uniqueness
      const timeVariation = Date.now() % 1000;
      const variedSeed = routeSeed + timeVariation;
      const topRecommendations = recData.data?.recommendations?.slice(0, 4) || [];
      const genreIndex = variedSeed % Math.max(topRecommendations.length, 1);
      const selectedGenre = topRecommendations[genreIndex]?.genre || 'ambient';
      
      console.log(`Generating ${selectedGenre} music for travel log ${log.id}`);

      // Generate AI composition with unique seed
      const compResponse = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/compose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          genre: selectedGenre,
          route_features: {
            distance: 5000 + (waypoints.length * 1000),
            latitude_range: 30 + (waypoints.length * 5),
            longitude_range: 50 + (waypoints.length * 10),
            direction: ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'][routeSeed % 8],
            seed: routeSeed + Date.now() // Add timestamp for uniqueness
          },
          duration: Math.max(30, waypoints.length * 10) // Longer music for more waypoints
        })
      });

      const compData = await compResponse.json();
      if (compData.success) {
        setCompositions(prev => ({ ...prev, [log.id]: compData.data }));
        toast({
          title: `🎵 ${selectedGenre.toUpperCase()} Music Generated`,
          description: `Unique composition for ${origin} → ${destination}`
        });
        
        // Auto-play the music
        playMusic(log.id, compData.data);
      }
    } catch (error) {
      console.error('Failed to generate music:', error);
      toast({
        title: 'Error',
        description: 'Failed to generate music.',
        variant: 'destructive'
      });
    } finally {
      setGeneratingMusic(null);
    }
  };

  const playMusic = async (logId: number, composition?: any) => {
    const comp = composition || compositions[logId];
    if (!comp?.note_sequence || comp.note_sequence.length === 0) {
      toast({
        title: 'No Music Available',
        description: 'Please generate music first',
        variant: 'destructive'
      });
      return;
    }

    try {
      setPlayingMusic(logId);
      
      const notes = comp.note_sequence.map((n: any) => ({
        note: n.pitch || n.note || 60,
        velocity: n.velocity || 80,
        duration: n.duration || 0.25,
        time: n.time || 0
      }));
      
      await audioPlayer.playComposition(notes, comp.tempo || 120);
      setPlayingMusic(null);
      
      toast({
        title: 'Playback Complete',
        description: `${comp.genre} music finished playing`
      });
    } catch (error) {
      console.error('Playback error:', error);
      setPlayingMusic(null);
      toast({
        title: 'Playback Failed',
        description: error instanceof Error ? error.message : 'Could not play music',
        variant: 'destructive'
      });
    }
  };

  const stopMusic = () => {
    audioPlayer.stop();
    setPlayingMusic(null);
    toast({
      title: 'Playback Stopped',
      description: 'Music playback stopped'
    });
  };

  return (
    <div className="min-h-screen pt-20">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-4">My Travel Logs</h1>
            <p className="text-muted-foreground text-lg">
              Document your journeys and convert them into musical compositions
            </p>
          </div>
          <Button onClick={() => setShowCreateForm(!showCreateForm)}>
            <Plus className="mr-2 h-4 w-4" />
            New Travel Log
        </Button>
      </div>

      {showCreateForm && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Create New Travel Log</CardTitle>
            <CardDescription>
              Add your travel waypoints to create a musical journey
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="title" className="text-white">Title</Label>
              <Input
                id="title"
                value={newLog.title}
                onChange={(e) => setNewLog({ ...newLog, title: e.target.value })}
                placeholder="My Amazing Trip"
                className="text-white"
              />
            </div>

            <div>
              <Label htmlFor="description" className="text-white">Description</Label>
              <Textarea
                id="description"
                value={newLog.description}
                onChange={(e) => setNewLog({ ...newLog, description: e.target.value })}
                placeholder="Describe your journey..."
                className="text-white"
              />
            </div>

            <div>
              <Label htmlFor="travel_date" className="text-white">Travel Date</Label>
              <Input
                id="travel_date"
                type="date"
                value={newLog.travel_date}
                onChange={(e) => setNewLog({ ...newLog, travel_date: e.target.value })}
                className="text-white"
              />
            </div>

            <div>
              <Label className="text-white">Waypoints</Label>
              <p className="text-sm text-white mb-2">
                Type airport code, name, or city to search
              </p>
              {newLog.waypoints.map((waypoint, index) => (
                <div key={index} className="flex gap-2 mb-2">
                  <AirportAutocomplete
                    value={waypoint.airport_code}
                    onChange={(code) => updateWaypoint(index, 'airport_code', code)}
                    placeholder="Search airport..."
                    className="w-48"
                  />
                  <Input
                    type="datetime-local"
                    value={waypoint.timestamp}
                    onChange={(e) => updateWaypoint(index, 'timestamp', e.target.value)}
                    className="flex-1 text-white"
                  />
                  <Input
                    placeholder="Notes"
                    value={waypoint.notes}
                    onChange={(e) => updateWaypoint(index, 'notes', e.target.value)}
                    className="flex-1 text-white"
                  />
                </div>
              ))}
              <Button variant="outline" size="sm" onClick={addWaypoint}>
                <Plus className="mr-2 h-4 w-4" />
                Add Waypoint
              </Button>
            </div>

            <div>
              <Label htmlFor="tags" className="text-white">Tags (comma-separated)</Label>
              <Input
                id="tags"
                value={newLog.tags}
                onChange={(e) => setNewLog({ ...newLog, tags: e.target.value })}
                placeholder="vacation, business, adventure"
                className="text-white"
              />
            </div>

            <div className="flex gap-2">
              <Button onClick={createTravelLog}>Create Travel Log</Button>
              <Button variant="outline" onClick={() => setShowCreateForm(false)}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {travelLogs.map((log) => (
          <Card key={log.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MapPin className="h-5 w-5" />
                {log.title}
              </CardTitle>
              <CardDescription>
                <Calendar className="inline h-4 w-4 mr-1" />
                {new Date(log.travel_date).toLocaleDateString()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-white mb-4">{log.description}</p>
              
              <div className="mb-4">
                <p className="text-sm font-semibold mb-2 text-white">Route:</p>
                <div className="flex flex-wrap gap-1">
                  {log.waypoints.map((wp, idx) => (
                    <span key={idx}>
                      <Badge variant="secondary">{wp.airport_code}</Badge>
                      {idx < log.waypoints.length - 1 && <span className="mx-1 text-white">→</span>}
                    </span>
                  ))}
                </div>
              </div>

              {log.tags.length > 0 && (
                <div className="mb-4">
                  <div className="flex flex-wrap gap-1">
                    {log.tags.map((tag, idx) => (
                      <Badge key={idx} variant="outline">{tag}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {compositions[log.id] && (
                <div className="mb-4 p-3 bg-muted rounded">
                  <div className="text-sm space-y-1">
                    <div className="flex items-center gap-2">
                      <Music className="h-4 w-4" />
                      <span className="font-semibold text-white capitalize">{compositions[log.id].genre}</span>
                      <Badge variant="secondary">{compositions[log.id].tempo} BPM</Badge>
                    </div>
                    <div className="text-xs text-white">
                      {compositions[log.id].note_sequence?.length || 0} notes • {compositions[log.id].duration}s
                    </div>
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                {playingMusic === log.id ? (
                  <Button size="sm" onClick={stopMusic} variant="destructive">
                    <Pause className="mr-2 h-4 w-4" />
                    Stop Music
                  </Button>
                ) : generatingMusic === log.id ? (
                  <Button size="sm" disabled>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </Button>
                ) : compositions[log.id] ? (
                  <Button size="sm" onClick={() => playMusic(log.id)}>
                    <Play className="mr-2 h-4 w-4" />
                    Play Music
                  </Button>
                ) : (
                  <Button size="sm" onClick={() => convertToMusic(log)}>
                    <Music className="mr-2 h-4 w-4" />
                    Generate Music
                  </Button>
                )}
                <Button size="sm" variant="outline">
                  <Share2 className="h-4 w-4" />
                </Button>
                <Button 
                  size="sm" 
                  variant="destructive" 
                  onClick={() => deleteTravelLog(log.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {travelLogs.length === 0 && !showCreateForm && (
        <div className="text-center py-12">
          <MapPin className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-xl font-semibold mb-2 text-white">No Travel Logs Yet</h3>
          <p className="text-white mb-4">
            Start documenting your journeys and turn them into music
          </p>
          <Button onClick={() => setShowCreateForm(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Your First Travel Log
          </Button>
        </div>
      )}
      </div>
    </div>
  );
}

