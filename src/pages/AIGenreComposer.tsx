import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Sparkles, Music2, Blend, Info, Play, Pause, Download } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { audioPlayer } from '@/lib/audioPlayer';
import { exportToMidi } from '@/lib/midiExport';

interface Genre {
  name: string;
  tempo_range: [number, number];
  scales: string[];
  complexity: number;
  dynamics: string;
}

export default function AIGenreComposer() {
  const [genres, setGenres] = useState<Record<string, Genre>>({});
  const [selectedGenre, setSelectedGenre] = useState('');
  const [primaryGenre, setPrimaryGenre] = useState('');
  const [secondaryGenre, setSecondaryGenre] = useState('');
  const [blendRatio, setBlendRatio] = useState([0.5]);
  const [distance, setDistance] = useState([5000]);
  const [latRange, setLatRange] = useState([30]);
  const [lonRange, setLonRange] = useState([50]);
  const [direction, setDirection] = useState('E');
  const [composition, setComposition] = useState<any>(null);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    fetchGenres();
  }, []);

  const fetchGenres = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/available`);
      const data = await response.json();
      if (data.success) {
        setGenres(data.data.genres);
        const genreNames = Object.keys(data.data.genres);
        if (genreNames.length > 0) {
          setSelectedGenre(genreNames[0]);
          setPrimaryGenre(genreNames[0]);
          setSecondaryGenre(genreNames[1] || genreNames[0]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch genres:', error);
    }
  };

  const generateComposition = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/compose`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          genre: selectedGenre,
          route_features: {
            distance: distance[0],
            latitude_range: latRange[0],
            longitude_range: lonRange[0],
            direction: direction
          },
          duration: 30
        })
      });

      const data = await response.json();
      if (data.success) {
        setComposition(data.data);
        toast({
          title: 'Composition Generated',
          description: `AI-generated ${selectedGenre} composition created successfully!`
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to generate composition.',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  };

  const getRecommendations = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          distance: distance[0],
          latitude_range: latRange[0],
          longitude_range: lonRange[0],
          direction: direction
        })
      });

      const data = await response.json();
      if (data.success) {
        setRecommendations(data.data.recommendations);
        toast({
          title: 'Recommendations Ready',
          description: `Top recommendation: ${data.data.top_genre}`
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to get recommendations.',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  };

  const blendGenres = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/blend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          primary_genre: primaryGenre,
          secondary_genre: secondaryGenre,
          blend_ratio: blendRatio[0],
          route_features: {
            distance: distance[0],
            latitude_range: latRange[0],
            longitude_range: lonRange[0],
            direction: direction
          }
        })
      });

      const data = await response.json();
      if (data.success) {
        setComposition(data.data);
        toast({
          title: 'Blend Created',
          description: `${primaryGenre} + ${secondaryGenre} blend generated!`
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to blend genres.',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  };

  const handlePlayMusic = async () => {
    if (!composition?.note_sequence || composition.note_sequence.length === 0) {
      toast({
        title: 'No Music Available',
        description: 'Please generate a composition first',
        variant: 'destructive'
      });
      return;
    }

    try {
      setIsPlaying(true);
      
      // Convert backend note format (pitch) to frontend format (note)
      const notes = composition.note_sequence.map((n: any) => ({
        note: n.pitch || n.note || 60,
        velocity: n.velocity || 80,
        duration: n.duration || 0.25,
        time: n.time || 0
      }));
      
      await audioPlayer.playComposition(
        notes,
        composition.tempo || 120,
        (progress) => {
          // Progress callback
        }
      );
      setIsPlaying(false);
      toast({
        title: 'Playback Complete',
        description: 'Music finished playing'
      });
    } catch (error) {
      console.error('Playback error:', error);
      setIsPlaying(false);
      toast({
        title: 'Playback Failed',
        description: error instanceof Error ? error.message : 'Could not play music',
        variant: 'destructive'
      });
    }
  };

  const handleStopMusic = () => {
    audioPlayer.stop();
    setIsPlaying(false);
    toast({
      title: 'Playback Stopped',
      description: 'Music playback stopped'
    });
  };

  const handleDownloadMidi = () => {
    if (!composition) {
      toast({
        title: 'No Composition',
        description: 'Please generate a composition first',
        variant: 'destructive'
      });
      return;
    }

    try {
      // Prepare composition data for MIDI export
      const midiData = {
        ...composition,
        midi_data: {
          notes: composition.note_sequence?.map((n: any) => ({
            midi: n.pitch || n.note || 60,
            velocity: (n.velocity || 80) / 127,
            time: n.time || 0,
            duration: n.duration || 0.25
          })) || []
        }
      };
      
      const filename = composition.blended 
        ? `${composition.primary_genre}_${composition.secondary_genre}_blend`
        : composition.genre;
      exportToMidi(midiData, filename);
      toast({
        title: 'Download Started',
        description: 'Your MIDI file is being downloaded.'
      });
    } catch (error) {
      console.error('MIDI export error:', error);
      toast({
        title: 'Download Failed',
        description: 'Failed to export MIDI file.',
        variant: 'destructive'
      });
    }
  };

  return (
    <div className="min-h-screen pt-20">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-2">
            <Sparkles className="h-8 w-8" />
            AI Genre Composer
          </h1>
          <p className="text-muted-foreground text-lg">
            Advanced PyTorch-powered music generation with genre-specific AI models
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Route Parameters</CardTitle>
              <CardDescription>
                Adjust route characteristics to influence the AI composition
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <label className="text-sm font-medium mb-2 block text-white">
                  Distance: {distance[0]} km
                </label>
                <Slider
                  value={distance}
                  onValueChange={setDistance}
                  min={500}
                  max={15000}
                  step={100}
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">
                  Latitude Range: {latRange[0]}°
                </label>
                <Slider
                  value={latRange}
                  onValueChange={setLatRange}
                  min={0}
                  max={180}
                  step={5}
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">
                  Longitude Range: {lonRange[0]}°
                </label>
                <Slider
                  value={lonRange}
                  onValueChange={setLonRange}
                  min={0}
                  max={360}
                  step={5}
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">Direction</label>
                <Select value={direction} onValueChange={setDirection}>
                  <SelectTrigger className="text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="N">North</SelectItem>
                    <SelectItem value="S">South</SelectItem>
                    <SelectItem value="E">East</SelectItem>
                    <SelectItem value="W">West</SelectItem>
                    <SelectItem value="NE">Northeast</SelectItem>
                    <SelectItem value="NW">Northwest</SelectItem>
                    <SelectItem value="SE">Southeast</SelectItem>
                    <SelectItem value="SW">Southwest</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button onClick={getRecommendations} className="w-full" variant="outline">
                <Info className="mr-2 h-4 w-4" />
                Get AI Recommendations
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Single Genre Composition</CardTitle>
              <CardDescription>Generate music in a specific genre</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block text-white">Select Genre</label>
                <Select value={selectedGenre} onValueChange={setSelectedGenre}>
                  <SelectTrigger className="text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.keys(genres).map((genre) => (
                      <SelectItem key={genre} value={genre}>
                        {genre.charAt(0).toUpperCase() + genre.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {selectedGenre && genres[selectedGenre] && (
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm mb-2 text-white">
                    <strong className="text-white">Tempo Range:</strong> {genres[selectedGenre].tempo_range[0]}-
                    {genres[selectedGenre].tempo_range[1]} BPM
                  </p>
                  <p className="text-sm mb-2 text-white">
                    <strong className="text-white">Complexity:</strong> {(genres[selectedGenre].complexity * 100).toFixed(0)}%
                  </p>
                  <p className="text-sm mb-2 text-white">
                    <strong className="text-white">Dynamics:</strong> {genres[selectedGenre].dynamics}
                  </p>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {genres[selectedGenre].scales.map((scale) => (
                      <Badge key={scale} variant="secondary">
                        {scale}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              <Button onClick={generateComposition} className="w-full" disabled={loading}>
                <Music2 className="mr-2 h-4 w-4" />
                {loading ? 'Generating...' : 'Generate Composition'}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Genre Blending</CardTitle>
              <CardDescription>Mix two genres to create unique sounds</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block text-white">Primary Genre</label>
                  <Select value={primaryGenre} onValueChange={setPrimaryGenre}>
                    <SelectTrigger className="text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.keys(genres).map((genre) => (
                        <SelectItem key={genre} value={genre}>
                          {genre.charAt(0).toUpperCase() + genre.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block text-white">Secondary Genre</label>
                  <Select value={secondaryGenre} onValueChange={setSecondaryGenre}>
                    <SelectTrigger className="text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.keys(genres).map((genre) => (
                        <SelectItem key={genre} value={genre}>
                          {genre.charAt(0).toUpperCase() + genre.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">
                  Blend Ratio: {(blendRatio[0] * 100).toFixed(0)}% {secondaryGenre}
                </label>
                <Slider
                  value={blendRatio}
                  onValueChange={setBlendRatio}
                  min={0}
                  max={1}
                  step={0.1}
                />
              </div>

              <Button onClick={blendGenres} className="w-full" variant="secondary" disabled={loading}>
                <Blend className="mr-2 h-4 w-4" />
                {loading ? 'Blending...' : 'Blend Genres'}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {recommendations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>AI Recommendations</CardTitle>
                <CardDescription>Best genres for your route</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {recommendations.slice(0, 5).map((rec, idx) => (
                    <div key={idx} className="flex justify-between items-center p-2 bg-muted rounded">
                      <span className="font-medium capitalize text-white">{rec.genre}</span>
                      <Badge>{(rec.confidence * 100).toFixed(1)}%</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {composition && (
            <Card>
              <CardHeader>
                <CardTitle>Generated Composition</CardTitle>
                <CardDescription>AI-powered music generation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="p-3 bg-muted rounded">
                  <p className="text-sm text-white"><strong className="text-white">Genre:</strong> {composition.genre}</p>
                  <p className="text-sm text-white"><strong className="text-white">Tempo:</strong> {composition.tempo} BPM</p>
                  <p className="text-sm text-white"><strong className="text-white">Scale:</strong> {composition.scale}</p>
                  <p className="text-sm text-white"><strong className="text-white">Key:</strong> {composition.key}</p>
                  <p className="text-sm text-white"><strong className="text-white">Duration:</strong> {composition.duration}s</p>
                  <p className="text-sm text-white">
                    <strong className="text-white">Complexity:</strong> {composition.complexity ? (composition.complexity * 100).toFixed(0) : '0'}%
                  </p>
                  <p className="text-sm text-white"><strong className="text-white">Dynamics:</strong> {composition.dynamics}</p>
                </div>

                {composition.blended && (
                  <Badge variant="outline" className="w-full justify-center">
                    Blended: {composition.primary_genre} + {composition.secondary_genre}
                  </Badge>
                )}

                <Badge variant="secondary" className="w-full justify-center">
                  <Sparkles className="mr-2 h-3 w-3" />
                  AI Generated with PyTorch
                </Badge>

                <p className="text-xs text-muted-foreground">
                  Notes: {composition.note_sequence?.length || 0} generated
                </p>

                <div className="flex gap-2 mt-4">
                  <Button 
                    onClick={isPlaying ? handleStopMusic : handlePlayMusic}
                    className="flex-1"
                    variant={isPlaying ? "destructive" : "default"}
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="mr-2 h-4 w-4" />
                        Stop
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Play
                      </>
                    )}
                  </Button>
                  <Button 
                    onClick={handleDownloadMidi}
                    variant="outline"
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>About AI Composer</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <p className="text-white">
                <strong className="text-white">Neural Networks:</strong> Uses PyTorch LSTM models for pattern generation
              </p>
              <p className="text-white">
                <strong className="text-white">Genre Embeddings:</strong> Learns genre-specific characteristics
              </p>
              <p className="text-white">
                <strong className="text-white">8 Genres:</strong> Classical, Jazz, Electronic, Ambient, Rock, World, Cinematic, Lofi
              </p>
              <p className="text-white">
                <strong className="text-white">Features:</strong> Genre blending, AI recommendations, dynamic composition
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
      </div>
    </div>
  );
}

