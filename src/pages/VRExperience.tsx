import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Glasses, Plane, Camera, Download, Play, Pause, Music2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Sphere, Stars, Trail } from '@react-three/drei';
import { audioPlayer } from '@/lib/audioPlayer';
import * as THREE from 'three';

interface FlightPathPoint {
  position: { x: number; y: number; z: number };
  geographic: { latitude: number; longitude: number; altitude: number };
  progress: number;
}

export default function VRExperience() {
  const [origin, setOrigin] = useState('JFK');
  const [destination, setDestination] = useState('LAX');
  const [experienceType, setExperienceType] = useState('immersive');
  const [cameraMode, setCameraMode] = useState('follow');
  const [experience, setExperience] = useState<any>(null);
  const [flightPath, setFlightPath] = useState<FlightPathPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [currentPoint, setCurrentPoint] = useState(0);
  const [musicPlaying, setMusicPlaying] = useState(false);
  const [composition, setComposition] = useState<any>(null);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [visualStyle, setVisualStyle] = useState<'default' | 'neon' | 'aurora' | 'cosmic'>('default');
  const [selectedGenre, setSelectedGenre] = useState<string>('auto');
  const [availableGenres, setAvailableGenres] = useState<string[]>([]);
  const { toast } = useToast();

  // Popular airports for easy selection
  const popularAirports = [
    { code: 'JFK', name: 'New York (JFK)', city: 'New York' },
    { code: 'LAX', name: 'Los Angeles (LAX)', city: 'Los Angeles' },
    { code: 'LHR', name: 'London Heathrow (LHR)', city: 'London' },
    { code: 'CDG', name: 'Paris Charles de Gaulle (CDG)', city: 'Paris' },
    { code: 'NRT', name: 'Tokyo Narita (NRT)', city: 'Tokyo' },
    { code: 'DXB', name: 'Dubai (DXB)', city: 'Dubai' },
    { code: 'SIN', name: 'Singapore (SIN)', city: 'Singapore' },
    { code: 'SYD', name: 'Sydney (SYD)', city: 'Sydney' },
    { code: 'FRA', name: 'Frankfurt (FRA)', city: 'Frankfurt' },
    { code: 'ORD', name: 'Chicago O\'Hare (ORD)', city: 'Chicago' },
    { code: 'DFW', name: 'Dallas/Fort Worth (DFW)', city: 'Dallas' },
    { code: 'ATL', name: 'Atlanta (ATL)', city: 'Atlanta' },
    { code: 'MIA', name: 'Miami (MIA)', city: 'Miami' },
    { code: 'SEA', name: 'Seattle (SEA)', city: 'Seattle' },
    { code: 'SFO', name: 'San Francisco (SFO)', city: 'San Francisco' },
    { code: 'BOS', name: 'Boston (BOS)', city: 'Boston' },
    { code: 'LAS', name: 'Las Vegas (LAS)', city: 'Las Vegas' },
    { code: 'PHX', name: 'Phoenix (PHX)', city: 'Phoenix' },
    { code: 'DEN', name: 'Denver (DEN)', city: 'Denver' },
    { code: 'MSP', name: 'Minneapolis (MSP)', city: 'Minneapolis' }
  ];
  const animationRef = useRef<number | null>(null);

  // Load available genres on component mount
  useEffect(() => {
    const loadGenres = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/available`);
        const data = await response.json();
        if (data.success) {
          setAvailableGenres(Object.keys(data.data.genres));
        }
      } catch (error) {
        console.error('Failed to load genres:', error);
      }
    };
    loadGenres();

    // Handle message channel errors from browser extensions
    const handleError = (event: ErrorEvent) => {
      if (event.message && event.message.includes('message channel closed')) {
        // Suppress browser extension message channel errors
        event.preventDefault();
        return false;
      }
    };

    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  const generateExperience = async () => {
    // Validate inputs
    if (!origin || !destination) {
      toast({
        title: 'Missing Information',
        description: 'Please select both origin and destination airports.',
        variant: 'destructive'
      });
      return;
    }

    if (origin === destination) {
      toast({
        title: 'Invalid Route',
        description: 'Origin and destination must be different.',
        variant: 'destructive'
      });
      return;
    }

    setLoading(true);
    try {
      console.log(`Generating VR experience: ${origin} → ${destination}`);
      
      // Generate VR experience
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/vr/vr-experiences/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          origin_code: origin,
          destination_code: destination,
          experience_type: experienceType,
          camera_mode: cameraMode
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('VR Experience API Response:', data);
      
      if (data.success) {
        setExperience(data.data);
        setFlightPath(data.data.flight_path);
        
        // Randomize visual style for unique experience
        const styles: Array<'default' | 'neon' | 'aurora' | 'cosmic'> = ['default', 'neon', 'aurora', 'cosmic'];
        setVisualStyle(styles[Math.floor(Math.random() * styles.length)]);
        
        // Randomize animation speed slightly
        setAnimationSpeed(0.8 + Math.random() * 0.4); // 0.8 to 1.2
        
        toast({
          title: 'VR Experience Created',
          description: `${origin} → ${destination} experience ready!`
        });
        
        // Generate AI music composition for this route
        await generateRouteMusic(data.data);
        
        // Auto-start the animation
        setTimeout(() => {
          playAnimation();
        }, 500);
      } else {
        throw new Error(data.message || 'Failed to create VR experience');
      }
    } catch (error) {
      console.error('VR Experience Generation Error:', error);
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to create VR experience. Please check your network connection and try again.',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  };

  const generateRouteMusic = async (experienceData: any) => {
    try {
      // Calculate route features for AI composer
      const distance = Math.sqrt(
        Math.pow(experienceData.route.destination_coords[0] - experienceData.route.origin_coords[0], 2) +
        Math.pow(experienceData.route.destination_coords[1] - experienceData.route.origin_coords[1], 2)
      ) * 111; // Rough km conversion
      
      const latRange = Math.abs(experienceData.route.destination_coords[0] - experienceData.route.origin_coords[0]);
      const lonRange = Math.abs(experienceData.route.destination_coords[1] - experienceData.route.origin_coords[1]);
      
      // Determine direction
      const latDiff = experienceData.route.destination_coords[0] - experienceData.route.origin_coords[0];
      const lonDiff = experienceData.route.destination_coords[1] - experienceData.route.origin_coords[1];
      let direction = 'E';
      if (Math.abs(latDiff) > Math.abs(lonDiff)) {
        direction = latDiff > 0 ? 'N' : 'S';
      } else {
        direction = lonDiff > 0 ? 'E' : 'W';
      }
      
      // Get AI recommendations for best genre
      const recResponse = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          distance: Math.min(distance, 15000),
          latitude_range: latRange,
          longitude_range: lonRange,
          direction: direction
        })
      });
      
      const recData = await recResponse.json();
      
      // Use route-specific seed with time variation for genre selection
      const routeSeed = `${experienceData.route.origin}-${experienceData.route.destination}`.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
      const timeVariation = Date.now() % 1000; // Add millisecond variation
      const variedSeed = routeSeed + timeVariation;
      
      console.log(`Route: ${experienceData.route.origin}-${experienceData.route.destination}, Seed: ${routeSeed}, Varied: ${variedSeed}`);
      console.log('Recommendations:', recData.data?.recommendations);
      
      // Use selected genre or AI recommendation
      let recommendedGenre: string;
      
      if (selectedGenre === 'auto') {
        // Select from top 3 recommendations with some randomness
        const topRecommendations = recData.data?.recommendations?.slice(0, 3) || [];
        const genreIndex = variedSeed % Math.max(topRecommendations.length, 1);
        recommendedGenre = topRecommendations[genreIndex]?.genre || 'cinematic';
        
        console.log(`AI Selected genre index ${genreIndex}: ${recommendedGenre} for route ${experienceData.route.origin}-${experienceData.route.destination}`);
      } else {
        // Use manually selected genre
        recommendedGenre = selectedGenre;
        console.log(`User selected genre: ${recommendedGenre} for route ${experienceData.route.origin}-${experienceData.route.destination}`);
      }
      
      // Generate composition with recommended genre and route-specific seed
      const compResponse = await fetch(`${import.meta.env.VITE_API_BASE_URL}/ai/ai-genres/compose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          genre: recommendedGenre,
          route_features: {
            distance: Math.min(distance, 15000),
            latitude_range: latRange,
            longitude_range: lonRange,
            direction: direction,
            seed: routeSeed  // Pass seed for reproducible but unique compositions
          },
          duration: 30
        })
      });
      
      const compData = await compResponse.json();
      if (compData.success) {
        setComposition(compData.data);
        console.log(`Generated ${recommendedGenre} composition for route:`, compData.data);
        
        // Show toast with genre info
        toast({
          title: `🎵 ${recommendedGenre.toUpperCase()} Music Generated`,
          description: `Unique composition for ${experienceData.route.origin} → ${experienceData.route.destination}`,
        });
      }
    } catch (error) {
      console.error('Failed to generate route music:', error);
    }
  };

  const playAnimation = async () => {
    if (playing) return;
    
    setPlaying(true);
    setCurrentPoint(0);
    
    // Start music if available
    if (composition && !musicPlaying) {
      try {
        setMusicPlaying(true);
        const notes = composition.note_sequence.map((n: any) => ({
          note: n.pitch || n.note || 60,
          velocity: n.velocity || 80,
          duration: n.duration || 0.25,
          time: n.time || 0
        }));
        
        // Play music in background
        audioPlayer.playComposition(notes, composition.tempo || 120).then(() => {
          setMusicPlaying(false);
        }).catch(() => {
          setMusicPlaying(false);
        });
        
        toast({
          title: 'Experience Started',
          description: `Playing ${composition.genre} music for your journey`,
        });
      } catch (error) {
        console.error('Music playback error:', error);
      }
    }
    
    // Animate through flight path
    let point = 0;
    const baseInterval = 50; // Base speed
    
    const animate = () => {
      point++;
      setCurrentPoint(point);
      
      if (point >= flightPath.length - 1) {
        setPlaying(false);
        setCurrentPoint(0);
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      } else {
        animationRef.current = window.setTimeout(() => {
          requestAnimationFrame(animate);
        }, baseInterval / animationSpeed);
      }
    };
    
    animate();
  };

  const stopAnimation = () => {
    setPlaying(false);
    setCurrentPoint(0);
    if (animationRef.current) {
      clearTimeout(animationRef.current);
      animationRef.current = null;
    }
    if (musicPlaying) {
      audioPlayer.stop();
      setMusicPlaying(false);
    }
  };

  // Animated plane component
  const AnimatedPlane = ({ position, rotation }: { position: [number, number, number], rotation: [number, number, number] }) => {
    const meshRef = useRef<THREE.Mesh>(null);
    
    useFrame((state) => {
      if (meshRef.current) {
        // Subtle bobbing animation
        meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      }
    });
    
    return (
      <mesh ref={meshRef} position={position} rotation={rotation}>
        <coneGeometry args={[0.3, 1, 4]} />
        <meshStandardMaterial 
          color={visualStyle === 'neon' ? '#00ff00' : visualStyle === 'aurora' ? '#ff00ff' : visualStyle === 'cosmic' ? '#ffaa00' : '#ff0000'} 
          emissive={visualStyle === 'neon' ? '#00ff00' : visualStyle === 'aurora' ? '#ff00ff' : visualStyle === 'cosmic' ? '#ffaa00' : '#ff0000'}
          emissiveIntensity={0.8}
        />
      </mesh>
    );
  };

  // Trail particles for visual effect
  const TrailParticles = ({ points, currentIndex }: { points: number[][], currentIndex: number }) => {
    const particlesRef = useRef<THREE.Points>(null);
    
    useFrame((state) => {
      if (particlesRef.current) {
        const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;
        for (let i = 0; i < positions.length; i += 3) {
          positions[i + 1] += Math.sin(state.clock.elapsedTime + i) * 0.01;
        }
        particlesRef.current.geometry.attributes.position.needsUpdate = true;
      }
    });
    
    const trailPoints = points.slice(Math.max(0, currentIndex - 20), currentIndex + 1);
    const positions = new Float32Array(trailPoints.length * 3);
    trailPoints.forEach((point, i) => {
      positions[i * 3] = point[0];
      positions[i * 3 + 1] = point[1];
      positions[i * 3 + 2] = point[2];
    });
    
    return (
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={positions.length / 3}
            array={positions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.2}
          color={visualStyle === 'neon' ? '#00ffff' : visualStyle === 'aurora' ? '#ff00ff' : visualStyle === 'cosmic' ? '#ffff00' : '#00ffff'}
          transparent
          opacity={0.6}
          sizeAttenuation
        />
      </points>
    );
  };

  const FlightPath3D = () => {
    if (!flightPath || flightPath.length === 0) return null;

    const points = flightPath.map(p => [p.position.x / 1000, p.position.y / 1000, p.position.z / 1000] as [number, number, number]);
    const currentPos = flightPath[currentPoint]?.position;
    
    // Calculate rotation for plane based on direction
    let rotation: [number, number, number] = [0, 0, 0];
    if (currentPoint > 0 && currentPoint < flightPath.length) {
      const prev = flightPath[currentPoint - 1].position;
      const curr = flightPath[currentPoint].position;
      const dx = curr.x - prev.x;
      const dy = curr.y - prev.y;
      const dz = curr.z - prev.z;
      rotation = [
        Math.atan2(dy, Math.sqrt(dx * dx + dz * dz)),
        Math.atan2(dx, dz),
        0
      ];
    }

    // Visual style colors
    const getPathColor = () => {
      switch (visualStyle) {
        case 'neon': return '#00ffff';
        case 'aurora': return '#ff00ff';
        case 'cosmic': return '#ffaa00';
        default: return '#00aaff';
      }
    };

    const getEarthColor = () => {
      switch (visualStyle) {
        case 'neon': return '#001a1a';
        case 'aurora': return '#1a001a';
        case 'cosmic': return '#1a0a00';
        default: return '#1e40af';
      }
    };

    return (
      <>
        {/* Background stars for cosmic effect */}
        {visualStyle === 'cosmic' && <Stars radius={100} depth={50} count={5000} factor={4} fade speed={1} />}
        
        {/* Flight path line */}
        <Line
          points={points}
          color={getPathColor()}
          lineWidth={3}
          transparent
          opacity={0.7}
        />
        
        {/* Trail particles */}
        {playing && currentPoint > 0 && (
          <TrailParticles points={points} currentIndex={currentPoint} />
        )}
        
        {/* Animated plane */}
        {currentPos && (
          <AnimatedPlane 
            position={[currentPos.x / 1000, currentPos.y / 1000, currentPos.z / 1000]}
            rotation={rotation}
          />
        )}
        
        {/* Progress markers along path */}
        {points.filter((_, i) => i % 10 === 0).map((point, idx) => (
          <Sphere key={idx} args={[0.15]} position={point as [number, number, number]}>
            <meshStandardMaterial 
              color={getPathColor()} 
              emissive={getPathColor()}
              emissiveIntensity={0.3}
              transparent
              opacity={0.5}
            />
          </Sphere>
        ))}
        
        {/* Earth */}
        <Sphere args={[6.371]} position={[0, 0, 0]}>
          <meshStandardMaterial 
            color={getEarthColor()} 
            wireframe 
            transparent
            opacity={0.3}
          />
        </Sphere>
        
        {/* Ambient glow around Earth */}
        <Sphere args={[6.5]} position={[0, 0, 0]}>
          <meshBasicMaterial 
            color={getPathColor()} 
            transparent
            opacity={0.05}
          />
        </Sphere>
      </>
    );
  };

  return (
    <div className="min-h-screen pt-20">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-2">
            <Glasses className="h-8 w-8" />
            VR/AR Flight Experience
          </h1>
          <p className="text-muted-foreground text-lg">
            Immersive 3D fly-through experiences with spatial audio
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Route Selection</CardTitle>
              <CardDescription>Choose your flight route</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block text-white">Origin Airport</label>
                <Select value={origin} onValueChange={setOrigin}>
                  <SelectTrigger className="text-white">
                    <SelectValue placeholder="Select origin airport" />
                  </SelectTrigger>
                  <SelectContent>
                    {popularAirports.map((airport) => (
                      <SelectItem key={airport.code} value={airport.code}>
                        {airport.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">Destination Airport</label>
                <Select value={destination} onValueChange={setDestination}>
                  <SelectTrigger className="text-white">
                    <SelectValue placeholder="Select destination airport" />
                  </SelectTrigger>
                  <SelectContent>
                    {popularAirports.map((airport) => (
                      <SelectItem key={airport.code} value={airport.code}>
                        {airport.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">Experience Type</label>
                <Select value={experienceType} onValueChange={setExperienceType}>
                  <SelectTrigger className="text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="immersive">Immersive</SelectItem>
                    <SelectItem value="cinematic">Cinematic</SelectItem>
                    <SelectItem value="educational">Educational</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">Camera Mode</label>
                <Select value={cameraMode} onValueChange={setCameraMode}>
                  <SelectTrigger className="text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="follow">Follow</SelectItem>
                    <SelectItem value="orbit">Orbit</SelectItem>
                    <SelectItem value="cinematic">Cinematic</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">Music Genre</label>
                <Select value={selectedGenre} onValueChange={setSelectedGenre}>
                  <SelectTrigger className="text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (AI Recommended)</SelectItem>
                    {availableGenres.map((genre) => (
                      <SelectItem key={genre} value={genre}>
                        {genre.charAt(0).toUpperCase() + genre.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block text-white">Quick Routes</label>
                <div className="grid grid-cols-2 gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => { setOrigin('JFK'); setDestination('LAX'); }}
                    className="text-xs"
                  >
                    NYC → LA
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => { setOrigin('LHR'); setDestination('JFK'); }}
                    className="text-xs"
                  >
                    London → NYC
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => { setOrigin('NRT'); setDestination('LAX'); }}
                    className="text-xs"
                  >
                    Tokyo → LA
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => { setOrigin('DXB'); setDestination('SIN'); }}
                    className="text-xs"
                  >
                    Dubai → Singapore
                  </Button>
                </div>
              </div>

              <Button 
                onClick={generateExperience} 
                className="w-full" 
                disabled={loading || !origin || !destination || origin === destination}
              >
                <Plane className="mr-2 h-4 w-4" />
                {loading ? 'Generating...' : 
                 !origin || !destination ? 'Select Origin & Destination' :
                 origin === destination ? 'Origin & Destination Must Differ' :
                 'Generate Experience'}
              </Button>
              
              {(!origin || !destination) && (
                <p className="text-xs text-muted-foreground text-center">
                  Please select both origin and destination airports
                </p>
              )}
              
              {origin === destination && origin && (
                <p className="text-xs text-red-400 text-center">
                  Origin and destination must be different
                </p>
              )}
            </CardContent>
          </Card>

          {experience && (
            <Card>
              <CardHeader>
                <CardTitle>Experience Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2 text-sm text-white">
                  <div><strong className="text-white">Route:</strong> {experience.route.origin} → {experience.route.destination}</div>
                  <div><strong className="text-white">Duration:</strong> {experience.duration}s</div>
                  <div><strong className="text-white">Path Points:</strong> {experience.flight_path.length}</div>
                  <div><strong className="text-white">Camera Keyframes:</strong> {experience.camera_animation.keyframes.length}</div>
                  <div><strong className="text-white">Audio Zones:</strong> {experience.spatial_audio.length}</div>
                  <div className="flex items-center gap-2">
                    <strong className="text-white">Visual Style:</strong>
                    <Badge variant="outline">{visualStyle}</Badge>
                  </div>
                  {composition && (
                    <div className="flex items-center gap-2">
                      <strong className="text-white">Music Genre:</strong>
                      <Badge variant="outline">{composition.genre}</Badge>
                    </div>
                  )}
                </div>

                <div className="flex flex-wrap gap-1 mt-3">
                  {experience.platforms.map((platform: string) => (
                    <Badge key={platform} variant="secondary" className="text-xs">
                      {platform}
                    </Badge>
                  ))}
                </div>

                <div className="flex gap-2 mt-4">
                  <Button 
                    onClick={playing ? stopAnimation : playAnimation} 
                    disabled={!flightPath.length} 
                    size="sm" 
                    className="flex-1"
                    variant={playing ? "destructive" : "default"}
                  >
                    {playing ? (
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
                  {composition && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={async () => {
                        if (musicPlaying) {
                          audioPlayer.stop();
                          setMusicPlaying(false);
                        } else {
                          setMusicPlaying(true);
                          const notes = composition.note_sequence.map((n: any) => ({
                            note: n.pitch || n.note || 60,
                            velocity: n.velocity || 80,
                            duration: n.duration || 0.25,
                            time: n.time || 0
                          }));
                          await audioPlayer.playComposition(notes, composition.tempo || 120);
                          setMusicPlaying(false);
                        }
                      }}
                    >
                      <Music2 className="h-4 w-4" />
                    </Button>
                  )}
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
                
                {musicPlaying && (
                  <div className="mt-2 p-2 bg-primary/10 rounded text-xs text-center">
                    🎵 Playing {composition?.genre} music...
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Supported Platforms</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">WebXR</Badge>
                  <span className="text-white">Browser-based VR</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline">Oculus</Badge>
                  <span className="text-white">Quest 2/3</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline">Unity</Badge>
                  <span className="text-white">Game engine</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline">ARKit</Badge>
                  <span className="text-white">iOS AR</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline">ARCore</Badge>
                  <span className="text-white">Android AR</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 3D Visualization */}
        <div className="lg:col-span-2">
          <Card className="h-[600px]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                3D Flight Path Preview
              </CardTitle>
              <CardDescription>
                Interactive 3D visualization of your flight route
              </CardDescription>
            </CardHeader>
            <CardContent className="h-[500px]">
              {flightPath.length > 0 ? (
                <Canvas 
                  camera={{ position: [15, 15, 15], fov: 60 }}
                  onCreated={({ gl }) => {
                    // Handle WebGL context loss
                    gl.domElement.addEventListener('webglcontextlost', (event) => {
                      event.preventDefault();
                      console.log('WebGL context lost, attempting to restore...');
                    });
                    
                    gl.domElement.addEventListener('webglcontextrestored', () => {
                      console.log('WebGL context restored');
                    });

                    // Suppress browser extension errors
                    const originalError = console.error;
                    console.error = (...args) => {
                      const message = args.join(' ');
                      if (message.includes('message channel closed') || 
                          message.includes('listener indicated an asynchronous response')) {
                        return; // Suppress these specific errors
                      }
                      originalError.apply(console, args);
                    };
                  }}
                  fallback={
                    <div className="h-full flex items-center justify-center text-muted-foreground">
                      <div className="text-center">
                        <div className="text-red-500 mb-2">WebGL Error</div>
                        <p>Unable to initialize 3D graphics</p>
                        <p className="text-xs mt-2">Try refreshing the page</p>
                      </div>
                    </div>
                  }
                >
                  <ambientLight intensity={visualStyle === 'neon' ? 0.3 : visualStyle === 'cosmic' ? 0.2 : 0.5} />
                  <pointLight position={[10, 10, 10]} intensity={visualStyle === 'aurora' ? 1.5 : 1} />
                  {visualStyle === 'aurora' && (
                    <>
                      <pointLight position={[-10, 5, -10]} color="#ff00ff" intensity={0.5} />
                      <pointLight position={[10, -5, 10]} color="#00ffff" intensity={0.5} />
                    </>
                  )}
                  <FlightPath3D />
                  <OrbitControls enableZoom={true} autoRotate={playing} autoRotateSpeed={0.5} />
                </Canvas>
              ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <Glasses className="h-16 w-16 mx-auto mb-4 opacity-50" />
                    <p>Generate an experience to see the 3D visualization</p>
                    <p className="text-xs mt-2">Each experience will be unique with different visuals and music!</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {experience && (
            <>
              <Card className="mt-4">
                <CardHeader>
                  <CardTitle>Unique Experience Features</CardTitle>
                  <CardDescription>Every journey is different!</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 gap-3">
                    <div className="p-3 bg-muted rounded">
                      <div className="font-semibold mb-1 text-white">🎨 Visual Style</div>
                      <div className="text-sm text-white capitalize">
                        {visualStyle} theme with dynamic lighting
                      </div>
                    </div>
                    <div className="p-3 bg-muted rounded">
                      <div className="font-semibold mb-1 text-white">🎵 AI Music</div>
                      <div className="text-sm text-white">
                        {composition ? `${composition.genre} composition` : 'Generating...'}
                      </div>
                    </div>
                    <div className="p-3 bg-muted rounded">
                      <div className="font-semibold mb-1 text-white">✨ Animation</div>
                      <div className="text-sm text-white">
                        Speed: {animationSpeed.toFixed(1)}x with particles
                      </div>
                    </div>
                    <div className="p-3 bg-muted rounded">
                      <div className="font-semibold mb-1 text-white">🌍 3D Path</div>
                      <div className="text-sm text-white">
                        {flightPath.length} points with trail effects
                      </div>
                    </div>
                  </div>
                  
                  {playing && (
                    <div className="mt-4">
                      <div className="flex justify-between text-xs text-muted-foreground mb-1">
                        <span>Progress</span>
                        <span>{Math.round((currentPoint / flightPath.length) * 100)}%</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2">
                        <div 
                          className="bg-primary h-2 rounded-full transition-all duration-300"
                          style={{ width: `${(currentPoint / flightPath.length) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
              
              <Card className="mt-4">
                <CardHeader>
                  <CardTitle>Standard Features</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 gap-3">
                    <div className="p-3 bg-muted rounded">
                      <div className="font-semibold mb-1 text-white">Camera Animation</div>
                      <div className="text-sm text-white">
                        Dynamic camera movements and angles
                      </div>
                    </div>
                    <div className="p-3 bg-muted rounded">
                      <div className="font-semibold mb-1 text-white">Spatial Audio</div>
                      <div className="text-sm text-white">
                        3D positioned music along the route
                      </div>
                    </div>
                    <div className="p-3 bg-muted rounded">
                      <div className="font-semibold mb-1 text-white">Interactive Hotspots</div>
                      <div className="text-sm text-white">
                        Information points along the journey
                      </div>
                    </div>
                    <div className="p-3 bg-muted rounded">
                      <div className="font-semibold mb-1 text-white">Multi-Platform</div>
                      <div className="text-sm text-white">
                        WebXR, Oculus, Unity, AR ready
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
      </div>
    </div>
  );
}

