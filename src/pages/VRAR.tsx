import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Box, Play, Pause, RotateCcw, Maximize, Volume2, Music } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import * as THREE from "three";
import { audioPlayer } from "@/lib/audioPlayer";
import { generateMusic } from "@/lib/api/music";

// Airport data with 3D coordinates
const airports = {
  JFK: { name: "New York JFK", lat: 40.6413, lng: -73.7781, color: "#3b82f6" },
  CDG: { name: "Paris CDG", lat: 49.0097, lng: 2.5479, color: "#8b5cf6" },
  LHR: { name: "London Heathrow", lat: 51.4700, lng: -0.4543, color: "#ec4899" },
  NRT: { name: "Tokyo Narita", lat: 35.7720, lng: 140.3929, color: "#f59e0b" },
  DXB: { name: "Dubai", lat: 25.2532, lng: 55.3657, color: "#10b981" },
  SYD: { name: "Sydney", lat: -33.9399, lng: 151.1753, color: "#06b6d4" },
  LAX: { name: "Los Angeles", lat: 33.9416, lng: -118.4085, color: "#f43f5e" },
  SIN: { name: "Singapore", lat: 1.3644, lng: 103.9915, color: "#a855f7" },
};

// Convert lat/lng to 3D sphere coordinates
const latLngToVector3 = (lat: number, lng: number, radius: number = 5) => {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lng + 180) * (Math.PI / 180);
  
  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);
  
  return new THREE.Vector3(x, y, z);
};

// Three.js Scene Component
const ThreeScene = ({ 
  canvasRef, 
  origin, 
  destination, 
  progress, 
  isPlaying 
}: { 
  canvasRef: React.RefObject<HTMLCanvasElement>;
  origin: keyof typeof airports; 
  destination: keyof typeof airports; 
  progress: number;
  isPlaying: boolean;
}) => {
  useEffect(() => {
    if (!canvasRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    const camera = new THREE.PerspectiveCamera(
      75,
      canvasRef.current.clientWidth / canvasRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 15;

    const renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef.current,
      antialias: true 
    });
    renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const pointLight1 = new THREE.PointLight(0xffffff, 1);
    pointLight1.position.set(10, 10, 10);
    scene.add(pointLight1);

    // Earth globe
    const earthGeometry = new THREE.SphereGeometry(5, 64, 64);
    const earthMaterial = new THREE.MeshStandardMaterial({
      color: 0x1e40af,
      transparent: true,
      opacity: 0.3,
      wireframe: false,
    });
    const earth = new THREE.Mesh(earthGeometry, earthMaterial);
    scene.add(earth);

    // Stars
    const starsGeometry = new THREE.BufferGeometry();
    const starsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.1 });
    const starsVertices = [];
    for (let i = 0; i < 5000; i++) {
      const x = (Math.random() - 0.5) * 200;
      const y = (Math.random() - 0.5) * 200;
      const z = (Math.random() - 0.5) * 200;
      starsVertices.push(x, y, z);
    }
    starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
    const stars = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(stars);

    // Airport markers
    const airportMarkers: THREE.Mesh[] = [];
    Object.entries(airports).forEach(([code, data]) => {
      const pos = latLngToVector3(data.lat, data.lng);
      const markerGeometry = new THREE.SphereGeometry(0.15, 16, 16);
      const markerMaterial = new THREE.MeshStandardMaterial({
        color: data.color,
        emissive: data.color,
        emissiveIntensity: 0.5,
      });
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.copy(pos);
      scene.add(marker);
      airportMarkers.push(marker);
    });

    // Flight path
    let flightPath: THREE.Line | null = null;
    let plane: THREE.Mesh | null = null;
    
    if (origin && destination) {
      const originPos = latLngToVector3(airports[origin].lat, airports[origin].lng);
      const destPos = latLngToVector3(airports[destination].lat, airports[destination].lng);
      
      const curve = new THREE.QuadraticBezierCurve3(
        originPos,
        new THREE.Vector3(
          (originPos.x + destPos.x) / 2,
          (originPos.y + destPos.y) / 2 + 2,
          (originPos.z + destPos.z) / 2
        ),
        destPos
      );
      
      const points = curve.getPoints(50);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0x60a5fa });
      flightPath = new THREE.Line(geometry, material);
      scene.add(flightPath);

      // Plane
      const planeGeometry = new THREE.ConeGeometry(0.1, 0.3, 8);
      const planeMaterial = new THREE.MeshStandardMaterial({ 
        color: 0xfbbf24,
        emissive: 0xf59e0b,
        emissiveIntensity: 0.5,
      });
      plane = new THREE.Mesh(planeGeometry, planeMaterial);
      const currentPoint = curve.getPoint(progress);
      plane.position.copy(currentPoint);
      scene.add(plane);
    }

    // Mouse controls
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    
    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };
    
    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const deltaX = e.clientX - previousMousePosition.x;
        const deltaY = e.clientY - previousMousePosition.y;
        
        earth.rotation.y += deltaX * 0.005;
        earth.rotation.x += deltaY * 0.005;
        
        previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    };
    
    const onMouseUp = () => {
      isDragging = false;
    };
    
    canvasRef.current.addEventListener('mousedown', onMouseDown);
    canvasRef.current.addEventListener('mousemove', onMouseMove);
    canvasRef.current.addEventListener('mouseup', onMouseUp);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      if (isPlaying) {
        earth.rotation.y += 0.001;
      }
      
      airportMarkers.forEach(marker => {
        marker.rotation.y += 0.01;
      });
      
      if (plane && origin && destination) {
        const originPos = latLngToVector3(airports[origin].lat, airports[origin].lng);
        const destPos = latLngToVector3(airports[destination].lat, airports[destination].lng);
        const curve = new THREE.QuadraticBezierCurve3(
          originPos,
          new THREE.Vector3(
            (originPos.x + destPos.x) / 2,
            (originPos.y + destPos.y) / 2 + 2,
            (originPos.z + destPos.z) / 2
          ),
          destPos
        );
        const currentPoint = curve.getPoint(progress);
        plane.position.copy(currentPoint);
      }
      
      renderer.render(scene, camera);
    };
    
    animate();

    // Cleanup
    return () => {
      canvasRef.current?.removeEventListener('mousedown', onMouseDown);
      canvasRef.current?.removeEventListener('mousemove', onMouseMove);
      canvasRef.current?.removeEventListener('mouseup', onMouseUp);
      renderer.dispose();
    };
  }, [canvasRef, origin, destination, progress, isPlaying]);

  return null;
};

const VRAR = () => {
  const { toast } = useToast();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedOrigin, setSelectedOrigin] = useState<keyof typeof airports>("JFK");
  const [selectedDestination, setSelectedDestination] = useState<keyof typeof airports>("CDG");
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState([1]);
  const [isVRSupported, setIsVRSupported] = useState(false);
  const [vrSession, setVrSession] = useState<any>(null);
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const [isLoadingSpatialAudio, setIsLoadingSpatialAudio] = useState(false);
  const [spatialAudioEnabled, setSpatialAudioEnabled] = useState(false);
  const [isLoadingMusic, setIsLoadingMusic] = useState(false);
  const [isMusicPlaying, setIsMusicPlaying] = useState(false);
  const [musicComposition, setMusicComposition] = useState<any>(null);

  const handleGenerateMusic = async () => {
    setIsLoadingMusic(true);
    
    try {
      const response = await generateMusic({
        origin: selectedOrigin,
        destination: selectedDestination,
        tempo: 120,
        music_style: 'ambient'
      });

      if (response.data) {
        setMusicComposition(response.data);
        toast({
          title: "Music Generated!",
          description: `Composition created for ${selectedOrigin} → ${selectedDestination}. Click Play Music to listen.`,
        });
        
        // Auto-play the music after generation
        const notes = response.data.music?.notes;
        if (notes && notes.length > 0) {
          setTimeout(async () => {
            try {
              setIsMusicPlaying(true);
              await audioPlayer.playComposition(
                notes,
                response.data.music.tempo || 120,
                (progress) => {
                  // Update progress if needed
                }
              );
              setIsMusicPlaying(false);
              toast({
                title: "Playback Complete",
                description: "Music finished playing",
              });
            } catch (error) {
              console.error('Auto-play error:', error);
              setIsMusicPlaying(false);
            }
          }, 500); // Small delay to ensure UI updates
        }
      } else if (response.error) {
        throw new Error(response.error.message);
      }
    } catch (error) {
      console.error('Music generation error:', error);
      toast({
        title: "Music Generation Failed",
        description: error instanceof Error ? error.message : "Could not generate music",
        variant: "destructive",
      });
    } finally {
      setIsLoadingMusic(false);
    }
  };

  const handlePlayMusic = async () => {
    if (!musicComposition || !musicComposition.music?.notes) {
      toast({
        title: "No Music Available",
        description: "Please generate music first",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsMusicPlaying(true);
      await audioPlayer.playComposition(
        musicComposition.music.notes,
        musicComposition.music.tempo || 120,
        (progress) => {
          // Could update a progress indicator here
        }
      );
      setIsMusicPlaying(false);
      toast({
        title: "Playback Complete",
        description: "Music finished playing",
      });
    } catch (error) {
      console.error('Playback error:', error);
      setIsMusicPlaying(false);
      toast({
        title: "Playback Failed",
        description: error instanceof Error ? error.message : "Could not play music",
        variant: "destructive",
      });
    }
  };

  const handleStopMusic = () => {
    audioPlayer.stop();
    setIsMusicPlaying(false);
    toast({
      title: "Playback Stopped",
      description: "Music playback stopped",
    });
  };

  const handleEnableSpatialAudio = async () => {
    setIsLoadingSpatialAudio(true);
    
    try {
      const { generateSpatialAudio } = await import('@/lib/api/vrar');
      
      const response = await generateSpatialAudio(
        selectedOrigin,
        selectedDestination,
        'ambient'
      );

      if (response.data) {
        setSpatialAudioEnabled(true);
        toast({
          title: "Spatial Audio Enabled!",
          description: "3D audio zones have been generated for your flight path",
        });
      } else if (response.error) {
        throw new Error(response.error.message);
      }
    } catch (error) {
      console.error('Spatial audio error:', error);
      toast({
        title: "Spatial Audio Failed",
        description: error instanceof Error ? error.message : "Could not enable spatial audio",
        variant: "destructive",
      });
    } finally {
      setIsLoadingSpatialAudio(false);
    }
  };

  useEffect(() => {
    // Check WebXR support
    if ('xr' in navigator) {
      (navigator as any).xr?.isSessionSupported('immersive-vr').then((supported: boolean) => {
        setIsVRSupported(supported);
      }).catch(() => {
        setIsVRSupported(false);
      });
    }
  }, []);

  useEffect(() => {
    let animationFrame: number;
    
    if (isPlaying) {
      const animate = () => {
        setProgress((prev) => {
          const newProgress = prev + (0.005 * speed[0]);
          if (newProgress >= 1) {
            setIsPlaying(false);
            return 1;
          }
          return newProgress;
        });
        animationFrame = requestAnimationFrame(animate);
      };
      animationFrame = requestAnimationFrame(animate);
    }
    
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [isPlaying, speed]);

  // Handle flight completion notification
  useEffect(() => {
    if (!isPlaying && progress >= 1) {
      toast({
        title: "Flight Complete!",
        description: `Journey from ${selectedOrigin} to ${selectedDestination} finished.`,
      });
    }
  }, [isPlaying, progress, selectedOrigin, selectedDestination, toast]);

  const handlePlayPause = () => {
    if (progress >= 1) {
      setProgress(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setProgress(0);
    setIsPlaying(false);
  };

  const handleEnterVR = async () => {
    if (!isVRSupported) {
      toast({
        title: "VR Not Supported",
        description: "Your browser or device doesn't support WebXR VR. Try using a VR headset with Chrome or Firefox.",
        variant: "destructive",
      });
      return;
    }

    setIsLoadingSession(true);
    
    try {
      // Import VR API
      const { createVRSession } = await import('@/lib/api/vrar');
      
      // Create VR session with backend
      const response = await createVRSession({
        origin: selectedOrigin,
        destination: selectedDestination,
        enable_spatial_audio: true,
        quality: 'high',
      });

      if (response.data) {
        setVrSession(response.data);
        
        toast({
          title: "VR Session Created!",
          description: "Put on your VR headset to experience the immersive flight.",
        });

        // Request VR session from browser
        if ('xr' in navigator) {
          const xr = (navigator as any).xr;
          const session = await xr.requestSession('immersive-vr', {
            requiredFeatures: ['local-floor'],
            optionalFeatures: ['hand-tracking', 'bounded-floor']
          });
          
          toast({
            title: "Entering VR Mode",
            description: `Flying from ${selectedOrigin} to ${selectedDestination}`,
          });
          
          // VR session is now active
          session.addEventListener('end', () => {
            toast({
              title: "VR Session Ended",
              description: "You've exited VR mode",
            });
          });
        }
      } else if (response.error) {
        throw new Error(response.error.message);
      }
    } catch (error) {
      console.error('VR session error:', error);
      toast({
        title: "VR Session Failed",
        description: error instanceof Error ? error.message : "Could not create VR session",
        variant: "destructive",
      });
    } finally {
      setIsLoadingSession(false);
    }
  };

  return (
    <div className="min-h-screen pt-20 py-24 px-4">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-8">
          <Box className="w-16 h-16 mx-auto mb-4 text-primary" />
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            VR/AR Immersive Experience
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Fly through your musical journey in stunning 3D visualization
          </p>
          {isVRSupported && (
            <Badge variant="secondary" className="mt-2">
              VR Ready
            </Badge>
          )}
        </div>

        <Tabs defaultValue="3d-view" className="space-y-6">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2">
            <TabsTrigger value="3d-view">3D Visualization</TabsTrigger>
            <TabsTrigger value="controls">Flight Controls</TabsTrigger>
          </TabsList>

          <TabsContent value="3d-view" className="space-y-6">
            {/* 3D Canvas */}
            <Card className="overflow-hidden">
              <CardContent className="p-0">
                <div className="w-full h-[600px] bg-black relative">
                  <canvas 
                    ref={canvasRef} 
                    className="w-full h-full"
                    style={{ display: 'block' }}
                  />
                  <ThreeScene
                    canvasRef={canvasRef}
                    origin={selectedOrigin}
                    destination={selectedDestination}
                    progress={progress}
                    isPlaying={isPlaying}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Playback Controls */}
            <Card>
              <CardHeader>
                <CardTitle>Playback Controls</CardTitle>
                <CardDescription>
                  Control your immersive flight experience
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Progress bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>{selectedOrigin}</span>
                    <span>{Math.round(progress * 100)}%</span>
                    <span>{selectedDestination}</span>
                  </div>
                  <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${progress * 100}%` }}
                    />
                  </div>
                </div>

                {/* Control buttons */}
                <div className="flex gap-4 justify-center">
                  <Button
                    size="lg"
                    onClick={handlePlayPause}
                    className="w-32"
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="w-4 h-4 mr-2" />
                        Pause
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Play
                      </>
                    )}
                  </Button>
                  
                  <Button
                    size="lg"
                    variant="outline"
                    onClick={handleReset}
                  >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Reset
                  </Button>

                  <Button
                    size="lg"
                    variant="secondary"
                    onClick={handleEnterVR}
                    disabled={!isVRSupported || isLoadingSession}
                  >
                    <Maximize className="w-4 h-4 mr-2" />
                    {isLoadingSession ? 'Loading...' : 'Enter VR'}
                  </Button>
                </div>

                {/* Speed control */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Flight Speed</span>
                    <span>{speed[0]}x</span>
                  </div>
                  <Slider
                    value={speed}
                    onValueChange={setSpeed}
                    min={0.5}
                    max={3}
                    step={0.5}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="controls" className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              {/* Route Selection */}
              <Card>
                <CardHeader>
                  <CardTitle>Select Route</CardTitle>
                  <CardDescription>Choose your flight path</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Origin</label>
                    <select
                      className="w-full p-2 border rounded-md bg-background"
                      value={selectedOrigin}
                      onChange={(e) => setSelectedOrigin(e.target.value as keyof typeof airports)}
                    >
                      {Object.entries(airports).map(([code, data]) => (
                        <option key={code} value={code}>
                          {code} - {data.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Destination</label>
                    <select
                      className="w-full p-2 border rounded-md bg-background"
                      value={selectedDestination}
                      onChange={(e) => setSelectedDestination(e.target.value as keyof typeof airports)}
                    >
                      {Object.entries(airports).map(([code, data]) => (
                        <option key={code} value={code}>
                          {code} - {data.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </CardContent>
              </Card>

              {/* VR/AR Features */}
              <Card>
                <CardHeader>
                  <CardTitle>Immersive Features</CardTitle>
                  <CardDescription>Enhanced VR/AR capabilities</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">3D Globe Rotation</span>
                      <Badge variant="secondary">Active</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Real-time Flight Path</span>
                      <Badge variant="secondary">Active</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">WebXR VR Support</span>
                      <Badge variant={isVRSupported ? "default" : "outline"}>
                        {isVRSupported ? "Supported" : "Not Available"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Spatial Audio</span>
                      <Badge variant={spatialAudioEnabled ? "default" : "outline"}>
                        {spatialAudioEnabled ? "Enabled" : "Disabled"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Music Composition</span>
                      <Badge variant={musicComposition ? "default" : "outline"}>
                        {musicComposition ? "Generated" : "Not Generated"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Music Playback</span>
                      <Badge variant={isMusicPlaying ? "default" : "outline"}>
                        {isMusicPlaying ? "Playing" : "Stopped"}
                      </Badge>
                    </div>
                  </div>

                  <div className="space-y-2 mt-4">
                    <Button 
                      className="w-full" 
                      variant="outline"
                      onClick={handleEnableSpatialAudio}
                      disabled={isLoadingSpatialAudio}
                    >
                      <Volume2 className="w-4 h-4 mr-2" />
                      {isLoadingSpatialAudio ? 'Loading...' : 'Enable Spatial Audio'}
                    </Button>

                    <Button 
                      className="w-full" 
                      variant="outline"
                      onClick={handleGenerateMusic}
                      disabled={isLoadingMusic}
                    >
                      <Music className="w-4 h-4 mr-2" />
                      {isLoadingMusic ? 'Generating...' : 'Generate Music'}
                    </Button>

                    {musicComposition && (
                      <Button 
                        className="w-full" 
                        variant={isMusicPlaying ? "destructive" : "default"}
                        onClick={isMusicPlaying ? handleStopMusic : handlePlayMusic}
                      >
                        {isMusicPlaying ? (
                          <>
                            <Pause className="w-4 h-4 mr-2" />
                            Stop Music
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 mr-2" />
                            Play Music
                          </>
                        )}
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* How to Use VR */}
            <Card className="border-primary/50 bg-primary/5">
              <CardHeader>
                <CardTitle>How to Use VR Mode</CardTitle>
                <CardDescription>Step-by-step guide for immersive experience</CardDescription>
              </CardHeader>
              <CardContent>
                <ol className="space-y-3 list-decimal list-inside">
                  <li className="text-sm">
                    <strong>Select your route</strong> - Choose origin and destination airports
                  </li>
                  <li className="text-sm">
                    <strong>Click "Enter VR"</strong> - This creates a VR session with the backend
                  </li>
                  <li className="text-sm">
                    <strong>Put on your VR headset</strong> - Meta Quest, HTC Vive, or any WebXR-compatible device
                  </li>
                  <li className="text-sm">
                    <strong>Use controllers</strong> - Navigate the 3D space and control playback
                  </li>
                  <li className="text-sm">
                    <strong>Enable Spatial Audio</strong> - For immersive 3D sound experience
                  </li>
                  <li className="text-sm">
                    <strong>Adjust flight speed</strong> - Control how fast you travel through the route
                  </li>
                </ol>
                
                <div className="mt-6 p-4 bg-background rounded-lg border">
                  <p className="text-sm font-medium mb-2">💡 Pro Tips:</p>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• Use Chrome or Firefox for best WebXR support</li>
                    <li>• Drag the globe with your mouse for 360° rotation</li>
                    <li>• Try different routes to see unique flight paths</li>
                    <li>• Spatial audio creates 3D sound zones along the route</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            {/* Features List */}
            <Card>
              <CardHeader>
                <CardTitle>VR/AR Capabilities</CardTitle>
                <CardDescription>What you can experience</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">Interactive 3D globe with real airport locations from OpenFlights dataset</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">Animated flight paths with curved trajectories based on real routes</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">Real-time progress tracking with visual indicators</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">Orbit controls for 360° viewing experience (drag with mouse)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">WebXR VR mode for immersive headset experience</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">Adjustable flight speed (0.5x to 3x) with smooth animations</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">Starfield background for space-like atmosphere</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">Color-coded airports with animated markers</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary">✓</span>
                    <span className="text-sm">Backend-generated spatial audio zones for 3D sound</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default VRAR;
