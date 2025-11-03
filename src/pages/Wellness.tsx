import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Heart, Wind, Waves, Moon } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { generateWellnessComposition } from "@/lib/api/wellness";
import { audioPlayer } from "@/lib/audioPlayer";

const Wellness = () => {
  const { toast } = useToast();
  const [calmLevel, setCalmLevel] = useState([50]);
  const [selectedTheme, setSelectedTheme] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentComposition, setCurrentComposition] = useState<any>(null);

  const handleGenerateWellness = async () => {
    if (!selectedTheme) return;
    
    setIsGenerating(true);
    const response = await generateWellnessComposition({
      theme: selectedTheme,
      calm_level: calmLevel[0],
      duration_minutes: 5,
    });
    
    if (response.data) {
      setCurrentComposition(response.data);
      
      toast({
        title: "Wellness Composition Ready",
        description: `Your ${selectedTheme} soundscape has been generated with ${response.data.notes?.length || 0} notes`,
      });
    } else if (response.error) {
      console.error('Wellness generation error:', response.error);
      toast({
        title: "Generation Failed",
        description: response.error.message || "Could not generate wellness composition",
        variant: "destructive",
      });
    }
    
    setIsGenerating(false);
  };

  const handlePlayWellness = async () => {
    if (!currentComposition || !currentComposition.notes) {
      toast({
        title: "No Composition",
        description: "Please generate a wellness composition first",
        variant: "destructive",
      });
      return;
    }
    
    if (isPlaying) {
      audioPlayer.stop();
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      try {
        await audioPlayer.playComposition(
          currentComposition.notes,
          60, // Slow tempo for wellness
          (progress) => {
            console.log('Wellness playback progress:', progress);
          }
        );
        setIsPlaying(false);
        toast({
          title: "Playback Complete",
          description: "Wellness session finished",
        });
      } catch (error) {
        console.error('Wellness playback error:', error);
        setIsPlaying(false);
        toast({
          title: "Playback Error",
          description: error instanceof Error ? error.message : "Could not play wellness composition",
          variant: "destructive",
        });
      }
    }
  };

  const themes = [
    {
      id: "ocean",
      title: "Ocean Breeze",
      description: "Calming coastal routes with gentle wave-like melodies",
      icon: Waves,
      routes: ["LAX → HNL", "MIA → CUN"],
    },
    {
      id: "mountain",
      title: "Mountain Serenity",
      description: "Peaceful mountain routes with ambient soundscapes",
      icon: Wind,
      routes: ["DEN → SLC", "GVA → INN"],
    },
    {
      id: "night",
      title: "Night Flight",
      description: "Soothing overnight routes for relaxation",
      icon: Moon,
      routes: ["JFK → LHR", "LAX → NRT"],
    },
  ];

  return (
    <div className="min-h-screen pt-20 py-24 px-4">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-12">
          <Heart className="w-16 h-16 mx-auto mb-4 text-primary" />
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Therapeutic & Wellness
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Generate calming sounds from serene routes for relaxation and meditation
          </p>
        </div>

        <div className="max-w-4xl mx-auto space-y-8">
          <Card>
            <CardHeader>
              <CardTitle>Calm Level</CardTitle>
              <CardDescription>
                Adjust the intensity of your therapeutic soundscape
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Slider
                value={calmLevel}
                onValueChange={setCalmLevel}
                max={100}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Gentle</span>
                <span>Moderate</span>
                <span>Deep</span>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6 md:grid-cols-3">
            {themes.map((theme) => {
              const Icon = theme.icon;
              return (
                <Card 
                  key={theme.id}
                  className={`cursor-pointer transition-all ${
                    selectedTheme === theme.id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => setSelectedTheme(theme.id)}
                >
                  <CardHeader>
                    <Icon className="w-12 h-12 mb-4 text-primary" />
                    <CardTitle>{theme.title}</CardTitle>
                    <CardDescription>{theme.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Suggested Routes:</p>
                      {theme.routes.map((route) => (
                        <div key={route} className="text-sm text-muted-foreground">
                          {route}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {selectedTheme && (
            <Card>
              <CardHeader>
                <CardTitle>Start Your Wellness Session</CardTitle>
                <CardDescription>
                  Generate a calming composition based on your selected theme
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button 
                  className="w-full" 
                  size="lg"
                  onClick={handleGenerateWellness}
                  disabled={isGenerating}
                >
                  {isGenerating ? "Generating..." : "Generate Calming Soundscape"}
                </Button>
                
                {currentComposition && (
                  <Button
                    className="w-full"
                    size="lg"
                    variant="outline"
                    onClick={handlePlayWellness}
                  >
                    {isPlaying ? "Stop" : "Play Wellness Composition"}
                  </Button>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default Wellness;
