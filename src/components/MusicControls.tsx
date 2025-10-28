import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Settings, Music, Sliders } from "lucide-react";

interface MusicControlsProps {
  onSettingsChange: (settings: MusicSettings) => void;
}

export interface MusicSettings {
  tempo: number;
  key: string;
  scale: string;
  complexity: number;
  harmonization: string;
}

const MusicControls = ({ onSettingsChange }: MusicControlsProps) => {
  const [tempo, setTempo] = useState([120]);
  const [complexity, setComplexity] = useState([50]);
  const [key, setKey] = useState("C");
  const [scale, setScale] = useState("major");
  const [harmonization, setHarmonization] = useState("triads");

  const handleChange = () => {
    onSettingsChange({
      tempo: tempo[0],
      key,
      scale,
      complexity: complexity[0],
      harmonization,
    });
  };

  return (
    <Card className="cockpit-panel p-6">
      <div className="flex items-center gap-2 mb-6">
        <Settings className="w-5 h-5 text-primary" />
        <h3 className="text-xl font-semibold">Music Parameters</h3>
      </div>

      <Tabs defaultValue="basic" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="basic">
            <Music className="w-4 h-4 mr-2" />
            Basic
          </TabsTrigger>
          <TabsTrigger value="advanced">
            <Sliders className="w-4 h-4 mr-2" />
            Advanced
          </TabsTrigger>
        </TabsList>

        <TabsContent value="basic" className="space-y-6 mt-6">
          {/* Tempo */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <Label>Tempo (BPM)</Label>
              <span className="text-sm text-primary font-medium">{tempo[0]}</span>
            </div>
            <Slider
              value={tempo}
              onValueChange={(val) => {
                setTempo(val);
                handleChange();
              }}
              min={60}
              max={200}
              step={1}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Controls the speed of the composition based on flight distance
            </p>
          </div>

          {/* Musical Key */}
          <div className="space-y-3">
            <Label>Musical Key</Label>
            <Select value={key} onValueChange={(val) => { setKey(val); handleChange(); }}>
              <SelectTrigger className="bg-background/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].map((k) => (
                  <SelectItem key={k} value={k}>{k}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Scale */}
          <div className="space-y-3">
            <Label>Scale Type</Label>
            <Select value={scale} onValueChange={(val) => { setScale(val); handleChange(); }}>
              <SelectTrigger className="bg-background/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="major">Major</SelectItem>
                <SelectItem value="minor">Minor</SelectItem>
                <SelectItem value="pentatonic">Pentatonic</SelectItem>
                <SelectItem value="blues">Blues</SelectItem>
                <SelectItem value="chromatic">Chromatic</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </TabsContent>

        <TabsContent value="advanced" className="space-y-6 mt-6">
          {/* Complexity */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <Label>Melodic Complexity</Label>
              <span className="text-sm text-primary font-medium">{complexity[0]}%</span>
            </div>
            <Slider
              value={complexity}
              onValueChange={(val) => {
                setComplexity(val);
                handleChange();
              }}
              min={0}
              max={100}
              step={1}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Higher values create more intricate melodic patterns
            </p>
          </div>

          {/* Harmonization */}
          <div className="space-y-3">
            <Label>Harmonization Style</Label>
            <Select value={harmonization} onValueChange={(val) => { setHarmonization(val); handleChange(); }}>
              <SelectTrigger className="bg-background/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None (Melody Only)</SelectItem>
                <SelectItem value="triads">Triads</SelectItem>
                <SelectItem value="seventh">Seventh Chords</SelectItem>
                <SelectItem value="extended">Extended Harmonies</SelectItem>
                <SelectItem value="layered">Layered Voices</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Adds harmonic layers for multi-stop routes
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </Card>
  );
};

export default MusicControls;
