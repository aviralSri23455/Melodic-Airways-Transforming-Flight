import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Pause, Download, RotateCcw } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { exportToMidi } from "@/lib/midiExport";
import { useToast } from "@/hooks/use-toast";

interface MusicPlayerProps {
  routeName?: string;
  isGenerating?: boolean;
  isPlaying?: boolean;
  onPlay?: () => void;
  canPlay?: boolean;
  duration?: number;
  composition?: any;
}

const MusicPlayer = ({ 
  routeName = "No route selected", 
  isGenerating = false,
  isPlaying = false,
  onPlay,
  canPlay = false,
  duration = 0,
  composition,
}: MusicPlayerProps) => {
  const { toast } = useToast();

  const handlePlayClick = () => {
    if (onPlay) onPlay();
  };

  const handleDownload = () => {
    if (!composition || !routeName) return;
    
    try {
      exportToMidi(composition, routeName);
      toast({
        title: "Download Started",
        description: "Your MIDI file is being downloaded.",
      });
    } catch (error) {
      toast({
        title: "Download Failed",
        description: "Failed to export MIDI file.",
        variant: "destructive",
      });
    }
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  };

  return (
    <Card className="cockpit-panel p-6">
      <div className="space-y-6">
        {/* Waveform visualization placeholder */}
        <div className="relative h-32 bg-background/50 rounded-lg overflow-hidden">
          <div className="absolute inset-0 flex items-center justify-center">
            {isGenerating ? (
              <div className="flex gap-1">
                {[...Array(40)].map((_, i) => (
                  <div
                    key={i}
                    className="w-1 bg-primary rounded-full animate-pulse"
                    style={{
                      height: `${Math.random() * 100}%`,
                      animationDelay: `${i * 0.05}s`,
                    }}
                  />
                ))}
              </div>
            ) : (
              <div className="text-muted-foreground text-sm">
                Select a route and generate music to see waveform
              </div>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <div className="space-y-2">
          <Progress value={isPlaying ? 50 : 0} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0:00</span>
            <span>{canPlay ? formatTime(duration) : "0:00"}</span>
          </div>
        </div>

        {/* Route info */}
        <div className="text-center">
          <p className="text-sm text-muted-foreground">Current Route</p>
          <p className="text-lg font-semibold">{routeName}</p>
        </div>

        {/* Musical Characteristics - Show Uniqueness */}
        {composition && (
          <div className="grid grid-cols-2 gap-3 p-4 bg-background/30 rounded-lg border border-primary/20">
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Scale</p>
              <p className="text-sm font-bold text-primary capitalize">
                {composition.scale || 'Major'}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Tempo</p>
              <p className="text-sm font-bold text-primary">
                {composition.tempo || 120} BPM
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Notes</p>
              <p className="text-sm font-bold text-primary">
                {composition.note_count || 0}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Tracks</p>
              <p className="text-sm font-bold text-primary">
                {composition.tracks ? 
                  `${Object.keys(composition.tracks).length} (M+H+B)` : 
                  '3'}
              </p>
            </div>
          </div>
        )}

        {/* Unique Composition ID */}
        {composition?.update_id && (
          <div className="text-center text-xs text-muted-foreground">
            <p>Unique ID: {composition.update_id.slice(-12)}</p>
          </div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-center gap-4">
          <Button
            variant="cockpit"
            size="icon"
            onClick={() => {}}
            disabled={!canPlay}
          >
            <RotateCcw className="w-5 h-5" />
          </Button>
          
          <Button
            variant="hero"
            size="lg"
            onClick={handlePlayClick}
            disabled={!canPlay}
            className="w-20 h-20 rounded-full"
          >
            {isPlaying ? (
              <Pause className="w-8 h-8" />
            ) : (
              <Play className="w-8 h-8 ml-1" />
            )}
          </Button>

          <Button
            variant="cockpit"
            size="icon"
            disabled={!canPlay}
            onClick={handleDownload}
          >
            <Download className="w-5 h-5" />
          </Button>
        </div>

        {!canPlay && (
          <p className="text-center text-sm text-muted-foreground">
            Generate a musical route to enable playback
          </p>
        )}
      </div>
    </Card>
  );
};

export default MusicPlayer;
