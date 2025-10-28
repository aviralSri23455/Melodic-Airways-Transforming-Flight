import { Card } from "@/components/ui/card";
import { Music, Waves, Gauge, Layers } from "lucide-react";

interface MusicDNAProps {
  composition?: any;
}

const MusicDNA = ({ composition }: MusicDNAProps) => {
  if (!composition) {
    return (
      <Card className="cockpit-panel p-6">
        <div className="text-center text-muted-foreground">
          <Music className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Generate music to see its unique DNA</p>
        </div>
      </Card>
    );
  }

  const scale = composition.scale || 'major';
  const tempo = composition.tempo || 120;
  const noteCount = composition.note_count || 0;
  const tracks = composition.tracks || { melody: 0, harmony: 0, bass: 0 };
  const rootNote = composition.root_note || 60;
  const updateId = composition.update_id || '';

  // Calculate uniqueness score based on characteristics
  const uniquenessScore = Math.min(100, 
    (noteCount / 100 * 30) + 
    (tempo / 200 * 20) + 
    (Object.keys(tracks).length * 15) + 
    5
  );

  // Map scale to color
  const scaleColors: Record<string, string> = {
    major: 'text-yellow-400',
    minor: 'text-blue-400',
    pentatonic: 'text-green-400',
    blues: 'text-purple-400',
    dorian: 'text-orange-400',
    phrygian: 'text-red-400',
  };

  const scaleColor = scaleColors[scale] || 'text-primary';

  return (
    <Card className="cockpit-panel p-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Waves className="w-5 h-5 text-primary" />
            Music DNA
          </h3>
          <div className="text-xs text-muted-foreground">
            Unique Composition
          </div>
        </div>

        {/* Uniqueness Score */}
        <div className="relative">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">Uniqueness</span>
            <span className="text-sm font-bold text-primary">{Math.round(uniquenessScore)}%</span>
          </div>
          <div className="h-2 bg-background/50 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
              style={{ width: `${uniquenessScore}%` }}
            />
          </div>
        </div>

        {/* Musical Characteristics Grid */}
        <div className="grid grid-cols-2 gap-3">
          {/* Scale */}
          <div className="p-3 bg-background/30 rounded-lg border border-primary/20">
            <div className="flex items-center gap-2 mb-1">
              <Music className="w-4 h-4 text-primary" />
              <span className="text-xs text-muted-foreground">Scale</span>
            </div>
            <p className={`text-lg font-bold capitalize ${scaleColor}`}>
              {scale}
            </p>
          </div>

          {/* Tempo */}
          <div className="p-3 bg-background/30 rounded-lg border border-primary/20">
            <div className="flex items-center gap-2 mb-1">
              <Gauge className="w-4 h-4 text-primary" />
              <span className="text-xs text-muted-foreground">Tempo</span>
            </div>
            <p className="text-lg font-bold text-primary">
              {tempo} <span className="text-xs">BPM</span>
            </p>
          </div>

          {/* Note Count */}
          <div className="p-3 bg-background/30 rounded-lg border border-primary/20">
            <div className="flex items-center gap-2 mb-1">
              <Waves className="w-4 h-4 text-primary" />
              <span className="text-xs text-muted-foreground">Notes</span>
            </div>
            <p className="text-lg font-bold text-primary">
              {noteCount}
            </p>
          </div>

          {/* Tracks */}
          <div className="p-3 bg-background/30 rounded-lg border border-primary/20">
            <div className="flex items-center gap-2 mb-1">
              <Layers className="w-4 h-4 text-primary" />
              <span className="text-xs text-muted-foreground">Layers</span>
            </div>
            <p className="text-lg font-bold text-primary">
              {Object.keys(tracks).length}
            </p>
          </div>
        </div>

        {/* Track Breakdown */}
        <div className="p-3 bg-background/30 rounded-lg border border-primary/20">
          <p className="text-xs text-muted-foreground mb-2">Track Composition</p>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">🎵 Melody</span>
              <span className="font-mono text-primary">{tracks.melody || 0} notes</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">🎼 Harmony</span>
              <span className="font-mono text-primary">{tracks.harmony || 0} notes</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">🎸 Bass</span>
              <span className="font-mono text-primary">{tracks.bass || 0} notes</span>
            </div>
          </div>
        </div>

        {/* Unique ID */}
        {updateId && (
          <div className="text-center p-2 bg-background/20 rounded border border-primary/10">
            <p className="text-xs text-muted-foreground mb-1">Composition ID</p>
            <p className="text-xs font-mono text-primary">
              {updateId.slice(-16)}
            </p>
          </div>
        )}

        {/* Scale Description */}
        <div className="text-xs text-muted-foreground text-center p-2 bg-background/20 rounded">
          {scale === 'major' && '✨ Bright & uplifting sound'}
          {scale === 'minor' && '🌙 Melancholic & emotional'}
          {scale === 'pentatonic' && '🎋 Asian-inspired simplicity'}
          {scale === 'blues' && '🎺 Soulful & expressive'}
          {scale === 'dorian' && '🎷 Jazz-influenced mood'}
          {scale === 'phrygian' && '🌶️ Spanish & exotic flavor'}
        </div>
      </div>
    </Card>
  );
};

export default MusicDNA;
