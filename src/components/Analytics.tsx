import { Card } from "@/components/ui/card";
import { BarChart3, TrendingUp, Music2, Layers } from "lucide-react";

interface AnalyticsProps {
  metrics?: {
    complexity: number;
    harmonicRichness: number;
    duration: number;
    uniqueNotes: number;
  };
}

// Helper function to format duration in seconds to MM:SS format
const formatDuration = (durationInSeconds: number): string => {
  const minutes = Math.floor(durationInSeconds / 60);
  const seconds = Math.round(durationInSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

const Analytics = ({ metrics }: AnalyticsProps) => {
  const defaultMetrics = {
    complexity: 0,
    harmonicRichness: 0,
    duration: 0,
    uniqueNotes: 0,
  };

  const data = metrics || defaultMetrics;
  const hasData = metrics !== undefined;

  const stats = [
    {
      icon: BarChart3,
      label: "Melodic Complexity",
      value: hasData ? `${data.complexity.toFixed(1)}%` : "—",
      description: "Pattern intricacy score",
      color: "text-primary",
    },
    {
      icon: Layers,
      label: "Harmonic Richness",
      value: hasData ? `${data.harmonicRichness.toFixed(1)}%` : "—",
      description: "Chord density measure",
      color: "text-accent",
    },
    {
      icon: TrendingUp,
      label: "Composition Length",
      value: hasData ? formatDuration(data.duration) : "—",
      description: "Total duration",
      color: "text-blue-400",
    },
    {
      icon: Music2,
      label: "Unique Notes",
      value: hasData ? String(data.uniqueNotes) : "—",
      description: "Distinct pitches used",
      color: "text-purple-400",
    },
  ];

  return (
    <Card className="cockpit-panel p-6">
      <div className="flex items-center gap-2 mb-6">
        <BarChart3 className="w-5 h-5 text-primary" />
        <h3 className="text-xl font-semibold">Musical Analytics</h3>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {stats.map((stat, index) => (
          <div
            key={index}
            className="p-4 rounded-lg bg-background/50 border border-border hover:border-primary/40 transition-colors"
          >
            <div className="flex items-center gap-2 mb-2">
              <stat.icon className={`w-5 h-5 ${stat.color}`} />
              <span className="text-sm text-muted-foreground">{stat.label}</span>
            </div>
            <div className="text-2xl font-bold mb-1">
              {stat.value}
            </div>
            <div className="text-xs text-muted-foreground">{stat.description}</div>
          </div>
        ))}
      </div>

      {!hasData && (
        <p className="text-center text-sm text-muted-foreground mt-6">
          Analytics will appear after generating a composition
        </p>
      )}
    </Card>
  );
};

export default Analytics;
