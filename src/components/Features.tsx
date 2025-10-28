import { motion } from "framer-motion";
import { Map, Music2, BarChart3, Download, Layers, Wand2 } from "lucide-react";
import { Card } from "@/components/ui/card";

const features = [
  {
    icon: Map,
    title: "Interactive Route Mapping",
    description: "Visualize flight paths on an interactive globe with real-time route selection and path optimization.",
  },
  {
    icon: Music2,
    title: "Algorithmic Composition",
    description: "Advanced algorithms convert route data into musical parameters: tempo, pitch, harmony, and rhythm.",
  },
  {
    icon: Layers,
    title: "Vector Embeddings",
    description: "MariaDB vector extensions enable semantic similarity searches for routes that 'sound similar'.",
  },
  {
    icon: Wand2,
    title: "Customizable Parameters",
    description: "Fine-tune tempo, key, harmony, and complexity to create your perfect musical interpretation.",
  },
  {
    icon: BarChart3,
    title: "Analytics Dashboard",
    description: "Track melodic complexity, harmonic richness, and other musical metrics for each composition.",
  },
  {
    icon: Download,
    title: "MIDI Export",
    description: "Download high-quality MIDI files ready for use in any DAW or music production software.",
  },
];

const Features = () => {
  return (
    <section id="features" className="py-24 px-4 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary to-transparent" />
      
      <div className="container mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Powerful Features
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Built with cutting-edge technology to transform aviation data into captivating musical experiences
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="cockpit-panel p-6 h-full hover:scale-105 transition-transform duration-300 group">
                <div className="mb-4 p-3 rounded-lg bg-primary/10 w-fit group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
