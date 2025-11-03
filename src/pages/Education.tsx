import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BookOpen, Map, Network, Music, Loader2, Play, Pause } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { getLessons, startLesson } from "@/lib/api/education";

const Education = () => {
  const { toast } = useToast();
  const [selectedLesson, setSelectedLesson] = useState<string | null>(null);
  const [lessons, setLessons] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [lessonContent, setLessonContent] = useState<any>(null);
  const [showLessonModal, setShowLessonModal] = useState(false);
  
  // Quiz state - track answers for each quiz
  const [quizAnswers, setQuizAnswers] = useState<Record<number, string>>({});
  const [quizRevealed, setQuizRevealed] = useState<Record<number, boolean>>({});
  
  // Interactive Lab state
  const [labOrigin, setLabOrigin] = useState("");
  const [labDestination, setLabDestination] = useState("");
  const [labComposition, setLabComposition] = useState<any>(null);
  const [isGeneratingLab, setIsGeneratingLab] = useState(false);
  const [isPlayingLab, setIsPlayingLab] = useState(false);
  const [activeTab, setActiveTab] = useState("lessons");

  // Fetch lessons from backend
  useEffect(() => {
    const fetchLessons = async () => {
      setIsLoading(true);
      const response = await getLessons();
      
      if (response.data) {
        // Backend returns {lessons: [...]} so we need to access the lessons array
        const lessonsArray = Array.isArray(response.data) ? response.data : (response.data as any).lessons || response.data;
        
        // Map backend lessons to UI format
        const mappedLessons = Array.isArray(lessonsArray) ? lessonsArray.map((lesson: any) => ({
          id: lesson.id,
          title: lesson.title,
          description: lesson.description,
          icon: lesson.id === 'geography' ? Map : lesson.id === 'graph-theory' ? Network : Music,
        })) : [];
        
        setLessons(mappedLessons);
      } else if (response.error) {
        console.error('Failed to load lessons:', response.error);
        toast({
          title: "Failed to Load Lessons",
          description: response.error.message,
          variant: "destructive",
        });
        // Fallback to default lessons
        setLessons([
          {
            id: "geography",
            title: "Geography Through Sound",
            description: "Learn about world geography by hearing the musical representation of flight routes",
            icon: Map,
          },
          {
            id: "graph-theory",
            title: "Graph Theory Visualization",
            description: "Understand graph algorithms through musical pathfinding",
            icon: Network,
          },
          {
            id: "music-theory",
            title: "Music Theory Basics",
            description: "Learn scales, tempo, and harmony through interactive examples",
            icon: Music,
          },
        ]);
      }
      setIsLoading(false);
    };

    fetchLessons();
  }, []);

  const handleStartLesson = async (lessonId: string) => {
    setIsStarting(true);
    setSelectedLesson(lessonId);
    
    const response = await startLesson(lessonId, lessonId, 'beginner');
    
    if (response.data) {
      setLessonContent(response.data);
      setShowLessonModal(true);
      // Reset quiz state for new lesson
      setQuizAnswers({});
      setQuizRevealed({});
      
      toast({
        title: "Lesson Loaded!",
        description: `${(response.data as any).title || 'Lesson'} is ready to explore`,
      });
    } else if (response.error) {
      toast({
        title: "Failed to Start Lesson",
        description: response.error.message,
        variant: "destructive",
      });
    }
    
    setIsStarting(false);
  };

  const handleQuizAnswer = (quizIndex: number, answer: string) => {
    setQuizAnswers(prev => ({ ...prev, [quizIndex]: answer }));
  };

  const handleRevealAnswer = (quizIndex: number) => {
    if (!quizAnswers[quizIndex]) {
      toast({
        title: "Select an Answer",
        description: "Please select an option before revealing the answer",
        variant: "destructive",
      });
      return;
    }
    setQuizRevealed(prev => ({ ...prev, [quizIndex]: true }));
  };

  const switchToInteractiveLab = () => {
    setShowLessonModal(false);
    setActiveTab("interactive");
  };

  const handleGenerateLabComposition = async () => {
    if (!labOrigin || !labDestination) return;
    
    setIsGeneratingLab(true);
    
    try {
      // Import music generation API
      const { generateMusic } = await import('@/lib/api/music');
      
      const response = await generateMusic({
        origin: labOrigin,
        destination: labDestination,
        music_style: 'major',
        tempo: 120,
      });

      if (response.data) {
        setLabComposition(response.data);
        
        toast({
          title: "Route Generated!",
          description: `Created musical route from ${labOrigin} to ${labDestination} with ${response.data.music.noteCount} notes`,
        });
      } else if (response.error) {
        throw new Error(response.error.message);
      }
    } catch (error) {
      console.error('Lab composition error:', error);
      toast({
        title: "Generation Failed",
        description: error instanceof Error ? error.message : "Could not generate route",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingLab(false);
    }
  };

  const handlePlayLabComposition = async () => {
    if (!labComposition?.music?.notes) {
      toast({
        title: "No Composition",
        description: "Please generate a route first",
        variant: "destructive",
      });
      return;
    }

    if (isPlayingLab) {
      const { audioPlayer } = await import('@/lib/audioPlayer');
      audioPlayer.stop();
      setIsPlayingLab(false);
    } else {
      setIsPlayingLab(true);
      
      try {
        const { audioPlayer } = await import('@/lib/audioPlayer');
        
        await audioPlayer.playComposition(
          labComposition.music.notes,
          labComposition.music.tempo,
          (progress) => {
            console.log('Lab playback progress:', progress);
          }
        );
        
        setIsPlayingLab(false);
        
        toast({
          title: "Playback Complete",
          description: `Played ${labComposition.music.noteCount} notes`,
        });
      } catch (error) {
        console.error('Lab playback error:', error);
        setIsPlayingLab(false);
        
        toast({
          title: "Playback Error",
          description: error instanceof Error ? error.message : "Could not play audio",
          variant: "destructive",
        });
      }
    }
  };

  return (
    <div className="min-h-screen pt-20 py-24 px-4">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-12">
          <BookOpen className="w-16 h-16 mx-auto mb-4 text-primary" />
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Educational Platform
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Teach geography and graph theory with sound-based visualization
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2">
            <TabsTrigger value="lessons">Lessons</TabsTrigger>
            <TabsTrigger value="interactive">Interactive Lab</TabsTrigger>
          </TabsList>

          <TabsContent value="lessons" className="space-y-6">
            {isLoading ? (
              <div className="flex justify-center items-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            ) : (
              <>
                <div className="grid gap-6 md:grid-cols-3">
                  {lessons.map((lesson) => {
                    const Icon = lesson.icon;
                    return (
                      <Card key={lesson.id} className="cursor-pointer hover:shadow-lg transition-shadow">
                        <CardHeader>
                          <Icon className="w-12 h-12 mb-4 text-primary" />
                          <CardTitle>{lesson.title}</CardTitle>
                          <CardDescription>{lesson.description}</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <Button 
                            onClick={() => handleStartLesson(lesson.id)}
                            className="w-full"
                            disabled={isStarting && selectedLesson === lesson.id}
                          >
                            {isStarting && selectedLesson === lesson.id ? (
                              <>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                Loading...
                              </>
                            ) : (
                              'Start Lesson'
                            )}
                          </Button>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>

                {/* Lesson Content Display */}
                {showLessonModal && lessonContent && (
                  <Card className="mt-8 border-2 border-primary">
                    <CardHeader>
                      <div className="flex justify-between items-start">
                        <div>
                          <CardTitle className="text-2xl">{lessonContent.title}</CardTitle>
                          <CardDescription className="mt-2">
                            {lessonContent.content.introduction}
                          </CardDescription>
                        </div>
                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => setShowLessonModal(false)}
                        >
                          Close
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Key Concepts */}
                      {lessonContent.content.key_concepts && (
                        <div className="p-4 bg-primary/5 rounded-lg border border-primary/20">
                          <h3 className="font-semibold mb-3 flex items-center gap-2">
                            <BookOpen className="w-5 h-5 text-primary" />
                            Key Concepts
                          </h3>
                          <ul className="space-y-2">
                            {lessonContent.content.key_concepts.map((concept: string, idx: number) => (
                              <li key={idx} className="text-sm flex items-start gap-2">
                                <span className="text-primary mt-1">•</span>
                                <span>{concept}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Examples */}
                      {lessonContent.content.examples && (
                        <div>
                          <h3 className="font-semibold mb-3">Real-World Examples</h3>
                          <div className="grid gap-4 md:grid-cols-2">
                            {lessonContent.content.examples.slice(0, 4).map((example: any, idx: number) => (
                              <div key={idx} className="p-4 border rounded-lg bg-background">
                                {example.route && (
                                  <p className="font-semibold text-primary mb-2">{example.route}</p>
                                )}
                                {example.scale && (
                                  <p className="font-semibold text-primary mb-2">{example.scale}</p>
                                )}
                                {example.airport && (
                                  <p className="font-semibold text-primary mb-2">{example.airport}</p>
                                )}
                                {example.distance_km && (
                                  <p className="text-sm text-muted-foreground">Distance: {example.distance_km.toFixed(0)} km</p>
                                )}
                                {example.connections && (
                                  <p className="text-sm text-muted-foreground">Connections: {example.connections}</p>
                                )}
                                {example.learning_point && (
                                  <p className="text-sm mt-2">{example.learning_point}</p>
                                )}
                                {example.mood && (
                                  <p className="text-sm mt-2">Mood: {example.mood}</p>
                                )}
                                {example.musical_representation && (
                                  <p className="text-sm mt-2">{example.musical_representation}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Dataset Facts */}
                      {lessonContent.content.dataset_facts && (
                        <div className="p-4 bg-secondary/20 rounded-lg">
                          <h3 className="font-semibold mb-3">📊 OpenFlights Dataset Facts</h3>
                          <ul className="space-y-2">
                            {lessonContent.content.dataset_facts.map((fact: string, idx: number) => (
                              <li key={idx} className="text-sm flex items-start gap-2">
                                <span className="text-primary mt-1">✓</span>
                                <span>{fact}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Interactive Elements */}
                      {lessonContent.interactive_elements && (
                        <div>
                          <h3 className="font-semibold mb-3">🎯 Interactive Activities</h3>
                          <div className="space-y-3">
                            {lessonContent.interactive_elements.map((element: any, idx: number) => (
                              <div key={idx} className="p-4 border rounded-lg hover:border-primary transition-colors">
                                <p className="font-medium mb-1">{element.type.replace(/_/g, ' ').toUpperCase()}</p>
                                <p className="text-sm text-muted-foreground mb-2">{element.description}</p>
                                {element.action && !element.question && (
                                  <Button 
                                    variant="outline" 
                                    size="sm" 
                                    className="mt-2"
                                    onClick={switchToInteractiveLab}
                                  >
                                    {element.action}
                                  </Button>
                                )}
                                {element.question && (
                                  <div className="mt-3 p-4 bg-primary/5 rounded-lg border border-primary/20">
                                    <p className="text-sm font-semibold mb-3">📝 Quiz: {element.question}</p>
                                    <div className="space-y-2 mb-3">
                                      {element.options?.map((option: string, optIdx: number) => {
                                        const isSelected = quizAnswers[idx] === option;
                                        const isCorrect = element.answer === option;
                                        const showResult = quizRevealed[idx];
                                        
                                        return (
                                          <button
                                            key={optIdx}
                                            onClick={() => handleQuizAnswer(idx, option)}
                                            disabled={quizRevealed[idx]}
                                            className={`w-full text-left p-3 rounded-lg border-2 transition-all ${
                                              isSelected && !showResult
                                                ? 'border-primary bg-primary/10'
                                                : showResult && isCorrect
                                                ? 'border-green-500 bg-green-50 dark:bg-green-950'
                                                : showResult && isSelected && !isCorrect
                                                ? 'border-red-500 bg-red-50 dark:bg-red-950'
                                                : 'border-border hover:border-primary/50'
                                            } ${quizRevealed[idx] ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                                          >
                                            <div className="flex items-center gap-2">
                                              <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                                                isSelected && !showResult
                                                  ? 'border-primary bg-primary'
                                                  : showResult && isCorrect
                                                  ? 'border-green-500 bg-green-500'
                                                  : showResult && isSelected && !isCorrect
                                                  ? 'border-red-500 bg-red-500'
                                                  : 'border-border'
                                              }`}>
                                                {isSelected && !showResult && (
                                                  <div className="w-2 h-2 rounded-full bg-white" />
                                                )}
                                                {showResult && isCorrect && (
                                                  <span className="text-white text-xs">✓</span>
                                                )}
                                                {showResult && isSelected && !isCorrect && (
                                                  <span className="text-white text-xs">✗</span>
                                                )}
                                              </div>
                                              <span className="text-sm">{option}</span>
                                            </div>
                                          </button>
                                        );
                                      })}
                                    </div>
                                    
                                    {!quizRevealed[idx] ? (
                                      <Button 
                                        size="sm" 
                                        onClick={() => handleRevealAnswer(idx)}
                                        disabled={!quizAnswers[idx]}
                                        className="w-full"
                                      >
                                        Check Answer
                                      </Button>
                                    ) : (
                                      <div className={`p-3 rounded-lg ${
                                        quizAnswers[idx] === element.answer
                                          ? 'bg-green-50 dark:bg-green-950 border border-green-500'
                                          : 'bg-red-50 dark:bg-red-950 border border-red-500'
                                      }`}>
                                        <p className={`text-sm font-semibold mb-1 ${
                                          quizAnswers[idx] === element.answer
                                            ? 'text-green-700 dark:text-green-300'
                                            : 'text-red-700 dark:text-red-300'
                                        }`}>
                                          {quizAnswers[idx] === element.answer ? '✓ Correct!' : '✗ Incorrect'}
                                        </p>
                                        {element.answer && (
                                          <p className="text-sm font-medium mb-1">
                                            Correct Answer: {element.answer}
                                          </p>
                                        )}
                                        {element.explanation && (
                                          <p className="text-sm text-muted-foreground">
                                            {element.explanation}
                                          </p>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Call to Action */}
                      <div className="flex gap-4 pt-4">
                        <Button 
                          className="flex-1"
                          onClick={switchToInteractiveLab}
                        >
                          Try Interactive Lab
                        </Button>
                        <Button 
                          variant="outline"
                          onClick={() => setShowLessonModal(false)}
                        >
                          Close Lesson
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            )}
          </TabsContent>

          <TabsContent value="interactive" className="space-y-6">
            {/* Interactive Route Builder */}
            <Card>
              <CardHeader>
                <CardTitle>Interactive Learning Lab</CardTitle>
                <CardDescription>
                  Experiment with routes and hear how geography translates to music in real-time
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Route Selection */}
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Origin Airport</label>
                    <select
                      className="w-full p-2 border rounded-md bg-background"
                      value={labOrigin}
                      onChange={(e) => setLabOrigin(e.target.value)}
                    >
                      <option value="">Select origin...</option>
                      <option value="JFK">JFK - New York</option>
                      <option value="LAX">LAX - Los Angeles</option>
                      <option value="LHR">LHR - London</option>
                      <option value="CDG">CDG - Paris</option>
                      <option value="NRT">NRT - Tokyo</option>
                      <option value="DXB">DXB - Dubai</option>
                      <option value="SYD">SYD - Sydney</option>
                    </select>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Destination Airport</label>
                    <select
                      className="w-full p-2 border rounded-md bg-background"
                      value={labDestination}
                      onChange={(e) => setLabDestination(e.target.value)}
                    >
                      <option value="">Select destination...</option>
                      <option value="JFK">JFK - New York</option>
                      <option value="LAX">LAX - Los Angeles</option>
                      <option value="LHR">LHR - London</option>
                      <option value="CDG">CDG - Paris</option>
                      <option value="NRT">NRT - Tokyo</option>
                      <option value="DXB">DXB - Dubai</option>
                      <option value="SYD">SYD - Sydney</option>
                    </select>
                  </div>
                </div>

                {/* Generate Button */}
                <Button 
                  className="w-full" 
                  size="lg"
                  onClick={handleGenerateLabComposition}
                  disabled={!labOrigin || !labDestination || isGeneratingLab}
                >
                  {isGeneratingLab ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Generating Musical Route...
                    </>
                  ) : (
                    <>
                      <Music className="w-4 h-4 mr-2" />
                      Generate & Visualize Route
                    </>
                  )}
                </Button>

                {/* Results Display */}
                {labComposition && (
                  <div className="space-y-4 p-4 border rounded-lg bg-secondary/20">
                    <div className="flex justify-between items-center">
                      <h3 className="font-semibold">
                        Route: {labOrigin} → {labDestination}
                      </h3>
                      <Button
                        size="sm"
                        onClick={handlePlayLabComposition}
                        disabled={isPlayingLab}
                      >
                        {isPlayingLab ? (
                          <>
                            <Pause className="w-4 h-4 mr-2" />
                            Stop
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 mr-2" />
                            Play
                          </>
                        )}
                      </Button>
                    </div>

                    {/* Educational Insights */}
                    <div className="grid md:grid-cols-3 gap-4">
                      <div className="p-3 bg-background rounded-lg">
                        <p className="text-xs text-muted-foreground">Distance</p>
                        <p className="text-lg font-bold text-primary">
                          {labComposition.route.distance.toFixed(0)} km
                        </p>
                      </div>
                      <div className="p-3 bg-background rounded-lg">
                        <p className="text-xs text-muted-foreground">Musical Notes</p>
                        <p className="text-lg font-bold text-primary">
                          {labComposition.music.noteCount}
                        </p>
                      </div>
                      <div className="p-3 bg-background rounded-lg">
                        <p className="text-xs text-muted-foreground">Duration</p>
                        <p className="text-lg font-bold text-primary">
                          {labComposition.music.duration.toFixed(1)}s
                        </p>
                      </div>
                    </div>

                    {/* Learning Points */}
                    <div className="p-4 bg-primary/5 rounded-lg border border-primary/20">
                      <h4 className="font-semibold mb-2 flex items-center gap-2">
                        <BookOpen className="w-4 h-4" />
                        What You're Learning:
                      </h4>
                      <ul className="space-y-2 text-sm">
                        <li>• <strong>Geography:</strong> Distance between {labComposition.route.origin.city} and {labComposition.route.destination.city}</li>
                        <li>• <strong>Graph Theory:</strong> Shortest path algorithm finds optimal route</li>
                        <li>• <strong>Music Theory:</strong> Distance maps to note count, creating unique melodies</li>
                        <li>• <strong>Data Visualization:</strong> Flight routes become audible patterns</li>
                      </ul>
                    </div>

                    {/* Route Details */}
                    <div className="text-sm text-muted-foreground">
                      <p><strong>From:</strong> {labComposition.route.origin.name}, {labComposition.route.origin.city}, {labComposition.route.origin.country}</p>
                      <p><strong>To:</strong> {labComposition.route.destination.name}, {labComposition.route.destination.city}, {labComposition.route.destination.country}</p>
                      <p><strong>Musical Scale:</strong> {labComposition.music.scale || 'Major'}</p>
                      <p><strong>Tempo:</strong> {labComposition.music.tempo} BPM</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Educational Tips */}
            <Card>
              <CardHeader>
                <CardTitle>Try These Experiments</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-semibold mb-2">🌍 Short vs Long Routes</h4>
                    <p className="text-sm text-muted-foreground">
                      Compare JFK→LAX (short) with JFK→SYD (long). Notice how longer distances create more complex melodies with more notes.
                    </p>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-semibold mb-2">🎵 East vs West Travel</h4>
                    <p className="text-sm text-muted-foreground">
                      Try LHR→NRT (eastward) vs LAX→LHR (westward). Different directions create different musical patterns.
                    </p>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-semibold mb-2">🔄 Reverse Routes</h4>
                    <p className="text-sm text-muted-foreground">
                      Generate JFK→CDG, then CDG→JFK. Same distance, but different musical interpretations!
                    </p>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-semibold mb-2">🌐 Cross-Hemisphere</h4>
                    <p className="text-sm text-muted-foreground">
                      Try routes crossing hemispheres like LHR→SYD. Latitude changes affect the musical scale used.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Education;
