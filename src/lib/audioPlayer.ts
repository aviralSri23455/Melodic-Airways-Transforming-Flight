// Web Audio API-based MIDI player
export class AudioPlayer {
  private audioContext: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private isPlaying = false;
  private scheduledNotes: number[] = [];

  constructor() {
    if (typeof window !== 'undefined') {
      try {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.gainNode = this.audioContext.createGain();
        this.gainNode.connect(this.audioContext.destination);
        this.gainNode.gain.value = 0.3;
        
        console.log('AudioContext initialized with state:', this.audioContext.state);
      } catch (error) {
        console.error('Failed to initialize AudioContext:', error);
      }
    }
  }

  // Convert MIDI note number to frequency
  private midiToFreq(note: number): number {
    return 440 * Math.pow(2, (note - 69) / 12);
  }

  // Play a single note with absolute timing
  private playNoteAbsolute(frequency: number, duration: number, velocity: number, absoluteStartTime: number) {
    if (!this.audioContext || !this.gainNode) return;

    const oscillator = this.audioContext.createOscillator();
    const noteGain = this.audioContext.createGain();

    oscillator.connect(noteGain);
    noteGain.connect(this.gainNode);

    // Use a more pleasant waveform
    oscillator.type = 'sine';
    oscillator.frequency.value = frequency;

    // ADSR envelope with absolute timing
    const normalizedVelocity = velocity / 127;
    
    noteGain.gain.setValueAtTime(0, absoluteStartTime);
    noteGain.gain.linearRampToValueAtTime(normalizedVelocity * 0.5, absoluteStartTime + 0.05); // Attack
    noteGain.gain.exponentialRampToValueAtTime(normalizedVelocity * 0.3, absoluteStartTime + duration * 0.5); // Decay
    noteGain.gain.exponentialRampToValueAtTime(0.01, absoluteStartTime + duration); // Release

    oscillator.start(absoluteStartTime);
    oscillator.stop(absoluteStartTime + duration);

    return oscillator;
  }

  // Play a single note
  private playNote(frequency: number, duration: number, velocity: number, startTime: number) {
    if (!this.audioContext || !this.gainNode) return;

    const oscillator = this.audioContext.createOscillator();
    const noteGain = this.audioContext.createGain();

    oscillator.connect(noteGain);
    noteGain.connect(this.gainNode);

    // Use a more pleasant waveform
    oscillator.type = 'sine';
    oscillator.frequency.value = frequency;

    // ADSR envelope
    const now = this.audioContext.currentTime + startTime;
    const normalizedVelocity = velocity / 127;
    
    noteGain.gain.setValueAtTime(0, now);
    noteGain.gain.linearRampToValueAtTime(normalizedVelocity * 0.5, now + 0.05); // Attack
    noteGain.gain.exponentialRampToValueAtTime(normalizedVelocity * 0.3, now + duration * 0.5); // Decay
    noteGain.gain.exponentialRampToValueAtTime(0.01, now + duration); // Release

    oscillator.start(now);
    oscillator.stop(now + duration);

    return oscillator;
  }

  // Ensure audio context is ready for playback
  async ensureAudioContextReady(): Promise<boolean> {
    if (!this.audioContext) {
      console.error('AudioContext not initialized');
      return false;
    }

    try {
      if (this.audioContext.state === 'suspended') {
        console.log('Resuming suspended AudioContext...');
        await this.audioContext.resume();
        console.log('AudioContext resumed successfully');
      }
      return this.audioContext.state === 'running';
    } catch (error) {
      console.error('Failed to resume AudioContext:', error);
      return false;
    }
  }

  async playComposition(notes: any[], tempo: number, onProgress?: (progress: number) => void): Promise<void> {
    if (!this.audioContext) {
      throw new Error('AudioContext not initialized');
    }

    // Ensure audio context is ready before starting
    const isReady = await this.ensureAudioContextReady();
    if (!isReady) {
      throw new Error('Could not start audio playback - AudioContext not ready');
    }

    this.stop(); // Stop any currently playing composition
    this.isPlaying = true;

    const ticksPerSecond = (tempo / 60) * 480; // 480 ticks = quarter note
    const maxTime = Math.max(...notes.map(n => n.time + n.duration));
    let totalDuration = maxTime / ticksPerSecond;

    // Store the start time for progress tracking
    const compositionStartTime = this.audioContext.currentTime;

    // Safety check: prevent extremely short compositions
    if (totalDuration < 2.0) {
      console.warn(`Composition too short (${totalDuration.toFixed(2)}s), extending to minimum 30 seconds`);
      // Extend notes proportionally to reach minimum duration
      const extensionFactor = 30.0 / totalDuration;
      notes.forEach(note => {
        note.time *= extensionFactor;
        note.duration *= extensionFactor;
      });
      totalDuration = 30.0; // Minimum 30 seconds for VR experience
    } else if (totalDuration < 10.0) {
      console.warn(`Composition is short (${totalDuration.toFixed(2)}s), but acceptable for testing`);
    }

    console.log(`Starting composition: ${notes.length} notes, ${totalDuration.toFixed(2)}s duration`);

    // Schedule all notes with absolute timing
    notes.forEach(note => {
      const startTime = compositionStartTime + (note.time / ticksPerSecond);
      const duration = note.duration / ticksPerSecond;
      const frequency = this.midiToFreq(note.note);

      if (this.isPlaying) {
        const oscillator = this.playNoteAbsolute(frequency, duration, note.velocity, startTime);
        if (oscillator) {
          this.scheduledNotes.push(oscillator as any);
        }
      }
    });

    // Track progress based on actual audio context time
    if (onProgress) {
      const progressInterval = setInterval(() => {
        if (!this.isPlaying || !this.audioContext) {
          clearInterval(progressInterval);
          return;
        }
        
        const elapsed = this.audioContext.currentTime - compositionStartTime;
        const progress = (elapsed / totalDuration) * 100;
        onProgress(Math.min(Math.max(progress, 0), 100));
        
        if (elapsed >= totalDuration) {
          clearInterval(progressInterval);
        }
      }, 50); // Update more frequently for smoother progress
    }

    // Wait for actual completion based on audio context time
    return new Promise((resolve) => {
      const checkCompletion = () => {
        if (!this.isPlaying || !this.audioContext) {
          resolve();
          return;
        }
        
        const elapsed = this.audioContext.currentTime - compositionStartTime;
        if (elapsed >= totalDuration) {
          console.log('Composition playback completed');
          this.isPlaying = false;
          this.scheduledNotes = [];
          resolve();
        } else {
          // Check again in 100ms
          setTimeout(checkCompletion, 100);
        }
      };
      
      // Start checking after the total duration plus a small buffer
      setTimeout(checkCompletion, totalDuration * 1000 + 100);
    });
  }

  stop() {
    this.isPlaying = false;
    this.scheduledNotes.forEach(note => {
      try {
        (note as any).stop();
      } catch (e) {
        // Note already stopped
      }
    });
    this.scheduledNotes = [];
  }

  getIsPlaying(): boolean {
    return this.isPlaying;
  }

  dispose() {
    this.stop();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

// Export singleton instance
export const audioPlayer = new AudioPlayer();
