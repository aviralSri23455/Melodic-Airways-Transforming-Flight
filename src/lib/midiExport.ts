// MIDI file export functionality
export const exportToMidi = (composition: any, routeName: string) => {
  if (!composition?.midi_data?.notes) {
    throw new Error('No MIDI data available');
  }

  const notes = composition.midi_data.notes;
  const tempo = composition.tempo || 120;
  
  // MIDI file structure
  const header = createMidiHeader();
  const tempoTrack = createTempoTrack(tempo);
  const noteTrack = createNoteTrack(notes);
  
  // Combine all parts
  const midiData = new Uint8Array([...header, ...tempoTrack, ...noteTrack]);
  
  // Create blob and download
  const blob = new Blob([midiData], { type: 'audio/midi' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${routeName.replace(/\s+/g, '_')}_composition.mid`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

function createMidiHeader(): number[] {
  return [
    0x4D, 0x54, 0x68, 0x64, // "MThd"
    0x00, 0x00, 0x00, 0x06, // Header length
    0x00, 0x01, // Format type 1
    0x00, 0x02, // Number of tracks
    0x01, 0xE0  // Ticks per quarter note (480)
  ];
}

function createTempoTrack(tempo: number): number[] {
  const microsecondsPerBeat = Math.round(60000000 / tempo);
  
  return [
    0x4D, 0x54, 0x72, 0x6B, // "MTrk"
    0x00, 0x00, 0x00, 0x0B, // Track length
    0x00, 0xFF, 0x51, 0x03, // Tempo meta event
    (microsecondsPerBeat >> 16) & 0xFF,
    (microsecondsPerBeat >> 8) & 0xFF,
    microsecondsPerBeat & 0xFF,
    0x00, 0xFF, 0x2F, 0x00  // End of track
  ];
}

function createNoteTrack(notes: any[]): number[] {
  const trackData: number[] = [0x4D, 0x54, 0x72, 0x6B]; // "MTrk"
  const events: number[] = [];
  
  let currentTime = 0;
  
  notes.forEach((note) => {
    const startTick = Math.round(note.time * 480);
    const duration = Math.round(note.duration * 480);
    const deltaTime = startTick - currentTime;
    
    // Note on event
    events.push(...encodeVariableLength(deltaTime));
    events.push(0x90); // Note on, channel 0
    events.push(note.midi || 60); // Note number
    events.push(Math.round((note.velocity || 0.8) * 127)); // Velocity
    
    // Note off event
    events.push(...encodeVariableLength(duration));
    events.push(0x80); // Note off, channel 0
    events.push(note.midi || 60);
    events.push(0x40); // Release velocity
    
    currentTime = startTick + duration;
  });
  
  // End of track
  events.push(0x00, 0xFF, 0x2F, 0x00);
  
  // Add track length
  const length = events.length;
  trackData.push(
    (length >> 24) & 0xFF,
    (length >> 16) & 0xFF,
    (length >> 8) & 0xFF,
    length & 0xFF
  );
  
  return [...trackData, ...events];
}

function encodeVariableLength(value: number): number[] {
  const bytes: number[] = [];
  bytes.push(value & 0x7F);
  
  value >>= 7;
  while (value > 0) {
    bytes.unshift((value & 0x7F) | 0x80);
    value >>= 7;
  }
  
  return bytes;
}
