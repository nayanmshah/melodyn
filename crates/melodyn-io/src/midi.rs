//! MIDI file I/O: reading and writing Standard MIDI Files (format 0 and 1).
//!
//! Based on MUMidiComponent, MUMidiIOConfiguration, MUMidiSynthesizer,
//! MUDocumentMidiImporter, MUPerformanceMidiExporter. Implements full binary
//! MIDI parsing including variable-length quantities, track chunks, tempo maps,
//! and conversion to/from DetectedNote.

use std::io::{self, Write};

// ---------------------------------------------------------------------------
// MIDI event types
// ---------------------------------------------------------------------------

/// MIDI channel voice message types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiMessage {
    NoteOff { channel: u8, note: u8, velocity: u8 },
    NoteOn { channel: u8, note: u8, velocity: u8 },
    PolyAftertouch { channel: u8, note: u8, pressure: u8 },
    ControlChange { channel: u8, controller: u8, value: u8 },
    ProgramChange { channel: u8, program: u8 },
    ChannelAftertouch { channel: u8, pressure: u8 },
    PitchBend { channel: u8, value: u16 },
    /// Meta events (tempo, time signature, key signature, text, etc.)
    Meta { meta_type: u8, data: Vec<u8> },
    /// System exclusive
    SysEx { data: Vec<u8> },
}

impl MidiMessage {
    /// Whether this is a note-on with velocity > 0.
    pub fn is_note_on(&self) -> bool {
        matches!(self, MidiMessage::NoteOn { velocity, .. } if *velocity > 0)
    }

    /// Whether this is a note-off (or note-on with velocity 0).
    pub fn is_note_off(&self) -> bool {
        matches!(self, MidiMessage::NoteOff { .. })
            || matches!(self, MidiMessage::NoteOn { velocity: 0, .. })
    }

    /// Get the MIDI channel (0-15) if applicable.
    pub fn channel(&self) -> Option<u8> {
        match self {
            MidiMessage::NoteOff { channel, .. }
            | MidiMessage::NoteOn { channel, .. }
            | MidiMessage::PolyAftertouch { channel, .. }
            | MidiMessage::ControlChange { channel, .. }
            | MidiMessage::ProgramChange { channel, .. }
            | MidiMessage::ChannelAftertouch { channel, .. }
            | MidiMessage::PitchBend { channel, .. } => Some(*channel),
            _ => None,
        }
    }
}

/// A MIDI event with timing.
#[derive(Debug, Clone)]
pub struct MidiEvent {
    /// Absolute time in ticks from the start of the track.
    pub tick: u64,
    /// The MIDI message.
    pub message: MidiMessage,
}

// ---------------------------------------------------------------------------
// MIDI track
// ---------------------------------------------------------------------------

/// A MIDI track containing an ordered sequence of events.
#[derive(Debug, Clone)]
pub struct MidiTrack {
    pub name: String,
    pub events: Vec<MidiEvent>,
}

impl MidiTrack {
    pub fn new(name: String) -> Self {
        Self {
            name,
            events: Vec::new(),
        }
    }

    /// Add an event, maintaining sorted order by tick.
    pub fn add_event(&mut self, event: MidiEvent) {
        let pos = self
            .events
            .binary_search_by_key(&event.tick, |e| e.tick)
            .unwrap_or_else(|p| p);
        self.events.insert(pos, event);
    }

    /// Get all note-on events.
    pub fn note_events(&self) -> Vec<&MidiEvent> {
        self.events.iter().filter(|e| e.message.is_note_on()).collect()
    }

    /// Duration in ticks.
    pub fn duration_ticks(&self) -> u64 {
        self.events.last().map(|e| e.tick).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Tempo map
// ---------------------------------------------------------------------------

/// A tempo change at a specific tick position.
#[derive(Debug, Clone)]
pub struct MidiTempoChange {
    pub tick: u64,
    pub microseconds_per_quarter: u32,
}

impl MidiTempoChange {
    pub fn bpm(&self) -> f64 {
        60_000_000.0 / self.microseconds_per_quarter as f64
    }
}

/// Tempo map for tick-to-time conversion.
#[derive(Debug, Clone)]
pub struct TempoMap {
    pub ticks_per_quarter: u16,
    pub changes: Vec<MidiTempoChange>,
}

impl TempoMap {
    pub fn new(ticks_per_quarter: u16) -> Self {
        Self {
            ticks_per_quarter,
            changes: vec![MidiTempoChange {
                tick: 0,
                microseconds_per_quarter: 500_000, // 120 BPM
            }],
        }
    }

    /// Convert tick position to seconds.
    pub fn tick_to_seconds(&self, tick: u64) -> f64 {
        let tpq = self.ticks_per_quarter as f64;
        let mut seconds = 0.0;
        let mut prev_tick = 0u64;
        let mut prev_uspq = self.changes[0].microseconds_per_quarter;

        for change in self.changes.iter().skip(1) {
            if change.tick >= tick {
                break;
            }
            let delta_ticks = change.tick - prev_tick;
            seconds += delta_ticks as f64 * prev_uspq as f64 / (tpq * 1_000_000.0);
            prev_tick = change.tick;
            prev_uspq = change.microseconds_per_quarter;
        }

        let remaining_ticks = tick - prev_tick;
        seconds += remaining_ticks as f64 * prev_uspq as f64 / (tpq * 1_000_000.0);
        seconds
    }

    /// Convert seconds to tick position.
    pub fn seconds_to_tick(&self, seconds: f64) -> u64 {
        let tpq = self.ticks_per_quarter as f64;
        let mut accumulated_seconds = 0.0;
        let mut prev_tick = 0u64;
        let mut prev_uspq = self.changes[0].microseconds_per_quarter;

        for change in self.changes.iter().skip(1) {
            let delta_ticks = change.tick - prev_tick;
            let segment_seconds = delta_ticks as f64 * prev_uspq as f64 / (tpq * 1_000_000.0);
            if accumulated_seconds + segment_seconds >= seconds {
                let remaining = seconds - accumulated_seconds;
                let ticks = remaining * tpq * 1_000_000.0 / prev_uspq as f64;
                return prev_tick + ticks as u64;
            }
            accumulated_seconds += segment_seconds;
            prev_tick = change.tick;
            prev_uspq = change.microseconds_per_quarter;
        }

        let remaining = seconds - accumulated_seconds;
        let ticks = remaining * tpq * 1_000_000.0 / prev_uspq as f64;
        prev_tick + ticks as u64
    }

    /// Add a tempo change.
    pub fn add_tempo_change(&mut self, tick: u64, microseconds_per_quarter: u32) {
        let pos = self
            .changes
            .binary_search_by_key(&tick, |c| c.tick)
            .unwrap_or_else(|p| p);
        self.changes.insert(
            pos,
            MidiTempoChange {
                tick,
                microseconds_per_quarter,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// MIDI file
// ---------------------------------------------------------------------------

/// MIDI file format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiFormat {
    /// Format 0: single track.
    SingleTrack,
    /// Format 1: multiple tracks, synchronous.
    MultiTrack,
}

/// A complete MIDI file.
#[derive(Debug, Clone)]
pub struct MidiFile {
    pub format: MidiFormat,
    pub ticks_per_quarter: u16,
    pub tracks: Vec<MidiTrack>,
}

impl MidiFile {
    pub fn new(format: MidiFormat, ticks_per_quarter: u16) -> Self {
        Self {
            format,
            ticks_per_quarter,
            tracks: Vec::new(),
        }
    }

    /// Build a tempo map from all meta events in the file.
    pub fn tempo_map(&self) -> TempoMap {
        let mut map = TempoMap::new(self.ticks_per_quarter);
        map.changes.clear();
        map.changes.push(MidiTempoChange {
            tick: 0,
            microseconds_per_quarter: 500_000,
        });

        for track in &self.tracks {
            for event in &track.events {
                if let MidiMessage::Meta {
                    meta_type: 0x51,
                    ref data,
                } = event.message
                {
                    if data.len() == 3 {
                        let uspq =
                            ((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32);
                        map.add_tempo_change(event.tick, uspq);
                    }
                }
            }
        }
        map
    }

    /// Total duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        let tempo_map = self.tempo_map();
        let max_tick = self
            .tracks
            .iter()
            .map(|t| t.duration_ticks())
            .max()
            .unwrap_or(0);
        tempo_map.tick_to_seconds(max_tick)
    }
}

// ---------------------------------------------------------------------------
// Variable-length quantity encoding/decoding
// ---------------------------------------------------------------------------

/// Read a variable-length quantity from a byte slice.
/// Returns (value, bytes_consumed).
fn read_vlq(data: &[u8]) -> Option<(u32, usize)> {
    let mut value: u32 = 0;
    let mut bytes_read = 0;
    for &byte in data.iter().take(4) {
        bytes_read += 1;
        value = (value << 7) | (byte & 0x7F) as u32;
        if byte & 0x80 == 0 {
            return Some((value, bytes_read));
        }
    }
    None
}

/// Write a variable-length quantity to a writer.
fn write_vlq<W: Write>(writer: &mut W, mut value: u32) -> io::Result<usize> {
    let mut buffer = [0u8; 4];
    let mut pos = 3;
    buffer[pos] = (value & 0x7F) as u8;
    value >>= 7;
    while value > 0 {
        pos -= 1;
        buffer[pos] = (value & 0x7F) as u8 | 0x80;
        value >>= 7;
    }
    let bytes = &buffer[pos..4];
    writer.write_all(bytes)?;
    Ok(bytes.len())
}

/// Encode a VLQ to bytes.
fn vlq_bytes(value: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    write_vlq(&mut buf, value).unwrap();
    buf
}

// ---------------------------------------------------------------------------
// MIDI file reader
// ---------------------------------------------------------------------------

/// Error type for MIDI parsing.
#[derive(Debug)]
pub enum MidiError {
    InvalidHeader,
    InvalidTrack,
    UnexpectedEof,
    UnsupportedFormat(u16),
    IoError(io::Error),
}

impl From<io::Error> for MidiError {
    fn from(e: io::Error) -> Self {
        MidiError::IoError(e)
    }
}

impl std::fmt::Display for MidiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MidiError::InvalidHeader => write!(f, "Invalid MIDI header"),
            MidiError::InvalidTrack => write!(f, "Invalid MIDI track"),
            MidiError::UnexpectedEof => write!(f, "Unexpected end of MIDI data"),
            MidiError::UnsupportedFormat(fmt) => write!(f, "Unsupported MIDI format: {}", fmt),
            MidiError::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for MidiError {}

/// Parse a Standard MIDI File from bytes.
pub fn read_midi(data: &[u8]) -> Result<MidiFile, MidiError> {
    if data.len() < 14 {
        return Err(MidiError::InvalidHeader);
    }

    // Parse header chunk: "MThd" + length(4) + format(2) + ntrks(2) + division(2)
    if &data[0..4] != b"MThd" {
        return Err(MidiError::InvalidHeader);
    }
    let header_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if header_len < 6 || data.len() < 8 + header_len {
        return Err(MidiError::InvalidHeader);
    }

    let format_num = u16::from_be_bytes([data[8], data[9]]);
    let num_tracks = u16::from_be_bytes([data[10], data[11]]) as usize;
    let division = u16::from_be_bytes([data[12], data[13]]);

    let format = match format_num {
        0 => MidiFormat::SingleTrack,
        1 => MidiFormat::MultiTrack,
        f => return Err(MidiError::UnsupportedFormat(f)),
    };

    // Division: if bit 15 is 0, it is ticks per quarter note
    let ticks_per_quarter = if division & 0x8000 == 0 {
        division
    } else {
        // SMPTE: negative frames-per-second in upper byte, ticks-per-frame in lower
        let fps = -(((division >> 8) as i8) as i16);
        let tpf = division & 0xFF;
        // Convert to approximate ticks-per-quarter at 120 BPM
        (fps as u16) * tpf / 2
    };

    let mut midi_file = MidiFile::new(format, ticks_per_quarter);
    let mut pos = 8 + header_len;

    for _ in 0..num_tracks {
        if pos + 8 > data.len() {
            return Err(MidiError::UnexpectedEof);
        }
        if &data[pos..pos + 4] != b"MTrk" {
            return Err(MidiError::InvalidTrack);
        }
        let track_len =
            u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        pos += 8;

        if pos + track_len > data.len() {
            return Err(MidiError::UnexpectedEof);
        }

        let track_data = &data[pos..pos + track_len];
        let track = parse_track(track_data)?;
        midi_file.tracks.push(track);
        pos += track_len;
    }

    Ok(midi_file)
}

/// Parse a single MIDI track from its data bytes.
fn parse_track(data: &[u8]) -> Result<MidiTrack, MidiError> {
    let mut track = MidiTrack::new(String::new());
    let mut pos = 0;
    let mut absolute_tick: u64 = 0;
    let mut running_status: u8 = 0;

    while pos < data.len() {
        // Read delta time
        let (delta, vlq_len) = read_vlq(&data[pos..]).ok_or(MidiError::UnexpectedEof)?;
        pos += vlq_len;
        absolute_tick += delta as u64;

        if pos >= data.len() {
            break;
        }

        let status_byte = data[pos];

        // Meta event
        if status_byte == 0xFF {
            pos += 1;
            if pos >= data.len() {
                return Err(MidiError::UnexpectedEof);
            }
            let meta_type = data[pos];
            pos += 1;
            let (length, vlq_len) = read_vlq(&data[pos..]).ok_or(MidiError::UnexpectedEof)?;
            pos += vlq_len;
            let length = length as usize;
            if pos + length > data.len() {
                return Err(MidiError::UnexpectedEof);
            }
            let meta_data = data[pos..pos + length].to_vec();
            pos += length;

            // Extract track name from meta type 0x03
            if meta_type == 0x03 {
                track.name = String::from_utf8_lossy(&meta_data).to_string();
            }

            track.events.push(MidiEvent {
                tick: absolute_tick,
                message: MidiMessage::Meta {
                    meta_type,
                    data: meta_data,
                },
            });

            // End of track
            if meta_type == 0x2F {
                break;
            }
            continue;
        }

        // SysEx
        if status_byte == 0xF0 || status_byte == 0xF7 {
            pos += 1;
            let (length, vlq_len) = read_vlq(&data[pos..]).ok_or(MidiError::UnexpectedEof)?;
            pos += vlq_len;
            let length = length as usize;
            if pos + length > data.len() {
                return Err(MidiError::UnexpectedEof);
            }
            let sysex_data = data[pos..pos + length].to_vec();
            pos += length;
            track.events.push(MidiEvent {
                tick: absolute_tick,
                message: MidiMessage::SysEx { data: sysex_data },
            });
            continue;
        }

        // Channel voice messages
        let (status, data_start) = if status_byte & 0x80 != 0 {
            running_status = status_byte;
            pos += 1;
            (status_byte, pos)
        } else {
            // Running status
            (running_status, pos)
        };

        let msg_type = status & 0xF0;
        let channel = status & 0x0F;

        let message = match msg_type {
            0x80 => {
                // Note Off
                if data_start + 2 > data.len() {
                    return Err(MidiError::UnexpectedEof);
                }
                let note = data[data_start];
                let velocity = data[data_start + 1];
                pos = data_start + 2;
                MidiMessage::NoteOff {
                    channel,
                    note,
                    velocity,
                }
            }
            0x90 => {
                // Note On
                if data_start + 2 > data.len() {
                    return Err(MidiError::UnexpectedEof);
                }
                let note = data[data_start];
                let velocity = data[data_start + 1];
                pos = data_start + 2;
                MidiMessage::NoteOn {
                    channel,
                    note,
                    velocity,
                }
            }
            0xA0 => {
                // Polyphonic Aftertouch
                if data_start + 2 > data.len() {
                    return Err(MidiError::UnexpectedEof);
                }
                let note = data[data_start];
                let pressure = data[data_start + 1];
                pos = data_start + 2;
                MidiMessage::PolyAftertouch {
                    channel,
                    note,
                    pressure,
                }
            }
            0xB0 => {
                // Control Change
                if data_start + 2 > data.len() {
                    return Err(MidiError::UnexpectedEof);
                }
                let controller = data[data_start];
                let value = data[data_start + 1];
                pos = data_start + 2;
                MidiMessage::ControlChange {
                    channel,
                    controller,
                    value,
                }
            }
            0xC0 => {
                // Program Change
                if data_start + 1 > data.len() {
                    return Err(MidiError::UnexpectedEof);
                }
                let program = data[data_start];
                pos = data_start + 1;
                MidiMessage::ProgramChange { channel, program }
            }
            0xD0 => {
                // Channel Aftertouch
                if data_start + 1 > data.len() {
                    return Err(MidiError::UnexpectedEof);
                }
                let pressure = data[data_start];
                pos = data_start + 1;
                MidiMessage::ChannelAftertouch { channel, pressure }
            }
            0xE0 => {
                // Pitch Bend
                if data_start + 2 > data.len() {
                    return Err(MidiError::UnexpectedEof);
                }
                let lsb = data[data_start] as u16;
                let msb = data[data_start + 1] as u16;
                let value = (msb << 7) | lsb;
                pos = data_start + 2;
                MidiMessage::PitchBend { channel, value }
            }
            _ => {
                // Unknown message type, skip one byte
                pos = data_start + 1;
                continue;
            }
        };

        track.events.push(MidiEvent {
            tick: absolute_tick,
            message,
        });
    }

    Ok(track)
}

// ---------------------------------------------------------------------------
// MIDI file writer
// ---------------------------------------------------------------------------

/// Write a MIDI file to bytes.
pub fn write_midi(midi_file: &MidiFile) -> Result<Vec<u8>, MidiError> {
    let mut out = Vec::new();

    // Header chunk
    out.extend_from_slice(b"MThd");
    out.extend_from_slice(&6u32.to_be_bytes());
    let format_num: u16 = match midi_file.format {
        MidiFormat::SingleTrack => 0,
        MidiFormat::MultiTrack => 1,
    };
    out.extend_from_slice(&format_num.to_be_bytes());
    out.extend_from_slice(&(midi_file.tracks.len() as u16).to_be_bytes());
    out.extend_from_slice(&midi_file.ticks_per_quarter.to_be_bytes());

    // Track chunks
    for track in &midi_file.tracks {
        let track_data = encode_track(track)?;
        out.extend_from_slice(b"MTrk");
        out.extend_from_slice(&(track_data.len() as u32).to_be_bytes());
        out.extend_from_slice(&track_data);
    }

    Ok(out)
}

/// Encode a track's events to bytes.
fn encode_track(track: &MidiTrack) -> Result<Vec<u8>, MidiError> {
    let mut out = Vec::new();
    let mut prev_tick: u64 = 0;

    // Write track name meta event if present
    if !track.name.is_empty() {
        out.extend_from_slice(&vlq_bytes(0)); // delta = 0
        out.push(0xFF);
        out.push(0x03);
        let name_bytes = track.name.as_bytes();
        out.extend_from_slice(&vlq_bytes(name_bytes.len() as u32));
        out.extend_from_slice(name_bytes);
    }

    for event in &track.events {
        let delta = (event.tick - prev_tick) as u32;
        prev_tick = event.tick;
        out.extend_from_slice(&vlq_bytes(delta));

        match &event.message {
            MidiMessage::NoteOff {
                channel,
                note,
                velocity,
            } => {
                out.push(0x80 | channel);
                out.push(*note);
                out.push(*velocity);
            }
            MidiMessage::NoteOn {
                channel,
                note,
                velocity,
            } => {
                out.push(0x90 | channel);
                out.push(*note);
                out.push(*velocity);
            }
            MidiMessage::PolyAftertouch {
                channel,
                note,
                pressure,
            } => {
                out.push(0xA0 | channel);
                out.push(*note);
                out.push(*pressure);
            }
            MidiMessage::ControlChange {
                channel,
                controller,
                value,
            } => {
                out.push(0xB0 | channel);
                out.push(*controller);
                out.push(*value);
            }
            MidiMessage::ProgramChange { channel, program } => {
                out.push(0xC0 | channel);
                out.push(*program);
            }
            MidiMessage::ChannelAftertouch { channel, pressure } => {
                out.push(0xD0 | channel);
                out.push(*pressure);
            }
            MidiMessage::PitchBend { channel, value } => {
                out.push(0xE0 | channel);
                out.push((value & 0x7F) as u8);
                out.push(((value >> 7) & 0x7F) as u8);
            }
            MidiMessage::Meta { meta_type, data } => {
                out.push(0xFF);
                out.push(*meta_type);
                out.extend_from_slice(&vlq_bytes(data.len() as u32));
                out.extend_from_slice(data);
            }
            MidiMessage::SysEx { data } => {
                out.push(0xF0);
                out.extend_from_slice(&vlq_bytes(data.len() as u32));
                out.extend_from_slice(data);
            }
        }
    }

    // End of track meta event
    out.extend_from_slice(&vlq_bytes(0));
    out.push(0xFF);
    out.push(0x2F);
    out.push(0x00);

    Ok(out)
}

// ---------------------------------------------------------------------------
// Conversion: DetectedNote <-> MIDI
// ---------------------------------------------------------------------------

/// Convert detected notes to MIDI events on a track.
///
/// Maps each DetectedNote to a NoteOn/NoteOff pair using the tempo map
/// for tick timing.
pub fn notes_to_midi_track(
    notes: &[melodyn_core::DetectedNote],
    tempo_map: &TempoMap,
    channel: u8,
) -> MidiTrack {
    let mut track = MidiTrack::new("Melodyn Export".to_string());

    for note in notes {
        let start_tick = tempo_map.seconds_to_tick(note.start_time);
        let end_tick = tempo_map.seconds_to_tick(note.end_time());
        let midi_note = note.midi_note.round().clamp(0.0, 127.0) as u8;
        let velocity = (note.amplitude * 127.0).clamp(1.0, 127.0) as u8;

        track.add_event(MidiEvent {
            tick: start_tick,
            message: MidiMessage::NoteOn {
                channel,
                note: midi_note,
                velocity,
            },
        });
        track.add_event(MidiEvent {
            tick: end_tick,
            message: MidiMessage::NoteOff {
                channel,
                note: midi_note,
                velocity: 0,
            },
        });
    }

    track
}

/// Convert detected notes to a complete MIDI file.
pub fn notes_to_midi_file(
    notes: &[melodyn_core::DetectedNote],
    ticks_per_quarter: u16,
    bpm: f64,
) -> MidiFile {
    let mut file = MidiFile::new(MidiFormat::SingleTrack, ticks_per_quarter);
    let tempo_map = TempoMap::new(ticks_per_quarter);

    // Create tempo track with desired BPM
    let mut tempo_track = MidiTrack::new("Tempo".to_string());
    let uspq = (60_000_000.0 / bpm) as u32;
    tempo_track.add_event(MidiEvent {
        tick: 0,
        message: MidiMessage::Meta {
            meta_type: 0x51,
            data: vec![
                ((uspq >> 16) & 0xFF) as u8,
                ((uspq >> 8) & 0xFF) as u8,
                (uspq & 0xFF) as u8,
            ],
        },
    });

    // Add notes
    let note_track = notes_to_midi_track(notes, &tempo_map, 0);
    // Merge tempo events into note track for format 0
    let mut merged = tempo_track;
    for event in note_track.events {
        merged.add_event(event);
    }

    file.tracks.push(merged);
    file
}

/// Convert MIDI note events to DetectedNote representation.
///
/// Pairs NoteOn with matching NoteOff to create notes with start time,
/// duration, and other properties.
pub fn midi_to_notes(
    track: &MidiTrack,
    tempo_map: &TempoMap,
) -> Vec<melodyn_core::DetectedNote> {
    let mut notes = Vec::new();
    // Track pending note-ons: (note_number, channel) -> (tick, velocity)
    let mut pending: std::collections::HashMap<(u8, u8), (u64, u8)> =
        std::collections::HashMap::new();

    for event in &track.events {
        match &event.message {
            MidiMessage::NoteOn {
                channel,
                note,
                velocity,
            } if *velocity > 0 => {
                pending.insert((*note, *channel), (event.tick, *velocity));
            }
            msg if msg.is_note_off() => {
                let (note_num, channel) = match msg {
                    MidiMessage::NoteOff { note, channel, .. } => (*note, *channel),
                    MidiMessage::NoteOn { note, channel, .. } => (*note, *channel),
                    _ => continue,
                };
                if let Some((start_tick, velocity)) = pending.remove(&(note_num, channel)) {
                    let start_time = tempo_map.tick_to_seconds(start_tick);
                    let end_time = tempo_map.tick_to_seconds(event.tick);
                    let duration = end_time - start_time;
                    let frequency = 440.0 * 2.0f64.powf((note_num as f64 - 69.0) / 12.0);
                    let amplitude = velocity as f64 / 127.0;

                    notes.push(melodyn_core::DetectedNote {
                        frequency,
                        midi_note: note_num as f64,
                        start_time,
                        duration,
                        start_sample: 0,
                        end_sample: 0,
                        amplitude,
                        peak_amplitude: amplitude,
                        confidence: 1.0,
                        pitch_contour: vec![frequency],
                        amplitude_contour: vec![amplitude],
                        vibrato_rate: 0.0,
                        vibrato_depth: 0.0,
                        attack_time: 0.0,
                        release_time: 0.0,
                    });
                }
            }
            _ => {}
        }
    }

    notes.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    notes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vlq_encode_decode() {
        let test_values = [0u32, 127, 128, 255, 16383, 16384, 2097151];
        for &val in &test_values {
            let bytes = vlq_bytes(val);
            let (decoded, len) = read_vlq(&bytes).unwrap();
            assert_eq!(decoded, val, "VLQ roundtrip failed for {}", val);
            assert_eq!(len, bytes.len());
        }
    }

    #[test]
    fn test_vlq_specific_values() {
        // 0x00 -> [0x00]
        assert_eq!(vlq_bytes(0), vec![0x00]);
        // 0x7F -> [0x7F]
        assert_eq!(vlq_bytes(0x7F), vec![0x7F]);
        // 0x80 -> [0x81, 0x00]
        assert_eq!(vlq_bytes(0x80), vec![0x81, 0x00]);
        // 0x3FFF -> [0xFF, 0x7F]
        assert_eq!(vlq_bytes(0x3FFF), vec![0xFF, 0x7F]);
        // 0x4000 -> [0x81, 0x80, 0x00]
        assert_eq!(vlq_bytes(0x4000), vec![0x81, 0x80, 0x00]);
    }

    #[test]
    fn test_tempo_map_basic() {
        let map = TempoMap::new(480);
        // 120 BPM: 1 quarter note = 0.5 seconds
        let seconds = map.tick_to_seconds(480);
        assert!((seconds - 0.5).abs() < 0.001);
        // And back
        let tick = map.seconds_to_tick(0.5);
        assert_eq!(tick, 480);
    }

    #[test]
    fn test_tempo_map_with_change() {
        let mut map = TempoMap::new(480);
        // Start at 120 BPM, change to 60 BPM at tick 960 (1 second)
        map.add_tempo_change(960, 1_000_000); // 60 BPM

        // First 960 ticks at 120 BPM = 1 second
        assert!((map.tick_to_seconds(960) - 1.0).abs() < 0.001);
        // Next 480 ticks at 60 BPM = 1 second
        assert!((map.tick_to_seconds(1440) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_midi_file_roundtrip() {
        let mut file = MidiFile::new(MidiFormat::SingleTrack, 480);
        let mut track = MidiTrack::new("Test Track".to_string());
        track.add_event(MidiEvent {
            tick: 0,
            message: MidiMessage::NoteOn {
                channel: 0,
                note: 60,
                velocity: 100,
            },
        });
        track.add_event(MidiEvent {
            tick: 480,
            message: MidiMessage::NoteOff {
                channel: 0,
                note: 60,
                velocity: 0,
            },
        });
        file.tracks.push(track);

        let bytes = write_midi(&file).unwrap();
        let parsed = read_midi(&bytes).unwrap();

        assert_eq!(parsed.format, MidiFormat::SingleTrack);
        assert_eq!(parsed.ticks_per_quarter, 480);
        assert_eq!(parsed.tracks.len(), 1);

        // Find note events (skip meta events like track name)
        let note_on_events: Vec<_> = parsed.tracks[0]
            .events
            .iter()
            .filter(|e| e.message.is_note_on())
            .collect();
        let note_off_events: Vec<_> = parsed.tracks[0]
            .events
            .iter()
            .filter(|e| e.message.is_note_off())
            .collect();
        assert_eq!(note_on_events.len(), 1);
        assert_eq!(note_off_events.len(), 1);
    }

    #[test]
    fn test_midi_format1_multiple_tracks() {
        let mut file = MidiFile::new(MidiFormat::MultiTrack, 480);
        // Tempo track
        let mut tempo = MidiTrack::new("Tempo".to_string());
        tempo.add_event(MidiEvent {
            tick: 0,
            message: MidiMessage::Meta {
                meta_type: 0x51,
                data: vec![0x07, 0xA1, 0x20], // 120 BPM
            },
        });
        file.tracks.push(tempo);
        // Note track
        let mut notes = MidiTrack::new("Piano".to_string());
        notes.add_event(MidiEvent {
            tick: 0,
            message: MidiMessage::NoteOn {
                channel: 0,
                note: 60,
                velocity: 80,
            },
        });
        notes.add_event(MidiEvent {
            tick: 480,
            message: MidiMessage::NoteOff {
                channel: 0,
                note: 60,
                velocity: 0,
            },
        });
        file.tracks.push(notes);

        let bytes = write_midi(&file).unwrap();
        let parsed = read_midi(&bytes).unwrap();
        assert_eq!(parsed.format, MidiFormat::MultiTrack);
        assert_eq!(parsed.tracks.len(), 2);
    }

    #[test]
    fn test_detected_note_to_midi_roundtrip() {
        let notes = vec![melodyn_core::DetectedNote {
            frequency: 440.0,
            midi_note: 69.0,
            start_time: 0.0,
            duration: 0.5,
            start_sample: 0,
            end_sample: 22050,
            amplitude: 0.8,
            peak_amplitude: 0.8,
            confidence: 0.95,
            pitch_contour: vec![440.0],
            amplitude_contour: vec![0.8],
            vibrato_rate: 0.0,
            vibrato_depth: 0.0,
            attack_time: 0.0,
            release_time: 0.0,
        }];

        let tempo_map = TempoMap::new(480);
        let track = notes_to_midi_track(&notes, &tempo_map, 0);
        let recovered = midi_to_notes(&track, &tempo_map);

        assert_eq!(recovered.len(), 1);
        assert!((recovered[0].frequency - 440.0).abs() < 0.01);
        assert!((recovered[0].start_time - 0.0).abs() < 0.01);
        assert!((recovered[0].duration - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_notes_to_midi_file() {
        let notes = vec![
            melodyn_core::DetectedNote {
                frequency: 261.63,
                midi_note: 60.0,
                start_time: 0.0,
                duration: 0.5,
                start_sample: 0,
                end_sample: 22050,
                amplitude: 0.7,
                peak_amplitude: 0.7,
                confidence: 0.9,
                pitch_contour: vec![261.63],
                amplitude_contour: vec![0.7],
                vibrato_rate: 0.0,
                vibrato_depth: 0.0,
                attack_time: 0.0,
                release_time: 0.0,
            },
            melodyn_core::DetectedNote {
                frequency: 329.63,
                midi_note: 64.0,
                start_time: 0.5,
                duration: 0.5,
                start_sample: 22050,
                end_sample: 44100,
                amplitude: 0.6,
                peak_amplitude: 0.6,
                confidence: 0.9,
                pitch_contour: vec![329.63],
                amplitude_contour: vec![0.6],
                vibrato_rate: 0.0,
                vibrato_depth: 0.0,
                attack_time: 0.0,
                release_time: 0.0,
            },
        ];

        let file = notes_to_midi_file(&notes, 480, 120.0);
        assert_eq!(file.tracks.len(), 1);

        let bytes = write_midi(&file).unwrap();
        let parsed = read_midi(&bytes).unwrap();
        assert_eq!(parsed.tracks.len(), 1);
    }

    #[test]
    fn test_midi_message_properties() {
        let on = MidiMessage::NoteOn {
            channel: 0,
            note: 60,
            velocity: 100,
        };
        assert!(on.is_note_on());
        assert!(!on.is_note_off());
        assert_eq!(on.channel(), Some(0));

        let off = MidiMessage::NoteOff {
            channel: 1,
            note: 60,
            velocity: 0,
        };
        assert!(!off.is_note_on());
        assert!(off.is_note_off());
        assert_eq!(off.channel(), Some(1));

        let vel0 = MidiMessage::NoteOn {
            channel: 0,
            note: 60,
            velocity: 0,
        };
        assert!(!vel0.is_note_on());
        assert!(vel0.is_note_off());

        let meta = MidiMessage::Meta {
            meta_type: 0x51,
            data: vec![0x07, 0xA1, 0x20],
        };
        assert_eq!(meta.channel(), None);
    }

    #[test]
    fn test_midi_track_sorted_insert() {
        let mut track = MidiTrack::new("test".to_string());
        track.add_event(MidiEvent {
            tick: 480,
            message: MidiMessage::NoteOn {
                channel: 0,
                note: 62,
                velocity: 100,
            },
        });
        track.add_event(MidiEvent {
            tick: 0,
            message: MidiMessage::NoteOn {
                channel: 0,
                note: 60,
                velocity: 100,
            },
        });
        track.add_event(MidiEvent {
            tick: 240,
            message: MidiMessage::NoteOn {
                channel: 0,
                note: 61,
                velocity: 100,
            },
        });
        assert_eq!(track.events[0].tick, 0);
        assert_eq!(track.events[1].tick, 240);
        assert_eq!(track.events[2].tick, 480);
    }
}
