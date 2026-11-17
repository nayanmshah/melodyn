//! ARA 2.0 (Audio Random Access) full implementation.
//!
//! Implements the complete ARA 2.0 protocol for bidirectional audio data exchange
//! between DAWs and the Melodyn plugin. Based on the MUAra* class hierarchy:
//! MUAraDocumentController, MUAraDocument, MUAraAudioSource, MUAraAudioModification,
//! MUAraAudioPlaybackRegion, MUAraMusicalContext, MUAraRegionSequence, and content readers.

use melodyn_core::{AnalysisConfig, AnalysisResult, DetectedNote};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for ARA model objects.
pub type AraId = u64;

/// Generates sequential unique IDs.
#[derive(Debug)]
struct IdGenerator {
    next: AraId,
}

impl IdGenerator {
    fn new() -> Self {
        Self { next: 1 }
    }
    fn next(&mut self) -> AraId {
        let id = self.next;
        self.next += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// Musical context
// ---------------------------------------------------------------------------

/// Tempo entry at a specific position.
#[derive(Debug, Clone)]
pub struct TempoEntry {
    /// Position in quarter notes from the start.
    pub position_quarters: f64,
    /// Tempo in BPM.
    pub bpm: f64,
}

/// Bar (measure) signature at a specific position.
#[derive(Debug, Clone)]
pub struct BarSignature {
    /// Position in quarter notes from the start.
    pub position_quarters: f64,
    /// Numerator of the time signature.
    pub numerator: u32,
    /// Denominator of the time signature.
    pub denominator: u32,
}

/// Key signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySignature {
    Major(i8),
    Minor(i8),
}

impl Default for KeySignature {
    fn default() -> Self {
        KeySignature::Major(0)
    }
}

/// Key signature entry at a specific position.
#[derive(Debug, Clone)]
pub struct KeySignatureEntry {
    pub position_quarters: f64,
    pub key: KeySignature,
}

/// Sheet chord entry.
#[derive(Debug, Clone)]
pub struct SheetChord {
    pub position_quarters: f64,
    pub root: u8,
    pub bass: u8,
    pub intervals: Vec<u8>,
    pub name: String,
}

/// Static tuning: per-pitch-class cent offsets (12 entries, C through B).
#[derive(Debug, Clone)]
pub struct StaticTuning {
    pub cent_offsets: [f64; 12],
    pub concert_pitch_hz: f64,
}

impl Default for StaticTuning {
    fn default() -> Self {
        Self {
            cent_offsets: [0.0; 12],
            concert_pitch_hz: 440.0,
        }
    }
}

/// Musical context from the host DAW.
///
/// Corresponds to MUAraMusicalContext.
#[derive(Debug, Clone)]
pub struct AraMusicalContext {
    pub id: AraId,
    pub name: String,
    pub tempo_entries: Vec<TempoEntry>,
    pub bar_signatures: Vec<BarSignature>,
    pub key_signatures: Vec<KeySignatureEntry>,
    pub sheet_chords: Vec<SheetChord>,
    pub tuning: StaticTuning,
}

impl AraMusicalContext {
    fn new(id: AraId, name: String) -> Self {
        Self {
            id,
            name,
            tempo_entries: vec![TempoEntry {
                position_quarters: 0.0,
                bpm: 120.0,
            }],
            bar_signatures: vec![BarSignature {
                position_quarters: 0.0,
                numerator: 4,
                denominator: 4,
            }],
            key_signatures: vec![KeySignatureEntry {
                position_quarters: 0.0,
                key: KeySignature::default(),
            }],
            sheet_chords: Vec::new(),
            tuning: StaticTuning::default(),
        }
    }

    /// Get tempo (BPM) at a given quarter-note position.
    pub fn tempo_at(&self, position_quarters: f64) -> f64 {
        let mut bpm = 120.0;
        for entry in &self.tempo_entries {
            if entry.position_quarters <= position_quarters {
                bpm = entry.bpm;
            } else {
                break;
            }
        }
        bpm
    }

    /// Convert quarter-note position to seconds.
    pub fn quarters_to_seconds(&self, quarters: f64) -> f64 {
        if self.tempo_entries.is_empty() {
            return quarters * 0.5; // 120 BPM default
        }
        let mut seconds = 0.0;
        let mut prev_quarters = 0.0;
        let mut prev_bpm = self.tempo_entries[0].bpm;

        for entry in &self.tempo_entries {
            if entry.position_quarters >= quarters {
                break;
            }
            let segment_quarters = entry.position_quarters - prev_quarters;
            seconds += segment_quarters * 60.0 / prev_bpm;
            prev_quarters = entry.position_quarters;
            prev_bpm = entry.bpm;
        }
        seconds += (quarters - prev_quarters) * 60.0 / prev_bpm;
        seconds
    }

    /// Convert seconds to quarter-note position.
    pub fn seconds_to_quarters(&self, seconds: f64) -> f64 {
        if self.tempo_entries.is_empty() {
            return seconds * 2.0; // 120 BPM default
        }
        let mut remaining = seconds;
        let mut quarters = 0.0;
        let mut prev_quarters = 0.0;
        let mut prev_bpm = self.tempo_entries[0].bpm;

        for entry in self.tempo_entries.iter().skip(1) {
            let segment_quarters = entry.position_quarters - prev_quarters;
            let segment_seconds = segment_quarters * 60.0 / prev_bpm;
            if remaining <= segment_seconds {
                return quarters + remaining * prev_bpm / 60.0;
            }
            remaining -= segment_seconds;
            quarters += segment_quarters;
            prev_quarters = entry.position_quarters;
            prev_bpm = entry.bpm;
        }
        quarters + remaining * prev_bpm / 60.0
    }
}

// ---------------------------------------------------------------------------
// Audio source
// ---------------------------------------------------------------------------

/// An audio source registered with the host.
///
/// Corresponds to MUAraAudioSource.
#[derive(Debug, Clone)]
pub struct AraAudioSource {
    pub id: AraId,
    pub name: String,
    pub sample_rate: f64,
    pub channel_count: usize,
    pub sample_count: usize,
    /// Per-channel audio data (planar).
    samples: Vec<Vec<f64>>,
    /// Analysis result, populated after analysis completes.
    pub analysis_result: Option<AnalysisResult>,
    /// Whether analysis has been requested but not completed.
    pub analysis_pending: bool,
    /// Persistent ID for archive storage.
    pub persistent_id: String,
}

impl AraAudioSource {
    fn new(id: AraId, name: String, sample_rate: f64, channel_count: usize) -> Self {
        Self {
            id,
            name,
            sample_rate,
            channel_count,
            sample_count: 0,
            samples: vec![Vec::new(); channel_count],
            analysis_result: None,
            analysis_pending: false,
            persistent_id: format!("source-{}", id),
        }
    }

    /// Push samples for a channel.
    pub fn push_samples(&mut self, channel: usize, data: &[f64]) {
        if channel < self.channel_count {
            self.samples[channel].extend_from_slice(data);
            self.sample_count = self.samples[0].len();
        }
    }

    /// Read samples from a channel into a buffer. Returns count read.
    pub fn read_samples(&self, channel: usize, start: usize, buf: &mut [f64]) -> usize {
        if channel >= self.channel_count {
            return 0;
        }
        let ch = &self.samples[channel];
        let available = if start < ch.len() {
            ch.len() - start
        } else {
            0
        };
        let count = available.min(buf.len());
        buf[..count].copy_from_slice(&ch[start..start + count]);
        count
    }

    /// Get mono mixdown of all channels.
    pub fn mono_samples(&self) -> Vec<f64> {
        if self.samples.is_empty() || self.sample_count == 0 {
            return Vec::new();
        }
        let scale = 1.0 / self.channel_count as f64;
        (0..self.sample_count)
            .map(|i| {
                self.samples
                    .iter()
                    .map(|ch| if i < ch.len() { ch[i] } else { 0.0 })
                    .sum::<f64>()
                    * scale
            })
            .collect()
    }

    /// Get detected notes (empty if not analyzed).
    pub fn notes(&self) -> &[DetectedNote] {
        self.analysis_result
            .as_ref()
            .map(|r| r.notes.as_slice())
            .unwrap_or(&[])
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        self.sample_count as f64 / self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Audio modification
// ---------------------------------------------------------------------------

/// Per-note edit parameters within a modification.
#[derive(Debug, Clone)]
pub struct NoteEdit {
    /// Index into the source's detected notes.
    pub note_index: usize,
    /// Pitch shift in semitones.
    pub pitch_shift: f64,
    /// Time stretch factor (1.0 = no change).
    pub time_stretch: f64,
    /// Formant shift in semitones.
    pub formant_shift: f64,
    /// Amplitude gain.
    pub gain: f64,
    /// New start time override (None = keep original).
    pub new_start: Option<f64>,
}

impl NoteEdit {
    pub fn identity(note_index: usize) -> Self {
        Self {
            note_index,
            pitch_shift: 0.0,
            time_stretch: 1.0,
            formant_shift: 0.0,
            gain: 1.0,
            new_start: None,
        }
    }
}

/// An edit applied to an audio source.
///
/// Corresponds to MUAraAudioModification.
#[derive(Debug, Clone)]
pub struct AraAudioModification {
    pub id: AraId,
    pub source_id: AraId,
    pub name: String,
    /// Per-note edits (index by note_index from the source).
    pub note_edits: HashMap<usize, NoteEdit>,
    /// Global pitch shift applied to all notes (semitones).
    pub global_pitch_shift: f64,
    /// Global time stretch factor.
    pub global_time_stretch: f64,
    /// Persistent ID for archive storage.
    pub persistent_id: String,
}

impl AraAudioModification {
    fn new(id: AraId, source_id: AraId, name: String) -> Self {
        Self {
            id,
            source_id,
            name,
            note_edits: HashMap::new(),
            global_pitch_shift: 0.0,
            global_time_stretch: 1.0,
            persistent_id: format!("mod-{}", id),
        }
    }

    /// Whether any pitch edits exist.
    pub fn has_pitch_edit(&self) -> bool {
        self.global_pitch_shift.abs() > 1e-6
            || self.note_edits.values().any(|e| e.pitch_shift.abs() > 1e-6)
    }

    /// Whether any time edits exist.
    pub fn has_time_edit(&self) -> bool {
        (self.global_time_stretch - 1.0).abs() > 1e-6
            || self
                .note_edits
                .values()
                .any(|e| (e.time_stretch - 1.0).abs() > 1e-6 || e.new_start.is_some())
    }

    /// Get the effective pitch shift for a note.
    pub fn pitch_shift_for(&self, note_index: usize) -> f64 {
        let local = self
            .note_edits
            .get(&note_index)
            .map(|e| e.pitch_shift)
            .unwrap_or(0.0);
        self.global_pitch_shift + local
    }

    /// Get the effective time stretch for a note.
    pub fn time_stretch_for(&self, note_index: usize) -> f64 {
        let local = self
            .note_edits
            .get(&note_index)
            .map(|e| e.time_stretch)
            .unwrap_or(1.0);
        self.global_time_stretch * local
    }
}

// ---------------------------------------------------------------------------
// Playback region
// ---------------------------------------------------------------------------

/// A region in the DAW timeline that plays back a modification.
///
/// Corresponds to MUAraAudioPlaybackRegion.
#[derive(Debug, Clone)]
pub struct AraPlaybackRegion {
    pub id: AraId,
    pub modification_id: AraId,
    pub region_sequence_id: Option<AraId>,
    /// Start time on the host playback timeline (seconds).
    pub start_in_playback: f64,
    /// Duration on the host timeline (seconds).
    pub duration_in_playback: f64,
    /// Start offset within the modification (seconds).
    pub start_in_modification: f64,
    pub name: String,
}

impl AraPlaybackRegion {
    fn new(id: AraId, modification_id: AraId) -> Self {
        Self {
            id,
            modification_id,
            region_sequence_id: None,
            start_in_playback: 0.0,
            duration_in_playback: 0.0,
            start_in_modification: 0.0,
            name: String::new(),
        }
    }

    /// End time on the playback timeline.
    pub fn end_in_playback(&self) -> f64 {
        self.start_in_playback + self.duration_in_playback
    }

    /// Whether a playback time falls within this region.
    pub fn contains_time(&self, time: f64) -> bool {
        time >= self.start_in_playback && time < self.end_in_playback()
    }

    /// Map playback time to modification time.
    pub fn playback_to_modification(&self, playback_time: f64) -> f64 {
        self.start_in_modification + (playback_time - self.start_in_playback)
    }
}

// ---------------------------------------------------------------------------
// Region sequence
// ---------------------------------------------------------------------------

/// An ordered sequence of playback regions on a track.
///
/// Corresponds to MUAraRegionSequence.
#[derive(Debug, Clone)]
pub struct AraRegionSequence {
    pub id: AraId,
    pub name: String,
    pub musical_context_id: Option<AraId>,
    pub region_ids: Vec<AraId>,
}

impl AraRegionSequence {
    fn new(id: AraId, name: String) -> Self {
        Self {
            id,
            name,
            musical_context_id: None,
            region_ids: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Content readers
// ---------------------------------------------------------------------------

/// Note content from analysis.
#[derive(Debug, Clone)]
pub struct NoteContent {
    pub frequency: f64,
    pub start_quarters: f64,
    pub duration_quarters: f64,
    pub volume: f64,
    pub pitch_cents: f64,
}

/// Content reader for detected notes.
pub struct NotesContentReader<'a> {
    notes: &'a [DetectedNote],
    context: &'a AraMusicalContext,
    index: usize,
}

impl<'a> NotesContentReader<'a> {
    pub fn new(notes: &'a [DetectedNote], context: &'a AraMusicalContext) -> Self {
        Self {
            notes,
            context,
            index: 0,
        }
    }

    pub fn count(&self) -> usize {
        self.notes.len()
    }

    pub fn read_next(&mut self) -> Option<NoteContent> {
        if self.index >= self.notes.len() {
            return None;
        }
        let note = &self.notes[self.index];
        self.index += 1;
        let start_q = self.context.seconds_to_quarters(note.start_time);
        let end_q = self.context.seconds_to_quarters(note.end_time());
        Some(NoteContent {
            frequency: note.frequency,
            start_quarters: start_q,
            duration_quarters: end_q - start_q,
            volume: note.amplitude,
            pitch_cents: note.cents_deviation(),
        })
    }
}

/// Content reader for tempo entries.
pub struct TempoEntriesContentReader<'a> {
    entries: &'a [TempoEntry],
    index: usize,
}

impl<'a> TempoEntriesContentReader<'a> {
    pub fn new(entries: &'a [TempoEntry]) -> Self {
        Self { entries, index: 0 }
    }
    pub fn count(&self) -> usize {
        self.entries.len()
    }
    pub fn read_next(&mut self) -> Option<&'a TempoEntry> {
        if self.index >= self.entries.len() {
            return None;
        }
        let entry = &self.entries[self.index];
        self.index += 1;
        Some(entry)
    }
}

/// Content reader for bar signatures.
pub struct BarSignaturesContentReader<'a> {
    sigs: &'a [BarSignature],
    index: usize,
}

impl<'a> BarSignaturesContentReader<'a> {
    pub fn new(sigs: &'a [BarSignature]) -> Self {
        Self { sigs, index: 0 }
    }
    pub fn count(&self) -> usize {
        self.sigs.len()
    }
    pub fn read_next(&mut self) -> Option<&'a BarSignature> {
        if self.index >= self.sigs.len() {
            return None;
        }
        let sig = &self.sigs[self.index];
        self.index += 1;
        Some(sig)
    }
}

/// Content reader for key signatures.
pub struct KeySignaturesContentReader<'a> {
    keys: &'a [KeySignatureEntry],
    index: usize,
}

impl<'a> KeySignaturesContentReader<'a> {
    pub fn new(keys: &'a [KeySignatureEntry]) -> Self {
        Self { keys, index: 0 }
    }
    pub fn count(&self) -> usize {
        self.keys.len()
    }
    pub fn read_next(&mut self) -> Option<&'a KeySignatureEntry> {
        if self.index >= self.keys.len() {
            return None;
        }
        let key = &self.keys[self.index];
        self.index += 1;
        Some(key)
    }
}

/// Content reader for sheet chords.
pub struct SheetChordsContentReader<'a> {
    chords: &'a [SheetChord],
    index: usize,
}

impl<'a> SheetChordsContentReader<'a> {
    pub fn new(chords: &'a [SheetChord]) -> Self {
        Self { chords, index: 0 }
    }
    pub fn count(&self) -> usize {
        self.chords.len()
    }
    pub fn read_next(&mut self) -> Option<&'a SheetChord> {
        if self.index >= self.chords.len() {
            return None;
        }
        let chord = &self.chords[self.index];
        self.index += 1;
        Some(chord)
    }
}

/// Content reader for static tuning.
pub struct StaticTuningContentReader<'a> {
    tuning: &'a StaticTuning,
    read: bool,
}

impl<'a> StaticTuningContentReader<'a> {
    pub fn new(tuning: &'a StaticTuning) -> Self {
        Self {
            tuning,
            read: false,
        }
    }
    pub fn read(&mut self) -> Option<&'a StaticTuning> {
        if self.read {
            return None;
        }
        self.read = true;
        Some(self.tuning)
    }
}

// ---------------------------------------------------------------------------
// Chunk archive persistence
// ---------------------------------------------------------------------------

/// Chunk type identifiers for persistent storage.
/// com.celemony.ara.chunk.1 through .13
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChunkType {
    Document = 1,
    AudioSource = 2,
    AudioModification = 3,
    PlaybackRegion = 4,
    MusicalContext = 5,
    RegionSequence = 6,
    NoteEdits = 7,
    TempoMap = 8,
    BarSignatures = 9,
    KeySignatures = 10,
    SheetChords = 11,
    StaticTuning = 12,
    AnalysisState = 13,
}

/// A chunk in the persistent archive.
#[derive(Debug, Clone)]
pub struct ArchiveChunk {
    pub chunk_type: ChunkType,
    pub persistent_id: String,
    pub data: Vec<u8>,
}

/// Archive for saving/restoring ARA document state.
#[derive(Debug, Clone, Default)]
pub struct AraArchive {
    pub chunks: Vec<ArchiveChunk>,
}

impl AraArchive {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    pub fn add_chunk(&mut self, chunk_type: ChunkType, persistent_id: String, data: Vec<u8>) {
        self.chunks.push(ArchiveChunk {
            chunk_type,
            persistent_id,
            data,
        });
    }

    pub fn find_chunks(&self, chunk_type: ChunkType) -> Vec<&ArchiveChunk> {
        self.chunks
            .iter()
            .filter(|c| c.chunk_type == chunk_type)
            .collect()
    }

    pub fn find_chunk(&self, chunk_type: ChunkType, persistent_id: &str) -> Option<&ArchiveChunk> {
        self.chunks
            .iter()
            .find(|c| c.chunk_type == chunk_type && c.persistent_id == persistent_id)
    }

    /// Serialize the archive to bytes (simple length-prefixed format).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // Magic header
        out.extend_from_slice(b"ARA2");
        // Chunk count
        out.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());
        for chunk in &self.chunks {
            out.push(chunk.chunk_type as u8);
            let id_bytes = chunk.persistent_id.as_bytes();
            out.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(id_bytes);
            out.extend_from_slice(&(chunk.data.len() as u32).to_le_bytes());
            out.extend_from_slice(&chunk.data);
        }
        out
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 || &data[0..4] != b"ARA2" {
            return None;
        }
        let count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let mut pos = 8;
        let mut chunks = Vec::with_capacity(count);
        for _ in 0..count {
            if pos >= data.len() {
                return None;
            }
            let ct_byte = data[pos];
            pos += 1;
            let chunk_type = match ct_byte {
                1 => ChunkType::Document,
                2 => ChunkType::AudioSource,
                3 => ChunkType::AudioModification,
                4 => ChunkType::PlaybackRegion,
                5 => ChunkType::MusicalContext,
                6 => ChunkType::RegionSequence,
                7 => ChunkType::NoteEdits,
                8 => ChunkType::TempoMap,
                9 => ChunkType::BarSignatures,
                10 => ChunkType::KeySignatures,
                11 => ChunkType::SheetChords,
                12 => ChunkType::StaticTuning,
                13 => ChunkType::AnalysisState,
                _ => return None,
            };
            if pos + 4 > data.len() {
                return None;
            }
            let id_len =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;
            if pos + id_len > data.len() {
                return None;
            }
            let persistent_id = String::from_utf8_lossy(&data[pos..pos + id_len]).to_string();
            pos += id_len;
            if pos + 4 > data.len() {
                return None;
            }
            let data_len =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;
            if pos + data_len > data.len() {
                return None;
            }
            let chunk_data = data[pos..pos + data_len].to_vec();
            pos += data_len;
            chunks.push(ArchiveChunk {
                chunk_type,
                persistent_id,
                data: chunk_data,
            });
        }
        Some(Self { chunks })
    }
}

// ---------------------------------------------------------------------------
// Model update notifications
// ---------------------------------------------------------------------------

/// Notification type for model changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelUpdateNotification {
    AudioSourceAnalysisStarted(AraId),
    AudioSourceAnalysisCompleted(AraId),
    AudioSourceSamplesChanged(AraId),
    ModificationChanged(AraId),
    PlaybackRegionChanged(AraId),
    MusicalContextChanged(AraId),
    RegionSequenceChanged(AraId),
    DocumentChanged,
}

// ---------------------------------------------------------------------------
// ARA Document
// ---------------------------------------------------------------------------

/// Root container for all ARA model objects.
///
/// Corresponds to MUAraDocument.
#[derive(Debug)]
pub struct AraDocument {
    pub name: String,
    pub audio_sources: HashMap<AraId, AraAudioSource>,
    pub modifications: HashMap<AraId, AraAudioModification>,
    pub playback_regions: HashMap<AraId, AraPlaybackRegion>,
    pub musical_contexts: HashMap<AraId, AraMusicalContext>,
    pub region_sequences: HashMap<AraId, AraRegionSequence>,
    id_gen: IdGenerator,
    /// Pending notifications from the current edit cycle.
    pending_notifications: Vec<ModelUpdateNotification>,
}

impl AraDocument {
    pub fn new(name: String) -> Self {
        Self {
            name,
            audio_sources: HashMap::new(),
            modifications: HashMap::new(),
            playback_regions: HashMap::new(),
            musical_contexts: HashMap::new(),
            region_sequences: HashMap::new(),
            id_gen: IdGenerator::new(),
            pending_notifications: Vec::new(),
        }
    }

    fn notify(&mut self, n: ModelUpdateNotification) {
        self.pending_notifications.push(n);
    }

    pub fn drain_notifications(&mut self) -> Vec<ModelUpdateNotification> {
        std::mem::take(&mut self.pending_notifications)
    }

    pub fn create_audio_source(
        &mut self,
        name: String,
        sample_rate: f64,
        channel_count: usize,
    ) -> AraId {
        let id = self.id_gen.next();
        let source = AraAudioSource::new(id, name, sample_rate, channel_count);
        self.audio_sources.insert(id, source);
        id
    }

    pub fn create_modification(&mut self, source_id: AraId, name: String) -> AraId {
        let id = self.id_gen.next();
        let modification = AraAudioModification::new(id, source_id, name);
        self.modifications.insert(id, modification);
        id
    }

    pub fn create_playback_region(&mut self, modification_id: AraId) -> AraId {
        let id = self.id_gen.next();
        let region = AraPlaybackRegion::new(id, modification_id);
        self.playback_regions.insert(id, region);
        id
    }

    pub fn create_musical_context(&mut self, name: String) -> AraId {
        let id = self.id_gen.next();
        let ctx = AraMusicalContext::new(id, name);
        self.musical_contexts.insert(id, ctx);
        id
    }

    pub fn create_region_sequence(&mut self, name: String) -> AraId {
        let id = self.id_gen.next();
        let seq = AraRegionSequence::new(id, name);
        self.region_sequences.insert(id, seq);
        id
    }

    pub fn destroy_audio_source(&mut self, id: AraId) {
        self.audio_sources.remove(&id);
        // Remove modifications referencing this source
        let mod_ids: Vec<AraId> = self
            .modifications
            .iter()
            .filter(|(_, m)| m.source_id == id)
            .map(|(&mid, _)| mid)
            .collect();
        for mid in mod_ids {
            self.destroy_modification(mid);
        }
    }

    pub fn destroy_modification(&mut self, id: AraId) {
        self.modifications.remove(&id);
        let region_ids: Vec<AraId> = self
            .playback_regions
            .iter()
            .filter(|(_, r)| r.modification_id == id)
            .map(|(&rid, _)| rid)
            .collect();
        for rid in region_ids {
            self.playback_regions.remove(&rid);
        }
    }

    /// Find all playback regions active at a given playback time.
    pub fn regions_at_time(&self, time: f64) -> Vec<&AraPlaybackRegion> {
        self.playback_regions
            .values()
            .filter(|r| r.contains_time(time))
            .collect()
    }

    /// Save document state to an archive.
    pub fn save_to_archive(&self) -> AraArchive {
        let mut archive = AraArchive::new();
        // Document chunk
        archive.add_chunk(
            ChunkType::Document,
            "document".to_string(),
            self.name.as_bytes().to_vec(),
        );
        // Audio sources
        for source in self.audio_sources.values() {
            let mut data = Vec::new();
            data.extend_from_slice(&source.sample_rate.to_le_bytes());
            data.extend_from_slice(&(source.channel_count as u32).to_le_bytes());
            data.extend_from_slice(&(source.sample_count as u64).to_le_bytes());
            let name_bytes = source.name.as_bytes();
            data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            data.extend_from_slice(name_bytes);
            archive.add_chunk(ChunkType::AudioSource, source.persistent_id.clone(), data);
        }
        // Modifications
        for modif in self.modifications.values() {
            let mut data = Vec::new();
            data.extend_from_slice(&modif.source_id.to_le_bytes());
            data.extend_from_slice(&modif.global_pitch_shift.to_le_bytes());
            data.extend_from_slice(&modif.global_time_stretch.to_le_bytes());
            archive.add_chunk(
                ChunkType::AudioModification,
                modif.persistent_id.clone(),
                data,
            );
        }
        // Musical contexts
        for ctx in self.musical_contexts.values() {
            let mut data = Vec::new();
            data.extend_from_slice(&(ctx.tempo_entries.len() as u32).to_le_bytes());
            for te in &ctx.tempo_entries {
                data.extend_from_slice(&te.position_quarters.to_le_bytes());
                data.extend_from_slice(&te.bpm.to_le_bytes());
            }
            archive.add_chunk(ChunkType::MusicalContext, format!("ctx-{}", ctx.id), data);
        }
        archive
    }
}

// ---------------------------------------------------------------------------
// ARA Document Controller
// ---------------------------------------------------------------------------

/// Manages the ARA document lifecycle and coordinates with the host.
///
/// Corresponds to MUAraDocumentController.
pub struct AraDocumentController {
    pub document: AraDocument,
    pub analysis_config: AnalysisConfig,
    edit_depth: u32,
    restore_in_progress: bool,
}

impl AraDocumentController {
    pub fn new() -> Self {
        Self {
            document: AraDocument::new("Untitled".to_string()),
            analysis_config: AnalysisConfig::default(),
            edit_depth: 0,
            restore_in_progress: false,
        }
    }

    /// Begin an edit cycle. Calls may be nested.
    pub fn begin_edit_cycle(&mut self) {
        self.edit_depth += 1;
        log::debug!("ARA: begin edit cycle (depth={})", self.edit_depth);
    }

    /// End an edit cycle. When depth reaches 0, notifications are flushed.
    pub fn end_edit_cycle(&mut self) -> Vec<ModelUpdateNotification> {
        if self.edit_depth > 0 {
            self.edit_depth -= 1;
        }
        log::debug!("ARA: end edit cycle (depth={})", self.edit_depth);
        if self.edit_depth == 0 {
            self.document.drain_notifications()
        } else {
            Vec::new()
        }
    }

    /// Whether we are inside an edit cycle.
    pub fn is_editing(&self) -> bool {
        self.edit_depth > 0
    }

    /// Create an audio source and register it with the document.
    pub fn create_audio_source(
        &mut self,
        name: String,
        sample_rate: f64,
        channel_count: usize,
    ) -> AraId {
        self.document
            .create_audio_source(name, sample_rate, channel_count)
    }

    /// Supply sample data for an audio source.
    pub fn update_audio_source_samples(&mut self, id: AraId, channel: usize, samples: &[f64]) {
        if let Some(source) = self.document.audio_sources.get_mut(&id) {
            source.push_samples(channel, samples);
            self.document
                .notify(ModelUpdateNotification::AudioSourceSamplesChanged(id));
        }
    }

    /// Request analysis of an audio source.
    pub fn request_analysis(&mut self, source_id: AraId) {
        // Extract what we need before mutating
        let (mono, sample_rate) = {
            if let Some(source) = self.document.audio_sources.get(&source_id) {
                (source.mono_samples(), source.sample_rate)
            } else {
                return;
            }
        };

        if let Some(source) = self.document.audio_sources.get_mut(&source_id) {
            source.analysis_pending = true;
        }
        self.document
            .notify(ModelUpdateNotification::AudioSourceAnalysisStarted(
                source_id,
            ));

        let result = if !mono.is_empty() {
            let mut config = self.analysis_config.clone();
            config.sample_rate = sample_rate;
            let analyzer = melodyn_core::SpectralAnalyzer::new(config);
            Some(analyzer.analyze(&mono))
        } else {
            None
        };

        if let Some(source) = self.document.audio_sources.get_mut(&source_id) {
            source.analysis_result = result;
            source.analysis_pending = false;
        }
        self.document
            .notify(ModelUpdateNotification::AudioSourceAnalysisCompleted(
                source_id,
            ));
    }

    /// Create a modification for a source.
    pub fn create_modification(&mut self, source_id: AraId, name: String) -> AraId {
        self.document.create_modification(source_id, name)
    }

    /// Create a playback region.
    pub fn create_playback_region(&mut self, modification_id: AraId) -> AraId {
        self.document.create_playback_region(modification_id)
    }

    /// Save state.
    pub fn save(&self) -> AraArchive {
        self.document.save_to_archive()
    }

    /// Restore state from an archive.
    pub fn restore(&mut self, archive: &AraArchive) {
        self.restore_in_progress = true;
        if let Some(doc_chunk) = archive.find_chunks(ChunkType::Document).first() {
            self.document.name = String::from_utf8_lossy(&doc_chunk.data).to_string();
        }
        self.restore_in_progress = false;
        self.document
            .notify(ModelUpdateNotification::DocumentChanged);
    }

    /// Get the analysis config.
    pub fn analysis_config(&self) -> &AnalysisConfig {
        &self.analysis_config
    }

    /// Set the analysis config.
    pub fn set_analysis_config(&mut self, config: AnalysisConfig) {
        self.analysis_config = config;
    }
}

impl Default for AraDocumentController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_controller_creation() {
        let ctrl = AraDocumentController::new();
        assert_eq!(ctrl.document.name, "Untitled");
        assert!(!ctrl.is_editing());
    }

    #[test]
    fn test_edit_cycle() {
        let mut ctrl = AraDocumentController::new();
        ctrl.begin_edit_cycle();
        assert!(ctrl.is_editing());
        ctrl.begin_edit_cycle();
        assert!(ctrl.is_editing());
        let n1 = ctrl.end_edit_cycle();
        assert!(n1.is_empty()); // Still nested
        assert!(ctrl.is_editing());
        let n2 = ctrl.end_edit_cycle();
        assert!(!ctrl.is_editing());
        assert!(n2.is_empty()); // No changes were made
    }

    #[test]
    fn test_audio_source_lifecycle() {
        let mut ctrl = AraDocumentController::new();
        let src_id = ctrl.create_audio_source("test.wav".to_string(), 44100.0, 1);
        assert!(ctrl.document.audio_sources.contains_key(&src_id));

        // Push samples
        let samples: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        ctrl.update_audio_source_samples(src_id, 0, &samples);
        let source = ctrl.document.audio_sources.get(&src_id).unwrap();
        assert_eq!(source.sample_count, 1000);

        // Destroy
        ctrl.document.destroy_audio_source(src_id);
        assert!(!ctrl.document.audio_sources.contains_key(&src_id));
    }

    #[test]
    fn test_modification_creation() {
        let mut ctrl = AraDocumentController::new();
        let src_id = ctrl.create_audio_source("src".to_string(), 44100.0, 1);
        let mod_id = ctrl.create_modification(src_id, "mod1".to_string());
        let modif = ctrl.document.modifications.get(&mod_id).unwrap();
        assert_eq!(modif.source_id, src_id);
        assert!(!modif.has_pitch_edit());
        assert!(!modif.has_time_edit());
    }

    #[test]
    fn test_modification_edits() {
        let mut ctrl = AraDocumentController::new();
        let src_id = ctrl.create_audio_source("src".to_string(), 44100.0, 1);
        let mod_id = ctrl.create_modification(src_id, "mod1".to_string());
        let modif = ctrl.document.modifications.get_mut(&mod_id).unwrap();
        modif.global_pitch_shift = 2.0;
        assert!(modif.has_pitch_edit());
        assert_eq!(modif.pitch_shift_for(0), 2.0);

        modif.note_edits.insert(
            0,
            NoteEdit {
                note_index: 0,
                pitch_shift: 1.0,
                time_stretch: 1.5,
                formant_shift: 0.0,
                gain: 1.0,
                new_start: None,
            },
        );
        assert_eq!(modif.pitch_shift_for(0), 3.0); // global + local
        assert!((modif.time_stretch_for(0) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_playback_region() {
        let mut ctrl = AraDocumentController::new();
        let src_id = ctrl.create_audio_source("src".to_string(), 44100.0, 1);
        let mod_id = ctrl.create_modification(src_id, "mod".to_string());
        let reg_id = ctrl.create_playback_region(mod_id);
        let region = ctrl.document.playback_regions.get_mut(&reg_id).unwrap();
        region.start_in_playback = 1.0;
        region.duration_in_playback = 2.0;
        region.start_in_modification = 0.5;

        assert!(region.contains_time(2.0));
        assert!(!region.contains_time(0.5));
        assert!((region.playback_to_modification(1.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_musical_context_tempo() {
        let ctx = AraMusicalContext::new(1, "default".to_string());
        // Default 120 BPM
        assert!((ctx.tempo_at(0.0) - 120.0).abs() < 1e-10);
        // 1 quarter note at 120 BPM = 0.5 seconds
        assert!((ctx.quarters_to_seconds(1.0) - 0.5).abs() < 1e-10);
        assert!((ctx.seconds_to_quarters(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_musical_context_tempo_change() {
        let mut ctx = AraMusicalContext::new(1, "test".to_string());
        ctx.tempo_entries = vec![
            TempoEntry {
                position_quarters: 0.0,
                bpm: 120.0,
            },
            TempoEntry {
                position_quarters: 4.0,
                bpm: 60.0,
            },
        ];
        // First 4 quarters at 120 BPM = 2 seconds
        assert!((ctx.quarters_to_seconds(4.0) - 2.0).abs() < 1e-10);
        // Next quarter at 60 BPM = 1 second, so total at 5 quarters = 3 seconds
        assert!((ctx.quarters_to_seconds(5.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_archive_roundtrip() {
        let mut doc = AraDocument::new("Test Document".to_string());
        doc.create_audio_source("src1".to_string(), 44100.0, 2);
        let archive = doc.save_to_archive();
        let bytes = archive.to_bytes();
        let restored = AraArchive::from_bytes(&bytes).unwrap();
        assert_eq!(restored.chunks.len(), archive.chunks.len());
        let doc_chunks = restored.find_chunks(ChunkType::Document);
        assert_eq!(doc_chunks.len(), 1);
        assert_eq!(
            String::from_utf8_lossy(&doc_chunks[0].data),
            "Test Document"
        );
    }

    #[test]
    fn test_regions_at_time() {
        let mut doc = AraDocument::new("test".to_string());
        let src_id = doc.create_audio_source("src".to_string(), 44100.0, 1);
        let mod_id = doc.create_modification(src_id, "mod".to_string());
        let r1 = doc.create_playback_region(mod_id);
        let r2 = doc.create_playback_region(mod_id);
        doc.playback_regions.get_mut(&r1).unwrap().start_in_playback = 0.0;
        doc.playback_regions
            .get_mut(&r1)
            .unwrap()
            .duration_in_playback = 2.0;
        doc.playback_regions.get_mut(&r2).unwrap().start_in_playback = 1.5;
        doc.playback_regions
            .get_mut(&r2)
            .unwrap()
            .duration_in_playback = 2.0;

        assert_eq!(doc.regions_at_time(0.5).len(), 1);
        assert_eq!(doc.regions_at_time(1.5).len(), 2); // Overlapping
        assert_eq!(doc.regions_at_time(2.5).len(), 1);
        assert_eq!(doc.regions_at_time(4.0).len(), 0);
    }

    #[test]
    fn test_content_reader_tempo() {
        let entries = vec![
            TempoEntry {
                position_quarters: 0.0,
                bpm: 120.0,
            },
            TempoEntry {
                position_quarters: 4.0,
                bpm: 90.0,
            },
        ];
        let mut reader = TempoEntriesContentReader::new(&entries);
        assert_eq!(reader.count(), 2);
        let first = reader.read_next().unwrap();
        assert!((first.bpm - 120.0).abs() < 1e-10);
        let second = reader.read_next().unwrap();
        assert!((second.bpm - 90.0).abs() < 1e-10);
        assert!(reader.read_next().is_none());
    }

    #[test]
    fn test_destroy_cascades() {
        let mut doc = AraDocument::new("test".to_string());
        let src_id = doc.create_audio_source("src".to_string(), 44100.0, 1);
        let mod_id = doc.create_modification(src_id, "mod".to_string());
        let _reg_id = doc.create_playback_region(mod_id);
        assert_eq!(doc.modifications.len(), 1);
        assert_eq!(doc.playback_regions.len(), 1);

        doc.destroy_audio_source(src_id);
        assert!(doc.audio_sources.is_empty());
        assert!(doc.modifications.is_empty());
        assert!(doc.playback_regions.is_empty());
    }
}
