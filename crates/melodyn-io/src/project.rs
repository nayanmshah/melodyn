//! Project file format: `.mpd` project files and `.mdd` detection data files.
//!
//! Implements the persistence layer for Melodyn documents, mapping the in-memory
//! `Document` model from `melodyn-core` to on-disk binary archives. The `.mpd`
//! format stores the complete project state (metadata, tracks, elements, notes,
//! composition). The `.mdd` format stores pre-computed analysis results linked
//! to source audio by content hash, allowing analysis to be skipped on reopen.

use crate::archive::{ArchiveError, BinaryKeyValueArchive, FileArchive};
use md5::{Digest, Md5};
use std::io::{Cursor, Read, Seek, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from project file operations.
#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("Archive error: {0}")]
    Archive(#[from] ArchiveError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Audio source hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("Invalid project version: {0}")]
    InvalidVersion(u32),
}

pub type ProjectResult<T> = Result<T, ProjectError>;

// ---------------------------------------------------------------------------
// Project metadata
// ---------------------------------------------------------------------------

/// Current project file format version.
const PROJECT_VERSION: u32 = 1;
/// Entry name for metadata in the file archive.
const ENTRY_METADATA: &str = "metadata";
/// Entry name for track data.
const ENTRY_TRACKS: &str = "tracks";
/// Entry name for composition data.
const ENTRY_COMPOSITION: &str = "composition";

/// Project metadata stored in the .mpd file.
#[derive(Debug, Clone)]
pub struct ProjectMetadata {
    /// Project name / title.
    pub name: String,
    /// Format version for forward compatibility.
    pub version: u32,
    /// Creation timestamp (seconds since Unix epoch, 0 if unknown).
    pub created_at: u64,
    /// Last modified timestamp.
    pub modified_at: u64,
    /// Author or creator name.
    pub author: String,
    /// Optional description / notes.
    pub description: String,
    /// Sample rate used by the project.
    pub sample_rate: f64,
}

impl Default for ProjectMetadata {
    fn default() -> Self {
        Self {
            name: "Untitled".to_string(),
            version: PROJECT_VERSION,
            created_at: 0,
            modified_at: 0,
            author: String::new(),
            description: String::new(),
            sample_rate: 44100.0,
        }
    }
}

impl ProjectMetadata {
    fn to_archive(&self) -> BinaryKeyValueArchive {
        let mut a = BinaryKeyValueArchive::new();
        a.set_string("name", &self.name);
        a.set_i32("version", self.version as i32);
        a.set_i64("created_at", self.created_at as i64);
        a.set_i64("modified_at", self.modified_at as i64);
        a.set_string("author", &self.author);
        a.set_string("description", &self.description);
        a.set_f64("sample_rate", self.sample_rate);
        a
    }

    fn from_archive(a: &BinaryKeyValueArchive) -> ProjectResult<Self> {
        Ok(Self {
            name: a.get_string("name").unwrap_or("Untitled").to_string(),
            version: a.get_i32("version").unwrap_or(1) as u32,
            created_at: a.get_i64("created_at").unwrap_or(0) as u64,
            modified_at: a.get_i64("modified_at").unwrap_or(0) as u64,
            author: a.get_string("author").unwrap_or("").to_string(),
            description: a.get_string("description").unwrap_or("").to_string(),
            sample_rate: a.get_f64("sample_rate").unwrap_or(44100.0),
        })
    }
}

// ---------------------------------------------------------------------------
// Serializable note representation
// ---------------------------------------------------------------------------

/// A note as stored in the project file. Mirrors `DetectedNote` fields
/// but uses a flat archive representation.
#[derive(Debug, Clone)]
pub struct SerializedNote {
    pub frequency: f64,
    pub midi_note: f64,
    pub start_time: f64,
    pub duration: f64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub amplitude: f64,
    pub peak_amplitude: f64,
    pub confidence: f64,
    pub vibrato_rate: f64,
    pub vibrato_depth: f64,
    pub attack_time: f64,
    pub release_time: f64,
    pub pitch_contour: Vec<f64>,
    pub amplitude_contour: Vec<f64>,
}

impl SerializedNote {
    fn to_archive(&self) -> BinaryKeyValueArchive {
        let mut a = BinaryKeyValueArchive::new();
        a.set_f64("frequency", self.frequency);
        a.set_f64("midi_note", self.midi_note);
        a.set_f64("start_time", self.start_time);
        a.set_f64("duration", self.duration);
        a.set_i64("start_sample", self.start_sample as i64);
        a.set_i64("end_sample", self.end_sample as i64);
        a.set_f64("amplitude", self.amplitude);
        a.set_f64("peak_amplitude", self.peak_amplitude);
        a.set_f64("confidence", self.confidence);
        a.set_f64("vibrato_rate", self.vibrato_rate);
        a.set_f64("vibrato_depth", self.vibrato_depth);
        a.set_f64("attack_time", self.attack_time);
        a.set_f64("release_time", self.release_time);
        // Store contours as packed f64 bytes.
        a.set_bytes("pitch_contour", f64_vec_to_bytes(&self.pitch_contour));
        a.set_bytes(
            "amplitude_contour",
            f64_vec_to_bytes(&self.amplitude_contour),
        );
        a
    }

    fn from_archive(a: &BinaryKeyValueArchive) -> ProjectResult<Self> {
        Ok(Self {
            frequency: a.get_f64("frequency").unwrap_or(440.0),
            midi_note: a.get_f64("midi_note").unwrap_or(69.0),
            start_time: a.get_f64("start_time").unwrap_or(0.0),
            duration: a.get_f64("duration").unwrap_or(0.0),
            start_sample: a.get_i64("start_sample").unwrap_or(0) as u64,
            end_sample: a.get_i64("end_sample").unwrap_or(0) as u64,
            amplitude: a.get_f64("amplitude").unwrap_or(0.0),
            peak_amplitude: a.get_f64("peak_amplitude").unwrap_or(0.0),
            confidence: a.get_f64("confidence").unwrap_or(0.0),
            vibrato_rate: a.get_f64("vibrato_rate").unwrap_or(0.0),
            vibrato_depth: a.get_f64("vibrato_depth").unwrap_or(0.0),
            attack_time: a.get_f64("attack_time").unwrap_or(0.0),
            release_time: a.get_f64("release_time").unwrap_or(0.0),
            pitch_contour: bytes_to_f64_vec(a.get_bytes("pitch_contour").unwrap_or(&[])),
            amplitude_contour: bytes_to_f64_vec(a.get_bytes("amplitude_contour").unwrap_or(&[])),
        })
    }
}

// ---------------------------------------------------------------------------
// Serializable element
// ---------------------------------------------------------------------------

/// A track element as stored in the project file.
#[derive(Debug, Clone)]
pub struct SerializedElement {
    pub id: u64,
    pub name: String,
    pub time_offset: f64,
    pub duration: f64,
    /// Relative path to source audio file (relative to project file location).
    pub source_path: Option<String>,
    pub muted: bool,
    pub gain: f64,
    pub notes: Vec<SerializedNote>,
}

impl SerializedElement {
    fn to_archive(&self) -> BinaryKeyValueArchive {
        let mut a = BinaryKeyValueArchive::new();
        a.set_i64("id", self.id as i64);
        a.set_string("name", &self.name);
        a.set_f64("time_offset", self.time_offset);
        a.set_f64("duration", self.duration);
        if let Some(ref path) = self.source_path {
            a.set_string("source_path", path);
        }
        a.set_bool("muted", self.muted);
        a.set_f64("gain", self.gain);
        a.set_i32("note_count", self.notes.len() as i32);
        for (i, note) in self.notes.iter().enumerate() {
            a.set_nested(&format!("note_{i}"), note.to_archive());
        }
        a
    }

    fn from_archive(a: &BinaryKeyValueArchive) -> ProjectResult<Self> {
        let note_count = a.get_i32("note_count").unwrap_or(0) as usize;
        let mut notes = Vec::with_capacity(note_count);
        for i in 0..note_count {
            if let Ok(nested) = a.get_nested(&format!("note_{i}")) {
                notes.push(SerializedNote::from_archive(nested)?);
            }
        }
        Ok(Self {
            id: a.get_i64("id").unwrap_or(0) as u64,
            name: a.get_string("name").unwrap_or("").to_string(),
            time_offset: a.get_f64("time_offset").unwrap_or(0.0),
            duration: a.get_f64("duration").unwrap_or(0.0),
            source_path: a.get_string("source_path").ok().map(|s| s.to_string()),
            muted: a.get_bool("muted").unwrap_or(false),
            gain: a.get_f64("gain").unwrap_or(1.0),
            notes,
        })
    }
}

// ---------------------------------------------------------------------------
// Serializable track
// ---------------------------------------------------------------------------

/// A track as stored in the project file.
#[derive(Debug, Clone)]
pub struct SerializedTrack {
    pub id: u64,
    pub name: String,
    pub color: [f32; 3],
    pub solo: bool,
    pub mute: bool,
    pub volume: f64,
    pub pan: f64,
    pub height: f32,
    pub elements: Vec<SerializedElement>,
}

impl SerializedTrack {
    fn to_archive(&self) -> BinaryKeyValueArchive {
        let mut a = BinaryKeyValueArchive::new();
        a.set_i64("id", self.id as i64);
        a.set_string("name", &self.name);
        a.set_f32("color_r", self.color[0]);
        a.set_f32("color_g", self.color[1]);
        a.set_f32("color_b", self.color[2]);
        a.set_bool("solo", self.solo);
        a.set_bool("mute", self.mute);
        a.set_f64("volume", self.volume);
        a.set_f64("pan", self.pan);
        a.set_f32("height", self.height);
        a.set_i32("element_count", self.elements.len() as i32);
        for (i, elem) in self.elements.iter().enumerate() {
            a.set_nested(&format!("element_{i}"), elem.to_archive());
        }
        a
    }

    fn from_archive(a: &BinaryKeyValueArchive) -> ProjectResult<Self> {
        let elem_count = a.get_i32("element_count").unwrap_or(0) as usize;
        let mut elements = Vec::with_capacity(elem_count);
        for i in 0..elem_count {
            if let Ok(nested) = a.get_nested(&format!("element_{i}")) {
                elements.push(SerializedElement::from_archive(nested)?);
            }
        }
        Ok(Self {
            id: a.get_i64("id").unwrap_or(0) as u64,
            name: a.get_string("name").unwrap_or("").to_string(),
            color: [
                a.get_f32("color_r").unwrap_or(0.5),
                a.get_f32("color_g").unwrap_or(0.7),
                a.get_f32("color_b").unwrap_or(1.0),
            ],
            solo: a.get_bool("solo").unwrap_or(false),
            mute: a.get_bool("mute").unwrap_or(false),
            volume: a.get_f64("volume").unwrap_or(1.0),
            pan: a.get_f64("pan").unwrap_or(0.0),
            height: a.get_f32("height").unwrap_or(100.0),
            elements,
        })
    }
}

// ---------------------------------------------------------------------------
// Serializable composition
// ---------------------------------------------------------------------------

/// A tempo mark in the serialized composition.
#[derive(Debug, Clone)]
pub struct SerializedTempoMark {
    pub beat_position: f64,
    pub bpm: f64,
}

/// Musical composition data as stored in the project file.
#[derive(Debug, Clone)]
pub struct SerializedComposition {
    pub tempo: f64,
    pub time_sig_numerator: u32,
    pub time_sig_denominator: u32,
    pub key_root: u8,
    pub key_minor: bool,
    pub tempo_changes: Vec<SerializedTempoMark>,
    pub duration: f64,
    pub sample_rate: f64,
}

impl Default for SerializedComposition {
    fn default() -> Self {
        Self {
            tempo: 120.0,
            time_sig_numerator: 4,
            time_sig_denominator: 4,
            key_root: 0,
            key_minor: false,
            tempo_changes: Vec::new(),
            duration: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl SerializedComposition {
    fn to_archive(&self) -> BinaryKeyValueArchive {
        let mut a = BinaryKeyValueArchive::new();
        a.set_f64("tempo", self.tempo);
        a.set_i32("time_sig_num", self.time_sig_numerator as i32);
        a.set_i32("time_sig_den", self.time_sig_denominator as i32);
        a.set_i32("key_root", self.key_root as i32);
        a.set_bool("key_minor", self.key_minor);
        a.set_f64("duration", self.duration);
        a.set_f64("sample_rate", self.sample_rate);
        a.set_i32("tempo_change_count", self.tempo_changes.len() as i32);
        for (i, tc) in self.tempo_changes.iter().enumerate() {
            let mut tca = BinaryKeyValueArchive::new();
            tca.set_f64("beat_position", tc.beat_position);
            tca.set_f64("bpm", tc.bpm);
            a.set_nested(&format!("tempo_change_{i}"), tca);
        }
        a
    }

    fn from_archive(a: &BinaryKeyValueArchive) -> ProjectResult<Self> {
        let tc_count = a.get_i32("tempo_change_count").unwrap_or(0) as usize;
        let mut tempo_changes = Vec::with_capacity(tc_count);
        for i in 0..tc_count {
            if let Ok(nested) = a.get_nested(&format!("tempo_change_{i}")) {
                tempo_changes.push(SerializedTempoMark {
                    beat_position: nested.get_f64("beat_position").unwrap_or(0.0),
                    bpm: nested.get_f64("bpm").unwrap_or(120.0),
                });
            }
        }
        Ok(Self {
            tempo: a.get_f64("tempo").unwrap_or(120.0),
            time_sig_numerator: a.get_i32("time_sig_num").unwrap_or(4) as u32,
            time_sig_denominator: a.get_i32("time_sig_den").unwrap_or(4) as u32,
            key_root: a.get_i32("key_root").unwrap_or(0) as u8,
            key_minor: a.get_bool("key_minor").unwrap_or(false),
            tempo_changes,
            duration: a.get_f64("duration").unwrap_or(0.0),
            sample_rate: a.get_f64("sample_rate").unwrap_or(44100.0),
        })
    }
}

// ---------------------------------------------------------------------------
// Project file (.mpd)
// ---------------------------------------------------------------------------

/// A complete project file, stored as a `FileArchive` with entries for
/// metadata, tracks, and composition.
#[derive(Debug, Clone)]
pub struct ProjectFile {
    pub metadata: ProjectMetadata,
    pub tracks: Vec<SerializedTrack>,
    pub composition: SerializedComposition,
}

impl ProjectFile {
    /// Create a new empty project file.
    pub fn new(name: &str) -> Self {
        Self {
            metadata: ProjectMetadata {
                name: name.to_string(),
                ..Default::default()
            },
            tracks: Vec::new(),
            composition: SerializedComposition::default(),
        }
    }

    /// Save the project to a writer using the file archive container format.
    pub fn save<W: Write>(&self, writer: &mut W) -> ProjectResult<()> {
        let mut archive = FileArchive::new();

        // Metadata entry.
        let meta_bytes = self.metadata.to_archive().to_bytes()?;
        archive.add_entry(ENTRY_METADATA, &meta_bytes, false);

        // Tracks entry: a wrapper archive with a track count + per-track nested archives.
        let mut tracks_archive = BinaryKeyValueArchive::new();
        tracks_archive.set_i32("track_count", self.tracks.len() as i32);
        for (i, track) in self.tracks.iter().enumerate() {
            tracks_archive.set_nested(&format!("track_{i}"), track.to_archive());
        }
        let tracks_bytes = tracks_archive.to_bytes()?;
        archive.add_entry(ENTRY_TRACKS, &tracks_bytes, true);

        // Composition entry.
        let comp_bytes = self.composition.to_archive().to_bytes()?;
        archive.add_entry(ENTRY_COMPOSITION, &comp_bytes, false);

        archive.write_to(writer)?;
        Ok(())
    }

    /// Save to a byte vector.
    pub fn to_bytes(&self) -> ProjectResult<Vec<u8>> {
        let mut buf = Vec::new();
        self.save(&mut buf)?;
        Ok(buf)
    }

    /// Load a project from a seekable reader.
    pub fn load<R: Read + Seek>(reader: &mut R) -> ProjectResult<Self> {
        let archive = FileArchive::read_from(reader)?;

        // Metadata.
        let meta_bytes = archive.read_entry(ENTRY_METADATA)?;
        let meta_archive = BinaryKeyValueArchive::from_bytes(&meta_bytes)?;
        let metadata = ProjectMetadata::from_archive(&meta_archive)?;

        if metadata.version > PROJECT_VERSION {
            return Err(ProjectError::InvalidVersion(metadata.version));
        }

        // Tracks.
        let tracks_bytes = archive.read_entry(ENTRY_TRACKS)?;
        let tracks_archive = BinaryKeyValueArchive::from_bytes(&tracks_bytes)?;
        let track_count = tracks_archive.get_i32("track_count").unwrap_or(0) as usize;
        let mut tracks = Vec::with_capacity(track_count);
        for i in 0..track_count {
            if let Ok(nested) = tracks_archive.get_nested(&format!("track_{i}")) {
                tracks.push(SerializedTrack::from_archive(nested)?);
            }
        }

        // Composition.
        let comp_bytes = archive.read_entry(ENTRY_COMPOSITION)?;
        let comp_archive = BinaryKeyValueArchive::from_bytes(&comp_bytes)?;
        let composition = SerializedComposition::from_archive(&comp_archive)?;

        Ok(Self {
            metadata,
            tracks,
            composition,
        })
    }

    /// Load from a byte slice.
    pub fn from_bytes(data: &[u8]) -> ProjectResult<Self> {
        let mut cursor = Cursor::new(data);
        Self::load(&mut cursor)
    }
}

// ---------------------------------------------------------------------------
// Detection data file (.mdd)
// ---------------------------------------------------------------------------

/// Entry names for detection data.
const MDD_ENTRY_HEADER: &str = "header";
const MDD_ENTRY_NOTES: &str = "notes";
const MDD_ENTRY_ANALYSIS: &str = "analysis";

/// Pre-computed detection/analysis data linked to a source audio file by content hash.
#[derive(Debug, Clone)]
pub struct DetectionData {
    /// Content hash of the source audio (hex-encoded MD5).
    pub source_hash: String,
    /// Original file name of the audio source.
    pub source_filename: String,
    /// Sample rate of the source audio.
    pub sample_rate: f64,
    /// Duration of the source audio in seconds.
    pub duration: f64,
    /// Number of channels in the source audio.
    pub channel_count: u32,
    /// Detected notes from the analysis.
    pub notes: Vec<SerializedNote>,
    /// Analysis parameters used (preset name, etc.).
    pub analysis_preset: String,
    /// Spectral analysis frame times (seconds).
    pub frame_times: Vec<f64>,
    /// Detected tempo in BPM (0.0 if not detected).
    pub detected_tempo: f64,
    /// Detected key root (0-11, or 255 if not detected).
    pub detected_key_root: u8,
    /// Whether the detected key is minor.
    pub detected_key_minor: bool,
}

impl DetectionData {
    /// Create new detection data for a source audio file.
    pub fn new(source_hash: &str, source_filename: &str) -> Self {
        Self {
            source_hash: source_hash.to_string(),
            source_filename: source_filename.to_string(),
            sample_rate: 44100.0,
            duration: 0.0,
            channel_count: 1,
            notes: Vec::new(),
            analysis_preset: "Automatic".to_string(),
            frame_times: Vec::new(),
            detected_tempo: 0.0,
            detected_key_root: 255,
            detected_key_minor: false,
        }
    }

    /// Save detection data to a writer.
    pub fn save<W: Write>(&self, writer: &mut W) -> ProjectResult<()> {
        let mut archive = FileArchive::new();

        // Header.
        let mut header = BinaryKeyValueArchive::new();
        header.set_string("source_hash", &self.source_hash);
        header.set_string("source_filename", &self.source_filename);
        header.set_f64("sample_rate", self.sample_rate);
        header.set_f64("duration", self.duration);
        header.set_i32("channel_count", self.channel_count as i32);
        header.set_string("analysis_preset", &self.analysis_preset);
        header.set_f64("detected_tempo", self.detected_tempo);
        header.set_i32("detected_key_root", self.detected_key_root as i32);
        header.set_bool("detected_key_minor", self.detected_key_minor);
        let header_bytes = header.to_bytes()?;
        archive.add_entry(MDD_ENTRY_HEADER, &header_bytes, false);

        // Notes.
        let mut notes_archive = BinaryKeyValueArchive::new();
        notes_archive.set_i32("note_count", self.notes.len() as i32);
        for (i, note) in self.notes.iter().enumerate() {
            notes_archive.set_nested(&format!("note_{i}"), note.to_archive());
        }
        let notes_bytes = notes_archive.to_bytes()?;
        archive.add_entry(MDD_ENTRY_NOTES, &notes_bytes, true);

        // Analysis frame times.
        let mut analysis = BinaryKeyValueArchive::new();
        analysis.set_bytes("frame_times", f64_vec_to_bytes(&self.frame_times));
        let analysis_bytes = analysis.to_bytes()?;
        archive.add_entry(MDD_ENTRY_ANALYSIS, &analysis_bytes, true);

        archive.write_to(writer)?;
        Ok(())
    }

    /// Save to a byte vector.
    pub fn to_bytes(&self) -> ProjectResult<Vec<u8>> {
        let mut buf = Vec::new();
        self.save(&mut buf)?;
        Ok(buf)
    }

    /// Load detection data from a seekable reader.
    pub fn load<R: Read + Seek>(reader: &mut R) -> ProjectResult<Self> {
        let archive = FileArchive::read_from(reader)?;

        let header_bytes = archive.read_entry(MDD_ENTRY_HEADER)?;
        let header = BinaryKeyValueArchive::from_bytes(&header_bytes)?;

        let source_hash = header.get_string("source_hash").unwrap_or("").to_string();
        let source_filename = header
            .get_string("source_filename")
            .unwrap_or("")
            .to_string();
        let sample_rate = header.get_f64("sample_rate").unwrap_or(44100.0);
        let duration = header.get_f64("duration").unwrap_or(0.0);
        let channel_count = header.get_i32("channel_count").unwrap_or(1) as u32;
        let analysis_preset = header
            .get_string("analysis_preset")
            .unwrap_or("Automatic")
            .to_string();
        let detected_tempo = header.get_f64("detected_tempo").unwrap_or(0.0);
        let detected_key_root = header.get_i32("detected_key_root").unwrap_or(255) as u8;
        let detected_key_minor = header.get_bool("detected_key_minor").unwrap_or(false);

        // Notes.
        let notes_bytes = archive.read_entry(MDD_ENTRY_NOTES)?;
        let notes_archive = BinaryKeyValueArchive::from_bytes(&notes_bytes)?;
        let note_count = notes_archive.get_i32("note_count").unwrap_or(0) as usize;
        let mut notes = Vec::with_capacity(note_count);
        for i in 0..note_count {
            if let Ok(nested) = notes_archive.get_nested(&format!("note_{i}")) {
                notes.push(SerializedNote::from_archive(nested)?);
            }
        }

        // Analysis.
        let analysis_bytes = archive.read_entry(MDD_ENTRY_ANALYSIS)?;
        let analysis = BinaryKeyValueArchive::from_bytes(&analysis_bytes)?;
        let frame_times = bytes_to_f64_vec(analysis.get_bytes("frame_times").unwrap_or(&[]));

        Ok(Self {
            source_hash,
            source_filename,
            sample_rate,
            duration,
            channel_count,
            notes,
            analysis_preset,
            frame_times,
            detected_tempo,
            detected_key_root,
            detected_key_minor,
        })
    }

    /// Load from a byte slice.
    pub fn from_bytes(data: &[u8]) -> ProjectResult<Self> {
        let mut cursor = Cursor::new(data);
        Self::load(&mut cursor)
    }

    /// Verify that this detection data matches a given audio source hash.
    pub fn verify_hash(&self, audio_data: &[u8]) -> ProjectResult<bool> {
        let hash = compute_audio_hash(audio_data);
        if hash != self.source_hash {
            return Err(ProjectError::HashMismatch {
                expected: self.source_hash.clone(),
                actual: hash,
            });
        }
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Audio content hashing
// ---------------------------------------------------------------------------

/// Compute the MD5 hash of audio data, returning a hex string.
pub fn compute_audio_hash(data: &[u8]) -> String {
    let mut hasher = Md5::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex_encode(&result)
}

/// Make a path relative to a base directory.
pub fn make_relative_path(path: &Path, base: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix(base) {
        stripped.to_path_buf()
    } else {
        path.to_path_buf()
    }
}

/// Resolve a relative path against a base directory.
pub fn resolve_path(relative: &Path, base: &Path) -> PathBuf {
    if relative.is_absolute() {
        relative.to_path_buf()
    } else {
        base.join(relative)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

fn f64_vec_to_bytes(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn bytes_to_f64_vec(data: &[u8]) -> Vec<f64> {
    let count = data.len() / 8;
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * 8;
        if offset + 8 <= data.len() {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&data[offset..offset + 8]);
            values.push(f64::from_le_bytes(buf));
        }
    }
    values
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_note(freq: f64, start: f64) -> SerializedNote {
        SerializedNote {
            frequency: freq,
            midi_note: 69.0 + 12.0 * (freq / 440.0).log2(),
            start_time: start,
            duration: 0.5,
            start_sample: (start * 44100.0) as u64,
            end_sample: ((start + 0.5) * 44100.0) as u64,
            amplitude: 0.6,
            peak_amplitude: 0.8,
            confidence: 0.95,
            vibrato_rate: 5.5,
            vibrato_depth: 15.0,
            attack_time: 0.02,
            release_time: 0.05,
            pitch_contour: vec![freq * 0.99, freq, freq * 1.01, freq, freq * 0.995],
            amplitude_contour: vec![0.1, 0.5, 0.8, 0.6, 0.2],
        }
    }

    fn make_test_project() -> ProjectFile {
        let mut project = ProjectFile::new("Test Project");
        project.metadata.author = "Test Author".to_string();
        project.metadata.sample_rate = 48000.0;
        project.metadata.created_at = 1700000000;
        project.metadata.modified_at = 1700001000;

        project.composition.tempo = 140.0;
        project.composition.time_sig_numerator = 3;
        project.composition.time_sig_denominator = 4;
        project.composition.key_root = 7; // G
        project.composition.key_minor = true;
        project.composition.duration = 180.0;
        project.composition.tempo_changes.push(SerializedTempoMark {
            beat_position: 32.0,
            bpm: 160.0,
        });

        let track = SerializedTrack {
            id: 1,
            name: "Vocals".to_string(),
            color: [0.9, 0.2, 0.3],
            solo: false,
            mute: false,
            volume: 0.85,
            pan: -0.1,
            height: 120.0,
            elements: vec![SerializedElement {
                id: 10,
                name: "Verse 1".to_string(),
                time_offset: 2.0,
                duration: 30.0,
                source_path: Some("audio/vocals.wav".to_string()),
                muted: false,
                gain: 1.0,
                notes: vec![
                    make_test_note(440.0, 2.0),
                    make_test_note(494.0, 2.5),
                    make_test_note(523.0, 3.0),
                ],
            }],
        };
        project.tracks.push(track);

        let track2 = SerializedTrack {
            id: 2,
            name: "Bass".to_string(),
            color: [0.2, 0.5, 0.9],
            solo: false,
            mute: true,
            volume: 1.0,
            pan: 0.0,
            height: 80.0,
            elements: vec![],
        };
        project.tracks.push(track2);

        project
    }

    #[test]
    fn test_project_save_load_roundtrip() {
        let project = make_test_project();
        let bytes = project.to_bytes().unwrap();
        let loaded = ProjectFile::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.metadata.name, "Test Project");
        assert_eq!(loaded.metadata.author, "Test Author");
        assert!((loaded.metadata.sample_rate - 48000.0).abs() < 1e-10);
        assert_eq!(loaded.metadata.created_at, 1700000000);
        assert_eq!(loaded.metadata.version, PROJECT_VERSION);

        assert_eq!(loaded.tracks.len(), 2);
        let t1 = &loaded.tracks[0];
        assert_eq!(t1.name, "Vocals");
        assert_eq!(t1.id, 1);
        assert!((t1.color[0] - 0.9).abs() < 1e-5);
        assert!((t1.volume - 0.85).abs() < 1e-10);
        assert!(!t1.mute);
        assert_eq!(t1.elements.len(), 1);

        let elem = &t1.elements[0];
        assert_eq!(elem.name, "Verse 1");
        assert!((elem.time_offset - 2.0).abs() < 1e-10);
        assert_eq!(elem.source_path.as_deref(), Some("audio/vocals.wav"));
        assert_eq!(elem.notes.len(), 3);

        let note = &elem.notes[0];
        assert!((note.frequency - 440.0).abs() < 1e-10);
        assert_eq!(note.pitch_contour.len(), 5);
        assert_eq!(note.amplitude_contour.len(), 5);

        let t2 = &loaded.tracks[1];
        assert_eq!(t2.name, "Bass");
        assert!(t2.mute);
        assert!(t2.elements.is_empty());

        assert!((loaded.composition.tempo - 140.0).abs() < 1e-10);
        assert_eq!(loaded.composition.time_sig_numerator, 3);
        assert_eq!(loaded.composition.key_root, 7);
        assert!(loaded.composition.key_minor);
        assert_eq!(loaded.composition.tempo_changes.len(), 1);
        assert!((loaded.composition.tempo_changes[0].bpm - 160.0).abs() < 1e-10);
    }

    #[test]
    fn test_detection_data_save_load_roundtrip() {
        let audio_data = b"fake audio content for hashing";
        let hash = compute_audio_hash(audio_data);

        let mut dd = DetectionData::new(&hash, "vocals.wav");
        dd.sample_rate = 48000.0;
        dd.duration = 60.0;
        dd.channel_count = 2;
        dd.analysis_preset = "Melodic".to_string();
        dd.detected_tempo = 128.0;
        dd.detected_key_root = 0; // C
        dd.detected_key_minor = false;
        dd.frame_times = vec![0.0, 0.023, 0.046, 0.069];
        dd.notes.push(make_test_note(440.0, 1.0));
        dd.notes.push(make_test_note(880.0, 2.0));

        let bytes = dd.to_bytes().unwrap();
        let loaded = DetectionData::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.source_hash, hash);
        assert_eq!(loaded.source_filename, "vocals.wav");
        assert!((loaded.sample_rate - 48000.0).abs() < 1e-10);
        assert!((loaded.duration - 60.0).abs() < 1e-10);
        assert_eq!(loaded.channel_count, 2);
        assert_eq!(loaded.analysis_preset, "Melodic");
        assert!((loaded.detected_tempo - 128.0).abs() < 1e-10);
        assert_eq!(loaded.detected_key_root, 0);
        assert!(!loaded.detected_key_minor);
        assert_eq!(loaded.frame_times.len(), 4);
        assert!((loaded.frame_times[2] - 0.046).abs() < 1e-10);
        assert_eq!(loaded.notes.len(), 2);
        assert!((loaded.notes[0].frequency - 440.0).abs() < 1e-10);
        assert!((loaded.notes[1].frequency - 880.0).abs() < 1e-10);
    }

    #[test]
    fn test_hash_based_audio_matching() {
        let audio_data = b"some audio samples here";
        let hash = compute_audio_hash(audio_data);

        let dd = DetectionData::new(&hash, "test.wav");
        assert!(dd.verify_hash(audio_data).unwrap());

        let different_audio = b"different audio data";
        assert!(dd.verify_hash(different_audio).is_err());
    }

    #[test]
    fn test_compute_audio_hash_deterministic() {
        let data = b"deterministic hash test data";
        let h1 = compute_audio_hash(data);
        let h2 = compute_audio_hash(data);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 32); // MD5 = 16 bytes = 32 hex chars.
    }

    #[test]
    fn test_f64_vec_roundtrip() {
        let values = vec![1.0, -2.5, std::f64::consts::PI, 0.0, f64::MAX];
        let bytes = f64_vec_to_bytes(&values);
        let loaded = bytes_to_f64_vec(&bytes);
        assert_eq!(values.len(), loaded.len());
        for (a, b) in values.iter().zip(loaded.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_relative_path_utilities() {
        let base = Path::new("/home/user/projects/song");
        let full = Path::new("/home/user/projects/song/audio/vocals.wav");
        let relative = make_relative_path(full, base);
        assert_eq!(relative, PathBuf::from("audio/vocals.wav"));

        let resolved = resolve_path(&relative, base);
        assert_eq!(
            resolved,
            PathBuf::from("/home/user/projects/song/audio/vocals.wav")
        );
    }

    #[test]
    fn test_empty_project_roundtrip() {
        let project = ProjectFile::new("Empty");
        let bytes = project.to_bytes().unwrap();
        let loaded = ProjectFile::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.metadata.name, "Empty");
        assert!(loaded.tracks.is_empty());
    }

    #[test]
    fn test_serialized_note_roundtrip() {
        let note = make_test_note(660.0, 5.0);
        let archive = note.to_archive();
        let bytes = archive.to_bytes().unwrap();
        let loaded_archive = BinaryKeyValueArchive::from_bytes(&bytes).unwrap();
        let loaded = SerializedNote::from_archive(&loaded_archive).unwrap();

        assert!((loaded.frequency - 660.0).abs() < 1e-10);
        assert!((loaded.start_time - 5.0).abs() < 1e-10);
        assert!((loaded.duration - 0.5).abs() < 1e-10);
        assert_eq!(loaded.pitch_contour.len(), 5);
    }

    #[test]
    fn test_detection_data_empty() {
        let dd = DetectionData::new("abc123", "empty.wav");
        let bytes = dd.to_bytes().unwrap();
        let loaded = DetectionData::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.source_hash, "abc123");
        assert!(loaded.notes.is_empty());
        assert!(loaded.frame_times.is_empty());
    }
}
