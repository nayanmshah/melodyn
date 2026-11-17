//! Melodyn Plugin -- ARA 2.0 plugin interface.
//!
//! This crate implements the ARA (Audio Random Access) 2.0 protocol for
//! bidirectional audio data exchange between DAWs and the Melodyn engine.
//!
//! The architecture follows Melodyne's MUAra* class hierarchy:
//! - AraDocumentController: manages the plugin's document state and edit cycles
//! - AraDocument: root container for all model objects
//! - AraAudioSource: represents an audio source registered with the host
//! - AraAudioModification: an edit applied to an audio source
//! - AraPlaybackRegion: a region of audio rendered to the host's timeline
//! - AraMusicalContext: tempo, key, time signature from the host
//! - AraRegionSequence: ordered sequence of regions on a track
//! - Content readers: for notes, tempo, bar signatures, key signatures, chords, tuning
//! - Archive persistence: save/restore via chunk archive system

pub mod ara;

// Re-export key types
pub use ara::{
    AraArchive, AraAudioModification, AraAudioSource, AraDocument, AraDocumentController,
    AraMusicalContext, AraPlaybackRegion, AraRegionSequence, ArchiveChunk, BarSignature,
    ChunkType, KeySignature, KeySignatureEntry, ModelUpdateNotification, NoteEdit, SheetChord,
    StaticTuning, TempoEntry,
};

/// Plugin version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Melodyn";
