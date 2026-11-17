//! Melodyn Plugin -- real-time audio pitch correction and ARA 2.0 interface.
//!
//! This crate provides:
//!
//! - **Plugin**: Trait-based plugin abstraction with parameters, suitable for
//!   wrapping with any SDK (VST3, CLAP, ARA, etc.). Includes the main
//!   `MelodynPlugin` processor that orchestrates analysis, correction, and synthesis.
//!
//! - **Real-time analysis** (`rt_analysis`): Ring-buffer-based pitch tracking
//!   using YIN detection on overlapping windows, providing low-latency pitch
//!   estimates for the correction engine.
//!
//! - **Pitch correction** (`correction`): Computes target pitch (nearest semitone
//!   or scale degree), applies smoothing and vibrato scaling, outputs the
//!   shift in semitones.
//!
//! - **Phase vocoder** (`phase_vocoder`): STFT-based real-time pitch shifting
//!   with phase unwrapping, bin interpolation, formant preservation, and
//!   overlap-add reconstruction.
//!
//! - **ARA 2.0** (`ara`): Full ARA protocol implementation for DAW integration
//!   (document controller, audio sources, modifications, playback regions,
//!   musical context, content readers, archive persistence).
//!
//! ## Architecture
//!
//! ```text
//! Audio Input
//!     |
//!     v
//! [RealTimeAnalysisCache] ---> detected pitch
//!     |                            |
//!     v                            v
//! [PhaseVocoder] <--- [PitchCorrectionProcessor]
//!     |                   (computes shift)
//!     v
//! Dry/Wet Mix
//!     |
//!     v
//! Audio Output
//! ```

pub mod ara;
pub mod correction;
pub mod phase_vocoder;
pub mod plugin;
pub mod rt_analysis;

// Re-export ARA types.
pub use ara::{
    AraArchive, AraAudioModification, AraAudioSource, AraDocument, AraDocumentController,
    AraMusicalContext, AraPlaybackRegion, AraRegionSequence, ArchiveChunk, BarSignature,
    ChunkType, KeySignature, KeySignatureEntry, ModelUpdateNotification, NoteEdit, SheetChord,
    StaticTuning, TempoEntry,
};

// Re-export plugin types.
pub use correction::{PitchCorrectionProcessor, ScaleConstraint};
pub use phase_vocoder::PhaseVocoder;
pub use plugin::{
    AudioBuffer, DetectionMode, MelodynPlugin, NullHost, ParamInfo, PluginHost, PluginParams,
};
pub use rt_analysis::{PitchEstimate, RealTimeAnalysisCache, RtAnalysisConfig};

/// Plugin version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Melodyn";
