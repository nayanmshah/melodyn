//! Melodyn Core -- DSP engine for spectral analysis, pitch detection, and
//! polyphonic separation.
//!
//! This crate implements the foundation of the Melodyn audio analysis pipeline:
//!
//! - **FFT**: Multi-resolution FFT with stacked analysis at multiple window sizes
//! - **Spectral**: Reassigned spectrogram with true frequency/time estimation
//! - **Analysis**: Complete spectral analysis pipeline orchestration
//! - **Pitch**: Monophonic (YIN + HPS) and polyphonic pitch detection
//! - **Energy Claim**: DNA-style polyphonic separation via energy competition
//! - **Decomposition**: 5-component audio decomposition (Tonal, Raspy, Consonants, Sibilant, Breath)
//! - **Note**: Detected note representation with full property set
//! - **Detection**: Analysis presets and configuration
//! - **Chord**: Chord detection, root analysis, and chord sequence analysis
//! - **Scale**: Scale/key detection, microtonal support, and tuning systems
//! - **Tempo**: Tempo detection, beat tracking, time signature, and groove analysis
//! - **Pitch Mapper**: Constraint-based pitch mapping and voice leading optimization
//! - **Separation**: Blind source separation (HPSS, NMF, spectral subtraction)
//! - **Multi-Mic**: Multi-microphone spatial source separation (GCC-PHAT, beamforming)
//! - **Vocal Isolation**: Vocal extraction via center-channel and harmonic masking

pub mod analysis;
pub mod chord;
pub mod decomposition;
pub mod detection;
pub mod energy_claim;
pub mod fft;
pub mod multimic;
pub mod note;
pub mod pitch;
pub mod pitch_mapper;
pub mod scale;
pub mod separation;
pub mod spectral;
pub mod tempo;
pub mod vocal_isolation;

// Re-export key types at crate root for convenience
pub use analysis::{AnalysisConfig, AnalysisResult, SpectralAnalyzer};
pub use chord::{
    ChordAnalyzer, ChordAnalyzerConfig, ChordDescription, ChordEntry, ChordFormatter,
    ChordSequence, ChordType, ChordTypeAnalyzer,
};
pub use decomposition::{AudioDecomposer, DecompositionComponent, DecompositionConfig};
pub use detection::{DetectionParams, DetectionPreset};
pub use energy_claim::{EnergyClaimConfig, EnergyClaimEngine};
pub use fft::{FftEngine, StackedFft, StackedFftConfig, WindowType};
pub use note::DetectedNote;
pub use pitch::{PitchCandidate, PitchDetector, PitchDetectorConfig};
pub use pitch_mapper::{NoteMapping, PitchMapper, PitchMapperConfig, Voicing};
pub use scale::{CandidateScale, ScaleMode, ScaleStretching, ScaleTuning};
pub use spectral::{
    ReassignedSpectrogram, ReassignedSpectrogramConfig, SpectralPoint, SpectralTimeSlice,
    TonalityClass, TransiencyInfo,
};
pub use tempo::{
    BarSignature, BeatGrid, TempoAnalysisResult, TempoAnalyzer, TempoAnalyzerConfig, TempoPeriod,
};

pub use separation::{
    HpssConfig, HpssResult, NmfConfig, NmfCostFunction, NmfInitialization, NmfResult,
    SpectralSubtractionConfig, SpectralSubtractionResult,
};
pub use multimic::{
    BeamformerConfig, MultichannelWienerConfig, MultichannelWienerResult, SpatialClusterConfig,
    SpatialClusterResult, TdoaConfig, TdoaResult,
};
pub use vocal_isolation::{
    CenterExtractionResult, SeparationQuality, SpectralMask, VocalIsolationConfig,
    VocalIsolationResult,
};
