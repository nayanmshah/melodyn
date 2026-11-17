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

pub mod analysis;
pub mod decomposition;
pub mod detection;
pub mod energy_claim;
pub mod fft;
pub mod note;
pub mod pitch;
pub mod spectral;

// Re-export key types at crate root for convenience
pub use analysis::{AnalysisConfig, AnalysisResult, SpectralAnalyzer};
pub use decomposition::{AudioDecomposer, DecompositionComponent, DecompositionConfig};
pub use detection::{DetectionParams, DetectionPreset};
pub use energy_claim::{EnergyClaimConfig, EnergyClaimEngine};
pub use fft::{FftEngine, StackedFft, StackedFftConfig, WindowType};
pub use note::DetectedNote;
pub use pitch::{PitchCandidate, PitchDetector, PitchDetectorConfig};
pub use spectral::{
    ReassignedSpectrogram, ReassignedSpectrogramConfig, SpectralPoint, SpectralTimeSlice,
    TonalityClass, TransiencyInfo,
};
