//! ARA (Audio Random Access) plugin interface skeleton.
//!
//! Defines the trait-based interface for ARA integration, corresponding to
//! the MUAra* class hierarchy in the original Melodyne engine.
//!
//! ARA allows the host DAW to share audio data and editing state with the
//! plugin for tight integration (e.g., Logic Pro, Studio One, Cubase).

use melodyn_core::{AnalysisConfig, AnalysisResult, DetectedNote};

// ---------------------------------------------------------------------------
// ARA Document Controller
// ---------------------------------------------------------------------------

/// The document controller manages the overall plugin state.
///
/// Corresponds to MUAraDocumentController.
pub trait AraDocumentController {
    /// Called when a new document is created or loaded.
    fn begin_editing(&mut self);

    /// Called when editing is complete.
    fn end_editing(&mut self);

    /// Get the current analysis configuration.
    fn analysis_config(&self) -> &AnalysisConfig;

    /// Set the analysis configuration.
    fn set_analysis_config(&mut self, config: AnalysisConfig);
}

// ---------------------------------------------------------------------------
// ARA Audio Source
// ---------------------------------------------------------------------------

/// Represents an audio source registered with the host.
///
/// Corresponds to MUAraAudioSource.
pub trait AraAudioSource {
    /// Unique identifier for this audio source.
    fn id(&self) -> u64;

    /// Name of the audio source.
    fn name(&self) -> &str;

    /// Sample rate of the source audio.
    fn sample_rate(&self) -> f64;

    /// Number of channels.
    fn channel_count(&self) -> usize;

    /// Total number of samples.
    fn sample_count(&self) -> usize;

    /// Read samples from the source into the buffer.
    /// Returns the number of samples actually read.
    fn read_samples(
        &self,
        channel: usize,
        start_sample: usize,
        buffer: &mut [f64],
    ) -> usize;

    /// Get the analysis result for this source (if analyzed).
    fn analysis_result(&self) -> Option<&AnalysisResult>;

    /// Get the detected notes (if analyzed).
    fn notes(&self) -> &[DetectedNote];
}

// ---------------------------------------------------------------------------
// ARA Audio Modification
// ---------------------------------------------------------------------------

/// An edit applied to an audio source.
///
/// Corresponds to MUAraAudioModification.
pub trait AraAudioModification {
    /// The audio source this modification applies to.
    fn source_id(&self) -> u64;

    /// Whether this modification changes the pitch.
    fn has_pitch_edit(&self) -> bool;

    /// Whether this modification changes timing.
    fn has_time_edit(&self) -> bool;

    /// Render modified audio into the buffer.
    /// Returns the number of samples rendered.
    fn render(
        &self,
        channel: usize,
        start_sample: usize,
        buffer: &mut [f64],
    ) -> usize;
}

// ---------------------------------------------------------------------------
// ARA Playback Region
// ---------------------------------------------------------------------------

/// A region of audio rendered to the host's timeline.
///
/// Corresponds to MUAraAudioPlaybackRegion.
pub trait AraPlaybackRegion {
    /// The modification this region uses.
    fn modification_id(&self) -> u64;

    /// Start time on the host timeline (seconds).
    fn start_in_playback(&self) -> f64;

    /// Duration on the host timeline (seconds).
    fn duration_in_playback(&self) -> f64;

    /// Start offset within the modification (seconds).
    fn start_in_modification(&self) -> f64;
}

// ---------------------------------------------------------------------------
// Placeholder implementation
// ---------------------------------------------------------------------------

/// Placeholder document controller for initial development.
pub struct PlaceholderDocumentController {
    config: AnalysisConfig,
}

impl PlaceholderDocumentController {
    /// Create a new placeholder controller.
    pub fn new() -> Self {
        Self {
            config: AnalysisConfig::default(),
        }
    }
}

impl Default for PlaceholderDocumentController {
    fn default() -> Self {
        Self::new()
    }
}

impl AraDocumentController for PlaceholderDocumentController {
    fn begin_editing(&mut self) {
        log::debug!("ARA: begin editing");
    }

    fn end_editing(&mut self) {
        log::debug!("ARA: end editing");
    }

    fn analysis_config(&self) -> &AnalysisConfig {
        &self.config
    }

    fn set_analysis_config(&mut self, config: AnalysisConfig) {
        self.config = config;
    }
}
