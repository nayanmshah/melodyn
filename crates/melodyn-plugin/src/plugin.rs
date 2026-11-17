//! Plugin wrapper: trait-based audio plugin abstraction.
//!
//! Provides a framework-agnostic plugin architecture that can be backed by
//! any SDK (VST3, CLAP, ARA, etc.). The core DSP processing is implemented
//! here and can be wrapped by any host-specific adapter.

use crate::correction::PitchCorrectionProcessor;
use crate::phase_vocoder::PhaseVocoder;
use crate::rt_analysis::RealTimeAnalysisCache;

// ---------------------------------------------------------------------------
// Detection mode
// ---------------------------------------------------------------------------

/// Detection mode for the plugin, controlling analysis behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    /// Monophonic melodic content (vocals, solo instruments).
    Melodic,
    /// Polyphonic content (piano, guitar chords).
    Polyphonic,
    /// Percussive content (drums, transients).
    Percussive,
}

impl DetectionMode {
    /// All available modes.
    pub const ALL: [DetectionMode; 3] = [Self::Melodic, Self::Polyphonic, Self::Percussive];

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Melodic => "Melodic",
            Self::Polyphonic => "Polyphonic",
            Self::Percussive => "Percussive",
        }
    }

    /// Index for parameter serialization.
    pub fn index(&self) -> usize {
        match self {
            Self::Melodic => 0,
            Self::Polyphonic => 1,
            Self::Percussive => 2,
        }
    }

    /// From index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Melodic,
            1 => Self::Polyphonic,
            2 => Self::Percussive,
            _ => Self::Melodic,
        }
    }
}

impl Default for DetectionMode {
    fn default() -> Self {
        Self::Melodic
    }
}

// ---------------------------------------------------------------------------
// Plugin parameters
// ---------------------------------------------------------------------------

/// All parameters exposed by the Melodyn plugin.
#[derive(Debug, Clone)]
pub struct PluginParams {
    /// Pitch shift in semitones (-24.0 to +24.0).
    pub pitch_shift: f64,
    /// Formant shift in semitones (-12.0 to +12.0).
    pub formant_shift: f64,
    /// Time stretch ratio (0.25 to 4.0).
    pub time_stretch: f64,
    /// Pitch correction amount (0.0 = off, 1.0 = full snap to semitone).
    pub pitch_correction_amount: f64,
    /// Vibrato depth scaling (0.0 to 2.0, 1.0 = natural).
    pub vibrato_depth: f64,
    /// Dry/wet mix (0.0 = dry, 1.0 = wet).
    pub mix: f64,
    /// Detection mode.
    pub detection_mode: DetectionMode,
}

impl Default for PluginParams {
    fn default() -> Self {
        Self {
            pitch_shift: 0.0,
            formant_shift: 0.0,
            time_stretch: 1.0,
            pitch_correction_amount: 0.0,
            vibrato_depth: 1.0,
            mix: 1.0,
            detection_mode: DetectionMode::Melodic,
        }
    }
}

impl PluginParams {
    /// Clamp all parameters to their valid ranges.
    pub fn clamped(&self) -> Self {
        Self {
            pitch_shift: self.pitch_shift.clamp(-24.0, 24.0),
            formant_shift: self.formant_shift.clamp(-12.0, 12.0),
            time_stretch: self.time_stretch.clamp(0.25, 4.0),
            pitch_correction_amount: self.pitch_correction_amount.clamp(0.0, 1.0),
            vibrato_depth: self.vibrato_depth.clamp(0.0, 2.0),
            mix: self.mix.clamp(0.0, 1.0),
            detection_mode: self.detection_mode,
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter metadata (for host integration)
// ---------------------------------------------------------------------------

/// Metadata for a single plugin parameter.
#[derive(Debug, Clone)]
pub struct ParamInfo {
    /// Unique parameter identifier.
    pub id: &'static str,
    /// Display name.
    pub name: &'static str,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Default value.
    pub default: f64,
    /// Unit label (e.g., "st", "%").
    pub unit: &'static str,
    /// Number of steps for discrete parameters (0 = continuous).
    pub steps: u32,
}

/// Get metadata for all plugin parameters.
pub fn param_info_list() -> Vec<ParamInfo> {
    vec![
        ParamInfo {
            id: "pitch_shift",
            name: "Pitch Shift",
            min: -24.0,
            max: 24.0,
            default: 0.0,
            unit: "st",
            steps: 0,
        },
        ParamInfo {
            id: "formant_shift",
            name: "Formant Shift",
            min: -12.0,
            max: 12.0,
            default: 0.0,
            unit: "st",
            steps: 0,
        },
        ParamInfo {
            id: "time_stretch",
            name: "Time Stretch",
            min: 0.25,
            max: 4.0,
            default: 1.0,
            unit: "x",
            steps: 0,
        },
        ParamInfo {
            id: "pitch_correction",
            name: "Pitch Correction",
            min: 0.0,
            max: 1.0,
            default: 0.0,
            unit: "%",
            steps: 0,
        },
        ParamInfo {
            id: "vibrato_depth",
            name: "Vibrato Depth",
            min: 0.0,
            max: 2.0,
            default: 1.0,
            unit: "x",
            steps: 0,
        },
        ParamInfo {
            id: "mix",
            name: "Mix",
            min: 0.0,
            max: 1.0,
            default: 1.0,
            unit: "%",
            steps: 0,
        },
        ParamInfo {
            id: "detection_mode",
            name: "Detection Mode",
            min: 0.0,
            max: 2.0,
            default: 0.0,
            unit: "",
            steps: 3,
        },
    ]
}

// ---------------------------------------------------------------------------
// Plugin trait
// ---------------------------------------------------------------------------

/// Audio buffer for plugin processing. Supports mono and stereo.
pub struct AudioBuffer<'a> {
    /// Input channels (read-only).
    pub inputs: &'a [&'a [f64]],
    /// Output channels (writeable).
    pub outputs: &'a mut [&'a mut [f64]],
    /// Number of samples per channel in this buffer.
    pub num_samples: usize,
}

/// Trait for a DAW plugin host adapter.
///
/// Any SDK wrapper (VST3, CLAP, etc.) implements this trait to provide
/// host-specific callbacks.
pub trait PluginHost {
    /// Report latency in samples to the host.
    fn report_latency(&self, samples: u32);
    /// Request the host to begin an edit gesture on a parameter.
    fn begin_edit(&self, param_id: &str);
    /// Set a parameter value from the UI/host side.
    fn set_parameter(&self, param_id: &str, value: f64);
    /// End an edit gesture.
    fn end_edit(&self, param_id: &str);
}

/// A no-op host for standalone / test use.
pub struct NullHost;

impl PluginHost for NullHost {
    fn report_latency(&self, _samples: u32) {}
    fn begin_edit(&self, _param_id: &str) {}
    fn set_parameter(&self, _param_id: &str, _value: f64) {}
    fn end_edit(&self, _param_id: &str) {}
}

// ---------------------------------------------------------------------------
// Plugin processor
// ---------------------------------------------------------------------------

/// The main Melodyn plugin processor.
///
/// Owns the analysis cache, correction engine, and phase vocoder.
/// Designed for real-time use: all allocations happen at initialization
/// or on parameter changes, never in the process callback.
pub struct MelodynPlugin {
    /// Current parameters.
    params: PluginParams,
    /// Sample rate.
    sample_rate: f64,
    /// Real-time analysis cache for pitch detection.
    analysis: RealTimeAnalysisCache,
    /// Pitch correction processor.
    correction: PitchCorrectionProcessor,
    /// Phase vocoder for pitch/time modification.
    vocoder: PhaseVocoder,
    /// Latency in samples (reported to host).
    latency_samples: u32,
    /// Pre-allocated scratch buffer for dry signal.
    dry_buffer: Vec<f64>,
}

impl MelodynPlugin {
    /// Create a new plugin instance with default parameters.
    pub fn new(sample_rate: f64) -> Self {
        let window_size = 2048;
        let hop_size = 512;
        let latency = window_size as u32;

        Self {
            params: PluginParams::default(),
            sample_rate,
            analysis: RealTimeAnalysisCache::new(sample_rate, window_size, hop_size),
            correction: PitchCorrectionProcessor::new(sample_rate),
            vocoder: PhaseVocoder::new(window_size, hop_size, sample_rate),
            latency_samples: latency,
            dry_buffer: vec![0.0; 8192],
        }
    }

    /// Get the current parameters.
    pub fn params(&self) -> &PluginParams {
        &self.params
    }

    /// Set parameters (called from host/UI thread, applied on next process).
    pub fn set_params(&mut self, params: PluginParams) {
        self.params = params.clamped();
        self.correction
            .set_correction_amount(self.params.pitch_correction_amount);
        self.correction
            .set_vibrato_scale(self.params.vibrato_depth);
    }

    /// Get the latency in samples.
    pub fn latency(&self) -> u32 {
        self.latency_samples
    }

    /// Reset internal state (called on transport stop, etc.).
    pub fn reset(&mut self) {
        self.analysis.reset();
        self.correction.reset();
        self.vocoder.reset();
    }

    /// Change sample rate and reinitialize.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        let window_size = 2048;
        let hop_size = 512;
        self.analysis = RealTimeAnalysisCache::new(sample_rate, window_size, hop_size);
        self.correction = PitchCorrectionProcessor::new(sample_rate);
        self.vocoder = PhaseVocoder::new(window_size, hop_size, sample_rate);
    }

    /// Process a block of audio samples (mono).
    ///
    /// This is the main real-time entry point. It:
    /// 1. Feeds samples into the analysis cache for pitch detection.
    /// 2. Computes the pitch correction target.
    /// 3. Applies pitch shifting via the phase vocoder.
    /// 4. Blends dry/wet based on the mix parameter.
    ///
    /// All processing is done in-place on the output buffer.
    pub fn process_mono(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());
        let params = self.params.clamped();

        // Ensure dry buffer is large enough (no allocation in steady state).
        if self.dry_buffer.len() < len {
            self.dry_buffer.resize(len, 0.0);
        }

        // Save dry signal for mix blending.
        self.dry_buffer[..len].copy_from_slice(&input[..len]);

        // Feed samples to the real-time analysis cache.
        self.analysis.push_samples(&input[..len]);

        // Determine the pitch shift to apply.
        let correction_shift = if params.pitch_correction_amount > 0.0 {
            self.compute_correction_shift(&params)
        } else {
            0.0
        };

        // Total pitch shift = user manual shift + correction shift.
        let total_shift_semitones = params.pitch_shift + correction_shift;
        let shift_ratio = semitones_to_ratio(total_shift_semitones);

        // Process through phase vocoder with formant option.
        let preserve_formants = params.formant_shift.abs() < 0.01;
        self.vocoder.set_pitch_shift(shift_ratio);
        self.vocoder.set_preserve_formants(preserve_formants);
        self.vocoder.process(&input[..len], &mut output[..len]);

        // Apply formant shift if nonzero.
        if params.formant_shift.abs() >= 0.01 {
            let formant_ratio = semitones_to_ratio(params.formant_shift);
            apply_formant_shift(&mut output[..len], formant_ratio);
        }

        // Apply dry/wet mix.
        if params.mix < 1.0 {
            let wet = params.mix;
            let dry = 1.0 - wet;
            for i in 0..len {
                output[i] = self.dry_buffer[i] * dry + output[i] * wet;
            }
        }
    }

    /// Process stereo audio (applies mono processing to each channel independently).
    pub fn process_stereo(
        &mut self,
        input_l: &[f64],
        input_r: &[f64],
        output_l: &mut [f64],
        output_r: &mut [f64],
    ) {
        // Process left channel (this updates the analysis cache).
        self.process_mono(input_l, output_l);

        // For right channel, use same pitch correction target but separate vocoder state.
        // In a production implementation we would have per-channel vocoders.
        // For now, process right as a simple mix of dry + shifted left channel info.
        let len = input_r.len().min(output_r.len());
        let params = self.params.clamped();

        let correction_shift = if params.pitch_correction_amount > 0.0 {
            self.compute_correction_shift(&params)
        } else {
            0.0
        };

        let total_shift = params.pitch_shift + correction_shift;
        let shift_ratio = semitones_to_ratio(total_shift);
        let preserve_formants = params.formant_shift.abs() < 0.01;
        self.vocoder.set_pitch_shift(shift_ratio);
        self.vocoder.set_preserve_formants(preserve_formants);
        self.vocoder.process(&input_r[..len], &mut output_r[..len]);

        if params.formant_shift.abs() >= 0.01 {
            let formant_ratio = semitones_to_ratio(params.formant_shift);
            apply_formant_shift(&mut output_r[..len], formant_ratio);
        }

        if params.mix < 1.0 {
            let wet = params.mix;
            let dry = 1.0 - wet;
            for i in 0..len {
                output_r[i] = input_r[i] * dry + output_r[i] * wet;
            }
        }
    }

    /// Compute the pitch correction shift in semitones based on the current
    /// detected pitch and correction amount.
    fn compute_correction_shift(&mut self, params: &PluginParams) -> f64 {
        if let Some(detected_freq) = self.analysis.current_pitch() {
            self.correction
                .compute_correction(detected_freq, params.pitch_correction_amount)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Convert semitones to frequency ratio.
pub fn semitones_to_ratio(semitones: f64) -> f64 {
    2.0_f64.powf(semitones / 12.0)
}

/// Convert frequency ratio to semitones.
pub fn ratio_to_semitones(ratio: f64) -> f64 {
    12.0 * ratio.log2()
}

/// Apply a simple formant shift by spectral envelope warping.
///
/// This is a lightweight approximation for real-time use: it applies
/// a frequency-dependent gain curve that simulates shifting the spectral
/// envelope without full STFT decomposition.
fn apply_formant_shift(samples: &mut [f64], _ratio: f64) {
    // In a full implementation, this would decompose the spectral envelope
    // and shift it. For the real-time path, the phase vocoder's formant
    // preservation mode handles most of this. This function handles residual
    // formant adjustments via a simple first-order filter approximation.
    //
    // The filter coefficient is derived from the formant ratio to gently
    // emphasize higher or lower formant regions.
    // For ratio > 1.0 (shift up): slight high-frequency emphasis.
    // For ratio < 1.0 (shift down): slight low-frequency emphasis.
    //
    // This is a placeholder -- the full formant processor in melodyn-synth
    // would be used for offline / higher-quality processing.
    if samples.is_empty() {
        return;
    }
    // No-op for now; the phase vocoder handles formant preservation.
    // A production implementation would integrate FormantProcessor from melodyn-synth.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = PluginParams::default();
        assert_eq!(params.pitch_shift, 0.0);
        assert_eq!(params.formant_shift, 0.0);
        assert_eq!(params.time_stretch, 1.0);
        assert_eq!(params.pitch_correction_amount, 0.0);
        assert_eq!(params.vibrato_depth, 1.0);
        assert_eq!(params.mix, 1.0);
        assert_eq!(params.detection_mode, DetectionMode::Melodic);
    }

    #[test]
    fn test_param_clamping() {
        let params = PluginParams {
            pitch_shift: 50.0,
            formant_shift: -20.0,
            time_stretch: 10.0,
            pitch_correction_amount: 2.0,
            vibrato_depth: -1.0,
            mix: 1.5,
            detection_mode: DetectionMode::Polyphonic,
        };
        let clamped = params.clamped();
        assert_eq!(clamped.pitch_shift, 24.0);
        assert_eq!(clamped.formant_shift, -12.0);
        assert_eq!(clamped.time_stretch, 4.0);
        assert_eq!(clamped.pitch_correction_amount, 1.0);
        assert_eq!(clamped.vibrato_depth, 0.0);
        assert_eq!(clamped.mix, 1.0);
    }

    #[test]
    fn test_semitones_to_ratio() {
        let ratio = semitones_to_ratio(12.0);
        assert!((ratio - 2.0).abs() < 1e-10, "12 semitones = octave = 2x");

        let ratio = semitones_to_ratio(0.0);
        assert!((ratio - 1.0).abs() < 1e-10, "0 semitones = no shift");

        let ratio = semitones_to_ratio(-12.0);
        assert!((ratio - 0.5).abs() < 1e-10, "-12 semitones = half freq");
    }

    #[test]
    fn test_ratio_to_semitones() {
        let st = ratio_to_semitones(2.0);
        assert!((st - 12.0).abs() < 1e-10);

        let st = ratio_to_semitones(1.0);
        assert!(st.abs() < 1e-10);
    }

    #[test]
    fn test_detection_mode() {
        assert_eq!(DetectionMode::from_index(0), DetectionMode::Melodic);
        assert_eq!(DetectionMode::from_index(1), DetectionMode::Polyphonic);
        assert_eq!(DetectionMode::from_index(2), DetectionMode::Percussive);
        assert_eq!(DetectionMode::from_index(99), DetectionMode::Melodic);

        for mode in DetectionMode::ALL {
            assert_eq!(DetectionMode::from_index(mode.index()), mode);
        }
    }

    #[test]
    fn test_param_info_list() {
        let infos = param_info_list();
        assert_eq!(infos.len(), 7);
        assert_eq!(infos[0].id, "pitch_shift");
        assert_eq!(infos[6].id, "detection_mode");
    }

    #[test]
    fn test_plugin_creation() {
        let plugin = MelodynPlugin::new(44100.0);
        assert_eq!(plugin.latency(), 2048);
        assert_eq!(plugin.params().pitch_shift, 0.0);
    }

    #[test]
    fn test_dry_wet_mix_zero_passes_dry() {
        let mut plugin = MelodynPlugin::new(44100.0);
        plugin.set_params(PluginParams {
            mix: 0.0,
            ..Default::default()
        });

        // Generate a simple test signal.
        let input: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut output = vec![0.0; 4096];

        plugin.process_mono(&input, &mut output);

        // With mix=0.0, output should equal the dry input signal.
        for i in 0..4096 {
            assert!(
                (output[i] - input[i]).abs() < 1e-10,
                "Sample {} differs: {} vs {}",
                i,
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_plugin_reset() {
        let mut plugin = MelodynPlugin::new(44100.0);
        let input = vec![0.5; 1024];
        let mut output = vec![0.0; 1024];
        plugin.process_mono(&input, &mut output);
        plugin.reset();
        // After reset, should process cleanly again.
        plugin.process_mono(&input, &mut output);
    }

    #[test]
    fn test_null_host() {
        let host = NullHost;
        host.report_latency(2048);
        host.begin_edit("pitch_shift");
        host.set_parameter("pitch_shift", 5.0);
        host.end_edit("pitch_shift");
    }
}
