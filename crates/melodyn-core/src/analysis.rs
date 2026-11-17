//! Spectral analysis pipeline.
//!
//! Coordinates the complete analysis workflow: stacked multi-resolution FFT,
//! reassigned spectrogram, peak tracking, and pitch candidate extraction.
//! This module corresponds to the MUElementAnalyzerEngine in the original.

use crate::detection::DetectionPreset;
use crate::fft::{StackedFft, StackedFftConfig, StackedFftResult};
use crate::note::DetectedNote;
use crate::pitch::{PitchDetector, PitchDetectorConfig};
use crate::spectral::{ReassignedSpectrogram, ReassignedSpectrogramConfig, SpectralTimeSlice};

// ---------------------------------------------------------------------------
// Analysis configuration
// ---------------------------------------------------------------------------

/// Master configuration for the spectral analysis pipeline.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Whether to use the stacked multi-resolution FFT.
    pub use_stacked_fft: bool,
    /// Stacked FFT configuration (if enabled).
    pub stacked_fft_config: StackedFftConfig,
    /// Reassigned spectrogram configuration.
    pub reassigned_config: ReassignedSpectrogramConfig,
    /// Pitch detector configuration.
    pub pitch_config: PitchDetectorConfig,
    /// Detection preset to use.
    pub preset: DetectionPreset,
    /// Minimum note duration in seconds to keep.
    pub min_note_duration: f64,
    /// Minimum note amplitude (linear) to keep.
    pub min_note_amplitude: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            use_stacked_fft: true,
            stacked_fft_config: StackedFftConfig::default(),
            reassigned_config: ReassignedSpectrogramConfig::default(),
            pitch_config: PitchDetectorConfig::default(),
            preset: DetectionPreset::Automatic,
            min_note_duration: 0.05,
            min_note_amplitude: 0.001,
        }
    }
}

impl AnalysisConfig {
    /// Create config tuned for the given preset.
    pub fn from_preset(preset: DetectionPreset, sample_rate: f64) -> Self {
        let mut config = Self {
            sample_rate,
            preset,
            ..Default::default()
        };
        config.stacked_fft_config.sample_rate = sample_rate;
        config.reassigned_config.sample_rate = sample_rate;
        config.pitch_config.sample_rate = sample_rate;

        match preset {
            DetectionPreset::Melodic | DetectionPreset::MelodicAdjustAttacks => {
                config.pitch_config.polyphonic = false;
                config.reassigned_config.fft_size = 4096;
                config.reassigned_config.hop_size = 1024;
            }
            DetectionPreset::MelodicNoSibilants => {
                config.pitch_config.polyphonic = false;
                config.pitch_config.min_freq = 80.0;
            }
            DetectionPreset::MelodicPercussive => {
                config.pitch_config.polyphonic = false;
                config.reassigned_config.onset_threshold = 0.3;
                config.min_note_duration = 0.02;
            }
            DetectionPreset::Percussive => {
                config.pitch_config.polyphonic = false;
                config.reassigned_config.onset_threshold = 0.2;
                config.min_note_duration = 0.01;
                config.min_note_amplitude = 0.01;
            }
            DetectionPreset::Polyphonic => {
                config.pitch_config.polyphonic = true;
                config.reassigned_config.fft_size = 8192;
                config.reassigned_config.hop_size = 2048;
                config.stacked_fft_config.sizes = vec![2048, 4096, 8192, 16384];
            }
            DetectionPreset::PolyphonicUnpitched => {
                config.pitch_config.polyphonic = true;
                config.pitch_config.min_freq = 40.0;
                config.min_note_amplitude = 0.005;
            }
            DetectionPreset::Automatic => {
                // Use defaults; auto-detect mono vs poly from content
            }
        }
        config
    }
}

// ---------------------------------------------------------------------------
// Analysis results
// ---------------------------------------------------------------------------

/// Complete analysis result for an audio buffer.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Sample rate of the analyzed audio.
    pub sample_rate: f64,
    /// Total number of samples analyzed.
    pub num_samples: usize,
    /// Duration in seconds.
    pub duration: f64,
    /// All spectral time slices from the reassigned spectrogram.
    pub spectral_slices: Vec<SpectralTimeSlice>,
    /// Stacked FFT results (if stacked FFT was enabled).
    pub stacked_results: Vec<StackedFftResult>,
    /// Detected notes.
    pub notes: Vec<DetectedNote>,
    /// Whether polyphonic detection was used.
    pub is_polyphonic: bool,
    /// Preset that was used.
    pub preset: DetectionPreset,
}

// ---------------------------------------------------------------------------
// Spectral analysis pipeline
// ---------------------------------------------------------------------------

/// The main spectral analysis pipeline.
///
/// Orchestrates the full analysis: FFT computation, reassigned spectrogram,
/// peak extraction, pitch detection, and note segmentation. This corresponds
/// to the MUElementAnalyzerEngine class.
pub struct SpectralAnalyzer {
    config: AnalysisConfig,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer with the given configuration.
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Create with default config for the given sample rate.
    pub fn with_sample_rate(sample_rate: f64) -> Self {
        let mut config = AnalysisConfig {
            sample_rate,
            ..Default::default()
        };
        config.stacked_fft_config.sample_rate = sample_rate;
        config.reassigned_config.sample_rate = sample_rate;
        config.pitch_config.sample_rate = sample_rate;
        Self { config }
    }

    /// Run the complete analysis pipeline on audio data.
    pub fn analyze(&self, audio: &[f64]) -> AnalysisResult {
        let sr = self.config.sample_rate;
        let num_samples = audio.len();
        let duration = num_samples as f64 / sr;

        // Step 1: Run stacked FFT if enabled
        let stacked_results = if self.config.use_stacked_fft {
            let stacked = StackedFft::new(self.config.stacked_fft_config.clone());
            stacked.analyze_all(audio)
        } else {
            vec![]
        };

        // Step 2: Run reassigned spectrogram
        let mut reassigned = ReassignedSpectrogram::new(self.config.reassigned_config.clone());
        let spectral_slices = reassigned.analyze(audio);

        // Step 3: Run pitch detection on each frame
        let pitch_detector = PitchDetector::new(self.config.pitch_config.clone());
        let mut notes = if self.config.pitch_config.polyphonic {
            pitch_detector.detect_polyphonic(audio, &spectral_slices)
        } else {
            pitch_detector.detect_monophonic(audio, &spectral_slices)
        };

        // Step 4: Filter notes by minimum duration and amplitude
        notes.retain(|note| {
            note.duration >= self.config.min_note_duration
                && note.amplitude >= self.config.min_note_amplitude
        });

        // Step 5: Sort notes by start time
        notes.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        AnalysisResult {
            sample_rate: sr,
            num_samples,
            duration,
            spectral_slices,
            stacked_results,
            notes,
            is_polyphonic: self.config.pitch_config.polyphonic,
            preset: self.config.preset,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &AnalysisConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Auto-detection of mono vs polyphonic content
// ---------------------------------------------------------------------------

/// Heuristic to determine if audio content is likely polyphonic.
///
/// Looks at the spectral slices to see if there are consistently
/// multiple strong harmonic series present simultaneously.
pub fn detect_polyphony(slices: &[SpectralTimeSlice]) -> bool {
    if slices.is_empty() {
        return false;
    }

    let mut multi_peak_count = 0;
    let total = slices.len();

    for slice in slices {
        let peaks = slice.find_peaks(0.01);
        if peaks.len() < 3 {
            continue;
        }

        // Check for multiple independent harmonic series
        // by looking for peaks that don't form a single harmonic ladder
        let mut freqs: Vec<f64> = peaks.iter().map(|&(bin, _)| slice.bin_to_freq(bin)).collect();
        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if has_multiple_fundamentals(&freqs) {
            multi_peak_count += 1;
        }
    }

    // If more than 30% of frames show multiple fundamentals, treat as polyphonic
    multi_peak_count as f64 / total as f64 > 0.3
}

/// Check if a sorted list of frequencies likely comes from multiple
/// independent harmonic series (multiple notes).
fn has_multiple_fundamentals(freqs: &[f64]) -> bool {
    if freqs.len() < 4 {
        return false;
    }

    // Try the lowest frequency as the fundamental
    let f0 = freqs[0];
    if f0 < 20.0 {
        return false;
    }

    let mut unexplained = 0;
    for &f in &freqs[1..] {
        let ratio = f / f0;
        let nearest_harmonic = ratio.round();
        let deviation = (ratio - nearest_harmonic).abs();
        // If a peak is not near a harmonic of f0, it might be from another note
        if deviation > 0.08 && nearest_harmonic > 0.5 {
            unexplained += 1;
        }
    }

    // If more than 40% of peaks are unexplained by one harmonic series,
    // likely polyphonic
    unexplained as f64 / freqs.len() as f64 > 0.4
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_wave(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_analysis_pipeline_basic() {
        let sr = 44100.0;
        let audio = sine_wave(440.0, sr, 44100);

        let analyzer = SpectralAnalyzer::with_sample_rate(sr);
        let result = analyzer.analyze(&audio);

        assert_eq!(result.sample_rate, sr);
        assert_eq!(result.num_samples, 44100);
        assert!((result.duration - 1.0).abs() < 0.001);
        assert!(!result.spectral_slices.is_empty());
    }

    #[test]
    fn test_analysis_detects_pitch() {
        let sr = 44100.0;
        let audio = sine_wave(440.0, sr, 88200); // 2 seconds

        let mut config = AnalysisConfig::default();
        config.sample_rate = sr;
        config.stacked_fft_config.sample_rate = sr;
        config.reassigned_config.sample_rate = sr;
        config.pitch_config.sample_rate = sr;
        config.use_stacked_fft = false; // faster for test

        let analyzer = SpectralAnalyzer::new(config);
        let result = analyzer.analyze(&audio);

        assert!(
            !result.notes.is_empty(),
            "Should detect at least one note"
        );

        // The detected pitch should be near 440 Hz
        if let Some(note) = result.notes.first() {
            assert!(
                (note.frequency - 440.0).abs() < 10.0,
                "Expected ~440 Hz, got {} Hz",
                note.frequency
            );
        }
    }

    #[test]
    fn test_preset_config() {
        let config = AnalysisConfig::from_preset(DetectionPreset::Polyphonic, 44100.0);
        assert!(config.pitch_config.polyphonic);
        assert_eq!(config.reassigned_config.fft_size, 8192);
    }

    #[test]
    fn test_polyphony_detection_mono() {
        let sr = 44100.0;
        let audio = sine_wave(440.0, sr, 44100);

        let mut analyzer = ReassignedSpectrogram::new(ReassignedSpectrogramConfig {
            sample_rate: sr,
            ..Default::default()
        });
        let slices = analyzer.analyze(&audio);
        assert!(!detect_polyphony(&slices));
    }

    #[test]
    fn test_has_multiple_fundamentals() {
        // Single harmonic series: 220, 440, 660, 880
        assert!(!has_multiple_fundamentals(&[220.0, 440.0, 660.0, 880.0]));

        // Two harmonic series: 220 + harmonics mixed with 330 + harmonics
        assert!(has_multiple_fundamentals(&[
            220.0, 330.0, 440.0, 495.0, 660.0, 770.0, 880.0, 990.0
        ]));
    }
}
