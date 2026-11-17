//! Window-based overlap-add (OLA) with phase continuity.
//!
//! Implements the overlap-add framework used by the LSS resynthesis engine.
//! Supports configurable analysis/synthesis windows, hop sizes, and
//! phase-locked processing for time stretching and pitch shifting.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for overlap-add processing.
#[derive(Debug, Clone)]
pub struct OverlapAddConfig {
    /// Window size in samples.
    pub window_size: usize,
    /// Hop size in samples (determines overlap factor).
    pub hop_size: usize,
    /// Analysis window coefficients.
    pub analysis_window: Vec<f64>,
    /// Synthesis window coefficients.
    pub synthesis_window: Vec<f64>,
    /// Sample rate.
    pub sample_rate: f64,
}

impl Default for OverlapAddConfig {
    fn default() -> Self {
        let window_size = 2048;
        let hop_size = 512;
        let analysis_window = hann_window(window_size);
        let synthesis_window = hann_window(window_size);
        Self {
            window_size,
            hop_size,
            analysis_window,
            synthesis_window,
            sample_rate: 44100.0,
        }
    }
}

impl OverlapAddConfig {
    /// Create a configuration with the given window and hop sizes.
    pub fn new(window_size: usize, hop_size: usize, sample_rate: f64) -> Self {
        Self {
            window_size,
            hop_size,
            analysis_window: hann_window(window_size),
            synthesis_window: hann_window(window_size),
            sample_rate,
        }
    }

    /// Overlap factor (e.g., 4 for 75% overlap with hop = window/4).
    pub fn overlap_factor(&self) -> f64 {
        self.window_size as f64 / self.hop_size as f64
    }

    /// Compute the COLA (Constant Overlap-Add) normalization factor.
    ///
    /// For perfect reconstruction, the sum of overlapping windows should
    /// be constant. This returns the normalization factor.
    pub fn cola_normalization(&self) -> f64 {
        let mut sum_at_center = 0.0;
        let steps = self.window_size.div_ceil(self.hop_size);
        for step in 0..steps {
            let offset = step * self.hop_size;
            let window_idx = self.window_size / 2;
            if let Some(idx) = window_idx.checked_sub(offset) {
                if idx < self.synthesis_window.len() {
                    sum_at_center += self.analysis_window[idx] * self.synthesis_window[idx];
                }
            }
        }
        if sum_at_center > 1e-10 {
            1.0 / sum_at_center
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

/// Generate a Hann window of the given size.
pub fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos()))
        .collect()
}

/// Generate a triangular (Bartlett) window.
pub fn triangular_window(size: usize) -> Vec<f64> {
    let half = size as f64 / 2.0;
    (0..size)
        .map(|i| 1.0 - ((i as f64 - half) / half).abs())
        .collect()
}

/// Generate a sine window (used in some MDCT implementations).
pub fn sine_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| (PI * (i as f64 + 0.5) / size as f64).sin())
        .collect()
}

// ---------------------------------------------------------------------------
// Overlap-Add processor
// ---------------------------------------------------------------------------

/// Overlap-add analysis-synthesis processor.
///
/// Breaks input audio into overlapping windowed frames, processes each
/// frame (via a user-supplied callback), and reconstructs the output
/// using overlap-add.
pub struct OverlapAddProcessor {
    config: OverlapAddConfig,
}

impl OverlapAddProcessor {
    /// Create a new OLA processor.
    pub fn new(config: OverlapAddConfig) -> Self {
        Self { config }
    }

    /// Break audio into overlapping analysis frames.
    pub fn analyze_frames(&self, audio: &[f64]) -> Vec<Vec<f64>> {
        let window_size = self.config.window_size;
        let hop_size = self.config.hop_size;
        let mut frames = Vec::new();

        let mut pos = 0;
        while pos + window_size <= audio.len() {
            let frame: Vec<f64> = audio[pos..pos + window_size]
                .iter()
                .zip(self.config.analysis_window.iter())
                .map(|(&s, &w)| s * w)
                .collect();
            frames.push(frame);
            pos += hop_size;
        }

        frames
    }

    /// Reconstruct audio from processed frames using overlap-add.
    pub fn synthesize_frames(&self, frames: &[Vec<f64>]) -> Vec<f64> {
        if frames.is_empty() {
            return vec![];
        }

        let window_size = self.config.window_size;
        let hop_size = self.config.hop_size;
        let total_len = (frames.len() - 1) * hop_size + window_size;
        let mut output = vec![0.0f64; total_len];
        let mut normalization = vec![0.0f64; total_len];

        for (frame_idx, frame) in frames.iter().enumerate() {
            let pos = frame_idx * hop_size;
            for i in 0..frame.len().min(window_size) {
                if pos + i < total_len {
                    let synth_w = if i < self.config.synthesis_window.len() {
                        self.config.synthesis_window[i]
                    } else {
                        1.0
                    };
                    output[pos + i] += frame[i] * synth_w;
                    normalization[pos + i] += synth_w * synth_w;
                }
            }
        }

        // Normalize for COLA
        for i in 0..total_len {
            if normalization[i] > 1e-10 {
                output[i] /= normalization[i];
            }
        }

        output
    }

    /// Process audio through an analysis-modify-synthesis pipeline.
    ///
    /// `process_fn` takes a windowed frame and returns a modified frame.
    pub fn process<F>(&self, audio: &[f64], process_fn: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let frames = self.analyze_frames(audio);
        let processed: Vec<Vec<f64>> = frames.iter().map(|f| process_fn(f)).collect();
        self.synthesize_frames(&processed)
    }

    /// Process audio with time stretching.
    ///
    /// `stretch_factor` > 1.0 makes the output longer (slower).
    /// Uses frame duplication/interpolation to achieve the stretch.
    pub fn time_stretch(&self, audio: &[f64], stretch_factor: f64) -> Vec<f64> {
        let frames = self.analyze_frames(audio);
        if frames.is_empty() {
            return vec![];
        }

        let num_output_frames = ((frames.len() as f64) * stretch_factor).ceil() as usize;
        let mut stretched_frames = Vec::with_capacity(num_output_frames);

        for i in 0..num_output_frames {
            let source_pos = i as f64 / stretch_factor;
            let frame_idx = source_pos.floor() as usize;
            let frac = source_pos - frame_idx as f64;

            if frame_idx + 1 < frames.len() {
                // Interpolate between adjacent frames
                let frame: Vec<f64> = frames[frame_idx]
                    .iter()
                    .zip(frames[frame_idx + 1].iter())
                    .map(|(&a, &b)| a * (1.0 - frac) + b * frac)
                    .collect();
                stretched_frames.push(frame);
            } else if frame_idx < frames.len() {
                stretched_frames.push(frames[frame_idx].clone());
            }
        }

        self.synthesize_frames(&stretched_frames)
    }

    /// Access the configuration.
    pub fn config(&self) -> &OverlapAddConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Phase-locked OLA (for better quality time stretching)
// ---------------------------------------------------------------------------

/// Phase-locked overlap-add state for a single partial.
#[derive(Debug, Clone)]
pub struct PhaseAccumulator {
    /// Current phase in radians.
    pub phase: f64,
    /// Previous frame's phase.
    pub prev_phase: f64,
    /// Expected phase advance per hop.
    pub expected_advance: f64,
}

impl PhaseAccumulator {
    /// Create a new phase accumulator for the given frequency.
    pub fn new(frequency: f64, hop_size: usize, sample_rate: f64) -> Self {
        let expected_advance = 2.0 * PI * frequency * hop_size as f64 / sample_rate;
        Self {
            phase: 0.0,
            prev_phase: 0.0,
            expected_advance,
        }
    }

    /// Update the phase accumulator with a new measured phase.
    ///
    /// Computes the instantaneous frequency from the phase difference,
    /// then advances the synthesis phase by the corrected amount.
    pub fn update(&mut self, measured_phase: f64) {
        // Phase difference
        let diff = measured_phase - self.prev_phase;
        // Remove expected advance
        let deviation = diff - self.expected_advance;
        // Wrap deviation to [-pi, pi]
        let wrapped = deviation - (deviation / (2.0 * PI)).round() * 2.0 * PI;
        // True phase advance
        let true_advance = self.expected_advance + wrapped;

        self.prev_phase = measured_phase;
        self.phase += true_advance;
    }

    /// Get the current synthesis phase.
    pub fn current_phase(&self) -> f64 {
        self.phase
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_hann_window_properties() {
        let w = hann_window(1024);
        assert_eq!(w.len(), 1024);
        assert!(w[0].abs() < 1e-10); // Zero at endpoints
        assert!((w[512] - 1.0).abs() < 0.01); // Peak at center
    }

    #[test]
    fn test_ola_roundtrip() {
        let config = OverlapAddConfig::new(1024, 256, 44100.0);
        let processor = OverlapAddProcessor::new(config);

        // Generate test signal
        let n = 44100;
        let audio: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        // Identity processing (no modification)
        let output = processor.process(&audio, |frame| frame.to_vec());

        // Check that the output roughly matches the input
        // (allowing for edge effects and windowing)
        let start = 1024; // Skip edge region
        let end = n.min(output.len()) - 1024;
        if end > start {
            let mut max_error = 0.0f64;
            for i in start..end {
                let err = (output[i] - audio[i]).abs();
                max_error = max_error.max(err);
            }
            assert!(
                max_error < 0.1,
                "OLA roundtrip max error {} should be < 0.1",
                max_error
            );
        }
    }

    #[test]
    fn test_time_stretch() {
        let config = OverlapAddConfig::new(2048, 512, 44100.0);
        let processor = OverlapAddProcessor::new(config);

        let n = 44100; // 1 second
        let audio: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        let stretched = processor.time_stretch(&audio, 2.0);
        // Output should be roughly twice as long
        assert!(
            stretched.len() > n,
            "Stretched output {} should be > {}",
            stretched.len(),
            n
        );
    }

    #[test]
    fn test_phase_accumulator() {
        let freq = 440.0;
        let hop = 512;
        let sr = 44100.0;
        let mut pa = PhaseAccumulator::new(freq, hop, sr);

        // Feed it steady-state phase advances
        let expected = 2.0 * PI * freq * hop as f64 / sr;
        for i in 1..10 {
            pa.update(expected * i as f64);
        }

        // Phase should have advanced roughly linearly
        let total_expected = expected * 9.0;
        assert!(
            (pa.current_phase() - total_expected).abs() < 0.1,
            "Phase {} should be near {}",
            pa.current_phase(),
            total_expected
        );
    }

    #[test]
    fn test_overlap_factor() {
        let config = OverlapAddConfig::new(2048, 512, 44100.0);
        assert!((config.overlap_factor() - 4.0).abs() < 1e-10);
    }
}
