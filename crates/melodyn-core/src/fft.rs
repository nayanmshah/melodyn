//! Multi-resolution FFT wrapper using rustfft.
//!
//! Provides stacked FFT analysis at multiple simultaneous window sizes
//! for different time-frequency tradeoffs, mirroring the DNA2Test_UseStackedFft
//! approach from Melodyne's spectral analysis.

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

/// Supported window function types for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Hann (raised cosine) window -- good general purpose.
    Hann,
    /// Blackman-Harris 4-term -- very low sidelobe leakage.
    BlackmanHarris,
    /// Hamming window -- slightly narrower main lobe than Hann.
    Hamming,
    /// Rectangular (no windowing).
    Rectangular,
}

/// Pre-compute a window function of the given type and length.
/// Returns a Vec<f64> of length `len`.
pub fn compute_window(window_type: WindowType, len: usize) -> Vec<f64> {
    let n = len as f64;
    (0..len)
        .map(|i| {
            let x = i as f64;
            match window_type {
                WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * x / n).cos()),
                WindowType::BlackmanHarris => {
                    let a0 = 0.35875;
                    let a1 = 0.48829;
                    let a2 = 0.14128;
                    let a3 = 0.01168;
                    a0 - a1 * (2.0 * PI * x / n).cos()
                        + a2 * (4.0 * PI * x / n).cos()
                        - a3 * (6.0 * PI * x / n).cos()
                }
                WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * x / n).cos(),
                WindowType::Rectangular => 1.0,
            }
        })
        .collect()
}

/// Compute the derivative window (time-ramped window) used for
/// reassigned spectrogram time correction.
/// d_window[i] = (i - N/2) * window[i]
pub fn compute_derivative_window(window: &[f64]) -> Vec<f64> {
    let n = window.len() as f64;
    let half = n / 2.0;
    window
        .iter()
        .enumerate()
        .map(|(i, &w)| (i as f64 - half) * w)
        .collect()
}

/// Compute the frequency-derivative window used for reassigned spectrogram
/// frequency correction. This is the derivative of the window function itself.
/// For Hann: d/dx [0.5*(1-cos(2*pi*x/N))] = (pi/N)*sin(2*pi*x/N)
pub fn compute_frequency_derivative_window(window_type: WindowType, len: usize) -> Vec<f64> {
    let n = len as f64;
    (0..len)
        .map(|i| {
            let x = i as f64;
            match window_type {
                WindowType::Hann => (PI / n) * (2.0 * PI * x / n).sin(),
                WindowType::BlackmanHarris => {
                    let a1 = 0.48829;
                    let a2 = 0.14128;
                    let a3 = 0.01168;
                    (2.0 * PI * a1 / n) * (2.0 * PI * x / n).sin()
                        - (4.0 * PI * a2 / n) * (4.0 * PI * x / n).sin()
                        + (6.0 * PI * a3 / n) * (6.0 * PI * x / n).sin()
                }
                WindowType::Hamming => (2.0 * PI * 0.46 / n) * (2.0 * PI * x / n).sin(),
                WindowType::Rectangular => 0.0,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Single-resolution FFT engine
// ---------------------------------------------------------------------------

/// A single FFT engine for one particular window size.
pub struct FftEngine {
    /// FFT size (number of bins).
    pub size: usize,
    /// Forward FFT plan (cached).
    forward: Arc<dyn Fft<f64>>,
    /// Inverse FFT plan (cached).
    inverse: Arc<dyn Fft<f64>>,
    /// Pre-computed window coefficients.
    pub window: Vec<f64>,
    /// Window type used.
    pub window_type: WindowType,
    /// Normalization factor: 1/N for inverse, 2/sum(window) for magnitude.
    pub window_sum: f64,
}

impl FftEngine {
    /// Create a new FFT engine with the given size and window type.
    pub fn new(size: usize, window_type: WindowType) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);
        let window = compute_window(window_type, size);
        let window_sum: f64 = window.iter().sum();

        Self {
            size,
            forward,
            inverse,
            window,
            window_type,
            window_sum,
        }
    }

    /// Apply the window function to real-valued input samples, returning
    /// a complex buffer ready for FFT.
    pub fn apply_window(&self, input: &[f64]) -> Vec<Complex<f64>> {
        assert!(
            input.len() >= self.size,
            "Input length {} < FFT size {}",
            input.len(),
            self.size
        );
        input
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect()
    }

    /// Compute forward FFT on windowed real input.
    /// Returns the full complex spectrum of length `size`.
    pub fn forward(&self, input: &[f64]) -> Vec<Complex<f64>> {
        let mut buffer = self.apply_window(input);
        self.forward.process(&mut buffer);
        buffer
    }

    /// Compute forward FFT on an already-windowed complex buffer (in-place).
    pub fn forward_complex_inplace(&self, buffer: &mut [Complex<f64>]) {
        assert_eq!(buffer.len(), self.size);
        self.forward.process(buffer);
    }

    /// Compute inverse FFT (in-place), including 1/N normalization.
    pub fn inverse_inplace(&self, buffer: &mut [Complex<f64>]) {
        assert_eq!(buffer.len(), self.size);
        self.inverse.process(buffer);
        let norm = 1.0 / self.size as f64;
        for c in buffer.iter_mut() {
            *c *= norm;
        }
    }

    /// Compute inverse FFT, returning real-valued output.
    pub fn inverse_real(&self, spectrum: &[Complex<f64>]) -> Vec<f64> {
        let mut buf = spectrum.to_vec();
        self.inverse_inplace(&mut buf);
        buf.iter().map(|c| c.re).collect()
    }

    /// Compute magnitude spectrum (first N/2+1 bins).
    pub fn magnitude_spectrum(spectrum: &[Complex<f64>]) -> Vec<f64> {
        let n = spectrum.len();
        let num_bins = n / 2 + 1;
        spectrum[..num_bins].iter().map(|c| c.norm()).collect()
    }

    /// Compute power spectrum in dB (first N/2+1 bins).
    pub fn power_spectrum_db(spectrum: &[Complex<f64>]) -> Vec<f64> {
        let n = spectrum.len();
        let num_bins = n / 2 + 1;
        spectrum[..num_bins]
            .iter()
            .map(|c| {
                let power = c.norm_sqr();
                if power > 1e-30 {
                    10.0 * power.log10()
                } else {
                    -300.0
                }
            })
            .collect()
    }

    /// Compute phase spectrum (first N/2+1 bins) in radians.
    pub fn phase_spectrum(spectrum: &[Complex<f64>]) -> Vec<f64> {
        let n = spectrum.len();
        let num_bins = n / 2 + 1;
        spectrum[..num_bins]
            .iter()
            .map(|c| c.im.atan2(c.re))
            .collect()
    }

    /// Convert bin index to frequency in Hz given sample rate.
    pub fn bin_to_freq(&self, bin: usize, sample_rate: f64) -> f64 {
        bin as f64 * sample_rate / self.size as f64
    }

    /// Convert frequency in Hz to (fractional) bin index.
    pub fn freq_to_bin(&self, freq: f64, sample_rate: f64) -> f64 {
        freq * self.size as f64 / sample_rate
    }
}

// ---------------------------------------------------------------------------
// Stacked multi-resolution FFT
// ---------------------------------------------------------------------------

/// Configuration for the stacked multi-resolution FFT.
#[derive(Debug, Clone)]
pub struct StackedFftConfig {
    /// FFT sizes to use, from smallest to largest.
    /// Smaller sizes give better time resolution; larger give better frequency resolution.
    pub sizes: Vec<usize>,
    /// Window type for all FFT layers.
    pub window_type: WindowType,
    /// Hop size as a fraction of the smallest FFT size (e.g. 0.25 for 75% overlap).
    pub hop_ratio: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
}

impl Default for StackedFftConfig {
    fn default() -> Self {
        Self {
            // 3-layer stack: 1024 for transients, 4096 for mid, 8192 for low-freq detail
            sizes: vec![1024, 4096, 8192],
            window_type: WindowType::BlackmanHarris,
            hop_ratio: 0.25,
            sample_rate: 44100.0,
        }
    }
}

/// Result of a single frame from one FFT layer.
#[derive(Debug, Clone)]
pub struct FftLayerResult {
    /// FFT size used for this layer.
    pub fft_size: usize,
    /// Complex spectrum (full N bins).
    pub spectrum: Vec<Complex<f64>>,
    /// Magnitude of first N/2+1 bins.
    pub magnitude: Vec<f64>,
    /// Phase of first N/2+1 bins (radians).
    pub phase: Vec<f64>,
}

/// Result of the stacked FFT for a single time frame.
#[derive(Debug, Clone)]
pub struct StackedFftResult {
    /// Per-layer FFT results, one per configured FFT size.
    pub layers: Vec<FftLayerResult>,
    /// Sample offset of the analysis frame center in the original audio.
    pub center_sample: usize,
}

/// Multi-resolution stacked FFT analyzer.
///
/// Runs multiple FFT sizes simultaneously on the same audio, mirroring the
/// DNA `UseStackedFft` approach. Smaller windows capture transient detail;
/// larger windows resolve closely-spaced harmonics.
pub struct StackedFft {
    engines: Vec<FftEngine>,
    config: StackedFftConfig,
}

impl StackedFft {
    /// Create a new stacked FFT analyzer from configuration.
    pub fn new(config: StackedFftConfig) -> Self {
        let engines = config
            .sizes
            .iter()
            .map(|&size| FftEngine::new(size, config.window_type))
            .collect();
        Self { engines, config }
    }

    /// Return the hop size in samples (based on smallest FFT size).
    pub fn hop_size(&self) -> usize {
        let min_size = self.config.sizes.iter().copied().min().unwrap_or(1024);
        (min_size as f64 * self.config.hop_ratio).round() as usize
    }

    /// Analyze a single frame centered at `center_sample`.
    /// The input `audio` must be long enough so that every FFT layer can
    /// extract its window centered at `center_sample`.
    pub fn analyze_frame(&self, audio: &[f64], center_sample: usize) -> StackedFftResult {
        let layers = self
            .engines
            .iter()
            .map(|engine| {
                let half = engine.size / 2;
                // Clamp and zero-pad if near edges
                let mut frame = vec![0.0f64; engine.size];
                for (i, frame_val) in frame.iter_mut().enumerate().take(engine.size) {
                    let sample_idx = center_sample as isize - half as isize + i as isize;
                    if sample_idx >= 0 && (sample_idx as usize) < audio.len() {
                        *frame_val = audio[sample_idx as usize];
                    }
                }
                let spectrum = engine.forward(&frame);
                let magnitude = FftEngine::magnitude_spectrum(&spectrum);
                let phase = FftEngine::phase_spectrum(&spectrum);
                FftLayerResult {
                    fft_size: engine.size,
                    spectrum,
                    magnitude,
                    phase,
                }
            })
            .collect();

        StackedFftResult {
            layers,
            center_sample,
        }
    }

    /// Analyze all frames across the entire audio buffer.
    pub fn analyze_all(&self, audio: &[f64]) -> Vec<StackedFftResult> {
        let hop = self.hop_size();
        let max_size = self.config.sizes.iter().copied().max().unwrap_or(0);
        let start = max_size / 2;
        let end = if audio.len() > max_size / 2 {
            audio.len() - max_size / 2
        } else {
            return vec![];
        };

        let mut results = Vec::new();
        let mut pos = start;
        while pos < end {
            results.push(self.analyze_frame(audio, pos));
            pos += hop;
        }
        results
    }

    /// Access the underlying FFT engines.
    pub fn engines(&self) -> &[FftEngine] {
        &self.engines
    }

    /// Access the configuration.
    pub fn config(&self) -> &StackedFftConfig {
        &self.config
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
    fn test_window_hann_endpoints() {
        let w = compute_window(WindowType::Hann, 1024);
        assert_eq!(w.len(), 1024);
        // Hann window is zero at endpoints
        assert!(w[0].abs() < 1e-10);
        // Peak near center
        assert!((w[512] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_window_blackman_harris() {
        let w = compute_window(WindowType::BlackmanHarris, 2048);
        assert_eq!(w.len(), 2048);
        // BH window has very small but nonzero endpoints
        assert!(w[0].abs() < 0.001);
        // Peak near center
        assert!(w[1024] > 0.9);
    }

    #[test]
    fn test_fft_roundtrip() {
        let engine = FftEngine::new(1024, WindowType::Rectangular);
        // Use rectangular window so roundtrip is exact
        let input: Vec<f64> = (0..1024).map(|i| (2.0 * PI * i as f64 / 1024.0).sin()).collect();
        let spectrum = engine.forward(&input);
        let reconstructed = engine.inverse_real(&spectrum);
        for (a, b) in input.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10, "Roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_fft_pure_tone_peak() {
        let size = 4096;
        let sample_rate = 44100.0;
        let freq = 440.0; // A4
        let engine = FftEngine::new(size, WindowType::Hann);

        let input: Vec<f64> = (0..size)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect();
        let spectrum = engine.forward(&input);
        let mag = FftEngine::magnitude_spectrum(&spectrum);

        // Find peak bin
        let peak_bin = mag
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let detected_freq = engine.bin_to_freq(peak_bin, sample_rate);
        assert!(
            (detected_freq - freq).abs() < 20.0,
            "Expected ~440 Hz, got {} Hz (bin {})",
            detected_freq,
            peak_bin
        );
    }

    #[test]
    fn test_stacked_fft_basic() {
        let config = StackedFftConfig {
            sizes: vec![512, 2048],
            window_type: WindowType::Hann,
            hop_ratio: 0.25,
            sample_rate: 44100.0,
        };
        let stacked = StackedFft::new(config);

        // Generate 1 second of 440 Hz sine
        let n = 44100;
        let audio: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();

        let results = stacked.analyze_all(&audio);
        assert!(!results.is_empty());
        assert_eq!(results[0].layers.len(), 2);
        assert_eq!(results[0].layers[0].fft_size, 512);
        assert_eq!(results[0].layers[1].fft_size, 2048);
    }

    #[test]
    fn test_derivative_window() {
        let w = compute_window(WindowType::Hann, 256);
        let dw = compute_derivative_window(&w);
        assert_eq!(dw.len(), 256);
        // At center (i=128), offset is 0, so dw[128] should be ~0
        assert!(dw[128].abs() < 1e-10);
    }
}
