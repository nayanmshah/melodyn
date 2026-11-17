//! Spectral time-slice representation and reassigned spectrogram.
//!
//! Models `MUSpectralTimeSlice` and `MUSpectralPointDNA3` from the original
//! Melodyne engine. Each time slice stores per-frame spectral data: complex
//! spectrum, magnitude, phase, tonality classification, transiency detection,
//! reassigned true frequencies, and reassigned time offsets.

use crate::fft::{
    compute_derivative_window, compute_frequency_derivative_window, compute_window, FftEngine,
    WindowType,
};
use num_complex::Complex;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Tonality / Transiency classification
// ---------------------------------------------------------------------------

/// Classification of a spectral bin's tonality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TonalityClass {
    /// Strongly tonal (clear harmonic partial).
    Tonal,
    /// Partially tonal with noise component.
    Mixed,
    /// Primarily noise-like.
    Noise,
    /// Transient onset energy.
    Transient,
}

/// Transiency detection result for a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransiencyInfo {
    /// Overall transiency score 0.0 .. 1.0 for this frame.
    pub score: f64,
    /// True if this frame is classified as a transient onset.
    pub is_onset: bool,
    /// Spectral flux value used to compute transiency.
    pub spectral_flux: f64,
}

// ---------------------------------------------------------------------------
// Spectral point (per-bin data)
// ---------------------------------------------------------------------------

/// Per-bin spectral data point, mirroring MUSpectralPointDNA3.
#[derive(Debug, Clone)]
pub struct SpectralPoint {
    /// Bin index in the FFT.
    pub bin: usize,
    /// Complex value from FFT.
    pub complex: Complex<f64>,
    /// Magnitude (linear).
    pub magnitude: f64,
    /// Phase in radians.
    pub phase: f64,
    /// Reassigned true frequency in Hz.
    pub true_freq: f64,
    /// Reassigned time offset in samples from frame center.
    pub true_time_offset: f64,
    /// Instantaneous frequency estimated from phase difference.
    pub instantaneous_freq: f64,
    /// Tonality classification.
    pub tonality: TonalityClass,
}

// ---------------------------------------------------------------------------
// Spectral time-slice
// ---------------------------------------------------------------------------

/// A single time frame of spectral analysis data.
///
/// This is the core data structure for the DNA analysis pipeline,
/// corresponding to MUSpectralTimeSlice. It holds all per-frame spectral
/// information needed for pitch detection, energy claiming, and decomposition.
#[derive(Debug, Clone)]
pub struct SpectralTimeSlice {
    /// Frame index in the analysis sequence.
    pub frame_index: usize,
    /// Center sample position in the original audio.
    pub center_sample: usize,
    /// Time in seconds of the frame center.
    pub time_seconds: f64,
    /// Sample rate used for analysis.
    pub sample_rate: f64,
    /// FFT size used for this slice.
    pub fft_size: usize,

    /// Full complex spectrum (N bins).
    pub spectrum: Vec<Complex<f64>>,
    /// Magnitude of first N/2+1 bins.
    pub magnitude: Vec<f64>,
    /// Phase of first N/2+1 bins (radians).
    pub phase: Vec<f64>,
    /// Power spectrum in dB.
    pub power_db: Vec<f64>,

    /// Reassigned true frequencies for each bin (Hz).
    /// Provides sub-bin frequency resolution.
    pub true_freq_data: Vec<f64>,
    /// Reassigned true time offsets for each bin (samples).
    /// Indicates where in time the energy actually occurs.
    pub true_time_sample_offset_data: Vec<f64>,
    /// Instantaneous frequency for each bin (Hz), from phase derivative.
    pub instantaneous_freq: Vec<f64>,

    /// Per-bin tonality classification.
    pub tonality: Vec<TonalityClass>,
    /// Frame-level transiency information.
    pub transiency: TransiencyInfo,

    /// Detailed per-bin spectral points (populated on demand).
    pub points: Vec<SpectralPoint>,
}

impl SpectralTimeSlice {
    /// Number of usable frequency bins (N/2 + 1).
    pub fn num_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Frequency resolution (Hz per bin).
    pub fn freq_resolution(&self) -> f64 {
        self.sample_rate / self.fft_size as f64
    }

    /// Get the frequency of a given bin index.
    pub fn bin_to_freq(&self, bin: usize) -> f64 {
        bin as f64 * self.freq_resolution()
    }

    /// Find the bin with the highest magnitude.
    pub fn peak_bin(&self) -> usize {
        self.magnitude
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Find all local maxima (spectral peaks) in the magnitude spectrum.
    /// Returns (bin_index, magnitude) pairs.
    pub fn find_peaks(&self, min_magnitude: f64) -> Vec<(usize, f64)> {
        let n = self.magnitude.len();
        if n < 3 {
            return vec![];
        }
        let mut peaks = Vec::new();
        for i in 1..n - 1 {
            if self.magnitude[i] > min_magnitude
                && self.magnitude[i] > self.magnitude[i - 1]
                && self.magnitude[i] > self.magnitude[i + 1]
            {
                peaks.push((i, self.magnitude[i]));
            }
        }
        peaks
    }

    /// Parabolic interpolation around a magnitude peak for sub-bin accuracy.
    /// Returns (true_bin, true_magnitude).
    pub fn interpolate_peak(&self, bin: usize) -> (f64, f64) {
        let n = self.magnitude.len();
        if bin == 0 || bin >= n - 1 {
            return (bin as f64, self.magnitude[bin]);
        }
        let alpha = self.magnitude[bin - 1].ln().max(-300.0);
        let beta = self.magnitude[bin].ln().max(-300.0);
        let gamma = self.magnitude[bin + 1].ln().max(-300.0);

        let denom = alpha - 2.0 * beta + gamma;
        if denom.abs() < 1e-30 {
            return (bin as f64, self.magnitude[bin]);
        }
        let p = 0.5 * (alpha - gamma) / denom;
        let true_bin = bin as f64 + p;
        let true_mag = (beta - 0.25 * (alpha - gamma) * p).exp();
        (true_bin, true_mag)
    }
}

// ---------------------------------------------------------------------------
// Reassigned spectrogram analyzer
// ---------------------------------------------------------------------------

/// Configuration for the reassigned spectrogram analyzer.
#[derive(Debug, Clone)]
pub struct ReassignedSpectrogramConfig {
    /// FFT size.
    pub fft_size: usize,
    /// Window type.
    pub window_type: WindowType,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Minimum magnitude threshold for tonality classification (linear).
    pub tonality_threshold: f64,
    /// Transiency onset threshold (spectral flux).
    pub onset_threshold: f64,
}

impl Default for ReassignedSpectrogramConfig {
    fn default() -> Self {
        Self {
            fft_size: 4096,
            window_type: WindowType::BlackmanHarris,
            hop_size: 1024,
            sample_rate: 44100.0,
            tonality_threshold: 0.001,
            onset_threshold: 0.5,
        }
    }
}

/// Reassigned spectrogram analyzer.
///
/// Implements the phase-vocoder-style reassigned spectrogram used by
/// Melodyne for true frequency and time estimation. Uses three FFTs per
/// frame: one with the analysis window, one with the time-ramped
/// (derivative) window, and one with the frequency-derivative window.
pub struct ReassignedSpectrogram {
    config: ReassignedSpectrogramConfig,
    engine: FftEngine,
    /// Time-ramped window: (n - N/2) * w(n)
    time_ramp_window: Vec<f64>,
    /// Frequency-derivative window: w'(n)
    freq_deriv_window: Vec<f64>,
    /// Previous frame magnitude for spectral flux computation.
    prev_magnitude: Option<Vec<f64>>,
}

impl ReassignedSpectrogram {
    /// Create a new reassigned spectrogram analyzer.
    pub fn new(config: ReassignedSpectrogramConfig) -> Self {
        let engine = FftEngine::new(config.fft_size, config.window_type);
        let base_window = compute_window(config.window_type, config.fft_size);
        let time_ramp_window = compute_derivative_window(&base_window);
        let freq_deriv_window =
            compute_frequency_derivative_window(config.window_type, config.fft_size);

        Self {
            config,
            engine,
            time_ramp_window,
            freq_deriv_window,
            prev_magnitude: None,
        }
    }

    /// Analyze a single frame centered at `center_sample`.
    ///
    /// Computes three FFTs:
    /// 1. Standard windowed FFT for the complex spectrum
    /// 2. Time-ramped window FFT for time reassignment
    /// 3. Frequency-derivative window FFT for frequency reassignment
    pub fn analyze_frame(
        &mut self,
        audio: &[f64],
        center_sample: usize,
        frame_index: usize,
    ) -> SpectralTimeSlice {
        let n = self.config.fft_size;
        let half = n / 2;
        let num_bins = n / 2 + 1;
        let sr = self.config.sample_rate;

        // Extract frame with zero-padding at edges
        let mut frame = vec![0.0f64; n];
        for (i, frame_val) in frame.iter_mut().enumerate().take(n) {
            let idx = center_sample as isize - half as isize + i as isize;
            if idx >= 0 && (idx as usize) < audio.len() {
                *frame_val = audio[idx as usize];
            }
        }

        // FFT 1: Standard windowed analysis
        let spectrum = self.engine.forward(&frame);
        let magnitude = FftEngine::magnitude_spectrum(&spectrum);
        let phase = FftEngine::phase_spectrum(&spectrum);
        let power_db = FftEngine::power_spectrum_db(&spectrum);

        // FFT 2: Time-ramped window for time reassignment
        let time_ramp_spectrum = {
            let mut buf: Vec<Complex<f64>> = frame
                .iter()
                .zip(self.time_ramp_window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            self.engine.forward_complex_inplace(&mut buf);
            buf
        };

        // FFT 3: Frequency-derivative window for frequency reassignment
        let freq_deriv_spectrum = {
            let mut buf: Vec<Complex<f64>> = frame
                .iter()
                .zip(self.freq_deriv_window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            self.engine.forward_complex_inplace(&mut buf);
            buf
        };

        // Compute reassigned frequencies and time offsets
        let mut true_freq_data = vec![0.0f64; num_bins];
        let mut true_time_data = vec![0.0f64; num_bins];
        let mut instantaneous_freq = vec![0.0f64; num_bins];

        for k in 0..num_bins {
            let nominal_freq = k as f64 * sr / n as f64;
            let mag_sq = spectrum[k].norm_sqr();

            if mag_sq > 1e-30 {
                // Frequency reassignment:
                // f_hat = f_k - Im(X_dw(k) / X_w(k)) * sr / (2*pi)
                // where X_dw is the freq-derivative-window FFT
                let ratio_freq = freq_deriv_spectrum[k] / spectrum[k];
                let freq_correction = -ratio_freq.im * sr / (2.0 * PI);
                true_freq_data[k] = nominal_freq + freq_correction;

                // Time reassignment:
                // t_hat = t_c + Re(X_tw(k) / X_w(k)) / sr
                // where X_tw is the time-ramped-window FFT
                let ratio_time = time_ramp_spectrum[k] / spectrum[k];
                true_time_data[k] = ratio_time.re / sr;

                // Instantaneous frequency from the standard reassignment
                instantaneous_freq[k] = true_freq_data[k];
            } else {
                true_freq_data[k] = nominal_freq;
                true_time_data[k] = 0.0;
                instantaneous_freq[k] = nominal_freq;
            }
        }

        // Classify tonality per bin
        let tonality = self.classify_tonality(&magnitude, &true_freq_data, num_bins);

        // Compute spectral flux for transiency detection
        let transiency = self.compute_transiency(&magnitude);

        // Update previous magnitude for next frame
        self.prev_magnitude = Some(magnitude.clone());

        // Build spectral points
        let points: Vec<SpectralPoint> = (0..num_bins)
            .map(|k| SpectralPoint {
                bin: k,
                complex: spectrum[k],
                magnitude: magnitude[k],
                phase: phase[k],
                true_freq: true_freq_data[k],
                true_time_offset: true_time_data[k],
                instantaneous_freq: instantaneous_freq[k],
                tonality: tonality[k],
            })
            .collect();

        SpectralTimeSlice {
            frame_index,
            center_sample,
            time_seconds: center_sample as f64 / sr,
            sample_rate: sr,
            fft_size: n,
            spectrum,
            magnitude,
            phase,
            power_db,
            true_freq_data,
            true_time_sample_offset_data: true_time_data,
            instantaneous_freq,
            tonality,
            transiency,
            points,
        }
    }

    /// Analyze the entire audio buffer, returning a sequence of time slices.
    pub fn analyze(&mut self, audio: &[f64]) -> Vec<SpectralTimeSlice> {
        let n = self.config.fft_size;
        let hop = self.config.hop_size;
        let half = n / 2;

        self.prev_magnitude = None;
        let mut slices = Vec::new();
        let mut frame_index = 0;
        let mut pos = half;

        while pos + half <= audio.len() {
            slices.push(self.analyze_frame(audio, pos, frame_index));
            pos += hop;
            frame_index += 1;
        }

        slices
    }

    /// Classify tonality for each bin based on reassignment coherence.
    ///
    /// A bin is tonal if its reassigned frequency is close to the nominal
    /// bin center frequency -- this indicates stable sinusoidal energy.
    fn classify_tonality(
        &self,
        magnitude: &[f64],
        true_freq: &[f64],
        num_bins: usize,
    ) -> Vec<TonalityClass> {
        let freq_res = self.config.sample_rate / self.config.fft_size as f64;

        (0..num_bins)
            .map(|k| {
                if magnitude[k] < self.config.tonality_threshold {
                    return TonalityClass::Noise;
                }
                let nominal = k as f64 * freq_res;
                let deviation = (true_freq[k] - nominal).abs() / freq_res;

                if deviation < 0.15 {
                    TonalityClass::Tonal
                } else if deviation < 0.4 {
                    TonalityClass::Mixed
                } else {
                    TonalityClass::Noise
                }
            })
            .collect()
    }

    /// Compute transiency score via spectral flux.
    fn compute_transiency(&self, magnitude: &[f64]) -> TransiencyInfo {
        let spectral_flux = match &self.prev_magnitude {
            Some(prev) => {
                let mut flux = 0.0;
                for (curr, prev_val) in magnitude.iter().zip(prev.iter()) {
                    let diff = curr - prev_val;
                    if diff > 0.0 {
                        flux += diff * diff;
                    }
                }
                flux.sqrt()
            }
            None => 0.0,
        };

        // Normalize flux relative to frame energy
        let energy: f64 = magnitude.iter().map(|m| m * m).sum::<f64>().sqrt();
        let normalized = if energy > 1e-10 {
            spectral_flux / energy
        } else {
            0.0
        };

        let score = normalized.min(1.0);
        let is_onset = score > self.config.onset_threshold;

        TransiencyInfo {
            score,
            is_onset,
            spectral_flux,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &ReassignedSpectrogramConfig {
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

    fn sine_wave(freq: f64, sample_rate: f64, num_samples: usize) -> Vec<f64> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_spectral_time_slice_basic() {
        let sr = 44100.0;
        let audio = sine_wave(440.0, sr, 44100);

        let config = ReassignedSpectrogramConfig {
            fft_size: 4096,
            window_type: WindowType::BlackmanHarris,
            hop_size: 1024,
            sample_rate: sr,
            ..Default::default()
        };
        let mut analyzer = ReassignedSpectrogram::new(config);
        let slices = analyzer.analyze(&audio);

        assert!(!slices.is_empty());

        let mid = &slices[slices.len() / 2];
        assert_eq!(mid.num_bins(), 2049);
        assert!((mid.freq_resolution() - sr / 4096.0).abs() < 1e-6);

        // The peak bin should be near 440 Hz
        let peak = mid.peak_bin();
        let peak_freq = mid.bin_to_freq(peak);
        assert!(
            (peak_freq - 440.0).abs() < 20.0,
            "Peak at {} Hz, expected ~440",
            peak_freq
        );
    }

    #[test]
    fn test_reassigned_frequency_accuracy() {
        let sr = 44100.0;
        let freq = 1000.0;
        let audio = sine_wave(freq, sr, 44100);

        let config = ReassignedSpectrogramConfig {
            fft_size: 4096,
            window_type: WindowType::Hann,
            hop_size: 1024,
            sample_rate: sr,
            tonality_threshold: 0.0001,
            onset_threshold: 0.5,
        };
        let mut analyzer = ReassignedSpectrogram::new(config);
        let slices = analyzer.analyze(&audio);

        // Pick a mid-stream frame
        let mid = &slices[slices.len() / 2];
        let peak = mid.peak_bin();

        // The reassigned true frequency should be closer to 1000 Hz
        // than the nominal bin frequency
        let nominal = mid.bin_to_freq(peak);
        let reassigned = mid.true_freq_data[peak];
        let nominal_error = (nominal - freq).abs();
        let reassigned_error = (reassigned - freq).abs();

        // Reassigned frequency should be reasonably accurate
        assert!(
            reassigned_error < 5.0,
            "Reassigned freq {} Hz, expected ~{} Hz (error {})",
            reassigned,
            freq,
            reassigned_error
        );
        // And generally better or comparable to nominal
        assert!(
            reassigned_error <= nominal_error + 1.0,
            "Reassigned error {} should be <= nominal error {}",
            reassigned_error,
            nominal_error
        );
    }

    #[test]
    fn test_tonality_classification() {
        let sr = 44100.0;
        let audio = sine_wave(440.0, sr, 44100);

        let config = ReassignedSpectrogramConfig {
            fft_size: 4096,
            window_type: WindowType::Hann,
            hop_size: 1024,
            sample_rate: sr,
            tonality_threshold: 0.001,
            onset_threshold: 0.5,
        };
        let mut analyzer = ReassignedSpectrogram::new(config);
        let slices = analyzer.analyze(&audio);

        let mid = &slices[slices.len() / 2];
        let peak = mid.peak_bin();

        // The peak bin of a pure sine should be classified as tonal
        assert_eq!(
            mid.tonality[peak],
            TonalityClass::Tonal,
            "Peak bin should be tonal"
        );
    }

    #[test]
    fn test_find_peaks() {
        let sr = 44100.0;
        // Two-tone signal
        let audio: Vec<f64> = (0..44100)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 880.0 * t).sin()
            })
            .collect();

        let config = ReassignedSpectrogramConfig::default();
        let mut analyzer = ReassignedSpectrogram::new(config);
        let slices = analyzer.analyze(&audio);
        let mid = &slices[slices.len() / 2];

        let peaks = mid.find_peaks(0.1);
        // Should find at least 2 peaks (440 and 880 Hz)
        assert!(
            peaks.len() >= 2,
            "Expected at least 2 peaks, found {}",
            peaks.len()
        );
    }

    #[test]
    fn test_transiency_detection() {
        let sr = 44100.0;
        let n = 44100;
        // Silence followed by sudden onset
        let mut audio = vec![0.0f64; n];
        for i in (n / 2)..n {
            audio[i] = (2.0 * PI * 440.0 * i as f64 / sr).sin();
        }

        let config = ReassignedSpectrogramConfig {
            fft_size: 2048,
            hop_size: 512,
            sample_rate: sr,
            onset_threshold: 0.3,
            ..Default::default()
        };
        let mut analyzer = ReassignedSpectrogram::new(config);
        let slices = analyzer.analyze(&audio);

        // There should be at least one onset detected near the midpoint
        let onsets: Vec<_> = slices.iter().filter(|s| s.transiency.is_onset).collect();
        assert!(!onsets.is_empty(), "Should detect at least one onset");
    }
}
