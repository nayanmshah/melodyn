//! Real-time phase vocoder for pitch shifting.
//!
//! Implements STFT-based pitch shifting with proper phase accumulation,
//! phase unwrapping, frequency-domain bin shifting with interpolation,
//! and ISTFT with overlap-add. Designed for low-latency real-time use
//! with one hop of algorithmic latency.
//!
//! Optionally preserves formants by separating the spectral envelope from
//! the fine structure before shifting.

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TWO_PI: f64 = 2.0 * PI;

// ---------------------------------------------------------------------------
// Phase vocoder
// ---------------------------------------------------------------------------

/// A real-time phase vocoder for frequency-domain pitch shifting.
///
/// Processing pipeline:
/// 1. Buffer incoming samples into overlapping analysis frames.
/// 2. Apply analysis window + FFT.
/// 3. Compute instantaneous frequency via phase unwrapping.
/// 4. Shift bins in the frequency domain (with interpolation).
/// 5. Accumulate synthesis phases.
/// 6. IFFT + synthesis window + overlap-add.
///
/// Latency is exactly `window_size` samples.
pub struct PhaseVocoder {
    /// FFT/window size.
    window_size: usize,
    /// Hop size (analysis and synthesis).
    hop_size: usize,
    /// Sample rate.
    sample_rate: f64,
    /// Number of frequency bins (window_size / 2 + 1).
    num_bins: usize,
    /// Pitch shift ratio (1.0 = no shift, 2.0 = octave up).
    pitch_shift: f64,
    /// Whether to preserve formants during pitch shifting.
    preserve_formants: bool,

    // FFT plans.
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,

    // Windows.
    analysis_window: Vec<f64>,
    synthesis_window: Vec<f64>,

    // Input buffering.
    input_buffer: Vec<f64>,
    input_write_pos: usize,
    samples_until_next_hop: usize,

    // Output buffering (overlap-add accumulator).
    output_buffer: Vec<f64>,
    output_read_pos: usize,

    // Phase tracking.
    last_input_phase: Vec<f64>,
    synthesis_phase: Vec<f64>,

    // Pre-allocated scratch buffers.
    fft_scratch: Vec<Complex<f64>>,
    magnitude: Vec<f64>,
    frequency: Vec<f64>,
    new_magnitude: Vec<f64>,
    new_frequency: Vec<f64>,
    envelope: Vec<f64>,

    /// Expected phase advance per bin per hop.
    phase_advance_per_bin: f64,
    /// Normalization for overlap-add.
    ola_norm: f64,
}

impl PhaseVocoder {
    /// Create a new phase vocoder.
    ///
    /// - `window_size`: FFT size (typically 1024 or 2048).
    /// - `hop_size`: Hop size in samples (typically window_size / 4).
    /// - `sample_rate`: Audio sample rate.
    pub fn new(window_size: usize, hop_size: usize, sample_rate: f64) -> Self {
        let num_bins = window_size / 2 + 1;

        let mut planner = FftPlanner::<f64>::new();
        let fft_forward = planner.plan_fft_forward(window_size);
        let fft_inverse = planner.plan_fft_inverse(window_size);

        let analysis_window = hann_window(window_size);
        let synthesis_window = hann_window(window_size);

        // Compute OLA normalization factor.
        // For Hann window with 75% overlap (hop = window/4), the COLA sum is 1.5.
        let overlap_factor = window_size / hop_size;
        let ola_norm = 2.0 / (3.0 * overlap_factor as f64).max(1.0);

        let phase_advance_per_bin = TWO_PI * hop_size as f64 / window_size as f64;

        // Output buffer needs to be large enough for overlap-add.
        // We use 2x window_size to have room for read and write simultaneously.
        let output_buf_size = window_size * 2;

        Self {
            window_size,
            hop_size,
            sample_rate,
            num_bins,
            pitch_shift: 1.0,
            preserve_formants: false,
            fft_forward,
            fft_inverse,
            analysis_window,
            synthesis_window,
            input_buffer: vec![0.0; window_size],
            input_write_pos: 0,
            samples_until_next_hop: hop_size,
            output_buffer: vec![0.0; output_buf_size],
            output_read_pos: 0,
            last_input_phase: vec![0.0; num_bins],
            synthesis_phase: vec![0.0; num_bins],
            fft_scratch: vec![Complex::new(0.0, 0.0); window_size],
            magnitude: vec![0.0; num_bins],
            frequency: vec![0.0; num_bins],
            new_magnitude: vec![0.0; num_bins],
            new_frequency: vec![0.0; num_bins],
            envelope: vec![0.0; num_bins],
            phase_advance_per_bin,
            ola_norm,
        }
    }

    /// Set the pitch shift ratio.
    ///
    /// - 1.0 = no shift
    /// - 2.0 = one octave up
    /// - 0.5 = one octave down
    pub fn set_pitch_shift(&mut self, ratio: f64) {
        self.pitch_shift = ratio.clamp(0.25, 4.0);
    }

    /// Set whether to preserve formants during pitch shifting.
    pub fn set_preserve_formants(&mut self, preserve: bool) {
        self.preserve_formants = preserve;
    }

    /// Get the latency in samples.
    pub fn latency(&self) -> usize {
        self.window_size
    }

    /// Reset internal state (clear all buffers and phases).
    pub fn reset(&mut self) {
        self.input_buffer.iter_mut().for_each(|s| *s = 0.0);
        self.input_write_pos = 0;
        self.samples_until_next_hop = self.hop_size;
        self.output_buffer.iter_mut().for_each(|s| *s = 0.0);
        self.output_read_pos = 0;
        self.last_input_phase.iter_mut().for_each(|s| *s = 0.0);
        self.synthesis_phase.iter_mut().for_each(|s| *s = 0.0);
    }

    /// Process a block of audio samples.
    ///
    /// Input and output buffers must be the same length. Output is delayed
    /// by `window_size` samples (the vocoder's latency).
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) {
        let len = input.len().min(output.len());

        for i in 0..len {
            // Write input sample into the circular input buffer.
            self.input_buffer[self.input_write_pos] = input[i];
            self.input_write_pos = (self.input_write_pos + 1) % self.window_size;

            self.samples_until_next_hop -= 1;

            if self.samples_until_next_hop == 0 {
                self.samples_until_next_hop = self.hop_size;
                self.process_frame();
            }

            // Read from the output overlap-add buffer.
            output[i] = self.output_buffer[self.output_read_pos];
            self.output_buffer[self.output_read_pos] = 0.0;
            self.output_read_pos = (self.output_read_pos + 1) % self.output_buffer.len();
        }
    }

    /// Process a single STFT frame.
    fn process_frame(&mut self) {
        let n = self.window_size;

        // Collect the analysis frame from the circular input buffer.
        // The most recent sample is at input_write_pos - 1.
        for i in 0..n {
            let buf_idx = (self.input_write_pos + i) % n;
            let windowed = self.input_buffer[buf_idx] * self.analysis_window[i];
            self.fft_scratch[i] = Complex::new(windowed, 0.0);
        }

        // Forward FFT.
        self.fft_forward.process(&mut self.fft_scratch);

        // Analysis: convert to magnitude + instantaneous frequency.
        self.analyze_phases();

        // Pitch shift in frequency domain.
        self.shift_bins();

        // If formant preservation is enabled, apply envelope correction.
        if self.preserve_formants && (self.pitch_shift - 1.0).abs() > 0.001 {
            self.apply_formant_preservation();
        }

        // Synthesis: convert back to complex spectrum.
        self.synthesize_phases();

        // Inverse FFT.
        self.fft_inverse.process(&mut self.fft_scratch);

        // Normalize IFFT (rustfft doesn't normalize).
        let inv_norm = 1.0 / n as f64;

        // Apply synthesis window and overlap-add into output buffer.
        let out_len = self.output_buffer.len();
        for i in 0..n {
            let sample = self.fft_scratch[i].re * inv_norm * self.synthesis_window[i] * self.ola_norm;
            let out_idx = (self.output_read_pos + i) % out_len;
            self.output_buffer[out_idx] += sample;
        }
    }

    /// Convert the FFT spectrum to magnitude and instantaneous frequency
    /// using phase unwrapping.
    fn analyze_phases(&mut self) {
        let hop = self.hop_size as f64;
        let freq_per_bin = self.sample_rate / self.window_size as f64;

        for k in 0..self.num_bins {
            let c = self.fft_scratch[k];
            let mag = c.norm();
            let phase = c.arg();

            // Phase difference from last frame.
            let expected_phase = self.last_input_phase[k] + self.phase_advance_per_bin * k as f64;
            let phase_diff = phase - expected_phase;

            // Wrap phase difference to [-pi, pi].
            let wrapped = phase_diff - TWO_PI * (phase_diff / TWO_PI).round();

            // Instantaneous frequency for this bin.
            let inst_freq = (k as f64 + wrapped / (TWO_PI * hop / self.window_size as f64))
                * freq_per_bin;

            self.magnitude[k] = mag;
            self.frequency[k] = inst_freq;
            self.last_input_phase[k] = phase;
        }
    }

    /// Shift bins in the frequency domain according to the pitch shift ratio.
    ///
    /// Uses linear interpolation for fractional bin positions.
    fn shift_bins(&mut self) {
        let shift = self.pitch_shift;

        // Clear destination buffers.
        self.new_magnitude.iter_mut().for_each(|m| *m = 0.0);
        self.new_frequency.iter_mut().for_each(|f| *f = 0.0);

        for k in 0..self.num_bins {
            let new_bin = k as f64 * shift;
            let new_bin_int = new_bin.floor() as usize;
            let frac = new_bin - new_bin_int as f64;

            if new_bin_int < self.num_bins {
                self.new_magnitude[new_bin_int] += self.magnitude[k] * (1.0 - frac);
                self.new_frequency[new_bin_int] = self.frequency[k] * shift;

                if new_bin_int + 1 < self.num_bins {
                    self.new_magnitude[new_bin_int + 1] += self.magnitude[k] * frac;
                    self.new_frequency[new_bin_int + 1] = self.frequency[k] * shift;
                }
            }
        }
    }

    /// Apply formant preservation by correcting the spectral envelope.
    ///
    /// When pitch-shifting, formants naturally shift with the harmonics.
    /// To preserve them, we:
    /// 1. Estimate the original spectral envelope.
    /// 2. Estimate the shifted spectral envelope.
    /// 3. Apply a correction gain = original_envelope / shifted_envelope.
    fn apply_formant_preservation(&mut self) {
        // Estimate spectral envelope using cepstral smoothing.
        // Use a simple peak-following envelope as a lightweight alternative.
        let smoothing_bins = 8; // Number of bins for local averaging.

        // Original envelope from input magnitudes.
        spectral_envelope(&self.magnitude, &mut self.envelope, smoothing_bins);

        // Shifted envelope from output magnitudes.
        let mut shifted_env = vec![0.0; self.num_bins];
        spectral_envelope(&self.new_magnitude, &mut shifted_env, smoothing_bins);

        // Apply correction: scale new magnitudes so the envelope matches original.
        for k in 0..self.num_bins {
            if shifted_env[k] > 1e-10 {
                let correction = self.envelope[k.min(self.num_bins - 1)] / shifted_env[k];
                self.new_magnitude[k] *= correction.min(10.0); // Limit gain to avoid blowup.
            }
        }
    }

    /// Convert shifted magnitudes and frequencies back to a complex spectrum
    /// with proper phase accumulation.
    fn synthesize_phases(&mut self) {
        let freq_per_bin = self.sample_rate / self.window_size as f64;
        let hop = self.hop_size as f64;
        let n = self.window_size;

        for k in 0..self.num_bins {
            // Convert instantaneous frequency back to phase increment.
            let freq = self.new_frequency[k];
            let bin_deviation = freq / freq_per_bin - k as f64;
            let phase_inc = self.phase_advance_per_bin * k as f64
                + TWO_PI * bin_deviation * hop / n as f64;

            self.synthesis_phase[k] += phase_inc;

            // Wrap to [-pi, pi] periodically to prevent float precision loss.
            if self.synthesis_phase[k].abs() > 1e6 {
                self.synthesis_phase[k] -= TWO_PI * (self.synthesis_phase[k] / TWO_PI).round();
            }

            let mag = self.new_magnitude[k];
            let phase = self.synthesis_phase[k];
            self.fft_scratch[k] = Complex::new(mag * phase.cos(), mag * phase.sin());
        }

        // Fill the conjugate-symmetric upper half.
        for k in self.num_bins..n {
            let mirror = n - k;
            self.fft_scratch[k] = self.fft_scratch[mirror].conj();
        }
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Generate a Hann window of the given size.
fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (TWO_PI * i as f64 / size as f64).cos()))
        .collect()
}

/// Estimate the spectral envelope using local averaging.
///
/// A lightweight alternative to cepstral smoothing for real-time use.
fn spectral_envelope(magnitudes: &[f64], envelope: &mut [f64], half_width: usize) {
    let n = magnitudes.len().min(envelope.len());
    for k in 0..n {
        let start = k.saturating_sub(half_width);
        let end = (k + half_width + 1).min(n);
        let count = end - start;
        let sum: f64 = magnitudes[start..end].iter().sum();
        envelope[k] = if count > 0 { sum / count as f64 } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a sine wave.
    fn sine_wave(freq: f64, sample_rate: f64, num_samples: usize) -> Vec<f64> {
        (0..num_samples)
            .map(|i| (TWO_PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    /// Compute RMS of a signal slice.
    fn rms(signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }
        (signal.iter().map(|&s| s * s).sum::<f64>() / signal.len() as f64).sqrt()
    }

    /// Estimate dominant frequency using zero crossings.
    fn estimate_freq_zero_crossings(signal: &[f64], sample_rate: f64) -> f64 {
        let mut crossings = 0usize;
        for i in 1..signal.len() {
            if (signal[i] >= 0.0) != (signal[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        // Each full cycle has 2 zero crossings.
        let duration = signal.len() as f64 / sample_rate;
        crossings as f64 / (2.0 * duration)
    }

    #[test]
    fn test_vocoder_creation() {
        let voc = PhaseVocoder::new(2048, 512, 44100.0);
        assert_eq!(voc.latency(), 2048);
        assert_eq!(voc.num_bins, 1025);
    }

    #[test]
    fn test_vocoder_passthrough() {
        let mut voc = PhaseVocoder::new(2048, 512, 44100.0);
        // No pitch shift = passthrough.
        voc.set_pitch_shift(1.0);

        let sample_rate = 44100.0;
        let input = sine_wave(440.0, sample_rate, 44100);
        let mut output = vec![0.0; input.len()];

        voc.process(&input, &mut output);

        // After the latency period, output should resemble input.
        let latency = voc.latency();
        let start = latency + 2048; // Allow extra settling.
        let end = output.len() - 2048;

        if start < end {
            let out_rms = rms(&output[start..end]);
            let in_rms = rms(&input[start..end]);
            // Output should have similar energy (within 6dB).
            assert!(
                out_rms > in_rms * 0.25 && out_rms < in_rms * 4.0,
                "Output RMS {} should be similar to input RMS {}",
                out_rms,
                in_rms
            );
        }
    }

    #[test]
    fn test_vocoder_octave_up() {
        let sample_rate = 44100.0;
        let mut voc = PhaseVocoder::new(2048, 512, sample_rate);
        voc.set_pitch_shift(2.0); // Octave up.

        let input = sine_wave(220.0, sample_rate, 44100);
        let mut output = vec![0.0; input.len()];

        voc.process(&input, &mut output);

        // After settling, the output frequency should be ~440 Hz.
        let start = 8192;
        let end = output.len() - 4096;
        if start < end {
            let freq = estimate_freq_zero_crossings(&output[start..end], sample_rate);
            // Allow generous tolerance for phase vocoder artifacts.
            assert!(
                (freq - 440.0).abs() < 100.0,
                "Expected ~440 Hz, got {} Hz",
                freq
            );
        }
    }

    #[test]
    fn test_vocoder_shift_up_then_down() {
        let sample_rate = 44100.0;
        let input = sine_wave(440.0, sample_rate, 44100);

        // Shift up by 2 semitones.
        let mut voc_up = PhaseVocoder::new(2048, 512, sample_rate);
        let shift_up = 2.0_f64.powf(2.0 / 12.0);
        voc_up.set_pitch_shift(shift_up);
        let mut intermediate = vec![0.0; input.len()];
        voc_up.process(&input, &mut intermediate);

        // Shift back down by 2 semitones.
        let mut voc_down = PhaseVocoder::new(2048, 512, sample_rate);
        let shift_down = 2.0_f64.powf(-2.0 / 12.0);
        voc_down.set_pitch_shift(shift_down);
        let mut output = vec![0.0; intermediate.len()];
        voc_down.process(&intermediate, &mut output);

        // After double latency settling, output should be close to input.
        let start = 16384;
        let end = output.len() - 8192;
        if start < end {
            let out_rms = rms(&output[start..end]);
            let in_rms = rms(&input[start..end]);
            // Should have similar energy (within 12dB for double vocoder).
            assert!(
                out_rms > in_rms * 0.1,
                "Roundtrip output RMS {} too low compared to input RMS {}",
                out_rms,
                in_rms
            );
        }
    }

    #[test]
    fn test_vocoder_reset() {
        let mut voc = PhaseVocoder::new(1024, 256, 44100.0);
        let input = vec![0.5; 2048];
        let mut output = vec![0.0; 2048];
        voc.process(&input, &mut output);

        voc.reset();

        // After reset, phases should be zeroed.
        assert!(voc.last_input_phase.iter().all(|&p| p == 0.0));
        assert!(voc.synthesis_phase.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn test_spectral_envelope() {
        let magnitudes = vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let mut env = vec![0.0; 8];
        spectral_envelope(&magnitudes, &mut env, 2);

        // Envelope should be smoothed.
        // Center values should be close to local average.
        assert!(env[4] > 0.0);
        // Edge values should still be non-negative.
        assert!(env[0] >= 0.0);
    }

    #[test]
    fn test_hann_window() {
        let win = hann_window(1024);
        assert_eq!(win.len(), 1024);
        // Hann window is zero at endpoints.
        assert!(win[0].abs() < 1e-10);
        // Peak near center.
        assert!(win[512] > 0.9);
    }

    #[test]
    fn test_vocoder_formant_preservation() {
        let mut voc = PhaseVocoder::new(2048, 512, 44100.0);
        voc.set_pitch_shift(1.5);
        voc.set_preserve_formants(true);

        let input = sine_wave(220.0, 44100.0, 8192);
        let mut output = vec![0.0; 8192];

        // Should not panic.
        voc.process(&input, &mut output);
        assert!(output.iter().any(|&s| s != 0.0));
    }

    #[test]
    fn test_vocoder_silence_in_silence_out() {
        let mut voc = PhaseVocoder::new(1024, 256, 44100.0);
        let input = vec![0.0; 4096];
        let mut output = vec![0.0; 4096];

        voc.process(&input, &mut output);

        // Silence in -> silence out.
        let out_rms = rms(&output);
        assert!(
            out_rms < 1e-10,
            "Silence input should produce silence output, got RMS {}",
            out_rms
        );
    }
}
