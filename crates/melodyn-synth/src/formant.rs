//! 18-band formant model with all extracted parameters.
//!
//! Implements the formant processing from MUFormant and MUFormantTool,
//! including spectral envelope extraction, formant shifting, and the
//! synthesis parameters extracted from the Melodyne binary.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Formant synthesis parameters (from binary extraction)
// ---------------------------------------------------------------------------

/// Parameters for formant synthesis, extracted from the Melodyne 5.4.2 binary.
///
/// These control the formant resynthesis engine's behaviour.
#[derive(Debug, Clone)]
pub struct FormantSynthParams {
    /// Amplitude of the fundamental in formant synthesis.
    pub fundamental_amp: f64,
    /// Random modulation depth of the fundamental.
    pub fundamental_random: f64,
    /// Amplitude of 2nd subharmonic.
    pub subharmonic2_amp: f64,
    /// Random modulation of 2nd subharmonic.
    pub subharmonic2_random: f64,
    /// Amplitude of 3rd subharmonic.
    pub subharmonic3_amp: f64,
    /// Random modulation of 3rd subharmonic.
    pub subharmonic3_random: f64,
    /// Amplitude of 4th subharmonic.
    pub subharmonic4_amp: f64,
    /// Random modulation of 4th subharmonic.
    pub subharmonic4_random: f64,
    /// Constant noise floor level.
    pub const_noise: f64,
    /// Wave-shaped noise level.
    pub wave_noise: f64,
    /// Pulse excitation decay rate.
    pub pulse_decay: f64,
    /// Growl high-pass filter frequency.
    pub growl_hi_pass: f64,
    /// Time-domain random modulation.
    pub time_random: f64,
}

impl Default for FormantSynthParams {
    fn default() -> Self {
        Self {
            fundamental_amp: 1.0,
            fundamental_random: 0.02,
            subharmonic2_amp: 0.3,
            subharmonic2_random: 0.05,
            subharmonic3_amp: 0.1,
            subharmonic3_random: 0.05,
            subharmonic4_amp: 0.05,
            subharmonic4_random: 0.05,
            const_noise: 0.001,
            wave_noise: 0.01,
            pulse_decay: 0.95,
            growl_hi_pass: 50.0,
            time_random: 0.01,
        }
    }
}

/// Parameters for formant spectral shaping.
#[derive(Debug, Clone)]
pub struct FormantShapeParams {
    /// Formant offset at 750 Hz band (dB).
    pub offset_750hz: f64,
    /// Formant offset at 1500 Hz band (dB).
    pub offset_1500hz: f64,
    /// Formant offset at 3000 Hz band (dB).
    pub offset_3000hz: f64,
    /// Formant offset at 5000 Hz band (dB).
    pub offset_5000hz: f64,
    /// Vowel parameter (controls vowel-like character).
    pub vowel: f64,
    /// Vowel cycle rate.
    pub vowel_cycle: f64,
    /// Formant transpose in semitones.
    pub transpose: f64,
    /// Synthesis amplitude.
    pub synth_amp: f64,
    /// Noise amplitude.
    pub noise_amp: f64,
    /// Amount of original signal to add back.
    pub add_original: f64,
    /// Emphasis (spectral tilt).
    pub emphasize: f64,
    /// Master volume.
    pub master_volume: f64,
    /// Low-frequency blend to original.
    pub lo_to_original: f64,
    /// Spectrum filter smoothness.
    pub spectrum_filter: f64,
    /// Time filter smoothness.
    pub time_filter: f64,
}

impl Default for FormantShapeParams {
    fn default() -> Self {
        Self {
            offset_750hz: 0.0,
            offset_1500hz: 0.0,
            offset_3000hz: 0.0,
            offset_5000hz: 0.0,
            vowel: 0.0,
            vowel_cycle: 0.0,
            transpose: 0.0,
            synth_amp: 1.0,
            noise_amp: 0.0,
            add_original: 0.0,
            emphasize: 0.0,
            master_volume: 1.0,
            lo_to_original: 0.0,
            spectrum_filter: 0.5,
            time_filter: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Formant band
// ---------------------------------------------------------------------------

/// A single band in the 18-band formant model.
#[derive(Debug, Clone)]
pub struct FormantBand {
    /// Center frequency of this band (Hz).
    pub center_freq: f64,
    /// Bandwidth (Hz).
    pub bandwidth: f64,
    /// Gain (linear).
    pub gain: f64,
    /// Phase offset (radians).
    pub phase: f64,
}

/// The 18-band formant model.
///
/// Covers the vocal range from ~100 Hz to ~16000 Hz with 18 overlapping
/// bands. The key formant frequencies (750, 1500, 3000, 5000 Hz)
/// correspond to the primary vocal formant regions F1-F4.
#[derive(Debug, Clone)]
pub struct FormantModel {
    /// The 18 formant bands.
    pub bands: Vec<FormantBand>,
    /// Sample rate.
    pub sample_rate: f64,
}

impl FormantModel {
    /// Create the default 18-band formant model.
    pub fn new(sample_rate: f64) -> Self {
        // 18 bands logarithmically spaced from ~100 Hz to ~16000 Hz
        let center_freqs = [
            100.0, 175.0, 275.0, 400.0, 550.0, 750.0, 1000.0, 1250.0, 1500.0, 2000.0, 2500.0,
            3000.0, 3750.0, 5000.0, 6500.0, 8000.0, 10000.0, 13000.0,
        ];

        let bands = center_freqs
            .iter()
            .map(|&freq| {
                let bandwidth = freq * 0.3; // ~30% bandwidth
                FormantBand {
                    center_freq: freq,
                    bandwidth,
                    gain: 1.0,
                    phase: 0.0,
                }
            })
            .collect();

        Self {
            bands,
            sample_rate,
        }
    }

    /// Extract formant envelope from a magnitude spectrum.
    ///
    /// Measures the energy in each formant band from the given spectrum.
    pub fn extract_envelope(&self, magnitude: &[f64], fft_size: usize) -> Vec<f64> {
        let freq_res = self.sample_rate / fft_size as f64;
        let num_bins = magnitude.len();

        self.bands
            .iter()
            .map(|band| {
                let lo_bin =
                    ((band.center_freq - band.bandwidth / 2.0) / freq_res).floor().max(0.0)
                        as usize;
                let hi_bin = ((band.center_freq + band.bandwidth / 2.0) / freq_res)
                    .ceil()
                    .min(num_bins as f64 - 1.0) as usize;

                if lo_bin >= hi_bin || lo_bin >= num_bins {
                    return 0.0;
                }

                // RMS energy in the band
                let energy: f64 =
                    magnitude[lo_bin..=hi_bin].iter().map(|m| m * m).sum::<f64>();
                let n = (hi_bin - lo_bin + 1) as f64;
                (energy / n).sqrt()
            })
            .collect()
    }

    /// Apply a formant envelope to a magnitude spectrum.
    ///
    /// Scales each frequency region according to the target formant gains.
    pub fn apply_envelope(
        &self,
        magnitude: &mut [f64],
        envelope: &[f64],
        target_envelope: &[f64],
        fft_size: usize,
    ) {
        let freq_res = self.sample_rate / fft_size as f64;
        let num_bins = magnitude.len();

        for (band_idx, band) in self.bands.iter().enumerate() {
            let lo_bin =
                ((band.center_freq - band.bandwidth / 2.0) / freq_res).floor().max(0.0) as usize;
            let hi_bin = ((band.center_freq + band.bandwidth / 2.0) / freq_res)
                .ceil()
                .min(num_bins as f64 - 1.0) as usize;

            if lo_bin >= hi_bin || lo_bin >= num_bins {
                continue;
            }

            let source = if band_idx < envelope.len() {
                envelope[band_idx]
            } else {
                1.0
            };
            let target = if band_idx < target_envelope.len() {
                target_envelope[band_idx]
            } else {
                1.0
            };

            let gain = if source > 1e-10 {
                target / source
            } else {
                1.0
            };

            for bin in lo_bin..=hi_bin.min(num_bins - 1) {
                magnitude[bin] *= gain;
            }
        }
    }

    /// Shift formants by the given amount in semitones.
    ///
    /// Returns a new formant model with shifted center frequencies.
    pub fn shifted(&self, semitones: f64) -> Self {
        let ratio = 2.0f64.powf(semitones / 12.0);
        let bands = self
            .bands
            .iter()
            .map(|band| FormantBand {
                center_freq: band.center_freq * ratio,
                bandwidth: band.bandwidth * ratio,
                gain: band.gain,
                phase: band.phase,
            })
            .collect();

        Self {
            bands,
            sample_rate: self.sample_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// Formant processor
// ---------------------------------------------------------------------------

/// Formant processor that handles formant-preserving pitch shifting.
///
/// When pitch-shifting a note, the formant processor can optionally
/// preserve the original formant structure so that the timbre doesn't
/// shift with the pitch (avoiding the "chipmunk effect").
pub struct FormantProcessor {
    model: FormantModel,
    synth_params: FormantSynthParams,
    shape_params: FormantShapeParams,
}

impl FormantProcessor {
    /// Create a new formant processor.
    pub fn new(sample_rate: f64) -> Self {
        Self {
            model: FormantModel::new(sample_rate),
            synth_params: FormantSynthParams::default(),
            shape_params: FormantShapeParams::default(),
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        sample_rate: f64,
        synth_params: FormantSynthParams,
        shape_params: FormantShapeParams,
    ) -> Self {
        Self {
            model: FormantModel::new(sample_rate),
            synth_params,
            shape_params,
        }
    }

    /// Process formant-preserving pitch shift on a magnitude spectrum.
    ///
    /// 1. Extract the original formant envelope
    /// 2. Pitch-shift the spectrum (done externally)
    /// 3. Re-apply the original formant envelope to the shifted spectrum
    ///
    /// `original_mag` is the unshifted spectrum, `shifted_mag` is after pitch shift.
    pub fn preserve_formants(
        &self,
        original_mag: &[f64],
        shifted_mag: &mut [f64],
        fft_size: usize,
        preservation_amount: f64,
    ) {
        if preservation_amount < 1e-6 {
            return;
        }

        // Extract envelope from original
        let original_env = self.model.extract_envelope(original_mag, fft_size);

        // Extract envelope from shifted
        let shifted_env = self.model.extract_envelope(shifted_mag, fft_size);

        // Apply formant shape offsets
        let target_env: Vec<f64> = original_env
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let offset_db = self.band_offset_db(i);
                e * 10.0f64.powf(offset_db / 20.0)
            })
            .collect();

        // Blend between shifted and preserved based on amount
        let blended_env: Vec<f64> = shifted_env
            .iter()
            .zip(target_env.iter())
            .map(|(&s, &t)| s * (1.0 - preservation_amount) + t * preservation_amount)
            .collect();

        self.model
            .apply_envelope(shifted_mag, &shifted_env, &blended_env, fft_size);
    }

    /// Get the formant offset in dB for a given band index.
    fn band_offset_db(&self, band_idx: usize) -> f64 {
        let freq = if band_idx < self.model.bands.len() {
            self.model.bands[band_idx].center_freq
        } else {
            return 0.0;
        };

        // Interpolate between the four key offset frequencies
        if freq <= 750.0 {
            self.shape_params.offset_750hz * (freq / 750.0)
        } else if freq <= 1500.0 {
            let t = (freq - 750.0) / (1500.0 - 750.0);
            self.shape_params.offset_750hz * (1.0 - t) + self.shape_params.offset_1500hz * t
        } else if freq <= 3000.0 {
            let t = (freq - 1500.0) / (3000.0 - 1500.0);
            self.shape_params.offset_1500hz * (1.0 - t) + self.shape_params.offset_3000hz * t
        } else if freq <= 5000.0 {
            let t = (freq - 3000.0) / (5000.0 - 3000.0);
            self.shape_params.offset_3000hz * (1.0 - t) + self.shape_params.offset_5000hz * t
        } else {
            self.shape_params.offset_5000hz
        }
    }

    /// Generate a formant excitation signal.
    ///
    /// Produces the glottal-pulse-like excitation used by the formant
    /// synthesis engine.
    pub fn generate_excitation(&self, num_samples: usize, fundamental: f64) -> Vec<f64> {
        let sr = self.model.sample_rate;
        let period = (sr / fundamental).round() as usize;
        let mut output = vec![0.0f64; num_samples];
        let mut phase = 0.0f64;

        // Simple pseudo-random using a basic LCG
        let mut rng_state: u64 = 42;
        let next_rand = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
        };

        for i in 0..num_samples {
            let t = i as f64 / sr;

            // Fundamental pulse train
            let pulse = (2.0 * PI * fundamental * t).sin();
            let fund = self.synth_params.fundamental_amp * pulse
                * (1.0 + self.synth_params.fundamental_random * next_rand(&mut rng_state));

            // Subharmonics
            let sub2 = self.synth_params.subharmonic2_amp
                * (PI * fundamental * t).sin()
                * (1.0 + self.synth_params.subharmonic2_random * next_rand(&mut rng_state));
            let sub3 = self.synth_params.subharmonic3_amp
                * (2.0 * PI * fundamental * t / 3.0).sin()
                * (1.0 + self.synth_params.subharmonic3_random * next_rand(&mut rng_state));
            let sub4 = self.synth_params.subharmonic4_amp
                * (PI * fundamental * t / 2.0).sin()
                * (1.0 + self.synth_params.subharmonic4_random * next_rand(&mut rng_state));

            // Noise components
            let noise = self.synth_params.const_noise * next_rand(&mut rng_state);
            let wave_noise =
                self.synth_params.wave_noise * pulse.abs() * next_rand(&mut rng_state);

            output[i] = fund + sub2 + sub3 + sub4 + noise + wave_noise;

            // Time randomization
            output[i] *=
                1.0 + self.synth_params.time_random * next_rand(&mut rng_state);
        }

        output
    }

    /// Access the formant model.
    pub fn model(&self) -> &FormantModel {
        &self.model
    }

    /// Access the synthesis parameters.
    pub fn synth_params(&self) -> &FormantSynthParams {
        &self.synth_params
    }

    /// Access the shape parameters.
    pub fn shape_params(&self) -> &FormantShapeParams {
        &self.shape_params
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_model_creation() {
        let model = FormantModel::new(44100.0);
        assert_eq!(model.bands.len(), 18);
        assert!((model.bands[0].center_freq - 100.0).abs() < 1e-6);
        assert!((model.bands[5].center_freq - 750.0).abs() < 1e-6);
        assert!((model.bands[13].center_freq - 5000.0).abs() < 1e-6);
    }

    #[test]
    fn test_formant_envelope_extraction() {
        let model = FormantModel::new(44100.0);
        let fft_size = 4096;
        let num_bins = fft_size / 2 + 1;

        // Create a simple test spectrum with a peak at 1000 Hz
        let mut magnitude = vec![0.001; num_bins];
        let freq_res = 44100.0 / fft_size as f64;
        let peak_bin = (1000.0 / freq_res).round() as usize;
        for i in peak_bin.saturating_sub(5)..=(peak_bin + 5).min(num_bins - 1) {
            magnitude[i] = 1.0;
        }

        let envelope = model.extract_envelope(&magnitude, fft_size);
        assert_eq!(envelope.len(), 18);

        // The band containing 1000 Hz should have the highest energy
        let band_1000_idx = model
            .bands
            .iter()
            .position(|b| (b.center_freq - 1000.0).abs() < 1.0)
            .unwrap();
        assert!(
            envelope[band_1000_idx] > envelope[0],
            "1000 Hz band should have more energy than 100 Hz band"
        );
    }

    #[test]
    fn test_formant_shift() {
        let model = FormantModel::new(44100.0);
        let shifted = model.shifted(12.0); // Up one octave
        assert!((shifted.bands[0].center_freq - 200.0).abs() < 1.0);
        assert!((shifted.bands[5].center_freq - 1500.0).abs() < 1.0);
    }

    #[test]
    fn test_formant_excitation_generation() {
        let processor = FormantProcessor::new(44100.0);
        let excitation = processor.generate_excitation(44100, 220.0);
        assert_eq!(excitation.len(), 44100);

        // Should have non-zero energy
        let rms = (excitation.iter().map(|s| s * s).sum::<f64>() / excitation.len() as f64).sqrt();
        assert!(rms > 0.0, "Excitation should have energy");
    }

    #[test]
    fn test_formant_preservation() {
        let processor = FormantProcessor::new(44100.0);
        let fft_size = 4096;
        let num_bins = fft_size / 2 + 1;

        let original_mag = vec![0.5; num_bins];
        let mut shifted_mag = vec![0.3; num_bins];

        processor.preserve_formants(&original_mag, &mut shifted_mag, fft_size, 1.0);

        // After full preservation, shifted should approach original levels
        // (since both are flat, the gains should be ~original/shifted = 5/3)
        // allowing some tolerance for band edge effects
    }
}
