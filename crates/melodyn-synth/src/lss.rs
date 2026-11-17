//! LSS (Local Sound Synthesis) generator and correlator.
//!
//! Implements per-partial sinusoidal synthesis with overlap-add and phase
//! continuity. This corresponds to MULSSGenerator and MULSSCorrelator from
//! the original Melodyne engine.
//!
//! The LSS model represents each note as a sum of sinusoidal partials,
//! each with time-varying amplitude and phase. The generator produces
//! audio from this model, while the correlator extracts the model
//! parameters from analyzed spectral data.

use melodyn_core::energy_claim::NoteEnergyProfile;
use melodyn_core::note::DetectedNote;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Partial (single sinusoidal component)
// ---------------------------------------------------------------------------

/// A single sinusoidal partial in the LSS model.
#[derive(Debug, Clone)]
pub struct LssPartial {
    /// Harmonic number (1 = fundamental, 2 = second harmonic, etc.).
    pub harmonic: usize,
    /// Frequency in Hz (may differ from exact harmonic due to inharmonicity).
    pub frequency: f64,
    /// Time-varying amplitude envelope (one value per synthesis frame).
    pub synth_amps: Vec<f64>,
    /// Time-varying phase values (one value per synthesis frame, radians).
    pub synth_phases: Vec<f64>,
    /// Whether this partial is active (has significant energy).
    pub active: bool,
}

impl LssPartial {
    /// Create a new partial with the given harmonic number and frame count.
    pub fn new(harmonic: usize, frequency: f64, num_frames: usize) -> Self {
        Self {
            harmonic,
            frequency,
            synth_amps: vec![0.0; num_frames],
            synth_phases: vec![0.0; num_frames],
            active: false,
        }
    }

    /// Amplitude at a fractional frame position (linear interpolation).
    pub fn amplitude_at(&self, frame: f64) -> f64 {
        if self.synth_amps.is_empty() {
            return 0.0;
        }
        let idx = frame.floor() as usize;
        let frac = frame - frame.floor();
        if idx + 1 < self.synth_amps.len() {
            self.synth_amps[idx] * (1.0 - frac) + self.synth_amps[idx + 1] * frac
        } else if idx < self.synth_amps.len() {
            self.synth_amps[idx]
        } else {
            0.0
        }
    }

    /// Phase at a fractional frame position (linear interpolation with unwrapping).
    pub fn phase_at(&self, frame: f64) -> f64 {
        if self.synth_phases.is_empty() {
            return 0.0;
        }
        let idx = frame.floor() as usize;
        let frac = frame - frame.floor();
        if idx + 1 < self.synth_phases.len() {
            let p0 = self.synth_phases[idx];
            let mut p1 = self.synth_phases[idx + 1];
            // Unwrap phase difference
            let diff = p1 - p0;
            let wrapped = diff - (diff / (2.0 * PI)).round() * 2.0 * PI;
            p1 = p0 + wrapped;
            p0 * (1.0 - frac) + p1 * frac
        } else if idx < self.synth_phases.len() {
            self.synth_phases[idx]
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// LSS model (per-note)
// ---------------------------------------------------------------------------

/// Complete LSS model for a single note.
#[derive(Debug, Clone)]
pub struct LssNoteModel {
    /// Reference to the detected note this model represents.
    pub note_index: usize,
    /// Fundamental frequency (Hz).
    pub fundamental: f64,
    /// All partials (harmonics) of this note.
    pub partials: Vec<LssPartial>,
    /// Number of synthesis frames.
    pub num_frames: usize,
    /// Synthesis frame rate (frames per second).
    pub frame_rate: f64,
    /// Synthesis hop size in samples.
    pub hop_size: usize,
    /// Sample rate.
    pub sample_rate: f64,
}

impl LssNoteModel {
    /// Create a new LSS model for a note with the given number of harmonics.
    pub fn new(
        note_index: usize,
        fundamental: f64,
        num_harmonics: usize,
        num_frames: usize,
        hop_size: usize,
        sample_rate: f64,
    ) -> Self {
        let partials = (1..=num_harmonics)
            .map(|h| LssPartial::new(h, fundamental * h as f64, num_frames))
            .collect();

        Self {
            note_index,
            fundamental,
            partials,
            num_frames,
            frame_rate: sample_rate / hop_size as f64,
            hop_size,
            sample_rate,
        }
    }

    /// Total duration in seconds.
    pub fn duration(&self) -> f64 {
        self.num_frames as f64 / self.frame_rate
    }

    /// Number of output samples.
    pub fn num_samples(&self) -> usize {
        self.num_frames * self.hop_size
    }
}

// ---------------------------------------------------------------------------
// LSS Correlator
// ---------------------------------------------------------------------------

/// Extracts LSS model parameters from energy claim profiles.
///
/// The correlator takes the per-note energy profiles from the energy claim
/// algorithm and converts them into per-partial amplitude and phase
/// trajectories for the LSS model.
pub struct LssCorrelator {
    /// Maximum number of harmonics to model.
    pub max_harmonics: usize,
    /// Synthesis hop size in samples.
    pub hop_size: usize,
    /// Sample rate.
    pub sample_rate: f64,
}

impl LssCorrelator {
    /// Create a new correlator.
    pub fn new(max_harmonics: usize, hop_size: usize, sample_rate: f64) -> Self {
        Self {
            max_harmonics,
            hop_size,
            sample_rate,
        }
    }

    /// Build an LSS model from energy claim profiles for a single note.
    ///
    /// Takes the sequence of per-frame energy profiles and constructs
    /// the amplitude and phase trajectories for each partial.
    pub fn build_model(
        &self,
        note: &DetectedNote,
        note_index: usize,
        energy_profiles: &[NoteEnergyProfile],
    ) -> LssNoteModel {
        let num_frames = energy_profiles.len();
        let mut model = LssNoteModel::new(
            note_index,
            note.frequency,
            self.max_harmonics,
            num_frames,
            self.hop_size,
            self.sample_rate,
        );

        // Fill in amplitude from energy profiles
        for (frame_idx, profile) in energy_profiles.iter().enumerate() {
            for (h_idx, partial) in model.partials.iter_mut().enumerate() {
                if h_idx < profile.harmonic_energies.len() {
                    partial.synth_amps[frame_idx] = profile.harmonic_energies[h_idx];
                    if profile.harmonic_energies[h_idx] > 1e-10 {
                        partial.active = true;
                    }
                }
            }
        }

        // Compute phase trajectories with continuity.
        // Phase advances by 2*pi*f*hop_size/sr per frame.
        for partial in &mut model.partials {
            if !partial.active {
                continue;
            }
            let freq = partial.frequency;
            let phase_inc = 2.0 * PI * freq * self.hop_size as f64 / self.sample_rate;
            let mut phase = 0.0f64;
            for i in 0..num_frames {
                partial.synth_phases[i] = phase;
                phase += phase_inc;
                // Keep phase in reasonable range to avoid precision loss
                if phase > 1000.0 * PI {
                    phase -= (phase / (2.0 * PI)).floor() * 2.0 * PI;
                }
            }
        }

        model
    }

    /// Build an LSS model directly from a detected note (without energy claims).
    /// Uses a simple harmonic model with 1/h amplitude decay.
    pub fn build_model_simple(&self, note: &DetectedNote, note_index: usize) -> LssNoteModel {
        let duration = note.duration;
        let num_frames = (duration * self.sample_rate / self.hop_size as f64).ceil() as usize;
        let num_frames = num_frames.max(1);

        let mut model = LssNoteModel::new(
            note_index,
            note.frequency,
            self.max_harmonics,
            num_frames,
            self.hop_size,
            self.sample_rate,
        );

        for partial in &mut model.partials {
            let h = partial.harmonic;
            let base_amp = note.amplitude / h as f64;

            // Simple amplitude envelope from the note's amplitude contour
            for i in 0..num_frames {
                let t = note.start_time + i as f64 / model.frame_rate;
                let env = note.amplitude_at(t) / note.amplitude.max(1e-10);
                partial.synth_amps[i] = base_amp * env;
            }

            // Phase trajectory
            let freq = partial.frequency;
            let phase_inc = 2.0 * PI * freq * self.hop_size as f64 / self.sample_rate;
            let mut phase = 0.0f64;
            for i in 0..num_frames {
                partial.synth_phases[i] = phase;
                phase += phase_inc;
                if phase > 1000.0 * PI {
                    phase -= (phase / (2.0 * PI)).floor() * 2.0 * PI;
                }
            }

            partial.active = base_amp > 1e-10;
        }

        model
    }
}

// ---------------------------------------------------------------------------
// LSS Generator
// ---------------------------------------------------------------------------

/// Configuration for the LSS generator.
#[derive(Debug, Clone)]
pub struct LssGeneratorConfig {
    /// Sample rate.
    pub sample_rate: f64,
    /// Synthesis hop size in samples.
    pub hop_size: usize,
    /// Maximum number of harmonics per note.
    pub max_harmonics: usize,
    /// Whether to apply pitch shifting.
    pub pitch_shift_semitones: f64,
    /// Whether to apply time stretching.
    pub time_stretch_factor: f64,
}

impl Default for LssGeneratorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            hop_size: 256,
            max_harmonics: 16,
            pitch_shift_semitones: 0.0,
            time_stretch_factor: 1.0,
        }
    }
}

/// LSS Generator -- synthesizes audio from LSS note models.
///
/// Corresponds to MULSSGenerator from the original engine.
/// Performs additive sinusoidal synthesis with per-partial amplitude
/// and phase control, overlap-add windowing, and phase continuity.
pub struct LssGenerator {
    config: LssGeneratorConfig,
}

impl LssGenerator {
    /// Create a new LSS generator.
    pub fn new(config: LssGeneratorConfig) -> Self {
        Self { config }
    }

    /// Synthesize a single note from its LSS model.
    ///
    /// Returns a buffer of audio samples.
    pub fn synthesize_note(&self, model: &LssNoteModel) -> Vec<f64> {
        let num_samples = model.num_samples();
        let mut output = vec![0.0f64; num_samples];
        let hop = self.config.hop_size;
        let sr = self.config.sample_rate;
        let pitch_ratio = 2.0f64.powf(self.config.pitch_shift_semitones / 12.0);

        for partial in &model.partials {
            if !partial.active {
                continue;
            }

            let freq = partial.frequency * pitch_ratio;
            if freq > sr / 2.0 {
                continue; // Skip partials above Nyquist
            }

            // Synthesize this partial using per-frame amplitude/phase interpolation
            let mut phase = 0.0f64;
            let omega = 2.0 * PI * freq / sr;

            for sample_idx in 0..num_samples {
                let frame_pos = sample_idx as f64 / hop as f64;
                let amp = partial.amplitude_at(frame_pos);

                output[sample_idx] += amp * phase.sin();
                phase += omega;

                // Prevent phase accumulation overflow
                if phase > 4.0 * PI {
                    phase -= 2.0 * PI;
                }
            }
        }

        output
    }

    /// Synthesize multiple notes and mix them together.
    pub fn synthesize_mix(&self, models: &[LssNoteModel]) -> Vec<f64> {
        if models.is_empty() {
            return vec![];
        }

        // Find total length needed
        let max_samples = models.iter().map(|m| m.num_samples()).max().unwrap_or(0);
        let mut output = vec![0.0f64; max_samples];

        for model in models {
            let note_audio = self.synthesize_note(model);
            for (i, &s) in note_audio.iter().enumerate() {
                if i < output.len() {
                    output[i] += s;
                }
            }
        }

        output
    }

    /// Access the configuration.
    pub fn config(&self) -> &LssGeneratorConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_note() -> DetectedNote {
        DetectedNote {
            frequency: 440.0,
            midi_note: 69.0,
            start_time: 0.0,
            duration: 0.5,
            start_sample: 0,
            end_sample: 22050,
            amplitude: 0.5,
            peak_amplitude: 0.5,
            confidence: 0.9,
            pitch_contour: vec![440.0; 10],
            amplitude_contour: vec![0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.1],
            vibrato_rate: 0.0,
            vibrato_depth: 0.0,
            attack_time: 0.0,
            release_time: 0.0,
        }
    }

    #[test]
    fn test_lss_partial_interpolation() {
        let mut partial = LssPartial::new(1, 440.0, 4);
        partial.synth_amps = vec![0.0, 0.5, 1.0, 0.5];
        partial.active = true;

        assert!((partial.amplitude_at(0.0) - 0.0).abs() < 1e-10);
        assert!((partial.amplitude_at(1.0) - 0.5).abs() < 1e-10);
        assert!((partial.amplitude_at(1.5) - 0.75).abs() < 1e-10);
        assert!((partial.amplitude_at(2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lss_model_creation() {
        let model = LssNoteModel::new(0, 440.0, 8, 100, 256, 44100.0);
        assert_eq!(model.partials.len(), 8);
        assert_eq!(model.partials[0].harmonic, 1);
        assert_eq!(model.partials[7].harmonic, 8);
        assert!((model.partials[2].frequency - 1320.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlator_simple_model() {
        let note = make_test_note();
        let correlator = LssCorrelator::new(8, 256, 44100.0);
        let model = correlator.build_model_simple(&note, 0);

        assert!(model.num_frames > 0);
        // Fundamental should have amplitude
        assert!(model.partials[0].active);
        assert!(model.partials[0].synth_amps.iter().any(|&a| a > 0.0));
    }

    #[test]
    fn test_lss_synthesis_produces_audio() {
        let note = make_test_note();
        let correlator = LssCorrelator::new(4, 256, 44100.0);
        let model = correlator.build_model_simple(&note, 0);

        let generator = LssGenerator::new(LssGeneratorConfig::default());
        let audio = generator.synthesize_note(&model);

        assert!(!audio.is_empty());
        // Should have non-silent audio
        let rms = (audio.iter().map(|s| s * s).sum::<f64>() / audio.len() as f64).sqrt();
        assert!(rms > 0.001, "RMS {} should be > 0.001", rms);
    }

    #[test]
    fn test_lss_synthesis_pitch_shift() {
        let note = make_test_note();
        let correlator = LssCorrelator::new(1, 256, 44100.0);
        let model = correlator.build_model_simple(&note, 0);

        // Synthesize at original pitch
        let gen_original = LssGenerator::new(LssGeneratorConfig {
            pitch_shift_semitones: 0.0,
            ..Default::default()
        });
        let audio_original = gen_original.synthesize_note(&model);

        // Synthesize shifted up 12 semitones (octave)
        let gen_shifted = LssGenerator::new(LssGeneratorConfig {
            pitch_shift_semitones: 12.0,
            ..Default::default()
        });
        let audio_shifted = gen_shifted.synthesize_note(&model);

        // Both should produce audio
        assert!(!audio_original.is_empty());
        assert!(!audio_shifted.is_empty());
    }
}
