//! Energy claim algorithm for polyphonic separation.
//!
//! Implements the core of Melodyne's DNA (Direct Note Access) polyphonic
//! separation. Notes compete to "claim" spectral energy from shared
//! frequency bins. This corresponds to
//! `MUAudioSourceDescription::assignEnergyClaimsInSampleRange()`.
//!
//! The algorithm:
//! 1. Each detected note generates expected harmonic energy at bins
//!    corresponding to its partials.
//! 2. Where harmonics of multiple notes overlap, energy is divided
//!    proportionally based on each note's expected contribution.
//! 3. The resulting per-note energy profiles drive the LSS resynthesis.

use crate::note::DetectedNote;
use crate::spectral::SpectralTimeSlice;

// ---------------------------------------------------------------------------
// Energy claim types
// ---------------------------------------------------------------------------

/// A claim by a single note on a frequency bin's energy.
#[derive(Debug, Clone)]
pub struct EnergyClaim {
    /// Index of the claiming note.
    pub note_index: usize,
    /// Frequency bin being claimed.
    pub bin: usize,
    /// Harmonic number this claim corresponds to (1 = fundamental).
    pub harmonic: usize,
    /// Expected relative strength of this harmonic (model-based).
    pub expected_weight: f64,
    /// Fraction of the bin's energy awarded to this note (0.0 .. 1.0).
    pub awarded_fraction: f64,
    /// Actual energy (magnitude) awarded.
    pub awarded_energy: f64,
}

/// Per-note energy profile across a time frame.
#[derive(Debug, Clone)]
pub struct NoteEnergyProfile {
    /// Index of the note.
    pub note_index: usize,
    /// Per-harmonic awarded energy (index 0 = fundamental, 1 = 2nd harmonic, etc.).
    pub harmonic_energies: Vec<f64>,
    /// Total energy for this note in this frame.
    pub total_energy: f64,
    /// Fundamental frequency used for the claim.
    pub fundamental: f64,
}

/// Result of energy claim assignment for a single time frame.
#[derive(Debug, Clone)]
pub struct EnergyClaimResult {
    /// Frame index.
    pub frame_index: usize,
    /// All individual claims made.
    pub claims: Vec<EnergyClaim>,
    /// Per-note energy profiles.
    pub note_profiles: Vec<NoteEnergyProfile>,
}

// ---------------------------------------------------------------------------
// Energy claim configuration
// ---------------------------------------------------------------------------

/// Configuration for the energy claim algorithm.
#[derive(Debug, Clone)]
pub struct EnergyClaimConfig {
    /// Maximum number of harmonics to consider per note.
    pub max_harmonics: usize,
    /// Harmonic decay model: amplitude of harmonic h relative to fundamental.
    /// Default: 1/h (sawtooth-like natural harmonic decay).
    pub harmonic_decay: HarmonicDecayModel,
    /// Bin radius around each harmonic peak to include in the claim.
    pub bin_radius: usize,
    /// Minimum energy in a bin to consider (noise floor).
    pub min_bin_energy: f64,
}

impl Default for EnergyClaimConfig {
    fn default() -> Self {
        Self {
            max_harmonics: 16,
            harmonic_decay: HarmonicDecayModel::InverseHarmonic,
            bin_radius: 2,
            min_bin_energy: 1e-6,
        }
    }
}

/// Model for expected harmonic decay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicDecayModel {
    /// 1/h decay (sawtooth-like).
    InverseHarmonic,
    /// 1/h^2 decay (softer, clarinet-like).
    InverseSquare,
    /// Flat (all harmonics equal -- useful for initial detection).
    Flat,
}

impl HarmonicDecayModel {
    /// Compute the expected relative weight for harmonic number h (1-based).
    pub fn weight(&self, h: usize) -> f64 {
        match self {
            Self::InverseHarmonic => 1.0 / h as f64,
            Self::InverseSquare => 1.0 / (h as f64 * h as f64),
            Self::Flat => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Energy claim engine
// ---------------------------------------------------------------------------

/// The energy claim engine that assigns spectral energy to competing notes.
pub struct EnergyClaimEngine {
    config: EnergyClaimConfig,
}

impl EnergyClaimEngine {
    /// Create a new energy claim engine.
    pub fn new(config: EnergyClaimConfig) -> Self {
        Self { config }
    }

    /// Assign energy claims for a single spectral time slice.
    ///
    /// Each note's harmonics "claim" energy from frequency bins. Where
    /// harmonics from multiple notes overlap, the energy is divided
    /// proportionally to each note's expected contribution.
    pub fn assign_claims(
        &self,
        slice: &SpectralTimeSlice,
        notes: &[DetectedNote],
    ) -> EnergyClaimResult {
        let num_bins = slice.num_bins();
        let freq_res = slice.freq_resolution();
        let sr = slice.sample_rate;
        let _fft_size = slice.fft_size;

        // Step 1: Build all claims
        let mut all_claims: Vec<EnergyClaim> = Vec::new();

        // Per-bin accumulator for total claimed weight
        let mut bin_total_weight = vec![0.0f64; num_bins];

        for (note_idx, note) in notes.iter().enumerate() {
            // Skip notes that don't overlap this time frame
            let frame_time = slice.time_seconds;
            if frame_time < note.start_time || frame_time > note.start_time + note.duration {
                continue;
            }

            let f0 = note.frequency;

            for h in 1..=self.config.max_harmonics {
                let harmonic_freq = f0 * h as f64;
                if harmonic_freq > sr / 2.0 {
                    break;
                }

                let center_bin = (harmonic_freq / freq_res).round() as usize;
                if center_bin >= num_bins {
                    break;
                }

                let expected_weight = self.config.harmonic_decay.weight(h);

                // Claim bins around the harmonic center
                let start_bin = center_bin.saturating_sub(self.config.bin_radius);
                let end_bin = (center_bin + self.config.bin_radius + 1).min(num_bins);

                for bin in start_bin..end_bin {
                    // Distance-weighted claim (Gaussian-like falloff from center)
                    let dist = (bin as f64 - center_bin as f64).abs();
                    let spatial_weight = (-dist * dist / 2.0).exp();
                    let claim_weight = expected_weight * spatial_weight;

                    all_claims.push(EnergyClaim {
                        note_index: note_idx,
                        bin,
                        harmonic: h,
                        expected_weight: claim_weight,
                        awarded_fraction: 0.0,
                        awarded_energy: 0.0,
                    });

                    bin_total_weight[bin] += claim_weight;
                }
            }
        }

        // Step 2: Divide energy proportionally
        for claim in &mut all_claims {
            let total = bin_total_weight[claim.bin];
            if total > 1e-30 && slice.magnitude[claim.bin] > self.config.min_bin_energy {
                claim.awarded_fraction = claim.expected_weight / total;
                claim.awarded_energy = slice.magnitude[claim.bin] * claim.awarded_fraction;
            }
        }

        // Step 3: Build per-note energy profiles
        let mut note_profiles: Vec<NoteEnergyProfile> = notes
            .iter()
            .enumerate()
            .map(|(i, note)| NoteEnergyProfile {
                note_index: i,
                harmonic_energies: vec![0.0; self.config.max_harmonics],
                total_energy: 0.0,
                fundamental: note.frequency,
            })
            .collect();

        for claim in &all_claims {
            if claim.harmonic > 0 && claim.harmonic <= self.config.max_harmonics {
                let profile = &mut note_profiles[claim.note_index];
                profile.harmonic_energies[claim.harmonic - 1] += claim.awarded_energy;
                profile.total_energy += claim.awarded_energy;
            }
        }

        EnergyClaimResult {
            frame_index: slice.frame_index,
            claims: all_claims,
            note_profiles,
        }
    }

    /// Assign energy claims across a range of spectral slices.
    pub fn assign_claims_in_range(
        &self,
        slices: &[SpectralTimeSlice],
        notes: &[DetectedNote],
    ) -> Vec<EnergyClaimResult> {
        slices
            .iter()
            .map(|slice| self.assign_claims(slice, notes))
            .collect()
    }

    /// Access the configuration.
    pub fn config(&self) -> &EnergyClaimConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::{ReassignedSpectrogram, ReassignedSpectrogramConfig};
    use std::f64::consts::PI;

    fn sine_wave(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (2.0 * PI * freq * i as f64 / sr).sin()).collect()
    }

    #[test]
    fn test_harmonic_decay_models() {
        assert!((HarmonicDecayModel::InverseHarmonic.weight(1) - 1.0).abs() < 1e-10);
        assert!((HarmonicDecayModel::InverseHarmonic.weight(2) - 0.5).abs() < 1e-10);
        assert!((HarmonicDecayModel::InverseSquare.weight(3) - 1.0 / 9.0).abs() < 1e-10);
        assert!((HarmonicDecayModel::Flat.weight(5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_claim_single_note() {
        let sr = 44100.0;
        let freq = 440.0;
        let audio = sine_wave(freq, sr, 44100);

        let config = ReassignedSpectrogramConfig {
            sample_rate: sr,
            fft_size: 4096,
            hop_size: 1024,
            ..Default::default()
        };
        let mut analyzer = ReassignedSpectrogram::new(config);
        let slices = analyzer.analyze(&audio);

        let note = DetectedNote {
            frequency: freq,
            midi_note: 69.0,
            start_time: 0.0,
            duration: 1.0,
            start_sample: 0,
            end_sample: 44100,
            amplitude: 0.5,
            peak_amplitude: 0.7,
            confidence: 0.9,
            pitch_contour: vec![freq],
            amplitude_contour: vec![0.5],
            vibrato_rate: 0.0,
            vibrato_depth: 0.0,
            attack_time: 0.0,
            release_time: 0.0,
        };

        let engine = EnergyClaimEngine::new(EnergyClaimConfig::default());
        let mid = &slices[slices.len() / 2];
        let result = engine.assign_claims(mid, &[note]);

        // Single note should get all energy
        assert!(!result.claims.is_empty());
        for claim in &result.claims {
            assert!((claim.awarded_fraction - 1.0).abs() < 1e-10);
        }
        assert_eq!(result.note_profiles.len(), 1);
        assert!(result.note_profiles[0].total_energy > 0.0);
    }

    #[test]
    fn test_energy_claim_two_notes_sharing() {
        let sr = 44100.0;
        // Two notes: A4 (440) and E5 (659.26) -- harmonics overlap at 1320 Hz
        let audio: Vec<f64> = (0..44100)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 440.0 * t).sin() + (2.0 * PI * 659.26 * t).sin()
            })
            .collect();

        let config = ReassignedSpectrogramConfig {
            sample_rate: sr,
            fft_size: 4096,
            hop_size: 1024,
            ..Default::default()
        };
        let mut analyzer = ReassignedSpectrogram::new(config);
        let slices = analyzer.analyze(&audio);

        let notes = vec![
            DetectedNote {
                frequency: 440.0,
                midi_note: 69.0,
                start_time: 0.0,
                duration: 1.0,
                start_sample: 0,
                end_sample: 44100,
                amplitude: 0.5,
                peak_amplitude: 0.5,
                confidence: 0.9,
                pitch_contour: vec![440.0],
                amplitude_contour: vec![0.5],
                vibrato_rate: 0.0,
                vibrato_depth: 0.0,
                attack_time: 0.0,
                release_time: 0.0,
            },
            DetectedNote {
                frequency: 659.26,
                midi_note: 76.0,
                start_time: 0.0,
                duration: 1.0,
                start_sample: 0,
                end_sample: 44100,
                amplitude: 0.5,
                peak_amplitude: 0.5,
                confidence: 0.9,
                pitch_contour: vec![659.26],
                amplitude_contour: vec![0.5],
                vibrato_rate: 0.0,
                vibrato_depth: 0.0,
                attack_time: 0.0,
                release_time: 0.0,
            },
        ];

        let engine = EnergyClaimEngine::new(EnergyClaimConfig::default());
        let mid = &slices[slices.len() / 2];
        let result = engine.assign_claims(mid, &notes);

        assert_eq!(result.note_profiles.len(), 2);
        // Both notes should have some energy
        assert!(result.note_profiles[0].total_energy > 0.0);
        assert!(result.note_profiles[1].total_energy > 0.0);

        // Where bins are shared, fractions should sum to ~1.0
        let shared_bins: Vec<usize> = result
            .claims
            .iter()
            .filter(|c| c.note_index == 0)
            .map(|c| c.bin)
            .collect();

        for &bin in &shared_bins {
            let total_fraction: f64 = result
                .claims
                .iter()
                .filter(|c| c.bin == bin)
                .map(|c| c.awarded_fraction)
                .sum();
            assert!(
                (total_fraction - 1.0).abs() < 0.01,
                "Bin {} fractions sum to {} (expected ~1.0)",
                bin,
                total_fraction
            );
        }
    }
}
