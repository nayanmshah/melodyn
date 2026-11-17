//! Five-component audio decomposition.
//!
//! Decomposes audio into five perceptual components mirroring
//! MUDecomposedAudioSignal:
//! - **Tonal**: Stable harmonic partials (the main pitched content)
//! - **Raspy**: Noisy but pitched content (vocal rasp, distortion)
//! - **Consonants**: Voiced consonants and fricatives
//! - **Sibilant**: High-frequency noise (s, sh, ch sounds)
//! - **Breath**: Aspirated noise, breath sounds
//!
//! The decomposition is driven by the spectral analysis: each bin's energy
//! is routed to the appropriate component based on tonality classification,
//! frequency range, and temporal behavior.

use crate::spectral::{SpectralTimeSlice, TonalityClass};

// ---------------------------------------------------------------------------
// Decomposition types
// ---------------------------------------------------------------------------

/// The five audio decomposition components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecompositionComponent {
    /// Stable harmonic partials -- the core pitched content.
    Tonal,
    /// Noisy but pitch-correlated content -- vocal rasp, growl, distortion.
    Raspy,
    /// Voiced consonants and low-mid frequency transient noise.
    Consonants,
    /// High-frequency unvoiced noise (sibilants: s, sh, ch, etc.).
    Sibilant,
    /// Aspirated noise, breath sounds, low-energy broadband noise.
    Breath,
}

impl DecompositionComponent {
    /// All five components in order.
    pub const ALL: [DecompositionComponent; 5] = [
        Self::Tonal,
        Self::Raspy,
        Self::Consonants,
        Self::Sibilant,
        Self::Breath,
    ];

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tonal => "Tonal",
            Self::Raspy => "Raspy",
            Self::Consonants => "Consonants",
            Self::Sibilant => "Sibilant",
            Self::Breath => "Breath",
        }
    }
}

/// Per-frame decomposition result: energy distribution across components.
#[derive(Debug, Clone)]
pub struct FrameDecomposition {
    /// Frame index.
    pub frame_index: usize,
    /// Time in seconds.
    pub time: f64,
    /// Energy per component (linear magnitude sum).
    pub component_energy: [f64; 5],
    /// Per-bin component assignment (which component owns each bin).
    pub bin_assignments: Vec<DecompositionComponent>,
    /// Per-bin component weights [bin][component] -- soft assignment.
    pub bin_weights: Vec<[f64; 5]>,
}

impl FrameDecomposition {
    /// Get energy for a specific component.
    pub fn energy(&self, component: DecompositionComponent) -> f64 {
        self.component_energy[component as usize]
    }

    /// Get the dominant component for this frame.
    pub fn dominant_component(&self) -> DecompositionComponent {
        let (idx, _) = self
            .component_energy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        DecompositionComponent::ALL[idx]
    }

    /// Total energy across all components.
    pub fn total_energy(&self) -> f64 {
        self.component_energy.iter().sum()
    }
}

/// Complete decomposition result for an audio buffer.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Per-frame decomposition.
    pub frames: Vec<FrameDecomposition>,
    /// Sample rate.
    pub sample_rate: f64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the audio decomposition.
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Frequency boundary between consonants and sibilants (Hz).
    pub sibilant_freq: f64,
    /// Frequency below which noise is classified as breath (Hz).
    pub breath_freq_upper: f64,
    /// Minimum magnitude to consider a bin (noise floor).
    pub noise_floor: f64,
    /// Raspy detection: how much noise energy coexists with tonal peaks.
    pub raspy_threshold: f64,
    /// Consonant detection: transiency score threshold.
    pub consonant_transiency_threshold: f64,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            sibilant_freq: 5000.0,
            breath_freq_upper: 500.0,
            noise_floor: 1e-6,
            raspy_threshold: 0.3,
            consonant_transiency_threshold: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// Decomposer
// ---------------------------------------------------------------------------

/// Audio decomposer that splits spectral energy into five components.
pub struct AudioDecomposer {
    config: DecompositionConfig,
}

impl AudioDecomposer {
    /// Create a new audio decomposer.
    pub fn new(config: DecompositionConfig) -> Self {
        Self { config }
    }

    /// Decompose a single spectral time slice into five components.
    pub fn decompose_frame(&self, slice: &SpectralTimeSlice) -> FrameDecomposition {
        let num_bins = slice.num_bins();
        let freq_res = slice.freq_resolution();
        let is_onset = slice.transiency.is_onset;
        let transiency_score = slice.transiency.score;

        let mut bin_assignments = vec![DecompositionComponent::Breath; num_bins];
        let mut bin_weights = vec![[0.0f64; 5]; num_bins];
        let mut component_energy = [0.0f64; 5];

        for k in 0..num_bins {
            let mag = slice.magnitude[k];
            if mag < self.config.noise_floor {
                // Below noise floor -- assign to breath with tiny weight
                bin_weights[k][DecompositionComponent::Breath as usize] = 1.0;
                continue;
            }

            let freq = k as f64 * freq_res;
            let tonality = slice.tonality[k];

            // Classify based on tonality, frequency, and transiency
            let weights = self.classify_bin(freq, tonality, mag, transiency_score, is_onset);
            bin_weights[k] = weights;

            // Find dominant component for hard assignment
            let (best_idx, _) = weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            bin_assignments[k] = DecompositionComponent::ALL[best_idx];

            // Accumulate weighted energy
            for c in 0..5 {
                component_energy[c] += mag * weights[c];
            }
        }

        FrameDecomposition {
            frame_index: slice.frame_index,
            time: slice.time_seconds,
            component_energy,
            bin_assignments,
            bin_weights,
        }
    }

    /// Decompose all spectral slices.
    pub fn decompose(&self, slices: &[SpectralTimeSlice]) -> DecompositionResult {
        let frames = slices.iter().map(|s| self.decompose_frame(s)).collect();
        DecompositionResult {
            frames,
            sample_rate: self.config.sample_rate,
        }
    }

    /// Classify a single bin, returning soft weights for each component.
    fn classify_bin(
        &self,
        freq: f64,
        tonality: TonalityClass,
        magnitude: f64,
        transiency_score: f64,
        is_onset: bool,
    ) -> [f64; 5] {
        let mut weights = [0.0f64; 5];

        match tonality {
            TonalityClass::Tonal => {
                // Primarily tonal, but check for raspy coexistence
                weights[DecompositionComponent::Tonal as usize] = 0.85;
                weights[DecompositionComponent::Raspy as usize] = 0.15;
            }
            TonalityClass::Mixed => {
                // Mixed: split between tonal and raspy based on frequency
                weights[DecompositionComponent::Tonal as usize] = 0.4;
                weights[DecompositionComponent::Raspy as usize] = 0.4;
                // Some consonant/sibilant contribution
                if freq > self.config.sibilant_freq {
                    weights[DecompositionComponent::Sibilant as usize] = 0.2;
                } else {
                    weights[DecompositionComponent::Consonants as usize] = 0.2;
                }
            }
            TonalityClass::Noise => {
                if freq > self.config.sibilant_freq {
                    // High-frequency noise -> sibilant
                    weights[DecompositionComponent::Sibilant as usize] = 0.8;
                    weights[DecompositionComponent::Breath as usize] = 0.2;
                } else if freq < self.config.breath_freq_upper {
                    // Low-frequency noise -> breath
                    weights[DecompositionComponent::Breath as usize] = 0.7;
                    weights[DecompositionComponent::Consonants as usize] = 0.3;
                } else {
                    // Mid-frequency noise -> consonants or raspy
                    weights[DecompositionComponent::Consonants as usize] = 0.5;
                    weights[DecompositionComponent::Raspy as usize] = 0.3;
                    weights[DecompositionComponent::Breath as usize] = 0.2;
                }
            }
            TonalityClass::Transient => {
                // Transient energy
                if freq > self.config.sibilant_freq {
                    weights[DecompositionComponent::Sibilant as usize] = 0.6;
                    weights[DecompositionComponent::Consonants as usize] = 0.4;
                } else {
                    weights[DecompositionComponent::Consonants as usize] = 0.7;
                    weights[DecompositionComponent::Raspy as usize] = 0.3;
                }
            }
        }

        // Modulate by transiency: onset frames push energy toward consonants
        if is_onset && transiency_score > self.config.consonant_transiency_threshold {
            let onset_boost = (transiency_score - self.config.consonant_transiency_threshold)
                .min(0.5);
            weights[DecompositionComponent::Consonants as usize] += onset_boost;
            // Reduce tonal proportionally
            weights[DecompositionComponent::Tonal as usize] =
                (weights[DecompositionComponent::Tonal as usize] - onset_boost * 0.5).max(0.0);
        }

        // Normalize weights to sum to 1.0
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }

    /// Access the configuration.
    pub fn config(&self) -> &DecompositionConfig {
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
    fn test_decomposition_component_names() {
        assert_eq!(DecompositionComponent::Tonal.name(), "Tonal");
        assert_eq!(DecompositionComponent::Sibilant.name(), "Sibilant");
    }

    #[test]
    fn test_decompose_pure_tone() {
        let sr = 44100.0;
        let audio = sine_wave(440.0, sr, 44100);

        let config = ReassignedSpectrogramConfig {
            sample_rate: sr,
            fft_size: 4096,
            hop_size: 1024,
            ..Default::default()
        };
        let mut analyzer = ReassignedSpectrogram::new(config);
        let slices = analyzer.analyze(&audio);

        let decomposer = AudioDecomposer::new(DecompositionConfig {
            sample_rate: sr,
            ..Default::default()
        });
        let result = decomposer.decompose(&slices);

        assert!(!result.frames.is_empty());

        // For a pure sine, most energy should be in the tonal component
        let mid = &result.frames[result.frames.len() / 2];
        let tonal = mid.energy(DecompositionComponent::Tonal);
        let total = mid.total_energy();
        assert!(
            tonal / total > 0.5,
            "Tonal fraction {:.2} should dominate for a pure sine",
            tonal / total
        );
        assert_eq!(mid.dominant_component(), DecompositionComponent::Tonal);
    }

    #[test]
    fn test_decompose_noise() {
        let sr = 44100.0;
        // White noise
        let audio: Vec<f64> = (0..44100)
            .map(|i| {
                // Simple pseudo-random using sin of large numbers
                (i as f64 * 12345.6789).sin()
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

        let decomposer = AudioDecomposer::new(DecompositionConfig {
            sample_rate: sr,
            ..Default::default()
        });
        let result = decomposer.decompose(&slices);

        // For noise, tonal should not dominate
        let mid = &result.frames[result.frames.len() / 2];
        let tonal = mid.energy(DecompositionComponent::Tonal);
        let total = mid.total_energy();
        // Noise spreads across noise-like components
        assert!(
            tonal / total < 0.7,
            "Tonal fraction {:.2} should not dominate for noise",
            tonal / total
        );
    }

    #[test]
    fn test_soft_weights_sum_to_one() {
        let decomposer = AudioDecomposer::new(DecompositionConfig::default());
        let weights = decomposer.classify_bin(1000.0, TonalityClass::Mixed, 0.5, 0.3, false);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights should sum to 1.0, got {}",
            sum
        );
    }
}
