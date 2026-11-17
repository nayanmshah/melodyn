//! Detection presets and configuration.
//!
//! Defines the eight detection presets from Melodyne and their associated
//! parameter tunings. These presets control how the analysis engine behaves
//! for different types of audio content.

// ---------------------------------------------------------------------------
// Detection presets
// ---------------------------------------------------------------------------

/// Detection preset, controlling analysis behavior for different content types.
///
/// These correspond to the presets found in Melodyne's detection mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectionPreset {
    /// Automatic mode: engine auto-detects mono/poly and content type.
    Automatic,
    /// Melodic: optimized for monophonic melodic content (vocals, solo instruments).
    Melodic,
    /// Melodic with adjusted attack detection for cleaner note onsets.
    MelodicAdjustAttacks,
    /// Melodic without sibilant detection (useful for non-vocal content).
    MelodicNoSibilants,
    /// Melodic-percussive hybrid for content with both pitched and percussive elements.
    MelodicPercussive,
    /// Percussive: optimized for drums and unpitched percussion.
    Percussive,
    /// Polyphonic: DNA analysis for multi-note content (piano, guitar, etc.).
    Polyphonic,
    /// Polyphonic unpitched: for complex layered sounds without clear pitch.
    PolyphonicUnpitched,
}

impl DetectionPreset {
    /// All available presets.
    pub const ALL: [DetectionPreset; 8] = [
        Self::Automatic,
        Self::Melodic,
        Self::MelodicAdjustAttacks,
        Self::MelodicNoSibilants,
        Self::MelodicPercussive,
        Self::Percussive,
        Self::Polyphonic,
        Self::PolyphonicUnpitched,
    ];

    /// Display name for the preset.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Automatic => "Automatic",
            Self::Melodic => "Melodic",
            Self::MelodicAdjustAttacks => "Melodic (Adjust Attacks)",
            Self::MelodicNoSibilants => "Melodic (No Sibilants)",
            Self::MelodicPercussive => "Melodic Percussive",
            Self::Percussive => "Percussive",
            Self::Polyphonic => "Polyphonic",
            Self::PolyphonicUnpitched => "Polyphonic Unpitched",
        }
    }

    /// Whether this preset uses polyphonic (DNA) analysis.
    pub fn is_polyphonic(&self) -> bool {
        matches!(self, Self::Polyphonic | Self::PolyphonicUnpitched)
    }

    /// Whether this preset expects pitched content.
    pub fn expects_pitched(&self) -> bool {
        !matches!(self, Self::Percussive | Self::PolyphonicUnpitched)
    }

    /// Whether sibilant detection is enabled.
    pub fn detect_sibilants(&self) -> bool {
        !matches!(self, Self::MelodicNoSibilants)
    }

    /// Whether attack adjustment is enabled.
    pub fn adjust_attacks(&self) -> bool {
        matches!(
            self,
            Self::MelodicAdjustAttacks | Self::MelodicPercussive | Self::Percussive
        )
    }

    /// Parse a preset from a string name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "automatic" | "auto" => Some(Self::Automatic),
            "melodic" => Some(Self::Melodic),
            "melodic_adjust_attacks" | "melodicadjustattacks" => Some(Self::MelodicAdjustAttacks),
            "melodic_no_sibilants" | "melodicnosibilants" => Some(Self::MelodicNoSibilants),
            "melodic_percussive" | "melodicpercussive" => Some(Self::MelodicPercussive),
            "percussive" => Some(Self::Percussive),
            "polyphonic" | "poly" => Some(Self::Polyphonic),
            "polyphonic_unpitched" | "polyphonicunpitched" => Some(Self::PolyphonicUnpitched),
            _ => None,
        }
    }
}

impl std::fmt::Display for DetectionPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Detection parameters (tunable per-preset)
// ---------------------------------------------------------------------------

/// Fine-grained detection parameters derived from a preset.
#[derive(Debug, Clone)]
pub struct DetectionParams {
    /// The preset these parameters were derived from.
    pub preset: DetectionPreset,
    /// FFT size for primary analysis.
    pub fft_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Minimum note frequency (Hz).
    pub min_frequency: f64,
    /// Maximum note frequency (Hz).
    pub max_frequency: f64,
    /// YIN aperiodicity threshold (lower = stricter pitch detection).
    pub yin_threshold: f64,
    /// Onset detection sensitivity (lower = more sensitive).
    pub onset_threshold: f64,
    /// Minimum note duration in seconds.
    pub min_note_duration: f64,
    /// Minimum note amplitude (linear).
    pub min_note_amplitude: f64,
    /// Whether to use polyphonic mode.
    pub polyphonic: bool,
    /// Maximum simultaneous notes (polyphonic mode).
    pub max_polyphony: usize,
    /// Whether sibilant detection is active.
    pub detect_sibilants: bool,
    /// Whether to adjust note attacks.
    pub adjust_attacks: bool,
    /// Stacked FFT sizes (empty = disabled).
    pub stacked_fft_sizes: Vec<usize>,
}

impl DetectionParams {
    /// Create detection parameters from a preset and sample rate.
    pub fn from_preset(preset: DetectionPreset, _sample_rate: f64) -> Self {
        let base = Self {
            preset,
            fft_size: 4096,
            hop_size: 1024,
            min_frequency: 60.0,
            max_frequency: 5000.0,
            yin_threshold: 0.15,
            onset_threshold: 0.5,
            min_note_duration: 0.05,
            min_note_amplitude: 0.001,
            polyphonic: false,
            max_polyphony: 1,
            detect_sibilants: true,
            adjust_attacks: false,
            stacked_fft_sizes: vec![],
        };

        match preset {
            DetectionPreset::Automatic => base,
            DetectionPreset::Melodic => Self {
                yin_threshold: 0.12,
                ..base
            },
            DetectionPreset::MelodicAdjustAttacks => Self {
                yin_threshold: 0.12,
                adjust_attacks: true,
                onset_threshold: 0.35,
                ..base
            },
            DetectionPreset::MelodicNoSibilants => Self {
                yin_threshold: 0.12,
                detect_sibilants: false,
                min_frequency: 80.0,
                ..base
            },
            DetectionPreset::MelodicPercussive => Self {
                yin_threshold: 0.18,
                adjust_attacks: true,
                onset_threshold: 0.3,
                min_note_duration: 0.02,
                ..base
            },
            DetectionPreset::Percussive => Self {
                yin_threshold: 0.3,
                adjust_attacks: true,
                onset_threshold: 0.2,
                min_note_duration: 0.01,
                min_note_amplitude: 0.01,
                min_frequency: 40.0,
                ..base
            },
            DetectionPreset::Polyphonic => Self {
                fft_size: 8192,
                hop_size: 2048,
                polyphonic: true,
                max_polyphony: 6,
                stacked_fft_sizes: vec![2048, 4096, 8192, 16384],
                ..base
            },
            DetectionPreset::PolyphonicUnpitched => Self {
                fft_size: 8192,
                hop_size: 2048,
                polyphonic: true,
                max_polyphony: 8,
                min_frequency: 40.0,
                min_note_amplitude: 0.005,
                stacked_fft_sizes: vec![2048, 4096, 8192],
                ..base
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_names() {
        assert_eq!(DetectionPreset::Automatic.name(), "Automatic");
        assert_eq!(DetectionPreset::Polyphonic.name(), "Polyphonic");
    }

    #[test]
    fn test_preset_from_name() {
        assert_eq!(
            DetectionPreset::from_name("auto"),
            Some(DetectionPreset::Automatic)
        );
        assert_eq!(
            DetectionPreset::from_name("polyphonic"),
            Some(DetectionPreset::Polyphonic)
        );
        assert_eq!(DetectionPreset::from_name("invalid"), None);
    }

    #[test]
    fn test_preset_properties() {
        assert!(DetectionPreset::Polyphonic.is_polyphonic());
        assert!(!DetectionPreset::Melodic.is_polyphonic());
        assert!(DetectionPreset::Melodic.expects_pitched());
        assert!(!DetectionPreset::Percussive.expects_pitched());
        assert!(!DetectionPreset::MelodicNoSibilants.detect_sibilants());
        assert!(DetectionPreset::Melodic.detect_sibilants());
        assert!(DetectionPreset::MelodicAdjustAttacks.adjust_attacks());
    }

    #[test]
    fn test_detection_params_from_preset() {
        let params = DetectionParams::from_preset(DetectionPreset::Polyphonic, 44100.0);
        assert!(params.polyphonic);
        assert_eq!(params.fft_size, 8192);
        assert_eq!(params.max_polyphony, 6);
        assert!(!params.stacked_fft_sizes.is_empty());
    }

    #[test]
    fn test_all_presets() {
        assert_eq!(DetectionPreset::ALL.len(), 8);
        for preset in &DetectionPreset::ALL {
            let _params = DetectionParams::from_preset(*preset, 44100.0);
            assert!(!preset.name().is_empty());
        }
    }
}
