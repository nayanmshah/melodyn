//! Scale and key analysis.
//!
//! Detects scales from pitch class distributions, identifies modes, supports
//! microtonal systems (Turkish Makam 53-TET, Gamelan Pelog/Slendro), and
//! performs key detection using the Krumhansl-Schmuckler algorithm. Models
//! MUCandidateScale, MUScaleMode, MUScalePitchSystem, MUScaleTuning, and
//! related classes.

use crate::note::DetectedNote;
use crate::pitch::PitchCandidate;

// ---------------------------------------------------------------------------
// Scale mode
// ---------------------------------------------------------------------------

/// A scale type / mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleMode {
    // Diatonic modes
    Ionian,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Aeolian,
    Locrian,
    // Additional Western scales
    HarmonicMinor,
    MelodicMinor,
    PentatonicMajor,
    PentatonicMinor,
    Blues,
    WholeTone,
    Chromatic,
    // Microtonal
    TurkishMakam,
    GamelanPelog,
    GamelanSlendro,
}

impl ScaleMode {
    /// Interval pattern as semitone offsets from the tonic for 12-TET scales.
    /// Returns None for microtonal scales (they use a different system).
    pub fn intervals(self) -> Option<&'static [u8]> {
        match self {
            Self::Ionian => Some(&[0, 2, 4, 5, 7, 9, 11]),
            Self::Dorian => Some(&[0, 2, 3, 5, 7, 9, 10]),
            Self::Phrygian => Some(&[0, 1, 3, 5, 7, 8, 10]),
            Self::Lydian => Some(&[0, 2, 4, 6, 7, 9, 11]),
            Self::Mixolydian => Some(&[0, 2, 4, 5, 7, 9, 10]),
            Self::Aeolian => Some(&[0, 2, 3, 5, 7, 8, 10]),
            Self::Locrian => Some(&[0, 1, 3, 5, 6, 8, 10]),
            Self::HarmonicMinor => Some(&[0, 2, 3, 5, 7, 8, 11]),
            Self::MelodicMinor => Some(&[0, 2, 3, 5, 7, 9, 11]),
            Self::PentatonicMajor => Some(&[0, 2, 4, 7, 9]),
            Self::PentatonicMinor => Some(&[0, 3, 5, 7, 10]),
            Self::Blues => Some(&[0, 3, 5, 6, 7, 10]),
            Self::WholeTone => Some(&[0, 2, 4, 6, 8, 10]),
            Self::Chromatic => Some(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            Self::TurkishMakam | Self::GamelanPelog | Self::GamelanSlendro => None,
        }
    }

    /// All 12-TET scale modes.
    pub const ALL_12TET: [ScaleMode; 14] = [
        Self::Ionian,
        Self::Dorian,
        Self::Phrygian,
        Self::Lydian,
        Self::Mixolydian,
        Self::Aeolian,
        Self::Locrian,
        Self::HarmonicMinor,
        Self::MelodicMinor,
        Self::PentatonicMajor,
        Self::PentatonicMinor,
        Self::Blues,
        Self::WholeTone,
        Self::Chromatic,
    ];

    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ionian => "Major (Ionian)",
            Self::Dorian => "Dorian",
            Self::Phrygian => "Phrygian",
            Self::Lydian => "Lydian",
            Self::Mixolydian => "Mixolydian",
            Self::Aeolian => "Natural Minor (Aeolian)",
            Self::Locrian => "Locrian",
            Self::HarmonicMinor => "Harmonic Minor",
            Self::MelodicMinor => "Melodic Minor",
            Self::PentatonicMajor => "Major Pentatonic",
            Self::PentatonicMinor => "Minor Pentatonic",
            Self::Blues => "Blues",
            Self::WholeTone => "Whole Tone",
            Self::Chromatic => "Chromatic",
            Self::TurkishMakam => "Turkish Makam",
            Self::GamelanPelog => "Gamelan Pelog",
            Self::GamelanSlendro => "Gamelan Slendro",
        }
    }
}

// ---------------------------------------------------------------------------
// Scale degree
// ---------------------------------------------------------------------------

/// A scale degree with optional alteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleModeDegree {
    /// Degree number (1-based: 1 = tonic, 2 = supertonic, etc.).
    pub degree: u8,
    /// Alteration in semitones (0 = natural, -1 = flat, +1 = sharp).
    pub alteration: i8,
}

// ---------------------------------------------------------------------------
// Diatonic degree mask
// ---------------------------------------------------------------------------

/// A bitmask identifying which of the 12 pitch classes belong to a diatonic scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiatonicDegreeMask(pub u16);

impl DiatonicDegreeMask {
    /// Create a mask from a scale mode and tonic.
    pub fn from_scale(tonic: u8, mode: ScaleMode) -> Self {
        let mut mask = 0u16;
        if let Some(intervals) = mode.intervals() {
            for &iv in intervals {
                let pc = (tonic + iv) % 12;
                mask |= 1 << pc;
            }
        }
        DiatonicDegreeMask(mask)
    }

    /// Check if a pitch class belongs to this scale.
    pub fn contains(&self, pc: u8) -> bool {
        (self.0 >> (pc % 12)) & 1 == 1
    }

    /// Count the number of pitch classes in the scale.
    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }
}

// ---------------------------------------------------------------------------
// Candidate scale
// ---------------------------------------------------------------------------

/// A candidate scale detected from the audio.
#[derive(Debug, Clone)]
pub struct CandidateScale {
    /// Tonic pitch class (0 = C, ..., 11 = B).
    pub tonic: u8,
    /// Scale mode.
    pub mode: ScaleMode,
    /// Correlation score from key-finding algorithm (higher = better fit).
    pub correlation: f64,
    /// Confidence (0.0 .. 1.0).
    pub confidence: f64,
}

impl CandidateScale {
    /// Map a MIDI note to its scale degree, or None if not in scale.
    pub fn note_to_degree(&self, midi_note: f64) -> Option<ScaleModeDegree> {
        let pc = ((midi_note.round() as i32).rem_euclid(12)) as u8;
        let intervals = self.mode.intervals()?;
        let relative = (pc + 12 - self.tonic) % 12;
        for (i, &iv) in intervals.iter().enumerate() {
            if iv == relative {
                return Some(ScaleModeDegree {
                    degree: (i + 1) as u8,
                    alteration: 0,
                });
            }
        }
        // Not an exact scale tone -- find nearest and report alteration.
        let mut best_degree = 0u8;
        let mut best_alt = 0i8;
        let mut best_dist = 12i8;
        for (i, &iv) in intervals.iter().enumerate() {
            let diff = relative as i8 - iv as i8;
            if diff.abs() < best_dist {
                best_dist = diff.abs();
                best_degree = (i + 1) as u8;
                best_alt = diff;
            }
        }
        Some(ScaleModeDegree {
            degree: best_degree,
            alteration: best_alt,
        })
    }

    /// Format the scale name, e.g. "C Major (Ionian)".
    pub fn name(&self) -> String {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        format!("{} {}", note_names[self.tonic as usize], self.mode.name())
    }
}

// ---------------------------------------------------------------------------
// Krumhansl-Schmuckler key-finding algorithm
// ---------------------------------------------------------------------------

/// Key profiles from Krumhansl & Kessler (1982).
/// Weights for each pitch class relative to the tonic.
const MAJOR_PROFILE: [f64; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

const MINOR_PROFILE: [f64; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Compute the Pearson correlation between two 12-element arrays.
fn pearson_correlation(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let mean_a: f64 = a.iter().sum::<f64>() / 12.0;
    let mean_b: f64 = b.iter().sum::<f64>() / 12.0;

    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for i in 0..12 {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den < 1e-12 {
        0.0
    } else {
        num / den
    }
}

/// Rotate an array by `shift` positions (for transposing key profiles).
fn rotate_profile(profile: &[f64; 12], shift: usize) -> [f64; 12] {
    let mut rotated = [0.0; 12];
    for i in 0..12 {
        rotated[i] = profile[(i + 12 - shift) % 12];
    }
    rotated
}

/// Detect the most likely key/scale from a pitch class distribution using
/// the Krumhansl-Schmuckler algorithm.
///
/// Returns candidates sorted by correlation (best first).
pub fn detect_key(distribution: &[f64; 12]) -> Vec<CandidateScale> {
    let mut candidates = Vec::with_capacity(24);

    for tonic in 0u8..12 {
        // Major
        let major_profile = rotate_profile(&MAJOR_PROFILE, tonic as usize);
        let major_corr = pearson_correlation(distribution, &major_profile);
        candidates.push(CandidateScale {
            tonic,
            mode: ScaleMode::Ionian,
            correlation: major_corr,
            confidence: 0.0, // set after normalization
        });

        // Minor
        let minor_profile = rotate_profile(&MINOR_PROFILE, tonic as usize);
        let minor_corr = pearson_correlation(distribution, &minor_profile);
        candidates.push(CandidateScale {
            tonic,
            mode: ScaleMode::Aeolian,
            correlation: minor_corr,
            confidence: 0.0,
        });
    }

    candidates.sort_by(|a, b| {
        b.correlation
            .partial_cmp(&a.correlation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Normalize confidence: best = 1.0, relative to range.
    if let (Some(best), Some(worst)) = (candidates.first(), candidates.last()) {
        let range = best.correlation - worst.correlation;
        if range > 1e-12 {
            for c in &mut candidates {
                c.confidence = ((c.correlation - worst.correlation) / range).clamp(0.0, 1.0);
            }
        }
    }

    candidates
}

// ---------------------------------------------------------------------------
// Extended scale detection (all modes)
// ---------------------------------------------------------------------------

/// Detect scale/mode from a pitch class distribution by template matching
/// against all 12-TET scale modes.
pub fn detect_scale(distribution: &[f64; 12]) -> Vec<CandidateScale> {
    let mut candidates = Vec::new();

    for &mode in &ScaleMode::ALL_12TET {
        if let Some(intervals) = mode.intervals() {
            for tonic in 0u8..12 {
                // Build a binary profile for this scale
                let mut profile = [0.0f64; 12];
                for &iv in intervals {
                    profile[(tonic as usize + iv as usize) % 12] = 1.0;
                }

                // Score = sum of distribution weight on scale tones minus off-scale weight
                let mut on_scale = 0.0;
                let mut off_scale = 0.0;
                for i in 0..12 {
                    if profile[i] > 0.5 {
                        on_scale += distribution[i];
                    } else {
                        off_scale += distribution[i];
                    }
                }

                let score = on_scale - off_scale * 0.5;

                candidates.push(CandidateScale {
                    tonic,
                    mode,
                    correlation: score,
                    confidence: on_scale.clamp(0.0, 1.0),
                });
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.correlation
            .partial_cmp(&a.correlation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Normalize confidence
    if let (Some(best), Some(worst)) = (candidates.first(), candidates.last()) {
        let range = best.correlation - worst.correlation;
        if range > 1e-12 {
            for c in &mut candidates {
                c.confidence = ((c.correlation - worst.correlation) / range).clamp(0.0, 1.0);
            }
        }
    }

    candidates
}

/// Build a pitch class distribution from detected notes.
pub fn distribution_from_notes(notes: &[DetectedNote]) -> [f64; 12] {
    let mut dist = [0.0f64; 12];
    for note in notes {
        let pc = ((note.midi_note.round() as i32).rem_euclid(12)) as usize;
        dist[pc] += note.amplitude * note.duration;
    }
    let total: f64 = dist.iter().sum();
    if total > 1e-12 {
        for d in &mut dist {
            *d /= total;
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// Turkish Makam (53-TET comma system)
// ---------------------------------------------------------------------------

/// A note in the Turkish 53-TET comma system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurkishNote {
    /// Comma position (0..52) within the octave.
    pub comma: u8,
    /// Traditional Turkish note name.
    pub name: &'static str,
}

/// A pentachord (5-note span) used in Turkish Makam theory.
#[derive(Debug, Clone)]
pub struct TurkishPentachord {
    /// Name of the pentachord type.
    pub name: &'static str,
    /// Interval pattern in commas.
    pub intervals: Vec<u8>,
}

/// Turkish Makam definition: a named scale built from pentachords and tetrachords.
#[derive(Debug, Clone)]
pub struct TurkishMakam {
    /// Makam name (e.g., "Rast", "Hicaz", "Huseyni").
    pub name: String,
    /// Comma positions of scale degrees within one octave (0..53).
    pub degrees: Vec<u8>,
    /// Tonic comma position.
    pub tonic_comma: u8,
}

impl TurkishMakam {
    /// Create the Rast makam (most fundamental Turkish makam).
    /// Rast: C-9-D-8-E(lowered)-5-F-9-G-9-A-8-B(lowered)-5-C
    pub fn rast() -> Self {
        Self {
            name: "Rast".to_string(),
            degrees: vec![0, 9, 17, 22, 31, 40, 48, 53],
            tonic_comma: 0,
        }
    }

    /// Create the Hicaz makam.
    /// Hicaz: C-5-Db-12-E-5-F-9-G-9-A-8-Bb-5-C
    pub fn hicaz() -> Self {
        Self {
            name: "Hicaz".to_string(),
            degrees: vec![0, 5, 17, 22, 31, 40, 48, 53],
            tonic_comma: 0,
        }
    }

    /// Create the Huseyni makam.
    pub fn huseyni() -> Self {
        Self {
            name: "Huseyni".to_string(),
            degrees: vec![0, 8, 13, 22, 31, 40, 44, 53],
            tonic_comma: 0,
        }
    }

    /// Convert a comma position to cents offset from the tonic.
    pub fn comma_to_cents(comma: u8) -> f64 {
        comma as f64 * (1200.0 / 53.0)
    }

    /// Convert a frequency to the nearest makam degree.
    /// `reference_freq` is the frequency of the tonic.
    pub fn freq_to_degree(&self, freq: f64, reference_freq: f64) -> Option<(usize, f64)> {
        if freq <= 0.0 || reference_freq <= 0.0 {
            return None;
        }
        let cents = 1200.0 * (freq / reference_freq).log2();
        let comma_pos = (cents / (1200.0 / 53.0)).round() as i32;
        let comma_in_octave = comma_pos.rem_euclid(53) as u8;

        let mut best_idx = 0;
        let mut best_dist = 53i32;
        for (i, &deg) in self.degrees.iter().enumerate() {
            let dist = ((comma_in_octave as i32 - deg as i32).abs()).min(
                (53 - (comma_in_octave as i32 - deg as i32).abs()),
            );
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        let deviation_cents = (comma_in_octave as f64 - self.degrees[best_idx] as f64)
            * (1200.0 / 53.0);
        Some((best_idx, deviation_cents))
    }
}

// ---------------------------------------------------------------------------
// Gamelan scales
// ---------------------------------------------------------------------------

/// Pelog scale (approximate cents from tonic for a common tuning).
pub const GAMELAN_PELOG_CENTS: [f64; 7] = [0.0, 120.0, 260.0, 540.0, 675.0, 780.0, 940.0];

/// Slendro scale (approximately equal 5-tone division of the octave).
pub const GAMELAN_SLENDRO_CENTS: [f64; 5] = [0.0, 240.0, 480.0, 720.0, 960.0];

/// Match a frequency to the nearest Pelog or Slendro scale degree.
/// Returns (degree_index, deviation_in_cents).
pub fn match_gamelan_degree(
    freq: f64,
    reference_freq: f64,
    scale_cents: &[f64],
) -> Option<(usize, f64)> {
    if freq <= 0.0 || reference_freq <= 0.0 || scale_cents.is_empty() {
        return None;
    }
    let cents = 1200.0 * (freq / reference_freq).log2();
    let cents_in_octave = cents.rem_euclid(1200.0);

    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    for (i, &sc) in scale_cents.iter().enumerate() {
        let dist = (cents_in_octave - sc).abs().min((cents_in_octave - sc + 1200.0).abs());
        let dist = dist.min((cents_in_octave - sc - 1200.0).abs());
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    let deviation = cents_in_octave - scale_cents[best_idx];
    Some((best_idx, deviation))
}

// ---------------------------------------------------------------------------
// Scale stretch tuning
// ---------------------------------------------------------------------------

/// Piano-style octave stretching parameters.
/// Real pianos have slightly sharp high notes and flat low notes due to
/// string inharmonicity. This models that for more realistic tuning.
#[derive(Debug, Clone)]
pub struct ScaleStretching {
    /// Stretch coefficient in cents per octave squared.
    /// Typical piano value: ~1.5 cents/octave^2.
    pub coefficient: f64,
    /// Reference MIDI note for zero stretch (typically A4 = 69).
    pub reference_note: f64,
}

impl Default for ScaleStretching {
    fn default() -> Self {
        Self {
            coefficient: 1.5,
            reference_note: 69.0,
        }
    }
}

impl ScaleStretching {
    /// Compute the stretch offset in cents for a given MIDI note.
    pub fn stretch_cents(&self, midi_note: f64) -> f64 {
        let octaves_from_ref = (midi_note - self.reference_note) / 12.0;
        self.coefficient * octaves_from_ref * octaves_from_ref.abs()
    }

    /// Compute the stretched frequency for a given MIDI note.
    pub fn stretched_freq(&self, midi_note: f64) -> f64 {
        let base_freq = PitchCandidate::midi_to_freq(midi_note);
        let stretch = self.stretch_cents(midi_note);
        base_freq * 2.0f64.powf(stretch / 1200.0)
    }
}

// ---------------------------------------------------------------------------
// Scale tuning (custom tuning table)
// ---------------------------------------------------------------------------

/// Custom tuning system: maps each pitch class to a cent offset from equal temperament.
#[derive(Debug, Clone)]
pub struct ScaleTuning {
    /// Name of the tuning system.
    pub name: String,
    /// Cent offsets for each pitch class (0 = C, ..., 11 = B).
    /// A value of 0.0 means equal temperament.
    pub offsets: [f64; 12],
}

impl ScaleTuning {
    /// Equal temperament (all offsets = 0).
    pub fn equal_temperament() -> Self {
        Self {
            name: "Equal Temperament".to_string(),
            offsets: [0.0; 12],
        }
    }

    /// Just intonation tuning (relative to C).
    pub fn just_intonation() -> Self {
        Self {
            name: "Just Intonation".to_string(),
            offsets: [
                0.0,    // C  (1/1)
                11.73,  // C# (16/15)
                3.91,   // D  (9/8)
                15.64,  // Eb (6/5)
                -13.69, // E  (5/4)
                -1.96,  // F  (4/3)
                -9.78,  // F# (45/32)
                1.96,   // G  (3/2)
                13.69,  // Ab (8/5)
                -15.64, // A  (5/3)
                -17.60, // Bb (9/5)
                -11.73, // B  (15/8)
            ],
        }
    }

    /// Pythagorean tuning.
    pub fn pythagorean() -> Self {
        Self {
            name: "Pythagorean".to_string(),
            offsets: [
                0.0,    // C
                -9.78,  // Db
                3.91,   // D
                -5.87,  // Eb
                7.82,   // E
                -1.96,  // F
                11.73,  // F#
                1.96,   // G
                -7.82,  // Ab
                5.87,   // A
                -3.91,  // Bb
                9.78,   // B
            ],
        }
    }

    /// Apply tuning to a MIDI note, returning adjusted frequency.
    pub fn apply(&self, midi_note: f64) -> f64 {
        let pc = ((midi_note.round() as i32).rem_euclid(12)) as usize;
        let offset_cents = self.offsets[pc];
        let base_freq = PitchCandidate::midi_to_freq(midi_note);
        base_freq * 2.0f64.powf(offset_cents / 1200.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_note(midi: f64, start: f64, dur: f64, amp: f64) -> DetectedNote {
        DetectedNote {
            frequency: PitchCandidate::midi_to_freq(midi),
            midi_note: midi,
            start_time: start,
            duration: dur,
            start_sample: 0,
            end_sample: 0,
            amplitude: amp,
            peak_amplitude: amp,
            confidence: 0.9,
            pitch_contour: vec![],
            amplitude_contour: vec![],
            vibrato_rate: 0.0,
            vibrato_depth: 0.0,
            attack_time: 0.0,
            release_time: 0.0,
        }
    }

    #[test]
    fn test_detect_key_c_major() {
        // Strong C major distribution: heavy on C, E, G
        let mut dist = [0.0f64; 12];
        dist[0] = 0.25; // C
        dist[2] = 0.10; // D
        dist[4] = 0.20; // E
        dist[5] = 0.08; // F
        dist[7] = 0.18; // G
        dist[9] = 0.10; // A
        dist[11] = 0.09; // B

        let candidates = detect_key(&dist);
        assert!(!candidates.is_empty());
        let best = &candidates[0];
        assert_eq!(best.tonic, 0, "Expected C as tonic, got {}", best.tonic);
        assert_eq!(best.mode, ScaleMode::Ionian);
    }

    #[test]
    fn test_detect_key_a_minor() {
        // A minor distribution: heavy on A, C, E
        let mut dist = [0.0f64; 12];
        dist[9] = 0.25; // A
        dist[0] = 0.18; // C
        dist[2] = 0.08; // D
        dist[4] = 0.20; // E
        dist[5] = 0.07; // F
        dist[7] = 0.12; // G
        dist[11] = 0.10; // B

        let candidates = detect_key(&dist);
        // A minor should be among top candidates
        let am = candidates
            .iter()
            .take(5)
            .find(|c| c.tonic == 9 && c.mode == ScaleMode::Aeolian);
        assert!(am.is_some(), "A minor should be in top 5 candidates");
    }

    #[test]
    fn test_detect_scale_modes() {
        // D Dorian distribution
        let mut dist = [0.0f64; 12];
        dist[2] = 0.20; // D
        dist[4] = 0.10; // E (Dorian has natural 6th and flat 7th)
        dist[5] = 0.15; // F
        dist[7] = 0.15; // G
        dist[9] = 0.15; // A
        dist[11] = 0.15; // B (natural 6th)
        dist[0] = 0.10; // C (flat 7th)

        let candidates = detect_scale(&dist);
        assert!(!candidates.is_empty());
        // D Dorian should score well
        let d_dorian = candidates
            .iter()
            .take(10)
            .find(|c| c.tonic == 2 && c.mode == ScaleMode::Dorian);
        assert!(d_dorian.is_some(), "D Dorian should be in top 10");
    }

    #[test]
    fn test_diatonic_degree_mask() {
        let mask = DiatonicDegreeMask::from_scale(0, ScaleMode::Ionian);
        assert!(mask.contains(0));  // C
        assert!(mask.contains(2));  // D
        assert!(mask.contains(4));  // E
        assert!(mask.contains(5));  // F
        assert!(mask.contains(7));  // G
        assert!(mask.contains(9));  // A
        assert!(mask.contains(11)); // B
        assert!(!mask.contains(1)); // C# not in C major
        assert_eq!(mask.count(), 7);
    }

    #[test]
    fn test_note_to_degree() {
        let scale = CandidateScale {
            tonic: 0,
            mode: ScaleMode::Ionian,
            correlation: 1.0,
            confidence: 1.0,
        };

        let d = scale.note_to_degree(60.0).unwrap(); // C4
        assert_eq!(d.degree, 1);
        assert_eq!(d.alteration, 0);

        let d = scale.note_to_degree(64.0).unwrap(); // E4
        assert_eq!(d.degree, 3);
        assert_eq!(d.alteration, 0);

        let d = scale.note_to_degree(67.0).unwrap(); // G4
        assert_eq!(d.degree, 5);
        assert_eq!(d.alteration, 0);
    }

    #[test]
    fn test_distribution_from_notes() {
        let notes = vec![
            make_note(60.0, 0.0, 1.0, 0.5), // C
            make_note(64.0, 0.0, 1.0, 0.5), // E
            make_note(67.0, 0.0, 1.0, 0.5), // G
        ];
        let dist = distribution_from_notes(&notes);
        assert!(dist[0] > 0.0); // C
        assert!(dist[4] > 0.0); // E
        assert!(dist[7] > 0.0); // G
        let total: f64 = dist.iter().sum();
        assert!((total - 1.0).abs() < 1e-10, "Should be normalized");
    }

    #[test]
    fn test_turkish_makam_rast() {
        let rast = TurkishMakam::rast();
        assert_eq!(rast.degrees.len(), 8); // 7 notes + octave

        // Test comma-to-cents conversion
        let cents_9 = TurkishMakam::comma_to_cents(9);
        assert!((cents_9 - 203.77).abs() < 1.0); // ~204 cents (whole tone)
    }

    #[test]
    fn test_turkish_makam_degree_matching() {
        let rast = TurkishMakam::rast();
        // Exact tonic should match degree 0
        let (deg, dev) = rast.freq_to_degree(440.0, 440.0).unwrap();
        assert_eq!(deg, 0);
        assert!(dev.abs() < 1.0);
    }

    #[test]
    fn test_gamelan_slendro() {
        // Exact scale tone should match with minimal deviation
        let reference = 440.0;
        let scale_tone_freq = reference * 2.0f64.powf(240.0 / 1200.0); // 2nd degree
        let (deg, dev) = match_gamelan_degree(scale_tone_freq, reference, &GAMELAN_SLENDRO_CENTS).unwrap();
        assert_eq!(deg, 1);
        assert!(dev.abs() < 5.0, "Deviation should be small: {}", dev);
    }

    #[test]
    fn test_scale_stretching() {
        let stretch = ScaleStretching::default();

        // At reference note, stretch should be zero
        let cents_at_ref = stretch.stretch_cents(69.0);
        assert!(cents_at_ref.abs() < 1e-10);

        // High notes should be sharp (positive cents)
        let cents_high = stretch.stretch_cents(108.0); // C8
        assert!(cents_high > 0.0, "High notes should be sharp");

        // Low notes should be flat (negative cents)
        let cents_low = stretch.stretch_cents(21.0); // A0
        assert!(cents_low < 0.0, "Low notes should be flat");
    }

    #[test]
    fn test_scale_tuning_equal() {
        let tuning = ScaleTuning::equal_temperament();
        let freq = tuning.apply(69.0); // A4
        assert!((freq - 440.0).abs() < 0.01);
    }

    #[test]
    fn test_scale_tuning_just_intonation() {
        let tuning = ScaleTuning::just_intonation();
        // G (perfect fifth from C) should be slightly different from ET
        let g_freq = tuning.apply(67.0);
        let g_et = PitchCandidate::midi_to_freq(67.0);
        // Just intonation G has +1.96 cents offset
        assert!((g_freq - g_et).abs() > 0.01, "JI should differ from ET");
    }

    #[test]
    fn test_candidate_scale_name() {
        let scale = CandidateScale {
            tonic: 0,
            mode: ScaleMode::Ionian,
            correlation: 0.9,
            confidence: 0.95,
        };
        assert_eq!(scale.name(), "C Major (Ionian)");
    }
}
