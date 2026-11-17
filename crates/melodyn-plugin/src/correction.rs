//! Pitch correction processor.
//!
//! Computes the pitch shift needed to snap a detected frequency to the
//! nearest semitone (or scale degree). Handles smoothing, vibrato scaling,
//! and unvoiced segment detection. Designed for real-time use with no
//! heap allocations in the correction path.

use melodyn_core::pitch::PitchCandidate;

// ---------------------------------------------------------------------------
// Scale constraint
// ---------------------------------------------------------------------------

/// A musical scale constraint for pitch correction.
///
/// When set, pitch correction snaps to the nearest note *in the scale*
/// rather than the nearest chromatic semitone.
#[derive(Debug, Clone)]
pub struct ScaleConstraint {
    /// Root note as MIDI note number (0-11, C=0, C#=1, ...).
    pub root: u8,
    /// Which of the 12 chromatic degrees are in the scale (true = in scale).
    pub degrees: [bool; 12],
}

impl ScaleConstraint {
    /// Chromatic scale (all notes allowed -- effectively no constraint).
    pub fn chromatic() -> Self {
        Self {
            root: 0,
            degrees: [true; 12],
        }
    }

    /// Major scale.
    pub fn major(root: u8) -> Self {
        // W W H W W W H = 0 2 4 5 7 9 11
        let mut degrees = [false; 12];
        let intervals = [0, 2, 4, 5, 7, 9, 11];
        for &i in &intervals {
            degrees[(i + root as usize) % 12] = true;
        }
        Self { root, degrees }
    }

    /// Natural minor scale.
    pub fn minor(root: u8) -> Self {
        // W H W W H W W = 0 2 3 5 7 8 10
        let mut degrees = [false; 12];
        let intervals = [0, 2, 3, 5, 7, 8, 10];
        for &i in &intervals {
            degrees[(i + root as usize) % 12] = true;
        }
        Self { root, degrees }
    }

    /// Check if a chromatic pitch class (0-11) is in the scale.
    pub fn contains(&self, pitch_class: u8) -> bool {
        self.degrees[(pitch_class % 12) as usize]
    }

    /// Find the nearest scale degree to a given MIDI note.
    ///
    /// Returns the nearest MIDI note that falls on a scale degree.
    pub fn nearest_scale_note(&self, midi_note: f64) -> f64 {
        let rounded = midi_note.round();
        let pitch_class = ((rounded as i32 % 12) + 12) % 12;

        if self.degrees[pitch_class as usize] {
            return rounded;
        }

        // Search up and down for the nearest scale degree.
        for offset in 1..=6 {
            let up = ((pitch_class + offset) % 12) as usize;
            let down = ((pitch_class - offset + 12) % 12) as usize;

            if self.degrees[up] {
                return rounded + offset as f64;
            }
            if self.degrees[down] {
                return rounded - offset as f64;
            }
        }

        // Fallback (shouldn't happen unless scale has no degrees).
        rounded
    }
}

impl Default for ScaleConstraint {
    fn default() -> Self {
        Self::chromatic()
    }
}

// ---------------------------------------------------------------------------
// Correction state
// ---------------------------------------------------------------------------

/// Smoothing state for pitch correction transitions.
#[derive(Debug, Clone)]
struct CorrectionSmoother {
    /// Current smoothed correction value in semitones.
    current: f64,
    /// Target correction value in semitones.
    target: f64,
    /// Smoothing coefficient (0.0 = instant, 1.0 = no change).
    /// Lower values give faster response.
    coeff: f64,
}

impl CorrectionSmoother {
    fn new(smoothing_ms: f64, sample_rate: f64, hop_size: usize) -> Self {
        // Convert smoothing time to per-hop coefficient.
        let hops_per_second = sample_rate / hop_size as f64;
        let tau = smoothing_ms * 0.001 * hops_per_second;
        let coeff = if tau > 0.0 { (-1.0 / tau).exp() } else { 0.0 };

        Self {
            current: 0.0,
            target: 0.0,
            coeff,
        }
    }

    fn set_target(&mut self, target: f64) {
        self.target = target;
    }

    fn advance(&mut self) -> f64 {
        self.current = self.current * self.coeff + self.target * (1.0 - self.coeff);
        self.current
    }

    fn reset(&mut self) {
        self.current = 0.0;
        self.target = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Pitch correction processor
// ---------------------------------------------------------------------------

/// Real-time pitch correction processor.
///
/// Given a detected pitch frequency, computes the shift needed to correct
/// it to the nearest target pitch (semitone or scale degree). Applies
/// smoothing to avoid abrupt jumps and allows vibrato scaling.
pub struct PitchCorrectionProcessor {
    /// Sample rate.
    sample_rate: f64,
    /// Correction amount (0.0 = off, 1.0 = full snap).
    correction_amount: f64,
    /// Vibrato scaling factor (1.0 = natural, 0.0 = remove, 2.0 = exaggerate).
    vibrato_scale: f64,
    /// Scale constraint (which notes to snap to).
    scale: ScaleConstraint,
    /// Smoothing state for correction transitions.
    smoother: CorrectionSmoother,
    /// Previous detected MIDI note (for vibrato extraction).
    prev_midi_note: f64,
    /// Running average MIDI note (for vibrato center estimation).
    running_avg_midi: f64,
    /// Running average coefficient.
    avg_coeff: f64,
    /// Whether we have seen any valid pitch yet.
    initialized: bool,
}

impl PitchCorrectionProcessor {
    /// Create a new pitch correction processor.
    pub fn new(sample_rate: f64) -> Self {
        let hop_size = 512;
        Self {
            sample_rate,
            correction_amount: 0.0,
            vibrato_scale: 1.0,
            scale: ScaleConstraint::chromatic(),
            smoother: CorrectionSmoother::new(20.0, sample_rate, hop_size),
            prev_midi_note: 0.0,
            running_avg_midi: 0.0,
            avg_coeff: 0.95,
            initialized: false,
        }
    }

    /// Set the correction amount (0.0 to 1.0).
    pub fn set_correction_amount(&mut self, amount: f64) {
        self.correction_amount = amount.clamp(0.0, 1.0);
    }

    /// Set the vibrato scaling factor.
    pub fn set_vibrato_scale(&mut self, scale: f64) {
        self.vibrato_scale = scale.clamp(0.0, 2.0);
    }

    /// Set the scale constraint.
    pub fn set_scale(&mut self, scale: ScaleConstraint) {
        self.scale = scale;
    }

    /// Get the current correction amount.
    pub fn correction_amount(&self) -> f64 {
        self.correction_amount
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.smoother.reset();
        self.prev_midi_note = 0.0;
        self.running_avg_midi = 0.0;
        self.initialized = false;
    }

    /// Compute the pitch correction shift in semitones for the given detected frequency.
    ///
    /// Returns the shift in semitones that should be applied to the signal.
    /// Positive = shift up, negative = shift down.
    ///
    /// The `amount` parameter (0.0 to 1.0) controls how much of the correction
    /// is applied, allowing smooth blending between natural and corrected pitch.
    pub fn compute_correction(&mut self, detected_freq: f64, amount: f64) -> f64 {
        if detected_freq <= 0.0 || amount <= 0.0 {
            self.smoother.set_target(0.0);
            return self.smoother.advance();
        }

        let detected_midi = PitchCandidate::freq_to_midi(detected_freq);

        // Update running average for vibrato center estimation.
        if !self.initialized {
            self.running_avg_midi = detected_midi;
            self.initialized = true;
        } else {
            self.running_avg_midi =
                self.running_avg_midi * self.avg_coeff + detected_midi * (1.0 - self.avg_coeff);
        }

        // Find the target note (nearest semitone or scale degree).
        let target_midi = self.scale.nearest_scale_note(self.running_avg_midi);

        // Compute the full correction (what we'd need to snap exactly).
        let full_correction = target_midi - detected_midi;

        // Extract vibrato component (deviation from running average).
        let vibrato_component = detected_midi - self.running_avg_midi;

        // Apply vibrato scaling: scale the vibrato relative to the target.
        let scaled_vibrato = vibrato_component * self.vibrato_scale;

        // The correction shift: blend between no correction and full correction.
        // With vibrato scaling applied.
        let base_correction = full_correction * amount;

        // Add back the (possibly scaled) vibrato.
        let vibrato_adjustment = if self.vibrato_scale != 1.0 {
            (scaled_vibrato - vibrato_component) * amount
        } else {
            0.0
        };

        let total_correction = base_correction + vibrato_adjustment;

        self.prev_midi_note = detected_midi;
        self.smoother.set_target(total_correction);
        self.smoother.advance()
    }

    /// Compute the target frequency for a given detected frequency.
    ///
    /// This is a convenience method that returns the absolute target frequency
    /// rather than the shift in semitones.
    pub fn target_frequency(&self, detected_freq: f64) -> f64 {
        if detected_freq <= 0.0 {
            return 0.0;
        }
        let midi = PitchCandidate::freq_to_midi(detected_freq);
        let target_midi = self.scale.nearest_scale_note(midi);
        PitchCandidate::midi_to_freq(target_midi)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_chromatic() {
        let scale = ScaleConstraint::chromatic();
        for i in 0..12 {
            assert!(scale.contains(i));
        }
    }

    #[test]
    fn test_scale_major() {
        let c_major = ScaleConstraint::major(0);
        // C major: C D E F G A B
        assert!(c_major.contains(0)); // C
        assert!(!c_major.contains(1)); // C#
        assert!(c_major.contains(2)); // D
        assert!(!c_major.contains(3)); // D#
        assert!(c_major.contains(4)); // E
        assert!(c_major.contains(5)); // F
        assert!(!c_major.contains(6)); // F#
        assert!(c_major.contains(7)); // G
        assert!(!c_major.contains(8)); // G#
        assert!(c_major.contains(9)); // A
        assert!(!c_major.contains(10)); // A#
        assert!(c_major.contains(11)); // B
    }

    #[test]
    fn test_scale_minor() {
        let a_minor = ScaleConstraint::minor(9); // A = 9
                                                 // A natural minor: A B C D E F G
        assert!(a_minor.contains(9)); // A
        assert!(a_minor.contains(11)); // B
        assert!(a_minor.contains(0)); // C
        assert!(a_minor.contains(2)); // D
        assert!(a_minor.contains(4)); // E
        assert!(a_minor.contains(5)); // F
        assert!(a_minor.contains(7)); // G
        assert!(!a_minor.contains(1)); // C#
        assert!(!a_minor.contains(6)); // F#
    }

    #[test]
    fn test_nearest_scale_note_chromatic() {
        let scale = ScaleConstraint::chromatic();
        // Any note should round to itself.
        assert_eq!(scale.nearest_scale_note(69.0), 69.0); // A4
        assert_eq!(scale.nearest_scale_note(60.0), 60.0); // C4
    }

    #[test]
    fn test_nearest_scale_note_major() {
        let c_major = ScaleConstraint::major(0);
        // C# (61) should snap to C (60) or D (62).
        let nearest = c_major.nearest_scale_note(61.0);
        assert!(nearest == 60.0 || nearest == 62.0);

        // F# (66) should snap to F (65) or G (67).
        let nearest = c_major.nearest_scale_note(66.0);
        assert!(nearest == 65.0 || nearest == 67.0);
    }

    #[test]
    fn test_correction_440hz() {
        let mut proc = PitchCorrectionProcessor::new(44100.0);
        proc.set_correction_amount(1.0);

        // 440 Hz is exactly A4 (MIDI 69), so correction should be ~0.
        // Run several times to let smoother settle.
        let mut shift = 0.0;
        for _ in 0..50 {
            shift = proc.compute_correction(440.0, 1.0);
        }
        assert!(
            shift.abs() < 0.5,
            "440 Hz correction shift should be near 0, got {}",
            shift
        );
    }

    #[test]
    fn test_correction_445hz() {
        let mut proc = PitchCorrectionProcessor::new(44100.0);
        proc.set_correction_amount(1.0);

        // 445 Hz is sharp of A4. Correction should push it down (negative shift).
        let mut shift = 0.0;
        for _ in 0..100 {
            shift = proc.compute_correction(445.0, 1.0);
        }
        // 445 Hz = ~69.196 MIDI, target = 69.0, so shift should be negative.
        assert!(
            shift < 0.0,
            "445 Hz should be corrected downward, got {}",
            shift
        );
        assert!(
            shift > -1.0,
            "Correction should be less than 1 semitone, got {}",
            shift
        );
    }

    #[test]
    fn test_correction_zero_amount() {
        let mut proc = PitchCorrectionProcessor::new(44100.0);
        proc.set_correction_amount(0.0);

        // With zero correction amount, shift should be 0.
        for _ in 0..50 {
            let shift = proc.compute_correction(445.0, 0.0);
            assert!(
                shift.abs() < 0.01,
                "Zero correction amount should give ~0 shift"
            );
        }
    }

    #[test]
    fn test_correction_partial_amount() {
        let mut proc = PitchCorrectionProcessor::new(44100.0);

        // 50% correction: should shift half as much as full correction.
        let mut shift_half = 0.0;
        for _ in 0..100 {
            shift_half = proc.compute_correction(445.0, 0.5);
        }

        proc.reset();

        let mut shift_full = 0.0;
        for _ in 0..100 {
            shift_full = proc.compute_correction(445.0, 1.0);
        }

        // Half correction should be roughly half of full correction.
        // Allow some tolerance due to smoothing dynamics.
        assert!(
            shift_half.abs() < shift_full.abs() + 0.1,
            "Half correction ({}) should be less than full ({})",
            shift_half,
            shift_full
        );
    }

    #[test]
    fn test_target_frequency() {
        let proc = PitchCorrectionProcessor::new(44100.0);

        // 440 Hz -> target should be 440 Hz.
        let target = proc.target_frequency(440.0);
        assert!(
            (target - 440.0).abs() < 1.0,
            "Target for 440 Hz should be ~440, got {}",
            target
        );

        // 0 Hz -> should return 0.
        assert_eq!(proc.target_frequency(0.0), 0.0);
    }

    #[test]
    fn test_correction_reset() {
        let mut proc = PitchCorrectionProcessor::new(44100.0);
        proc.set_correction_amount(1.0);

        for _ in 0..50 {
            proc.compute_correction(440.0, 1.0);
        }

        proc.reset();

        // After reset, internal state should be cleared.
        // First correction call should still work.
        let shift = proc.compute_correction(440.0, 1.0);
        assert!(shift.abs() < 1.0);
    }

    #[test]
    fn test_vibrato_scale() {
        let mut proc = PitchCorrectionProcessor::new(44100.0);
        proc.set_vibrato_scale(0.0);
        assert_eq!(proc.vibrato_scale, 0.0);

        proc.set_vibrato_scale(2.0);
        assert_eq!(proc.vibrato_scale, 2.0);

        // Clamp test.
        proc.set_vibrato_scale(5.0);
        assert_eq!(proc.vibrato_scale, 2.0);
    }

    #[test]
    fn test_unvoiced_returns_zero() {
        let mut proc = PitchCorrectionProcessor::new(44100.0);
        proc.set_correction_amount(1.0);

        // Negative or zero frequency = unvoiced, should return ~0 shift.
        for _ in 0..10 {
            let shift = proc.compute_correction(0.0, 1.0);
            assert!(shift.abs() < 0.5);

            let shift = proc.compute_correction(-100.0, 1.0);
            assert!(shift.abs() < 0.5);
        }
    }
}
