//! Detected note representation.
//!
//! Models `MUAudioSourcePrincipalItem` from the original Melodyne engine.
//! Each DetectedNote captures all the properties of a detected note:
//! pitch, amplitude, timing, vibrato, attack/release, formant profile,
//! and pitch contour across its duration.

use crate::pitch::PitchCandidate;

// ---------------------------------------------------------------------------
// Detected note
// ---------------------------------------------------------------------------

/// A detected note in the audio, corresponding to MUAudioSourcePrincipalItem.
///
/// This is the primary output of the analysis pipeline -- each instance
/// represents a single detected pitched event with all its properties.
#[derive(Debug, Clone)]
pub struct DetectedNote {
    /// Fundamental frequency in Hz (weighted average across the note).
    pub frequency: f64,
    /// MIDI note number (fractional for microtuning).
    pub midi_note: f64,
    /// Start time in seconds.
    pub start_time: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Start sample index.
    pub start_sample: usize,
    /// End sample index.
    pub end_sample: usize,
    /// Average amplitude (linear).
    pub amplitude: f64,
    /// Peak amplitude (linear).
    pub peak_amplitude: f64,
    /// Detection confidence (0.0 .. 1.0).
    pub confidence: f64,
    /// Pitch contour: per-frame frequency values across the note.
    pub pitch_contour: Vec<f64>,
    /// Amplitude contour: per-frame amplitude values across the note.
    pub amplitude_contour: Vec<f64>,
    /// Vibrato rate in Hz (modulation frequency).
    pub vibrato_rate: f64,
    /// Vibrato depth in cents (modulation amplitude).
    pub vibrato_depth: f64,
    /// Attack time in seconds (onset to steady state).
    pub attack_time: f64,
    /// Release time in seconds (steady state to silence).
    pub release_time: f64,
}

impl DetectedNote {
    /// End time in seconds.
    pub fn end_time(&self) -> f64 {
        self.start_time + self.duration
    }

    /// Note name (e.g., "A4", "C#3").
    pub fn note_name(&self) -> String {
        PitchCandidate::note_name(self.midi_note)
    }

    /// Pitch deviation from nearest semitone in cents.
    pub fn cents_deviation(&self) -> f64 {
        (self.midi_note - self.midi_note.round()) * 100.0
    }

    /// Whether this note overlaps with another note in time.
    pub fn overlaps(&self, other: &DetectedNote) -> bool {
        self.start_time < other.end_time() && self.end_time() > other.start_time
    }

    /// Compute the attack time from the amplitude contour.
    /// Attack = time from 10% to 90% of peak amplitude.
    pub fn compute_attack_time(&self, frame_duration: f64) -> f64 {
        if self.amplitude_contour.is_empty() {
            return 0.0;
        }

        let peak = self
            .amplitude_contour
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        if peak < 1e-10 {
            return 0.0;
        }

        let threshold_lo = peak * 0.1;
        let threshold_hi = peak * 0.9;

        let mut start_frame = None;
        let mut end_frame = None;

        for (i, &amp) in self.amplitude_contour.iter().enumerate() {
            if start_frame.is_none() && amp >= threshold_lo {
                start_frame = Some(i);
            }
            if start_frame.is_some() && end_frame.is_none() && amp >= threshold_hi {
                end_frame = Some(i);
                break;
            }
        }

        match (start_frame, end_frame) {
            (Some(s), Some(e)) => (e - s) as f64 * frame_duration,
            _ => 0.0,
        }
    }

    /// Compute the release time from the amplitude contour.
    /// Release = time from 90% to 10% of peak amplitude after the peak.
    pub fn compute_release_time(&self, frame_duration: f64) -> f64 {
        if self.amplitude_contour.is_empty() {
            return 0.0;
        }

        let peak = self
            .amplitude_contour
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        if peak < 1e-10 {
            return 0.0;
        }

        let peak_idx = self
            .amplitude_contour
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let threshold_hi = peak * 0.9;
        let threshold_lo = peak * 0.1;

        let mut start_frame = None;
        let mut end_frame = None;

        for i in peak_idx..self.amplitude_contour.len() {
            if start_frame.is_none() && self.amplitude_contour[i] <= threshold_hi {
                start_frame = Some(i);
            }
            if start_frame.is_some() && end_frame.is_none() && self.amplitude_contour[i] <= threshold_lo {
                end_frame = Some(i);
                break;
            }
        }

        match (start_frame, end_frame) {
            (Some(s), Some(e)) => (e - s) as f64 * frame_duration,
            _ => 0.0,
        }
    }

    /// Create a new note with computed attack/release from its contours.
    pub fn with_computed_envelope(mut self, frame_duration: f64) -> Self {
        self.attack_time = self.compute_attack_time(frame_duration);
        self.release_time = self.compute_release_time(frame_duration);
        self
    }

    /// Frequency at a given time (interpolated from pitch contour).
    pub fn frequency_at(&self, time: f64) -> f64 {
        if self.pitch_contour.is_empty() {
            return self.frequency;
        }
        if self.duration <= 0.0 {
            return self.frequency;
        }

        let t = (time - self.start_time) / self.duration;
        let t = t.clamp(0.0, 1.0);
        let idx_f = t * (self.pitch_contour.len() - 1) as f64;
        let idx = idx_f.floor() as usize;
        let frac = idx_f - idx as f64;

        if idx + 1 < self.pitch_contour.len() {
            self.pitch_contour[idx] * (1.0 - frac) + self.pitch_contour[idx + 1] * frac
        } else {
            self.pitch_contour[idx.min(self.pitch_contour.len() - 1)]
        }
    }

    /// Amplitude at a given time (interpolated from amplitude contour).
    pub fn amplitude_at(&self, time: f64) -> f64 {
        if self.amplitude_contour.is_empty() {
            return self.amplitude;
        }
        if self.duration <= 0.0 {
            return self.amplitude;
        }

        let t = (time - self.start_time) / self.duration;
        let t = t.clamp(0.0, 1.0);
        let idx_f = t * (self.amplitude_contour.len() - 1) as f64;
        let idx = idx_f.floor() as usize;
        let frac = idx_f - idx as f64;

        if idx + 1 < self.amplitude_contour.len() {
            self.amplitude_contour[idx] * (1.0 - frac) + self.amplitude_contour[idx + 1] * frac
        } else {
            self.amplitude_contour[idx.min(self.amplitude_contour.len() - 1)]
        }
    }
}

// ---------------------------------------------------------------------------
// Note editing (pitch shifting, time stretching)
// ---------------------------------------------------------------------------

/// Parameters for editing a detected note's pitch.
#[derive(Debug, Clone)]
pub struct PitchEditParams {
    /// Target frequency in Hz (None = keep original).
    pub target_frequency: Option<f64>,
    /// Pitch shift in semitones (applied on top of target_frequency).
    pub semitone_shift: f64,
    /// Pitch drift correction factor (0.0 = no correction, 1.0 = snap to target).
    pub drift_correction: f64,
    /// Vibrato depth multiplier (1.0 = unchanged).
    pub vibrato_depth_scale: f64,
}

impl Default for PitchEditParams {
    fn default() -> Self {
        Self {
            target_frequency: None,
            semitone_shift: 0.0,
            drift_correction: 0.0,
            vibrato_depth_scale: 1.0,
        }
    }
}

/// Parameters for editing a detected note's timing.
#[derive(Debug, Clone)]
pub struct TimeEditParams {
    /// Time stretch factor (1.0 = unchanged, 2.0 = double duration).
    pub time_stretch: f64,
    /// New start time (None = keep original).
    pub new_start_time: Option<f64>,
    /// Attack time adjustment factor.
    pub attack_scale: f64,
    /// Release time adjustment factor.
    pub release_scale: f64,
}

impl Default for TimeEditParams {
    fn default() -> Self {
        Self {
            time_stretch: 1.0,
            new_start_time: None,
            attack_scale: 1.0,
            release_scale: 1.0,
        }
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
            start_time: 1.0,
            duration: 0.5,
            start_sample: 44100,
            end_sample: 66150,
            amplitude: 0.5,
            peak_amplitude: 0.8,
            confidence: 0.95,
            pitch_contour: vec![438.0, 440.0, 442.0, 440.0, 439.0],
            amplitude_contour: vec![0.1, 0.5, 0.8, 0.6, 0.2],
            vibrato_rate: 5.0,
            vibrato_depth: 20.0,
            attack_time: 0.0,
            release_time: 0.0,
        }
    }

    #[test]
    fn test_note_end_time() {
        let note = make_test_note();
        assert!((note.end_time() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_note_name() {
        let note = make_test_note();
        assert_eq!(note.note_name(), "A4");
    }

    #[test]
    fn test_cents_deviation() {
        let mut note = make_test_note();
        note.midi_note = 69.5; // A4 + 50 cents
        assert!((note.cents_deviation().abs() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_at_interpolation() {
        let note = make_test_note();
        // At start
        let f_start = note.frequency_at(1.0);
        assert!((f_start - 438.0).abs() < 1.0);
        // At end
        let f_end = note.frequency_at(1.5);
        assert!((f_end - 439.0).abs() < 1.0);
        // At midpoint
        let f_mid = note.frequency_at(1.25);
        assert!(f_mid > 438.0 && f_mid < 443.0);
    }

    #[test]
    fn test_amplitude_at_interpolation() {
        let note = make_test_note();
        let a_start = note.amplitude_at(1.0);
        assert!((a_start - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_overlap_detection() {
        let note1 = make_test_note();
        let mut note2 = make_test_note();
        note2.start_time = 1.3; // overlaps
        assert!(note1.overlaps(&note2));

        let mut note3 = make_test_note();
        note3.start_time = 2.0; // no overlap
        assert!(!note1.overlaps(&note3));
    }

    #[test]
    fn test_attack_time_computation() {
        let note = make_test_note();
        let frame_dur = 0.1; // 100ms per frame
        let attack = note.compute_attack_time(frame_dur);
        // Peak is 0.8: 10% = 0.08, 90% = 0.72
        // amp[0]=0.1 > 0.08, amp[2]=0.8 > 0.72 -> attack = (2-0)*0.1 = 0.2
        assert!((attack - 0.2).abs() < 0.01);
    }
}
