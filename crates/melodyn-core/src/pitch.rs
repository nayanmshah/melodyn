//! Pitch detection: monophonic and polyphonic paths.
//!
//! Implements autocorrelation-based pitch detection (YIN-style) and
//! Harmonic Product Spectrum (HPS) for robust fundamental frequency estimation.
//! Polyphonic detection uses iterative spectral subtraction.
//!
//! Corresponds to MUElementAnalyzer's makeMonophonicItems() and
//! findPolyphonicDetectionItems() methods.

use crate::note::DetectedNote;
use crate::spectral::SpectralTimeSlice;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the pitch detector.
#[derive(Debug, Clone)]
pub struct PitchDetectorConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Minimum detectable frequency in Hz.
    pub min_freq: f64,
    /// Maximum detectable frequency in Hz.
    pub max_freq: f64,
    /// Whether to use polyphonic detection.
    pub polyphonic: bool,
    /// Maximum number of simultaneous notes in polyphonic mode.
    pub max_polyphony: usize,
    /// YIN threshold for aperiodicity (lower = stricter).
    pub yin_threshold: f64,
    /// Number of HPS downsampling stages.
    pub hps_harmonics: usize,
    /// Minimum confidence to accept a pitch candidate.
    pub min_confidence: f64,
    /// Minimum gap in seconds between distinct notes.
    pub min_note_gap: f64,
}

impl Default for PitchDetectorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            min_freq: 60.0,
            max_freq: 5000.0,
            polyphonic: false,
            max_polyphony: 6,
            yin_threshold: 0.15,
            hps_harmonics: 5,
            min_confidence: 0.3,
            min_note_gap: 0.03,
        }
    }
}

// ---------------------------------------------------------------------------
// Pitch candidate
// ---------------------------------------------------------------------------

/// A pitch candidate detected in a single frame.
#[derive(Debug, Clone)]
pub struct PitchCandidate {
    /// Estimated fundamental frequency in Hz.
    pub frequency: f64,
    /// MIDI note number (fractional for microtuning).
    pub midi_note: f64,
    /// Confidence score 0.0 .. 1.0.
    pub confidence: f64,
    /// Amplitude of the detected pitch (linear).
    pub amplitude: f64,
    /// Frame index where this was detected.
    pub frame_index: usize,
    /// Time in seconds.
    pub time: f64,
}

impl PitchCandidate {
    /// Convert frequency to MIDI note number.
    /// A4 = 440 Hz = MIDI 69.
    pub fn freq_to_midi(freq: f64) -> f64 {
        if freq <= 0.0 {
            return 0.0;
        }
        69.0 + 12.0 * (freq / 440.0).log2()
    }

    /// Convert MIDI note number to frequency.
    pub fn midi_to_freq(midi: f64) -> f64 {
        440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
    }

    /// Note name (e.g., "A4", "C#3").
    pub fn note_name(midi: f64) -> String {
        let note_num = midi.round() as i32;
        let octave = note_num / 12 - 1;
        let note_idx = ((note_num % 12) + 12) % 12;
        let names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let cents = ((midi - midi.round()) * 100.0).round() as i32;
        if cents == 0 {
            format!("{}{}", names[note_idx as usize], octave)
        } else if cents > 0 {
            format!("{}{}+{}c", names[note_idx as usize], octave, cents)
        } else {
            format!("{}{}{}c", names[note_idx as usize], octave, cents)
        }
    }
}

// ---------------------------------------------------------------------------
// Pitch detector
// ---------------------------------------------------------------------------

/// Main pitch detector implementing autocorrelation (YIN) and
/// Harmonic Product Spectrum (HPS) methods.
pub struct PitchDetector {
    config: PitchDetectorConfig,
}

impl PitchDetector {
    /// Create a new pitch detector.
    pub fn new(config: PitchDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect pitch in a single audio frame using the YIN algorithm.
    ///
    /// YIN is an autocorrelation-based method with cumulative mean normalized
    /// difference function, providing robust monophonic pitch detection.
    ///
    /// Returns the best pitch candidate, if any.
    pub fn detect_yin(&self, frame: &[f64]) -> Option<PitchCandidate> {
        let sr = self.config.sample_rate;
        let min_lag = (sr / self.config.max_freq).ceil() as usize;
        let max_lag = (sr / self.config.min_freq).floor() as usize;
        let frame_len = frame.len();

        if max_lag >= frame_len / 2 {
            return None;
        }

        let w = frame_len / 2;

        // Step 1: Compute the difference function
        // d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
        let mut diff = vec![0.0f64; max_lag + 1];
        for tau in 1..=max_lag {
            let mut sum = 0.0;
            for j in 0..w {
                if j + tau < frame_len {
                    let d = frame[j] - frame[j + tau];
                    sum += d * d;
                }
            }
            diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalized difference function (CMNDF)
        // d'(tau) = d(tau) / [(1/tau) * sum_{j=1}^{tau} d(j)]
        let mut cmndf = vec![0.0f64; max_lag + 1];
        cmndf[0] = 1.0;
        let mut running_sum = 0.0;
        for tau in 1..=max_lag {
            running_sum += diff[tau];
            if running_sum > 0.0 {
                cmndf[tau] = diff[tau] * tau as f64 / running_sum;
            } else {
                cmndf[tau] = 0.0;
            }
        }

        // Step 3: Absolute threshold - find first dip below threshold
        let mut best_tau = 0;
        let mut best_val = 1.0;

        for tau in min_lag..=max_lag {
            if cmndf[tau] < self.config.yin_threshold {
                // Find the local minimum in this valley
                let mut local_tau = tau;
                while local_tau + 1 <= max_lag && cmndf[local_tau + 1] < cmndf[local_tau] {
                    local_tau += 1;
                }
                if cmndf[local_tau] < best_val {
                    best_val = cmndf[local_tau];
                    best_tau = local_tau;
                }
                break;
            }
        }

        // If no dip below threshold, find global minimum
        if best_tau == 0 {
            for tau in min_lag..=max_lag {
                if cmndf[tau] < best_val {
                    best_val = cmndf[tau];
                    best_tau = tau;
                }
            }
        }

        if best_tau == 0 {
            return None;
        }

        // Step 4: Parabolic interpolation around the minimum
        let true_tau = if best_tau > 0 && best_tau < max_lag {
            let alpha = cmndf[best_tau - 1];
            let beta = cmndf[best_tau];
            let gamma = cmndf[best_tau + 1];
            let denom = alpha - 2.0 * beta + gamma;
            if denom.abs() > 1e-10 {
                best_tau as f64 + 0.5 * (alpha - gamma) / denom
            } else {
                best_tau as f64
            }
        } else {
            best_tau as f64
        };

        let freq = sr / true_tau;
        let confidence = 1.0 - best_val.min(1.0);

        // Compute amplitude (RMS of the frame)
        let rms = (frame.iter().map(|&s| s * s).sum::<f64>() / frame.len() as f64).sqrt();

        if confidence >= self.config.min_confidence && freq >= self.config.min_freq && freq <= self.config.max_freq {
            Some(PitchCandidate {
                frequency: freq,
                midi_note: PitchCandidate::freq_to_midi(freq),
                confidence,
                amplitude: rms,
                frame_index: 0,
                time: 0.0,
            })
        } else {
            None
        }
    }

    /// Detect pitch using Harmonic Product Spectrum (HPS).
    ///
    /// Multiplies downsampled versions of the magnitude spectrum to reinforce
    /// the fundamental frequency.
    pub fn detect_hps(&self, magnitude: &[f64], sample_rate: f64, fft_size: usize) -> Option<PitchCandidate> {
        let num_bins = magnitude.len();
        let min_bin = (self.config.min_freq * fft_size as f64 / sample_rate).ceil() as usize;
        let max_bin = (self.config.max_freq * fft_size as f64 / sample_rate)
            .floor()
            .min((num_bins - 1) as f64) as usize;
        let harmonics = self.config.hps_harmonics;

        if min_bin >= max_bin || max_bin == 0 {
            return None;
        }

        // Compute HPS: product of downsampled magnitude spectra
        let usable_bins = num_bins / harmonics;
        if usable_bins <= min_bin {
            return None;
        }

        let mut hps = vec![1.0f64; usable_bins];
        for h in 1..=harmonics {
            for k in 0..usable_bins {
                let idx = k * h;
                if idx < num_bins {
                    hps[k] *= magnitude[idx];
                }
            }
        }

        // Find peak in HPS within valid range
        let search_end = usable_bins.min(max_bin + 1);
        let mut best_bin = min_bin;
        let mut best_val = hps[min_bin];
        for k in min_bin..search_end {
            if hps[k] > best_val {
                best_val = hps[k];
                best_bin = k;
            }
        }

        if best_val < 1e-20 {
            return None;
        }

        // Parabolic interpolation
        let true_bin = if best_bin > 0 && best_bin + 1 < usable_bins {
            let alpha = if hps[best_bin - 1] > 0.0 {
                hps[best_bin - 1].ln()
            } else {
                -300.0
            };
            let beta = if hps[best_bin] > 0.0 {
                hps[best_bin].ln()
            } else {
                -300.0
            };
            let gamma = if hps[best_bin + 1] > 0.0 {
                hps[best_bin + 1].ln()
            } else {
                -300.0
            };
            let denom = alpha - 2.0 * beta + gamma;
            if denom.abs() > 1e-10 {
                best_bin as f64 + 0.5 * (alpha - gamma) / denom
            } else {
                best_bin as f64
            }
        } else {
            best_bin as f64
        };

        let freq = true_bin * sample_rate / fft_size as f64;

        // Confidence from peak prominence
        let mean_hps = hps[min_bin..search_end].iter().sum::<f64>() / (search_end - min_bin) as f64;
        let confidence = if mean_hps > 0.0 {
            (best_val / mean_hps).log10().max(0.0).min(1.0) / 3.0
        } else {
            0.0
        };

        if freq >= self.config.min_freq && freq <= self.config.max_freq {
            Some(PitchCandidate {
                frequency: freq,
                midi_note: PitchCandidate::freq_to_midi(freq),
                confidence: confidence.max(0.1), // HPS always gives some confidence
                amplitude: magnitude[best_bin],
                frame_index: 0,
                time: 0.0,
            })
        } else {
            None
        }
    }

    /// Run monophonic pitch detection across all frames.
    ///
    /// Uses YIN on the time-domain signal for each frame, with HPS as a
    /// cross-check from the spectral data. Groups consecutive frames
    /// into note segments.
    pub fn detect_monophonic(
        &self,
        audio: &[f64],
        spectral_slices: &[SpectralTimeSlice],
    ) -> Vec<DetectedNote> {
        let sr = self.config.sample_rate;
        let frame_size = if !spectral_slices.is_empty() {
            spectral_slices[0].fft_size
        } else {
            4096
        };
        let hop_size = if spectral_slices.len() > 1 {
            spectral_slices[1].center_sample - spectral_slices[0].center_sample
        } else {
            1024
        };

        // Detect pitch per frame
        let mut frame_pitches: Vec<Option<PitchCandidate>> = Vec::new();

        for (i, slice) in spectral_slices.iter().enumerate() {
            // Extract audio frame for YIN
            let start = slice.center_sample.saturating_sub(frame_size / 2);
            let end = (start + frame_size).min(audio.len());
            let frame = &audio[start..end];

            // Run YIN
            let mut yin_result = self.detect_yin(frame);

            // Run HPS as cross-check
            let hps_result =
                self.detect_hps(&slice.magnitude, sr, slice.fft_size);

            // Combine: prefer YIN if confident, fall back to HPS
            let candidate = match (&mut yin_result, &hps_result) {
                (Some(yin), Some(hps)) => {
                    // If both agree (within a semitone), boost confidence
                    let ratio = yin.frequency / hps.frequency;
                    if (ratio - 1.0).abs() < 0.06 {
                        yin.confidence = (yin.confidence + hps.confidence * 0.5).min(1.0);
                    }
                    yin.frame_index = i;
                    yin.time = slice.time_seconds;
                    Some(yin.clone())
                }
                (Some(yin), None) => {
                    yin.frame_index = i;
                    yin.time = slice.time_seconds;
                    Some(yin.clone())
                }
                (None, Some(hps)) => {
                    let mut h = hps.clone();
                    h.frame_index = i;
                    h.time = slice.time_seconds;
                    Some(h)
                }
                (None, None) => None,
            };

            frame_pitches.push(candidate);
        }

        // Group consecutive pitched frames into notes
        self.group_into_notes(&frame_pitches, hop_size)
    }

    /// Run polyphonic pitch detection.
    ///
    /// Uses iterative spectral peak finding and subtraction: detect the
    /// strongest pitch, subtract its harmonics from the spectrum, repeat.
    pub fn detect_polyphonic(
        &self,
        audio: &[f64],
        spectral_slices: &[SpectralTimeSlice],
    ) -> Vec<DetectedNote> {
        let sr = self.config.sample_rate;
        let hop_size = if spectral_slices.len() > 1 {
            spectral_slices[1].center_sample - spectral_slices[0].center_sample
        } else {
            1024
        };

        // For each frame, detect multiple pitches
        let mut all_frame_candidates: Vec<Vec<PitchCandidate>> = Vec::new();

        for (i, slice) in spectral_slices.iter().enumerate() {
            let mut residual_mag = slice.magnitude.clone();
            let mut candidates = Vec::new();

            for _ in 0..self.max_polyphony() {
                if let Some(mut cand) =
                    self.detect_hps(&residual_mag, sr, slice.fft_size)
                {
                    cand.frame_index = i;
                    cand.time = slice.time_seconds;

                    if cand.confidence >= self.config.min_confidence {
                        // Subtract harmonics of detected pitch from residual
                        subtract_harmonics(
                            &mut residual_mag,
                            cand.frequency,
                            sr,
                            slice.fft_size,
                            8,
                        );
                        candidates.push(cand);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            all_frame_candidates.push(candidates);
        }

        // Track pitches across frames and group into notes
        self.track_polyphonic_notes(&all_frame_candidates, hop_size)
    }

    fn max_polyphony(&self) -> usize {
        self.config.max_polyphony
    }

    /// Group consecutive pitched frames into note segments.
    fn group_into_notes(
        &self,
        frame_pitches: &[Option<PitchCandidate>],
        hop_size: usize,
    ) -> Vec<DetectedNote> {
        let sr = self.config.sample_rate;
        let mut notes = Vec::new();
        let mut current_note: Option<NoteAccumulator> = None;

        for (i, pitch) in frame_pitches.iter().enumerate() {
            match (pitch, &mut current_note) {
                (Some(p), Some(acc)) => {
                    // Check if this continues the current note (within ~1 semitone)
                    let ratio = p.frequency / acc.avg_frequency();
                    if (ratio - 1.0).abs() < 0.06 {
                        acc.add_frame(p);
                    } else {
                        // New note -- finish the old one
                        notes.push(acc.to_note(hop_size, sr));
                        current_note = Some(NoteAccumulator::new(p));
                    }
                }
                (Some(p), None) => {
                    current_note = Some(NoteAccumulator::new(p));
                }
                (None, Some(acc)) => {
                    // Gap -- finish current note
                    notes.push(acc.to_note(hop_size, sr));
                    current_note = None;
                }
                (None, None) => {}
            }
        }

        // Finish last note
        if let Some(acc) = current_note {
            notes.push(acc.to_note(hop_size, sr));
        }

        notes
    }

    /// Track polyphonic pitch candidates across frames into note segments.
    fn track_polyphonic_notes(
        &self,
        frame_candidates: &[Vec<PitchCandidate>],
        hop_size: usize,
    ) -> Vec<DetectedNote> {
        let sr = self.config.sample_rate;
        let mut active_tracks: Vec<NoteAccumulator> = Vec::new();
        let mut finished_notes: Vec<DetectedNote> = Vec::new();

        for (frame_idx, candidates) in frame_candidates.iter().enumerate() {
            let mut matched = vec![false; candidates.len()];
            let mut track_matched = vec![false; active_tracks.len()];

            // Match candidates to existing tracks
            for (ci, cand) in candidates.iter().enumerate() {
                let mut best_track = None;
                let mut best_distance = f64::MAX;

                for (ti, track) in active_tracks.iter().enumerate() {
                    if track_matched[ti] {
                        continue;
                    }
                    let ratio = cand.frequency / track.avg_frequency();
                    let distance = (ratio - 1.0).abs();
                    if distance < 0.06 && distance < best_distance {
                        best_distance = distance;
                        best_track = Some(ti);
                    }
                }

                if let Some(ti) = best_track {
                    active_tracks[ti].add_frame(cand);
                    matched[ci] = true;
                    track_matched[ti] = true;
                }
            }

            // Start new tracks for unmatched candidates
            for (ci, cand) in candidates.iter().enumerate() {
                if !matched[ci] {
                    active_tracks.push(NoteAccumulator::new(cand));
                    track_matched.push(true);
                }
            }

            // Finish tracks that weren't matched this frame
            let mut i = 0;
            while i < active_tracks.len() {
                if i < track_matched.len() && !track_matched[i] {
                    let acc = active_tracks.remove(i);
                    if track_matched.len() > i {
                        track_matched.remove(i);
                    }
                    finished_notes.push(acc.to_note(hop_size, sr));
                } else {
                    i += 1;
                }
            }
        }

        // Finish remaining tracks
        for acc in active_tracks {
            finished_notes.push(acc.to_note(hop_size, sr));
        }

        finished_notes
    }

    /// Access the configuration.
    pub fn config(&self) -> &PitchDetectorConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helper: subtract harmonics from magnitude spectrum
// ---------------------------------------------------------------------------

/// Subtract the harmonics of a detected pitch from the magnitude spectrum.
/// This enables iterative polyphonic detection by removing already-found notes.
fn subtract_harmonics(
    magnitude: &mut [f64],
    fundamental: f64,
    sample_rate: f64,
    fft_size: usize,
    num_harmonics: usize,
) {
    let freq_res = sample_rate / fft_size as f64;
    let num_bins = magnitude.len();

    for h in 1..=num_harmonics {
        let freq = fundamental * h as f64;
        let bin = (freq / freq_res).round() as usize;

        if bin >= num_bins {
            break;
        }

        // Zero out bins around the harmonic (width depends on window bandwidth)
        let bandwidth = 3; // bins on each side
        let start = bin.saturating_sub(bandwidth);
        let end = (bin + bandwidth + 1).min(num_bins);

        for k in start..end {
            magnitude[k] *= 0.1; // Attenuate rather than zero to avoid artifacts
        }
    }
}

// ---------------------------------------------------------------------------
// Note accumulator
// ---------------------------------------------------------------------------

/// Accumulates pitch candidates across frames to form a note.
struct NoteAccumulator {
    frequencies: Vec<f64>,
    confidences: Vec<f64>,
    amplitudes: Vec<f64>,
    start_frame: usize,
    end_frame: usize,
    start_time: f64,
}

impl NoteAccumulator {
    fn new(candidate: &PitchCandidate) -> Self {
        Self {
            frequencies: vec![candidate.frequency],
            confidences: vec![candidate.confidence],
            amplitudes: vec![candidate.amplitude],
            start_frame: candidate.frame_index,
            end_frame: candidate.frame_index,
            start_time: candidate.time,
        }
    }

    fn add_frame(&mut self, candidate: &PitchCandidate) {
        self.frequencies.push(candidate.frequency);
        self.confidences.push(candidate.confidence);
        self.amplitudes.push(candidate.amplitude);
        self.end_frame = candidate.frame_index;
    }

    fn avg_frequency(&self) -> f64 {
        // Weighted average by confidence
        let total_weight: f64 = self.confidences.iter().sum();
        if total_weight < 1e-10 {
            return self.frequencies.iter().sum::<f64>() / self.frequencies.len() as f64;
        }
        self.frequencies
            .iter()
            .zip(self.confidences.iter())
            .map(|(&f, &c)| f * c)
            .sum::<f64>()
            / total_weight
    }

    fn to_note(&self, hop_size: usize, sample_rate: f64) -> DetectedNote {
        let freq = self.avg_frequency();
        let avg_confidence =
            self.confidences.iter().sum::<f64>() / self.confidences.len() as f64;
        let peak_amplitude = self
            .amplitudes
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        let avg_amplitude =
            self.amplitudes.iter().sum::<f64>() / self.amplitudes.len() as f64;

        let num_frames = self.end_frame - self.start_frame + 1;
        let duration = num_frames as f64 * hop_size as f64 / sample_rate;

        DetectedNote {
            frequency: freq,
            midi_note: PitchCandidate::freq_to_midi(freq),
            start_time: self.start_time,
            duration,
            start_sample: self.start_frame * hop_size,
            end_sample: (self.end_frame + 1) * hop_size,
            amplitude: avg_amplitude,
            peak_amplitude,
            confidence: avg_confidence,
            pitch_contour: self.frequencies.clone(),
            amplitude_contour: self.amplitudes.clone(),
            vibrato_rate: estimate_vibrato_rate(&self.frequencies, sample_rate, hop_size),
            vibrato_depth: estimate_vibrato_depth(&self.frequencies),
            attack_time: 0.0,   // Computed later in note.rs
            release_time: 0.0,  // Computed later in note.rs
        }
    }
}

// ---------------------------------------------------------------------------
// Vibrato estimation
// ---------------------------------------------------------------------------

/// Estimate vibrato rate from a pitch contour (Hz modulation rate).
fn estimate_vibrato_rate(freqs: &[f64], sample_rate: f64, hop_size: usize) -> f64 {
    if freqs.len() < 8 {
        return 0.0;
    }

    // Convert to cents deviation from mean
    let mean_freq = freqs.iter().sum::<f64>() / freqs.len() as f64;
    let cents: Vec<f64> = freqs
        .iter()
        .map(|&f| 1200.0 * (f / mean_freq).log2())
        .collect();

    // Count zero crossings in the cents deviation
    let mut crossings = 0;
    for i in 1..cents.len() {
        if (cents[i] >= 0.0) != (cents[i - 1] >= 0.0) {
            crossings += 1;
        }
    }

    // Vibrato rate = zero_crossings / (2 * duration)
    let frame_rate = sample_rate / hop_size as f64;
    let duration = freqs.len() as f64 / frame_rate;
    if duration > 0.0 {
        crossings as f64 / (2.0 * duration)
    } else {
        0.0
    }
}

/// Estimate vibrato depth in cents.
fn estimate_vibrato_depth(freqs: &[f64]) -> f64 {
    if freqs.len() < 4 {
        return 0.0;
    }
    let mean_freq = freqs.iter().sum::<f64>() / freqs.len() as f64;
    if mean_freq < 1.0 {
        return 0.0;
    }
    let cents: Vec<f64> = freqs
        .iter()
        .map(|&f| 1200.0 * (f / mean_freq).log2())
        .collect();

    // Depth = standard deviation of cents
    let mean_cents = cents.iter().sum::<f64>() / cents.len() as f64;
    let variance = cents.iter().map(|&c| (c - mean_cents).powi(2)).sum::<f64>() / cents.len() as f64;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_wave(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_freq_to_midi() {
        assert!((PitchCandidate::freq_to_midi(440.0) - 69.0).abs() < 0.01);
        assert!((PitchCandidate::freq_to_midi(261.63) - 60.0).abs() < 0.1);
        assert!((PitchCandidate::freq_to_midi(880.0) - 81.0).abs() < 0.01);
    }

    #[test]
    fn test_midi_to_freq() {
        assert!((PitchCandidate::midi_to_freq(69.0) - 440.0).abs() < 0.01);
        assert!((PitchCandidate::midi_to_freq(60.0) - 261.63).abs() < 0.5);
    }

    #[test]
    fn test_note_name() {
        assert_eq!(PitchCandidate::note_name(69.0), "A4");
        assert_eq!(PitchCandidate::note_name(60.0), "C4");
        assert_eq!(PitchCandidate::note_name(72.0), "C5");
    }

    #[test]
    fn test_yin_pure_tone() {
        let sr = 44100.0;
        let freq = 440.0;
        let audio = sine_wave(freq, sr, 4096);

        let config = PitchDetectorConfig {
            sample_rate: sr,
            yin_threshold: 0.2,
            ..Default::default()
        };
        let detector = PitchDetector::new(config);
        let result = detector.detect_yin(&audio);

        assert!(result.is_some(), "Should detect pitch in pure tone");
        let p = result.unwrap();
        assert!(
            (p.frequency - freq).abs() < 5.0,
            "Expected ~{} Hz, got {} Hz",
            freq,
            p.frequency
        );
        assert!(p.confidence > 0.5, "Confidence {} should be high", p.confidence);
    }

    #[test]
    fn test_yin_different_pitches() {
        let sr = 44100.0;
        let config = PitchDetectorConfig {
            sample_rate: sr,
            yin_threshold: 0.2,
            ..Default::default()
        };
        let detector = PitchDetector::new(config);

        for freq in [220.0, 440.0, 880.0, 1000.0] {
            let audio = sine_wave(freq, sr, 4096);
            let result = detector.detect_yin(&audio);
            assert!(result.is_some(), "Should detect {} Hz", freq);
            let p = result.unwrap();
            assert!(
                (p.frequency - freq).abs() < 10.0,
                "Expected ~{} Hz, got {} Hz",
                freq,
                p.frequency
            );
        }
    }

    #[test]
    fn test_hps_pure_tone() {
        let sr = 44100.0;
        let freq = 440.0;
        let fft_size = 4096;
        let audio = sine_wave(freq, sr, fft_size);

        // Compute magnitude spectrum
        use crate::fft::{FftEngine, WindowType};
        let engine = FftEngine::new(fft_size, WindowType::Hann);
        let spectrum = engine.forward(&audio);
        let magnitude = FftEngine::magnitude_spectrum(&spectrum);

        let config = PitchDetectorConfig {
            sample_rate: sr,
            ..Default::default()
        };
        let detector = PitchDetector::new(config);
        let result = detector.detect_hps(&magnitude, sr, fft_size);

        assert!(result.is_some(), "HPS should detect pitch");
        let p = result.unwrap();
        assert!(
            (p.frequency - freq).abs() < 20.0,
            "HPS: Expected ~{} Hz, got {} Hz",
            freq,
            p.frequency
        );
    }

    #[test]
    fn test_vibrato_estimation() {
        // Simulate vibrato: 440 Hz with 6 Hz vibrato, 50 cents depth
        let sr = 44100.0;
        let hop = 512;
        let num_frames = 200;
        let vibrato_rate = 6.0;
        let vibrato_depth_cents = 50.0;

        let frame_rate = sr / hop as f64;
        let freqs: Vec<f64> = (0..num_frames)
            .map(|i| {
                let t = i as f64 / frame_rate;
                let cents_offset = vibrato_depth_cents * (2.0 * PI * vibrato_rate * t).sin();
                440.0 * 2.0f64.powf(cents_offset / 1200.0)
            })
            .collect();

        let rate = estimate_vibrato_rate(&freqs, sr, hop);
        assert!(
            (rate - vibrato_rate).abs() < 2.0,
            "Expected vibrato rate ~{}, got {}",
            vibrato_rate,
            rate
        );

        let depth = estimate_vibrato_depth(&freqs);
        assert!(
            (depth - vibrato_depth_cents / 2.0_f64.sqrt()).abs() < 15.0,
            "Expected vibrato depth ~{}, got {}",
            vibrato_depth_cents / 2.0_f64.sqrt(),
            depth
        );
    }
}
