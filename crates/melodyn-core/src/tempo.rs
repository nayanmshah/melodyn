//! Tempo detection and beat tracking system.
//!
//! Implements onset detection (energy-based, spectral flux), inter-onset
//! interval analysis, autocorrelation-based tempo estimation, beat grid
//! generation, time signature detection, bar/measure mapping, tempo curve
//! extraction, and groove pattern analysis. Models MUTempoAnalyzer,
//! MUBeatGrid, MUBarSignatureMap, and related classes.

use crate::note::DetectedNote;
use crate::spectral::SpectralTimeSlice;

// ---------------------------------------------------------------------------
// Onset detection
// ---------------------------------------------------------------------------

/// A detected onset event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnsetEvent {
    /// Time in seconds.
    pub time: f64,
    /// Onset strength (0.0 .. 1.0).
    pub strength: f64,
    /// Frame index in the analysis.
    pub frame_index: usize,
}

/// Detect onsets from spectral time slices using spectral flux.
/// Spectral flux = sum of positive magnitude differences between consecutive frames.
pub fn detect_onsets_spectral(slices: &[SpectralTimeSlice], threshold: f64) -> Vec<OnsetEvent> {
    if slices.len() < 2 {
        return vec![];
    }

    // Compute spectral flux for each frame
    let mut flux = Vec::with_capacity(slices.len());
    flux.push(0.0);
    for i in 1..slices.len() {
        let mut sf = 0.0f64;
        let prev = &slices[i - 1].magnitude;
        let curr = &slices[i].magnitude;
        let len = prev.len().min(curr.len());
        for j in 0..len {
            let diff = curr[j] - prev[j];
            if diff > 0.0 {
                sf += diff;
            }
        }
        flux.push(sf);
    }

    // Normalize flux
    let max_flux = flux.iter().copied().fold(0.0f64, f64::max);
    if max_flux < 1e-12 {
        return vec![];
    }
    let norm_flux: Vec<f64> = flux.iter().map(|&f| f / max_flux).collect();

    // Adaptive thresholding: onset if flux exceeds local median + threshold
    let window = 10;
    let mut onsets = Vec::new();

    for i in 1..norm_flux.len() {
        let start = i.saturating_sub(window);
        let end = (i + window + 1).min(norm_flux.len());
        let mut local: Vec<f64> = norm_flux[start..end].to_vec();
        local.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = local[local.len() / 2];

        if norm_flux[i] > median + threshold && norm_flux[i] > norm_flux[i - 1] {
            // Check it is a local peak
            let is_peak = if i + 1 < norm_flux.len() {
                norm_flux[i] >= norm_flux[i + 1]
            } else {
                true
            };
            if is_peak {
                onsets.push(OnsetEvent {
                    time: slices[i].time_seconds,
                    strength: norm_flux[i],
                    frame_index: i,
                });
            }
        }
    }

    onsets
}

/// Detect onsets from detected notes (note-onset based).
pub fn detect_onsets_from_notes(notes: &[DetectedNote]) -> Vec<OnsetEvent> {
    let mut onsets: Vec<OnsetEvent> = notes
        .iter()
        .enumerate()
        .map(|(i, n)| OnsetEvent {
            time: n.start_time,
            strength: n.amplitude.min(1.0),
            frame_index: i,
        })
        .collect();
    onsets.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    onsets
}

// ---------------------------------------------------------------------------
// Tempo estimation
// ---------------------------------------------------------------------------

/// Tempo detection range.
#[derive(Debug, Clone, Copy)]
pub struct TempoDetectionRange {
    /// Minimum BPM to consider.
    pub min_bpm: f64,
    /// Maximum BPM to consider.
    pub max_bpm: f64,
}

impl Default for TempoDetectionRange {
    fn default() -> Self {
        Self {
            min_bpm: 40.0,
            max_bpm: 240.0,
        }
    }
}

/// A detected tempo period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoPeriod {
    /// Tempo in BPM.
    pub bpm: f64,
    /// Period in seconds (60/bpm).
    pub period: f64,
    /// Confidence / strength of this tempo estimate.
    pub strength: f64,
}

/// Estimate tempo from onset times using autocorrelation of the onset signal.
///
/// 1. Create an onset strength signal (time-quantized).
/// 2. Compute autocorrelation.
/// 3. Find peaks corresponding to tempo periods.
pub fn estimate_tempo(
    onsets: &[OnsetEvent],
    range: &TempoDetectionRange,
    duration: f64,
) -> Vec<TempoPeriod> {
    if onsets.len() < 3 || duration < 1.0 {
        return vec![];
    }

    // Quantize onsets to a grid for autocorrelation.
    // Resolution: 10ms bins.
    let resolution = 0.01;
    let num_bins = (duration / resolution).ceil() as usize + 1;
    let mut onset_signal = vec![0.0f64; num_bins];

    for onset in onsets {
        let bin = (onset.time / resolution).round() as usize;
        if bin < num_bins {
            onset_signal[bin] += onset.strength;
        }
    }

    // Compute autocorrelation for lag range corresponding to BPM range
    let min_lag = (60.0 / range.max_bpm / resolution).ceil() as usize;
    let max_lag = (60.0 / range.min_bpm / resolution).floor() as usize;
    let max_lag = max_lag.min(num_bins / 2);

    if min_lag >= max_lag {
        return vec![];
    }

    let mut autocorr = vec![0.0f64; max_lag + 1];
    let energy: f64 = onset_signal.iter().map(|x| x * x).sum();
    if energy < 1e-12 {
        return vec![];
    }

    for lag in min_lag..=max_lag {
        let mut sum = 0.0;
        for i in 0..(num_bins - lag) {
            sum += onset_signal[i] * onset_signal[i + lag];
        }
        autocorr[lag] = sum / energy;
    }

    // Find peaks in the autocorrelation
    let mut peaks = Vec::new();
    for lag in (min_lag + 1)..max_lag {
        if autocorr[lag] > autocorr[lag - 1] && autocorr[lag] > autocorr[lag + 1] {
            let period = lag as f64 * resolution;
            let bpm = 60.0 / period;
            if bpm >= range.min_bpm && bpm <= range.max_bpm {
                peaks.push(TempoPeriod {
                    bpm,
                    period,
                    strength: autocorr[lag],
                });
            }
        }
    }

    peaks.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Prefer tempos that are integer multiples of stronger candidates
    // (e.g., prefer 120 BPM over 60 BPM if both are detected).
    if peaks.len() > 1 {
        let top_bpm = peaks[0].bpm;
        for p in &mut peaks[1..] {
            let ratio = p.bpm / top_bpm;
            if (ratio - 2.0).abs() < 0.1 || (ratio - 0.5).abs() < 0.05 {
                // This is a double or half of the top tempo -- slightly boost
                p.strength *= 0.8;
            }
        }
        peaks.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    peaks
}

/// Simpler tempo estimation using inter-onset intervals (IOIs).
/// Clusters IOIs and returns the most common tempo.
pub fn estimate_tempo_from_ioi(
    onsets: &[OnsetEvent],
    range: &TempoDetectionRange,
) -> Option<TempoPeriod> {
    if onsets.len() < 3 {
        return None;
    }

    let min_ioi = 60.0 / range.max_bpm;
    let max_ioi = 60.0 / range.min_bpm;

    // Collect inter-onset intervals
    let mut iois = Vec::new();
    for i in 1..onsets.len() {
        let ioi = onsets[i].time - onsets[i - 1].time;
        if ioi >= min_ioi && ioi <= max_ioi {
            iois.push(ioi);
        }
    }

    if iois.is_empty() {
        return None;
    }

    // Cluster IOIs using a simple histogram approach
    let num_buckets = 100;
    let bucket_width = (max_ioi - min_ioi) / num_buckets as f64;
    let mut histogram = vec![0usize; num_buckets];

    for &ioi in &iois {
        let bucket = ((ioi - min_ioi) / bucket_width).floor() as usize;
        if bucket < num_buckets {
            histogram[bucket] += 1;
        }
    }

    // Find the peak bucket
    let (best_bucket, &best_count) = histogram
        .iter()
        .enumerate()
        .max_by_key(|(_, &count)| count)?;

    if best_count == 0 {
        return None;
    }

    let period = min_ioi + (best_bucket as f64 + 0.5) * bucket_width;
    let bpm = 60.0 / period;
    let strength = best_count as f64 / iois.len() as f64;

    Some(TempoPeriod {
        bpm,
        period,
        strength,
    })
}

// ---------------------------------------------------------------------------
// Beat grid
// ---------------------------------------------------------------------------

/// A single beat in the beat grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyzedBeat {
    /// Time of this beat in seconds.
    pub time: f64,
    /// Beat number within the bar (0-based).
    pub beat_in_bar: u8,
    /// Bar number (0-based).
    pub bar: usize,
    /// Whether this is a downbeat (beat 0 of a bar).
    pub is_downbeat: bool,
    /// Confidence that this beat aligns with actual audio events.
    pub confidence: f64,
}

/// A beat grid: regular pulse positions derived from tempo analysis.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// All beats in the grid.
    pub beats: Vec<AnalyzedBeat>,
    /// Tempo in BPM.
    pub bpm: f64,
    /// Beats per bar (numerator of time signature).
    pub beats_per_bar: u8,
    /// Start time offset of the grid (phase).
    pub phase_offset: f64,
}

impl BeatGrid {
    /// Generate a beat grid from a tempo estimate and time signature.
    pub fn generate(
        bpm: f64,
        beats_per_bar: u8,
        start_time: f64,
        end_time: f64,
        phase_offset: f64,
    ) -> Self {
        let period = 60.0 / bpm;
        let mut beats = Vec::new();
        let mut t = start_time + phase_offset;
        let mut bar = 0usize;
        let mut beat_in_bar = 0u8;

        while t <= end_time {
            beats.push(AnalyzedBeat {
                time: t,
                beat_in_bar,
                bar,
                is_downbeat: beat_in_bar == 0,
                confidence: 1.0,
            });
            t += period;
            beat_in_bar += 1;
            if beat_in_bar >= beats_per_bar {
                beat_in_bar = 0;
                bar += 1;
            }
        }

        BeatGrid {
            beats,
            bpm,
            beats_per_bar,
            phase_offset,
        }
    }

    /// Find the beat grid phase that best aligns with detected onsets.
    pub fn align_to_onsets(
        bpm: f64,
        beats_per_bar: u8,
        onsets: &[OnsetEvent],
        duration: f64,
    ) -> Self {
        let period = 60.0 / bpm;
        let num_phases = 32;
        let phase_step = period / num_phases as f64;
        let mut best_phase = 0.0;
        let mut best_score = f64::NEG_INFINITY;

        for p in 0..num_phases {
            let phase = p as f64 * phase_step;
            let score = Self::phase_alignment_score(bpm, phase, onsets, duration);
            if score > best_score {
                best_score = score;
                best_phase = phase;
            }
        }

        Self::generate(bpm, beats_per_bar, 0.0, duration, best_phase)
    }

    /// Score how well a phase aligns the beat grid with onset events.
    fn phase_alignment_score(bpm: f64, phase: f64, onsets: &[OnsetEvent], _duration: f64) -> f64 {
        let period = 60.0 / bpm;
        let tolerance = period * 0.15; // 15% of beat period
        let mut score = 0.0;

        for onset in onsets {
            // Find distance to nearest beat
            let beats_from_start = (onset.time - phase) / period;
            let nearest_beat_time = phase + beats_from_start.round() * period;
            let dist = (onset.time - nearest_beat_time).abs();
            if dist < tolerance {
                score += onset.strength * (1.0 - dist / tolerance);
            }
        }

        score
    }

    /// Get the beat at or just before a given time.
    pub fn beat_at(&self, time: f64) -> Option<&AnalyzedBeat> {
        self.beats.iter().rev().find(|b| b.time <= time)
    }

    /// Get all downbeats.
    pub fn downbeats(&self) -> Vec<&AnalyzedBeat> {
        self.beats.iter().filter(|b| b.is_downbeat).collect()
    }
}

// ---------------------------------------------------------------------------
// Time signature detection
// ---------------------------------------------------------------------------

/// A time signature (bar signature).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarSignature {
    /// Numerator (beats per bar).
    pub numerator: u8,
    /// Denominator (beat unit: 4 = quarter note, 8 = eighth note).
    pub denominator: u8,
}

impl BarSignature {
    pub fn new(num: u8, den: u8) -> Self {
        Self {
            numerator: num,
            denominator: den,
        }
    }

    /// Common 4/4 time.
    pub fn four_four() -> Self {
        Self::new(4, 4)
    }

    /// 3/4 time (waltz).
    pub fn three_four() -> Self {
        Self::new(3, 4)
    }

    /// 6/8 time.
    pub fn six_eight() -> Self {
        Self::new(6, 8)
    }
}

/// Detect time signature from onsets and tempo.
///
/// Checks whether downbeat accents align better with groups of 2, 3, or 4 beats.
pub fn detect_time_signature(onsets: &[OnsetEvent], bpm: f64) -> BarSignature {
    if onsets.len() < 4 {
        return BarSignature::four_four();
    }

    let period = 60.0 / bpm;

    // For each candidate grouping (2, 3, 4, 6), check accent strength
    let candidates = [(2u8, 4u8), (3, 4), (4, 4), (6, 8)];
    let mut best_sig = BarSignature::four_four();
    let mut best_score = f64::NEG_INFINITY;

    for &(num, den) in &candidates {
        let bar_duration = num as f64 * period * if den == 8 { 0.5 } else { 1.0 };
        if bar_duration < 0.1 {
            continue;
        }

        let mut score = 0.0;
        let mut count = 0;

        for onset in onsets {
            let pos_in_bar = (onset.time % bar_duration) / bar_duration;
            // Downbeat should be near position 0 (or 1)
            let downbeat_dist = pos_in_bar.min(1.0 - pos_in_bar);
            if downbeat_dist < 0.1 {
                score += onset.strength * 2.0;
                count += 1;
            }
            // Check for strong beats at divisions
            let beat_pos = pos_in_bar * num as f64;
            let beat_dist = (beat_pos - beat_pos.round()).abs();
            if beat_dist < 0.15 {
                score += onset.strength;
                count += 1;
            }
        }

        if count > 0 {
            let avg_score = score / count as f64;
            if avg_score > best_score {
                best_score = avg_score;
                best_sig = BarSignature::new(num, den);
            }
        }
    }

    best_sig
}

// ---------------------------------------------------------------------------
// Bar signature map
// ---------------------------------------------------------------------------

/// A map of time signature changes over the course of a piece.
#[derive(Debug, Clone)]
pub struct BarSignatureMap {
    /// (time, signature) pairs, sorted by time.
    pub changes: Vec<(f64, BarSignature)>,
}

impl BarSignatureMap {
    /// Create a map with a single time signature for the entire piece.
    pub fn constant(sig: BarSignature) -> Self {
        Self {
            changes: vec![(0.0, sig)],
        }
    }

    /// Get the time signature active at a given time.
    pub fn signature_at(&self, time: f64) -> BarSignature {
        self.changes
            .iter()
            .rev()
            .find(|(t, _)| *t <= time)
            .map(|(_, sig)| *sig)
            .unwrap_or(BarSignature::four_four())
    }
}

// ---------------------------------------------------------------------------
// Tempo curve (rubato / tempo changes)
// ---------------------------------------------------------------------------

/// A point on a tempo curve (instantaneous tempo at a given time).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoPoint {
    pub time: f64,
    pub bpm: f64,
}

/// Extract a tempo curve from onset events, tracking how tempo varies over time.
/// Uses a sliding window of IOIs.
pub fn extract_tempo_curve(onsets: &[OnsetEvent], window_size: usize) -> Vec<TempoPoint> {
    if onsets.len() < window_size + 1 {
        return vec![];
    }

    let mut curve = Vec::new();

    for i in 0..(onsets.len() - window_size) {
        let window_onsets = &onsets[i..i + window_size + 1];
        let mut iois = Vec::with_capacity(window_size);
        for j in 1..window_onsets.len() {
            let ioi = window_onsets[j].time - window_onsets[j - 1].time;
            if ioi > 0.01 {
                iois.push(ioi);
            }
        }

        if iois.is_empty() {
            continue;
        }

        // Median IOI for robustness
        iois.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ioi = iois[iois.len() / 2];
        let bpm = 60.0 / median_ioi;

        let center_time =
            (window_onsets.first().unwrap().time + window_onsets.last().unwrap().time) / 2.0;

        curve.push(TempoPoint {
            time: center_time,
            bpm,
        });
    }

    curve
}

// ---------------------------------------------------------------------------
// Groove pattern analysis
// ---------------------------------------------------------------------------

/// A groove template describing micro-timing deviations from a straight grid.
#[derive(Debug, Clone)]
pub struct GrooveTemplate {
    /// Name of the groove pattern.
    pub name: String,
    /// Number of subdivisions per beat.
    pub subdivisions: u8,
    /// Timing deviation for each subdivision as a fraction of one subdivision
    /// duration (-0.5 .. +0.5). Index 0 is the beat itself.
    pub deviations: Vec<f64>,
    /// Velocity scale for each subdivision (0.0 .. 1.0).
    pub velocity_scale: Vec<f64>,
}

impl GrooveTemplate {
    /// A straight (no swing) groove.
    pub fn straight(subdivisions: u8) -> Self {
        let n = subdivisions as usize;
        Self {
            name: "Straight".to_string(),
            subdivisions,
            deviations: vec![0.0; n],
            velocity_scale: vec![1.0; n],
        }
    }

    /// A swing groove (triplet feel on 8th notes).
    /// `swing_amount`: 0.0 = straight, 1.0 = full triplet swing.
    pub fn swing(swing_amount: f64) -> Self {
        let deviation = swing_amount * (1.0 / 6.0); // Max swing = 1/6 of beat
        Self {
            name: format!("Swing {:.0}%", swing_amount * 100.0),
            subdivisions: 2,
            deviations: vec![0.0, deviation],
            velocity_scale: vec![1.0, 0.7],
        }
    }
}

/// Detect swing amount from onset timing relative to a beat grid.
/// Returns swing amount (0.0 = straight, ~0.67 = triplet swing).
pub fn detect_swing(onsets: &[OnsetEvent], bpm: f64) -> f64 {
    if onsets.len() < 4 {
        return 0.0;
    }

    let beat_period = 60.0 / bpm;
    let _eighth_note = beat_period / 2.0;
    let mut swing_ratios = Vec::new();

    // Look at pairs of onsets that are roughly an 8th note apart
    for i in 0..onsets.len().saturating_sub(1) {
        let ioi = onsets[i + 1].time - onsets[i].time;
        // Check if the pair spans roughly one beat (two 8th notes)
        if (ioi - beat_period).abs() < beat_period * 0.3 {
            // Look for the note between them
            let midpoint = onsets[i].time + beat_period / 2.0;
            // Find onset closest to this midpoint
            let closest = onsets.iter().min_by(|a, b| {
                let da = (a.time - midpoint).abs();
                let db = (b.time - midpoint).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(mid) = closest {
                let first_half = mid.time - onsets[i].time;
                if first_half > 0.01 && ioi > 0.01 {
                    let ratio = first_half / ioi;
                    if ratio > 0.4 && ratio < 0.8 {
                        swing_ratios.push(ratio);
                    }
                }
            }
        }
    }

    if swing_ratios.is_empty() {
        return 0.0;
    }

    let avg_ratio = swing_ratios.iter().sum::<f64>() / swing_ratios.len() as f64;
    // Convert ratio to swing amount: 0.5 = straight, 0.667 = full triplet
    ((avg_ratio - 0.5) / (2.0 / 3.0 - 0.5)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Quarter note assignment
// ---------------------------------------------------------------------------

/// Quantize note onset times to the nearest beat grid position.
pub fn quantize_to_grid(notes: &[DetectedNote], grid: &BeatGrid) -> Vec<(usize, f64)> {
    notes
        .iter()
        .enumerate()
        .map(|(i, note)| {
            let nearest_beat = grid
                .beats
                .iter()
                .min_by(|a, b| {
                    let da = (a.time - note.start_time).abs();
                    let db = (b.time - note.start_time).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|b| b.time)
                .unwrap_or(note.start_time);
            (i, nearest_beat)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Full tempo analyzer
// ---------------------------------------------------------------------------

/// Configuration for the tempo analyzer.
#[derive(Debug, Clone)]
pub struct TempoAnalyzerConfig {
    /// Tempo detection range.
    pub range: TempoDetectionRange,
    /// Onset detection threshold (0.0 .. 1.0).
    pub onset_threshold: f64,
    /// Window size for tempo curve extraction (number of onsets).
    pub curve_window: usize,
}

impl Default for TempoAnalyzerConfig {
    fn default() -> Self {
        Self {
            range: TempoDetectionRange::default(),
            onset_threshold: 0.15,
            curve_window: 8,
        }
    }
}

/// Complete tempo analysis result.
#[derive(Debug, Clone)]
pub struct TempoAnalysisResult {
    /// Detected onsets.
    pub onsets: Vec<OnsetEvent>,
    /// Primary tempo estimate.
    pub primary_tempo: Option<TempoPeriod>,
    /// All tempo candidates.
    pub tempo_candidates: Vec<TempoPeriod>,
    /// Beat grid aligned to onsets.
    pub beat_grid: Option<BeatGrid>,
    /// Detected time signature.
    pub time_signature: BarSignature,
    /// Tempo curve (if tempo varies).
    pub tempo_curve: Vec<TempoPoint>,
    /// Detected swing amount.
    pub swing: f64,
}

/// Full tempo analysis pipeline.
pub struct TempoAnalyzer {
    config: TempoAnalyzerConfig,
}

impl TempoAnalyzer {
    pub fn new(config: TempoAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Analyze tempo from spectral time slices.
    pub fn analyze_spectral(
        &self,
        slices: &[SpectralTimeSlice],
        duration: f64,
    ) -> TempoAnalysisResult {
        let onsets = detect_onsets_spectral(slices, self.config.onset_threshold);
        self.analyze_from_onsets(onsets, duration)
    }

    /// Analyze tempo from detected notes.
    pub fn analyze_notes(&self, notes: &[DetectedNote], duration: f64) -> TempoAnalysisResult {
        let onsets = detect_onsets_from_notes(notes);
        self.analyze_from_onsets(onsets, duration)
    }

    fn analyze_from_onsets(&self, onsets: Vec<OnsetEvent>, duration: f64) -> TempoAnalysisResult {
        // Estimate tempo
        let tempo_candidates = estimate_tempo(&onsets, &self.config.range, duration);
        let primary_tempo = tempo_candidates.first().copied();

        // Detect time signature
        let time_signature = if let Some(ref tp) = primary_tempo {
            detect_time_signature(&onsets, tp.bpm)
        } else {
            BarSignature::four_four()
        };

        // Build beat grid
        let beat_grid = primary_tempo.map(|tp| {
            BeatGrid::align_to_onsets(tp.bpm, time_signature.numerator, &onsets, duration)
        });

        // Extract tempo curve
        let tempo_curve = extract_tempo_curve(&onsets, self.config.curve_window);

        // Detect swing
        let swing = if let Some(ref tp) = primary_tempo {
            detect_swing(&onsets, tp.bpm)
        } else {
            0.0
        };

        TempoAnalysisResult {
            onsets,
            primary_tempo,
            tempo_candidates,
            beat_grid,
            time_signature,
            tempo_curve,
            swing,
        }
    }
}

// ---------------------------------------------------------------------------
// Tempo formatter
// ---------------------------------------------------------------------------

/// Format a tempo value for display.
pub fn format_tempo(bpm: f64) -> String {
    if (bpm - bpm.round()).abs() < 0.05 {
        format!("{} BPM", bpm.round() as i32)
    } else {
        format!("{:.1} BPM", bpm)
    }
}

/// Format a time signature for display.
pub fn format_time_signature(sig: &BarSignature) -> String {
    format!("{}/{}", sig.numerator, sig.denominator)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchCandidate;

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

    fn make_regular_onsets(bpm: f64, num_beats: usize) -> Vec<OnsetEvent> {
        let period = 60.0 / bpm;
        (0..num_beats)
            .map(|i| OnsetEvent {
                time: i as f64 * period,
                strength: if i % 4 == 0 { 1.0 } else { 0.7 },
                frame_index: i,
            })
            .collect()
    }

    #[test]
    fn test_onset_detection_from_notes() {
        let notes = vec![
            make_note(60.0, 0.0, 0.5, 0.8),
            make_note(64.0, 0.5, 0.5, 0.6),
            make_note(67.0, 1.0, 0.5, 0.7),
        ];
        let onsets = detect_onsets_from_notes(&notes);
        assert_eq!(onsets.len(), 3);
        assert!((onsets[0].time - 0.0).abs() < 1e-10);
        assert!((onsets[1].time - 0.5).abs() < 1e-10);
        assert!((onsets[2].time - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tempo_estimation_120bpm() {
        let onsets = make_regular_onsets(120.0, 32);
        let duration = 32.0 * 0.5; // 16 seconds at 120 BPM
        let range = TempoDetectionRange::default();
        let tempos = estimate_tempo(&onsets, &range, duration);

        assert!(!tempos.is_empty(), "Should detect tempo");
        let best = &tempos[0];
        assert!(
            (best.bpm - 120.0).abs() < 5.0,
            "Expected ~120 BPM, got {:.1}",
            best.bpm
        );
    }

    #[test]
    fn test_tempo_estimation_ioi() {
        let onsets = make_regular_onsets(90.0, 20);
        let range = TempoDetectionRange::default();
        let tempo = estimate_tempo_from_ioi(&onsets, &range);
        assert!(tempo.is_some());
        let tp = tempo.unwrap();
        assert!(
            (tp.bpm - 90.0).abs() < 5.0,
            "Expected ~90 BPM, got {:.1}",
            tp.bpm
        );
    }

    #[test]
    fn test_beat_grid_generation() {
        let grid = BeatGrid::generate(120.0, 4, 0.0, 4.0, 0.0);
        assert_eq!(grid.bpm, 120.0);
        assert_eq!(grid.beats_per_bar, 4);
        // At 120 BPM, 4 seconds = 8 beats
        assert_eq!(grid.beats.len(), 9); // 0.0, 0.5, 1.0, ..., 4.0
        assert!(grid.beats[0].is_downbeat);
        assert!(!grid.beats[1].is_downbeat);
        assert!(grid.beats[4].is_downbeat); // beat 4 = bar 1 downbeat
    }

    #[test]
    fn test_beat_grid_alignment() {
        let onsets = make_regular_onsets(120.0, 16);
        let grid = BeatGrid::align_to_onsets(120.0, 4, &onsets, 8.0);
        // Phase should be near 0 since onsets start at t=0
        assert!(
            grid.phase_offset.abs() < 0.1,
            "Phase {} should be near 0",
            grid.phase_offset
        );
    }

    #[test]
    fn test_time_signature_detection() {
        // Create onsets with strong downbeats every 3 beats (3/4 time)
        let bpm = 120.0;
        let period = 60.0 / bpm;
        let onsets: Vec<OnsetEvent> = (0..24)
            .map(|i| OnsetEvent {
                time: i as f64 * period,
                strength: if i % 3 == 0 { 1.0 } else { 0.4 },
                frame_index: i,
            })
            .collect();
        let sig = detect_time_signature(&onsets, bpm);
        // Should detect 3/4
        assert_eq!(
            sig.numerator, 3,
            "Expected 3/4, got {}/{}",
            sig.numerator, sig.denominator
        );
    }

    #[test]
    fn test_tempo_curve() {
        // Gradually accelerating onsets
        let mut onsets = Vec::new();
        let mut t = 0.0;
        for i in 0..20 {
            onsets.push(OnsetEvent {
                time: t,
                strength: 0.8,
                frame_index: i,
            });
            let bpm = 100.0 + i as f64 * 2.0; // 100 to 138 BPM
            t += 60.0 / bpm;
        }

        let curve = extract_tempo_curve(&onsets, 4);
        assert!(!curve.is_empty());
        // Tempo should generally increase
        if curve.len() >= 2 {
            assert!(
                curve.last().unwrap().bpm > curve.first().unwrap().bpm,
                "Tempo should increase"
            );
        }
    }

    #[test]
    fn test_groove_template_straight() {
        let tmpl = GrooveTemplate::straight(4);
        assert_eq!(tmpl.subdivisions, 4);
        assert!(tmpl.deviations.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn test_groove_template_swing() {
        let tmpl = GrooveTemplate::swing(0.5);
        assert_eq!(tmpl.subdivisions, 2);
        assert!(tmpl.deviations[0] == 0.0);
        assert!(tmpl.deviations[1] > 0.0);
    }

    #[test]
    fn test_format_tempo() {
        assert_eq!(format_tempo(120.0), "120 BPM");
        assert_eq!(format_tempo(120.5), "120.5 BPM");
    }

    #[test]
    fn test_format_time_signature() {
        assert_eq!(format_time_signature(&BarSignature::four_four()), "4/4");
        assert_eq!(format_time_signature(&BarSignature::three_four()), "3/4");
    }

    #[test]
    fn test_bar_signature_map() {
        let map = BarSignatureMap::constant(BarSignature::four_four());
        assert_eq!(map.signature_at(5.0), BarSignature::four_four());
    }

    #[test]
    fn test_quantize_to_grid() {
        let grid = BeatGrid::generate(120.0, 4, 0.0, 4.0, 0.0);
        let notes = vec![
            make_note(60.0, 0.03, 0.4, 0.8), // slightly after beat 0
            make_note(64.0, 0.52, 0.4, 0.6), // slightly after beat 1
        ];
        let quantized = quantize_to_grid(&notes, &grid);
        assert_eq!(quantized.len(), 2);
        assert!((quantized[0].1 - 0.0).abs() < 0.01);
        assert!((quantized[1].1 - 0.5).abs() < 0.01);
    }
}
