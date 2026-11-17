//! Pitch mapping system with constraint-based optimization.
//!
//! Implements the 12-criteria pitch optimization engine that maps detected
//! notes to target pitch classes, chords, and voicings. Models MUPitchMapper,
//! MUPitchMapperCriterion, MUPitchClassMapper, MUBlockVoicingMapper,
//! MUVoiceMapper, and related classes.

use crate::chord::{ChordDescription, ChordType, PitchClass};
use crate::note::DetectedNote;
use crate::pitch::PitchCandidate;
use crate::scale::{CandidateScale, ScaleMode};

// ---------------------------------------------------------------------------
// Note mapping
// ---------------------------------------------------------------------------

/// A mapping from an original note to a new pitch.
#[derive(Debug, Clone)]
pub struct NoteMapping {
    /// Index of the original note.
    pub note_index: usize,
    /// Original MIDI note number.
    pub original_midi: f64,
    /// Mapped MIDI note number.
    pub mapped_midi: f64,
    /// Voice assignment index.
    pub voice: usize,
    /// Mapping confidence / quality score.
    pub quality: f64,
}

impl NoteMapping {
    /// Pitch shift in semitones.
    pub fn shift(&self) -> f64 {
        self.mapped_midi - self.original_midi
    }
}

// ---------------------------------------------------------------------------
// Voicing
// ---------------------------------------------------------------------------

/// A voicing: a set of target pitches for a chord.
#[derive(Debug, Clone)]
pub struct Voicing {
    /// MIDI note numbers of each voice (sorted low to high).
    pub pitches: Vec<f64>,
    /// Voice labels (e.g., "bass", "tenor", "alto", "soprano").
    pub labels: Vec<String>,
}

impl Voicing {
    /// Create from a list of MIDI pitches.
    pub fn from_pitches(pitches: &[f64]) -> Self {
        let mut sorted = pitches.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let labels: Vec<String> = (0..sorted.len()).map(|i| format!("voice_{}", i)).collect();
        Self {
            pitches: sorted,
            labels,
        }
    }

    /// Generate a close voicing for a chord around a center pitch.
    pub fn close_voicing(chord: &ChordDescription, center_midi: f64) -> Self {
        let intervals = chord.chord_type.intervals().unwrap_or(&[]);
        let root = chord.root as f64;
        // Find the octave closest to center for the root
        let root_midi = {
            let base = root + 12.0; // start from octave 1
            let mut best = base;
            while best < 127.0 {
                if (best - center_midi).abs() < (best + 12.0 - center_midi).abs() {
                    break;
                }
                best += 12.0;
            }
            // Adjust to nearest below center
            while best > center_midi + 6.0 {
                best -= 12.0;
            }
            while best < center_midi - 18.0 {
                best += 12.0;
            }
            best
        };

        let mut pitches = vec![root_midi];
        for &iv in intervals {
            let p = root_midi + iv as f64;
            pitches.push(p);
        }
        // Keep all within a reasonable range of center
        for p in &mut pitches {
            while *p > center_midi + 12.0 {
                *p -= 12.0;
            }
            while *p < center_midi - 12.0 {
                *p += 12.0;
            }
        }
        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Self::from_pitches(&pitches)
    }
}

// ---------------------------------------------------------------------------
// Pitch mapper criteria
// ---------------------------------------------------------------------------

/// Weight for a pitch mapping criterion.
#[derive(Debug, Clone, Copy)]
pub struct CriterionWeight {
    pub weight: f64,
    pub enabled: bool,
}

impl CriterionWeight {
    pub fn new(weight: f64) -> Self {
        Self {
            weight,
            enabled: true,
        }
    }

    pub fn disabled() -> Self {
        Self {
            weight: 0.0,
            enabled: false,
        }
    }
}

/// Configuration for all pitch mapping criteria.
#[derive(Debug, Clone)]
pub struct PitchMapperConfig {
    pub chord_coverage: CriterionWeight,
    pub double_count: CriterionWeight,
    pub has_crossings: CriterionWeight,
    pub has_minor_second: CriterionWeight,
    pub lowest_interval_fourth: CriterionWeight,
    pub lowest_interval_fifth: CriterionWeight,
    pub match_voicing: CriterionWeight,
    pub voicing_distance: CriterionWeight,
    pub weighted_distance: CriterionWeight,
    pub transposition: CriterionWeight,
    pub exceeds_limits: CriterionWeight,
    /// Voice range limits (min_midi, max_midi).
    pub voice_range: (f64, f64),
    /// Target chord (if any).
    pub target_chord: Option<ChordDescription>,
    /// Target voicing template (if any).
    pub target_voicing: Option<Voicing>,
}

impl Default for PitchMapperConfig {
    fn default() -> Self {
        Self {
            chord_coverage: CriterionWeight::new(2.0),
            double_count: CriterionWeight::new(1.5),
            has_crossings: CriterionWeight::new(3.0),
            has_minor_second: CriterionWeight::new(1.0),
            lowest_interval_fourth: CriterionWeight::new(0.8),
            lowest_interval_fifth: CriterionWeight::new(0.8),
            match_voicing: CriterionWeight::new(1.0),
            voicing_distance: CriterionWeight::new(1.5),
            weighted_distance: CriterionWeight::new(2.0),
            transposition: CriterionWeight::new(1.0),
            exceeds_limits: CriterionWeight::new(5.0),
            voice_range: (36.0, 96.0), // C2 to C7
            target_chord: None,
            target_voicing: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Individual criterion evaluation functions
// ---------------------------------------------------------------------------

/// Score how well the mapped notes cover the target chord tones.
/// Returns a penalty (lower = better, 0 = all chord tones covered).
fn chord_coverage_penalty(mappings: &[NoteMapping], chord: &ChordDescription) -> f64 {
    let chord_pcs = chord.pitch_classes();
    let mapped_pcs: Vec<PitchClass> = mappings
        .iter()
        .map(|m| ((m.mapped_midi.round() as i32).rem_euclid(12)) as PitchClass)
        .collect();

    let mut uncovered = 0;
    for &cpc in &chord_pcs {
        if !mapped_pcs.contains(&cpc) {
            uncovered += 1;
        }
    }
    uncovered as f64 / chord_pcs.len().max(1) as f64
}

/// Penalty for multiple voices on the same pitch.
fn double_count_penalty(mappings: &[NoteMapping]) -> f64 {
    let mut rounded: Vec<i32> = mappings.iter().map(|m| m.mapped_midi.round() as i32).collect();
    rounded.sort();
    let total = rounded.len();
    rounded.dedup();
    let duplicates = total - rounded.len();
    duplicates as f64 / total.max(1) as f64
}

/// Penalty for voice crossings (higher voice below lower voice).
fn crossings_penalty(mappings: &[NoteMapping]) -> f64 {
    if mappings.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0;
    let mut pairs = 0;
    for i in 0..mappings.len() {
        for j in (i + 1)..mappings.len() {
            if mappings[i].voice < mappings[j].voice {
                pairs += 1;
                if mappings[i].mapped_midi > mappings[j].mapped_midi {
                    crossings += 1;
                }
            } else if mappings[i].voice > mappings[j].voice {
                pairs += 1;
                if mappings[i].mapped_midi < mappings[j].mapped_midi {
                    crossings += 1;
                }
            }
        }
    }
    if pairs == 0 {
        0.0
    } else {
        crossings as f64 / pairs as f64
    }
}

/// Penalty for minor second intervals between voices.
fn minor_second_penalty(mappings: &[NoteMapping]) -> f64 {
    if mappings.len() < 2 {
        return 0.0;
    }
    let mut count = 0;
    let mut pairs = 0;
    for i in 0..mappings.len() {
        for j in (i + 1)..mappings.len() {
            pairs += 1;
            let interval = (mappings[i].mapped_midi - mappings[j].mapped_midi).abs();
            let interval_mod = interval % 12.0;
            if (interval_mod - 1.0).abs() < 0.5 || (interval_mod - 11.0).abs() < 0.5 {
                count += 1;
            }
        }
    }
    if pairs == 0 {
        0.0
    } else {
        count as f64 / pairs as f64
    }
}

/// Penalty if the lowest interval is less than a fourth (5 semitones).
fn lowest_interval_fourth_penalty(mappings: &[NoteMapping]) -> f64 {
    if mappings.len() < 2 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = mappings.iter().map(|m| m.mapped_midi).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lowest_interval = sorted[1] - sorted[0];
    if lowest_interval < 5.0 {
        (5.0 - lowest_interval) / 5.0
    } else {
        0.0
    }
}

/// Penalty if the lowest interval is less than a fifth (7 semitones).
fn lowest_interval_fifth_penalty(mappings: &[NoteMapping]) -> f64 {
    if mappings.len() < 2 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = mappings.iter().map(|m| m.mapped_midi).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lowest_interval = sorted[1] - sorted[0];
    if lowest_interval < 7.0 {
        (7.0 - lowest_interval) / 7.0
    } else {
        0.0
    }
}

/// Penalty for distance from a target voicing template.
fn match_voicing_penalty(mappings: &[NoteMapping], voicing: &Voicing) -> f64 {
    if mappings.is_empty() || voicing.pitches.is_empty() {
        return 0.0;
    }
    let mut total_dist = 0.0;
    let count = mappings.len().min(voicing.pitches.len());
    let mut sorted_mappings: Vec<f64> = mappings.iter().map(|m| m.mapped_midi).collect();
    sorted_mappings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..count {
        let dist = (sorted_mappings[i] - voicing.pitches[i]).abs();
        total_dist += dist;
    }

    total_dist / (count as f64 * 12.0) // Normalize to roughly 0..1 range
}

/// Penalty for total pitch movement (sum of |shift| for all notes).
fn voicing_distance_penalty(mappings: &[NoteMapping]) -> f64 {
    if mappings.is_empty() {
        return 0.0;
    }
    let total: f64 = mappings.iter().map(|m| m.shift().abs()).sum();
    total / (mappings.len() as f64 * 12.0)
}

/// Weighted distance penalty (larger shifts penalized more).
fn weighted_distance_penalty(mappings: &[NoteMapping]) -> f64 {
    if mappings.is_empty() {
        return 0.0;
    }
    let total: f64 = mappings.iter().map(|m| m.shift().abs().powi(2)).sum();
    (total / mappings.len() as f64).sqrt() / 12.0
}

/// Penalty for non-simple transpositions (prefers uniform shifts).
fn transposition_penalty(mappings: &[NoteMapping]) -> f64 {
    if mappings.len() < 2 {
        return 0.0;
    }
    let shifts: Vec<f64> = mappings.iter().map(|m| m.shift()).collect();
    let mean_shift = shifts.iter().sum::<f64>() / shifts.len() as f64;
    let variance = shifts.iter().map(|&s| (s - mean_shift).powi(2)).sum::<f64>() / shifts.len() as f64;
    (variance.sqrt() / 12.0).min(1.0)
}

/// Penalty for notes exceeding voice range limits.
fn exceeds_limits_penalty(mappings: &[NoteMapping], range: (f64, f64)) -> f64 {
    if mappings.is_empty() {
        return 0.0;
    }
    let mut violations = 0;
    for m in mappings {
        if m.mapped_midi < range.0 || m.mapped_midi > range.1 {
            violations += 1;
        }
    }
    violations as f64 / mappings.len() as f64
}

// ---------------------------------------------------------------------------
// Combined scoring
// ---------------------------------------------------------------------------

/// Compute the total penalty score for a set of mappings.
fn total_penalty(mappings: &[NoteMapping], config: &PitchMapperConfig) -> f64 {
    let mut penalty = 0.0;

    if config.chord_coverage.enabled {
        if let Some(ref chord) = config.target_chord {
            penalty += config.chord_coverage.weight * chord_coverage_penalty(mappings, chord);
        }
    }
    if config.double_count.enabled {
        penalty += config.double_count.weight * double_count_penalty(mappings);
    }
    if config.has_crossings.enabled {
        penalty += config.has_crossings.weight * crossings_penalty(mappings);
    }
    if config.has_minor_second.enabled {
        penalty += config.has_minor_second.weight * minor_second_penalty(mappings);
    }
    if config.lowest_interval_fourth.enabled {
        penalty += config.lowest_interval_fourth.weight * lowest_interval_fourth_penalty(mappings);
    }
    if config.lowest_interval_fifth.enabled {
        penalty += config.lowest_interval_fifth.weight * lowest_interval_fifth_penalty(mappings);
    }
    if config.match_voicing.enabled {
        if let Some(ref voicing) = config.target_voicing {
            penalty += config.match_voicing.weight * match_voicing_penalty(mappings, voicing);
        }
    }
    if config.voicing_distance.enabled {
        penalty += config.voicing_distance.weight * voicing_distance_penalty(mappings);
    }
    if config.weighted_distance.enabled {
        penalty += config.weighted_distance.weight * weighted_distance_penalty(mappings);
    }
    if config.transposition.enabled {
        penalty += config.transposition.weight * transposition_penalty(mappings);
    }
    if config.exceeds_limits.enabled {
        penalty += config.exceeds_limits.weight * exceeds_limits_penalty(mappings, config.voice_range);
    }

    penalty
}

// ---------------------------------------------------------------------------
// Pitch class mapper
// ---------------------------------------------------------------------------

/// Map notes to target scale/chord pitch classes (nearest available pitch class).
pub struct PitchClassMapper;

impl PitchClassMapper {
    /// Map each note to the nearest pitch class in the target set.
    pub fn map(notes: &[DetectedNote], target_pcs: &[PitchClass]) -> Vec<NoteMapping> {
        if target_pcs.is_empty() {
            return notes
                .iter()
                .enumerate()
                .map(|(i, n)| NoteMapping {
                    note_index: i,
                    original_midi: n.midi_note,
                    mapped_midi: n.midi_note,
                    voice: i,
                    quality: 1.0,
                })
                .collect();
        }

        notes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let midi = n.midi_note;
                let pc = ((midi.round() as i32).rem_euclid(12)) as u8;

                // If already on a target pitch class, keep it
                if target_pcs.contains(&pc) {
                    return NoteMapping {
                        note_index: i,
                        original_midi: midi,
                        mapped_midi: midi.round(),
                        voice: i,
                        quality: 1.0,
                    };
                }

                // Find nearest target pitch class (minimum semitone distance)
                let mut best_shift = 0i32;
                let mut best_dist = 12i32;
                for &tpc in target_pcs {
                    let up = (tpc as i32 - pc as i32).rem_euclid(12);
                    let down = (pc as i32 - tpc as i32).rem_euclid(12);
                    if up <= down && up < best_dist {
                        best_dist = up;
                        best_shift = up;
                    } else if down < up && down < best_dist {
                        best_dist = down;
                        best_shift = -(down);
                    }
                }

                NoteMapping {
                    note_index: i,
                    original_midi: midi,
                    mapped_midi: midi.round() + best_shift as f64,
                    voice: i,
                    quality: 1.0 - (best_dist as f64 / 6.0),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Transposition mapper
// ---------------------------------------------------------------------------

/// Simple uniform transposition by a fixed interval.
pub struct TranspositionMapper;

impl TranspositionMapper {
    /// Transpose all notes by the given number of semitones.
    pub fn map(notes: &[DetectedNote], semitones: f64) -> Vec<NoteMapping> {
        notes
            .iter()
            .enumerate()
            .map(|(i, n)| NoteMapping {
                note_index: i,
                original_midi: n.midi_note,
                mapped_midi: n.midi_note + semitones,
                voice: i,
                quality: 1.0,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Pitch approach mapper
// ---------------------------------------------------------------------------

/// Approach note type for pitch transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproachType {
    /// Direct jump to target pitch.
    Direct,
    /// Slide (portamento) from previous pitch.
    Slide,
    /// Anticipation: arrive at target pitch slightly early.
    Anticipation,
    /// Grace note approach from a chromatic neighbor.
    GraceNote,
}

/// Map approach notes for pitch transitions between consecutive notes.
pub struct PitchApproachMapper;

impl PitchApproachMapper {
    /// Determine approach types for a sequence of note mappings.
    pub fn analyze_approaches(
        mappings: &[NoteMapping],
        approach_type: ApproachType,
    ) -> Vec<(usize, ApproachType, f64)> {
        if mappings.len() < 2 {
            return vec![];
        }

        let mut approaches = Vec::new();
        for i in 1..mappings.len() {
            let interval = (mappings[i].mapped_midi - mappings[i - 1].mapped_midi).abs();
            let approach = match approach_type {
                ApproachType::Direct => ApproachType::Direct,
                ApproachType::Slide => {
                    if interval <= 2.0 {
                        ApproachType::Slide
                    } else {
                        ApproachType::Direct
                    }
                }
                ApproachType::Anticipation => ApproachType::Anticipation,
                ApproachType::GraceNote => {
                    if interval >= 1.0 {
                        ApproachType::GraceNote
                    } else {
                        ApproachType::Direct
                    }
                }
            };
            approaches.push((i, approach, interval));
        }
        approaches
    }
}

// ---------------------------------------------------------------------------
// Pitch run mapper
// ---------------------------------------------------------------------------

/// Detect and handle melodic runs (fast passages of stepwise motion).
pub struct PitchRunMapper;

impl PitchRunMapper {
    /// Detect runs in a note sequence (3+ consecutive stepwise notes).
    pub fn detect_runs(notes: &[DetectedNote], max_step: f64) -> Vec<(usize, usize)> {
        if notes.len() < 3 {
            return vec![];
        }

        let mut runs = Vec::new();
        let mut run_start = 0;
        let mut in_run = false;

        for i in 1..notes.len() {
            let interval = (notes[i].midi_note - notes[i - 1].midi_note).abs();
            let is_step = interval >= 0.5 && interval <= max_step;

            if is_step {
                if !in_run {
                    run_start = i - 1;
                    in_run = true;
                }
            } else if in_run {
                if i - run_start >= 3 {
                    runs.push((run_start, i - 1));
                }
                in_run = false;
            }
        }

        if in_run && notes.len() - run_start >= 3 {
            runs.push((run_start, notes.len() - 1));
        }

        runs
    }

    /// Map a melodic run to a target scale, preserving direction and step motion.
    pub fn map_run(
        notes: &[DetectedNote],
        start: usize,
        end: usize,
        scale: &CandidateScale,
    ) -> Vec<NoteMapping> {
        let intervals = match scale.mode.intervals() {
            Some(iv) => iv,
            None => return Vec::new(),
        };

        (start..=end)
            .map(|i| {
                let midi = notes[i].midi_note;
                let rounded = midi.round();
                let pc = ((rounded as i32).rem_euclid(12)) as u8;
                let rel = (pc + 12 - scale.tonic) % 12;

                // Find nearest scale tone
                let mut best_iv = 0u8;
                let mut best_dist = 12u8;
                for &iv in intervals {
                    let dist = ((rel as i8 - iv as i8).abs() as u8).min(12 - (rel as i8 - iv as i8).unsigned_abs() as u8);
                    if dist < best_dist {
                        best_dist = dist;
                        best_iv = iv;
                    }
                }

                let target_pc = (scale.tonic + best_iv) % 12;
                let shift = (target_pc as i8 - pc as i8) as f64;
                let shift = if shift > 6.0 {
                    shift - 12.0
                } else if shift < -6.0 {
                    shift + 12.0
                } else {
                    shift
                };

                NoteMapping {
                    note_index: i,
                    original_midi: midi,
                    mapped_midi: rounded + shift,
                    voice: 0,
                    quality: 1.0 - (best_dist as f64 / 6.0),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Block voicing mapper
// ---------------------------------------------------------------------------

/// Map a set of simultaneous notes to a target voicing (block chord style).
pub struct BlockVoicingMapper;

impl BlockVoicingMapper {
    /// Map notes to match a target voicing, minimizing total pitch movement.
    pub fn map(
        notes: &[DetectedNote],
        target: &Voicing,
        config: &PitchMapperConfig,
    ) -> Vec<NoteMapping> {
        if notes.is_empty() || target.pitches.is_empty() {
            return Vec::new();
        }

        let n = notes.len();
        let t = target.pitches.len();

        // Simple greedy assignment: for each target pitch, find the nearest note
        let mut used = vec![false; n];
        let mut mappings = Vec::new();

        for (voice_idx, &target_pitch) in target.pitches.iter().enumerate() {
            let mut best_note = None;
            let mut best_dist = f64::MAX;

            for (ni, note) in notes.iter().enumerate() {
                if used[ni] {
                    continue;
                }
                let dist = (note.midi_note - target_pitch).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_note = Some(ni);
                }
            }

            if let Some(ni) = best_note {
                used[ni] = true;
                mappings.push(NoteMapping {
                    note_index: ni,
                    original_midi: notes[ni].midi_note,
                    mapped_midi: target_pitch,
                    voice: voice_idx,
                    quality: 1.0 - (best_dist / 24.0).min(1.0),
                });
            }
        }

        // Map any remaining unassigned notes to nearest target
        for (ni, note) in notes.iter().enumerate() {
            if used[ni] {
                continue;
            }
            let nearest = target
                .pitches
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (note.midi_note - **a).abs();
                    let db = (note.midi_note - **b).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(vi, &p)| (vi, p))
                .unwrap_or((0, note.midi_note));

            mappings.push(NoteMapping {
                note_index: ni,
                original_midi: note.midi_note,
                mapped_midi: nearest.1,
                voice: nearest.0,
                quality: 0.5,
            });
        }

        mappings
    }
}

// ---------------------------------------------------------------------------
// Voice mapper (voice leading optimization)
// ---------------------------------------------------------------------------

/// Assign voice indices to simultaneous notes for smooth voice leading.
pub struct VoiceMapper;

impl VoiceMapper {
    /// Assign voices to notes based on pitch ordering (lowest = voice 0).
    pub fn assign_voices(notes: &[DetectedNote]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..notes.len()).collect();
        indices.sort_by(|&a, &b| {
            notes[a]
                .midi_note
                .partial_cmp(&notes[b].midi_note)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut voices = vec![0usize; notes.len()];
        for (voice, &note_idx) in indices.iter().enumerate() {
            voices[note_idx] = voice;
        }
        voices
    }

    /// Optimize voice leading between two consecutive sets of note mappings.
    /// Minimizes total pitch movement while avoiding crossings.
    pub fn optimize_voice_leading(
        prev: &[NoteMapping],
        current_notes: &[DetectedNote],
        config: &PitchMapperConfig,
    ) -> Vec<NoteMapping> {
        if prev.is_empty() || current_notes.is_empty() {
            return current_notes
                .iter()
                .enumerate()
                .map(|(i, n)| NoteMapping {
                    note_index: i,
                    original_midi: n.midi_note,
                    mapped_midi: n.midi_note,
                    voice: i,
                    quality: 1.0,
                })
                .collect();
        }

        let n = current_notes.len();
        let p = prev.len();

        // Greedy matching: assign each current note to the nearest previous voice
        let mut used_prev = vec![false; p];
        let mut mappings = Vec::with_capacity(n);

        for (ci, note) in current_notes.iter().enumerate() {
            let mut best_pi = None;
            let mut best_dist = f64::MAX;

            for (pi, pm) in prev.iter().enumerate() {
                if used_prev[pi] {
                    continue;
                }
                let dist = (note.midi_note - pm.mapped_midi).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_pi = Some(pi);
                }
            }

            let voice = best_pi.map(|pi| {
                used_prev[pi] = true;
                prev[pi].voice
            }).unwrap_or(ci);

            mappings.push(NoteMapping {
                note_index: ci,
                original_midi: note.midi_note,
                mapped_midi: note.midi_note,
                voice,
                quality: 1.0 - (best_dist / 24.0).min(0.5),
            });
        }

        mappings
    }
}

// ---------------------------------------------------------------------------
// Full pitch mapper
// ---------------------------------------------------------------------------

/// The main pitch mapper: applies pitch class mapping with constraint optimization.
pub struct PitchMapper {
    config: PitchMapperConfig,
}

impl PitchMapper {
    pub fn new(config: PitchMapperConfig) -> Self {
        Self { config }
    }

    /// Map notes to a target chord, using the pitch class mapper and
    /// then optimizing with the constraint criteria.
    pub fn map_to_chord(
        &self,
        notes: &[DetectedNote],
        chord: &ChordDescription,
    ) -> Vec<NoteMapping> {
        let target_pcs = chord.pitch_classes();
        let mut mappings = PitchClassMapper::map(notes, &target_pcs);

        // Assign voices
        let voices = VoiceMapper::assign_voices(notes);
        for (m, &v) in mappings.iter_mut().zip(voices.iter()) {
            m.voice = v;
        }

        // Evaluate and try to improve with single-note adjustments
        let mut best_penalty = {
            let mut cfg = self.config.clone();
            cfg.target_chord = Some(chord.clone());
            total_penalty(&mappings, &cfg)
        };

        // Hill-climbing: try shifting each note by -2..+2 semitones
        for iteration in 0..3 {
            let mut improved = false;
            for i in 0..mappings.len() {
                let original = mappings[i].mapped_midi;
                for shift in &[-2.0, -1.0, 1.0, 2.0] {
                    mappings[i].mapped_midi = original + shift;

                    let mut cfg = self.config.clone();
                    cfg.target_chord = Some(chord.clone());
                    let penalty = total_penalty(&mappings, &cfg);

                    if penalty < best_penalty {
                        best_penalty = penalty;
                        improved = true;
                    } else {
                        mappings[i].mapped_midi = original;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        // Set quality scores based on final penalty
        for m in &mut mappings {
            m.quality = (1.0 - best_penalty / mappings.len().max(1) as f64).clamp(0.0, 1.0);
        }

        mappings
    }

    /// Map notes to a target scale.
    pub fn map_to_scale(
        &self,
        notes: &[DetectedNote],
        scale: &CandidateScale,
    ) -> Vec<NoteMapping> {
        let intervals = match scale.mode.intervals() {
            Some(iv) => iv,
            None => {
                return notes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| NoteMapping {
                        note_index: i,
                        original_midi: n.midi_note,
                        mapped_midi: n.midi_note,
                        voice: i,
                        quality: 1.0,
                    })
                    .collect();
            }
        };

        let target_pcs: Vec<PitchClass> = intervals
            .iter()
            .map(|&iv| (scale.tonic + iv) % 12)
            .collect();

        PitchClassMapper::map(notes, &target_pcs)
    }

    /// Simple transposition.
    pub fn transpose(&self, notes: &[DetectedNote], semitones: f64) -> Vec<NoteMapping> {
        TranspositionMapper::map(notes, semitones)
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
    fn test_pitch_class_mapper_c_major() {
        let notes = vec![
            make_note(61.0, 0.0, 1.0, 0.8), // C#4 -> should map to C or D
            make_note(63.0, 0.0, 1.0, 0.6), // Eb4 -> should map to E or D
            make_note(66.0, 0.0, 1.0, 0.6), // F#4 -> should map to G or F
        ];
        let target_pcs: Vec<PitchClass> = vec![0, 4, 7]; // C, E, G
        let mappings = PitchClassMapper::map(&notes, &target_pcs);

        assert_eq!(mappings.len(), 3);
        for m in &mappings {
            let mapped_pc = ((m.mapped_midi.round() as i32).rem_euclid(12)) as u8;
            assert!(
                target_pcs.contains(&mapped_pc),
                "Mapped PC {} should be in target set",
                mapped_pc
            );
        }
    }

    #[test]
    fn test_transposition_mapper() {
        let notes = vec![
            make_note(60.0, 0.0, 1.0, 0.8),
            make_note(64.0, 0.0, 1.0, 0.6),
        ];
        let mappings = TranspositionMapper::map(&notes, 7.0);
        assert!((mappings[0].mapped_midi - 67.0).abs() < 0.01);
        assert!((mappings[1].mapped_midi - 71.0).abs() < 0.01);
    }

    #[test]
    fn test_voice_assignment() {
        let notes = vec![
            make_note(67.0, 0.0, 1.0, 0.5), // G4
            make_note(60.0, 0.0, 1.0, 0.5), // C4
            make_note(64.0, 0.0, 1.0, 0.5), // E4
        ];
        let voices = VoiceMapper::assign_voices(&notes);
        // C4 should be voice 0 (lowest), E4 voice 1, G4 voice 2
        assert_eq!(voices[0], 2); // G4 = highest
        assert_eq!(voices[1], 0); // C4 = lowest
        assert_eq!(voices[2], 1); // E4 = middle
    }

    #[test]
    fn test_crossings_penalty() {
        // No crossings
        let mappings = vec![
            NoteMapping {
                note_index: 0,
                original_midi: 60.0,
                mapped_midi: 60.0,
                voice: 0,
                quality: 1.0,
            },
            NoteMapping {
                note_index: 1,
                original_midi: 64.0,
                mapped_midi: 64.0,
                voice: 1,
                quality: 1.0,
            },
        ];
        assert!(crossings_penalty(&mappings) < 0.01);

        // Crossing: voice 0 above voice 1
        let crossed = vec![
            NoteMapping {
                note_index: 0,
                original_midi: 60.0,
                mapped_midi: 70.0,
                voice: 0,
                quality: 1.0,
            },
            NoteMapping {
                note_index: 1,
                original_midi: 64.0,
                mapped_midi: 60.0,
                voice: 1,
                quality: 1.0,
            },
        ];
        assert!(crossings_penalty(&crossed) > 0.5);
    }

    #[test]
    fn test_double_count_penalty() {
        let no_doubles = vec![
            NoteMapping {
                note_index: 0,
                original_midi: 60.0,
                mapped_midi: 60.0,
                voice: 0,
                quality: 1.0,
            },
            NoteMapping {
                note_index: 1,
                original_midi: 64.0,
                mapped_midi: 64.0,
                voice: 1,
                quality: 1.0,
            },
        ];
        assert!(double_count_penalty(&no_doubles) < 0.01);

        let doubles = vec![
            NoteMapping {
                note_index: 0,
                original_midi: 60.0,
                mapped_midi: 60.0,
                voice: 0,
                quality: 1.0,
            },
            NoteMapping {
                note_index: 1,
                original_midi: 64.0,
                mapped_midi: 60.0,
                voice: 1,
                quality: 1.0,
            },
        ];
        assert!(double_count_penalty(&doubles) > 0.3);
    }

    #[test]
    fn test_chord_coverage_penalty() {
        let chord = ChordDescription {
            root: 0,
            chord_type: ChordType::Major,
            bass_note: None,
            confidence: 1.0,
        };

        // All chord tones covered
        let full = vec![
            NoteMapping { note_index: 0, original_midi: 60.0, mapped_midi: 60.0, voice: 0, quality: 1.0 },
            NoteMapping { note_index: 1, original_midi: 64.0, mapped_midi: 64.0, voice: 1, quality: 1.0 },
            NoteMapping { note_index: 2, original_midi: 67.0, mapped_midi: 67.0, voice: 2, quality: 1.0 },
        ];
        assert!(chord_coverage_penalty(&full, &chord) < 0.01);

        // Missing the fifth
        let partial = vec![
            NoteMapping { note_index: 0, original_midi: 60.0, mapped_midi: 60.0, voice: 0, quality: 1.0 },
            NoteMapping { note_index: 1, original_midi: 64.0, mapped_midi: 64.0, voice: 1, quality: 1.0 },
        ];
        assert!(chord_coverage_penalty(&partial, &chord) > 0.2);
    }

    #[test]
    fn test_voicing_close() {
        let chord = ChordDescription {
            root: 0,
            chord_type: ChordType::Major,
            bass_note: None,
            confidence: 1.0,
        };
        let voicing = Voicing::close_voicing(&chord, 60.0);
        assert_eq!(voicing.pitches.len(), 3);
        // Should contain C, E, G near MIDI 60
        for &p in &voicing.pitches {
            assert!(p >= 48.0 && p <= 72.0);
        }
    }

    #[test]
    fn test_pitch_mapper_to_chord() {
        let notes = vec![
            make_note(61.0, 0.0, 1.0, 0.8), // C#4
            make_note(66.0, 0.0, 1.0, 0.6), // F#4
            make_note(70.0, 0.0, 1.0, 0.6), // Bb4
        ];
        let chord = ChordDescription {
            root: 0,
            chord_type: ChordType::Major,
            bass_note: None,
            confidence: 1.0,
        };
        let mapper = PitchMapper::new(PitchMapperConfig::default());
        let mappings = mapper.map_to_chord(&notes, &chord);

        assert_eq!(mappings.len(), 3);
        // All mapped notes should be chord tones (C, E, G)
        let chord_pcs = chord.pitch_classes();
        for m in &mappings {
            let pc = ((m.mapped_midi.round() as i32).rem_euclid(12)) as u8;
            assert!(
                chord_pcs.contains(&pc),
                "Mapped MIDI {} (pc {}) should be a chord tone",
                m.mapped_midi,
                pc
            );
        }
    }

    #[test]
    fn test_pitch_mapper_to_scale() {
        let notes = vec![
            make_note(61.0, 0.0, 0.5, 0.8), // C#4
            make_note(63.0, 0.5, 0.5, 0.6), // Eb4
        ];
        let scale = CandidateScale {
            tonic: 0,
            mode: ScaleMode::Ionian,
            correlation: 0.9,
            confidence: 0.9,
        };
        let mapper = PitchMapper::new(PitchMapperConfig::default());
        let mappings = mapper.map_to_scale(&notes, &scale);

        // All mapped notes should be in C major scale
        let scale_pcs: Vec<u8> = vec![0, 2, 4, 5, 7, 9, 11];
        for m in &mappings {
            let pc = ((m.mapped_midi.round() as i32).rem_euclid(12)) as u8;
            assert!(
                scale_pcs.contains(&pc),
                "Mapped MIDI {} (pc {}) should be in C major",
                m.mapped_midi,
                pc
            );
        }
    }

    #[test]
    fn test_exceeds_limits_penalty() {
        let in_range = vec![
            NoteMapping { note_index: 0, original_midi: 60.0, mapped_midi: 60.0, voice: 0, quality: 1.0 },
        ];
        assert!(exceeds_limits_penalty(&in_range, (36.0, 96.0)) < 0.01);

        let out_of_range = vec![
            NoteMapping { note_index: 0, original_midi: 60.0, mapped_midi: 20.0, voice: 0, quality: 1.0 },
        ];
        assert!(exceeds_limits_penalty(&out_of_range, (36.0, 96.0)) > 0.5);
    }

    #[test]
    fn test_pitch_run_detection() {
        let notes = vec![
            make_note(60.0, 0.0, 0.1, 0.5), // C4
            make_note(62.0, 0.1, 0.1, 0.5), // D4
            make_note(64.0, 0.2, 0.1, 0.5), // E4
            make_note(65.0, 0.3, 0.1, 0.5), // F4
            make_note(67.0, 0.4, 0.1, 0.5), // G4
            make_note(80.0, 0.5, 0.5, 0.5), // E5 (jump, not a step)
        ];
        let runs = PitchRunMapper::detect_runs(&notes, 3.0);
        assert!(!runs.is_empty(), "Should detect at least one run");
        assert_eq!(runs[0].0, 0); // starts at index 0
        assert_eq!(runs[0].1, 4); // ends at index 4
    }

    #[test]
    fn test_block_voicing_mapper() {
        let notes = vec![
            make_note(60.0, 0.0, 1.0, 0.8), // C4
            make_note(64.0, 0.0, 1.0, 0.6), // E4
            make_note(67.0, 0.0, 1.0, 0.6), // G4
        ];
        let target = Voicing::from_pitches(&[48.0, 55.0, 60.0]); // C3, G3, C4
        let config = PitchMapperConfig::default();
        let mappings = BlockVoicingMapper::map(&notes, &target, &config);

        assert_eq!(mappings.len(), 3);
        // Each mapping should target one of the voicing pitches
        for m in &mappings {
            assert!(
                target.pitches.contains(&m.mapped_midi),
                "Mapped {} should be in target voicing",
                m.mapped_midi
            );
        }
    }

    #[test]
    fn test_approach_analysis() {
        let mappings = vec![
            NoteMapping { note_index: 0, original_midi: 60.0, mapped_midi: 60.0, voice: 0, quality: 1.0 },
            NoteMapping { note_index: 1, original_midi: 62.0, mapped_midi: 62.0, voice: 0, quality: 1.0 },
            NoteMapping { note_index: 2, original_midi: 72.0, mapped_midi: 72.0, voice: 0, quality: 1.0 },
        ];
        let approaches = PitchApproachMapper::analyze_approaches(&mappings, ApproachType::Slide);
        assert_eq!(approaches.len(), 2);
        // First transition (2 semitones) should be a slide
        assert_eq!(approaches[0].1, ApproachType::Slide);
        // Second transition (10 semitones) should be direct (too far for slide)
        assert_eq!(approaches[1].1, ApproachType::Direct);
    }
}
