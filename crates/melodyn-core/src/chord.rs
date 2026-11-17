//! Chord analysis system.
//!
//! Detects chords from sets of pitch classes, analyzes chord roots using
//! lowest-interval and bass-note heuristics, sequences chords over time,
//! and formats chord labels. Models MUChord, MUChordAnalyzer,
//! MUChordRootAnalyzer, MUChordTypeAnalyzer, and related classes.

use crate::note::DetectedNote;

// ---------------------------------------------------------------------------
// Pitch class utilities
// ---------------------------------------------------------------------------

/// A pitch class (0 = C, 1 = C#, ..., 11 = B).
pub type PitchClass = u8;

/// Extract the pitch class from a MIDI note number.
pub fn midi_to_pitch_class(midi: f64) -> PitchClass {
    ((midi.round() as i32).rem_euclid(12)) as PitchClass
}

/// Build a 12-element pitch class set from a collection of MIDI note numbers.
/// Each element is true if that pitch class is present.
pub fn pitch_class_set(midi_notes: &[f64]) -> [bool; 12] {
    let mut set = [false; 12];
    for &m in midi_notes {
        set[midi_to_pitch_class(m) as usize] = true;
    }
    set
}

/// Build a weighted pitch class distribution from notes (weight = amplitude * duration).
pub fn pitch_class_distribution(notes: &[DetectedNote]) -> [f64; 12] {
    let mut dist = [0.0f64; 12];
    for note in notes {
        let pc = midi_to_pitch_class(note.midi_note) as usize;
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
// Chord type
// ---------------------------------------------------------------------------

/// Recognized chord quality / type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChordType {
    Major,
    Minor,
    Diminished,
    Augmented,
    Sus2,
    Sus4,
    Dom7,
    Maj7,
    Min7,
    MinMaj7,
    Dim7,
    HalfDim7,
    Aug7,
    Dom9,
    Maj9,
    Min9,
    Add9,
    Dom11,
    Dom13,
    Power5,
    /// No recognized chord type.
    Unknown,
}

impl ChordType {
    /// Interval template as semitone offsets from root (not including root).
    /// Returns None for Unknown.
    pub fn intervals(self) -> Option<&'static [u8]> {
        match self {
            Self::Major => Some(&[4, 7]),
            Self::Minor => Some(&[3, 7]),
            Self::Diminished => Some(&[3, 6]),
            Self::Augmented => Some(&[4, 8]),
            Self::Sus2 => Some(&[2, 7]),
            Self::Sus4 => Some(&[5, 7]),
            Self::Dom7 => Some(&[4, 7, 10]),
            Self::Maj7 => Some(&[4, 7, 11]),
            Self::Min7 => Some(&[3, 7, 10]),
            Self::MinMaj7 => Some(&[3, 7, 11]),
            Self::Dim7 => Some(&[3, 6, 9]),
            Self::HalfDim7 => Some(&[3, 6, 10]),
            Self::Aug7 => Some(&[4, 8, 10]),
            Self::Dom9 => Some(&[4, 7, 10, 14]),
            Self::Maj9 => Some(&[4, 7, 11, 14]),
            Self::Min9 => Some(&[3, 7, 10, 14]),
            Self::Add9 => Some(&[4, 7, 14]),
            Self::Dom11 => Some(&[4, 7, 10, 14, 17]),
            Self::Dom13 => Some(&[4, 7, 10, 14, 17, 21]),
            Self::Power5 => Some(&[7]),
            Self::Unknown => None,
        }
    }

    /// All chord types in preference order (simpler chords first).
    pub const ALL: [ChordType; 20] = [
        Self::Major,
        Self::Minor,
        Self::Dom7,
        Self::Min7,
        Self::Maj7,
        Self::Diminished,
        Self::Augmented,
        Self::Sus2,
        Self::Sus4,
        Self::Power5,
        Self::Dim7,
        Self::HalfDim7,
        Self::MinMaj7,
        Self::Aug7,
        Self::Dom9,
        Self::Maj9,
        Self::Min9,
        Self::Add9,
        Self::Dom11,
        Self::Dom13,
    ];
}

// ---------------------------------------------------------------------------
// Chord naming / formatting
// ---------------------------------------------------------------------------

/// How to display accidentals in chord names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccidentalStyle {
    /// Sharp: C#, D#, F#, G#, A#
    Sharp,
    /// Flat: Db, Eb, Gb, Ab, Bb
    Flat,
}

/// How to display augmented/diminished symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AugDimStyle {
    /// Written out: "aug", "dim"
    Word,
    /// Symbols: "+", "o"
    Symbol,
}

/// How to display major 7th.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maj7Style {
    /// "maj7"
    Maj7,
    /// Triangle symbol: "M7"
    M7,
}

/// Chord formatting configuration.
#[derive(Debug, Clone)]
pub struct ChordFormatter {
    pub accidental_style: AccidentalStyle,
    pub aug_dim_style: AugDimStyle,
    pub maj7_style: Maj7Style,
}

impl Default for ChordFormatter {
    fn default() -> Self {
        Self {
            accidental_style: AccidentalStyle::Sharp,
            aug_dim_style: AugDimStyle::Word,
            maj7_style: Maj7Style::Maj7,
        }
    }
}

impl ChordFormatter {
    /// Format a root pitch class as a note name.
    pub fn format_root(&self, root: PitchClass) -> &'static str {
        match self.accidental_style {
            AccidentalStyle::Sharp => {
                const NAMES: [&str; 12] = [
                    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                ];
                NAMES[root as usize % 12]
            }
            AccidentalStyle::Flat => {
                const NAMES: [&str; 12] = [
                    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
                ];
                NAMES[root as usize % 12]
            }
        }
    }

    /// Format a chord type suffix.
    pub fn format_type(&self, chord_type: ChordType) -> &'static str {
        match chord_type {
            ChordType::Major => "",
            ChordType::Minor => "m",
            ChordType::Diminished => match self.aug_dim_style {
                AugDimStyle::Word => "dim",
                AugDimStyle::Symbol => "o",
            },
            ChordType::Augmented => match self.aug_dim_style {
                AugDimStyle::Word => "aug",
                AugDimStyle::Symbol => "+",
            },
            ChordType::Sus2 => "sus2",
            ChordType::Sus4 => "sus4",
            ChordType::Dom7 => "7",
            ChordType::Maj7 => match self.maj7_style {
                Maj7Style::Maj7 => "maj7",
                Maj7Style::M7 => "M7",
            },
            ChordType::Min7 => "m7",
            ChordType::MinMaj7 => "m(maj7)",
            ChordType::Dim7 => match self.aug_dim_style {
                AugDimStyle::Word => "dim7",
                AugDimStyle::Symbol => "o7",
            },
            ChordType::HalfDim7 => "m7b5",
            ChordType::Aug7 => match self.aug_dim_style {
                AugDimStyle::Word => "aug7",
                AugDimStyle::Symbol => "+7",
            },
            ChordType::Dom9 => "9",
            ChordType::Maj9 => "maj9",
            ChordType::Min9 => "m9",
            ChordType::Add9 => "add9",
            ChordType::Dom11 => "11",
            ChordType::Dom13 => "13",
            ChordType::Power5 => "5",
            ChordType::Unknown => "?",
        }
    }

    /// Format a complete chord description.
    pub fn format(&self, desc: &ChordDescription) -> String {
        let root_str = self.format_root(desc.root);
        let type_str = self.format_type(desc.chord_type);
        if let Some(bass) = desc.bass_note {
            if bass != desc.root {
                let bass_str = self.format_root(bass);
                return format!("{}{}/{}", root_str, type_str, bass_str);
            }
        }
        format!("{}{}", root_str, type_str)
    }
}

// ---------------------------------------------------------------------------
// Chord description and candidate
// ---------------------------------------------------------------------------

/// A fully described chord with root, type, and optional bass note.
#[derive(Debug, Clone, PartialEq)]
pub struct ChordDescription {
    /// Root pitch class.
    pub root: PitchClass,
    /// Chord quality / type.
    pub chord_type: ChordType,
    /// Bass note pitch class (for slash chords / inversions). None means root is bass.
    pub bass_note: Option<PitchClass>,
    /// Confidence / likelihood score (0.0 .. 1.0).
    pub confidence: f64,
}

impl ChordDescription {
    /// Pitch classes present in this chord (including root).
    pub fn pitch_classes(&self) -> Vec<PitchClass> {
        let mut pcs = vec![self.root];
        if let Some(intervals) = self.chord_type.intervals() {
            for &iv in intervals {
                pcs.push((self.root + iv) % 12);
            }
        }
        pcs
    }
}

/// A candidate chord with a score for ranking alternatives.
#[derive(Debug, Clone)]
pub struct CandidateChord {
    pub description: ChordDescription,
    /// How well this chord explains the observed pitch classes (higher = better).
    pub coverage_score: f64,
    /// Penalty for complexity (fewer notes in template = simpler = less penalty).
    pub complexity_penalty: f64,
    /// Combined score (coverage - penalty).
    pub total_score: f64,
}

// ---------------------------------------------------------------------------
// Chord type analyzer
// ---------------------------------------------------------------------------

/// Detects chord type from a set of pitch classes, trying all 12 roots.
pub struct ChordTypeAnalyzer;

impl ChordTypeAnalyzer {
    /// Given a pitch class set (boolean array), return all candidate chords
    /// ranked by likelihood.
    pub fn analyze(pc_set: &[bool; 12]) -> Vec<CandidateChord> {
        let active: Vec<PitchClass> = (0..12u8).filter(|&i| pc_set[i as usize]).collect();
        if active.is_empty() {
            return vec![];
        }

        let mut candidates = Vec::new();

        // Try every root and every chord type
        for root in 0u8..12 {
            for &ct in &ChordType::ALL {
                if let Some(intervals) = ct.intervals() {
                    let template: Vec<PitchClass> = std::iter::once(0u8)
                        .chain(intervals.iter().map(|&iv| iv % 12))
                        .collect();

                    // Count how many template notes are present
                    let template_count = template.len();
                    let mut matched = 0usize;
                    for &t in &template {
                        let pc = (root + t) % 12;
                        if pc_set[pc as usize] {
                            matched += 1;
                        }
                    }

                    // Count extra notes not in the template
                    let mut extra = 0usize;
                    for &a in &active {
                        let rel = (a + 12 - root) % 12;
                        if !template.contains(&rel) {
                            extra += 1;
                        }
                    }

                    // All template notes must be present for a full match
                    if matched == template_count {
                        let coverage = matched as f64 / active.len().max(1) as f64;
                        let complexity = (template_count as f64 - 3.0).max(0.0) * 0.05;
                        let extra_penalty = extra as f64 * 0.15;
                        let total = coverage - complexity - extra_penalty;

                        candidates.push(CandidateChord {
                            description: ChordDescription {
                                root,
                                chord_type: ct,
                                bass_note: None,
                                confidence: total.clamp(0.0, 1.0),
                            },
                            coverage_score: coverage,
                            complexity_penalty: complexity + extra_penalty,
                            total_score: total,
                        });
                    }
                }
            }
        }

        candidates.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }
}

// ---------------------------------------------------------------------------
// Chord root analyzer
// ---------------------------------------------------------------------------

/// Determines the most likely root using the lowest-interval heuristic
/// and bass note weighting.
pub struct ChordRootAnalyzer;

impl ChordRootAnalyzer {
    /// Given detected notes sounding simultaneously, determine the best root.
    /// Uses the "lowest interval" heuristic: the root of the lowest interval
    /// (perfect fifth, major/minor third) is likely the chord root.
    /// Bass note (lowest pitch) also gets extra weight.
    pub fn analyze_root(notes: &[DetectedNote]) -> Option<PitchClass> {
        if notes.is_empty() {
            return None;
        }

        let mut scores = [0.0f64; 12];

        // Bass note weighting: the lowest sounding note gets a root bonus.
        let bass_note = notes
            .iter()
            .min_by(|a, b| {
                a.midi_note
                    .partial_cmp(&b.midi_note)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let bass_pc = midi_to_pitch_class(bass_note.midi_note);
        scores[bass_pc as usize] += 2.0;

        // Lowest-interval heuristic: for each pair of notes, the root of
        // common intervals (P5, M3, m3) gets a vote.
        for i in 0..notes.len() {
            for j in (i + 1)..notes.len() {
                let pc_i = midi_to_pitch_class(notes[i].midi_note);
                let pc_j = midi_to_pitch_class(notes[j].midi_note);
                let interval = (pc_j as i8 - pc_i as i8).rem_euclid(12) as u8;

                match interval {
                    7 => {
                        // Perfect fifth: lower note is likely root
                        scores[pc_i as usize] += 1.5;
                    }
                    5 => {
                        // Perfect fourth (inverted fifth): upper note is root
                        scores[pc_j as usize] += 1.5;
                    }
                    4 => {
                        // Major third: lower note is likely root
                        scores[pc_i as usize] += 1.0;
                    }
                    3 => {
                        // Minor third: lower note is likely root
                        scores[pc_i as usize] += 1.0;
                    }
                    8 => {
                        // Minor sixth (inverted M3): upper note is root
                        scores[pc_j as usize] += 1.0;
                    }
                    9 => {
                        // Major sixth (inverted m3): upper note is root
                        scores[pc_j as usize] += 1.0;
                    }
                    _ => {}
                }
            }
        }

        // Amplitude weighting: louder notes are more likely roots.
        for note in notes {
            let pc = midi_to_pitch_class(note.midi_note) as usize;
            scores[pc] += note.amplitude * 0.5;
        }

        // Return the pitch class with the highest score.
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(pc, _)| pc as PitchClass)
    }
}

// ---------------------------------------------------------------------------
// Chord entry and sequence
// ---------------------------------------------------------------------------

/// A chord occurring at a specific time range.
#[derive(Debug, Clone)]
pub struct ChordEntry {
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
    /// Best chord description.
    pub chord: ChordDescription,
    /// Alternative chord interpretations, ranked by score.
    pub alternatives: Vec<CandidateChord>,
}

/// A sequence of chords over time.
#[derive(Debug, Clone)]
pub struct ChordSequence {
    pub entries: Vec<ChordEntry>,
}

impl ChordSequence {
    /// Find the chord active at a given time.
    pub fn chord_at(&self, time: f64) -> Option<&ChordEntry> {
        self.entries
            .iter()
            .find(|e| time >= e.start_time && time < e.end_time)
    }

    /// Total duration of the sequence.
    pub fn duration(&self) -> f64 {
        self.entries.last().map(|e| e.end_time).unwrap_or(0.0)
            - self.entries.first().map(|e| e.start_time).unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Chord analyzer (full pipeline)
// ---------------------------------------------------------------------------

/// Configuration for the chord analyzer.
#[derive(Debug, Clone)]
pub struct ChordAnalyzerConfig {
    /// Time resolution for chord analysis in seconds.
    pub time_resolution: f64,
    /// Minimum number of distinct pitch classes to attempt chord detection.
    pub min_pitch_classes: usize,
    /// Maximum number of alternative chords to keep per time slot.
    pub max_alternatives: usize,
}

impl Default for ChordAnalyzerConfig {
    fn default() -> Self {
        Self {
            time_resolution: 0.5,
            min_pitch_classes: 2,
            max_alternatives: 5,
        }
    }
}

/// Full chord analysis pipeline: segments notes into time windows,
/// detects chords, assigns roots, and builds a chord sequence.
pub struct ChordAnalyzer {
    config: ChordAnalyzerConfig,
}

impl ChordAnalyzer {
    pub fn new(config: ChordAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Analyze a list of detected notes and produce a chord sequence.
    pub fn analyze(&self, notes: &[DetectedNote]) -> ChordSequence {
        if notes.is_empty() {
            return ChordSequence {
                entries: Vec::new(),
            };
        }

        let start = notes
            .iter()
            .map(|n| n.start_time)
            .fold(f64::INFINITY, f64::min);
        let end = notes
            .iter()
            .map(|n| n.end_time())
            .fold(f64::NEG_INFINITY, f64::max);

        let mut entries = Vec::new();
        let mut t = start;

        while t < end {
            let window_end = (t + self.config.time_resolution).min(end);

            // Collect notes sounding in this window.
            let active: Vec<&DetectedNote> = notes
                .iter()
                .filter(|n| n.start_time < window_end && n.end_time() > t)
                .collect();

            let midi_notes: Vec<f64> = active.iter().map(|n| n.midi_note).collect();
            let pc_set = pitch_class_set(&midi_notes);
            let active_count = pc_set.iter().filter(|&&b| b).count();

            if active_count >= self.config.min_pitch_classes {
                let mut candidates = ChordTypeAnalyzer::analyze(&pc_set);

                // Refine root using bass-note heuristic.
                let active_owned: Vec<DetectedNote> = active.iter().map(|n| (*n).clone()).collect();
                if let Some(root) = ChordRootAnalyzer::analyze_root(&active_owned) {
                    // Boost candidates whose root matches.
                    for c in &mut candidates {
                        if c.description.root == root {
                            c.total_score += 0.3;
                            c.description.confidence = (c.description.confidence + 0.15).min(1.0);
                        }
                    }
                    candidates.sort_by(|a, b| {
                        b.total_score
                            .partial_cmp(&a.total_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                // Assign bass note for inversion detection.
                let bass_pc = active
                    .iter()
                    .min_by(|a, b| {
                        a.midi_note
                            .partial_cmp(&b.midi_note)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|n| midi_to_pitch_class(n.midi_note));

                if let Some(best) = candidates.first() {
                    let mut chord = best.description.clone();
                    if let Some(bp) = bass_pc {
                        if bp != chord.root {
                            chord.bass_note = Some(bp);
                        }
                    }
                    let alternatives: Vec<CandidateChord> = candidates
                        .iter()
                        .take(self.config.max_alternatives)
                        .cloned()
                        .collect();

                    entries.push(ChordEntry {
                        start_time: t,
                        end_time: window_end,
                        chord,
                        alternatives,
                    });
                }
            }

            t = window_end;
        }

        // Merge consecutive identical chords.
        let merged = Self::merge_consecutive(entries);
        ChordSequence { entries: merged }
    }

    /// Merge consecutive entries with the same root and type.
    fn merge_consecutive(entries: Vec<ChordEntry>) -> Vec<ChordEntry> {
        let mut merged: Vec<ChordEntry> = Vec::new();
        for entry in entries {
            if let Some(last) = merged.last_mut() {
                if last.chord.root == entry.chord.root
                    && last.chord.chord_type == entry.chord.chord_type
                {
                    last.end_time = entry.end_time;
                    continue;
                }
            }
            merged.push(entry);
        }
        merged
    }
}

// ---------------------------------------------------------------------------
// Chord harmony note (for voicing analysis)
// ---------------------------------------------------------------------------

/// A note within a chord context, tagged with its harmonic function.
#[derive(Debug, Clone)]
pub struct ChordHarmonyNote {
    /// The detected note.
    pub midi_note: f64,
    /// Pitch class.
    pub pitch_class: PitchClass,
    /// Scale degree relative to chord root (0 = root, 4 = major third, 7 = fifth, etc.).
    pub chord_degree: u8,
    /// Whether this is a chord tone (vs. a non-chord tone / passing tone).
    pub is_chord_tone: bool,
}

/// Tag notes with their harmonic function relative to a chord.
pub fn tag_harmony_notes(
    notes: &[DetectedNote],
    chord: &ChordDescription,
) -> Vec<ChordHarmonyNote> {
    let chord_pcs = chord.pitch_classes();
    notes
        .iter()
        .map(|n| {
            let pc = midi_to_pitch_class(n.midi_note);
            let degree = (pc + 12 - chord.root) % 12;
            let is_chord_tone = chord_pcs.contains(&pc);
            ChordHarmonyNote {
                midi_note: n.midi_note,
                pitch_class: pc,
                chord_degree: degree,
                is_chord_tone,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Chord parser (from string)
// ---------------------------------------------------------------------------

/// Parse a chord symbol string into a ChordDescription.
/// Supports formats like "C", "Am", "F#m7", "Bb/D", "Gdim", "Caug".
pub fn parse_chord(input: &str) -> Option<ChordDescription> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }

    let bytes = input.as_bytes();
    let mut pos = 0;

    // Parse root note
    let root_letter = match bytes.get(pos)? {
        b'C' => 0u8,
        b'D' => 2,
        b'E' => 4,
        b'F' => 5,
        b'G' => 7,
        b'A' => 9,
        b'B' => 11,
        _ => return None,
    };
    pos += 1;

    let root = if pos < bytes.len() && bytes[pos] == b'#' {
        pos += 1;
        (root_letter + 1) % 12
    } else if pos < bytes.len() && bytes[pos] == b'b' {
        pos += 1;
        (root_letter + 11) % 12
    } else {
        root_letter
    };

    let suffix = &input[pos..];

    // Check for slash chord
    let (type_part, bass_part) = if let Some(slash_pos) = suffix.find('/') {
        (&suffix[..slash_pos], Some(&suffix[slash_pos + 1..]))
    } else {
        (suffix, None)
    };

    // Parse chord type from suffix
    let chord_type = match type_part {
        "" | "maj" | "M" => ChordType::Major,
        "m" | "min" | "-" => ChordType::Minor,
        "dim" | "o" => ChordType::Diminished,
        "aug" | "+" => ChordType::Augmented,
        "sus2" => ChordType::Sus2,
        "sus4" | "sus" => ChordType::Sus4,
        "7" | "dom7" => ChordType::Dom7,
        "maj7" | "M7" => ChordType::Maj7,
        "m7" | "min7" | "-7" => ChordType::Min7,
        "m(maj7)" | "mM7" => ChordType::MinMaj7,
        "dim7" | "o7" => ChordType::Dim7,
        "m7b5" => ChordType::HalfDim7,
        "aug7" | "+7" => ChordType::Aug7,
        "9" => ChordType::Dom9,
        "maj9" | "M9" => ChordType::Maj9,
        "m9" | "min9" => ChordType::Min9,
        "add9" => ChordType::Add9,
        "11" => ChordType::Dom11,
        "13" => ChordType::Dom13,
        "5" => ChordType::Power5,
        _ => ChordType::Unknown,
    };

    // Parse bass note
    let bass_note = bass_part.and_then(|bp| {
        let bp = bp.trim();
        let bb = bp.as_bytes();
        if bb.is_empty() {
            return None;
        }
        let bl = match bb[0] {
            b'C' => 0u8,
            b'D' => 2,
            b'E' => 4,
            b'F' => 5,
            b'G' => 7,
            b'A' => 9,
            b'B' => 11,
            _ => return None,
        };
        let bass = if bb.len() > 1 && bb[1] == b'#' {
            (bl + 1) % 12
        } else if bb.len() > 1 && bb[1] == b'b' {
            (bl + 11) % 12
        } else {
            bl
        };
        Some(bass)
    });

    Some(ChordDescription {
        root,
        chord_type,
        bass_note,
        confidence: 1.0,
    })
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

    #[test]
    fn test_pitch_class_set_c_major() {
        // C4=60, E4=64, G4=67
        let set = pitch_class_set(&[60.0, 64.0, 67.0]);
        assert!(set[0]); // C
        assert!(set[4]); // E
        assert!(set[7]); // G
        assert!(!set[1]); // C# absent
    }

    #[test]
    fn test_chord_type_detection_major() {
        let mut pcs = [false; 12];
        pcs[0] = true; // C
        pcs[4] = true; // E
        pcs[7] = true; // G
        let candidates = ChordTypeAnalyzer::analyze(&pcs);
        assert!(!candidates.is_empty());
        let best = &candidates[0];
        assert_eq!(best.description.root, 0);
        assert_eq!(best.description.chord_type, ChordType::Major);
    }

    #[test]
    fn test_chord_type_detection_minor() {
        let mut pcs = [false; 12];
        pcs[9] = true; // A
        pcs[0] = true; // C
        pcs[4] = true; // E
        let candidates = ChordTypeAnalyzer::analyze(&pcs);
        assert!(!candidates.is_empty());
        // Should find Am among the top candidates
        let am = candidates
            .iter()
            .find(|c| c.description.root == 9 && c.description.chord_type == ChordType::Minor);
        assert!(am.is_some(), "Should detect A minor");
    }

    #[test]
    fn test_chord_type_detection_dom7() {
        let mut pcs = [false; 12];
        pcs[7] = true; // G
        pcs[11] = true; // B
        pcs[2] = true; // D
        pcs[5] = true; // F
        let candidates = ChordTypeAnalyzer::analyze(&pcs);
        let g7 = candidates
            .iter()
            .find(|c| c.description.root == 7 && c.description.chord_type == ChordType::Dom7);
        assert!(g7.is_some(), "Should detect G7");
    }

    #[test]
    fn test_root_analysis_bass_weight() {
        // C major chord: C3, E4, G4 -- C is lowest, should be root
        let notes = vec![
            make_note(48.0, 0.0, 1.0, 0.8), // C3
            make_note(64.0, 0.0, 1.0, 0.5), // E4
            make_note(67.0, 0.0, 1.0, 0.5), // G4
        ];
        let root = ChordRootAnalyzer::analyze_root(&notes);
        assert_eq!(root, Some(0)); // C
    }

    #[test]
    fn test_chord_formatter() {
        let fmt = ChordFormatter::default();
        let desc = ChordDescription {
            root: 0,
            chord_type: ChordType::Major,
            bass_note: None,
            confidence: 1.0,
        };
        assert_eq!(fmt.format(&desc), "C");

        let desc2 = ChordDescription {
            root: 9,
            chord_type: ChordType::Min7,
            bass_note: None,
            confidence: 1.0,
        };
        assert_eq!(fmt.format(&desc2), "Am7");

        // Slash chord
        let desc3 = ChordDescription {
            root: 0,
            chord_type: ChordType::Major,
            bass_note: Some(4),
            confidence: 1.0,
        };
        assert_eq!(fmt.format(&desc3), "C/E");
    }

    #[test]
    fn test_parse_chord() {
        let c = parse_chord("C").unwrap();
        assert_eq!(c.root, 0);
        assert_eq!(c.chord_type, ChordType::Major);

        let am7 = parse_chord("Am7").unwrap();
        assert_eq!(am7.root, 9);
        assert_eq!(am7.chord_type, ChordType::Min7);

        let fsharp = parse_chord("F#m").unwrap();
        assert_eq!(fsharp.root, 6);
        assert_eq!(fsharp.chord_type, ChordType::Minor);

        let slash = parse_chord("C/E").unwrap();
        assert_eq!(slash.root, 0);
        assert_eq!(slash.bass_note, Some(4));
    }

    #[test]
    fn test_chord_analyzer_sequence() {
        // Two chords: C major (notes at t=0..1) then G major (notes at t=1..2)
        let notes = vec![
            make_note(60.0, 0.0, 1.0, 0.8), // C4
            make_note(64.0, 0.0, 1.0, 0.6), // E4
            make_note(67.0, 0.0, 1.0, 0.6), // G4
            make_note(67.0, 1.0, 1.0, 0.8), // G4
            make_note(71.0, 1.0, 1.0, 0.6), // B4
            make_note(74.0, 1.0, 1.0, 0.6), // D5
        ];
        let analyzer = ChordAnalyzer::new(ChordAnalyzerConfig {
            time_resolution: 1.0,
            ..Default::default()
        });
        let seq = analyzer.analyze(&notes);
        assert!(seq.entries.len() >= 2, "Should detect at least 2 chords");
    }

    #[test]
    fn test_tag_harmony_notes() {
        let chord = ChordDescription {
            root: 0,
            chord_type: ChordType::Major,
            bass_note: None,
            confidence: 1.0,
        };
        let notes = vec![
            make_note(60.0, 0.0, 1.0, 0.8), // C4 -- root
            make_note(64.0, 0.0, 1.0, 0.5), // E4 -- third
            make_note(66.0, 0.0, 1.0, 0.3), // F#4 -- non-chord tone
        ];
        let tagged = tag_harmony_notes(&notes, &chord);
        assert!(tagged[0].is_chord_tone);
        assert!(tagged[1].is_chord_tone);
        assert!(!tagged[2].is_chord_tone);
        assert_eq!(tagged[0].chord_degree, 0); // root
        assert_eq!(tagged[1].chord_degree, 4); // major third
    }

    #[test]
    fn test_chord_description_pitch_classes() {
        let desc = ChordDescription {
            root: 0,
            chord_type: ChordType::Dom7,
            bass_note: None,
            confidence: 1.0,
        };
        let pcs = desc.pitch_classes();
        assert!(pcs.contains(&0)); // C
        assert!(pcs.contains(&4)); // E
        assert!(pcs.contains(&7)); // G
        assert!(pcs.contains(&10)); // Bb
    }
}
