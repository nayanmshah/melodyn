//! Document model: Document, Track, Element, Composition, Selection, Undo/Redo.
//!
//! Based on MUDocument, MUComposition, MUTrack, MUTrackSet, MUElement,
//! MUSequence, MUModelObject, MUEditorSelection. Provides the top-level
//! data model for the Melodyn editor with undo/redo via command pattern.

use crate::note::DetectedNote;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Identifiers and common types
// ---------------------------------------------------------------------------

/// Unique identifier for model objects.
pub type ObjectId = u64;

/// Color for track display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Default for TrackColor {
    fn default() -> Self {
        Self {
            r: 0.5,
            g: 0.7,
            b: 1.0,
        }
    }
}

/// ID generator for model objects.
#[derive(Debug)]
struct IdGen {
    next: ObjectId,
}

impl IdGen {
    fn new() -> Self {
        Self { next: 1 }
    }
    fn next(&mut self) -> ObjectId {
        let id = self.next;
        self.next += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// Composition (musical structure)
// ---------------------------------------------------------------------------

/// Musical time signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSignature {
    pub numerator: u32,
    pub denominator: u32,
}

impl Default for TimeSignature {
    fn default() -> Self {
        Self {
            numerator: 4,
            denominator: 4,
        }
    }
}

/// Key signature for the composition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Key {
    /// Root note (0=C, 1=C#, ..., 11=B).
    pub root: u8,
    /// Whether the key is minor (false = major).
    pub minor: bool,
}

/// Tempo change at a specific position.
#[derive(Debug, Clone)]
pub struct TempoMark {
    /// Position in beats from the start.
    pub beat_position: f64,
    /// Tempo in BPM.
    pub bpm: f64,
}

/// Musical structure of the document.
#[derive(Debug, Clone)]
pub struct Composition {
    /// Initial tempo in BPM.
    pub tempo: f64,
    /// Time signature.
    pub time_signature: TimeSignature,
    /// Key signature.
    pub key: Key,
    /// Tempo changes throughout the piece.
    pub tempo_changes: Vec<TempoMark>,
    /// Total duration in seconds (from audio content).
    pub duration: f64,
    /// Sample rate.
    pub sample_rate: f64,
}

impl Default for Composition {
    fn default() -> Self {
        Self {
            tempo: 120.0,
            time_signature: TimeSignature::default(),
            key: Key::default(),
            tempo_changes: Vec::new(),
            duration: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl Composition {
    /// Convert beat position to seconds using the tempo map.
    pub fn beats_to_seconds(&self, beats: f64) -> f64 {
        if self.tempo_changes.is_empty() {
            return beats * 60.0 / self.tempo;
        }
        let mut seconds = 0.0;
        let mut prev_beat = 0.0;
        let mut prev_bpm = self.tempo;

        for mark in &self.tempo_changes {
            if mark.beat_position >= beats {
                break;
            }
            let segment_beats = mark.beat_position - prev_beat;
            seconds += segment_beats * 60.0 / prev_bpm;
            prev_beat = mark.beat_position;
            prev_bpm = mark.bpm;
        }
        seconds += (beats - prev_beat) * 60.0 / prev_bpm;
        seconds
    }

    /// Convert seconds to beat position.
    pub fn seconds_to_beats(&self, seconds: f64) -> f64 {
        if self.tempo_changes.is_empty() {
            return seconds * self.tempo / 60.0;
        }
        let mut remaining = seconds;
        let mut beats = 0.0;
        let mut prev_beat = 0.0;
        let mut prev_bpm = self.tempo;

        for mark in &self.tempo_changes {
            let segment_beats = mark.beat_position - prev_beat;
            let segment_seconds = segment_beats * 60.0 / prev_bpm;
            if remaining <= segment_seconds {
                return beats + remaining * prev_bpm / 60.0;
            }
            remaining -= segment_seconds;
            beats += segment_beats;
            prev_beat = mark.beat_position;
            prev_bpm = mark.bpm;
        }
        beats + remaining * prev_bpm / 60.0
    }

    /// Duration of one measure in seconds at the initial tempo.
    pub fn measure_duration(&self) -> f64 {
        let beats_per_measure =
            self.time_signature.numerator as f64 * 4.0 / self.time_signature.denominator as f64;
        beats_per_measure * 60.0 / self.tempo
    }
}

// ---------------------------------------------------------------------------
// Element (analyzed audio region on a track)
// ---------------------------------------------------------------------------

/// An analyzed audio region placed on a track.
#[derive(Debug, Clone)]
pub struct Element {
    pub id: ObjectId,
    /// Name/label for this element.
    pub name: String,
    /// Time offset on the track (seconds).
    pub time_offset: f64,
    /// Original audio duration (seconds).
    pub duration: f64,
    /// Source file path (if loaded from file).
    pub source_path: Option<String>,
    /// Detected notes from analysis.
    pub notes: Vec<DetectedNote>,
    /// Whether this element is muted.
    pub muted: bool,
    /// Gain applied to this element (linear).
    pub gain: f64,
}

impl Element {
    fn new(id: ObjectId, name: String) -> Self {
        Self {
            id,
            name,
            time_offset: 0.0,
            duration: 0.0,
            source_path: None,
            notes: Vec::new(),
            muted: false,
            gain: 1.0,
        }
    }

    /// End time of this element on the track.
    pub fn end_time(&self) -> f64 {
        self.time_offset + self.duration
    }

    /// Whether this element overlaps a time range.
    pub fn overlaps(&self, start: f64, end: f64) -> bool {
        self.time_offset < end && self.end_time() > start
    }

    /// Find notes within a time range (relative to the element's start).
    pub fn notes_in_range(&self, start: f64, end: f64) -> Vec<&DetectedNote> {
        self.notes
            .iter()
            .filter(|n| {
                let abs_start = self.time_offset + n.start_time;
                let abs_end = abs_start + n.duration;
                abs_start < end && abs_end > start
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

/// A track in the document, containing elements.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: ObjectId,
    pub name: String,
    pub color: TrackColor,
    pub elements: Vec<Element>,
    /// Whether the track is soloed.
    pub solo: bool,
    /// Whether the track is muted.
    pub mute: bool,
    /// Track volume (linear, 0.0 to 2.0, default 1.0).
    pub volume: f64,
    /// Track pan (-1.0 left to 1.0 right).
    pub pan: f64,
    /// Track height in the UI (pixels).
    pub height: f32,
}

impl Track {
    fn new(id: ObjectId, name: String) -> Self {
        Self {
            id,
            name,
            color: TrackColor::default(),
            elements: Vec::new(),
            solo: false,
            mute: false,
            volume: 1.0,
            pan: 0.0,
            height: 100.0,
        }
    }

    /// Add an element to this track.
    pub fn add_element(&mut self, element: Element) {
        self.elements.push(element);
        self.elements
            .sort_by(|a, b| a.time_offset.partial_cmp(&b.time_offset).unwrap());
    }

    /// Remove an element by ID.
    pub fn remove_element(&mut self, element_id: ObjectId) -> Option<Element> {
        if let Some(pos) = self.elements.iter().position(|e| e.id == element_id) {
            Some(self.elements.remove(pos))
        } else {
            None
        }
    }

    /// Get an element by ID.
    pub fn element(&self, id: ObjectId) -> Option<&Element> {
        self.elements.iter().find(|e| e.id == id)
    }

    /// Get a mutable element by ID.
    pub fn element_mut(&mut self, id: ObjectId) -> Option<&mut Element> {
        self.elements.iter_mut().find(|e| e.id == id)
    }

    /// Track duration (end time of the last element).
    pub fn duration(&self) -> f64 {
        self.elements
            .iter()
            .map(|e| e.end_time())
            .fold(0.0, f64::max)
    }

    /// Whether this track should produce audio (not muted, or soloed).
    pub fn is_audible(&self, any_solo: bool) -> bool {
        if any_solo {
            self.solo
        } else {
            !self.mute
        }
    }

    /// Find all elements overlapping a time range.
    pub fn elements_in_range(&self, start: f64, end: f64) -> Vec<&Element> {
        self.elements
            .iter()
            .filter(|e| e.overlaps(start, end))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Multi-element selection support.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    /// Selected track IDs.
    pub track_ids: HashSet<ObjectId>,
    /// Selected element IDs.
    pub element_ids: HashSet<ObjectId>,
    /// Selected note indices (element_id, note_index).
    pub note_selections: HashSet<(ObjectId, usize)>,
    /// Time range selection (start, end) in seconds.
    pub time_range: Option<(f64, f64)>,
}

impl Selection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.track_ids.clear();
        self.element_ids.clear();
        self.note_selections.clear();
        self.time_range = None;
    }

    pub fn select_track(&mut self, id: ObjectId) {
        self.track_ids.insert(id);
    }

    pub fn deselect_track(&mut self, id: ObjectId) {
        self.track_ids.remove(&id);
    }

    pub fn select_element(&mut self, id: ObjectId) {
        self.element_ids.insert(id);
    }

    pub fn deselect_element(&mut self, id: ObjectId) {
        self.element_ids.remove(&id);
    }

    pub fn select_note(&mut self, element_id: ObjectId, note_index: usize) {
        self.note_selections.insert((element_id, note_index));
    }

    pub fn deselect_note(&mut self, element_id: ObjectId, note_index: usize) {
        self.note_selections.remove(&(element_id, note_index));
    }

    pub fn set_time_range(&mut self, start: f64, end: f64) {
        self.time_range = Some((start, end));
    }

    pub fn is_empty(&self) -> bool {
        self.track_ids.is_empty()
            && self.element_ids.is_empty()
            && self.note_selections.is_empty()
            && self.time_range.is_none()
    }

    pub fn has_tracks(&self) -> bool {
        !self.track_ids.is_empty()
    }

    pub fn has_elements(&self) -> bool {
        !self.element_ids.is_empty()
    }

    pub fn has_notes(&self) -> bool {
        !self.note_selections.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Undo/Redo command pattern
// ---------------------------------------------------------------------------

/// A reversible edit command.
pub trait EditCommand: std::fmt::Debug {
    /// Execute the command, returning true on success.
    fn execute(&mut self, doc: &mut Document) -> bool;
    /// Undo the command.
    fn undo(&mut self, doc: &mut Document);
    /// Description of the command for display.
    fn description(&self) -> &str;
}

/// Command: rename a track.
#[derive(Debug)]
pub struct RenameTrackCommand {
    track_id: ObjectId,
    new_name: String,
    old_name: String,
}

impl RenameTrackCommand {
    pub fn new(track_id: ObjectId, new_name: String) -> Self {
        Self {
            track_id,
            new_name,
            old_name: String::new(),
        }
    }
}

impl EditCommand for RenameTrackCommand {
    fn execute(&mut self, doc: &mut Document) -> bool {
        if let Some(track) = doc.track_mut(self.track_id) {
            self.old_name = track.name.clone();
            track.name = self.new_name.clone();
            true
        } else {
            false
        }
    }
    fn undo(&mut self, doc: &mut Document) {
        if let Some(track) = doc.track_mut(self.track_id) {
            track.name = self.old_name.clone();
        }
    }
    fn description(&self) -> &str {
        "Rename Track"
    }
}

/// Command: move an element to a new time offset.
#[derive(Debug)]
pub struct MoveElementCommand {
    track_id: ObjectId,
    element_id: ObjectId,
    new_offset: f64,
    old_offset: f64,
}

impl MoveElementCommand {
    pub fn new(track_id: ObjectId, element_id: ObjectId, new_offset: f64) -> Self {
        Self {
            track_id,
            element_id,
            new_offset,
            old_offset: 0.0,
        }
    }
}

impl EditCommand for MoveElementCommand {
    fn execute(&mut self, doc: &mut Document) -> bool {
        if let Some(track) = doc.track_mut(self.track_id) {
            if let Some(elem) = track.element_mut(self.element_id) {
                self.old_offset = elem.time_offset;
                elem.time_offset = self.new_offset;
                return true;
            }
        }
        false
    }
    fn undo(&mut self, doc: &mut Document) {
        if let Some(track) = doc.track_mut(self.track_id) {
            if let Some(elem) = track.element_mut(self.element_id) {
                elem.time_offset = self.old_offset;
            }
        }
    }
    fn description(&self) -> &str {
        "Move Element"
    }
}

/// Command: set track volume.
#[derive(Debug)]
pub struct SetTrackVolumeCommand {
    track_id: ObjectId,
    new_volume: f64,
    old_volume: f64,
}

impl SetTrackVolumeCommand {
    pub fn new(track_id: ObjectId, new_volume: f64) -> Self {
        Self {
            track_id,
            new_volume,
            old_volume: 1.0,
        }
    }
}

impl EditCommand for SetTrackVolumeCommand {
    fn execute(&mut self, doc: &mut Document) -> bool {
        if let Some(track) = doc.track_mut(self.track_id) {
            self.old_volume = track.volume;
            track.volume = self.new_volume;
            true
        } else {
            false
        }
    }
    fn undo(&mut self, doc: &mut Document) {
        if let Some(track) = doc.track_mut(self.track_id) {
            track.volume = self.old_volume;
        }
    }
    fn description(&self) -> &str {
        "Set Track Volume"
    }
}

/// Undo/redo history manager.
#[derive(Debug)]
pub struct UndoManager {
    undo_stack: Vec<Box<dyn EditCommand>>,
    redo_stack: Vec<Box<dyn EditCommand>>,
    max_history: usize,
}

impl UndoManager {
    pub fn new(max_history: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_history,
        }
    }

    /// Execute a command and push it onto the undo stack.
    pub fn execute(&mut self, mut cmd: Box<dyn EditCommand>, doc: &mut Document) -> bool {
        if cmd.execute(doc) {
            self.redo_stack.clear();
            self.undo_stack.push(cmd);
            if self.undo_stack.len() > self.max_history {
                self.undo_stack.remove(0);
            }
            true
        } else {
            false
        }
    }

    /// Undo the last command.
    pub fn undo(&mut self, doc: &mut Document) -> bool {
        if let Some(mut cmd) = self.undo_stack.pop() {
            cmd.undo(doc);
            self.redo_stack.push(cmd);
            true
        } else {
            false
        }
    }

    /// Redo the last undone command.
    pub fn redo(&mut self, doc: &mut Document) -> bool {
        if let Some(mut cmd) = self.redo_stack.pop() {
            if cmd.execute(doc) {
                self.undo_stack.push(cmd);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Whether undo is available.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Whether redo is available.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Description of the next undo action.
    pub fn undo_description(&self) -> Option<&str> {
        self.undo_stack.last().map(|c| c.description())
    }

    /// Description of the next redo action.
    pub fn redo_description(&self) -> Option<&str> {
        self.redo_stack.last().map(|c| c.description())
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// Top-level document container.
#[derive(Debug)]
pub struct Document {
    pub name: String,
    pub tracks: Vec<Track>,
    pub composition: Composition,
    pub selection: Selection,
    id_gen: IdGen,
    /// Whether the document has unsaved changes.
    pub modified: bool,
}

impl Document {
    pub fn new(name: String) -> Self {
        Self {
            name,
            tracks: Vec::new(),
            composition: Composition::default(),
            selection: Selection::new(),
            id_gen: IdGen::new(),
            modified: false,
        }
    }

    /// Create a new track and add it to the document.
    pub fn add_track(&mut self, name: String) -> ObjectId {
        let id = self.id_gen.next();
        self.tracks.push(Track::new(id, name));
        self.modified = true;
        id
    }

    /// Remove a track by ID.
    pub fn remove_track(&mut self, id: ObjectId) -> Option<Track> {
        if let Some(pos) = self.tracks.iter().position(|t| t.id == id) {
            self.modified = true;
            self.selection.deselect_track(id);
            Some(self.tracks.remove(pos))
        } else {
            None
        }
    }

    /// Get a track by ID.
    pub fn track(&self, id: ObjectId) -> Option<&Track> {
        self.tracks.iter().find(|t| t.id == id)
    }

    /// Get a mutable track by ID.
    pub fn track_mut(&mut self, id: ObjectId) -> Option<&mut Track> {
        self.tracks.iter_mut().find(|t| t.id == id)
    }

    /// Create a new element on a track.
    pub fn add_element(&mut self, track_id: ObjectId, name: String) -> Option<ObjectId> {
        let elem_id = self.id_gen.next();
        if let Some(track) = self.track_mut(track_id) {
            let element = Element::new(elem_id, name);
            track.add_element(element);
            self.modified = true;
            Some(elem_id)
        } else {
            None
        }
    }

    /// Remove an element from a track.
    pub fn remove_element(
        &mut self,
        track_id: ObjectId,
        element_id: ObjectId,
    ) -> Option<Element> {
        self.selection.deselect_element(element_id);
        self.modified = true;
        if let Some(track) = self.track_mut(track_id) {
            track.remove_element(element_id)
        } else {
            None
        }
    }

    /// Total document duration (max of all track durations).
    pub fn duration(&self) -> f64 {
        self.tracks
            .iter()
            .map(|t| t.duration())
            .fold(self.composition.duration, f64::max)
    }

    /// Whether any track is soloed.
    pub fn any_solo(&self) -> bool {
        self.tracks.iter().any(|t| t.solo)
    }

    /// Get all audible tracks.
    pub fn audible_tracks(&self) -> Vec<&Track> {
        let any_solo = self.any_solo();
        self.tracks
            .iter()
            .filter(|t| t.is_audible(any_solo))
            .collect()
    }

    /// Move a track to a new position in the list.
    pub fn reorder_track(&mut self, track_id: ObjectId, new_index: usize) {
        if let Some(pos) = self.tracks.iter().position(|t| t.id == track_id) {
            let track = self.tracks.remove(pos);
            let insert_at = new_index.min(self.tracks.len());
            self.tracks.insert(insert_at, track);
            self.modified = true;
        }
    }

    /// Find an element across all tracks by ID.
    pub fn find_element(&self, element_id: ObjectId) -> Option<(&Track, &Element)> {
        for track in &self.tracks {
            if let Some(elem) = track.element(element_id) {
                return Some((track, elem));
            }
        }
        None
    }

    /// Total number of notes across all tracks and elements.
    pub fn total_notes(&self) -> usize {
        self.tracks
            .iter()
            .flat_map(|t| t.elements.iter())
            .map(|e| e.notes.len())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_note(start: f64, freq: f64) -> DetectedNote {
        DetectedNote {
            frequency: freq,
            midi_note: 69.0,
            start_time: start,
            duration: 0.5,
            start_sample: 0,
            end_sample: 22050,
            amplitude: 0.5,
            peak_amplitude: 0.5,
            confidence: 0.9,
            pitch_contour: vec![freq],
            amplitude_contour: vec![0.5],
            vibrato_rate: 0.0,
            vibrato_depth: 0.0,
            attack_time: 0.0,
            release_time: 0.0,
        }
    }

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Test".to_string());
        assert_eq!(doc.name, "Test");
        assert!(doc.tracks.is_empty());
        assert!(!doc.modified);
    }

    #[test]
    fn test_add_remove_track() {
        let mut doc = Document::new("Test".to_string());
        let id = doc.add_track("Track 1".to_string());
        assert_eq!(doc.tracks.len(), 1);
        assert!(doc.modified);

        let removed = doc.remove_track(id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "Track 1");
        assert!(doc.tracks.is_empty());
    }

    #[test]
    fn test_add_element() {
        let mut doc = Document::new("Test".to_string());
        let track_id = doc.add_track("Track 1".to_string());
        let elem_id = doc.add_element(track_id, "Region 1".to_string()).unwrap();
        assert_eq!(doc.tracks[0].elements.len(), 1);
        assert_eq!(doc.tracks[0].elements[0].id, elem_id);
    }

    #[test]
    fn test_element_time_overlap() {
        let mut elem = Element::new(1, "test".to_string());
        elem.time_offset = 1.0;
        elem.duration = 2.0;

        assert!(elem.overlaps(0.5, 1.5));
        assert!(elem.overlaps(2.0, 4.0));
        assert!(!elem.overlaps(3.5, 4.0));
        assert!(!elem.overlaps(0.0, 0.5));
    }

    #[test]
    fn test_track_duration() {
        let mut doc = Document::new("Test".to_string());
        let track_id = doc.add_track("Track 1".to_string());
        let elem_id = doc.add_element(track_id, "Region".to_string()).unwrap();
        let track = doc.track_mut(track_id).unwrap();
        let elem = track.element_mut(elem_id).unwrap();
        elem.time_offset = 1.0;
        elem.duration = 3.0;
        assert!((track.duration() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_track_solo_mute() {
        let mut doc = Document::new("Test".to_string());
        let t1 = doc.add_track("Track 1".to_string());
        let t2 = doc.add_track("Track 2".to_string());

        // No solo: both audible
        assert_eq!(doc.audible_tracks().len(), 2);

        // Solo track 1: only track 1 audible
        doc.track_mut(t1).unwrap().solo = true;
        assert_eq!(doc.audible_tracks().len(), 1);
        assert_eq!(doc.audible_tracks()[0].id, t1);

        // Unsolo, mute track 2
        doc.track_mut(t1).unwrap().solo = false;
        doc.track_mut(t2).unwrap().mute = true;
        assert_eq!(doc.audible_tracks().len(), 1);
        assert_eq!(doc.audible_tracks()[0].id, t1);
    }

    #[test]
    fn test_selection() {
        let mut sel = Selection::new();
        assert!(sel.is_empty());

        sel.select_track(1);
        sel.select_element(10);
        sel.select_note(10, 0);
        assert!(sel.has_tracks());
        assert!(sel.has_elements());
        assert!(sel.has_notes());

        sel.deselect_track(1);
        assert!(!sel.has_tracks());

        sel.clear();
        assert!(sel.is_empty());
    }

    #[test]
    fn test_selection_time_range() {
        let mut sel = Selection::new();
        sel.set_time_range(1.0, 3.0);
        assert!(!sel.is_empty());
        assert_eq!(sel.time_range, Some((1.0, 3.0)));
    }

    #[test]
    fn test_composition_tempo() {
        let comp = Composition::default();
        // 120 BPM: 1 beat = 0.5 seconds
        assert!((comp.beats_to_seconds(1.0) - 0.5).abs() < 1e-10);
        assert!((comp.seconds_to_beats(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_composition_tempo_change() {
        let mut comp = Composition {
            tempo: 120.0,
            ..Default::default()
        };
        comp.tempo_changes.push(TempoMark {
            beat_position: 4.0,
            bpm: 60.0,
        });
        // First 4 beats at 120 BPM = 2 seconds
        assert!((comp.beats_to_seconds(4.0) - 2.0).abs() < 1e-10);
        // Next beat at 60 BPM = 1 second
        assert!((comp.beats_to_seconds(5.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_composition_measure_duration() {
        let comp = Composition::default(); // 4/4 at 120 BPM
        // 4 beats * 0.5 sec/beat = 2 seconds
        assert!((comp.measure_duration() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_undo_redo_rename() {
        let mut doc = Document::new("Test".to_string());
        let track_id = doc.add_track("Original".to_string());
        let mut undo = UndoManager::new(100);

        let cmd = Box::new(RenameTrackCommand::new(track_id, "Renamed".to_string()));
        undo.execute(cmd, &mut doc);
        assert_eq!(doc.track(track_id).unwrap().name, "Renamed");
        assert!(undo.can_undo());

        undo.undo(&mut doc);
        assert_eq!(doc.track(track_id).unwrap().name, "Original");
        assert!(undo.can_redo());

        undo.redo(&mut doc);
        assert_eq!(doc.track(track_id).unwrap().name, "Renamed");
    }

    #[test]
    fn test_undo_redo_move_element() {
        let mut doc = Document::new("Test".to_string());
        let track_id = doc.add_track("Track".to_string());
        let elem_id = doc.add_element(track_id, "Region".to_string()).unwrap();
        let mut undo = UndoManager::new(100);

        let cmd = Box::new(MoveElementCommand::new(track_id, elem_id, 2.0));
        undo.execute(cmd, &mut doc);
        assert!(
            (doc.track(track_id).unwrap().element(elem_id).unwrap().time_offset - 2.0).abs()
                < 1e-10
        );

        undo.undo(&mut doc);
        assert!(
            (doc.track(track_id).unwrap().element(elem_id).unwrap().time_offset - 0.0).abs()
                < 1e-10
        );
    }

    #[test]
    fn test_undo_clears_redo() {
        let mut doc = Document::new("Test".to_string());
        let track_id = doc.add_track("Track".to_string());
        let mut undo = UndoManager::new(100);

        let cmd1 = Box::new(RenameTrackCommand::new(track_id, "A".to_string()));
        undo.execute(cmd1, &mut doc);
        undo.undo(&mut doc);
        assert!(undo.can_redo());

        // New command should clear redo stack
        let cmd2 = Box::new(RenameTrackCommand::new(track_id, "B".to_string()));
        undo.execute(cmd2, &mut doc);
        assert!(!undo.can_redo());
    }

    #[test]
    fn test_undo_descriptions() {
        let mut doc = Document::new("Test".to_string());
        let track_id = doc.add_track("Track".to_string());
        let mut undo = UndoManager::new(100);

        assert!(undo.undo_description().is_none());
        let cmd = Box::new(RenameTrackCommand::new(track_id, "X".to_string()));
        undo.execute(cmd, &mut doc);
        assert_eq!(undo.undo_description(), Some("Rename Track"));
    }

    #[test]
    fn test_find_element() {
        let mut doc = Document::new("Test".to_string());
        let t1 = doc.add_track("T1".to_string());
        let _t2 = doc.add_track("T2".to_string());
        let e1 = doc.add_element(t1, "E1".to_string()).unwrap();

        let (track, elem) = doc.find_element(e1).unwrap();
        assert_eq!(track.id, t1);
        assert_eq!(elem.id, e1);
        assert!(doc.find_element(999).is_none());
    }

    #[test]
    fn test_reorder_track() {
        let mut doc = Document::new("Test".to_string());
        let t1 = doc.add_track("T1".to_string());
        let t2 = doc.add_track("T2".to_string());
        let t3 = doc.add_track("T3".to_string());

        doc.reorder_track(t3, 0);
        assert_eq!(doc.tracks[0].id, t3);
        assert_eq!(doc.tracks[1].id, t1);
        assert_eq!(doc.tracks[2].id, t2);
    }

    #[test]
    fn test_total_notes() {
        let mut doc = Document::new("Test".to_string());
        let track_id = doc.add_track("Track".to_string());
        let elem_id = doc.add_element(track_id, "Region".to_string()).unwrap();
        let track = doc.track_mut(track_id).unwrap();
        let elem = track.element_mut(elem_id).unwrap();
        elem.notes.push(make_test_note(0.0, 440.0));
        elem.notes.push(make_test_note(0.5, 330.0));
        assert_eq!(doc.total_notes(), 2);
    }

    #[test]
    fn test_element_notes_in_range() {
        let mut elem = Element::new(1, "test".to_string());
        elem.time_offset = 1.0;
        elem.notes.push(make_test_note(0.0, 440.0)); // absolute: 1.0 to 1.5
        elem.notes.push(make_test_note(1.0, 330.0)); // absolute: 2.0 to 2.5

        let found = elem.notes_in_range(0.5, 1.5);
        assert_eq!(found.len(), 1);
        assert!((found[0].frequency - 440.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_track_volume_undo() {
        let mut doc = Document::new("Test".to_string());
        let track_id = doc.add_track("Track".to_string());
        let mut undo = UndoManager::new(100);

        let cmd = Box::new(SetTrackVolumeCommand::new(track_id, 0.5));
        undo.execute(cmd, &mut doc);
        assert!((doc.track(track_id).unwrap().volume - 0.5).abs() < 1e-10);

        undo.undo(&mut doc);
        assert!((doc.track(track_id).unwrap().volume - 1.0).abs() < 1e-10);
    }
}
