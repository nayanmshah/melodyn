//! Clipboard support for the Melodyn editor.
//!
//! Provides internal clipboard operations for notes: copy, cut, paste,
//! and duplicate. Notes are stored as `EditableNote` values with their
//! time positions relative to the earliest copied note.

use crate::app::{AppState, EditableNote};

// ---------------------------------------------------------------------------
// Clipboard state
// ---------------------------------------------------------------------------

/// Internal clipboard holding copied notes.
#[derive(Debug, Clone, Default)]
pub struct Clipboard {
    /// Copied notes with their original positions.
    notes: Vec<EditableNote>,
    /// The time of the earliest note in the clipboard (used to compute
    /// relative offsets when pasting at the cursor).
    earliest_time: f64,
}

impl Clipboard {
    /// Create an empty clipboard.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the clipboard has content to paste.
    pub fn has_content(&self) -> bool {
        !self.notes.is_empty()
    }

    /// Number of notes in the clipboard.
    pub fn len(&self) -> usize {
        self.notes.len()
    }

    /// Whether the clipboard is empty.
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Clipboard operations
// ---------------------------------------------------------------------------

/// Copy the currently selected notes into the clipboard.
///
/// The copied notes retain their full state; the earliest start time is
/// recorded so that paste can offset relative to the cursor position.
pub fn copy_selected(state: &AppState, clipboard: &mut Clipboard) {
    if state.selected_notes.is_empty() {
        return;
    }

    let mut copied: Vec<EditableNote> = state
        .selected_notes
        .iter()
        .filter_map(|&i| state.notes.get(i).cloned())
        .collect();

    if copied.is_empty() {
        return;
    }

    let earliest = copied
        .iter()
        .map(|n| n.effective_start())
        .fold(f64::MAX, f64::min);

    // Sort by start time for predictable paste ordering
    copied.sort_by(|a, b| {
        a.effective_start()
            .partial_cmp(&b.effective_start())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    clipboard.notes = copied;
    clipboard.earliest_time = earliest;
}

/// Cut the currently selected notes (copy then delete).
pub fn cut_selected(state: &mut AppState, clipboard: &mut Clipboard) {
    copy_selected(state, clipboard);
    state.delete_selected();
}

/// Paste clipboard contents at the current playback position.
///
/// Each note's time offset is adjusted so that the earliest copied note
/// lands at `state.playback_position`. Pitch and other offsets are
/// preserved as-is.
pub fn paste_at_cursor(state: &mut AppState, clipboard: &Clipboard) {
    if clipboard.is_empty() {
        return;
    }

    state.push_undo("Paste Notes");

    let cursor = state.playback_position;
    let time_shift = cursor - clipboard.earliest_time;

    let first_new_idx = state.notes.len();
    state.selected_notes.clear();

    for note in &clipboard.notes {
        let mut pasted = note.clone();
        pasted.time_offset += time_shift;
        state.notes.push(pasted);
    }

    // Select the newly pasted notes
    for i in first_new_idx..state.notes.len() {
        state.selected_notes.insert(i);
    }

    let count = clipboard.len();
    state.status_message = format!("Pasted {} note{}", count, if count > 1 { "s" } else { "" });
}

/// Duplicate selected notes with a small time offset.
///
/// Equivalent to copy + paste at (earliest note time + small delta).
pub fn duplicate_selected(state: &mut AppState) {
    if state.selected_notes.is_empty() {
        return;
    }

    let selected: Vec<EditableNote> = state
        .selected_notes
        .iter()
        .filter_map(|&i| state.notes.get(i).cloned())
        .collect();

    if selected.is_empty() {
        return;
    }

    state.push_undo("Duplicate Notes");

    // Small time offset for the duplicate (100ms)
    let offset = 0.1;

    let first_new_idx = state.notes.len();
    state.selected_notes.clear();

    for note in &selected {
        let mut dup = note.clone();
        dup.time_offset += offset;
        state.notes.push(dup);
    }

    // Select the duplicated notes
    for i in first_new_idx..state.notes.len() {
        state.selected_notes.insert(i);
    }

    let count = selected.len();
    state.status_message = format!(
        "Duplicated {} note{}",
        count,
        if count > 1 { "s" } else { "" }
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use melodyn_core::DetectedNote;

    fn make_note(start: f64, midi: f64) -> EditableNote {
        EditableNote::from_detected(DetectedNote {
            frequency: melodyn_core::PitchCandidate::midi_to_freq(midi),
            midi_note: midi,
            start_time: start,
            duration: 0.5,
            start_sample: 0,
            end_sample: 22050,
            amplitude: 0.5,
            peak_amplitude: 0.5,
            confidence: 0.9,
            pitch_contour: vec![440.0],
            amplitude_contour: vec![0.5],
            vibrato_rate: 0.0,
            vibrato_depth: 0.0,
            attack_time: 0.0,
            release_time: 0.0,
        })
    }

    #[test]
    fn test_copy_paste() {
        let mut state = AppState::default();
        state.notes.push(make_note(1.0, 60.0));
        state.notes.push(make_note(2.0, 64.0));
        state.selected_notes.insert(0);
        state.selected_notes.insert(1);
        state.playback_position = 5.0;

        let mut clipboard = Clipboard::new();
        copy_selected(&state, &mut clipboard);
        assert_eq!(clipboard.len(), 2);
        assert!((clipboard.earliest_time - 1.0).abs() < 1e-10);

        paste_at_cursor(&mut state, &clipboard);
        assert_eq!(state.notes.len(), 4);
        // Pasted notes should start at cursor (5.0)
        let pasted_start = state.notes[2].effective_start();
        assert!(
            (pasted_start - 5.0).abs() < 1e-10,
            "pasted start={}, expected 5.0",
            pasted_start
        );
    }

    #[test]
    fn test_cut() {
        let mut state = AppState::default();
        state.notes.push(make_note(1.0, 60.0));
        state.selected_notes.insert(0);

        let mut clipboard = Clipboard::new();
        cut_selected(&mut state, &mut clipboard);
        assert_eq!(clipboard.len(), 1);
        assert!(state.notes.is_empty());
    }

    #[test]
    fn test_duplicate() {
        let mut state = AppState::default();
        state.notes.push(make_note(1.0, 60.0));
        state.selected_notes.insert(0);

        duplicate_selected(&mut state);
        assert_eq!(state.notes.len(), 2);
        // Duplicate should be offset by 0.1s
        let delta = state.notes[1].effective_start() - state.notes[0].effective_start();
        assert!((delta - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_empty_clipboard() {
        let clipboard = Clipboard::new();
        assert!(clipboard.is_empty());
        assert!(!clipboard.has_content());
    }
}
