//! Enhanced undo/redo system with compound operations, named steps, configurable
//! depth, and integration with the collaborative editing operation model.
//!
//! Builds on the basic `UndoManager` from `document.rs`, adding:
//! - Compound operations (group multiple edits as a single undo step)
//! - Named undo steps for UI display ("Move notes", "Change pitch", etc.)
//! - Configurable max history depth
//! - Branch-aware: undo + new edit discards old redo history
//! - Integration with `collab::EditOp` for operation-based undo

use crate::collab::EditOp;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Undo step
// ---------------------------------------------------------------------------

/// A single atomic change that can be undone/redone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoableEdit {
    /// The forward operation (what was done).
    pub forward: EditOp,
    /// The inverse operation (how to undo it).
    pub inverse: EditOp,
}

impl UndoableEdit {
    /// Create a new undoable edit with explicit forward and inverse operations.
    pub fn new(forward: EditOp, inverse: EditOp) -> Self {
        Self { forward, inverse }
    }

    /// Create an undoable MovePitch edit.
    pub fn move_pitch(note_id: u64, delta_semitones: f64) -> Self {
        Self {
            forward: EditOp::MovePitch {
                note_id,
                delta_semitones,
            },
            inverse: EditOp::MovePitch {
                note_id,
                delta_semitones: -delta_semitones,
            },
        }
    }

    /// Create an undoable MoveTime edit.
    pub fn move_time(note_id: u64, delta_seconds: f64) -> Self {
        Self {
            forward: EditOp::MoveTime {
                note_id,
                delta_seconds,
            },
            inverse: EditOp::MoveTime {
                note_id,
                delta_seconds: -delta_seconds,
            },
        }
    }

    /// Create an undoable SetAmplitude edit.
    pub fn set_amplitude(note_id: u64, old_amplitude: f64, new_amplitude: f64) -> Self {
        Self {
            forward: EditOp::SetAmplitude {
                note_id,
                amplitude: new_amplitude,
            },
            inverse: EditOp::SetAmplitude {
                note_id,
                amplitude: old_amplitude,
            },
        }
    }

    /// Create an undoable SetFormant edit.
    pub fn set_formant(note_id: u64, old_shift: f64, new_shift: f64) -> Self {
        Self {
            forward: EditOp::SetFormant {
                note_id,
                shift: new_shift,
            },
            inverse: EditOp::SetFormant {
                note_id,
                shift: old_shift,
            },
        }
    }

    /// Create an undoable SetTempo edit.
    pub fn set_tempo(position: f64, old_bpm: f64, new_bpm: f64) -> Self {
        Self {
            forward: EditOp::SetTempo {
                position,
                bpm: new_bpm,
            },
            inverse: EditOp::SetTempo {
                position,
                bpm: old_bpm,
            },
        }
    }

    /// Create an undoable DeleteNote / CreateNote pair.
    pub fn delete_note(note_id: u64, pitch: f64, time: f64, duration: f64) -> Self {
        Self {
            forward: EditOp::DeleteNote { note_id },
            inverse: EditOp::CreateNote {
                note_id,
                pitch,
                time,
                duration,
            },
        }
    }

    /// Create an undoable CreateNote / DeleteNote pair.
    pub fn create_note(note_id: u64, pitch: f64, time: f64, duration: f64) -> Self {
        Self {
            forward: EditOp::CreateNote {
                note_id,
                pitch,
                time,
                duration,
            },
            inverse: EditOp::DeleteNote { note_id },
        }
    }
}

// ---------------------------------------------------------------------------
// Compound undo step
// ---------------------------------------------------------------------------

/// A named group of edits that form a single undo step.
/// For example, "Move selected notes" might contain multiple MovePitch
/// and MoveTime edits that should all be undone together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoStep {
    /// Human-readable name for this undo step (shown in Edit menu).
    pub name: String,
    /// The individual edits in this compound step, in order of application.
    pub edits: Vec<UndoableEdit>,
    /// Monotonic step ID for ordering.
    pub step_id: u64,
}

impl UndoStep {
    /// Create a new undo step with a name and no edits.
    pub fn new(name: &str, step_id: u64) -> Self {
        Self {
            name: name.to_string(),
            edits: Vec::new(),
            step_id,
        }
    }

    /// Add an edit to this compound step.
    pub fn add_edit(&mut self, edit: UndoableEdit) {
        self.edits.push(edit);
    }

    /// Get all forward operations in order.
    pub fn forward_ops(&self) -> Vec<&EditOp> {
        self.edits.iter().map(|e| &e.forward).collect()
    }

    /// Get all inverse operations in reverse order (for undo).
    pub fn inverse_ops(&self) -> Vec<&EditOp> {
        self.edits.iter().rev().map(|e| &e.inverse).collect()
    }

    /// Whether this step has any edits.
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Number of individual edits.
    pub fn edit_count(&self) -> usize {
        self.edits.len()
    }
}

// ---------------------------------------------------------------------------
// Undo Manager
// ---------------------------------------------------------------------------

/// Enhanced undo/redo manager with compound operations, configurable depth,
/// and branch-aware redo.
#[derive(Debug)]
pub struct EnhancedUndoManager {
    /// The undo stack (most recent at the end).
    undo_stack: Vec<UndoStep>,
    /// The redo stack (most recent undo at the end).
    redo_stack: Vec<UndoStep>,
    /// Maximum number of undo steps to keep.
    max_depth: usize,
    /// Next step ID counter.
    next_step_id: u64,
    /// Currently accumulating compound step (if any).
    building_step: Option<UndoStep>,
}

impl EnhancedUndoManager {
    /// Create a new undo manager with the given maximum history depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth,
            next_step_id: 1,
            building_step: None,
        }
    }

    /// Begin building a compound undo step. All edits added until `end_compound`
    /// will be grouped into a single step.
    pub fn begin_compound(&mut self, name: &str) {
        let id = self.next_step_id;
        self.next_step_id += 1;
        self.building_step = Some(UndoStep::new(name, id));
    }

    /// Add an edit to the current compound step, or create a single-edit step if
    /// no compound is in progress.
    pub fn add_edit(&mut self, name: &str, edit: UndoableEdit) {
        if let Some(ref mut step) = self.building_step {
            step.add_edit(edit);
        } else {
            // Single-edit step.
            let id = self.next_step_id;
            self.next_step_id += 1;
            let mut step = UndoStep::new(name, id);
            step.add_edit(edit);
            self.push_step(step);
        }
    }

    /// End the current compound step and push it onto the undo stack.
    /// Returns the forward operations of the step, or None if no compound was in progress.
    pub fn end_compound(&mut self) -> Option<Vec<EditOp>> {
        if let Some(step) = self.building_step.take() {
            if step.is_empty() {
                return None;
            }
            let ops: Vec<EditOp> = step.edits.iter().map(|e| e.forward.clone()).collect();
            self.push_step(step);
            Some(ops)
        } else {
            None
        }
    }

    /// Cancel the current compound step without pushing it.
    pub fn cancel_compound(&mut self) {
        self.building_step = None;
    }

    /// Push a completed step onto the undo stack.
    fn push_step(&mut self, step: UndoStep) {
        // Any new edit clears the redo stack (branch-aware).
        self.redo_stack.clear();
        self.undo_stack.push(step);
        // Enforce max depth.
        while self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    /// Undo the most recent step. Returns the inverse operations to apply, or None.
    pub fn undo(&mut self) -> Option<Vec<EditOp>> {
        if let Some(step) = self.undo_stack.pop() {
            let ops: Vec<EditOp> = step.edits.iter().rev().map(|e| e.inverse.clone()).collect();
            self.redo_stack.push(step);
            Some(ops)
        } else {
            None
        }
    }

    /// Redo the most recently undone step. Returns the forward operations, or None.
    pub fn redo(&mut self) -> Option<Vec<EditOp>> {
        if let Some(step) = self.redo_stack.pop() {
            let ops: Vec<EditOp> = step.edits.iter().map(|e| e.forward.clone()).collect();
            self.undo_stack.push(step);
            Some(ops)
        } else {
            None
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

    /// Name of the next undo step (for UI: "Undo Move Notes").
    pub fn undo_name(&self) -> Option<&str> {
        self.undo_stack.last().map(|s| s.name.as_str())
    }

    /// Name of the next redo step (for UI: "Redo Move Notes").
    pub fn redo_name(&self) -> Option<&str> {
        self.redo_stack.last().map(|s| s.name.as_str())
    }

    /// Total number of undo steps.
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Total number of redo steps.
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }

    /// Clear all undo and redo history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.building_step = None;
    }

    /// Get the current max depth.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Change the max depth, truncating if necessary.
    pub fn set_max_depth(&mut self, depth: usize) {
        self.max_depth = depth;
        while self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    /// Get a read-only view of the undo stack for display.
    pub fn undo_history(&self) -> Vec<(&str, usize)> {
        self.undo_stack
            .iter()
            .rev()
            .map(|s| (s.name.as_str(), s.edit_count()))
            .collect()
    }

    /// Get a read-only view of the redo stack for display.
    pub fn redo_history(&self) -> Vec<(&str, usize)> {
        self.redo_stack
            .iter()
            .rev()
            .map(|s| (s.name.as_str(), s.edit_count()))
            .collect()
    }

    /// Whether a compound step is currently being built.
    pub fn is_building_compound(&self) -> bool {
        self.building_step.is_some()
    }

    /// Serialize the full undo/redo state to JSON (for session persistence).
    pub fn to_json(&self) -> Result<String, String> {
        let state = UndoState {
            undo_stack: self.undo_stack.clone(),
            redo_stack: self.redo_stack.clone(),
            max_depth: self.max_depth,
            next_step_id: self.next_step_id,
        };
        serde_json::to_string_pretty(&state).map_err(|e| e.to_string())
    }

    /// Restore from JSON.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let state: UndoState = serde_json::from_str(json).map_err(|e| e.to_string())?;
        Ok(Self {
            undo_stack: state.undo_stack,
            redo_stack: state.redo_stack,
            max_depth: state.max_depth,
            next_step_id: state.next_step_id,
            building_step: None,
        })
    }
}

/// Serializable state for the undo manager.
#[derive(Debug, Serialize, Deserialize)]
struct UndoState {
    undo_stack: Vec<UndoStep>,
    redo_stack: Vec<UndoStep>,
    max_depth: usize,
    next_step_id: u64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undo_redo_basic() {
        let mut mgr = EnhancedUndoManager::new(100);

        // Single edit: move pitch up 2 semitones.
        mgr.add_edit("Move Pitch", UndoableEdit::move_pitch(1, 2.0));

        assert!(mgr.can_undo());
        assert!(!mgr.can_redo());
        assert_eq!(mgr.undo_name(), Some("Move Pitch"));

        // Undo should produce inverse: move pitch down 2.
        let inverse = mgr.undo().unwrap();
        assert_eq!(inverse.len(), 1);
        if let EditOp::MovePitch {
            note_id,
            delta_semitones,
        } = &inverse[0]
        {
            assert_eq!(*note_id, 1);
            assert!((*delta_semitones - (-2.0)).abs() < 1e-10);
        } else {
            panic!("Expected MovePitch");
        }

        assert!(!mgr.can_undo());
        assert!(mgr.can_redo());
        assert_eq!(mgr.redo_name(), Some("Move Pitch"));

        // Redo should produce forward: move pitch up 2.
        let forward = mgr.redo().unwrap();
        assert_eq!(forward.len(), 1);
        if let EditOp::MovePitch {
            delta_semitones, ..
        } = &forward[0]
        {
            assert!((*delta_semitones - 2.0).abs() < 1e-10);
        }

        assert!(mgr.can_undo());
        assert!(!mgr.can_redo());
    }

    #[test]
    fn test_compound_undo() {
        let mut mgr = EnhancedUndoManager::new(100);

        // Begin compound: select and move 3 notes.
        mgr.begin_compound("Move Selected Notes");
        assert!(mgr.is_building_compound());

        mgr.add_edit("", UndoableEdit::move_pitch(1, 3.0));
        mgr.add_edit("", UndoableEdit::move_pitch(2, 3.0));
        mgr.add_edit("", UndoableEdit::move_pitch(3, 3.0));
        let ops = mgr.end_compound().unwrap();
        assert_eq!(ops.len(), 3);
        assert!(!mgr.is_building_compound());

        // Should be a single undo step with 3 edits.
        assert_eq!(mgr.undo_depth(), 1);
        assert_eq!(mgr.undo_name(), Some("Move Selected Notes"));

        // Undo all 3 at once (in reverse order).
        let inverse = mgr.undo().unwrap();
        assert_eq!(inverse.len(), 3);
        // Reversed: note 3, 2, 1.
        if let EditOp::MovePitch {
            note_id,
            delta_semitones,
        } = &inverse[0]
        {
            assert_eq!(*note_id, 3);
            assert!((*delta_semitones - (-3.0)).abs() < 1e-10);
        }
        if let EditOp::MovePitch { note_id, .. } = &inverse[1] {
            assert_eq!(*note_id, 2);
        }
        if let EditOp::MovePitch { note_id, .. } = &inverse[2] {
            assert_eq!(*note_id, 1);
        }
    }

    #[test]
    fn test_redo_cleared_on_new_edit() {
        let mut mgr = EnhancedUndoManager::new(100);

        mgr.add_edit("Edit A", UndoableEdit::move_pitch(1, 1.0));
        mgr.add_edit("Edit B", UndoableEdit::move_pitch(2, 2.0));

        // Undo B.
        mgr.undo();
        assert!(mgr.can_redo());
        assert_eq!(mgr.redo_name(), Some("Edit B"));

        // New edit C: should clear redo.
        mgr.add_edit("Edit C", UndoableEdit::move_pitch(3, 3.0));
        assert!(!mgr.can_redo());
        assert_eq!(mgr.undo_depth(), 2); // A, C
    }

    #[test]
    fn test_max_depth_enforcement() {
        let mut mgr = EnhancedUndoManager::new(3);

        mgr.add_edit("A", UndoableEdit::move_pitch(1, 1.0));
        mgr.add_edit("B", UndoableEdit::move_pitch(2, 2.0));
        mgr.add_edit("C", UndoableEdit::move_pitch(3, 3.0));
        assert_eq!(mgr.undo_depth(), 3);

        // Adding a 4th should evict the oldest.
        mgr.add_edit("D", UndoableEdit::move_pitch(4, 4.0));
        assert_eq!(mgr.undo_depth(), 3);

        // The undo history should be B, C, D (A was evicted).
        let history = mgr.undo_history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].0, "D");
        assert_eq!(history[1].0, "C");
        assert_eq!(history[2].0, "B");
    }

    #[test]
    fn test_set_max_depth_truncates() {
        let mut mgr = EnhancedUndoManager::new(10);
        for i in 0..5 {
            mgr.add_edit(
                &format!("Step {i}"),
                UndoableEdit::move_pitch(i as u64, 1.0),
            );
        }
        assert_eq!(mgr.undo_depth(), 5);

        mgr.set_max_depth(2);
        assert_eq!(mgr.undo_depth(), 2);
        assert_eq!(mgr.max_depth(), 2);
    }

    #[test]
    fn test_undo_redo_multiple_steps() {
        let mut mgr = EnhancedUndoManager::new(100);

        mgr.add_edit("A", UndoableEdit::move_pitch(1, 1.0));
        mgr.add_edit("B", UndoableEdit::set_amplitude(2, 0.5, 0.8));
        mgr.add_edit("C", UndoableEdit::move_time(3, 0.1));

        assert_eq!(mgr.undo_depth(), 3);

        // Undo C.
        let inv_c = mgr.undo().unwrap();
        if let EditOp::MoveTime { delta_seconds, .. } = &inv_c[0] {
            assert!((*delta_seconds - (-0.1)).abs() < 1e-10);
        }

        // Undo B.
        let inv_b = mgr.undo().unwrap();
        if let EditOp::SetAmplitude { amplitude, .. } = &inv_b[0] {
            assert!((*amplitude - 0.5).abs() < 1e-10); // restored to old value
        }

        assert_eq!(mgr.undo_depth(), 1);
        assert_eq!(mgr.redo_depth(), 2);

        // Redo B.
        let fwd_b = mgr.redo().unwrap();
        if let EditOp::SetAmplitude { amplitude, .. } = &fwd_b[0] {
            assert!((*amplitude - 0.8).abs() < 1e-10); // new value
        }
    }

    #[test]
    fn test_undoable_edit_constructors() {
        let edit = UndoableEdit::delete_note(5, 440.0, 1.0, 0.5);
        assert!(matches!(edit.forward, EditOp::DeleteNote { note_id: 5 }));
        if let EditOp::CreateNote {
            note_id,
            pitch,
            time,
            duration,
        } = &edit.inverse
        {
            assert_eq!(*note_id, 5);
            assert!((*pitch - 440.0).abs() < 1e-10);
            assert!((*time - 1.0).abs() < 1e-10);
            assert!((*duration - 0.5).abs() < 1e-10);
        } else {
            panic!("Expected CreateNote");
        }

        let edit2 = UndoableEdit::create_note(6, 880.0, 2.0, 1.0);
        assert!(matches!(
            edit2.forward,
            EditOp::CreateNote { note_id: 6, .. }
        ));
        assert!(matches!(edit2.inverse, EditOp::DeleteNote { note_id: 6 }));
    }

    #[test]
    fn test_undo_step_operations() {
        let mut step = UndoStep::new("Test", 1);
        assert!(step.is_empty());

        step.add_edit(UndoableEdit::move_pitch(1, 2.0));
        step.add_edit(UndoableEdit::move_pitch(2, 3.0));
        assert_eq!(step.edit_count(), 2);

        let fwd = step.forward_ops();
        assert_eq!(fwd.len(), 2);
        if let EditOp::MovePitch { note_id, .. } = fwd[0] {
            assert_eq!(*note_id, 1);
        }

        let inv = step.inverse_ops();
        assert_eq!(inv.len(), 2);
        // Inverse is reversed.
        if let EditOp::MovePitch { note_id, .. } = inv[0] {
            assert_eq!(*note_id, 2);
        }
    }

    #[test]
    fn test_clear() {
        let mut mgr = EnhancedUndoManager::new(100);
        mgr.add_edit("A", UndoableEdit::move_pitch(1, 1.0));
        mgr.undo();
        assert!(mgr.can_redo());

        mgr.clear();
        assert!(!mgr.can_undo());
        assert!(!mgr.can_redo());
        assert_eq!(mgr.undo_depth(), 0);
    }

    #[test]
    fn test_cancel_compound() {
        let mut mgr = EnhancedUndoManager::new(100);
        mgr.begin_compound("Test");
        mgr.add_edit("", UndoableEdit::move_pitch(1, 1.0));
        mgr.cancel_compound();
        assert!(!mgr.is_building_compound());
        assert_eq!(mgr.undo_depth(), 0);
    }

    #[test]
    fn test_empty_compound() {
        let mut mgr = EnhancedUndoManager::new(100);
        mgr.begin_compound("Empty");
        let result = mgr.end_compound();
        assert!(result.is_none());
        assert_eq!(mgr.undo_depth(), 0);
    }

    #[test]
    fn test_undo_nothing() {
        let mut mgr = EnhancedUndoManager::new(100);
        assert!(mgr.undo().is_none());
        assert!(mgr.redo().is_none());
        assert!(mgr.undo_name().is_none());
        assert!(mgr.redo_name().is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut mgr = EnhancedUndoManager::new(50);
        mgr.add_edit("Move pitch", UndoableEdit::move_pitch(1, 2.0));
        mgr.add_edit("Set amplitude", UndoableEdit::set_amplitude(2, 0.5, 0.9));
        mgr.undo(); // Set amplitude is now in redo.

        let json = mgr.to_json().unwrap();
        let restored = EnhancedUndoManager::from_json(&json).unwrap();

        assert_eq!(restored.undo_depth(), 1);
        assert_eq!(restored.redo_depth(), 1);
        assert_eq!(restored.max_depth(), 50);
        assert_eq!(restored.undo_name(), Some("Move pitch"));
        assert_eq!(restored.redo_name(), Some("Set amplitude"));
    }

    #[test]
    fn test_undo_history_display() {
        let mut mgr = EnhancedUndoManager::new(100);
        mgr.add_edit("A", UndoableEdit::move_pitch(1, 1.0));

        mgr.begin_compound("B compound");
        mgr.add_edit("", UndoableEdit::move_pitch(2, 2.0));
        mgr.add_edit("", UndoableEdit::move_pitch(3, 3.0));
        mgr.end_compound();

        let history = mgr.undo_history();
        assert_eq!(history.len(), 2);
        // Most recent first.
        assert_eq!(history[0].0, "B compound");
        assert_eq!(history[0].1, 2); // 2 edits in compound
        assert_eq!(history[1].0, "A");
        assert_eq!(history[1].1, 1);
    }

    #[test]
    fn test_set_formant_undo() {
        let mut mgr = EnhancedUndoManager::new(100);
        mgr.add_edit("Set formant", UndoableEdit::set_formant(1, 0.0, 1.5));

        let inv = mgr.undo().unwrap();
        if let EditOp::SetFormant { shift, .. } = &inv[0] {
            assert!((*shift - 0.0).abs() < 1e-10);
        } else {
            panic!("Expected SetFormant");
        }

        let fwd = mgr.redo().unwrap();
        if let EditOp::SetFormant { shift, .. } = &fwd[0] {
            assert!((*shift - 1.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_set_tempo_undo() {
        let mut mgr = EnhancedUndoManager::new(100);
        mgr.add_edit("Change tempo", UndoableEdit::set_tempo(0.0, 120.0, 140.0));

        let inv = mgr.undo().unwrap();
        if let EditOp::SetTempo { bpm, .. } = &inv[0] {
            assert!((*bpm - 120.0).abs() < 1e-10);
        }

        let fwd = mgr.redo().unwrap();
        if let EditOp::SetTempo { bpm, .. } = &fwd[0] {
            assert!((*bpm - 140.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_branch_aware_undo_redo() {
        let mut mgr = EnhancedUndoManager::new(100);
        mgr.add_edit("A", UndoableEdit::move_pitch(1, 1.0));
        mgr.add_edit("B", UndoableEdit::move_pitch(2, 2.0));
        mgr.add_edit("C", UndoableEdit::move_pitch(3, 3.0));

        // Undo C, B.
        mgr.undo();
        mgr.undo();
        assert_eq!(mgr.redo_depth(), 2);

        // New edit D: should discard B and C from redo.
        mgr.add_edit("D", UndoableEdit::move_pitch(4, 4.0));
        assert_eq!(mgr.redo_depth(), 0);
        assert_eq!(mgr.undo_depth(), 2); // A, D
        assert_eq!(mgr.undo_name(), Some("D"));

        // Undo D.
        mgr.undo();
        assert_eq!(mgr.undo_name(), Some("A"));
        assert_eq!(mgr.redo_name(), Some("D"));
    }
}
