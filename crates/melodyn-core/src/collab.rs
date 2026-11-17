//! Collaborative editing sync system with operational transformation.
//!
//! Implements `GNEditingSyncServer` / `GNEditingSyncClient` semantics: an append-only
//! operation log, operational transformation for concurrent edits, conflict resolution
//! with last-writer-wins, session management with optimistic local apply and rebase,
//! and sync protocol message types for a future network layer.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CollabError {
    #[error("Operation {0} not found in log")]
    OpNotFound(u64),
    #[error("Note {0} not found")]
    NoteNotFound(u64),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("Session already exists for author {0}")]
    SessionExists(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type CollabResult<T> = Result<T, CollabError>;

// ---------------------------------------------------------------------------
// Note ID and Author ID
// ---------------------------------------------------------------------------

/// Unique identifier for a note in the editing model.
pub type NoteId = u64;
/// Unique identifier for a collaborating author/client.
pub type AuthorId = String;

// ---------------------------------------------------------------------------
// Edit Operations
// ---------------------------------------------------------------------------

/// An atomic unit of editing change. Each variant captures the full delta
/// needed to apply or transform the operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditOp {
    /// Shift a note's pitch by the given number of semitones.
    MovePitch { note_id: NoteId, delta_semitones: f64 },
    /// Shift a note's time position by the given number of seconds.
    MoveTime { note_id: NoteId, delta_seconds: f64 },
    /// Set a note's amplitude to an absolute value.
    SetAmplitude { note_id: NoteId, amplitude: f64 },
    /// Set a note's formant shift.
    SetFormant { note_id: NoteId, shift: f64 },
    /// Split a note at the given time (seconds from note start).
    SplitNote { note_id: NoteId, split_time: f64 },
    /// Merge two or more notes into one.
    MergeNotes { note_ids: Vec<NoteId> },
    /// Delete a note.
    DeleteNote { note_id: NoteId },
    /// Create a new note with the given properties.
    CreateNote {
        note_id: NoteId,
        pitch: f64,
        time: f64,
        duration: f64,
    },
    /// Set the tempo at a position (in seconds).
    SetTempo { position: f64, bpm: f64 },
    /// Set the scale at a position.
    SetScale { position: f64, scale: String },
}

impl EditOp {
    /// Return the note ID(s) this operation targets, if any.
    pub fn target_note_ids(&self) -> Vec<NoteId> {
        match self {
            Self::MovePitch { note_id, .. }
            | Self::MoveTime { note_id, .. }
            | Self::SetAmplitude { note_id, .. }
            | Self::SetFormant { note_id, .. }
            | Self::SplitNote { note_id, .. }
            | Self::DeleteNote { note_id } => vec![*note_id],
            Self::MergeNotes { note_ids } => note_ids.clone(),
            Self::CreateNote { note_id, .. } => vec![*note_id],
            Self::SetTempo { .. } | Self::SetScale { .. } => vec![],
        }
    }

    /// Whether this operation and another conflict (target the same note with
    /// incompatible edits).
    pub fn conflicts_with(&self, other: &EditOp) -> bool {
        let my_ids = self.target_note_ids();
        let other_ids = other.target_note_ids();
        if my_ids.is_empty() || other_ids.is_empty() {
            // Tempo/scale ops don't conflict with note ops.
            // Two tempo ops at the same position do conflict.
            return match (self, other) {
                (
                    Self::SetTempo { position: p1, .. },
                    Self::SetTempo { position: p2, .. },
                ) => (p1 - p2).abs() < 1e-6,
                (
                    Self::SetScale { position: p1, .. },
                    Self::SetScale { position: p2, .. },
                ) => (p1 - p2).abs() < 1e-6,
                _ => false,
            };
        }
        // Check for overlapping note IDs.
        my_ids.iter().any(|id| other_ids.contains(id))
    }
}

// ---------------------------------------------------------------------------
// Edit Operation with metadata
// ---------------------------------------------------------------------------

/// A timestamped, attributed edit operation with a unique sequential ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditOperation {
    /// Unique sequential operation ID (assigned by the server/log).
    pub op_id: u64,
    /// The edit operation.
    pub op: EditOp,
    /// Author who created this operation.
    pub author: AuthorId,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
}

// ---------------------------------------------------------------------------
// Operation Log
// ---------------------------------------------------------------------------

/// An append-only log of edit operations, supporting replay from any point.
#[derive(Debug, Clone)]
pub struct OperationLog {
    operations: Vec<EditOperation>,
    next_op_id: u64,
}

impl Default for OperationLog {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationLog {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            next_op_id: 1,
        }
    }

    /// Append an operation to the log, assigning it the next sequential ID.
    /// Returns the assigned operation ID.
    pub fn append(&mut self, op: EditOp, author: AuthorId, timestamp_ms: u64) -> u64 {
        let op_id = self.next_op_id;
        self.next_op_id += 1;
        self.operations.push(EditOperation {
            op_id,
            op,
            author,
            timestamp_ms,
        });
        op_id
    }

    /// Append a pre-built operation (must have op_id already set).
    pub fn append_operation(&mut self, operation: EditOperation) {
        if operation.op_id >= self.next_op_id {
            self.next_op_id = operation.op_id + 1;
        }
        self.operations.push(operation);
    }

    /// Get all operations after (not including) the given op_id.
    pub fn operations_after(&self, op_id: u64) -> Vec<&EditOperation> {
        self.operations.iter().filter(|o| o.op_id > op_id).collect()
    }

    /// Get all operations in the log.
    pub fn all_operations(&self) -> &[EditOperation] {
        &self.operations
    }

    /// Get a specific operation by ID.
    pub fn get(&self, op_id: u64) -> Option<&EditOperation> {
        self.operations.iter().find(|o| o.op_id == op_id)
    }

    /// The ID of the last operation, or 0 if empty.
    pub fn last_op_id(&self) -> u64 {
        self.operations.last().map(|o| o.op_id).unwrap_or(0)
    }

    /// Total number of operations.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Replay all operations from a given point, returning cloned operations.
    pub fn replay_from(&self, from_op_id: u64) -> Vec<EditOperation> {
        self.operations
            .iter()
            .filter(|o| o.op_id > from_op_id)
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Operational Transformation
// ---------------------------------------------------------------------------

/// Transform operation `op` against a concurrent operation `against`, producing
/// an adjusted `op` that accounts for the effect of `against` having been applied first.
///
/// This is the core OT function: if two clients independently produce operations
/// A and B from the same state, then `transform(A, B)` yields A' such that
/// applying B then A' produces the same state as applying A then transform(B, A).
pub fn transform(op: &EditOp, against: &EditOp) -> EditOp {
    // If the operations don't conflict, the operation is unchanged.
    if !op.conflicts_with(against) {
        return op.clone();
    }

    match (op, against) {
        // Both move pitch on the same note: accumulate deltas.
        (
            EditOp::MovePitch { note_id, delta_semitones },
            EditOp::MovePitch { note_id: other_id, .. },
        ) if note_id == other_id => {
            // The other move has already been applied, so our move is still relative.
            // Keep our original delta (it's additive).
            EditOp::MovePitch {
                note_id: *note_id,
                delta_semitones: *delta_semitones,
            }
        }

        // Both move time on the same note: keep our delta (additive).
        (
            EditOp::MoveTime { note_id, delta_seconds },
            EditOp::MoveTime { note_id: other_id, .. },
        ) if note_id == other_id => EditOp::MoveTime {
            note_id: *note_id,
            delta_seconds: *delta_seconds,
        },

        // Last-writer-wins for absolute-value operations on the same note.
        (
            EditOp::SetAmplitude { note_id, amplitude },
            EditOp::SetAmplitude { note_id: other_id, .. },
        ) if note_id == other_id => {
            // Our value wins (we are the later writer in this transform direction).
            EditOp::SetAmplitude {
                note_id: *note_id,
                amplitude: *amplitude,
            }
        }

        (
            EditOp::SetFormant { note_id, shift },
            EditOp::SetFormant { note_id: other_id, .. },
        ) if note_id == other_id => EditOp::SetFormant {
            note_id: *note_id,
            shift: *shift,
        },

        // If the other operation deleted the note we're editing, our edit becomes a no-op.
        (_, EditOp::DeleteNote { note_id: deleted_id })
            if op.target_note_ids().contains(deleted_id) =>
        {
            // Convert to a no-op by keeping the same shape but zeroing the effect.
            match op {
                EditOp::MovePitch { note_id, .. } => EditOp::MovePitch {
                    note_id: *note_id,
                    delta_semitones: 0.0,
                },
                EditOp::MoveTime { note_id, .. } => EditOp::MoveTime {
                    note_id: *note_id,
                    delta_seconds: 0.0,
                },
                EditOp::SetAmplitude { note_id, .. } => EditOp::SetAmplitude {
                    note_id: *note_id,
                    amplitude: 0.0,
                },
                EditOp::SetFormant { note_id, .. } => EditOp::SetFormant {
                    note_id: *note_id,
                    shift: 0.0,
                },
                EditOp::SplitNote { note_id, .. } => EditOp::MovePitch {
                    note_id: *note_id,
                    delta_semitones: 0.0,
                },
                EditOp::DeleteNote { note_id } => EditOp::MovePitch {
                    note_id: *note_id,
                    delta_semitones: 0.0,
                },
                other => other.clone(),
            }
        }

        // Two tempo sets at the same position: last-writer-wins.
        (
            EditOp::SetTempo { position, bpm },
            EditOp::SetTempo { .. },
        ) => EditOp::SetTempo {
            position: *position,
            bpm: *bpm,
        },

        // Two scale sets at the same position: last-writer-wins.
        (
            EditOp::SetScale { position, scale },
            EditOp::SetScale { .. },
        ) => EditOp::SetScale {
            position: *position,
            scale: scale.clone(),
        },

        // Default: return op unchanged (no transformation needed).
        _ => op.clone(),
    }
}

/// Transform a list of operations against a single concurrent operation.
pub fn transform_list(ops: &[EditOp], against: &EditOp) -> Vec<EditOp> {
    ops.iter().map(|op| transform(op, against)).collect()
}

// ---------------------------------------------------------------------------
// Conflict Resolution
// ---------------------------------------------------------------------------

/// Describes a detected conflict between two operations.
#[derive(Debug, Clone)]
pub struct Conflict {
    /// The local operation that conflicts.
    pub local_op: EditOperation,
    /// The remote operation that conflicts.
    pub remote_op: EditOperation,
    /// How the conflict was resolved.
    pub resolution: ConflictResolution,
}

/// How a conflict was resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictResolution {
    /// The local operation's value was kept (last-writer-wins, local is later).
    LocalWins,
    /// The remote operation's value was kept.
    RemoteWins,
    /// Both operations were applied (they are compatible, e.g., additive deltas).
    BothApplied,
    /// The local operation was dropped (target was deleted by remote).
    LocalDropped,
}

/// Resolve conflicts between a local pending operation and a remote confirmed operation.
/// Returns the conflict description and the transformed local operation.
pub fn resolve_conflict(
    local: &EditOperation,
    remote: &EditOperation,
) -> (Conflict, EditOp) {
    let transformed = transform(&local.op, &remote.op);
    let resolution = determine_resolution(&local.op, &remote.op, &transformed);
    let conflict = Conflict {
        local_op: local.clone(),
        remote_op: remote.clone(),
        resolution,
    };
    (conflict, transformed)
}

fn determine_resolution(
    local: &EditOp,
    remote: &EditOp,
    transformed: &EditOp,
) -> ConflictResolution {
    // If the target was deleted by remote, local is dropped.
    if let EditOp::DeleteNote { .. } = remote {
        if local.target_note_ids().iter().any(|id| remote.target_note_ids().contains(id)) {
            return ConflictResolution::LocalDropped;
        }
    }

    // For additive operations (MovePitch, MoveTime), both are applied.
    match (local, remote) {
        (EditOp::MovePitch { .. }, EditOp::MovePitch { .. })
        | (EditOp::MoveTime { .. }, EditOp::MoveTime { .. }) => {
            return ConflictResolution::BothApplied;
        }
        _ => {}
    }

    // For absolute-value operations, the transform keeps local's value = local wins.
    if transformed == local {
        ConflictResolution::LocalWins
    } else {
        ConflictResolution::RemoteWins
    }
}

// ---------------------------------------------------------------------------
// Sync Protocol Messages
// ---------------------------------------------------------------------------

/// Messages for the sync protocol (future network layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncMessage {
    /// Client requests all operations after last_known_op_id.
    SyncRequest { last_known_op_id: u64 },
    /// Server responds with operations the client is missing.
    SyncResponse { operations: Vec<EditOperation> },
    /// Client pushes a new operation to the server.
    PushOperation { operation: EditOperation },
    /// Server acknowledges a client's operation with its assigned ID.
    AckOperation { op_id: u64 },
    /// Server sends a full model snapshot for initial sync.
    ModelSnapshot {
        /// The state as a JSON blob (kept opaque for flexibility).
        state_json: String,
        /// The last op_id included in this snapshot.
        last_op_id: u64,
    },
}

impl SyncMessage {
    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> CollabResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| CollabError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> CollabResult<Self> {
        serde_json::from_slice(data).map_err(|e| CollabError::SerializationError(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Edit Session
// ---------------------------------------------------------------------------

/// A local editing session that tracks pending (unconfirmed) and confirmed operations.
/// Supports optimistic local application and rebase when remote operations arrive.
#[derive(Debug, Clone)]
pub struct EditSession {
    /// The author ID for this session.
    pub author: AuthorId,
    /// The ID of the last confirmed operation from the server.
    pub last_confirmed_op_id: u64,
    /// Operations confirmed by the server (the shared truth).
    confirmed_ops: Vec<EditOperation>,
    /// Operations applied locally but not yet confirmed by the server.
    pending_ops: Vec<EditOp>,
    /// Conflicts detected during rebase.
    pub conflicts: Vec<Conflict>,
    /// Monotonic local timestamp counter for ordering.
    local_timestamp: u64,
}

impl EditSession {
    /// Create a new editing session for the given author.
    pub fn new(author: AuthorId) -> Self {
        Self {
            author,
            last_confirmed_op_id: 0,
            confirmed_ops: Vec::new(),
            pending_ops: Vec::new(),
            conflicts: Vec::new(),
            local_timestamp: 0,
        }
    }

    /// Apply an operation locally (optimistic). It is added to the pending queue.
    pub fn apply_local(&mut self, op: EditOp) {
        self.pending_ops.push(op);
    }

    /// Get the pending operations.
    pub fn pending_ops(&self) -> &[EditOp] {
        &self.pending_ops
    }

    /// Get confirmed operations.
    pub fn confirmed_ops(&self) -> &[EditOperation] {
        &self.confirmed_ops
    }

    /// Create a PushOperation message for the next pending operation.
    pub fn next_push_message(&mut self) -> Option<SyncMessage> {
        if self.pending_ops.is_empty() {
            return None;
        }
        let op = self.pending_ops[0].clone();
        self.local_timestamp += 1;
        Some(SyncMessage::PushOperation {
            operation: EditOperation {
                op_id: 0, // Server will assign the real ID.
                op,
                author: self.author.clone(),
                timestamp_ms: self.local_timestamp,
            },
        })
    }

    /// Handle acknowledgment from the server: move the first pending op to confirmed.
    pub fn handle_ack(&mut self, op_id: u64) {
        if let Some(op) = self.pending_ops.first().cloned() {
            self.pending_ops.remove(0);
            self.confirmed_ops.push(EditOperation {
                op_id,
                op,
                author: self.author.clone(),
                timestamp_ms: self.local_timestamp,
            });
            self.last_confirmed_op_id = op_id;
        }
    }

    /// Receive remote confirmed operations and rebase local pending ops.
    /// This is the core of optimistic concurrency: when the server confirms
    /// operations from other clients, we transform our pending ops against them.
    pub fn receive_remote_ops(&mut self, remote_ops: &[EditOperation]) {
        for remote in remote_ops {
            if remote.author == self.author {
                // This is our own operation being confirmed; handle as ack.
                self.handle_ack(remote.op_id);
                continue;
            }

            // Rebase each pending operation against the remote operation.
            let mut new_pending = Vec::with_capacity(self.pending_ops.len());
            for pending in &self.pending_ops {
                let local_edit_op = EditOperation {
                    op_id: 0,
                    op: pending.clone(),
                    author: self.author.clone(),
                    timestamp_ms: self.local_timestamp,
                };
                if pending.conflicts_with(&remote.op) {
                    let (conflict, transformed) = resolve_conflict(&local_edit_op, remote);
                    self.conflicts.push(conflict);
                    new_pending.push(transformed);
                } else {
                    new_pending.push(pending.clone());
                }
            }
            self.pending_ops = new_pending;
            self.confirmed_ops.push(remote.clone());
            self.last_confirmed_op_id = remote.op_id;
        }
    }

    /// Create a sync request message to get missed operations.
    pub fn sync_request(&self) -> SyncMessage {
        SyncMessage::SyncRequest {
            last_known_op_id: self.last_confirmed_op_id,
        }
    }

    /// Total number of operations (confirmed + pending).
    pub fn total_ops(&self) -> usize {
        self.confirmed_ops.len() + self.pending_ops.len()
    }

    /// Serialize the session state to JSON for crash recovery.
    pub fn to_json(&self) -> CollabResult<String> {
        let state = SessionState {
            author: self.author.clone(),
            last_confirmed_op_id: self.last_confirmed_op_id,
            confirmed_ops: self.confirmed_ops.clone(),
            pending_ops: self.pending_ops.clone(),
            local_timestamp: self.local_timestamp,
        };
        serde_json::to_string_pretty(&state)
            .map_err(|e| CollabError::SerializationError(e.to_string()))
    }

    /// Restore a session from JSON.
    pub fn from_json(json: &str) -> CollabResult<Self> {
        let state: SessionState =
            serde_json::from_str(json).map_err(|e| CollabError::SerializationError(e.to_string()))?;
        Ok(Self {
            author: state.author,
            last_confirmed_op_id: state.last_confirmed_op_id,
            confirmed_ops: state.confirmed_ops,
            pending_ops: state.pending_ops,
            conflicts: Vec::new(),
            local_timestamp: state.local_timestamp,
        })
    }
}

/// Internal serializable session state for crash recovery.
#[derive(Debug, Serialize, Deserialize)]
struct SessionState {
    author: AuthorId,
    last_confirmed_op_id: u64,
    confirmed_ops: Vec<EditOperation>,
    pending_ops: Vec<EditOp>,
    local_timestamp: u64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op(op_id: u64, op: EditOp, author: &str) -> EditOperation {
        EditOperation {
            op_id,
            op,
            author: author.to_string(),
            timestamp_ms: op_id * 1000,
        }
    }

    // -- Operation serialization --

    #[test]
    fn test_edit_op_serialization_roundtrip() {
        let ops = vec![
            EditOp::MovePitch { note_id: 1, delta_semitones: 2.5 },
            EditOp::MoveTime { note_id: 2, delta_seconds: -0.1 },
            EditOp::SetAmplitude { note_id: 3, amplitude: 0.8 },
            EditOp::SetFormant { note_id: 4, shift: 1.5 },
            EditOp::SplitNote { note_id: 5, split_time: 0.3 },
            EditOp::MergeNotes { note_ids: vec![6, 7, 8] },
            EditOp::DeleteNote { note_id: 9 },
            EditOp::CreateNote { note_id: 10, pitch: 440.0, time: 1.0, duration: 0.5 },
            EditOp::SetTempo { position: 0.0, bpm: 120.0 },
            EditOp::SetScale { position: 0.0, scale: "C major".to_string() },
        ];

        for op in &ops {
            let json = serde_json::to_string(op).unwrap();
            let loaded: EditOp = serde_json::from_str(&json).unwrap();
            assert_eq!(op, &loaded);
        }
    }

    #[test]
    fn test_edit_operation_serialization() {
        let operation = make_op(
            1,
            EditOp::MovePitch { note_id: 42, delta_semitones: 3.0 },
            "alice",
        );
        let json = serde_json::to_string(&operation).unwrap();
        let loaded: EditOperation = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.op_id, 1);
        assert_eq!(loaded.author, "alice");
        assert_eq!(loaded.op, operation.op);
    }

    // -- Operation Log --

    #[test]
    fn test_operation_log_append_and_replay() {
        let mut log = OperationLog::new();
        assert!(log.is_empty());

        let id1 = log.append(
            EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 },
            "alice".to_string(),
            1000,
        );
        let id2 = log.append(
            EditOp::MoveTime { note_id: 1, delta_seconds: 0.1 },
            "bob".to_string(),
            2000,
        );
        let id3 = log.append(
            EditOp::SetAmplitude { note_id: 2, amplitude: 0.5 },
            "alice".to_string(),
            3000,
        );

        assert_eq!(log.len(), 3);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        assert_eq!(log.last_op_id(), 3);

        // Replay from op 1 should give ops 2 and 3.
        let replay = log.replay_from(1);
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].op_id, 2);
        assert_eq!(replay[1].op_id, 3);

        // operations_after works the same way.
        let after = log.operations_after(2);
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].op_id, 3);

        // Get specific op.
        let op2 = log.get(2).unwrap();
        assert_eq!(op2.author, "bob");
    }

    // -- Operational Transform --

    #[test]
    fn test_transform_non_conflicting_independent() {
        let op_a = EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 };
        let op_b = EditOp::MovePitch { note_id: 2, delta_semitones: -1.0 };

        // Different notes: no conflict.
        let transformed = transform(&op_a, &op_b);
        assert_eq!(transformed, op_a);
    }

    #[test]
    fn test_transform_concurrent_pitch_moves_same_note() {
        let op_a = EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 };
        let op_b = EditOp::MovePitch { note_id: 1, delta_semitones: 3.0 };

        // Both move the same note. After B is applied, A should still apply its delta.
        let a_prime = transform(&op_a, &op_b);
        let b_prime = transform(&op_b, &op_a);

        // Both produce additive results:
        // State + B + A' should equal State + A + B'
        // With additive semantics: total shift = 2 + 3 = 5 regardless of order.
        if let EditOp::MovePitch { delta_semitones, .. } = a_prime {
            assert!((delta_semitones - 2.0).abs() < 1e-10);
        } else {
            panic!("Expected MovePitch");
        }
        if let EditOp::MovePitch { delta_semitones, .. } = b_prime {
            assert!((delta_semitones - 3.0).abs() < 1e-10);
        } else {
            panic!("Expected MovePitch");
        }
    }

    #[test]
    fn test_transform_concurrent_time_moves_same_note() {
        let op_a = EditOp::MoveTime { note_id: 5, delta_seconds: 0.1 };
        let op_b = EditOp::MoveTime { note_id: 5, delta_seconds: -0.2 };

        let a_prime = transform(&op_a, &op_b);
        if let EditOp::MoveTime { delta_seconds, .. } = a_prime {
            assert!((delta_seconds - 0.1).abs() < 1e-10);
        } else {
            panic!("Expected MoveTime");
        }
    }

    #[test]
    fn test_transform_set_amplitude_last_writer_wins() {
        let op_a = EditOp::SetAmplitude { note_id: 1, amplitude: 0.3 };
        let op_b = EditOp::SetAmplitude { note_id: 1, amplitude: 0.7 };

        // Transform A against B: A is the "later" writer, so A's value is kept.
        let a_prime = transform(&op_a, &op_b);
        if let EditOp::SetAmplitude { amplitude, .. } = a_prime {
            assert!((amplitude - 0.3).abs() < 1e-10);
        } else {
            panic!("Expected SetAmplitude");
        }
    }

    #[test]
    fn test_transform_edit_against_delete() {
        let op_a = EditOp::MovePitch { note_id: 1, delta_semitones: 5.0 };
        let op_b = EditOp::DeleteNote { note_id: 1 };

        // B deletes the note A wants to edit. A becomes a no-op.
        let a_prime = transform(&op_a, &op_b);
        if let EditOp::MovePitch { delta_semitones, .. } = a_prime {
            assert!((delta_semitones).abs() < 1e-10);
        } else {
            panic!("Expected no-op MovePitch");
        }
    }

    #[test]
    fn test_transform_concurrent_nonconflicting_produces_same_result() {
        // Two edits on different notes: applying in either order should work.
        let op_a = EditOp::MovePitch { note_id: 1, delta_semitones: 3.0 };
        let op_b = EditOp::SetAmplitude { note_id: 2, amplitude: 0.5 };

        let a_prime = transform(&op_a, &op_b);
        let b_prime = transform(&op_b, &op_a);

        // Both are unchanged since they don't conflict.
        assert_eq!(a_prime, op_a);
        assert_eq!(b_prime, op_b);
    }

    #[test]
    fn test_transform_concurrent_tempo_sets() {
        let op_a = EditOp::SetTempo { position: 0.0, bpm: 120.0 };
        let op_b = EditOp::SetTempo { position: 0.0, bpm: 140.0 };

        let a_prime = transform(&op_a, &op_b);
        if let EditOp::SetTempo { bpm, .. } = a_prime {
            assert!((bpm - 120.0).abs() < 1e-10); // A wins as the "later" writer.
        } else {
            panic!("Expected SetTempo");
        }
    }

    // -- Conflict Resolution --

    #[test]
    fn test_conflict_resolution_same_note() {
        let local = make_op(
            0,
            EditOp::SetAmplitude { note_id: 1, amplitude: 0.3 },
            "alice",
        );
        let remote = make_op(
            5,
            EditOp::SetAmplitude { note_id: 1, amplitude: 0.9 },
            "bob",
        );

        let (conflict, transformed) = resolve_conflict(&local, &remote);
        assert_eq!(conflict.resolution, ConflictResolution::LocalWins);
        if let EditOp::SetAmplitude { amplitude, .. } = transformed {
            assert!((amplitude - 0.3).abs() < 1e-10);
        }
    }

    #[test]
    fn test_conflict_resolution_additive() {
        let local = make_op(
            0,
            EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 },
            "alice",
        );
        let remote = make_op(
            5,
            EditOp::MovePitch { note_id: 1, delta_semitones: -1.0 },
            "bob",
        );

        let (conflict, _) = resolve_conflict(&local, &remote);
        assert_eq!(conflict.resolution, ConflictResolution::BothApplied);
    }

    #[test]
    fn test_conflict_resolution_local_dropped() {
        let local = make_op(
            0,
            EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 },
            "alice",
        );
        let remote = make_op(5, EditOp::DeleteNote { note_id: 1 }, "bob");

        let (conflict, _) = resolve_conflict(&local, &remote);
        assert_eq!(conflict.resolution, ConflictResolution::LocalDropped);
    }

    // -- Session Management --

    #[test]
    fn test_session_apply_local_and_push() {
        let mut session = EditSession::new("alice".to_string());

        session.apply_local(EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 });
        session.apply_local(EditOp::SetAmplitude { note_id: 2, amplitude: 0.5 });

        assert_eq!(session.pending_ops().len(), 2);
        assert_eq!(session.confirmed_ops().len(), 0);

        let msg = session.next_push_message().unwrap();
        if let SyncMessage::PushOperation { operation } = msg {
            assert_eq!(operation.author, "alice");
            if let EditOp::MovePitch { delta_semitones, .. } = operation.op {
                assert!((delta_semitones - 2.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_session_handle_ack() {
        let mut session = EditSession::new("alice".to_string());
        session.apply_local(EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 });
        session.apply_local(EditOp::SetAmplitude { note_id: 2, amplitude: 0.5 });

        session.handle_ack(10);
        assert_eq!(session.pending_ops().len(), 1);
        assert_eq!(session.confirmed_ops().len(), 1);
        assert_eq!(session.last_confirmed_op_id, 10);
    }

    #[test]
    fn test_session_rebase_with_remote_ops() {
        let mut session = EditSession::new("alice".to_string());

        // Alice makes a local edit.
        session.apply_local(EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 });

        // Bob's edit arrives from the server: move the same note.
        let remote = make_op(
            5,
            EditOp::MovePitch { note_id: 1, delta_semitones: -1.0 },
            "bob",
        );
        session.receive_remote_ops(&[remote]);

        // Alice's pending op should still be there (transformed).
        assert_eq!(session.pending_ops().len(), 1);
        // The remote op should now be confirmed.
        assert_eq!(session.confirmed_ops().len(), 1);
        assert_eq!(session.last_confirmed_op_id, 5);

        // A conflict should have been recorded.
        assert_eq!(session.conflicts.len(), 1);
        assert_eq!(session.conflicts[0].resolution, ConflictResolution::BothApplied);
    }

    #[test]
    fn test_session_rebase_own_op_confirmed() {
        let mut session = EditSession::new("alice".to_string());
        session.apply_local(EditOp::SetAmplitude { note_id: 1, amplitude: 0.5 });

        // Server confirms Alice's own operation.
        let confirmed = make_op(
            7,
            EditOp::SetAmplitude { note_id: 1, amplitude: 0.5 },
            "alice",
        );
        session.receive_remote_ops(&[confirmed]);

        // Pending should be cleared, confirmed should have the op.
        assert_eq!(session.pending_ops().len(), 0);
        assert_eq!(session.confirmed_ops().len(), 1);
        assert_eq!(session.last_confirmed_op_id, 7);
    }

    #[test]
    fn test_session_serialization_crash_recovery() {
        let mut session = EditSession::new("alice".to_string());
        session.apply_local(EditOp::MovePitch { note_id: 1, delta_semitones: 2.0 });
        session.handle_ack(1);
        session.apply_local(EditOp::SetAmplitude { note_id: 2, amplitude: 0.8 });

        let json = session.to_json().unwrap();
        let restored = EditSession::from_json(&json).unwrap();

        assert_eq!(restored.author, "alice");
        assert_eq!(restored.last_confirmed_op_id, 1);
        assert_eq!(restored.confirmed_ops().len(), 1);
        assert_eq!(restored.pending_ops().len(), 1);
    }

    // -- Sync Protocol Messages --

    #[test]
    fn test_sync_message_serialization() {
        let messages = vec![
            SyncMessage::SyncRequest { last_known_op_id: 42 },
            SyncMessage::SyncResponse {
                operations: vec![make_op(1, EditOp::DeleteNote { note_id: 5 }, "bob")],
            },
            SyncMessage::PushOperation {
                operation: make_op(0, EditOp::SetTempo { position: 0.0, bpm: 128.0 }, "alice"),
            },
            SyncMessage::AckOperation { op_id: 99 },
            SyncMessage::ModelSnapshot {
                state_json: "{\"notes\":[]}".to_string(),
                last_op_id: 50,
            },
        ];

        for msg in &messages {
            let bytes = msg.to_json().unwrap();
            let loaded = SyncMessage::from_json(&bytes).unwrap();
            // Verify round-trip by re-serializing.
            let bytes2 = loaded.to_json().unwrap();
            assert_eq!(bytes, bytes2);
        }
    }

    #[test]
    fn test_edit_op_target_note_ids() {
        assert_eq!(
            EditOp::MovePitch { note_id: 1, delta_semitones: 0.0 }.target_note_ids(),
            vec![1]
        );
        assert_eq!(
            EditOp::MergeNotes { note_ids: vec![1, 2, 3] }.target_note_ids(),
            vec![1, 2, 3]
        );
        assert!(EditOp::SetTempo { position: 0.0, bpm: 120.0 }
            .target_note_ids()
            .is_empty());
    }

    #[test]
    fn test_conflicts_with() {
        let a = EditOp::MovePitch { note_id: 1, delta_semitones: 1.0 };
        let b = EditOp::SetAmplitude { note_id: 1, amplitude: 0.5 };
        let c = EditOp::MovePitch { note_id: 2, delta_semitones: 1.0 };

        assert!(a.conflicts_with(&b)); // Same note.
        assert!(!a.conflicts_with(&c)); // Different notes.
    }

    #[test]
    fn test_operation_log_empty() {
        let log = OperationLog::new();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
        assert_eq!(log.last_op_id(), 0);
        assert!(log.replay_from(0).is_empty());
    }

    #[test]
    fn test_transform_list() {
        let ops = vec![
            EditOp::MovePitch { note_id: 1, delta_semitones: 1.0 },
            EditOp::MovePitch { note_id: 2, delta_semitones: 2.0 },
        ];
        let against = EditOp::DeleteNote { note_id: 1 };

        let transformed = transform_list(&ops, &against);
        // First op should be a no-op (note deleted), second unchanged.
        if let EditOp::MovePitch { delta_semitones, .. } = &transformed[0] {
            assert!(delta_semitones.abs() < 1e-10);
        }
        assert_eq!(transformed[1], ops[1]);
    }
}
