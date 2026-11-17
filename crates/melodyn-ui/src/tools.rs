//! Tool system for the Melodyn editor.
//!
//! Based on Melodyne's tool types: MUToolBoxTool, MUCombiTool, MUPitchTool,
//! MUFormantTool, MUAmplitudeTool, MUTimeTool, MUTimeStretchTool,
//! MUDetectionSeparationTool, and others.
//!
//! Each tool modifies different note properties when the user drags on a note
//! in the piano roll editor.

use crate::app::AppState;
use eframe::egui;

// ---------------------------------------------------------------------------
// Tool enum
// ---------------------------------------------------------------------------

/// The active editing tool, controlling how drag interactions affect notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tool {
    /// Combined tool: pitch on vertical drag, time on horizontal drag.
    Combi,
    /// Drag up/down to shift pitch in semitones.
    Pitch,
    /// Drag to adjust formant character independently of pitch.
    Formant,
    /// Drag up/down to change note amplitude.
    Amplitude,
    /// Drag left/right to move notes in time.
    Time,
    /// Drag left/right on note edges to stretch or compress duration.
    TimeStretch,
    /// Click to split detected notes at a point; shift-click to merge.
    Separation,
}

impl Tool {
    /// All available tools in toolbar order.
    pub const ALL: [Tool; 7] = [
        Tool::Combi,
        Tool::Pitch,
        Tool::Formant,
        Tool::Amplitude,
        Tool::Time,
        Tool::TimeStretch,
        Tool::Separation,
    ];

    /// Display label for the tool.
    pub fn label(self) -> &'static str {
        match self {
            Tool::Combi => "Combi",
            Tool::Pitch => "Pitch",
            Tool::Formant => "Formant",
            Tool::Amplitude => "Amplitude",
            Tool::Time => "Time",
            Tool::TimeStretch => "Stretch",
            Tool::Separation => "Separate",
        }
    }

    /// Keyboard shortcut hint.
    pub fn shortcut(self) -> &'static str {
        match self {
            Tool::Combi => "1",
            Tool::Pitch => "2",
            Tool::Formant => "3",
            Tool::Amplitude => "4",
            Tool::Time => "5",
            Tool::TimeStretch => "6",
            Tool::Separation => "7",
        }
    }

    /// Short description of what the tool does.
    pub fn tooltip(self) -> &'static str {
        match self {
            Tool::Combi => "Combined: pitch (vertical) + time (horizontal)",
            Tool::Pitch => "Drag up/down to shift pitch",
            Tool::Formant => "Drag to adjust formant character",
            Tool::Amplitude => "Drag up/down to change volume",
            Tool::Time => "Drag left/right to move in time",
            Tool::TimeStretch => "Drag to stretch/compress duration",
            Tool::Separation => "Click to split notes, Shift+click to merge",
        }
    }

    /// Description for undo stack.
    pub fn undo_description(self) -> &'static str {
        match self {
            Tool::Combi => "Move Notes",
            Tool::Pitch => "Adjust Pitch",
            Tool::Formant => "Adjust Formant",
            Tool::Amplitude => "Adjust Amplitude",
            Tool::Time => "Move in Time",
            Tool::TimeStretch => "Time Stretch",
            Tool::Separation => "Split/Merge Notes",
        }
    }
}

// ---------------------------------------------------------------------------
// Snap mode
// ---------------------------------------------------------------------------

/// Grid snapping mode for note edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapMode {
    /// No snapping; free movement.
    Off,
    /// Snap to beat grid positions.
    Beat,
    /// Snap to nearest semitone.
    Semitone,
    /// Snap to nearest scale degree (requires detected key).
    ScaleDegree,
}

impl SnapMode {
    pub const ALL: [SnapMode; 4] = [
        SnapMode::Off,
        SnapMode::Beat,
        SnapMode::Semitone,
        SnapMode::ScaleDegree,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SnapMode::Off => "Off",
            SnapMode::Beat => "Beat",
            SnapMode::Semitone => "Semitone",
            SnapMode::ScaleDegree => "Scale",
        }
    }
}

// ---------------------------------------------------------------------------
// Tool drag application
// ---------------------------------------------------------------------------

/// Apply a tool drag to the selected notes in the editor.
///
/// `delta` is the raw screen-space drag vector. The function converts it to
/// musical units (time and pitch) based on the current view scale, then
/// applies tool-specific modifications to each selected note.
pub fn apply_tool_drag(state: &mut AppState, delta: egui::Vec2) {
    let dt = delta.x as f64 / state.time_scale;
    let dp = -(delta.y as f64) / state.pitch_scale;

    let selected: Vec<usize> = state.selected_notes.iter().copied().collect();
    if selected.is_empty() {
        return;
    }

    for idx in selected {
        if idx >= state.notes.len() {
            continue;
        }
        match state.active_tool {
            Tool::Combi => {
                state.notes[idx].pitch_offset += dp;
                state.notes[idx].time_offset += dt;
            }
            Tool::Pitch => {
                state.notes[idx].pitch_offset += dp;
            }
            Tool::Time => {
                state.notes[idx].time_offset += dt;
            }
            Tool::Amplitude => {
                let scale_change = -delta.y as f64 * 0.003;
                state.notes[idx].amplitude_scale =
                    (state.notes[idx].amplitude_scale + scale_change).clamp(0.0, 4.0);
            }
            Tool::Formant => {
                state.notes[idx].formant_shift += dp * 0.5;
            }
            Tool::TimeStretch => {
                let scale_change = delta.x as f64 * 0.003;
                state.notes[idx].duration_scale =
                    (state.notes[idx].duration_scale + scale_change).clamp(0.1, 4.0);
            }
            Tool::Separation => {
                // Separation is handled on click, not drag.
            }
        }
    }
}

/// Apply snap to a pitch offset value.
pub fn snap_pitch(raw_midi: f64, mode: SnapMode, _detected_key_tonic: Option<u8>) -> f64 {
    match mode {
        SnapMode::Off => raw_midi,
        SnapMode::Beat => raw_midi,
        SnapMode::Semitone => raw_midi.round(),
        SnapMode::ScaleDegree => {
            // For scale snapping, round to nearest semitone for now.
            // A full implementation would use the detected scale intervals.
            raw_midi.round()
        }
    }
}

/// Apply snap to a time offset value.
pub fn snap_time(raw_time: f64, mode: SnapMode, state: &AppState) -> f64 {
    match mode {
        SnapMode::Off | SnapMode::Semitone | SnapMode::ScaleDegree => raw_time,
        SnapMode::Beat => {
            if let Some(ref grid) = state.beat_grid {
                // Find nearest beat
                let mut best_time = raw_time;
                let mut best_dist = f64::MAX;
                for beat in &grid.beats {
                    let dist = (beat.time - raw_time).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_time = beat.time;
                    }
                }
                best_time
            } else {
                raw_time
            }
        }
    }
}

/// Handle the separation tool click: split the note at the cursor time.
pub fn handle_separation_click(state: &mut AppState, note_idx: usize, click_time: f64) {
    if note_idx >= state.notes.len() {
        return;
    }

    let note = &state.notes[note_idx];
    let start = note.effective_start();
    let end = start + note.effective_duration();

    // Only split if click is within the note and not too close to edges
    let margin = note.effective_duration() * 0.1;
    if click_time <= start + margin || click_time >= end - margin {
        return;
    }

    // Calculate split point relative to original note
    let split_ratio = (click_time - start) / (end - start);
    let orig = note.clone();

    // First half: same start, shorter duration
    let mut first = orig.clone();
    first.duration_scale *= split_ratio;

    // Second half: later start, shorter duration
    let mut second = orig.clone();
    second.time_offset += note.effective_duration() * split_ratio;
    second.duration_scale *= 1.0 - split_ratio;

    // Replace original with the two halves
    state.notes[note_idx] = first;
    state.notes.insert(note_idx + 1, second);

    // Update selection
    state.selected_notes.clear();
    state.selected_notes.insert(note_idx);
    state.selected_notes.insert(note_idx + 1);
}

// ---------------------------------------------------------------------------
// Toolbar rendering
// ---------------------------------------------------------------------------

/// Draw the toolbar with tool selection buttons and snap mode.
pub fn toolbar(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        ui.label("Tools:");

        for tool in Tool::ALL {
            let is_active = state.active_tool == tool;
            let button = egui::Button::new(tool.label())
                .selected(is_active);
            let response = ui.add(button).on_hover_text(
                format!("{} [{}]", tool.tooltip(), tool.shortcut()),
            );
            if response.clicked() {
                state.active_tool = tool;
            }
        }

        ui.separator();
        ui.label("Snap:");

        for mode in SnapMode::ALL {
            let is_active = state.snap_mode == mode;
            let button = egui::Button::new(mode.label()).selected(is_active);
            if ui.add(button).clicked() {
                state.snap_mode = mode;
            }
        }

        ui.separator();

        // Zoom controls
        if ui.button("Zoom In").on_hover_text("Ctrl +").clicked() {
            state.time_scale = (state.time_scale * 1.25).min(2000.0);
        }
        if ui.button("Zoom Out").on_hover_text("Ctrl -").clicked() {
            state.time_scale = (state.time_scale / 1.25).max(20.0);
        }
        if ui.button("Fit").on_hover_text("Fit to window").clicked() {
            state.time_offset = 0.0;
            if state.duration > 0.0 {
                // Will be adjusted based on editor width in next frame
                state.time_scale = 100.0;
            }
        }
    });
}

/// Handle keyboard shortcuts for tool selection.
pub fn handle_tool_shortcuts(ctx: &egui::Context, state: &mut AppState) {
    ctx.input(|i| {
        if i.key_pressed(egui::Key::Num1) {
            state.active_tool = Tool::Combi;
        }
        if i.key_pressed(egui::Key::Num2) {
            state.active_tool = Tool::Pitch;
        }
        if i.key_pressed(egui::Key::Num3) {
            state.active_tool = Tool::Formant;
        }
        if i.key_pressed(egui::Key::Num4) {
            state.active_tool = Tool::Amplitude;
        }
        if i.key_pressed(egui::Key::Num5) {
            state.active_tool = Tool::Time;
        }
        if i.key_pressed(egui::Key::Num6) {
            state.active_tool = Tool::TimeStretch;
        }
        if i.key_pressed(egui::Key::Num7) {
            state.active_tool = Tool::Separation;
        }
    });
}
