//! Piano roll / note editor -- the core editing view.
//!
//! Based on MUEditorView, MUEditorNoteView, MUEditorScrollView, and
//! MUEditorSubview from the original Melodyne engine.
//!
//! Layout:
//! - Left strip: piano keyboard (vertical pitch labels)
//! - Main area: note blobs with pitch contour, beat grid overlay, selection
//! - Interactions: click/shift-click select, drag to edit, scroll/zoom

use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

use crate::app::{AppState, EditableNote};
use crate::tools::{self, Tool};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PIANO_WIDTH: f32 = 50.0;
const NOTE_MIN_HEIGHT_FRAC: f64 = 0.45;

/// Background color for the editor area.
const BG_COLOR: Color32 = Color32::from_rgb(28, 28, 38);
/// Piano white key color.
const PIANO_WHITE: Color32 = Color32::from_rgb(230, 230, 230);
/// Piano black key color.
const PIANO_BLACK: Color32 = Color32::from_rgb(45, 45, 50);
/// Piano separator line.
const PIANO_LINE: Color32 = Color32::from_rgb(60, 60, 70);
/// Grid line color (semitone).
const GRID_LINE: Color32 = Color32::from_rgba_premultiplied(55, 55, 70, 128);
/// Beat grid line.
const BEAT_LINE: Color32 = Color32::from_rgba_premultiplied(70, 70, 90, 150);
/// Downbeat grid line.
const DOWNBEAT_LINE: Color32 = Color32::from_rgba_premultiplied(110, 110, 130, 200);
/// Note body fill (unselected).
const NOTE_FILL: Color32 = Color32::from_rgb(210, 140, 60);
/// Note body fill (selected).
const NOTE_FILL_SELECTED: Color32 = Color32::from_rgb(255, 190, 80);
/// Note body fill (hovered).
const NOTE_FILL_HOVERED: Color32 = Color32::from_rgb(240, 170, 70);
/// Note outline (selected).
const NOTE_OUTLINE_SELECTED: Color32 = Color32::from_rgb(255, 255, 255);
/// Pitch contour line inside a note.
const CONTOUR_COLOR: Color32 = Color32::from_rgb(160, 90, 25);
/// Playback cursor color.
const CURSOR_COLOR: Color32 = Color32::from_rgb(255, 70, 70);

// ---------------------------------------------------------------------------
// Coordinate conversion
// ---------------------------------------------------------------------------

/// Convert time (seconds) to screen X coordinate within the editor rect.
fn time_to_x(time: f64, rect: Rect, state: &AppState) -> f32 {
    rect.left() + ((time - state.time_offset) * state.time_scale) as f32
}

/// Convert screen X coordinate to time (seconds).
fn x_to_time(x: f32, rect: Rect, state: &AppState) -> f64 {
    state.time_offset + (x - rect.left()) as f64 / state.time_scale
}

/// Convert MIDI note number to screen Y coordinate (higher pitch = higher up).
fn midi_to_y(midi: f64, rect: Rect, state: &AppState) -> f32 {
    rect.bottom() - ((midi - state.pitch_offset) * state.pitch_scale) as f32
}

/// Convert screen Y coordinate to MIDI note number.
#[allow(dead_code)]
fn y_to_midi(y: f32, rect: Rect, state: &AppState) -> f64 {
    state.pitch_offset + (rect.bottom() - y) as f64 / state.pitch_scale
}

/// Compute the screen rectangle for a note blob.
fn note_screen_rect(note: &EditableNote, rect: Rect, state: &AppState) -> Rect {
    let x_start = time_to_x(note.effective_start(), rect, state);
    let x_end = time_to_x(note.effective_end(), rect, state);
    let y_center = midi_to_y(note.effective_midi(), rect, state);
    let half_h = (state.pitch_scale * NOTE_MIN_HEIGHT_FRAC) as f32;

    Rect::from_min_max(
        Pos2::new(x_start, y_center - half_h),
        Pos2::new(x_end.max(x_start + 3.0), y_center + half_h),
    )
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

/// Find the topmost note at a screen position, returning its index.
fn find_note_at(pos: Pos2, rect: Rect, state: &AppState) -> Option<usize> {
    // Iterate in reverse so topmost (last drawn) is found first.
    for (i, note) in state.notes.iter().enumerate().rev() {
        let nr = note_screen_rect(note, rect, state);
        if nr.contains(pos) {
            return Some(i);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

/// Draw the piano keyboard strip on the left side.
fn draw_piano_keyboard(painter: &egui::Painter, rect: Rect, state: &AppState) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(50, 50, 55));

    let bottom_midi = state.pitch_offset.floor() as i32 - 1;
    let top_midi =
        (state.pitch_offset + rect.height() as f64 / state.pitch_scale).ceil() as i32 + 1;

    let black_notes: [i32; 5] = [1, 3, 6, 8, 10]; // C#, D#, F#, G#, A#

    for midi in bottom_midi..=top_midi {
        let y = midi_to_y(midi as f64, rect, state);
        let y_top = midi_to_y(midi as f64 + 0.5, rect, state);
        let y_bot = midi_to_y(midi as f64 - 0.5, rect, state);

        if y_bot < rect.top() || y_top > rect.bottom() {
            continue;
        }

        let pc = ((midi % 12) + 12) % 12;
        let is_black = black_notes.contains(&pc);

        let key_rect = Rect::from_min_max(
            Pos2::new(rect.left(), y_top.max(rect.top())),
            Pos2::new(rect.right(), y_bot.min(rect.bottom())),
        );

        let fill = if is_black { PIANO_BLACK } else { PIANO_WHITE };
        painter.rect_filled(key_rect, 0.0, fill);
        painter.line_segment(
            [
                Pos2::new(rect.left(), y_bot),
                Pos2::new(rect.right(), y_bot),
            ],
            Stroke::new(0.5_f32, PIANO_LINE),
        );

        // Label C notes
        if pc == 0 {
            let octave = midi / 12 - 1;
            let label = format!("C{}", octave);
            let text_color = if is_black {
                Color32::WHITE
            } else {
                Color32::from_rgb(40, 40, 50)
            };
            painter.text(
                Pos2::new(rect.left() + 4.0, y),
                egui::Align2::LEFT_CENTER,
                label,
                egui::FontId::proportional(10.0),
                text_color,
            );
        }
    }
}

/// Draw the pitch and beat grid in the editor area.
fn draw_grid(painter: &egui::Painter, rect: Rect, state: &AppState) {
    // Background
    painter.rect_filled(rect, 0.0, BG_COLOR);

    // Horizontal lines: one per semitone
    let bottom_midi = state.pitch_offset.floor() as i32 - 1;
    let top_midi =
        (state.pitch_offset + rect.height() as f64 / state.pitch_scale).ceil() as i32 + 1;

    let black_notes: [i32; 5] = [1, 3, 6, 8, 10];

    for midi in bottom_midi..=top_midi {
        let y = midi_to_y(midi as f64, rect, state);
        if y < rect.top() || y > rect.bottom() {
            continue;
        }
        let pc = ((midi % 12) + 12) % 12;
        // Slightly shade black-key rows
        if black_notes.contains(&pc) {
            let y_top = midi_to_y(midi as f64 + 0.5, rect, state).max(rect.top());
            let y_bot = midi_to_y(midi as f64 - 0.5, rect, state).min(rect.bottom());
            let shade_rect = Rect::from_min_max(
                Pos2::new(rect.left(), y_top),
                Pos2::new(rect.right(), y_bot),
            );
            painter.rect_filled(
                shade_rect,
                0.0,
                Color32::from_rgba_premultiplied(0, 0, 0, 20),
            );
        }
        // C notes get a brighter line
        let stroke = if pc == 0 {
            Stroke::new(1.0_f32, BEAT_LINE)
        } else {
            Stroke::new(0.5_f32, GRID_LINE)
        };
        painter.line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            stroke,
        );
    }

    // Vertical lines: beat grid (if available)
    if let Some(ref grid) = state.beat_grid {
        for beat in &grid.beats {
            let x = time_to_x(beat.time, rect, state);
            if x < rect.left() || x > rect.right() {
                continue;
            }
            let stroke = if beat.is_downbeat {
                Stroke::new(1.0_f32, DOWNBEAT_LINE)
            } else {
                Stroke::new(0.5_f32, BEAT_LINE)
            };
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                stroke,
            );
        }
    } else {
        // No beat grid: draw second markers
        let t_start = x_to_time(rect.left(), rect, state);
        let t_end = x_to_time(rect.right(), rect, state);
        let interval = choose_time_interval(t_end - t_start, rect.width());

        let mut t = (t_start / interval).floor() * interval;
        while t <= t_end {
            if t >= 0.0 {
                let x = time_to_x(t, rect, state);
                painter.line_segment(
                    [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                    Stroke::new(0.5_f32, GRID_LINE),
                );
            }
            t += interval;
        }
    }
}

/// Choose a readable time interval for grid lines based on visible range.
fn choose_time_interval(visible_range: f64, width_px: f32) -> f64 {
    let target_spacing = 80.0; // pixels between grid lines
    let raw = visible_range * target_spacing / width_px as f64;
    let intervals = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0];
    for &iv in &intervals {
        if iv >= raw {
            return iv;
        }
    }
    60.0
}

/// Draw all note blobs with pitch contour visualization.
fn draw_notes(painter: &egui::Painter, rect: Rect, state: &AppState) {
    for (i, note) in state.notes.iter().enumerate() {
        let nr = note_screen_rect(note, rect, state);

        // Skip if off-screen
        if nr.right() < rect.left() || nr.left() > rect.right() {
            continue;
        }
        if nr.bottom() < rect.top() || nr.top() > rect.bottom() {
            continue;
        }

        let is_selected = state.selected_notes.contains(&i);
        let is_hovered = state.hovered_note == Some(i);

        // Note body opacity scales with amplitude
        let alpha = (note.effective_amplitude() * 180.0 + 75.0).min(255.0) as u8;

        let base = if is_selected {
            NOTE_FILL_SELECTED
        } else if is_hovered {
            NOTE_FILL_HOVERED
        } else {
            NOTE_FILL
        };
        let fill = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha);

        // Draw rounded rect
        painter.rect_filled(nr, 3.0, fill);

        // Selected outline
        if is_selected {
            painter.rect_stroke(nr, 3.0, Stroke::new(1.5_f32, NOTE_OUTLINE_SELECTED));
        }

        // Draw pitch contour line inside the note
        if !note.detected.pitch_contour.is_empty() && nr.width() > 8.0 {
            let contour = &note.detected.pitch_contour;
            let n = contour.len();
            let step_x = nr.width() / n.max(1) as f32;

            for j in 0..n.saturating_sub(1) {
                let freq1 = contour[j];
                let freq2 = contour[j + 1];
                let midi1 = melodyn_core::PitchCandidate::freq_to_midi(freq1) + note.pitch_offset;
                let midi2 = melodyn_core::PitchCandidate::freq_to_midi(freq2) + note.pitch_offset;

                let x1 = nr.left() + j as f32 * step_x;
                let x2 = nr.left() + (j + 1) as f32 * step_x;
                let y1 = midi_to_y(midi1, rect, state);
                let y2 = midi_to_y(midi2, rect, state);

                // Clamp to note rect
                let y1c = y1.clamp(nr.top(), nr.bottom());
                let y2c = y2.clamp(nr.top(), nr.bottom());

                painter.line_segment(
                    [Pos2::new(x1, y1c), Pos2::new(x2, y2c)],
                    Stroke::new(1.2_f32, CONTOUR_COLOR),
                );
            }
        }

        // Draw pitch center line
        let y_center = midi_to_y(note.effective_midi(), rect, state);
        if y_center >= nr.top() && y_center <= nr.bottom() {
            painter.line_segment(
                [
                    Pos2::new(nr.left() + 2.0, y_center),
                    Pos2::new(nr.right() - 2.0, y_center),
                ],
                Stroke::new(0.5_f32, Color32::from_rgba_premultiplied(255, 255, 255, 60)),
            );
        }
    }
}

/// Draw the playback position cursor.
fn draw_playback_cursor(painter: &egui::Painter, rect: Rect, state: &AppState) {
    let x = time_to_x(state.playback_position, rect, state);
    if x >= rect.left() && x <= rect.right() {
        painter.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(1.5_f32, CURSOR_COLOR),
        );
        // Small triangle at top
        let tri_size = 5.0;
        let tri = vec![
            Pos2::new(x, rect.top()),
            Pos2::new(x - tri_size, rect.top() - tri_size),
            Pos2::new(x + tri_size, rect.top() - tri_size),
        ];
        painter.add(egui::Shape::convex_polygon(tri, CURSOR_COLOR, Stroke::NONE));
    }
}

// ---------------------------------------------------------------------------
// Interaction handling
// ---------------------------------------------------------------------------

/// Handle mouse interactions in the editor area.
fn handle_editor_input(response: &egui::Response, rect: Rect, state: &mut AppState, ui: &egui::Ui) {
    let modifiers = ui.input(|i| i.modifiers);

    // Update hovered note
    state.hovered_note = response
        .hover_pos()
        .and_then(|pos| find_note_at(pos, rect, state));

    // Click to select
    if response.clicked() {
        if state.active_tool == Tool::Separation {
            // Separation tool: split note at click position
            if let (Some(idx), Some(pos)) = (state.hovered_note, response.interact_pointer_pos()) {
                let click_time = x_to_time(pos.x, rect, state);
                state.push_undo("Split Note");
                tools::handle_separation_click(state, idx, click_time);
            }
        } else if let Some(idx) = state.hovered_note {
            if !modifiers.shift {
                state.selected_notes.clear();
            }
            if state.selected_notes.contains(&idx) && modifiers.shift {
                state.selected_notes.remove(&idx);
            } else {
                state.selected_notes.insert(idx);
            }
        } else {
            if !modifiers.shift {
                state.selected_notes.clear();
            }
            // Click on empty space: set playback position
            if let Some(pos) = response.interact_pointer_pos() {
                state.playback_position = x_to_time(pos.x, rect, state).max(0.0);
            }
        }
    }

    // Drag start: save undo and set drag state
    if response.drag_started() {
        if let Some(idx) = state.hovered_note {
            if !state.selected_notes.contains(&idx) {
                state.selected_notes.clear();
                state.selected_notes.insert(idx);
            }
            state.push_undo(state.active_tool.undo_description());
            state.drag_note_origins = state
                .selected_notes
                .iter()
                .map(|&i| {
                    if i < state.notes.len() {
                        (
                            state.notes[i].effective_midi(),
                            state.notes[i].effective_start(),
                        )
                    } else {
                        (0.0, 0.0)
                    }
                })
                .collect();
            state.dragging = true;
        }
    }

    // Dragging: apply tool
    if response.dragged() && state.dragging && !state.selected_notes.is_empty() {
        let delta = response.drag_delta();
        tools::apply_tool_drag(state, delta);
    }

    // Drag end
    if response.drag_stopped() {
        // Apply snap after drag
        if state.snap_mode != crate::tools::SnapMode::Off {
            let selected: Vec<usize> = state.selected_notes.iter().copied().collect();
            for idx in selected {
                if idx < state.notes.len() {
                    let raw_midi = state.notes[idx].effective_midi();
                    let snapped =
                        tools::snap_pitch(raw_midi, state.snap_mode, state.detected_key_tonic);
                    state.notes[idx].pitch_offset += snapped - raw_midi;

                    let raw_time = state.notes[idx].effective_start();
                    let snapped_t = tools::snap_time(raw_time, state.snap_mode, state);
                    state.notes[idx].time_offset += snapped_t - raw_time;
                }
            }
        }
        state.dragging = false;
        state.drag_note_origins.clear();
    }

    // Scroll and zoom
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta);

        if scroll != Vec2::ZERO {
            if modifiers.command {
                // Ctrl+scroll: horizontal zoom toward cursor
                let factor = 1.0 + (scroll.y / 300.0) as f64;
                let factor = factor.clamp(0.5, 2.0);
                if let Some(cursor) = response.hover_pos() {
                    let time_at_cursor = x_to_time(cursor.x, rect, state);
                    state.time_scale = (state.time_scale * factor).clamp(20.0, 2000.0);
                    // Adjust offset so cursor stays at same time
                    state.time_offset =
                        time_at_cursor - (cursor.x - rect.left()) as f64 / state.time_scale;
                }
            } else {
                // Scroll: vertical = pitch, horizontal/shift = time
                state.pitch_offset += (scroll.y as f64) / state.pitch_scale;
                state.time_offset -= (scroll.x as f64) / state.time_scale;
            }

            // Clamp view
            state.time_offset = state.time_offset.max(-1.0);
            state.pitch_offset = state.pitch_offset.clamp(0.0, 127.0);
        }
    }

    // Set cursor style
    if state.hovered_note.is_some() {
        match state.active_tool {
            Tool::Pitch => {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
            }
            Tool::Time | Tool::TimeStretch => {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
            Tool::Combi => {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Move);
            }
            Tool::Separation => {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Cell);
            }
            _ => {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Draw the complete piano roll editor view.
///
/// Allocates a horizontal layout with a piano keyboard strip on the left
/// and the main note editor area on the right. Handles all rendering and
/// mouse interaction.
pub fn piano_roll_editor(ui: &mut egui::Ui, state: &mut AppState) {
    let available = ui.available_size();
    let editor_width = (available.x - PIANO_WIDTH).max(100.0);
    let height = available.y.max(100.0);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);

        // Piano keyboard strip
        let (piano_resp, piano_painter) =
            ui.allocate_painter(egui::vec2(PIANO_WIDTH, height), egui::Sense::hover());
        draw_piano_keyboard(&piano_painter, piano_resp.rect, state);

        // Main editor area
        let (editor_resp, editor_painter) = ui.allocate_painter(
            egui::vec2(editor_width, height),
            egui::Sense::click_and_drag(),
        );
        let editor_rect = editor_resp.rect;

        draw_grid(&editor_painter, editor_rect, state);
        draw_notes(&editor_painter, editor_rect, state);
        draw_playback_cursor(&editor_painter, editor_rect, state);
        handle_editor_input(&editor_resp, editor_rect, state, ui);
    });
}
