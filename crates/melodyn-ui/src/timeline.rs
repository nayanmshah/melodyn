//! Timeline ruler displayed above the piano roll editor.
//!
//! Based on MUTimelineRulerView and MUSignatureEditorView from the original
//! Melodyne engine. Shows time markers (seconds or bar:beat), tempo, and
//! key/scale information aligned with the editor's horizontal scroll.

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};

use crate::app::AppState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BG_COLOR: Color32 = Color32::from_rgb(38, 38, 48);
const TICK_COLOR: Color32 = Color32::from_rgb(120, 120, 140);
const MAJOR_TICK: Color32 = Color32::from_rgb(160, 160, 180);
const TEXT_COLOR: Color32 = Color32::from_rgb(180, 180, 200);
const CURSOR_COLOR: Color32 = Color32::from_rgb(255, 70, 70);
const RULER_HEIGHT: f32 = 28.0;

// ---------------------------------------------------------------------------
// Timeline ruler
// ---------------------------------------------------------------------------

/// Draw the timeline ruler above the editor.
///
/// Allocates a fixed-height strip and draws time/beat markers synchronized
/// with the editor's horizontal scroll and zoom.
pub fn timeline_ruler(ui: &mut egui::Ui, state: &AppState) {
    let desired = egui::vec2(ui.available_width(), RULER_HEIGHT);
    let (rect, _response) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // Background
    painter.rect_filled(rect, 0.0, BG_COLOR);

    // Bottom edge line
    painter.line_segment(
        [
            Pos2::new(rect.left(), rect.bottom()),
            Pos2::new(rect.right(), rect.bottom()),
        ],
        Stroke::new(1.0_f32, Color32::from_rgb(60, 60, 70)),
    );

    // Skip piano-width area on the left to align with editor
    let piano_width = 50.0;
    let ruler_rect = Rect::from_min_max(Pos2::new(rect.left() + piano_width, rect.top()), rect.max);

    if state.beat_grid.is_some() {
        draw_beat_ruler(&painter, ruler_rect, state);
    } else {
        draw_time_ruler(&painter, ruler_rect, state);
    }

    // Playback cursor
    let cursor_x = ruler_rect.left()
        + ((state.playback_position - state.time_offset) * state.time_scale) as f32;
    if cursor_x >= ruler_rect.left() && cursor_x <= ruler_rect.right() {
        painter.line_segment(
            [
                Pos2::new(cursor_x, rect.top()),
                Pos2::new(cursor_x, rect.bottom()),
            ],
            Stroke::new(1.5_f32, CURSOR_COLOR),
        );
        // Small triangle
        let tri = 4.0;
        painter.add(egui::Shape::convex_polygon(
            vec![
                Pos2::new(cursor_x, rect.bottom()),
                Pos2::new(cursor_x - tri, rect.bottom() + tri),
                Pos2::new(cursor_x + tri, rect.bottom() + tri),
            ],
            CURSOR_COLOR,
            Stroke::NONE,
        ));
    }

    // Key / tempo labels on the left (in the piano-width zone)
    let label_rect = Rect::from_min_max(
        rect.min,
        Pos2::new(rect.left() + piano_width, rect.bottom()),
    );

    if let Some(tempo) = state.detected_tempo {
        painter.text(
            Pos2::new(label_rect.left() + 4.0, label_rect.top() + 8.0),
            egui::Align2::LEFT_TOP,
            format!("{:.0}", tempo),
            egui::FontId::proportional(9.0),
            TEXT_COLOR,
        );
    }
    if let Some(ref key) = state.detected_key {
        // Truncate key name for small space
        let short = if key.len() > 6 { &key[..6] } else { key };
        painter.text(
            Pos2::new(label_rect.left() + 4.0, label_rect.bottom() - 8.0),
            egui::Align2::LEFT_BOTTOM,
            short,
            egui::FontId::proportional(9.0),
            TEXT_COLOR,
        );
    }
}

// ---------------------------------------------------------------------------
// Time-based ruler (seconds)
// ---------------------------------------------------------------------------

/// Draw ruler ticks in seconds when no beat grid is available.
fn draw_time_ruler(painter: &egui::Painter, rect: Rect, state: &AppState) {
    let t_start = state.time_offset - 0.5;
    let t_end = state.time_offset + rect.width() as f64 / state.time_scale + 0.5;

    let interval = choose_time_interval(t_end - t_start, rect.width());
    let sub_interval = interval / 4.0;

    // Major ticks
    let mut t = (t_start / interval).floor() * interval;
    while t <= t_end {
        if t >= 0.0 {
            let x = rect.left() + ((t - state.time_offset) * state.time_scale) as f32;
            if x >= rect.left() && x <= rect.right() {
                // Major tick line
                painter.line_segment(
                    [
                        Pos2::new(x, rect.bottom() - 10.0),
                        Pos2::new(x, rect.bottom()),
                    ],
                    Stroke::new(1.0_f32, MAJOR_TICK),
                );

                // Label
                let label = format_time_label(t, interval);
                painter.text(
                    Pos2::new(x + 3.0, rect.top() + 4.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::proportional(10.0),
                    TEXT_COLOR,
                );
            }
        }
        t += interval;
    }

    // Minor ticks
    t = (t_start / sub_interval).floor() * sub_interval;
    while t <= t_end {
        if t >= 0.0 {
            let x = rect.left() + ((t - state.time_offset) * state.time_scale) as f32;
            if x >= rect.left() && x <= rect.right() {
                painter.line_segment(
                    [
                        Pos2::new(x, rect.bottom() - 5.0),
                        Pos2::new(x, rect.bottom()),
                    ],
                    Stroke::new(0.5_f32, TICK_COLOR),
                );
            }
        }
        t += sub_interval;
    }
}

// ---------------------------------------------------------------------------
// Beat-based ruler
// ---------------------------------------------------------------------------

/// Draw ruler ticks aligned to the beat grid.
fn draw_beat_ruler(painter: &egui::Painter, rect: Rect, state: &AppState) {
    let grid = match state.beat_grid {
        Some(ref g) => g,
        None => return,
    };

    for beat in &grid.beats {
        let x = rect.left() + ((beat.time - state.time_offset) * state.time_scale) as f32;
        if x < rect.left() || x > rect.right() {
            continue;
        }

        if beat.is_downbeat {
            // Major tick + bar number
            painter.line_segment(
                [
                    Pos2::new(x, rect.bottom() - 12.0),
                    Pos2::new(x, rect.bottom()),
                ],
                Stroke::new(1.0_f32, MAJOR_TICK),
            );

            let label = format!("{}", beat.bar + 1);
            painter.text(
                Pos2::new(x + 3.0, rect.top() + 4.0),
                egui::Align2::LEFT_TOP,
                label,
                egui::FontId::proportional(11.0),
                TEXT_COLOR,
            );
        } else {
            // Minor tick + beat number
            painter.line_segment(
                [
                    Pos2::new(x, rect.bottom() - 6.0),
                    Pos2::new(x, rect.bottom()),
                ],
                Stroke::new(0.5_f32, TICK_COLOR),
            );

            // Show beat number if zoomed in enough
            let px_per_beat = (60.0 / grid.bpm) * state.time_scale;
            if px_per_beat > 30.0 {
                let label = format!("{}", beat.beat_in_bar + 1);
                painter.text(
                    Pos2::new(x + 2.0, rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    label,
                    egui::FontId::proportional(9.0),
                    Color32::from_rgb(120, 120, 140),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Choose a readable time interval for ruler ticks.
fn choose_time_interval(visible_range: f64, width_px: f32) -> f64 {
    let target_spacing = 100.0;
    let raw = visible_range * target_spacing / width_px as f64;
    let intervals = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0];
    for &iv in &intervals {
        if iv >= raw {
            return iv;
        }
    }
    60.0
}

/// Format a time value as a readable label.
fn format_time_label(seconds: f64, interval: f64) -> String {
    if interval < 1.0 {
        format!("{:.1}s", seconds)
    } else if seconds < 60.0 {
        format!("{:.0}s", seconds)
    } else {
        let m = (seconds / 60.0).floor() as u32;
        let s = seconds % 60.0;
        format!("{}:{:02.0}", m, s)
    }
}
