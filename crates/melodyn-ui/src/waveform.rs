//! Waveform overview display.
//!
//! Shows the full audio waveform at a reduced resolution with a visible
//! region indicator that syncs with the main editor's scroll position.
//! Click to seek, drag to scroll the editor view.
//!
//! Based on MUFunctionView / waveform display in the original Melodyne.

use eframe::egui::{self, Color32, Pos2, Rect, Stroke};

use crate::app::AppState;

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

const BG_COLOR: Color32 = Color32::from_rgb(24, 24, 32);
const WAVE_COLOR: Color32 = Color32::from_rgb(80, 160, 220);
const WAVE_FILL: Color32 = Color32::from_rgba_premultiplied(60, 120, 180, 100);
const REGION_FILL: Color32 = Color32::from_rgba_premultiplied(255, 255, 255, 25);
const REGION_BORDER: Color32 = Color32::from_rgba_premultiplied(255, 255, 255, 100);
const CURSOR_COLOR: Color32 = Color32::from_rgb(255, 70, 70);
const ZERO_LINE: Color32 = Color32::from_rgba_premultiplied(80, 80, 100, 100);

// ---------------------------------------------------------------------------
// Waveform overview
// ---------------------------------------------------------------------------

/// Draw the waveform overview panel.
///
/// Shows the full waveform with a highlight overlay indicating the region
/// currently visible in the main editor. Click to seek; drag to scroll.
pub fn waveform_overview(ui: &mut egui::Ui, state: &mut AppState) {
    let available = ui.available_size();
    let (response, painter) =
        ui.allocate_painter(available, egui::Sense::click_and_drag());
    let rect = response.rect;

    // Background
    painter.rect_filled(rect, 2.0, BG_COLOR);

    if state.waveform_peaks.is_empty() || state.duration <= 0.0 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No audio loaded",
            egui::FontId::proportional(12.0),
            Color32::from_rgb(100, 100, 120),
        );
        return;
    }

    // Zero line
    let y_center = rect.center().y;
    painter.line_segment(
        [Pos2::new(rect.left(), y_center), Pos2::new(rect.right(), y_center)],
        Stroke::new(0.5_f32, ZERO_LINE),
    );

    // Draw waveform peaks
    let num_peaks = state.waveform_peaks.len();
    let px_per_peak = rect.width() / num_peaks as f32;
    let half_height = rect.height() * 0.45;

    for (i, &(min, max)) in state.waveform_peaks.iter().enumerate() {
        let x = rect.left() + (i as f32 + 0.5) * px_per_peak;

        // Filled area
        let y_min = y_center - max * half_height;
        let y_max = y_center - min * half_height;

        if px_per_peak > 1.0 {
            let peak_rect = Rect::from_min_max(
                Pos2::new(x - px_per_peak * 0.4, y_min),
                Pos2::new(x + px_per_peak * 0.4, y_max),
            );
            painter.rect_filled(peak_rect, 0.0, WAVE_FILL);
        }

        // Outline
        painter.line_segment(
            [Pos2::new(x, y_min), Pos2::new(x, y_max)],
            Stroke::new(px_per_peak.min(1.5), WAVE_COLOR),
        );
    }

    // Visible region overlay
    let region_x_start = time_to_overview_x(state.time_offset, rect, state.duration);
    let visible_seconds = rect.width() as f64 / state.time_scale;
    let region_x_end =
        time_to_overview_x(state.time_offset + visible_seconds, rect, state.duration);

    let region_rect = Rect::from_min_max(
        Pos2::new(region_x_start.max(rect.left()), rect.top()),
        Pos2::new(region_x_end.min(rect.right()), rect.bottom()),
    );
    painter.rect_filled(region_rect, 0.0, REGION_FILL);
    painter.rect_stroke(region_rect, 0.0, Stroke::new(1.0_f32, REGION_BORDER));

    // Playback cursor
    let cursor_x = time_to_overview_x(state.playback_position, rect, state.duration);
    if cursor_x >= rect.left() && cursor_x <= rect.right() {
        painter.line_segment(
            [Pos2::new(cursor_x, rect.top()), Pos2::new(cursor_x, rect.bottom())],
            Stroke::new(1.5_f32, CURSOR_COLOR),
        );
    }

    // Interaction: click to seek, drag to scroll
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let time = overview_x_to_time(pos.x, rect, state.duration);
            state.playback_position = time.clamp(0.0, state.duration);
        }
    }

    if response.dragged() {
        if let Some(pos) = response.interact_pointer_pos() {
            let time = overview_x_to_time(pos.x, rect, state.duration);
            // Center the editor view on the dragged position
            let visible_seconds = rect.width() as f64 / state.time_scale;
            state.time_offset = (time - visible_seconds * 0.5).max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Coordinate conversion
// ---------------------------------------------------------------------------

/// Convert a time (seconds) to X position in the overview rect.
fn time_to_overview_x(time: f64, rect: Rect, duration: f64) -> f32 {
    if duration <= 0.0 {
        return rect.left();
    }
    rect.left() + (time / duration * rect.width() as f64) as f32
}

/// Convert an X position in the overview rect to time (seconds).
fn overview_x_to_time(x: f32, rect: Rect, duration: f64) -> f64 {
    if rect.width() <= 0.0 {
        return 0.0;
    }
    ((x - rect.left()) as f64 / rect.width() as f64) * duration
}
