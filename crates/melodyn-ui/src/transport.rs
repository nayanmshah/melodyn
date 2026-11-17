//! Transport / playback bar.
//!
//! Based on MUPlaybackView and the transport controls in the original
//! Melodyne engine. Provides play/stop buttons, position display, tempo,
//! loop controls, and a master volume slider.

use eframe::egui;

use crate::app::AppState;

// ---------------------------------------------------------------------------
// Time formatting
// ---------------------------------------------------------------------------

/// Format a time in seconds as "M:SS.mmm" (minutes, seconds, milliseconds).
fn format_time(seconds: f64) -> String {
    if seconds < 0.0 {
        return "0:00.000".to_string();
    }
    let mins = (seconds / 60.0).floor() as u32;
    let secs = seconds % 60.0;
    format!("{}:{:05.3}", mins, secs)
}

/// Format a time in seconds as "bar:beat" given a tempo.
fn format_bar_beat(seconds: f64, bpm: f64, beats_per_bar: u32) -> String {
    if bpm <= 0.0 || beats_per_bar == 0 {
        return "--:--".to_string();
    }
    let total_beats = seconds * bpm / 60.0;
    let bar = (total_beats / beats_per_bar as f64).floor() as u32 + 1;
    let beat = (total_beats % beats_per_bar as f64).floor() as u32 + 1;
    format!("{}:{}", bar, beat)
}

// ---------------------------------------------------------------------------
// Transport bar rendering
// ---------------------------------------------------------------------------

/// Draw the transport bar with playback controls, position, and volume.
pub fn transport_bar(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;

        // Go to start
        if ui
            .button("|<")
            .on_hover_text("Go to start (Home)")
            .clicked()
        {
            state.playback_position = 0.0;
            state.time_offset = 0.0;
        }

        // Play/Stop toggle
        let play_label = if state.playing { "Stop" } else { "Play" };
        if ui
            .button(play_label)
            .on_hover_text("Toggle playback (Space)")
            .clicked()
        {
            state.playing = !state.playing;
        }

        ui.separator();

        // Position display (time)
        let time_str = format_time(state.playback_position);
        ui.monospace(&time_str);

        // Position display (bar:beat) if tempo is detected
        if let Some(tempo) = state.detected_tempo {
            let bar_beat = format_bar_beat(state.playback_position, tempo, 4);
            ui.label("|");
            ui.monospace(&bar_beat);
        }

        ui.separator();

        // Tempo display
        if let Some(tempo) = state.detected_tempo {
            ui.label(format!("{:.1} BPM", tempo));
        } else {
            ui.label("-- BPM");
        }

        ui.separator();

        // Loop toggle
        let loop_label = if state.loop_enabled {
            "Loop: ON"
        } else {
            "Loop: OFF"
        };
        if ui.selectable_label(state.loop_enabled, loop_label).clicked() {
            state.loop_enabled = !state.loop_enabled;
        }

        // Loop range (editable)
        if state.loop_enabled {
            ui.label("from");
            ui.add(
                egui::DragValue::new(&mut state.loop_start)
                    .speed(0.1)
                    .range(0.0..=state.duration)
                    .suffix("s"),
            );
            ui.label("to");
            ui.add(
                egui::DragValue::new(&mut state.loop_end)
                    .speed(0.1)
                    .range(0.0..=state.duration)
                    .suffix("s"),
            );
        }

        ui.separator();

        // Volume slider
        ui.label("Vol:");
        ui.add(
            egui::Slider::new(&mut state.volume, 0.0..=1.0)
                .show_value(false)
                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
        );

        // Duration display (right aligned)
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if state.duration > 0.0 {
                let dur_str = format_time(state.duration);
                ui.monospace(format!("/ {}", dur_str));
            }
        });
    });
}
