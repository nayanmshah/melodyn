//! Inspector panel for viewing and editing properties of selected notes.
//!
//! Based on the property inspector in the original Melodyne UI. When one
//! or more notes are selected, their properties are displayed in editable
//! fields. Edits through the inspector apply to all selected notes.

use eframe::egui;

use crate::app::{AppState, EditableNote};

// ---------------------------------------------------------------------------
// Inspector panel
// ---------------------------------------------------------------------------

/// Draw the inspector panel showing properties of selected notes.
pub fn inspector_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Inspector");
    ui.separator();

    if state.selected_notes.is_empty() {
        ui.label("No notes selected.");
        ui.add_space(8.0);
        ui.label("Click a note in the editor to select it, or use Ctrl+A to select all.");
        return;
    }

    let count = state.selected_notes.len();
    ui.label(format!("{} note{} selected", count, if count > 1 { "s" } else { "" }));
    ui.add_space(4.0);

    if count == 1 {
        // Single note: show all properties
        let idx = *state.selected_notes.iter().next().unwrap();
        if idx < state.notes.len() {
            single_note_inspector(ui, state, idx);
        }
    } else {
        // Multiple notes: show summary and batch edit controls
        multi_note_inspector(ui, state);
    }
}

/// Inspector view for a single selected note.
fn single_note_inspector(ui: &mut egui::Ui, state: &mut AppState, idx: usize) {
    let note = &state.notes[idx];

    // Note name and pitch
    ui.group(|ui| {
        ui.label("Pitch");
        ui.horizontal(|ui| {
            ui.label("Note:");
            ui.strong(note.effective_note_name());
        });
        ui.horizontal(|ui| {
            ui.label("MIDI:");
            ui.monospace(format!("{:.2}", note.effective_midi()));
        });
        ui.horizontal(|ui| {
            ui.label("Freq:");
            ui.monospace(format!("{:.1} Hz", note.effective_frequency()));
        });
        ui.horizontal(|ui| {
            ui.label("Cents:");
            ui.monospace(format!("{:+.1}", note.effective_cents()));
        });
    });

    ui.add_space(4.0);

    // Timing
    ui.group(|ui| {
        ui.label("Time");
        ui.horizontal(|ui| {
            ui.label("Start:");
            ui.monospace(format!("{:.3}s", note.effective_start()));
        });
        ui.horizontal(|ui| {
            ui.label("Duration:");
            ui.monospace(format!("{:.3}s", note.effective_duration()));
        });
        ui.horizontal(|ui| {
            ui.label("End:");
            ui.monospace(format!("{:.3}s", note.effective_end()));
        });
    });

    ui.add_space(4.0);

    // Amplitude
    ui.group(|ui| {
        ui.label("Amplitude");
        ui.horizontal(|ui| {
            ui.label("Level:");
            ui.monospace(format!("{:.1} dB", note.amplitude_db()));
        });
        ui.horizontal(|ui| {
            ui.label("Linear:");
            ui.monospace(format!("{:.3}", note.effective_amplitude()));
        });
        ui.horizontal(|ui| {
            ui.label("Confidence:");
            ui.monospace(format!("{:.1}%", note.detected.confidence * 100.0));
        });
    });

    ui.add_space(4.0);

    // Vibrato
    ui.group(|ui| {
        ui.label("Vibrato");
        ui.horizontal(|ui| {
            ui.label("Rate:");
            ui.monospace(format!("{:.1} Hz", note.detected.vibrato_rate));
        });
        ui.horizontal(|ui| {
            ui.label("Depth:");
            ui.monospace(format!("{:.1} cents", note.detected.vibrato_depth));
        });
    });

    ui.add_space(8.0);

    // Editable offsets
    ui.heading("Edit Offsets");
    ui.separator();

    // Create mutable reference for editing
    let note = &mut state.notes[idx];

    ui.horizontal(|ui| {
        ui.label("Pitch shift:");
        let old = note.pitch_offset;
        ui.add(
            egui::DragValue::new(&mut note.pitch_offset)
                .speed(0.1)
                .range(-48.0..=48.0)
                .suffix(" st"),
        );
        if note.pitch_offset != old {
            state.modified = true;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Time shift:");
        let old = note.time_offset;
        ui.add(
            egui::DragValue::new(&mut note.time_offset)
                .speed(0.01)
                .suffix(" s"),
        );
        if note.time_offset != old {
            state.modified = true;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Duration scale:");
        let old = note.duration_scale;
        ui.add(
            egui::DragValue::new(&mut note.duration_scale)
                .speed(0.01)
                .range(0.1..=4.0)
                .suffix("x"),
        );
        if note.duration_scale != old {
            state.modified = true;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Amplitude scale:");
        let old = note.amplitude_scale;
        ui.add(
            egui::DragValue::new(&mut note.amplitude_scale)
                .speed(0.01)
                .range(0.0..=4.0)
                .suffix("x"),
        );
        if note.amplitude_scale != old {
            state.modified = true;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Formant shift:");
        let old = note.formant_shift;
        ui.add(
            egui::DragValue::new(&mut note.formant_shift)
                .speed(0.1)
                .range(-24.0..=24.0)
                .suffix(" st"),
        );
        if note.formant_shift != old {
            state.modified = true;
        }
    });

    ui.add_space(8.0);

    if ui.button("Reset Offsets").clicked() {
        state.push_undo("Reset Note Offsets");
        let note = &mut state.notes[idx];
        note.pitch_offset = 0.0;
        note.time_offset = 0.0;
        note.duration_scale = 1.0;
        note.amplitude_scale = 1.0;
        note.formant_shift = 0.0;
    }
}

/// Inspector view for multiple selected notes (batch operations).
fn multi_note_inspector(ui: &mut egui::Ui, state: &mut AppState) {
    let selected: Vec<usize> = state.selected_notes.iter().copied().collect();

    // Compute summary statistics
    let notes: Vec<&EditableNote> = selected
        .iter()
        .filter_map(|&i| state.notes.get(i))
        .collect();

    if notes.is_empty() {
        ui.label("Invalid selection.");
        return;
    }

    let min_midi = notes
        .iter()
        .map(|n| n.effective_midi())
        .fold(f64::MAX, f64::min);
    let max_midi = notes
        .iter()
        .map(|n| n.effective_midi())
        .fold(f64::MIN, f64::max);
    let min_time = notes
        .iter()
        .map(|n| n.effective_start())
        .fold(f64::MAX, f64::min);
    let max_time = notes
        .iter()
        .map(|n| n.effective_end())
        .fold(f64::MIN, f64::max);
    let avg_amp = notes.iter().map(|n| n.effective_amplitude()).sum::<f64>() / notes.len() as f64;

    ui.group(|ui| {
        ui.label("Range Summary");
        ui.horizontal(|ui| {
            ui.label("Pitch:");
            ui.monospace(format!(
                "{} - {}",
                melodyn_core::PitchCandidate::note_name(min_midi),
                melodyn_core::PitchCandidate::note_name(max_midi),
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Time:");
            ui.monospace(format!("{:.2}s - {:.2}s", min_time, max_time));
        });
        ui.horizontal(|ui| {
            ui.label("Avg amplitude:");
            ui.monospace(format!("{:.3}", avg_amp));
        });
    });

    ui.add_space(8.0);
    ui.heading("Batch Edit");
    ui.separator();

    // Batch pitch shift
    ui.horizontal(|ui| {
        ui.label("Shift pitch:");
        if ui.button("-1 st").clicked() {
            state.push_undo("Batch Pitch Down");
            for &i in &selected {
                if i < state.notes.len() {
                    state.notes[i].pitch_offset -= 1.0;
                }
            }
        }
        if ui.button("+1 st").clicked() {
            state.push_undo("Batch Pitch Up");
            for &i in &selected {
                if i < state.notes.len() {
                    state.notes[i].pitch_offset += 1.0;
                }
            }
        }
        if ui.button("-12 st").clicked() {
            state.push_undo("Batch Octave Down");
            for &i in &selected {
                if i < state.notes.len() {
                    state.notes[i].pitch_offset -= 12.0;
                }
            }
        }
        if ui.button("+12 st").clicked() {
            state.push_undo("Batch Octave Up");
            for &i in &selected {
                if i < state.notes.len() {
                    state.notes[i].pitch_offset += 12.0;
                }
            }
        }
    });

    // Batch amplitude scale
    ui.horizontal(|ui| {
        ui.label("Amplitude:");
        if ui.button("-3 dB").clicked() {
            state.push_undo("Batch Volume Down");
            let factor = 10.0f64.powf(-3.0 / 20.0);
            for &i in &selected {
                if i < state.notes.len() {
                    state.notes[i].amplitude_scale *= factor;
                }
            }
        }
        if ui.button("+3 dB").clicked() {
            state.push_undo("Batch Volume Up");
            let factor = 10.0f64.powf(3.0 / 20.0);
            for &i in &selected {
                if i < state.notes.len() {
                    state.notes[i].amplitude_scale =
                        (state.notes[i].amplitude_scale * factor).min(4.0);
                }
            }
        }
    });

    ui.add_space(8.0);

    if ui.button("Reset All Offsets").clicked() {
        state.push_undo("Reset Selection Offsets");
        for &i in &selected {
            if i < state.notes.len() {
                state.notes[i].pitch_offset = 0.0;
                state.notes[i].time_offset = 0.0;
                state.notes[i].duration_scale = 1.0;
                state.notes[i].amplitude_scale = 1.0;
                state.notes[i].formant_shift = 0.0;
            }
        }
    }
}
