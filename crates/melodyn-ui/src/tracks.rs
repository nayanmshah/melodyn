//! Multi-track panel for the Melodyn editor.
//!
//! Provides a left-side panel showing all tracks in the document with
//! controls for solo, mute, volume, color, adding/removing tracks,
//! reordering via drag, and selecting the active track.

use eframe::egui::{self, Color32, Stroke, Vec2};

use crate::app::AppState;

// ---------------------------------------------------------------------------
// Track UI state
// ---------------------------------------------------------------------------

/// Per-track color palette (cycled for new tracks).
const TRACK_COLORS: &[(f32, f32, f32)] = &[
    (0.82, 0.55, 0.24), // Orange
    (0.31, 0.65, 0.87), // Blue
    (0.55, 0.78, 0.32), // Green
    (0.78, 0.35, 0.65), // Purple
    (0.90, 0.72, 0.25), // Gold
    (0.40, 0.80, 0.75), // Teal
    (0.85, 0.40, 0.40), // Red
    (0.60, 0.60, 0.80), // Lavender
];

/// UI state for one track in the track list.
#[derive(Debug, Clone)]
pub struct TrackUiState {
    /// Display name.
    pub name: String,
    /// Track color (RGB, 0.0-1.0).
    pub color: (f32, f32, f32),
    /// Whether the track is soloed.
    pub solo: bool,
    /// Whether the track is muted.
    pub mute: bool,
    /// Track volume (0.0 to 2.0).
    pub volume: f32,
    /// Number of notes in this track.
    pub note_count: usize,
    /// Duration of the track audio in seconds.
    pub duration: f64,
    /// Source file path (if loaded from file).
    pub source_path: Option<String>,
}

impl TrackUiState {
    /// Create a new track UI state with default values.
    pub fn new(name: String, color_index: usize) -> Self {
        let palette = TRACK_COLORS[color_index % TRACK_COLORS.len()];
        Self {
            name,
            color: palette,
            solo: false,
            mute: false,
            volume: 1.0,
            note_count: 0,
            duration: 0.0,
            source_path: None,
        }
    }

    /// Convert the track color to an egui Color32.
    pub fn color32(&self) -> Color32 {
        Color32::from_rgb(
            (self.color.0 * 255.0) as u8,
            (self.color.1 * 255.0) as u8,
            (self.color.2 * 255.0) as u8,
        )
    }
}

// ---------------------------------------------------------------------------
// Track list state
// ---------------------------------------------------------------------------

/// State for the multi-track panel.
#[derive(Debug, Clone)]
pub struct TrackListState {
    /// All tracks in display order.
    pub tracks: Vec<TrackUiState>,
    /// Index of the currently active track (shown in the editor).
    pub active_track: usize,
    /// Track being dragged for reorder (if any).
    _drag_source: Option<usize>,
    /// Whether the color picker popup is open and for which track.
    color_picker_track: Option<usize>,
    /// Pending confirmation for track removal.
    confirm_remove: Option<usize>,
}

impl Default for TrackListState {
    fn default() -> Self {
        Self {
            tracks: vec![TrackUiState::new("Track 1".to_string(), 0)],
            active_track: 0,
            _drag_source: None,
            color_picker_track: None,
            confirm_remove: None,
        }
    }
}

impl TrackListState {
    /// Add a new track with the next available color.
    pub fn add_track(&mut self, name: String) -> usize {
        let idx = self.tracks.len();
        self.tracks.push(TrackUiState::new(name, idx));
        idx
    }

    /// Remove a track by index, shifting the active track if needed.
    pub fn remove_track(&mut self, index: usize) -> bool {
        if self.tracks.len() <= 1 || index >= self.tracks.len() {
            return false;
        }
        self.tracks.remove(index);
        if self.active_track >= self.tracks.len() {
            self.active_track = self.tracks.len() - 1;
        }
        true
    }

    /// Move a track from one position to another.
    pub fn reorder(&mut self, from: usize, to: usize) {
        if from == to || from >= self.tracks.len() || to >= self.tracks.len() {
            return;
        }
        let track = self.tracks.remove(from);
        self.tracks.insert(to, track);
        // Update active track index
        if self.active_track == from {
            self.active_track = to;
        } else if from < self.active_track && to >= self.active_track {
            self.active_track = self.active_track.saturating_sub(1);
        } else if from > self.active_track && to <= self.active_track {
            self.active_track += 1;
        }
    }

    /// Whether any track is soloed.
    pub fn any_solo(&self) -> bool {
        self.tracks.iter().any(|t| t.solo)
    }

    /// Whether a track should produce audio.
    pub fn is_audible(&self, index: usize) -> bool {
        if let Some(track) = self.tracks.get(index) {
            if self.any_solo() {
                track.solo
            } else {
                !track.mute
            }
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Track list panel rendering
// ---------------------------------------------------------------------------

/// Height of each track row in the panel.
const TRACK_ROW_HEIGHT: f32 = 72.0;
/// Width of the solo/mute buttons.
const BUTTON_SIZE: f32 = 22.0;

/// Draw the track list panel.
///
/// Shows a scrollable list of tracks with controls for each:
/// - Color indicator and name (click to select as active)
/// - Solo (S) and Mute (M) toggle buttons
/// - Volume slider
/// - Track drag handle for reordering
pub fn track_list_panel(
    ui: &mut egui::Ui,
    track_state: &mut TrackListState,
    app_state: &mut AppState,
) {
    ui.heading("Tracks");
    ui.separator();

    // Add Track button
    ui.horizontal(|ui| {
        if ui.button("+ Add Track").clicked() {
            let idx = track_state.tracks.len() + 1;
            let name = format!("Track {}", idx);
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("WAV files", &["wav"])
                .pick_file()
            {
                let track_idx = track_state.add_track(name);
                track_state.tracks[track_idx].source_path =
                    Some(path.to_string_lossy().to_string());
                track_state.active_track = track_idx;
                app_state.status_message = format!("Added track from {}", path.display());
            }
        }
    });

    ui.add_space(4.0);

    // Confirm remove dialog
    if let Some(remove_idx) = track_state.confirm_remove {
        ui.horizontal(|ui| {
            ui.label(format!(
                "Remove \"{}\"?",
                track_state.tracks.get(remove_idx).map_or("?", |t| &t.name)
            ));
            if ui.button("Yes").clicked() {
                track_state.remove_track(remove_idx);
                track_state.confirm_remove = None;
                app_state.status_message = "Track removed".to_string();
            }
            if ui.button("No").clicked() {
                track_state.confirm_remove = None;
            }
        });
        ui.separator();
    }

    // Track list (scrollable)
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let num_tracks = track_state.tracks.len();

            for i in 0..num_tracks {
                let is_active = track_state.active_track == i;
                let is_audible = track_state.is_audible(i);

                let track_color = track_state.tracks[i].color32();
                let bg = if is_active {
                    Color32::from_rgba_premultiplied(
                        track_color.r() / 4,
                        track_color.g() / 4,
                        track_color.b() / 4,
                        40,
                    )
                } else {
                    Color32::TRANSPARENT
                };

                // Track row frame
                let frame = egui::Frame::default()
                    .fill(bg)
                    .inner_margin(egui::Margin::same(4.0))
                    .stroke(if is_active {
                        Stroke::new(1.0_f32, track_color)
                    } else {
                        Stroke::NONE
                    });

                frame.show(ui, |ui| {
                    ui.set_min_height(TRACK_ROW_HEIGHT);

                    // Color bar + name row
                    ui.horizontal(|ui| {
                        // Color indicator (clickable for color picker)
                        let (color_resp, color_painter) = ui.allocate_painter(
                            Vec2::new(8.0, TRACK_ROW_HEIGHT - 16.0),
                            egui::Sense::click(),
                        );
                        color_painter.rect_filled(color_resp.rect, 2.0, track_color);
                        if color_resp.clicked() {
                            track_state.color_picker_track = Some(i);
                        }

                        ui.vertical(|ui| {
                            // Name (click to make active)
                            let name_resp =
                                ui.selectable_label(is_active, &track_state.tracks[i].name);
                            if name_resp.clicked() {
                                track_state.active_track = i;
                            }

                            // Solo / Mute buttons
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 2.0;

                                let solo_color = if track_state.tracks[i].solo {
                                    Color32::from_rgb(255, 220, 50)
                                } else {
                                    Color32::from_rgb(120, 120, 120)
                                };
                                let solo_btn = egui::Button::new(
                                    egui::RichText::new("S").color(solo_color).small(),
                                )
                                .min_size(Vec2::new(BUTTON_SIZE, BUTTON_SIZE));
                                if ui.add(solo_btn).on_hover_text("Solo").clicked() {
                                    track_state.tracks[i].solo = !track_state.tracks[i].solo;
                                }

                                let mute_color = if track_state.tracks[i].mute {
                                    Color32::from_rgb(255, 80, 80)
                                } else {
                                    Color32::from_rgb(120, 120, 120)
                                };
                                let mute_btn = egui::Button::new(
                                    egui::RichText::new("M").color(mute_color).small(),
                                )
                                .min_size(Vec2::new(BUTTON_SIZE, BUTTON_SIZE));
                                if ui.add(mute_btn).on_hover_text("Mute").clicked() {
                                    track_state.tracks[i].mute = !track_state.tracks[i].mute;
                                }

                                // Audibility indicator
                                if !is_audible {
                                    ui.label(
                                        egui::RichText::new("(silent)")
                                            .small()
                                            .color(Color32::from_rgb(180, 80, 80)),
                                    );
                                }
                            });

                            // Volume slider
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Vol").small());
                                ui.add(
                                    egui::Slider::new(&mut track_state.tracks[i].volume, 0.0..=2.0)
                                        .show_value(false)
                                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                                );
                            });

                            // Note count + remove button
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} notes",
                                        track_state.tracks[i].note_count
                                    ))
                                    .small()
                                    .color(Color32::from_rgb(150, 150, 160)),
                                );

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if num_tracks > 1
                                            && ui
                                                .small_button("x")
                                                .on_hover_text("Remove track")
                                                .clicked()
                                        {
                                            track_state.confirm_remove = Some(i);
                                        }
                                    },
                                );
                            });
                        });
                    });
                });

                ui.separator();
            }
        });

    // Color picker popup
    if let Some(picker_idx) = track_state.color_picker_track {
        egui::Window::new("Track Color")
            .collapsible(false)
            .resizable(false)
            .show(ui.ctx(), |ui| {
                ui.horizontal_wrapped(|ui| {
                    for &(r, g, b) in TRACK_COLORS.iter() {
                        let color = Color32::from_rgb(
                            (r * 255.0) as u8,
                            (g * 255.0) as u8,
                            (b * 255.0) as u8,
                        );
                        let (resp, painter) =
                            ui.allocate_painter(Vec2::new(24.0, 24.0), egui::Sense::click());
                        painter.rect_filled(resp.rect, 3.0, color);
                        let is_current = picker_idx < track_state.tracks.len()
                            && track_state.tracks[picker_idx].color == (r, g, b);
                        if is_current {
                            painter.rect_stroke(
                                resp.rect,
                                3.0,
                                Stroke::new(2.0_f32, Color32::WHITE),
                            );
                        }
                        if resp.clicked() {
                            if picker_idx < track_state.tracks.len() {
                                track_state.tracks[picker_idx].color = (r, g, b);
                            }
                            track_state.color_picker_track = None;
                        }
                    }
                });
                if ui.button("Close").clicked() {
                    track_state.color_picker_track = None;
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_list_default() {
        let state = TrackListState::default();
        assert_eq!(state.tracks.len(), 1);
        assert_eq!(state.active_track, 0);
    }

    #[test]
    fn test_add_remove_track() {
        let mut state = TrackListState::default();
        let idx = state.add_track("Track 2".to_string());
        assert_eq!(idx, 1);
        assert_eq!(state.tracks.len(), 2);

        assert!(state.remove_track(0));
        assert_eq!(state.tracks.len(), 1);
        assert_eq!(state.active_track, 0);
    }

    #[test]
    fn test_cannot_remove_last_track() {
        let mut state = TrackListState::default();
        assert!(!state.remove_track(0));
        assert_eq!(state.tracks.len(), 1);
    }

    #[test]
    fn test_reorder_tracks() {
        let mut state = TrackListState::default();
        state.add_track("Track 2".to_string());
        state.add_track("Track 3".to_string());
        state.active_track = 0;

        state.reorder(0, 2);
        assert_eq!(state.tracks[2].name, "Track 1");
        assert_eq!(state.active_track, 2);
    }

    #[test]
    fn test_solo_mute() {
        let mut state = TrackListState::default();
        state.add_track("Track 2".to_string());

        // No solo: both audible
        assert!(state.is_audible(0));
        assert!(state.is_audible(1));

        // Solo track 0
        state.tracks[0].solo = true;
        assert!(state.is_audible(0));
        assert!(!state.is_audible(1));

        // Unsolo, mute track 1
        state.tracks[0].solo = false;
        state.tracks[1].mute = true;
        assert!(state.is_audible(0));
        assert!(!state.is_audible(1));
    }

    #[test]
    fn test_track_color() {
        let track = TrackUiState::new("Test".to_string(), 0);
        let c = track.color32();
        assert!(c.r() > 0);
    }
}
