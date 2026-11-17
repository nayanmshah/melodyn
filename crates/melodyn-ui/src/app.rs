//! Main application struct and state management.
//!
//! Contains the `MelodynApp` struct (implementing `eframe::App`), the
//! `AppState` for all editor state, `EditableNote` wrapping detected notes
//! with edit offsets, undo/redo, file I/O, analysis pipeline, and menu bar.

use std::collections::HashSet;

use eframe::egui;
use melodyn_core::{
    AnalysisConfig, AnalysisResult, DetectedNote, DetectionPreset, PitchCandidate,
    SpectralAnalyzer, TempoAnalyzer, TempoAnalyzerConfig,
};
use melodyn_core::scale::{detect_key, distribution_from_notes};
use melodyn_core::tempo::BeatGrid;
use melodyn_io::{AudioBuffer, read_wav, write_wav, WavWriteConfig};

use crate::tools::{SnapMode, Tool};

// ---------------------------------------------------------------------------
// Editable note
// ---------------------------------------------------------------------------

/// A note with editable offsets layered on top of the original `DetectedNote`.
///
/// All edits are non-destructive: the original analysis data is preserved
/// and offsets are applied on top to produce effective values.
#[derive(Clone, Debug)]
pub struct EditableNote {
    /// The original detected note from analysis.
    pub detected: DetectedNote,
    /// Pitch offset in semitones (applied on top of original MIDI note).
    pub pitch_offset: f64,
    /// Time offset in seconds (applied on top of original start time).
    pub time_offset: f64,
    /// Duration scale factor (1.0 = original duration).
    pub duration_scale: f64,
    /// Amplitude scale factor (1.0 = original amplitude).
    pub amplitude_scale: f64,
    /// Formant shift in semitones.
    pub formant_shift: f64,
}

impl EditableNote {
    /// Create from a detected note with no edits applied.
    pub fn from_detected(note: DetectedNote) -> Self {
        Self {
            detected: note,
            pitch_offset: 0.0,
            time_offset: 0.0,
            duration_scale: 1.0,
            amplitude_scale: 1.0,
            formant_shift: 0.0,
        }
    }

    /// Effective MIDI note number after pitch offset.
    pub fn effective_midi(&self) -> f64 {
        self.detected.midi_note + self.pitch_offset
    }

    /// Effective frequency after pitch offset.
    pub fn effective_frequency(&self) -> f64 {
        PitchCandidate::midi_to_freq(self.effective_midi())
    }

    /// Effective start time after time offset.
    pub fn effective_start(&self) -> f64 {
        self.detected.start_time + self.time_offset
    }

    /// Effective duration after scaling.
    pub fn effective_duration(&self) -> f64 {
        self.detected.duration * self.duration_scale
    }

    /// Effective end time.
    pub fn effective_end(&self) -> f64 {
        self.effective_start() + self.effective_duration()
    }

    /// Effective amplitude after scaling.
    pub fn effective_amplitude(&self) -> f64 {
        (self.detected.amplitude * self.amplitude_scale).clamp(0.0, 1.0)
    }

    /// Note name after edits (e.g., "A4", "C#3+12c").
    pub fn effective_note_name(&self) -> String {
        PitchCandidate::note_name(self.effective_midi())
    }

    /// Cents deviation from nearest semitone after edits.
    pub fn effective_cents(&self) -> f64 {
        (self.effective_midi() - self.effective_midi().round()) * 100.0
    }

    /// Amplitude in decibels.
    pub fn amplitude_db(&self) -> f64 {
        let amp = self.effective_amplitude();
        if amp < 1e-10 {
            -100.0
        } else {
            20.0 * amp.log10()
        }
    }
}

// ---------------------------------------------------------------------------
// Undo/redo
// ---------------------------------------------------------------------------

/// A snapshot of all note states for undo/redo.
#[derive(Clone)]
pub struct UndoSnapshot {
    pub description: String,
    pub notes: Vec<EditableNote>,
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// Complete editor state shared across all UI panels.
pub struct AppState {
    // -- Audio data --
    /// The loaded audio buffer (all channels).
    pub audio_buffer: Option<AudioBuffer>,
    /// Mono mixdown for analysis and display.
    pub audio_mono: Vec<f64>,
    /// Sample rate of the loaded audio.
    pub sample_rate: f64,
    /// Path of the loaded file.
    pub file_path: Option<String>,

    // -- Analysis results --
    /// All notes with editable overlays.
    pub notes: Vec<EditableNote>,
    /// Total audio duration in seconds.
    pub duration: f64,
    /// Detected tempo in BPM.
    pub detected_tempo: Option<f64>,
    /// Detected key/scale name.
    pub detected_key: Option<String>,
    /// Detected key tonic (0=C, 11=B).
    pub detected_key_tonic: Option<u8>,
    /// Beat grid aligned to detected onsets.
    pub beat_grid: Option<BeatGrid>,
    /// Analysis result (retained for reference).
    pub analysis_result: Option<AnalysisResult>,

    // -- Waveform peaks (precomputed for overview display) --
    pub waveform_peaks: Vec<(f32, f32)>,

    // -- View state --
    /// Left edge of visible time range (seconds).
    pub time_offset: f64,
    /// Horizontal zoom: pixels per second.
    pub time_scale: f64,
    /// Bottom edge of visible pitch range (MIDI note).
    pub pitch_offset: f64,
    /// Vertical zoom: pixels per semitone.
    pub pitch_scale: f64,

    // -- Selection --
    /// Indices of selected notes.
    pub selected_notes: HashSet<usize>,
    /// Index of note under cursor (if any).
    pub hovered_note: Option<usize>,

    // -- Transport --
    /// Whether playback is active.
    pub playing: bool,
    /// Current playback position in seconds.
    pub playback_position: f64,
    /// Whether loop mode is on.
    pub loop_enabled: bool,
    /// Loop start in seconds.
    pub loop_start: f64,
    /// Loop end in seconds.
    pub loop_end: f64,
    /// Master volume (0.0 to 1.0).
    pub volume: f32,

    // -- Active tool --
    pub active_tool: Tool,
    pub snap_mode: SnapMode,

    // -- Drag state --
    pub dragging: bool,
    pub drag_note_origins: Vec<(f64, f64)>,

    // -- Undo/redo --
    pub undo_stack: Vec<UndoSnapshot>,
    pub redo_stack: Vec<UndoSnapshot>,

    // -- Status --
    pub status_message: String,
    pub modified: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            audio_buffer: None,
            audio_mono: Vec::new(),
            sample_rate: 44100.0,
            file_path: None,

            notes: Vec::new(),
            duration: 0.0,
            detected_tempo: None,
            detected_key: None,
            detected_key_tonic: None,
            beat_grid: None,
            analysis_result: None,

            waveform_peaks: Vec::new(),

            time_offset: 0.0,
            time_scale: 150.0,
            pitch_offset: 48.0,
            pitch_scale: 12.0,

            selected_notes: HashSet::new(),
            hovered_note: None,

            playing: false,
            playback_position: 0.0,
            loop_enabled: false,
            loop_start: 0.0,
            loop_end: 0.0,
            volume: 0.8,

            active_tool: Tool::Combi,
            snap_mode: SnapMode::Off,

            dragging: false,
            drag_note_origins: Vec::new(),

            undo_stack: Vec::new(),
            redo_stack: Vec::new(),

            status_message: "Ready -- open a WAV file to begin".to_string(),
            modified: false,
        }
    }
}

impl AppState {
    /// Push an undo snapshot of the current note state.
    pub fn push_undo(&mut self, description: &str) {
        self.undo_stack.push(UndoSnapshot {
            description: description.to_string(),
            notes: self.notes.clone(),
        });
        self.redo_stack.clear();
        if self.undo_stack.len() > 200 {
            self.undo_stack.remove(0);
        }
        self.modified = true;
    }

    /// Undo the last action.
    pub fn undo(&mut self) {
        if let Some(snapshot) = self.undo_stack.pop() {
            self.redo_stack.push(UndoSnapshot {
                description: snapshot.description.clone(),
                notes: self.notes.clone(),
            });
            self.notes = snapshot.notes;
            self.status_message = format!("Undo: {}", snapshot.description);
        }
    }

    /// Redo the last undone action.
    pub fn redo(&mut self) {
        if let Some(snapshot) = self.redo_stack.pop() {
            self.undo_stack.push(UndoSnapshot {
                description: snapshot.description.clone(),
                notes: self.notes.clone(),
            });
            self.notes = snapshot.notes;
            self.status_message = format!("Redo: {}", snapshot.description);
        }
    }

    /// Select all notes.
    pub fn select_all(&mut self) {
        for i in 0..self.notes.len() {
            self.selected_notes.insert(i);
        }
    }

    /// Delete all selected notes.
    pub fn delete_selected(&mut self) {
        if self.selected_notes.is_empty() {
            return;
        }
        self.push_undo("Delete Notes");
        let mut indices: Vec<usize> = self.selected_notes.iter().copied().collect();
        indices.sort_unstable_by(|a, b| b.cmp(a)); // reverse order
        for idx in indices {
            if idx < self.notes.len() {
                self.notes.remove(idx);
            }
        }
        self.selected_notes.clear();
    }

    /// Advance playback position (called each frame when playing).
    pub fn advance_playback(&mut self, dt: f64) {
        if !self.playing {
            return;
        }
        self.playback_position += dt;
        if self.loop_enabled && self.loop_end > self.loop_start {
            if self.playback_position >= self.loop_end {
                self.playback_position = self.loop_start;
            }
        } else if self.playback_position >= self.duration {
            self.playing = false;
            self.playback_position = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Waveform peak computation
// ---------------------------------------------------------------------------

/// Compute (min, max) peaks from mono audio for waveform display.
fn compute_waveform_peaks(mono: &[f64], target_peaks: usize) -> Vec<(f32, f32)> {
    if mono.is_empty() || target_peaks == 0 {
        return Vec::new();
    }
    let samples_per_peak = (mono.len() / target_peaks).max(1);
    mono.chunks(samples_per_peak)
        .map(|chunk| {
            let min = chunk.iter().copied().fold(f64::MAX, f64::min) as f32;
            let max = chunk.iter().copied().fold(f64::MIN, f64::max) as f32;
            (min, max)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// File I/O and analysis
// ---------------------------------------------------------------------------

/// Open a WAV file, run analysis, and populate the app state.
fn open_file(state: &mut AppState, path: &str) {
    state.status_message = format!("Loading {}...", path);

    // Read WAV
    let buffer = match read_wav(path) {
        Ok(buf) => buf,
        Err(e) => {
            state.status_message = format!("Error loading file: {}", e);
            return;
        }
    };

    let sample_rate = buffer.sample_rate();
    let mono = buffer.to_mono();
    let duration = buffer.duration();

    // Run spectral analysis
    state.status_message = "Analyzing audio...".to_string();
    let config = AnalysisConfig::from_preset(DetectionPreset::Automatic, sample_rate);
    let analyzer = SpectralAnalyzer::new(config);
    let result = analyzer.analyze(&mono);

    // Convert detected notes to editable notes
    let notes: Vec<EditableNote> = result
        .notes
        .iter()
        .cloned()
        .map(EditableNote::from_detected)
        .collect();

    // Run tempo analysis
    let tempo_analyzer = TempoAnalyzer::new(TempoAnalyzerConfig::default());
    let tempo_result = tempo_analyzer.analyze_notes(&result.notes, duration);
    let detected_tempo = tempo_result.primary_tempo.map(|t| t.bpm);
    let beat_grid = tempo_result.beat_grid.clone();

    // Detect key
    let (detected_key, detected_key_tonic) = if !result.notes.is_empty() {
        let dist = distribution_from_notes(&result.notes);
        let candidates = detect_key(&dist);
        if let Some(best) = candidates.first() {
            (Some(best.name()), Some(best.tonic))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    // Compute waveform peaks
    let waveform_peaks = compute_waveform_peaks(&mono, 4000);

    // Compute initial view range from notes
    let (pitch_min, _pitch_max) = if notes.is_empty() {
        (48.0, 84.0)
    } else {
        let min = notes
            .iter()
            .map(|n| n.detected.midi_note)
            .fold(f64::MAX, f64::min);
        let max = notes
            .iter()
            .map(|n| n.detected.midi_note)
            .fold(f64::MIN, f64::max);
        (min - 4.0, max + 4.0)
    };

    // Populate state
    state.audio_buffer = Some(buffer);
    state.audio_mono = mono;
    state.sample_rate = sample_rate;
    state.file_path = Some(path.to_string());
    state.notes = notes;
    state.duration = duration;
    state.detected_tempo = detected_tempo;
    state.detected_key = detected_key;
    state.detected_key_tonic = detected_key_tonic;
    state.beat_grid = beat_grid;
    state.analysis_result = Some(result);
    state.waveform_peaks = waveform_peaks;

    // Set view to fit content
    state.time_offset = 0.0;
    state.pitch_offset = pitch_min;
    state.pitch_scale = 12.0;
    state.playback_position = 0.0;
    state.playing = false;
    state.selected_notes.clear();
    state.undo_stack.clear();
    state.redo_stack.clear();
    state.modified = false;
    state.loop_start = 0.0;
    state.loop_end = duration;

    let note_count = state.notes.len();
    let tempo_str = detected_tempo
        .map(|t| format!("{:.1} BPM", t))
        .unwrap_or_else(|| "N/A".to_string());
    let key_str = state
        .detected_key
        .clone()
        .unwrap_or_else(|| "N/A".to_string());
    state.status_message = format!(
        "Loaded: {} notes, {:.1}s, tempo: {}, key: {}",
        note_count, duration, tempo_str, key_str
    );
}

/// Export the current audio to a WAV file.
fn export_wav(state: &AppState, path: &str) {
    if let Some(ref buffer) = state.audio_buffer {
        let config = WavWriteConfig::default();
        match write_wav(path, buffer, &config) {
            Ok(()) => {
                log::info!("Exported to {}", path);
            }
            Err(e) => {
                log::error!("Export failed: {}", e);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Menu bar
// ---------------------------------------------------------------------------

/// Draw the application menu bar.
pub fn menu_bar(ui: &mut egui::Ui, state: &mut AppState) {
    egui::menu::bar(ui, |ui| {
        ui.menu_button("File", |ui| {
            if ui
                .add(egui::Button::new("Open...").shortcut_text("Ctrl+O"))
                .clicked()
            {
                ui.close_menu();
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV files", &["wav"])
                    .add_filter("All files", &["*"])
                    .pick_file()
                {
                    open_file(state, &path.to_string_lossy());
                }
            }

            if ui
                .add(
                    egui::Button::new("Export WAV...")
                        .shortcut_text("Ctrl+Shift+E"),
                )
                .clicked()
            {
                ui.close_menu();
                if state.audio_buffer.is_some() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("WAV files", &["wav"])
                        .save_file()
                    {
                        export_wav(state, &path.to_string_lossy());
                        state.status_message =
                            format!("Exported to {}", path.display());
                    }
                }
            }

            ui.separator();

            if ui.button("Quit").clicked() {
                std::process::exit(0);
            }
        });

        ui.menu_button("Edit", |ui| {
            let can_undo = !state.undo_stack.is_empty();
            let undo_label = if let Some(last) = state.undo_stack.last() {
                format!("Undo: {}", last.description)
            } else {
                "Undo".to_string()
            };
            if ui
                .add_enabled(
                    can_undo,
                    egui::Button::new(undo_label).shortcut_text("Ctrl+Z"),
                )
                .clicked()
            {
                ui.close_menu();
                state.undo();
            }

            let can_redo = !state.redo_stack.is_empty();
            if ui
                .add_enabled(
                    can_redo,
                    egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z"),
                )
                .clicked()
            {
                ui.close_menu();
                state.redo();
            }

            ui.separator();

            if ui
                .add(egui::Button::new("Select All").shortcut_text("Ctrl+A"))
                .clicked()
            {
                ui.close_menu();
                state.select_all();
            }

            if ui
                .add_enabled(
                    !state.selected_notes.is_empty(),
                    egui::Button::new("Delete Selected").shortcut_text("Delete"),
                )
                .clicked()
            {
                ui.close_menu();
                state.delete_selected();
            }
        });

        ui.menu_button("View", |ui| {
            if ui
                .add(egui::Button::new("Zoom In").shortcut_text("Ctrl+="))
                .clicked()
            {
                state.time_scale = (state.time_scale * 1.25).min(2000.0);
                ui.close_menu();
            }
            if ui
                .add(egui::Button::new("Zoom Out").shortcut_text("Ctrl+-"))
                .clicked()
            {
                state.time_scale = (state.time_scale / 1.25).max(20.0);
                ui.close_menu();
            }
            ui.separator();
            if ui
                .add(
                    egui::Button::new("Zoom Vertical In")
                        .shortcut_text("Ctrl+Shift+="),
                )
                .clicked()
            {
                state.pitch_scale = (state.pitch_scale * 1.25).min(60.0);
                ui.close_menu();
            }
            if ui
                .add(
                    egui::Button::new("Zoom Vertical Out")
                        .shortcut_text("Ctrl+Shift+-"),
                )
                .clicked()
            {
                state.pitch_scale = (state.pitch_scale / 1.25).max(4.0);
                ui.close_menu();
            }
        });

        ui.menu_button("Analysis", |ui| {
            if ui
                .add_enabled(
                    state.audio_buffer.is_some(),
                    egui::Button::new("Re-analyze (Automatic)"),
                )
                .clicked()
            {
                ui.close_menu();
                if let Some(ref path) = state.file_path.clone() {
                    open_file(state, path);
                }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// MelodynApp
// ---------------------------------------------------------------------------

/// Main application struct implementing `eframe::App`.
pub struct MelodynApp {
    pub state: AppState,
}

impl MelodynApp {
    /// Create the application, optionally loading initial state.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            state: AppState::default(),
        }
    }

    /// Handle global keyboard shortcuts.
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        // Tool shortcuts
        crate::tools::handle_tool_shortcuts(ctx, &mut self.state);

        ctx.input(|i| {
            // Space: toggle play
            if i.key_pressed(egui::Key::Space) {
                self.state.playing = !self.state.playing;
                if self.state.playing {
                    self.state.status_message = "Playing".to_string();
                } else {
                    self.state.status_message = format!(
                        "Stopped at {:.2}s",
                        self.state.playback_position
                    );
                }
            }

            // Ctrl+O: open
            if i.modifiers.command && i.key_pressed(egui::Key::O) {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV files", &["wav"])
                    .pick_file()
                {
                    open_file(&mut self.state, &path.to_string_lossy());
                }
            }

            // Ctrl+Z: undo, Ctrl+Shift+Z: redo
            if i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z) {
                self.state.undo();
            }
            if i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Z) {
                self.state.redo();
            }

            // Ctrl+A: select all
            if i.modifiers.command && i.key_pressed(egui::Key::A) {
                self.state.select_all();
            }

            // Delete/Backspace: delete selected
            if i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace) {
                self.state.delete_selected();
            }

            // Home: go to start
            if i.key_pressed(egui::Key::Home) {
                self.state.playback_position = 0.0;
                self.state.time_offset = 0.0;
            }

            // Ctrl+=/-: zoom
            if i.modifiers.command && i.key_pressed(egui::Key::Equals) {
                self.state.time_scale = (self.state.time_scale * 1.25).min(2000.0);
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Minus) {
                self.state.time_scale = (self.state.time_scale / 1.25).max(20.0);
            }
        });
    }
}

impl eframe::App for MelodynApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Advance playback
        if self.state.playing {
            let dt = ctx.input(|i| i.predicted_dt as f64);
            self.state.advance_playback(dt);
            ctx.request_repaint();
        }

        self.handle_shortcuts(ctx);

        // Top: menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            menu_bar(ui, &mut self.state);
        });

        // Top: toolbar
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            crate::tools::toolbar(ui, &mut self.state);
        });

        // Bottom: status bar
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(22.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(&self.state.status_message);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some(ref key) = self.state.detected_key {
                            ui.label(format!("Key: {}", key));
                        }
                        if let Some(tempo) = self.state.detected_tempo {
                            ui.label(format!("{:.1} BPM", tempo));
                        }
                        ui.label(format!("{} notes", self.state.notes.len()));
                    });
                });
            });

        // Bottom: transport
        egui::TopBottomPanel::bottom("transport")
            .exact_height(44.0)
            .show(ctx, |ui| {
                crate::transport::transport_bar(ui, &mut self.state);
            });

        // Bottom: waveform overview
        egui::TopBottomPanel::bottom("waveform")
            .exact_height(60.0)
            .show(ctx, |ui| {
                crate::waveform::waveform_overview(ui, &mut self.state);
            });

        // Right: inspector
        egui::SidePanel::right("inspector")
            .resizable(true)
            .default_width(230.0)
            .min_width(180.0)
            .show(ctx, |ui| {
                crate::inspector::inspector_panel(ui, &mut self.state);
            });

        // Central: timeline ruler + piano roll editor
        egui::CentralPanel::default().show(ctx, |ui| {
            crate::timeline::timeline_ruler(ui, &self.state);
            ui.separator();
            crate::editor::piano_roll_editor(ui, &mut self.state);
        });
    }
}
