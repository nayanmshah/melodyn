//! Main application struct and state management.
//!
//! Contains the `MelodynApp` struct (implementing `eframe::App`), the
//! `AppState` for all editor state, `EditableNote` wrapping detected notes
//! with edit offsets, undo/redo, file I/O, analysis pipeline, menu bar,
//! and real-time audio playback via `PlaybackEngine`.

use std::collections::HashSet;

use eframe::egui;
use melodyn_core::{
    AnalysisConfig, AnalysisResult, DetectedNote, DetectionPreset, PitchCandidate,
    SpectralAnalyzer, TempoAnalyzer, TempoAnalyzerConfig,
};
use melodyn_core::playback::{NoteEditSnapshot, PlaybackEngine};
use melodyn_core::scale::{detect_key, distribution_from_notes};
use melodyn_core::tempo::BeatGrid;
use melodyn_io::{AudioBuffer, read_wav, write_wav, WavWriteConfig};

use crate::clipboard::Clipboard;
use crate::tools::{SnapMode, Tool};
use crate::tracks::TrackListState;

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

    /// Convert to a lightweight snapshot for the render path.
    pub fn to_edit_snapshot(&self) -> NoteEditSnapshot {
        NoteEditSnapshot {
            frequency: self.detected.frequency,
            midi_note: self.detected.midi_note,
            start_time: self.detected.start_time,
            duration: self.detected.duration,
            amplitude: self.detected.amplitude,
            pitch_offset: self.pitch_offset,
            time_offset: self.time_offset,
            duration_scale: self.duration_scale,
            amplitude_scale: self.amplitude_scale,
            formant_shift: self.formant_shift,
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

    // -- Clipboard --
    pub clipboard: Clipboard,

    // -- Multi-track --
    pub track_list: TrackListState,

    // -- Drag-and-drop --
    pub drop_hover: bool,

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

            clipboard: Clipboard::new(),

            track_list: TrackListState::default(),

            drop_hover: false,

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
/// Returns the stereo channel pair for the playback engine.
fn open_file(state: &mut AppState, path: &str) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    state.status_message = format!("Loading {}...", path);

    // Read WAV
    let buffer = match read_wav(path) {
        Ok(buf) => buf,
        Err(e) => {
            state.status_message = format!("Error loading file: {}", e);
            return None;
        }
    };

    let sample_rate = buffer.sample_rate();
    let mono = buffer.to_mono();
    let duration = buffer.duration();

    // Extract stereo channels for the playback engine before moving buffer
    let left = buffer.channel(0).to_vec();
    let right = if buffer.num_channels() >= 2 {
        buffer.channel(1).to_vec()
    } else {
        left.clone()
    };

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

    Some((left, right, sample_rate))
}

/// Export the current audio to a WAV file, applying note edits.
///
/// Collects note edit snapshots and renders through the offline mix,
/// then writes via the melodyn-io WAV writer.
fn export_wav_with_edits(state: &AppState, path: &str) {
    let buffer = match state.audio_buffer.as_ref() {
        Some(b) => b,
        None => return,
    };

    let left = buffer.channel(0).to_vec();
    let right = if buffer.num_channels() >= 2 {
        buffer.channel(1).to_vec()
    } else {
        left.clone()
    };

    // Collect note edit snapshots
    let snapshots: Vec<NoteEditSnapshot> = state
        .notes
        .iter()
        .map(|n| n.to_edit_snapshot())
        .collect();

    // Render offline mix with edits applied
    let (out_left, out_right) = PlaybackEngine::render_offline_mix(
        &left,
        &right,
        &snapshots,
        state.sample_rate,
    );

    // Build an AudioBuffer and write
    let out_buffer = AudioBuffer::from_stereo(out_left, out_right, state.sample_rate);
    let config = WavWriteConfig::default();
    match write_wav(path, &out_buffer, &config) {
        Ok(()) => {
            log::info!("Exported to {} with edits applied", path);
        }
        Err(e) => {
            log::error!("Export failed: {}", e);
        }
    }
}

// ---------------------------------------------------------------------------
// Menu bar
// ---------------------------------------------------------------------------

/// Draw the application menu bar.
pub fn menu_bar(ui: &mut egui::Ui, state: &mut AppState, engine: &mut Option<PlaybackEngine>) {
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
                    if let Some((left, right, sr)) =
                        open_file(state, &path.to_string_lossy())
                    {
                        let new_engine = PlaybackEngine::new(left, right, sr);
                        new_engine.set_volume(state.volume);
                        new_engine.set_loop(
                            state.loop_enabled,
                            state.loop_start,
                            state.loop_end,
                        );
                        *engine = Some(new_engine);
                    }
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
                        export_wav_with_edits(state, &path.to_string_lossy());
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
                    if let Some((left, right, sr)) = open_file(state, path) {
                        let new_engine = PlaybackEngine::new(left, right, sr);
                        new_engine.set_volume(state.volume);
                        *engine = Some(new_engine);
                    }
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
    /// Real-time playback engine (created when a file is loaded).
    pub playback_engine: Option<PlaybackEngine>,
}

impl MelodynApp {
    /// Create the application, optionally loading initial state.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            state: AppState::default(),
            playback_engine: None,
        }
    }

    /// Synchronise the playback engine with the UI state.
    ///
    /// - Pushes volume and loop settings from UI to engine.
    /// - Reads the atomic playback position back from the engine.
    /// - Pumps audio data into the ring buffer.
    fn sync_playback(&mut self) {
        let engine = match self.playback_engine.as_mut() {
            Some(e) => e,
            None => return,
        };

        // Push volume to engine
        engine.set_volume(self.state.volume);

        // Push loop state to engine
        engine.set_loop(
            self.state.loop_enabled,
            self.state.loop_start,
            self.state.loop_end,
        );

        // Read position back from the engine's atomic counter
        if self.state.playing {
            let engine_pos = engine.position_seconds();
            self.state.playback_position = engine_pos;

            // Check if engine stopped itself (e.g., reached end of audio)
            if !engine.shared_state().is_playing() {
                self.state.playing = false;
                self.state.status_message = format!(
                    "Stopped at {:.2}s",
                    self.state.playback_position
                );
            }

            // Feed audio into the ring buffer
            engine.pump();
        }
    }

    /// Start playback through the audio engine.
    fn start_playback(&mut self) {
        self.state.playing = true;
        if let Some(ref engine) = self.playback_engine {
            // Seek the engine to the current UI position before playing
            engine.seek(self.state.playback_position);
            engine.play();
        }
        self.state.status_message = "Playing".to_string();
    }

    /// Stop playback through the audio engine.
    fn stop_playback(&mut self) {
        self.state.playing = false;
        if let Some(ref engine) = self.playback_engine {
            engine.pause();
        }
        self.state.status_message = format!(
            "Stopped at {:.2}s",
            self.state.playback_position
        );
    }

    /// Handle files dropped onto the application window.
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        // Check for hover state (files being dragged over)
        let has_hover = ctx.input(|i| !i.raw.hovered_files.is_empty());
        self.state.drop_hover = has_hover;

        // Check for dropped files
        let dropped: Vec<egui::DroppedFile> =
            ctx.input(|i| i.raw.dropped_files.clone());

        for file in &dropped {
            if let Some(ref path) = file.path {
                let path_str = path.to_string_lossy();
                // Only accept .wav files
                if path_str.to_lowercase().ends_with(".wav") {
                    if let Some((left, right, sr)) =
                        open_file(&mut self.state, &path_str)
                    {
                        let new_engine = PlaybackEngine::new(left, right, sr);
                        new_engine.set_volume(self.state.volume);
                        self.playback_engine = Some(new_engine);
                    }
                } else {
                    self.state.status_message = format!(
                        "Unsupported file type: {}. Only .wav files are supported.",
                        path.display()
                    );
                }
            }
        }

        if !dropped.is_empty() {
            self.state.drop_hover = false;
        }
    }

    /// Handle global keyboard shortcuts.
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        // Tool shortcuts
        crate::tools::handle_tool_shortcuts(ctx, &mut self.state);

        let mut toggle_play = false;
        let mut do_open = false;
        let mut do_undo = false;
        let mut do_redo = false;
        let mut do_select_all = false;
        let mut do_deselect_all = false;
        let mut do_delete = false;
        let mut do_home = false;
        let mut do_zoom_in = false;
        let mut do_zoom_out = false;
        let mut do_copy = false;
        let mut do_cut = false;
        let mut do_paste = false;
        let mut do_duplicate = false;
        let mut do_escape = false;
        let mut do_fit_view = false;
        let mut nudge_pitch: f64 = 0.0;
        let mut nudge_time: f64 = 0.0;

        ctx.input(|i| {
            // Space: toggle play
            if i.key_pressed(egui::Key::Space) {
                toggle_play = true;
            }

            // Ctrl+O: open
            if i.modifiers.command && i.key_pressed(egui::Key::O) {
                do_open = true;
            }

            // Ctrl+Z: undo, Ctrl+Shift+Z: redo
            if i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z) {
                do_undo = true;
            }
            if i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Z) {
                do_redo = true;
            }

            // Ctrl+A: select all
            if i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::A) {
                do_select_all = true;
            }

            // Ctrl+Shift+A: deselect all
            if i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::A) {
                do_deselect_all = true;
            }

            // Ctrl+C: copy
            if i.modifiers.command && i.key_pressed(egui::Key::C) {
                do_copy = true;
            }

            // Ctrl+X: cut
            if i.modifiers.command && i.key_pressed(egui::Key::X) {
                do_cut = true;
            }

            // Ctrl+V: paste
            if i.modifiers.command && i.key_pressed(egui::Key::V) {
                do_paste = true;
            }

            // Ctrl+D: duplicate
            if i.modifiers.command && i.key_pressed(egui::Key::D) {
                do_duplicate = true;
            }

            // Delete/Backspace: delete selected
            if i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace) {
                do_delete = true;
            }

            // Escape: deselect all / cancel
            if i.key_pressed(egui::Key::Escape) {
                do_escape = true;
            }

            // Home: go to start
            if i.key_pressed(egui::Key::Home) {
                do_home = true;
            }

            // Ctrl+=/-: zoom
            if i.modifiers.command && i.key_pressed(egui::Key::Equals) {
                do_zoom_in = true;
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Minus) {
                do_zoom_out = true;
            }

            // +/- keys (without Ctrl): zoom in/out
            if !i.modifiers.command && i.key_pressed(egui::Key::Equals) {
                do_zoom_in = true;
            }
            if !i.modifiers.command && i.key_pressed(egui::Key::Minus) {
                do_zoom_out = true;
            }

            // F key: fit view to show all notes
            if i.key_pressed(egui::Key::F) && !i.modifiers.command {
                do_fit_view = true;
            }

            // Arrow keys: nudge selected notes
            let big_step = i.modifiers.shift;
            if i.key_pressed(egui::Key::ArrowUp) && !i.modifiers.command {
                nudge_pitch = if big_step { 12.0 } else { 1.0 };
            }
            if i.key_pressed(egui::Key::ArrowDown) && !i.modifiers.command {
                nudge_pitch = if big_step { -12.0 } else { -1.0 };
            }
            if i.key_pressed(egui::Key::ArrowRight) && !i.modifiers.command {
                nudge_time = if big_step { 0.1 } else { 0.01 };
            }
            if i.key_pressed(egui::Key::ArrowLeft) && !i.modifiers.command {
                nudge_time = if big_step { -0.1 } else { -0.01 };
            }
        });

        if toggle_play {
            if self.state.playing {
                self.stop_playback();
            } else {
                self.start_playback();
            }
        }

        if do_open {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("WAV files", &["wav"])
                .pick_file()
            {
                if let Some((left, right, sr)) =
                    open_file(&mut self.state, &path.to_string_lossy())
                {
                    let new_engine = PlaybackEngine::new(left, right, sr);
                    new_engine.set_volume(self.state.volume);
                    self.playback_engine = Some(new_engine);
                }
            }
        }

        if do_undo {
            self.state.undo();
        }
        if do_redo {
            self.state.redo();
        }
        if do_select_all {
            self.state.select_all();
        }
        if do_deselect_all || do_escape {
            self.state.selected_notes.clear();
            self.state.status_message = "Deselected all".to_string();
        }
        if do_delete {
            self.state.delete_selected();
        }
        if do_home {
            self.state.playback_position = 0.0;
            self.state.time_offset = 0.0;
            if let Some(ref engine) = self.playback_engine {
                engine.seek(0.0);
            }
        }
        if do_zoom_in {
            self.state.time_scale = (self.state.time_scale * 1.25).min(2000.0);
        }
        if do_zoom_out {
            self.state.time_scale = (self.state.time_scale / 1.25).max(20.0);
        }

        // Clipboard operations
        if do_copy {
            let mut clipboard = self.state.clipboard.clone();
            crate::clipboard::copy_selected(&self.state, &mut clipboard);
            let count = clipboard.len();
            self.state.clipboard = clipboard;
            if count > 0 {
                self.state.status_message = format!(
                    "Copied {} note{}",
                    count,
                    if count > 1 { "s" } else { "" }
                );
            }
        }
        if do_cut {
            let mut clipboard = self.state.clipboard.clone();
            crate::clipboard::cut_selected(&mut self.state, &mut clipboard);
            self.state.clipboard = clipboard;
        }
        if do_paste {
            let clipboard = self.state.clipboard.clone();
            crate::clipboard::paste_at_cursor(&mut self.state, &clipboard);
        }
        if do_duplicate {
            crate::clipboard::duplicate_selected(&mut self.state);
        }

        // Arrow key nudging
        if (nudge_pitch.abs() > 0.001 || nudge_time.abs() > 0.001)
            && !self.state.selected_notes.is_empty()
        {
            self.state.push_undo("Nudge Notes");
            let selected: Vec<usize> = self.state.selected_notes.iter().copied().collect();
            for idx in selected {
                if idx < self.state.notes.len() {
                    self.state.notes[idx].pitch_offset += nudge_pitch;
                    self.state.notes[idx].time_offset += nudge_time;
                }
            }
        }

        // Fit view
        if do_fit_view {
            fit_view_to_notes(&mut self.state);
        }
    }
}

impl eframe::App for MelodynApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Sync playback engine: push volume/loop, read position, pump audio
        self.sync_playback();

        // If playing, request continuous repaints so the cursor moves smoothly
        // and the ring buffer gets pumped frequently.
        if self.state.playing {
            ctx.request_repaint();
        }

        // Handle drag-and-drop
        self.handle_dropped_files(ctx);

        self.handle_shortcuts(ctx);

        // Top: menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            menu_bar(ui, &mut self.state, &mut self.playback_engine);
        });

        // Top: toolbar
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            crate::tools::toolbar(ui, &mut self.state);
        });

        // Bottom: status bar with detailed info
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(22.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(&self.state.status_message);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // File name
                        if let Some(ref path) = self.state.file_path {
                            if let Some(name) = std::path::Path::new(path)
                                .file_name()
                                .and_then(|n| n.to_str())
                            {
                                ui.label(
                                    egui::RichText::new(name)
                                        .small()
                                        .color(egui::Color32::from_rgb(160, 160, 180)),
                                );
                                ui.separator();
                            }
                        }
                        // Sample rate and duration
                        if self.state.duration > 0.0 {
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.0} Hz | {:.1}s",
                                    self.state.sample_rate, self.state.duration
                                ))
                                .small(),
                            );
                            ui.separator();
                        }
                        // Key
                        if let Some(ref key) = self.state.detected_key {
                            ui.label(format!("Key: {}", key));
                            ui.separator();
                        }
                        // Tempo
                        if let Some(tempo) = self.state.detected_tempo {
                            ui.label(format!("{:.1} BPM", tempo));
                            ui.separator();
                        }
                        // Current tool name
                        ui.label(
                            egui::RichText::new(self.state.active_tool.label())
                                .small()
                                .color(egui::Color32::from_rgb(180, 160, 100)),
                        );
                        ui.separator();
                        // Selected / total notes
                        let selected = self.state.selected_notes.len();
                        let total = self.state.notes.len();
                        if selected > 0 {
                            ui.label(format!("{}/{} notes", selected, total));
                        } else {
                            ui.label(format!("{} notes", total));
                        }
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

        // Left: track list panel
        // We temporarily take the track list out to avoid double borrow.
        let mut track_list = std::mem::take(&mut self.state.track_list);
        egui::SidePanel::left("track_list")
            .resizable(true)
            .default_width(200.0)
            .min_width(150.0)
            .show(ctx, |ui| {
                crate::tracks::track_list_panel(
                    ui,
                    &mut track_list,
                    &mut self.state,
                );
            });
        self.state.track_list = track_list;

        // Drag-and-drop overlay indicator
        if self.state.drop_hover {
            egui::Area::new(egui::Id::new("drop_overlay"))
                .fixed_pos(egui::pos2(0.0, 0.0))
                .show(ctx, |ui| {
                    let screen = ctx.screen_rect();
                    let painter = ui.painter();
                    painter.rect_filled(
                        screen,
                        0.0,
                        egui::Color32::from_rgba_premultiplied(40, 80, 160, 60),
                    );
                    painter.text(
                        screen.center(),
                        egui::Align2::CENTER_CENTER,
                        "Drop WAV file to open",
                        egui::FontId::proportional(28.0),
                        egui::Color32::WHITE,
                    );
                });
        }

        // Central: timeline ruler + piano roll editor
        egui::CentralPanel::default().show(ctx, |ui| {
            crate::timeline::timeline_ruler(ui, &self.state);
            ui.separator();
            crate::editor::piano_roll_editor(ui, &mut self.state);
        });
    }
}

// ---------------------------------------------------------------------------
// Fit view helper
// ---------------------------------------------------------------------------

/// Adjust the view to show all notes in the editor.
fn fit_view_to_notes(state: &mut AppState) {
    if state.notes.is_empty() {
        state.time_offset = 0.0;
        state.pitch_offset = 48.0;
        return;
    }

    let min_time = state
        .notes
        .iter()
        .map(|n| n.effective_start())
        .fold(f64::MAX, f64::min);
    let max_time = state
        .notes
        .iter()
        .map(|n| n.effective_end())
        .fold(f64::MIN, f64::max);
    let min_midi = state
        .notes
        .iter()
        .map(|n| n.effective_midi())
        .fold(f64::MAX, f64::min);
    let max_midi = state
        .notes
        .iter()
        .map(|n| n.effective_midi())
        .fold(f64::MIN, f64::max);

    let time_range = (max_time - min_time).max(1.0);
    state.time_offset = min_time - time_range * 0.05;
    // time_scale will be adjusted to fit content (approximate)
    state.time_scale = 100.0; // A reasonable default; editor width-dependent

    state.pitch_offset = min_midi - 4.0;
    let pitch_range = (max_midi - min_midi).max(12.0);
    state.pitch_scale = (400.0 / pitch_range).clamp(4.0, 60.0);

    state.status_message = "View fitted to notes".to_string();
}
