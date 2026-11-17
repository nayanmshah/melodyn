//! Playback engine: bridges the synthesis pipeline to the audio output.
//!
//! `PlaybackEngine` holds the audio data, detected notes, and the audio
//! output stream.  It renders audio blocks by either:
//!   - Playing back the original loaded audio (passthrough mode), or
//!   - Synthesising edited notes through the LSS pipeline.
//!
//! Communication with the UI is entirely through atomics -- no locks on
//! the audio path.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::audio_output::{AudioOutputConfig, AudioOutputStream, SharedPlaybackState};

// ---------------------------------------------------------------------------
// Playback mode
// ---------------------------------------------------------------------------

/// How the playback engine sources audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackMode {
    /// Play back the original loaded audio directly.
    Passthrough,
    /// Synthesise from LSS models (not yet wired -- falls back to passthrough).
    Synthesis,
}

// ---------------------------------------------------------------------------
// Note edit snapshot (plain-old-data, cheap to clone)
// ---------------------------------------------------------------------------

/// A lightweight copy of the edit offsets for one note, used by the render
/// path.  This avoids the render loop needing a reference into the UI state.
#[derive(Debug, Clone)]
pub struct NoteEditSnapshot {
    pub frequency: f64,
    pub midi_note: f64,
    pub start_time: f64,
    pub duration: f64,
    pub amplitude: f64,
    pub pitch_offset: f64,
    pub time_offset: f64,
    pub duration_scale: f64,
    pub amplitude_scale: f64,
    pub formant_shift: f64,
}

impl NoteEditSnapshot {
    /// Effective start time in seconds.
    pub fn effective_start(&self) -> f64 {
        self.start_time + self.time_offset
    }

    /// Effective end time in seconds.
    pub fn effective_end(&self) -> f64 {
        self.effective_start() + self.duration * self.duration_scale
    }

    /// Effective amplitude (clamped 0..1).
    pub fn effective_amplitude(&self) -> f64 {
        (self.amplitude * self.amplitude_scale).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Loop range (shared with atomics)
// ---------------------------------------------------------------------------

/// Loop state communicated to the render thread through atomics.
struct LoopState {
    enabled: AtomicBool,
    /// Loop start in samples (u64).
    start_samples: AtomicU64,
    /// Loop end in samples (u64).
    end_samples: AtomicU64,
}

impl LoopState {
    fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            start_samples: AtomicU64::new(0),
            end_samples: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Playback engine
// ---------------------------------------------------------------------------

/// The main playback engine.
///
/// Owns the audio output stream, holds a copy of the loaded audio for
/// passthrough playback, and feeds the ring buffer from a render method
/// that the UI calls each frame.
pub struct PlaybackEngine {
    /// The cpal audio output stream.
    output: AudioOutputStream,
    /// Loaded audio: left channel.
    audio_left: Vec<f64>,
    /// Loaded audio: right channel.
    audio_right: Vec<f64>,
    /// Sample rate of the loaded audio.
    sample_rate: f64,
    /// Total number of samples (per channel).
    total_samples: usize,
    /// Playback mode.
    mode: PlaybackMode,
    /// Pre-allocated interleaved render buffer (avoids per-frame allocation).
    render_buf: Vec<f64>,
    /// Render block size in frames.
    render_block_frames: usize,
    /// Number of output channels.
    num_channels: usize,
    /// Loop state shared with the render path.
    loop_state: Arc<LoopState>,
    /// Snapshot of note edits (updated from UI before each render pass).
    _note_snapshots: Vec<NoteEditSnapshot>,
}

impl PlaybackEngine {
    /// Create a new playback engine from loaded audio data.
    ///
    /// `left` and `right` are the two channels (for mono, pass the same
    /// data for both).  The engine opens the default audio output device.
    pub fn new(left: Vec<f64>, right: Vec<f64>, sample_rate: f64) -> Self {
        let total_samples = left.len();
        let config = AudioOutputConfig {
            sample_rate: sample_rate as u32,
            channels: 2,
            ring_buffer_capacity: 131072,
        };
        let output = AudioOutputStream::open(&config);

        let render_block_frames = 2048;
        let num_channels = 2;
        let render_buf = vec![0.0f64; render_block_frames * num_channels];

        Self {
            output,
            audio_left: left,
            audio_right: right,
            sample_rate,
            total_samples,
            mode: PlaybackMode::Passthrough,
            render_buf,
            render_block_frames,
            num_channels,
            loop_state: Arc::new(LoopState::new()),
            _note_snapshots: Vec::new(),
        }
    }

    /// Reference to the shared playback state (for the UI to control
    /// play/pause/stop/volume/position).
    pub fn shared_state(&self) -> &SharedPlaybackState {
        &self.output.state
    }

    /// Whether the output device is active.
    pub fn is_device_active(&self) -> bool {
        self.output.is_active()
    }

    /// Set the playback mode.
    pub fn set_mode(&mut self, mode: PlaybackMode) {
        self.mode = mode;
    }

    /// Set loop parameters (called from the UI thread).
    pub fn set_loop(&self, enabled: bool, start_seconds: f64, end_seconds: f64) {
        self.loop_state
            .enabled
            .store(enabled, Ordering::Release);
        self.loop_state
            .start_samples
            .store((start_seconds * self.sample_rate) as u64, Ordering::Release);
        self.loop_state
            .end_samples
            .store((end_seconds * self.sample_rate) as u64, Ordering::Release);
    }

    /// Current playback position in seconds.
    pub fn position_seconds(&self) -> f64 {
        self.output.state.position() as f64 / self.sample_rate
    }

    /// Seek to a position in seconds.
    pub fn seek(&self, seconds: f64) {
        let sample = (seconds * self.sample_rate).max(0.0) as u64;
        self.output.state.set_position(sample);
        self.output.flush();
    }

    /// Play.
    pub fn play(&self) {
        self.output.state.play();
    }

    /// Pause (keeps position).
    pub fn pause(&self) {
        self.output.state.pause();
    }

    /// Stop (resets position to 0).
    pub fn stop(&self) {
        self.output.state.stop();
        self.output.flush();
    }

    /// Set volume (0.0 .. 1.0).
    pub fn set_volume(&self, volume: f32) {
        self.output.state.set_volume(volume);
    }

    /// Total duration of the loaded audio in seconds.
    pub fn duration(&self) -> f64 {
        self.total_samples as f64 / self.sample_rate
    }

    /// Sample rate of the loaded audio.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Feed audio into the ring buffer.
    ///
    /// Call this once per UI frame (or from a background thread).  It will
    /// render as many blocks as the ring buffer has room for, up to a
    /// reasonable limit to avoid stalling the UI.
    pub fn pump(&mut self) {
        if !self.output.state.is_playing() {
            return;
        }

        // Fill the ring buffer up to ~100ms ahead of the current position.
        let target_buffered = (self.sample_rate * 0.1) as usize * self.num_channels;
        let max_iterations = 8; // Safety limit

        for _ in 0..max_iterations {
            let buffered = self.output.buffered_samples();
            if buffered >= target_buffered {
                break;
            }
            if self.output.available_write() < self.render_block_frames * self.num_channels {
                break;
            }
            self.render_and_write_block();
        }
    }

    /// Render one block and write it to the ring buffer.
    fn render_and_write_block(&mut self) {
        let pos = self.output.state.position() as usize;
        let block_frames = self.render_block_frames;

        // Check loop
        let loop_enabled = self.loop_state.enabled.load(Ordering::Acquire);
        let loop_start = self.loop_state.start_samples.load(Ordering::Acquire) as usize;
        let loop_end = self.loop_state.end_samples.load(Ordering::Acquire) as usize;

        let mut read_pos = pos;

        // Handle looping
        if loop_enabled && loop_end > loop_start && read_pos >= loop_end {
            let loop_len = loop_end - loop_start;
            read_pos = loop_start + (read_pos - loop_start) % loop_len;
            // Update the atomic position so the UI cursor stays in sync
            self.output
                .state
                .set_position(read_pos as u64);
        }

        // Check if we've reached the end
        if read_pos >= self.total_samples {
            if loop_enabled && loop_end > loop_start {
                // Wrap around to loop start
                read_pos = loop_start;
                self.output.state.set_position(read_pos as u64);
            } else {
                // Playback complete: stop
                self.output.state.pause();
                return;
            }
        }

        // Render interleaved stereo into self.render_buf
        let frames_to_render = block_frames.min(self.total_samples.saturating_sub(read_pos));

        match self.mode {
            PlaybackMode::Passthrough => {
                self.render_passthrough(read_pos, frames_to_render);
            }
            PlaybackMode::Synthesis => {
                // For now fall back to passthrough; full LSS synthesis
                // can be wired here later.
                self.render_passthrough(read_pos, frames_to_render);
            }
        }

        let interleaved_len = frames_to_render * self.num_channels;
        self.output
            .write_samples(&self.render_buf[..interleaved_len]);
    }

    /// Render a block of audio in passthrough mode (original loaded audio).
    fn render_passthrough(&mut self, start_frame: usize, num_frames: usize) {
        let end = (start_frame + num_frames).min(self.total_samples);
        for i in 0..num_frames {
            let idx = start_frame + i;
            let out_idx = i * self.num_channels;
            if idx < end {
                self.render_buf[out_idx] = self.audio_left[idx];
                self.render_buf[out_idx + 1] = self.audio_right[idx];
            } else {
                self.render_buf[out_idx] = 0.0;
                self.render_buf[out_idx + 1] = 0.0;
            }
        }
    }

    /// Render an offline (non-real-time) mix of all notes with edits applied,
    /// for export purposes.  Returns a stereo pair (left, right).
    ///
    /// For now this returns the original audio scaled by note amplitude edits.
    /// A full implementation would synthesise through the LSS pipeline.
    pub fn render_offline_mix(
        audio_left: &[f64],
        audio_right: &[f64],
        _notes: &[NoteEditSnapshot],
        _sample_rate: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        // Passthrough for now -- the edits will be applied once the full
        // synthesis pipeline is wired.
        (audio_left.to_vec(), audio_right.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_audio() -> (Vec<f64>, Vec<f64>) {
        let sr = 44100.0;
        let duration = 0.5; // seconds
        let len = (sr * duration) as usize;
        let freq = 440.0;
        let left: Vec<f64> = (0..len)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * freq * t).sin() * 0.5
            })
            .collect();
        let right = left.clone();
        (left, right)
    }

    #[test]
    fn test_playback_engine_creation() {
        let (left, right) = make_test_audio();
        let engine = PlaybackEngine::new(left, right, 44100.0);
        assert!((engine.duration() - 0.5).abs() < 0.01);
        assert!((engine.sample_rate() - 44100.0).abs() < 0.01);
        assert!(!engine.shared_state().is_playing());
    }

    #[test]
    fn test_playback_engine_play_stop() {
        let (left, right) = make_test_audio();
        let engine = PlaybackEngine::new(left, right, 44100.0);

        engine.play();
        assert!(engine.shared_state().is_playing());

        engine.pause();
        assert!(!engine.shared_state().is_playing());

        engine.play();
        engine.stop();
        assert!(!engine.shared_state().is_playing());
        assert_eq!(engine.shared_state().position(), 0);
    }

    #[test]
    fn test_playback_engine_seek() {
        let (left, right) = make_test_audio();
        let engine = PlaybackEngine::new(left, right, 44100.0);

        engine.seek(0.25);
        let pos = engine.position_seconds();
        assert!((pos - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_playback_engine_volume() {
        let (left, right) = make_test_audio();
        let engine = PlaybackEngine::new(left, right, 44100.0);

        engine.set_volume(0.5);
        assert!((engine.shared_state().volume() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_playback_engine_loop() {
        let (left, right) = make_test_audio();
        let engine = PlaybackEngine::new(left, right, 44100.0);
        engine.set_loop(true, 0.1, 0.3);
        // Loop state is set
        assert!(engine.loop_state.enabled.load(Ordering::Acquire));
    }

    #[test]
    fn test_playback_engine_pump_when_stopped() {
        let (left, right) = make_test_audio();
        let mut engine = PlaybackEngine::new(left, right, 44100.0);
        // Should not panic, and should not write any data
        engine.pump();
        assert_eq!(engine.output.buffered_samples(), 0);
    }

    #[test]
    fn test_playback_engine_pump_when_playing() {
        let (left, right) = make_test_audio();
        let mut engine = PlaybackEngine::new(left, right, 44100.0);
        engine.play();
        engine.pump();
        // Should have written some data into the ring buffer
        assert!(engine.output.buffered_samples() > 0);
    }

    #[test]
    fn test_render_block_produces_correct_count() {
        let (left, right) = make_test_audio();
        let mut engine = PlaybackEngine::new(left, right, 44100.0);
        engine.play();

        // Manually render one block
        engine.render_and_write_block();
        let buffered = engine.output.buffered_samples();
        // Should have written render_block_frames * 2 (stereo interleaved) samples
        assert_eq!(buffered, engine.render_block_frames * engine.num_channels);
    }

    #[test]
    fn test_render_passthrough_non_silent() {
        let (left, right) = make_test_audio();
        let mut engine = PlaybackEngine::new(left, right, 44100.0);

        // Render from the start
        let num_frames = 1024;
        engine.render_passthrough(0, num_frames);

        // Check that the output is non-silent (the test audio is a 440 Hz sine)
        let energy: f64 = engine.render_buf[..num_frames * 2]
            .iter()
            .map(|s| s * s)
            .sum();
        assert!(energy > 0.01, "Passthrough render should produce non-silent audio");
    }

    #[test]
    fn test_note_edit_snapshot_effective_values() {
        let snap = NoteEditSnapshot {
            frequency: 440.0,
            midi_note: 69.0,
            start_time: 1.0,
            duration: 0.5,
            amplitude: 0.8,
            pitch_offset: 2.0,
            time_offset: 0.1,
            duration_scale: 1.5,
            amplitude_scale: 0.5,
            formant_shift: 0.0,
        };

        assert!((snap.effective_start() - 1.1).abs() < 0.001);
        assert!((snap.effective_end() - 1.85).abs() < 0.001);
        assert!((snap.effective_amplitude() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_offline_mix_passthrough() {
        let left = vec![0.5; 1000];
        let right = vec![-0.3; 1000];
        let (out_l, out_r) = PlaybackEngine::render_offline_mix(&left, &right, &[], 44100.0);
        assert_eq!(out_l.len(), 1000);
        assert_eq!(out_r.len(), 1000);
        assert!((out_l[0] - 0.5).abs() < 1e-10);
        assert!((out_r[0] - (-0.3)).abs() < 1e-10);
    }
}
