//! Cross-platform audio output via `cpal`.
//!
//! Provides `AudioOutputStream` which opens the default audio output device,
//! accepts audio data through a lock-free ring buffer, and supports
//! play/pause/stop control, volume, and position reporting -- all via atomics
//! so the real-time audio callback never touches a lock.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};

use crate::audio_engine::AudioRingBuffer;

// ---------------------------------------------------------------------------
// Playback state flags shared between UI and audio callback
// ---------------------------------------------------------------------------

/// Shared atomic state between the audio callback and the controlling thread.
///
/// All fields use atomics so the real-time callback never blocks.
#[derive(Clone)]
pub struct SharedPlaybackState {
    /// Whether the stream is actively playing.
    pub playing: Arc<AtomicBool>,
    /// Master volume as fixed-point (u32 bits reinterpreted as f32).
    volume_bits: Arc<AtomicU32>,
    /// Current playback position in samples (written by callback, read by UI).
    pub position_samples: Arc<AtomicU64>,
    /// Total number of samples rendered since the stream was created.
    pub total_rendered: Arc<AtomicU64>,
    /// Signal to shut down the stream callback.
    pub shutdown: Arc<AtomicBool>,
}

impl SharedPlaybackState {
    /// Create a new shared state with default values.
    pub fn new() -> Self {
        Self {
            playing: Arc::new(AtomicBool::new(false)),
            volume_bits: Arc::new(AtomicU32::new(f32::to_bits(0.8))),
            position_samples: Arc::new(AtomicU64::new(0)),
            total_rendered: Arc::new(AtomicU64::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the volume (0.0 to 1.0).
    pub fn set_volume(&self, volume: f32) {
        let v = volume.clamp(0.0, 1.0);
        self.volume_bits.store(f32::to_bits(v), Ordering::Release);
    }

    /// Get the current volume.
    pub fn volume(&self) -> f32 {
        f32::from_bits(self.volume_bits.load(Ordering::Acquire))
    }

    /// Get the current playback position in samples.
    pub fn position(&self) -> u64 {
        self.position_samples.load(Ordering::Acquire)
    }

    /// Set the playback position (used for seek).
    pub fn set_position(&self, pos: u64) {
        self.position_samples.store(pos, Ordering::Release);
    }

    /// Whether the stream is playing.
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Acquire)
    }

    /// Start playback.
    pub fn play(&self) {
        self.playing.store(true, Ordering::Release);
    }

    /// Pause playback.
    pub fn pause(&self) {
        self.playing.store(false, Ordering::Release);
    }

    /// Stop playback and reset position.
    pub fn stop(&self) {
        self.playing.store(false, Ordering::Release);
        self.position_samples.store(0, Ordering::Release);
    }
}

impl Default for SharedPlaybackState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Audio output stream configuration
// ---------------------------------------------------------------------------

/// Configuration for the audio output stream.
#[derive(Debug, Clone)]
pub struct AudioOutputConfig {
    /// Desired sample rate (will fall back to device default).
    pub sample_rate: u32,
    /// Number of output channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Ring buffer capacity in samples (per channel, interleaved).
    pub ring_buffer_capacity: usize,
}

impl Default for AudioOutputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            ring_buffer_capacity: 131072, // ~1.5 seconds at stereo 44100
        }
    }
}

// ---------------------------------------------------------------------------
// Audio output stream
// ---------------------------------------------------------------------------

/// Cross-platform audio output stream.
///
/// Opens the default output device via `cpal`, reads audio data from a
/// lock-free ring buffer, and plays it. The ring buffer must be fed
/// by calling `write_samples()` from outside the audio callback (e.g.,
/// from a dedicated render thread or from the main thread between frames).
pub struct AudioOutputStream {
    /// The cpal stream handle (keeps the stream alive).
    _stream: Option<Stream>,
    /// Ring buffer shared with the audio callback.
    ring_buffer: Arc<AudioRingBuffer>,
    /// Shared playback state.
    pub state: SharedPlaybackState,
    /// Actual sample rate negotiated with the device.
    pub actual_sample_rate: u32,
    /// Actual channel count negotiated with the device.
    pub actual_channels: u16,
}

impl AudioOutputStream {
    /// Open the default audio output device and create a stream.
    ///
    /// Returns `Ok` even if no audio device is available -- in that case
    /// `is_active()` returns false and writes are silently dropped.
    pub fn open(config: &AudioOutputConfig) -> Self {
        let state = SharedPlaybackState::new();
        let ring_buffer = Arc::new(AudioRingBuffer::new(config.ring_buffer_capacity));

        let host = cpal::default_host();

        let device = match host.default_output_device() {
            Some(d) => d,
            None => {
                log::warn!("No audio output device found -- playback disabled");
                return Self {
                    _stream: None,
                    ring_buffer,
                    state,
                    actual_sample_rate: config.sample_rate,
                    actual_channels: config.channels,
                };
            }
        };

        let supported_config = match device.default_output_config() {
            Ok(c) => c,
            Err(e) => {
                log::warn!(
                    "Cannot get default output config: {} -- playback disabled",
                    e
                );
                return Self {
                    _stream: None,
                    ring_buffer,
                    state,
                    actual_sample_rate: config.sample_rate,
                    actual_channels: config.channels,
                };
            }
        };

        let sample_format = supported_config.sample_format();
        let actual_sample_rate = supported_config.sample_rate().0;
        let actual_channels = supported_config.channels();

        let stream_config: StreamConfig = supported_config.into();

        let rb = Arc::clone(&ring_buffer);
        let st = state.clone();

        let stream = match sample_format {
            SampleFormat::F32 => Self::build_stream_f32(&device, &stream_config, rb, st),
            SampleFormat::I16 => Self::build_stream_i16(&device, &stream_config, rb, st),
            SampleFormat::U16 => Self::build_stream_u16(&device, &stream_config, rb, st),
            _ => {
                log::warn!(
                    "Unsupported sample format {:?} -- playback disabled",
                    sample_format
                );
                None
            }
        };

        if let Some(ref s) = stream {
            if let Err(e) = s.play() {
                log::warn!("Failed to start audio stream: {}", e);
            }
        }

        Self {
            _stream: stream,
            ring_buffer,
            state,
            actual_sample_rate,
            actual_channels,
        }
    }

    /// Build a cpal stream that outputs f32 samples.
    fn build_stream_f32(
        device: &cpal::Device,
        config: &StreamConfig,
        ring_buffer: Arc<AudioRingBuffer>,
        state: SharedPlaybackState,
    ) -> Option<Stream> {
        let channels = config.channels as usize;
        // Pre-allocate a read buffer sized for a reasonable callback (2048 frames).
        let max_read = 4096 * channels;
        let mut read_buf = vec![0.0f64; max_read];

        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let num_frames = data.len() / channels;
                    let total_interleaved = num_frames * channels;

                    if state.shutdown.load(Ordering::Acquire) {
                        data.fill(0.0);
                        return;
                    }

                    if !state.playing.load(Ordering::Acquire) {
                        data.fill(0.0);
                        return;
                    }

                    let volume = f32::from_bits(state.volume_bits.load(Ordering::Acquire));

                    // Read interleaved f64 from ring buffer
                    let to_read = total_interleaved.min(read_buf.len());
                    let read_count = ring_buffer.read(&mut read_buf[..to_read]);

                    // Convert f64 -> f32 with volume, fill output
                    for i in 0..data.len() {
                        if i < read_count {
                            data[i] = (read_buf[i] as f32) * volume;
                        } else {
                            data[i] = 0.0;
                        }
                    }

                    // Update position: advance by the number of frames consumed
                    let frames_consumed = read_count / channels;
                    state
                        .position_samples
                        .fetch_add(frames_consumed as u64, Ordering::Release);
                    state
                        .total_rendered
                        .fetch_add(frames_consumed as u64, Ordering::Release);
                },
                move |err| {
                    log::error!("Audio output stream error: {}", err);
                },
                None,
            )
            .ok();

        stream
    }

    /// Build a cpal stream that outputs i16 samples.
    fn build_stream_i16(
        device: &cpal::Device,
        config: &StreamConfig,
        ring_buffer: Arc<AudioRingBuffer>,
        state: SharedPlaybackState,
    ) -> Option<Stream> {
        let channels = config.channels as usize;
        let max_read = 4096 * channels;
        let mut read_buf = vec![0.0f64; max_read];

        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let num_frames = data.len() / channels;
                    let total_interleaved = num_frames * channels;

                    if state.shutdown.load(Ordering::Acquire)
                        || !state.playing.load(Ordering::Acquire)
                    {
                        data.fill(0);
                        return;
                    }

                    let volume = f32::from_bits(state.volume_bits.load(Ordering::Acquire)) as f64;

                    let to_read = total_interleaved.min(read_buf.len());
                    let read_count = ring_buffer.read(&mut read_buf[..to_read]);

                    for i in 0..data.len() {
                        if i < read_count {
                            let val = (read_buf[i] * volume).clamp(-1.0, 1.0);
                            data[i] = (val * 32767.0) as i16;
                        } else {
                            data[i] = 0;
                        }
                    }

                    let frames_consumed = read_count / channels;
                    state
                        .position_samples
                        .fetch_add(frames_consumed as u64, Ordering::Release);
                    state
                        .total_rendered
                        .fetch_add(frames_consumed as u64, Ordering::Release);
                },
                move |err| {
                    log::error!("Audio output stream error: {}", err);
                },
                None,
            )
            .ok();

        stream
    }

    /// Build a cpal stream that outputs u16 samples.
    fn build_stream_u16(
        device: &cpal::Device,
        config: &StreamConfig,
        ring_buffer: Arc<AudioRingBuffer>,
        state: SharedPlaybackState,
    ) -> Option<Stream> {
        let channels = config.channels as usize;
        let max_read = 4096 * channels;
        let mut read_buf = vec![0.0f64; max_read];

        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                    let num_frames = data.len() / channels;
                    let total_interleaved = num_frames * channels;

                    if state.shutdown.load(Ordering::Acquire)
                        || !state.playing.load(Ordering::Acquire)
                    {
                        data.fill(32768);
                        return;
                    }

                    let volume = f32::from_bits(state.volume_bits.load(Ordering::Acquire)) as f64;

                    let to_read = total_interleaved.min(read_buf.len());
                    let read_count = ring_buffer.read(&mut read_buf[..to_read]);

                    for i in 0..data.len() {
                        if i < read_count {
                            let val = (read_buf[i] * volume).clamp(-1.0, 1.0);
                            data[i] = ((val * 0.5 + 0.5) * 65535.0) as u16;
                        } else {
                            data[i] = 32768;
                        }
                    }

                    let frames_consumed = read_count / channels;
                    state
                        .position_samples
                        .fetch_add(frames_consumed as u64, Ordering::Release);
                    state
                        .total_rendered
                        .fetch_add(frames_consumed as u64, Ordering::Release);
                },
                move |err| {
                    log::error!("Audio output stream error: {}", err);
                },
                None,
            )
            .ok();

        stream
    }

    /// Whether the stream is active (device was opened successfully).
    pub fn is_active(&self) -> bool {
        self._stream.is_some()
    }

    /// Write interleaved f64 samples into the ring buffer.
    ///
    /// Returns the number of samples actually written. If the ring buffer
    /// is full, samples are silently dropped (the caller should produce
    /// data at approximately the right rate).
    pub fn write_samples(&self, data: &[f64]) -> usize {
        self.ring_buffer.write(data)
    }

    /// How many samples can be written without blocking.
    pub fn available_write(&self) -> usize {
        self.ring_buffer.available_write()
    }

    /// How many samples are queued for playback.
    pub fn buffered_samples(&self) -> usize {
        self.ring_buffer.available_read()
    }

    /// Reset the ring buffer (e.g., on seek).
    pub fn flush(&self) {
        self.ring_buffer.reset();
    }
}

impl Drop for AudioOutputStream {
    fn drop(&mut self) {
        self.state.shutdown.store(true, Ordering::Release);
        // The cpal Stream is dropped automatically, stopping the callback.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_playback_state_defaults() {
        let state = SharedPlaybackState::new();
        assert!(!state.is_playing());
        assert!((state.volume() - 0.8).abs() < 0.001);
        assert_eq!(state.position(), 0);
    }

    #[test]
    fn test_shared_playback_state_play_pause_stop() {
        let state = SharedPlaybackState::new();

        state.play();
        assert!(state.is_playing());

        state.pause();
        assert!(!state.is_playing());

        state.set_position(44100);
        assert_eq!(state.position(), 44100);

        state.stop();
        assert!(!state.is_playing());
        assert_eq!(state.position(), 0);
    }

    #[test]
    fn test_shared_playback_state_volume() {
        let state = SharedPlaybackState::new();

        state.set_volume(0.5);
        assert!((state.volume() - 0.5).abs() < 0.001);

        // Clamp to range
        state.set_volume(2.0);
        assert!((state.volume() - 1.0).abs() < 0.001);

        state.set_volume(-0.5);
        assert!(state.volume().abs() < 0.001);
    }

    #[test]
    fn test_audio_output_config_defaults() {
        let config = AudioOutputConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert!(config.ring_buffer_capacity > 0);
    }

    #[test]
    fn test_audio_output_stream_creation() {
        // This test may fail in CI where no audio device is available,
        // but should not panic.
        let config = AudioOutputConfig::default();
        let stream = AudioOutputStream::open(&config);
        // On machines without audio, is_active() returns false.
        // On machines with audio, it returns true.
        // Either way, the struct should be usable.
        let _ = stream.is_active();
        assert_eq!(stream.buffered_samples(), 0);
    }

    #[test]
    fn test_audio_output_write_and_buffer() {
        let config = AudioOutputConfig::default();
        let stream = AudioOutputStream::open(&config);

        let samples = vec![0.5f64; 1024];
        let written = stream.write_samples(&samples);
        assert_eq!(written, 1024);
        assert_eq!(stream.buffered_samples(), 1024);

        stream.flush();
        assert_eq!(stream.buffered_samples(), 0);
    }
}
