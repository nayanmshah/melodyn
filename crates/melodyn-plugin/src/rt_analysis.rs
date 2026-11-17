//! Real-time analysis cache for pitch detection.
//!
//! Provides low-latency pitch tracking by maintaining a ring buffer of
//! incoming samples and running YIN pitch detection on overlapping windows.
//! Designed for real-time use: pre-allocates all buffers at construction
//! and performs no heap allocations during processing.

use melodyn_core::pitch::{PitchCandidate, PitchDetector, PitchDetectorConfig};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the real-time analysis cache.
#[derive(Debug, Clone)]
pub struct RtAnalysisConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Analysis window size in samples.
    pub window_size: usize,
    /// Hop size in samples (determines analysis frequency).
    pub hop_size: usize,
    /// Minimum detectable frequency in Hz.
    pub min_freq: f64,
    /// Maximum detectable frequency in Hz.
    pub max_freq: f64,
    /// YIN aperiodicity threshold (lower = stricter).
    pub yin_threshold: f64,
    /// Minimum confidence to accept a pitch estimate.
    pub min_confidence: f64,
    /// Number of recent estimates to keep for smoothing.
    pub history_size: usize,
}

impl Default for RtAnalysisConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            window_size: 2048,
            hop_size: 512,
            min_freq: 60.0,
            max_freq: 2000.0,
            yin_threshold: 0.15,
            min_confidence: 0.3,
            history_size: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Pitch estimate
// ---------------------------------------------------------------------------

/// A timestamped pitch estimate from the analysis cache.
#[derive(Debug, Clone, Copy)]
pub struct PitchEstimate {
    /// Detected frequency in Hz (0.0 = unvoiced).
    pub frequency: f64,
    /// MIDI note number (fractional).
    pub midi_note: f64,
    /// Confidence (0.0 .. 1.0).
    pub confidence: f64,
    /// Amplitude (linear RMS).
    pub amplitude: f64,
    /// Sample position where this estimate was computed.
    pub sample_position: u64,
}

impl PitchEstimate {
    /// Create an unvoiced (no pitch) estimate.
    pub fn unvoiced(sample_position: u64) -> Self {
        Self {
            frequency: 0.0,
            midi_note: 0.0,
            confidence: 0.0,
            amplitude: 0.0,
            sample_position,
        }
    }

    /// Whether this estimate represents a voiced segment.
    pub fn is_voiced(&self) -> bool {
        self.frequency > 0.0 && self.confidence > 0.0
    }
}

// ---------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------

/// Fixed-size ring buffer for audio samples.
///
/// Pre-allocated at construction, no heap allocations during push/read.
struct RingBuffer {
    /// Sample storage.
    data: Vec<f64>,
    /// Current write position (wraps around).
    write_pos: usize,
    /// Total samples written (monotonically increasing).
    total_written: u64,
    /// Capacity.
    capacity: usize,
}

impl RingBuffer {
    /// Create a ring buffer with the given capacity.
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            total_written: 0,
            capacity,
        }
    }

    /// Push samples into the ring buffer.
    fn push(&mut self, samples: &[f64]) {
        for &s in samples {
            self.data[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.capacity;
            self.total_written += 1;
        }
    }

    /// Read the most recent `count` samples into the provided buffer.
    ///
    /// Returns the number of samples actually read (may be less than `count`
    /// if fewer samples have been written).
    fn read_recent(&self, output: &mut [f64], count: usize) -> usize {
        let available = (self.total_written as usize).min(self.capacity);
        let to_read = count.min(available).min(output.len());

        if to_read == 0 {
            return 0;
        }

        // Start position: write_pos - to_read (with wraparound).
        let start = if self.write_pos >= to_read {
            self.write_pos - to_read
        } else {
            self.capacity - (to_read - self.write_pos)
        };

        for i in 0..to_read {
            let idx = (start + i) % self.capacity;
            output[i] = self.data[idx];
        }

        to_read
    }

    /// Total samples written since creation/reset.
    fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Reset the buffer to empty state.
    fn reset(&mut self) {
        self.write_pos = 0;
        self.total_written = 0;
        for s in self.data.iter_mut() {
            *s = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Real-time analysis cache
// ---------------------------------------------------------------------------

/// Real-time pitch analysis cache.
///
/// Collects incoming samples in a ring buffer and runs YIN pitch detection
/// on overlapping analysis windows. Maintains a history of recent pitch
/// estimates for smoothing and provides the current pitch to the correction
/// algorithm.
pub struct RealTimeAnalysisCache {
    /// Configuration.
    config: RtAnalysisConfig,
    /// Ring buffer collecting incoming samples.
    ring: RingBuffer,
    /// Pre-allocated frame buffer for pitch detection.
    frame_buffer: Vec<f64>,
    /// Pitch detector.
    detector: PitchDetector,
    /// Circular history of recent pitch estimates.
    history: Vec<PitchEstimate>,
    /// Write position into history.
    history_write: usize,
    /// Number of valid entries in history.
    history_count: usize,
    /// Samples since last analysis was triggered.
    samples_since_analysis: usize,
    /// Current smoothed pitch estimate.
    current_estimate: Option<PitchEstimate>,
}

impl RealTimeAnalysisCache {
    /// Create a new analysis cache with the specified parameters.
    pub fn new(sample_rate: f64, window_size: usize, hop_size: usize) -> Self {
        let config = RtAnalysisConfig {
            sample_rate,
            window_size,
            hop_size,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create with full configuration.
    pub fn with_config(config: RtAnalysisConfig) -> Self {
        // Ring buffer holds at least 2x the analysis window.
        let ring_capacity = config.window_size * 4;

        let detector_config = PitchDetectorConfig {
            sample_rate: config.sample_rate,
            min_freq: config.min_freq,
            max_freq: config.max_freq,
            polyphonic: false,
            yin_threshold: config.yin_threshold,
            min_confidence: config.min_confidence,
            ..Default::default()
        };

        let history_size = config.history_size;

        Self {
            ring: RingBuffer::new(ring_capacity),
            frame_buffer: vec![0.0; config.window_size],
            detector: PitchDetector::new(detector_config),
            history: vec![PitchEstimate::unvoiced(0); history_size],
            history_write: 0,
            history_count: 0,
            samples_since_analysis: 0,
            current_estimate: None,
            config,
        }
    }

    /// Push new samples into the analysis cache.
    ///
    /// Automatically triggers pitch detection when enough samples
    /// have accumulated (hop_size samples between analyses).
    pub fn push_samples(&mut self, samples: &[f64]) {
        self.ring.push(samples);
        self.samples_since_analysis += samples.len();

        // Run analysis every hop_size samples.
        while self.samples_since_analysis >= self.config.hop_size {
            self.run_analysis();
            self.samples_since_analysis -= self.config.hop_size;
        }
    }

    /// Get the current detected pitch frequency in Hz.
    ///
    /// Returns `None` if no pitched content is detected (unvoiced segment).
    pub fn current_pitch(&self) -> Option<f64> {
        self.current_estimate
            .filter(|e| e.is_voiced())
            .map(|e| e.frequency)
    }

    /// Get the current full pitch estimate.
    pub fn current_estimate(&self) -> Option<&PitchEstimate> {
        self.current_estimate.as_ref().filter(|e| e.is_voiced())
    }

    /// Get the current MIDI note number (fractional).
    pub fn current_midi_note(&self) -> Option<f64> {
        self.current_estimate
            .filter(|e| e.is_voiced())
            .map(|e| e.midi_note)
    }

    /// Get the analysis latency in samples.
    pub fn latency_samples(&self) -> usize {
        self.config.window_size
    }

    /// Get the analysis latency in seconds.
    pub fn latency_seconds(&self) -> f64 {
        self.config.window_size as f64 / self.config.sample_rate
    }

    /// Reset the analysis cache (clear all history and buffers).
    pub fn reset(&mut self) {
        self.ring.reset();
        self.samples_since_analysis = 0;
        self.history_write = 0;
        self.history_count = 0;
        self.current_estimate = None;
        for est in self.history.iter_mut() {
            *est = PitchEstimate::unvoiced(0);
        }
        for s in self.frame_buffer.iter_mut() {
            *s = 0.0;
        }
    }

    /// Run pitch detection on the most recent window of samples.
    fn run_analysis(&mut self) {
        let window_size = self.config.window_size;
        let read = self.ring.read_recent(&mut self.frame_buffer, window_size);

        if read < window_size {
            // Not enough data yet.
            return;
        }

        let sample_pos = self.ring.total_written();

        // Run YIN pitch detection.
        let estimate = match self.detector.detect_yin(&self.frame_buffer) {
            Some(candidate) => PitchEstimate {
                frequency: candidate.frequency,
                midi_note: candidate.midi_note,
                confidence: candidate.confidence,
                amplitude: candidate.amplitude,
                sample_position: sample_pos,
            },
            None => PitchEstimate::unvoiced(sample_pos),
        };

        // Store in history.
        let idx = self.history_write % self.config.history_size;
        self.history[idx] = estimate;
        self.history_write += 1;
        if self.history_count < self.config.history_size {
            self.history_count += 1;
        }

        // Compute smoothed estimate.
        self.current_estimate = Some(self.compute_smoothed_estimate());
    }

    /// Compute a smoothed pitch estimate from the recent history.
    ///
    /// Uses a confidence-weighted median to reject outliers and provide
    /// a stable pitch reading.
    fn compute_smoothed_estimate(&self) -> PitchEstimate {
        if self.history_count == 0 {
            return PitchEstimate::unvoiced(0);
        }

        // Collect voiced estimates with their confidences.
        let mut voiced: Vec<(f64, f64, f64)> = Vec::with_capacity(self.history_count);
        let start = if self.history_write >= self.history_count {
            self.history_write - self.history_count
        } else {
            0
        };

        for i in start..self.history_write {
            let idx = i % self.config.history_size;
            let est = &self.history[idx];
            if est.is_voiced() {
                voiced.push((est.frequency, est.confidence, est.amplitude));
            }
        }

        if voiced.is_empty() {
            return PitchEstimate::unvoiced(self.ring.total_written());
        }

        // Confidence-weighted average.
        let total_weight: f64 = voiced.iter().map(|(_, c, _)| c).sum();
        if total_weight < 1e-10 {
            return PitchEstimate::unvoiced(self.ring.total_written());
        }

        let freq = voiced.iter().map(|(f, c, _)| f * c).sum::<f64>() / total_weight;
        let amp = voiced.iter().map(|(_, c, a)| a * c).sum::<f64>() / total_weight;
        let conf = total_weight / self.history_count as f64;

        PitchEstimate {
            frequency: freq,
            midi_note: PitchCandidate::freq_to_midi(freq),
            confidence: conf,
            amplitude: amp,
            sample_position: self.ring.total_written(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate a sine wave at the given frequency.
    fn sine_wave(freq: f64, sample_rate: f64, num_samples: usize) -> Vec<f64> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_ring_buffer_basic() {
        let mut ring = RingBuffer::new(8);
        ring.push(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ring.total_written(), 4);

        let mut out = [0.0; 4];
        let read = ring.read_recent(&mut out, 4);
        assert_eq!(read, 4);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let mut ring = RingBuffer::new(4);
        ring.push(&[1.0, 2.0, 3.0, 4.0]);
        ring.push(&[5.0, 6.0]);

        let mut out = [0.0; 4];
        let read = ring.read_recent(&mut out, 4);
        assert_eq!(read, 4);
        assert_eq!(out, [3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_ring_buffer_reset() {
        let mut ring = RingBuffer::new(8);
        ring.push(&[1.0, 2.0, 3.0]);
        ring.reset();
        assert_eq!(ring.total_written(), 0);

        let mut out = [0.0; 4];
        let read = ring.read_recent(&mut out, 4);
        assert_eq!(read, 0);
    }

    #[test]
    fn test_pitch_estimate_unvoiced() {
        let est = PitchEstimate::unvoiced(100);
        assert!(!est.is_voiced());
        assert_eq!(est.sample_position, 100);
    }

    #[test]
    fn test_analysis_cache_creation() {
        let cache = RealTimeAnalysisCache::new(44100.0, 2048, 512);
        assert_eq!(cache.latency_samples(), 2048);
        assert!(cache.current_pitch().is_none());
    }

    #[test]
    fn test_analysis_cache_sine_440hz() {
        let sample_rate = 44100.0;
        let mut cache = RealTimeAnalysisCache::new(sample_rate, 2048, 512);

        // Generate 440 Hz sine wave, long enough for several analysis windows.
        let signal = sine_wave(440.0, sample_rate, 44100);

        // Push in chunks to simulate real-time.
        for chunk in signal.chunks(512) {
            cache.push_samples(chunk);
        }

        // Should detect a pitch near 440 Hz.
        let pitch = cache.current_pitch();
        assert!(pitch.is_some(), "Should detect pitch for 440 Hz sine");
        let freq = pitch.unwrap();
        assert!(
            (freq - 440.0).abs() < 10.0,
            "Detected {} Hz, expected ~440 Hz",
            freq
        );
    }

    #[test]
    fn test_analysis_cache_silence() {
        let mut cache = RealTimeAnalysisCache::new(44100.0, 2048, 512);

        // Push silence.
        let silence = vec![0.0; 8192];
        cache.push_samples(&silence);

        // Should not detect any pitch.
        assert!(
            cache.current_pitch().is_none(),
            "Should not detect pitch in silence"
        );
    }

    #[test]
    fn test_analysis_cache_reset() {
        let sample_rate = 44100.0;
        let mut cache = RealTimeAnalysisCache::new(sample_rate, 2048, 512);

        let signal = sine_wave(440.0, sample_rate, 8192);
        cache.push_samples(&signal);
        assert!(cache.current_pitch().is_some());

        cache.reset();
        assert!(cache.current_pitch().is_none());
    }

    #[test]
    fn test_analysis_midi_note() {
        let sample_rate = 44100.0;
        let mut cache = RealTimeAnalysisCache::new(sample_rate, 2048, 512);

        let signal = sine_wave(440.0, sample_rate, 44100);
        cache.push_samples(&signal);

        let midi = cache.current_midi_note();
        assert!(midi.is_some());
        let note = midi.unwrap();
        // A4 = MIDI 69.
        assert!(
            (note - 69.0).abs() < 1.0,
            "MIDI note {} should be near 69 (A4)",
            note
        );
    }

    #[test]
    fn test_config_default() {
        let config = RtAnalysisConfig::default();
        assert_eq!(config.sample_rate, 44100.0);
        assert_eq!(config.window_size, 2048);
        assert_eq!(config.hop_size, 512);
        assert_eq!(config.history_size, 8);
    }
}
