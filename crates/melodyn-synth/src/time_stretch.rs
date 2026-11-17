//! Warp function time stretching.
//!
//! Implements time stretching via warp functions that map output time to
//! input time. This allows non-uniform time stretching (e.g., stretching
//! only specific portions) and is the foundation for Melodyne's time
//! editing capabilities.

use crate::overlap_add::{OverlapAddConfig, OverlapAddProcessor};

// ---------------------------------------------------------------------------
// Warp function
// ---------------------------------------------------------------------------

/// A warp function maps output time to input time.
///
/// For uniform stretching by factor S:
///   warp(t_out) = t_out / S
///
/// For non-uniform stretching, the warp function can have arbitrary shape
/// defined by breakpoints.
#[derive(Debug, Clone)]
pub struct WarpFunction {
    /// Breakpoints: (output_time, input_time) pairs, sorted by output_time.
    breakpoints: Vec<(f64, f64)>,
}

impl WarpFunction {
    /// Create a uniform time stretch warp function.
    ///
    /// `factor` > 1.0 means slower (output is longer).
    /// `duration` is the input duration in seconds.
    pub fn uniform(factor: f64, duration: f64) -> Self {
        Self {
            breakpoints: vec![(0.0, 0.0), (duration * factor, duration)],
        }
    }

    /// Create a warp function from explicit breakpoints.
    ///
    /// Breakpoints are (output_time, input_time) pairs.
    /// They must be sorted by output_time.
    pub fn from_breakpoints(mut breakpoints: Vec<(f64, f64)>) -> Self {
        breakpoints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self { breakpoints }
    }

    /// Create an identity warp function (no stretching).
    pub fn identity(duration: f64) -> Self {
        Self {
            breakpoints: vec![(0.0, 0.0), (duration, duration)],
        }
    }

    /// Map output time to input time.
    pub fn map(&self, output_time: f64) -> f64 {
        if self.breakpoints.is_empty() {
            return output_time;
        }

        // Before first breakpoint
        if output_time <= self.breakpoints[0].0 {
            return self.breakpoints[0].1;
        }

        // After last breakpoint
        let last = self.breakpoints.last().unwrap();
        if output_time >= last.0 {
            return last.1;
        }

        // Find the segment
        for i in 0..self.breakpoints.len() - 1 {
            let (out0, in0) = self.breakpoints[i];
            let (out1, in1) = self.breakpoints[i + 1];

            if output_time >= out0 && output_time <= out1 {
                let t = if (out1 - out0).abs() > 1e-10 {
                    (output_time - out0) / (out1 - out0)
                } else {
                    0.0
                };
                return in0 + t * (in1 - in0);
            }
        }

        output_time
    }

    /// Compute the local stretch factor at a given output time.
    ///
    /// Returns dt_input/dt_output, the inverse of the local stretch.
    /// Values < 1.0 mean stretching (slowing), > 1.0 mean compressing (speeding).
    pub fn local_rate(&self, output_time: f64) -> f64 {
        if self.breakpoints.len() < 2 {
            return 1.0;
        }

        for i in 0..self.breakpoints.len() - 1 {
            let (out0, in0) = self.breakpoints[i];
            let (out1, in1) = self.breakpoints[i + 1];

            if output_time >= out0 && output_time <= out1 {
                let d_out = out1 - out0;
                let d_in = in1 - in0;
                return if d_out.abs() > 1e-10 {
                    d_in / d_out
                } else {
                    1.0
                };
            }
        }

        1.0
    }

    /// Output duration (last breakpoint's output time).
    pub fn output_duration(&self) -> f64 {
        self.breakpoints.last().map(|b| b.0).unwrap_or(0.0)
    }

    /// Input duration (last breakpoint's input time).
    pub fn input_duration(&self) -> f64 {
        self.breakpoints.last().map(|b| b.1).unwrap_or(0.0)
    }

    /// Average stretch factor.
    pub fn average_stretch(&self) -> f64 {
        let in_dur = self.input_duration();
        let out_dur = self.output_duration();
        if in_dur > 1e-10 {
            out_dur / in_dur
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// Time stretcher
// ---------------------------------------------------------------------------

/// Configuration for time stretching.
#[derive(Debug, Clone)]
pub struct TimeStretchConfig {
    /// Overlap-add configuration.
    pub ola_config: OverlapAddConfig,
    /// Whether to use phase-locked processing.
    pub phase_locked: bool,
    /// Crossfade duration at frame boundaries (samples).
    pub crossfade_samples: usize,
}

impl Default for TimeStretchConfig {
    fn default() -> Self {
        Self {
            ola_config: OverlapAddConfig::default(),
            phase_locked: true,
            crossfade_samples: 64,
        }
    }
}

/// Time stretcher using warp-function-driven overlap-add.
pub struct TimeStretcher {
    config: TimeStretchConfig,
}

impl TimeStretcher {
    /// Create a new time stretcher.
    pub fn new(config: TimeStretchConfig) -> Self {
        Self { config }
    }

    /// Apply uniform time stretching to audio.
    pub fn stretch_uniform(&self, audio: &[f64], factor: f64) -> Vec<f64> {
        let processor = OverlapAddProcessor::new(self.config.ola_config.clone());
        processor.time_stretch(audio, factor)
    }

    /// Apply warp-function-driven time stretching.
    ///
    /// Reads from the input audio at positions determined by the warp
    /// function, using windowed overlap-add for smooth reconstruction.
    pub fn stretch_warp(&self, audio: &[f64], warp: &WarpFunction) -> Vec<f64> {
        let sr = self.config.ola_config.sample_rate;
        let hop = self.config.ola_config.hop_size;
        let window_size = self.config.ola_config.window_size;
        let analysis_window = &self.config.ola_config.analysis_window;
        let synthesis_window = &self.config.ola_config.synthesis_window;

        let output_duration = warp.output_duration();
        let output_samples = (output_duration * sr).ceil() as usize;
        let mut output = vec![0.0f64; output_samples];
        let mut normalization = vec![0.0f64; output_samples];

        let half_window = window_size / 2;

        // Process in overlapping frames
        let mut out_pos = 0;
        while out_pos + window_size <= output_samples {
            let out_time = (out_pos + half_window) as f64 / sr;
            let in_time = warp.map(out_time);
            let in_center = (in_time * sr).round() as isize;

            // Extract and window the input frame
            let mut frame = vec![0.0f64; window_size];
            for i in 0..window_size {
                let in_idx = in_center - half_window as isize + i as isize;
                let sample = if in_idx >= 0 && (in_idx as usize) < audio.len() {
                    audio[in_idx as usize]
                } else {
                    0.0
                };
                let aw = if i < analysis_window.len() {
                    analysis_window[i]
                } else {
                    1.0
                };
                frame[i] = sample * aw;
            }

            // Apply synthesis window and add to output
            for i in 0..window_size {
                let sw = if i < synthesis_window.len() {
                    synthesis_window[i]
                } else {
                    1.0
                };
                output[out_pos + i] += frame[i] * sw;
                normalization[out_pos + i] += sw * sw;
            }

            out_pos += hop;
        }

        // Normalize
        for i in 0..output_samples {
            if normalization[i] > 1e-10 {
                output[i] /= normalization[i];
            }
        }

        output
    }

    /// Access the configuration.
    pub fn config(&self) -> &TimeStretchConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_warp_function_uniform() {
        let warp = WarpFunction::uniform(2.0, 1.0);
        // Output duration is 2 seconds for 1 second input
        assert!((warp.output_duration() - 2.0).abs() < 1e-10);
        assert!((warp.input_duration() - 1.0).abs() < 1e-10);
        // At output t=1.0, input should be t=0.5
        assert!((warp.map(1.0) - 0.5).abs() < 1e-10);
        // Local rate should be 0.5 everywhere
        assert!((warp.local_rate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_warp_function_identity() {
        let warp = WarpFunction::identity(1.0);
        assert!((warp.map(0.5) - 0.5).abs() < 1e-10);
        assert!((warp.local_rate(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_warp_function_nonuniform() {
        // First half normal speed, second half double speed
        let warp = WarpFunction::from_breakpoints(vec![
            (0.0, 0.0),
            (0.5, 0.5), // normal speed for first half
            (1.0, 1.5), // double speed for second half (covers 1.0 seconds in 0.5)
        ]);

        assert!((warp.map(0.25) - 0.25).abs() < 1e-10);
        assert!((warp.map(0.75) - 1.0).abs() < 1e-10);
        assert!((warp.local_rate(0.25) - 1.0).abs() < 1e-10); // Normal speed
        assert!((warp.local_rate(0.75) - 2.0).abs() < 1e-10); // Double speed
    }

    #[test]
    fn test_uniform_stretch() {
        let config = TimeStretchConfig {
            ola_config: OverlapAddConfig::new(2048, 512, 44100.0),
            ..Default::default()
        };
        let stretcher = TimeStretcher::new(config);

        let sr = 44100.0;
        let audio: Vec<f64> = (0..44100)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let stretched = stretcher.stretch_uniform(&audio, 1.5);
        // Should be roughly 1.5x longer
        assert!(
            stretched.len() > audio.len(),
            "Stretched {} should be > original {}",
            stretched.len(),
            audio.len()
        );
    }

    #[test]
    fn test_warp_stretch() {
        let config = TimeStretchConfig {
            ola_config: OverlapAddConfig::new(2048, 512, 44100.0),
            ..Default::default()
        };
        let stretcher = TimeStretcher::new(config);

        let sr = 44100.0;
        let duration = 1.0;
        let audio: Vec<f64> = (0..(sr as usize))
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let warp = WarpFunction::uniform(2.0, duration);
        let stretched = stretcher.stretch_warp(&audio, &warp);

        // Should be roughly 2x longer
        assert!(
            stretched.len() > audio.len(),
            "Warped output {} should be > input {}",
            stretched.len(),
            audio.len()
        );
    }
}
