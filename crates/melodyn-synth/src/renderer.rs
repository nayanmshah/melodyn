//! Component and element renderers.
//!
//! Implements MULSSComponentRenderer and MULSSElementRenderer from the
//! original Melodyne engine. The component renderer handles individual
//! partials, while the element renderer orchestrates all partials of a
//! note with proper overlap-add windowing.

use crate::lss::{LssNoteModel, LssPartial};
use crate::overlap_add::OverlapAddConfig;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Render request
// ---------------------------------------------------------------------------

/// A request to render a portion of audio from an LSS model.
///
/// Corresponds to MULSSElementRenderRequest.
#[derive(Debug, Clone)]
pub struct RenderRequest {
    /// Start sample in the output buffer.
    pub start_sample: usize,
    /// Number of samples to render.
    pub num_samples: usize,
    /// Sample rate.
    pub sample_rate: f64,
    /// Pitch shift in semitones.
    pub pitch_shift: f64,
    /// Time stretch factor (1.0 = no stretch).
    pub time_stretch: f64,
    /// Formant preservation amount (0.0 = no preservation, 1.0 = full).
    pub formant_preservation: f64,
}

impl Default for RenderRequest {
    fn default() -> Self {
        Self {
            start_sample: 0,
            num_samples: 0,
            sample_rate: 44100.0,
            pitch_shift: 0.0,
            time_stretch: 1.0,
            formant_preservation: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Component renderer (single partial)
// ---------------------------------------------------------------------------

/// Renders a single sinusoidal partial with amplitude and phase interpolation.
///
/// Corresponds to MULSSComponentRenderer.
pub struct ComponentRenderer {
    /// Current phase accumulator.
    phase: f64,
    /// Previous frame amplitude (for interpolation).
    prev_amplitude: f64,
}

impl ComponentRenderer {
    /// Create a new component renderer.
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            prev_amplitude: 0.0,
        }
    }

    /// Reset the renderer state.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.prev_amplitude = 0.0;
    }

    /// Render one synthesis frame of a partial into the output buffer.
    ///
    /// Uses linear amplitude interpolation across the frame and continuous
    /// phase accumulation for glitch-free output.
    pub fn render_frame(
        &mut self,
        partial: &LssPartial,
        frame_index: usize,
        frequency: f64,
        sample_rate: f64,
        hop_size: usize,
        output: &mut [f64],
        output_offset: usize,
    ) {
        let target_amp = if frame_index < partial.synth_amps.len() {
            partial.synth_amps[frame_index]
        } else {
            0.0
        };

        let omega = 2.0 * PI * frequency / sample_rate;

        for i in 0..hop_size {
            if output_offset + i >= output.len() {
                break;
            }

            // Linear amplitude ramp within the frame
            let t = i as f64 / hop_size as f64;
            let amp = self.prev_amplitude * (1.0 - t) + target_amp * t;

            output[output_offset + i] += amp * self.phase.sin();
            self.phase += omega;
        }

        // Normalize phase
        while self.phase > 2.0 * PI {
            self.phase -= 2.0 * PI;
        }

        self.prev_amplitude = target_amp;
    }
}

impl Default for ComponentRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Element renderer (full note)
// ---------------------------------------------------------------------------

/// Renders all partials of a single note, managing per-partial component
/// renderers and applying overlap-add windowing.
///
/// Corresponds to MULSSElementRenderer.
pub struct ElementRenderer {
    /// Per-partial component renderers.
    component_renderers: Vec<ComponentRenderer>,
    /// Overlap-add configuration.
    ola_config: OverlapAddConfig,
}

impl ElementRenderer {
    /// Create a new element renderer for the given number of harmonics.
    pub fn new(max_harmonics: usize, ola_config: OverlapAddConfig) -> Self {
        let component_renderers = (0..max_harmonics).map(|_| ComponentRenderer::new()).collect();
        Self {
            component_renderers,
            ola_config,
        }
    }

    /// Reset all component renderers.
    pub fn reset(&mut self) {
        for cr in &mut self.component_renderers {
            cr.reset();
        }
    }

    /// Render a complete note from its LSS model.
    pub fn render(&mut self, model: &LssNoteModel, request: &RenderRequest) -> Vec<f64> {
        let num_samples = if request.num_samples > 0 {
            request.num_samples
        } else {
            model.num_samples()
        };
        let mut output = vec![0.0f64; num_samples];
        let hop = model.hop_size;
        let sr = model.sample_rate;
        let pitch_ratio = 2.0f64.powf(request.pitch_shift / 12.0);

        // Ensure we have enough component renderers
        while self.component_renderers.len() < model.partials.len() {
            self.component_renderers.push(ComponentRenderer::new());
        }

        // Render frame by frame
        for frame_idx in 0..model.num_frames {
            let output_offset = frame_idx * hop;
            if output_offset >= num_samples {
                break;
            }

            for (p_idx, partial) in model.partials.iter().enumerate() {
                if !partial.active {
                    continue;
                }

                let freq = partial.frequency * pitch_ratio;
                if freq > sr / 2.0 {
                    continue;
                }

                self.component_renderers[p_idx].render_frame(
                    partial,
                    frame_idx,
                    freq,
                    sr,
                    hop,
                    &mut output,
                    output_offset,
                );
            }
        }

        output
    }

    /// Render a complete note with overlap-add windowing applied.
    pub fn render_windowed(&mut self, model: &LssNoteModel, request: &RenderRequest) -> Vec<f64> {
        let raw = self.render(model, request);
        // Apply overlap-add synthesis window
        apply_synthesis_window(&raw, model.hop_size, &self.ola_config)
    }
}

/// Apply a synthesis window for overlap-add reconstruction.
///
/// In OLA synthesis, each frame is windowed and overlapped. For perfect
/// reconstruction with a Hann window and 50% overlap, the window is
/// applied and the overlapping regions sum to unity.
fn apply_synthesis_window(audio: &[f64], hop_size: usize, config: &OverlapAddConfig) -> Vec<f64> {
    let window_size = config.window_size;
    let window = &config.synthesis_window;

    if window.is_empty() || audio.is_empty() {
        return audio.to_vec();
    }

    let num_samples = audio.len();
    let mut output = vec![0.0f64; num_samples];

    let mut pos = 0;
    while pos + window_size <= num_samples {
        for i in 0..window_size {
            output[pos + i] += audio[pos + i] * window[i];
        }
        pos += hop_size;
    }

    // Handle remaining samples
    if pos < num_samples {
        let remaining = num_samples - pos;
        for i in 0..remaining.min(window_size) {
            let w = if i < window.len() { window[i] } else { 1.0 };
            output[pos + i] += audio[pos + i] * w;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Generator renderer (multi-note orchestrator)
// ---------------------------------------------------------------------------

/// Orchestrates rendering of multiple notes into a single audio buffer.
///
/// Corresponds to MULSSGeneratorRenderer.
pub struct GeneratorRenderer {
    /// Per-note element renderers.
    element_renderers: Vec<ElementRenderer>,
    /// Maximum harmonics per note.
    max_harmonics: usize,
    /// OLA configuration.
    ola_config: OverlapAddConfig,
}

impl GeneratorRenderer {
    /// Create a new generator renderer.
    pub fn new(max_harmonics: usize, ola_config: OverlapAddConfig) -> Self {
        Self {
            element_renderers: Vec::new(),
            max_harmonics,
            ola_config,
        }
    }

    /// Render all notes and mix into a single buffer.
    pub fn render_all(
        &mut self,
        models: &[LssNoteModel],
        total_samples: usize,
        request: &RenderRequest,
    ) -> Vec<f64> {
        let mut output = vec![0.0f64; total_samples];

        // Ensure we have enough element renderers
        while self.element_renderers.len() < models.len() {
            self.element_renderers.push(ElementRenderer::new(
                self.max_harmonics,
                self.ola_config.clone(),
            ));
        }

        for (i, model) in models.iter().enumerate() {
            self.element_renderers[i].reset();
            let note_audio = self.element_renderers[i].render(model, request);

            // Mix into output at the correct position
            let start = model.note_index * model.hop_size; // Approximate start
            for (j, &s) in note_audio.iter().enumerate() {
                let idx = start + j;
                if idx < output.len() {
                    output[idx] += s;
                }
            }
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lss::{LssCorrelator, LssNoteModel, LssPartial};
    use melodyn_core::note::DetectedNote;

    fn make_test_note() -> DetectedNote {
        DetectedNote {
            frequency: 440.0,
            midi_note: 69.0,
            start_time: 0.0,
            duration: 0.25,
            start_sample: 0,
            end_sample: 11025,
            amplitude: 0.5,
            peak_amplitude: 0.5,
            confidence: 0.9,
            pitch_contour: vec![440.0; 5],
            amplitude_contour: vec![0.1, 0.5, 0.5, 0.5, 0.1],
            vibrato_rate: 0.0,
            vibrato_depth: 0.0,
            attack_time: 0.0,
            release_time: 0.0,
        }
    }

    #[test]
    fn test_component_renderer_basic() {
        let mut cr = ComponentRenderer::new();
        let mut partial = LssPartial::new(1, 440.0, 4);
        partial.synth_amps = vec![0.5, 0.5, 0.5, 0.5];
        partial.active = true;

        let mut output = vec![0.0f64; 1024];
        cr.render_frame(&partial, 0, 440.0, 44100.0, 256, &mut output, 0);

        // Should produce non-zero output
        let energy: f64 = output[..256].iter().map(|s| s * s).sum();
        assert!(energy > 0.0, "Component renderer should produce audio");
    }

    #[test]
    fn test_element_renderer() {
        let note = make_test_note();
        let correlator = LssCorrelator::new(4, 256, 44100.0);
        let model = correlator.build_model_simple(&note, 0);

        let ola_config = OverlapAddConfig::default();
        let mut renderer = ElementRenderer::new(4, ola_config);

        let request = RenderRequest {
            sample_rate: 44100.0,
            ..Default::default()
        };
        let audio = renderer.render(&model, &request);

        assert!(!audio.is_empty());
        let rms = (audio.iter().map(|s| s * s).sum::<f64>() / audio.len() as f64).sqrt();
        assert!(rms > 0.001, "Element renderer should produce audible output");
    }

    #[test]
    fn test_pitch_shift_rendering() {
        let note = make_test_note();
        let correlator = LssCorrelator::new(2, 256, 44100.0);
        let model = correlator.build_model_simple(&note, 0);

        let ola_config = OverlapAddConfig::default();
        let mut renderer = ElementRenderer::new(2, ola_config);

        let request = RenderRequest {
            sample_rate: 44100.0,
            pitch_shift: 12.0, // One octave up
            ..Default::default()
        };
        let audio = renderer.render(&model, &request);
        assert!(!audio.is_empty());
    }
}
