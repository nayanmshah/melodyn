//! Melodyn Synth -- LSS resynthesis engine.
//!
//! This crate implements the Local Sound Synthesis (LSS) engine:
//!
//! - **LSS**: Per-partial sinusoidal model (generator and correlator)
//! - **Renderer**: Component and element renderers with amplitude/phase control
//! - **Formant**: 18-band formant model with preservation and shifting
//! - **Overlap Add**: Window-based overlap-add framework
//! - **Time Stretch**: Warp-function-driven time stretching

pub mod formant;
pub mod lss;
pub mod overlap_add;
pub mod renderer;
pub mod time_stretch;

// Re-export key types
pub use formant::{FormantModel, FormantProcessor, FormantShapeParams, FormantSynthParams};
pub use lss::{LssCorrelator, LssGenerator, LssGeneratorConfig, LssNoteModel, LssPartial};
pub use overlap_add::{OverlapAddConfig, OverlapAddProcessor};
pub use renderer::{ComponentRenderer, ElementRenderer, GeneratorRenderer, RenderRequest};
pub use time_stretch::{TimeStretchConfig, TimeStretcher, WarpFunction};
