//! Audio effects chain: parametric EQ, gain, panning, compressor, reverb.
//!
//! Based on MUAudioEffect, MUAudioEffectChain, MUFilterEngine, MUGainTransformer,
//! MUStereoPanner. Provides a trait-based effect system with serial chaining,
//! biquad filter engine, and built-in effects.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Audio effect trait
// ---------------------------------------------------------------------------

/// Trait for all audio effects.
pub trait AudioEffect: Send {
    /// Process a stereo buffer in-place. Each slice is one channel.
    fn process(&mut self, left: &mut [f64], right: &mut [f64], sample_rate: f64);
    /// Reset internal state (e.g., filter memory, delay lines).
    fn reset(&mut self);
    /// Effect name for display.
    fn name(&self) -> &str;
    /// Whether the effect is bypassed.
    fn is_bypassed(&self) -> bool;
    /// Set bypass state.
    fn set_bypassed(&mut self, bypassed: bool);
    /// Wet/dry mix (0.0 = fully dry, 1.0 = fully wet).
    fn wet_dry(&self) -> f64;
    /// Set wet/dry mix.
    fn set_wet_dry(&mut self, mix: f64);
}

// ---------------------------------------------------------------------------
// Biquad filter types
// ---------------------------------------------------------------------------

/// Standard biquad filter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiquadType {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Allpass,
    Peaking,
    LowShelf,
    HighShelf,
}

/// Second-order biquad filter (direct form II transposed).
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl BiquadFilter {
    /// Create a new filter with the given parameters.
    pub fn new(filter_type: BiquadType, freq: f64, q: f64, gain_db: f64, sample_rate: f64) -> Self {
        let mut f = Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        };
        f.compute_coefficients(filter_type, freq, q, gain_db, sample_rate);
        f
    }

    /// Recompute coefficients for new parameters.
    pub fn compute_coefficients(
        &mut self,
        filter_type: BiquadType,
        freq: f64,
        q: f64,
        gain_db: f64,
        sample_rate: f64,
    ) {
        let w0 = 2.0 * PI * freq / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let a_lin = 10.0f64.powf(gain_db / 40.0);

        let (b0, b1, b2, a0, a1, a2) = match filter_type {
            BiquadType::Lowpass => {
                let b1 = 1.0 - cos_w0;
                let b0 = b1 / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Highpass => {
                let b1 = -(1.0 + cos_w0);
                let b0 = (1.0 + cos_w0) / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Bandpass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Allpass => {
                let b0 = 1.0 - alpha;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 + alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Peaking => {
                let b0 = 1.0 + alpha * a_lin;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a_lin;
                let a0 = 1.0 + alpha / a_lin;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a_lin;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::LowShelf => {
                let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;
                let b0 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = 2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
                let b2 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = -2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
                let a2 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighShelf => {
                let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;
                let b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
                let b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
                let b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
                let a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
                let a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
                let a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    /// Process a single sample (direct form II transposed).
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }

    /// Process a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Filter engine (parametric EQ)
// ---------------------------------------------------------------------------

/// A band in the parametric EQ.
#[derive(Debug, Clone)]
pub struct EqBand {
    pub filter_type: BiquadType,
    pub frequency: f64,
    pub q: f64,
    pub gain_db: f64,
    pub enabled: bool,
}

/// Multi-band parametric EQ using cascaded biquad filters.
pub struct ParametricEq {
    bands: Vec<EqBand>,
    filters_l: Vec<BiquadFilter>,
    filters_r: Vec<BiquadFilter>,
    sample_rate: f64,
    bypassed: bool,
    wet_dry: f64,
}

impl ParametricEq {
    pub fn new(bands: Vec<EqBand>, sample_rate: f64) -> Self {
        let filters_l: Vec<BiquadFilter> = bands
            .iter()
            .map(|b| BiquadFilter::new(b.filter_type, b.frequency, b.q, b.gain_db, sample_rate))
            .collect();
        let filters_r = filters_l.clone();
        Self {
            bands,
            filters_l,
            filters_r,
            sample_rate,
            bypassed: false,
            wet_dry: 1.0,
        }
    }

    /// Update a band's parameters.
    pub fn set_band(&mut self, index: usize, band: EqBand) {
        if index < self.bands.len() {
            self.filters_l[index].compute_coefficients(
                band.filter_type,
                band.frequency,
                band.q,
                band.gain_db,
                self.sample_rate,
            );
            self.filters_r[index].compute_coefficients(
                band.filter_type,
                band.frequency,
                band.q,
                band.gain_db,
                self.sample_rate,
            );
            self.bands[index] = band;
        }
    }
}

impl AudioEffect for ParametricEq {
    fn process(&mut self, left: &mut [f64], right: &mut [f64], _sample_rate: f64) {
        if self.bypassed {
            return;
        }
        let dry_mix = 1.0 - self.wet_dry;
        for i in 0..left.len() {
            let dry_l = left[i];
            let dry_r = right[i];
            let mut wet_l = left[i];
            let mut wet_r = right[i];
            for (idx, band) in self.bands.iter().enumerate() {
                if band.enabled {
                    wet_l = self.filters_l[idx].process_sample(wet_l);
                    wet_r = self.filters_r[idx].process_sample(wet_r);
                }
            }
            left[i] = wet_l * self.wet_dry + dry_l * dry_mix;
            right[i] = wet_r * self.wet_dry + dry_r * dry_mix;
        }
    }
    fn reset(&mut self) {
        for f in &mut self.filters_l {
            f.reset();
        }
        for f in &mut self.filters_r {
            f.reset();
        }
    }
    fn name(&self) -> &str {
        "Parametric EQ"
    }
    fn is_bypassed(&self) -> bool {
        self.bypassed
    }
    fn set_bypassed(&mut self, b: bool) {
        self.bypassed = b;
    }
    fn wet_dry(&self) -> f64 {
        self.wet_dry
    }
    fn set_wet_dry(&mut self, mix: f64) {
        self.wet_dry = mix.clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Gain transformer
// ---------------------------------------------------------------------------

/// Simple gain/trim effect.
pub struct GainTransformer {
    pub gain_db: f64,
    bypassed: bool,
    wet_dry: f64,
}

impl GainTransformer {
    pub fn new(gain_db: f64) -> Self {
        Self {
            gain_db,
            bypassed: false,
            wet_dry: 1.0,
        }
    }

    fn linear_gain(&self) -> f64 {
        10.0f64.powf(self.gain_db / 20.0)
    }
}

impl AudioEffect for GainTransformer {
    fn process(&mut self, left: &mut [f64], right: &mut [f64], _sample_rate: f64) {
        if self.bypassed {
            return;
        }
        let gain = self.linear_gain();
        let wet = self.wet_dry;
        let dry = 1.0 - wet;
        for s in left.iter_mut() {
            *s = *s * gain * wet + *s * dry;
        }
        for s in right.iter_mut() {
            *s = *s * gain * wet + *s * dry;
        }
    }
    fn reset(&mut self) {}
    fn name(&self) -> &str {
        "Gain"
    }
    fn is_bypassed(&self) -> bool {
        self.bypassed
    }
    fn set_bypassed(&mut self, b: bool) {
        self.bypassed = b;
    }
    fn wet_dry(&self) -> f64 {
        self.wet_dry
    }
    fn set_wet_dry(&mut self, mix: f64) {
        self.wet_dry = mix.clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Stereo panner
// ---------------------------------------------------------------------------

/// Stereo panner using constant-power panning law.
pub struct StereoPanner {
    /// Pan position: -1.0 = hard left, 0.0 = center, 1.0 = hard right.
    pub pan: f64,
    bypassed: bool,
    wet_dry: f64,
}

impl StereoPanner {
    pub fn new(pan: f64) -> Self {
        Self {
            pan: pan.clamp(-1.0, 1.0),
            bypassed: false,
            wet_dry: 1.0,
        }
    }

    fn gains(&self) -> (f64, f64) {
        // Constant-power pan law
        let angle = (self.pan + 1.0) * 0.25 * PI; // 0 to PI/2
        (angle.cos(), angle.sin())
    }
}

impl AudioEffect for StereoPanner {
    fn process(&mut self, left: &mut [f64], right: &mut [f64], _sample_rate: f64) {
        if self.bypassed {
            return;
        }
        let (gain_l, gain_r) = self.gains();
        let wet = self.wet_dry;
        let dry = 1.0 - wet;
        for i in 0..left.len().min(right.len()) {
            let mono = (left[i] + right[i]) * 0.5;
            left[i] = (mono * gain_l) * wet + left[i] * dry;
            right[i] = (mono * gain_r) * wet + right[i] * dry;
        }
    }
    fn reset(&mut self) {}
    fn name(&self) -> &str {
        "Stereo Panner"
    }
    fn is_bypassed(&self) -> bool {
        self.bypassed
    }
    fn set_bypassed(&mut self, b: bool) {
        self.bypassed = b;
    }
    fn wet_dry(&self) -> f64 {
        self.wet_dry
    }
    fn set_wet_dry(&mut self, mix: f64) {
        self.wet_dry = mix.clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Simple compressor
// ---------------------------------------------------------------------------

/// Simple feed-forward compressor with attack/release envelope.
pub struct Compressor {
    pub threshold_db: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub makeup_gain_db: f64,
    envelope: f64,
    bypassed: bool,
    wet_dry: f64,
}

impl Compressor {
    pub fn new(threshold_db: f64, ratio: f64, attack_ms: f64, release_ms: f64) -> Self {
        Self {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_gain_db: 0.0,
            envelope: 0.0,
            bypassed: false,
            wet_dry: 1.0,
        }
    }

    fn compute_gain(&self, level_db: f64) -> f64 {
        if level_db > self.threshold_db {
            let over = level_db - self.threshold_db;
            let compressed = over / self.ratio;
            self.threshold_db + compressed - level_db + self.makeup_gain_db
        } else {
            self.makeup_gain_db
        }
    }
}

impl AudioEffect for Compressor {
    fn process(&mut self, left: &mut [f64], right: &mut [f64], sample_rate: f64) {
        if self.bypassed {
            return;
        }
        let attack_coeff = (-1.0 / (self.attack_ms * 0.001 * sample_rate)).exp();
        let release_coeff = (-1.0 / (self.release_ms * 0.001 * sample_rate)).exp();
        let wet = self.wet_dry;
        let dry = 1.0 - wet;

        for i in 0..left.len().min(right.len()) {
            let peak = left[i].abs().max(right[i].abs());
            let level_db = if peak > 1e-10 {
                20.0 * peak.log10()
            } else {
                -120.0
            };

            let target = self.compute_gain(level_db);
            let coeff = if target < self.envelope {
                attack_coeff
            } else {
                release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * target;

            let gain_linear = 10.0f64.powf(self.envelope / 20.0);
            left[i] = left[i] * gain_linear * wet + left[i] * dry;
            right[i] = right[i] * gain_linear * wet + right[i] * dry;
        }
    }
    fn reset(&mut self) {
        self.envelope = 0.0;
    }
    fn name(&self) -> &str {
        "Compressor"
    }
    fn is_bypassed(&self) -> bool {
        self.bypassed
    }
    fn set_bypassed(&mut self, b: bool) {
        self.bypassed = b;
    }
    fn wet_dry(&self) -> f64 {
        self.wet_dry
    }
    fn set_wet_dry(&mut self, mix: f64) {
        self.wet_dry = mix.clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Freeverb reverb
// ---------------------------------------------------------------------------

/// Allpass filter for the Freeverb algorithm.
struct AllpassDelay {
    buffer: Vec<f64>,
    index: usize,
    feedback: f64,
}

impl AllpassDelay {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            index: 0,
            feedback: 0.5,
        }
    }

    #[inline]
    fn process(&mut self, input: f64) -> f64 {
        let buffered = self.buffer[self.index];
        let output = -input + buffered;
        self.buffer[self.index] = input + buffered * self.feedback;
        self.index = (self.index + 1) % self.buffer.len();
        output
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
    }
}

/// Comb filter for the Freeverb algorithm.
struct CombFilter {
    buffer: Vec<f64>,
    index: usize,
    feedback: f64,
    damp1: f64,
    damp2: f64,
    filter_store: f64,
}

impl CombFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            index: 0,
            feedback: 0.5,
            damp1: 0.5,
            damp2: 0.5,
            filter_store: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, input: f64) -> f64 {
        let output = self.buffer[self.index];
        self.filter_store = output * self.damp2 + self.filter_store * self.damp1;
        self.buffer[self.index] = input + self.filter_store * self.feedback;
        self.index = (self.index + 1) % self.buffer.len();
        output
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
        self.filter_store = 0.0;
    }
}

/// Freeverb reverb algorithm implementation.
pub struct Reverb {
    combs_l: Vec<CombFilter>,
    combs_r: Vec<CombFilter>,
    allpasses_l: Vec<AllpassDelay>,
    allpasses_r: Vec<AllpassDelay>,
    pub room_size: f64,
    pub damping: f64,
    bypassed: bool,
    wet_dry: f64,
}

// Freeverb tuning constants (at 44100 Hz)
const COMB_TUNING: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASS_TUNING: [usize; 4] = [556, 441, 341, 225];
const STEREO_SPREAD: usize = 23;

impl Reverb {
    pub fn new(room_size: f64, damping: f64) -> Self {
        let mut rev = Self {
            combs_l: COMB_TUNING.iter().map(|&s| CombFilter::new(s)).collect(),
            combs_r: COMB_TUNING
                .iter()
                .map(|&s| CombFilter::new(s + STEREO_SPREAD))
                .collect(),
            allpasses_l: ALLPASS_TUNING
                .iter()
                .map(|&s| AllpassDelay::new(s))
                .collect(),
            allpasses_r: ALLPASS_TUNING
                .iter()
                .map(|&s| AllpassDelay::new(s + STEREO_SPREAD))
                .collect(),
            room_size,
            damping,
            bypassed: false,
            wet_dry: 0.3,
        };
        rev.update_parameters();
        rev
    }

    fn update_parameters(&mut self) {
        let feedback = self.room_size * 0.28 + 0.7;
        let damp = self.damping * 0.4;
        for comb in self.combs_l.iter_mut().chain(self.combs_r.iter_mut()) {
            comb.feedback = feedback;
            comb.damp1 = damp;
            comb.damp2 = 1.0 - damp;
        }
    }

    pub fn set_room_size(&mut self, size: f64) {
        self.room_size = size.clamp(0.0, 1.0);
        self.update_parameters();
    }

    pub fn set_damping(&mut self, damp: f64) {
        self.damping = damp.clamp(0.0, 1.0);
        self.update_parameters();
    }
}

impl AudioEffect for Reverb {
    fn process(&mut self, left: &mut [f64], right: &mut [f64], _sample_rate: f64) {
        if self.bypassed {
            return;
        }
        let wet = self.wet_dry;
        let dry = 1.0 - wet;

        for i in 0..left.len().min(right.len()) {
            let input = (left[i] + right[i]) * 0.5;
            let dry_l = left[i];
            let dry_r = right[i];

            let mut out_l = 0.0;
            let mut out_r = 0.0;

            for comb in &mut self.combs_l {
                out_l += comb.process(input);
            }
            for comb in &mut self.combs_r {
                out_r += comb.process(input);
            }

            for ap in &mut self.allpasses_l {
                out_l = ap.process(out_l);
            }
            for ap in &mut self.allpasses_r {
                out_r = ap.process(out_r);
            }

            left[i] = out_l * wet + dry_l * dry;
            right[i] = out_r * wet + dry_r * dry;
        }
    }
    fn reset(&mut self) {
        for c in &mut self.combs_l {
            c.reset();
        }
        for c in &mut self.combs_r {
            c.reset();
        }
        for a in &mut self.allpasses_l {
            a.reset();
        }
        for a in &mut self.allpasses_r {
            a.reset();
        }
    }
    fn name(&self) -> &str {
        "Reverb"
    }
    fn is_bypassed(&self) -> bool {
        self.bypassed
    }
    fn set_bypassed(&mut self, b: bool) {
        self.bypassed = b;
    }
    fn wet_dry(&self) -> f64 {
        self.wet_dry
    }
    fn set_wet_dry(&mut self, mix: f64) {
        self.wet_dry = mix.clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Audio effect chain
// ---------------------------------------------------------------------------

/// Serial chain of audio effects with insert/remove/reorder.
pub struct AudioEffectChain {
    effects: Vec<Box<dyn AudioEffect>>,
    bypassed: bool,
}

impl AudioEffectChain {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            bypassed: false,
        }
    }

    /// Add an effect to the end of the chain.
    pub fn push(&mut self, effect: Box<dyn AudioEffect>) {
        self.effects.push(effect);
    }

    /// Insert an effect at the given position.
    pub fn insert(&mut self, index: usize, effect: Box<dyn AudioEffect>) {
        let idx = index.min(self.effects.len());
        self.effects.insert(idx, effect);
    }

    /// Remove the effect at the given position.
    pub fn remove(&mut self, index: usize) -> Option<Box<dyn AudioEffect>> {
        if index < self.effects.len() {
            Some(self.effects.remove(index))
        } else {
            None
        }
    }

    /// Swap two effects in the chain.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a < self.effects.len() && b < self.effects.len() {
            self.effects.swap(a, b);
        }
    }

    /// Number of effects in the chain.
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Set global bypass.
    pub fn set_bypassed(&mut self, bypassed: bool) {
        self.bypassed = bypassed;
    }

    /// Process stereo buffers through the entire chain.
    pub fn process(&mut self, left: &mut [f64], right: &mut [f64], sample_rate: f64) {
        if self.bypassed {
            return;
        }
        for effect in &mut self.effects {
            effect.process(left, right, sample_rate);
        }
    }

    /// Reset all effects.
    pub fn reset(&mut self) {
        for effect in &mut self.effects {
            effect.reset();
        }
    }
}

impl Default for AudioEffectChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Effect matrix for routing
// ---------------------------------------------------------------------------

/// Routing point in the effect matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNode {
    Input(usize),
    Effect(usize),
    Output(usize),
}

/// A routing connection in the matrix.
#[derive(Debug, Clone)]
pub struct MatrixConnection {
    pub from: MatrixNode,
    pub to: MatrixNode,
    pub gain: f64,
}

/// Effect matrix for flexible routing between effects.
pub struct EffectMatrix {
    chains: Vec<AudioEffectChain>,
    connections: Vec<MatrixConnection>,
    num_inputs: usize,
    num_outputs: usize,
}

impl EffectMatrix {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            chains: Vec::new(),
            connections: Vec::new(),
            num_inputs,
            num_outputs,
        }
    }

    /// Add an effect chain to the matrix, returning its index.
    pub fn add_chain(&mut self, chain: AudioEffectChain) -> usize {
        let idx = self.chains.len();
        self.chains.push(chain);
        idx
    }

    /// Add a routing connection.
    pub fn connect(&mut self, from: MatrixNode, to: MatrixNode, gain: f64) {
        self.connections.push(MatrixConnection { from, to, gain });
    }

    /// Process all chains according to the routing. Buffers are indexed by
    /// channel pair (stereo).
    pub fn process(
        &mut self,
        inputs: &[(&[f64], &[f64])],
        outputs: &mut [(Vec<f64>, Vec<f64>)],
        sample_rate: f64,
    ) {
        if inputs.is_empty() || outputs.is_empty() {
            return;
        }
        let buf_len = inputs[0].0.len();

        // Intermediate buffers for each chain
        let mut chain_bufs: Vec<(Vec<f64>, Vec<f64>)> = self
            .chains
            .iter()
            .map(|_| (vec![0.0; buf_len], vec![0.0; buf_len]))
            .collect();

        // Route inputs to chains
        for conn in &self.connections {
            if let (MatrixNode::Input(src), MatrixNode::Effect(dst)) = (conn.from, conn.to) {
                if src < inputs.len() && dst < chain_bufs.len() {
                    for i in 0..buf_len {
                        chain_bufs[dst].0[i] += inputs[src].0[i] * conn.gain;
                        chain_bufs[dst].1[i] += inputs[src].1[i] * conn.gain;
                    }
                }
            }
        }

        // Process each chain
        for (idx, chain) in self.chains.iter_mut().enumerate() {
            chain.process(&mut chain_bufs[idx].0, &mut chain_bufs[idx].1, sample_rate);
        }

        // Route chains to outputs
        for out in outputs.iter_mut() {
            out.0.fill(0.0);
            out.1.fill(0.0);
        }
        for conn in &self.connections {
            if let (MatrixNode::Effect(src), MatrixNode::Output(dst)) = (conn.from, conn.to) {
                if src < chain_bufs.len() && dst < outputs.len() {
                    for i in 0..buf_len {
                        outputs[dst].0[i] += chain_bufs[src].0[i] * conn.gain;
                        outputs[dst].1[i] += chain_bufs[src].1[i] * conn.gain;
                    }
                }
            }
        }
    }

    pub fn num_chains(&self) -> usize {
        self.chains.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_lowpass() {
        let mut filter = BiquadFilter::new(BiquadType::Lowpass, 1000.0, 0.707, 0.0, 44100.0);
        let mut signal: Vec<f64> = (0..1000)
            .map(|i| {
                let t = i as f64 / 44100.0;
                (2.0 * PI * 440.0 * t).sin() + (2.0 * PI * 8000.0 * t).sin()
            })
            .collect();
        filter.process_buffer(&mut signal);
        // High frequency content should be attenuated
        let energy: f64 = signal.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_biquad_allpass_unity() {
        let mut filter = BiquadFilter::new(BiquadType::Allpass, 1000.0, 0.707, 0.0, 44100.0);
        let input: Vec<f64> = (0..1024)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let mut output = input.clone();
        filter.process_buffer(&mut output);
        // Allpass should preserve energy (after transient)
        let in_energy: f64 = input[100..].iter().map(|s| s * s).sum();
        let out_energy: f64 = output[100..].iter().map(|s| s * s).sum();
        assert!((in_energy - out_energy).abs() / in_energy < 0.05);
    }

    #[test]
    fn test_gain_transformer() {
        let mut gain = GainTransformer::new(6.0); // +6dB
        let mut left = vec![0.5; 100];
        let mut right = vec![0.5; 100];
        gain.process(&mut left, &mut right, 44100.0);
        // +6dB is roughly 2x gain
        assert!((left[0] - 0.5 * 10.0f64.powf(6.0 / 20.0)).abs() < 0.01);
    }

    #[test]
    fn test_gain_bypass() {
        let mut gain = GainTransformer::new(12.0);
        gain.set_bypassed(true);
        let mut left = vec![0.5; 100];
        let mut right = vec![0.5; 100];
        gain.process(&mut left, &mut right, 44100.0);
        assert!((left[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stereo_panner_center() {
        let mut panner = StereoPanner::new(0.0);
        let mut left = vec![1.0; 100];
        let mut right = vec![1.0; 100];
        panner.process(&mut left, &mut right, 44100.0);
        // Center pan: both channels should be equal
        assert!((left[0] - right[0]).abs() < 0.01);
    }

    #[test]
    fn test_stereo_panner_hard_left() {
        let mut panner = StereoPanner::new(-1.0);
        let mut left = vec![1.0; 100];
        let mut right = vec![1.0; 100];
        panner.process(&mut left, &mut right, 44100.0);
        assert!(left[0].abs() > right[0].abs());
    }

    #[test]
    fn test_compressor_below_threshold() {
        let mut comp = Compressor::new(-10.0, 4.0, 1.0, 50.0);
        let mut left = vec![0.01; 1000]; // Well below threshold
        let mut right = vec![0.01; 1000];
        comp.process(&mut left, &mut right, 44100.0);
        // Should be nearly unchanged (only makeup gain, which is 0)
        assert!((left[500] - 0.01).abs() < 0.005);
    }

    #[test]
    fn test_reverb_produces_tail() {
        let mut reverb = Reverb::new(0.8, 0.5);
        let mut left = vec![0.0; 4000];
        let mut right = vec![0.0; 4000];
        // Impulse
        left[0] = 1.0;
        right[0] = 1.0;
        reverb.process(&mut left, &mut right, 44100.0);
        // Should have a tail after the impulse
        let tail_energy: f64 = left[100..].iter().map(|s| s * s).sum();
        assert!(tail_energy > 0.001, "Reverb should produce a tail");
    }

    #[test]
    fn test_effect_chain() {
        let mut chain = AudioEffectChain::new();
        chain.push(Box::new(GainTransformer::new(0.0)));
        chain.push(Box::new(StereoPanner::new(0.0)));
        assert_eq!(chain.len(), 2);

        let mut left = vec![0.5; 100];
        let mut right = vec![0.5; 100];
        chain.process(&mut left, &mut right, 44100.0);
        // With 0dB gain and center pan, output should be close to input
        assert!((left[50] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_effect_chain_insert_remove() {
        let mut chain = AudioEffectChain::new();
        chain.push(Box::new(GainTransformer::new(0.0)));
        chain.push(Box::new(GainTransformer::new(0.0)));
        assert_eq!(chain.len(), 2);

        chain.insert(1, Box::new(StereoPanner::new(0.0)));
        assert_eq!(chain.len(), 3);

        chain.remove(0);
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_parametric_eq() {
        let bands = vec![
            EqBand {
                filter_type: BiquadType::LowShelf,
                frequency: 100.0,
                q: 0.707,
                gain_db: 6.0,
                enabled: true,
            },
            EqBand {
                filter_type: BiquadType::Peaking,
                frequency: 1000.0,
                q: 1.0,
                gain_db: -3.0,
                enabled: true,
            },
            EqBand {
                filter_type: BiquadType::HighShelf,
                frequency: 8000.0,
                q: 0.707,
                gain_db: 3.0,
                enabled: true,
            },
        ];
        let mut eq = ParametricEq::new(bands, 44100.0);
        let mut left: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin())
            .collect();
        let mut right = left.clone();
        eq.process(&mut left, &mut right, 44100.0);
        let energy: f64 = left.iter().map(|s| s * s).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_effect_matrix_passthrough() {
        let mut matrix = EffectMatrix::new(1, 1);
        let chain = AudioEffectChain::new(); // Empty chain = passthrough
        let chain_idx = matrix.add_chain(chain);
        matrix.connect(MatrixNode::Input(0), MatrixNode::Effect(chain_idx), 1.0);
        matrix.connect(MatrixNode::Effect(chain_idx), MatrixNode::Output(0), 1.0);

        let input_l = vec![0.5; 100];
        let input_r = vec![0.3; 100];
        let mut outputs = vec![(vec![0.0; 100], vec![0.0; 100])];
        matrix.process(&[(&input_l, &input_r)], &mut outputs, 44100.0);
        assert!((outputs[0].0[50] - 0.5).abs() < 1e-10);
        assert!((outputs[0].1[50] - 0.3).abs() < 1e-10);
    }
}
