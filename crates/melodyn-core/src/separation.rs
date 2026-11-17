//! Source separation algorithms for single-channel audio.
//!
//! Implements three blind source separation approaches:
//! - **HPSS**: Harmonic/Percussive Separation via median filtering
//! - **NMF**: Non-negative Matrix Factorization with multiplicative updates
//! - **Iterative Spectral Subtraction**: Pitch-guided harmonic removal
//!
//! These correspond to the separation entries in
//! `MUAudioSourceDescriptionSeparationEntry` and `MURawNoteSeparation`.

use crate::fft::{compute_window, FftEngine, WindowType};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// HPSS: Harmonic/Percussive Source Separation
// ---------------------------------------------------------------------------

/// Configuration for Harmonic/Percussive Source Separation.
#[derive(Debug, Clone)]
pub struct HpssConfig {
    /// FFT size for the STFT.
    pub fft_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Window type for STFT analysis/synthesis.
    pub window_type: WindowType,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Kernel size for horizontal (time) median filter (must be odd).
    /// Larger values capture longer harmonic structures.
    pub harmonic_kernel: usize,
    /// Kernel size for vertical (frequency) median filter (must be odd).
    /// Larger values capture broader percussive transients.
    pub percussive_kernel: usize,
    /// Power exponent for Wiener mask computation (typically 1 or 2).
    pub mask_power: f64,
}

impl Default for HpssConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            window_type: WindowType::Hann,
            sample_rate: 44100.0,
            harmonic_kernel: 31,
            percussive_kernel: 31,
            mask_power: 2.0,
        }
    }
}

/// Result of harmonic/percussive separation.
#[derive(Debug, Clone)]
pub struct HpssResult {
    /// The harmonic (tonal) component signal.
    pub harmonic: Vec<f64>,
    /// The percussive (transient) component signal.
    pub percussive: Vec<f64>,
    /// Residual signal (original - harmonic - percussive).
    pub residual: Vec<f64>,
}

/// Perform Harmonic/Percussive Source Separation via median filtering.
///
/// The algorithm:
/// 1. Compute STFT magnitude spectrogram
/// 2. Apply horizontal median filter -> harmonic-enhanced spectrogram
/// 3. Apply vertical median filter -> percussive-enhanced spectrogram
/// 4. Compute soft Wiener masks from the enhanced spectrograms
/// 5. Apply masks to complex STFT and reconstruct via inverse STFT
pub fn hpss(audio: &[f64], config: &HpssConfig) -> HpssResult {
    let n = config.fft_size;
    let hop = config.hop_size;
    let num_bins = n / 2 + 1;

    // Compute STFT
    let (stft, num_frames) = compute_stft(audio, n, hop, config.window_type);

    // Build magnitude spectrogram: [frame][bin]
    let mut mag_spec: Vec<Vec<f64>> = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let offset = frame_idx * n;
        let frame_mag: Vec<f64> = (0..num_bins)
            .map(|k| stft[offset + k].norm())
            .collect();
        mag_spec.push(frame_mag);
    }

    // Horizontal median filtering (along time axis) -> harmonic enhanced
    let harmonic_enhanced = median_filter_horizontal(&mag_spec, config.harmonic_kernel);

    // Vertical median filtering (along frequency axis) -> percussive enhanced
    let percussive_enhanced = median_filter_vertical(&mag_spec, config.percussive_kernel);

    // Compute Wiener masks
    let p = config.mask_power;
    let mut harmonic_stft = vec![Complex::new(0.0, 0.0); num_frames * n];
    let mut percussive_stft = vec![Complex::new(0.0, 0.0); num_frames * n];

    for t in 0..num_frames {
        for k in 0..num_bins {
            let h_val = harmonic_enhanced[t][k].powf(p);
            let p_val = percussive_enhanced[t][k].powf(p);
            let denom = h_val + p_val + 1e-30;

            let h_mask = h_val / denom;
            let p_mask = p_val / denom;

            let orig = stft[t * n + k];
            harmonic_stft[t * n + k] = orig * h_mask;
            percussive_stft[t * n + k] = orig * p_mask;

            // Fill conjugate symmetric part
            if k > 0 && k < num_bins - 1 {
                harmonic_stft[t * n + (n - k)] = harmonic_stft[t * n + k].conj();
                percussive_stft[t * n + (n - k)] = percussive_stft[t * n + k].conj();
            }
        }
    }

    // Inverse STFT with overlap-add
    let harmonic = inverse_stft(&harmonic_stft, n, hop, num_frames, audio.len(), config.window_type);
    let percussive = inverse_stft(&percussive_stft, n, hop, num_frames, audio.len(), config.window_type);

    // Residual
    let residual: Vec<f64> = audio
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let h = if i < harmonic.len() { harmonic[i] } else { 0.0 };
            let p = if i < percussive.len() { percussive[i] } else { 0.0 };
            s - h - p
        })
        .collect();

    HpssResult {
        harmonic,
        percussive,
        residual,
    }
}

// ---------------------------------------------------------------------------
// NMF: Non-negative Matrix Factorization
// ---------------------------------------------------------------------------

/// Cost function for NMF optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmfCostFunction {
    /// Euclidean distance: ||V - WH||^2
    Euclidean,
    /// Generalized KL divergence: D_KL(V || WH)
    KlDivergence,
}

/// Initialization strategy for NMF bases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmfInitialization {
    /// Deterministic seeded initialization.
    Seeded,
    /// Harmonic comb initialization (each basis is a harmonic series).
    HarmonicComb,
}

/// Configuration for NMF source separation.
#[derive(Debug, Clone)]
pub struct NmfConfig {
    /// FFT size for the STFT.
    pub fft_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Window type.
    pub window_type: WindowType,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of NMF components (sources).
    pub num_components: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold (relative cost decrease).
    pub convergence_threshold: f64,
    /// Cost function to minimize.
    pub cost_function: NmfCostFunction,
    /// Initialization strategy.
    pub initialization: NmfInitialization,
    /// Base frequencies for harmonic comb initialization (Hz).
    /// Only used when initialization == HarmonicComb.
    pub base_frequencies: Vec<f64>,
}

impl Default for NmfConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            window_type: WindowType::Hann,
            sample_rate: 44100.0,
            num_components: 2,
            max_iterations: 200,
            convergence_threshold: 1e-6,
            cost_function: NmfCostFunction::Euclidean,
            initialization: NmfInitialization::Seeded,
            base_frequencies: vec![],
        }
    }
}

/// Result of NMF decomposition.
#[derive(Debug, Clone)]
pub struct NmfResult {
    /// Basis spectra W: [num_bins x num_components].
    pub bases: Vec<Vec<f64>>,
    /// Activation matrix H: [num_components x num_frames].
    pub activations: Vec<Vec<f64>>,
    /// Per-component reconstructed audio signals.
    pub components: Vec<Vec<f64>>,
    /// Cost function value at each iteration.
    pub cost_history: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Perform NMF-based source separation.
///
/// Decomposes the magnitude spectrogram V into V ~ W * H where:
/// - W (num_bins x K): basis spectra (spectral templates)
/// - H (K x num_frames): activation coefficients
///
/// Per-source audio is reconstructed using Wiener filtering on the complex STFT.
pub fn nmf_separate(audio: &[f64], config: &NmfConfig) -> NmfResult {
    let n = config.fft_size;
    let hop = config.hop_size;
    let num_bins = n / 2 + 1;
    let k = config.num_components;

    // Compute STFT
    let (stft, num_frames) = compute_stft(audio, n, hop, config.window_type);

    // Build magnitude spectrogram V: [num_bins][num_frames]
    let mut v_mat: Vec<Vec<f64>> = vec![vec![0.0; num_frames]; num_bins];
    for t in 0..num_frames {
        for b in 0..num_bins {
            v_mat[b][t] = stft[t * n + b].norm().max(1e-30);
        }
    }

    // Initialize W and H
    let (mut w_mat, mut h_mat) = initialize_nmf(
        num_bins,
        num_frames,
        k,
        config.initialization,
        &config.base_frequencies,
        config.sample_rate,
        n,
    );

    // Multiplicative update iterations
    let mut cost_history = Vec::with_capacity(config.max_iterations);

    for _iter in 0..config.max_iterations {
        // Compute approximation WH
        let wh = mat_mul_wh(&w_mat, &h_mat, num_bins, num_frames, k);

        // Compute cost
        let cost = compute_nmf_cost(&v_mat, &wh, num_bins, num_frames, config.cost_function);
        cost_history.push(cost);

        // Check convergence
        if cost_history.len() >= 2 {
            let prev = cost_history[cost_history.len() - 2];
            if prev > 0.0 && (prev - cost).abs() / prev < config.convergence_threshold {
                break;
            }
        }

        // Update rules
        match config.cost_function {
            NmfCostFunction::Euclidean => {
                // Update H: H <- H .* (W^T V) ./ (W^T W H + eps)
                update_h_euclidean(&mut h_mat, &w_mat, &v_mat, &wh, num_bins, num_frames, k);
                // Recompute WH after H update
                let wh2 = mat_mul_wh(&w_mat, &h_mat, num_bins, num_frames, k);
                // Update W: W <- W .* (V H^T) ./ (W H H^T + eps)
                update_w_euclidean(&mut w_mat, &h_mat, &v_mat, &wh2, num_bins, num_frames, k);
            }
            NmfCostFunction::KlDivergence => {
                // Update H: H <- H .* (W^T (V ./ WH)) ./ sum_f(W)
                update_h_kl(&mut h_mat, &w_mat, &v_mat, &wh, num_bins, num_frames, k);
                let wh2 = mat_mul_wh(&w_mat, &h_mat, num_bins, num_frames, k);
                // Update W: W <- W .* ((V ./ WH) H^T) ./ sum_t(H)
                update_w_kl(&mut w_mat, &h_mat, &v_mat, &wh2, num_bins, num_frames, k);
            }
        }
    }

    let iterations = cost_history.len();

    // Reconstruct per-component audio using Wiener filtering
    let wh_final = mat_mul_wh(&w_mat, &h_mat, num_bins, num_frames, k);
    let mut components = Vec::with_capacity(k);

    for c in 0..k {
        // Component spectrogram: W[:,c] * H[c,:]
        let mut comp_stft = vec![Complex::new(0.0, 0.0); num_frames * n];
        for t in 0..num_frames {
            for b in 0..num_bins {
                let comp_mag = w_mat[b][c] * h_mat[c][t];
                let denom = wh_final[b][t] + 1e-30;
                let mask = comp_mag / denom;
                let orig = stft[t * n + b];
                comp_stft[t * n + b] = orig * mask;
                if b > 0 && b < num_bins - 1 {
                    comp_stft[t * n + (n - b)] = comp_stft[t * n + b].conj();
                }
            }
        }
        let signal = inverse_stft(&comp_stft, n, hop, num_frames, audio.len(), config.window_type);
        components.push(signal);
    }

    // Format bases and activations for output
    let bases = (0..k)
        .map(|c| (0..num_bins).map(|b| w_mat[b][c]).collect())
        .collect();
    let activations = h_mat.clone();

    NmfResult {
        bases,
        activations,
        components,
        cost_history,
        iterations,
    }
}

// ---------------------------------------------------------------------------
// Iterative Spectral Subtraction
// ---------------------------------------------------------------------------

/// Configuration for iterative spectral subtraction.
#[derive(Debug, Clone)]
pub struct SpectralSubtractionConfig {
    /// FFT size for the STFT.
    pub fft_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Window type.
    pub window_type: WindowType,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Maximum number of sources to extract.
    pub max_sources: usize,
    /// Number of harmonics to subtract per source.
    pub num_harmonics: usize,
    /// Minimum frequency for pitch detection (Hz).
    pub min_freq: f64,
    /// Maximum frequency for pitch detection (Hz).
    pub max_freq: f64,
    /// Attenuation factor for subtracted harmonics (0.0 to 1.0).
    pub attenuation: f64,
    /// Bandwidth of each harmonic in bins (for subtraction).
    pub harmonic_bandwidth: usize,
}

impl Default for SpectralSubtractionConfig {
    fn default() -> Self {
        Self {
            fft_size: 4096,
            hop_size: 1024,
            window_type: WindowType::Hann,
            sample_rate: 44100.0,
            max_sources: 3,
            num_harmonics: 12,
            min_freq: 80.0,
            max_freq: 4000.0,
            attenuation: 0.9,
            harmonic_bandwidth: 3,
        }
    }
}

/// Result of iterative spectral subtraction.
#[derive(Debug, Clone)]
pub struct SpectralSubtractionResult {
    /// Extracted source signals, one per detected dominant pitch.
    pub sources: Vec<ExtractedSource>,
    /// Residual signal after all extractions.
    pub residual: Vec<f64>,
}

/// A single extracted source from spectral subtraction.
#[derive(Debug, Clone)]
pub struct ExtractedSource {
    /// Estimated fundamental frequency (Hz).
    pub fundamental: f64,
    /// Reconstructed audio signal for this source.
    pub signal: Vec<f64>,
}

/// Perform iterative spectral subtraction.
///
/// Algorithm:
/// 1. Compute STFT of the input
/// 2. Find dominant pitch in the magnitude spectrum (via HPS)
/// 3. Build a harmonic mask for that pitch
/// 4. Extract the masked signal, subtract from residual
/// 5. Repeat for next source
pub fn spectral_subtraction(audio: &[f64], config: &SpectralSubtractionConfig) -> SpectralSubtractionResult {
    let n = config.fft_size;
    let hop = config.hop_size;
    let num_bins = n / 2 + 1;

    let (stft, num_frames) = compute_stft(audio, n, hop, config.window_type);

    // Work with a mutable copy of the magnitude spectrogram
    let mut residual_mag: Vec<Vec<f64>> = Vec::with_capacity(num_frames);
    for t in 0..num_frames {
        let frame_mag: Vec<f64> = (0..num_bins)
            .map(|k| stft[t * n + k].norm())
            .collect();
        residual_mag.push(frame_mag);
    }

    let mut sources = Vec::new();

    for _ in 0..config.max_sources {
        // Estimate dominant pitch from average magnitude spectrum
        let avg_mag = average_spectrum(&residual_mag, num_bins);
        let fundamental = detect_dominant_pitch_hps(
            &avg_mag,
            config.sample_rate,
            n,
            config.min_freq,
            config.max_freq,
        );

        if fundamental.is_none() {
            break;
        }
        let f0 = fundamental.unwrap();

        // Build harmonic mask for this pitch across all frames
        let mut source_stft = vec![Complex::new(0.0, 0.0); num_frames * n];

        for t in 0..num_frames {
            for h in 1..=config.num_harmonics {
                let harm_freq = f0 * h as f64;
                if harm_freq >= config.sample_rate / 2.0 {
                    break;
                }
                let center_bin = (harm_freq * n as f64 / config.sample_rate).round() as usize;
                if center_bin >= num_bins {
                    break;
                }

                let bw = config.harmonic_bandwidth;
                let start = center_bin.saturating_sub(bw);
                let end = (center_bin + bw + 1).min(num_bins);

                for k in start..end {
                    let dist = (k as f64 - center_bin as f64).abs();
                    let weight = (-dist * dist / (bw as f64 * 0.5)).exp();
                    let extract_amount = weight * config.attenuation;

                    source_stft[t * n + k] = stft[t * n + k] * extract_amount;
                    if k > 0 && k < num_bins - 1 {
                        source_stft[t * n + (n - k)] = source_stft[t * n + k].conj();
                    }

                    // Subtract from residual magnitude
                    residual_mag[t][k] *= (1.0 - extract_amount).max(0.0);
                }
            }
        }

        let signal = inverse_stft(&source_stft, n, hop, num_frames, audio.len(), config.window_type);
        sources.push(ExtractedSource {
            fundamental: f0,
            signal,
        });
    }

    // Reconstruct residual from remaining magnitude + original phase
    let mut residual_stft = vec![Complex::new(0.0, 0.0); num_frames * n];
    for t in 0..num_frames {
        for k in 0..num_bins {
            let orig = stft[t * n + k];
            let orig_mag = orig.norm();
            let phase = if orig_mag > 1e-30 { orig / orig_mag } else { Complex::new(1.0, 0.0) };
            residual_stft[t * n + k] = phase * residual_mag[t][k];
            if k > 0 && k < num_bins - 1 {
                residual_stft[t * n + (n - k)] = residual_stft[t * n + k].conj();
            }
        }
    }
    let residual = inverse_stft(&residual_stft, n, hop, num_frames, audio.len(), config.window_type);

    SpectralSubtractionResult { sources, residual }
}

// ---------------------------------------------------------------------------
// STFT / inverse STFT utilities
// ---------------------------------------------------------------------------

/// Compute the Short-Time Fourier Transform of an audio signal.
/// Returns (flat complex spectrum array, number of frames).
/// Layout: frame t, bin k -> index [t * fft_size + k].
pub fn compute_stft(
    audio: &[f64],
    fft_size: usize,
    hop_size: usize,
    window_type: WindowType,
) -> (Vec<Complex<f64>>, usize) {
    let engine = FftEngine::new(fft_size, window_type);
    let half = fft_size / 2;

    // Count frames
    let mut num_frames = 0;
    let mut pos = 0usize;
    while pos + fft_size <= audio.len() + half {
        num_frames += 1;
        pos += hop_size;
    }
    if num_frames == 0 {
        return (vec![], 0);
    }

    let mut stft = vec![Complex::new(0.0, 0.0); num_frames * fft_size];
    let mut pos = 0usize;
    for t in 0..num_frames {
        // Extract frame with zero-padding at edges
        let mut frame = vec![0.0f64; fft_size];
        for i in 0..fft_size {
            let idx = pos + i;
            if idx < audio.len() {
                frame[i] = audio[idx];
            }
        }
        let spectrum = engine.forward(&frame);
        let offset = t * fft_size;
        stft[offset..offset + fft_size].copy_from_slice(&spectrum);
        pos += hop_size;
    }

    (stft, num_frames)
}

/// Inverse STFT with overlap-add synthesis.
pub fn inverse_stft(
    stft: &[Complex<f64>],
    fft_size: usize,
    hop_size: usize,
    num_frames: usize,
    output_len: usize,
    window_type: WindowType,
) -> Vec<f64> {
    let engine = FftEngine::new(fft_size, window_type);
    let window = compute_window(window_type, fft_size);

    let total_len = (num_frames - 1) * hop_size + fft_size;
    let out_len = total_len.max(output_len);
    let mut output = vec![0.0f64; out_len];
    let mut window_sum = vec![0.0f64; out_len];

    for t in 0..num_frames {
        let offset = t * fft_size;
        let frame_spectrum = &stft[offset..offset + fft_size];
        let frame_signal = engine.inverse_real(frame_spectrum);

        let pos = t * hop_size;
        for i in 0..fft_size {
            if pos + i < out_len {
                output[pos + i] += frame_signal[i] * window[i];
                window_sum[pos + i] += window[i] * window[i];
            }
        }
    }

    // Normalize by window sum (overlap-add normalization)
    for i in 0..output.len().min(output_len) {
        if window_sum[i] > 1e-10 {
            output[i] /= window_sum[i];
        }
    }

    output.truncate(output_len);
    output
}

// ---------------------------------------------------------------------------
// Median filtering helpers
// ---------------------------------------------------------------------------

/// Horizontal median filter (across time dimension) on the spectrogram.
fn median_filter_horizontal(spec: &[Vec<f64>], kernel: usize) -> Vec<Vec<f64>> {
    let num_frames = spec.len();
    if num_frames == 0 {
        return vec![];
    }
    let num_bins = spec[0].len();
    let half_k = kernel / 2;

    let mut result = vec![vec![0.0; num_bins]; num_frames];

    for b in 0..num_bins {
        for t in 0..num_frames {
            let start = t.saturating_sub(half_k);
            let end = (t + half_k + 1).min(num_frames);
            let mut vals: Vec<f64> = (start..end).map(|tt| spec[tt][b]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            result[t][b] = vals[vals.len() / 2];
        }
    }
    result
}

/// Vertical median filter (across frequency dimension) on the spectrogram.
fn median_filter_vertical(spec: &[Vec<f64>], kernel: usize) -> Vec<Vec<f64>> {
    let num_frames = spec.len();
    if num_frames == 0 {
        return vec![];
    }
    let num_bins = spec[0].len();
    let half_k = kernel / 2;

    let mut result = vec![vec![0.0; num_bins]; num_frames];

    for t in 0..num_frames {
        for b in 0..num_bins {
            let start = b.saturating_sub(half_k);
            let end = (b + half_k + 1).min(num_bins);
            let mut vals: Vec<f64> = (start..end).map(|bb| spec[t][bb]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            result[t][b] = vals[vals.len() / 2];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// NMF helpers
// ---------------------------------------------------------------------------

/// Initialize NMF matrices W and H.
/// W: [num_bins][k], H: [k][num_frames]
fn initialize_nmf(
    num_bins: usize,
    num_frames: usize,
    k: usize,
    init: NmfInitialization,
    base_frequencies: &[f64],
    sample_rate: f64,
    fft_size: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut w = vec![vec![0.0f64; k]; num_bins];
    let mut h = vec![vec![0.0f64; num_frames]; k];

    match init {
        NmfInitialization::Seeded => {
            // Deterministic pseudo-random initialization using a simple LCG
            let mut seed: u64 = 42;
            for b in 0..num_bins {
                for c in 0..k {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    w[b][c] = ((seed >> 33) as f64 / u32::MAX as f64) * 0.1 + 0.01;
                }
            }
            for c in 0..k {
                for t in 0..num_frames {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    h[c][t] = ((seed >> 33) as f64 / u32::MAX as f64) * 0.1 + 0.01;
                }
            }
        }
        NmfInitialization::HarmonicComb => {
            let freq_res = sample_rate / fft_size as f64;
            for c in 0..k {
                let f0 = if c < base_frequencies.len() {
                    base_frequencies[c]
                } else {
                    // Default: spread fundamentals across the range
                    100.0 * (c + 1) as f64
                };

                // Create harmonic comb for this component
                for h_num in 1..=20 {
                    let harm_freq = f0 * h_num as f64;
                    if harm_freq >= sample_rate / 2.0 {
                        break;
                    }
                    let bin = (harm_freq / freq_res).round() as usize;
                    if bin < num_bins {
                        let weight = 1.0 / h_num as f64;
                        // Gaussian spread around the harmonic bin
                        for b in bin.saturating_sub(3)..(bin + 4).min(num_bins) {
                            let dist = (b as f64 - bin as f64).abs();
                            w[b][c] += weight * (-dist * dist / 2.0).exp();
                        }
                    }
                }

                // Initialize H uniformly
                for t in 0..num_frames {
                    h[c][t] = 0.1;
                }
            }

            // Ensure no zeros in W
            for b in 0..num_bins {
                for c in 0..k {
                    w[b][c] = w[b][c].max(1e-6);
                }
            }
        }
    }

    (w, h)
}

/// Compute WH matrix product: result[b][t] = sum_c W[b][c] * H[c][t]
fn mat_mul_wh(
    w: &[Vec<f64>],
    h: &[Vec<f64>],
    num_bins: usize,
    num_frames: usize,
    k: usize,
) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0f64; num_frames]; num_bins];
    for b in 0..num_bins {
        for c in 0..k {
            let w_bc = w[b][c];
            if w_bc > 1e-30 {
                for t in 0..num_frames {
                    result[b][t] += w_bc * h[c][t];
                }
            }
        }
    }
    result
}

/// Compute NMF cost function value.
fn compute_nmf_cost(
    v: &[Vec<f64>],
    wh: &[Vec<f64>],
    num_bins: usize,
    num_frames: usize,
    cost_fn: NmfCostFunction,
) -> f64 {
    let mut cost = 0.0;
    match cost_fn {
        NmfCostFunction::Euclidean => {
            for b in 0..num_bins {
                for t in 0..num_frames {
                    let diff = v[b][t] - wh[b][t];
                    cost += diff * diff;
                }
            }
            cost * 0.5
        }
        NmfCostFunction::KlDivergence => {
            for b in 0..num_bins {
                for t in 0..num_frames {
                    let v_bt = v[b][t].max(1e-30);
                    let wh_bt = wh[b][t].max(1e-30);
                    cost += v_bt * (v_bt / wh_bt).ln() - v_bt + wh_bt;
                }
            }
            cost
        }
    }
}

/// Euclidean update for H: H <- H .* (W^T V) ./ (W^T WH + eps)
fn update_h_euclidean(
    h: &mut [Vec<f64>],
    w: &[Vec<f64>],
    v: &[Vec<f64>],
    wh: &[Vec<f64>],
    num_bins: usize,
    num_frames: usize,
    k: usize,
) {
    for c in 0..k {
        for t in 0..num_frames {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for b in 0..num_bins {
                numerator += w[b][c] * v[b][t];
                denominator += w[b][c] * wh[b][t];
            }
            h[c][t] *= numerator / (denominator + 1e-30);
            h[c][t] = h[c][t].max(1e-30);
        }
    }
}

/// Euclidean update for W: W <- W .* (V H^T) ./ (WH H^T + eps)
fn update_w_euclidean(
    w: &mut [Vec<f64>],
    h: &[Vec<f64>],
    v: &[Vec<f64>],
    wh: &[Vec<f64>],
    num_bins: usize,
    num_frames: usize,
    k: usize,
) {
    for b in 0..num_bins {
        for c in 0..k {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for t in 0..num_frames {
                numerator += v[b][t] * h[c][t];
                denominator += wh[b][t] * h[c][t];
            }
            w[b][c] *= numerator / (denominator + 1e-30);
            w[b][c] = w[b][c].max(1e-30);
        }
    }
}

/// KL-divergence update for H: H <- H .* (W^T (V ./ WH)) ./ sum_b(W[:,c])
fn update_h_kl(
    h: &mut [Vec<f64>],
    w: &[Vec<f64>],
    v: &[Vec<f64>],
    wh: &[Vec<f64>],
    num_bins: usize,
    num_frames: usize,
    k: usize,
) {
    // Precompute column sums of W
    let w_col_sums: Vec<f64> = (0..k)
        .map(|c| (0..num_bins).map(|b| w[b][c]).sum::<f64>().max(1e-30))
        .collect();

    for c in 0..k {
        for t in 0..num_frames {
            let mut numerator = 0.0;
            for b in 0..num_bins {
                let ratio = v[b][t] / (wh[b][t] + 1e-30);
                numerator += w[b][c] * ratio;
            }
            h[c][t] *= numerator / w_col_sums[c];
            h[c][t] = h[c][t].max(1e-30);
        }
    }
}

/// KL-divergence update for W: W <- W .* ((V ./ WH) H^T) ./ sum_t(H[c,:])
fn update_w_kl(
    w: &mut [Vec<f64>],
    h: &[Vec<f64>],
    v: &[Vec<f64>],
    wh: &[Vec<f64>],
    num_bins: usize,
    num_frames: usize,
    k: usize,
) {
    // Precompute row sums of H
    let h_row_sums: Vec<f64> = (0..k)
        .map(|c| (0..num_frames).map(|t| h[c][t]).sum::<f64>().max(1e-30))
        .collect();

    for b in 0..num_bins {
        for c in 0..k {
            let mut numerator = 0.0;
            for t in 0..num_frames {
                let ratio = v[b][t] / (wh[b][t] + 1e-30);
                numerator += ratio * h[c][t];
            }
            w[b][c] *= numerator / h_row_sums[c];
            w[b][c] = w[b][c].max(1e-30);
        }
    }
}

// ---------------------------------------------------------------------------
// Pitch detection helper for spectral subtraction
// ---------------------------------------------------------------------------

/// Detect dominant pitch by finding the highest spectral peak in the magnitude array.
///
/// Uses simple peak-picking with parabolic interpolation. This is more robust
/// than HPS for signals that may be pure tones without harmonics (e.g., after
/// partial spectral subtraction).
fn detect_dominant_pitch_hps(
    magnitude: &[f64],
    sample_rate: f64,
    fft_size: usize,
    min_freq: f64,
    max_freq: f64,
) -> Option<f64> {
    let num_bins = magnitude.len();
    let min_bin = (min_freq * fft_size as f64 / sample_rate).ceil() as usize;
    let max_bin = ((max_freq * fft_size as f64 / sample_rate).floor() as usize).min(num_bins - 1);

    if min_bin >= max_bin || min_bin + 1 >= num_bins {
        return None;
    }

    // Find the highest local maximum (peak) in the valid frequency range
    let mut best_bin = min_bin;
    let mut best_val = magnitude[min_bin];

    for k in (min_bin + 1)..max_bin {
        if magnitude[k] > best_val
            && magnitude[k] >= magnitude[k - 1]
            && magnitude[k] >= magnitude[k + 1]
        {
            best_val = magnitude[k];
            best_bin = k;
        }
    }

    // Also check the edges of the range
    if magnitude[min_bin] > best_val {
        best_val = magnitude[min_bin];
        best_bin = min_bin;
    }
    if max_bin < num_bins && magnitude[max_bin] > best_val {
        best_val = magnitude[max_bin];
        best_bin = max_bin;
    }

    if best_val < 1e-20 {
        return None;
    }

    // Parabolic interpolation for sub-bin accuracy
    let true_bin = if best_bin > 0 && best_bin + 1 < num_bins {
        let alpha = magnitude[best_bin - 1].max(1e-30).ln();
        let beta = magnitude[best_bin].max(1e-30).ln();
        let gamma = magnitude[best_bin + 1].max(1e-30).ln();
        let denom = alpha - 2.0 * beta + gamma;
        if denom.abs() > 1e-10 {
            best_bin as f64 + 0.5 * (alpha - gamma) / denom
        } else {
            best_bin as f64
        }
    } else {
        best_bin as f64
    };

    Some(true_bin * sample_rate / fft_size as f64)
}

/// Compute average magnitude spectrum across all frames.
fn average_spectrum(spec: &[Vec<f64>], num_bins: usize) -> Vec<f64> {
    let num_frames = spec.len();
    if num_frames == 0 {
        return vec![0.0; num_bins];
    }
    let mut avg = vec![0.0f64; num_bins];
    for frame in spec {
        for (k, &m) in frame.iter().enumerate() {
            if k < num_bins {
                avg[k] += m;
            }
        }
    }
    let scale = 1.0 / num_frames as f64;
    for v in &mut avg {
        *v *= scale;
    }
    avg
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_wave(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    fn click_signal(position: usize, length: usize) -> Vec<f64> {
        let mut signal = vec![0.0; length];
        if position < length {
            signal[position] = 1.0;
            if position + 1 < length {
                signal[position + 1] = -0.5;
            }
        }
        signal
    }

    #[test]
    fn test_hpss_separates_tone_and_click() {
        let sr = 44100.0;
        let n = 44100; // 1 second

        // Create a signal with a sustained harmonic tone + a click
        let tone = sine_wave(440.0, sr, n);
        let click_pos = n / 2;
        let click = click_signal(click_pos, n);

        let mixed: Vec<f64> = tone.iter().zip(click.iter()).map(|(&t, &c)| t + c).collect();

        let config = HpssConfig {
            fft_size: 2048,
            hop_size: 512,
            sample_rate: sr,
            harmonic_kernel: 17,
            percussive_kernel: 17,
            mask_power: 2.0,
            ..Default::default()
        };

        let result = hpss(&mixed, &config);

        // The harmonic component should have more energy in the tone region
        let tone_energy: f64 = result.harmonic.iter().map(|s| s * s).sum();
        let perc_energy: f64 = result.percussive.iter().map(|s| s * s).sum();

        assert!(tone_energy > 0.0, "Harmonic component should have energy");
        assert!(perc_energy > 0.0, "Percussive component should have energy");

        // The harmonic component should capture most of the sustained tone energy
        let orig_tone_energy: f64 = tone.iter().map(|s| s * s).sum();
        assert!(
            tone_energy > orig_tone_energy * 0.1,
            "Harmonic should capture significant tone energy: {:.4} vs {:.4}",
            tone_energy,
            orig_tone_energy
        );
    }

    #[test]
    fn test_nmf_cost_decreases() {
        let sr = 44100.0;
        let n = 22050; // 0.5 seconds

        // Two-tone signal
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                0.7 * (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 660.0 * t).sin()
            })
            .collect();

        let config = NmfConfig {
            fft_size: 1024,
            hop_size: 256,
            sample_rate: sr,
            num_components: 2,
            max_iterations: 50,
            convergence_threshold: 1e-8,
            cost_function: NmfCostFunction::Euclidean,
            initialization: NmfInitialization::Seeded,
            ..Default::default()
        };

        let result = nmf_separate(&signal, &config);

        // Cost should decrease monotonically (or nearly so)
        assert!(
            result.cost_history.len() > 2,
            "Should run more than 2 iterations"
        );

        let first_cost = result.cost_history[0];
        let last_cost = result.cost_history[result.cost_history.len() - 1];
        assert!(
            last_cost < first_cost,
            "Cost should decrease: first={}, last={}",
            first_cost,
            last_cost
        );

        // Check that cost generally decreases (allow small numerical increases)
        let mut decreasing_count = 0;
        for i in 1..result.cost_history.len() {
            if result.cost_history[i] <= result.cost_history[i - 1] + 1e-6 {
                decreasing_count += 1;
            }
        }
        let ratio = decreasing_count as f64 / (result.cost_history.len() - 1) as f64;
        assert!(
            ratio > 0.8,
            "Cost should decrease in most iterations: {:.1}% decreasing",
            ratio * 100.0
        );
    }

    #[test]
    fn test_nmf_reconstruction_error() {
        let sr = 44100.0;
        let n = 11025; // 0.25 seconds

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 440.0 * t).sin()
            })
            .collect();

        let config = NmfConfig {
            fft_size: 1024,
            hop_size: 256,
            sample_rate: sr,
            num_components: 2,
            max_iterations: 100,
            cost_function: NmfCostFunction::Euclidean,
            initialization: NmfInitialization::Seeded,
            ..Default::default()
        };

        let result = nmf_separate(&signal, &config);

        // Sum of all components should approximate the original
        let recon_len = signal.len().min(result.components[0].len());
        let mut recon_error = 0.0;
        let mut signal_energy = 0.0;
        for i in 0..recon_len {
            let recon: f64 = result.components.iter().map(|c| c[i]).sum();
            let err = signal[i] - recon;
            recon_error += err * err;
            signal_energy += signal[i] * signal[i];
        }

        let snr = if recon_error > 0.0 {
            10.0 * (signal_energy / recon_error).log10()
        } else {
            100.0
        };

        assert!(
            snr > 5.0,
            "NMF reconstruction SNR should be reasonable: {:.1} dB",
            snr
        );
    }

    #[test]
    fn test_nmf_kl_divergence() {
        let sr = 44100.0;
        let n = 11025;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 330.0 * t).sin() + 0.5 * (2.0 * PI * 550.0 * t).sin()
            })
            .collect();

        let config = NmfConfig {
            fft_size: 1024,
            hop_size: 256,
            sample_rate: sr,
            num_components: 2,
            max_iterations: 50,
            cost_function: NmfCostFunction::KlDivergence,
            initialization: NmfInitialization::Seeded,
            ..Default::default()
        };

        let result = nmf_separate(&signal, &config);

        let first_cost = result.cost_history[0];
        let last_cost = result.cost_history[result.cost_history.len() - 1];
        assert!(
            last_cost < first_cost,
            "KL cost should decrease: first={}, last={}",
            first_cost,
            last_cost
        );
    }

    #[test]
    fn test_spectral_subtraction_removes_tone() {
        let sr = 44100.0;
        let n = 44100;

        // Two tones at very different frequencies
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 440.0 * t).sin() + 0.8 * (2.0 * PI * 1000.0 * t).sin()
            })
            .collect();

        let config = SpectralSubtractionConfig {
            fft_size: 4096,
            hop_size: 1024,
            sample_rate: sr,
            max_sources: 2,
            num_harmonics: 8,
            attenuation: 0.85,
            ..Default::default()
        };

        let result = spectral_subtraction(&signal, &config);

        assert!(
            !result.sources.is_empty(),
            "Should extract at least one source"
        );

        // The first extracted source should be near one of the two frequencies
        let f0 = result.sources[0].fundamental;
        let close_to_440 = (f0 - 440.0).abs() < 50.0;
        let close_to_1000 = (f0 - 1000.0).abs() < 50.0;
        assert!(
            close_to_440 || close_to_1000,
            "Extracted fundamental {:.1} Hz should be near 440 or 1000 Hz",
            f0
        );

        // Residual should have less energy than original
        let orig_energy: f64 = signal.iter().map(|s| s * s).sum();
        let residual_energy: f64 = result.residual.iter().map(|s| s * s).sum();
        assert!(
            residual_energy < orig_energy,
            "Residual energy {:.4} should be less than original {:.4}",
            residual_energy,
            orig_energy
        );
    }

    #[test]
    fn test_stft_roundtrip() {
        let sr = 44100.0;
        let n = 8192;
        let signal = sine_wave(440.0, sr, n);

        let fft_size = 1024;
        let hop = 256;

        let (stft_data, num_frames) = compute_stft(&signal, fft_size, hop, WindowType::Hann);
        let recon = inverse_stft(&stft_data, fft_size, hop, num_frames, n, WindowType::Hann);

        // Check reconstruction accuracy in the well-covered middle region
        let start = fft_size;
        let end = n - fft_size;
        let mut error = 0.0;
        let mut energy = 0.0;
        for i in start..end {
            let e = signal[i] - recon[i];
            error += e * e;
            energy += signal[i] * signal[i];
        }

        let snr = if error > 0.0 {
            10.0 * (energy / error).log10()
        } else {
            100.0
        };

        assert!(
            snr > 30.0,
            "STFT roundtrip SNR should be high: {:.1} dB",
            snr
        );
    }

    #[test]
    fn test_median_filter() {
        // Simple test: median of [1, 5, 3] = 3
        let spec = vec![vec![1.0, 2.0], vec![5.0, 8.0], vec![3.0, 4.0]];
        let filtered = median_filter_horizontal(&spec, 3);
        assert!((filtered[1][0] - 3.0).abs() < 1e-10);
        assert!((filtered[1][1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nmf_harmonic_comb_init() {
        let sr = 44100.0;
        let n = 11025;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 440.0 * t).sin() + 0.6 * (2.0 * PI * 660.0 * t).sin()
            })
            .collect();

        let config = NmfConfig {
            fft_size: 1024,
            hop_size: 256,
            sample_rate: sr,
            num_components: 2,
            max_iterations: 50,
            cost_function: NmfCostFunction::Euclidean,
            initialization: NmfInitialization::HarmonicComb,
            base_frequencies: vec![440.0, 660.0],
            ..Default::default()
        };

        let result = nmf_separate(&signal, &config);

        // With harmonic comb init, we should get reasonable separation
        assert_eq!(result.components.len(), 2);
        for comp in &result.components {
            let energy: f64 = comp.iter().map(|s| s * s).sum();
            assert!(energy > 0.0, "Each component should have some energy");
        }
    }
}
