//! Vocal isolation and extraction.
//!
//! Higher-level interface combining separation tools for vocal extraction,
//! corresponding to the vocal-focused paths in Melodyne's separation engine:
//!
//! - **Center-channel extraction**: Sum/difference for stereo vocal isolation
//! - **Harmonic mask generation**: Pitch-guided spectral masking
//! - **Vocal range filtering**: Frequency constraints for vocal fundamentals
//! - **Quality metrics**: SDR, SIR for evaluating separation quality

use crate::fft::WindowType;
use crate::separation::{compute_stft, inverse_stft};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for vocal isolation.
#[derive(Debug, Clone)]
pub struct VocalIsolationConfig {
    /// FFT size for analysis.
    pub fft_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Window type.
    pub window_type: WindowType,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Minimum vocal fundamental frequency in Hz.
    pub vocal_min_freq: f64,
    /// Maximum vocal fundamental frequency in Hz.
    pub vocal_max_freq: f64,
    /// Number of harmonics to include in the vocal mask.
    pub num_harmonics: usize,
    /// Harmonic mask bandwidth in bins around each harmonic.
    pub harmonic_bandwidth: usize,
    /// Mask smoothing kernel size (in time frames).
    pub mask_smoothing_frames: usize,
    /// Mask smoothing kernel size (in frequency bins).
    pub mask_smoothing_bins: usize,
    /// Minimum mask value (to avoid complete zeroing artifacts).
    pub mask_floor: f64,
}

impl Default for VocalIsolationConfig {
    fn default() -> Self {
        Self {
            fft_size: 4096,
            hop_size: 1024,
            window_type: WindowType::Hann,
            sample_rate: 44100.0,
            vocal_min_freq: 80.0,
            vocal_max_freq: 1200.0,
            num_harmonics: 20,
            harmonic_bandwidth: 3,
            mask_smoothing_frames: 3,
            mask_smoothing_bins: 3,
            mask_floor: 0.02,
        }
    }
}

// ---------------------------------------------------------------------------
// Center-channel extraction (stereo)
// ---------------------------------------------------------------------------

/// Result of center-channel extraction.
#[derive(Debug, Clone)]
pub struct CenterExtractionResult {
    /// Center channel (sum: typically contains the vocal).
    pub center: Vec<f64>,
    /// Sides channel (difference: typically contains panned instruments).
    pub sides: Vec<f64>,
}

/// Extract the center channel from a stereo signal.
///
/// For stereo recordings, vocals are typically panned center. The center
/// content is approximated by (L + R) / 2, and sides by (L - R) / 2.
///
/// This simple approach works well when the vocal is truly centered and
/// the mix has good stereo separation.
pub fn extract_center_channel(left: &[f64], right: &[f64]) -> CenterExtractionResult {
    let len = left.len().min(right.len());
    let mut center = vec![0.0f64; len];
    let mut sides = vec![0.0f64; len];

    for i in 0..len {
        center[i] = (left[i] + right[i]) * 0.5;
        sides[i] = (left[i] - right[i]) * 0.5;
    }

    CenterExtractionResult { center, sides }
}

/// Frequency-domain center extraction with spectral masking.
///
/// More sophisticated than simple sum/difference: computes a per-bin mask
/// based on the correlation between left and right channels. Bins where
/// L and R are similar (correlated) are assigned to center; bins where
/// they differ are assigned to sides.
pub fn extract_center_spectral(
    left: &[f64],
    right: &[f64],
    config: &VocalIsolationConfig,
) -> CenterExtractionResult {
    let n = config.fft_size;
    let hop = config.hop_size;
    let num_bins = n / 2 + 1;

    let (stft_l, num_frames) = compute_stft(left, n, hop, config.window_type);
    let (stft_r, _) = compute_stft(right, n, hop, config.window_type);

    let signal_len = left.len().min(right.len());

    let mut center_stft = vec![Complex::new(0.0, 0.0); num_frames * n];
    let mut sides_stft = vec![Complex::new(0.0, 0.0); num_frames * n];

    for t in 0..num_frames {
        for k in 0..num_bins {
            let l = stft_l[t * n + k];
            let r = stft_r[t * n + k];

            // Compute panning coefficient: how "centered" is this bin?
            // Using correlation: Re(L * conj(R)) / (|L| * |R|)
            let l_mag = l.norm();
            let r_mag = r.norm();
            let denom = l_mag * r_mag;

            let correlation = if denom > 1e-30 {
                (l * r.conj()).re / denom
            } else {
                0.0
            };

            // Convert correlation to center mask (high correlation = center)
            // Correlation ranges from -1 to 1; center content has correlation ~ 1
            let center_mask = ((correlation + 1.0) / 2.0).clamp(0.0, 1.0);
            let sides_mask = 1.0 - center_mask;

            // Apply mask to sum and difference
            let sum = (l + r) * 0.5;
            let diff = (l - r) * 0.5;

            center_stft[t * n + k] = sum * center_mask + diff * (1.0 - center_mask) * 0.1;
            sides_stft[t * n + k] = diff * sides_mask + sum * (1.0 - sides_mask) * 0.1;

            if k > 0 && k < num_bins - 1 {
                center_stft[t * n + (n - k)] = center_stft[t * n + k].conj();
                sides_stft[t * n + (n - k)] = sides_stft[t * n + k].conj();
            }
        }
    }

    let center = inverse_stft(&center_stft, n, hop, num_frames, signal_len, config.window_type);
    let sides = inverse_stft(&sides_stft, n, hop, num_frames, signal_len, config.window_type);

    CenterExtractionResult { center, sides }
}

// ---------------------------------------------------------------------------
// Harmonic mask generation
// ---------------------------------------------------------------------------

/// A spectral mask for vocal isolation.
#[derive(Debug, Clone)]
pub struct SpectralMask {
    /// Mask values: [num_frames][num_bins], each in range [mask_floor, 1.0].
    pub mask: Vec<Vec<f64>>,
    /// Number of time frames.
    pub num_frames: usize,
    /// Number of frequency bins.
    pub num_bins: usize,
}

/// Generate a harmonic mask for a known pitch contour.
///
/// Creates a spectral mask that passes energy at harmonics of the given
/// fundamental frequency contour, suppressing non-harmonic content.
///
/// The mask is smoothed in both time and frequency to reduce artifacts.
pub fn generate_harmonic_mask(
    pitch_contour: &[f64],
    num_frames: usize,
    num_bins: usize,
    config: &VocalIsolationConfig,
) -> SpectralMask {
    let freq_res = config.sample_rate / config.fft_size as f64;
    let bw = config.harmonic_bandwidth;
    let floor = config.mask_floor;

    let mut mask = vec![vec![floor; num_bins]; num_frames];

    for (t, &f0) in pitch_contour.iter().enumerate() {
        if t >= num_frames || f0 < config.vocal_min_freq || f0 > config.vocal_max_freq {
            continue;
        }

        for h in 1..=config.num_harmonics {
            let harm_freq = f0 * h as f64;
            if harm_freq >= config.sample_rate / 2.0 {
                break;
            }

            let center_bin = (harm_freq / freq_res).round() as usize;
            if center_bin >= num_bins {
                break;
            }

            // Gaussian-shaped mask around the harmonic
            let start = center_bin.saturating_sub(bw);
            let end = (center_bin + bw + 1).min(num_bins);

            for (k, mask_val) in mask[t].iter_mut().enumerate().take(end).skip(start) {
                let dist = (k as f64 - center_bin as f64).abs();
                let weight = (-dist * dist / (2.0 * (bw as f64 / 2.0).powi(2))).exp();
                // Harmonic decay: higher harmonics get weaker mask
                let harmonic_weight = 1.0 / (1.0 + 0.1 * (h - 1) as f64);
                let new_val = weight * harmonic_weight;
                *mask_val = mask_val.max(new_val).min(1.0);
            }
        }
    }

    // Smooth the mask in time
    if config.mask_smoothing_frames > 1 {
        smooth_mask_time(&mut mask, num_frames, num_bins, config.mask_smoothing_frames);
    }

    // Smooth the mask in frequency
    if config.mask_smoothing_bins > 1 {
        smooth_mask_frequency(&mut mask, num_frames, num_bins, config.mask_smoothing_bins);
    }

    // Enforce floor
    for mask_row in mask.iter_mut().take(num_frames) {
        for mask_val in mask_row.iter_mut().take(num_bins) {
            *mask_val = mask_val.max(floor);
        }
    }

    SpectralMask {
        mask,
        num_frames,
        num_bins,
    }
}

/// Generate a vocal-range frequency mask.
///
/// Creates a mask that passes frequencies within the expected vocal
/// range (fundamentals between vocal_min_freq and vocal_max_freq,
/// plus harmonics up to Nyquist) with smooth roll-off at the boundaries.
pub fn generate_vocal_range_mask(
    num_frames: usize,
    num_bins: usize,
    config: &VocalIsolationConfig,
) -> SpectralMask {
    let freq_res = config.sample_rate / config.fft_size as f64;
    let floor = config.mask_floor;

    // Vocal energy extends beyond the fundamental range due to harmonics.
    // Allow up to ~10 kHz for vocal overtones and sibilance.
    let vocal_low = config.vocal_min_freq;
    let vocal_high = 10000.0f64.min(config.sample_rate / 2.0);

    // Transition bandwidth in Hz
    let transition_bw = 50.0;

    let mask_row: Vec<f64> = (0..num_bins)
        .map(|k| {
            let freq = k as f64 * freq_res;

            if freq < vocal_low - transition_bw || freq > vocal_high + transition_bw {
                floor
            } else if freq < vocal_low {
                // Smooth ramp up
                let t = (freq - (vocal_low - transition_bw)) / transition_bw;
                floor + (1.0 - floor) * t * t
            } else if freq > vocal_high {
                // Smooth ramp down
                let t = ((vocal_high + transition_bw) - freq) / transition_bw;
                floor + (1.0 - floor) * t * t
            } else {
                1.0
            }
        })
        .collect();

    let mask = vec![mask_row; num_frames];

    SpectralMask {
        mask,
        num_frames,
        num_bins,
    }
}

// ---------------------------------------------------------------------------
// Vocal isolation pipeline
// ---------------------------------------------------------------------------

/// Result of vocal isolation.
#[derive(Debug, Clone)]
pub struct VocalIsolationResult {
    /// Extracted vocal signal.
    pub vocal: Vec<f64>,
    /// Accompaniment (everything except vocal).
    pub accompaniment: Vec<f64>,
    /// The spectral mask used for isolation.
    pub mask: SpectralMask,
}

/// Isolate vocals from a mono signal using harmonic masking.
///
/// Combines pitch tracking with harmonic mask generation and
/// vocal range constraints to extract the vocal content.
pub fn isolate_vocal_harmonic(
    audio: &[f64],
    pitch_contour: &[f64],
    config: &VocalIsolationConfig,
) -> VocalIsolationResult {
    let n = config.fft_size;
    let hop = config.hop_size;
    let num_bins = n / 2 + 1;

    let (stft, num_frames) = compute_stft(audio, n, hop, config.window_type);

    // Generate harmonic mask from pitch contour
    let harmonic_mask = generate_harmonic_mask(pitch_contour, num_frames, num_bins, config);

    // Also apply vocal range constraint
    let range_mask = generate_vocal_range_mask(num_frames, num_bins, config);

    // Combine masks (multiply)
    let mut combined_mask = SpectralMask {
        mask: vec![vec![config.mask_floor; num_bins]; num_frames],
        num_frames,
        num_bins,
    };
    for t in 0..num_frames {
        for k in 0..num_bins {
            combined_mask.mask[t][k] =
                (harmonic_mask.mask[t][k] * range_mask.mask[t][k]).max(config.mask_floor);
        }
    }

    // Apply mask to STFT
    let mut vocal_stft = vec![Complex::new(0.0, 0.0); num_frames * n];
    let mut accomp_stft = vec![Complex::new(0.0, 0.0); num_frames * n];

    for t in 0..num_frames {
        for k in 0..num_bins {
            let mask_val = combined_mask.mask[t][k];
            let orig = stft[t * n + k];

            vocal_stft[t * n + k] = orig * mask_val;
            accomp_stft[t * n + k] = orig * (1.0 - mask_val);

            if k > 0 && k < num_bins - 1 {
                vocal_stft[t * n + (n - k)] = vocal_stft[t * n + k].conj();
                accomp_stft[t * n + (n - k)] = accomp_stft[t * n + k].conj();
            }
        }
    }

    let vocal = inverse_stft(&vocal_stft, n, hop, num_frames, audio.len(), config.window_type);
    let accompaniment =
        inverse_stft(&accomp_stft, n, hop, num_frames, audio.len(), config.window_type);

    VocalIsolationResult {
        vocal,
        accompaniment,
        mask: combined_mask,
    }
}

/// Isolate vocals from a stereo signal combining center extraction and
/// harmonic masking.
pub fn isolate_vocal_stereo(
    left: &[f64],
    right: &[f64],
    pitch_contour: &[f64],
    config: &VocalIsolationConfig,
) -> VocalIsolationResult {
    // Step 1: Center extraction
    let center_result = extract_center_channel(left, right);

    // Step 2: Harmonic masking on the center channel
    isolate_vocal_harmonic(&center_result.center, pitch_contour, config)
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

/// Quality metrics for evaluating source separation.
#[derive(Debug, Clone)]
pub struct SeparationQuality {
    /// Signal-to-Distortion Ratio in dB.
    /// Measures overall separation quality.
    pub sdr: f64,
    /// Signal-to-Interference Ratio in dB.
    /// Measures how well interfering sources are rejected.
    pub sir: f64,
    /// Signal-to-Artifact Ratio in dB.
    /// Measures artifact level introduced by the separation process.
    pub sar: f64,
}

/// Compute separation quality metrics (SDR, SIR, SAR).
///
/// Based on BSS Eval methodology:
/// - SDR: Signal-to-Distortion Ratio
/// - SIR: Signal-to-Interference Ratio
/// - SAR: Signal-to-Artifact Ratio
///
/// `reference` is the clean reference signal, `estimated` is the separated
/// signal, and `interference` is the original interfering signal (if known).
pub fn compute_quality_metrics(
    reference: &[f64],
    estimated: &[f64],
    interference: Option<&[f64]>,
) -> SeparationQuality {
    let len = reference.len().min(estimated.len());
    if len == 0 {
        return SeparationQuality {
            sdr: 0.0,
            sir: 0.0,
            sar: 0.0,
        };
    }

    let ref_slice = &reference[..len];
    let est_slice = &estimated[..len];

    // Reference energy
    let ref_energy: f64 = ref_slice.iter().map(|s| s * s).sum();
    if ref_energy < 1e-30 {
        return SeparationQuality {
            sdr: 0.0,
            sir: 0.0,
            sar: 0.0,
        };
    }

    // SDR: 10 * log10(||s_ref||^2 / ||s_ref - s_est||^2)
    let distortion_energy: f64 = ref_slice
        .iter()
        .zip(est_slice.iter())
        .map(|(&r, &e)| {
            let d = r - e;
            d * d
        })
        .sum();

    let sdr = if distortion_energy > 1e-30 {
        10.0 * (ref_energy / distortion_energy).log10()
    } else {
        100.0
    };

    // SIR: requires interference signal
    let sir = if let Some(interf) = interference {
        let interf_len = len.min(interf.len());
        let interf_slice = &interf[..interf_len];

        // Project estimated onto interference subspace
        // s_interf = <est, interf> / <interf, interf> * interf
        let interf_energy: f64 = interf_slice.iter().map(|s| s * s).sum();
        if interf_energy > 1e-30 {
            let cross: f64 = est_slice
                .iter()
                .zip(interf_slice.iter())
                .map(|(&e, &i)| e * i)
                .sum();
            let proj_scale = cross / interf_energy;
            let interf_component_energy: f64 = interf_slice
                .iter()
                .map(|&i| {
                    let proj = proj_scale * i;
                    proj * proj
                })
                .sum();

            // SIR = 10 * log10(||s_target_in_est||^2 / ||s_interf_in_est||^2)
            // Approximate target component as: est - interference projection
            let target_component_energy = ref_energy;

            if interf_component_energy > 1e-30 {
                10.0 * (target_component_energy / interf_component_energy).log10()
            } else {
                100.0
            }
        } else {
            100.0
        }
    } else {
        // Without interference signal, estimate SIR from SDR
        sdr + 3.0 // Rough approximation
    };

    // SAR: Signal-to-Artifact Ratio
    // Artifact = distortion - interference component
    // Approximate: SAR = SDR + 6 dB (heuristic when exact decomposition unavailable)
    let sar = if interference.is_some() {
        // e_artif = e_est - s_target_proj - s_interf_proj
        // For simplicity, use: SAR = 10 * log10(||s_target||^2 / ||e_artif||^2)
        // where e_artif = est - reference (approximate)
        let artif_energy: f64 = ref_slice
            .iter()
            .zip(est_slice.iter())
            .map(|(&r, &e)| {
                let a = e - r;
                a * a
            })
            .sum();
        if artif_energy > 1e-30 {
            10.0 * (ref_energy / artif_energy).log10()
        } else {
            100.0
        }
    } else {
        sdr // Without more info, SAR equals SDR
    };

    SeparationQuality { sdr, sir, sar }
}

/// Compute SDR (Signal-to-Distortion Ratio) only.
///
/// A simpler version when only SDR is needed.
pub fn compute_sdr(reference: &[f64], estimated: &[f64]) -> f64 {
    let len = reference.len().min(estimated.len());
    if len == 0 {
        return 0.0;
    }

    let ref_energy: f64 = reference[..len].iter().map(|s| s * s).sum();
    let error_energy: f64 = reference[..len]
        .iter()
        .zip(estimated[..len].iter())
        .map(|(&r, &e)| (r - e).powi(2))
        .sum();

    if error_energy > 1e-30 {
        10.0 * (ref_energy / error_energy).log10()
    } else if ref_energy > 1e-30 {
        100.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Mask smoothing helpers
// ---------------------------------------------------------------------------

/// Smooth mask along the time axis using a moving average.
fn smooth_mask_time(
    mask: &mut [Vec<f64>],
    num_frames: usize,
    num_bins: usize,
    kernel: usize,
) {
    let half = kernel / 2;
    let mut temp = vec![0.0f64; num_frames];

    for k in 0..num_bins {
        // Extract column
        for (t, temp_val) in temp.iter_mut().enumerate().take(num_frames) {
            *temp_val = mask[t][k];
        }
        // Smooth
        for (t, mask_row) in mask.iter_mut().enumerate().take(num_frames) {
            let start = t.saturating_sub(half);
            let end = (t + half + 1).min(num_frames);
            let count = (end - start) as f64;
            let sum: f64 = temp[start..end].iter().sum();
            mask_row[k] = sum / count;
        }
    }
}

/// Smooth mask along the frequency axis using a moving average.
fn smooth_mask_frequency(
    mask: &mut [Vec<f64>],
    num_frames: usize,
    num_bins: usize,
    kernel: usize,
) {
    let half = kernel / 2;
    let mut temp = vec![0.0f64; num_bins];

    for mask_row in mask.iter_mut().take(num_frames) {
        // Copy row
        temp[..num_bins].copy_from_slice(&mask_row[..num_bins]);
        // Smooth
        for (k, mask_val) in mask_row.iter_mut().enumerate().take(num_bins) {
            let start = k.saturating_sub(half);
            let end = (k + half + 1).min(num_bins);
            let count = (end - start) as f64;
            let sum: f64 = temp[start..end].iter().sum();
            *mask_val = sum / count;
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

    fn sine_wave(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_center_extraction_centered_vocal() {
        let sr = 44100.0;
        let n = 44100;

        // Vocal: centered (same in L and R)
        let vocal = sine_wave(440.0, sr, n);

        // Instrument: panned left
        let instrument: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f64 / sr).sin())
            .collect();

        let left: Vec<f64> = vocal.iter().zip(instrument.iter()).map(|(&v, &i)| v + i).collect();
        let right: Vec<f64> = vocal.iter().zip(instrument.iter()).map(|(&v, &_i)| v).collect();

        let result = extract_center_channel(&left, &right);

        // Center should be close to the vocal
        let center_energy: f64 = result.center.iter().map(|s| s * s).sum();
        let vocal_energy: f64 = vocal.iter().map(|s| s * s).sum();

        assert!(
            center_energy > 0.0,
            "Center should have energy"
        );

        // SDR of center vs vocal
        let sdr = compute_sdr(&vocal, &result.center);
        assert!(
            sdr > 0.0,
            "SDR should be positive for centered vocal: {:.1} dB",
            sdr
        );
    }

    #[test]
    fn test_center_extraction_symmetry() {
        let sr = 44100.0;
        let n = 22050;

        // Perfectly centered signal
        let signal = sine_wave(440.0, sr, n);
        let left = signal.clone();
        let right = signal.clone();

        let result = extract_center_channel(&left, &right);

        // Center should equal the signal
        let error: f64 = result
            .center
            .iter()
            .zip(signal.iter())
            .map(|(&c, &s)| (c - s).powi(2))
            .sum();
        let energy: f64 = signal.iter().map(|s| s * s).sum();

        assert!(
            error / energy < 0.01,
            "Centered signal should pass through perfectly"
        );

        // Sides should be near zero
        let sides_energy: f64 = result.sides.iter().map(|s| s * s).sum();
        assert!(
            sides_energy / energy < 0.01,
            "Sides should be near zero for centered signal"
        );
    }

    #[test]
    fn test_harmonic_mask_generation() {
        let config = VocalIsolationConfig {
            fft_size: 4096,
            sample_rate: 44100.0,
            vocal_min_freq: 80.0,
            vocal_max_freq: 1200.0,
            num_harmonics: 10,
            harmonic_bandwidth: 3,
            mask_smoothing_frames: 1,
            mask_smoothing_bins: 1,
            mask_floor: 0.02,
            ..Default::default()
        };

        let freq_res = config.sample_rate / config.fft_size as f64;
        let num_frames = 10;
        let num_bins = config.fft_size / 2 + 1;

        // Constant pitch at 440 Hz
        let pitch_contour = vec![440.0; num_frames];

        let mask = generate_harmonic_mask(&pitch_contour, num_frames, num_bins, &config);

        assert_eq!(mask.num_frames, num_frames);
        assert_eq!(mask.num_bins, num_bins);

        // Check that mask has peaks at harmonics of 440 Hz
        let mid_frame = num_frames / 2;
        for h in 1..=5 {
            let harm_freq = 440.0 * h as f64;
            let bin = (harm_freq / freq_res).round() as usize;
            if bin < num_bins {
                assert!(
                    mask.mask[mid_frame][bin] > 0.3,
                    "Mask at harmonic {} (bin {}) should be > 0.3, got {:.3}",
                    h,
                    bin,
                    mask.mask[mid_frame][bin]
                );
            }
        }

        // Check that mask at non-harmonic frequencies is low
        let non_harmonic_bin = ((440.0 * 1.5) / freq_res).round() as usize;
        assert!(
            mask.mask[mid_frame][non_harmonic_bin] < 0.5,
            "Mask between harmonics should be lower: {:.3}",
            mask.mask[mid_frame][non_harmonic_bin]
        );
    }

    #[test]
    fn test_vocal_range_mask() {
        let config = VocalIsolationConfig::default();
        let num_frames = 5;
        let num_bins = config.fft_size / 2 + 1;

        let mask = generate_vocal_range_mask(num_frames, num_bins, &config);

        let freq_res = config.sample_rate / config.fft_size as f64;

        // Within vocal range: should be close to 1.0
        let vocal_bin = (500.0 / freq_res).round() as usize;
        assert!(
            mask.mask[0][vocal_bin] > 0.8,
            "Within vocal range (500 Hz) should have high mask: {:.3}",
            mask.mask[0][vocal_bin]
        );

        // Well below vocal range: should be at floor
        let low_bin = (20.0 / freq_res).round() as usize;
        if low_bin > 0 {
            assert!(
                mask.mask[0][low_bin] < 0.3,
                "Below vocal range (20 Hz) should have low mask: {:.3}",
                mask.mask[0][low_bin]
            );
        }

        // Above 10 kHz: should be at floor
        let high_bin = (15000.0 / freq_res).round() as usize;
        if high_bin < num_bins {
            assert!(
                mask.mask[0][high_bin] < 0.3,
                "Above vocal range (15 kHz) should have low mask: {:.3}",
                mask.mask[0][high_bin]
            );
        }
    }

    #[test]
    fn test_vocal_isolation_harmonic() {
        let sr = 44100.0;
        let n = 44100; // 1 second
        let hop = 1024;

        // Vocal: 440 Hz with harmonics
        let vocal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 440.0 * t).sin()
                    + 0.5 * (2.0 * PI * 880.0 * t).sin()
                    + 0.3 * (2.0 * PI * 1320.0 * t).sin()
            })
            .collect();

        // Non-vocal: 2500 Hz tone (above typical vocal fundamental)
        let noise: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 2500.0 * i as f64 / sr).sin())
            .collect();

        let mixed: Vec<f64> = vocal.iter().zip(noise.iter()).map(|(&v, &n)| v + n).collect();

        // Number of frames
        let num_frames = (n - 4096) / hop + 1;
        let pitch_contour = vec![440.0; num_frames];

        let config = VocalIsolationConfig {
            fft_size: 4096,
            hop_size: hop,
            sample_rate: sr,
            vocal_min_freq: 80.0,
            vocal_max_freq: 1200.0,
            num_harmonics: 10,
            harmonic_bandwidth: 4,
            mask_smoothing_frames: 1,
            mask_smoothing_bins: 1,
            mask_floor: 0.02,
            ..Default::default()
        };

        let result = isolate_vocal_harmonic(&mixed, &pitch_contour, &config);

        let vocal_energy: f64 = result.vocal.iter().map(|s| s * s).sum();
        let accomp_energy: f64 = result.accompaniment.iter().map(|s| s * s).sum();

        assert!(vocal_energy > 0.0, "Vocal output should have energy");
        assert!(accomp_energy > 0.0, "Accompaniment should have energy");

        // The vocal output should correlate more with the original vocal
        // than with the noise
        let vocal_correlation: f64 = result
            .vocal
            .iter()
            .zip(vocal.iter())
            .map(|(&v, &r)| v * r)
            .sum::<f64>()
            .abs();
        let noise_correlation: f64 = result
            .vocal
            .iter()
            .zip(noise.iter())
            .map(|(&v, &n)| v * n)
            .sum::<f64>()
            .abs();

        assert!(
            vocal_correlation > noise_correlation,
            "Vocal output should correlate more with vocal ({:.2}) than noise ({:.2})",
            vocal_correlation,
            noise_correlation
        );
    }

    #[test]
    fn test_quality_metrics_perfect() {
        // Perfect reconstruction
        let signal = sine_wave(440.0, 44100.0, 4096);
        let metrics = compute_quality_metrics(&signal, &signal, None);

        assert!(
            metrics.sdr > 90.0,
            "Perfect match should have very high SDR: {:.1} dB",
            metrics.sdr
        );
    }

    #[test]
    fn test_quality_metrics_scaled() {
        let signal = sine_wave(440.0, 44100.0, 4096);
        // Scale by 0.5 (6 dB attenuation)
        let scaled: Vec<f64> = signal.iter().map(|&s| s * 0.5).collect();

        let metrics = compute_quality_metrics(&signal, &scaled, None);

        // SDR should be finite and positive-ish (depends on scale mismatch)
        assert!(
            metrics.sdr.is_finite(),
            "SDR should be finite: {}",
            metrics.sdr
        );
    }

    #[test]
    fn test_quality_metrics_with_interference() {
        let sr = 44100.0;
        let n = 44100;

        let reference = sine_wave(440.0, sr, n);
        let interference = sine_wave(1000.0, sr, n);

        // Estimated = reference + small interference leak
        let estimated: Vec<f64> = reference
            .iter()
            .zip(interference.iter())
            .map(|(&r, &i)| r + 0.1 * i)
            .collect();

        let metrics = compute_quality_metrics(&reference, &estimated, Some(&interference));

        assert!(
            metrics.sdr > 10.0,
            "SDR should be good with small interference: {:.1} dB",
            metrics.sdr
        );
        assert!(
            metrics.sir > 5.0,
            "SIR should be positive: {:.1} dB",
            metrics.sir
        );
    }

    #[test]
    fn test_sdr_computation() {
        let signal = sine_wave(440.0, 44100.0, 8192);

        // SDR of signal vs itself
        let sdr_perfect = compute_sdr(&signal, &signal);
        assert!(sdr_perfect > 90.0, "SDR of identical signals should be very high");

        // SDR of signal vs noisy version
        let noisy: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.1 * (i as f64 * 123.456).sin())
            .collect();
        let sdr_noisy = compute_sdr(&signal, &noisy);
        assert!(
            sdr_noisy > 0.0 && sdr_noisy < 90.0,
            "SDR of noisy signal should be moderate: {:.1} dB",
            sdr_noisy
        );
    }

    #[test]
    fn test_spectral_center_extraction() {
        let sr = 44100.0;
        let n = 22050;

        // Centered vocal
        let vocal = sine_wave(440.0, sr, n);

        // Panned instrument (only in left)
        let instrument: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 1500.0 * i as f64 / sr).sin())
            .collect();

        let left: Vec<f64> = vocal.iter().zip(instrument.iter()).map(|(&v, &i)| v + i).collect();
        let right = vocal.clone();

        let config = VocalIsolationConfig {
            fft_size: 2048,
            hop_size: 512,
            sample_rate: sr,
            ..Default::default()
        };

        let result = extract_center_spectral(&left, &right, &config);

        let center_energy: f64 = result.center.iter().map(|s| s * s).sum();
        assert!(center_energy > 0.0, "Spectral center should have energy");
    }

    #[test]
    fn test_mask_smoothing() {
        let num_frames = 10;
        let num_bins = 20;
        let mut mask = vec![vec![0.0; num_bins]; num_frames];

        // Single point peak
        mask[5][10] = 1.0;

        smooth_mask_time(&mut mask, num_frames, num_bins, 3);

        // After smoothing, the peak should spread to neighbors
        assert!(mask[4][10] > 0.0, "Smoothing should spread temporally");
        assert!(mask[6][10] > 0.0, "Smoothing should spread temporally");
        assert!(mask[5][10] < 1.0, "Peak should be reduced by smoothing");
    }

    #[test]
    fn test_stereo_vocal_isolation() {
        let sr = 44100.0;
        let n = 22050;
        let hop = 1024;

        let vocal = sine_wave(440.0, sr, n);
        let instrument: Vec<f64> = (0..n)
            .map(|i| 0.3 * (2.0 * PI * 2000.0 * i as f64 / sr).sin())
            .collect();

        // Vocal centered, instrument panned
        let left: Vec<f64> = vocal.iter().zip(instrument.iter()).map(|(&v, &i)| v + i).collect();
        let right: Vec<f64> = vocal.iter().map(|&v| v).collect();

        let num_frames = (n - 4096) / hop + 1;
        let pitch_contour = vec![440.0; num_frames];

        let config = VocalIsolationConfig {
            fft_size: 4096,
            hop_size: hop,
            sample_rate: sr,
            ..Default::default()
        };

        let result = isolate_vocal_stereo(&left, &right, &pitch_contour, &config);

        let vocal_energy: f64 = result.vocal.iter().map(|s| s * s).sum();
        assert!(vocal_energy > 0.0, "Stereo vocal isolation should produce output");
    }
}
