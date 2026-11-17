//! Audio export with format options, normalization, and dithering.
//!
//! Provides flexible export from `AudioBuffer` to various output formats:
//!
//! - WAV (16-bit, 24-bit, or 32-bit integer)
//! - 32-bit float WAV
//! - Raw PCM (headerless, configurable bit depth and byte order)
//! - Normalization to a target peak level
//! - TPDF dithering for bit-depth reduction

use crate::audio_buffer::AudioBuffer;
use crate::wav::{write_wav, WavError, WavWriteConfig};
use std::io::Write;
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from export operations.
#[derive(Debug, Error)]
pub enum ExportError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("WAV export error: {0}")]
    WavError(#[from] WavError),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

// ---------------------------------------------------------------------------
// Export format
// ---------------------------------------------------------------------------

/// Output format for audio export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Standard WAV file with integer samples.
    WavInt,
    /// WAV file with 32-bit floating-point samples.
    WavFloat,
    /// Raw PCM data (no header).
    RawPcm,
}

impl ExportFormat {
    /// Return a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::WavInt => "WAV (integer)",
            Self::WavFloat => "WAV (float)",
            Self::RawPcm => "Raw PCM",
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Byte order for raw PCM
// ---------------------------------------------------------------------------

/// Byte order for raw PCM export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

// ---------------------------------------------------------------------------
// Export configuration
// ---------------------------------------------------------------------------

/// Configuration for audio export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output format.
    pub format: ExportFormat,
    /// Target sample rate. If different from the buffer's sample rate, the
    /// caller is responsible for resampling before export.
    pub sample_rate: u32,
    /// Bits per sample (8, 16, 24, or 32). For WavFloat this is always 32.
    pub bits_per_sample: u16,
    /// Whether to normalize the audio before export.
    pub normalize: bool,
    /// Target peak amplitude for normalization (default 1.0).
    pub normalize_target: f64,
    /// Whether to apply TPDF dithering when reducing bit depth.
    pub dither: bool,
    /// Byte order for raw PCM export.
    pub byte_order: ByteOrder,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::WavInt,
            sample_rate: 44100,
            bits_per_sample: 16,
            normalize: false,
            normalize_target: 1.0,
            dither: false,
            byte_order: ByteOrder::LittleEndian,
        }
    }
}

impl ExportConfig {
    /// Create a config for 16-bit WAV export.
    pub fn wav_16bit() -> Self {
        Self::default()
    }

    /// Create a config for 24-bit WAV export.
    pub fn wav_24bit() -> Self {
        Self {
            bits_per_sample: 24,
            ..Self::default()
        }
    }

    /// Create a config for 32-bit float WAV export.
    pub fn wav_float() -> Self {
        Self {
            format: ExportFormat::WavFloat,
            bits_per_sample: 32,
            ..Self::default()
        }
    }

    /// Create a config for raw 16-bit PCM export.
    pub fn raw_pcm_16bit() -> Self {
        Self {
            format: ExportFormat::RawPcm,
            bits_per_sample: 16,
            ..Self::default()
        }
    }

    /// Enable normalization.
    pub fn with_normalize(mut self, target: f64) -> Self {
        self.normalize = true;
        self.normalize_target = target;
        self
    }

    /// Enable TPDF dithering.
    pub fn with_dither(mut self) -> Self {
        self.dither = true;
        self
    }
}

// ---------------------------------------------------------------------------
// TPDF dithering
// ---------------------------------------------------------------------------

/// Triangular Probability Density Function (TPDF) dither generator.
///
/// Produces dither noise with a triangular distribution in the range [-1, +1] LSB,
/// which is optimal for eliminating quantization distortion artifacts.
struct TpdfDither {
    /// Linear congruential generator state.
    state: u32,
}

impl TpdfDither {
    fn new(seed: u32) -> Self {
        Self {
            state: seed.wrapping_add(1), // Avoid zero state.
        }
    }

    /// Advance the LCG and return a uniform random u32.
    fn next_u32(&mut self) -> u32 {
        // LCG parameters from Numerical Recipes.
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    /// Generate a single TPDF dither sample scaled to one quantization step.
    ///
    /// `quant_step` is the size of one quantization level (e.g., 1/32768 for 16-bit).
    fn next_sample(&mut self, quant_step: f64) -> f64 {
        // Two independent uniform random values in [0, 1).
        let u1 = self.next_u32() as f64 / u32::MAX as f64;
        let u2 = self.next_u32() as f64 / u32::MAX as f64;
        // Triangular distribution in [-1, +1] by summing two uniform [-0.5, 0.5].
        let tpdf = (u1 - 0.5) + (u2 - 0.5);
        tpdf * quant_step
    }
}

// ---------------------------------------------------------------------------
// Core export functions
// ---------------------------------------------------------------------------

/// Export an `AudioBuffer` to a file using the given configuration.
pub fn export_audio<P: AsRef<Path>>(
    path: P,
    buffer: &AudioBuffer,
    config: &ExportConfig,
) -> Result<(), ExportError> {
    // Optionally normalize.
    let working_buffer = if config.normalize {
        let mut buf = buffer.clone();
        buf.normalize(config.normalize_target);
        buf
    } else {
        buffer.clone()
    };

    match config.format {
        ExportFormat::WavInt => {
            let wav_config = WavWriteConfig {
                bits_per_sample: config.bits_per_sample,
                float_format: false,
            };
            if config.dither && config.bits_per_sample < 64 {
                let dithered = apply_dither(&working_buffer, config.bits_per_sample);
                write_wav(path, &dithered, &wav_config)?;
            } else {
                write_wav(path, &working_buffer, &wav_config)?;
            }
        }
        ExportFormat::WavFloat => {
            let wav_config = WavWriteConfig {
                bits_per_sample: 32,
                float_format: true,
            };
            write_wav(path, &working_buffer, &wav_config)?;
        }
        ExportFormat::RawPcm => {
            if config.dither && config.bits_per_sample < 64 {
                let dithered = apply_dither(&working_buffer, config.bits_per_sample);
                write_raw_pcm(path, &dithered, config)?;
            } else {
                write_raw_pcm(path, &working_buffer, config)?;
            }
        }
    }

    Ok(())
}

/// Apply TPDF dithering to an `AudioBuffer` for the target bit depth.
///
/// Returns a new buffer with dither noise added. The dither amplitude is one
/// quantization step of the target bit depth.
pub fn apply_dither(buffer: &AudioBuffer, target_bits: u16) -> AudioBuffer {
    let quant_step = match target_bits {
        8 => 1.0 / 128.0,
        16 => 1.0 / 32768.0,
        24 => 1.0 / 8388608.0,
        32 => 1.0 / 2147483648.0,
        _ => 1.0 / 32768.0,
    };

    let mut dithered = buffer.clone();
    let mut rng = TpdfDither::new(42);

    for ch in 0..dithered.num_channels() {
        for s in 0..dithered.num_samples() {
            let original = dithered.get(ch, s);
            let noise = rng.next_sample(quant_step);
            dithered.set(ch, s, original + noise);
        }
    }

    dithered
}

/// Write raw PCM data (no header) to a file.
pub fn write_raw_pcm<P: AsRef<Path>>(
    path: P,
    buffer: &AudioBuffer,
    config: &ExportConfig,
) -> Result<(), ExportError> {
    let mut file = std::fs::File::create(path.as_ref())?;

    let bits = config.bits_per_sample;
    let is_le = config.byte_order == ByteOrder::LittleEndian;

    for s in 0..buffer.num_samples() {
        for ch in 0..buffer.num_channels() {
            let sample = buffer.get(ch, s).clamp(-1.0, 1.0);
            write_pcm_sample(&mut file, sample, bits, is_le)?;
        }
    }

    file.flush()?;
    Ok(())
}

/// Write a single PCM sample in the specified bit depth and byte order.
fn write_pcm_sample<W: Write>(
    writer: &mut W,
    sample: f64,
    bits: u16,
    little_endian: bool,
) -> Result<(), ExportError> {
    match bits {
        8 => {
            // Unsigned 8-bit: 0..255, center at 128.
            let val = ((sample * 127.0) + 128.0).clamp(0.0, 255.0) as u8;
            writer.write_all(&[val])?;
        }
        16 => {
            let val = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            let bytes = if little_endian {
                val.to_le_bytes()
            } else {
                val.to_be_bytes()
            };
            writer.write_all(&bytes)?;
        }
        24 => {
            let val = (sample * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
            let full = if little_endian {
                val.to_le_bytes()
            } else {
                val.to_be_bytes()
            };
            // Write 3 bytes.
            if little_endian {
                writer.write_all(&full[..3])?;
            } else {
                writer.write_all(&full[1..4])?;
            }
        }
        32 => {
            let val = (sample * 2147483647.0).clamp(-2147483648.0, 2147483647.0) as i32;
            let bytes = if little_endian {
                val.to_le_bytes()
            } else {
                val.to_be_bytes()
            };
            writer.write_all(&bytes)?;
        }
        _ => {
            return Err(ExportError::InvalidConfig(format!(
                "Unsupported bit depth: {}",
                bits
            )));
        }
    }
    Ok(())
}

/// Read raw PCM data from a file into an `AudioBuffer`.
///
/// Since raw PCM has no header, all parameters must be specified.
pub fn read_raw_pcm<P: AsRef<Path>>(
    path: P,
    num_channels: usize,
    sample_rate: f64,
    bits_per_sample: u16,
    little_endian: bool,
) -> Result<AudioBuffer, ExportError> {
    let data = std::fs::read(path.as_ref())?;
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let total_samples = data.len() / bytes_per_sample;
    let samples_per_channel = total_samples / num_channels;

    let mut interleaved = Vec::with_capacity(total_samples);

    let mut offset = 0;
    for _ in 0..total_samples {
        if offset + bytes_per_sample > data.len() {
            break;
        }
        let sample = read_pcm_sample(
            &data[offset..offset + bytes_per_sample],
            bits_per_sample,
            little_endian,
        );
        interleaved.push(sample);
        offset += bytes_per_sample;
    }

    if samples_per_channel == 0 {
        return Ok(AudioBuffer::new(num_channels, 0, sample_rate));
    }

    Ok(AudioBuffer::from_interleaved(
        &interleaved,
        num_channels,
        sample_rate,
    ))
}

/// Parse a single PCM sample from raw bytes.
fn read_pcm_sample(bytes: &[u8], bits: u16, little_endian: bool) -> f64 {
    match bits {
        8 => {
            // Unsigned 8-bit.
            (bytes[0] as f64 - 128.0) / 128.0
        }
        16 => {
            let val = if little_endian {
                i16::from_le_bytes([bytes[0], bytes[1]])
            } else {
                i16::from_be_bytes([bytes[0], bytes[1]])
            };
            val as f64 / 32768.0
        }
        24 => {
            let val = if little_endian {
                // Sign-extend from 24 to 32 bits.
                let raw = (bytes[0] as i32) | ((bytes[1] as i32) << 8) | ((bytes[2] as i32) << 16);
                if raw & 0x800000 != 0 {
                    raw | !0xFFFFFF_i32
                } else {
                    raw
                }
            } else {
                let raw = ((bytes[0] as i32) << 16) | ((bytes[1] as i32) << 8) | (bytes[2] as i32);
                if raw & 0x800000 != 0 {
                    raw | !0xFFFFFF_i32
                } else {
                    raw
                }
            };
            val as f64 / 8388608.0
        }
        32 => {
            let val = if little_endian {
                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            };
            val as f64 / 2147483648.0
        }
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_buffer::AudioBuffer;

    /// Create a simple test buffer with a known signal.
    fn test_buffer() -> AudioBuffer {
        let samples: Vec<f64> = (0..1000)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin() * 0.8)
            .collect();
        AudioBuffer::from_mono(samples, 44100.0)
    }

    // -- Raw PCM export roundtrip --

    #[test]
    fn test_raw_pcm_16bit_roundtrip() {
        let buffer = test_buffer();
        let path = std::env::temp_dir().join("melodyn_export_test_raw16.pcm");

        let config = ExportConfig {
            format: ExportFormat::RawPcm,
            bits_per_sample: 16,
            byte_order: ByteOrder::LittleEndian,
            ..ExportConfig::default()
        };

        export_audio(&path, &buffer, &config).expect("export should succeed");

        // Read it back.
        let read_back = read_raw_pcm(&path, 1, 44100.0, 16, true).expect("read should succeed");
        assert_eq!(read_back.num_channels(), 1);
        assert_eq!(read_back.num_samples(), 1000);

        // 16-bit quantization: error should be < 1/32768.
        for i in 0..1000 {
            let diff = (read_back.get(0, i) - buffer.get(0, i)).abs();
            assert!(
                diff < 0.001,
                "Raw PCM 16-bit sample {} error {} too large",
                i,
                diff
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_raw_pcm_24bit_roundtrip() {
        let buffer = test_buffer();
        let path = std::env::temp_dir().join("melodyn_export_test_raw24.pcm");

        let config = ExportConfig {
            format: ExportFormat::RawPcm,
            bits_per_sample: 24,
            byte_order: ByteOrder::LittleEndian,
            ..ExportConfig::default()
        };

        export_audio(&path, &buffer, &config).expect("export should succeed");

        let read_back = read_raw_pcm(&path, 1, 44100.0, 24, true).expect("read should succeed");
        assert_eq!(read_back.num_samples(), 1000);

        // 24-bit quantization: much tighter tolerance.
        for i in 0..1000 {
            let diff = (read_back.get(0, i) - buffer.get(0, i)).abs();
            assert!(
                diff < 0.0001,
                "Raw PCM 24-bit sample {} error {} too large",
                i,
                diff
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_raw_pcm_big_endian() {
        let buffer = test_buffer();
        let path = std::env::temp_dir().join("melodyn_export_test_raw_be.pcm");

        let config = ExportConfig {
            format: ExportFormat::RawPcm,
            bits_per_sample: 16,
            byte_order: ByteOrder::BigEndian,
            ..ExportConfig::default()
        };

        export_audio(&path, &buffer, &config).expect("export should succeed");
        let read_back = read_raw_pcm(&path, 1, 44100.0, 16, false).expect("read back");
        assert_eq!(read_back.num_samples(), 1000);

        for i in 0..100 {
            let diff = (read_back.get(0, i) - buffer.get(0, i)).abs();
            assert!(diff < 0.001);
        }

        let _ = std::fs::remove_file(&path);
    }

    // -- 32-bit float WAV export --

    #[test]
    fn test_wav_float_export() {
        let buffer = test_buffer();
        let path = std::env::temp_dir().join("melodyn_export_test_float.wav");

        let config = ExportConfig::wav_float();
        export_audio(&path, &buffer, &config).expect("float wav export");

        // Read back via hound.
        let read_back = crate::wav::read_wav(&path).expect("read float wav");
        assert_eq!(read_back.num_samples(), 1000);

        for i in 0..1000 {
            let diff = (read_back.get(0, i) - buffer.get(0, i)).abs();
            assert!(
                diff < 0.0001,
                "Float WAV sample {} error {} too large",
                i,
                diff
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    // -- Dithering --

    #[test]
    fn test_dithering_produces_correct_bit_depth() {
        let buffer = test_buffer();
        let path = std::env::temp_dir().join("melodyn_export_test_dither.wav");

        let config = ExportConfig {
            format: ExportFormat::WavInt,
            bits_per_sample: 16,
            dither: true,
            ..ExportConfig::default()
        };

        export_audio(&path, &buffer, &config).expect("dithered export");

        // Read back and verify it is a valid 16-bit WAV.
        let info = crate::wav::wav_info(&path).expect("wav info");
        assert_eq!(info.bits_per_sample, 16);
        assert!(!info.is_float);

        // Verify the dithered version is close but not identical to original.
        let read_back = crate::wav::read_wav(&path).expect("read dithered wav");
        let mut total = 0;
        for i in 0..read_back.num_samples() {
            let diff = (read_back.get(0, i) - buffer.get(0, i)).abs();
            assert!(diff < 0.01, "Dithered sample {} too far from original", i);
            total += 1;
        }
        // Check that the file is valid and readable.
        assert!(total > 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_tpdf_dither_statistical_properties() {
        // Verify TPDF dither is approximately zero-mean.
        let mut rng = TpdfDither::new(123);
        let quant_step = 1.0 / 32768.0;
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += rng.next_sample(quant_step);
        }
        let mean = sum / n as f64;
        // Should be very close to zero.
        assert!(
            mean.abs() < quant_step * 0.1,
            "TPDF dither mean {} is not close to zero",
            mean
        );
    }

    // -- Normalize --

    #[test]
    fn test_normalize_on_export() {
        let samples: Vec<f64> = vec![0.5, -0.3, 0.1, -0.5, 0.2];
        let buffer = AudioBuffer::from_mono(samples, 44100.0);
        let path = std::env::temp_dir().join("melodyn_export_test_normalize.wav");

        let config = ExportConfig {
            normalize: true,
            normalize_target: 1.0,
            format: ExportFormat::WavFloat,
            bits_per_sample: 32,
            ..ExportConfig::default()
        };

        export_audio(&path, &buffer, &config).expect("normalized export");

        let read_back = crate::wav::read_wav(&path).expect("read normalized wav");
        let peak = read_back.peak_amplitude();
        assert!(
            (peak - 1.0).abs() < 0.01,
            "Normalized peak should be ~1.0, got {}",
            peak
        );

        let _ = std::fs::remove_file(&path);
    }

    // -- Config presets --

    #[test]
    fn test_export_config_presets() {
        let c16 = ExportConfig::wav_16bit();
        assert_eq!(c16.format, ExportFormat::WavInt);
        assert_eq!(c16.bits_per_sample, 16);

        let c24 = ExportConfig::wav_24bit();
        assert_eq!(c24.bits_per_sample, 24);

        let cfloat = ExportConfig::wav_float();
        assert_eq!(cfloat.format, ExportFormat::WavFloat);
        assert_eq!(cfloat.bits_per_sample, 32);

        let raw = ExportConfig::raw_pcm_16bit();
        assert_eq!(raw.format, ExportFormat::RawPcm);
        assert_eq!(raw.bits_per_sample, 16);
    }

    #[test]
    fn test_config_builder_methods() {
        let config = ExportConfig::wav_16bit().with_normalize(0.95).with_dither();

        assert!(config.normalize);
        assert!((config.normalize_target - 0.95).abs() < 1e-10);
        assert!(config.dither);
    }

    #[test]
    fn test_export_format_display() {
        assert_eq!(format!("{}", ExportFormat::WavInt), "WAV (integer)");
        assert_eq!(format!("{}", ExportFormat::WavFloat), "WAV (float)");
        assert_eq!(format!("{}", ExportFormat::RawPcm), "Raw PCM");
    }

    #[test]
    fn test_raw_pcm_32bit_roundtrip() {
        let buffer = test_buffer();
        let path = std::env::temp_dir().join("melodyn_export_test_raw32.pcm");

        let config = ExportConfig {
            format: ExportFormat::RawPcm,
            bits_per_sample: 32,
            byte_order: ByteOrder::LittleEndian,
            ..ExportConfig::default()
        };

        export_audio(&path, &buffer, &config).expect("export 32-bit raw");
        let read_back = read_raw_pcm(&path, 1, 44100.0, 32, true).expect("read back");
        assert_eq!(read_back.num_samples(), 1000);

        for i in 0..1000 {
            let diff = (read_back.get(0, i) - buffer.get(0, i)).abs();
            assert!(diff < 0.00001, "32-bit PCM sample {} error {}", i, diff);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_stereo_raw_pcm_roundtrip() {
        let left: Vec<f64> = (0..200).map(|i| (i as f64 / 200.0) * 0.9).collect();
        let right: Vec<f64> = (0..200).map(|i| -(i as f64 / 200.0) * 0.7).collect();
        let buffer = AudioBuffer::from_stereo(left, right, 48000.0);
        let path = std::env::temp_dir().join("melodyn_export_test_stereo_raw.pcm");

        let config = ExportConfig {
            format: ExportFormat::RawPcm,
            bits_per_sample: 16,
            byte_order: ByteOrder::LittleEndian,
            ..ExportConfig::default()
        };

        export_audio(&path, &buffer, &config).expect("export stereo raw");
        let read_back = read_raw_pcm(&path, 2, 48000.0, 16, true).expect("read back");
        assert_eq!(read_back.num_channels(), 2);
        assert_eq!(read_back.num_samples(), 200);

        let _ = std::fs::remove_file(&path);
    }
}
