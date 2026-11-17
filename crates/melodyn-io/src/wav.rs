//! WAV file reading and writing using the `hound` crate.

use crate::audio_buffer::AudioBuffer;
use hound::{SampleFormat as HoundSampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from WAV I/O operations.
#[derive(Debug, Error)]
pub enum WavError {
    #[error("Failed to open WAV file: {0}")]
    OpenError(String),
    #[error("Failed to write WAV file: {0}")]
    WriteError(String),
    #[error("Unsupported WAV format: {0}")]
    UnsupportedFormat(String),
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Hound error: {0}")]
    HoundError(#[from] hound::Error),
}

// ---------------------------------------------------------------------------
// WAV info
// ---------------------------------------------------------------------------

/// Information about a WAV file.
#[derive(Debug, Clone)]
pub struct WavInfo {
    /// Number of channels.
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bits per sample.
    pub bits_per_sample: u16,
    /// Total number of samples (per channel).
    pub num_samples: usize,
    /// Duration in seconds.
    pub duration: f64,
    /// Sample format (integer or float).
    pub is_float: bool,
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Read a WAV file and return an AudioBuffer.
pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer, WavError> {
    let reader = WavReader::open(path.as_ref())
        .map_err(|e| WavError::OpenError(format!("{}: {}", path.as_ref().display(), e)))?;

    let spec = reader.spec();
    let num_channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as f64;
    let bits_per_sample = spec.bits_per_sample;

    match spec.sample_format {
        HoundSampleFormat::Int => read_wav_int(reader, num_channels, sample_rate, bits_per_sample),
        HoundSampleFormat::Float => read_wav_float(reader, num_channels, sample_rate),
    }
}

/// Read integer-format WAV.
fn read_wav_int(
    mut reader: WavReader<std::io::BufReader<std::fs::File>>,
    num_channels: usize,
    sample_rate: f64,
    bits_per_sample: u16,
) -> Result<AudioBuffer, WavError> {
    let scale = match bits_per_sample {
        8 => 128.0,
        16 => 32768.0,
        24 => 8388608.0,
        32 => 2147483648.0,
        _ => {
            return Err(WavError::UnsupportedFormat(format!(
                "{}-bit integer",
                bits_per_sample
            )))
        }
    };

    let samples: Vec<f64> = reader
        .samples::<i32>()
        .map(|s| s.map(|v| v as f64 / scale))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(AudioBuffer::from_interleaved(
        &samples,
        num_channels,
        sample_rate,
    ))
}

/// Read float-format WAV.
fn read_wav_float(
    mut reader: WavReader<std::io::BufReader<std::fs::File>>,
    num_channels: usize,
    sample_rate: f64,
) -> Result<AudioBuffer, WavError> {
    let samples: Vec<f64> = reader
        .samples::<f32>()
        .map(|s| s.map(|v| v as f64))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(AudioBuffer::from_interleaved(
        &samples,
        num_channels,
        sample_rate,
    ))
}

/// Get information about a WAV file without reading all samples.
pub fn wav_info<P: AsRef<Path>>(path: P) -> Result<WavInfo, WavError> {
    let reader = WavReader::open(path.as_ref())
        .map_err(|e| WavError::OpenError(format!("{}: {}", path.as_ref().display(), e)))?;

    let spec = reader.spec();
    let total_samples = reader.len() as usize;
    let num_channels = spec.channels as usize;
    let samples_per_channel = if num_channels > 0 {
        total_samples / num_channels
    } else {
        0
    };
    let duration = samples_per_channel as f64 / spec.sample_rate as f64;

    Ok(WavInfo {
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: spec.bits_per_sample,
        num_samples: samples_per_channel,
        duration,
        is_float: spec.sample_format == HoundSampleFormat::Float,
    })
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write configuration for WAV output.
#[derive(Debug, Clone)]
pub struct WavWriteConfig {
    /// Bits per sample (16, 24, or 32).
    pub bits_per_sample: u16,
    /// Whether to write as float format.
    pub float_format: bool,
}

impl Default for WavWriteConfig {
    fn default() -> Self {
        Self {
            bits_per_sample: 16,
            float_format: false,
        }
    }
}

/// Write an AudioBuffer to a WAV file.
pub fn write_wav<P: AsRef<Path>>(
    path: P,
    buffer: &AudioBuffer,
    config: &WavWriteConfig,
) -> Result<(), WavError> {
    let spec = WavSpec {
        channels: buffer.num_channels() as u16,
        sample_rate: buffer.sample_rate() as u32,
        bits_per_sample: config.bits_per_sample,
        sample_format: if config.float_format {
            HoundSampleFormat::Float
        } else {
            HoundSampleFormat::Int
        },
    };

    let mut writer = WavWriter::create(path.as_ref(), spec)
        .map_err(|e| WavError::WriteError(format!("{}: {}", path.as_ref().display(), e)))?;

    if config.float_format {
        write_wav_float(&mut writer, buffer)?;
    } else {
        write_wav_int(&mut writer, buffer, config.bits_per_sample)?;
    }

    writer.finalize()?;
    Ok(())
}

/// Write integer samples.
fn write_wav_int(
    writer: &mut WavWriter<std::io::BufWriter<std::fs::File>>,
    buffer: &AudioBuffer,
    bits_per_sample: u16,
) -> Result<(), WavError> {
    let scale = match bits_per_sample {
        8 => 127.0,
        16 => 32767.0,
        24 => 8388607.0,
        32 => 2147483647.0,
        _ => {
            return Err(WavError::UnsupportedFormat(format!(
                "{}-bit integer",
                bits_per_sample
            )))
        }
    };

    let max_val = scale as i32;
    let min_val = -(max_val + 1);

    for s in 0..buffer.num_samples() {
        for ch in 0..buffer.num_channels() {
            let val = (buffer.get(ch, s) * scale) as i32;
            let clamped = val.clamp(min_val, max_val);
            writer.write_sample(clamped)?;
        }
    }

    Ok(())
}

/// Write float samples.
fn write_wav_float(
    writer: &mut WavWriter<std::io::BufWriter<std::fs::File>>,
    buffer: &AudioBuffer,
) -> Result<(), WavError> {
    for s in 0..buffer.num_samples() {
        for ch in 0..buffer.num_channels() {
            writer.write_sample(buffer.get(ch, s) as f32)?;
        }
    }
    Ok(())
}

/// Write a mono f64 slice to a WAV file (convenience function).
pub fn write_mono_wav<P: AsRef<Path>>(
    path: P,
    samples: &[f64],
    sample_rate: u32,
) -> Result<(), WavError> {
    let buffer = AudioBuffer::from_mono(samples.to_vec(), sample_rate as f64);
    write_wav(path, &buffer, &WavWriteConfig::default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_write_config_default() {
        let config = WavWriteConfig::default();
        assert_eq!(config.bits_per_sample, 16);
        assert!(!config.float_format);
    }

    // Integration tests that require file I/O would go here.
    // They would create temp files, write, read back, and compare.
    // Skipped in unit tests as they need filesystem access.

    #[test]
    fn test_wav_roundtrip() {
        let samples: Vec<f64> = (0..1000)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin() * 0.9)
            .collect();
        let buffer = AudioBuffer::from_mono(samples.clone(), 44100.0);

        // Write to a temp file
        let path = std::env::temp_dir().join("melodyn_test_roundtrip.wav");
        let config = WavWriteConfig {
            bits_per_sample: 16,
            float_format: false,
        };
        write_wav(&path, &buffer, &config).expect("Write should succeed");

        // Read it back
        let read_back = read_wav(&path).expect("Read should succeed");
        assert_eq!(read_back.num_channels(), 1);
        assert_eq!(read_back.num_samples(), 1000);
        assert!((read_back.sample_rate() - 44100.0).abs() < 1.0);

        // Check samples are approximately correct (16-bit quantization)
        for i in 0..1000 {
            let error = (read_back.get(0, i) - samples[i]).abs();
            assert!(error < 0.001, "Sample {} error {} too large", i, error);
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wav_info() {
        let samples: Vec<f64> = vec![0.0; 44100];
        let buffer = AudioBuffer::from_mono(samples, 44100.0);
        let path = std::env::temp_dir().join("melodyn_test_info.wav");
        write_wav(&path, &buffer, &WavWriteConfig::default()).expect("Write should succeed");

        let info = wav_info(&path).expect("Info should succeed");
        assert_eq!(info.channels, 1);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.num_samples, 44100);
        assert!((info.duration - 1.0).abs() < 0.01);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wav_float_roundtrip() {
        let samples: Vec<f64> = (0..500)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 44100.0).sin() * 0.5)
            .collect();
        let buffer = AudioBuffer::from_mono(samples.clone(), 44100.0);

        let path = std::env::temp_dir().join("melodyn_test_float.wav");
        let config = WavWriteConfig {
            bits_per_sample: 32,
            float_format: true,
        };
        write_wav(&path, &buffer, &config).expect("Write should succeed");

        let read_back = read_wav(&path).expect("Read should succeed");
        for i in 0..500 {
            let error = (read_back.get(0, i) - samples[i]).abs();
            assert!(error < 0.0001, "Float sample {} error {} too large", i, error);
        }

        let _ = std::fs::remove_file(&path);
    }
}
