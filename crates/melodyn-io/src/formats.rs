//! Multi-format audio reader using `symphonia` for decoding.
//!
//! Supports automatic format detection and decoding of WAV, FLAC, MP3,
//! OGG/Vorbis, and AIFF files into the planar f64 `AudioBuffer`.

use crate::audio_buffer::AudioBuffer;
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the multi-format audio reader.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("Failed to open file: {0}")]
    OpenError(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("No audio track found in file")]
    NoAudioTrack,
    #[error("Decode error: {0}")]
    DecodeError(String),
    #[error("Probe error: {0}")]
    ProbeError(String),
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Audio format enumeration
// ---------------------------------------------------------------------------

/// Known audio file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Flac,
    Mp3,
    OggVorbis,
    Aiff,
    Unknown,
}

impl AudioFormat {
    /// Return a human-readable name for the format.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wav => "WAV",
            Self::Flac => "FLAC",
            Self::Mp3 => "MP3",
            Self::OggVorbis => "OGG/Vorbis",
            Self::Aiff => "AIFF",
            Self::Unknown => "Unknown",
        }
    }

    /// Detect the format from a file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "wav" | "wave" => Self::Wav,
            "flac" => Self::Flac,
            "mp3" => Self::Mp3,
            "ogg" | "oga" => Self::OggVorbis,
            "aif" | "aiff" | "aifc" => Self::Aiff,
            _ => Self::Unknown,
        }
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Audio file info / metadata
// ---------------------------------------------------------------------------

/// Metadata and technical information about an audio file.
#[derive(Debug, Clone)]
pub struct AudioFileInfo {
    /// Detected file format.
    pub format: AudioFormat,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Duration in seconds.
    pub duration: f64,
    /// Total number of samples per channel.
    pub num_samples: u64,
    /// Bits per sample (if applicable; 0 for compressed formats like MP3).
    pub bits_per_sample: u32,
    /// Title tag (if present).
    pub title: Option<String>,
    /// Artist tag (if present).
    pub artist: Option<String>,
    /// Album tag (if present).
    pub album: Option<String>,
}

// ---------------------------------------------------------------------------
// Multi-format reader
// ---------------------------------------------------------------------------

/// Read an audio file in any supported format, auto-detecting the format.
///
/// Supported formats: WAV, FLAC, MP3, OGG/Vorbis, AIFF.
///
/// Returns a planar f64 `AudioBuffer` suitable for DSP processing.
pub fn read_audio_file<P: AsRef<Path>>(path: P) -> Result<AudioBuffer, FormatError> {
    let path = path.as_ref();

    let file = File::open(path)
        .map_err(|e| FormatError::OpenError(format!("{}: {}", path.display(), e)))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Provide a hint from the file extension.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| FormatError::ProbeError(format!("{}: {}", path.display(), e)))?;

    let mut format_reader = probed.format;

    // Find the first audio track.
    let track = format_reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or(FormatError::NoAudioTrack)?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| FormatError::DecodeError("Missing sample rate".to_string()))?;

    let channels = codec_params.channels.map(|ch| ch.count()).unwrap_or(1);

    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &decoder_opts)
        .map_err(|e| FormatError::DecodeError(format!("Failed to create decoder: {}", e)))?;

    // Decode all packets into interleaved f32 samples, then convert to f64.
    let mut all_samples: Vec<f64> = Vec::new();
    let num_channels = channels;

    loop {
        let packet = match format_reader.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                // Some formats signal end-of-stream this way.
                break;
            }
            Err(e) => {
                return Err(FormatError::DecodeError(format!(
                    "Packet read error: {}",
                    e
                )));
            }
        };

        // Skip packets for other tracks.
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(e)) => {
                log::warn!("Decode error on packet, skipping: {}", e);
                continue;
            }
            Err(e) => {
                return Err(FormatError::DecodeError(format!("Decode error: {}", e)));
            }
        };

        // Convert to interleaved f32 via SampleBuffer, then to f64.
        let spec = *decoded.spec();
        let num_frames = decoded.frames();
        if num_frames == 0 {
            continue;
        }

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        for &s in sample_buf.samples() {
            all_samples.push(s as f64);
        }
    }

    let num_samples_per_channel = all_samples.len().checked_div(num_channels).unwrap_or(0);

    if num_samples_per_channel == 0 {
        return Ok(AudioBuffer::new(num_channels, 0, sample_rate as f64));
    }

    Ok(AudioBuffer::from_interleaved(
        &all_samples,
        num_channels,
        sample_rate as f64,
    ))
}

/// Get information about an audio file without decoding the full audio.
///
/// Reads headers and metadata tags to populate `AudioFileInfo`.
pub fn audio_file_info<P: AsRef<Path>>(path: P) -> Result<AudioFileInfo, FormatError> {
    let path = path.as_ref();

    let format_from_ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(AudioFormat::from_extension)
        .unwrap_or(AudioFormat::Unknown);

    let file = File::open(path)
        .map_err(|e| FormatError::OpenError(format!("{}: {}", path.display(), e)))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions {
        limit_metadata_bytes: symphonia::core::meta::Limit::Default,
        limit_visual_bytes: symphonia::core::meta::Limit::None,
    };

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| FormatError::ProbeError(format!("{}: {}", path.display(), e)))?;

    let mut format_reader = probed.format;

    let track = format_reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or(FormatError::NoAudioTrack)?;

    let codec_params = &track.codec_params;

    let sample_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|ch| ch.count()).unwrap_or(1) as u16;
    let bits_per_sample = codec_params.bits_per_sample.unwrap_or(0);

    let n_frames = codec_params.n_frames.unwrap_or(0);
    let duration = if sample_rate > 0 {
        n_frames as f64 / sample_rate as f64
    } else {
        0.0
    };

    // Extract metadata tags.
    let mut title = None;
    let mut artist = None;
    let mut album = None;

    if let Some(metadata) = format_reader.metadata().current() {
        for tag in metadata.tags() {
            if let Some(std_key) = tag.std_key {
                match std_key {
                    symphonia::core::meta::StandardTagKey::TrackTitle => {
                        title = Some(tag.value.to_string());
                    }
                    symphonia::core::meta::StandardTagKey::Artist => {
                        artist = Some(tag.value.to_string());
                    }
                    symphonia::core::meta::StandardTagKey::Album => {
                        album = Some(tag.value.to_string());
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(AudioFileInfo {
        format: format_from_ext,
        sample_rate,
        channels,
        duration,
        num_samples: n_frames,
        bits_per_sample,
        title,
        artist,
        album,
    })
}

/// Detect the audio format of a file from its extension.
pub fn detect_format<P: AsRef<Path>>(path: P) -> AudioFormat {
    path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .map(AudioFormat::from_extension)
        .unwrap_or(AudioFormat::Unknown)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::{write_wav, WavWriteConfig};

    /// Helper: create a test WAV file and return its path.
    fn create_test_wav(name: &str) -> std::path::PathBuf {
        let samples: Vec<f64> = (0..2000)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin() * 0.8)
            .collect();
        let buffer = AudioBuffer::from_mono(samples, 44100.0);
        let path = std::env::temp_dir().join(format!("melodyn_formats_test_{}.wav", name));
        write_wav(
            &path,
            &buffer,
            &WavWriteConfig {
                bits_per_sample: 16,
                float_format: false,
            },
        )
        .expect("write test wav");
        path
    }

    #[test]
    fn test_read_wav_via_multi_format_reader() {
        let path = create_test_wav("multi_read");

        // Read via the new multi-format reader.
        let buffer = read_audio_file(&path).expect("read_audio_file should succeed");

        assert_eq!(buffer.num_channels(), 1);
        assert_eq!(buffer.num_samples(), 2000);
        assert!((buffer.sample_rate() - 44100.0).abs() < 1.0);

        // Compare against direct WAV read.
        let direct = crate::wav::read_wav(&path).expect("direct wav read");
        assert_eq!(buffer.num_samples(), direct.num_samples());

        // The samples should be very close (both decode the same 16-bit WAV).
        for i in 0..buffer.num_samples() {
            let diff = (buffer.get(0, i) - direct.get(0, i)).abs();
            assert!(
                diff < 0.01,
                "Sample {} differs by {} between multi-format and direct WAV readers",
                i,
                diff
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_audio_file_info_wav() {
        let path = create_test_wav("info");

        let info = audio_file_info(&path).expect("audio_file_info should succeed");

        assert_eq!(info.format, AudioFormat::Wav);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 1);
        assert!(info.bits_per_sample == 16);
        assert!(info.num_samples > 0);
        assert!(info.duration > 0.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_unsupported_format_error() {
        // Try to read a non-existent file.
        let result = read_audio_file("/tmp/nonexistent_melodyn_test.xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_format_from_extension() {
        assert_eq!(detect_format("song.wav"), AudioFormat::Wav);
        assert_eq!(detect_format("song.flac"), AudioFormat::Flac);
        assert_eq!(detect_format("song.mp3"), AudioFormat::Mp3);
        assert_eq!(detect_format("song.ogg"), AudioFormat::OggVorbis);
        assert_eq!(detect_format("song.aiff"), AudioFormat::Aiff);
        assert_eq!(detect_format("song.aif"), AudioFormat::Aiff);
        assert_eq!(detect_format("song.xyz"), AudioFormat::Unknown);
        assert_eq!(detect_format("noext"), AudioFormat::Unknown);
    }

    #[test]
    fn test_audio_format_name() {
        assert_eq!(AudioFormat::Wav.name(), "WAV");
        assert_eq!(AudioFormat::Flac.name(), "FLAC");
        assert_eq!(AudioFormat::Mp3.name(), "MP3");
        assert_eq!(AudioFormat::OggVorbis.name(), "OGG/Vorbis");
        assert_eq!(AudioFormat::Aiff.name(), "AIFF");
        assert_eq!(AudioFormat::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_audio_format_display() {
        assert_eq!(format!("{}", AudioFormat::Wav), "WAV");
        assert_eq!(format!("{}", AudioFormat::Mp3), "MP3");
    }

    #[test]
    fn test_read_float_wav_via_multi_format_reader() {
        // Create a 32-bit float WAV.
        let samples: Vec<f64> = (0..1000)
            .map(|i| (2.0 * std::f64::consts::PI * 880.0 * i as f64 / 44100.0).sin() * 0.5)
            .collect();
        let buffer = AudioBuffer::from_mono(samples.clone(), 44100.0);
        let path = std::env::temp_dir().join("melodyn_formats_test_float32.wav");
        write_wav(
            &path,
            &buffer,
            &WavWriteConfig {
                bits_per_sample: 32,
                float_format: true,
            },
        )
        .expect("write float wav");

        let read_back = read_audio_file(&path).expect("read float wav via multi-format");
        assert_eq!(read_back.num_channels(), 1);
        assert_eq!(read_back.num_samples(), 1000);

        for i in 0..read_back.num_samples() {
            let diff = (read_back.get(0, i) - samples[i]).abs();
            assert!(diff < 0.001, "Float WAV sample {} differs by {}", i, diff);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_stereo_wav_via_multi_format_reader() {
        let left: Vec<f64> = (0..500).map(|i| (i as f64 / 500.0) * 0.8).collect();
        let right: Vec<f64> = (0..500).map(|i| -(i as f64 / 500.0) * 0.6).collect();
        let buffer = AudioBuffer::from_stereo(left.clone(), right.clone(), 48000.0);
        let path = std::env::temp_dir().join("melodyn_formats_test_stereo.wav");
        write_wav(
            &path,
            &buffer,
            &WavWriteConfig {
                bits_per_sample: 16,
                float_format: false,
            },
        )
        .expect("write stereo wav");

        let read_back = read_audio_file(&path).expect("read stereo wav via multi-format");
        assert_eq!(read_back.num_channels(), 2);
        assert_eq!(read_back.num_samples(), 500);
        assert!((read_back.sample_rate() - 48000.0).abs() < 1.0);

        let _ = std::fs::remove_file(&path);
    }
}
