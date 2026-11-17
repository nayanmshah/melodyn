//! Audio buffer types: interleaved/planar, various bit depths, format conversion.

// ---------------------------------------------------------------------------
// Sample format
// ---------------------------------------------------------------------------

/// Supported audio sample formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 16-bit signed integer.
    I16,
    /// 24-bit signed integer (stored in i32).
    I24,
    /// 32-bit signed integer.
    I32,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
}

impl SampleFormat {
    /// Bits per sample.
    pub fn bits(&self) -> u16 {
        match self {
            Self::I16 => 16,
            Self::I24 => 24,
            Self::I32 => 32,
            Self::F32 => 32,
            Self::F64 => 64,
        }
    }

    /// Whether this is a floating-point format.
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

// ---------------------------------------------------------------------------
// Audio buffer
// ---------------------------------------------------------------------------

/// Audio buffer storing samples in planar (non-interleaved) f64 format.
///
/// Internal storage is always f64 for maximum precision during DSP.
/// Conversion to/from other formats happens at I/O boundaries.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Per-channel sample data. channels[c][s] = channel c, sample s.
    channels: Vec<Vec<f64>>,
    /// Number of channels.
    num_channels: usize,
    /// Number of samples per channel.
    num_samples: usize,
    /// Sample rate in Hz.
    sample_rate: f64,
}

impl AudioBuffer {
    /// Create a new empty audio buffer.
    pub fn new(num_channels: usize, num_samples: usize, sample_rate: f64) -> Self {
        Self {
            channels: vec![vec![0.0; num_samples]; num_channels],
            num_channels,
            num_samples,
            sample_rate,
        }
    }

    /// Create a mono buffer from a Vec of samples.
    pub fn from_mono(samples: Vec<f64>, sample_rate: f64) -> Self {
        let num_samples = samples.len();
        Self {
            channels: vec![samples],
            num_channels: 1,
            num_samples,
            sample_rate,
        }
    }

    /// Create a stereo buffer from two Vecs of samples.
    pub fn from_stereo(left: Vec<f64>, right: Vec<f64>, sample_rate: f64) -> Self {
        assert_eq!(
            left.len(),
            right.len(),
            "Stereo channels must be same length"
        );
        let num_samples = left.len();
        Self {
            channels: vec![left, right],
            num_channels: 2,
            num_samples,
            sample_rate,
        }
    }

    /// Create from interleaved samples (e.g., [L0, R0, L1, R1, ...]).
    pub fn from_interleaved(interleaved: &[f64], num_channels: usize, sample_rate: f64) -> Self {
        let num_samples = interleaved.len() / num_channels;
        let mut channels = vec![vec![0.0; num_samples]; num_channels];

        for (i, &sample) in interleaved.iter().enumerate() {
            let ch = i % num_channels;
            let s = i / num_channels;
            if s < num_samples {
                channels[ch][s] = sample;
            }
        }

        Self {
            channels,
            num_channels,
            num_samples,
            sample_rate,
        }
    }

    /// Create from interleaved i16 samples.
    pub fn from_interleaved_i16(
        interleaved: &[i16],
        num_channels: usize,
        sample_rate: f64,
    ) -> Self {
        let f64_samples: Vec<f64> = interleaved.iter().map(|&s| s as f64 / 32768.0).collect();
        Self::from_interleaved(&f64_samples, num_channels, sample_rate)
    }

    /// Create from interleaved i32 samples (24-bit in 32-bit container).
    pub fn from_interleaved_i24(
        interleaved: &[i32],
        num_channels: usize,
        sample_rate: f64,
    ) -> Self {
        let f64_samples: Vec<f64> = interleaved.iter().map(|&s| s as f64 / 8388608.0).collect();
        Self::from_interleaved(&f64_samples, num_channels, sample_rate)
    }

    /// Create from interleaved f32 samples.
    pub fn from_interleaved_f32(
        interleaved: &[f32],
        num_channels: usize,
        sample_rate: f64,
    ) -> Self {
        let f64_samples: Vec<f64> = interleaved.iter().map(|&s| s as f64).collect();
        Self::from_interleaved(&f64_samples, num_channels, sample_rate)
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Number of samples per channel.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
    }

    /// Whether this is a mono buffer.
    pub fn is_mono(&self) -> bool {
        self.num_channels == 1
    }

    /// Whether this is a stereo buffer.
    pub fn is_stereo(&self) -> bool {
        self.num_channels == 2
    }

    /// Get a reference to a channel's samples.
    pub fn channel(&self, ch: usize) -> &[f64] {
        &self.channels[ch]
    }

    /// Get a mutable reference to a channel's samples.
    pub fn channel_mut(&mut self, ch: usize) -> &mut [f64] {
        &mut self.channels[ch]
    }

    /// Get a sample value.
    pub fn get(&self, ch: usize, sample: usize) -> f64 {
        self.channels[ch][sample]
    }

    /// Set a sample value.
    pub fn set(&mut self, ch: usize, sample: usize, value: f64) {
        self.channels[ch][sample] = value;
    }

    /// Convert to mono by averaging all channels.
    pub fn to_mono(&self) -> Vec<f64> {
        if self.num_channels == 1 {
            return self.channels[0].clone();
        }

        let scale = 1.0 / self.num_channels as f64;
        (0..self.num_samples)
            .map(|s| self.channels.iter().map(|ch| ch[s]).sum::<f64>() * scale)
            .collect()
    }

    /// Convert to interleaved f64 samples.
    pub fn to_interleaved(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.num_channels * self.num_samples];
        for s in 0..self.num_samples {
            for ch in 0..self.num_channels {
                out[s * self.num_channels + ch] = self.channels[ch][s];
            }
        }
        out
    }

    /// Convert to interleaved i16 samples (with clipping).
    pub fn to_interleaved_i16(&self) -> Vec<i16> {
        let mut out = vec![0i16; self.num_channels * self.num_samples];
        for s in 0..self.num_samples {
            for ch in 0..self.num_channels {
                let val = self.channels[ch][s].clamp(-1.0, 1.0);
                out[s * self.num_channels + ch] = (val * 32767.0) as i16;
            }
        }
        out
    }

    /// Convert to interleaved f32 samples.
    pub fn to_interleaved_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.num_channels * self.num_samples];
        for s in 0..self.num_samples {
            for ch in 0..self.num_channels {
                out[s * self.num_channels + ch] = self.channels[ch][s] as f32;
            }
        }
        out
    }

    /// Peak amplitude across all channels.
    pub fn peak_amplitude(&self) -> f64 {
        self.channels
            .iter()
            .flat_map(|ch| ch.iter())
            .map(|s| s.abs())
            .fold(0.0f64, f64::max)
    }

    /// RMS amplitude across all channels.
    pub fn rms_amplitude(&self) -> f64 {
        let total_samples = self.num_channels * self.num_samples;
        if total_samples == 0 {
            return 0.0;
        }
        let sum_sq: f64 = self
            .channels
            .iter()
            .flat_map(|ch| ch.iter())
            .map(|s| s * s)
            .sum();
        (sum_sq / total_samples as f64).sqrt()
    }

    /// Normalize the buffer so peak amplitude = target (default 1.0).
    pub fn normalize(&mut self, target: f64) {
        let peak = self.peak_amplitude();
        if peak < 1e-10 {
            return;
        }
        let gain = target / peak;
        for ch in &mut self.channels {
            for s in ch.iter_mut() {
                *s *= gain;
            }
        }
    }

    /// Apply gain to all samples.
    pub fn apply_gain(&mut self, gain: f64) {
        for ch in &mut self.channels {
            for s in ch.iter_mut() {
                *s *= gain;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buf = AudioBuffer::new(2, 1024, 44100.0);
        assert_eq!(buf.num_channels(), 2);
        assert_eq!(buf.num_samples(), 1024);
        assert!((buf.sample_rate() - 44100.0).abs() < 1e-10);
        assert!(buf.is_stereo());
    }

    #[test]
    fn test_mono_buffer() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let buf = AudioBuffer::from_mono(samples.clone(), 44100.0);
        assert!(buf.is_mono());
        assert_eq!(buf.channel(0), &samples[..]);
    }

    #[test]
    fn test_interleaved_roundtrip() {
        let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = AudioBuffer::from_interleaved(&interleaved, 2, 44100.0);

        assert_eq!(buf.channel(0), &[1.0, 3.0, 5.0]);
        assert_eq!(buf.channel(1), &[2.0, 4.0, 6.0]);

        let roundtrip = buf.to_interleaved();
        assert_eq!(roundtrip, interleaved);
    }

    #[test]
    fn test_to_mono() {
        let left = vec![1.0, 0.0, -1.0];
        let right = vec![0.0, 1.0, -1.0];
        let buf = AudioBuffer::from_stereo(left, right, 44100.0);

        let mono = buf.to_mono();
        assert!((mono[0] - 0.5).abs() < 1e-10);
        assert!((mono[1] - 0.5).abs() < 1e-10);
        assert!((mono[2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_i16_conversion() {
        let i16_samples: Vec<i16> = vec![0, 16384, -16384, 32767];
        let buf = AudioBuffer::from_interleaved_i16(&i16_samples, 1, 44100.0);

        assert!(buf.channel(0)[0].abs() < 0.001);
        assert!((buf.channel(0)[1] - 0.5).abs() < 0.01);
        assert!((buf.channel(0)[2] - (-0.5)).abs() < 0.01);

        let back = buf.to_interleaved_i16();
        assert_eq!(back[0], 0);
    }

    #[test]
    fn test_normalize() {
        let mut buf = AudioBuffer::from_mono(vec![0.5, -0.3, 0.1], 44100.0);
        buf.normalize(1.0);
        assert!((buf.peak_amplitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_duration() {
        let buf = AudioBuffer::new(1, 44100, 44100.0);
        assert!((buf.duration() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rms_amplitude() {
        // DC signal of 1.0 should have RMS of 1.0
        let buf = AudioBuffer::from_mono(vec![1.0; 100], 44100.0);
        assert!((buf.rms_amplitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_format() {
        assert_eq!(SampleFormat::I16.bits(), 16);
        assert_eq!(SampleFormat::F32.bits(), 32);
        assert!(SampleFormat::F64.is_float());
        assert!(!SampleFormat::I16.is_float());
    }
}
