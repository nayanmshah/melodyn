//! Melodyn I/O -- Audio file reading and writing, buffer management.
//!
//! - **audio_buffer**: Planar/interleaved audio buffers with format conversion
//! - **wav**: WAV file reading and writing via hound

pub mod audio_buffer;
pub mod midi;
pub mod wav;

pub use audio_buffer::{AudioBuffer, SampleFormat};
pub use wav::{read_wav, wav_info, write_mono_wav, write_wav, WavError, WavInfo, WavWriteConfig};
