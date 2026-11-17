//! Melodyn I/O -- Audio file reading and writing, buffer management, serialization,
//! and project file persistence.
//!
//! - **audio_buffer**: Planar/interleaved audio buffers with format conversion
//! - **wav**: WAV file reading and writing via hound
//! - **archive**: Binary key-value, XML, and file archive container serialization
//! - **project**: `.mpd` project files and `.mdd` detection data persistence

pub mod archive;
pub mod audio_buffer;
pub mod midi;
pub mod project;
pub mod wav;

pub use archive::{
    ArchiveError, ArchiveValue, BinaryKeyValueArchive, CompressionMethod, FileArchive,
    XmlKeyValueArchive,
};
pub use audio_buffer::{AudioBuffer, SampleFormat};
pub use project::{
    DetectionData, ProjectError, ProjectFile, ProjectMetadata, SerializedComposition,
    SerializedElement, SerializedNote, SerializedTrack,
};
pub use wav::{read_wav, wav_info, write_mono_wav, write_wav, WavError, WavInfo, WavWriteConfig};
