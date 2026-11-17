//! Melodyn Plugin -- Future ARA/VST3 plugin interface.
//!
//! This crate is a skeleton for the plugin layer that will expose Melodyn's
//! analysis and synthesis engines as an ARA plugin and/or VST3 instrument.
//!
//! The architecture follows Melodyne's MUAra* class hierarchy:
//! - AraDocumentController: manages the plugin's document state
//! - AraAudioSource: represents an audio source registered with the host
//! - AraAudioModification: an edit applied to an audio source
//! - AraPlaybackRegion: a region of audio rendered to the host's timeline

pub mod ara;

/// Plugin version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Melodyn";
