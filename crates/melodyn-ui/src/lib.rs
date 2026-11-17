//! Melodyn UI -- Desktop application for audio pitch correction.
//!
//! This crate provides a full egui/eframe-based desktop UI for the Melodyn
//! audio analysis and editing engine. The interface is modelled after
//! Celemony Melodyne's editor layout:
//!
//! - **app**: Main application struct, state management, file I/O, undo/redo
//! - **editor**: Piano roll / note editor with custom Painter rendering
//! - **tools**: Tool system (Pitch, Time, Formant, Amplitude, Stretch, Separation)
//! - **transport**: Playback transport bar with position display and volume
//! - **inspector**: Property inspector for selected notes
//! - **waveform**: Full audio waveform overview with visible region indicator
//! - **timeline**: Time/beat ruler above the editor
//!
//! # Layout
//!
//! ```text
//! +---------------------------------------------------+
//! | Menu Bar  (File, Edit, View, Analysis)             |
//! +---------------------------------------------------+
//! | Toolbar   (Tool buttons, Snap mode, Zoom)          |
//! +--------------------------------------+------------+
//! | Timeline Ruler                       |            |
//! +------+-------------------------------+ Inspector  |
//! | Piano|                               |   Panel    |
//! | Keys | Note Editor (piano roll)      |            |
//! |      |                               |            |
//! +------+-------------------------------+------------+
//! | Waveform Overview                                  |
//! +---------------------------------------------------+
//! | Transport (Play/Stop, Position, Tempo, Loop, Vol)  |
//! +---------------------------------------------------+
//! | Status Bar                                         |
//! +---------------------------------------------------+
//! ```

pub mod app;
pub mod editor;
pub mod inspector;
pub mod timeline;
pub mod tools;
pub mod transport;
pub mod waveform;

pub use app::MelodynApp;
