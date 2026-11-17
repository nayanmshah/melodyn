//! Melodyn Desktop Application
//!
//! Entry point for the eframe-based Melodyn UI. Launches the native window
//! with the full editor experience: piano roll, transport, inspector, and
//! waveform overview.

fn main() -> eframe::Result {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Melodyn",
        options,
        Box::new(|cc| Ok(Box::new(melodyn_ui::MelodynApp::new(cc)))),
    )
}
