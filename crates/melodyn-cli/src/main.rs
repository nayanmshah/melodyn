//! Melodyn CLI -- Command-line tool for audio analysis and pitch detection.
//!
//! Usage:
//!   melodyn-cli analyze <input.wav> [--preset <preset>] [--output <output.txt>]
//!   melodyn-cli info <input.wav>
//!   melodyn-cli synth <input.wav> --pitch-shift <semitones> --output <output.wav>

use clap::{Parser, Subcommand};
use melodyn_core::{
    AnalysisConfig, AudioDecomposer, DecompositionComponent, DecompositionConfig, DetectionPreset,
    SpectralAnalyzer,
};
use melodyn_io::{read_wav, wav_info, write_wav, AudioBuffer, WavWriteConfig};
use melodyn_synth::{LssCorrelator, LssGenerator, LssGeneratorConfig};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "melodyn-cli")]
#[command(about = "Melodyn audio analysis and pitch correction engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze a WAV file and detect pitches.
    Analyze {
        /// Input WAV file path.
        input: String,

        /// Detection preset (automatic, melodic, polyphonic, percussive, etc.).
        #[arg(short, long, default_value = "automatic")]
        preset: String,

        /// Output file for analysis results (text format).
        #[arg(short, long)]
        output: Option<String>,

        /// Show decomposition analysis.
        #[arg(long)]
        decompose: bool,
    },

    /// Show information about a WAV file.
    Info {
        /// Input WAV file path.
        input: String,
    },

    /// Resynthesize audio with optional pitch shift.
    Synth {
        /// Input WAV file path.
        input: String,

        /// Output WAV file path.
        #[arg(short, long)]
        output: String,

        /// Pitch shift in semitones.
        #[arg(long, default_value = "0.0")]
        pitch_shift: f64,

        /// Time stretch factor (1.0 = no change).
        #[arg(long, default_value = "1.0")]
        time_stretch: f64,

        /// Maximum number of harmonics in resynthesis.
        #[arg(long, default_value = "16")]
        harmonics: usize,
    },
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze {
            input,
            preset,
            output,
            decompose,
        } => cmd_analyze(&input, &preset, output.as_deref(), decompose),
        Commands::Info { input } => cmd_info(&input),
        Commands::Synth {
            input,
            output,
            pitch_shift,
            time_stretch,
            harmonics,
        } => cmd_synth(&input, &output, pitch_shift, time_stretch, harmonics),
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_info(input: &str) {
    match wav_info(input) {
        Ok(info) => {
            println!("File: {}", input);
            println!("  Channels:        {}", info.channels);
            println!("  Sample rate:     {} Hz", info.sample_rate);
            println!("  Bits per sample: {}", info.bits_per_sample);
            println!("  Samples:         {}", info.num_samples);
            println!("  Duration:        {:.3} s", info.duration);
            println!(
                "  Format:          {}",
                if info.is_float { "Float" } else { "Integer" }
            );
        }
        Err(e) => {
            eprintln!("Error reading {}: {}", input, e);
            std::process::exit(1);
        }
    }
}

fn cmd_analyze(input: &str, preset_name: &str, output: Option<&str>, decompose: bool) {
    // Read WAV
    let buffer = match read_wav(input) {
        Ok(buf) => buf,
        Err(e) => {
            eprintln!("Error reading {}: {}", input, e);
            std::process::exit(1);
        }
    };

    let sr = buffer.sample_rate();
    let mono = buffer.to_mono();

    println!(
        "Analyzing {} ({:.1}s, {} Hz, {} channels)",
        input,
        buffer.duration(),
        sr as u32,
        buffer.num_channels()
    );

    // Parse preset
    let preset = DetectionPreset::from_name(preset_name).unwrap_or_else(|| {
        eprintln!(
            "Unknown preset '{}'. Available: automatic, melodic, polyphonic, percussive, \
             melodic_adjust_attacks, melodic_no_sibilants, melodic_percussive, polyphonic_unpitched",
            preset_name
        );
        std::process::exit(1);
    });

    println!("Using preset: {}", preset);

    // Run analysis
    let config = AnalysisConfig::from_preset(preset, sr);
    let analyzer = SpectralAnalyzer::new(config);
    let result = analyzer.analyze(&mono);

    // Print results
    println!("\n--- Analysis Results ---");
    println!("Spectral frames: {}", result.spectral_slices.len());
    println!("Notes detected:  {}", result.notes.len());
    println!();

    if result.notes.is_empty() {
        println!("No notes detected.");
    } else {
        println!(
            "{:<6} {:<8} {:<10} {:<10} {:<10} {:<8}",
            "#", "Note", "Freq (Hz)", "Start (s)", "Dur (s)", "Conf"
        );
        println!("{}", "-".repeat(60));

        for (i, note) in result.notes.iter().enumerate() {
            println!(
                "{:<6} {:<8} {:<10.1} {:<10.3} {:<10.3} {:<8.2}",
                i + 1,
                note.note_name(),
                note.frequency,
                note.start_time,
                note.duration,
                note.confidence,
            );
        }
    }

    // Optional decomposition
    if decompose && !result.spectral_slices.is_empty() {
        println!("\n--- Decomposition Analysis ---");
        let decomposer = AudioDecomposer::new(DecompositionConfig {
            sample_rate: sr,
            ..Default::default()
        });
        let decomp = decomposer.decompose(&result.spectral_slices);

        // Average energy per component
        let mut avg_energy = [0.0f64; 5];
        for frame in &decomp.frames {
            for c in 0..5 {
                avg_energy[c] += frame.component_energy[c];
            }
        }
        let n = decomp.frames.len() as f64;
        for avg in &mut avg_energy {
            *avg /= n;
        }

        let total: f64 = avg_energy.iter().sum();
        println!("\nComponent energy distribution:");
        for comp in &DecompositionComponent::ALL {
            let e = avg_energy[*comp as usize];
            let pct = if total > 0.0 { e / total * 100.0 } else { 0.0 };
            println!("  {:<12} {:.1}%", comp.name(), pct);
        }
    }

    // Write to file if requested
    if let Some(out_path) = output {
        let mut lines = Vec::new();
        lines.push(format!("# Melodyn Analysis: {}", input));
        lines.push(format!(
            "# Duration: {:.3}s, Sample Rate: {}, Preset: {}",
            buffer.duration(),
            sr as u32,
            preset
        ));
        lines.push(String::new());
        lines.push("note\tfrequency\tstart\tduration\tconfidence\tmidi".to_string());
        for note in &result.notes {
            lines.push(format!(
                "{}\t{:.2}\t{:.4}\t{:.4}\t{:.3}\t{:.1}",
                note.note_name(),
                note.frequency,
                note.start_time,
                note.duration,
                note.confidence,
                note.midi_note,
            ));
        }

        match std::fs::write(out_path, lines.join("\n") + "\n") {
            Ok(()) => println!("\nResults written to {}", out_path),
            Err(e) => eprintln!("Error writing {}: {}", out_path, e),
        }
    }
}

fn cmd_synth(input: &str, output: &str, pitch_shift: f64, time_stretch: f64, max_harmonics: usize) {
    // Read WAV
    let buffer = match read_wav(input) {
        Ok(buf) => buf,
        Err(e) => {
            eprintln!("Error reading {}: {}", input, e);
            std::process::exit(1);
        }
    };

    let sr = buffer.sample_rate();
    let mono = buffer.to_mono();

    println!(
        "Analyzing {} ({:.1}s, {} Hz)...",
        input,
        buffer.duration(),
        sr as u32
    );

    // Run analysis
    let config = AnalysisConfig::from_preset(DetectionPreset::Melodic, sr);
    let analyzer = SpectralAnalyzer::new(config);
    let result = analyzer.analyze(&mono);

    println!("Detected {} notes", result.notes.len());

    if result.notes.is_empty() {
        eprintln!("No notes detected -- cannot resynthesize.");
        std::process::exit(1);
    }

    // Build LSS models
    let hop_size = 256;
    let correlator = LssCorrelator::new(max_harmonics, hop_size, sr);
    let models: Vec<_> = result
        .notes
        .iter()
        .enumerate()
        .map(|(i, note)| correlator.build_model_simple(note, i))
        .collect();

    // Synthesize
    println!(
        "Resynthesizing with {} harmonics, pitch shift = {:.1} semitones...",
        max_harmonics, pitch_shift
    );

    let gen_config = LssGeneratorConfig {
        sample_rate: sr,
        hop_size,
        max_harmonics,
        pitch_shift_semitones: pitch_shift,
        time_stretch_factor: time_stretch,
    };
    let generator = LssGenerator::new(gen_config);
    let synth_audio = generator.synthesize_mix(&models);

    // Write output
    let out_buffer = AudioBuffer::from_mono(synth_audio, sr);
    let write_config = WavWriteConfig {
        bits_per_sample: 16,
        float_format: false,
    };

    match write_wav(output, &out_buffer, &write_config) {
        Ok(()) => println!("Output written to {}", output),
        Err(e) => {
            eprintln!("Error writing {}: {}", output, e);
            std::process::exit(1);
        }
    }
}
