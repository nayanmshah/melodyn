# Melodyn

Open-source audio pitch correction engine in Rust, reimplementing core DSP algorithms from polyphonic pitch detection and correction.

## Build

```bash
# Build all crates
cargo build

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run CLI
cargo run -p melodyn-cli -- analyze input.wav --preset melodic
cargo run -p melodyn-cli -- info input.wav
cargo run -p melodyn-cli -- synth input.wav --output output.wav --pitch-shift 2.0
```

## Architecture

### Crate structure

```
melodyn/
  crates/
    melodyn-core/     Core DSP: FFT, spectral analysis, pitch detection, energy claims
    melodyn-synth/    LSS resynthesis: sinusoidal synthesis, formants, overlap-add
    melodyn-io/       Audio file I/O: WAV reading/writing, buffer management
    melodyn-plugin/   Plugin interface skeleton (future ARA/VST3)
    melodyn-cli/      Command-line tool for analysis and resynthesis
```

### Core DSP pipeline (melodyn-core)

1. **Multi-resolution FFT** (`fft.rs`) -- Stacked FFT at multiple window sizes (1024, 4096, 8192) for different time-frequency tradeoffs. Uses rustfft.
2. **Reassigned spectrogram** (`spectral.rs`) -- Three FFTs per frame (standard, time-ramped, freq-derivative) for sub-bin frequency and time estimation.
3. **Spectral analysis pipeline** (`analysis.rs`) -- Orchestrates stacked FFT, reassigned spectrogram, peak tracking, and pitch candidate extraction.
4. **Pitch detection** (`pitch.rs`) -- YIN autocorrelation for monophonic, Harmonic Product Spectrum for spectral domain, iterative subtraction for polyphonic.
5. **Energy claim** (`energy_claim.rs`) -- Notes compete for spectral bin energy proportionally to expected harmonic weights. Core of polyphonic separation.
6. **Decomposition** (`decomposition.rs`) -- Five-component split: Tonal, Raspy, Consonants, Sibilant, Breath.
7. **Detection presets** (`detection.rs`) -- Eight presets controlling analysis behaviour.

### Synthesis pipeline (melodyn-synth)

1. **LSS model** (`lss.rs`) -- Per-partial sinusoidal model with time-varying amplitude and phase trajectories.
2. **Renderers** (`renderer.rs`) -- Component (single partial) and element (full note) renderers with phase continuity.
3. **Formant model** (`formant.rs`) -- 18-band formant processor with preservation, shifting, and excitation synthesis.
4. **Overlap-add** (`overlap_add.rs`) -- Windowed OLA framework for analysis-modify-synthesis.
5. **Time stretch** (`time_stretch.rs`) -- Warp-function-driven non-uniform time stretching.

### Key design decisions

- All internal processing uses f64 for maximum precision
- Sample rate is configurable (default 44100 Hz)
- Audio buffers use planar (non-interleaved) layout internally
- FFT plans are cached and reused via rustfft's planner

## Dependencies

- `rustfft` -- FFT operations
- `num-complex` -- Complex number types
- `hound` -- WAV file I/O
- `clap` -- CLI argument parsing
- `rayon` -- Parallel processing
- `thiserror` -- Error types
- `log` / `env_logger` -- Logging

## Code conventions

- Use `cargo fmt` and `cargo clippy` before committing
- Each module includes unit tests in a `#[cfg(test)] mod tests` block
- Public APIs have doc comments
- Error handling uses `thiserror` for typed errors, `Result` for fallible operations
