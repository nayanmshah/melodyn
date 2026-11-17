//! Real-time audio engine: render callback, ring buffer, audio node graph,
//! playback controller, render cache, and performance renderer.
//!
//! Based on MUAudioRenderThread, MUAudioRenderInfo, MUAudioRingBuffer,
//! MUAudioNode, MUPlaybackController, MURenderCacheData, MUPerformanceRenderer.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Transport state
// ---------------------------------------------------------------------------

/// Playback transport state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportState {
    Stopped,
    Playing,
    Paused,
    Recording,
}

/// Transport position and loop points.
#[derive(Debug, Clone)]
pub struct TransportInfo {
    pub state: TransportState,
    /// Current playback position in seconds.
    pub position_seconds: f64,
    /// Current playback position in samples.
    pub position_samples: u64,
    /// Tempo in BPM.
    pub tempo: f64,
    /// Sample rate.
    pub sample_rate: f64,
    /// Whether looping is enabled.
    pub loop_enabled: bool,
    /// Loop start in seconds.
    pub loop_start: f64,
    /// Loop end in seconds.
    pub loop_end: f64,
}

impl Default for TransportInfo {
    fn default() -> Self {
        Self {
            state: TransportState::Stopped,
            position_seconds: 0.0,
            position_samples: 0,
            tempo: 120.0,
            sample_rate: 44100.0,
            loop_enabled: false,
            loop_start: 0.0,
            loop_end: 0.0,
        }
    }
}

impl TransportInfo {
    /// Advance position by the given number of samples, respecting loop points.
    pub fn advance(&mut self, num_samples: usize) {
        if self.state != TransportState::Playing && self.state != TransportState::Recording {
            return;
        }
        self.position_samples += num_samples as u64;
        self.position_seconds = self.position_samples as f64 / self.sample_rate;

        if self.loop_enabled && self.position_seconds >= self.loop_end && self.loop_end > self.loop_start {
            let loop_len = self.loop_end - self.loop_start;
            self.position_seconds =
                self.loop_start + (self.position_seconds - self.loop_start) % loop_len;
            self.position_samples = (self.position_seconds * self.sample_rate) as u64;
        }
    }
}

// ---------------------------------------------------------------------------
// Audio render info
// ---------------------------------------------------------------------------

/// Information passed to the render callback each cycle.
#[derive(Debug, Clone)]
pub struct AudioRenderInfo {
    /// Number of output channels.
    pub num_channels: usize,
    /// Number of samples per channel in this render call.
    pub buffer_size: usize,
    /// Sample rate.
    pub sample_rate: f64,
    /// Current transport info.
    pub transport: TransportInfo,
}

impl AudioRenderInfo {
    /// Duration of this render buffer in seconds.
    pub fn buffer_duration(&self) -> f64 {
        self.buffer_size as f64 / self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Lock-free ring buffer (SPSC)
// ---------------------------------------------------------------------------

/// Lock-free single-producer single-consumer ring buffer for audio I/O.
///
/// Uses atomic indices for thread-safe communication between the audio
/// render thread (producer) and the main thread (consumer), or vice versa.
pub struct AudioRingBuffer {
    buffer: Vec<f64>,
    capacity: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
}

impl AudioRingBuffer {
    /// Create a new ring buffer with the given capacity (in samples).
    pub fn new(capacity: usize) -> Self {
        // Use power-of-two sizing for efficient modulo via bitmask.
        let capacity = capacity.next_power_of_two();
        Self {
            buffer: vec![0.0; capacity],
            capacity,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
        }
    }

    /// Number of samples available to read.
    pub fn available_read(&self) -> usize {
        let w = self.write_pos.load(Ordering::Acquire);
        let r = self.read_pos.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    /// Number of samples that can be written.
    pub fn available_write(&self) -> usize {
        self.capacity - 1 - self.available_read()
    }

    /// Write samples into the ring buffer. Returns number of samples written.
    pub fn write(&self, data: &[f64]) -> usize {
        let available = self.available_write();
        let to_write = data.len().min(available);
        let w = self.write_pos.load(Ordering::Relaxed);

        for i in 0..to_write {
            let idx = (w + i) & (self.capacity - 1);
            // SAFETY: single producer, index is within bounds
            unsafe {
                let ptr = self.buffer.as_ptr() as *mut f64;
                *ptr.add(idx) = data[i];
            }
        }

        self.write_pos
            .store(w.wrapping_add(to_write), Ordering::Release);
        to_write
    }

    /// Read samples from the ring buffer. Returns number of samples read.
    pub fn read(&self, output: &mut [f64]) -> usize {
        let available = self.available_read();
        let to_read = output.len().min(available);
        let r = self.read_pos.load(Ordering::Relaxed);

        for i in 0..to_read {
            let idx = (r + i) & (self.capacity - 1);
            output[i] = self.buffer[idx];
        }

        self.read_pos
            .store(r.wrapping_add(to_read), Ordering::Release);
        to_read
    }

    /// Reset the ring buffer (clear all data).
    pub fn reset(&self) {
        self.write_pos.store(0, Ordering::Release);
        self.read_pos.store(0, Ordering::Release);
    }

    /// Capacity in samples.
    pub fn capacity(&self) -> usize {
        self.capacity - 1
    }
}

// ---------------------------------------------------------------------------
// Audio node graph
// ---------------------------------------------------------------------------

/// Unique identifier for audio nodes.
pub type NodeId = u64;

/// Audio processing node trait.
pub trait AudioNode: Send {
    /// Process audio: read from inputs, write to outputs.
    fn process(
        &mut self,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
        info: &AudioRenderInfo,
    );
    /// Number of input channels.
    fn num_inputs(&self) -> usize;
    /// Number of output channels.
    fn num_outputs(&self) -> usize;
    /// Reset internal state.
    fn reset(&mut self);
    /// Node name for debugging.
    fn name(&self) -> &str;
}

/// A connection between two nodes.
#[derive(Debug, Clone)]
pub struct NodeConnection {
    pub source_id: NodeId,
    pub source_channel: usize,
    pub dest_id: NodeId,
    pub dest_channel: usize,
}

/// Audio node graph with topological processing order.
pub struct AudioGraph {
    nodes: HashMap<NodeId, Box<dyn AudioNode>>,
    connections: Vec<NodeConnection>,
    processing_order: Vec<NodeId>,
    next_id: NodeId,
    buffers: HashMap<NodeId, Vec<Vec<f64>>>,
    buffer_size: usize,
}

impl AudioGraph {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            connections: Vec::new(),
            processing_order: Vec::new(),
            next_id: 1,
            buffers: HashMap::new(),
            buffer_size,
        }
    }

    /// Add a node to the graph, returning its ID.
    pub fn add_node(&mut self, node: Box<dyn AudioNode>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let num_outputs = node.num_outputs();
        self.buffers
            .insert(id, vec![vec![0.0; self.buffer_size]; num_outputs]);
        self.nodes.insert(id, node);
        self.update_processing_order();
        id
    }

    /// Remove a node from the graph.
    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.remove(&id);
        self.buffers.remove(&id);
        self.connections
            .retain(|c| c.source_id != id && c.dest_id != id);
        self.update_processing_order();
    }

    /// Connect two nodes.
    pub fn connect(
        &mut self,
        source_id: NodeId,
        source_channel: usize,
        dest_id: NodeId,
        dest_channel: usize,
    ) {
        self.connections.push(NodeConnection {
            source_id,
            source_channel,
            dest_id,
            dest_channel,
        });
        self.update_processing_order();
    }

    /// Update the topological processing order using Kahn's algorithm.
    fn update_processing_order(&mut self) {
        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        let mut in_degree: HashMap<NodeId, usize> = ids.iter().map(|&id| (id, 0)).collect();

        for conn in &self.connections {
            if let Some(deg) = in_degree.get_mut(&conn.dest_id) {
                *deg += 1;
            }
        }

        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort(); // Deterministic order

        let mut order = Vec::new();
        while let Some(id) = queue.pop() {
            order.push(id);
            for conn in &self.connections {
                if conn.source_id == id {
                    if let Some(deg) = in_degree.get_mut(&conn.dest_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(conn.dest_id);
                        }
                    }
                }
            }
        }

        self.processing_order = order;
    }

    /// Process all nodes in topological order.
    pub fn process(&mut self, info: &AudioRenderInfo) {
        let buffer_size = info.buffer_size.min(self.buffer_size);

        // Clear all buffers
        for bufs in self.buffers.values_mut() {
            for buf in bufs.iter_mut() {
                buf[..buffer_size].fill(0.0);
            }
        }

        let order = self.processing_order.clone();
        for &id in &order {
            // Gather inputs from connections
            let input_conns: Vec<NodeConnection> = self
                .connections
                .iter()
                .filter(|c| c.dest_id == id)
                .cloned()
                .collect();

            let num_inputs = self.nodes.get(&id).map(|n| n.num_inputs()).unwrap_or(0);
            let mut input_bufs = vec![vec![0.0f64; buffer_size]; num_inputs];

            for conn in &input_conns {
                if let Some(src_bufs) = self.buffers.get(&conn.source_id) {
                    if conn.source_channel < src_bufs.len()
                        && conn.dest_channel < input_bufs.len()
                    {
                        for i in 0..buffer_size {
                            input_bufs[conn.dest_channel][i] += src_bufs[conn.source_channel][i];
                        }
                    }
                }
            }

            let num_outputs = self.nodes.get(&id).map(|n| n.num_outputs()).unwrap_or(0);
            let mut output_bufs = vec![vec![0.0f64; buffer_size]; num_outputs];

            if let Some(node) = self.nodes.get_mut(&id) {
                let input_refs: Vec<&[f64]> = input_bufs.iter().map(|b| b.as_slice()).collect();
                let mut output_refs: Vec<&mut [f64]> =
                    output_bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
                node.process(&input_refs, &mut output_refs, info);
            }

            if let Some(bufs) = self.buffers.get_mut(&id) {
                for (ch, out) in output_bufs.into_iter().enumerate() {
                    if ch < bufs.len() {
                        bufs[ch] = out;
                    }
                }
            }
        }
    }

    /// Get the output buffer of a node.
    pub fn node_output(&self, id: NodeId, channel: usize) -> Option<&[f64]> {
        self.buffers
            .get(&id)
            .and_then(|bufs| bufs.get(channel))
            .map(|b| b.as_slice())
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Reset all nodes.
    pub fn reset(&mut self) {
        for node in self.nodes.values_mut() {
            node.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Playback controller
// ---------------------------------------------------------------------------

/// Controls playback: play, pause, stop, seek, loop.
pub struct PlaybackController {
    transport: TransportInfo,
    playing: Arc<AtomicBool>,
    position: Arc<AtomicU64>,
}

impl PlaybackController {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            transport: TransportInfo {
                sample_rate,
                ..Default::default()
            },
            playing: Arc::new(AtomicBool::new(false)),
            position: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn play(&mut self) {
        self.transport.state = TransportState::Playing;
        self.playing.store(true, Ordering::Release);
    }

    pub fn pause(&mut self) {
        self.transport.state = TransportState::Paused;
        self.playing.store(false, Ordering::Release);
    }

    pub fn stop(&mut self) {
        self.transport.state = TransportState::Stopped;
        self.transport.position_seconds = 0.0;
        self.transport.position_samples = 0;
        self.playing.store(false, Ordering::Release);
        self.position.store(0, Ordering::Release);
    }

    pub fn seek(&mut self, position_seconds: f64) {
        self.transport.position_seconds = position_seconds;
        self.transport.position_samples =
            (position_seconds * self.transport.sample_rate) as u64;
        self.position
            .store(self.transport.position_samples, Ordering::Release);
    }

    pub fn set_loop(&mut self, enabled: bool, start: f64, end: f64) {
        self.transport.loop_enabled = enabled;
        self.transport.loop_start = start;
        self.transport.loop_end = end;
    }

    pub fn set_tempo(&mut self, bpm: f64) {
        self.transport.tempo = bpm;
    }

    pub fn transport(&self) -> &TransportInfo {
        &self.transport
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Acquire)
    }

    /// Advance transport and return render info.
    pub fn begin_render(&mut self, buffer_size: usize) -> AudioRenderInfo {
        let info = AudioRenderInfo {
            num_channels: 2,
            buffer_size,
            sample_rate: self.transport.sample_rate,
            transport: self.transport.clone(),
        };
        self.transport.advance(buffer_size);
        self.position
            .store(self.transport.position_samples, Ordering::Release);
        info
    }
}

// ---------------------------------------------------------------------------
// Render cache
// ---------------------------------------------------------------------------

/// Cache key for pre-rendered audio segments.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    pub source_id: u64,
    pub segment_index: usize,
}

/// Pre-computed audio segment for the render cache.
#[derive(Debug, Clone)]
pub struct CachedSegment {
    /// Start time in seconds.
    pub start_time: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Pre-rendered audio (stereo interleaved).
    pub samples: Vec<f64>,
    /// Number of channels.
    pub num_channels: usize,
}

/// Cache for pre-computed audio segments.
pub struct RenderCache {
    segments: HashMap<CacheKey, CachedSegment>,
    max_entries: usize,
}

impl RenderCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            segments: HashMap::new(),
            max_entries,
        }
    }

    /// Insert a segment into the cache.
    pub fn insert(&mut self, key: CacheKey, segment: CachedSegment) {
        if self.segments.len() >= self.max_entries {
            // Evict oldest entry (simple FIFO by picking first key)
            if let Some(first_key) = self.segments.keys().next().cloned() {
                self.segments.remove(&first_key);
            }
        }
        self.segments.insert(key, segment);
    }

    /// Look up a cached segment.
    pub fn get(&self, key: &CacheKey) -> Option<&CachedSegment> {
        self.segments.get(key)
    }

    /// Invalidate a specific entry.
    pub fn invalidate(&mut self, key: &CacheKey) {
        self.segments.remove(key);
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.segments.clear();
    }

    /// Number of cached segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Performance renderer
// ---------------------------------------------------------------------------

/// Ties analysis and synthesis for real-time output.
///
/// Based on MUPerformanceRenderer. Manages the audio graph, playback
/// controller, and render cache to produce real-time audio output from
/// analyzed and edited audio sources.
pub struct PerformanceRenderer {
    pub graph: AudioGraph,
    pub controller: PlaybackController,
    pub cache: RenderCache,
    output_buffer_l: Vec<f64>,
    output_buffer_r: Vec<f64>,
}

impl PerformanceRenderer {
    pub fn new(sample_rate: f64, buffer_size: usize) -> Self {
        Self {
            graph: AudioGraph::new(buffer_size),
            controller: PlaybackController::new(sample_rate),
            cache: RenderCache::new(256),
            output_buffer_l: vec![0.0; buffer_size],
            output_buffer_r: vec![0.0; buffer_size],
        }
    }

    /// Render the next buffer of audio.
    ///
    /// Returns references to the left and right output channels.
    pub fn render(&mut self, buffer_size: usize) -> (&[f64], &[f64]) {
        let info = self.controller.begin_render(buffer_size);

        // Ensure output buffers are large enough
        if self.output_buffer_l.len() < buffer_size {
            self.output_buffer_l.resize(buffer_size, 0.0);
            self.output_buffer_r.resize(buffer_size, 0.0);
        }

        self.output_buffer_l[..buffer_size].fill(0.0);
        self.output_buffer_r[..buffer_size].fill(0.0);

        if info.transport.state == TransportState::Playing
            || info.transport.state == TransportState::Recording
        {
            self.graph.process(&info);
        }

        (&self.output_buffer_l[..buffer_size], &self.output_buffer_r[..buffer_size])
    }

    /// Reset the renderer state.
    pub fn reset(&mut self) {
        self.graph.reset();
        self.cache.clear();
        self.output_buffer_l.fill(0.0);
        self.output_buffer_r.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Built-in audio nodes
// ---------------------------------------------------------------------------

/// A silence/null source node (produces zeros).
pub struct SilenceNode {
    num_outputs: usize,
}

impl SilenceNode {
    pub fn new(num_outputs: usize) -> Self {
        Self { num_outputs }
    }
}

impl AudioNode for SilenceNode {
    fn process(
        &mut self,
        _inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
        _info: &AudioRenderInfo,
    ) {
        for out in outputs.iter_mut() {
            out.fill(0.0);
        }
    }
    fn num_inputs(&self) -> usize {
        0
    }
    fn num_outputs(&self) -> usize {
        self.num_outputs
    }
    fn reset(&mut self) {}
    fn name(&self) -> &str {
        "Silence"
    }
}

/// A gain node that scales its input.
pub struct GainNode {
    pub gain: f64,
}

impl GainNode {
    pub fn new(gain: f64) -> Self {
        Self { gain }
    }
}

impl AudioNode for GainNode {
    fn process(
        &mut self,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
        _info: &AudioRenderInfo,
    ) {
        for (ch, out) in outputs.iter_mut().enumerate() {
            if ch < inputs.len() {
                for (i, sample) in out.iter_mut().enumerate() {
                    *sample = inputs[ch][i] * self.gain;
                }
            }
        }
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        2
    }
    fn reset(&mut self) {}
    fn name(&self) -> &str {
        "Gain"
    }
}

/// A mixer node that sums all inputs.
pub struct MixerNode {
    num_channels: usize,
}

impl MixerNode {
    pub fn new(num_channels: usize) -> Self {
        Self { num_channels }
    }
}

impl AudioNode for MixerNode {
    fn process(
        &mut self,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
        _info: &AudioRenderInfo,
    ) {
        for out in outputs.iter_mut() {
            out.fill(0.0);
        }
        // Sum all input channels into the corresponding output channel
        for (ch, out) in outputs.iter_mut().enumerate() {
            if ch < inputs.len() {
                for (i, sample) in out.iter_mut().enumerate() {
                    *sample += inputs[ch][i];
                }
            }
        }
    }
    fn num_inputs(&self) -> usize {
        self.num_channels
    }
    fn num_outputs(&self) -> usize {
        self.num_channels
    }
    fn reset(&mut self) {}
    fn name(&self) -> &str {
        "Mixer"
    }
}

/// A buffer source node that plays back pre-loaded audio.
pub struct BufferSourceNode {
    samples: Vec<Vec<f64>>,
    position: usize,
    looping: bool,
}

impl BufferSourceNode {
    pub fn new(samples: Vec<Vec<f64>>, looping: bool) -> Self {
        Self {
            samples,
            position: 0,
            looping,
        }
    }

    pub fn from_mono(data: Vec<f64>) -> Self {
        Self::new(vec![data.clone(), data], false)
    }
}

impl AudioNode for BufferSourceNode {
    fn process(
        &mut self,
        _inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
        _info: &AudioRenderInfo,
    ) {
        if self.samples.is_empty() {
            for out in outputs.iter_mut() {
                out.fill(0.0);
            }
            return;
        }
        let total_len = self.samples[0].len();
        for (ch, out) in outputs.iter_mut().enumerate() {
            let src = if ch < self.samples.len() {
                &self.samples[ch]
            } else {
                &self.samples[0]
            };
            for sample in out.iter_mut() {
                if self.position < total_len {
                    *sample = src[self.position];
                } else if self.looping && total_len > 0 {
                    *sample = src[self.position % total_len];
                } else {
                    *sample = 0.0;
                }
            }
            // Only increment position on first channel
            if ch == 0 {
                self.position += out.len();
                if self.looping && total_len > 0 {
                    self.position %= total_len;
                }
            }
        }
    }
    fn num_inputs(&self) -> usize {
        0
    }
    fn num_outputs(&self) -> usize {
        self.samples.len().max(1)
    }
    fn reset(&mut self) {
        self.position = 0;
    }
    fn name(&self) -> &str {
        "BufferSource"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_advance() {
        let mut transport = TransportInfo {
            state: TransportState::Playing,
            sample_rate: 44100.0,
            ..Default::default()
        };
        transport.advance(44100);
        assert!((transport.position_seconds - 1.0).abs() < 0.001);
        assert_eq!(transport.position_samples, 44100);
    }

    #[test]
    fn test_transport_loop() {
        let mut transport = TransportInfo {
            state: TransportState::Playing,
            sample_rate: 44100.0,
            loop_enabled: true,
            loop_start: 0.0,
            loop_end: 1.0,
            ..Default::default()
        };
        // Advance past loop end
        transport.advance(66150); // 1.5 seconds
        assert!(transport.position_seconds >= 0.0);
        assert!(transport.position_seconds < 1.0);
    }

    #[test]
    fn test_transport_stopped_no_advance() {
        let mut transport = TransportInfo::default();
        transport.advance(44100);
        assert_eq!(transport.position_samples, 0);
    }

    #[test]
    fn test_ring_buffer_write_read() {
        let rb = AudioRingBuffer::new(1024);
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let written = rb.write(&data);
        assert_eq!(written, 100);
        assert_eq!(rb.available_read(), 100);

        let mut output = vec![0.0; 100];
        let read = rb.read(&mut output);
        assert_eq!(read, 100);
        assert_eq!(output, data);
        assert_eq!(rb.available_read(), 0);
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let rb = AudioRingBuffer::new(16); // Will be 16 (already power of 2)
        let data = vec![1.0; 10];
        rb.write(&data);
        let mut out = vec![0.0; 10];
        rb.read(&mut out);

        // Write again, this will wrap around
        let data2 = vec![2.0; 10];
        let written = rb.write(&data2);
        assert_eq!(written, 10);
        let mut out2 = vec![0.0; 10];
        let read = rb.read(&mut out2);
        assert_eq!(read, 10);
        assert!(out2.iter().all(|&v| (v - 2.0).abs() < 1e-10));
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let rb = AudioRingBuffer::new(8); // capacity = 8, usable = 7
        let data = vec![1.0; 20]; // More than capacity
        let written = rb.write(&data);
        assert!(written < 20);
        assert_eq!(written, rb.capacity());
    }

    #[test]
    fn test_ring_buffer_reset() {
        let rb = AudioRingBuffer::new(32);
        rb.write(&[1.0; 10]);
        assert_eq!(rb.available_read(), 10);
        rb.reset();
        assert_eq!(rb.available_read(), 0);
    }

    #[test]
    fn test_audio_graph_basic() {
        let mut graph = AudioGraph::new(256);
        let silence_id = graph.add_node(Box::new(SilenceNode::new(2)));
        let gain_id = graph.add_node(Box::new(GainNode::new(0.5)));
        graph.connect(silence_id, 0, gain_id, 0);
        graph.connect(silence_id, 1, gain_id, 1);

        assert_eq!(graph.num_nodes(), 2);

        let info = AudioRenderInfo {
            num_channels: 2,
            buffer_size: 256,
            sample_rate: 44100.0,
            transport: TransportInfo::default(),
        };
        graph.process(&info);

        // Output should be all zeros (silence * gain)
        if let Some(output) = graph.node_output(gain_id, 0) {
            assert!(output.iter().all(|&v| v.abs() < 1e-10));
        }
    }

    #[test]
    fn test_audio_graph_remove_node() {
        let mut graph = AudioGraph::new(256);
        let id1 = graph.add_node(Box::new(SilenceNode::new(1)));
        let id2 = graph.add_node(Box::new(GainNode::new(1.0)));
        graph.connect(id1, 0, id2, 0);
        assert_eq!(graph.num_nodes(), 2);

        graph.remove_node(id1);
        assert_eq!(graph.num_nodes(), 1);
    }

    #[test]
    fn test_playback_controller() {
        let mut ctrl = PlaybackController::new(44100.0);
        assert!(!ctrl.is_playing());

        ctrl.play();
        assert!(ctrl.is_playing());
        assert_eq!(ctrl.transport().state, TransportState::Playing);

        ctrl.seek(1.0);
        assert!((ctrl.transport().position_seconds - 1.0).abs() < 0.001);

        ctrl.pause();
        assert!(!ctrl.is_playing());
        assert_eq!(ctrl.transport().state, TransportState::Paused);

        ctrl.stop();
        assert_eq!(ctrl.transport().position_seconds, 0.0);
    }

    #[test]
    fn test_playback_controller_loop() {
        let mut ctrl = PlaybackController::new(44100.0);
        ctrl.set_loop(true, 0.0, 1.0);
        ctrl.play();
        // Render enough to pass loop point
        let _info = ctrl.begin_render(66150); // 1.5 seconds
        assert!(ctrl.transport().position_seconds < 1.0);
    }

    #[test]
    fn test_render_cache() {
        let mut cache = RenderCache::new(10);
        let key = CacheKey {
            source_id: 1,
            segment_index: 0,
        };
        let segment = CachedSegment {
            start_time: 0.0,
            duration: 1.0,
            samples: vec![0.5; 1000],
            num_channels: 1,
        };
        cache.insert(key.clone(), segment);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&key).is_some());

        cache.invalidate(&key);
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_render_cache_eviction() {
        let mut cache = RenderCache::new(3);
        for i in 0..5 {
            cache.insert(
                CacheKey {
                    source_id: i,
                    segment_index: 0,
                },
                CachedSegment {
                    start_time: 0.0,
                    duration: 1.0,
                    samples: vec![0.0; 10],
                    num_channels: 1,
                },
            );
        }
        assert!(cache.len() <= 3);
    }

    #[test]
    fn test_performance_renderer() {
        let mut renderer = PerformanceRenderer::new(44100.0, 512);
        let (left, right) = renderer.render(256);
        assert_eq!(left.len(), 256);
        assert_eq!(right.len(), 256);
        // Stopped, so output should be silence
        assert!(left.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_render_info_duration() {
        let info = AudioRenderInfo {
            num_channels: 2,
            buffer_size: 512,
            sample_rate: 44100.0,
            transport: TransportInfo::default(),
        };
        assert!((info.buffer_duration() - 512.0 / 44100.0).abs() < 1e-6);
    }

    #[test]
    fn test_gain_node_processing() {
        let mut gain = GainNode::new(0.5);
        let input = vec![1.0f64; 10];
        let inputs: Vec<&[f64]> = vec![&input, &input];
        let mut out_l = vec![0.0f64; 10];
        let mut out_r = vec![0.0f64; 10];
        let mut outputs: Vec<&mut [f64]> = vec![&mut out_l, &mut out_r];
        let info = AudioRenderInfo {
            num_channels: 2,
            buffer_size: 10,
            sample_rate: 44100.0,
            transport: TransportInfo::default(),
        };
        gain.process(&inputs, &mut outputs, &info);
        assert!((outputs[0][0] - 0.5).abs() < 1e-10);
    }
}
