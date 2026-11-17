//! TCP-based network transport for the collaborative editing sync system.
//!
//! Provides a `SyncServer` that accepts multiple client connections and
//! broadcasts confirmed operations, and a `SyncClient` that connects to a
//! server and relays operations. The wire format is length-prefixed JSON over
//! TCP (4-byte big-endian length prefix followed by a JSON payload).
//!
//! A `CollabSession` combines a `SyncClient` with an `EditSession` for a
//! convenient start/join/leave workflow with automatic sync on connect.

use std::collections::HashMap;
use std::io::{self, Read as _, Write as _};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::collab::{
    CollabError, EditOp, EditOperation, EditSession, OperationLog, SyncMessage,
};

// ---------------------------------------------------------------------------
// Network errors
// ---------------------------------------------------------------------------

/// Errors specific to network transport operations.
#[derive(Debug)]
pub enum NetworkError {
    /// An I/O error occurred on the socket.
    Io(io::Error),
    /// A serialization or deserialization error.
    Serialization(String),
    /// The connection was closed by the remote peer.
    ConnectionClosed,
    /// The server is not running or could not start.
    ServerNotRunning,
    /// The client is not connected.
    NotConnected,
    /// A channel send/recv error.
    ChannelError(String),
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Network I/O error: {}", e),
            Self::Serialization(e) => write!(f, "Serialization error: {}", e),
            Self::ConnectionClosed => write!(f, "Connection closed"),
            Self::ServerNotRunning => write!(f, "Server not running"),
            Self::NotConnected => write!(f, "Client not connected"),
            Self::ChannelError(e) => write!(f, "Channel error: {}", e),
        }
    }
}

impl std::error::Error for NetworkError {}

impl From<io::Error> for NetworkError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<CollabError> for NetworkError {
    fn from(e: CollabError) -> Self {
        Self::Serialization(e.to_string())
    }
}

pub type NetworkResult<T> = Result<T, NetworkError>;

// ---------------------------------------------------------------------------
// Wire protocol: length-prefixed JSON
// ---------------------------------------------------------------------------

/// Send a `SyncMessage` over a TCP stream using the length-prefixed wire format.
///
/// The format is a 4-byte big-endian length prefix followed by a JSON payload.
pub fn send_message(stream: &mut TcpStream, msg: &SyncMessage) -> NetworkResult<()> {
    let payload = msg.to_json().map_err(NetworkError::from)?;
    let len = payload.len() as u32;
    let len_bytes = len.to_be_bytes();

    stream.write_all(&len_bytes)?;
    stream.write_all(&payload)?;
    stream.flush()?;
    Ok(())
}

/// Receive a `SyncMessage` from a TCP stream using the length-prefixed wire format.
///
/// The stream **must not** have a read timeout set when using this function.
/// For non-blocking usage, use `StreamReader` instead.
pub fn recv_message(stream: &mut TcpStream) -> NetworkResult<SyncMessage> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(NetworkError::ConnectionClosed);
        }
        Err(e) => return Err(NetworkError::Io(e)),
    }

    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 {
        return Err(NetworkError::Serialization(
            "Zero-length message".to_string(),
        ));
    }
    if len > 16 * 1024 * 1024 {
        return Err(NetworkError::Serialization(format!(
            "Message too large: {} bytes",
            len
        )));
    }

    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    SyncMessage::from_json(&payload).map_err(NetworkError::from)
}

// ---------------------------------------------------------------------------
// StreamReader -- buffered non-blocking message reader
// ---------------------------------------------------------------------------

/// A buffered reader for length-prefixed messages on a non-blocking TCP stream.
///
/// Accumulates bytes from non-blocking reads and parses complete messages.
/// This avoids the partial-read corruption that occurs when combining
/// `read_exact` with socket timeouts.
struct StreamReader {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl StreamReader {
    fn new(stream: TcpStream) -> io::Result<Self> {
        stream.set_nonblocking(true)?;
        Ok(Self {
            stream,
            buf: Vec::with_capacity(4096),
        })
    }

    /// Try to read available bytes from the socket into the internal buffer.
    fn fill_buf(&mut self) -> Result<bool, NetworkError> {
        let mut tmp = [0u8; 8192];
        match self.stream.read(&mut tmp) {
            Ok(0) => Err(NetworkError::ConnectionClosed),
            Ok(n) => {
                self.buf.extend_from_slice(&tmp[..n]);
                Ok(true)
            }
            Err(ref e)
                if e.kind() == io::ErrorKind::WouldBlock
                    || e.kind() == io::ErrorKind::TimedOut =>
            {
                Ok(false)
            }
            Err(e) => Err(NetworkError::Io(e)),
        }
    }

    /// Try to parse a complete message from the internal buffer.
    fn try_parse_message(&mut self) -> Result<Option<SyncMessage>, NetworkError> {
        if self.buf.len() < 4 {
            return Ok(None);
        }

        let len =
            u32::from_be_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]) as usize;

        if len == 0 {
            self.buf.drain(..4);
            return Err(NetworkError::Serialization(
                "Zero-length message".to_string(),
            ));
        }
        if len > 16 * 1024 * 1024 {
            self.buf.drain(..4);
            return Err(NetworkError::Serialization(format!(
                "Message too large: {} bytes",
                len
            )));
        }

        let total = 4 + len;
        if self.buf.len() < total {
            return Ok(None);
        }

        let payload = &self.buf[4..total];
        let msg = SyncMessage::from_json(payload).map_err(NetworkError::from)?;
        self.buf.drain(..total);
        Ok(Some(msg))
    }

    /// Non-blocking attempt to receive a message: reads available bytes, then
    /// tries to parse a complete message.
    fn try_recv(&mut self) -> Result<Option<SyncMessage>, NetworkError> {
        if let Some(msg) = self.try_parse_message()? {
            return Ok(Some(msg));
        }
        self.fill_buf()?;
        self.try_parse_message()
    }
}

// ---------------------------------------------------------------------------
// Client handle (server-side per-client bookkeeping)
// ---------------------------------------------------------------------------

struct ClientHandle {
    sender: Sender<SyncMessage>,
    alive: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// SyncServer
// ---------------------------------------------------------------------------

/// A TCP-based synchronization server that:
///
/// - Listens on a configurable port for client connections.
/// - Assigns each client a unique identifier.
/// - Maintains the authoritative `OperationLog`.
/// - Receives `PushOperation` messages, assigns sequential global IDs, and
///   broadcasts confirmed operations to all other connected clients.
/// - Handles `SyncRequest` (sends all operations since the requested ID).
/// - Handles `ModelSnapshot` requests.
pub struct SyncServer {
    local_addr: std::net::SocketAddr,
    operation_log: Arc<Mutex<OperationLog>>,
    clients: Arc<Mutex<HashMap<String, ClientHandle>>>,
    running: Arc<AtomicBool>,
    accept_handle: Option<JoinHandle<()>>,
    #[allow(dead_code)]
    next_client_id: Arc<Mutex<u64>>,
    model_snapshot: Arc<Mutex<Option<String>>>,
}

impl SyncServer {
    /// Create and start a new sync server listening on the given address.
    pub fn start<A: ToSocketAddrs>(addr: A) -> NetworkResult<Self> {
        let listener = TcpListener::bind(addr)?;
        let local_addr = listener.local_addr()?;

        let operation_log = Arc::new(Mutex::new(OperationLog::new()));
        let clients: Arc<Mutex<HashMap<String, ClientHandle>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let running = Arc::new(AtomicBool::new(true));
        let next_client_id = Arc::new(Mutex::new(1u64));
        let model_snapshot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        let log_clone = Arc::clone(&operation_log);
        let clients_clone = Arc::clone(&clients);
        let running_clone = Arc::clone(&running);
        let next_id_clone = Arc::clone(&next_client_id);
        let snapshot_clone = Arc::clone(&model_snapshot);

        listener
            .set_nonblocking(true)
            .expect("set_nonblocking on listener");

        let accept_handle = thread::spawn(move || {
            while running_clone.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _peer_addr)) => {
                        let client_id = {
                            let mut id = next_id_clone.lock().unwrap();
                            let cid = format!("client-{}", *id);
                            *id += 1;
                            cid
                        };
                        Self::spawn_client_handler(
                            stream,
                            client_id,
                            Arc::clone(&log_clone),
                            Arc::clone(&clients_clone),
                            Arc::clone(&running_clone),
                            Arc::clone(&snapshot_clone),
                        );
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => {
                        if !running_clone.load(Ordering::Relaxed) {
                            break;
                        }
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        });

        Ok(Self {
            local_addr,
            operation_log,
            clients,
            running,
            accept_handle: Some(accept_handle),
            next_client_id,
            model_snapshot,
        })
    }

    /// The local socket address the server is bound to.
    pub fn local_addr(&self) -> std::net::SocketAddr {
        self.local_addr
    }

    /// The number of currently connected clients.
    pub fn client_count(&self) -> usize {
        let clients = self.clients.lock().unwrap();
        clients
            .values()
            .filter(|c| c.alive.load(Ordering::Relaxed))
            .count()
    }

    /// Set a model snapshot JSON that will be sent to clients on `ModelSnapshot` request.
    pub fn set_model_snapshot(&self, json: String) {
        let mut snap = self.model_snapshot.lock().unwrap();
        *snap = Some(json);
    }

    /// Get a clone of the authoritative operation log.
    pub fn operation_log(&self) -> OperationLog {
        self.operation_log.lock().unwrap().clone()
    }

    /// Stop the server and disconnect all clients.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        {
            let mut clients = self.clients.lock().unwrap();
            for (_, handle) in clients.drain() {
                handle.alive.store(false, Ordering::Relaxed);
            }
        }
        if let Some(handle) = self.accept_handle.take() {
            let _ = handle.join();
        }
    }

    fn spawn_client_handler(
        stream: TcpStream,
        client_id: String,
        operation_log: Arc<Mutex<OperationLog>>,
        clients: Arc<Mutex<HashMap<String, ClientHandle>>>,
        running: Arc<AtomicBool>,
        model_snapshot: Arc<Mutex<Option<String>>>,
    ) {
        let (tx, rx): (Sender<SyncMessage>, Receiver<SyncMessage>) = mpsc::channel();
        let alive = Arc::new(AtomicBool::new(true));

        {
            let mut map = clients.lock().unwrap();
            map.insert(
                client_id.clone(),
                ClientHandle {
                    sender: tx,
                    alive: Arc::clone(&alive),
                },
            );
        }

        // --- Writer thread ---
        let mut write_stream = stream.try_clone().expect("clone TcpStream for writer");
        let alive_w = Arc::clone(&alive);
        let _writer_handle = thread::spawn(move || {
            while alive_w.load(Ordering::Relaxed) {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(msg) => {
                        if send_message(&mut write_stream, &msg).is_err() {
                            alive_w.store(false, Ordering::Relaxed);
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });

        // --- Reader thread (uses StreamReader to avoid partial-read issues) ---
        let alive_r = Arc::clone(&alive);
        let cid = client_id;
        let _reader_handle = thread::spawn(move || {
            let mut reader = match StreamReader::new(stream) {
                Ok(r) => r,
                Err(_) => {
                    alive_r.store(false, Ordering::Relaxed);
                    return;
                }
            };

            while alive_r.load(Ordering::Relaxed) && running.load(Ordering::Relaxed) {
                match reader.try_recv() {
                    Ok(Some(msg)) => {
                        Self::handle_client_message(
                            &msg,
                            &cid,
                            &operation_log,
                            &clients,
                            &model_snapshot,
                        );
                    }
                    Ok(None) => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(NetworkError::ConnectionClosed) => {
                        alive_r.store(false, Ordering::Relaxed);
                        break;
                    }
                    Err(_) => {
                        alive_r.store(false, Ordering::Relaxed);
                        break;
                    }
                }
            }

            let mut map = clients.lock().unwrap();
            map.remove(&cid);
        });
    }

    fn handle_client_message(
        msg: &SyncMessage,
        sender_id: &str,
        operation_log: &Arc<Mutex<OperationLog>>,
        clients: &Arc<Mutex<HashMap<String, ClientHandle>>>,
        model_snapshot: &Arc<Mutex<Option<String>>>,
    ) {
        match msg {
            SyncMessage::PushOperation { operation } => {
                let confirmed_op = {
                    let mut log = operation_log.lock().unwrap();
                    let op_id = log.append(
                        operation.op.clone(),
                        operation.author.clone(),
                        operation.timestamp_ms,
                    );
                    EditOperation {
                        op_id,
                        op: operation.op.clone(),
                        author: operation.author.clone(),
                        timestamp_ms: operation.timestamp_ms,
                    }
                };

                let ack = SyncMessage::AckOperation {
                    op_id: confirmed_op.op_id,
                };
                Self::send_to_client(clients, sender_id, ack);

                let broadcast = SyncMessage::SyncResponse {
                    operations: vec![confirmed_op],
                };
                Self::broadcast_except(clients, sender_id, broadcast);
            }

            SyncMessage::SyncRequest { last_known_op_id } => {
                let ops = {
                    let log = operation_log.lock().unwrap();
                    log.replay_from(*last_known_op_id)
                };
                let response = SyncMessage::SyncResponse { operations: ops };
                Self::send_to_client(clients, sender_id, response);
            }

            SyncMessage::ModelSnapshot { .. } => {
                let snap = model_snapshot.lock().unwrap();
                let log = operation_log.lock().unwrap();
                let response = SyncMessage::ModelSnapshot {
                    state_json: snap.clone().unwrap_or_else(|| "{}".to_string()),
                    last_op_id: log.last_op_id(),
                };
                Self::send_to_client(clients, sender_id, response);
            }

            _ => {}
        }
    }

    fn send_to_client(
        clients: &Arc<Mutex<HashMap<String, ClientHandle>>>,
        client_id: &str,
        msg: SyncMessage,
    ) {
        let map = clients.lock().unwrap();
        if let Some(handle) = map.get(client_id) {
            if handle.alive.load(Ordering::Relaxed) {
                let _ = handle.sender.send(msg);
            }
        }
    }

    fn broadcast_except(
        clients: &Arc<Mutex<HashMap<String, ClientHandle>>>,
        except_id: &str,
        msg: SyncMessage,
    ) {
        let map = clients.lock().unwrap();
        for (id, handle) in map.iter() {
            if id != except_id && handle.alive.load(Ordering::Relaxed) {
                let _ = handle.sender.send(msg.clone());
            }
        }
    }
}

impl Drop for SyncServer {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// SyncClient
// ---------------------------------------------------------------------------

/// A TCP-based synchronization client that connects to a `SyncServer`.
///
/// Sends local operations as `PushOperation` messages and receives confirmed
/// operations from the server in a background thread. On connect the client
/// automatically sends a `SyncRequest` to retrieve any missed operations.
pub struct SyncClient {
    write_stream: Mutex<TcpStream>,
    running: Arc<AtomicBool>,
    recv_handle: Option<JoinHandle<()>>,
    incoming: Mutex<Receiver<SyncMessage>>,
    last_known_op_id: Mutex<u64>,
}

impl SyncClient {
    /// Connect to a sync server at the given address.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> NetworkResult<Self> {
        Self::connect_with_last_op(addr, 0)
    }

    /// Connect to a sync server and request operations after `last_known_op_id`.
    pub fn connect_with_last_op<A: ToSocketAddrs>(
        addr: A,
        last_known_op_id: u64,
    ) -> NetworkResult<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;

        let running = Arc::new(AtomicBool::new(true));
        let (tx, rx): (Sender<SyncMessage>, Receiver<SyncMessage>) = mpsc::channel();

        let read_stream = stream.try_clone()?;
        let running_clone = Arc::clone(&running);

        let recv_handle = thread::spawn(move || {
            let mut reader = match StreamReader::new(read_stream) {
                Ok(r) => r,
                Err(_) => {
                    running_clone.store(false, Ordering::Relaxed);
                    return;
                }
            };

            while running_clone.load(Ordering::Relaxed) {
                match reader.try_recv() {
                    Ok(Some(msg)) => {
                        if tx.send(msg).is_err() {
                            break;
                        }
                    }
                    Ok(None) => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(NetworkError::ConnectionClosed) => {
                        running_clone.store(false, Ordering::Relaxed);
                        break;
                    }
                    Err(_) => {
                        running_clone.store(false, Ordering::Relaxed);
                        break;
                    }
                }
            }
        });

        let client = Self {
            write_stream: Mutex::new(stream),
            running,
            recv_handle: Some(recv_handle),
            incoming: Mutex::new(rx),
            last_known_op_id: Mutex::new(last_known_op_id),
        };

        // Send initial sync request.
        client.send(SyncMessage::SyncRequest { last_known_op_id })?;

        Ok(client)
    }

    /// Send a `SyncMessage` to the server.
    pub fn send(&self, msg: SyncMessage) -> NetworkResult<()> {
        let mut stream = self.write_stream.lock().map_err(|e| {
            NetworkError::ChannelError(format!("Failed to lock write stream: {}", e))
        })?;
        send_message(&mut stream, &msg)
    }

    /// Push a local edit operation to the server.
    pub fn push_operation(&self, operation: EditOperation) -> NetworkResult<()> {
        self.send(SyncMessage::PushOperation { operation })
    }

    /// Try to receive the next message from the server (non-blocking).
    pub fn try_recv(&self) -> NetworkResult<Option<SyncMessage>> {
        let rx = self.incoming.lock().map_err(|e| {
            NetworkError::ChannelError(format!("Failed to lock incoming: {}", e))
        })?;
        match rx.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err(NetworkError::ConnectionClosed),
        }
    }

    /// Receive the next message, blocking until one arrives or timeout elapses.
    pub fn recv_timeout(&self, timeout: Duration) -> NetworkResult<Option<SyncMessage>> {
        let rx = self.incoming.lock().map_err(|e| {
            NetworkError::ChannelError(format!("Failed to lock incoming: {}", e))
        })?;
        match rx.recv_timeout(timeout) {
            Ok(msg) => Ok(Some(msg)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(NetworkError::ConnectionClosed),
        }
    }

    /// Drain all currently available messages from the server (non-blocking).
    pub fn drain_messages(&self) -> NetworkResult<Vec<SyncMessage>> {
        let rx = self.incoming.lock().map_err(|e| {
            NetworkError::ChannelError(format!("Failed to lock incoming: {}", e))
        })?;
        let mut msgs = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(msg) => msgs.push(msg),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        Ok(msgs)
    }

    /// Whether the client is still connected to the server.
    pub fn is_connected(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get the last known confirmed operation ID.
    pub fn last_known_op_id(&self) -> u64 {
        *self.last_known_op_id.lock().unwrap()
    }

    /// Update the last known operation ID.
    pub fn set_last_known_op_id(&self, op_id: u64) {
        *self.last_known_op_id.lock().unwrap() = op_id;
    }

    /// Disconnect from the server.
    pub fn disconnect(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        // Shut down the socket to unblock the reader thread.
        if let Ok(stream) = self.write_stream.lock() {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
        if let Some(handle) = self.recv_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SyncClient {
    fn drop(&mut self) {
        self.disconnect();
    }
}

// ---------------------------------------------------------------------------
// CollabSession
// ---------------------------------------------------------------------------

/// Combines a `SyncClient` with an `EditSession` for a convenient
/// collaborative editing workflow.
pub struct CollabSession {
    client: SyncClient,
    session: EditSession,
    /// Number of pending ops that have been pushed but not yet acknowledged.
    sent_count: usize,
}

impl CollabSession {
    /// Join a collaborative session by connecting to the server.
    pub fn join<A: ToSocketAddrs>(addr: A, author: String) -> NetworkResult<Self> {
        let client = SyncClient::connect(addr)?;
        let session = EditSession::new(author);
        Ok(Self {
            client,
            session,
            sent_count: 0,
        })
    }

    /// Join with a known last operation ID (for reconnection).
    pub fn rejoin<A: ToSocketAddrs>(
        addr: A,
        author: String,
        last_known_op_id: u64,
    ) -> NetworkResult<Self> {
        let client = SyncClient::connect_with_last_op(addr, last_known_op_id)?;
        let mut session = EditSession::new(author);
        session.last_confirmed_op_id = last_known_op_id;
        Ok(Self {
            client,
            session,
            sent_count: 0,
        })
    }

    /// Apply a local edit operation (optimistic application).
    pub fn apply_local(&mut self, op: EditOp) {
        self.session.apply_local(op);
    }

    /// Synchronize with the server: push unsent pending operations and process
    /// incoming confirmed operations.
    ///
    /// Returns the number of remote operations received.
    pub fn sync(&mut self) -> NetworkResult<usize> {
        let pending = self.session.pending_ops();
        let total_pending = pending.len();
        while self.sent_count < total_pending {
            let op = pending[self.sent_count].clone();
            let msg = SyncMessage::PushOperation {
                operation: EditOperation {
                    op_id: 0,
                    op,
                    author: self.session.author.clone(),
                    timestamp_ms: 0,
                },
            };
            self.client.send(msg)?;
            self.sent_count += 1;
        }

        let messages = self.client.drain_messages()?;
        let mut remote_count = 0;

        for msg in messages {
            match msg {
                SyncMessage::AckOperation { op_id } => {
                    self.session.handle_ack(op_id);
                    self.client.set_last_known_op_id(op_id);
                    if self.sent_count > 0 {
                        self.sent_count -= 1;
                    }
                }
                SyncMessage::SyncResponse { operations } => {
                    remote_count += operations.len();
                    self.session.receive_remote_ops(&operations);
                    if let Some(last) = operations.last() {
                        self.client.set_last_known_op_id(last.op_id);
                    }
                }
                _ => {}
            }
        }

        Ok(remote_count)
    }

    /// Get a reference to the underlying edit session.
    pub fn session(&self) -> &EditSession {
        &self.session
    }

    /// Get a mutable reference to the underlying edit session.
    pub fn session_mut(&mut self) -> &mut EditSession {
        &mut self.session
    }

    /// Whether the client is still connected.
    pub fn is_connected(&self) -> bool {
        self.client.is_connected()
    }

    /// Disconnect and leave the collaborative session.
    pub fn leave(&mut self) {
        self.client.disconnect();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn start_test_server() -> SyncServer {
        SyncServer::start("127.0.0.1:0").expect("server should start")
    }

    fn settle() {
        thread::sleep(Duration::from_millis(200));
    }

    /// Poll until a condition is met or timeout expires.
    fn wait_for<F: Fn() -> bool>(condition: F, timeout_ms: u64) -> bool {
        let start = std::time::Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        while start.elapsed() < timeout {
            if condition() {
                return true;
            }
            thread::sleep(Duration::from_millis(20));
        }
        condition()
    }

    // -- Wire format --

    #[test]
    fn test_wire_format_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client_thread = thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            let msg = SyncMessage::SyncRequest {
                last_known_op_id: 42,
            };
            send_message(&mut stream, &msg).unwrap();
            recv_message(&mut stream).unwrap()
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        let received = recv_message(&mut server_stream).unwrap();

        if let SyncMessage::SyncRequest { last_known_op_id } = &received {
            assert_eq!(*last_known_op_id, 42);
        } else {
            panic!("Expected SyncRequest, got {:?}", received);
        }

        let response = SyncMessage::AckOperation { op_id: 100 };
        send_message(&mut server_stream, &response).unwrap();

        let client_response = client_thread.join().unwrap();
        if let SyncMessage::AckOperation { op_id } = &client_response {
            assert_eq!(*op_id, 100);
        } else {
            panic!("Expected AckOperation, got {:?}", client_response);
        }
    }

    // -- Server start/stop --

    #[test]
    fn test_server_start_and_stop() {
        let mut server = start_test_server();
        let addr = server.local_addr();
        assert_ne!(addr.port(), 0);
        assert_eq!(server.client_count(), 0);
        server.stop();
    }

    // -- Client connect/disconnect --

    #[test]
    fn test_client_connect_disconnect() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        let mut client = SyncClient::connect(addr).expect("client should connect");
        wait_for(|| server.client_count() >= 1, 2000);

        assert!(client.is_connected());
        assert!(server.client_count() >= 1);

        client.disconnect();
        settle();

        server.stop();
    }

    // -- Operation round-trip --

    #[test]
    fn test_operation_roundtrip_through_server() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        let client_a = SyncClient::connect(addr).expect("client A connect");
        let client_b = SyncClient::connect(addr).expect("client B connect");
        wait_for(|| server.client_count() >= 2, 2000);
        settle();

        let _ = client_a.drain_messages();
        let _ = client_b.drain_messages();

        let op = EditOperation {
            op_id: 0,
            op: EditOp::MovePitch {
                note_id: 1,
                delta_semitones: 3.0,
            },
            author: "alice".to_string(),
            timestamp_ms: 1000,
        };
        client_a.push_operation(op).expect("push should succeed");

        wait_for(|| server.operation_log().len() >= 1, 2000);
        settle();

        let a_msgs = client_a.drain_messages().unwrap();
        let has_ack = a_msgs
            .iter()
            .any(|m| matches!(m, SyncMessage::AckOperation { .. }));
        assert!(has_ack, "Client A should receive an AckOperation");

        let b_msgs = client_b.drain_messages().unwrap();
        let has_sync = b_msgs.iter().any(|m| {
            if let SyncMessage::SyncResponse { operations } = m {
                operations
                    .iter()
                    .any(|o| matches!(&o.op, EditOp::MovePitch { note_id: 1, .. }))
            } else {
                false
            }
        });
        assert!(
            has_sync,
            "Client B should receive the broadcasted operation"
        );

        server.stop();
    }

    // -- Reconnection --

    #[test]
    fn test_reconnection_gets_missed_operations() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        let client_a = SyncClient::connect(addr).expect("client A connect");
        wait_for(|| server.client_count() >= 1, 2000);
        settle();
        let _ = client_a.drain_messages();

        let op = EditOperation {
            op_id: 0,
            op: EditOp::SetAmplitude {
                note_id: 10,
                amplitude: 0.75,
            },
            author: "alice".to_string(),
            timestamp_ms: 2000,
        };
        client_a.push_operation(op).expect("push should succeed");
        wait_for(|| server.operation_log().len() >= 1, 2000);

        let client_b = SyncClient::connect(addr).expect("client B connect");
        settle();

        let b_msgs = client_b.drain_messages().unwrap();
        let found = b_msgs.iter().any(|m| {
            if let SyncMessage::SyncResponse { operations } = m {
                operations.iter().any(|o| {
                    matches!(
                        &o.op,
                        EditOp::SetAmplitude {
                            note_id: 10,
                            ..
                        }
                    )
                })
            } else {
                false
            }
        });
        assert!(
            found,
            "Client B should receive missed operations on reconnect"
        );

        server.stop();
    }

    // -- Multiple clients concurrent editing --

    #[test]
    fn test_multiple_clients_concurrent_edits() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        let client_a = SyncClient::connect(addr).expect("client A connect");
        let client_b = SyncClient::connect(addr).expect("client B connect");
        wait_for(|| server.client_count() >= 2, 2000);
        settle();

        let _ = client_a.drain_messages();
        let _ = client_b.drain_messages();

        let op_a = EditOperation {
            op_id: 0,
            op: EditOp::MovePitch {
                note_id: 1,
                delta_semitones: 2.0,
            },
            author: "alice".to_string(),
            timestamp_ms: 100,
        };
        let op_b = EditOperation {
            op_id: 0,
            op: EditOp::MoveTime {
                note_id: 2,
                delta_seconds: 0.5,
            },
            author: "bob".to_string(),
            timestamp_ms: 100,
        };

        client_a.push_operation(op_a).unwrap();
        client_b.push_operation(op_b).unwrap();

        let got_both = wait_for(|| server.operation_log().len() >= 2, 3000);
        assert!(
            got_both,
            "Server log should have at least 2 ops, got {}",
            server.operation_log().len()
        );

        settle();

        let a_msgs = client_a.drain_messages().unwrap();
        let b_msgs = client_b.drain_messages().unwrap();

        let a_has_ack = a_msgs
            .iter()
            .any(|m| matches!(m, SyncMessage::AckOperation { .. }));
        assert!(a_has_ack, "Client A should get an ack");

        let a_has_bob_op = a_msgs.iter().any(|m| {
            if let SyncMessage::SyncResponse { operations } = m {
                operations.iter().any(|o| o.author == "bob")
            } else {
                false
            }
        });
        assert!(a_has_bob_op, "Client A should see Bob's operation");

        let b_has_ack = b_msgs
            .iter()
            .any(|m| matches!(m, SyncMessage::AckOperation { .. }));
        assert!(b_has_ack, "Client B should get an ack");

        let b_has_alice_op = b_msgs.iter().any(|m| {
            if let SyncMessage::SyncResponse { operations } = m {
                operations.iter().any(|o| o.author == "alice")
            } else {
                false
            }
        });
        assert!(b_has_alice_op, "Client B should see Alice's operation");

        server.stop();
    }

    // -- CollabSession --

    #[test]
    fn test_collab_session_join_and_edit() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        let mut session_a =
            CollabSession::join(addr, "alice".to_string()).expect("join should succeed");
        wait_for(|| server.client_count() >= 1, 2000);
        settle();

        let _ = session_a.sync();

        session_a.apply_local(EditOp::MovePitch {
            note_id: 1,
            delta_semitones: 5.0,
        });
        assert_eq!(session_a.session().pending_ops().len(), 1);

        session_a.sync().expect("sync should succeed");
        wait_for(|| server.operation_log().len() >= 1, 2000);
        settle();

        session_a.sync().expect("second sync should succeed");

        let log = server.operation_log();
        assert!(log.len() >= 1);

        session_a.leave();
        server.stop();
    }

    #[test]
    fn test_collab_session_two_users() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        let mut session_a =
            CollabSession::join(addr, "alice".to_string()).expect("alice join");
        let mut session_b =
            CollabSession::join(addr, "bob".to_string()).expect("bob join");
        wait_for(|| server.client_count() >= 2, 2000);
        settle();

        let _ = session_a.sync();
        let _ = session_b.sync();

        session_a.apply_local(EditOp::SetAmplitude {
            note_id: 1,
            amplitude: 0.6,
        });
        session_a.sync().expect("alice sync");
        wait_for(|| server.operation_log().len() >= 1, 2000);
        settle();

        let remote_count = session_b.sync().expect("bob sync");
        assert!(
            remote_count >= 1 || session_b.session().confirmed_ops().len() >= 1,
            "Bob should see at least one remote operation"
        );

        session_a.leave();
        session_b.leave();
        server.stop();
    }

    // -- Sync request for missed ops --

    #[test]
    fn test_sync_request_returns_missed_ops() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        let client_a = SyncClient::connect(addr).expect("client A");
        wait_for(|| server.client_count() >= 1, 2000);
        settle();
        let _ = client_a.drain_messages();

        for i in 1..=3u64 {
            let op = EditOperation {
                op_id: 0,
                op: EditOp::MovePitch {
                    note_id: i,
                    delta_semitones: i as f64,
                },
                author: "alice".to_string(),
                timestamp_ms: i * 1000,
            };
            client_a.push_operation(op).unwrap();
            let expected = i as usize;
            wait_for(|| server.operation_log().len() >= expected, 2000);
        }

        assert!(server.operation_log().len() >= 3);

        let client_b = SyncClient::connect(addr).expect("client B");
        settle();

        let b_msgs = client_b.drain_messages().unwrap();
        let total_ops: usize = b_msgs
            .iter()
            .filter_map(|m| {
                if let SyncMessage::SyncResponse { operations } = m {
                    Some(operations.len())
                } else {
                    None
                }
            })
            .sum();

        assert!(
            total_ops >= 3,
            "Client B should receive all 3 missed operations, got {}",
            total_ops
        );

        server.stop();
    }
}
