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
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::collab::{
    CollabError, CollabResult, EditOp, EditOperation, EditSession, OperationLog, SyncMessage,
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
/// Blocks until a complete message is available or the connection is closed.
pub fn recv_message(stream: &mut TcpStream) -> NetworkResult<SyncMessage> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(NetworkError::ConnectionClosed);
        }
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
            return Err(NetworkError::Io(e));
        }
        Err(e) => return Err(NetworkError::Io(e)),
    }

    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 {
        return Err(NetworkError::Serialization(
            "Zero-length message".to_string(),
        ));
    }
    // Reject unreasonably large messages (16 MiB limit).
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
// Client handle (server-side per-client bookkeeping)
// ---------------------------------------------------------------------------

/// A handle to a connected client, held by the server.
struct ClientHandle {
    /// Unique client identifier (assigned by the server).
    id: String,
    /// Sender half of a channel used to queue outgoing messages.
    sender: Sender<SyncMessage>,
    /// Whether this client handle is still alive.
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
    /// The address the server is listening on (after bind).
    local_addr: std::net::SocketAddr,
    /// Shared authoritative operation log.
    operation_log: Arc<Mutex<OperationLog>>,
    /// Connected client handles keyed by client ID.
    clients: Arc<Mutex<HashMap<String, ClientHandle>>>,
    /// Whether the server accept loop is running.
    running: Arc<AtomicBool>,
    /// Handle to the accept-loop thread.
    accept_handle: Option<JoinHandle<()>>,
    /// Counter for assigning unique client IDs.
    next_client_id: Arc<Mutex<u64>>,
    /// Optional model snapshot JSON for initial sync.
    model_snapshot: Arc<Mutex<Option<String>>>,
}

impl SyncServer {
    /// Create and start a new sync server listening on the given address.
    ///
    /// The server immediately begins accepting connections in a background thread.
    pub fn start<A: ToSocketAddrs>(addr: A) -> NetworkResult<Self> {
        let listener = TcpListener::bind(addr)?;
        let local_addr = listener.local_addr()?;
        listener.set_nonblocking(false)?;

        let operation_log = Arc::new(Mutex::new(OperationLog::new()));
        let clients: Arc<Mutex<HashMap<String, ClientHandle>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let running = Arc::new(AtomicBool::new(true));
        let next_client_id = Arc::new(Mutex::new(1u64));
        let model_snapshot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        // Clone everything the accept thread needs.
        let log_clone = Arc::clone(&operation_log);
        let clients_clone = Arc::clone(&clients);
        let running_clone = Arc::clone(&running);
        let next_id_clone = Arc::clone(&next_client_id);
        let snapshot_clone = Arc::clone(&model_snapshot);

        // Set a timeout on the listener so we can check the running flag.
        listener
            .set_nonblocking(false)
            .expect("set_nonblocking on listener");

        let accept_handle = thread::spawn(move || {
            // Use a short accept timeout so we can check the `running` flag.
            listener
                .set_nonblocking(true)
                .expect("set_nonblocking on listener");

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
                        // No pending connection; sleep briefly and retry.
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => {
                        // A real error on accept; if we are shutting down, break.
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
        clients.values().filter(|c| c.alive.load(Ordering::Relaxed)).count()
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

        // Drop all client senders to unblock their write loops.
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

    /// Spawn reader and writer threads for a newly connected client.
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

        // Register the client.
        {
            let mut map = clients.lock().unwrap();
            map.insert(
                client_id.clone(),
                ClientHandle {
                    id: client_id.clone(),
                    sender: tx,
                    alive: Arc::clone(&alive),
                },
            );
        }

        // --- Writer thread: drains the channel and sends messages to the client. ---
        let mut write_stream = stream.try_clone().expect("clone TcpStream for writer");
        let alive_w = Arc::clone(&alive);
        let _writer_handle = thread::spawn(move || {
            while alive_w.load(Ordering::Relaxed) {
                match rx.recv_timeout(Duration::from_millis(100)) {
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

        // --- Reader thread: reads messages from the client and processes them. ---
        let mut read_stream = stream;
        let alive_r = Arc::clone(&alive);
        let cid = client_id.clone();
        let _reader_handle = thread::spawn(move || {
            // Set a short read timeout so we can check the alive flag.
            let _ = read_stream.set_read_timeout(Some(Duration::from_millis(200)));

            while alive_r.load(Ordering::Relaxed) && running.load(Ordering::Relaxed) {
                let msg = match recv_message(&mut read_stream) {
                    Ok(m) => m,
                    Err(NetworkError::ConnectionClosed) => {
                        alive_r.store(false, Ordering::Relaxed);
                        break;
                    }
                    Err(NetworkError::Io(ref e))
                        if e.kind() == io::ErrorKind::WouldBlock
                            || e.kind() == io::ErrorKind::TimedOut =>
                    {
                        continue;
                    }
                    Err(_) => {
                        alive_r.store(false, Ordering::Relaxed);
                        break;
                    }
                };

                Self::handle_client_message(
                    &msg,
                    &cid,
                    &operation_log,
                    &clients,
                    &model_snapshot,
                );
            }

            // Remove the client on disconnect.
            let mut map = clients.lock().unwrap();
            map.remove(&cid);
        });
    }

    /// Process a single message received from a client.
    fn handle_client_message(
        msg: &SyncMessage,
        sender_id: &str,
        operation_log: &Arc<Mutex<OperationLog>>,
        clients: &Arc<Mutex<HashMap<String, ClientHandle>>>,
        model_snapshot: &Arc<Mutex<Option<String>>>,
    ) {
        match msg {
            SyncMessage::PushOperation { operation } => {
                // Assign a global sequential ID and add to the authoritative log.
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

                // Send Ack to the sender.
                let ack = SyncMessage::AckOperation {
                    op_id: confirmed_op.op_id,
                };
                Self::send_to_client(clients, sender_id, ack);

                // Broadcast the confirmed op to all *other* clients.
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
                // Client requests a model snapshot.
                let snap = model_snapshot.lock().unwrap();
                let log = operation_log.lock().unwrap();
                let response = SyncMessage::ModelSnapshot {
                    state_json: snap.clone().unwrap_or_else(|| "{}".to_string()),
                    last_op_id: log.last_op_id(),
                };
                Self::send_to_client(clients, sender_id, response);
            }

            // AckOperation and SyncResponse are server-to-client only.
            _ => {}
        }
    }

    /// Send a message to a specific client by ID.
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

    /// Broadcast a message to all clients except the one with `except_id`.
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
    /// The TCP stream used for sending messages (the reader thread has its own clone).
    write_stream: Mutex<TcpStream>,
    /// Whether the background receive loop is running.
    running: Arc<AtomicBool>,
    /// Handle to the background receive thread.
    recv_handle: Option<JoinHandle<()>>,
    /// Receiver for incoming `SyncMessage` values from the background thread.
    incoming: Mutex<Receiver<SyncMessage>>,
    /// The last confirmed operation ID this client knows about.
    last_known_op_id: Mutex<u64>,
}

impl SyncClient {
    /// Connect to a sync server at the given address.
    ///
    /// A background thread is started to receive messages from the server.
    /// After connecting, an initial `SyncRequest` is sent with
    /// `last_known_op_id = 0` to retrieve the full operation history.
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

        // Clone for the reader thread.
        let mut read_stream = stream.try_clone()?;
        let running_clone = Arc::clone(&running);

        let recv_handle = thread::spawn(move || {
            let _ = read_stream.set_read_timeout(Some(Duration::from_millis(200)));

            while running_clone.load(Ordering::Relaxed) {
                match recv_message(&mut read_stream) {
                    Ok(msg) => {
                        if tx.send(msg).is_err() {
                            break;
                        }
                    }
                    Err(NetworkError::ConnectionClosed) => {
                        running_clone.store(false, Ordering::Relaxed);
                        break;
                    }
                    Err(NetworkError::Io(ref e))
                        if e.kind() == io::ErrorKind::WouldBlock
                            || e.kind() == io::ErrorKind::TimedOut =>
                    {
                        continue;
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
    ///
    /// Returns `Ok(None)` if no message is currently available.
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

    /// Receive the next message from the server, blocking until one arrives or
    /// the given timeout elapses.
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
///
/// Usage:
///   1. `CollabSession::join(addr, author)` to connect.
///   2. `apply_local(op)` to make a local edit (optimistically applied).
///   3. `sync()` to push pending ops and receive remote ops.
///   4. `leave()` to disconnect.
pub struct CollabSession {
    /// The underlying sync client.
    client: SyncClient,
    /// The local edit session (tracks pending and confirmed ops).
    session: EditSession,
}

impl CollabSession {
    /// Join a collaborative session by connecting to the server.
    ///
    /// Sends a sync request to retrieve any existing operations.
    pub fn join<A: ToSocketAddrs>(addr: A, author: String) -> NetworkResult<Self> {
        let client = SyncClient::connect(addr)?;
        let session = EditSession::new(author);
        Ok(Self { client, session })
    }

    /// Join with a known last operation ID (for reconnection).
    pub fn rejoin<A: ToSocketAddrs>(
        addr: A,
        author: String,
        last_known_op_id: u64,
    ) -> NetworkResult<Self> {
        let client = SyncClient::connect_with_last_op(addr, last_known_op_id)?;
        let mut session = EditSession::new(author);
        // Set the session's confirmed state to match.
        // The sync response will fill in the gap.
        session.last_confirmed_op_id = last_known_op_id;
        Ok(Self { client, session })
    }

    /// Apply a local edit operation (optimistic application).
    pub fn apply_local(&mut self, op: EditOp) {
        self.session.apply_local(op);
    }

    /// Synchronize with the server: push pending operations and process
    /// incoming confirmed operations.
    ///
    /// Returns the number of remote operations received.
    pub fn sync(&mut self) -> NetworkResult<usize> {
        // Push all pending operations.
        while let Some(msg) = self.session.next_push_message() {
            self.client.send(msg)?;
        }

        // Process all incoming messages.
        let messages = self.client.drain_messages()?;
        let mut remote_count = 0;

        for msg in messages {
            match msg {
                SyncMessage::AckOperation { op_id } => {
                    self.session.handle_ack(op_id);
                    self.client.set_last_known_op_id(op_id);
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

    /// Helper: start a server on an OS-assigned port and return it.
    fn start_test_server() -> SyncServer {
        SyncServer::start("127.0.0.1:0").expect("server should start")
    }

    /// Helper: wait a short time for async operations to settle.
    fn settle() {
        thread::sleep(Duration::from_millis(150));
    }

    // -- Wire format --

    #[test]
    fn test_wire_format_roundtrip() {
        // Create a connected pair using a temporary server.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client_thread = thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            let msg = SyncMessage::SyncRequest {
                last_known_op_id: 42,
            };
            send_message(&mut stream, &msg).unwrap();

            let response = recv_message(&mut stream).unwrap();
            response
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        let received = recv_message(&mut server_stream).unwrap();

        // Verify received message.
        if let SyncMessage::SyncRequest { last_known_op_id } = &received {
            assert_eq!(*last_known_op_id, 42);
        } else {
            panic!("Expected SyncRequest, got {:?}", received);
        }

        // Send a response back.
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
        settle();

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

        // Client A sends an operation.
        let client_a = SyncClient::connect(addr).expect("client A connect");
        settle();

        // Client B connects.
        let client_b = SyncClient::connect(addr).expect("client B connect");
        settle();

        // Drain the initial SyncResponse from the server.
        let _ = client_a.drain_messages();
        let _ = client_b.drain_messages();

        // Client A pushes an operation.
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

        // Wait for the server to process and broadcast.
        settle();

        // Client A should get an Ack.
        let a_msgs = client_a.drain_messages().unwrap();
        let has_ack = a_msgs.iter().any(|m| matches!(m, SyncMessage::AckOperation { .. }));
        assert!(has_ack, "Client A should receive an AckOperation");

        // Client B should get a SyncResponse with the confirmed operation.
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

        // The server's log should have the operation.
        let log = server.operation_log();
        assert!(log.len() >= 1);

        server.stop();
    }

    // -- Reconnection --

    #[test]
    fn test_reconnection_gets_missed_operations() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        // Client A connects and pushes an operation.
        let client_a = SyncClient::connect(addr).expect("client A connect");
        settle();

        // Drain initial sync.
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
        settle();

        // Now client B connects *after* the operation was pushed.
        // It sends SyncRequest with last_known_op_id=0 on connect.
        let client_b = SyncClient::connect(addr).expect("client B connect");
        settle();

        // Client B should receive the missed operation in the initial SyncResponse.
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
        assert!(found, "Client B should receive missed operations on reconnect");

        server.stop();
    }

    // -- Multiple clients concurrent editing --

    #[test]
    fn test_multiple_clients_concurrent_edits() {
        let mut server = start_test_server();
        let addr = server.local_addr();

        let client_a = SyncClient::connect(addr).expect("client A connect");
        let client_b = SyncClient::connect(addr).expect("client B connect");
        settle();

        // Drain initial messages.
        let _ = client_a.drain_messages();
        let _ = client_b.drain_messages();

        // Both clients push operations concurrently.
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
        settle();

        // The server should have both operations.
        let log = server.operation_log();
        assert_eq!(log.len(), 2);

        // Both clients should eventually see both operations.
        let a_msgs = client_a.drain_messages().unwrap();
        let b_msgs = client_b.drain_messages().unwrap();

        // Client A should get an ack for its own op + a sync for B's op.
        let a_has_ack = a_msgs.iter().any(|m| matches!(m, SyncMessage::AckOperation { .. }));
        assert!(a_has_ack, "Client A should get an ack");

        let a_has_bob_op = a_msgs.iter().any(|m| {
            if let SyncMessage::SyncResponse { operations } = m {
                operations.iter().any(|o| o.author == "bob")
            } else {
                false
            }
        });
        assert!(a_has_bob_op, "Client A should see Bob's operation");

        // Client B should get an ack for its own op + a sync for A's op.
        let b_has_ack = b_msgs.iter().any(|m| matches!(m, SyncMessage::AckOperation { .. }));
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
        settle();

        // Apply a local edit.
        session_a.apply_local(EditOp::MovePitch {
            note_id: 1,
            delta_semitones: 5.0,
        });
        assert_eq!(session_a.session().pending_ops().len(), 1);

        // Sync to push the edit.
        session_a.sync().expect("sync should succeed");
        settle();

        // After sync, the pending op should have been pushed. Try syncing again
        // to process the ack.
        session_a.sync().expect("second sync should succeed");

        // The server should have the operation.
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
        settle();

        // Drain initial sync.
        let _ = session_a.sync();
        let _ = session_b.sync();

        // Alice edits.
        session_a.apply_local(EditOp::SetAmplitude {
            note_id: 1,
            amplitude: 0.6,
        });
        session_a.sync().expect("alice sync");
        settle();

        // Bob syncs and should see Alice's operation.
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

        // Push some ops via client A.
        let client_a = SyncClient::connect(addr).expect("client A");
        settle();
        let _ = client_a.drain_messages();

        for i in 1..=3 {
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
            settle();
        }

        // Client B connects and asks for ops since 0.
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
