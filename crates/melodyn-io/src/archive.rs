//! Serialization archive system: Binary Key-Value, XML, and File Archive Container.
//!
//! Implements `GNBinaryKeyValueArchive`, `GNXMLKeyValueArchive`, and `GNFileArchive`
//! from the Melodyne engine. Provides typed key-value storage with chunked binary
//! format, XML serialization with namespace support, and a multi-entry file container
//! with compression and CRC32 integrity checks.

use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use std::collections::BTreeMap;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from archive operations.
#[derive(Debug, Error)]
pub enum ArchiveError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid magic bytes")]
    InvalidMagic,
    #[error("Unsupported archive version: {0}")]
    UnsupportedVersion(u32),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("Type mismatch for key '{0}': expected {1}")]
    TypeMismatch(String, &'static str),
    #[error("CRC32 mismatch: expected {expected:#010x}, got {actual:#010x}")]
    CrcMismatch { expected: u32, actual: u32 },
    #[error("Invalid chunk type: {0:?}")]
    InvalidChunkType([u8; 4]),
    #[error("Entry not found: {0}")]
    EntryNotFound(String),
    #[error("Decompression error: {0}")]
    DecompressionError(String),
    #[error("XML error: {0}")]
    XmlError(String),
    #[error("Data truncated")]
    Truncated,
}

pub type ArchiveResult<T> = Result<T, ArchiveError>;

// ---------------------------------------------------------------------------
// CRC32 (IEEE / ITU-T V.42, same polynomial as zlib)
// ---------------------------------------------------------------------------

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// Compute CRC32 over a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// Value types for the key-value archive
// ---------------------------------------------------------------------------

/// A typed value stored in a key-value archive.
#[derive(Debug, Clone, PartialEq)]
pub enum ArchiveValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
    Nested(BinaryKeyValueArchive),
}

impl ArchiveValue {
    fn type_tag(&self) -> u8 {
        match self {
            Self::I32(_) => 1,
            Self::I64(_) => 2,
            Self::F32(_) => 3,
            Self::F64(_) => 4,
            Self::Bool(_) => 5,
            Self::Str(_) => 6,
            Self::Bytes(_) => 7,
            Self::Nested(_) => 8,
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::Bool(_) => "bool",
            Self::Str(_) => "String",
            Self::Bytes(_) => "Vec<u8>",
            Self::Nested(_) => "Nested",
        }
    }
}

// ---------------------------------------------------------------------------
// Binary Key-Value Archive
// ---------------------------------------------------------------------------

/// Magic bytes for binary key-value archive: "BKVA".
const BKVA_MAGIC: [u8; 4] = *b"BKVA";
/// Current format version.
const BKVA_VERSION: u32 = 1;

/// A binary key-value archive with typed values, chunked storage, and CRC32 integrity.
///
/// Each key-value pair is stored as a chunk:
///   [4-byte chunk type][4-byte key length][key bytes][1-byte type tag][value bytes][4-byte CRC32]
///
/// The chunk type is "KVPR" for key-value pair, "NEST" for nested archives.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryKeyValueArchive {
    entries: BTreeMap<String, ArchiveValue>,
}

impl Default for BinaryKeyValueArchive {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryKeyValueArchive {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    // -- Setters --

    pub fn set_i32(&mut self, key: &str, value: i32) {
        self.entries.insert(key.to_string(), ArchiveValue::I32(value));
    }

    pub fn set_i64(&mut self, key: &str, value: i64) {
        self.entries.insert(key.to_string(), ArchiveValue::I64(value));
    }

    pub fn set_f32(&mut self, key: &str, value: f32) {
        self.entries.insert(key.to_string(), ArchiveValue::F32(value));
    }

    pub fn set_f64(&mut self, key: &str, value: f64) {
        self.entries.insert(key.to_string(), ArchiveValue::F64(value));
    }

    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.entries
            .insert(key.to_string(), ArchiveValue::Bool(value));
    }

    pub fn set_string(&mut self, key: &str, value: &str) {
        self.entries
            .insert(key.to_string(), ArchiveValue::Str(value.to_string()));
    }

    pub fn set_bytes(&mut self, key: &str, value: Vec<u8>) {
        self.entries
            .insert(key.to_string(), ArchiveValue::Bytes(value));
    }

    pub fn set_nested(&mut self, key: &str, value: BinaryKeyValueArchive) {
        self.entries
            .insert(key.to_string(), ArchiveValue::Nested(value));
    }

    // -- Getters --

    pub fn get(&self, key: &str) -> Option<&ArchiveValue> {
        self.entries.get(key)
    }

    pub fn get_i32(&self, key: &str) -> ArchiveResult<i32> {
        match self.entries.get(key) {
            Some(ArchiveValue::I32(v)) => Ok(*v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_i64(&self, key: &str) -> ArchiveResult<i64> {
        match self.entries.get(key) {
            Some(ArchiveValue::I64(v)) => Ok(*v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_f32(&self, key: &str) -> ArchiveResult<f32> {
        match self.entries.get(key) {
            Some(ArchiveValue::F32(v)) => Ok(*v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_f64(&self, key: &str) -> ArchiveResult<f64> {
        match self.entries.get(key) {
            Some(ArchiveValue::F64(v)) => Ok(*v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_bool(&self, key: &str) -> ArchiveResult<bool> {
        match self.entries.get(key) {
            Some(ArchiveValue::Bool(v)) => Ok(*v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_string(&self, key: &str) -> ArchiveResult<&str> {
        match self.entries.get(key) {
            Some(ArchiveValue::Str(v)) => Ok(v.as_str()),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_bytes(&self, key: &str) -> ArchiveResult<&[u8]> {
        match self.entries.get(key) {
            Some(ArchiveValue::Bytes(v)) => Ok(v.as_slice()),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_nested(&self, key: &str) -> ArchiveResult<&BinaryKeyValueArchive> {
        match self.entries.get(key) {
            Some(ArchiveValue::Nested(v)) => Ok(v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    /// Return all keys in sorted order.
    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the archive is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -- Serialization --

    /// Serialize to binary format with header, chunks, and CRC32 integrity.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> ArchiveResult<()> {
        writer.write_all(&BKVA_MAGIC)?;
        writer.write_all(&BKVA_VERSION.to_le_bytes())?;
        writer.write_all(&(self.entries.len() as u32).to_le_bytes())?;

        for (key, value) in &self.entries {
            let chunk_type = match value {
                ArchiveValue::Nested(_) => *b"NEST",
                _ => *b"KVPR",
            };
            // Build chunk payload: key_len(4) + key + type_tag(1) + value_bytes
            let mut payload = Vec::new();
            let key_bytes = key.as_bytes();
            payload.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(key_bytes);
            payload.push(value.type_tag());
            write_value_bytes(&mut payload, value)?;

            let checksum = crc32(&payload);
            writer.write_all(&chunk_type)?;
            writer.write_all(&(payload.len() as u32).to_le_bytes())?;
            writer.write_all(&payload)?;
            writer.write_all(&checksum.to_le_bytes())?;
        }
        Ok(())
    }

    /// Serialize to a byte vector.
    pub fn to_bytes(&self) -> ArchiveResult<Vec<u8>> {
        let mut buf = Vec::new();
        self.write_to(&mut buf)?;
        Ok(buf)
    }

    /// Deserialize from a reader.
    pub fn read_from<R: Read>(reader: &mut R) -> ArchiveResult<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|_| ArchiveError::Truncated)?;
        if magic != BKVA_MAGIC {
            return Err(ArchiveError::InvalidMagic);
        }
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf).map_err(|_| ArchiveError::Truncated)?;
        let version = u32::from_le_bytes(version_buf);
        if version != BKVA_VERSION {
            return Err(ArchiveError::UnsupportedVersion(version));
        }
        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf).map_err(|_| ArchiveError::Truncated)?;
        let count = u32::from_le_bytes(count_buf) as usize;

        let mut archive = Self::new();
        for _ in 0..count {
            let mut chunk_type = [0u8; 4];
            reader.read_exact(&mut chunk_type).map_err(|_| ArchiveError::Truncated)?;
            if chunk_type != *b"KVPR" && chunk_type != *b"NEST" {
                return Err(ArchiveError::InvalidChunkType(chunk_type));
            }
            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf).map_err(|_| ArchiveError::Truncated)?;
            let payload_len = u32::from_le_bytes(len_buf) as usize;

            let mut payload = vec![0u8; payload_len];
            reader.read_exact(&mut payload).map_err(|_| ArchiveError::Truncated)?;

            let mut crc_buf = [0u8; 4];
            reader.read_exact(&mut crc_buf).map_err(|_| ArchiveError::Truncated)?;
            let expected_crc = u32::from_le_bytes(crc_buf);
            let actual_crc = crc32(&payload);
            if expected_crc != actual_crc {
                return Err(ArchiveError::CrcMismatch {
                    expected: expected_crc,
                    actual: actual_crc,
                });
            }

            let mut cursor = Cursor::new(&payload);
            let mut klen_buf = [0u8; 4];
            cursor.read_exact(&mut klen_buf)?;
            let key_len = u32::from_le_bytes(klen_buf) as usize;
            let mut key_bytes = vec![0u8; key_len];
            cursor.read_exact(&mut key_bytes)?;
            let key = String::from_utf8_lossy(&key_bytes).into_owned();

            let mut tag_buf = [0u8; 1];
            cursor.read_exact(&mut tag_buf)?;
            let value = read_value_bytes(&mut cursor, tag_buf[0])?;
            archive.entries.insert(key, value);
        }
        Ok(archive)
    }

    /// Deserialize from a byte slice.
    pub fn from_bytes(data: &[u8]) -> ArchiveResult<Self> {
        let mut cursor = Cursor::new(data);
        Self::read_from(&mut cursor)
    }
}

/// Write value bytes (excluding key and type tag).
fn write_value_bytes<W: Write>(writer: &mut W, value: &ArchiveValue) -> ArchiveResult<()> {
    match value {
        ArchiveValue::I32(v) => writer.write_all(&v.to_le_bytes())?,
        ArchiveValue::I64(v) => writer.write_all(&v.to_le_bytes())?,
        ArchiveValue::F32(v) => writer.write_all(&v.to_le_bytes())?,
        ArchiveValue::F64(v) => writer.write_all(&v.to_le_bytes())?,
        ArchiveValue::Bool(v) => writer.write_all(&[if *v { 1 } else { 0 }])?,
        ArchiveValue::Str(v) => {
            let bytes = v.as_bytes();
            writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
            writer.write_all(bytes)?;
        }
        ArchiveValue::Bytes(v) => {
            writer.write_all(&(v.len() as u32).to_le_bytes())?;
            writer.write_all(v)?;
        }
        ArchiveValue::Nested(v) => {
            let nested_bytes = v.to_bytes()?;
            writer.write_all(&(nested_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(&nested_bytes)?;
        }
    }
    Ok(())
}

/// Read value bytes given a type tag.
fn read_value_bytes<R: Read>(reader: &mut R, tag: u8) -> ArchiveResult<ArchiveValue> {
    match tag {
        1 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(ArchiveValue::I32(i32::from_le_bytes(buf)))
        }
        2 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(ArchiveValue::I64(i64::from_le_bytes(buf)))
        }
        3 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(ArchiveValue::F32(f32::from_le_bytes(buf)))
        }
        4 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(ArchiveValue::F64(f64::from_le_bytes(buf)))
        }
        5 => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(ArchiveValue::Bool(buf[0] != 0))
        }
        6 => {
            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;
            Ok(ArchiveValue::Str(String::from_utf8_lossy(&data).into_owned()))
        }
        7 => {
            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;
            Ok(ArchiveValue::Bytes(data))
        }
        8 => {
            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;
            let nested = BinaryKeyValueArchive::from_bytes(&data)?;
            Ok(ArchiveValue::Nested(nested))
        }
        _ => Err(ArchiveError::TypeMismatch(
            format!("tag={tag}"),
            "valid type tag (1-8)",
        )),
    }
}

// ---------------------------------------------------------------------------
// XML Key-Value Archive
// ---------------------------------------------------------------------------

/// XML namespace for the archive format.
pub const XML_ARCHIVE_NS: &str = "https://dev.celemony.com/xmlns/2022/archive";
/// XML namespace for the model data.
pub const XML_MODEL_NS: &str = "https://dev.celemony.com/xmlns/2022/model";

/// An XML key-value archive with namespace support, human-readable output.
#[derive(Debug, Clone, PartialEq)]
pub struct XmlKeyValueArchive {
    entries: BTreeMap<String, ArchiveValue>,
    /// Optional archive-level metadata (e.g., format version).
    pub version: u32,
}

impl Default for XmlKeyValueArchive {
    fn default() -> Self {
        Self::new()
    }
}

impl XmlKeyValueArchive {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            version: 1,
        }
    }

    /// Set a value. Same type system as the binary archive.
    pub fn set(&mut self, key: &str, value: ArchiveValue) {
        self.entries.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<&ArchiveValue> {
        self.entries.get(key)
    }

    pub fn get_i32(&self, key: &str) -> ArchiveResult<i32> {
        match self.entries.get(key) {
            Some(ArchiveValue::I32(v)) => Ok(*v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_f64(&self, key: &str) -> ArchiveResult<f64> {
        match self.entries.get(key) {
            Some(ArchiveValue::F64(v)) => Ok(*v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_string(&self, key: &str) -> ArchiveResult<&str> {
        match self.entries.get(key) {
            Some(ArchiveValue::Str(v)) => Ok(v.as_str()),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn get_bool(&self, key: &str) -> ArchiveResult<bool> {
        match self.entries.get(key) {
            Some(ArchiveValue::Bool(v)) => Ok(*v),
            Some(other) => Err(ArchiveError::TypeMismatch(key.to_string(), other.type_name())),
            None => Err(ArchiveError::KeyNotFound(key.to_string())),
        }
    }

    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize to a human-readable XML string with namespace declarations.
    pub fn to_xml_string(&self) -> ArchiveResult<String> {
        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str(&format!(
            "<archive xmlns=\"{}\" xmlns:model=\"{}\" version=\"{}\">\n",
            XML_ARCHIVE_NS, XML_MODEL_NS, self.version
        ));
        for (key, value) in &self.entries {
            write_xml_entry(&mut out, key, value, 1);
        }
        out.push_str("</archive>\n");
        Ok(out)
    }

    /// Deserialize from an XML string.
    pub fn from_xml_string(xml: &str) -> ArchiveResult<Self> {
        let mut archive = Self::new();
        // Simple line-based parser for the specific format we produce.
        let mut lines = xml.lines().peekable();

        // Skip XML declaration.
        while let Some(line) = lines.peek() {
            let trimmed = line.trim();
            if trimmed.starts_with("<?xml") || trimmed.is_empty() {
                lines.next();
                continue;
            }
            break;
        }

        // Parse <archive ...> tag for version.
        if let Some(line) = lines.next() {
            let trimmed = line.trim();
            if let Some(pos) = trimmed.find("version=\"") {
                let rest = &trimmed[pos + 9..];
                if let Some(end) = rest.find('"') {
                    if let Ok(v) = rest[..end].parse::<u32>() {
                        archive.version = v;
                    }
                }
            }
        }

        // Parse entries until </archive>.
        parse_xml_entries(&mut lines, &mut archive.entries)?;
        Ok(archive)
    }
}

fn write_xml_entry(out: &mut String, key: &str, value: &ArchiveValue, indent: usize) {
    let pad: String = "  ".repeat(indent);
    match value {
        ArchiveValue::I32(v) => {
            out.push_str(&format!("{pad}<entry key=\"{key}\" type=\"i32\">{v}</entry>\n"));
        }
        ArchiveValue::I64(v) => {
            out.push_str(&format!("{pad}<entry key=\"{key}\" type=\"i64\">{v}</entry>\n"));
        }
        ArchiveValue::F32(v) => {
            out.push_str(&format!("{pad}<entry key=\"{key}\" type=\"f32\">{v}</entry>\n"));
        }
        ArchiveValue::F64(v) => {
            out.push_str(&format!("{pad}<entry key=\"{key}\" type=\"f64\">{v}</entry>\n"));
        }
        ArchiveValue::Bool(v) => {
            out.push_str(&format!("{pad}<entry key=\"{key}\" type=\"bool\">{v}</entry>\n"));
        }
        ArchiveValue::Str(v) => {
            let escaped = xml_escape(v);
            out.push_str(&format!("{pad}<entry key=\"{key}\" type=\"string\">{escaped}</entry>\n"));
        }
        ArchiveValue::Bytes(v) => {
            let hex: String = v.iter().map(|b| format!("{b:02x}")).collect();
            out.push_str(&format!("{pad}<entry key=\"{key}\" type=\"bytes\">{hex}</entry>\n"));
        }
        ArchiveValue::Nested(nested) => {
            out.push_str(&format!("{pad}<entry key=\"{key}\" type=\"nested\">\n"));
            for (k, v) in &nested.entries {
                write_xml_entry(out, k, v, indent + 1);
            }
            out.push_str(&format!("{pad}</entry>\n"));
        }
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Parse `<entry ...>` elements from lines into the entries map.
fn parse_xml_entries<'a, I>(
    lines: &mut std::iter::Peekable<I>,
    entries: &mut BTreeMap<String, ArchiveValue>,
) -> ArchiveResult<()>
where
    I: Iterator<Item = &'a str>,
{
    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.starts_with("</") {
            // End of parent element.
            lines.next();
            return Ok(());
        }
        if !trimmed.starts_with("<entry ") {
            lines.next();
            continue;
        }
        let trimmed = trimmed.to_string();
        lines.next();

        // Extract key and type attributes.
        let key = extract_attr(&trimmed, "key").unwrap_or_default();
        let type_str = extract_attr(&trimmed, "type").unwrap_or_default();

        if type_str == "nested" {
            // Recurse for nested entries.
            let mut nested_entries = BTreeMap::new();
            parse_xml_entries(lines, &mut nested_entries)?;
            let nested = BinaryKeyValueArchive {
                entries: nested_entries,
            };
            entries.insert(key, ArchiveValue::Nested(nested));
        } else {
            // Self-closing or inline value.
            let content = extract_content(&trimmed);
            let value = parse_xml_value(&type_str, &content)?;
            entries.insert(key, value);
        }
    }
    Ok(())
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let pos = tag.find(&needle)?;
    let rest = &tag[pos + needle.len()..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_content(tag: &str) -> String {
    // Find content between > and </entry>
    if let Some(start) = tag.find('>') {
        let rest = &tag[start + 1..];
        if let Some(end) = rest.find("</entry>") {
            return rest[..end].to_string();
        }
    }
    String::new()
}

fn parse_xml_value(type_str: &str, content: &str) -> ArchiveResult<ArchiveValue> {
    match type_str {
        "i32" => {
            let v: i32 = content.trim().parse().map_err(|_| {
                ArchiveError::XmlError(format!("invalid i32: {content}"))
            })?;
            Ok(ArchiveValue::I32(v))
        }
        "i64" => {
            let v: i64 = content.trim().parse().map_err(|_| {
                ArchiveError::XmlError(format!("invalid i64: {content}"))
            })?;
            Ok(ArchiveValue::I64(v))
        }
        "f32" => {
            let v: f32 = content.trim().parse().map_err(|_| {
                ArchiveError::XmlError(format!("invalid f32: {content}"))
            })?;
            Ok(ArchiveValue::F32(v))
        }
        "f64" => {
            let v: f64 = content.trim().parse().map_err(|_| {
                ArchiveError::XmlError(format!("invalid f64: {content}"))
            })?;
            Ok(ArchiveValue::F64(v))
        }
        "bool" => {
            let v = content.trim() == "true";
            Ok(ArchiveValue::Bool(v))
        }
        "string" => Ok(ArchiveValue::Str(xml_unescape(content))),
        "bytes" => {
            let hex = content.trim();
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .filter_map(|i| {
                    if i + 2 <= hex.len() {
                        u8::from_str_radix(&hex[i..i + 2], 16).ok()
                    } else {
                        None
                    }
                })
                .collect();
            Ok(ArchiveValue::Bytes(bytes))
        }
        _ => Err(ArchiveError::XmlError(format!("unknown type: {type_str}"))),
    }
}

// ---------------------------------------------------------------------------
// File Archive Container
// ---------------------------------------------------------------------------

/// Magic for the file archive container.
const FILE_ARCHIVE_MAGIC: [u8; 8] = *b"FmpD@-\x00\x00";
/// File archive version.
const FILE_ARCHIVE_VERSION: u32 = 1;

/// Compression method for file archive entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
    None = 0,
    Deflate = 1,
}

impl CompressionMethod {
    fn from_u8(v: u8) -> ArchiveResult<Self> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Deflate),
            _ => Err(ArchiveError::DecompressionError(format!(
                "unknown compression method: {v}"
            ))),
        }
    }
}

/// An entry in the file archive directory.
#[derive(Debug, Clone)]
struct FileArchiveEntry {
    name: String,
    offset: u64,
    compressed_size: u64,
    uncompressed_size: u64,
    compression: CompressionMethod,
    crc32: u32,
}

/// A multi-entry file archive container with compression, CRC32, and an end-of-file
/// directory for random access.
///
/// Layout:
///   [8-byte magic][4-byte version]
///   [entry data blocks...]
///   [directory entries]
///   [4-byte directory offset (from start)][4-byte entry count]
pub struct FileArchive {
    entries: Vec<(FileArchiveEntry, Vec<u8>)>,
}

impl Default for FileArchive {
    fn default() -> Self {
        Self::new()
    }
}

impl FileArchive {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an entry with optional compression.
    pub fn add_entry(&mut self, name: &str, data: &[u8], compress: bool) {
        let crc = crc32(data);
        let (compressed_data, compression) = if compress {
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(data).expect("deflate write");
            let compressed = encoder.finish().expect("deflate finish");
            if compressed.len() < data.len() {
                (compressed, CompressionMethod::Deflate)
            } else {
                (data.to_vec(), CompressionMethod::None)
            }
        } else {
            (data.to_vec(), CompressionMethod::None)
        };

        let entry = FileArchiveEntry {
            name: name.to_string(),
            offset: 0, // Filled during write.
            compressed_size: compressed_data.len() as u64,
            uncompressed_size: data.len() as u64,
            compression,
            crc32: crc,
        };
        self.entries.push((entry, compressed_data));
    }

    /// Write the complete archive to a writer.
    pub fn write_to<W: Write>(&mut self, writer: &mut W) -> ArchiveResult<()> {
        writer.write_all(&FILE_ARCHIVE_MAGIC)?;
        writer.write_all(&FILE_ARCHIVE_VERSION.to_le_bytes())?;

        // Track position: magic(8) + version(4) = 12 bytes already written.
        let mut pos: u64 = 12;

        // Write entry data blocks.
        for (entry, data) in &mut self.entries {
            entry.offset = pos;
            writer.write_all(data)?;
            pos += data.len() as u64;
        }

        // Write the directory.
        let dir_offset = pos;
        for (entry, _) in &self.entries {
            let name_bytes = entry.name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;
            writer.write_all(&entry.offset.to_le_bytes())?;
            writer.write_all(&entry.compressed_size.to_le_bytes())?;
            writer.write_all(&entry.uncompressed_size.to_le_bytes())?;
            writer.write_all(&[entry.compression as u8])?;
            writer.write_all(&entry.crc32.to_le_bytes())?;
        }

        // Write directory footer: offset + count.
        writer.write_all(&dir_offset.to_le_bytes())?;
        writer.write_all(&(self.entries.len() as u32).to_le_bytes())?;
        Ok(())
    }

    /// Write to a byte vector.
    pub fn to_bytes(&mut self) -> ArchiveResult<Vec<u8>> {
        let mut buf = Vec::new();
        self.write_to(&mut buf)?;
        Ok(buf)
    }

    /// Read a file archive from a seekable reader.
    pub fn read_from<R: Read + Seek>(reader: &mut R) -> ArchiveResult<Self> {
        // Read and verify magic.
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).map_err(|_| ArchiveError::Truncated)?;
        if magic != FILE_ARCHIVE_MAGIC {
            return Err(ArchiveError::InvalidMagic);
        }

        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf).map_err(|_| ArchiveError::Truncated)?;
        let version = u32::from_le_bytes(version_buf);
        if version != FILE_ARCHIVE_VERSION {
            return Err(ArchiveError::UnsupportedVersion(version));
        }

        // Read directory footer from the end of the stream.
        reader.seek(SeekFrom::End(-12))?;
        let mut dir_offset_buf = [0u8; 8];
        reader.read_exact(&mut dir_offset_buf)?;
        let dir_offset = u64::from_le_bytes(dir_offset_buf);

        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;

        // Read directory entries.
        reader.seek(SeekFrom::Start(dir_offset))?;
        let mut dir_entries = Vec::with_capacity(count);
        for _ in 0..count {
            let mut name_len_buf = [0u8; 4];
            reader.read_exact(&mut name_len_buf)?;
            let name_len = u32::from_le_bytes(name_len_buf) as usize;
            let mut name_bytes = vec![0u8; name_len];
            reader.read_exact(&mut name_bytes)?;
            let name = String::from_utf8_lossy(&name_bytes).into_owned();

            let mut offset_buf = [0u8; 8];
            reader.read_exact(&mut offset_buf)?;
            let offset = u64::from_le_bytes(offset_buf);

            let mut cs_buf = [0u8; 8];
            reader.read_exact(&mut cs_buf)?;
            let compressed_size = u64::from_le_bytes(cs_buf);

            let mut us_buf = [0u8; 8];
            reader.read_exact(&mut us_buf)?;
            let uncompressed_size = u64::from_le_bytes(us_buf);

            let mut comp_buf = [0u8; 1];
            reader.read_exact(&mut comp_buf)?;
            let compression = CompressionMethod::from_u8(comp_buf[0])?;

            let mut crc_buf = [0u8; 4];
            reader.read_exact(&mut crc_buf)?;
            let entry_crc = u32::from_le_bytes(crc_buf);

            dir_entries.push(FileArchiveEntry {
                name,
                offset,
                compressed_size,
                uncompressed_size,
                compression,
                crc32: entry_crc,
            });
        }

        // Read entry data.
        let mut archive = FileArchive::new();
        for entry in dir_entries {
            reader.seek(SeekFrom::Start(entry.offset))?;
            let mut data = vec![0u8; entry.compressed_size as usize];
            reader.read_exact(&mut data)?;
            archive.entries.push((entry, data));
        }

        Ok(archive)
    }

    /// Read from a byte slice.
    pub fn from_bytes(data: &[u8]) -> ArchiveResult<Self> {
        let mut cursor = Cursor::new(data);
        Self::read_from(&mut cursor)
    }

    /// List all entry names.
    pub fn entry_names(&self) -> Vec<&str> {
        self.entries.iter().map(|(e, _)| e.name.as_str()).collect()
    }

    /// Read and decompress an entry by name.
    pub fn read_entry(&self, name: &str) -> ArchiveResult<Vec<u8>> {
        let (entry, data) = self
            .entries
            .iter()
            .find(|(e, _)| e.name == name)
            .ok_or_else(|| ArchiveError::EntryNotFound(name.to_string()))?;

        let decompressed = match entry.compression {
            CompressionMethod::None => data.clone(),
            CompressionMethod::Deflate => {
                let mut decoder = DeflateDecoder::new(data.as_slice());
                let mut out = Vec::with_capacity(entry.uncompressed_size as usize);
                decoder.read_to_end(&mut out).map_err(|e| {
                    ArchiveError::DecompressionError(format!("deflate: {e}"))
                })?;
                out
            }
        };

        // Verify CRC32.
        let actual_crc = crc32(&decompressed);
        if actual_crc != entry.crc32 {
            return Err(ArchiveError::CrcMismatch {
                expected: entry.crc32,
                actual: actual_crc,
            });
        }

        Ok(decompressed)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the archive has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_known_value() {
        // "123456789" should produce CRC32 = 0xCBF43926
        let crc = crc32(b"123456789");
        assert_eq!(crc, 0xCBF4_3926);
    }

    #[test]
    fn test_binary_archive_roundtrip_all_types() {
        let mut archive = BinaryKeyValueArchive::new();
        archive.set_i32("int32", -42);
        archive.set_i64("int64", 1_000_000_000_000i64);
        archive.set_f32("float32", std::f32::consts::PI);
        archive.set_f64("float64", std::f64::consts::E);
        archive.set_bool("flag", true);
        archive.set_string("greeting", "Hello, Melodyn!");
        archive.set_bytes("raw", vec![0xDE, 0xAD, 0xBE, 0xEF]);

        let bytes = archive.to_bytes().unwrap();
        let loaded = BinaryKeyValueArchive::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.get_i32("int32").unwrap(), -42);
        assert_eq!(loaded.get_i64("int64").unwrap(), 1_000_000_000_000i64);
        assert!((loaded.get_f32("float32").unwrap() - std::f32::consts::PI).abs() < 1e-7);
        assert!((loaded.get_f64("float64").unwrap() - std::f64::consts::E).abs() < 1e-15);
        assert!(loaded.get_bool("flag").unwrap());
        assert_eq!(loaded.get_string("greeting").unwrap(), "Hello, Melodyn!");
        assert_eq!(loaded.get_bytes("raw").unwrap(), &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_binary_archive_nested() {
        let mut inner = BinaryKeyValueArchive::new();
        inner.set_i32("depth", 1);
        inner.set_string("label", "inner");

        let mut outer = BinaryKeyValueArchive::new();
        outer.set_string("label", "outer");
        outer.set_nested("child", inner);

        let bytes = outer.to_bytes().unwrap();
        let loaded = BinaryKeyValueArchive::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.get_string("label").unwrap(), "outer");
        let child = loaded.get_nested("child").unwrap();
        assert_eq!(child.get_i32("depth").unwrap(), 1);
        assert_eq!(child.get_string("label").unwrap(), "inner");
    }

    #[test]
    fn test_binary_archive_key_not_found() {
        let archive = BinaryKeyValueArchive::new();
        assert!(archive.get_i32("missing").is_err());
    }

    #[test]
    fn test_binary_archive_type_mismatch() {
        let mut archive = BinaryKeyValueArchive::new();
        archive.set_i32("num", 42);
        assert!(archive.get_string("num").is_err());
    }

    #[test]
    fn test_binary_archive_corrupt_crc() {
        let mut archive = BinaryKeyValueArchive::new();
        archive.set_i32("val", 100);
        let mut bytes = archive.to_bytes().unwrap();

        // Corrupt a byte in the payload (somewhere after the header).
        if bytes.len() > 20 {
            bytes[18] ^= 0xFF;
        }
        let result = BinaryKeyValueArchive::from_bytes(&bytes);
        assert!(result.is_err());
        if let Err(ArchiveError::CrcMismatch { .. }) = result {
            // Expected.
        } else {
            panic!("Expected CrcMismatch error, got: {:?}", result);
        }
    }

    #[test]
    fn test_binary_archive_invalid_magic() {
        let data = b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00";
        let result = BinaryKeyValueArchive::from_bytes(data);
        assert!(matches!(result, Err(ArchiveError::InvalidMagic)));
    }

    #[test]
    fn test_xml_archive_roundtrip() {
        let mut archive = XmlKeyValueArchive::new();
        archive.set("count", ArchiveValue::I32(42));
        archive.set("ratio", ArchiveValue::F64(3.14159));
        archive.set("name", ArchiveValue::Str("test <project>".to_string()));
        archive.set("active", ArchiveValue::Bool(true));
        archive.set("data", ArchiveValue::Bytes(vec![0xCA, 0xFE]));

        let xml = archive.to_xml_string().unwrap();
        assert!(xml.contains(XML_ARCHIVE_NS));
        assert!(xml.contains(XML_MODEL_NS));

        let loaded = XmlKeyValueArchive::from_xml_string(&xml).unwrap();
        assert_eq!(loaded.get_i32("count").unwrap(), 42);
        assert!((loaded.get_f64("ratio").unwrap() - 3.14159).abs() < 1e-5);
        assert_eq!(loaded.get_string("name").unwrap(), "test <project>");
        assert!(loaded.get_bool("active").unwrap());
        if let Some(ArchiveValue::Bytes(b)) = loaded.get("data") {
            assert_eq!(b, &[0xCA, 0xFE]);
        } else {
            panic!("Expected bytes");
        }
    }

    #[test]
    fn test_xml_archive_nested() {
        let mut inner = BinaryKeyValueArchive::new();
        inner.set_i32("x", 10);

        let mut archive = XmlKeyValueArchive::new();
        archive.set("child", ArchiveValue::Nested(inner));

        let xml = archive.to_xml_string().unwrap();
        let loaded = XmlKeyValueArchive::from_xml_string(&xml).unwrap();

        if let Some(ArchiveValue::Nested(child)) = loaded.get("child") {
            assert_eq!(child.get_i32("x").unwrap(), 10);
        } else {
            panic!("Expected nested archive");
        }
    }

    #[test]
    fn test_file_archive_multiple_entries() {
        let mut archive = FileArchive::new();
        archive.add_entry("metadata.json", b"{\"version\":1}", false);
        archive.add_entry("audio.dat", &[0u8; 1024], false);
        archive.add_entry("notes.bin", b"note data here", false);

        let bytes = archive.to_bytes().unwrap();
        let loaded = FileArchive::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.len(), 3);
        let names = loaded.entry_names();
        assert!(names.contains(&"metadata.json"));
        assert!(names.contains(&"audio.dat"));
        assert!(names.contains(&"notes.bin"));

        assert_eq!(loaded.read_entry("metadata.json").unwrap(), b"{\"version\":1}");
        assert_eq!(loaded.read_entry("audio.dat").unwrap(), vec![0u8; 1024]);
        assert_eq!(loaded.read_entry("notes.bin").unwrap(), b"note data here");
    }

    #[test]
    fn test_file_archive_random_access() {
        let mut archive = FileArchive::new();
        archive.add_entry("first", b"AAAA", false);
        archive.add_entry("second", b"BBBB", false);
        archive.add_entry("third", b"CCCC", false);

        let bytes = archive.to_bytes().unwrap();
        let loaded = FileArchive::from_bytes(&bytes).unwrap();

        // Access out of order to test random access.
        assert_eq!(loaded.read_entry("third").unwrap(), b"CCCC");
        assert_eq!(loaded.read_entry("first").unwrap(), b"AAAA");
        assert_eq!(loaded.read_entry("second").unwrap(), b"BBBB");
    }

    #[test]
    fn test_file_archive_compression_roundtrip() {
        // Create data with enough repetition for deflate to actually compress.
        let original: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();

        let mut archive = FileArchive::new();
        archive.add_entry("compressed.dat", &original, true);
        archive.add_entry("plain.dat", b"not compressed", false);

        let bytes = archive.to_bytes().unwrap();
        let loaded = FileArchive::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.read_entry("compressed.dat").unwrap(), original);
        assert_eq!(loaded.read_entry("plain.dat").unwrap(), b"not compressed");
    }

    #[test]
    fn test_file_archive_entry_not_found() {
        let archive = FileArchive::new();
        assert!(matches!(
            archive.read_entry("missing"),
            Err(ArchiveError::EntryNotFound(_))
        ));
    }

    #[test]
    fn test_file_archive_corrupt_data_crc() {
        let mut archive = FileArchive::new();
        archive.add_entry("test", b"original data", false);
        let mut bytes = archive.to_bytes().unwrap();

        // Corrupt the entry data (after the 12-byte header).
        if bytes.len() > 14 {
            bytes[13] ^= 0xFF;
        }
        let loaded = FileArchive::from_bytes(&bytes).unwrap();
        let result = loaded.read_entry("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_file_archive_invalid_magic() {
        let data = b"XXXXXXXX\x01\x00\x00\x00";
        let result = FileArchive::from_bytes(data);
        assert!(matches!(result, Err(ArchiveError::InvalidMagic)));
    }

    #[test]
    fn test_binary_archive_keys_and_len() {
        let mut archive = BinaryKeyValueArchive::new();
        assert!(archive.is_empty());
        archive.set_i32("b", 2);
        archive.set_i32("a", 1);
        assert_eq!(archive.len(), 2);
        // BTreeMap keeps sorted order.
        assert_eq!(archive.keys(), vec!["a", "b"]);
    }

    #[test]
    fn test_binary_archive_empty_roundtrip() {
        let archive = BinaryKeyValueArchive::new();
        let bytes = archive.to_bytes().unwrap();
        let loaded = BinaryKeyValueArchive::from_bytes(&bytes).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_xml_archive_version() {
        let mut archive = XmlKeyValueArchive::new();
        archive.version = 3;
        archive.set("x", ArchiveValue::I32(1));
        let xml = archive.to_xml_string().unwrap();
        let loaded = XmlKeyValueArchive::from_xml_string(&xml).unwrap();
        assert_eq!(loaded.version, 3);
    }
}
