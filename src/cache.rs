//! SQLite-backed embeddings cache stored in `.clawgrep/` (or a user-chosen
//! directory).
//!
//! This is "just a cache" — it can be thrown away at any time without data
//! loss. The app must never fail because the cache is corrupt or outdated.
//!
//! Schema versioning: a single `SCHEMA_VERSION` constant is stored in the
//! DB. If the app is newer than the DB, we nuke the entire DB and rebuild.
//! No migrations.
//!
//! Error resilience: any cache error (corruption, I/O, deserialization)
//! causes the DB to be deleted and recreated. If that also fails, the app
//! runs without caching.
//!
//! Each row stores embeddings for one file, keyed by
//! `(canonical_path, embedding_model)`.
//!
//! What is stored per row:
//! - path, embedding model identifier
//! - mtime epoch-millis and file size (for change detection)
//! - chunk boundaries: a bincode blob of `Vec<ChunkBoundary>` (start_line,
//!   end_line, boost per chunk)
//! - raw embedding floats as a byte blob (num_chunks × dim)
//! - dim and num_chunks scalars
//!
//! What is NOT stored:
//! - chunk text (read from the live file at query time)
//! - keyword tokens (computed from live text at query time)
//!
//! Concurrency:
//! - WAL journal mode allows concurrent readers + serialised writers.
//! - `busy_timeout` lets a blocked writer retry for up to 5 seconds.
//! - Upserts use `WHERE excluded.mtime_ms >= cache_entries.mtime_ms` so
//!   a slower process never overwrites a newer result.
//!
//! Change detection uses **mtime + size** which is fast and portable.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use log::warn;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};

/// Default name of the hidden cache directory.
pub const CACHE_DIR_NAME: &str = ".clawgrep";

/// Name of the SQLite database file inside the cache directory.
const DB_FILE: &str = "cache.db";

/// Bump this when changing the DB schema. The cache is disposable: if the
/// app's version is newer than the DB's, we nuke all data and recreate.
const SCHEMA_VERSION: u32 = 1;

// ── Serialised types ────────────────────────────────────────────────────

/// Per-chunk boundary info stored as a bincode blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkBoundary {
    /// 1-based line number of the first line in this chunk.
    pub start_line: usize,
    /// 1-based line number of the last line in this chunk.
    pub end_line: usize,
    /// Score multiplier (1.0 for content chunks, path_boost for path chunks).
    pub boost: f32,
}

/// An in-memory cache entry loaded from SQLite.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub path: String,
    pub mtime_ms: u64,
    pub file_size: u64,
    pub num_chunks: usize,
    pub dim: usize,
    pub embedding_model: String,
    pub chunks: Vec<ChunkBoundary>,
    /// Flat embedding data: `num_chunks * dim` floats.
    pub embeddings: Vec<f32>,
}

// ── Public helpers ──────────────────────────────────────────────────────

/// Return `(mtime_ms, size)` for a file.
pub fn file_stamp(path: &Path) -> Result<(u64, u64)> {
    let md = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let mtime = md
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    Ok((mtime, md.len()))
}

/// Resolve the cache directory.
/// Priority: custom path > `~/.cache/clawgrep/` (or platform equivalent).
pub fn cache_dir(custom: Option<&Path>) -> PathBuf {
    match custom {
        Some(p) => p.to_path_buf(),
        None => default_cache_dir(),
    }
}

/// Default cache directory: `~/.cache/clawgrep/` (or platform equivalent).
fn default_cache_dir() -> PathBuf {
    if let Some(cache) = dirs::cache_dir() {
        cache.join("clawgrep")
    } else {
        // Fallback: use temp dir.
        std::env::temp_dir().join("clawgrep_cache")
    }
}

/// Check whether a cached entry is still fresh for the file at `path`.
pub fn is_fresh(entry: &CacheEntry, path: &Path) -> bool {
    match file_stamp(path) {
        Ok((mtime_ms, size)) => entry.mtime_ms == mtime_ms && entry.file_size == size,
        Err(_) => false,
    }
}

// ── Database ────────────────────────────────────────────────────────────

/// Path to the DB file for a given cache directory.
fn db_path(custom_cache: Option<&Path>) -> PathBuf {
    cache_dir(custom_cache).join(DB_FILE)
}

/// Delete the cache DB file (and WAL/SHM sidecars). Best-effort.
fn nuke_db_file(custom_cache: Option<&Path>) {
    let base = db_path(custom_cache);
    for suffix in &["", "-wal", "-shm"] {
        let mut p = base.as_os_str().to_owned();
        p.push(suffix);
        let _ = fs::remove_file(PathBuf::from(p));
    }
}

/// Create the schema tables (including `schema_info`).
fn create_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_info (
            version INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cache_entries (
            path             TEXT    NOT NULL,
            embedding_model  TEXT    NOT NULL,
            mtime_ms         INTEGER NOT NULL,
            file_size        INTEGER NOT NULL,
            num_chunks       INTEGER NOT NULL,
            dim              INTEGER NOT NULL,
            chunks_blob      BLOB    NOT NULL,
            embeddings_blob  BLOB    NOT NULL,
            PRIMARY KEY (path, embedding_model)
        );",
    )
    .context("creating cache tables")?;
    Ok(())
}

/// Check the schema version. If it's outdated, drop everything and recreate.
fn ensure_schema_version(conn: &Connection) -> Result<()> {
    let version: Option<u32> = conn
        .query_row("SELECT version FROM schema_info LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()?;

    match version {
        Some(v) if v == SCHEMA_VERSION => {}
        Some(v) => {
            warn!("cache schema v{v} != app v{SCHEMA_VERSION}; nuking cached data");
            conn.execute_batch(
                "DROP TABLE IF EXISTS cache_entries;
                 DROP TABLE IF EXISTS schema_info;",
            )?;
            create_tables(conn)?;
            conn.execute(
                "INSERT INTO schema_info (version) VALUES (?1)",
                params![SCHEMA_VERSION],
            )?;
        }
        None => {
            conn.execute(
                "INSERT INTO schema_info (version) VALUES (?1)",
                params![SCHEMA_VERSION],
            )?;
        }
    }
    Ok(())
}

/// Open (or create) the cache database with WAL mode and busy timeout.
/// Checks schema version, nuking data if outdated.
pub fn open_db(custom_cache: Option<&Path>) -> Result<Connection> {
    let dir = cache_dir(custom_cache);
    fs::create_dir_all(&dir).with_context(|| format!("creating cache dir {}", dir.display()))?;
    let path = dir.join(DB_FILE);

    let conn = Connection::open_with_flags(
        &path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("opening cache db {}", path.display()))?;

    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "busy_timeout", 5000)?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;

    create_tables(&conn)?;
    ensure_schema_version(&conn)?;

    Ok(conn)
}

/// Open the cache DB with full resilience. Any error (corruption, schema
/// mismatch, I/O failure) causes the DB file to be deleted and recreated.
/// If even that fails, returns `None` and the app runs without caching.
pub fn open_db_resilient(custom_cache: Option<&Path>) -> Option<Connection> {
    match open_db(custom_cache) {
        Ok(conn) => Some(conn),
        Err(e) => {
            warn!("cache error: {e:#}; deleting cache and retrying");
            eprintln!("clawgrep: cache error, rebuilding cache");
            nuke_db_file(custom_cache);
            match open_db(custom_cache) {
                Ok(conn) => Some(conn),
                Err(e) => {
                    warn!("cache still broken after reset: {e:#}; running without cache");
                    eprintln!("clawgrep: cache unavailable, running without cache");
                    None
                }
            }
        }
    }
}

/// Look up a single cache entry. Returns `None` if not found.
pub fn get_entry(conn: &Connection, path: &str, model: &str) -> Result<Option<CacheEntry>> {
    let mut stmt = conn.prepare_cached(
        "SELECT mtime_ms, file_size, num_chunks, dim, chunks_blob, embeddings_blob
         FROM cache_entries
         WHERE path = ?1 AND embedding_model = ?2",
    )?;

    let mut rows = stmt.query(params![path, model])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };

    let mtime_ms: i64 = row.get(0)?;
    let file_size: i64 = row.get(1)?;
    let num_chunks: i64 = row.get(2)?;
    let dim: i64 = row.get(3)?;
    let chunks_blob: Vec<u8> = row.get(4)?;
    let embeddings_blob: Vec<u8> = row.get(5)?;

    let chunks: Vec<ChunkBoundary> =
        bincode::deserialize(&chunks_blob).context("deserializing chunk boundaries")?;
    let embeddings = blob_to_f32(&embeddings_blob);

    Ok(Some(CacheEntry {
        path: path.to_string(),
        mtime_ms: mtime_ms as u64,
        file_size: file_size as u64,
        num_chunks: num_chunks as usize,
        dim: dim as usize,
        embedding_model: model.to_string(),
        chunks,
        embeddings,
    }))
}

/// Upsert a cache entry. Only overwrites if the new mtime_ms >= existing.
pub fn upsert_entry(conn: &Connection, entry: &CacheEntry) -> Result<()> {
    let chunks_blob = bincode::serialize(&entry.chunks).context("serializing chunk boundaries")?;
    let embeddings_blob = f32_to_blob(&entry.embeddings);

    conn.execute(
        "INSERT INTO cache_entries
            (path, embedding_model, mtime_ms, file_size, num_chunks, dim, chunks_blob, embeddings_blob)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(path, embedding_model) DO UPDATE SET
            mtime_ms = excluded.mtime_ms,
            file_size = excluded.file_size,
            num_chunks = excluded.num_chunks,
            dim = excluded.dim,
            chunks_blob = excluded.chunks_blob,
            embeddings_blob = excluded.embeddings_blob
         WHERE excluded.mtime_ms >= cache_entries.mtime_ms",
        params![
            entry.path,
            entry.embedding_model,
            entry.mtime_ms as i64,
            entry.file_size as i64,
            entry.num_chunks as i64,
            entry.dim as i64,
            chunks_blob,
            embeddings_blob,
        ],
    )?;

    Ok(())
}

/// Load all cache entries for a given model.
pub fn load_all_entries(conn: &Connection, model: &str) -> Result<Vec<CacheEntry>> {
    let mut stmt = conn.prepare_cached(
        "SELECT path, mtime_ms, file_size, num_chunks, dim, chunks_blob, embeddings_blob
         FROM cache_entries
         WHERE embedding_model = ?1",
    )?;

    let mut entries = Vec::new();
    let mut rows = stmt.query(params![model])?;
    while let Some(row) = rows.next()? {
        let path: String = row.get(0)?;
        let mtime_ms: i64 = row.get(1)?;
        let file_size: i64 = row.get(2)?;
        let num_chunks: i64 = row.get(3)?;
        let dim: i64 = row.get(4)?;
        let chunks_blob: Vec<u8> = row.get(5)?;
        let embeddings_blob: Vec<u8> = row.get(6)?;

        let chunks: Vec<ChunkBoundary> =
            bincode::deserialize(&chunks_blob).context("deserializing chunk boundaries")?;
        let embeddings = blob_to_f32(&embeddings_blob);

        entries.push(CacheEntry {
            path,
            mtime_ms: mtime_ms as u64,
            file_size: file_size as u64,
            num_chunks: num_chunks as usize,
            dim: dim as usize,
            embedding_model: model.to_string(),
            chunks,
            embeddings,
        });
    }

    Ok(entries)
}

// ── Safe wrappers (cache errors never propagate) ────────────────────────

/// Like `get_entry`, but returns `None` on any error instead of propagating.
pub fn get_entry_safe(conn: &Connection, path: &str, model: &str) -> Option<CacheEntry> {
    match get_entry(conn, path, model) {
        Ok(entry) => entry,
        Err(e) => {
            warn!("cache read error for {path}: {e:#}");
            None
        }
    }
}

/// Like `upsert_entry`, but silently logs a warning on failure.
pub fn upsert_entry_safe(conn: &Connection, entry: &CacheEntry) {
    if let Err(e) = upsert_entry(conn, entry) {
        warn!("cache write error for {}: {e:#}", entry.path);
    }
}

// ── Blob conversion helpers ─────────────────────────────────────────────

fn f32_to_blob(data: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(data.len() * 4);
    for &v in data {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    blob
}

fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn roundtrip_entry() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_db(Some(dir.path())).unwrap();

        let entry = CacheEntry {
            path: "test.txt".to_string(),
            mtime_ms: 1000,
            file_size: 42,
            num_chunks: 1,
            dim: 4,
            embedding_model: "model-a".to_string(),
            chunks: vec![ChunkBoundary {
                start_line: 1,
                end_line: 3,
                boost: 1.0,
            }],
            embeddings: vec![0.1, 0.2, 0.3, 0.4],
        };

        upsert_entry(&conn, &entry).unwrap();
        let loaded = get_entry(&conn, "test.txt", "model-a").unwrap().unwrap();
        assert_eq!(loaded.num_chunks, 1);
        assert_eq!(loaded.dim, 4);
        assert_eq!(loaded.chunks.len(), 1);
        assert_eq!(loaded.embeddings.len(), 4);
        assert!((loaded.embeddings[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn multiple_models_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_db(Some(dir.path())).unwrap();

        for model in &["model-a", "model-b"] {
            let entry = CacheEntry {
                path: "test.txt".to_string(),
                mtime_ms: 1000,
                file_size: 42,
                num_chunks: 1,
                dim: 4,
                embedding_model: model.to_string(),
                chunks: vec![],
                embeddings: vec![],
            };
            upsert_entry(&conn, &entry).unwrap();
        }

        let a = get_entry(&conn, "test.txt", "model-a").unwrap();
        let b = get_entry(&conn, "test.txt", "model-b").unwrap();
        assert!(a.is_some());
        assert!(b.is_some());
    }

    #[test]
    fn custom_cache_dir() {
        let dir = tempfile::tempdir().unwrap();
        let custom = dir.path().join("my-cache");
        let _conn = open_db(Some(&custom)).unwrap();
        assert!(custom.join(DB_FILE).exists());
    }

    #[test]
    fn freshness_check() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        {
            let mut f = fs::File::create(&file).unwrap();
            writeln!(f, "line1").unwrap();
        }

        let (mtime_ms, size) = file_stamp(&file).unwrap();
        let entry = CacheEntry {
            path: file.to_string_lossy().into(),
            mtime_ms,
            file_size: size,
            num_chunks: 0,
            dim: 4,
            embedding_model: "m".to_string(),
            chunks: vec![],
            embeddings: vec![],
        };
        assert!(is_fresh(&entry, &file));

        // mutate the file → stale
        {
            let mut f = fs::OpenOptions::new().append(true).open(&file).unwrap();
            writeln!(f, "line2 extra data to change size").unwrap();
        }
        assert!(!is_fresh(&entry, &file));
    }

    #[test]
    fn optimistic_concurrency_newer_wins() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_db(Some(dir.path())).unwrap();

        let entry1 = CacheEntry {
            path: "test.txt".to_string(),
            mtime_ms: 2000,
            file_size: 42,
            num_chunks: 1,
            dim: 4,
            embedding_model: "model".to_string(),
            chunks: vec![ChunkBoundary {
                start_line: 1,
                end_line: 1,
                boost: 1.0,
            }],
            embeddings: vec![1.0, 1.0, 1.0, 1.0],
        };
        upsert_entry(&conn, &entry1).unwrap();

        // Older mtime should be rejected
        let entry_old = CacheEntry {
            path: "test.txt".to_string(),
            mtime_ms: 1000,
            file_size: 30,
            num_chunks: 1,
            dim: 4,
            embedding_model: "model".to_string(),
            chunks: vec![ChunkBoundary {
                start_line: 1,
                end_line: 1,
                boost: 1.0,
            }],
            embeddings: vec![0.0, 0.0, 0.0, 0.0],
        };
        upsert_entry(&conn, &entry_old).unwrap();

        let loaded = get_entry(&conn, "test.txt", "model").unwrap().unwrap();
        assert_eq!(loaded.mtime_ms, 2000);
        assert!((loaded.embeddings[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn load_all_entries_filters_by_model() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_db(Some(dir.path())).unwrap();

        for (path, model) in &[("a.txt", "m1"), ("b.txt", "m1"), ("c.txt", "m2")] {
            upsert_entry(
                &conn,
                &CacheEntry {
                    path: path.to_string(),
                    mtime_ms: 1000,
                    file_size: 10,
                    num_chunks: 0,
                    dim: 4,
                    embedding_model: model.to_string(),
                    chunks: vec![],
                    embeddings: vec![],
                },
            )
            .unwrap();
        }

        assert_eq!(load_all_entries(&conn, "m1").unwrap().len(), 2);
        assert_eq!(load_all_entries(&conn, "m2").unwrap().len(), 1);
    }

    #[test]
    fn blob_roundtrip() {
        let original = vec![0.1f32, -0.5, 3.14, 0.0];
        let blob = f32_to_blob(&original);
        let restored = blob_to_f32(&blob);
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
