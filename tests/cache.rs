//! SQLite cache tests.

use std::fs;

use rusqlite::{params, Connection, OpenFlags};
use tempfile::TempDir;

use clawgrep::cache::*;

#[test]
fn cache_roundtrip_preserves_data() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();

    let entry = CacheEntry {
        path: "file_a.txt".to_string(),
        mtime_ms: 1700000000000,
        file_size: 100,
        num_chunks: 1,
        dim: 8,
        embedding_model: "model-x".to_string(),
        chunks: vec![ChunkBoundary {
            start_line: 1,
            end_line: 5,
            boost: 1.0,
        }],
        embeddings: vec![0.1; 8],
    };

    upsert_entry(&conn, &entry).unwrap();
    assert!(cache_dir(Some(dir.path())).exists());
    let loaded = get_entry(&conn, "file_a.txt", "model-x").unwrap().unwrap();
    assert_eq!(loaded.num_chunks, 1);
    assert_eq!(loaded.dim, 8);
    assert_eq!(loaded.chunks.len(), 1);
    assert_eq!(loaded.embeddings.len(), 8);
}

#[test]
fn cache_empty_db_returns_none() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();
    assert!(get_entry(&conn, "nonexistent.txt", "model")
        .unwrap()
        .is_none());
}

#[test]
fn cache_freshness_detects_unchanged() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("stable.txt");
    fs::write(&file, "unchanged content").unwrap();
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
}

#[test]
fn cache_freshness_detects_content_change() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("changing.txt");
    fs::write(&file, "version 1").unwrap();
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
    std::thread::sleep(std::time::Duration::from_millis(50));
    fs::write(&file, "version 2 with more data added here").unwrap();
    assert!(!is_fresh(&entry, &file));
}

#[test]
fn cache_freshness_stale_if_file_deleted() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("ephemeral.txt");
    fs::write(&file, "soon gone").unwrap();
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
    fs::remove_file(&file).unwrap();
    assert!(!is_fresh(&entry, &file));
}

#[test]
fn cache_dir_uses_home_cache() {
    let cd = cache_dir(None);
    assert!(cd.ends_with("clawgrep"));
}

#[test]
fn cache_custom_dir() {
    let dir = TempDir::new().unwrap();
    let custom = dir.path().join("custom-cache");
    let _conn = open_db(Some(&custom)).unwrap();
    assert!(custom.join("cache.db").exists());
}

#[test]
fn cache_multi_model_same_file() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();

    for model in &["model-a", "model-b"] {
        upsert_entry(
            &conn,
            &CacheEntry {
                path: "f.txt".to_string(),
                mtime_ms: 0,
                file_size: 0,
                num_chunks: 0,
                dim: 4,
                embedding_model: model.to_string(),
                chunks: vec![],
                embeddings: vec![],
            },
        )
        .unwrap();
    }

    let entries = load_all_entries(&conn, "model-a").unwrap();
    assert_eq!(entries.len(), 1);
    let entries = load_all_entries(&conn, "model-b").unwrap();
    assert_eq!(entries.len(), 1);
}

#[test]
fn cache_multiple_files_roundtrip() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();

    for i in 0..10u64 {
        upsert_entry(
            &conn,
            &CacheEntry {
                path: format!("file_{i}.txt"),
                mtime_ms: 1700000000000 + i * 1000,
                file_size: 50 + i * 10,
                num_chunks: 1,
                dim: 4,
                embedding_model: "model".to_string(),
                chunks: vec![ChunkBoundary {
                    start_line: 1,
                    end_line: 2,
                    boost: 1.0,
                }],
                embeddings: vec![i as f32 * 0.1; 4],
            },
        )
        .unwrap();
    }

    let loaded = load_all_entries(&conn, "model").unwrap();
    assert_eq!(loaded.len(), 10);
}

#[test]
fn cache_upsert_overwrites_with_newer() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();

    upsert_entry(
        &conn,
        &CacheEntry {
            path: "file.txt".to_string(),
            mtime_ms: 1000,
            file_size: 10,
            num_chunks: 1,
            dim: 4,
            embedding_model: "m".to_string(),
            chunks: vec![ChunkBoundary {
                start_line: 1,
                end_line: 1,
                boost: 1.0,
            }],
            embeddings: vec![0.0; 4],
        },
    )
    .unwrap();

    upsert_entry(
        &conn,
        &CacheEntry {
            path: "file.txt".to_string(),
            mtime_ms: 2000,
            file_size: 20,
            num_chunks: 1,
            dim: 4,
            embedding_model: "m".to_string(),
            chunks: vec![ChunkBoundary {
                start_line: 1,
                end_line: 1,
                boost: 1.0,
            }],
            embeddings: vec![1.0; 4],
        },
    )
    .unwrap();

    let loaded = get_entry(&conn, "file.txt", "m").unwrap().unwrap();
    assert_eq!(loaded.mtime_ms, 2000);
    assert!((loaded.embeddings[0] - 1.0).abs() < 1e-6);
}

#[test]
fn cache_upsert_rejects_older() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();

    upsert_entry(
        &conn,
        &CacheEntry {
            path: "file.txt".to_string(),
            mtime_ms: 2000,
            file_size: 20,
            num_chunks: 1,
            dim: 4,
            embedding_model: "m".to_string(),
            chunks: vec![ChunkBoundary {
                start_line: 1,
                end_line: 1,
                boost: 1.0,
            }],
            embeddings: vec![1.0; 4],
        },
    )
    .unwrap();

    // Try to overwrite with older mtime — should be rejected
    upsert_entry(
        &conn,
        &CacheEntry {
            path: "file.txt".to_string(),
            mtime_ms: 1000,
            file_size: 10,
            num_chunks: 1,
            dim: 4,
            embedding_model: "m".to_string(),
            chunks: vec![ChunkBoundary {
                start_line: 1,
                end_line: 1,
                boost: 1.0,
            }],
            embeddings: vec![0.0; 4],
        },
    )
    .unwrap();

    let loaded = get_entry(&conn, "file.txt", "m").unwrap().unwrap();
    assert_eq!(loaded.mtime_ms, 2000);
    assert!((loaded.embeddings[0] - 1.0).abs() < 1e-6);
}

// ── Schema versioning tests ─────────────────────────────────────────────

#[test]
fn cache_schema_version_written_on_fresh_db() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();
    let version: u32 = conn
        .query_row("SELECT version FROM schema_info LIMIT 1", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(version, 1);
}

#[test]
fn cache_wrong_schema_version_nukes_data() {
    let dir = TempDir::new().unwrap();

    // Populate the cache with data under a valid schema.
    {
        let conn = open_db(Some(dir.path())).unwrap();
        upsert_entry(
            &conn,
            &CacheEntry {
                path: "old.txt".to_string(),
                mtime_ms: 1000,
                file_size: 10,
                num_chunks: 1,
                dim: 4,
                embedding_model: "m".to_string(),
                chunks: vec![ChunkBoundary {
                    start_line: 1,
                    end_line: 1,
                    boost: 1.0,
                }],
                embeddings: vec![1.0; 4],
            },
        )
        .unwrap();
        // Confirm the data is there.
        assert!(get_entry(&conn, "old.txt", "m").unwrap().is_some());
    }

    // Manually set schema_info to a future version to simulate the DB
    // being from a newer (or just different) app version.
    {
        let db_path = cache_dir(Some(dir.path())).join("cache.db");
        let conn = Connection::open_with_flags(
            &db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .unwrap();
        conn.execute("UPDATE schema_info SET version = 9999", [])
            .unwrap();
    }

    // Reopen — should detect mismatch and nuke the data.
    {
        let conn = open_db(Some(dir.path())).unwrap();
        // Old data should be gone.
        assert!(get_entry(&conn, "old.txt", "m").unwrap().is_none());
        // Schema version should now be current.
        let version: u32 = conn
            .query_row("SELECT version FROM schema_info LIMIT 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(version, 1);
    }
}

// ── Corruption resilience tests ─────────────────────────────────────────

#[test]
fn cache_corrupt_db_file_recovers_via_resilient() {
    let dir = TempDir::new().unwrap();
    let cache_path = cache_dir(Some(dir.path()));
    fs::create_dir_all(&cache_path).unwrap();
    let db_file = cache_path.join("cache.db");

    // Write garbage into the DB file.
    fs::write(&db_file, b"this is not a sqlite database").unwrap();

    // open_db_resilient should recover by nuking and recreating.
    let conn = open_db_resilient(Some(dir.path()));
    assert!(conn.is_some(), "should recover from corrupt DB");

    // Should be a working empty cache.
    let conn = conn.unwrap();
    assert!(get_entry(&conn, "anything", "m").unwrap().is_none());
}

#[test]
fn cache_get_entry_safe_returns_none_on_bad_blob() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();

    // Manually insert a row with garbage blob data.
    conn.execute(
        "INSERT INTO cache_entries
            (path, embedding_model, mtime_ms, file_size, num_chunks, dim, chunks_blob, embeddings_blob)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params!["bad.txt", "m", 1000i64, 10i64, 1i64, 4i64, b"garbage".to_vec(), b"garbage".to_vec()],
    )
    .unwrap();

    // The safe wrapper should return None, not panic or error.
    let result = get_entry_safe(&conn, "bad.txt", "m");
    assert!(result.is_none());
}

#[test]
fn cache_upsert_entry_safe_swallows_errors() {
    let dir = TempDir::new().unwrap();
    let conn = open_db(Some(dir.path())).unwrap();

    // Drop the table to cause writes to fail.
    conn.execute_batch("DROP TABLE cache_entries").unwrap();

    let entry = CacheEntry {
        path: "file.txt".to_string(),
        mtime_ms: 1000,
        file_size: 10,
        num_chunks: 1,
        dim: 4,
        embedding_model: "m".to_string(),
        chunks: vec![ChunkBoundary {
            start_line: 1,
            end_line: 1,
            boost: 1.0,
        }],
        embeddings: vec![1.0; 4],
    };

    // Should not panic — just log a warning.
    upsert_entry_safe(&conn, &entry);
}

#[test]
fn cache_missing_db_dir_resilient_returns_some() {
    let dir = TempDir::new().unwrap();
    let nested = dir.path().join("a").join("b").join("c");
    // Dir doesn't exist yet — open_db_resilient should create it.
    let conn = open_db_resilient(Some(&nested));
    assert!(conn.is_some());
}
