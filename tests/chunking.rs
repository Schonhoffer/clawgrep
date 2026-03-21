//! Chunking tests.

use std::fs;

use tempfile::TempDir;

use clawgrep::index::chunk_file;

#[test]
fn chunk_file_basic() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("test.txt");
    fs::write(&file, "alpha\n\n  \nbeta\n\ngamma\n").unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_line, 1);
    assert!(chunks[0].text.contains("alpha"));
    assert!(chunks[0].text.contains("gamma"));
}

#[test]
fn chunk_file_preserves_indentation() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("code.py");
    fs::write(&file, "def foo():\n    return 42\n").unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].text.contains("    return 42"));
}

#[test]
fn chunk_file_empty_returns_empty() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("empty.txt");
    fs::write(&file, "").unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn chunk_file_large_produces_multiple_chunks() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("big.txt");
    let mut content = String::new();
    for i in 0..50 {
        content.push_str(&format!("line {i}\n"));
    }
    fs::write(&file, &content).unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(chunks.len() > 1, "50 lines should produce multiple chunks");
    assert_eq!(chunks[0].start_line, 1);
}

#[test]
fn chunk_file_has_searchable_text() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("tokens.txt");
    fs::write(&file, "error connecting to database\n").unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(chunks[0].text.contains("error"));
    assert!(chunks[0].text.contains("database"));
}

#[test]
fn unicode_file_content() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("unicode.txt");
    fs::write(&file, "\u{65E5}\u{672C}\u{8A9E}\n\u{4E2D}\u{6587}\n").unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert_eq!(chunks.len(), 1);
}

#[test]
fn very_long_lines() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("long.txt");
    let long_line = "x".repeat(10_000);
    fs::write(&file, format!("{long_line}\nshort\n")).unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(!chunks.is_empty());
    assert!(chunks[0].text.len() >= 10_000);
}

#[test]
fn single_line_no_trailing_newline() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("noeol.txt");
    fs::write(&file, "no newline at end").unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "no newline at end");
    assert_eq!(chunks[0].start_line, 1);
}

#[test]
fn windows_line_endings() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("crlf.txt");
    fs::write(&file, "line one\r\nline two\r\n\r\nline four\r\n").unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(!chunks.is_empty());
}
