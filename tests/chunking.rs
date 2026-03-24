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
    assert!(chunks[0].text.contains("\u{65E5}\u{672C}\u{8A9E}"));
}

#[test]
fn cjk_dense_text_chunks_reasonably() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("cjk.txt");
    // Create dense CJK text: 30 lines of ~80 CJK chars each.
    // With char-based token estimation, each line ≈ 20 tokens,
    // so ~5 lines should reach the token limit.
    let mut content = String::new();
    for _ in 0..30 {
        for _ in 0..80 {
            content.push('漢');
        }
        content.push('\n');
    }
    fs::write(&file, &content).unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(
        chunks.len() >= 2,
        "dense CJK text should produce multiple chunks, got {}",
        chunks.len()
    );
}

#[test]
fn cyrillic_text_chunks() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("cyrillic.txt");
    let mut content = String::new();
    for i in 0..30 {
        content.push_str(&format!(
            "Строка номер {i} содержит текст на русском языке для тестирования работы системы\n"
        ));
    }
    fs::write(&file, &content).unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(
        chunks.len() >= 2,
        "30 lines of Cyrillic should produce multiple chunks, got {}",
        chunks.len()
    );
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

#[test]
fn splits_at_blank_line_boundary() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("paragraphs.md");
    // Build two paragraphs of ~60 tokens each (~240 chars), separated
    // by a blank line.  Together they exceed TARGET_CHUNK_TOKENS (100),
    // so the chunker should prefer splitting at the blank line.
    let para1 = "word ".repeat(60); // ~300 chars, ~75 tokens
    let para2 = "text ".repeat(60);
    let content = format!("{para1}\n\n{para2}\n");
    fs::write(&file, &content).unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(
        chunks.len() >= 2,
        "should split into at least 2 chunks at the blank line"
    );
    // The first chunk should NOT contain text from para2.
    assert!(
        !chunks[0].text.contains("text text text"),
        "first chunk should end before the second paragraph"
    );
}

#[test]
fn splits_at_markdown_header() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("doc.md");
    let section1 = "content line\n".repeat(12); // ~12 lines, ~24 tokens
    let section2 = "more content line\n".repeat(12);
    let content = format!("{section1}## New Section\n{section2}");
    fs::write(&file, &content).unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(
        chunks.len() >= 2,
        "should split when hitting line cap near header"
    );
}

#[test]
fn dense_prose_splits_by_token_count() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("dense.txt");
    // 6 lines of ~80 chars each = ~120 tokens per line. Each line is
    // ~20 tokens, so ~6 lines to hit 100 tokens.  Well under the
    // 20-line cap, so token limit triggers first.
    let mut content = String::new();
    for i in 0..15 {
        content.push_str(&format!(
            "This is a fairly long line number {i} with enough words to use up meaningful tokens in the model context window.\n"
        ));
    }
    fs::write(&file, &content).unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(
        chunks.len() >= 2,
        "dense prose should split by token count before hitting line cap"
    );
    // Each chunk should have fewer than 20 lines (token limit triggers first).
    for chunk in &chunks {
        let line_count = chunk.text.lines().count();
        assert!(
            line_count <= 20,
            "chunk has {line_count} lines, should be <= 20"
        );
    }
}

#[test]
fn chunks_have_overlap() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("overlap.txt");
    let mut content = String::new();
    for i in 0..50 {
        content.push_str(&format!("unique_marker_{i}\n"));
    }
    fs::write(&file, &content).unwrap();
    let chunks = chunk_file(&file).unwrap();
    assert!(chunks.len() >= 2);
    // The last few lines of chunk 0 should appear at the start of chunk 1.
    let c0_lines: Vec<&str> = chunks[0].text.lines().collect();
    let c1_lines: Vec<&str> = chunks[1].text.lines().collect();
    let last_of_c0 = c0_lines.last().unwrap();
    assert!(
        c1_lines.iter().any(|l| l == last_of_c0),
        "chunks should overlap: last line of chunk 0 ({last_of_c0}) not found in chunk 1"
    );
}
