//! Keyword search tests.

mod common;

use std::fs;

use tempfile::TempDir;

use clawgrep::index::discover_files;
use clawgrep::keyword::keyword_search;

use common::{make_project, test_discover_opts};

#[test]
fn keyword_exact_substring_match() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("data.txt"),
        "The barcode is UPC-A 012345678901\nUnrelated line\n",
    )
    .unwrap();
    let files = vec![dir.path().join("data.txt")];
    let hits = keyword_search("012345678901", &files);
    assert!(!hits.is_empty());
    assert!(hits[0].score > 0.9, "exact match should score high");
}

#[test]
fn keyword_case_insensitive() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "Database Connection Error\n").unwrap();
    let files = vec![dir.path().join("a.txt")];
    let hits = keyword_search("database connection error", &files);
    assert!(!hits.is_empty());
    assert!(hits[0].score > 0.9);
}

#[test]
fn keyword_no_match_returns_empty() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "sunny weather today\n").unwrap();
    let files = vec![dir.path().join("a.txt")];
    let hits = keyword_search("xyzzy_nonexistent", &files);
    assert!(hits.is_empty());
}

#[test]
fn keyword_partial_word_match() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("a.txt"),
        "The database connection was established successfully\n",
    )
    .unwrap();
    let files = vec![dir.path().join("a.txt")];
    let hits = keyword_search("database error", &files);
    // "database" matches but "error" doesn't — should get a partial score
    assert!(!hits.is_empty());
    assert!(hits[0].score > 0.0);
    assert!(
        hits[0].score < 1.0,
        "partial match should not be full score"
    );
}

#[test]
fn keyword_stemming_works() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("a.txt"),
        "The server is connecting to the host\n",
    )
    .unwrap();
    let files = vec![dir.path().join("a.txt")];
    // "connection" should match "connecting" via stemming
    let hits = keyword_search("connection", &files);
    assert!(
        !hits.is_empty(),
        "stemming should match 'connecting' to 'connection'"
    );
}

#[test]
fn keyword_regex_match() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "Serial: ABC-12345-XYZ\n").unwrap();
    let files = vec![dir.path().join("a.txt")];
    let hits = keyword_search(r"[A-Z]+-\d+-[A-Z]+", &files);
    assert!(!hits.is_empty());
    assert!(hits[0].score > 0.8, "regex match should score high");
}

#[test]
fn keyword_search_on_project_files() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let hits = keyword_search("barcode UPC", &files);
    assert!(!hits.is_empty());
    let top = hits[0].text.to_lowercase();
    assert!(top.contains("barcode") || top.contains("upc"), "top: {top}");
}

#[test]
fn keyword_search_scores_descending() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let hits = keyword_search("barcode", &files);
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[test]
fn keyword_empty_query() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "hello world\n").unwrap();
    let files = vec![dir.path().join("a.txt")];
    let hits = keyword_search("", &files);
    // Empty query should not crash; may or may not return results
    let _ = hits;
}
