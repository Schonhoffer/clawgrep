//! Hybrid search: semantic (embedding) + keyword (substring/regex).
//!
//! Scoring for each chunk:
//!
//! ```text
//!     score = semantic_weight * cosine(query_emb, chunk_emb)
//!           + keyword_weight  * keyword_score(query, chunk_text)
//! ```
//!
//! Semantic scores come from cached embeddings (no file I/O).
//! Keyword scores come from reading files from disk and doing
//! substring / regex / word-stem matching.
//!
//! By default `semantic_weight = 0.7` and `keyword_weight = 0.3`.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use log::debug;
use rayon::prelude::*;

use crate::cache::CacheEntry;
use crate::embed::{cosine_similarity, Embedder};
use crate::index::{Index, CHUNK_LINES, CHUNK_OVERLAP};
use crate::keyword::keyword_search;

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file: PathBuf,
    /// 1-based start line of the matching chunk.
    pub line_num: usize,
    /// End line of the matching chunk.
    pub end_line: usize,
    /// The chunk text (read from disk).
    pub text: String,
    /// Combined score (higher = more relevant).
    pub score: f32,
    /// Semantic component of the score.
    pub semantic_score: f32,
    /// Keyword component of the score.
    pub keyword_score: f32,
}

/// Options that control how results are ranked.
#[derive(Debug, Clone)]
pub struct SearchOpts {
    /// Number of results to return.
    pub top_k: usize,
    /// Minimum combined score cutoff.
    pub min_score: Option<f32>,
    /// Weight for semantic similarity [0..1].
    pub semantic_weight: f32,
    /// Weight for keyword (substring) similarity [0..1].
    pub keyword_weight: f32,
}

impl Default for SearchOpts {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: None,
            semantic_weight: 0.7,
            keyword_weight: 0.3,
        }
    }
}

/// Run a hybrid search over an already-built index.
///
/// The `index` contains embeddings and chunk boundaries (no text).
/// Text is read from disk as needed for keyword scoring and result display.
pub fn hybrid_search(
    query: &str,
    index: &Index,
    files: &[PathBuf],
    embedder: &Embedder,
    opts: &SearchOpts,
) -> Result<Vec<SearchResult>> {
    if index.entries.is_empty() {
        return Ok(vec![]);
    }

    // 1. Embed the query.
    let query_emb = embedder.embed_one(query)?;

    // 2. Semantic scoring: score every chunk from cached embeddings.
    let sem_scores = semantic_scores(&query_emb, &index.entries);
    debug!("scored {} chunks semantically", sem_scores.len());

    // 3. Keyword scoring: read files from disk and do substring/regex matching.
    let kw_hits = if opts.keyword_weight > 0.0 {
        keyword_search(query, files, CHUNK_LINES, CHUNK_OVERLAP)
    } else {
        vec![]
    };

    // Build lookup: (file, start_line) → (keyword score, end_line, text).
    let mut kw_map: HashMap<(String, usize), (f32, usize, String)> = HashMap::new();
    for hit in &kw_hits {
        let key = (hit.file.to_string_lossy().to_string(), hit.start_line);
        let existing = kw_map
            .entry(key)
            .or_insert((0.0, hit.end_line, hit.text.clone()));
        if hit.score > existing.0 {
            *existing = (hit.score, hit.end_line, hit.text.clone());
        }
    }

    // 4. Combine scores.
    let mut candidates: Vec<(String, usize, usize, f32, f32, f32, f32)> = Vec::new();
    let mut seen: HashSet<(String, usize)> = HashSet::new();

    // From semantic side: every cached chunk gets a semantic score.
    for (file, start_line, end_line, boost, sem) in &sem_scores {
        let kw_key = (file.clone(), *start_line);
        let kw = kw_map.get(&kw_key).map(|v| v.0).unwrap_or(0.0);
        let combined = (opts.semantic_weight * sem + opts.keyword_weight * kw) * boost;
        if combined > 0.0 {
            seen.insert(kw_key);
            candidates.push((
                file.clone(),
                *start_line,
                *end_line,
                combined,
                *sem,
                kw,
                *boost,
            ));
        }
    }

    // From keyword side: hits that don't have a matching semantic chunk.
    for ((file, start_line), (kw_score, end_line, _text)) in &kw_map {
        let key = (file.clone(), *start_line);
        if seen.contains(&key) {
            continue;
        }
        let combined = opts.keyword_weight * kw_score;
        if combined > 0.0 {
            candidates.push((
                file.clone(),
                *start_line,
                *end_line,
                combined,
                0.0,
                *kw_score,
                1.0,
            ));
        }
    }

    // 5. Sort by combined score descending.
    candidates.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

    // 6. Take top candidates and build results (read text from disk).
    let mut results: Vec<SearchResult> = Vec::new();
    for (file, start_line, end_line, combined, sem, kw, _boost) in candidates {
        if let Some(min) = opts.min_score {
            if combined < min {
                continue;
            }
        }
        if results.len() >= opts.top_k {
            break;
        }

        // Read chunk text from the file.
        let text = read_chunk_text(&file, start_line, end_line);

        results.push(SearchResult {
            file: PathBuf::from(&file),
            line_num: start_line,
            end_line,
            text,
            score: combined,
            semantic_score: sem,
            keyword_score: kw,
        });
    }

    Ok(results)
}

/// Score every chunk in the index against the query embedding.
/// Returns: (file_path, start_line, end_line, boost, cosine_score).
fn semantic_scores(
    query_emb: &[f32],
    entries: &[CacheEntry],
) -> Vec<(String, usize, usize, f32, f32)> {
    entries
        .par_iter()
        .flat_map(|entry| {
            let dim = entry.dim;
            entry
                .chunks
                .iter()
                .enumerate()
                .filter_map(|(i, cb)| {
                    let start = i * dim;
                    let end = start + dim;
                    if end > entry.embeddings.len() {
                        return None;
                    }
                    let chunk_emb = &entry.embeddings[start..end];
                    let sim = cosine_similarity(query_emb, chunk_emb);
                    Some((
                        entry.path.clone(),
                        cb.start_line,
                        cb.end_line,
                        cb.boost,
                        sim,
                    ))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Read a specific chunk's text from disk.
fn read_chunk_text(file: &str, start_line: usize, end_line: usize) -> String {
    let Ok(content) = fs::read_to_string(file) else {
        return String::new();
    };
    let lines: Vec<&str> = content.lines().collect();
    if start_line == 0 || start_line > lines.len() {
        return String::new();
    }
    let s = start_line - 1;
    let e = end_line.min(lines.len());
    lines[s..e].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_opts_defaults() {
        let opts = SearchOpts::default();
        assert_eq!(opts.top_k, 5);
        assert!((opts.semantic_weight - 0.7).abs() < 1e-6);
        assert!((opts.keyword_weight - 0.3).abs() < 1e-6);
    }

    #[test]
    fn read_chunk_text_basic() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("test.txt");
        std::fs::write(&f, "line1\nline2\nline3\nline4\n").unwrap();
        let text = read_chunk_text(f.to_str().unwrap(), 2, 3);
        assert_eq!(text, "line2\nline3");
    }

    #[test]
    fn read_chunk_text_missing_file() {
        let text = read_chunk_text("/nonexistent/file.txt", 1, 1);
        assert!(text.is_empty());
    }
}
