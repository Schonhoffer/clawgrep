//! Keyword search: substring, regex, and word-level matching.
//!
//! Designed to find exact identifiers, barcodes, serial numbers, and other
//! literal strings that semantic (embedding) search is bad at.
//!
//! Scoring strategy for a chunk against a query:
//! 1. Full query as exact case-insensitive substring → highest score.
//! 2. Full query as regex (if it compiles) → high score.
//! 3. Individual query words matched as substrings → partial score,
//!    with basic suffix-stripping (stemming) for flexibility.
//!
//! The scores are normalised to [0, 1].

use std::fs;
use std::path::PathBuf;

use rayon::prelude::*;
use regex::RegexBuilder;

/// A keyword match result for one chunk of one file.
#[derive(Debug, Clone)]
pub struct KeywordHit {
    pub file: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub score: f32,
}

/// Score weight for a full exact substring match.
const EXACT_WEIGHT: f32 = 1.0;
/// Score weight for a regex match.
const REGEX_WEIGHT: f32 = 0.9;
/// Score weight for word-level matches (fraction of words matched).
const WORD_WEIGHT: f32 = 0.6;

/// Run keyword search over the given files, reading them from disk.
/// Returns hits sorted by score descending.
pub fn keyword_search(
    query: &str,
    files: &[PathBuf],
    chunk_lines: usize,
    chunk_overlap: usize,
) -> Vec<KeywordHit> {
    let query_lower = query.to_lowercase();
    let query_words = split_words(&query_lower);
    let stemmed_query: Vec<String> = query_words.iter().map(|w| stem(w)).collect();

    // Try to compile query as regex (case-insensitive).
    let query_regex = RegexBuilder::new(query).case_insensitive(true).build().ok();

    let hits: Vec<Vec<KeywordHit>> = files
        .par_iter()
        .filter_map(|path| {
            let content = fs::read_to_string(path).ok()?;
            let lines: Vec<&str> = content.lines().collect();
            if lines.is_empty() {
                return None;
            }
            let mut file_hits = Vec::new();
            let step = chunk_lines.saturating_sub(chunk_overlap).max(1);
            let mut start = 0;
            while start < lines.len() {
                let end = (start + chunk_lines).min(lines.len());
                let chunk_text = lines[start..end].join("\n");
                let score = score_chunk(
                    &chunk_text,
                    &query_lower,
                    &query_words,
                    &stemmed_query,
                    &query_regex,
                );
                if score > 0.0 {
                    file_hits.push(KeywordHit {
                        file: path.clone(),
                        start_line: start + 1,
                        end_line: end,
                        text: chunk_text,
                        score,
                    });
                }
                start += step;
                if end == lines.len() {
                    break;
                }
            }
            Some(file_hits)
        })
        .collect();

    let mut all_hits: Vec<KeywordHit> = hits.into_iter().flatten().collect();
    all_hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_hits
}

/// Score a single chunk of text against the query.
fn score_chunk(
    chunk_text: &str,
    query_lower: &str,
    query_words: &[String],
    stemmed_query: &[String],
    query_regex: &Option<regex::Regex>,
) -> f32 {
    let chunk_lower = chunk_text.to_lowercase();

    // 1. Exact substring match (case-insensitive).
    if !query_lower.is_empty() && chunk_lower.contains(query_lower) {
        return EXACT_WEIGHT;
    }

    // 2. Regex match.
    if let Some(re) = query_regex {
        if re.is_match(chunk_text) {
            return REGEX_WEIGHT;
        }
    }

    // 3. Word-level matching with stemming.
    if query_words.is_empty() {
        return 0.0;
    }
    let chunk_words = split_words(&chunk_lower);
    let stemmed_chunk: Vec<String> = chunk_words.iter().map(|w| stem(w)).collect();

    let mut matched = 0usize;
    for sq in stemmed_query {
        // Check if any chunk word stem matches, or if the raw query word
        // appears as a substring anywhere in the chunk.
        let word_match = stemmed_chunk.iter().any(|cw| cw == sq);
        let substr_match = chunk_lower.contains(sq.as_str());
        if word_match || substr_match {
            matched += 1;
        }
    }

    if matched == 0 {
        return 0.0;
    }

    let fraction = matched as f32 / stemmed_query.len() as f32;
    WORD_WEIGHT * fraction
}

/// Split text into lowercase words on whitespace and punctuation,
/// keeping hyphens, underscores, and dots within words (for barcodes,
/// serial numbers, etc.).
fn split_words(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || ",:;!?\"'()[]{}|<>".contains(c))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// Very simple suffix-stripping stemmer.  Handles common English suffixes
/// to allow "connecting" to match "connection", "errors" to match "error",
/// etc.  Not trying to be a full Porter stemmer — just enough to be useful.
fn stem(word: &str) -> String {
    let w = word.to_lowercase();
    // Order matters: try longest suffixes first.
    for suffix in &[
        "ation", "tion", "sion", "ment", "ness", "ence", "ance", "ible", "able", "ing", "ied",
        "ies", "ers", "est", "ful", "ous", "ive", "ize", "ise", "ely", "ly", "ed", "er", "es",
        "al", "en", "ty", "ry", "s",
    ] {
        if w.len() > suffix.len() + 2 && w.ends_with(suffix) {
            return w[..w.len() - suffix.len()].to_string();
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_words_basic() {
        let words = split_words("hello, world! foo-bar_baz 123");
        assert_eq!(words, vec!["hello", "world", "foo-bar_baz", "123"]);
    }

    #[test]
    fn split_words_preserves_dots() {
        let words = split_words("UPC-A EAN-13 v2.3.1");
        assert!(words.contains(&"UPC-A".to_string()));
        assert!(words.contains(&"v2.3.1".to_string()));
    }

    #[test]
    fn stem_basic() {
        assert_eq!(stem("connecting"), "connect");
        assert_eq!(stem("errors"), "error");
        assert_eq!(stem("deployment"), "deploy");
    }

    #[test]
    fn exact_substring_scores_highest() {
        let score = score_chunk(
            "The barcode scanner found UPC-A 012345678901",
            "upc-a 012345678901",
            &split_words("upc-a 012345678901"),
            &split_words("upc-a 012345678901")
                .iter()
                .map(|w| stem(w))
                .collect::<Vec<_>>(),
            &None,
        );
        assert!((score - EXACT_WEIGHT).abs() < 1e-6);
    }

    #[test]
    fn no_match_scores_zero() {
        let score = score_chunk(
            "The weather is sunny today",
            "database error",
            &split_words("database error"),
            &split_words("database error")
                .iter()
                .map(|w| stem(w))
                .collect::<Vec<_>>(),
            &None,
        );
        assert!(score < 1e-6);
    }

    #[test]
    fn partial_word_match_scores_intermediate() {
        let score = score_chunk(
            "The database connection was established",
            "database error",
            &split_words("database error"),
            &split_words("database error")
                .iter()
                .map(|w| stem(w))
                .collect::<Vec<_>>(),
            &None,
        );
        assert!(score > 0.0);
        assert!(score < EXACT_WEIGHT);
    }

    #[test]
    fn regex_match_works() {
        let re = RegexBuilder::new(r"\d{12}")
            .case_insensitive(true)
            .build()
            .ok();
        let score = score_chunk(
            "Barcode: 012345678901",
            "\\d{12}",
            &split_words("\\d{12}"),
            &split_words("\\d{12}")
                .iter()
                .map(|w| stem(w))
                .collect::<Vec<_>>(),
            &re,
        );
        assert!((score - REGEX_WEIGHT).abs() < 1e-6);
    }
}
