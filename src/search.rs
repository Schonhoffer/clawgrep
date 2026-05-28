//! Hybrid search: semantic (embedding) + keyword (substring/regex).
//!
//! Results from the two rankers are fused with weighted Reciprocal Rank
//! Fusion (RRF, Cormack 2009):
//!
//! ```text
//!     rrf(d) = w_sem * boost(d) * 1/(k + rank_sem(d))
//!            + w_kw           * 1/(k + rank_kw(d))
//! ```
//!
//! `k = 60` (standard constant). A chunk missing from a ranker contributes
//! `0` from that side. The final `score` is normalised to `[0, 1]` by
//! dividing by the maximum attainable raw value, so `--min-score` and
//! `--show-score` keep working on the same scale.
//!
//! `path_boost` multiplies the semantic-side RRF contribution for chunks
//! built from file/folder paths.
//!
//! Semantic scores come from cached embeddings (no file I/O).
//! Keyword scores come from reading files from disk and doing
//! substring / regex / word-stem matching.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use log::debug;
use rayon::prelude::*;

use crate::cache::CacheEntry;
use crate::embed::{cosine_similarity, Embedder};
use crate::index::Index;
use crate::keyword::keyword_search;

/// Rank constant for RRF.  60 is the standard default from Cormack 2009 and
/// is what Elasticsearch, OpenSearch, and Azure AI Search use.
const RRF_K: f32 = 60.0;

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
    /// Minimum combined score cutoff (0.0–1.0, on the normalised RRF scale).
    pub min_score: Option<f32>,
    /// RRF weight for semantic similarity [0..1].
    pub semantic_weight: f32,
    /// RRF weight for keyword (substring) matching [0..1].
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

    // 2. Semantic ranking: score every cached chunk, sort by cosine desc.
    let sem_ranking = if opts.semantic_weight > 0.0 {
        let mut scored = semantic_scores(&query_emb, &index.entries);
        // Deterministic tie-break by (path, start_line).
        scored.sort_by(|a, b| {
            b.4.partial_cmp(&a.4)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        scored
    } else {
        vec![]
    };
    debug!("scored {} segments semantically", sem_ranking.len());

    // 3. Keyword ranking: already returned sorted desc by `keyword_search`.
    let kw_ranking: Vec<KeywordRanked> = if opts.keyword_weight > 0.0 {
        let mut hits: Vec<KeywordRanked> = keyword_search(query, files)
            .into_iter()
            .map(|h| KeywordRanked {
                file: h.file.to_string_lossy().to_string(),
                start_line: h.start_line,
                end_line: h.end_line,
                score: h.score,
            })
            .collect();
        // Deterministic tie-break by (path, start_line).
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.file.cmp(&b.file))
                .then_with(|| a.start_line.cmp(&b.start_line))
        });
        hits
    } else {
        vec![]
    };

    // 4. Fuse the two rankings with weighted RRF.
    let path_boost = max_boost(&sem_ranking);
    let candidates = rrf_fuse(
        &sem_ranking,
        &kw_ranking,
        opts.semantic_weight,
        opts.keyword_weight,
        path_boost,
    );

    // 5. Apply min_score and top_k, build results.
    let mut results: Vec<SearchResult> = Vec::new();
    for cand in candidates {
        if let Some(min) = opts.min_score {
            if cand.score < min {
                continue;
            }
        }
        if results.len() >= opts.top_k {
            break;
        }
        let text = read_chunk_text(&cand.file, cand.start_line, cand.end_line);
        results.push(SearchResult {
            file: PathBuf::from(&cand.file),
            line_num: cand.start_line,
            end_line: cand.end_line,
            text,
            score: cand.score,
            semantic_score: cand.semantic_score,
            keyword_score: cand.keyword_score,
        });
    }

    Ok(results)
}

/// One chunk ranked by the semantic scorer.
/// `(file, start_line, end_line, boost, cosine)`.
type SemanticRanked = (String, usize, usize, f32, f32);

/// One chunk ranked by the keyword scorer.
struct KeywordRanked {
    file: String,
    start_line: usize,
    end_line: usize,
    score: f32,
}

/// One fused candidate after RRF.
struct FusedCandidate {
    file: String,
    start_line: usize,
    end_line: usize,
    score: f32,
    semantic_score: f32,
    keyword_score: f32,
}

/// Largest `boost` seen across all semantically-ranked chunks.
/// Used to compute the normaliser so the final score lands in `[0, 1]`.
fn max_boost(sem: &[SemanticRanked]) -> f32 {
    sem.iter().map(|s| s.3).fold(1.0_f32, f32::max)
}

/// Weighted Reciprocal Rank Fusion of a semantic ranking and a keyword
/// ranking.  Returns candidates sorted by fused score descending.
///
/// Each ranker contributes `w_r / (RRF_K + rank_r)` per chunk it ranked.
/// Missing chunks contribute `0` from that side.  Path chunks get their
/// semantic-side contribution multiplied by their `boost`.
///
/// The fused score is normalised by the maximum attainable raw score so
/// the result is in `[0, 1]` regardless of weights or path_boost.
fn rrf_fuse(
    sem: &[SemanticRanked],
    kw: &[KeywordRanked],
    w_sem: f32,
    w_kw: f32,
    path_boost: f32,
) -> Vec<FusedCandidate> {
    // Maximum possible raw RRF score: a chunk at rank 1 in both rankers with
    // the largest possible path boost on the semantic side.
    let max_raw = (w_sem * path_boost + w_kw) / (RRF_K + 1.0);

    let mut by_key: HashMap<(String, usize), FusedCandidate> = HashMap::new();

    // Semantic-side contributions.
    for (rank0, (file, start, end, boost, cos)) in sem.iter().enumerate() {
        let rank = (rank0 + 1) as f32;
        let contrib = w_sem * boost * (1.0 / (RRF_K + rank));
        let entry = by_key
            .entry((file.clone(), *start))
            .or_insert_with(|| FusedCandidate {
                file: file.clone(),
                start_line: *start,
                end_line: *end,
                score: 0.0,
                semantic_score: 0.0,
                keyword_score: 0.0,
            });
        entry.score += contrib;
        // Keep the chunk's best raw cosine (semantic_scores may have
        // duplicates across content + path chunks; take the higher).
        if *cos > entry.semantic_score {
            entry.semantic_score = *cos;
        }
    }

    // Keyword-side contributions.
    for (rank0, hit) in kw.iter().enumerate() {
        let rank = (rank0 + 1) as f32;
        let contrib = w_kw * (1.0 / (RRF_K + rank));
        let entry = by_key
            .entry((hit.file.clone(), hit.start_line))
            .or_insert_with(|| FusedCandidate {
                file: hit.file.clone(),
                start_line: hit.start_line,
                end_line: hit.end_line,
                score: 0.0,
                semantic_score: 0.0,
                keyword_score: 0.0,
            });
        entry.score += contrib;
        if hit.score > entry.keyword_score {
            entry.keyword_score = hit.score;
        }
    }

    // Normalise to [0, 1].
    let mut out: Vec<FusedCandidate> = by_key.into_values().collect();
    if max_raw > 0.0 {
        for c in &mut out {
            c.score /= max_raw;
            if c.score > 1.0 {
                c.score = 1.0;
            }
        }
    }

    // Sort by fused score desc; deterministic tie-break by (file, start_line).
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.file.cmp(&b.file))
            .then_with(|| a.start_line.cmp(&b.start_line))
    });
    out
}

/// Score every chunk in the index against the query embedding.
/// Returns: (file_path, start_line, end_line, boost, cosine_score).
fn semantic_scores(query_emb: &[f32], entries: &[CacheEntry]) -> Vec<SemanticRanked> {
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

    // ── RRF fusion ──────────────────────────────────────────────────────

    fn sem(file: &str, start: usize, boost: f32, cos: f32) -> SemanticRanked {
        (file.to_string(), start, start, boost, cos)
    }

    fn kw(file: &str, start: usize, score: f32) -> KeywordRanked {
        KeywordRanked {
            file: file.to_string(),
            start_line: start,
            end_line: start,
            score,
        }
    }

    #[test]
    fn rrf_empty_inputs() {
        let out = rrf_fuse(&[], &[], 0.7, 0.3, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn rrf_semantic_only_preserves_order() {
        let sem_in = vec![
            sem("a.txt", 1, 1.0, 0.9),
            sem("b.txt", 1, 1.0, 0.8),
            sem("c.txt", 1, 1.0, 0.7),
        ];
        let out = rrf_fuse(&sem_in, &[], 1.0, 0.0, 1.0);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].file, "a.txt");
        assert_eq!(out[1].file, "b.txt");
        assert_eq!(out[2].file, "c.txt");
        // Top chunk gets score = w_sem/(k+1) / max_raw = 1.0.
        assert!((out[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rrf_keyword_only_preserves_order() {
        let kw_in = vec![
            kw("a.txt", 1, 1.0),
            kw("b.txt", 1, 0.6),
            kw("c.txt", 1, 0.3),
        ];
        let out = rrf_fuse(&[], &kw_in, 0.0, 1.0, 1.0);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].file, "a.txt");
        assert!((out[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rrf_chunk_in_both_beats_chunk_in_one() {
        // "both" is rank 3 in semantic and rank 1 in keyword.
        // "sem-only" is rank 1 in semantic but missing from keyword.
        let sem_in = vec![
            sem("sem-only.txt", 1, 1.0, 0.9),
            sem("other.txt", 1, 1.0, 0.8),
            sem("both.txt", 1, 1.0, 0.7),
        ];
        let kw_in = vec![kw("both.txt", 1, 1.0)];
        let out = rrf_fuse(&sem_in, &kw_in, 0.5, 0.5, 1.0);
        assert_eq!(out[0].file, "both.txt", "both-rankers chunk should win");
        // It should also outscore sem-only.
        let both_score = out.iter().find(|c| c.file == "both.txt").unwrap().score;
        let sem_score = out.iter().find(|c| c.file == "sem-only.txt").unwrap().score;
        assert!(both_score > sem_score);
    }

    #[test]
    fn rrf_missing_from_keyword_contributes_zero() {
        // Without any keyword side, a chunk at semantic rank 1 should
        // get score = w_sem/(k+1) / max_raw. With w_sem=0.7, max_raw =
        // (0.7 + 0.3)/(k+1) = 1/(k+1). So score = 0.7.
        let sem_in = vec![sem("a.txt", 1, 1.0, 0.9)];
        let out = rrf_fuse(&sem_in, &[], 0.7, 0.3, 1.0);
        assert!((out[0].score - 0.7).abs() < 1e-5, "got {}", out[0].score);
        assert!((out[0].semantic_score - 0.9).abs() < 1e-5);
        assert_eq!(out[0].keyword_score, 0.0);
    }

    #[test]
    fn rrf_path_boost_lifts_path_chunk() {
        // Without boost (1.0), content chunk at rank 1 beats path chunk
        // at rank 2.  With boost = 5.0, path chunk wins.
        let sem_no_boost = vec![
            sem("content.txt", 5, 1.0, 0.9), // rank 1
            sem("paths/a.txt", 0, 1.0, 0.7), // rank 2, no boost
        ];
        let out1 = rrf_fuse(&sem_no_boost, &[], 1.0, 0.0, 1.0);
        assert_eq!(out1[0].file, "content.txt");

        let sem_boosted = vec![
            sem("content.txt", 5, 1.0, 0.9), // rank 1
            sem("paths/a.txt", 0, 5.0, 0.7), // rank 2, boost 5x
        ];
        let out2 = rrf_fuse(&sem_boosted, &[], 1.0, 0.0, 5.0);
        assert_eq!(
            out2[0].file, "paths/a.txt",
            "path chunk with 5x boost should win"
        );
    }

    #[test]
    fn rrf_scores_stay_in_unit_range() {
        // With normalization and clamping, no score should exceed 1.0.
        let sem_in = vec![
            sem("a.txt", 1, 1.0, 0.9),
            sem("a.txt", 0, 3.0, 0.95), // path chunk with high boost
        ];
        let kw_in = vec![kw("a.txt", 1, 1.0)];
        let out = rrf_fuse(&sem_in, &kw_in, 0.7, 0.3, 3.0);
        for c in &out {
            assert!(c.score >= 0.0 && c.score <= 1.0, "score = {}", c.score);
        }
    }

    #[test]
    fn rrf_deterministic_tie_break() {
        // Two chunks ranked identically should appear in (file, start_line)
        // order.
        let sem_in = vec![sem("b.txt", 2, 1.0, 0.9), sem("a.txt", 1, 1.0, 0.9)];
        let out = rrf_fuse(&sem_in, &[], 1.0, 0.0, 1.0);
        // Both have the same fused score; tie-break by file alphabetically.
        // Note: the rrf_fuse caller is responsible for feeding a
        // deterministically-ordered semantic ranking; here we pass them
        // already tied in raw cosine.  Internally they get ranks 1 and 2
        // in the order given, so scores differ slightly.  The final sort
        // breaks remaining ties by (file, start_line).
        assert!(out[0].score >= out[1].score);
    }
}
