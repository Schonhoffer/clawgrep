use std::path::PathBuf;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use clawgrep::embed::Embedder;
use clawgrep::index::{build_index, discover_files, DiscoverOpts, IndexOpts};
use clawgrep::search::{hybrid_search, SearchOpts};

/// A single search result returned to JavaScript.
#[napi(object)]
pub struct JsSearchResult {
    /// File path containing the match.
    pub file: String,
    /// 1-based start line of the matching chunk.
    pub line_num: u32,
    /// End line of the matching chunk.
    pub end_line: u32,
    /// The matched chunk text.
    pub text: String,
    /// Combined relevance score (higher = more relevant).
    pub score: f64,
    /// Semantic component of the score.
    pub semantic_score: f64,
    /// Keyword component of the score.
    pub keyword_score: f64,
}

/// Options for the search function.
#[napi(object)]
pub struct JsSearchOptions {
    /// Number of results to return (default: 5).
    pub top_k: Option<u32>,
    /// Minimum combined score cutoff (0.0–1.0).
    pub min_score: Option<f64>,
    /// Weight for semantic similarity (default: 0.7).
    pub semantic_weight: Option<f64>,
    /// Weight for keyword matching (default: 0.3).
    pub keyword_weight: Option<f64>,
    /// Boost factor for file/folder path matches (default: 1.0).
    pub path_boost: Option<f64>,
    /// Force re-embed all files.
    pub reindex: Option<bool>,
    /// Don't read or write caches.
    pub no_cache: Option<bool>,
    /// Custom cache directory.
    pub cache_dir: Option<String>,
    /// Do not respect .gitignore files.
    pub no_gitignore: Option<bool>,
}

/// Search files by meaning and keywords.
///
/// `query` — natural language or keyword search string.
/// `paths` — array of file or directory paths to search.
/// `options` — optional search configuration.
#[napi]
pub fn search(
    query: String,
    paths: Vec<String>,
    options: Option<JsSearchOptions>,
) -> Result<Vec<JsSearchResult>> {
    let opts = options.unwrap_or(JsSearchOptions {
        top_k: None,
        min_score: None,
        semantic_weight: None,
        keyword_weight: None,
        path_boost: None,
        reindex: None,
        no_cache: None,
        cache_dir: None,
        no_gitignore: None,
    });

    let search_opts = SearchOpts {
        top_k: opts.top_k.unwrap_or(5) as usize,
        min_score: opts.min_score.map(|v| v as f32),
        semantic_weight: opts.semantic_weight.unwrap_or(0.7) as f32,
        keyword_weight: opts.keyword_weight.unwrap_or(0.3) as f32,
    };

    let path_boost = opts.path_boost.unwrap_or(1.0) as f32;
    let reindex = opts.reindex.unwrap_or(false);
    let no_cache = opts.no_cache.unwrap_or(false);
    let cache_dir_path = opts.cache_dir.map(PathBuf::from);
    let no_gitignore = opts.no_gitignore.unwrap_or(false);

    let discover_opts = DiscoverOpts {
        use_gitignore: !no_gitignore,
        custom_ignore_files: &[],
    };

    // Discover files from all given paths.
    let mut files = Vec::new();
    for p in &paths {
        let path = PathBuf::from(p);
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            files.extend(discover_files(&path, &discover_opts));
        }
    }

    if files.is_empty() {
        return Ok(vec![]);
    }

    let embedder =
        Embedder::new(cache_dir_path.as_deref()).map_err(|e| Error::from_reason(format!("{e}")))?;

    let index_opts = IndexOpts {
        reindex,
        no_cache,
        custom_cache: cache_dir_path.as_deref(),
        path_boost,
        verbose: false,
    };

    let index = build_index(&files, &embedder, &index_opts)
        .map_err(|e| Error::from_reason(format!("{e}")))?;

    let results = hybrid_search(&query, &index, &files, &embedder, &search_opts)
        .map_err(|e| Error::from_reason(format!("{e}")))?;

    Ok(results
        .into_iter()
        .map(|r| JsSearchResult {
            file: r.file.to_string_lossy().to_string(),
            line_num: r.line_num as u32,
            end_line: r.end_line as u32,
            text: r.text,
            score: r.score as f64,
            semantic_score: r.semantic_score as f64,
            keyword_score: r.keyword_score as f64,
        })
        .collect())
}
