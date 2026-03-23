use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use clawgrep::embed::Embedder;
use clawgrep::index::{build_index, discover_files, DiscoverOpts, IndexOpts};
use clawgrep::search::{hybrid_search, SearchOpts};

/// A single search result.
#[pyclass]
#[derive(Clone)]
pub struct SearchResult {
    #[pyo3(get)]
    pub file: String,
    #[pyo3(get)]
    pub line_num: usize,
    #[pyo3(get)]
    pub end_line: usize,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub semantic_score: f64,
    #[pyo3(get)]
    pub keyword_score: f64,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(file={:?}, line_num={}, score={:.4})",
            self.file, self.line_num, self.score
        )
    }
}

/// Search files by meaning and keywords.
///
/// Args:
///     query: Natural language or keyword search string.
///     paths: List of file or directory paths to search.
///     top_k: Number of results to return (default: 5).
///     min_score: Minimum combined score cutoff (0.0-1.0).
///     semantic_weight: Weight for semantic similarity (default: 0.7).
///     keyword_weight: Weight for keyword matching (default: 0.3).
///     path_boost: Boost for file/folder path matches (default: 1.0).
///     reindex: Force re-embed all files.
///     no_cache: Don't read or write caches.
///     cache_dir: Custom cache directory path.
///     no_gitignore: Do not respect .gitignore files.
///
/// Returns:
///     List of SearchResult objects sorted by relevance.
#[pyfunction]
#[pyo3(signature = (
    query,
    paths,
    *,
    top_k = 5,
    min_score = None,
    semantic_weight = 0.7,
    keyword_weight = 0.3,
    path_boost = 1.0,
    reindex = false,
    no_cache = false,
    cache_dir = None,
    no_gitignore = false,
))]
#[allow(clippy::too_many_arguments)]
fn search(
    query: &str,
    paths: Vec<String>,
    top_k: usize,
    min_score: Option<f64>,
    semantic_weight: f64,
    keyword_weight: f64,
    path_boost: f64,
    reindex: bool,
    no_cache: bool,
    cache_dir: Option<String>,
    no_gitignore: bool,
) -> PyResult<Vec<SearchResult>> {
    let search_opts = SearchOpts {
        top_k,
        min_score: min_score.map(|v| v as f32),
        semantic_weight: semantic_weight as f32,
        keyword_weight: keyword_weight as f32,
    };

    let cache_dir_path = cache_dir.map(PathBuf::from);

    let discover_opts = DiscoverOpts {
        use_gitignore: !no_gitignore,
        custom_ignore_files: &[],
    };

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

    let embedder = Embedder::new(cache_dir_path.as_deref())
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

    let index_opts = IndexOpts {
        reindex,
        no_cache,
        custom_cache: cache_dir_path.as_deref(),
        path_boost: path_boost as f32,
        verbose: false,
    };

    let index = build_index(&files, &embedder, &index_opts)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

    let results = hybrid_search(query, &index, &files, &embedder, &search_opts)
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

    Ok(results
        .into_iter()
        .map(|r| SearchResult {
            file: r.file.to_string_lossy().to_string(),
            line_num: r.line_num,
            end_line: r.end_line,
            text: r.text,
            score: r.score as f64,
            semantic_score: r.semantic_score as f64,
            keyword_score: r.keyword_score as f64,
        })
        .collect())
}

/// clawgrep Python module.
#[pymodule(name = "clawgrep")]
fn clawgrep_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_class::<SearchResult>()?;
    Ok(())
}
