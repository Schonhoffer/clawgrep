//! File discovery and incremental indexing.
//!
//! Uses the `ignore` crate to respect `.gitignore` (and `.clawgrepignore`)
//! rules.  Files are split into paragraph-sized chunks using token-count
//! estimation and natural-break detection for better embedding quality.
//! Only stale files are re-embedded.
//!
//! Embeddings are persisted to a SQLite database with periodic checkpointing
//! so that interrupted indexing resumes from roughly where it stopped.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use log::{debug, info, warn};

use crate::cache::{
    file_stamp, get_entry_safe, is_fresh, open_db_resilient, upsert_entry_safe, CacheEntry,
    ChunkBoundary, CACHE_DIR_NAME,
};
use crate::embed::Embedder;

/// Maximum lines in a single chunk, to keep results granular even when
/// token counts are low (e.g. sparse code).
const MAX_CHUNK_LINES: usize = 20;

/// Target tokens per chunk.  The embedding model accepts up to 128 tokens;
/// aiming for ~100 leaves headroom for [CLS]/[SEP] special tokens.
const TARGET_CHUNK_TOKENS: usize = 100;

/// Number of overlapping lines between consecutive chunks.
const CHUNK_OVERLAP: usize = 5;

/// How many lines to search backward from the target split point for a
/// natural break (blank line or section header).
const BOUNDARY_WINDOW: usize = 5;

/// How many files to embed before committing a checkpoint to the database.
const CHECKPOINT_INTERVAL: usize = 25;

/// Options controlling file discovery.
pub struct DiscoverOpts<'a> {
    /// Respect .gitignore (default: true).
    pub use_gitignore: bool,
    /// Extra ignore-file names to load (e.g. ".clawgrepignore").
    pub custom_ignore_files: &'a [String],
}

impl Default for DiscoverOpts<'_> {
    fn default() -> Self {
        Self {
            use_gitignore: true,
            custom_ignore_files: &[],
        }
    }
}

/// Returns `true` if the file looks like a text file we want to index.
/// Heuristic: readable with no NUL bytes in the first 8 KiB.
fn is_text_file(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    use std::io::Read;
    let mut buf = [0u8; 8192];
    let n = match file.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    !buf[..n].contains(&0)
}

/// Discover all text files under `root`, respecting ignore rules.
pub fn discover_files(root: &Path, opts: &DiscoverOpts) -> Vec<PathBuf> {
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(false)
        .git_ignore(opts.use_gitignore)
        .git_global(opts.use_gitignore)
        .git_exclude(opts.use_gitignore)
        .follow_links(true);

    builder.filter_entry(|e| {
        let name = e.file_name().to_string_lossy();
        if name == CACHE_DIR_NAME {
            return false;
        }
        if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            return !matches!(name.as_ref(), ".git" | ".hg" | ".svn");
        }
        true
    });

    for ignore_name in opts.custom_ignore_files {
        builder.add_custom_ignore_filename(ignore_name);
    }

    let mut files = Vec::new();
    for entry in builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                // WalkBuilder emits errors for symlink loops, permission
                // issues, etc.  Log them so they are visible with --verbose
                // but never let them stop the walk.
                warn!("skipping entry during walk: {e}");
                continue;
            }
        };
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.into_path();
        if is_text_file(&path) {
            files.push(path);
        }
    }
    files
}

/// A chunk's text and its boundary info, used during indexing.
pub struct TextChunk {
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub boost: f32,
}

/// Split file content into overlapping, boundary-aware chunks.
pub fn chunk_file(path: &Path) -> Result<Vec<TextChunk>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let all_lines: Vec<&str> = content.lines().collect();
    if all_lines.is_empty() {
        return Ok(vec![]);
    }
    Ok(make_chunks(&all_lines))
}

/// Rough token count for a line (~4 chars per token for BERT tokenizers).
fn estimate_tokens(line: &str) -> usize {
    if line.is_empty() {
        return 0;
    }
    (line.len() / 4).max(1)
}

/// A blank line or markdown header — natural places to split chunks.
fn is_natural_break(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty() || trimmed.starts_with('#')
}

/// Core chunking logic.  Accumulates lines until *either* the estimated
/// token count reaches `TARGET_CHUNK_TOKENS` or the line count reaches
/// `MAX_CHUNK_LINES`.  Before splitting, it searches backward up to
/// `BOUNDARY_WINDOW` lines for a natural break (blank line or heading)
/// and prefers that as the split point.
fn make_chunks(lines: &[&str]) -> Vec<TextChunk> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < lines.len() {
        // Accumulate lines until we hit a limit.
        let mut tokens = 0;
        let mut end = start;
        while end < lines.len() {
            let lt = estimate_tokens(lines[end]);
            if (tokens + lt > TARGET_CHUNK_TOKENS || end - start >= MAX_CHUNK_LINES) && end > start
            {
                break;
            }
            tokens += lt;
            end += 1;
        }

        // If we stopped before EOF, look back for a natural break.
        if end < lines.len() && end > start + 1 {
            let search_from = end.saturating_sub(BOUNDARY_WINDOW).max(start + 1);
            for i in (search_from..end).rev() {
                if is_natural_break(lines[i]) {
                    end = i;
                    break;
                }
            }
        }

        // Must always make progress.
        if end == start {
            end = start + 1;
        }

        let text = lines[start..end].join("\n");
        chunks.push(TextChunk {
            start_line: start + 1,
            end_line: end,
            text,
            boost: 1.0,
        });

        if end >= lines.len() {
            break;
        }

        // Next chunk starts CHUNK_OVERLAP lines before the end.
        start = end.saturating_sub(CHUNK_OVERLAP).max(start + 1);
    }

    chunks
}

/// Options for `build_index`.
pub struct IndexOpts<'a> {
    /// Force re-embed all files.
    pub reindex: bool,
    /// Don't read or write cache.
    pub no_cache: bool,
    /// Optional custom cache directory.
    pub custom_cache: Option<&'a Path>,
    /// Boost factor for file/folder path match chunks.
    /// 1.0 = same weight as content, 0.0 = no path chunks.
    pub path_boost: f32,
    /// Print per-file progress to stderr.
    pub verbose: bool,
}

/// The result of `build_index`: a list of per-file cache entries holding
/// embeddings and chunk boundaries (but not chunk text).
pub struct Index {
    pub entries: Vec<CacheEntry>,
}

/// Build or update the embeddings index for the given files.
///
/// Returns an `Index` containing cache entries for every file
/// (from cache if fresh, or freshly embedded if stale).
///
/// Checkpoints to SQLite every `CHECKPOINT_INTERVAL` files so that
/// interrupted runs can resume.
pub fn build_index(files: &[PathBuf], embedder: &Embedder, opts: &IndexOpts) -> Result<Index> {
    let model = embedder.model_name();

    // If no_cache, skip DB entirely — embed everything in memory.
    if opts.no_cache {
        return build_index_no_cache(files, embedder, opts);
    }

    // Open cache resiliently: if it fails, fall back to no-cache mode.
    let conn = open_db_resilient(opts.custom_cache);
    let Some(conn) = conn else {
        return build_index_no_cache(files, embedder, opts);
    };

    // Determine which files need (re-)embedding.
    let mut fresh_entries: Vec<CacheEntry> = Vec::new();
    let mut to_embed: Vec<PathBuf> = Vec::new();

    for file in files {
        let path_str = file.to_string_lossy().to_string();
        if !opts.reindex {
            if let Some(entry) = get_entry_safe(&conn, &path_str, model) {
                if is_fresh(&entry, file) {
                    fresh_entries.push(entry);
                    continue;
                }
            }
        }
        to_embed.push(file.clone());
    }

    if !to_embed.is_empty() {
        info!("indexing {} files", to_embed.len());
        eprintln!("clawgrep: indexing {} files...", to_embed.len());

        // Embed in batches of CHECKPOINT_INTERVAL, committing after each batch.
        for (batch_idx, batch) in to_embed.chunks(CHECKPOINT_INTERVAL).enumerate() {
            let batch_start = batch_idx * CHECKPOINT_INTERVAL;
            let new_entries = embed_files(
                batch,
                embedder,
                opts.path_boost,
                opts.verbose,
                batch_start,
                to_embed.len(),
            )?;

            // Checkpoint: write this batch to SQLite.
            for entry in &new_entries {
                upsert_entry_safe(&conn, entry);
            }

            fresh_entries.extend(new_entries);
        }
    }

    Ok(Index {
        entries: fresh_entries,
    })
}

/// Build index without any cache (everything in memory).
fn build_index_no_cache(files: &[PathBuf], embedder: &Embedder, opts: &IndexOpts) -> Result<Index> {
    if files.is_empty() {
        return Ok(Index {
            entries: Vec::new(),
        });
    }
    let entries = embed_files(
        files,
        embedder,
        opts.path_boost,
        opts.verbose,
        0,
        files.len(),
    )?;
    Ok(Index { entries })
}

/// Build an index from raw text content (e.g. stdin).  No caching.
/// The `label` is used as the synthetic file path in results.
pub fn build_index_from_content(
    label: &str,
    content: &str,
    embedder: &Embedder,
    path_boost: f32,
) -> Result<Index> {
    let all_lines: Vec<&str> = content.lines().collect();
    if all_lines.is_empty() {
        return Ok(Index {
            entries: Vec::new(),
        });
    }
    let mut chunks = make_chunks(&all_lines);
    if path_boost > 0.0 {
        chunks.insert(
            0,
            TextChunk {
                start_line: 0,
                end_line: 0,
                text: label.to_string(),
                boost: path_boost,
            },
        );
    }

    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = embedder.embed_batch(&texts, false)?;
    let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);

    let mut flat: Vec<f32> = Vec::with_capacity(chunks.len() * dim);
    for emb in &embeddings {
        flat.extend_from_slice(emb);
    }

    let boundaries: Vec<ChunkBoundary> = chunks
        .iter()
        .map(|c| ChunkBoundary {
            start_line: c.start_line,
            end_line: c.end_line,
            boost: c.boost,
        })
        .collect();

    let entry = CacheEntry {
        path: label.to_string(),
        mtime_ms: 0,
        file_size: content.len() as u64,
        num_chunks: chunks.len(),
        dim,
        embedding_model: embedder.model_name().to_string(),
        chunks: boundaries,
        embeddings: flat,
    };

    Ok(Index {
        entries: vec![entry],
    })
}

/// Build a synthetic chunk from a file's path for path-based matching.
fn path_chunk(path: &Path, boost: f32) -> TextChunk {
    let path_str = path.to_string_lossy();
    TextChunk {
        start_line: 0,
        end_line: 0,
        text: path_str.into_owned(),
        boost,
    }
}

/// Embed a set of files and return cache entries.
fn embed_files(
    files: &[PathBuf],
    embedder: &Embedder,
    path_boost: f32,
    verbose: bool,
    offset: usize,
    total: usize,
) -> Result<Vec<CacheEntry>> {
    let model = embedder.model_name().to_string();

    // Collect all chunks across files so we can batch-embed.
    let mut file_chunks: Vec<(PathBuf, Vec<TextChunk>)> = Vec::new();
    let mut all_texts: Vec<String> = Vec::new();
    let mut offsets: Vec<(usize, usize)> = Vec::new();

    for (fi, path) in files.iter().enumerate() {
        if verbose {
            eprintln!(
                "clawgrep: [{}/{}] indexing {}",
                offset + fi + 1,
                total,
                path.display()
            );
        }
        let mut chunks = match chunk_file(path) {
            Ok(c) => c,
            Err(e) => {
                warn!("{}: {e}", path.display());
                eprintln!("clawgrep: warning: {}: {e}", path.display());
                continue;
            }
        };
        if path_boost > 0.0 {
            chunks.insert(0, path_chunk(path, path_boost));
        }
        let start = all_texts.len();
        for cm in &chunks {
            all_texts.push(cm.text.clone());
        }
        offsets.push((start, chunks.len()));
        file_chunks.push((path.clone(), chunks));
    }

    if all_texts.is_empty() {
        return Ok(Vec::new());
    }

    if verbose {
        eprintln!("clawgrep: embedding {} total segments...", all_texts.len());
    }

    let text_refs: Vec<&str> = all_texts.iter().map(|s| s.as_str()).collect();
    let all_embeddings = embedder.embed_batch(&text_refs, verbose)?;
    let dim = all_embeddings.first().map(|v| v.len()).unwrap_or(0);

    let mut entries = Vec::new();
    for ((path, chunks), &(start, count)) in file_chunks.iter().zip(offsets.iter()) {
        let mut flat: Vec<f32> = Vec::with_capacity(count * dim);
        for emb in &all_embeddings[start..start + count] {
            flat.extend_from_slice(emb);
        }

        let (mtime_ms, size) = file_stamp(path)?;
        let boundaries: Vec<ChunkBoundary> = chunks
            .iter()
            .map(|c| ChunkBoundary {
                start_line: c.start_line,
                end_line: c.end_line,
                boost: c.boost,
            })
            .collect();

        debug!("cached {} ({} segments)", path.display(), count);
        entries.push(CacheEntry {
            path: path.to_string_lossy().into(),
            mtime_ms,
            file_size: size,
            num_chunks: count,
            dim,
            embedding_model: model.clone(),
            chunks: boundaries,
            embeddings: flat,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_skips_hidden_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        fs::write(root.join("hello.txt"), "hello world\n").unwrap();

        let sf = root.join(CACHE_DIR_NAME);
        fs::create_dir_all(&sf).unwrap();
        fs::write(sf.join("cache.db"), "data\n").unwrap();

        let found = discover_files(
            root,
            &DiscoverOpts {
                use_gitignore: false,
                ..Default::default()
            },
        );
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("hello.txt"));
    }

    #[test]
    fn chunk_basic() {
        let lines: Vec<&str> = (0..50).map(|_| "some text").collect();
        let chunks = make_chunks(&lines);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 20);
    }

    #[test]
    fn chunk_small_file() {
        let lines = vec!["hello", "world"];
        let chunks = make_chunks(&lines);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 2);
    }

    #[test]
    fn chunk_preserves_text() {
        let lines = vec!["error connecting to database"];
        let chunks = make_chunks(&lines);
        assert!(chunks[0].text.contains("error"));
        assert!(chunks[0].text.contains("database"));
    }
}
