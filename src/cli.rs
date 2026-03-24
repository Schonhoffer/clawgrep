//! CLI entry point logic for clawgrep.
//!
//! This module contains the full CLI implementation, callable from both
//! the standalone binary (`main.rs`) and foreign-language entry points
//! (e.g. the Python console script).

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::config::load_config;
use crate::embed::Embedder;
use crate::index::{
    build_index, build_index_from_content, discover_files, DiscoverOpts, IndexOpts,
};
use crate::search::{hybrid_search, SearchOpts};

// ── CLI argument validation ─────────────────────────────────────────────

fn parse_weight(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err("value must be between 0.0 and 1.0".into())
    }
}

fn parse_min_score(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err("value must be between 0.0 and 1.0".into())
    }
}

fn parse_path_boost(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if v >= 0.0 {
        Ok(v)
    } else {
        Err("value must be >= 0.0".into())
    }
}

/// Grep-like CLI with hybrid semantic and keyword search.
///
/// Combines embedding-based semantic search with substring/regex keyword matching.
/// Output is grep-compatible: file:line:text
#[derive(Parser, Debug)]
#[command(name = "clawgrep", version, about)]
struct Cli {
    /// Search query (natural language or keywords).
    query: String,

    /// Files or directories to search.  When a directory is given all text
    /// files underneath it are searched recursively.  If omitted, reads
    /// from stdin.
    paths: Vec<PathBuf>,

    /// Number of results to return.
    #[arg(short = 'k', long = "top-k", default_value_t = 5)]
    top_k: usize,

    /// Lines of context to show before each match (like grep -B).
    #[arg(short = 'B', long = "before-context", default_value_t = 0)]
    before: usize,

    /// Lines of context to show after each match (like grep -A).
    #[arg(short = 'A', long = "after-context", default_value_t = 0)]
    after: usize,

    /// Lines of context to show before and after each match (like grep -C).
    #[arg(short = 'C', long = "context", default_value_t = 0)]
    context: usize,

    /// Minimum combined score threshold (0.0–1.0).
    #[arg(long = "min-score", value_parser = parse_min_score)]
    min_score: Option<f32>,

    /// Weight for semantic (embedding) similarity [0.0–1.0].
    #[arg(long = "semantic-weight", default_value_t = 0.7, value_parser = parse_weight)]
    semantic_weight: f32,

    /// Weight for keyword (substring/regex) similarity [0.0–1.0].
    #[arg(long = "keyword-weight", default_value_t = 0.3, value_parser = parse_weight)]
    keyword_weight: f32,

    /// Force re-embed all files (ignore cache).
    #[arg(long)]
    reindex: bool,

    /// Don't read or write the cache.
    #[arg(long)]
    no_cache: bool,

    /// Custom cache directory (default: .clawgrep/ under the search root).
    #[arg(long = "cache-dir")]
    cache_dir: Option<PathBuf>,

    /// Print only filenames of matching files (like grep -l).
    #[arg(short = 'l', long = "files-with-matches")]
    list_files: bool,

    /// Print only a count of matching lines per file (like grep -c).
    #[arg(short = 'c', long = "count")]
    count: bool,

    /// Suppress all normal output; exit with 0 if any match (like grep -q).
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,

    /// Never use colours in output.
    #[arg(long = "no-color")]
    no_color: bool,

    /// Do not recurse into subdirectories.
    #[arg(long = "no-recursive")]
    no_recursive: bool,

    /// Do not respect .gitignore files.
    #[arg(long = "no-gitignore")]
    no_gitignore: bool,

    /// Additional ignore file names (same syntax as .gitignore).
    /// Can be specified multiple times.
    #[arg(long = "ignore-file", action = clap::ArgAction::Append)]
    ignore_file: Vec<String>,

    /// Boost factor for file/folder path matches vs content matches.
    /// 1.0 = same weight, 2.0 = path matches count double, 0.0 = ignore paths.
    #[arg(long = "path-boost", default_value_t = 1.0, value_parser = parse_path_boost)]
    path_boost: f32,

    /// Print progress and diagnostic information to stderr.
    /// Also enabled by setting CLAWGREP_VERBOSE=1.
    #[arg(long = "verbose")]
    verbose: bool,

    /// Print the relevance score after each match line.
    #[arg(long = "show-score")]
    show_score: bool,

    /// Accepted for compatibility with grep but ignored.
    /// clawgrep is always case-insensitive.
    #[arg(short = 'i', long = "ignore-case", hide = true)]
    _ignore_case: bool,
}

// ── ANSI helpers ────────────────────────────────────────────────────────

fn magenta(s: &str, color: bool) -> String {
    if color {
        format!("\x1b[35m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
fn green(s: &str, color: bool) -> String {
    if color {
        format!("\x1b[32m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

fn red_bold(s: &str, color: bool) -> String {
    if color {
        format!("\x1b[1;31m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
fn cyan(s: &str, color: bool) -> String {
    if color {
        format!("\x1b[36m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

// ── Context printing ────────────────────────────────────────────────────

fn read_file_lines(path: &PathBuf) -> Vec<String> {
    fs::read_to_string(path)
        .map(|c| c.lines().map(String::from).collect())
        .unwrap_or_default()
}

/// Highlight occurrences of query words in `text` using red bold.
fn highlight_matches(text: &str, query: &str, color: bool) -> String {
    if !color || query.is_empty() {
        return text.to_string();
    }
    let words: Vec<&str> = query.split_whitespace().filter(|w| !w.is_empty()).collect();
    if words.is_empty() {
        return text.to_string();
    }
    let escaped: Vec<String> = words.iter().map(|w| regex::escape(w)).collect();
    let pattern = escaped.join("|");
    let Ok(re) = regex::RegexBuilder::new(&pattern)
        .case_insensitive(true)
        .build()
    else {
        return text.to_string();
    };
    let mut result = String::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        result.push_str(&text[last..m.start()]);
        result.push_str(&red_bold(m.as_str(), true));
        last = m.end();
    }
    result.push_str(&text[last..]);
    result
}

// ── Public entry point ──────────────────────────────────────────────────

/// Run the clawgrep CLI with the given arguments.
///
/// Returns an exit code: 0 = match found, 1 = no match, 2 = error.
pub fn run<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let args: Vec<OsString> = args.into_iter().map(|a| a.into()).collect();
    let cli = match Cli::try_parse_from(&args) {
        Ok(c) => c,
        Err(e) => {
            let _ = e.print();
            return match e.kind() {
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => 0,
                _ => 2,
            };
        }
    };

    let raw_args: Vec<String> = args
        .iter()
        .filter_map(|a| a.to_str().map(String::from))
        .collect();

    match run_inner(cli, &raw_args) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(e) => {
            eprintln!("clawgrep: {e:#}");
            2
        }
    }
}

fn run_inner(cli: Cli, raw_args: &[String]) -> Result<bool> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .try_init();

    let verbose = cli.verbose || std::env::var("CLAWGREP_VERBOSE").is_ok_and(|v| v == "1");
    let cfg = load_config();

    // Merge: CLI flags take precedence over config file.
    let semantic_weight = if raw_args.iter().any(|a| a == "--semantic-weight") {
        cli.semantic_weight
    } else {
        cfg.semantic_weight.unwrap_or(cli.semantic_weight)
    };
    let keyword_weight = if raw_args.iter().any(|a| a == "--keyword-weight") {
        cli.keyword_weight
    } else {
        cfg.keyword_weight.unwrap_or(cli.keyword_weight)
    };
    let top_k = if raw_args.iter().any(|a| a == "--top-k" || a == "-k") {
        cli.top_k
    } else {
        cfg.top_k.unwrap_or(cli.top_k)
    };
    let min_score = if raw_args.iter().any(|a| a == "--min-score") {
        cli.min_score
    } else {
        cli.min_score.or(cfg.min_score)
    };
    let cache_dir = if raw_args.iter().any(|a| a == "--cache-dir") {
        cli.cache_dir.clone()
    } else {
        cfg.cache_dir
            .clone()
            .or_else(|| std::env::var("CLAWGREP_CACHE_DIR").ok().map(PathBuf::from))
    };
    let no_gitignore = if raw_args.iter().any(|a| a == "--no-gitignore") {
        cli.no_gitignore
    } else {
        cfg.no_gitignore.unwrap_or(cli.no_gitignore)
    };
    let path_boost = if raw_args.iter().any(|a| a == "--path-boost") {
        cli.path_boost
    } else {
        cfg.path_boost.unwrap_or(cli.path_boost)
    };

    let color =
        !cli.no_color && std::env::var_os("NO_COLOR").is_none() && io::stdout().is_terminal();
    let reading_stdin = cli.paths.is_empty();

    // Load model.
    if verbose {
        eprintln!("clawgrep: loading model...");
    }
    let embedder = Embedder::new(cache_dir.as_deref())?;
    if verbose {
        eprintln!("clawgrep: model loaded");
    }

    let search_opts = SearchOpts {
        top_k,
        min_score,
        semantic_weight,
        keyword_weight,
    };

    // ── stdin vs file paths ──────────────────────────────────────────────
    let (results, files, stdin_lines) = if reading_stdin {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| anyhow::anyhow!("reading stdin: {e}"))?;
        let lines: Vec<String> = buf.lines().map(String::from).collect();
        if verbose {
            eprintln!("clawgrep: read {} lines from stdin", lines.len());
        }
        let index = build_index_from_content("(standard input)", &buf, &embedder, path_boost)?;
        let stdin_path = PathBuf::from("(standard input)");
        let files = vec![stdin_path];
        let results = hybrid_search(&cli.query, &index, &files, &embedder, &search_opts)?;
        (results, files, Some(lines))
    } else {
        let discover_opts = DiscoverOpts {
            use_gitignore: !no_gitignore,
            custom_ignore_files: &cli.ignore_file,
        };

        let mut files: Vec<PathBuf> = Vec::new();
        for p in &cli.paths {
            if p.is_dir() {
                if cli.no_recursive {
                    if let Ok(rd) = fs::read_dir(p) {
                        for entry in rd.flatten() {
                            let path = entry.path();
                            if path.is_file() {
                                files.push(path);
                            }
                        }
                    }
                } else {
                    files.extend(discover_files(p, &discover_opts));
                }
            } else if p.is_file() {
                files.push(p.clone());
            } else {
                eprintln!("clawgrep: {}: No such file or directory", p.display());
            }
        }
        if files.is_empty() {
            eprintln!("clawgrep: no files to search");
            return Ok(false);
        }
        if verbose {
            eprintln!("clawgrep: discovered {} files", files.len());
        }

        let index_opts = IndexOpts {
            reindex: cli.reindex,
            no_cache: cli.no_cache,
            custom_cache: cache_dir.as_deref(),
            path_boost,
            verbose,
        };
        if verbose {
            eprintln!("clawgrep: building index...");
        }
        let index = build_index(&files, &embedder, &index_opts)?;
        if verbose {
            eprintln!("clawgrep: index built ({} entries)", index.entries.len());
        }

        if verbose {
            eprintln!("clawgrep: searching...");
        }
        let results = hybrid_search(&cli.query, &index, &files, &embedder, &search_opts)?;
        (results, files, None)
    };

    if results.is_empty() {
        return Ok(false);
    }

    if cli.quiet {
        return Ok(true);
    }

    let show_filename = files.len() > 1;

    // ── Output modes ────────────────────────────────────────────────────
    if cli.list_files {
        let mut seen = std::collections::HashSet::new();
        for r in &results {
            let f = r.file.display().to_string();
            if seen.insert(f.clone()) {
                println!("{}", f);
            }
        }
        return Ok(true);
    }

    if cli.count {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for r in &results {
            *counts.entry(r.file.display().to_string()).or_default() += 1;
        }
        for (file, n) in &counts {
            if show_filename {
                println!("{file}:{n}");
            } else {
                println!("{n}");
            }
        }
        return Ok(true);
    }

    // ── Normal / context output ──────────────────────────────────────────
    let before = if cli.context > 0 {
        cli.context
    } else {
        cli.before
    };
    let after = if cli.context > 0 {
        cli.context
    } else {
        cli.after
    };
    let show_ctx = before > 0 || after > 0;

    let mut file_lines_cache: std::collections::HashMap<PathBuf, Vec<String>> =
        std::collections::HashMap::new();

    if let Some(ref slines) = stdin_lines {
        file_lines_cache.insert(PathBuf::from("(standard input)"), slines.clone());
    }

    let fmt_prefix = |file: &std::path::Path, ln: usize, sep: &str| -> String {
        if show_filename {
            format!(
                "{}{}{}{}",
                magenta(&file.display().to_string(), color),
                cyan(sep, color),
                green(&ln.to_string(), color),
                cyan(sep, color),
            )
        } else {
            format!("{}{}", green(&ln.to_string(), color), cyan(sep, color),)
        }
    };

    for (idx, r) in results.iter().enumerate() {
        let lines = file_lines_cache
            .entry(r.file.clone())
            .or_insert_with(|| read_file_lines(&r.file));

        let match_start = r.line_num.saturating_sub(1);
        let match_end = r.end_line.min(lines.len());

        if show_ctx {
            let ctx_start = match_start.saturating_sub(before);
            let ctx_end = (match_end + after).min(lines.len());
            for i in ctx_start..ctx_end {
                let ln = i + 1;
                let text = &lines[i];
                if ln >= r.line_num && ln <= r.end_line {
                    let prefix = fmt_prefix(&r.file, ln, ":");
                    let highlighted = highlight_matches(text, &cli.query, color);
                    if cli.show_score {
                        println!("{}{}\t({:.3})", prefix, highlighted, r.score);
                    } else {
                        println!("{}{}", prefix, highlighted);
                    }
                } else {
                    let prefix = fmt_prefix(&r.file, ln, "-");
                    println!("{}{}", prefix, text);
                }
            }
            if idx + 1 < results.len() {
                println!("{}", cyan("--", color));
            }
        } else if match_start < lines.len() {
            let ln = r.line_num;
            let text = &lines[match_start];
            let prefix = fmt_prefix(&r.file, ln, ":");
            let highlighted = highlight_matches(text, &cli.query, color);
            if cli.show_score {
                println!("{}{}\t({:.3})", prefix, highlighted, r.score);
            } else {
                println!("{}{}", prefix, highlighted);
            }
        }
    }

    Ok(true)
}
