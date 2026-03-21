//! CLI entry point for clawgrep.
//!
//! Output format is grep-compatible:
//!
//!     file:line:text
//!
//! Exit codes follow grep conventions:
//! - 0: at least one match found
//! - 1: no matches found
//! - 2: error

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;

use clawgrep::config::load_config;
use clawgrep::embed::Embedder;
use clawgrep::index::{
    build_index, build_index_from_content, discover_files, DiscoverOpts, IndexOpts,
};
use clawgrep::search::{hybrid_search, SearchOpts};

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

/// AI grep — search files by meaning, not pattern.
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
    // Build a regex that matches any query word (case-insensitive).
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

fn run() -> Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();
    let verbose = cli.verbose || std::env::var("CLAWGREP_VERBOSE").map_or(false, |v| v == "1");
    let cfg = load_config();

    // Merge: CLI flags take precedence over config file.
    let raw_args: Vec<String> = std::env::args().collect();
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
        cli.cache_dir.clone().or(cfg.cache_dir.clone())
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

    let color = !cli.no_color && io::stdout().is_terminal();
    let reading_stdin = cli.paths.is_empty();

    // Load model.
    if verbose {
        eprintln!("clawgrep: loading model...");
    }
    let embedder = Embedder::new()?;
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
        // Read all of stdin into memory; build index without caching.
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

        // Resolve files: expand directories into their text files.
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

    // Determine whether to show filenames. grep behaviour: omit filename
    // when searching a single file (or stdin), show when multiple files.
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

    // Cache of file lines for context display.
    let mut file_lines_cache: std::collections::HashMap<PathBuf, Vec<String>> =
        std::collections::HashMap::new();

    // If we read from stdin, pre-populate the line cache.
    if let Some(ref slines) = stdin_lines {
        file_lines_cache.insert(PathBuf::from("(standard input)"), slines.clone());
    }

    // Format helpers that respect show_filename.
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

        // The result points at a chunk (line_num..end_line). We output
        // individual matching lines, not the whole chunk. For each line
        // in the chunk, print it as a match line.
        let match_start = r.line_num.saturating_sub(1);
        let match_end = r.end_line.min(lines.len());

        if show_ctx {
            let ctx_start = match_start.saturating_sub(before);
            let ctx_end = (match_end + after).min(lines.len());
            for i in ctx_start..ctx_end {
                let ln = i + 1;
                let text = &lines[i];
                if ln >= r.line_num && ln <= r.end_line {
                    // Match line: use ":" separator.
                    let prefix = fmt_prefix(&r.file, ln, ":");
                    let highlighted = highlight_matches(text, &cli.query, color);
                    if cli.show_score {
                        println!("{}{}\t({:.3})", prefix, highlighted, r.score);
                    } else {
                        println!("{}{}", prefix, highlighted);
                    }
                } else {
                    // Context line: use "-" separator (grep convention).
                    let prefix = fmt_prefix(&r.file, ln, "-");
                    println!("{}{}", prefix, text);
                }
            }
            if idx + 1 < results.len() {
                println!("{}", cyan("--", color));
            }
        } else {
            // No context: print the first line of the matching chunk.
            if match_start < lines.len() {
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
    }

    Ok(true)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("clawgrep: {e:#}");
            ExitCode::from(2)
        }
    }
}
