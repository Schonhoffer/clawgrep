# clawgrep

Grep-like CLI with hybrid semantic and keyword search.

Combines embedding-based semantic search with substring/regex keyword matching. Output is grep-compatible. Runs fully local.

Semantic search is awesome for searching by the "meaning" which makes it very flexible, but can struggle with long non-semantic sequences like serial numbers. This utility uses both embedding space cosine similarity and keyword search, and returns a combined search result.

The goal of making the input arguments and output format similar to grep is to make it easy to approach by pre-trained LLMs that generate the commands and interpreting results.

## Installation

Though `clawgrep` is primarily used as a CLI utility, it is distributed through language-specific package managers.

```bash
cargo install clawgrep        # Rust/Cargo
npm install -g clawgrep        # Node.js
pip install clawgrep           # Python
```

Or download a pre-built binary from [GitHub Releases](https://github.com/Schonhoffer/clawgrep/releases) (Linux, macOS, Windows — x86_64 and aarch64).

## Examples

```bash
# Search a directory
clawgrep "database connection timeout" ./src

# Search specific files
clawgrep "error handling" main.rs lib.rs

# Read from stdin
cat logs/*.txt | clawgrep "connection refused"

# Context lines (like grep -C)
clawgrep -C 3 "authentication logic" ./src

# Top 10 results instead of default 5
clawgrep -k 10 "memory leak" ./logs

# Only filenames
clawgrep -l "TODO" ./src

# Quiet mode (exit code 0 or 1)
clawgrep -q "security vulnerability" ./audit

# Keyword-heavy search (finds barcodes, serial numbers, exact strings)
clawgrep --keyword-weight 0.8 --semantic-weight 0.2 "UPC-A 012345678901" ./src

# Force re-index
clawgrep --reindex "startup sequence" ./src

# Skip cache entirely
clawgrep --no-cache "one-off query" ./tmp

# Don't follow .gitignore
clawgrep --no-gitignore "build artifact" .

# Use a custom ignore file
clawgrep --ignore-file .clawgrepignore "todo" .

# Custom cache directory
clawgrep --cache-dir /tmp/clawgrep-cache "query" ./src

# Non-recursive (top-level files only)
clawgrep --no-recursive "config" ./src

# Boost path/filename matches
clawgrep --path-boost 2.0 "utils" ./src

# Show relevance scores
clawgrep --show-score "database" ./src
```

## Output format

Output follows the same `file:line:text` format as grep, one match per line:

    docs/setup.md:12:Configure the database connection string in your environment
    docs/troubleshooting.md:45:If the connection is refused, check firewall rules
    docs/architecture.md:8:The connection pool manages up to 20 concurrent sessions

Results are sorted by relevance score (highest first), not by file position. This differs from grep, which prints matches in file order.

When searching a single file or stdin, the filename prefix is omitted, matching grep behavior. Context lines (from `-A`, `-B`, `-C`) use `-` as their separator instead of `:`, also matching grep: `file-line-text`.

With `--show-score`, a tab-separated relevance score is appended to each line: `file:line:text\t(0.847)`. This is the only output variation that departs from grep format.

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | At least one match found |
| `1`  | No matches found |
| `2`  | Error |

These match grep conventions.

## How it works

1. **Discover** files recursively, respecting `.gitignore` and `.clawgrepignore`.
2. **Index** files by splitting them into overlapping segments sized by estimated token count (~100 tokens, matching the embedding model's budget). Prefers splitting at natural boundaries (blank lines, section headers) when they fall near the target size; a hard cap of 20 lines per chunk keeps results granular for token-sparse content like code. Embeddings are computed with a local ONNX model (BAAI/bge-small-en-v1.5, 384d) using tract (pure-Rust inference). Model weights (~65 MB) are downloaded on first use into the `models/` subdirectory of the cache directory.
3. **Cache** embeddings and model weights in a platform-specific cache directory (or `--cache-dir`). The SQLite database and downloaded model files share the same directory. WAL mode allows concurrent readers and serialised writers; optimistic concurrency ensures newer embeddings always win.
4. **Checkpoint** every 25 files during indexing, so interrupted runs resume from roughly where they stopped.
5. **Keyword search** reads files from disk and does substring matching, regex matching, and basic stemming. This runs independently of embeddings and finds exact strings like barcodes, serial numbers, and error codes.
6. **Rank** by combining scores: `score = semantic_weight * cosine(query, segment) + keyword_weight * keyword_match(query, segment)`. Results are sorted by combined score and truncated to top-k.

Subsequent searches reuse cached embeddings. Only changed files are re-embedded. Multiple clawgrep processes can share the same cache without corruption.

### Cache directory

The default cache location follows OS conventions:

| OS | Default path |
|----|-------------|
| Linux | `~/.cache/clawgrep/` |
| macOS | `~/Library/Caches/clawgrep/` |
| Windows | `C:\Users\<user>\AppData\Local\clawgrep\` |

Override with `--cache-dir`, `CLAWGREP_CACHE_DIR`, or the `cache_dir` field in the config file. This is useful in Docker containers or non-standard environments where the OS default may not be writable:

```bash
# Docker / CI
export CLAWGREP_CACHE_DIR=/workspace/.clawgrep-cache
clawgrep "query" ./src

# Or in ~/.clawgrep.toml
# cache_dir = "/workspace/.clawgrep-cache"
```

## Configuration

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `-k`, `--top-k` | 5 | Number of results |
| `-B`, `--before-context` | 0 | Lines before match |
| `-A`, `--after-context` | 0 | Lines after match |
| `-C`, `--context` | 0 | Lines before and after |
| `--min-score` | none | Minimum score threshold (0.0-1.0) |
| `--semantic-weight` | 0.7 | Embedding similarity weight (0.0-1.0) |
| `--keyword-weight` | 0.3 | Keyword (substring/regex) weight (0.0-1.0) |
| `--reindex` | false | Force re-embedding |
| `--no-cache` | false | Don't read or write cache |
| `--cache-dir` | platform default | Custom cache directory |
| `--no-gitignore` | false | Don't respect .gitignore |
| `--ignore-file` | none | Additional ignore file (repeatable) |
| `--no-recursive` | false | Don't recurse into subdirectories |
| `--path-boost` | 1.0 | Boost factor for file/folder path matches (0.0 disables) |
| `-l` | - | Print only matching filenames |
| `-c` | - | Print match count per file |
| `-q` | - | Quiet; exit 0 if any match |
| `--no-color` | false | Disable colored output |
| `--show-score` | false | Append relevance score to each output line |
| `--verbose` | false | Print diagnostic info to stderr |
| `-i` | - | Accepted but ignored (always case-insensitive) |

### Ignore files

By default, `.gitignore` rules are respected. To add project-specific ignore rules without modifying `.gitignore`, create a `.clawgrepignore` file (same syntax) and pass `--ignore-file .clawgrepignore`.

To disable all gitignore filtering: `--no-gitignore`.

### Logging

Set `RUST_LOG` to control debug output:

```bash
RUST_LOG=clawgrep=debug clawgrep "query" .
RUST_LOG=clawgrep=info clawgrep "query" .
```

Uses the standard Rust `log` + `env_logger` stack. Logs go to stderr.

### Environment variables

| Variable | Description |
|----------|-------------|
| `CLAWGREP_CONFIG` | Path to config file (default: `~/.clawgrep.toml`) |
| `CLAWGREP_CACHE_DIR` | Cache directory (overrides platform default) |
| `CLAWGREP_VERBOSE` | Set to `1` to enable verbose output |
| `NO_COLOR` | Disable colored output (any value) |

### Configuration file

Settings can be placed in `~/.clawgrep.toml` to avoid passing flags every time. Set the `CLAWGREP_CONFIG` environment variable to use a different path.

Example `~/.clawgrep.toml`:

```toml
semantic_weight = 0.7
keyword_weight = 0.3
top_k = 10
min_score = 0.3
path_boost = 1.5
no_gitignore = false
cache_dir = "/tmp/clawgrep-cache"
```

All fields are optional. Precedence: CLI flags > config file > environment variables.

## OpenClaw

clawgrep works well as a tool for OpenClaw or other AI agents that need to search a workspace. See [OPENCLAW.md](OPENCLAW.md) for detailed setup instructions, Docker configuration, and troubleshooting. Recommended setup:

1. Pre-configure `~/.clawgrep.toml` with your preferred weights and cache directory. The AI only needs to form a simple command.
2. Use `--no-color` when parsing output programmatically.
3. Output format is `file:line:text`, one result per line. Same as grep. Use `--show-score` to append relevance scores.
4. Exit codes follow grep conventions (0 = match, 1 = no match, 2 = error), so the AI can check success without parsing stdout.
5. For repeated searches in a session, let the cache persist (don't use `--no-cache`). Only changed files are re-embedded.
6. Use `-k` to control how many results the AI receives. Lower values reduce token usage.
7. Use `--path-boost` to help the AI find files by name. Setting it above 1.0 ranks filename matches higher than content matches.
8. Use `-q` for existence checks when the AI just needs to know whether something exists in the codebase.
9. For finding exact identifiers, barcodes, or serial numbers, increase `--keyword-weight` (e.g. `--keyword-weight 0.8 --semantic-weight 0.2`).
10. Piping content via stdin is supported. Embeddings are generated on-the-fly (no caching).

## Library usage

The npm and pip packages also work as libraries.

### Node.js

```js
const { search } = require('clawgrep');
const results = search('database connection', ['./src'], {
  topK: 10,
  semanticWeight: 0.7,
  keywordWeight: 0.3,
});
// results: [{ file, lineNum, endLine, text, score, semanticScore, keywordScore }]
```

### Python

```python
import clawgrep

results = clawgrep.search(
    "database connection",
    ["./src"],
    top_k=10,
    semantic_weight=0.7,
    keyword_weight=0.3,
)
# results: list[SearchResult(file, line_num, end_line, text, score, semantic_score, keyword_score)]
```

## Contributing

```bash
# Run all tests (~65 MB model download on first run, cached afterwards)
cargo test

# Format
cargo fmt

# Build release
cargo build --release
```

Tests are organized by concern:
- `tests/discovery.rs` — file discovery and ignore rules
- `tests/chunking.rs` — text segmentation
- `tests/cache.rs` — SQLite cache operations
- `tests/keyword.rs` — keyword search (substring, regex, stemming)
- `tests/search.rs` — hybrid search with the embedding model
- `tests/cli.rs` — CLI binary integration tests

Shared helpers live in `tests/common/mod.rs`. Unit tests are co-located in each module. Prefer E2E tests; add unit tests only for self-contained logic.

## License

Dual-licensed under MIT or Apache 2.0, at your option. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
