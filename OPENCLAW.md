# Using clawgrep with OpenClaw

Guide for setting up and using clawgrep as a search tool inside OpenClaw.

## Installation

Install clawgrep so it is on PATH before starting OpenClaw:

```bash
cargo install clawgrep
# or
npm install -g clawgrep
# or
pip install clawgrep
```

Verify:

```bash
clawgrep --version
```

If running OpenClaw inside Docker, install clawgrep in the Dockerfile or as part of the container entrypoint so the binary is available at runtime. Example Dockerfile snippet:

```dockerfile
RUN cargo install clawgrep
```

Or use a pre-built binary from [GitHub Releases](https://github.com/user/clawgrep/releases) to avoid compiling from source:

```dockerfile
RUN curl -fsSL https://github.com/user/clawgrep/releases/latest/download/clawgrep-x86_64-unknown-linux-gnu.tar.gz \
    | tar xz -C /usr/local/bin
```


All fields are optional. CLI flags override config file values.

### Docker: storing the cache in the workspace

By default clawgrep caches embeddings in a platform-specific directory (`~/.cache/clawgrep/` on Linux, `~/Library/Caches/clawgrep/` on macOS). In Docker containers, the home directory is often ephemeral — the cache is lost when the container restarts and must be rebuilt from scratch on the next search.

To persist the cache across container restarts, point it at a directory inside the mounted workspace volume:

**Option 1 — config file:**

```toml
cache_dir = "/workspace/.clawgrep-cache"
```

**Option 2 — environment variable:**

```bash
export CLAWGREP_CACHE_DIR=/workspace/.clawgrep-cache
```

Add `.clawgrep-cache/` to `.gitignore` so the cache directory is not committed.

If you mount the workspace to a different path, adjust accordingly — the cache directory just needs to be on a volume that survives container restarts.

### Recommended .clawgrep.toml for Docker

```toml
semantic_weight = 0.7
keyword_weight = 0.3
top_k = 10
path_boost = 1.5
cache_dir = "/workspace/.clawgrep-cache"
```

Place this file at `~/.clawgrep.toml` inside the container image, or set `CLAWGREP_CONFIG` to point at a config file in the workspace.

## Usage in OpenClaw

OpenClaw can invoke clawgrep as a shell command. Always pass `--no-color` when parsing output programmatically.

### Semantic search (finding code by intent)

```bash
clawgrep --no-color "retry logic with exponential backoff" ./src
```

### Finding exact identifiers

Increase keyword weight for function names, error codes, or exact strings:

```bash
clawgrep --no-color --keyword-weight 0.8 --semantic-weight 0.2 "handleUserAuth" ./src
```

### Existence checks

Use `-q` when OpenClaw just needs to know if something exists:

```bash
clawgrep -q "database migration" ./src && echo "found" || echo "not found"
```

### Reducing token usage

Keep `-k` low (default is 5). Higher values return more results but consume more tokens:

```bash
clawgrep --no-color -k 3 "authentication" ./src
```

### Finding files by name

Boost path matches to surface files whose names match the query:

```bash
clawgrep --no-color --path-boost 2.0 "utils" ./src
```

### Output format

Output is grep-compatible, one result per line:

```
file:line:text
```

Exit codes: 0 = match found, 1 = no match, 2 = error.

## Compatibility notes

- clawgrep is always case-insensitive. The `-i` flag is accepted but ignored.
- First run on a workspace downloads the embedding model (~65 MB) and indexes all files. This takes longer than subsequent runs. Plan for this in container startup if latency on first search matters.
- The embedding model (BAAI/bge-small-en-v1.5) runs locally via tract (pure Rust, no Python or GPU required). No API keys or network access needed after the model is downloaded.
- Concurrent clawgrep processes can share the same cache safely (SQLite WAL mode).
- stdin is supported: `cat file.txt | clawgrep --no-color "query"`. Stdin input is not cached.

## Common problems in Docker

**Cache lost on container restart.** Set `cache_dir` to a path on a mounted volume. See [Docker: storing the cache in the workspace](#docker-storing-the-cache-in-the-workspace).

**Model download fails or is slow.** The first run downloads the ONNX model. If the container has restricted network access, download the model during image build or mount it in. The model is cached alongside embeddings in the cache directory.

**Permission errors on cache directory.** Ensure the user running clawgrep has write access to the cache directory. In Docker, this means the mounted volume permissions must allow writes from the container user.

**Binary not found.** Make sure clawgrep is installed and on PATH. In Docker, install in the Dockerfile rather than at runtime so it is available immediately.

**Indexing is slow on first run.** Expected — all files must be embedded. Subsequent runs only re-embed changed files. Keep the cache on a persistent volume to avoid re-indexing.
