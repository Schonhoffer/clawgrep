---
name: clawgrep
description: >
  Search files by meaning, not just pattern. Combines semantic embedding search
  with keyword matching for high-quality code and document retrieval. Use when
  finding relevant code or documentation where the exact wording is unknown,
  locating functions or logic by intent, searching large codebases where grep
  misses results, or finding exact identifiers alongside semantic matches.
  Requires clawgrep CLI installed. Output is grep-compatible.
compatibility: >
  Requires the clawgrep binary on PATH. Works on Linux, macOS, and Windows.
  No API keys or network access needed. Install via cargo, npm, or pip.
license: MIT
metadata:
  version: "0.1"
---

# clawgrep

Semantic + keyword file search. Output is grep-compatible. Runs locally, no
API keys.

## Check availability

```bash
clawgrep --version
```

If not found, install:

```bash
cargo install clawgrep        # Rust (recommended)
npm install -g clawgrep        # Node.js
pip install clawgrep           # Python
```

## Basic usage

```bash
clawgrep --no-color "query" <path>
```

Always pass `--no-color` when parsing output programmatically.

### Search a workspace

```bash
clawgrep --no-color "database connection timeout" ./src
```

### Output format

Grep-compatible, one result per line:

```
file:line:text
```

Results are ranked by relevance (best first). Context lines use `-` separator
like grep (`file-line-text`).

### Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Match found |
| `1`  | No match |
| `2`  | Error |

Same as grep. Use `-q` for existence checks without output.

## Choosing search mode

Default weights: 70% semantic, 30% keyword.

**Concept search** (don't know exact wording):

```bash
clawgrep --no-color "retry logic with exponential backoff" ./src
```

**Exact identifier search** (function names, error codes, serial numbers):

```bash
clawgrep --no-color --keyword-weight 0.8 --semantic-weight 0.2 "handleUserAuth" ./src
```

## Key flags

| Flag | Purpose |
|------|---------|
| `-k N` | Number of results (default: 5) |
| `-C N` | Context lines before and after |
| `-l` | Print only matching filenames |
| `-q` | Quiet; just set exit code |
| `--show-score` | Append relevance score |
| `--path-boost N` | Boost filename matches (>1.0 = higher) |
| `--min-score N` | Filter low-relevance results (0.0–1.0) |

See [CLI reference](references/cli-reference.md) for all flags.

## Best practices

1. Use `--no-color` always when parsing output.
2. Keep `-k` small (3–5) to reduce output. Increase only when needed.
3. Check exit codes instead of parsing stdout when possible.
4. Let the cache persist — don't use `--no-cache` unless searching throwaway
   content. First run indexes; subsequent runs are fast.
5. Search the narrowest relevant directory, not the whole filesystem.
6. Pre-configure `~/.clawgrep.toml` so commands stay short. See
   [configuration reference](references/cli-reference.md#configuration-file).

## More information

- [CLI reference](references/cli-reference.md) — all flags, config file format, grep compatibility details
- [Examples](references/examples.md) — input/output examples for common scenarios
