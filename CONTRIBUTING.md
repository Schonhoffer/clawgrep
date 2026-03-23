# Contributing to clawgrep

## Getting started

```bash
git clone https://github.com/Schonhoffer/clawgrep.git
cd clawgrep
cargo build
cargo test    # ~65 MB model download on first run, cached afterwards
```

## Development workflow

```bash
cargo fmt         # format
cargo test --all  # run all tests
cargo build --release
```

## Project structure

- `src/` — core library and CLI binary
- `bindings/node/` — Node.js (napi-rs) bindings
- `bindings/python/` — Python (PyO3/maturin) bindings
- `tests/` — integration / E2E tests
- `dist/clawgrep/` — agent skill distribution

## Testing

Prefer E2E tests over unit tests. Unit tests are for self-contained logic that is hard to exercise end-to-end. Avoid mocking — use the real thing.

Tests are organized by concern:
- `tests/cli.rs` — CLI integration tests
- `tests/search.rs` — hybrid search with the embedding model
- `tests/keyword.rs` — keyword search
- `tests/cache.rs` — SQLite cache operations
- `tests/chunking.rs` — text chunking
- `tests/discovery.rs` — file discovery and ignore rules

Shared helpers live in `tests/common/mod.rs`.

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Make your changes. Keep diffs focused.
3. Add tests for new behavior.
4. Run `cargo fmt` and `cargo test --all` before submitting.
5. Open a pull request against `main`.

## Code style

- Keep files small.
- Write straightforward code. Avoid clever abstractions.
- Follow existing patterns in the codebase.

## Reporting bugs

Open a GitHub issue with:
- What you expected vs. what happened
- Steps to reproduce
- OS and clawgrep version (`clawgrep --version`)

## License

By contributing, you agree that your contributions will be dual-licensed under the MIT and Apache 2.0 licenses.
