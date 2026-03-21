# clawgrep â€” Agent Instructions

Overall goal of this project: **A CLI utility for LLMs that should be like grep for compatibility, but incorporate semantic search.**

Hero scenario: OpenClaw using the CLI command to semantic search on it's workspace folder.

Want to optimize the experience for the hero scenario, but also want to not have any specific openclaw features or semantics, it should be equally usable by any AI coding harness like Codex, Claude Code, or Github Copilot.

## Design principles

- Designed for AI to call as a shell command when searching a workspace. Like for OpenClaw to search its own markdown files.
- Grep-compatible where possible (flags, output, exit codes). This is to make it easy for an LLM to form the commands and parse the outputs.
- When used with normal grep inputs, it should return normal grep-compatible outputs. It can be possible to return grep-incompatible results when the user has supplied clawgrep-specific CLI arguments.
- Results returned should be in order of relevance, not the order that the search was run, that way an LLM can take top N and prefer the first results.
- Support a combination of semantic / embeddings search and keyword traditional search.
  - The semantic ranking is to search the files based on semantic embeddings.
  - The keyword search is to find things like barcodes and serial numbers that the semantic search is bad at.
- It should have a feature to also include the names and folder paths of the files as part of the index, both the semantic index and TF/IDF. Matches on those should be able to be ranked higher than matches on file contents, or the same if desired.
- Zero configuration â€” works out of the box with no API keys or setup
- Support reading most of the configurations, like which embedding model to use, or ranking configuration from a dotfile in the home folder.
  - That path to that config file can also be set with an optional environment variable.
  - The goal of this design is so that the user configuring the AI can decide those parameters ahead of time, and still have the AI generate a "simple" command for clawgrep.
  - Optional CLI commands should still be able to override the config from the file, taking precedence.
- Single binary with no native runtime dependencies. Uses tract (pure Rust) for ONNX model inference.
- Cache aggressively, invalidate precisely
- High performance, fully local.
- Use a cache of the indexed embeddings to increase performance during repeated searches. The cache should be stored at `~/.cache/clawgrep/` but that path should be customizable using toml config or environment variable or CLI arg.
- Library crate (`clawgrep`) exposes all building blocks for reuse
- Support Linux, Windows, macOS on x86 and ARM
- Ability to be installed globally with a Cargo command so that it would appear on the user's path.

## Packaging

Distribute clawgrep four ways from a single codebase:

1. Portable binary archives (tar.gz/zip) for each OS/arch, attached to GitHub Releases.
2. Cargo crate published to crates.io (`cargo install clawgrep`).
3. npm package (`npm install -g clawgrep`) that provides both a CLI binary and a JS library API returning native objects.
4. Python wheels (`pip install clawgrep`) that provide both a CLI entry point and a Python library API returning native objects.

The Node and Python bindings are thin wrapper crates that depend on the core library crate. They convert Rust types to JS/Python objects and expose a `search()` function with the same options as the CLI.

All artifacts are built by a single GitHub Actions workflow triggered on version tags. Binary, Node, and Python builds run in parallel across the target matrix. Registry publishes (crates.io, npm, PyPI) run after their respective builds. The GitHub Release is created last, after all builds complete.

## Cache philosophy

The cache is "just a cache" — disposable, never critical. The app must work correctly (just slower) if the cache is deleted, corrupt, or unavailable.

- Schema versioning uses a single `SCHEMA_VERSION` integer in the DB. No migrations. If the app's version is newer than the DB's version, nuke the entire DB contents and let it rebuild.
- Any cache error (corruption, I/O failure, deserialization, unexpected schema) should be caught. The response is: delete the DB file and retry once. If it still fails, run without caching.
- Never let a cache problem propagate as an application error. Cache reads return `None` on failure; cache writes are best-effort.
- Use safe wrappers (`open_db_resilient`, `get_entry_safe`, `upsert_entry_safe`) in application code. The raw `Result`-returning functions exist for the library crate and tests.

## Instructions for AI
- Don't over-use bullets or emoji.
- Avoid emphatic or emotional tone.
- Be concise and terse when authoring markdown files.
- Don't write fancy code that is hard to read. Code should read like a children's book.
- Prefer E2Es over unit tests, author unit tests only when E2E is impossible to cover the function or when the unit itself is large and self-contained.
- Cover every feature with E2E tests. Unit tests only as needed.
- Always confirm that the project can build, all tests pass, and there are no formatting errors before completing any task. Do not stop until the tests pass.
- Avoid mocking in tests if possible, use the real thing.
- Avoid large individual files. Keep files small so they can all easily fit in context windows.
- It's better that the tests be slow and conclusive than fast and inconclusive.
- Don't remove or skip tests as a method for getting the tests to pass.

## Keep the README.md updated
- Keep the README.md up-to-date. When changing any user-facing part of the code like allowed arguments, or configuration, update the README.md accordingly.
- Write the README.md with a clinical, dispassionate tone. Be concise and terse.
- The README.md should have these sections:
  - Goal of the project in a few sentences.
  - Simple installation instructions
  - A few example commands to get people started. These should mirror grep.
  - Quick discussion of the exit codes and stdout/stderr.
  - Advanced configuration and examples.
  - Specific advice for use and best practices with OpenClaw.
  - Contributing guide

## Distributable skill (`dist/clawgrep/`)

`dist/clawgrep/` contains an Agent Skills spec-compliant skill that teaches AI agents how to install and use clawgrep. It is distributed separately (e.g. via clawdhub, manual install, etc.) and is **not** loaded by agents working on this repo.

Goals: tell an agent how to check if clawgrep is available, install it if missing, search a workspace effectively, and parse the grep-compatible output. Must not mention any specific agent product by name. Must work equally well in any AI coding harness.

Structure follows progressive disclosure per the Agent Skills spec:
1. Frontmatter (`name`, `description`) — loaded at startup for skill matching.
2. SKILL.md body (<500 lines) — availability check, installation, basic usage, key flags, best practices.
3. `references/` — full CLI reference and detailed input/output examples, loaded on demand.

Keep the skill up to date when changing user-facing behavior: CLI flags, output format, exit codes, config file fields, or installation methods. The SKILL.md, `references/cli-reference.md`, and `references/examples.md` should reflect the current CLI.
