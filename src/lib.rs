//! clawgrep — Grep-like CLI with hybrid semantic and keyword search.
//!
//! This is the library crate that exposes the core building blocks:
//!
//! - [`embed`] — load & run the embedding model
//! - [`cache`] — SQLite-backed embeddings cache with concurrent-safe writes
//! - [`index`] — file discovery and incremental re-indexing with checkpointing
//! - [`keyword`] — substring/regex/stemming keyword search on live files
//! - [`search`] — hybrid (semantic + keyword) search and ranking
//! - [`cli`] — full CLI entry point, reusable from bindings

pub mod cache;
pub mod cli;
pub mod config;
pub mod embed;
pub mod index;
pub mod keyword;
pub mod search;
