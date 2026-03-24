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

use std::process::ExitCode;

fn main() -> ExitCode {
    ExitCode::from(clawgrep::cli::run(std::env::args()) as u8)
}
