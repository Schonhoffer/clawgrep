//! Configuration file support.
//!
//! Reads settings from a TOML file at `~/.clawgrep.toml` (or the path in
//! `CLAWGREP_CONFIG`).  CLI flags take precedence over file values.

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

/// Configuration values that can be set in the config file.
/// All fields are optional — absent fields use CLI defaults.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct FileConfig {
    pub semantic_weight: Option<f32>,
    pub keyword_weight: Option<f32>,
    pub top_k: Option<usize>,
    pub min_score: Option<f32>,
    pub cache_dir: Option<PathBuf>,
    pub no_gitignore: Option<bool>,
    pub path_boost: Option<f32>,
}

/// Load the config file.  Returns `FileConfig::default()` if no file exists.
pub fn load_config() -> FileConfig {
    let path = config_path();
    let Some(path) = path else {
        return FileConfig::default();
    };
    if !path.exists() {
        return FileConfig::default();
    }
    let Ok(content) = fs::read_to_string(&path) else {
        eprintln!(
            "clawgrep: warning: could not read config {}",
            path.display()
        );
        return FileConfig::default();
    };
    match toml::from_str(&content) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("clawgrep: warning: bad config {}: {e}", path.display());
            FileConfig::default()
        }
    }
}

/// Resolve the config file path.
/// Priority: `CLAWGREP_CONFIG` env var, then `~/.clawgrep.toml`.
fn config_path() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("CLAWGREP_CONFIG") {
        return Some(PathBuf::from(v));
    }
    dirs::home_dir().map(|h| h.join(".clawgrep.toml"))
}
