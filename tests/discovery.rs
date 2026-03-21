//! File discovery tests.

mod common;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use tempfile::TempDir;

use clawgrep::cache::CACHE_DIR_NAME;
use clawgrep::index::{discover_files, DiscoverOpts};

use common::{make_project, test_discover_opts};

#[test]
fn discover_finds_text_files() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    assert!(
        files.len() >= 7,
        "expected >=7 text files, got {}: {:?}",
        files.len(),
        files
    );
}

#[test]
fn discover_skips_git_directory() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let git_internal: Vec<_> = files
        .iter()
        .filter(|f| {
            let s = f.to_string_lossy();
            s.contains(".git\\") || s.contains(".git/")
        })
        .collect();
    assert!(
        git_internal.is_empty(),
        "should skip .git dir contents: {:?}",
        git_internal
    );
}

#[test]
fn discover_skips_node_modules_with_gitignore() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let opts = DiscoverOpts {
        use_gitignore: true,
        custom_ignore_files: &[],
    };
    let files = discover_files(dir.path(), &opts);
    let nm: Vec<_> = files
        .iter()
        .filter(|f| f.to_string_lossy().contains("node_modules"))
        .collect();
    assert!(
        nm.is_empty(),
        "should skip node_modules via .gitignore: {:?}",
        nm
    );
}

#[test]
fn discover_skips_clawgrep_cache() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let cache = dir.path().join(CACHE_DIR_NAME);
    fs::create_dir_all(&cache).unwrap();
    fs::write(cache.join("cache.db"), "fake data").unwrap();
    let files = discover_files(dir.path(), &test_discover_opts());
    let cached: Vec<_> = files
        .iter()
        .filter(|f| f.to_string_lossy().contains(CACHE_DIR_NAME))
        .collect();
    assert!(cached.is_empty(), "should skip .clawgrep: {:?}", cached);
}

#[test]
fn discover_skips_binary_files() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let bins: Vec<_> = files
        .iter()
        .filter(|f| f.extension().map(|e| e == "bin").unwrap_or(false))
        .collect();
    assert!(bins.is_empty(), "should skip binary files: {:?}", bins);
}

#[test]
fn discover_includes_nested_text() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let names: HashSet<String> = files
        .iter()
        .filter_map(|f| f.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect();
    for expected in &[
        "main.rs",
        "lib.rs",
        "guide.md",
        "faq.md",
        "deploy.log",
        "errors.log",
    ] {
        assert!(names.contains(*expected), "missing {expected}");
    }
}

#[test]
fn discover_handles_empty_directory() {
    let dir = TempDir::new().unwrap();
    let files = discover_files(dir.path(), &test_discover_opts());
    assert!(files.is_empty());
}

#[test]
fn discover_custom_ignore_file() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("keep.txt"), "keep\n").unwrap();
    fs::write(dir.path().join("skip.log"), "skip\n").unwrap();
    fs::write(dir.path().join(".clawgrepignore"), "*.log\n").unwrap();

    let ignore_files = vec![".clawgrepignore".to_string()];
    let opts = DiscoverOpts {
        use_gitignore: false,
        custom_ignore_files: &ignore_files,
    };
    let files = discover_files(dir.path(), &opts);
    let names: HashSet<String> = files
        .iter()
        .filter_map(|f| f.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect();
    assert!(names.contains("keep.txt"));
    assert!(
        !names.contains("skip.log"),
        "should skip .log via .clawgrepignore"
    );
}

#[test]
fn discover_across_subdirectories() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let mut by_dir: HashMap<String, Vec<String>> = HashMap::new();
    for f in &files {
        let parent = f
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        by_dir
            .entry(parent)
            .or_default()
            .push(f.file_name().unwrap().to_string_lossy().to_string());
    }
    assert!(by_dir.contains_key("src"));
    assert!(by_dir.contains_key("docs"));
    assert!(by_dir.contains_key("logs"));
}

#[test]
fn many_files_in_flat_directory() {
    let dir = TempDir::new().unwrap();
    for i in 0..50 {
        fs::write(
            dir.path().join(format!("file_{i:03}.txt")),
            format!("content of file {i}\n"),
        )
        .unwrap();
    }
    let files = discover_files(dir.path(), &test_discover_opts());
    assert_eq!(files.len(), 50);
}

#[test]
fn deeply_nested_directory() {
    let dir = TempDir::new().unwrap();
    let mut path = dir.path().to_path_buf();
    for i in 0..10 {
        path = path.join(format!("level{i}"));
    }
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("deep.txt"), "found at the bottom\n").unwrap();
    let files = discover_files(dir.path(), &test_discover_opts());
    assert_eq!(files.len(), 1);
    assert!(files[0].to_string_lossy().contains("deep.txt"));
}

#[test]
fn file_stamp_returns_valid_values() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("stamped.txt");
    fs::write(&file, "some content here").unwrap();
    let (mtime_ms, size) = clawgrep::cache::file_stamp(&file).unwrap();
    assert!(mtime_ms > 0);
    assert_eq!(size, 17);
}

#[test]
fn file_stamp_changes_on_write() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("mutable.txt");
    fs::write(&file, "v1").unwrap();
    let (_, size1) = clawgrep::cache::file_stamp(&file).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    fs::write(&file, "version two is longer").unwrap();
    let (_, size2) = clawgrep::cache::file_stamp(&file).unwrap();
    assert_ne!(size1, size2);
}

#[test]
fn file_stamp_errors_on_missing_file() {
    assert!(clawgrep::cache::file_stamp(Path::new("/nonexistent/path/foo.txt")).is_err());
}

/// Verify that a symlink loop does not cause discover_files to hang.
/// The `ignore` crate detects cycles when `follow_links(true)` is set and
/// emits an error entry that discover_files logs and skips.
#[test]
fn discover_does_not_loop_on_symlink_cycle() {
    let dir = TempDir::new().unwrap();
    let child = dir.path().join("child");
    fs::create_dir(&child).unwrap();
    fs::write(child.join("real.txt"), "real content\n").unwrap();

    // Create a symlink inside child/ that points back to the root,
    // forming an infinite cycle: root -> child -> link -> root -> ...
    let link = child.join("loop_link");
    let created = symlink_dir(dir.path(), &link);
    if !created {
        // Symlink creation can fail without elevated privileges on some
        // platforms; skip the test rather than false-pass.
        eprintln!("skipping: could not create directory symlink (needs privileges)");
        return;
    }

    // discover_files must terminate and return the real file.
    let files = discover_files(dir.path(), &test_discover_opts());
    let names: Vec<String> = files
        .iter()
        .filter_map(|f| f.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect();
    assert!(
        names.contains(&"real.txt".to_string()),
        "should find real.txt despite symlink loop: {:?}",
        names
    );
}

/// Cross-platform helper to create a directory symlink.
/// Returns false if the OS refused (e.g. missing privileges on Windows).
fn symlink_dir(target: &Path, link: &Path) -> bool {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).is_ok()
    }
    #[cfg(windows)]
    {
        // Try a junction first (no admin needed), fall back to symlink_dir.
        junction::create(target, link).is_ok()
            || std::os::windows::fs::symlink_dir(target, link).is_ok()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, link);
        false
    }
}
