//! CLI binary tests.

mod common;

use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};

use tempfile::TempDir;

use common::{make_project, shared_embedder};

#[test]
fn cli_basic_search() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment issue",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.is_empty());
    assert!(stdout.contains(":"));
}

#[test]
fn cli_no_results_exit_code_1() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "hello world\n").unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "xyzzy_nonexistent_12345",
            dir.path().to_str().unwrap(),
            "--no-cache",
            "--min-score",
            "0.99",
            "--no-color",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn cli_top_k_flag() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "error",
            dir.path().to_str().unwrap(),
            "-k",
            "2",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line_count = stdout.lines().count();
    assert!(line_count <= 2);
}

#[test]
fn cli_quiet_flag() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "-q",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn cli_version_flag() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("clawgrep"));
}

#[test]
fn cli_invalid_weight_rejected() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "test\n").unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "query",
            dir.path().to_str().unwrap(),
            "--semantic-weight",
            "5.0",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success(), "weight > 1.0 should be rejected");
}

#[test]
fn cli_search_long_file_does_not_hang() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    let long_content: String = (0..800).map(|i| format!("uniqueword{i} ")).collect();
    fs::write(dir.path().join("huge.txt"), &long_content).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "uniqueword0",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.is_empty());
}

#[test]
fn cli_path_boost_flag() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "main",
            dir.path().to_str().unwrap(),
            "--path-boost",
            "2.0",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "-k",
            "3",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.is_empty());
}

// ── Config file tests ──────────────────────────────────────────────────

#[test]
fn config_file_sets_top_k() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());

    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("clawgrep.toml");
    fs::write(&config_path, "top_k = 1\n").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .env("CLAWGREP_CONFIG", config_path.to_str().unwrap())
        .args([
            "error",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "top_k=1 from config should yield 1 line");
}

#[test]
fn config_file_cli_overrides_config() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());

    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("clawgrep.toml");
    fs::write(&config_path, "top_k = 1\n").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .env("CLAWGREP_CONFIG", config_path.to_str().unwrap())
        .args([
            "error",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "-k",
            "3",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(lines.len() <= 3, "CLI -k 3 should override config top_k=1");
    assert!(lines.len() > 1, "should get more than 1 result with -k 3");
}

#[test]
fn config_file_missing_is_ok() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .env("CLAWGREP_CONFIG", "/nonexistent/path/clawgrep.toml")
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
}

#[test]
fn config_file_sets_weights() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());

    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("clawgrep.toml");
    fs::write(
        &config_path,
        "semantic_weight = 0.0\nkeyword_weight = 1.0\ntop_k = 3\n",
    )
    .unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .env("CLAWGREP_CONFIG", config_path.to_str().unwrap())
        .args([
            "barcode",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "--path-boost",
            "0",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.is_empty(), "should have results");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "config top_k=3 should yield 3 lines");
}

// ── Score output tests ─────────────────────────────────────────────────

#[test]
fn cli_no_score_by_default() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("\t("),
        "score should not appear by default: {stdout}"
    );
}

#[test]
fn cli_show_score_flag() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "--show-score",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\t("),
        "score should appear with --show-score: {stdout}"
    );
}

// ── Flag compatibility tests ───────────────────────────────────────────

#[test]
fn cli_ignore_case_flag_accepted() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "-i",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "-i should be accepted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cli_verbose_no_short_v() {
    // -v should NOT be a valid flag (it conflicted with grep's -v for invert).
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "test\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args(["query", dir.path().to_str().unwrap(), "-v"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "-v should not be a recognised flag"
    );
}

// ── stdin tests ────────────────────────────────────────────────────────

#[test]
fn cli_stdin_search() {
    shared_embedder();
    let mut child = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args(["database connection", "--no-color"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(b"The database connection failed at startup.\nNothing relevant here.\nAnother line about connecting to the database.\n")
            .unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "stdin search should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.is_empty(), "should have results from stdin");
    // Should not contain a filename prefix (single source).
    assert!(
        !stdout.contains("(standard input):"),
        "single source should not show filename: {stdout}"
    );
}

// ── Context output format tests ────────────────────────────────────────

#[test]
fn cli_context_uses_dash_separator() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "-C",
            "1",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Context lines should use "-" separator (grep convention).
    // At least one context line with "-" should appear.
    let has_dash_sep = stdout.lines().any(|l| {
        // A context line has the form: "number-text" (no filename for single dir).
        // or "file-number-text" for multi-file. Either way, look for "-" after digits.
        l.contains('-')
    });
    assert!(
        has_dash_sep,
        "context output should use - separator: {stdout}"
    );
}

// ── Single file: no filename prefix ────────────────────────────────────

#[test]
fn cli_single_file_no_filename() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    let file_path = dir.path().join("test.txt");
    fs::write(
        &file_path,
        "The database connection was established.\nServer started on port 8080.\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "database",
            file_path.to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Single file search should NOT include filename in output (grep convention).
    // The line should start with a line number, not a file path.
    let first_line = stdout.lines().next().unwrap_or("");
    let first_char = first_line.chars().next().unwrap_or(' ');
    assert!(
        first_char.is_ascii_digit(),
        "single file output should start with line number, got: {first_line}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Grep compatibility tests
//
// Each test below is tagged with the gap ID from GREP_COMPAT_ANALYSIS.md.
// ═══════════════════════════════════════════════════════════════════════

// ── G1: -v is not claimed by clawgrep ──────────────────────────────────

#[test]
fn grep_compat_short_v_rejected() {
    // -v must NOT be accepted. grep uses -v for --invert-match.
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "hello\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args(["hello", dir.path().to_str().unwrap(), "-v"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(2),
        "-v should be rejected with exit code 2"
    );
}

#[test]
fn grep_compat_long_verbose_works() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--verbose",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("clawgrep:"),
        "--verbose should produce stderr diagnostics: {stderr}"
    );
}

// ── G2: -m is not claimed by clawgrep ──────────────────────────────────

#[test]
fn grep_compat_short_m_rejected() {
    // -m must NOT be accepted. grep uses -m for --max-count.
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "hello\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args(["hello", dir.path().to_str().unwrap(), "-m", "0.5"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "-m should not be a recognised flag"
    );
}

#[test]
fn grep_compat_long_min_score_works() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--min-score",
            "0.01",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "--min-score should work: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ── G3: output format matches grep (file:line:text) ────────────────────

#[test]
fn grep_compat_output_format_multi_file() {
    // grep format for multi-file: "file:line:text"
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next().unwrap();
    // Must match pattern: file:number:text (no tab+score).
    // Use regex to handle Windows paths that contain ":" (e.g. C:\...).
    let re = regex::Regex::new(r"^(.+):(\d+):(.*)$").unwrap();
    assert!(
        re.is_match(line),
        "line should match file:line:text pattern — got: {line}"
    );
    assert!(
        !line.contains("\t("),
        "should not have score suffix: {line}"
    );
}

#[test]
fn grep_compat_output_format_single_file() {
    // grep format for single file: "line:text" (no filename)
    shared_embedder();
    let dir = TempDir::new().unwrap();
    let f = dir.path().join("single.txt");
    fs::write(
        &f,
        "alpha bravo charlie\ndelta echo foxtrot\ngolf hotel india\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "alpha bravo",
            f.to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next().unwrap();
    // Should be "number:text" with no filename
    let parts: Vec<&str> = line.splitn(2, ':').collect();
    assert_eq!(
        parts.len(),
        2,
        "single-file line should be num:text — got: {line}"
    );
    assert!(
        parts[0].parse::<usize>().is_ok(),
        "first field should be line number: {line}"
    );
}

#[test]
fn grep_compat_show_score_format() {
    // With --show-score: "file:line:text\t(0.xxx)"
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "--show-score",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next().unwrap();
    assert!(line.contains('\t'), "score line should contain tab: {line}");
    // Score format is (N.NNN) where N is a digit.
    let re = regex::Regex::new(r"\t\(\d+\.\d+\)$").unwrap();
    assert!(
        re.is_match(line),
        "score line should end with tab+(N.NNN): {line}"
    );
}

// ── G4: context line separator ─────────────────────────────────────────

#[test]
fn grep_compat_context_colon_for_match_dash_for_context() {
    // grep convention: match lines use ":" separator, context lines use "-".
    // Note: clawgrep matches at chunk granularity (~20 lines). All lines in
    // the chunk use ":" (they are match lines). Only lines OUTSIDE the chunk
    // range but inside the context window use "-".
    shared_embedder();
    let dir = TempDir::new().unwrap();
    let f = dir.path().join("ctx.txt");
    // We need exactly 3 lines so the whole file is one tiny chunk.
    // That way -B1/-A1 gives us clear context lines outside the match.
    // But a 3-line file forms a single chunk (lines 1-3). We want the
    // match chunk at line 2 only. Use a file small enough that the chunk
    // is at most 1 line by placing the unique text very early.
    fs::write(
        &f,
        "padding line one\nxyzzy_ctx_target_99 found here\npadding line three\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "xyzzy_ctx_target_99",
            f.to_str().unwrap(),
            "-B",
            "1",
            "-A",
            "1",
            "--no-color",
            "--no-cache",
            "-k",
            "1",
            "--keyword-weight",
            "1.0",
            "--semantic-weight",
            "0.0",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    // The chunk covers all 3 lines (too small to split). All lines in the
    // chunk are match lines (using ":"). The -B1/-A1 context window may
    // extend beyond the chunk, or not if the chunk already covers the file.
    // In either case, ALL lines that are in the chunk MUST use ":".
    for line in &lines {
        if *line == "--" {
            continue;
        }
        // For a single file, format is "number<sep>text".
        let first_non_digit = line.find(|c: char| !c.is_ascii_digit());
        if let Some(pos) = first_non_digit {
            let sep = line.as_bytes()[pos];
            // sep should be ':' for match lines or '-' for context lines.
            assert!(
                sep == b':' || sep == b'-',
                "separator should be ':' or '-': {line}"
            );
        }
    }

    // Verify match line with unique text uses ":".
    let ml = lines.iter().find(|l| l.contains("xyzzy_ctx_target_99"));
    assert!(ml.is_some(), "should find match line: {stdout}");
    let ml = ml.unwrap();
    let first_non_digit = ml.find(|c: char| !c.is_ascii_digit());
    assert_eq!(
        ml.as_bytes()[first_non_digit.unwrap()],
        b':',
        "match line should use ':' separator: {ml}"
    );
}

#[test]
fn grep_compat_context_group_separator() {
    // grep uses "--" between context groups.
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "-C",
            "1",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "-k",
            "2",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // With k=2 and context, there should be a "--" separator between groups.
    assert!(
        stdout.contains("\n--\n"),
        "should have group separator '--' between context blocks: {stdout}"
    );
}

// ── G5: stdin support ──────────────────────────────────────────────────

#[test]
fn grep_compat_stdin_quiet_mode() {
    // Like: echo "hello" | grep -q "hello"
    shared_embedder();
    let mut child = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args(["database connection", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(b"The database connection was established.\n")
            .unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "stdin + -q should succeed");
    assert!(
        output.stdout.is_empty(),
        "quiet mode should produce no stdout"
    );
}

#[test]
fn grep_compat_stdin_no_match_exit_1() {
    // Like: echo "hello" | grep "zzz" → exit 1
    shared_embedder();
    let mut child = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "xyzzy_impossible_query_9999",
            "--min-score",
            "0.99",
            "--no-color",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"hello world\n").unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdin with no match should exit 1"
    );
}

// ── G7: filename shown only for multiple files ─────────────────────────

#[test]
fn grep_compat_multi_file_shows_filename() {
    // grep shows filename when multiple files are searched.
    shared_embedder();
    let dir = TempDir::new().unwrap();
    let f1 = dir.path().join("one.txt");
    let f2 = dir.path().join("two.txt");
    fs::write(&f1, "The database connection was established.\n").unwrap();
    fs::write(&f2, "Server deployment finished successfully.\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "database deployment",
            f1.to_str().unwrap(),
            f2.to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "-k",
            "2",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Each line should contain the filename somewhere.
    for line in stdout.lines() {
        assert!(
            line.contains("one.txt") || line.contains("two.txt"),
            "multi-file output should include filename: {line}"
        );
        // Should match file:num:text pattern.
        let re = regex::Regex::new(r"^(.+):(\d+):(.*)$").unwrap();
        assert!(
            re.is_match(line),
            "multi-file line should match file:num:text: {line}"
        );
    }
}

#[test]
fn grep_compat_single_file_no_filename() {
    // grep omits filename for single-file search.
    shared_embedder();
    let dir = TempDir::new().unwrap();
    let f = dir.path().join("only.txt");
    fs::write(&f, "The database connection was established.\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "database",
            f.to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next().unwrap();
    // Should NOT contain the filename; should start with digit.
    assert!(
        !line.contains("only.txt"),
        "single file output should omit filename: {line}"
    );
    assert!(
        line.starts_with(|c: char| c.is_ascii_digit()),
        "should start with line number: {line}"
    );
}

// ── G16: -i accepted as no-op ──────────────────────────────────────────

#[test]
fn grep_compat_i_long_and_short_accepted() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    // Short -i
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "-i",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "-i short should work");

    // Long --ignore-case
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--ignore-case",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "--ignore-case should work");
}

// ── Exit codes ─────────────────────────────────────────────────────────

#[test]
fn grep_compat_exit_0_on_match() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "should exit 0 on match");
}

#[test]
fn grep_compat_exit_1_on_no_match() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.txt"), "hello world\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "xyzzy_completely_impossible_match",
            dir.path().to_str().unwrap(),
            "--min-score",
            "0.99",
            "--no-color",
            "--no-cache",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "should exit 1 on no match");
}

#[test]
fn grep_compat_exit_2_on_error() {
    // Giving a nonexistent path should result in exit code 2 (but only if it
    // fully fails — e.g. no files found at all). We use a missing directory.
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args(["query", "/nonexistent/path/that/does/not/exist"])
        .output()
        .unwrap();
    // Should be exit 1 (no files to search → no match) or exit 2.
    let code = output.status.code().unwrap();
    assert!(
        code == 1 || code == 2,
        "nonexistent path should exit 1 or 2, got {code}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("clawgrep:"),
        "error output should go to stderr with 'clawgrep:' prefix: {stderr}"
    );
}

// ── -l output format ───────────────────────────────────────────────────

#[test]
fn grep_compat_list_files_format() {
    // grep -l outputs one filename per line, nothing else.
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "-l",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        // Each line should be a filename — no line numbers, no score.
        // Don't check for absence of ':' because Windows paths have 'C:'.
        assert!(
            !line.contains("\t("),
            "-l output should not have score: {line}"
        );
        // Should not look like a normal result line (file:num:text).
        let re = regex::Regex::new(r":\d+:").unwrap();
        assert!(
            !re.is_match(line),
            "-l output should not have :number: pattern: {line}"
        );
    }
}

// ── -c output format ───────────────────────────────────────────────────

#[test]
fn grep_compat_count_format_multi_file() {
    // grep -c with multiple files: "file:count"
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "-c",
            "--no-color",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.rsplitn(2, ':').collect();
        assert_eq!(
            parts.len(),
            2,
            "-c multi-file format should be file:count — got: {line}"
        );
        assert!(
            parts[0].parse::<usize>().is_ok(),
            "count should be a number: {line}"
        );
    }
}

#[test]
fn grep_compat_count_format_single_file() {
    // grep -c with single file: just "count" (no filename)
    shared_embedder();
    let dir = TempDir::new().unwrap();
    let f = dir.path().join("count.txt");
    fs::write(
        &f,
        "The database connection was established.\nServer started.\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "database",
            f.to_str().unwrap(),
            "-c",
            "--no-color",
            "--no-cache",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim();
    // Single file: should be just a number, no filename.
    assert!(
        line.parse::<usize>().is_ok(),
        "single file -c should output just a number: {line}"
    );
}

// ── -q (quiet) ─────────────────────────────────────────────────────────

#[test]
fn grep_compat_quiet_no_stdout() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "-q",
            "--no-cache",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert!(
        output.stdout.is_empty(),
        "-q should produce zero stdout bytes"
    );
}

// ── -B and -A (separate from -C) ──────────────────────────────────────

#[test]
fn grep_compat_before_after_context() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    let f = dir.path().join("ba.txt");
    let mut content = String::new();
    for i in 1..=40 {
        content.push_str(&format!("line_{i}_padding\n"));
    }
    content.push_str("xyzzy_unique_before_after_target here\n");
    for i in 42..=50 {
        content.push_str(&format!("line_{i}_padding\n"));
    }
    fs::write(&f, &content).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "xyzzy_unique_before_after_target",
            f.to_str().unwrap(),
            "-B",
            "2",
            "-A",
            "2",
            "--no-color",
            "--no-cache",
            "-k",
            "1",
            "--keyword-weight",
            "1.0",
            "--semantic-weight",
            "0.0",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let out_lines: Vec<&str> = stdout.lines().collect();

    // With -B2 -A2, we expect at least 5 lines (2 before + match + 2 after).
    assert!(
        out_lines.len() >= 5,
        "expected >= 5 lines with -B2 -A2, got {}: {stdout}",
        out_lines.len()
    );

    // Verify the match line is present.
    assert!(
        stdout.contains("xyzzy_unique_before_after_target"),
        "match text should appear: {stdout}"
    );
}

// ── Error messages go to stderr with "clawgrep:" prefix ────────────────

#[test]
fn grep_compat_error_messages_to_stderr() {
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args(["query", "/no/such/directory/ever"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("clawgrep:"),
        "stderr should contain 'clawgrep:' prefix: {stderr}"
    );
    // stdout should be empty on error.
    assert!(output.stdout.is_empty(), "stdout should be empty on error");
}

// ── No-color produces no ANSI escape sequences ────────────────────────

#[test]
fn grep_compat_no_color_clean_output() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args([
            "deployment",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-cache",
            "--no-gitignore",
            "-k",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("\x1b["),
        "--no-color should not produce ANSI codes: {stdout}"
    );
}

// ── Stdin with -l shows "(standard input)" ─────────────────────────────

#[test]
fn grep_compat_stdin_list_files() {
    // grep -l with stdin outputs "(standard input)"
    shared_embedder();
    let mut child = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .args(["database connection", "-l", "--no-color"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(b"The database connection was established.\n")
            .unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        "(standard input)",
        "stdin + -l should output '(standard input)'"
    );
}

// ── CLAWGREP_CACHE_DIR environment variable ────────────────────────────

#[test]
fn env_cache_dir_is_used() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());

    let cache_dir = TempDir::new().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .env("CLAWGREP_CACHE_DIR", cache_dir.path().to_str().unwrap())
        .args([
            "error",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    // Cache DB should have been created inside the env var directory.
    assert!(
        cache_dir.path().join("cache.db").exists(),
        "cache.db should be created in CLAWGREP_CACHE_DIR"
    );
}

#[test]
fn env_cache_dir_overridden_by_cli_flag() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());

    let env_cache = TempDir::new().unwrap();
    let cli_cache = TempDir::new().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .env("CLAWGREP_CACHE_DIR", env_cache.path().to_str().unwrap())
        .args([
            "error",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-gitignore",
            "--cache-dir",
            cli_cache.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    // CLI flag should win: cache goes to cli_cache, not env_cache.
    assert!(
        cli_cache.path().join("cache.db").exists(),
        "cache.db should be in --cache-dir, not CLAWGREP_CACHE_DIR"
    );
    assert!(
        !env_cache.path().join("cache.db").exists(),
        "env dir should NOT get cache when --cache-dir is set"
    );
}

#[test]
fn env_cache_dir_overridden_by_config_file() {
    shared_embedder();
    let dir = TempDir::new().unwrap();
    make_project(dir.path());

    let env_cache = TempDir::new().unwrap();
    let toml_cache = TempDir::new().unwrap();
    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("clawgrep.toml");
    fs::write(
        &config_path,
        format!("cache_dir = {:?}\n", toml_cache.path().to_str().unwrap()),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_clawgrep"))
        .env("CLAWGREP_CONFIG", config_path.to_str().unwrap())
        .env("CLAWGREP_CACHE_DIR", env_cache.path().to_str().unwrap())
        .args([
            "error",
            dir.path().to_str().unwrap(),
            "--no-color",
            "--no-gitignore",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    // Config file should win over env var.
    assert!(
        toml_cache.path().join("cache.db").exists(),
        "cache.db should be in config cache_dir, not CLAWGREP_CACHE_DIR"
    );
    assert!(
        !env_cache.path().join("cache.db").exists(),
        "env dir should NOT get cache when config has cache_dir"
    );
}
