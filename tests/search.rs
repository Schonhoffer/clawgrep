//! Model and hybrid search tests — require the real embedding model.

mod common;

use std::fs;

use tempfile::TempDir;

use clawgrep::embed::{cosine_similarity, DEFAULT_DIM};
use clawgrep::index::{build_index, discover_files, IndexOpts};
use clawgrep::search::{hybrid_search, SearchOpts};

use common::{make_project, shared_embedder, test_discover_opts, test_index_opts};

// ── Embedding model ────────────────────────────────────────────────────

#[test]
fn model_embed_single_text() {
    let embedder = shared_embedder();
    assert_eq!(
        embedder.embed_one("hello world").unwrap().len(),
        DEFAULT_DIM
    );
}

#[test]
fn model_embed_batch() {
    let embedder = shared_embedder();
    let vecs = embedder
        .embed_batch(&["first", "second", "third"], false)
        .unwrap();
    assert_eq!(vecs.len(), 3);
    for v in &vecs {
        assert_eq!(v.len(), DEFAULT_DIM);
    }
}

#[test]
fn model_embed_empty_batch() {
    assert!(shared_embedder()
        .embed_batch(&[], false)
        .unwrap()
        .is_empty());
}

#[test]
fn model_similar_texts_score_high() {
    let embedder = shared_embedder();
    let v1 = embedder
        .embed_one("the database connection failed")
        .unwrap();
    let v2 = embedder
        .embed_one("could not connect to the database")
        .unwrap();
    let v3 = embedder.embed_one("sunny weather at the beach").unwrap();
    assert!(cosine_similarity(&v1, &v2) > cosine_similarity(&v1, &v3));
    assert!(cosine_similarity(&v1, &v2) > 0.5);
}

#[test]
fn model_embed_long_text_truncates_without_error() {
    let embedder = shared_embedder();
    let long_text: String = (0..800).map(|i| format!("word{i} ")).collect();
    let vec = embedder.embed_one(&long_text).unwrap();
    assert_eq!(vec.len(), DEFAULT_DIM);
}

// ── Index build ────────────────────────────────────────────────────────

#[test]
fn model_build_index_creates_cache() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let cache_tmp = TempDir::new().unwrap();
    let custom_cache = cache_tmp.path().join("test-cache");
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let opts = IndexOpts {
        reindex: false,
        no_cache: false,
        custom_cache: Some(custom_cache.as_path()),
        path_boost: 1.0,
        verbose: false,
    };
    let index = build_index(&files, &embedder, &opts).unwrap();
    assert!(custom_cache.join("cache.db").exists());
    assert!(!index.entries.is_empty());
}

#[test]
fn model_incremental_index_skips_unchanged() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let cache_tmp = TempDir::new().unwrap();
    let custom_cache = cache_tmp.path().join("test-cache");
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let opts = IndexOpts {
        reindex: false,
        no_cache: false,
        custom_cache: Some(custom_cache.as_path()),
        path_boost: 1.0,
        verbose: false,
    };
    let m1 = build_index(&files, &embedder, &opts).unwrap();
    let m2 = build_index(&files, &embedder, &opts).unwrap();
    assert_eq!(m1.entries.len(), m2.entries.len());
}

#[test]
fn model_reindex_flag_ignores_cache() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("test.txt"), "hello world\n").unwrap();
    let files = vec![dir.path().join("test.txt")];
    let embedder = shared_embedder();
    let _ = build_index(
        &files,
        &embedder,
        &IndexOpts {
            reindex: false,
            no_cache: false,
            custom_cache: None,
            path_boost: 1.0,
            verbose: false,
        },
    )
    .unwrap();
    let m2 = build_index(
        &files,
        &embedder,
        &IndexOpts {
            reindex: true,
            no_cache: false,
            custom_cache: None,
            path_boost: 1.0,
            verbose: false,
        },
    )
    .unwrap();
    assert!(!m2.entries.is_empty());
}

#[test]
fn model_no_cache_flag_doesnt_persist() {
    let dir = TempDir::new().unwrap();
    let cache_tmp = TempDir::new().unwrap();
    let custom_cache = cache_tmp.path().join("no-cache-test");
    fs::write(dir.path().join("test.txt"), "hello world\n").unwrap();
    let files = vec![dir.path().join("test.txt")];
    let embedder = shared_embedder();
    // no_cache=true means no db is written
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    assert!(!index.entries.is_empty());
    assert!(!custom_cache.join("cache.db").exists());
}

// ── Hybrid search ──────────────────────────────────────────────────────

#[test]
fn model_hybrid_search_deployment_issue() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "deployment issue",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 5,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty());
    let top = results[0].text.to_lowercase();
    assert!(
        top.contains("deploy")
            || top.contains("docker")
            || top.contains("kubernetes")
            || top.contains("production")
            || top.contains("build"),
        "top: {top}"
    );
}

#[test]
fn model_hybrid_search_barcode_upc() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "UPC barcode scanner",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 5,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(
        |r| r.text.to_lowercase().contains("barcode") || r.text.to_lowercase().contains("upc")
    ));
}

#[test]
fn model_hybrid_search_database_connection() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "database connection failed",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 3,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty());
    let top = results[0].text.to_lowercase();
    assert!(
        top.contains("database") || top.contains("connect") || top.contains("postgres"),
        "top: {top}"
    );
}

#[test]
fn model_hybrid_search_min_score_filter() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "random",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 100,
            min_score: Some(0.99),
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(results.len() < 3, "got {}", results.len());
}

#[test]
fn model_hybrid_search_top_k_limits_results() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "error",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 2,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(results.len() <= 2);
}

#[test]
fn model_hybrid_search_scores_descending() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "permission error",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 10,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    for w in results.windows(2) {
        assert!(w[0].score >= w[1].score, "{} >= {}", w[0].score, w[1].score);
    }
}

#[test]
fn model_keyword_only_search() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "barcode",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 3,
            semantic_weight: 0.0,
            keyword_weight: 1.0,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty());
    for r in &results {
        assert!(
            r.text.to_lowercase().contains("barcode"),
            "kw-only: {}",
            r.text
        );
    }
}

#[test]
fn model_semantic_only_search() {
    let dir = TempDir::new().unwrap();
    make_project(dir.path());
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "problems with authorization",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 3,
            semantic_weight: 1.0,
            keyword_weight: 0.0,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| {
        let t = r.text.to_lowercase();
        t.contains("permission") || t.contains("password") || t.contains("denied")
    }));
}

#[test]
fn model_search_empty_directory() {
    let dir = TempDir::new().unwrap();
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    assert!(hybrid_search(
        "anything",
        &index,
        &files,
        &embedder,
        &SearchOpts::default()
    )
    .unwrap()
    .is_empty());
}

#[test]
fn model_search_single_file() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("lonely.txt");
    fs::write(
        &file,
        "The server crashed at midnight\nAll tests are passing\nDisk space is running low\n",
    )
    .unwrap();
    let files = vec![file];
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "server error",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 1,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert_eq!(results.len(), 1);
    let t = results[0].text.to_lowercase();
    assert!(t.contains("crash") || t.contains("server"), "got: {t}");
}

#[test]
fn model_index_file_exceeding_token_limit() {
    let dir = TempDir::new().unwrap();
    let long_content: String = (0..800).map(|i| format!("token{i} ")).collect();
    let file = dir.path().join("big.txt");
    fs::write(&file, &long_content).unwrap();
    let files = vec![file];

    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    assert!(!index.entries.is_empty());

    let results = hybrid_search(
        "token0 token1",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 1,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty());
}

// ── Path indexing ──────────────────────────────────────────────────────

#[test]
fn path_indexing_includes_path_chunks() {
    let dir = TempDir::new().unwrap();
    let sub = dir.path().join("special_module");
    fs::create_dir_all(&sub).unwrap();
    fs::write(sub.join("handler.rs"), "fn handle() {}\n").unwrap();

    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(
        &files,
        &embedder,
        &IndexOpts {
            reindex: false,
            no_cache: true,
            custom_cache: None,
            path_boost: 1.0,
            verbose: false,
        },
    )
    .unwrap();

    // Path chunk should exist: a chunk whose file path contains "special_module"
    // and has a boost that might differ. But since we no longer store text in
    // cache entries, we check that the file was indexed and has a path chunk
    // by counting chunks (should be > 1 for a 1-line file: 1 content + 1 path).
    let entry = index
        .entries
        .iter()
        .find(|e| e.path.contains("handler.rs"))
        .expect("should have entry for handler.rs");
    assert!(
        entry.num_chunks >= 2,
        "with path_boost=1.0, should have content + path chunk"
    );
}

#[test]
fn path_boost_zero_skips_path_chunks() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("test.txt"), "hello world\n").unwrap();

    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(
        &files,
        &embedder,
        &IndexOpts {
            reindex: false,
            no_cache: true,
            custom_cache: None,
            path_boost: 0.0,
            verbose: false,
        },
    )
    .unwrap();

    let has_boosted = index
        .entries
        .iter()
        .any(|entry| entry.chunks.iter().any(|c| c.boost != 1.0));
    assert!(!has_boosted, "path_boost=0 should produce no path chunks");
}

#[test]
fn path_boost_affects_ranking() {
    let dir = TempDir::new().unwrap();
    let sub = dir.path().join("database_utils");
    fs::create_dir_all(&sub).unwrap();
    fs::write(
        sub.join("connector.rs"),
        "fn unrelated_function() {\n    let x = 42;\n    println!(\"no db here\");\n}\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("notes.txt"),
        "The database connector handles pooling and retry logic.\n",
    )
    .unwrap();

    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(
        &files,
        &embedder,
        &IndexOpts {
            reindex: false,
            no_cache: true,
            custom_cache: None,
            path_boost: 3.0,
            verbose: false,
        },
    )
    .unwrap();
    let results = hybrid_search(
        "database connector",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 5,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty());
    let has_path_result = results
        .iter()
        .any(|r| r.file.to_string_lossy().contains("database_utils"));
    assert!(has_path_result, "high path_boost should surface path match");
}

// ── RRF fusion ─────────────────────────────────────────────────────────

#[test]
fn model_rrf_prefers_chunk_ranked_in_both_signals() {
    // Three files:
    // - both.md: matches the keyword "kubernetes" AND is semantically
    //   related to "container orchestration cluster".
    // - sem_only.md: semantically related, but does not contain the
    //   literal word "kubernetes".
    // - kw_only.md: contains "kubernetes" but in an off-topic sentence.
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("both.md"),
        "Our kubernetes cluster runs multiple containerized microservices \
         in production and auto-scales based on load.\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("sem_only.md"),
        "We orchestrate our container workloads across a managed cluster \
         that automatically schedules pods and balances load.\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("kw_only.md"),
        "The word kubernetes appears here exactly once and otherwise this \
         document is about baking sourdough bread.\n",
    )
    .unwrap();

    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "kubernetes container orchestration cluster",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 3,
            semantic_weight: 0.5,
            keyword_weight: 0.5,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert_eq!(results.len(), 3);
    assert!(
        results[0].file.ends_with("both.md"),
        "RRF should rank the chunk ranked highly by both signals first; got {:?}",
        results[0].file
    );
    // All scores should land in [0, 1].
    for r in &results {
        assert!(r.score >= 0.0 && r.score <= 1.0, "score = {}", r.score);
    }
}

// ── Search opts ────────────────────────────────────────────────────────

#[test]
fn search_opts_default_weights() {
    let opts = SearchOpts::default();
    assert_eq!(opts.top_k, 5);
    assert!((opts.semantic_weight - 0.7).abs() < 1e-6);
    assert!((opts.keyword_weight - 0.3).abs() < 1e-6);
}

// ── Non-English / Unicode ──────────────────────────────────────────────

#[test]
fn model_embed_cjk_text() {
    let embedder = shared_embedder();
    let vec = embedder.embed_one("数据库连接失败").unwrap();
    assert_eq!(vec.len(), DEFAULT_DIM);
    // Should produce a non-zero embedding.
    assert!(vec.iter().any(|&v| v.abs() > 1e-6));
}

#[test]
fn model_embed_cyrillic_text() {
    let embedder = shared_embedder();
    let vec = embedder.embed_one("база данных недоступна").unwrap();
    assert_eq!(vec.len(), DEFAULT_DIM);
    assert!(vec.iter().any(|&v| v.abs() > 1e-6));
}

#[test]
fn model_embed_arabic_text() {
    let embedder = shared_embedder();
    let vec = embedder.embed_one("اتصال قاعدة البيانات فشل").unwrap();
    assert_eq!(vec.len(), DEFAULT_DIM);
    assert!(vec.iter().any(|&v| v.abs() > 1e-6));
}

#[test]
fn model_similar_cjk_texts_score_high() {
    let embedder = shared_embedder();
    let v1 = embedder.embed_one("数据库连接失败").unwrap();
    let v2 = embedder.embed_one("无法连接到数据库").unwrap();
    let v3 = embedder.embed_one("今天天气很好").unwrap();
    assert!(
        cosine_similarity(&v1, &v2) > cosine_similarity(&v1, &v3),
        "similar CJK texts should score higher than unrelated"
    );
}

#[test]
fn model_hybrid_search_cjk_content() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("zh.md"),
        "# 系统日志\n\n数据库连接失败，请检查配置。\n\n服务器已成功启动。\n",
    )
    .unwrap();
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "数据库",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 5,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty(), "should find CJK content");
}

#[test]
fn model_hybrid_search_cyrillic_content() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("ru.md"),
        "# Журнал ошибок\n\nОшибка подключения к базе данных.\n\nСервер запущен успешно.\n",
    )
    .unwrap();
    let files = discover_files(dir.path(), &test_discover_opts());
    let embedder = shared_embedder();
    let index = build_index(&files, &embedder, &test_index_opts()).unwrap();
    let results = hybrid_search(
        "ошибка базы данных",
        &index,
        &files,
        &embedder,
        &SearchOpts {
            top_k: 5,
            ..SearchOpts::default()
        },
    )
    .unwrap();
    assert!(!results.is_empty(), "should find Cyrillic content");
}
