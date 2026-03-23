#![allow(dead_code)]
//! Shared test helpers for clawgrep integration tests.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;

use clawgrep::embed::Embedder;
use clawgrep::index::{DiscoverOpts, IndexOpts};

/// Shared embedder loaded once across all tests.  Avoids parallel tests
/// competing to download the model simultaneously.
pub fn shared_embedder() -> &'static Embedder {
    static EMBEDDER: OnceLock<Embedder> = OnceLock::new();
    EMBEDDER.get_or_init(|| Embedder::new(None).expect("failed to load embedding model"))
}

pub fn test_discover_opts() -> DiscoverOpts<'static> {
    DiscoverOpts {
        use_gitignore: false,
        custom_ignore_files: &[],
    }
}

pub fn test_index_opts() -> IndexOpts<'static> {
    IndexOpts {
        reindex: false,
        no_cache: true,
        custom_cache: None,
        path_boost: 1.0,
        verbose: false,
    }
}

/// Create a temp directory tree that looks like a realistic project.
pub fn make_project(dir: &Path) {
    fs::write(
        dir.join("README.md"),
        "# My Project\n\nA tool for managing inventory.\n\nSee docs/guide.md for usage.\n",
    )
    .unwrap();

    fs::write(dir.join("empty.txt"), "").unwrap();

    {
        let mut f = fs::File::create(dir.join("binary.bin")).unwrap();
        f.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x00, 0x00, 0x01])
            .unwrap();
    }

    let src = dir.join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("main.rs"),
        "fn main() {\n    println!(\"starting inventory server\");\n    let db = connect_database(\"postgres://localhost/inventory\");\n    run_server(db);\n}\n",
    ).unwrap();
    fs::write(
        src.join("lib.rs"),
        "pub fn connect_database(url: &str) -> Database {\n    // open a persistent connection pool\n    Database::new(url)\n}\n\npub fn scan_barcode(upc: &str) -> Option<Product> {\n    // lookup UPC barcode in the product catalog\n    catalog::find_by_upc(upc)\n}\n\npub fn process_return(order_id: u64) -> Result<(), Error> {\n    // handle product return and restock inventory\n    let order = orders::get(order_id)?;\n    inventory::restock(order.items());\n    Ok(())\n}\n",
    ).unwrap();

    let docs = dir.join("docs");
    fs::create_dir_all(&docs).unwrap();
    fs::write(
        docs.join("guide.md"),
        "# Usage Guide\n\n## Installation\nDownload the binary and add it to your PATH.\n\n## Searching Inventory\nUse the search command to find products by name or UPC barcode.\n\n## Deployment\nDeploy using Docker: `docker compose up -d`\nMake sure all environment variables are set in .env file.\n\n## Troubleshooting\nIf the server fails to start, check the database connection string.\nPermission errors usually mean the data directory is not writable.\n",
    ).unwrap();
    fs::write(
        docs.join("faq.md"),
        "# FAQ\n\nQ: How do I reset my password?\nA: Use the `forgot-password` endpoint or contact support.\n\nQ: What barcode formats are supported?\nA: We support UPC-A, EAN-13, Code 128, and QR codes.\n\nQ: Can I export data to CSV?\nA: Yes, use `clawgrep export --format csv`.\n",
    ).unwrap();

    let logs = dir.join("logs");
    fs::create_dir_all(&logs).unwrap();
    fs::write(
        logs.join("deploy.log"),
        "[2026-01-15 10:30] INFO: Starting deployment to production\n[2026-01-15 10:31] INFO: Building Docker image inventory:v2.3.1\n[2026-01-15 10:33] ERROR: Build failed - missing environment variable DATABASE_URL\n[2026-01-15 10:35] INFO: Fixed .env, retrying build\n[2026-01-15 10:37] INFO: Image built successfully\n[2026-01-15 10:38] INFO: Deploying to kubernetes cluster prod-east\n[2026-01-15 10:40] INFO: Deployment complete, all pods healthy\n",
    ).unwrap();
    fs::write(
        logs.join("errors.log"),
        "[2026-01-16 08:00] ERROR: Connection refused to postgres://db:5432/inventory\n[2026-01-16 08:01] WARN: Retrying database connection (attempt 2/5)\n[2026-01-16 08:02] ERROR: Permission denied reading /var/data/inventory.db\n[2026-01-16 08:03] ERROR: Failed to scan barcode - invalid UPC format: 12345\n[2026-01-16 08:05] INFO: Database connection restored after failover\n",
    ).unwrap();

    let git = dir.join(".git");
    fs::create_dir_all(&git).unwrap();
    fs::write(
        git.join("config"),
        "[core]\n\trepositoryformatversion = 0\n",
    )
    .unwrap();

    let nm = dir.join("node_modules");
    fs::create_dir_all(&nm).unwrap();
    fs::write(
        nm.join("leftpad.js"),
        "module.exports = (s,n) => s.padStart(n);\n",
    )
    .unwrap();

    fs::write(dir.join(".gitignore"), "node_modules/\n").unwrap();
}
