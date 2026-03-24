#!/usr/bin/env node

const { spawnSync } = require("child_process");
const path = require("path");
const os = require("os");
const fs = require("fs");

const PLATFORM_PACKAGES = {
  "darwin-arm64": "@clawgrep/clawgrep-darwin-arm64",
  "darwin-x64": "@clawgrep/clawgrep-darwin-x64",
  "linux-arm64": "@clawgrep/clawgrep-linux-arm64-gnu",
  "linux-x64": "@clawgrep/clawgrep-linux-x64-gnu",
  "win32-x64": "@clawgrep/clawgrep-win32-x64-msvc",
};

function getBinaryPath() {
  const platform = os.platform();
  const arch = os.arch();
  const ext = platform === "win32" ? ".exe" : "";
  const binaryName = `clawgrep${ext}`;

  // Try the platform-specific napi package first.
  const pkg = PLATFORM_PACKAGES[`${platform}-${arch}`];
  if (pkg) {
    try {
      const pkgDir = path.dirname(require.resolve(`${pkg}/package.json`));
      const candidate = path.join(pkgDir, binaryName);
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    } catch {
      // Package not installed — fall through.
    }
  }

  // Fall back to a binary next to this script (local dev / standalone).
  const local = path.join(__dirname, binaryName);
  if (fs.existsSync(local)) {
    return local;
  }

  console.error(
    `clawgrep: binary not found for ${platform}-${arch}.\n` +
      `Install the platform package: npm install ${pkg || "clawgrep"}`
  );
  process.exit(2);
}

const result = spawnSync(getBinaryPath(), process.argv.slice(2), {
  stdio: "inherit",
});
process.exit(result.status ?? 2);
