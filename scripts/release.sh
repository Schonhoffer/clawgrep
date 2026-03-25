#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new-version>" >&2
    echo "Example: $0 0.2.0" >&2
    exit 1
fi

NEW_VERSION="$1"

# Validate version format
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: version must be in semver format (e.g. 0.2.0)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Read current version from root Cargo.toml
OLD_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')

if [ "$OLD_VERSION" = "$NEW_VERSION" ]; then
    echo "Error: new version ($NEW_VERSION) is the same as current version" >&2
    exit 1
fi

echo "Bumping $OLD_VERSION -> $NEW_VERSION"

# Escape dots for sed
OLD_ESC=$(echo "$OLD_VERSION" | sed 's/\./\\./g')
NEW_ESC="$NEW_VERSION"

# --- Cargo.toml files (only the version = "..." line near the top) ---
for f in Cargo.toml bindings/node/Cargo.toml bindings/python/Cargo.toml; do
    sed -i "s/^version = \"$OLD_ESC\"/version = \"$NEW_ESC\"/" "$f"
done

# --- pyproject.toml ---
sed -i "s/^version = \"$OLD_ESC\"/version = \"$NEW_ESC\"/" bindings/python/pyproject.toml

# --- Node package.json files (main + platform sub-packages) ---
for f in \
    bindings/node/package.json \
    bindings/node/npm/win32-x64-msvc/package.json \
    bindings/node/npm/darwin-x64/package.json \
    bindings/node/npm/darwin-arm64/package.json \
    bindings/node/npm/linux-x64-gnu/package.json \
    bindings/node/npm/linux-arm64-gnu/package.json; do
    sed -i "s/\"$OLD_ESC\"/\"$NEW_ESC\"/g" "$f"
done

# --- package-lock.json ---
sed -i "s/\"$OLD_ESC\"/\"$NEW_ESC\"/g" bindings/node/package-lock.json

# --- Update Cargo.lock via cargo (avoids hand-editing) ---
cargo update --workspace
echo "Cargo.lock updated"

# --- Verify ---
echo ""
echo "Changed files:"
git diff --name-only

echo ""
echo "Remaining references to old version:"
if grep -rn "\"$OLD_ESC\"" Cargo.toml bindings/ --include='*.toml' --include='*.json' 2>/dev/null; then
    echo "Warning: old version still found in some files" >&2
    exit 1
else
    echo "(none)"
fi

# --- Commit, tag, push ---
git add -A
git commit -m "chore: bump version to $NEW_VERSION"
git push

git tag "v$NEW_VERSION"
git push origin "v$NEW_VERSION"

echo ""
echo "Released v$NEW_VERSION"
