# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in clawgrep, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, open a [GitHub Security Advisory](https://github.com/Schonhoffer/clawgrep/security/advisories/new).

Include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential mitigations you've identified

You should receive an acknowledgement within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

clawgrep runs fully locally with no network calls at runtime (after model download). Relevant threat areas:

- Path traversal via crafted filenames or symlinks
- Cache poisoning or arbitrary file writes through the SQLite cache
- Denial of service via malformed input files or model files
- Command injection through CLI arguments

## Supported Versions

Only the latest release is actively supported with security fixes.
