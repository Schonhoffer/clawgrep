# Examples

## Basic search

Search a directory for a concept:

```bash
clawgrep --no-color "database connection timeout" ./src
```

Output:

```
src/db.rs:42:let pool = ConnectionPool::new(config.timeout)
src/db.rs:78:connection.reconnect_with_backoff()
src/db.rs:15:use crate::pool::{ConnectionPool, PoolConfig};
src/config.rs:23:pub db_pool_size: usize,
src/config.rs:24:pub db_timeout_ms: u64,
```

Results are ordered by relevance (best match first), not by file position.

## Context lines

Show surrounding code with `-C`:

```bash
clawgrep --no-color -C 2 "authentication middleware" ./src
```

Output:

```
src/auth.rs-10-use crate::token::verify_jwt;
src/auth.rs-11-
src/auth.rs:12:pub async fn auth_middleware(req: Request) -> Result<Request> {
src/auth.rs-13-    let token = req.header("Authorization");
src/auth.rs-14-    let claims = verify_jwt(token)?;
```

Match lines use `:` as separator. Context lines use `-`. Identical to grep.

## Relevance scores

Append scores with `--show-score`:

```bash
clawgrep --no-color --show-score "database connection" ./src
```

Output:

```
src/db.rs:42:let pool = ConnectionPool::new(config.timeout)	(0.912)
src/db.rs:78:connection.reconnect_with_backoff()	(0.847)
```

Tab-separated score (0.0–1.0) appended to each line.

## Exact identifier search

Find a specific function name by shifting weights toward keyword:

```bash
clawgrep --no-color --keyword-weight 0.8 --semantic-weight 0.2 "handleUserAuth" ./src
```

Output:

```
src/auth.rs:45:pub fn handleUserAuth(req: &Request) -> AuthResult {
src/routes.rs:12:use crate::auth::handleUserAuth;
src/routes.rs:78:let result = handleUserAuth(&req);
```

## Concept search

Find code by intent when you don't know the exact wording:

```bash
clawgrep --no-color "retry logic with exponential backoff" ./src
```

Output:

```
src/http.rs:91:async fn retry_request(url: &str, max_attempts: u32) -> Result<Response> {
src/http.rs:95:    let delay = Duration::from_millis(100 * 2u64.pow(attempt));
src/http.rs:88:/// Retries failed HTTP requests with increasing delays.
```

## Listing matching files

Get only filenames with `-l`:

```bash
clawgrep --no-color -l "database" ./src
```

Output:

```
src/db.rs
src/config.rs
src/migrations.rs
```

## Match count

Count matches per file with `-c`:

```bash
clawgrep --no-color -c "error" ./src
```

Output:

```
src/handler.rs:3
src/db.rs:2
src/config.rs:1
```

## Quiet / existence check

Check whether something exists without output:

```bash
clawgrep -q "TODO" ./src
echo $?   # 0 if found, 1 if not
```

Useful in conditionals:

```bash
if clawgrep -q "security vulnerability" ./audit; then
  echo "Issues found"
fi
```

## Boosting path matches

Rank filename matches higher with `--path-boost`:

```bash
clawgrep --no-color --path-boost 2.0 "utils" ./src
```

Output:

```
src/utils.rs:1://! General utility functions.
src/utils.rs:15:pub fn format_duration(d: Duration) -> String {
src/handler.rs:3:use crate::utils::format_duration;
```

The file named `utils.rs` is ranked first because `--path-boost 2.0` doubles
the weight of path matches.

## More results

Get more results with `-k`:

```bash
clawgrep --no-color -k 20 "error handling" ./src
```

Returns up to 20 results instead of the default 5.

## Minimum score threshold

Filter out low-quality results:

```bash
clawgrep --no-color --min-score 0.5 "authentication" ./src
```

Only returns results with a combined score of 0.5 or higher.

## Piping stdin

Search piped content (no caching):

```bash
cat error.log | clawgrep --no-color "timeout"
git diff | clawgrep --no-color "security"
```

When reading from stdin with a single source, the filename prefix is omitted
(matching grep behavior):

```
42:connection timed out after 30s
78:timeout exceeded for request /api/health
```

## Custom ignore file

Add project-specific ignore rules:

```bash
clawgrep --no-color --ignore-file .clawgrepignore "todo" .
```

The ignore file uses the same syntax as `.gitignore`.

## Force re-index

If results seem stale after file changes:

```bash
clawgrep --no-color --reindex "startup sequence" ./src
```

## No-cache mode

Skip the cache entirely for throwaway searches:

```bash
clawgrep --no-color --no-cache "one-off query" ./tmp
```
