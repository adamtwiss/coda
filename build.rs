//! Build-time version stamping (single source of truth: Cargo.toml version +
//! git tags). The UCI `id name` and release artifacts all derive from this.
//!
//! - Built exactly on an engine release tag (`vX.Y.Z`)   -> "X.Y.Z"
//! - Any other commit -> `git describe` output, e.g. "1.0.0-37-gabc1234-dirty"
//!   or, before any engine tag exists, "0.9.0-dev+abc1234"
//! - No .git available (source tarball)                  -> "X.Y.Z-nogit"
//!
//! Net-asset release tags (`*-nets*`) are excluded so they can never be
//! mistaken for engine versions (the talkchess "net_v0.7.0" incident).

use std::process::Command;

fn main() {
    let pkg = env!("CARGO_PKG_VERSION");
    let desc = Command::new("git")
        .args(["describe", "--tags", "--dirty", "--always", "--exclude", "*-nets*"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    let version = match desc {
        // Tag-based describe contains a '.', a bare fallback sha does not.
        Some(d) if d.contains('.') => d.strip_prefix('v').unwrap_or(&d).to_string(),
        Some(sha) => format!("{pkg}-dev+{sha}"),
        None => format!("{pkg}-nogit"),
    };

    println!("cargo:rustc-env=CODA_VERSION={version}");
    // Re-stamp when HEAD moves (branch switch / new commit) — also narrows the
    // stale-relink window that produced the #2573 wrong-bench incident.
    //
    // HEAD alone is not enough: on a branch it only says "ref: refs/heads/main",
    // so a `git pull` or new commit moves refs/heads/main (or packed-refs) and
    // leaves HEAD untouched — the version stayed stale until `make clean`.
    // Watch the ref HEAD points at and packed-refs too. Paths come from git
    // itself so linked worktrees (whose HEAD lives elsewhere) work as well.
    let git_path = |p: &str| {
        Command::new("git")
            .args(["rev-parse", "--git-path", p])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let mut watch = vec!["HEAD".to_string(), "packed-refs".to_string(), "refs/tags".to_string()];
    if let Some(r) = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    {
        watch.push(r); // e.g. refs/heads/main
    }
    for w in watch {
        if let Some(p) = git_path(&w) {
            println!("cargo:rerun-if-changed={p}");
        }
    }
}
