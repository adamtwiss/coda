# Distributed SPRT Testing Framework Analysis

**Date**: 2026-03-22
**Context**: 3-5 machines (Intel/AMD, US/Europe), Go chess engine with NNUE, security-conscious (Tailscale VPN)

## OpenBench Assessment

### Architecture
OpenBench is a Django web app (server) with Python workers that poll for work via HTTP. The server stores all state in a database (SQLite or PostgreSQL). Workers authenticate with username/password, request workloads, build engines locally from source, download NNUE nets from the server, run games with fastchess (their cutechess replacement), and report W/D/L results back. The server aggregates results and computes SPRT statistics.

The server-side data model includes: Engines (with repo URL, build config, benchmark hash), Tests (dev vs base engine, branches, SPRT params, time control), Results (per-machine W/D/L), Machines (worker metadata, NPS), Networks (NNUE files stored by SHA256), and Profiles (user accounts with approver status).

### How Tests Are Submitted
Tests are created via web UI or API. You specify:
- Engine name (must be pre-configured in `Engines/*.json`)
- Dev branch and base branch (git refs on the engine's repo)
- Test preset (STC/LTC/SMP, which sets TC, hash, book, SPRT bounds)
- Optional: custom UCI options, network file overrides

Each engine has a JSON config file defining its repo URL, build path, compiler requirements, CPU flags, and test/tune presets.

### How Binaries Are Built
Workers clone the engine repo, checkout the specified branch, and run `make -j EXE=<path>` in the configured build directory. The build system is make-centric. Compilers supported: gcc, clang, and cargo (Rust). The worker auto-detects available compilers and matches them against the engine's config.

**Critical limitation for Go**: OpenBench's build pipeline is hardcoded around `make` with CC/CXX flags. There is no `go build` support. The `makefile_command()` function in `utils.py` constructs make commands with compiler variables. A Go engine would need either:
1. A Makefile wrapper that calls `go build` (simplest)
2. Forking OpenBench to add Go build support (non-trivial)

### Security Model
- Workers authenticate with username/password over HTTP (no TLS enforced by OpenBench itself)
- Workers download and execute arbitrary code from git repos configured on the server
- Private engines use credential files for GitHub token auth
- No worker isolation, sandboxing, or code review step
- The server trusts workers to report accurate results

For our use case (trusted machines on Tailscale), the security model is adequate, but it's also not providing anything we don't already get from Tailscale + our own framework.

### NNUE Network File Handling
Networks are uploaded to the server and stored by SHA256 hash. Workers download them to a local `Networks/` directory. Engine configs reference networks via UCI option (typically `EvalFile`). The server tracks which SHA belongs to which engine.

This works fine but requires all nets to be uploaded to the Django server first. There's no direct rsync/URL-based distribution.

### Self-Hosting Complexity
Running your own OpenBench instance requires:
- Python 3.8+, Django, PostgreSQL (or SQLite for small scale)
- Web server (typically Nginx + Gunicorn for production)
- Domain/TLS setup (or skip for LAN/VPN)
- Engine JSON configs for each engine
- Network file management through the web UI
- User account management
- Worker Python environments on each machine (requests, psutil, etc.)

The [public instance](https://chess.grantnet.us) serves dozens of engines with hundreds of workers. For 3-5 private machines, this is significant infrastructure overhead.

### Would We Need to Fork/Modify?
Yes, substantially:
1. **Go build support**: Need to add `go build` to the worker's build pipeline, or maintain a Makefile wrapper
2. **Engine config**: Create `Engines/GoChess.json` with appropriate build/test presets
3. **Network distribution**: Adapt to our model file workflow (nets are large, change frequently during training)
4. **fastchess dependency**: OpenBench has moved from cutechess-cli to fastchess; we'd need to ensure fastchess works with our engine
5. **NPS scaling**: OpenBench scales time controls based on measured NPS vs reference; needs calibration for Go (which is slower than C++)

---

## Our Custom Framework Assessment

### What's Already Built (`cmd/sprt/`)
The existing framework is surprisingly complete:

**Coordinator (server)**:
- HTTP API: create/list/delete experiments, claim work, report results
- SPRT math: LLR computation (Bernoulli approximation), Elo estimation with 95% CI
- State persistence: JSON file (atomic write with rename)
- Work distribution: fill-one-first priority, batch-based claims
- Stale batch reaper: reclaims work from dead workers (30min timeout)
- Error tracking: auto-pauses experiments after 3 consecutive errors
- Web dashboard: real-time polling UI with LLR progress bars, experiment cards, create/cancel forms

**Worker**:
- Polls coordinator for work
- Git fetch + branch resolution (tries multiple ref strategies)
- Builds engines in git worktrees with `go build` (commit-hash caching)
- Runs cutechess-cli batches with proper argument construction
- NNUE file path resolution (per-engine or shared)
- Parses cutechess output for W/D/L
- Exponential backoff on errors
- Cleanup of worktrees after build

**Test creation CLI**:
- `sprt create -id lmr-c1.25 -branch feature -elo0 0 -elo1 10 -nnue net.nnue`

### What's Missing for Production Use

1. **NNUE file distribution**: Workers resolve NNUE paths relative to the local repo. No mechanism to sync new .nnue files to remote workers. Need: coordinator serves files via HTTP, or rsync-based pre-distribution.

2. **Authentication**: No auth at all currently. Any HTTP client can create/delete experiments and report fake results. Need: at minimum, a shared secret/API key. On Tailscale, this is less critical.

3. **TLS/transport security**: Plain HTTP. On Tailscale, traffic is already encrypted, so this is fine.

4. **Opening book distribution**: Assumes `testdata/noob_3moves.epd` exists locally. Need: either ship in git (already there) or distribute separately.

5. **Auto-logging**: No automatic logging of completed experiments to `experiments.md`. Currently results are in the JSON state file and server logs.

6. **Pentanomial SPRT**: Uses Bernoulli (score-based) LLR, not pentanomial (game-pair-based). Pentanomial is more accurate but the Bernoulli approximation is fine for our scale.

7. **Multi-experiment scheduling**: Currently fill-one-first (all workers on the highest-priority experiment). No round-robin or multi-experiment parallelism.

8. **Worker status monitoring**: No heartbeat or worker list in the dashboard. Can only infer activity from batch reports.

9. **Benchmark validation**: No NPS sanity check. A slow machine could produce time-pressure artifacts.

### Estimated Work to Finish

| Feature | Effort | Priority |
|---------|--------|----------|
| NNUE file distribution (HTTP serve from coordinator) | 2-3 hours | High |
| API key auth (shared secret header) | 1 hour | Medium |
| Auto-log results to experiments.md | 1-2 hours | Medium |
| Worker heartbeat + status page | 2-3 hours | Low |
| Benchmark NPS check | 1-2 hours | Low |
| Multi-experiment scheduling | 2-3 hours | Low |

**Total for production-ready**: ~5-8 hours of focused work.

---

## Requirements Analysis

### Security: Tailscale VPN
All machines are on Tailscale. This gives us:
- Encrypted tunnels between all machines (WireGuard)
- Identity-based access (each machine has a Tailscale identity)
- No exposed ports on the public internet

**Implication**: We can bind the coordinator to the Tailscale interface only (`-addr 100.x.x.x:8080`). Authentication becomes optional since only VPN members can reach the service. This dramatically simplifies our framework and makes OpenBench's user/password system redundant overhead.

### Multi-Machine: Workers in Different DCs
Workers in US and Europe means:
- 100-200ms latency between coordinator and some workers (irrelevant for batch-based work)
- No shared filesystem (correct -- our framework already uses HTTP APIs)
- NNUE files need explicit distribution (HTTP download from coordinator, or rsync)
- Git repos need to be independently fetchable (workers fetch from GitHub, not from coordinator)

Our framework handles this well. Workers clone the repo locally and fetch from origin.

### Model Files: NNUE Distribution
NNUE nets are 1-5 MB, change with training iterations. Options:
1. **Coordinator serves files**: Add a `/api/files/{name}` endpoint. Worker downloads before building. Simple, works over Tailscale.
2. **Git LFS**: Store nets in git. Bloats repo, complex setup.
3. **rsync pre-distribution**: Script to rsync nets to all workers. Manual but reliable.
4. **HTTP URL in experiment config**: Worker downloads from any URL. Flexible.

Option 1 is the simplest for our scale. Option 4 is the most flexible.

### Ease of Use
Current workflow: `sprt create -id test1 -branch feature-x -nnue net.nnue`, then watch the dashboard.
Missing: auto-log results when SPRT concludes. Easy to add as a post-conclusion hook.

### Maintainability
Our framework is ~800 lines of Go in 7 files. Zero external dependencies beyond the stdlib and cutechess-cli. No database, no Django, no Python environments to manage. State is a single JSON file.

OpenBench is ~10K+ lines of Python/Django/HTML/JS with database migrations, user management, engine configs, and a Python worker runtime. Maintaining a fork with Go support adds ongoing merge conflict risk.

---

## Recommendation: Build on Our Own Framework

**Decision**: Extend `cmd/sprt/` rather than adopting OpenBench.

### Rationale

1. **Already 80% done**: The coordinator, worker, SPRT math, web dashboard, and CLI are all working. The remaining gaps are small (NNUE serving, auth, auto-logging).

2. **Go-native**: Our worker builds engines with `go build` in worktrees with commit-hash caching. OpenBench would need a Makefile wrapper or fork modifications.

3. **Minimal infrastructure**: Single Go binary, JSON state file, no database. Deploy with `scp` and `systemd`. OpenBench needs Django + database + Python on every worker.

4. **Tailscale eliminates auth complexity**: On a private VPN, we don't need OpenBench's user management. A simple API key (or nothing) suffices.

5. **Right-sized for 3-5 machines**: OpenBench is designed for dozens of engines and hundreds of workers. Its complexity (engine configs, user profiles, approval workflows, NPS scaling) is overhead for our use case.

6. **Maintainability**: 800 lines of Go we fully understand vs. maintaining a Django fork. No Python dependency management on workers.

7. **We control the protocol**: We can add features tailored to our workflow (auto-log to experiments.md, Claude-friendly status output, custom NNUE distribution).

### When OpenBench Would Be Better
- If we had 10+ engines or 20+ workers
- If we needed multi-user access control
- If we wanted to contribute tests to a public testing community
- If we used C++ and wanted zero setup effort

### Key Design Decisions

1. **Coordinator binds to Tailscale IP only** (`-addr 100.x.x.x:8080`). No public exposure.
2. **NNUE files served by coordinator** via `/api/files/` endpoint. Workers download before each batch if the SHA has changed.
3. **Shared API key** as a request header for basic auth. Prevents accidental interaction from other Tailscale services.
4. **Keep cutechess-cli** (already installed everywhere, well-tested with our engine). No need to switch to fastchess.
5. **Auto-append to experiments.md** when an SPRT concludes. Include date, branch, Elo, LLR, game count.
6. **Worker ID from Tailscale hostname** (already the default behavior).

### Estimated Effort

| Task | Hours |
|------|-------|
| NNUE file serving (coordinator endpoint + worker download) | 3 |
| API key auth middleware | 1 |
| Auto-log to experiments.md on conclusion | 2 |
| Systemd service files for coordinator + worker | 1 |
| Deploy to all machines, test end-to-end | 2 |
| **Total** | **~9 hours** |

Compare with OpenBench adoption: Django setup, database, engine config, Go build support fork, worker Python setup on 3-5 machines, testing -- easily 20+ hours with ongoing maintenance burden.
