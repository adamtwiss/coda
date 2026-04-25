#!/usr/bin/env bash
# One-time setup for nps_microbench.sh on a new host.
#
# Ensures:
#   - Reckless clone exists at $RECKLESS_DIR with the Coda-comparison
#     evalbench patch applied (idempotent — re-running is a no-op).
#   - v9 production net is present at the path nps_microbench.sh expects.
#
# Usage (from inside the Coda checkout):
#   ./scripts/nps_microbench_setup.sh
#   ./scripts/nps_microbench.sh          # produces CSV + appends to doc
#
# Env overrides:
#   CODA_DIR       default: $HOME/code/coda
#   RECKLESS_DIR   default: $HOME/chess/engines/Reckless
#   RECKLESS_REPO  default: https://github.com/codedeliveryservice/Reckless.git

set -euo pipefail

CODA_DIR=${CODA_DIR:-$HOME/code/coda}
RECKLESS_DIR=${RECKLESS_DIR:-$HOME/chess/engines/Reckless}
RECKLESS_REPO=${RECKLESS_REPO:-https://github.com/codedeliveryservice/Reckless.git}

# 1. Reckless clone
if [[ ! -d "$RECKLESS_DIR/.git" ]]; then
    echo "==== Cloning Reckless -> $RECKLESS_DIR ===="
    mkdir -p "$(dirname "$RECKLESS_DIR")"
    git clone "$RECKLESS_REPO" "$RECKLESS_DIR"
else
    echo "==== Pulling Reckless (already cloned) ===="
    (cd "$RECKLESS_DIR" && git pull --ff-only)
fi

# 2. Apply evalbench patch — idempotent check via file presence.
if [[ -f "$RECKLESS_DIR/src/tools/eval_bench.rs" ]]; then
    echo "==== Reckless evalbench patch already applied ===="
else
    echo "==== Applying Reckless evalbench patch ===="
    (cd "$RECKLESS_DIR" && git apply "$CODA_DIR/scripts/reckless_evalbench.patch")
fi

# 3. v9 production net
NET_PATH="$CODA_DIR/net-v9-768th16x32-kb10-w15-e800s800-crelu.nnue"
if [[ -f "$NET_PATH" ]]; then
    echo "==== v9 net present at $NET_PATH ===="
else
    # Try existing nets/ subdirectory first (common on dev hosts).
    if [[ -f "$CODA_DIR/nets/net-v9-768th16x32-kb10-w15-e800s800-crelu.nnue" ]]; then
        ln -sf "$CODA_DIR/nets/net-v9-768th16x32-kb10-w15-e800s800-crelu.nnue" "$NET_PATH"
        echo "==== Symlinked v9 net from nets/ ===="
    else
        echo "==== Fetching v9 net via make net ===="
        (cd "$CODA_DIR" && make net)
    fi
fi

echo ""
echo "Setup complete. Next:  ./scripts/nps_microbench.sh"
