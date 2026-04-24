# Coda Chess Engine — Makefile
# Supports: manual builds, OpenBench integration, PGO builds
#
# Usage:
#   make                  Build with native CPU optimizations
#   make EXE=coda-v2      Build with custom output name
#   make pgo              PGO-optimized build (only helps v5 on main branch — see note below)
#   make openbench        OpenBench-compatible build target
#   make net              Download the production NNUE net
#   make clean            Remove build artifacts

# Configuration
EXE := coda
NET_URL := $(shell cat net.txt 2>/dev/null)
# EVALFILE: defaults to the filename from net.txt (e.g. net-v5-768pw-w7-e800s800-filtered-lowestlr.nnue)
# OB overrides this with an absolute path to the network file.
EVALFILE := $(if $(NET_URL),$(notdir $(NET_URL)),net.nnue)
MIN_RUST_VERSION := 1.70.0

# Platform detection
ifeq ($(OS),Windows_NT)
    NAME := $(EXE).exe
    RM := del /q
else
    NAME := $(EXE)
    RM := rm -f
endif

# Rust flags
export RUSTFLAGS := -Ctarget-cpu=native

# Default: build with embedded NNUE net (downloads from net.txt if needed)
# EVALFILE may be overridden by OpenBench with an absolute path to the network
rule: check-rust net
	CODA_EVALFILE=$(abspath $(EVALFILE)) cargo rustc --release --features embedded-net -- --emit link=$(NAME)

# Alias for OpenBench compatibility
openbench: rule

# PGO build (profile-guided optimization).
#
# Status (2026-04-17):
#   v5 on main:          +3-5% NPS. Use this.
#   v5 on threat branch: -5% NPS regression.
#   v9 on threat branch: -10 to -12% NPS regression. DO NOT USE.
#
# Why it regresses on the threat branch: the PGO *instrument* build inserts
# entry/branch counters that slow the bench profile run by ~20%. That profile
# captures a counter-burdened hot path (small SIMD functions dominate), and
# PGO's inlining/layout decisions are made against that degraded view. The
# final optimised binary over-inlines small functions into the delta-generation
# hot path (push_threats_for_piece and friends), bloating those functions and
# hurting icache behaviour. The regression is independent of binary size —
# confirmed by testing without the embedded net (2.5MB binary regresses the
# same amount as the 72MB one) and independent of which NNUE net is profiled
# (v5 profile + v9 runtime regresses the same as v9 profile + v9 runtime).
#
# If we want profile-guided optimisation for v9 later, try AutoFDO (sampling
# via perf record) instead — its profile reflects actual uncounted execution.
#
# Requires: rustup component add llvm-tools-preview
TARGET_TUPLE := $(shell rustc --print host-tuple 2>/dev/null)
pgo: check-rust net
	CODA_EVALFILE=$(abspath $(EVALFILE)) cargo pgo instrument build -- --features embedded-net
	LLVM_PROFILE_FILE=target/pgo-profiles/coda_%m_%p.profraw ./target/$(TARGET_TUPLE)/release/coda bench 13
	CODA_EVALFILE=$(abspath $(EVALFILE)) cargo pgo optimize build -- --features embedded-net
	cp target/$(TARGET_TUPLE)/release/coda $(NAME)

# Download production NNUE net (uses actual filename from net.txt, not generic net.nnue)
net:
	@if [ ! -f "$(EVALFILE)" ] && [ -n "$(NET_URL)" ]; then \
		echo "Downloading NNUE net from $(NET_URL)..."; \
		curl -sL "$(NET_URL)" -o "$(EVALFILE)"; \
		echo "Downloaded $(EVALFILE)"; \
	elif [ -f "$(EVALFILE)" ]; then \
		echo "$(EVALFILE) already exists"; \
	else \
		echo "Warning: no net.txt found and no $(EVALFILE) present"; \
	fi

# Check Rust toolchain version
check-rust:
	@command -v cargo >/dev/null 2>&1 || { echo "Error: cargo not found. Install Rust from https://rustup.rs"; exit 1; }
	@RUST_VERSION=$$(rustc --version | sed 's/rustc \([0-9]*\.[0-9]*\.[0-9]*\).*/\1/'); \
	MIN="$(MIN_RUST_VERSION)"; \
	if [ "$$(printf '%s\n' "$$MIN" "$$RUST_VERSION" | sort -V | head -n1)" != "$$MIN" ]; then \
		echo "Error: Rust $$RUST_VERSION is too old. Need >= $$MIN. Run: rustup update"; \
		exit 1; \
	fi

clean:
	cargo clean
	$(RM) $(NAME)

.PHONY: rule openbench pgo net check-rust clean
