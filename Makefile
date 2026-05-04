# Coda Chess Engine — Makefile
# Supports: manual builds, OpenBench integration
#
# Usage:
#   make                  Build with native CPU optimizations
#   make EXE=coda-v2      Build with custom output name
#   make openbench        OpenBench-compatible build target
#   make net              Download the production NNUE net
#   make clean            Remove build artifacts
#
# `make pgo` was disabled 2026-05-04 — see commented-out rule below and
# docs/pgo_v9_regression_2026-05-04.md. PGO regresses v9 by 12-13% NPS.

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

# PGO build — DISABLED 2026-05-04.
#
# Status:
#   v5 on main:          +3-5% NPS. Worked.
#   v5 on threat branch: -5% NPS regression.
#   v9 on main (current): -12.5% NPS regression. DO NOT USE.
#
# 2026-05-04 investigation: docs/pgo_v9_regression_2026-05-04.md
# Headline: the regression is +12.4% executed instructions. PGO's icache
# and iTLB are actually *better* than no-PGO; IPC is unchanged. The
# slowdown is from PGO's profile-driven inlining decisions — small SIMD
# helpers (simd512_pairwise_pack_fused, finny_batch_apply,
# push_threats_for_piece, MovePicker::pick_best, etc.) get aggressively
# inlined into hot callers, and the inlined versions execute more total
# instructions than the standalone-call versions did. None of the LLVM
# inline-threshold knobs recover more than ~0.5%.
#
# Cargo-pgo bug also discovered: `cargo pgo instrument` doesn't inherit
# `rustflags = ["-C", "target-cpu=native"]` from .cargo/config.toml, so
# the instrumented binary lacks AVX-512 functions on AVX-512 hosts. Setting
# RUSTFLAGS explicitly in the pgo target below fixes that part. Doesn't
# help the inlining-driven regression but is correct defensive practice
# if/when the rule is re-enabled.
#
# To re-enable: uncomment the rule + .PHONY entry below. Validate with a
# bench delta + SPRT before merging.
#
# If we want profile-guided optimisation later, try AutoFDO (sampling via
# perf record) instead — sampling-based profile reflects actual uncounted
# execution and may not trigger the same over-inlining behaviour.
#
# Requires: rustup component add llvm-tools-preview && cargo install cargo-pgo
#
# TARGET_TUPLE := $(shell rustc --print host-tuple 2>/dev/null)
# pgo: check-rust net
# 	RUSTFLAGS="-Ctarget-cpu=native" CODA_EVALFILE=$(abspath $(EVALFILE)) cargo pgo instrument build -- --features embedded-net
# 	LLVM_PROFILE_FILE=target/pgo-profiles/coda_%m_%p.profraw ./target/$(TARGET_TUPLE)/release/coda bench 13
# 	RUSTFLAGS="-Ctarget-cpu=native" CODA_EVALFILE=$(abspath $(EVALFILE)) cargo pgo optimize build -- --features embedded-net
# 	cp target/$(TARGET_TUPLE)/release/coda $(NAME)

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

.PHONY: rule openbench net check-rust clean
