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
# SF-style dual-net approach: optional embedded small net. Defaults to the
# in-repo file when present (temporary hosting; moves to the release page +
# small-net.txt on merge). Override/disable: make SMALL_EVALFILE=
SMALL_EVALFILE ?= $(wildcard nets/smallnet-256pw-mat400-s400.nnue)
# EVALFILE: defaults to the filename from net.txt (e.g. net-v5-768pw-w7-e800s800-filtered-lowestlr.nnue)
# OB overrides this with an absolute path to the network file.
EVALFILE := $(if $(NET_URL),$(notdir $(NET_URL)),net.nnue)
MIN_RUST_VERSION := 1.70.0
comma := ,

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
	CODA_EVALFILE=$(abspath $(EVALFILE)) $(if $(SMALL_EVALFILE),CODA_SMALL_EVALFILE=$(abspath $(SMALL_EVALFILE))) cargo rustc --release --features embedded-net$(if $(TUNE),$(comma)tune)$(if $(SMALL_EVALFILE),$(comma)embedded-small-net) -- --emit link=$(NAME)

# OpenBench build — enables the `tune` feature so the SPSA tunable UCI options are
# advertised (required for OpenBench SPSA to setoption them). Normal `make` and the
# release build keep them hidden. Point OpenBench's Coda build command at `make openbench`.
openbench: TUNE := 1
openbench: rule

# PGO build (profile-guided optimization).
#
# Status (2026-05-17): builds cleanly with cgu=16 (the v9-era crash that
# the prior comment described is gone), but PGO with bench-13 profile is
# fleet-fragile and not shipped as default. Production-class hardware
# (Lichess host, "fast ionos" cohort) shows ~0 Elo; Zen 5 hosts (zeus)
# regress -18.5 Elo. Older silicon (Coffee Lake Xeon, Zen 1) wins.
#
# See docs/pgo_fleet_finding_2026-05-17.md for the full per-machine data
# and why we're not using PGO by default.
#
# `make pgo` here remains available for one-off local experiments and as
# scaffolding for future profile-strategy work (richer profile, AutoFDO).
#
# Requires: cargo install cargo-pgo; rustup component add llvm-tools-preview
TARGET_TUPLE := $(shell rustc --print host-tuple 2>/dev/null)
pgo: check-rust net
	CODA_EVALFILE=$(abspath $(EVALFILE)) cargo pgo instrument build -- --features embedded-net
	LLVM_PROFILE_FILE=target/pgo-profiles/coda_%m_%p.profraw ./target/$(TARGET_TUPLE)/release/coda bench 13
	CODA_EVALFILE=$(abspath $(EVALFILE)) cargo pgo optimize build -- --features embedded-net
	cp target/$(TARGET_TUPLE)/release/coda $(NAME)

# Download production NNUE net (uses actual filename from net.txt, not generic net.nnue)
net:
	@if [ ! -s "$(EVALFILE)" ] && [ -n "$(NET_URL)" ]; then \
		set -e; \
		tmp="$(EVALFILE).tmp"; \
		trap 'rm -f "$$tmp"' EXIT; \
		echo "Downloading NNUE net from $(NET_URL)..."; \
		curl -fsSL --retry 3 --retry-delay 2 "$(NET_URL)" -o "$$tmp"; \
		test -s "$$tmp"; \
		mv "$$tmp" "$(EVALFILE)"; \
		trap - EXIT; \
		echo "Downloaded $(EVALFILE)"; \
	elif [ -s "$(EVALFILE)" ]; then \
		echo "$(EVALFILE) already exists"; \
	else \
		echo "Warning: no net.txt found and no $(EVALFILE) present"; \
	fi

# Check Rust toolchain version.
# Version compare uses awk rather than `sort -V`: sort's version-sort flag is a
# GNU coreutils extension absent from Git Bash / minimal Windows Unix shells,
# where it silently mis-sorted and broke the check.
check-rust:
	@command -v cargo >/dev/null 2>&1 || { echo "Error: cargo not found. Install Rust from https://rustup.rs"; exit 1; }
	@RUST_VERSION=$$(rustc --version | sed 's/rustc \([0-9]*\.[0-9]*\.[0-9]*\).*/\1/'); \
	MIN="$(MIN_RUST_VERSION)"; \
	awk -v min="$$MIN" -v cur="$$RUST_VERSION" ' \
		function cmp(a,b) { \
			split(a,A,"."); split(b,B,"."); \
			for(i=1;i<=3;i++){ \
				if(A[i]+0 < B[i]+0) return -1; \
				if(A[i]+0 > B[i]+0) return 1; \
			} \
			return 0; \
		} \
		BEGIN { \
			if(cmp(cur,min) < 0) { \
				print "Error: Rust " cur " is too old. Need >= " min ". Run: rustup update"; \
				exit 1; \
			} \
		}'

clean:
	cargo clean
	$(RM) $(NAME)

.PHONY: rule openbench pgo net check-rust clean
