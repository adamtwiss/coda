# Coda Chess Engine — Makefile
# Supports: manual builds, OpenBench integration, PGO builds
#
# Usage:
#   make                  Build with native CPU optimizations
#   make EXE=coda-v2      Build with custom output name
#   make pgo              PGO-optimized build (slower compile, ~5% faster binary)
#   make openbench        OpenBench-compatible build target
#   make net              Download the production NNUE net
#   make clean            Remove build artifacts

# Configuration
EXE := coda
EVALFILE := net.nnue
NET_URL := $(shell cat net.txt 2>/dev/null)
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

# Default: build with native optimizations
rule: check-rust
	cargo rustc --release -- --emit link=$(NAME)

# OpenBench target
openbench: check-rust net
	cargo rustc --release -- --emit link=$(NAME)

# PGO build (profile-guided optimization)
pgo: check-rust net
	cargo pgo instrument build
	cargo pgo run -- bench
	cargo pgo optimize build
	mv target/release/coda $(NAME) 2>/dev/null || true

# Download production NNUE net
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
	@RUST_VERSION=$$(rustc --version | grep -oP '\d+\.\d+\.\d+'); \
	MIN="$(MIN_RUST_VERSION)"; \
	if [ "$$(printf '%s\n' "$$MIN" "$$RUST_VERSION" | sort -V | head -n1)" != "$$MIN" ]; then \
		echo "Error: Rust $$RUST_VERSION is too old. Need >= $$MIN. Run: rustup update"; \
		exit 1; \
	fi

clean:
	cargo clean
	$(RM) $(NAME)

.PHONY: rule openbench pgo net check-rust clean
