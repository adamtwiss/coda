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

# Default: build with embedded NNUE net (downloads from net.txt if needed)
# EVALFILE may be overridden by OpenBench with an absolute path to the network
rule: check-rust net
	CODA_EVALFILE=$(abspath $(EVALFILE)) cargo rustc --release --features embedded-net -- --emit link=$(NAME)

# Alias for OpenBench compatibility
openbench: rule

# PGO build (profile-guided optimization, ~3% NPS gain)
TARGET_TUPLE := $(shell rustc --print host-tuple 2>/dev/null)
# Find llvm-profdata matching rustc's LLVM version (needed for PGO)
RUSTC_LLVM_VER := $(shell rustc --version --verbose 2>/dev/null | awk '/LLVM version:/ {split($$3,v,"."); print v[1]}')
BREW_LLVM_MATCH := $(shell brew --prefix llvm@$(RUSTC_LLVM_VER) 2>/dev/null)
BREW_LLVM_ANY := $(shell brew --prefix llvm 2>/dev/null)
PGO_LLVM := $(or $(wildcard $(BREW_LLVM_MATCH)/bin/llvm-profdata),$(wildcard $(BREW_LLVM_ANY)/bin/llvm-profdata))
PGO_PATH := $(if $(PGO_LLVM),$(dir $(PGO_LLVM)):$(PATH),$(PATH))
pgo: check-rust net
	cargo pgo instrument build
	LLVM_PROFILE_FILE=target/pgo-profiles/coda_%m_%p.profraw ./target/$(TARGET_TUPLE)/release/coda bench 13
	PATH="$(PGO_PATH)" cargo pgo optimize build
	cp target/$(TARGET_TUPLE)/release/coda $(NAME)

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
