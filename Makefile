.PHONY: all build clean test rust-build rust-clean go-build go-test go-clean help

# Directories
RUST_DIR := rust
BUILD_DIR := build
INCLUDE_DIR := include
GO_PKG := fastembed

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIB_EXT := dylib
	LIB_PREFIX := lib
else ifeq ($(UNAME_S),Linux)
	LIB_EXT := so
	LIB_PREFIX := lib
else
	LIB_EXT := dll
	LIB_PREFIX :=
endif

# Library names
RUST_LIB := $(LIB_PREFIX)fastembed_c.$(LIB_EXT)
STATIC_LIB := $(LIB_PREFIX)fastembed_c.a
TARGET_LIB := $(BUILD_DIR)/$(RUST_LIB)
TARGET_STATIC := $(BUILD_DIR)/$(STATIC_LIB)

# Default target
all: build

## help: Display this help message
help:
	@echo "FastEmbed Go Bindings - Makefile Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk '/^##/ { \
		helpMessage = substr($$0, 4); \
		getline; \
		target = $$1; \
		gsub(/:/, "", target); \
		printf "  %-20s %s\n", target, helpMessage; \
	}' $(MAKEFILE_LIST)

## build: Build both Rust library and Go bindings
build: rust-build go-build

## rust-build: Build the Rust C library
rust-build:
	@echo "Building Rust library..."
	@if [ -d "/opt/homebrew/opt/libiconv" ]; then \
		export LIBRARY_PATH="/opt/homebrew/opt/libiconv/lib:$$LIBRARY_PATH"; \
	fi; \
	cd $(RUST_DIR) && cargo build --release
	@mkdir -p $(BUILD_DIR)
	@cp $(RUST_DIR)/target/release/$(RUST_LIB) $(TARGET_LIB) || true
	@cp $(RUST_DIR)/target/release/$(STATIC_LIB) $(TARGET_STATIC) || true
	@echo "Rust library built successfully"

## rust-clean: Clean Rust build artifacts
rust-clean:
	@echo "Cleaning Rust artifacts..."
	@cd $(RUST_DIR) && cargo clean
	@rm -f $(TARGET_LIB) $(TARGET_STATIC)
	@echo "Rust artifacts cleaned"

## go-build: Build Go package
go-build: rust-build
	@echo "Building Go package..."
	@cd $(GO_PKG) && CGO_ENABLED=1 go build -v
	@echo "Go package built successfully"

## go-test: Run Go tests
go-test: rust-build
	@echo "Running Go tests..."
	@cd $(GO_PKG) && CGO_ENABLED=1 go test -v

## go-clean: Clean Go build artifacts
go-clean:
	@echo "Cleaning Go artifacts..."
	@cd $(GO_PKG) && go clean
	@rm -rf $(GO_PKG)/*.test
	@echo "Go artifacts cleaned"

## test: Run all tests
test: go-test

## clean: Clean all build artifacts
clean: rust-clean go-clean
	@echo "Removing build directory..."
	@rm -rf $(BUILD_DIR)
	@echo "All artifacts cleaned"

## fmt: Format Rust and Go code
fmt:
	@echo "Formatting Rust code..."
	@cd $(RUST_DIR) && cargo fmt
	@echo "Formatting Go code..."
	@cd $(GO_PKG) && go fmt ./...

## lint: Lint Rust and Go code
lint:
	@echo "Linting Rust code..."
	@cd $(RUST_DIR) && cargo clippy -- -D warnings
	@echo "Linting Go code..."
	@cd $(GO_PKG) && go vet ./...

## check: Run format and lint checks
check: fmt lint

## install-deps: Install required dependencies
install-deps:
	@echo "Installing Rust dependencies..."
	@cd $(RUST_DIR) && cargo fetch
	@echo "Installing Go dependencies..."
	@go mod download
	@echo "Dependencies installed"

## example: Build and run example (placeholder)
example: build
	@echo "Example target - implement your example here"

.DEFAULT_GOAL := help
