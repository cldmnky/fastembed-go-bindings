#!/bin/bash
set -e

echo "Building FastEmbed Go Bindings example..."

# Build the Rust library first
cd ../..
make rust-build

# Build the example
cd examples/basic
CGO_ENABLED=1 go build -v -o example main.go

echo "Example built successfully: examples/basic/example"
echo ""
echo "To run the example:"
echo "  ./examples/basic/example"
