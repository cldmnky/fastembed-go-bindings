# Quick Start Guide

Get up and running with FastEmbed Go Bindings in 5 minutes!

## Step 1: Install Prerequisites

### macOS
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install Go (if not already installed)
brew install go
```

### Linux (Ubuntu/Debian)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install build essentials
sudo apt-get update
sudo apt-get install build-essential

# Install Go (if not already installed)
sudo apt-get install golang-go
```

## Step 2: Clone and Build

```bash
# Clone the repository
git clone https://github.com/cldmnky/fastembed-go-bindings.git
cd fastembed-go-bindings

# Build everything
make build
```

This will:
- Download dependencies
- Build the Rust library
- Build the Go package

## Step 3: Run the Example

```bash
# Build the example
cd examples/basic
./build.sh

# Run it
./example
```

## Step 4: Use in Your Project

### Add to your Go module

```bash
# In your Go project
go get github.com/cldmnky/fastembed-go-bindings/fastembed
```

### Simple Example

Create `main.go`:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/cldmnky/fastembed-go-bindings/fastembed"
)

func main() {
    // Create model
    model, err := fastembed.NewTextEmbedding("BGESmallENV15")
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()
    
    // Generate embeddings
    texts := []string{"Hello, World!", "How are you?"}
    embeddings, err := model.Embed(texts, 0)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Generated %d embeddings\n", len(embeddings))
    fmt.Printf("Embedding dimension: %d\n", len(embeddings[0]))
}
```

### Build and Run

```bash
# Make sure the library is built
cd /path/to/fastembed-go-bindings
make build

# In your project
CGO_ENABLED=1 go run main.go
```

## Common Commands

```bash
# Build
make build

# Run tests
make test

# Clean
make clean

# Format code
make fmt

# See all options
make help
```

## Troubleshooting

### "library not found"

Make sure you've built the Rust library:
```bash
make rust-build
```

### "cannot find package"

Ensure your Go module is set up:
```bash
go mod init your-project-name
go mod tidy
```

### Model download slow/fails

Models are downloaded on first use. They're cached in `~/.fastembed_cache/`.
Subsequent uses will be faster.

## Next Steps

- Read the [API Documentation](docs/API.md)
- Check out more [Examples](examples/)
- Read the [Development Guide](docs/DEVELOPMENT.md)
- See the full [README](README.md)

## Getting Help

- Check the [Documentation](docs/)
- See [fastembed-rs docs](https://github.com/Anush008/fastembed-rs)
- Open an issue on GitHub

Happy embedding! 🚀
