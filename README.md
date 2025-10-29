# FastEmbed Go Bindings

Go bindings for [fastembed-rs](https://github.com/Anush008/fastembed-rs), a fast, lightweight library for generating text embeddings, sparse embeddings, image embeddings, and reranking.

## Features

- **Text Embeddings**: Generate dense vector embeddings for text using various models
- **Sparse Text Embeddings**: Generate sparse vector embeddings for text
- **Image Embeddings**: Generate vector embeddings for images
- **Text Reranking**: Rerank documents based on relevance to a query
- **Fast Performance**: Built on top of fastembed-rs using ONNX Runtime
- **Multiple Models**: Support for various embedding and reranking models

## Architecture

This library consists of three main components:

1. **Rust Library** (`rust/`): A C FFI wrapper around fastembed-rs
2. **C Headers** (`include/`): C header files defining the API
3. **Go Package** (`fastembed/`): CGo bindings providing a Go-friendly API

## Prerequisites

- Rust (latest stable version)
- Go 1.21 or later
- C compiler (gcc, clang, or MSVC)
- Make

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cldmnky/fastembed-go-bindings.git
cd fastembed-go-bindings
```

2. Build the project:
```bash
make build
```

This will:
- Build the Rust library
- Copy the compiled library to the `build/` directory
- Build the Go package

## Usage

### Text Embeddings

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/cldmnky/fastembed-go-bindings/fastembed"
)

func main() {
    // Create a text embedding model with default model
    model, err := fastembed.NewTextEmbedding("")
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()
    
    // Or use a specific model
    // model, err := fastembed.NewTextEmbedding("Xenova/all-MiniLM-L6-v2")
    
    // Generate embeddings
    texts := []string{
        "Hello, World!",
        "This is a test document.",
    }
    
    embeddings, err := model.Embed(texts, 0) // 0 = default batch size
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Generated %d embeddings\n", len(embeddings))
    fmt.Printf("First embedding dimension: %d\n", len(embeddings[0]))
}
```

### Model Selection

List all available models and select a specific one:

```go
// List all available text embedding models
models := fastembed.ListTextEmbeddingModels()
for _, m := range models {
    fmt.Printf("%s (dim=%d): %s\n", m.ModelCode, m.Dimension, m.Description)
}

// Use a specific model
model, err := fastembed.NewTextEmbedding("Xenova/all-MiniLM-L6-v2")
if err != nil {
    log.Fatal(err)
}
defer model.Close()
```

See [docs/MODELS.md](docs/MODELS.md) for a complete list of all supported models.

### Sparse Text Embeddings

```go
model, err := fastembed.NewSparseTextEmbedding("")
if err != nil {
    log.Fatal(err)
}
defer model.Close()

embeddings, err := model.Embed(texts, 0)
if err != nil {
    log.Fatal(err)
}

for i, emb := range embeddings {
    fmt.Printf("Embedding %d: %d non-zero values\n", i, len(emb.Indices))
}
```

### Image Embeddings

```go
model, err := fastembed.NewImageEmbedding("")
if err != nil {
    log.Fatal(err)
}
defer model.Close()

imagePaths := []string{
    "path/to/image1.jpg",
    "path/to/image2.png",
}

embeddings, err := model.Embed(imagePaths, 0)
if err != nil {
    log.Fatal(err)
}
```

### Document Reranking

```go
model, err := fastembed.NewTextRerank()
if err != nil {
    log.Fatal(err)
}
defer model.Close()

query := "What is a panda?"
documents := []string{
    "The giant panda is a bear species endemic to China.",
    "Panda is an animal.",
    "I don't know.",
}

results, err := model.Rerank(query, documents, true, 0)
if err != nil {
    log.Fatal(err)
}

for _, result := range results {
    fmt.Printf("Index: %d, Score: %.4f, Doc: %s\n", 
        result.Index, result.Score, result.Document)
}
```

## Development

### Building

Build the entire project:
```bash
make build
```

Build only the Rust library:
```bash
make rust-build
```

Build only the Go package:
```bash
make go-build
```

### Testing

Run all tests:
```bash
make test
```

Run only Go tests:
```bash
make go-test
```

### Code Quality

Format code:
```bash
make fmt
```

Lint code:
```bash
make lint
```

Run all checks:
```bash
make check
```

### Cleaning

Clean all build artifacts:
```bash
make clean
```

Clean only Rust artifacts:
```bash
make rust-clean
```

Clean only Go artifacts:
```bash
make go-clean
```

## Project Structure

```
.
├── Makefile              # Build automation
├── README.md             # This file
├── AGENTS.md             # Agent development guide
├── go.mod                # Go module file
├── rust/                 # Rust C FFI library
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs        # Rust implementation
├── include/              # C header files
│   └── fastembed.h       # C API definitions
├── fastembed/            # Go package
│   ├── fastembed.go      # Go bindings
│   └── fastembed_test.go # Go tests
├── build/                # Compiled libraries (created during build)
└── docs/                 # Additional documentation
```

## Supported Models

### Text Embedding Models
- AllMiniLML6V2
- BGESmallENV15 (default)
- BGEBaseENV15
- BGELargeENV15
- And many more from fastembed-rs

### Sparse Embedding Models
- SPLADE++

### Image Embedding Models
- CLIP ViT-B/32

### Reranking Models
- BGE Reranker Base

For a complete list of supported models, see the [fastembed-rs documentation](https://github.com/Anush008/fastembed-rs).

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a feature branch from `main`
2. Use TDD practices when implementing features
3. Run tests and code quality checks before submitting
4. Update documentation as needed

See [AGENTS.md](AGENTS.md) for more details on development practices.

## License

This project is licensed under the same license as the upstream fastembed-rs library.

## Acknowledgments

- [fastembed-rs](https://github.com/Anush008/fastembed-rs) - The underlying Rust library
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference engine
- [Hugging Face](https://huggingface.co/) - Model hosting and tokenizers