# Development Guide

## Getting Started

This guide provides detailed information for developers working on the fastembed-go-bindings project.

## Prerequisites

Ensure you have the following installed:

- **Rust**: Latest stable version (install via [rustup](https://rustup.rs/))
- **Go**: Version 1.21 or later
- **C Compiler**: 
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: GCC or Clang
  - Windows: MSVC or MinGW
- **Make**: Build automation tool

## Project Architecture

The project consists of three layers:

### 1. Rust FFI Layer (`rust/`)

The Rust library wraps fastembed-rs and exports a C-compatible API.

**Key files:**
- `Cargo.toml`: Rust dependencies and build configuration
- `src/lib.rs`: C FFI implementations

**Building:**
```bash
make rust-build
```

The compiled library will be placed in `build/`.

### 2. C Header Layer (`include/`)

C header files define the FFI interface between Rust and Go.

**Key files:**
- `fastembed.h`: C API declarations

### 3. Go Bindings Layer (`fastembed/`)

Go package that uses CGo to call the Rust library through the C interface.

**Key files:**
- `fastembed.go`: Go bindings implementation
- `fastembed_test.go`: Unit tests

## Building

### Full Build

Build everything:
```bash
make build
```

This will:
1. Build the Rust library
2. Copy artifacts to `build/`
3. Build the Go package

### Incremental Builds

Build only Rust:
```bash
make rust-build
```

Build only Go (requires Rust library):
```bash
make go-build
```

## Testing

### Running Tests

Run all tests:
```bash
make test
```

Run only Go tests:
```bash
make go-test
```

### Writing Tests

Tests are located in `fastembed/fastembed_test.go`. Follow TDD principles:

1. Write a failing test
2. Implement the feature
3. Make the test pass
4. Refactor

Example test structure:
```go
func TestFeature(t *testing.T) {
    // Arrange
    model, err := NewTextEmbedding("BGESmallENV15")
    if err != nil {
        t.Fatalf("Setup failed: %v", err)
    }
    defer model.Close()
    
    // Act
    result, err := model.SomeMethod()
    
    // Assert
    if err != nil {
        t.Fatalf("Method failed: %v", err)
    }
    if result != expected {
        t.Errorf("Expected %v, got %v", expected, result)
    }
}
```

## Code Quality

### Formatting

Format Rust code:
```bash
cd rust && cargo fmt
```

Format Go code:
```bash
cd fastembed && go fmt ./...
```

Format all code:
```bash
make fmt
```

### Linting

Lint Rust code:
```bash
cd rust && cargo clippy -- -D warnings
```

Lint Go code:
```bash
cd fastembed && go vet ./...
```

Lint all code:
```bash
make lint
```

### Pre-commit Checks

Run before committing:
```bash
make check
```

## Debugging

### Debugging Rust Code

1. Enable debug logging in Rust:
```rust
#[cfg(debug_assertions)]
eprintln!("Debug: {:?}", value);
```

2. Build with debug symbols:
```bash
cd rust && cargo build
```

### Debugging Go Code

1. Use Go's built-in debugger (Delve):
```bash
dlv test ./fastembed
```

2. Add debug prints:
```go
log.Printf("Debug: %+v\n", value)
```

### Debugging CGo Interface

1. Check CGo compilation:
```bash
CGO_ENABLED=1 go build -x -v ./fastembed
```

2. Verify library loading:
```bash
otool -L build/libfastembed_c.dylib  # macOS
ldd build/libfastembed_c.so          # Linux
```

## Memory Management

### Rust Side

- All exported functions use `Box::into_raw()` to transfer ownership to C
- Free functions use `Box::from_raw()` to reclaim ownership and drop

### Go Side

- Use `defer` to ensure cleanup:
```go
model, err := NewTextEmbedding("model")
if err != nil {
    return err
}
defer model.Close()
```

- Finalizers are set but should not be relied upon
- Always explicitly call `Close()`

## Adding New Features

Follow this workflow:

1. **Update Rust FFI** (`rust/src/lib.rs`):
   - Add new C-compatible function
   - Handle errors properly
   - Manage memory correctly

2. **Update C Header** (`include/fastembed.h`):
   - Add function declaration
   - Document parameters and return values

3. **Update Go Bindings** (`fastembed/fastembed.go`):
   - Add Go wrapper function
   - Convert between C and Go types
   - Handle errors

4. **Add Tests** (`fastembed/fastembed_test.go`):
   - Test success cases
   - Test error cases
   - Test edge cases

5. **Update Documentation**:
   - Update `README.md` with usage examples
   - Update `docs/API.md` with API details

## Branching Strategy

- `main`: Stable code
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes following TDD
3. Run `make check` to verify code quality
4. Run `make test` to verify all tests pass
5. Update documentation
6. Create a pull request

## Common Issues

### Library Not Found

**Error**: `library not found for -lfastembed_c`

**Solution**:
```bash
make rust-build
```

### CGo Compilation Fails

**Error**: Various CGo errors

**Solution**:
1. Ensure C compiler is installed
2. Check that `build/` contains the library
3. Verify `CGO_ENABLED=1` is set

### Model Download Fails

**Error**: Model download timeout or failure

**Solution**:
1. Check internet connection
2. Models are cached in `~/.fastembed_cache`
3. May need to manually download models

## Performance Tips

1. **Batch Processing**: Use appropriate batch sizes for better throughput
2. **Model Caching**: Models are loaded once and reused
3. **Parallel Processing**: Create multiple model instances for parallelism
4. **Memory**: Monitor memory usage with large batch sizes

## Resources

- [fastembed-rs](https://github.com/Anush008/fastembed-rs): Upstream Rust library
- [CGo Documentation](https://pkg.go.dev/cmd/cgo): Go's C interop
- [The Rustonomicon](https://doc.rust-lang.org/nomicon/): Unsafe Rust and FFI
