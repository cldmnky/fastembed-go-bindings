# Architecture Overview

## Layer Stack

```
┌─────────────────────────────────────────┐
│         Go Application Layer            │
│  (Your code using fastembed package)    │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│      Go Bindings (fastembed/)           │
│  - Type-safe Go API                     │
│  - Error handling                       │
│  - Resource management                  │
│  - C-to-Go type conversion              │
└─────────────────────────────────────────┘
                   ↓ CGo
┌─────────────────────────────────────────┐
│      C Header Layer (include/)          │
│  - fastembed.h                          │
│  - C type definitions                   │
│  - Function declarations                │
└─────────────────────────────────────────┘
                   ↓ FFI
┌─────────────────────────────────────────┐
│    Rust C Library (rust/src/lib.rs)    │
│  - C FFI exports                        │
│  - Memory management                    │
│  - Error conversion                     │
│  - fastembed-rs wrapper                 │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│       fastembed-rs Library              │
│  - Text embeddings                      │
│  - Sparse embeddings                    │
│  - Image embeddings                     │
│  - Reranking                            │
│  - ONNX Runtime                         │
└─────────────────────────────────────────┘
```

## Data Flow Example (Text Embedding)

```
Go Application
    ↓
    texts := []string{"Hello", "World"}
    model.Embed(texts, 0)
    ↓
Go Bindings (fastembed.go)
    ↓
    - Convert Go []string to []*C.char
    - Call C.fastembed_text_embedding_embed()
    ↓
C FFI Boundary
    ↓
Rust Library (lib.rs)
    ↓
    - Convert *const *const c_char to Vec<String>
    - Call TextEmbedding::embed()
    ↓
fastembed-rs
    ↓
    - Tokenize text
    - Run ONNX inference
    - Return Vec<Vec<f32>>
    ↓
Rust Library
    ↓
    - Convert to FloatArrayVec (C-compatible)
    - Return pointer
    ↓
C FFI Boundary
    ↓
Go Bindings
    ↓
    - Convert FloatArrayVec to [][]float32
    - Free C memory
    - Return Go slices
    ↓
Go Application
    ↓
    embeddings := [][]float32{...}
```

## Memory Management

### Rust → Go

```
Rust allocates:
    Box::new(data)
    Box::into_raw() → C pointer

Go receives:
    C pointer
    Convert to Go types
    Copy data
    
Go frees C memory:
    C.free_function()
    (Rust: Box::from_raw())
```

### Ownership Rules

1. **Rust owns**: Model handles, internal state
2. **C transfers**: Result data (embeddings, errors)
3. **Go owns**: Converted data, application data
4. **Cleanup**: Go calls C free functions → Rust drops

## Error Handling Flow

```
Rust Error (anyhow::Error)
    ↓
Convert to C string
    CString::new(error.to_string())
    ↓
Return FastEmbedError*
    ↓
Go checks error pointer
    if cErr != nil
    ↓
Convert to Go error
    newError(cErr) → error
    ↓
Free C error
    C.fastembed_error_free()
```

## Thread Safety

```
┌─────────────┐     ┌─────────────┐
│ Goroutine 1 │     │ Goroutine 2 │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ↓                   ↓
┌──────────────┐    ┌──────────────┐
│   Model A    │    │   Model B    │
│   (Thread 1) │    │   (Thread 2) │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └────────┬──────────┘
                ↓
         ONNX Runtime
         (Thread-safe)
```

**Key Points**:
- Each model instance is independent
- Models are NOT thread-safe individually
- Use separate instances for parallelism
- Or synchronize access with mutex

## Build Process

```
1. Cargo builds Rust
   ├─→ Downloads fastembed-rs
   ├─→ Compiles to C library
   └─→ Outputs: libfastembed_c.{so,dylib,dll}

2. Makefile copies to build/
   └─→ build/libfastembed_c.{so,dylib,dll}

3. Go build uses CGo
   ├─→ Includes: include/fastembed.h
   ├─→ Links: build/libfastembed_c.*
   └─→ Creates: Go package
```

## Deployment

### Development
```
Repository
├─→ Rust source
├─→ C headers
├─→ Go source
└─→ Build locally
```

### Distribution Options

**Option 1**: Source distribution
```
User clones repo
User runs: make build
User imports Go package
```

**Option 2**: Pre-built library (future)
```
Download pre-built library
Place in build/
Import Go package
```

**Option 3**: Go module with lib (future)
```
go get package
Auto-download library
Import and use
```

## Model Loading

```
First Use:
    NewTextEmbedding("BGESmallENV15")
    ↓
    Check cache: ~/.fastembed_cache
    ↓
    If not cached:
        Download from Hugging Face
        Save to cache
    ↓
    Load ONNX model
    Load tokenizer
    ↓
    Return handle

Subsequent Uses:
    Load from cache (fast)
```

## Performance Characteristics

### Initialization
- **First time**: Slow (download + load)
- **Subsequent**: Medium (load from cache)
- **Reuse instance**: Fast (already loaded)

### Inference
- **Small batches**: ~ms per text
- **Large batches**: Better throughput
- **Batch size**: Trade-off memory vs speed

### Memory
- **Model size**: 100MB - 1GB depending on model
- **Per request**: ~KB per text
- **Batch processing**: Linear with batch size

## API Design Patterns

### Resource Acquisition
```go
model, err := NewXXX()
if err != nil {
    return err
}
defer model.Close()  // RAII pattern
```

### Error Handling
```go
result, err := model.Method()
if err != nil {
    return nil, err  // Explicit errors
}
```

### Type Safety
```go
// Go types, not C types
embeddings [][]float32
results []RerankResult
```
