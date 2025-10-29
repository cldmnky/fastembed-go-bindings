# FastEmbed Go Bindings - API Documentation

## Overview

This document provides detailed API documentation for the Go bindings.

## Text Embeddings

### TextEmbedding

Text embedding model for generating dense vector embeddings from text.

#### Constructor

```go
func NewTextEmbedding(modelName string) (*TextEmbedding, error)
```

Creates a new text embedding model instance.

**Parameters:**
- `modelName`: Name of the model to use (e.g., "BGESmallENV15", "AllMiniLML6V2")

**Returns:**
- `*TextEmbedding`: A new text embedding instance
- `error`: Error if model initialization fails

#### Methods

##### Embed

```go
func (te *TextEmbedding) Embed(texts []string, batchSize int) ([][]float32, error)
```

Generates embeddings for the given texts.

**Parameters:**
- `texts`: Slice of strings to embed
- `batchSize`: Batch size for processing (0 for default)

**Returns:**
- `[][]float32`: Slice of embeddings, one per input text
- `error`: Error if embedding generation fails

##### Close

```go
func (te *TextEmbedding) Close()
```

Releases resources associated with the model. Should be called when done using the model.

## Sparse Text Embeddings

### SparseTextEmbedding

Sparse text embedding model for generating sparse vector embeddings.

#### Constructor

```go
func NewSparseTextEmbedding() (*SparseTextEmbedding, error)
```

Creates a new sparse text embedding model instance.

**Returns:**
- `*SparseTextEmbedding`: A new sparse text embedding instance
- `error`: Error if model initialization fails

#### Methods

##### Embed

```go
func (ste *SparseTextEmbedding) Embed(texts []string, batchSize int) ([]SparseEmbedding, error)
```

Generates sparse embeddings for the given texts.

**Parameters:**
- `texts`: Slice of strings to embed
- `batchSize`: Batch size for processing (0 for default)

**Returns:**
- `[]SparseEmbedding`: Slice of sparse embeddings
- `error`: Error if embedding generation fails

##### Close

```go
func (ste *SparseTextEmbedding) Close()
```

Releases resources associated with the model.

### SparseEmbedding

Structure representing a sparse embedding result.

**Fields:**
- `Indices []int`: Indices of non-zero values
- `Values []float32`: Non-zero values

## Image Embeddings

### ImageEmbedding

Image embedding model for generating vector embeddings from images.

#### Constructor

```go
func NewImageEmbedding() (*ImageEmbedding, error)
```

Creates a new image embedding model instance.

**Returns:**
- `*ImageEmbedding`: A new image embedding instance
- `error`: Error if model initialization fails

#### Methods

##### Embed

```go
func (ie *ImageEmbedding) Embed(imagePaths []string, batchSize int) ([][]float32, error)
```

Generates embeddings for images at the given paths.

**Parameters:**
- `imagePaths`: Slice of file paths to images
- `batchSize`: Batch size for processing (0 for default)

**Returns:**
- `[][]float32`: Slice of embeddings, one per image
- `error`: Error if embedding generation fails

##### Close

```go
func (ie *ImageEmbedding) Close()
```

Releases resources associated with the model.

## Text Reranking

### TextRerank

Text reranking model for scoring document relevance to a query.

#### Constructor

```go
func NewTextRerank() (*TextRerank, error)
```

Creates a new text reranking model instance.

**Returns:**
- `*TextRerank`: A new text rerank instance
- `error`: Error if model initialization fails

#### Methods

##### Rerank

```go
func (tr *TextRerank) Rerank(query string, documents []string, returnDocuments bool, batchSize int) ([]RerankResult, error)
```

Reranks documents based on relevance to the query.

**Parameters:**
- `query`: Query string
- `documents`: Slice of documents to rerank
- `returnDocuments`: Whether to include document text in results
- `batchSize`: Batch size for processing (0 for default)

**Returns:**
- `[]RerankResult`: Slice of rerank results, sorted by score (descending)
- `error`: Error if reranking fails

##### Close

```go
func (tr *TextRerank) Close()
```

Releases resources associated with the model.

### RerankResult

Structure representing a reranking result.

**Fields:**
- `Index int`: Original index of the document
- `Score float32`: Relevance score
- `Document string`: Document text (if returnDocuments was true)

## Error Handling

All functions that can fail return an `error` as their last return value. Errors are wrapped in a custom `Error` type that implements the standard Go `error` interface.

### Error

Custom error type for FastEmbed errors.

**Methods:**

```go
func (e *Error) Error() string
```

Returns the error message.

```go
func (e *Error) String() string
```

Returns a formatted error string with "FastEmbed error:" prefix.

## Resource Management

All model types implement a `Close()` method that should be called when done using the model. The bindings also set up finalizers to automatically clean up resources, but it's best practice to explicitly call `Close()` using defer:

```go
model, err := fastembed.NewTextEmbedding("BGESmallENV15")
if err != nil {
    log.Fatal(err)
}
defer model.Close()
```

## Thread Safety

The Go bindings are **not thread-safe**. If you need to use a model from multiple goroutines, you should either:

1. Create separate model instances per goroutine
2. Use synchronization primitives (e.g., `sync.Mutex`) to serialize access

## Performance Considerations

- **Batch Size**: Use larger batch sizes for better throughput when processing many items
- **Model Loading**: Model initialization can be slow as it downloads and loads model files
- **Memory**: Models are loaded into memory; ensure sufficient RAM for your chosen models
- **Cache**: Models are cached in `~/.fastembed_cache` by default

## Examples

See the main [README.md](../README.md) for usage examples.
