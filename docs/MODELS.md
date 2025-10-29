# Supported Models

This document lists all models supported by fastembed-go-bindings.

## Text Embedding Models (31 models)

The default model for text embeddings is `BAAI/bge-small-en-v1.5`.

### Usage

```go
// Use default model
model, err := fastembed.NewTextEmbedding("")

// Use specific model
model, err := fastembed.NewTextEmbedding("Xenova/all-MiniLM-L6-v2")

// List all available models
models := fastembed.ListTextEmbeddingModels()
for _, m := range models {
    fmt.Printf("%s - %s (dim=%d)\n", m.ModelCode, m.Description, m.Dimension)
}
```

### Popular Models

- **BAAI/bge-small-en-v1.5** (dim=384) - Fast and default English model
- **Xenova/all-MiniLM-L6-v2** (dim=384) - Lightweight sentence transformer
- **mixedbread-ai/mxbai-embed-large-v1** (dim=1024) - Large English embedding
- **nomic-ai/nomic-embed-text-v1.5** (dim=768) - 8192 context length
- **intfloat/multilingual-e5-small** (dim=384) - Multilingual model

## Sparse Text Embedding Models (1 model)

The default model for sparse embeddings is `Qdrant/Splade_PP_en_v1`.

### Usage

```go
// Use default model
model, err := fastembed.NewSparseTextEmbedding("")

// Use specific model (same as default currently)
model, err := fastembed.NewSparseTextEmbedding("Qdrant/Splade_PP_en_v1")

// List all available models
models := fastembed.ListSparseTextEmbeddingModels()
```

### Available Models

- **Qdrant/Splade_PP_en_v1** - Splade sparse vector model for commercial use

## Image Embedding Models (5 models)

The default model for image embeddings is `Qdrant/clip-ViT-B-32-vision`.

### Usage

```go
// Use default model
model, err := fastembed.NewImageEmbedding("")

// Use specific model
model, err := fastembed.NewImageEmbedding("nomic-ai/nomic-embed-vision-v1.5")

// List all available models
models := fastembed.ListImageEmbeddingModels()
```

### Available Models

- **Qdrant/clip-ViT-B-32-vision** (dim=512) - CLIP vision encoder
- **Qdrant/resnet50-onnx** (dim=2048) - ResNet-50 model
- **Qdrant/Unicom-ViT-B-16** (dim=768) - Unicom vision model
- **Qdrant/Unicom-ViT-B-32** (dim=512) - Unicom vision model
- **nomic-ai/nomic-embed-vision-v1.5** (dim=768) - Nomic vision embeddings

## Text Reranking Models (4 models)

The default model for reranking is `BAAI/bge-reranker-base`.

### Usage

```go
// Use default model
model, err := fastembed.NewTextRerank("")

// Use specific model
model, err := fastembed.NewTextRerank("jinaai/jina-reranker-v1-turbo-en")

// List all available models
models := fastembed.ListTextRerankModels()
```

### Available Models

- **BAAI/bge-reranker-base** - Reranker for English and Chinese
- **rozgo/bge-reranker-v2-m3** - Multilingual reranker v2
- **jinaai/jina-reranker-v1-turbo-en** - Fast English reranker
- **jinaai/jina-reranker-v2-base-multilingual** - Multilingual reranker

## Programmatic Access

You can list all models programmatically:

```go
package main

import (
    "fmt"
    "github.com/cldmnky/fastembed-go-bindings/fastembed"
)

func main() {
    // Text embedding models
    fmt.Println("Text Embedding Models:")
    for _, model := range fastembed.ListTextEmbeddingModels() {
        fmt.Printf("  %s (dim=%d)\n", model.ModelCode, model.Dimension)
    }

    // Sparse embedding models
    fmt.Println("\nSparse Embedding Models:")
    for _, model := range fastembed.ListSparseTextEmbeddingModels() {
        fmt.Printf("  %s\n", model.ModelCode)
    }

    // Image embedding models
    fmt.Println("\nImage Embedding Models:")
    for _, model := range fastembed.ListImageEmbeddingModels() {
        fmt.Printf("  %s (dim=%d)\n", model.ModelCode, model.Dimension)
    }

    // Reranking models
    fmt.Println("\nReranking Models:")
    for _, model := range fastembed.ListTextRerankModels() {
        fmt.Printf("  %s\n", model.ModelCode)
    }
}
```

## Model Selection Guidelines

### When to use which model:

**Text Embeddings:**
- **Small/Fast**: `BAAI/bge-small-en-v1.5`, `Xenova/all-MiniLM-L6-v2`
- **Large/Accurate**: `mixedbread-ai/mxbai-embed-large-v1`, `BAAI/bge-large-en-v1.5`
- **Multilingual**: `intfloat/multilingual-e5-base`, `Xenova/paraphrase-multilingual-mpnet-base-v2`
- **Long Context**: `nomic-ai/nomic-embed-text-v1.5` (8192 tokens)

**Sparse Embeddings:**
- Use `Qdrant/Splade_PP_en_v1` for information retrieval tasks

**Image Embeddings:**
- **General Purpose**: `Qdrant/clip-ViT-B-32-vision`
- **High Dimension**: `Qdrant/resnet50-onnx` (2048 dim)
- **Multimodal**: `nomic-ai/nomic-embed-vision-v1.5` (pairs with text model)

**Reranking:**
- **English**: `jinaai/jina-reranker-v1-turbo-en`
- **Multilingual**: `jinaai/jina-reranker-v2-base-multilingual`
- **General**: `BAAI/bge-reranker-base`

## Notes

- Empty string `""` as model name uses the default model for each type
- All models are downloaded from HuggingFace on first use and cached locally
- Model dimensions vary - check `Dimension` field in `ModelInfo`
- Reranking models don't have dimensions as they output relevance scores
