#ifndef FASTEMBED_H
#define FASTEMBED_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef struct TextEmbeddingHandle TextEmbeddingHandle;
typedef struct SparseTextEmbeddingHandle SparseTextEmbeddingHandle;
typedef struct ImageEmbeddingHandle ImageEmbeddingHandle;
typedef struct TextRerankHandle TextRerankHandle;

// Error handling
typedef struct {
    char* message;
} FastEmbedError;

void fastembed_error_free(FastEmbedError* error);

// Result types
typedef struct {
    float* data;
    size_t len;
} FloatArray;

typedef struct {
    FloatArray* arrays;
    size_t len;
} FloatArrayVec;

typedef struct {
    size_t* indices;
    float* values;
    size_t len;
} SparseEmbeddingC;

typedef struct {
    SparseEmbeddingC* embeddings;
    size_t len;
} SparseEmbeddingVec;

typedef struct {
    size_t index;
    float score;
    char* document;
} RerankResultC;

typedef struct {
    RerankResultC* results;
    size_t len;
} RerankResultVec;

// Text Embedding API
TextEmbeddingHandle* fastembed_text_embedding_new(
    const char* model_name,
    FastEmbedError** error
);

FloatArrayVec* fastembed_text_embedding_embed(
    TextEmbeddingHandle* handle,
    const char** texts,
    size_t num_texts,
    size_t batch_size,
    FastEmbedError** error
);

void fastembed_text_embedding_free(TextEmbeddingHandle* handle);

// Sparse Text Embedding API
SparseTextEmbeddingHandle* fastembed_sparse_text_embedding_new(
    const char* model_name,
    FastEmbedError** error
);

SparseEmbeddingVec* fastembed_sparse_text_embedding_embed(
    SparseTextEmbeddingHandle* handle,
    const char** texts,
    size_t num_texts,
    size_t batch_size,
    FastEmbedError** error
);

void fastembed_sparse_text_embedding_free(SparseTextEmbeddingHandle* handle);

// Image Embedding API
ImageEmbeddingHandle* fastembed_image_embedding_new(
    const char* model_name,
    FastEmbedError** error
);

FloatArrayVec* fastembed_image_embedding_embed(
    ImageEmbeddingHandle* handle,
    const char** image_paths,
    size_t num_images,
    size_t batch_size,
    FastEmbedError** error
);

void fastembed_image_embedding_free(ImageEmbeddingHandle* handle);

// Text Reranking API
TextRerankHandle* fastembed_text_rerank_new(
    const char* model_name,
    FastEmbedError** error
);

RerankResultVec* fastembed_text_rerank_rerank(
    TextRerankHandle* handle,
    const char* query,
    const char** documents,
    size_t num_documents,
    bool return_documents,
    size_t batch_size,
    FastEmbedError** error
);

void fastembed_text_rerank_free(TextRerankHandle* handle);

// Model Information
typedef struct {
    char* model_code;
    char* description;
    size_t dim;
} ModelInfoC;

typedef struct {
    ModelInfoC* models;
    size_t len;
} ModelInfoVec;

// Model listing functions
ModelInfoVec* fastembed_text_embedding_list_supported_models(void);
ModelInfoVec* fastembed_sparse_text_embedding_list_supported_models(void);
ModelInfoVec* fastembed_image_embedding_list_supported_models(void);
ModelInfoVec* fastembed_text_rerank_list_supported_models(void);

// Memory cleanup
void fastembed_float_array_vec_free(FloatArrayVec* vec);
void fastembed_sparse_embedding_vec_free(SparseEmbeddingVec* vec);
void fastembed_rerank_result_vec_free(RerankResultVec* vec);
void fastembed_model_info_vec_free(ModelInfoVec* vec);

#ifdef __cplusplus
}
#endif

#endif // FASTEMBED_H
