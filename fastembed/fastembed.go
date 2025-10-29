package fastembed

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo LDFLAGS: -L${SRCDIR}/../build -lfastembed_c
#include "fastembed.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Error represents a FastEmbed error
type Error struct {
	message string
}

func (e *Error) Error() string {
	return e.message
}

// newError creates a new Error from a C error pointer
func newError(cErr *C.FastEmbedError) error {
	if cErr == nil {
		return nil
	}
	defer C.fastembed_error_free(cErr)
	return &Error{message: C.GoString(cErr.message)}
}

// TextEmbedding represents a text embedding model
type TextEmbedding struct {
	handle *C.TextEmbeddingHandle
}

// NewTextEmbedding creates a new text embedding model instance
func NewTextEmbedding(modelName string) (*TextEmbedding, error) {
	var cErr *C.FastEmbedError
	var cModelName *C.char
	if modelName != "" {
		cModelName = C.CString(modelName)
		defer C.free(unsafe.Pointer(cModelName))
	}

	handle := C.fastembed_text_embedding_new(cModelName, &cErr)
	if handle == nil {
		return nil, newError(cErr)
	}

	te := &TextEmbedding{handle: handle}
	runtime.SetFinalizer(te, func(t *TextEmbedding) {
		t.Close()
	})
	return te, nil
}

// Embed generates embeddings for the given texts
func (te *TextEmbedding) Embed(texts []string, batchSize int) ([][]float32, error) {
	if te.handle == nil {
		return nil, &Error{message: "TextEmbedding handle is nil"}
	}

	// Convert Go strings to C strings
	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}

	var cErr *C.FastEmbedError
	result := C.fastembed_text_embedding_embed(
		te.handle,
		(**C.char)(unsafe.Pointer(&cTexts[0])),
		C.size_t(len(texts)),
		C.size_t(batchSize),
		&cErr,
	)
	if result == nil {
		return nil, newError(cErr)
	}
	defer C.fastembed_float_array_vec_free(result)

	// Convert C result to Go slices
	embeddings := make([][]float32, int(result.len))
	arrays := (*[1 << 30]C.FloatArray)(unsafe.Pointer(result.arrays))[:result.len:result.len]

	for i, array := range arrays {
		embedding := make([]float32, int(array.len))
		data := (*[1 << 30]C.float)(unsafe.Pointer(array.data))[:array.len:array.len]
		for j, v := range data {
			embedding[j] = float32(v)
		}
		embeddings[i] = embedding
	}

	return embeddings, nil
}

// Close releases the resources associated with the text embedding model
func (te *TextEmbedding) Close() {
	if te.handle != nil {
		C.fastembed_text_embedding_free(te.handle)
		te.handle = nil
	}
}

// SparseEmbedding represents a sparse embedding result
type SparseEmbedding struct {
	Indices []int
	Values  []float32
}

// SparseTextEmbedding represents a sparse text embedding model
type SparseTextEmbedding struct {
	handle *C.SparseTextEmbeddingHandle
}

// NewSparseTextEmbedding creates a new sparse text embedding model instance
func NewSparseTextEmbedding(modelName string) (*SparseTextEmbedding, error) {
	var cErr *C.FastEmbedError
	var cModelName *C.char
	if modelName != "" {
		cModelName = C.CString(modelName)
		defer C.free(unsafe.Pointer(cModelName))
	}

	handle := C.fastembed_sparse_text_embedding_new(cModelName, &cErr)
	if handle == nil {
		return nil, newError(cErr)
	}

	ste := &SparseTextEmbedding{handle: handle}
	runtime.SetFinalizer(ste, func(s *SparseTextEmbedding) {
		s.Close()
	})
	return ste, nil
}

// Embed generates sparse embeddings for the given texts
func (ste *SparseTextEmbedding) Embed(texts []string, batchSize int) ([]SparseEmbedding, error) {
	if ste.handle == nil {
		return nil, &Error{message: "SparseTextEmbedding handle is nil"}
	}

	// Convert Go strings to C strings
	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}

	var cErr *C.FastEmbedError
	result := C.fastembed_sparse_text_embedding_embed(
		ste.handle,
		(**C.char)(unsafe.Pointer(&cTexts[0])),
		C.size_t(len(texts)),
		C.size_t(batchSize),
		&cErr,
	)
	if result == nil {
		return nil, newError(cErr)
	}
	defer C.fastembed_sparse_embedding_vec_free(result)

	// Convert C result to Go slices
	embeddings := make([]SparseEmbedding, int(result.len))
	cEmbeddings := (*[1 << 30]C.SparseEmbeddingC)(unsafe.Pointer(result.embeddings))[:result.len:result.len]

	for i, cEmb := range cEmbeddings {
		indices := make([]int, int(cEmb.len))
		values := make([]float32, int(cEmb.len))

		cIndices := (*[1 << 30]C.size_t)(unsafe.Pointer(cEmb.indices))[:cEmb.len:cEmb.len]
		cValues := (*[1 << 30]C.float)(unsafe.Pointer(cEmb.values))[:cEmb.len:cEmb.len]

		for j := range indices {
			indices[j] = int(cIndices[j])
			values[j] = float32(cValues[j])
		}

		embeddings[i] = SparseEmbedding{
			Indices: indices,
			Values:  values,
		}
	}

	return embeddings, nil
}

// Close releases the resources associated with the sparse text embedding model
func (ste *SparseTextEmbedding) Close() {
	if ste.handle != nil {
		C.fastembed_sparse_text_embedding_free(ste.handle)
		ste.handle = nil
	}
}

// ImageEmbedding represents an image embedding model
type ImageEmbedding struct {
	handle *C.ImageEmbeddingHandle
}

// NewImageEmbedding creates a new image embedding model instance
func NewImageEmbedding(modelName string) (*ImageEmbedding, error) {
	var cErr *C.FastEmbedError
	var cModelName *C.char
	if modelName != "" {
		cModelName = C.CString(modelName)
		defer C.free(unsafe.Pointer(cModelName))
	}

	handle := C.fastembed_image_embedding_new(cModelName, &cErr)
	if handle == nil {
		return nil, newError(cErr)
	}

	ie := &ImageEmbedding{handle: handle}
	runtime.SetFinalizer(ie, func(i *ImageEmbedding) {
		i.Close()
	})
	return ie, nil
}

// Embed generates embeddings for the given image paths
func (ie *ImageEmbedding) Embed(imagePaths []string, batchSize int) ([][]float32, error) {
	if ie.handle == nil {
		return nil, &Error{message: "ImageEmbedding handle is nil"}
	}

	// Convert Go strings to C strings
	cPaths := make([]*C.char, len(imagePaths))
	for i, path := range imagePaths {
		cPaths[i] = C.CString(path)
		defer C.free(unsafe.Pointer(cPaths[i]))
	}

	var cErr *C.FastEmbedError
	result := C.fastembed_image_embedding_embed(
		ie.handle,
		(**C.char)(unsafe.Pointer(&cPaths[0])),
		C.size_t(len(imagePaths)),
		C.size_t(batchSize),
		&cErr,
	)
	if result == nil {
		return nil, newError(cErr)
	}
	defer C.fastembed_float_array_vec_free(result)

	// Convert C result to Go slices
	embeddings := make([][]float32, int(result.len))
	arrays := (*[1 << 30]C.FloatArray)(unsafe.Pointer(result.arrays))[:result.len:result.len]

	for i, array := range arrays {
		embedding := make([]float32, int(array.len))
		data := (*[1 << 30]C.float)(unsafe.Pointer(array.data))[:array.len:array.len]
		for j, v := range data {
			embedding[j] = float32(v)
		}
		embeddings[i] = embedding
	}

	return embeddings, nil
}

// Close releases the resources associated with the image embedding model
func (ie *ImageEmbedding) Close() {
	if ie.handle != nil {
		C.fastembed_image_embedding_free(ie.handle)
		ie.handle = nil
	}
}

// RerankResult represents a reranking result
type RerankResult struct {
	Index    int
	Score    float32
	Document string
}

// TextRerank represents a text reranking model
type TextRerank struct {
	handle *C.TextRerankHandle
}

// NewTextRerank creates a new text reranking model instance
func NewTextRerank(modelName string) (*TextRerank, error) {
	var cErr *C.FastEmbedError
	var cModelName *C.char
	if modelName != "" {
		cModelName = C.CString(modelName)
		defer C.free(unsafe.Pointer(cModelName))
	}

	handle := C.fastembed_text_rerank_new(cModelName, &cErr)
	if handle == nil {
		return nil, newError(cErr)
	}

	tr := &TextRerank{handle: handle}
	runtime.SetFinalizer(tr, func(t *TextRerank) {
		t.Close()
	})
	return tr, nil
}

// Rerank reranks documents based on their relevance to the query
func (tr *TextRerank) Rerank(query string, documents []string, returnDocuments bool, batchSize int) ([]RerankResult, error) {
	if tr.handle == nil {
		return nil, &Error{message: "TextRerank handle is nil"}
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	// Convert Go strings to C strings
	cDocs := make([]*C.char, len(documents))
	for i, doc := range documents {
		cDocs[i] = C.CString(doc)
		defer C.free(unsafe.Pointer(cDocs[i]))
	}

	var cErr *C.FastEmbedError
	result := C.fastembed_text_rerank_rerank(
		tr.handle,
		cQuery,
		(**C.char)(unsafe.Pointer(&cDocs[0])),
		C.size_t(len(documents)),
		C.bool(returnDocuments),
		C.size_t(batchSize),
		&cErr,
	)
	if result == nil {
		return nil, newError(cErr)
	}
	defer C.fastembed_rerank_result_vec_free(result)

	// Convert C result to Go slices
	results := make([]RerankResult, int(result.len))
	cResults := (*[1 << 30]C.RerankResultC)(unsafe.Pointer(result.results))[:result.len:result.len]

	for i, cResult := range cResults {
		results[i] = RerankResult{
			Index: int(cResult.index),
			Score: float32(cResult.score),
		}
		if cResult.document != nil {
			results[i].Document = C.GoString(cResult.document)
		}
	}

	return results, nil
}

// Close releases the resources associated with the text reranking model
func (tr *TextRerank) Close() {
	if tr.handle != nil {
		C.fastembed_text_rerank_free(tr.handle)
		tr.handle = nil
	}
}

// String returns a string representation of the error
func (e *Error) String() string {
	return fmt.Sprintf("FastEmbed error: %s", e.message)
}

// ModelInfo represents information about a supported model
type ModelInfo struct {
	ModelCode   string
	Description string
	Dimension   int
}

// ListTextEmbeddingModels returns a list of all supported text embedding models
func ListTextEmbeddingModels() []ModelInfo {
	cVec := C.fastembed_text_embedding_list_supported_models()
	if cVec == nil {
		return nil
	}
	defer C.fastembed_model_info_vec_free(cVec)

	models := (*[1 << 30]C.ModelInfoC)(unsafe.Pointer(cVec.models))[:cVec.len:cVec.len]
	result := make([]ModelInfo, cVec.len)

	for i, model := range models {
		result[i] = ModelInfo{
			ModelCode:   C.GoString(model.model_code),
			Description: C.GoString(model.description),
			Dimension:   int(model.dim),
		}
	}

	return result
}

// ListSparseTextEmbeddingModels returns a list of all supported sparse text embedding models
func ListSparseTextEmbeddingModels() []ModelInfo {
	cVec := C.fastembed_sparse_text_embedding_list_supported_models()
	if cVec == nil {
		return nil
	}
	defer C.fastembed_model_info_vec_free(cVec)

	models := (*[1 << 30]C.ModelInfoC)(unsafe.Pointer(cVec.models))[:cVec.len:cVec.len]
	result := make([]ModelInfo, cVec.len)

	for i, model := range models {
		result[i] = ModelInfo{
			ModelCode:   C.GoString(model.model_code),
			Description: C.GoString(model.description),
			Dimension:   int(model.dim),
		}
	}

	return result
}

// ListImageEmbeddingModels returns a list of all supported image embedding models
func ListImageEmbeddingModels() []ModelInfo {
	cVec := C.fastembed_image_embedding_list_supported_models()
	if cVec == nil {
		return nil
	}
	defer C.fastembed_model_info_vec_free(cVec)

	models := (*[1 << 30]C.ModelInfoC)(unsafe.Pointer(cVec.models))[:cVec.len:cVec.len]
	result := make([]ModelInfo, cVec.len)

	for i, model := range models {
		result[i] = ModelInfo{
			ModelCode:   C.GoString(model.model_code),
			Description: C.GoString(model.description),
			Dimension:   int(model.dim),
		}
	}

	return result
}

// ListTextRerankModels returns a list of all supported text reranking models
func ListTextRerankModels() []ModelInfo {
	cVec := C.fastembed_text_rerank_list_supported_models()
	if cVec == nil {
		return nil
	}
	defer C.fastembed_model_info_vec_free(cVec)

	models := (*[1 << 30]C.ModelInfoC)(unsafe.Pointer(cVec.models))[:cVec.len:cVec.len]
	result := make([]ModelInfo, cVec.len)

	for i, model := range models {
		result[i] = ModelInfo{
			ModelCode:   C.GoString(model.model_code),
			Description: C.GoString(model.description),
			Dimension:   int(model.dim),
		}
	}

	return result
}
