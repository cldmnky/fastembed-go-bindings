package fastembed

import (
	"testing"
)

// TestTextEmbedding_New tests creating a new text embedding model
func TestTextEmbedding_New(t *testing.T) {

	te, err := NewTextEmbedding("BGESmallENV15")
	if err != nil {
		t.Fatalf("Failed to create text embedding: %v", err)
	}
	defer te.Close()

	if te.handle == nil {
		t.Error("Expected non-nil handle")
	}
}

// TestTextEmbedding_Embed tests embedding text
func TestTextEmbedding_Embed(t *testing.T) {

	te, err := NewTextEmbedding("BGESmallENV15")
	if err != nil {
		t.Fatalf("Failed to create text embedding: %v", err)
	}
	defer te.Close()

	texts := []string{
		"Hello, World!",
		"This is a test.",
	}

	embeddings, err := te.Embed(texts, 0)
	if err != nil {
		t.Fatalf("Failed to embed texts: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	for i, emb := range embeddings {
		if len(emb) == 0 {
			t.Errorf("Embedding %d is empty", i)
		}
	}
}

// TestSparseTextEmbedding_New tests creating a new sparse text embedding model
func TestSparseTextEmbedding_New(t *testing.T) {

	ste, err := NewSparseTextEmbedding("")
	if err != nil {
		t.Fatalf("Failed to create sparse text embedding: %v", err)
	}
	defer ste.Close()

	if ste.handle == nil {
		t.Error("Expected non-nil handle")
	}
}

// TestSparseTextEmbedding_Embed tests embedding text with sparse model
func TestSparseTextEmbedding_Embed(t *testing.T) {

	ste, err := NewSparseTextEmbedding("")
	if err != nil {
		t.Fatalf("Failed to create sparse text embedding: %v", err)
	}
	defer ste.Close()

	texts := []string{
		"Hello, World!",
		"This is a test.",
	}

	embeddings, err := ste.Embed(texts, 0)
	if err != nil {
		t.Fatalf("Failed to embed texts: %v", err)
	}

	if len(embeddings) != len(texts) {
		t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
	}

	for i, emb := range embeddings {
		if len(emb.Indices) == 0 {
			t.Errorf("Embedding %d has no indices", i)
		}
		if len(emb.Values) == 0 {
			t.Errorf("Embedding %d has no values", i)
		}
		if len(emb.Indices) != len(emb.Values) {
			t.Errorf("Embedding %d has mismatched indices and values", i)
		}
	}
}

// TestImageEmbedding_New tests creating a new image embedding model
func TestImageEmbedding_New(t *testing.T) {

	ie, err := NewImageEmbedding("")
	if err != nil {
		t.Fatalf("Failed to create image embedding: %v", err)
	}
	defer ie.Close()

	if ie.handle == nil {
		t.Error("Expected non-nil handle")
	}
}

// TestTextRerank_New tests creating a new text reranking model
func TestTextRerank_New(t *testing.T) {

	tr, err := NewTextRerank("")
	if err != nil {
		t.Fatalf("Failed to create text rerank: %v", err)
	}
	defer tr.Close()

	if tr.handle == nil {
		t.Error("Expected non-nil handle")
	}
}

// TestTextRerank_Rerank tests reranking documents
func TestTextRerank_Rerank(t *testing.T) {

	tr, err := NewTextRerank("")
	if err != nil {
		t.Fatalf("Failed to create text rerank: %v", err)
	}
	defer tr.Close()

	query := "What is a panda?"
	documents := []string{
		"The giant panda is a bear species endemic to China.",
		"Panda is an animal.",
		"I don't know.",
	}

	results, err := tr.Rerank(query, documents, true, 0)
	if err != nil {
		t.Fatalf("Failed to rerank documents: %v", err)
	}

	if len(results) != len(documents) {
		t.Errorf("Expected %d results, got %d", len(documents), len(results))
	}

	// Results should be sorted by score in descending order
	for i := 1; i < len(results); i++ {
		if results[i-1].Score < results[i].Score {
			t.Error("Results are not sorted by score in descending order")
		}
	}
}
