package fastembed

import (
	"testing"
)

func TestListTextEmbeddingModels(t *testing.T) {
	models := ListTextEmbeddingModels()
	if len(models) == 0 {
		t.Fatal("Expected at least one text embedding model")
	}

	t.Logf("Found %d text embedding models:", len(models))
	for i, model := range models {
		t.Logf("  %d. %s (dim=%d)", i+1, model.ModelCode, model.Dimension)
		t.Logf("     %s", model.Description)
		
		if model.ModelCode == "" {
			t.Errorf("Model %d has empty model code", i)
		}
		if model.Dimension == 0 {
			t.Errorf("Model %s has zero dimension", model.ModelCode)
		}
	}
}

func TestListSparseTextEmbeddingModels(t *testing.T) {
	models := ListSparseTextEmbeddingModels()
	if len(models) == 0 {
		t.Fatal("Expected at least one sparse text embedding model")
	}

	t.Logf("Found %d sparse text embedding models:", len(models))
	for i, model := range models {
		t.Logf("  %d. %s (dim=%d)", i+1, model.ModelCode, model.Dimension)
		t.Logf("     %s", model.Description)
		
		if model.ModelCode == "" {
			t.Errorf("Model %d has empty model code", i)
		}
	}
}

func TestListImageEmbeddingModels(t *testing.T) {
	models := ListImageEmbeddingModels()
	if len(models) == 0 {
		t.Fatal("Expected at least one image embedding model")
	}

	t.Logf("Found %d image embedding models:", len(models))
	for i, model := range models {
		t.Logf("  %d. %s (dim=%d)", i+1, model.ModelCode, model.Dimension)
		t.Logf("     %s", model.Description)
		
		if model.ModelCode == "" {
			t.Errorf("Model %d has empty model code", i)
		}
		if model.Dimension == 0 {
			t.Errorf("Model %s has zero dimension", model.ModelCode)
		}
	}
}

func TestListTextRerankModels(t *testing.T) {
	models := ListTextRerankModels()
	if len(models) == 0 {
		t.Fatal("Expected at least one text reranking model")
	}

	t.Logf("Found %d text reranking models:", len(models))
	for i, model := range models {
		t.Logf("  %d. %s", i+1, model.ModelCode)
		t.Logf("     %s", model.Description)
		
		if model.ModelCode == "" {
			t.Errorf("Model %d has empty model code", i)
		}
	}
}

// Test that we can use different models by their model codes
func TestNewSparseTextEmbeddingWithModelCode(t *testing.T) {
	models := ListSparseTextEmbeddingModels()
	if len(models) == 0 {
		t.Skip("No sparse models available")
	}

	// Try using the model code from the list
	modelCode := models[0].ModelCode
	t.Logf("Testing with model code: %s", modelCode)

	model, err := NewSparseTextEmbedding(modelCode)
	if err != nil {
		t.Fatalf("Failed to create sparse text embedding with model code %s: %v", modelCode, err)
	}
	defer model.Close()

	// Test a simple embedding
	texts := []string{"Hello, world!"}
	embeddings, err := model.Embed(texts, 0)
	if err != nil {
		t.Fatalf("Failed to embed with model %s: %v", modelCode, err)
	}

	if len(embeddings) != 1 {
		t.Errorf("Expected 1 embedding, got %d", len(embeddings))
	}
}

func TestNewImageEmbeddingWithModelCode(t *testing.T) {
	models := ListImageEmbeddingModels()
	if len(models) == 0 {
		t.Skip("No image models available")
	}

	// Try using the model code from the list
	modelCode := models[0].ModelCode
	t.Logf("Testing with model code: %s", modelCode)

	model, err := NewImageEmbedding(modelCode)
	if err != nil {
		t.Fatalf("Failed to create image embedding with model code %s: %v", modelCode, err)
	}
	defer model.Close()

	t.Logf("Successfully created image embedding model: %s", modelCode)
}

func TestNewTextRerankWithModelCode(t *testing.T) {
	models := ListTextRerankModels()
	if len(models) == 0 {
		t.Skip("No reranking models available")
	}

	// Try the first model from the list
	modelCode := models[0].ModelCode
	t.Logf("Testing with model code: %s", modelCode)

	model, err := NewTextRerank(modelCode)
	if err != nil {
		t.Fatalf("Failed to create text reranker with model code %s: %v", modelCode, err)
	}
	defer model.Close()

	// Test a simple reranking
	query := "What is a panda?"
	documents := []string{
		"Panda is an animal.",
		"I don't know.",
	}

	results, err := model.Rerank(query, documents, true, 0)
	if err != nil {
		t.Fatalf("Failed to rerank with model %s: %v", modelCode, err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	t.Logf("Reranking results:")
	for i, result := range results {
		t.Logf("  %d. Score: %.4f, Index: %d, Document: %s", i+1, result.Score, result.Index, result.Document)
	}
}
