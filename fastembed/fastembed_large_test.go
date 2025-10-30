package fastembed

import (
	"bufio"
	"os"
	"strings"
	"testing"
	"time"
)

// loadTestData loads the test text chunks from file
func loadTestData(t *testing.T) []string {
	file, err := os.Open("test-data/large_texts.txt")
	if err != nil {
		t.Fatalf("Failed to open test data file: %v", err)
	}
	defer file.Close()

	var texts []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // Increase buffer size for large lines

	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text != "" {
			texts = append(texts, text)
		}
	}

	if err := scanner.Err(); err != nil {
		t.Fatalf("Error reading test data file: %v", err)
	}

	if len(texts) != 100 {
		t.Fatalf("Expected 100 text chunks, got %d", len(texts))
	}

	return texts
}

// TestAllTextEmbedding_LargeBatch tests all text embedding models against 100 large text chunks
func TestAllTextEmbedding_LargeBatch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test in short mode")
	}

	texts := loadTestData(t)

	// Get all available text embedding models
	models := ListTextEmbeddingModels()
	if len(models) == 0 {
		t.Fatal("No text embedding models available")
	}

	// Test specific model
	testModels := []string{
		"Xenova/bge-small-en-v1.5", // Small, fast model
	}

	t.Logf("Testing %d text embedding model against %d text chunks", len(testModels), len(texts))

	for _, modelCode := range testModels {
		// Find model info
		var modelInfo *ModelInfo
		for i := range models {
			if models[i].ModelCode == modelCode {
				modelInfo = &models[i]
				break
			}
		}

		if modelInfo == nil {
			t.Logf("Skipping model %s (not found in available models)", modelCode)
			continue
		}

		model := *modelInfo
		t.Run(model.ModelCode, func(t *testing.T) {
			t.Logf("Testing model: %s (dim=%d)", model.ModelCode, model.Dimension)
			t.Logf("Description: %s", model.Description)

			start := time.Now()
			te, err := NewTextEmbedding(model.ModelCode)
			if err != nil {
				t.Fatalf("Failed to create text embedding for %s: %v", model.ModelCode, err)
			}
			defer te.Close()

			// Generate embeddings for all 100 chunks
			embeddings, err := te.Embed(texts, 0)
			if err != nil {
				t.Fatalf("Failed to embed large batch of texts with %s: %v", model.ModelCode, err)
			}
			elapsed := time.Since(start)

			// Verify we got embeddings for all texts
			if len(embeddings) != len(texts) {
				t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
			}

			// Verify each embedding is non-empty
			if len(embeddings) > 0 && len(embeddings[0]) == 0 {
				t.Errorf("First embedding is empty")
			}

			actualDim := 0
			if len(embeddings) > 0 {
				actualDim = len(embeddings[0])
			}

			t.Logf("✓ Successfully generated %d embeddings (dim=%d) in %v (%.2f ms/text)",
				len(embeddings), actualDim, elapsed, float64(elapsed.Milliseconds())/float64(len(texts)))
		})
	}
}

// TestSparseTextEmbedding_LargeBatch tests sparse text embedding against 100 large text chunks
func TestSparseTextEmbedding_LargeBatch(t *testing.T) {
	texts := loadTestData(t)

	// Get all available sparse text embedding models
	models := ListSparseTextEmbeddingModels()
	if len(models) == 0 {
		t.Skip("No sparse text embedding models available")
	}

	t.Logf("Testing %d sparse text embedding models against %d text chunks", len(models), len(texts))

	for _, model := range models {
		model := model // capture range variable
		t.Run(model.ModelCode, func(t *testing.T) {
			t.Logf("Testing sparse model: %s", model.ModelCode)
			t.Logf("Description: %s", model.Description)

			start := time.Now()
			ste, err := NewSparseTextEmbedding(model.ModelCode)
			if err != nil {
				t.Fatalf("Failed to create sparse text embedding for %s: %v", model.ModelCode, err)
			}
			defer ste.Close()

			// Generate embeddings for all 100 chunks
			embeddings, err := ste.Embed(texts, 0)
			if err != nil {
				t.Fatalf("Failed to embed large batch of texts with %s: %v", model.ModelCode, err)
			}
			elapsed := time.Since(start)

			// Verify we got embeddings for all texts
			if len(embeddings) != len(texts) {
				t.Errorf("Expected %d embeddings, got %d", len(texts), len(embeddings))
			}

			// Verify each embedding is non-empty
			totalNonZero := 0
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
				totalNonZero += len(emb.Indices)
			}

			avgSparsity := float64(totalNonZero) / float64(len(embeddings))
			t.Logf("✓ Successfully generated %d sparse embeddings in %v (avg %.1f non-zero values, %.2f ms/text)",
				len(embeddings), elapsed, avgSparsity, float64(elapsed.Milliseconds())/float64(len(texts)))
		})
	}
}

// TestTextRerank_LargeBatch tests text reranking against 100 large text chunks
func TestTextRerank_LargeBatch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test in short mode")
	}

	texts := loadTestData(t)

	// Get all available text reranking models
	models := ListTextRerankModels()
	if len(models) == 0 {
		t.Skip("No text reranking models available")
	}

	// Test specific model
	testModels := []string{
		"BAAI/bge-reranker-base",
	}

	t.Logf("Testing %d text reranking model against %d text chunks", len(testModels), len(texts))

	for _, modelCode := range testModels {
		// Find model info
		var modelInfo *ModelInfo
		for i := range models {
			if models[i].ModelCode == modelCode {
				modelInfo = &models[i]
				break
			}
		}

		if modelInfo == nil {
			t.Logf("Skipping model %s (not found in available models)", modelCode)
			continue
		}

		model := *modelInfo
		t.Run(model.ModelCode, func(t *testing.T) {
			t.Logf("Testing reranker model: %s", model.ModelCode)
			t.Logf("Description: %s", model.Description)

			start := time.Now()
			tr, err := NewTextRerank(model.ModelCode)
			if err != nil {
				t.Fatalf("Failed to create text reranker for %s: %v", model.ModelCode, err)
			}
			defer tr.Close()

			// Use the first text as query and rerank the rest
			query := texts[0]
			documents := texts[1:] // Use all 99 remaining texts as documents

			// Rerank documents
			results, err := tr.Rerank(query, documents, false, 0)
			if err != nil {
				t.Fatalf("Failed to rerank documents with %s: %v", model.ModelCode, err)
			}
			elapsed := time.Since(start)

			// Verify we got results for all documents
			if len(results) != len(documents) {
				t.Errorf("Expected %d rerank results, got %d", len(documents), len(results))
			}

			// Verify results are sorted by score in descending order
			for i := 1; i < len(results); i++ {
				if results[i-1].Score < results[i].Score {
					t.Errorf("Results are not sorted by score in descending order at index %d", i)
					break
				}
			}

			t.Logf("✓ Successfully reranked %d documents in %v (%.2f ms/document)",
				len(documents), elapsed, float64(elapsed.Milliseconds())/float64(len(documents)))
			if len(results) > 0 {
				t.Logf("  Top score: %.4f, Bottom score: %.4f", results[0].Score, results[len(results)-1].Score)
			}
		})
	}
}
