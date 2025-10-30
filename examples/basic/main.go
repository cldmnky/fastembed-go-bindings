package main

import (
	"fmt"
	"log"

	"github.com/cldmnky/fastembed-go-bindings/fastembed"
)

func main() {
	fmt.Println("FastEmbed Go Bindings Example")
	fmt.Println("==============================\n")

	// Example 1: Text Embeddings
	textEmbeddingExample()

	// Example 2: Sparse Text Embeddings
	sparseEmbeddingExample()

	// Example 3: Text Reranking
	rerankExample()
}

func textEmbeddingExample() {
	fmt.Println("1. Text Embeddings Example")
	fmt.Println("--------------------------")

	// Create a text embedding model
	model, err := fastembed.NewTextEmbedding("BGESmallENV15")
	if err != nil {
		log.Printf("Failed to create text embedding model: %v\n", err)
		return
	}
	defer model.Close()

	// Generate embeddings
	texts := []string{
		"Hello, World!",
		"This is a test document.",
		"FastEmbed is fast and efficient.",
	}

	embeddings, err := model.Embed(texts, 0)
	if err != nil {
		log.Printf("Failed to generate embeddings: %v\n", err)
		return
	}

	fmt.Printf("Generated %d embeddings\n", len(embeddings))
	for i, emb := range embeddings {
		fmt.Printf("  Text %d: dimension=%d, first 5 values=%v\n",
			i, len(emb), emb[:min(5, len(emb))])
	}
	fmt.Println()
}

func sparseEmbeddingExample() {
	fmt.Println("2. Sparse Text Embeddings Example")
	fmt.Println("----------------------------------")

	model, err := fastembed.NewSparseTextEmbedding("BGESmallENV15")
	if err != nil {
		log.Printf("Failed to create sparse text embedding model: %v\n", err)
		return
	}
	defer model.Close()

	texts := []string{
		"Hello, World!",
		"This is a test document.",
	}

	embeddings, err := model.Embed(texts, 0)
	if err != nil {
		log.Printf("Failed to generate sparse embeddings: %v\n", err)
		return
	}

	fmt.Printf("Generated %d sparse embeddings\n", len(embeddings))
	for i, emb := range embeddings {
		fmt.Printf("  Text %d: non-zero values=%d\n", i, len(emb.Indices))
		if len(emb.Indices) > 0 {
			fmt.Printf("    First index=%d, value=%.4f\n",
				emb.Indices[0], emb.Values[0])
		}
	}
	fmt.Println()
}

func rerankExample() {
	fmt.Println("3. Text Reranking Example")
	fmt.Println("-------------------------")

	model, err := fastembed.NewTextRerank("BGERerankerBase")
	if err != nil {
		log.Printf("Failed to create text rerank model: %v\n", err)
		return
	}
	defer model.Close()

	query := "What is a panda?"
	documents := []string{
		"The giant panda is a bear species endemic to China.",
		"Panda is an animal.",
		"I don't know what you're talking about.",
		"A type of mammal found in bamboo forests.",
	}

	results, err := model.Rerank(query, documents, true, 0)
	if err != nil {
		log.Printf("Failed to rerank documents: %v\n", err)
		return
	}

	fmt.Printf("Reranked %d documents\n", len(results))
	fmt.Println("Results (sorted by relevance):")
	for i, result := range results {
		fmt.Printf("  %d. Score: %.4f, Index: %d\n     Document: %s\n",
			i+1, result.Score, result.Index, result.Document)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
