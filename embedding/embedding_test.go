package embedding

import (
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

func TestEmbeddingForwardAndBackward(t *testing.T) {
	embedding := NewEmbedding("tokens", 5, 3)
	outputs := embedding.Forward([]int{2, 4, 2})
	if !reflect.DeepEqual(outputs.Row(0), embedding.Table.Row(2)) || !reflect.DeepEqual(outputs.Row(1), embedding.Table.Row(4)) {
		t.Errorf("Forward did not copy the right rows of the table")
	}

	embedding.Backward(vectormath.MatrixFromRows([]vectormath.Vector{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}))
	if !reflect.DeepEqual(tokenGradients(embedding, 2), vectormath.Vector{4, 4, 4}) {
		t.Errorf("token 2 appears twice so its gradients should add up to 4, got %v", tokenGradients(embedding, 2))
	}
	if !reflect.DeepEqual(tokenGradients(embedding, 0), vectormath.Vector{0, 0, 0}) {
		t.Errorf("token 0 was never used but has gradients %v", tokenGradients(embedding, 0))
	}
}

func TestEmbeddingRejectsUnknownIDs(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("token ID 5 in a vocabulary of 5 did not panic")
		}
	}()
	NewEmbedding("tokens", 5, 3).Forward([]int{5})
}

func TestPositionalEncoding(t *testing.T) {
	encoding := PositionalEncoding(10, 6)
	if encoding.Get(0, 0) != 0 || encoding.Get(0, 1) != 1 {
		t.Errorf("position 0 should be sin(0)=0, cos(0)=1, got %v", encoding.Row(0))
	}
	if math.Abs(encoding.Get(3, 0)-math.Sin(3)) > 1e-12 {
		t.Errorf("position 3 value 0 = %v, want sin(3)", encoding.Get(3, 0))
	}
}

func TestVocabulary(t *testing.T) {
	vocabulary := BuildVocabulary(SplitIntoWords("the cat sat on the mat"))
	if vocabulary.Size() != 6 {
		t.Errorf("vocabulary size = %d, want 6 (5 words plus %s)", vocabulary.Size(), UnknownToken)
	}
	ids := vocabulary.Encode([]string{"the", "dog", "sat"})
	if ids[1] != vocabulary.TokenToID[UnknownToken] {
		t.Errorf("unknown word got ID %d", ids[1])
	}
	if got := vocabulary.Decode(ids); !reflect.DeepEqual(got, []string{"the", UnknownToken, "sat"}) {
		t.Errorf("Decode = %v", got)
	}
}

func TestVocabularyFileRoundTrip(t *testing.T) {
	vocabulary := BuildVocabulary(SplitIntoCharacters("hi there\n\"quoted\"\t"))
	path := filepath.Join(t.TempDir(), "vocabulary.txt")
	if err := vocabulary.SaveToFile(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadVocabularyFromFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded.IDToToken, vocabulary.IDToToken) {
		t.Errorf("loaded %q, saved %q", loaded.IDToToken, vocabulary.IDToToken)
	}
}

func tokenGradients(embedding *Embedding, tokenID int) vectormath.Vector {
	size := embedding.VectorSize()
	return embedding.TableGradients[tokenID*size : (tokenID+1)*size]
}
