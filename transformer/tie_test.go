package transformer

import (
	"os"
	"path/filepath"
	"testing"
)

// TestTiedSaveLoad checks that a tied model saves and loads (the output weight is
// omitted from the file and re-tied on load) and that the output projection and
// embedding share one matrix.
func TestTiedSaveLoad(t *testing.T) {
	settings := tinySettings()
	settings.TieEmbeddings = true
	model := newModel(t, settings)

	// The output weights and the embedding table must be the same backing array.
	if &model.OutputLayer.Weights.Values[0] != &model.TokenEmbedding.Table.Values[0] {
		t.Fatal("tied model: output weights and embedding table are not shared")
	}
	// A tied model has fewer parameters than the same model untied.
	untied := tinySettings()
	if newModel(t, untied).NumberOfParameters() <= model.NumberOfParameters() {
		t.Error("tied model should have fewer parameters than untied")
	}

	tokenIDs := []int{1, 5, 2, 9, 3, 1}
	before := model.Forward(tokenIDs)

	path := filepath.Join(t.TempDir(), "tied.weights")
	if err := model.Save(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if &loaded.OutputLayer.Weights.Values[0] != &loaded.TokenEmbedding.Table.Values[0] {
		t.Fatal("loaded tied model is not shared")
	}
	after := loaded.Forward(tokenIDs)
	for i := range before.Values {
		if before.Values[i] != after.Values[i] {
			t.Fatalf("output differs after save/load at %d: %v vs %v", i, before.Values[i], after.Values[i])
		}
	}
}
