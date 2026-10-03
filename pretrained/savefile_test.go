package pretrained

import (
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/weightfile"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func sameScoresExactly(t *testing.T, name string, first []float64, second []float64) {
	t.Helper()
	if len(first) != len(second) {
		t.Fatalf("%s: %d scores vs %d scores", name, len(first), len(second))
	}
	for i := range first {
		if math.Float64bits(first[i]) != math.Float64bits(second[i]) {
			t.Fatalf("%s score %d: %v vs %v", name, i, first[i], second[i])
		}
	}
}

func TestSaveModelAndTokenizer(t *testing.T) {
	trainedTokenizer := tokenizer.Train("the cat sat on the mat and the cat ate the rat", 40)
	settings := transformer.SmallSettings(trainedTokenizer.VocabularySize())
	settings.WeightPrecision = lowprecision.Int8
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "small.weights")
	if err := SaveModelAndTokenizer(path, model, trainedTokenizer, weightfile.Float64); err != nil {
		t.Fatal(err)
	}
	loadedModel, loadedTokenizer, err := LoadModelAndTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	text := "the cat sat on the rat"
	if !reflect.DeepEqual(trainedTokenizer.Encode(text), loadedTokenizer.Encode(text)) {
		t.Fatalf("tokenizer encodes %q as %v before saving and %v after loading", text, trainedTokenizer.Encode(text), loadedTokenizer.Encode(text))
	}
	tokenIDs := trainedTokenizer.Encode(text)
	sameScoresExactly(t, "small Int8 model", model.Forward(tokenIDs).Values, loadedModel.Forward(tokenIDs).Values)
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	fileInfo, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return fileInfo.Size()
}

func TestSmolLM2SavedAsInt8LoadsTheSame(t *testing.T) {
	folder := needDownloadedModel(t)
	startedLoadingFromSafetensors := time.Now()
	model, loadedTokenizer, err := LoadLlamaWithPrecision(folder, lowprecision.Int8)
	if err != nil {
		t.Fatal(err)
	}
	loadingFromSafetensors := time.Since(startedLoadingFromSafetensors)

	path := filepath.Join(t.TempDir(), "smollm2-int8.weights")
	if err := SaveModelAndTokenizer(path, model, loadedTokenizer, weightfile.Float64); err != nil {
		t.Fatal(err)
	}
	startedLoadingSaved := time.Now()
	loadedModel, reloadedTokenizer, err := LoadModelAndTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	loadingSaved := time.Since(startedLoadingSaved)
	t.Logf("Int8 SmolLM2: %d parameters, file %d bytes; loading from safetensors took %v, loading the saved file took %v", model.NumberOfParameters(), fileSize(t, path), loadingFromSafetensors, loadingSaved)

	text := "The capital of France is"
	tokenIDs := loadedTokenizer.Encode(text)
	if !reflect.DeepEqual(tokenIDs, reloadedTokenizer.Encode(text)) {
		t.Fatalf("tokenizer encodes %q as %v before saving and %v after loading", text, tokenIDs, reloadedTokenizer.Encode(text))
	}
	sameScoresExactly(t, "Int8 SmolLM2", model.Forward(tokenIDs).Values, loadedModel.Forward(tokenIDs).Values)
	originalTokens := model.Generate(tokenIDs, 5, 0)
	loadedTokens := loadedModel.Generate(tokenIDs, 5, 0)
	if !reflect.DeepEqual(originalTokens, loadedTokens) {
		t.Fatalf("generated %v before saving and %v after loading", originalTokens, loadedTokens)
	}
}

func TestSmolLM2SavedAsBFloat16IsLossless(t *testing.T) {
	folder := needDownloadedModel(t)
	model, loadedTokenizer, err := LoadLlama(folder)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "smollm2-bfloat16.weights")
	if err := SaveModelAndTokenizer(path, model, loadedTokenizer, weightfile.BFloat16); err != nil {
		t.Fatal(err)
	}
	t.Logf("BFloat16 SmolLM2: %d parameters, file %d bytes", model.NumberOfParameters(), fileSize(t, path))
	loadedModel, _, err := LoadModelAndTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	loadedParameters := loadedModel.Parameters()
	for i, current := range model.Parameters() {
		sameScoresExactly(t, current.Name, current.Values, loadedParameters[i].Values)
	}
}
