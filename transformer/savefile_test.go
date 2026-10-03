package transformer

import (
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/vectormath"
	"github.com/javanhut/GoTransformers/weightfile"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func saveAndLoad(t *testing.T, model *Model, precision weightfile.StoragePrecision) (*Model, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.weights")
	if err := model.SaveWithPrecision(path, precision); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	return loaded, path
}

func sameBits(t *testing.T, name string, first []float64, second []float64) {
	t.Helper()
	if len(first) != len(second) {
		t.Fatalf("%s: %d values vs %d values", name, len(first), len(second))
	}
	for i := range first {
		if math.Float64bits(first[i]) != math.Float64bits(second[i]) {
			t.Fatalf("%s value %d: %v vs %v", name, i, first[i], second[i])
		}
	}
}

func sameGeneration(t *testing.T, name string, original *Model, loaded *Model) {
	t.Helper()
	prompt := []int{1, 5, 2}
	tokenIDs := []int{3, 1, 4, 1, 5, 9}
	sameBits(t, name+" forward scores", original.Forward(tokenIDs).Values, loaded.Forward(tokenIDs).Values)
	originalTokens := original.Generate(prompt, 8, 0)
	loadedTokens := loaded.Generate(prompt, 8, 0)
	if !reflect.DeepEqual(originalTokens, loadedTokens) {
		t.Fatalf("%s: generated %v before saving and %v after loading", name, originalTokens, loadedTokens)
	}
	originalAnswer := original.Answer(prompt, GenerationOptions{MaximumNewTokens: 6})
	loadedAnswer := loaded.Answer(prompt, GenerationOptions{MaximumNewTokens: 6})
	if !reflect.DeepEqual(originalAnswer, loadedAnswer) {
		t.Fatalf("%s: answered %+v before saving and %+v after loading", name, originalAnswer, loadedAnswer)
	}
}

func sameCompressedWeights(t *testing.T, name string, original *Model, loaded *Model) {
	t.Helper()
	originalLayers := original.allLayers()
	loadedLayers := loaded.allLayers()
	for i, layer := range originalLayers {
		if layer.IsCompressed() != loadedLayers[i].IsCompressed() {
			t.Fatalf("%s: layer %s compressed %v before saving and %v after loading", name, layer.Name, layer.IsCompressed(), loadedLayers[i].IsCompressed())
		}
		if layer.IsCompressed() && !reflect.DeepEqual(layer.CompressedWeights.Snapshot(), loadedLayers[i].CompressedWeights.Snapshot()) {
			t.Fatalf("%s: layer %s has different compressed weights after loading", name, layer.Name)
		}
	}
	if original.TokenEmbedding.IsCompressed() != loaded.TokenEmbedding.IsCompressed() {
		t.Fatalf("%s: token table compressed %v before saving and %v after loading", name, original.TokenEmbedding.IsCompressed(), loaded.TokenEmbedding.IsCompressed())
	}
	if original.TokenEmbedding.IsCompressed() && !reflect.DeepEqual(original.TokenEmbedding.CompressedTable.Snapshot(), loaded.TokenEmbedding.CompressedTable.Snapshot()) {
		t.Fatalf("%s: token table has different compressed values after loading", name)
	}
}

func TestOldModelFileStillLoads(t *testing.T) {
	loaded, err := LoadModel(filepath.Join("testdata", "oldmodel.weights"))
	if err != nil {
		t.Fatal(err)
	}
	savedValues, err := weightfile.ReadText(filepath.Join("testdata", "oldmodel.weights.txt"))
	if err != nil {
		t.Fatal(err)
	}
	for _, current := range loaded.Parameters() {
		sameBits(t, current.Name, savedValues[current.Name], current.Values)
	}
}

func TestSaveAtEachStoragePrecision(t *testing.T) {
	model := newModel(t, SmallSettings(200))
	for _, precision := range []weightfile.StoragePrecision{weightfile.Float64, weightfile.Float32, weightfile.BFloat16} {
		loaded, path := saveAndLoad(t, model, precision)
		loadedParameters := loaded.Parameters()
		for i, current := range model.Parameters() {
			for j, value := range current.Values {
				if math.Float64bits(loadedParameters[i].Values[j]) != math.Float64bits(precision.Round(value)) {
					t.Fatalf("%v: %s[%d] = %v loaded as %v", precision, current.Name, j, value, loadedParameters[i].Values[j])
				}
			}
		}
		fileInfo, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		bytesPerParameter := float64(fileInfo.Size()) / float64(model.NumberOfParameters())
		wanted := float64(precision.BytesPerValue())
		if bytesPerParameter < wanted || bytesPerParameter > wanted+0.05 {
			t.Errorf("%v: file uses %.3f bytes per parameter, expected about %v", precision, bytesPerParameter, wanted)
		}
		t.Logf("%v: %d parameters in %d bytes (%.3f bytes each)", precision, model.NumberOfParameters(), fileInfo.Size(), bytesPerParameter)
	}
}

func TestCompressedModelSavesAndLoadsExactly(t *testing.T) {
	for _, compression := range []lowprecision.Precision{lowprecision.Float32, lowprecision.Int8, lowprecision.FP4} {
		settings := tinySettings()
		settings.WeightPrecision = compression
		model := newModel(t, settings)
		loaded, _ := saveAndLoad(t, model, weightfile.Float64)
		sameCompressedWeights(t, compression.String(), model, loaded)
		sameGeneration(t, compression.String(), model, loaded)

		smallerLoaded, _ := saveAndLoad(t, model, weightfile.BFloat16)
		sameCompressedWeights(t, compression.String()+" with bfloat16 for the rest", model, smallerLoaded)
	}
}

func TestModelCompressedAfterCreationLoadsCompressed(t *testing.T) {
	model := newModel(t, tinySettings())
	model.CompressWeights(lowprecision.Int8)
	loaded, _ := saveAndLoad(t, model, weightfile.Float64)
	if !loaded.IsCompressed() || loaded.Settings.WeightPrecision != lowprecision.Float64 {
		t.Fatalf("loaded model compressed %v with settings precision %v", loaded.IsCompressed(), loaded.Settings.WeightPrecision)
	}
	sameCompressedWeights(t, "compressed after creation", model, loaded)
	sameGeneration(t, "compressed after creation", model, loaded)
}

func TestCompressedFileIsSmaller(t *testing.T) {
	sizes := map[lowprecision.Precision]int64{}
	for _, compression := range []lowprecision.Precision{lowprecision.Float64, lowprecision.Float32, lowprecision.Int8, lowprecision.FP4} {
		settings := SmallSettings(200)
		settings.WeightPrecision = compression
		model := newModel(t, settings)
		_, path := saveAndLoad(t, model, weightfile.Float64)
		fileInfo, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		sizes[compression] = fileInfo.Size()
		t.Logf("%v: %d bytes", compression, fileInfo.Size())
	}
	if !(sizes[lowprecision.FP4] < sizes[lowprecision.Int8] && sizes[lowprecision.Int8] < sizes[lowprecision.Float32] && sizes[lowprecision.Float32] < sizes[lowprecision.Float64]) {
		t.Errorf("compressed files are not getting smaller: %v", sizes)
	}
	if float64(sizes[lowprecision.Int8]) > 0.2*float64(sizes[lowprecision.Float64]) {
		t.Errorf("an Int8 file is %d bytes, more than a fifth of the %d byte Float64 file", sizes[lowprecision.Int8], sizes[lowprecision.Float64])
	}
}

func TestCompressedModelWithAdaptersSaves(t *testing.T) {
	settings := tinySettings()
	settings.WeightPrecision = lowprecision.Int8
	model := newModel(t, settings)
	model.AddLowRankAdapters(2, 4)
	adam := optimizer.NewAdam(0.02)
	for range 3 {
		model.TrainStep([]int{1, 2, 3, 4, 5}, adam)
	}
	loaded, _ := saveAndLoad(t, model, weightfile.Float64)
	if loaded.Settings.AdapterRank != 2 {
		t.Fatalf("loaded model has adapter rank %d", loaded.Settings.AdapterRank)
	}
	sameCompressedWeights(t, "Int8 with adapters", model, loaded)
	sameGeneration(t, "Int8 with adapters", model, loaded)
}

func TestCompressedModelWithAdaptersResumesFromCheckpoint(t *testing.T) {
	tokenIDs := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10}
	settings := tinySettings()
	settings.WeightPrecision = lowprecision.Int8
	settings.AdapterRank = 2
	settings.AdapterAlpha = 4

	vectormath.SetRandomSeed(7)
	straightThrough := newModel(t, settings)
	trainSteps(straightThrough, optimizer.NewAdam(0.01), tokenIDs, 6)

	vectormath.SetRandomSeed(7)
	firstHalf := newModel(t, settings)
	firstOptimizer := optimizer.NewAdam(0.01)
	trainSteps(firstHalf, firstOptimizer, tokenIDs, 3)
	folder := filepath.Join(t.TempDir(), "checkpoint")
	if err := firstHalf.SaveCheckpoint(folder, firstOptimizer, 3); err != nil {
		t.Fatal(err)
	}

	vectormath.SetRandomSeed(12345)
	resumedOptimizer := optimizer.NewAdam(0.01)
	resumed, _, err := LoadCheckpoint(folder, resumedOptimizer)
	if err != nil {
		t.Fatal(err)
	}
	trainSteps(resumed, resumedOptimizer, tokenIDs, 3)
	sameCompressedWeights(t, "resumed QLoRA checkpoint", straightThrough, resumed)
	wanted := straightThrough.Parameters()
	for i, current := range resumed.Parameters() {
		sameBits(t, current.Name, wanted[i].Values, current.Values)
	}
}

func TestTiedModelSavesAndLoadsCompressedOrNot(t *testing.T) {
	settings := tinySettings()
	settings.TieOutputToEmbedding = true
	model := newModel(t, settings)
	loaded, path := saveAndLoad(t, model, weightfile.Float64)
	sameGeneration(t, "tied", model, loaded)
	if &loaded.OutputLayer.Weights.Values[0] != &loaded.TokenEmbedding.Table.Values[0] {
		t.Error("the loaded output layer no longer shares the token table")
	}
	savedValues, _, err := weightfile.ReadBinaryKeepingCompressed(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, found := savedValues["output.weights"]; found {
		t.Error("an uncompressed tied model saved output.weights as well as tokens.table")
	}

	for _, compression := range []lowprecision.Precision{lowprecision.Int8, lowprecision.FP4} {
		compressedSettings := settings
		compressedSettings.WeightPrecision = compression
		compressedModel := newModel(t, compressedSettings)
		compressedLoaded, compressedPath := saveAndLoad(t, compressedModel, weightfile.Float64)
		sameCompressedWeights(t, "tied "+compression.String(), compressedModel, compressedLoaded)
		sameGeneration(t, "tied "+compression.String(), compressedModel, compressedLoaded)
		_, savedCompressedValues, err := weightfile.ReadBinaryKeepingCompressed(compressedPath)
		if err != nil {
			t.Fatal(err)
		}
		if _, found := savedCompressedValues["output.weights"]; !found {
			t.Errorf("tied %v: the compressed output layer was not saved", compression)
		}
	}

	compressedAfterCreation := newModel(t, settings)
	compressedAfterCreation.CompressWeights(lowprecision.Int8)
	loadedAfterCreation, _ := saveAndLoad(t, compressedAfterCreation, weightfile.Float64)
	sameCompressedWeights(t, "tied and compressed after creation", compressedAfterCreation, loadedAfterCreation)
	sameGeneration(t, "tied and compressed after creation", compressedAfterCreation, loadedAfterCreation)
	loadedAfterCreation.DecompressWeights()
	if &loadedAfterCreation.OutputLayer.Weights.Values[0] != &loadedAfterCreation.TokenEmbedding.Table.Values[0] {
		t.Error("after decompressing, the output layer no longer shares the token table")
	}
}
