package transformer

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"
	"transformer/attention"
	"transformer/gradientcheck"
	"transformer/lossfunction"
	"transformer/lowprecision"
	"transformer/optimizer"
	"transformer/parameter"
	"transformer/vectormath"
)

func tinySettings() Settings {
	settings := SmallSettings(11)
	settings.VectorSize = 8
	settings.NumberOfHeads = 2
	settings.FeedForwardSize = 12
	settings.NumberOfBlocks = 3
	return settings
}

func testSettings() map[string]Settings {
	all := map[string]Settings{}
	all["plain"] = tinySettings()

	sineWaves := tinySettings()
	sineWaves.UseRotaryPositions = false
	all["sine-wave positions"] = sineWaves

	windowed := tinySettings()
	windowed.WindowSize = 3
	windowed.FullAttentionEvery = 3
	all["window 3, full every 3rd block"] = windowed

	topK := tinySettings()
	topK.TopK = 2
	all["top 2"] = topK

	sharing := tinySettings()
	sharing.BlocksPerKeyValueGroup = 3
	sharing.GroupSharingMode = attention.BorrowKeysAndValues
	all["3 blocks share keys and values"] = sharing

	sharingChoices := tinySettings()
	sharingChoices.BlocksPerKeyValueGroup = 2
	sharingChoices.GroupSharingMode = attention.BorrowKeysValuesAndChoices
	sharingChoices.TopK = 2
	sharingChoices.WindowSize = 4
	all["2 blocks share keys, values and top-2 choices in a window"] = sharingChoices

	compressedMix := tinySettings()
	compressedMix.AttentionPattern = []AttentionKind{StandardAttention, CompressedSparseAttention, HeavilyCompressedAttention}
	compressedMix.CompressionRate = 2
	compressedMix.HeavyCompressionRate = 3
	compressedMix.CompressedWindowSize = 2
	compressedMix.CompressedTopK = 1
	compressedMix.UseAttentionSink = true
	compressedMix.NormalizeQueriesAndKeys = true
	all["standard, compressed sparse and heavily compressed blocks"] = compressedMix

	smallerCache := tinySettings()
	smallerCache.NumberOfKeyValueHeads = 1
	smallerCache.ShareKeyAsValue = true
	smallerCache.RotaryDimensions = 2
	smallerCache.QueryRank = 3
	smallerCache.BlocksPerKeyValueGroup = 2
	all["multi query, key doubles as value, partial rotary, low rank queries, sharing"] = smallerCache

	experts := tinySettings()
	experts.UseMixtureOfExperts = true
	experts.NumberOfRoutedExperts = 3
	experts.ExpertsPerToken = 2
	experts.ExpertHiddenSize = 6
	experts.HashRoutedBlocks = 1
	all["mixture of experts with a hash routed first block"] = experts

	streams := tinySettings()
	streams.NumberOfResidualStreams = 2
	all["mHC with 2 streams"] = streams

	multiToken := tinySettings()
	multiToken.MultiTokenPrediction = true
	all["multi-token prediction"] = multiToken

	withDropout := tinySettings()
	withDropout.ResidualDropout = 0.1
	withDropout.AttentionDropout = 0.1
	all["dropout everywhere"] = withDropout

	deepSeek := DeepSeekStyleSettings(11)
	deepSeek.VectorSize = 8
	deepSeek.NumberOfHeads = 2
	deepSeek.RotaryDimensions = 2
	deepSeek.FeedForwardSize = 12
	deepSeek.ExpertHiddenSize = 6
	deepSeek.NumberOfRoutedExperts = 3
	deepSeek.CompressionRate = 2
	deepSeek.HeavyCompressionRate = 3
	deepSeek.CompressedWindowSize = 2
	deepSeek.CompressedTopK = 1
	deepSeek.WindowSize = 3
	deepSeek.NumberOfResidualStreams = 2
	all["DeepSeek style, everything on"] = deepSeek

	fp4 := tinySettings()
	fp4.CachePrecision = lowprecision.FP4
	fp4.TrainAtCachePrecision = true
	fp4.FeedForwardClampLimit = 10
	all["FP4 cache, clamped SwiGLU"] = fp4

	return all
}

func newModel(t *testing.T, settings Settings) *Model {
	t.Helper()
	model, err := NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestGeneratingMatchesTraining(t *testing.T) {
	tokenIDs := []int{1, 5, 2, 9, 9, 3, 0, 7, 4, 10}
	for name, settings := range testSettings() {
		model := newModel(t, settings)
		allAtOnce := model.Forward(tokenIDs)

		model.StartGenerating()
		for position, tokenID := range tokenIDs {
			oneAtATime := model.NextTokenScores(tokenID)
			for i, score := range oneAtATime {
				if math.Abs(score-allAtOnce.Get(position, i)) > 1e-9 {
					t.Errorf("%s position %d score %d: one at a time %v, all at once %v", name, position, i, score, allAtOnce.Get(position, i))
					break
				}
			}
		}
	}
}

func TestBlockGradients(t *testing.T) {
	tokenIDs := []int{1, 5, 2, 9, 3}
	for name, settings := range testSettings() {
		if settings.TrainAtCachePrecision {
			continue
		}
		model := newModel(t, settings)
		forward := func(inputs vectormath.Matrix) vectormath.Matrix {
			values := inputs
			for _, block := range model.Blocks {
				values = block.Forward(values, tokenIDs)
			}
			return values
		}
		backward := func(outputGradients vectormath.Matrix) vectormath.Matrix {
			gradients := outputGradients
			for i := len(model.Blocks) - 1; i >= 0; i-- {
				gradients = model.Blocks[i].Backward(gradients)
			}
			return gradients
		}
		var parameters []parameter.Parameter
		for _, block := range model.Blocks {
			parameters = append(parameters, block.Parameters()...)
		}
		inputs := vectormath.NewRandomMatrix(len(tokenIDs), settings.VectorSize*settings.residualStreams(), -1, 1)
		for _, problem := range gradientcheck.Compare(forward, backward, parameters, inputs) {
			t.Errorf("%s: %s", name, problem)
		}
	}
}

func TestWholeModelGradients(t *testing.T) {
	multiToken := tinySettings()
	multiToken.MultiTokenPrediction = true
	multiToken.NumberOfResidualStreams = 2
	checkWholeModelGradients(t, "plain", tinySettings())
	checkWholeModelGradients(t, "multi-token prediction and mHC", multiToken)
}

func checkWholeModelGradients(t *testing.T, name string, settings Settings) {
	const stepSize = 1e-6
	model := newModel(t, settings)
	tokenIDs := []int{1, 5, 2, 9, 3, 1}

	parameters := model.Parameters()
	parameter.ZeroGradients(parameters)
	model.ComputeGradients(tokenIDs)

	for _, current := range parameters {
		for i := 0; i < len(current.Values); i += 7 {
			original := current.Values[i]
			current.Values[i] = original + stepSize
			higher := model.TrainingLoss(tokenIDs)
			current.Values[i] = original - stepSize
			lower := model.TrainingLoss(tokenIDs)
			current.Values[i] = original
			numerical := (higher - lower) / (2 * stepSize)
			if math.Abs(numerical-current.Gradients()[i]) > 1e-5*math.Max(1, math.Abs(numerical)) {
				t.Errorf("%s: %s[%d]: backward gave %v, finite difference gave %v", name, current.Name, i, current.Gradients()[i], numerical)
			}
		}
	}
}

func TestTrainingLowersTheLoss(t *testing.T) {
	for name, settings := range testSettings() {
		model := newModel(t, settings)
		tokenIDs := []int{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}
		adam := optimizer.NewAdam(0.01)
		firstLoss := model.TrainStep(tokenIDs, adam)
		var lastLoss float64
		for step := 0; step < 150; step++ {
			lastLoss = model.TrainStep(tokenIDs, adam)
		}
		if lastLoss > firstLoss/4 {
			t.Errorf("%s: loss only went from %v to %v", name, firstLoss, lastLoss)
		}
	}
}

func TestSaveAndLoad(t *testing.T) {
	settings := testSettings()["2 blocks share keys, values and top-2 choices in a window"]
	model := newModel(t, settings)
	path := filepath.Join(t.TempDir(), "model.weights")
	if err := model.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded.Settings, model.Settings) {
		t.Errorf("loaded settings %+v, saved %+v", loaded.Settings, model.Settings)
	}
	tokenIDs := []int{3, 1, 4, 1, 5}
	original := model.Forward(tokenIDs)
	fromFile := loaded.Forward(tokenIDs)
	for i := range original.Values {
		if original.Values[i] != fromFile.Values[i] {
			t.Fatalf("score %d: original %v, loaded from file %v", i, original.Values[i], fromFile.Values[i])
		}
	}
}

func TestGenerate(t *testing.T) {
	model := newModel(t, tinySettings())
	generated := model.Generate([]int{1, 2}, 6, 0)
	if len(generated) != 6 {
		t.Errorf("asked for 6 tokens, got %v", generated)
	}
	sampled := model.Generate([]int{1, 2}, 6, 1.0)
	for _, tokenID := range sampled {
		if tokenID < 0 || tokenID >= model.Settings.VocabularySize {
			t.Errorf("sampled token %d is outside the vocabulary", tokenID)
		}
	}
}

func TestBadSettings(t *testing.T) {
	broken := map[string]func(*Settings){
		"heads don't divide vector size": func(settings *Settings) { settings.NumberOfHeads = 3 },
		"no blocks":                      func(settings *Settings) { settings.NumberOfBlocks = 0 },
		"negative window":                func(settings *Settings) { settings.WindowSize = -1 },
		"zero blocks per group":          func(settings *Settings) { settings.BlocksPerKeyValueGroup = 0 },
		"sharing mode own with groups": func(settings *Settings) {
			settings.BlocksPerKeyValueGroup = 2
			settings.GroupSharingMode = attention.OwnKeysAndValues
		},
	}
	for name, breakIt := range broken {
		settings := tinySettings()
		breakIt(&settings)
		if _, err := NewModel(settings); err == nil {
			t.Errorf("%s: NewModel did not return an error", name)
		}
	}
}

func lossGradient(scores vectormath.Matrix, targets vectormath.Matrix) vectormath.Matrix {
	return lossfunction.SoftmaxCrossEntropy.Gradient(scores, targets)
}

func TestSaveAndLoadGenerationState(t *testing.T) {
	for name, settings := range testSettings() {
		model := newModel(t, settings)
		prompt := []int{3, 1, 4, 1, 5, 9, 2, 6}
		path := filepath.Join(t.TempDir(), "prompt.cache")

		model.StartGenerating()
		model.Feed(prompt)
		if err := model.SaveGenerationState(path); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		straightThrough := model.Feed([]int{5, 3, 5})

		model.StartGenerating()
		model.Feed([]int{7, 7})
		if err := model.LoadGenerationState(path); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		afterLoading := model.Feed([]int{5, 3, 5})
		for i := range straightThrough {
			if straightThrough[i] != afterLoading[i] {
				t.Errorf("%s score %d: without saving %v, after loading the cache %v", name, i, straightThrough[i], afterLoading[i])
				break
			}
		}
	}
}

func TestGenerationStateRejectsOtherModels(t *testing.T) {
	model := newModel(t, tinySettings())
	other := newModel(t, tinySettings())
	path := filepath.Join(t.TempDir(), "prompt.cache")
	model.StartGenerating()
	model.Feed([]int{1, 2, 3})
	if err := model.SaveGenerationState(path); err != nil {
		t.Fatal(err)
	}
	if err := other.LoadGenerationState(path); err == nil {
		t.Error("a model with different weights loaded the cache without an error")
	}
}
