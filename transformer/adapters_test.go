package transformer

import (
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func sameScores(t *testing.T, name string, first vectormath.Matrix, second vectormath.Matrix, allowed float64) {
	t.Helper()
	for i := range first.Values {
		if math.Abs(first.Values[i]-second.Values[i]) > allowed {
			t.Fatalf("%s score %d: %v vs %v", name, i, first.Values[i], second.Values[i])
		}
	}
}

func TestNewAdaptersDoNotChangeTheModel(t *testing.T) {
	model := newModel(t, tinySettings())
	tokenIDs := []int{1, 5, 2, 9}
	before := model.Forward(tokenIDs)
	model.AddLowRankAdapters(2, 4)
	sameScores(t, "after adding adapters", before, model.Forward(tokenIDs), 0)
}

func TestAdapterGradientsMatchFiniteDifference(t *testing.T) {
	const stepSize = 1e-6
	model := newModel(t, tinySettings())
	model.AddLowRankAdapters(2, 4)
	for _, current := range model.AdapterParameters() {
		for i := range current.Values {
			current.Values[i] = vectormath.RandomNumberBetween(-0.3, 0.3)
		}
	}
	example := Example{PromptIDs: []int{1, 5, 2}, AnswerIDs: []int{9, 3, 4}}
	parameter.ZeroGradients(model.Parameters())
	model.ComputeAnswerGradients(example)
	for _, current := range model.AdapterParameters() {
		for i := 0; i < len(current.Values); i += 3 {
			original := current.Values[i]
			current.Values[i] = original + stepSize
			higher := model.AnswerLoss(example)
			current.Values[i] = original - stepSize
			lower := model.AnswerLoss(example)
			current.Values[i] = original
			numerical := (higher - lower) / (2 * stepSize)
			if math.Abs(numerical-current.Gradients()[i]) > 1e-5*math.Max(1, math.Abs(numerical)) {
				t.Errorf("%s[%d]: backward gave %v, finite difference gave %v", current.Name, i, current.Gradients()[i], numerical)
			}
		}
	}
}

func baseValues(model *Model) map[string][]float64 {
	values := map[string][]float64{}
	for _, current := range model.Parameters() {
		if !strings.Contains(current.Name, ".lora.") {
			values[current.Name] = append([]float64(nil), current.Values...)
		}
	}
	return values
}

func TestAdaptersLearnWhileTheBaseStaysFrozen(t *testing.T) {
	for _, precision := range []lowprecision.Precision{lowprecision.Float64, lowprecision.Int8} {
		settings := tinySettings()
		settings.WeightPrecision = precision
		model := newModel(t, settings)
		model.AddLowRankAdapters(4, 8)
		before := baseValues(model)

		examples := []Example{
			{PromptIDs: []int{1, 2, 3}, AnswerIDs: []int{7, 8, 0}},
			{PromptIDs: []int{4, 5, 6}, AnswerIDs: []int{9, 10, 0}},
		}
		adam := optimizer.NewAdam(0.02)
		for step := 0; step < 200; step++ {
			model.TrainOnExamples(examples, adam)
		}
		for _, example := range examples {
			answer := model.Answer(example.PromptIDs, GenerationOptions{MaximumNewTokens: 5, StopTokenIDs: []int{0}})
			if !reflect.DeepEqual(answer.TokenIDs, example.AnswerIDs[:2]) {
				t.Errorf("%v base: prompt %v answered %v, wanted %v", precision, example.PromptIDs, answer.TokenIDs, example.AnswerIDs[:2])
			}
		}
		if !reflect.DeepEqual(before, baseValues(model)) {
			t.Errorf("%v base: training adapters changed the base model", precision)
		}
		if model.TokenEmbedding.TableGradients != nil {
			t.Errorf("%v base: the frozen embedding table got gradient memory", precision)
		}
		adapterBytes := 0
		for _, current := range model.AdapterParameters() {
			adapterBytes += len(current.Values) * 8
		}
		if model.GradientBytes() != adapterBytes {
			t.Errorf("%v base: gradient memory is %d bytes, the adapters alone need %d", precision, model.GradientBytes(), adapterBytes)
		}
	}
}

func TestSaveAndLoadAdapters(t *testing.T) {
	vectormath.SetRandomSeed(3)
	trained := newModel(t, tinySettings())
	vectormath.SetRandomSeed(3)
	freshCopy := newModel(t, tinySettings())

	trained.AddLowRankAdapters(2, 4)
	adam := optimizer.NewAdam(0.02)
	for step := 0; step < 20; step++ {
		trained.TrainStep([]int{1, 2, 3, 4, 5}, adam)
	}
	path := filepath.Join(t.TempDir(), "task.adapters")
	if err := trained.SaveAdapters(path); err != nil {
		t.Fatal(err)
	}
	if err := freshCopy.LoadAdapters(path); err != nil {
		t.Fatal(err)
	}
	tokenIDs := []int{3, 1, 4}
	sameScores(t, "loaded adapters", trained.Forward(tokenIDs), freshCopy.Forward(tokenIDs), 0)

	merged := trained.Forward(tokenIDs)
	trained.MergeAdapters()
	sameScores(t, "merged adapters", merged, trained.Forward(tokenIDs), 1e-12)
	if trained.Settings.AdapterRank != 0 {
		t.Error("merging should clear the adapter settings")
	}
}

func TestSavedModelKeepsItsAdapters(t *testing.T) {
	model := newModel(t, tinySettings())
	model.AddLowRankAdapters(2, 4)
	model.TrainStep([]int{1, 2, 3, 4}, optimizer.NewAdam(0.02))
	path := filepath.Join(t.TempDir(), "model.weights")
	if err := model.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	tokenIDs := []int{3, 1, 4}
	sameScores(t, "model loaded with adapters", model.Forward(tokenIDs), loaded.Forward(tokenIDs), 0)
}
