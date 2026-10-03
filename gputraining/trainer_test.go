package gputraining

import (
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"strings"
	"testing"
)

const testLearningRate = 0.001
const testWeightDecay = 0.1

func openTestDevice(t *testing.T) *gpu.Device {
	t.Helper()
	device, err := gpu.OpenBest()
	if err != nil {
		t.Skipf("no Vulkan GPU available: %v", err)
	}
	t.Cleanup(device.Close)
	return device
}

func smallTestSettings() transformer.Settings {
	settings := transformer.SmallSettings(23)
	settings.VectorSize = 32
	settings.NumberOfHeads = 4
	settings.FeedForwardSize = 40
	settings.NumberOfBlocks = 2
	return settings
}

func twinModels(t *testing.T, settings transformer.Settings) (*transformer.Model, *transformer.Model) {
	t.Helper()
	makeModel := func() *transformer.Model {
		vectormath.SetRandomSeed(7)
		model, err := transformer.NewModel(settings)
		if err != nil {
			t.Fatal(err)
		}
		for _, current := range model.Parameters() {
			if current.IsMatrix() {
				continue
			}
			for i := range current.Values {
				current.Values[i] += vectormath.RandomNumberBetween(-0.2, 0.2)
			}
		}
		return model
	}
	first := makeModel()
	second := makeModel()
	firstParameters := first.Parameters()
	secondParameters := second.Parameters()
	for index, current := range firstParameters {
		for i, value := range current.Values {
			if secondParameters[index].Values[i] != value {
				t.Fatalf("the two models did not start the same at %s[%d]", current.Name, i)
			}
		}
	}
	return first, second
}

func gpuGradients(t *testing.T, trainer *Trainer) map[string][]float64 {
	t.Helper()
	gradients := map[string][]float64{}
	for _, current := range trainer.parameters {
		downloaded := make([]float64, len(current.cpuValues))
		if err := current.gradients.Download(downloaded); err != nil {
			t.Fatal(err)
		}
		gradients[current.name] = downloaded
	}
	return gradients
}

func gradientScale(cpuModel *transformer.Model, gradients []float64) float64 {
	largestInModel := 0.0
	for _, current := range cpuModel.Parameters() {
		largestInModel = math.Max(largestInModel, largestMagnitude(current.Gradients()))
	}
	return math.Max(largestMagnitude(gradients), 1e-4*largestInModel)
}

func largestMagnitude(values []float64) float64 {
	largest := 0.0
	for _, value := range values {
		largest = math.Max(largest, math.Abs(value))
	}
	return largest
}

type comparison struct {
	worstGradientError float64
	worstGradientName  string
	worstWeightError   float64
	worstWeightName    string
	flippedTinySteps   int
}

func compareGradients(t *testing.T, cpuModel *transformer.Model, gpuGradientsByName map[string][]float64, result *comparison) {
	t.Helper()
	for _, current := range cpuModel.Parameters() {
		gpuValues, found := gpuGradientsByName[current.Name]
		if !found {
			t.Fatalf("the GPU trainer has no gradients for %s", current.Name)
		}
		cpuValues := current.Gradients()
		scale := gradientScale(cpuModel, cpuValues)
		for i := range cpuValues {
			relativeError := math.Abs(gpuValues[i]-cpuValues[i]) / scale
			if relativeError > result.worstGradientError {
				result.worstGradientError = relativeError
				result.worstGradientName = current.Name
			}
		}
	}
}

func compareWeights(t *testing.T, cpuModel *transformer.Model, gpuModel *transformer.Model, startingValues map[string][]float64, result *comparison) {
	t.Helper()
	gpuParameters := map[string]parameter.Parameter{}
	for _, current := range gpuModel.Parameters() {
		gpuParameters[current.Name] = current
	}
	for _, current := range cpuModel.Parameters() {
		gpuValues := gpuParameters[current.Name].Values
		gradients := current.Gradients()
		scale := gradientScale(cpuModel, gradients)
		for i, cpuValue := range current.Values {
			difference := math.Abs(gpuValues[i] - cpuValue)
			if math.Abs(gradients[i]) < 1e-3*scale {
				if difference > 2*testLearningRate+1e-6 {
					t.Errorf("%s[%d]: GPU moved to %v, CPU to %v, from %v", current.Name, i, gpuValues[i], cpuValue, startingValues[current.Name][i])
				}
				if difference > 1e-5 {
					result.flippedTinySteps++
				}
				continue
			}
			relativeError := difference / testLearningRate
			if relativeError > result.worstWeightError {
				result.worstWeightError = relativeError
				result.worstWeightName = current.Name
			}
		}
	}
}

func snapshot(model *transformer.Model) map[string][]float64 {
	values := map[string][]float64{}
	for _, current := range model.Parameters() {
		values[current.Name] = append([]float64(nil), current.Values...)
	}
	return values
}

func randomSequence(length int, vocabularySize int) []int {
	sequence := make([]int, length)
	for i := range sequence {
		sequence[i] = int(vectormath.RandomNumberBetween(0, float64(vocabularySize)-0.001))
	}
	return sequence
}

func TestOneStepMatchesCPU(t *testing.T) {
	device := openTestDevice(t)

	groupedKeys := smallTestSettings()
	groupedKeys.NumberOfKeyValueHeads = 2

	partialHalves := smallTestSettings()
	partialHalves.RotaryDimensions = 4
	partialHalves.RotateHalves = true
	partialHalves.RotaryBase = 500

	sineWaves := smallTestSettings()
	sineWaves.UseRotaryPositions = false

	clamped := smallTestSettings()
	clamped.FeedForwardClampLimit = 0.05

	tied := smallTestSettings()
	tied.TieEmbeddings = true

	cases := []struct {
		name     string
		settings transformer.Settings
		lengths  []int
		answers  bool
	}{
		{"plain", smallTestSettings(), []int{12, 12, 12}, false},
		{"grouped keys and values", groupedKeys, []int{10, 10}, false},
		{"partial rotary with halves", partialHalves, []int{11, 11}, false},
		{"sine-wave positions", sineWaves, []int{9, 9}, false},
		{"clamped SwiGLU", clamped, []int{10, 10}, false},
		{"padded uneven batch", groupedKeys, []int{13, 5, 9, 2}, false},
		{"answer-only examples", partialHalves, []int{12, 7, 9}, true},
		{"tied embeddings", tied, []int{10, 10}, false},
		{"tied embeddings answer-only", tied, []int{12, 7, 9}, true},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			cpuModel, gpuModel := twinModels(t, testCase.settings)
			vectormath.SetRandomSeed(99)
			var examples []transformer.Example
			for index, length := range testCase.lengths {
				sequence := randomSequence(length, testCase.settings.VocabularySize)
				example := transformer.Example{AnswerIDs: sequence}
				if testCase.answers {
					promptLength := []int{5, 1, 0}[index%3]
					example = transformer.Example{PromptIDs: sequence[:promptLength], AnswerIDs: sequence[promptLength:]}
				}
				examples = append(examples, example)
			}
			sequences := make([][]int, len(examples))
			for i, example := range examples {
				sequences[i] = append(append([]int(nil), example.PromptIDs...), example.AnswerIDs...)
			}

			trainer, err := NewTrainer(device, gpuModel, DefaultTrainerOptions(testLearningRate, testWeightDecay))
			if err != nil {
				t.Fatal(err)
			}
			defer trainer.Close()
			cpuOptimizer := optimizer.NewAdamW(testLearningRate, testWeightDecay)
			startingValues := snapshot(cpuModel)

			for step := 1; step <= 2; step++ {
				var cpuLoss, gpuLoss float64
				if testCase.answers {
					cpuLoss = cpuModel.TrainOnExamples(examples, cpuOptimizer)
					gpuLoss = trainer.TrainOnExamples(examples)
				} else {
					cpuLoss = cpuModel.TrainBatch(sequences, cpuOptimizer)
					gpuLoss = trainer.TrainBatch(sequences)
				}
				lossError := math.Abs(gpuLoss-cpuLoss) / cpuLoss
				result := comparison{}
				compareGradients(t, cpuModel, gpuGradients(t, trainer), &result)
				if err := trainer.CopyWeightsToModel(); err != nil {
					t.Fatal(err)
				}
				if step == 1 {
					compareWeights(t, cpuModel, gpuModel, startingValues, &result)
				}
				t.Logf("step %d: loss CPU %.6f GPU %.6f (relative error %.1e), worst gradient error %.1e of the largest gradient (%s), worst weight error %.1e of the learning rate (%s), %d tiny-gradient steps differ",
					step, cpuLoss, gpuLoss, lossError, result.worstGradientError, result.worstGradientName, result.worstWeightError, result.worstWeightName, result.flippedTinySteps)
				if lossError > 1e-4 {
					t.Errorf("step %d: loss differs: CPU %v GPU %v", step, cpuLoss, gpuLoss)
				}
				if result.worstGradientError > 1e-3 {
					t.Errorf("step %d: gradients of %s differ by %.1e of the largest gradient", step, result.worstGradientName, result.worstGradientError)
				}
				if result.worstWeightError > 1e-2 {
					t.Errorf("step %d: updated weights of %s differ by %.1e of the learning rate", step, result.worstWeightName, result.worstWeightError)
				}
			}
		})
	}
}

func TestLossGoesDownOnRepetitiveText(t *testing.T) {
	device := openTestDevice(t)
	settings := smallTestSettings()
	settings.VocabularySize = 8
	vectormath.SetRandomSeed(3)
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(device, model, DefaultTrainerOptions(0.003, 0.01))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	pattern := []int{1, 2, 3, 4, 5, 6, 7, 3, 2, 1}
	var text []int
	for len(text) < 400 {
		text = append(text, pattern...)
	}
	firstLoss := 0.0
	lastLoss := 0.0
	for step := 1; step <= 300; step++ {
		lastLoss = trainer.TrainBatch(transformer.RandomChunks(text, 24, 4))
		if step == 1 {
			firstLoss = lastLoss
		}
	}
	t.Logf("loss went from %.4f to %.4f in 300 steps", firstLoss, lastLoss)
	if lastLoss > 0.1 || lastLoss > firstLoss/10 {
		t.Errorf("loss only went from %.4f to %.4f", firstLoss, lastLoss)
	}
	if err := trainer.CopyWeightsToModel(); err != nil {
		t.Fatal(err)
	}
	cpuLoss := model.Loss(text[:25])
	if cpuLoss > 0.2 {
		t.Errorf("after copying the weights back the CPU model's loss is %.4f", cpuLoss)
	}
}

func TestUnsupportedSettingsAreListed(t *testing.T) {
	device := openTestDevice(t)
	settings := smallTestSettings()
	settings.WindowSize = 4
	settings.TopK = 2
	settings.UseAttentionSink = true
	settings.MultiTokenPrediction = true
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewTrainer(device, model, DefaultTrainerOptions(0.001, 0))
	if err == nil {
		t.Fatal("expected an error")
	}
	for _, expected := range []string{"WindowSize", "TopK", "UseAttentionSink", "MultiTokenPrediction"} {
		if !strings.Contains(err.Error(), expected) {
			t.Errorf("error %q does not mention %s", err, expected)
		}
	}
}

func TestUploadWeightsFromModelResetsGPUWeights(t *testing.T) {
	device := openTestDevice(t)
	vectormath.SetRandomSeed(5)
	model, err := transformer.NewModel(smallTestSettings())
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(device, model, DefaultTrainerOptions(0.01, 0))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	before := snapshot(model)
	sequence := []int{1, 2, 3, 4, 5, 6}
	firstLoss := trainer.TrainBatch([][]int{sequence})
	if err := trainer.UploadWeightsFromModel(); err != nil {
		t.Fatal(err)
	}
	if err := trainer.CopyWeightsToModel(); err != nil {
		t.Fatal(err)
	}
	for _, current := range trainer.parameters {
		for i, value := range current.cpuValues {
			if math.Abs(value-before[current.name][i]) > 1e-6*math.Max(1, math.Abs(value)) {
				t.Fatalf("%s[%d] is %v after uploading, expected %v", current.name, i, value, before[current.name][i])
			}
		}
	}
	if firstLoss <= 0 {
		t.Errorf("loss %v", firstLoss)
	}
}

func TestChangingBatchShapesMatchCPU(t *testing.T) {
	device := openTestDevice(t)
	settings := smallTestSettings()
	settings.NumberOfKeyValueHeads = 2
	cpuModel, gpuModel := twinModels(t, settings)
	trainer, err := NewTrainer(device, gpuModel, DefaultTrainerOptions(testLearningRate, testWeightDecay))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	cpuOptimizer := optimizer.NewAdamW(testLearningRate, testWeightDecay)
	vectormath.SetRandomSeed(11)
	for step, lengths := range [][]int{{8, 8}, {5, 9, 7}, {20}, {4, 4, 4, 4, 4}, {6, 3}} {
		var sequences [][]int
		for _, length := range lengths {
			sequences = append(sequences, randomSequence(length, settings.VocabularySize))
		}
		cpuLoss := cpuModel.TrainBatch(sequences, cpuOptimizer)
		gpuLoss := trainer.TrainBatch(sequences)
		if math.Abs(gpuLoss-cpuLoss)/cpuLoss > 1e-4 {
			t.Errorf("step %d with lengths %v: CPU loss %v, GPU loss %v", step+1, lengths, cpuLoss, gpuLoss)
		}
	}
}
