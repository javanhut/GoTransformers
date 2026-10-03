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

func TestAnswerGradientsMatchFiniteDifference(t *testing.T) {
	const stepSize = 1e-6
	multiToken := tinySettings()
	multiToken.MultiTokenPrediction = true
	for name, settings := range map[string]Settings{"plain": tinySettings(), "multi-token prediction": multiToken} {
		model := newModel(t, settings)
		example := Example{PromptIDs: []int{1, 5, 2}, AnswerIDs: []int{9, 3, 4}}

		parameters := model.Parameters()
		parameter.ZeroGradients(parameters)
		model.ComputeAnswerGradients(example)

		for _, current := range parameters {
			for i := 0; i < len(current.Values); i += 11 {
				original := current.Values[i]
				current.Values[i] = original + stepSize
				higher := model.AnswerLoss(example)
				current.Values[i] = original - stepSize
				lower := model.AnswerLoss(example)
				current.Values[i] = original
				numerical := (higher - lower) / (2 * stepSize)
				if math.Abs(numerical-current.Gradients()[i]) > 1e-5*math.Max(1, math.Abs(numerical)) {
					t.Errorf("%s: %s[%d]: backward gave %v, finite difference gave %v", name, current.Name, i, current.Gradients()[i], numerical)
				}
			}
		}
	}
}

func TestAnswerLossOnlyCountsTheAnswer(t *testing.T) {
	model := newModel(t, tinySettings())
	prompt := []int{1, 5, 2, 7}
	answer := []int{9, 3}
	score := model.ScoreAnswer(prompt, answer)
	wantLoss := -score.LogProbability / float64(len(answer))
	if gotLoss := model.AnswerLoss(Example{PromptIDs: prompt, AnswerIDs: answer}); math.Abs(gotLoss-wantLoss) > 1e-12 {
		t.Errorf("answer loss %v, but minus the average log probability of the answer is %v", gotLoss, wantLoss)
	}
	wholeSequence := append(append([]int(nil), prompt...), answer...)
	if model.Loss(wholeSequence) == model.AnswerLoss(Example{PromptIDs: prompt, AnswerIDs: answer}) {
		t.Error("answer loss should differ from the loss over the whole sequence")
	}
}

func TestFrozenParametersDoNotMove(t *testing.T) {
	model := newModel(t, tinySettings())
	model.Freeze("block1.")
	model.Freeze("tokens")
	before := map[string][]float64{}
	for _, current := range model.Parameters() {
		before[current.Name] = append([]float64(nil), current.Values...)
	}
	adam := optimizer.NewAdam(0.01)
	for range 3 {
		model.TrainStep([]int{1, 2, 3, 4, 5, 6}, adam)
	}
	for _, current := range model.Parameters() {
		unchanged := reflect.DeepEqual(before[current.Name], current.Values)
		shouldBeFrozen := strings.HasPrefix(current.Name, "block1.") || strings.HasPrefix(current.Name, "tokens")
		if shouldBeFrozen && !unchanged {
			t.Errorf("%s is frozen but changed", current.Name)
		}
		if !shouldBeFrozen && unchanged && !strings.Contains(current.Name, "Norm") {
			t.Errorf("%s is not frozen but did not change", current.Name)
		}
	}
}

func TestBatchUsesTheAverageGradient(t *testing.T) {
	model := newModel(t, tinySettings())
	sequences := [][]int{{1, 2, 3, 4}, {5, 6, 7}, {8, 9, 10, 1, 2}}

	averageGradients := map[string][]float64{}
	for _, sequence := range sequences {
		parameters := model.Parameters()
		parameter.ZeroGradients(parameters)
		model.ComputeGradients(sequence)
		for _, current := range parameters {
			if averageGradients[current.Name] == nil {
				averageGradients[current.Name] = make([]float64, len(current.Gradients()))
			}
			for i, gradient := range current.Gradients() {
				averageGradients[current.Name][i] += gradient / float64(len(sequences))
			}
		}
	}

	before := map[string][]float64{}
	for _, current := range model.Parameters() {
		before[current.Name] = append([]float64(nil), current.Values...)
	}
	const learningRate = 0.1
	model.TrainBatch(sequences, optimizer.NewSGD(learningRate))
	for _, current := range model.Parameters() {
		for i, value := range current.Values {
			wanted := before[current.Name][i] - learningRate*averageGradients[current.Name][i]
			if math.Abs(value-wanted) > 1e-12 {
				t.Fatalf("%s[%d]: after the batch step %v, expected %v", current.Name, i, value, wanted)
			}
		}
	}
}

func trainSteps(model *Model, chosenOptimizer optimizer.Optimizer, tokenIDs []int, steps int) {
	for range steps {
		model.TrainBatch(RandomChunks(tokenIDs, 6, 2), chosenOptimizer)
	}
}

func TestCheckpointResumesExactly(t *testing.T) {
	tokenIDs := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10}
	makeOptimizers := map[string]func() optimizer.Resumable{
		"Adam": func() optimizer.Resumable { return optimizer.NewAdam(0.01) },
		"Muon": func() optimizer.Resumable { return optimizer.NewMuon(0.01) },
	}
	for name, makeOptimizer := range makeOptimizers {
		vectormath.SetRandomSeed(7)
		straightThrough := newModel(t, tinySettings())
		straightThrough.Freeze("block2.")
		trainSteps(straightThrough, makeOptimizer(), tokenIDs, 6)

		vectormath.SetRandomSeed(7)
		firstHalf := newModel(t, tinySettings())
		firstHalf.Freeze("block2.")
		firstOptimizer := makeOptimizer()
		trainSteps(firstHalf, firstOptimizer, tokenIDs, 3)
		folder := filepath.Join(t.TempDir(), "checkpoint")
		if err := firstHalf.SaveCheckpoint(folder, firstOptimizer, 3); err != nil {
			t.Fatal(err)
		}

		vectormath.SetRandomSeed(12345)
		resumedOptimizer := makeOptimizer()
		resumed, stepsDone, err := LoadCheckpoint(folder, resumedOptimizer)
		if err != nil {
			t.Fatal(err)
		}
		if stepsDone != 3 || !reflect.DeepEqual(resumed.FrozenNamePrefixes, []string{"block2."}) {
			t.Errorf("%s: resumed at step %d with frozen %v", name, stepsDone, resumed.FrozenNamePrefixes)
		}
		trainSteps(resumed, resumedOptimizer, tokenIDs, 3)

		wanted := straightThrough.Parameters()
		for i, current := range resumed.Parameters() {
			if !reflect.DeepEqual(current.Values, wanted[i].Values) {
				t.Errorf("%s: %s differs after resuming from the checkpoint", name, current.Name)
				break
			}
		}
	}
}

func TestAnswerConfidenceAndAbstaining(t *testing.T) {
	model := newModel(t, tinySettings())
	prompt := []int{1, 5, 2}

	answer := model.Answer(prompt, GenerationOptions{MaximumNewTokens: 4})
	if len(answer.TokenIDs) != 4 || answer.Abstained {
		t.Fatalf("expected 4 tokens and no abstaining, got %+v", answer)
	}
	scored := model.ScoreAnswer(prompt, answer.TokenIDs)
	if math.Abs(scored.Probability-answer.Score.Probability) > 1e-9 {
		t.Errorf("generation said the answer has probability %v, scoring it says %v", answer.Score.Probability, scored.Probability)
	}

	stopped := model.Answer(prompt, GenerationOptions{MaximumNewTokens: 4, StopTokenIDs: []int{answer.TokenIDs[0]}})
	if len(stopped.TokenIDs) != 0 || len(stopped.Score.TokenProbabilities) != 1 {
		t.Errorf("stopping on the first token should give no tokens and 1 probability, got %+v", stopped)
	}

	abstainIDs := []int{0}
	unsure := model.Answer(prompt, GenerationOptions{MaximumNewTokens: 4, ConfidenceTarget: 0.99, AbstainTokenIDs: abstainIDs})
	if !unsure.Abstained || !reflect.DeepEqual(unsure.TokenIDs, abstainIDs) || !reflect.DeepEqual(unsure.RejectedTokenIDs, answer.TokenIDs) {
		t.Errorf("an untrained model should abstain at a 0.99 confidence target, got %+v", unsure)
	}
}

func TestFineTuningLearnsAnswersAndBecomesConfident(t *testing.T) {
	model := newModel(t, tinySettings())
	examples := []Example{
		{PromptIDs: []int{1, 2, 3}, AnswerIDs: []int{7, 8, 0}},
		{PromptIDs: []int{4, 5, 6}, AnswerIDs: []int{9, 10, 0}},
	}
	adam := optimizer.NewAdam(0.02)
	for range 150 {
		model.TrainOnExamples(examples, adam)
	}
	for _, example := range examples {
		answer := model.Answer(example.PromptIDs, GenerationOptions{MaximumNewTokens: 5, StopTokenIDs: []int{0}, ConfidenceTarget: 0.5, AbstainTokenIDs: []int{0}})
		if answer.Abstained || !reflect.DeepEqual(answer.TokenIDs, example.AnswerIDs[:2]) {
			t.Errorf("prompt %v: wanted %v, got %+v", example.PromptIDs, example.AnswerIDs[:2], answer)
		}
	}
}

func TestCompressedModelStillGenerates(t *testing.T) {
	for name, settings := range testSettings() {
		model := newModel(t, settings)
		tokenIDs := []int{1, 5, 2, 9, 3}
		before := model.Forward(tokenIDs)
		weightBytesBefore := model.WeightBytes()
		model.CompressWeights(lowprecision.Float32)
		after := model.Forward(tokenIDs)
		for i := range before.Values {
			if math.Abs(before.Values[i]-after.Values[i]) > 1e-4 {
				t.Errorf("%s score %d: full precision %v, Float32 weights %v", name, i, before.Values[i], after.Values[i])
				break
			}
		}
		if model.WeightBytes()*10 > weightBytesBefore*6 {
			t.Errorf("%s: Float32 weights use %d bytes, Float64 used %d", name, model.WeightBytes(), weightBytesBefore)
		}
		model.Generate([]int{1, 2}, 3, 0)
		if model.GradientBytes() != 0 {
			t.Errorf("%s: a compressed model should hold no gradient memory, has %d bytes", name, model.GradientBytes())
		}
	}
}

func TestTrainingCreatesGradientsAndReleaseFreesThem(t *testing.T) {
	model := newModel(t, tinySettings())
	if model.GradientBytes() != 0 {
		t.Fatalf("a new model should hold no gradient memory, has %d bytes", model.GradientBytes())
	}
	model.TrainStep([]int{1, 2, 3, 4}, optimizer.NewAdam(0.01))
	if model.GradientBytes() == 0 {
		t.Fatal("training should create gradient memory")
	}
	model.ReleaseGradients()
	if model.GradientBytes() != 0 {
		t.Fatalf("ReleaseGradients should free gradient memory, still %d bytes", model.GradientBytes())
	}
	model.TrainStep([]int{1, 2, 3, 4}, optimizer.NewAdam(0.01))
}

func TestDecompressingLetsTrainingContinue(t *testing.T) {
	model := newModel(t, tinySettings())
	model.CompressWeights(lowprecision.Int8)
	model.DecompressWeights()
	model.TrainStep([]int{1, 2, 3, 4}, optimizer.NewAdam(0.01))
}
