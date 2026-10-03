package transformer

import (
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"math"
	"path/filepath"
	"testing"
)

func tiedSettings() Settings {
	settings := tinySettings()
	settings.TieOutputToEmbedding = true
	return settings
}

func TestTiedModelSharesOneTable(t *testing.T) {
	model := newModel(t, tiedSettings())
	model.TokenEmbedding.Table.Values[3] = 0.125
	if model.OutputLayer.Weights.Values[3] != 0.125 {
		t.Fatal("the output layer does not read the embedding table")
	}
	untied := newModel(t, tinySettings())
	parameterCount := untied.NumberOfParameters() - model.NumberOfParameters()
	if parameterCount != untied.Settings.VocabularySize*untied.Settings.VectorSize {
		t.Fatalf("tying removed %d parameters, want %d", parameterCount, untied.Settings.VocabularySize*untied.Settings.VectorSize)
	}
	for _, current := range model.Parameters() {
		if current.Name == "output.weights" {
			t.Fatal("output.weights is still listed as its own parameter")
		}
	}
}

func TestTiedGradientsMatchFiniteDifference(t *testing.T) {
	const stepSize = 1e-6
	multiToken := tiedSettings()
	multiToken.MultiTokenPrediction = true
	for name, settings := range map[string]Settings{"tied": tiedSettings(), "tied with multi-token prediction": multiToken} {
		model := newModel(t, settings)
		example := Example{PromptIDs: []int{1, 5, 2}, AnswerIDs: []int{9, 3, 4}}
		parameters := model.Parameters()
		parameter.ZeroGradients(parameters)
		model.ComputeAnswerGradients(example)
		for _, current := range parameters {
			if current.Name != "tokens.table" {
				continue
			}
			for i := 0; i < len(current.Values); i += 3 {
				original := current.Values[i]
				current.Values[i] = original + stepSize
				higher := model.AnswerLoss(example)
				current.Values[i] = original - stepSize
				lower := model.AnswerLoss(example)
				current.Values[i] = original
				numerical := (higher - lower) / (2 * stepSize)
				if math.Abs(numerical-current.Gradients()[i]) > 1e-5*math.Max(1, math.Abs(numerical)) {
					t.Errorf("%s: tokens.table[%d]: backward gave %v, finite difference gave %v", name, i, current.Gradients()[i], numerical)
				}
			}
		}
	}
}

func TestTiedModelStaysTiedAfterLoading(t *testing.T) {
	model := newModel(t, tiedSettings())
	path := filepath.Join(t.TempDir(), "tied.weights")
	if err := model.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	loaded.TokenEmbedding.Table.Values[0] = 7
	if loaded.OutputLayer.Weights.Values[0] != 7 {
		t.Fatal("the loaded model is no longer tied")
	}
	loaded.TokenEmbedding.Table.Values[0] = model.TokenEmbedding.Table.Values[0]
	tokenIDs := []int{1, 2, 3, 4}
	if math.Abs(model.Loss(tokenIDs)-loaded.Loss(tokenIDs)) > 1e-12 {
		t.Fatal("the loaded tied model gives a different loss")
	}
}

func TestTrainerWithScheduleAndClippingLowersTheLoss(t *testing.T) {
	model := newModel(t, tiedSettings())
	adamW := optimizer.NewAdamW(0.01, 0.01)
	trainer, err := NewTrainer(model, adamW, TrainerOptions{
		Schedule:            optimizer.WarmupThenCosine(5, 60, 0.1),
		MaximumGradientNorm: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	sequences := [][]int{{1, 2, 3, 4, 5, 6}, {6, 5, 4, 3, 2, 1}}
	before := trainer.EvaluationLoss(sequences)
	for range 60 {
		trainer.TrainBatch(sequences)
		if trainer.LastGradientNorm() <= 0 {
			t.Fatal("the trainer did not report a gradient norm")
		}
	}
	after := trainer.EvaluationLoss(sequences)
	if after >= before*0.5 {
		t.Fatalf("loss went from %v to %v", before, after)
	}
	if trainer.StepsTaken() != 60 || math.Abs(trainer.LearningRate()-0.001) > 1e-12 || adamW.LearningRate >= 0.0011 {
		t.Fatalf("after 60 steps: %d steps taken, next learning rate %v (want 0.001), last used %v", trainer.StepsTaken(), trainer.LearningRate(), adamW.LearningRate)
	}
}

func TestTrainerClipsTheUpdate(t *testing.T) {
	model := newModel(t, tinySettings())
	startingValues := map[string][]float64{}
	for _, current := range model.Parameters() {
		startingValues[current.Name] = append([]float64(nil), current.Values...)
	}
	trainer, err := NewTrainer(model, optimizer.NewSGD(1), TrainerOptions{MaximumGradientNorm: 1e-3})
	if err != nil {
		t.Fatal(err)
	}
	trainer.TrainBatch([][]int{{1, 2, 3, 4, 5}})
	if trainer.LastGradientNorm() <= 1e-3 {
		t.Fatalf("gradient norm %v is too small for this test", trainer.LastGradientNorm())
	}
	movedSquared := 0.0
	for _, current := range model.Parameters() {
		for i, value := range current.Values {
			difference := value - startingValues[current.Name][i]
			movedSquared += difference * difference
		}
	}
	if math.Abs(math.Sqrt(movedSquared)-1e-3) > 1e-6 {
		t.Fatalf("SGD with rate 1 moved the clipped weights by %v, want 1e-3", math.Sqrt(movedSquared))
	}
}

func TestTiedWeightBytesCountTheTableOnce(t *testing.T) {
	tied := newModel(t, tiedSettings())
	untied := newModel(t, tinySettings())
	tableBytes := untied.TokenEmbedding.TableBytes()
	if untied.WeightBytes()-tied.WeightBytes() != tableBytes {
		t.Fatalf("tied model uses %d bytes, untied %d, the difference should be the table's %d", tied.WeightBytes(), untied.WeightBytes(), tableBytes)
	}
}
