package gputraining

import (
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"path/filepath"
	"testing"
)

func TestTiedClippedScheduledStepsMatchCPU(t *testing.T) {
	device := openTestDevice(t)
	tied := smallTestSettings()
	tied.TieEmbeddings = true
	sineWaves := smallTestSettings()
	sineWaves.UseRotaryPositions = false
	sineWaves.TieEmbeddings = true
	for name, settings := range map[string]transformer.Settings{"tied": tied, "tied with sine-wave positions": sineWaves, "untied": smallTestSettings()} {
		t.Run(name, func(t *testing.T) {
			cpuModel, gpuModel := twinModels(t, settings)
			options := DefaultTrainerOptions(testLearningRate, testWeightDecay)
			options.Schedule = optimizer.WarmupThenCosine(2, 6, 0.1)
			options.MaximumGradientNorm = 0.5
			gpuTrainer, err := NewTrainer(device, gpuModel, options)
			if err != nil {
				t.Fatal(err)
			}
			defer gpuTrainer.Close()
			cpuTrainer, err := transformer.NewTrainer(cpuModel, optimizer.NewAdamW(testLearningRate, testWeightDecay), transformer.TrainerOptions{Schedule: options.Schedule, MaximumGradientNorm: options.MaximumGradientNorm})
			if err != nil {
				t.Fatal(err)
			}
			vectormath.SetRandomSeed(3)
			sequences := [][]int{randomSequence(12, settings.VocabularySize), randomSequence(12, settings.VocabularySize), randomSequence(12, settings.VocabularySize)}
			for step := 1; step <= 4; step++ {
				if math.Abs(cpuTrainer.LearningRate()-gpuTrainer.LearningRate()) > 1e-15 {
					t.Fatalf("step %d: learning rates differ: CPU %v GPU %v", step, cpuTrainer.LearningRate(), gpuTrainer.LearningRate())
				}
				startingValues := snapshot(cpuModel)
				cpuLoss := cpuTrainer.TrainBatch(sequences)
				gpuLoss := gpuTrainer.TrainBatch(sequences)
				if math.Abs(gpuLoss-cpuLoss)/cpuLoss > 1e-4 {
					t.Errorf("step %d: loss CPU %v GPU %v", step, cpuLoss, gpuLoss)
				}
				normError := math.Abs(gpuTrainer.LastGradientNorm()-cpuTrainer.LastGradientNorm()) / cpuTrainer.LastGradientNorm()
				if normError > 1e-3 {
					t.Errorf("step %d: gradient norm CPU %v GPU %v", step, cpuTrainer.LastGradientNorm(), gpuTrainer.LastGradientNorm())
				}
				if step == 1 && cpuTrainer.LastGradientNorm() <= options.MaximumGradientNorm {
					t.Fatalf("gradient norm %v is below the clip limit, the test doesn't check clipping", cpuTrainer.LastGradientNorm())
				}
				if err := gpuTrainer.CopyWeightsToModel(); err != nil {
					t.Fatal(err)
				}
				result := comparison{}
				compareWeights(t, cpuModel, gpuModel, startingValues, &result)
				t.Logf("step %d: loss CPU %.6f GPU %.6f, norm CPU %.6f GPU %.6f, worst weight error %.1e of the learning rate (%s)", step, cpuLoss, gpuLoss, cpuTrainer.LastGradientNorm(), gpuTrainer.LastGradientNorm(), result.worstWeightError, result.worstWeightName)
				if result.worstWeightError > 2e-2 {
					t.Errorf("step %d: weights of %s differ by %.1e of the learning rate", step, result.worstWeightName, result.worstWeightError)
				}
				cpuModel.SetWeights(snapshot(gpuModel))
			}
		})
	}
}

func TestEvaluationLossMatchesCPUAndChangesNothing(t *testing.T) {
	device := openTestDevice(t)
	settings := smallTestSettings()
	settings.TieEmbeddings = true
	settings.ResidualDropout = 0.2
	settings.AttentionDropout = 0.2
	cpuModel, gpuModel := twinModels(t, settings)
	trainer, err := NewTrainer(device, gpuModel, DefaultTrainerOptions(testLearningRate, testWeightDecay))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	vectormath.SetRandomSeed(11)
	sequences := [][]int{randomSequence(9, settings.VocabularySize), randomSequence(9, settings.VocabularySize)}
	cpuLoss := (cpuModel.Loss(sequences[0]) + cpuModel.Loss(sequences[1])) / 2
	first := trainer.EvaluationLoss(sequences)
	second := trainer.EvaluationLoss(sequences)
	if first != second {
		t.Fatalf("evaluation is not repeatable (dropout leaked in?): %v then %v", first, second)
	}
	if math.Abs(first-cpuLoss)/cpuLoss > 1e-4 {
		t.Fatalf("evaluation loss GPU %v, CPU %v", first, cpuLoss)
	}
	if trainer.StepsTaken() != 0 {
		t.Fatal("evaluation counted as a training step")
	}
}

func TestDropoutTrainingIsRandomButStillLearns(t *testing.T) {
	device := openTestDevice(t)
	settings := smallTestSettings()
	settings.ResidualDropout = 0.1
	settings.AttentionDropout = 0.1
	vectormath.SetRandomSeed(4)
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(device, model, DefaultTrainerOptions(0.01, 0))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	pattern := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	sequences := [][]int{pattern[:12], pattern[4:16], pattern[8:]}
	before := trainer.EvaluationLoss(sequences)
	var losses []float64
	for range 150 {
		losses = append(losses, trainer.TrainBatch(sequences))
	}
	after := trainer.EvaluationLoss(sequences)
	if after > before*0.3 {
		t.Fatalf("evaluation loss went from %v to only %v", before, after)
	}
	sameAsNext := 0
	for i := 1; i < len(losses); i++ {
		if losses[i] == losses[i-1] {
			sameAsNext++
		}
	}
	if sameAsNext > 5 {
		t.Fatalf("%d training losses repeated exactly, dropout masks don't seem to change", sameAsNext)
	}
}

func TestFrozenTokenTableDoesNotMove(t *testing.T) {
	device := openTestDevice(t)
	settings := smallTestSettings()
	settings.TieEmbeddings = true
	vectormath.SetRandomSeed(8)
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	model.Freeze("tokens.")
	before := append([]float64(nil), model.TokenEmbedding.Table.Values...)
	trainer, err := NewTrainer(device, model, DefaultTrainerOptions(0.01, 0.1))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	for range 3 {
		trainer.TrainBatch([][]int{{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}})
	}
	if err := trainer.CopyWeightsToModel(); err != nil {
		t.Fatal(err)
	}
	for i, value := range model.TokenEmbedding.Table.Values {
		if math.Abs(value-before[i]) > 1e-6 {
			t.Fatalf("frozen table value %d moved from %v to %v", i, before[i], value)
		}
	}
}

func TestSaveModelWritesTheTrainedWeights(t *testing.T) {
	device := openTestDevice(t)
	settings := smallTestSettings()
	settings.TieEmbeddings = true
	vectormath.SetRandomSeed(9)
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(device, model, DefaultTrainerOptions(0.01, 0))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	sequences := [][]int{{1, 2, 3, 4, 5, 6}}
	for range 20 {
		trainer.TrainBatch(sequences)
	}
	path := filepath.Join(t.TempDir(), "trained.weights")
	if err := trainer.SaveModel(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := transformer.LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(loaded.Loss(sequences[0])-trainer.EvaluationLoss(sequences)) > 1e-4 {
		t.Fatalf("saved model loss %v, trainer evaluation loss %v", loaded.Loss(sequences[0]), trainer.EvaluationLoss(sequences))
	}
}
