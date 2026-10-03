package training

import (
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/transformer"
	"os"
	"path/filepath"
	"testing"
)

func newTinyTrainer(t *testing.T) *transformer.Trainer {
	settings := transformer.SmallSettings(12)
	settings.VectorSize = 16
	settings.NumberOfHeads = 2
	settings.FeedForwardSize = 32
	settings.NumberOfBlocks = 2
	settings.TieOutputToEmbedding = true
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := transformer.NewTrainer(model, optimizer.NewAdamW(0.01, 0.01), transformer.TrainerOptions{
		Schedule:            optimizer.WarmupThenCosine(5, 80, 0.1),
		MaximumGradientNorm: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	return trainer
}

func TestLoopEvaluatesAndSavesTheBestModel(t *testing.T) {
	trainer := newTinyTrainer(t)
	pattern := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	bestPath := filepath.Join(t.TempDir(), "best.weights")
	var evaluations []EvaluationReport
	stepsSeen := 0
	loop := Loop{
		Trainer:             trainer,
		NumberOfSteps:       80,
		NextBatch:           func() [][]int { return transformer.RandomChunks(pattern, 6, 4) },
		EvaluationSequences: [][]int{pattern[:6], pattern[3:9], pattern[5:]},
		EvaluationBatchSize: 2,
		EvaluateEvery:       20,
		BestModelPath:       bestPath,
		OnStep: func(report StepReport) {
			stepsSeen++
			if report.LearningRate <= 0 || report.GradientNorm <= 0 {
				t.Errorf("step %d reported learning rate %v and gradient norm %v", report.Step, report.LearningRate, report.GradientNorm)
			}
		},
		OnEvaluation: func(report EvaluationReport) {
			evaluations = append(evaluations, report)
		},
	}
	result, err := loop.Run()
	if err != nil {
		t.Fatal(err)
	}
	if stepsSeen != 80 || result.StepsTaken != 80 || len(evaluations) != 4 {
		t.Fatalf("saw %d steps and %d evaluations, want 80 and 4", stepsSeen, len(evaluations))
	}
	if evaluations[len(evaluations)-1].EvaluationLoss >= evaluations[0].EvaluationLoss {
		t.Fatalf("evaluation loss did not go down: %+v", evaluations)
	}
	if _, err := os.Stat(bestPath); err != nil {
		t.Fatalf("the best model was not saved: %v", err)
	}
	bestModel, err := transformer.LoadModel(bestPath)
	if err != nil {
		t.Fatal(err)
	}
	bestTrainer, _ := transformer.NewTrainer(bestModel, optimizer.NewSGD(0), transformer.TrainerOptions{})
	reloadedLoss := EvaluationLossInBatches(bestTrainer, loop.EvaluationSequences, 0)
	if difference := reloadedLoss - result.BestEvaluationLoss; difference > 1e-9 || difference < -1e-9 {
		t.Fatalf("the saved best model has loss %v, the loop reported %v", reloadedLoss, result.BestEvaluationLoss)
	}
}

type flatTrainer struct {
	steps int
}

func (trainer *flatTrainer) TrainBatch(sequences [][]int) float64 {
	trainer.steps++
	return 1
}
func (trainer *flatTrainer) EvaluationLoss(sequences [][]int) float64 { return 1 }
func (trainer *flatTrainer) LearningRate() float64                    { return 0.1 }
func (trainer *flatTrainer) LastGradientNorm() float64                { return 1 }
func (trainer *flatTrainer) StepsTaken() int                          { return trainer.steps }
func (trainer *flatTrainer) SaveModel(path string) error              { return nil }

func TestLoopStopsWhenEvaluationStopsImproving(t *testing.T) {
	trainer := &flatTrainer{}
	result, err := Loop{
		Trainer:                              trainer,
		NumberOfSteps:                        1000,
		NextBatch:                            func() [][]int { return nil },
		EvaluationSequences:                  [][]int{{1, 2}},
		EvaluateEvery:                        10,
		StopAfterEvaluationsWithoutImproving: 3,
	}.Run()
	if err != nil {
		t.Fatal(err)
	}
	if !result.StoppedEarly || result.StepsTaken != 40 || result.BestStep != 10 {
		t.Fatalf("got %+v, want an early stop at step 40 with the best at step 10", result)
	}
}

func TestLoopRejectsBadSetups(t *testing.T) {
	trainer := &flatTrainer{}
	badLoops := []Loop{
		{NumberOfSteps: 1, NextBatch: func() [][]int { return nil }},
		{Trainer: trainer, NumberOfSteps: 1},
		{Trainer: trainer, NumberOfSteps: 0, NextBatch: func() [][]int { return nil }},
		{Trainer: trainer, NumberOfSteps: 1, NextBatch: func() [][]int { return nil }, EvaluateEvery: 5},
		{Trainer: trainer, NumberOfSteps: 1, NextBatch: func() [][]int { return nil }, BestModelPath: "x"},
	}
	for i, loop := range badLoops {
		if _, err := loop.Run(); err == nil {
			t.Errorf("bad loop %d was accepted", i)
		}
	}
}
