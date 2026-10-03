package gputraining

import (
	"path/filepath"
	"testing"

	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
)

// TestTrainerStateRoundTrip trains a few steps, saves the optimizer state to a
// file, restores it into a fresh trainer holding an identical model, and checks
// the step count and every moment buffer survive the GPU download/upload round
// trip unchanged.
func TestTrainerStateRoundTrip(t *testing.T) {
	device := openTestDevice(t)
	settings := smallTestSettings()

	vectormath.SetRandomSeed(7)
	model, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(device, model, DefaultTrainerOptions(testLearningRate, testWeightDecay))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()

	examples := []transformer.Example{
		{PromptIDs: []int{1, 2, 3}, AnswerIDs: []int{4, 5, 6}},
		{PromptIDs: []int{7, 8}, AnswerIDs: []int{9, 10, 11}},
	}
	for step := 0; step < 3; step++ {
		trainer.TrainOnExamples(examples)
	}
	if trainer.StepsTaken() != 3 {
		t.Fatalf("StepsTaken = %d, want 3", trainer.StepsTaken())
	}

	saved, err := trainer.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "optimizer.state")
	if err := trainer.SaveStateToFile(path); err != nil {
		t.Fatal(err)
	}

	// Fresh trainer with an identical model, then restore from the file.
	vectormath.SetRandomSeed(7)
	model2, err := transformer.NewModel(settings)
	if err != nil {
		t.Fatal(err)
	}
	trainer2, err := NewTrainer(device, model2, DefaultTrainerOptions(testLearningRate, testWeightDecay))
	if err != nil {
		t.Fatal(err)
	}
	defer trainer2.Close()
	if err := trainer2.LoadStateFromFile(path); err != nil {
		t.Fatal(err)
	}

	if trainer2.StepsTaken() != 3 {
		t.Errorf("restored StepsTaken = %d, want 3", trainer2.StepsTaken())
	}
	restored, err := trainer2.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range saved.AverageGradients {
		got := restored.AverageGradients[name]
		if len(got) != len(want) {
			t.Fatalf("param %q averageGradients length %d, want %d", name, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("param %q averageGradients[%d] = %v, want %v", name, i, got[i], want[i])
			}
		}
	}
	if restored.Embedding.StepsTaken != saved.Embedding.StepsTaken {
		t.Errorf("embedding StepsTaken = %d, want %d", restored.Embedding.StepsTaken, saved.Embedding.StepsTaken)
	}
}
