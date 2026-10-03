package gputraining

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"os"

	"github.com/javanhut/GoTransformers/optimizer"
)

// TrainerState is the GPU trainer's resumable optimizer state: the AdamW step
// count, each parameter's first- and second-moment buffers (read back from the
// GPU), and the token embedding's own optimizer state. Model weights are not
// included here; save them separately with Model.Save and reload them before
// restoring this state.
//
// Resuming a run: LoadModel the saved weights, build a Trainer with NewTrainer
// (which uploads those weights and zeroes the moments), then LoadStateFromFile to
// restore the moments and step count so AdamW continues exactly where it left off
// instead of restarting its momentum from zero.
type TrainerState struct {
	StepsTaken              int
	AverageGradients        map[string][]float64
	AverageSquaredGradients map[string][]float64
	Embedding               optimizer.State
}

// SaveState reads the optimizer moments back from the GPU into a TrainerState.
func (trainer *Trainer) SaveState() (TrainerState, error) {
	if trainer.closed {
		return TrainerState{}, fmt.Errorf("the trainer is closed")
	}
	state := TrainerState{
		StepsTaken:              trainer.stepsTaken,
		AverageGradients:        make(map[string][]float64, len(trainer.parameters)),
		AverageSquaredGradients: make(map[string][]float64, len(trainer.parameters)),
		Embedding:               trainer.embeddingOptimizer.SaveState(),
	}
	for _, current := range trainer.parameters {
		averageGradients := make([]float64, len(current.cpuValues))
		if err := current.averageGradients.Download(averageGradients); err != nil {
			return TrainerState{}, err
		}
		averageSquaredGradients := make([]float64, len(current.cpuValues))
		if err := current.averageSquaredGradients.Download(averageSquaredGradients); err != nil {
			return TrainerState{}, err
		}
		state.AverageGradients[current.name] = averageGradients
		state.AverageSquaredGradients[current.name] = averageSquaredGradients
	}
	return state, nil
}

// RestoreState uploads the saved moments back onto the GPU and restores the step
// count and embedding optimizer. The trainer must hold the same model (same
// parameter names and sizes) the state was saved from.
func (trainer *Trainer) RestoreState(state TrainerState) error {
	if trainer.closed {
		return fmt.Errorf("the trainer is closed")
	}
	for _, current := range trainer.parameters {
		averageGradients, err := momentsFor(state.AverageGradients, "averageGradients", current)
		if err != nil {
			return err
		}
		if err := current.averageGradients.Upload(averageGradients); err != nil {
			return err
		}
		averageSquaredGradients, err := momentsFor(state.AverageSquaredGradients, "averageSquaredGradients", current)
		if err != nil {
			return err
		}
		if err := current.averageSquaredGradients.Upload(averageSquaredGradients); err != nil {
			return err
		}
	}
	if err := trainer.embeddingOptimizer.RestoreState(state.Embedding); err != nil {
		return err
	}
	trainer.stepsTaken = state.StepsTaken
	return nil
}

func momentsFor(saved map[string][]float64, which string, current *parameterOnGPU) ([]float64, error) {
	values, found := saved[current.name]
	if !found {
		return nil, fmt.Errorf("saved state has no %s for parameter %q", which, current.name)
	}
	if len(values) != len(current.cpuValues) {
		return nil, fmt.Errorf("saved %s for %q has %d values, want %d", which, current.name, len(values), len(current.cpuValues))
	}
	return values, nil
}

// SaveStateToFile writes the resumable state to a file with gob.
func (trainer *Trainer) SaveStateToFile(path string) error {
	state, err := trainer.SaveState()
	if err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	if err := gob.NewEncoder(writer).Encode(state); err != nil {
		file.Close()
		return err
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

// LoadStateFromFile reads a state written by SaveStateToFile and restores it.
func (trainer *Trainer) LoadStateFromFile(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	var state TrainerState
	if err := gob.NewDecoder(bufio.NewReader(file)).Decode(&state); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return trainer.RestoreState(state)
}
