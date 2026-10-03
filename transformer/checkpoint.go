package transformer

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/vectormath"
	"os"
	"path/filepath"
)

type TrainingProgress struct {
	StepsDone          int
	FrozenNamePrefixes []string
	RandomState        []byte
}

func checkpointFiles(folder string) (string, string, string) {
	return filepath.Join(folder, "model.weights"), filepath.Join(folder, "optimizer.state"), filepath.Join(folder, "progress.json")
}

func (model *Model) SaveCheckpoint(folder string, chosenOptimizer optimizer.Resumable, stepsDone int) error {
	if err := os.MkdirAll(folder, 0o755); err != nil {
		return err
	}
	modelPath, optimizerPath, progressPath := checkpointFiles(folder)
	if err := model.Save(modelPath); err != nil {
		return err
	}

	optimizerFile, err := os.Create(optimizerPath)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(optimizerFile)
	if err := gob.NewEncoder(writer).Encode(chosenOptimizer.SaveState()); err != nil {
		optimizerFile.Close()
		return err
	}
	if err := writer.Flush(); err != nil {
		optimizerFile.Close()
		return err
	}
	if err := optimizerFile.Close(); err != nil {
		return err
	}

	randomState, err := vectormath.SaveRandomState()
	if err != nil {
		return err
	}
	progress := TrainingProgress{StepsDone: stepsDone, FrozenNamePrefixes: model.FrozenNamePrefixes, RandomState: randomState}
	progressText, err := json.MarshalIndent(progress, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(progressPath, progressText, 0o644)
}

func LoadCheckpoint(folder string, chosenOptimizer optimizer.Resumable) (*Model, int, error) {
	modelPath, optimizerPath, progressPath := checkpointFiles(folder)
	model, err := LoadModel(modelPath)
	if err != nil {
		return nil, 0, err
	}

	optimizerFile, err := os.Open(optimizerPath)
	if err != nil {
		return nil, 0, err
	}
	defer optimizerFile.Close()
	var state optimizer.State
	if err := gob.NewDecoder(bufio.NewReader(optimizerFile)).Decode(&state); err != nil {
		return nil, 0, fmt.Errorf("%s: %w", optimizerPath, err)
	}
	if err := chosenOptimizer.RestoreState(state); err != nil {
		return nil, 0, err
	}

	progressText, err := os.ReadFile(progressPath)
	if err != nil {
		return nil, 0, err
	}
	var progress TrainingProgress
	if err := json.Unmarshal(progressText, &progress); err != nil {
		return nil, 0, fmt.Errorf("%s: %w", progressPath, err)
	}
	if err := vectormath.RestoreRandomState(progress.RandomState); err != nil {
		return nil, 0, err
	}
	model.FrozenNamePrefixes = progress.FrozenNamePrefixes
	return model, progress.StepsDone, nil
}
