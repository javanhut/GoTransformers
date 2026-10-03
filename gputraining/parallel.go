package gputraining

import (
	"fmt"
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/vectormath"
	"sync"
)

type DataParallelTrainer struct {
	ResynchronizeEvery int

	trainers         []*Trainer
	gradientCopies   [][][]float64
	weightCopies     [][]float64
	lastGradientNorm float64
}

func NewDataParallelTrainer(devices []*gpu.Device, model *transformer.Model, options TrainerOptions) (*DataParallelTrainer, error) {
	if len(devices) == 0 {
		return nil, fmt.Errorf("data-parallel training needs at least 1 GPU")
	}
	parallel := &DataParallelTrainer{ResynchronizeEvery: 200}
	for _, device := range devices {
		trainer, err := NewTrainer(device, model, options)
		if err != nil {
			parallel.Close()
			return nil, err
		}
		parallel.trainers = append(parallel.trainers, trainer)
		var copies [][]float64
		for _, current := range trainer.parameters {
			copies = append(copies, make([]float64, len(current.cpuValues)))
		}
		parallel.gradientCopies = append(parallel.gradientCopies, copies)
	}
	return parallel, nil
}

func (parallel *DataParallelTrainer) NumberOfGPUs() int {
	return len(parallel.trainers)
}

func (parallel *DataParallelTrainer) Close() {
	for _, trainer := range parallel.trainers {
		trainer.Close()
	}
}

func splitAcrossGPUs(examples []transformer.Example, numberOfGPUs int) [][]transformer.Example {
	parts := make([][]transformer.Example, numberOfGPUs)
	for gpuIndex := range numberOfGPUs {
		first := gpuIndex * len(examples) / numberOfGPUs
		last := (gpuIndex + 1) * len(examples) / numberOfGPUs
		parts[gpuIndex] = examples[first:last]
	}
	return parts
}

func runOnEveryGPU(numberOfGPUs int, work func(gpuIndex int) error) error {
	errors := make([]error, numberOfGPUs)
	var waitGroup sync.WaitGroup
	for gpuIndex := range numberOfGPUs {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			errors[gpuIndex] = work(gpuIndex)
		}()
	}
	waitGroup.Wait()
	for gpuIndex, err := range errors {
		if err != nil {
			return fmt.Errorf("GPU %d: %w", gpuIndex, err)
		}
	}
	return nil
}

func (trainer *Trainer) computeGradients(examples []transformer.Example, sequencesInWholeBatch int, seed uint32, gradientCopies [][]float64) (float64, error) {
	batch, err := trainer.startBatch(examples, sequencesInWholeBatch)
	if err != nil {
		return 0, err
	}
	trainer.stepSeed = seed
	trainer.recordForward(batch, true)
	trainer.recordBackward(batch)
	rowLosses := make([]float64, batch.numberOfSequences*batch.sequenceLength)
	trainer.recorder.Download(trainer.shared.rowLosses, rowLosses)
	for parameterIndex, current := range trainer.parameters {
		trainer.recorder.Download(current.gradients, gradientCopies[parameterIndex])
	}
	if err := trainer.recorder.Submit(); err != nil {
		return 0, err
	}
	return sumOf(rowLosses), nil
}

func (trainer *Trainer) applyGradients(gradients [][]float64) error {
	if trainer.closed {
		return fmt.Errorf("the trainer is closed")
	}
	trainer.recorder.Begin()
	for parameterIndex, current := range trainer.parameters {
		if trainer.isFrozen(current.name) {
			continue
		}
		if err := trainer.recorder.Upload(current.gradients, gradients[parameterIndex]); err != nil {
			return err
		}
	}
	clipResult := trainer.recordOptimizerStep()
	if err := trainer.recorder.Submit(); err != nil {
		return err
	}
	trainer.lastGradientNorm = clipResult[1]
	return nil
}

func (parallel *DataParallelTrainer) TrainBatch(sequences [][]int) float64 {
	return parallel.TrainOnExamples(examplesFromSequences(sequences))
}

func (parallel *DataParallelTrainer) TrainOnExamples(examples []transformer.Example) float64 {
	loss, err := parallel.trainOnExamples(examples)
	if err != nil {
		panic(fmt.Sprintf("gputraining: %v", err))
	}
	return loss
}

func (parallel *DataParallelTrainer) trainOnExamples(examples []transformer.Example) (float64, error) {
	if len(examples) == 0 {
		return 0, fmt.Errorf("no examples given")
	}
	numberOfGPUs := len(parallel.trainers)
	parts := splitAcrossGPUs(examples, numberOfGPUs)
	seeds := make([]uint32, numberOfGPUs)
	for gpuIndex := range seeds {
		seeds[gpuIndex] = uint32(vectormath.RandomNumberBetween(0, 4294967295))
	}
	losses := make([]float64, numberOfGPUs)
	err := runOnEveryGPU(numberOfGPUs, func(gpuIndex int) error {
		if len(parts[gpuIndex]) == 0 {
			return nil
		}
		loss, err := parallel.trainers[gpuIndex].computeGradients(parts[gpuIndex], len(examples), seeds[gpuIndex], parallel.gradientCopies[gpuIndex])
		losses[gpuIndex] = loss
		return err
	})
	if err != nil {
		return 0, err
	}
	summedGradients := parallel.sumGradients(parts)
	err = runOnEveryGPU(numberOfGPUs, func(gpuIndex int) error {
		return parallel.trainers[gpuIndex].applyGradients(summedGradients)
	})
	if err != nil {
		return 0, err
	}
	parallel.lastGradientNorm = parallel.trainers[0].lastGradientNorm
	if parallel.ResynchronizeEvery > 0 && parallel.StepsTaken()%parallel.ResynchronizeEvery == 0 {
		if err := parallel.ResynchronizeWeights(); err != nil {
			return 0, err
		}
	}
	return sumOf(losses), nil
}

func (parallel *DataParallelTrainer) sumGradients(parts [][]transformer.Example) [][]float64 {
	var usedGPUs []int
	for gpuIndex, part := range parts {
		if len(part) > 0 {
			usedGPUs = append(usedGPUs, gpuIndex)
		}
	}
	summed := parallel.gradientCopies[usedGPUs[0]]
	for _, gpuIndex := range usedGPUs[1:] {
		for parameterIndex, gradients := range parallel.gradientCopies[gpuIndex] {
			target := summed[parameterIndex]
			for i, gradient := range gradients {
				target[i] += gradient
			}
		}
	}
	return summed
}

func (parallel *DataParallelTrainer) ResynchronizeWeights() error {
	first := parallel.trainers[0]
	if parallel.weightCopies == nil {
		for _, current := range first.parameters {
			parallel.weightCopies = append(parallel.weightCopies, make([]float64, len(current.cpuValues)))
		}
	}
	for parameterIndex, current := range first.parameters {
		if err := current.values.Download(parallel.weightCopies[parameterIndex]); err != nil {
			return err
		}
	}
	for _, trainer := range parallel.trainers[1:] {
		for parameterIndex, current := range trainer.parameters {
			if err := current.values.Upload(parallel.weightCopies[parameterIndex]); err != nil {
				return err
			}
		}
	}
	return nil
}

func (parallel *DataParallelTrainer) EvaluationLoss(sequences [][]int) float64 {
	examples := examplesFromSequences(sequences)
	parts := splitAcrossGPUs(examples, len(parallel.trainers))
	losses := make([]float64, len(parallel.trainers))
	err := runOnEveryGPU(len(parallel.trainers), func(gpuIndex int) error {
		if len(parts[gpuIndex]) == 0 {
			return nil
		}
		loss, err := parallel.trainers[gpuIndex].evaluationLossInWholeBatch(parts[gpuIndex], len(examples))
		losses[gpuIndex] = loss
		return err
	})
	if err != nil {
		panic(fmt.Sprintf("gputraining: %v", err))
	}
	return sumOf(losses)
}

func (parallel *DataParallelTrainer) LearningRate() float64 {
	return parallel.trainers[0].LearningRate()
}

func (parallel *DataParallelTrainer) LastGradientNorm() float64 {
	return parallel.lastGradientNorm
}

func (parallel *DataParallelTrainer) StepsTaken() int {
	return parallel.trainers[0].StepsTaken()
}

func (parallel *DataParallelTrainer) CopyWeightsToModel() error {
	return parallel.trainers[0].CopyWeightsToModel()
}

func (parallel *DataParallelTrainer) UploadWeightsFromModel() error {
	for _, trainer := range parallel.trainers {
		if err := trainer.UploadWeightsFromModel(); err != nil {
			return err
		}
	}
	return nil
}

func (parallel *DataParallelTrainer) SaveModel(path string) error {
	return parallel.trainers[0].SaveModel(path)
}

func (parallel *DataParallelTrainer) SaveStateToFile(path string) error {
	return parallel.trainers[0].SaveStateToFile(path)
}

func (parallel *DataParallelTrainer) LoadStateFromFile(path string) error {
	for _, trainer := range parallel.trainers {
		if err := trainer.LoadStateFromFile(path); err != nil {
			return err
		}
	}
	return nil
}
