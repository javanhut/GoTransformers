package main

import (
	"fmt"
	"github.com/javanhut/GoTransformers/datafile"
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/gputraining"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/training"
	"github.com/javanhut/GoTransformers/transformer"
	"github.com/javanhut/GoTransformers/weightfile"
	"os"
	"strconv"
	"strings"
)

type gpuTrainer interface {
	training.Trainer
	CopyWeightsToModel() error
	Close()
}

func chooseSchedule(shape string, warmupSteps int, totalSteps int, finalFraction float64) optimizer.LearningRateSchedule {
	switch shape {
	case "none":
		return optimizer.LearningRateSchedule{WarmupSteps: warmupSteps}
	case "cosine":
		return optimizer.WarmupThenCosine(warmupSteps, totalSteps, finalFraction)
	case "linear":
		return optimizer.WarmupThenLinear(warmupSteps, totalSteps, finalFraction)
	case "wsd":
		return optimizer.WarmupStableDecay(warmupSteps, totalSteps, totalSteps/5, finalFraction)
	}
	fmt.Println("-schedule must be none, cosine, linear or wsd")
	os.Exit(1)
	return optimizer.LearningRateSchedule{}
}

func chooseStoragePrecision(name string) weightfile.StoragePrecision {
	switch name {
	case "float64":
		return weightfile.Float64
	case "float32":
		return weightfile.Float32
	case "bfloat16":
		return weightfile.BFloat16
	}
	fmt.Println("-save-precision must be float64, float32 or bfloat16")
	os.Exit(1)
	return weightfile.Float64
}

func openDevices(choice string) []*gpu.Device {
	var indexes []int
	switch choice {
	case "best":
		device, err := gpu.OpenBest()
		if err != nil {
			fmt.Println("no Vulkan GPU found:", err)
			os.Exit(1)
		}
		return []*gpu.Device{device}
	case "all":
		devices, err := gpu.ListDevices()
		if err != nil {
			fmt.Println("no Vulkan GPU found:", err)
			os.Exit(1)
		}
		for _, device := range devices {
			if device.Kind == "discrete" {
				indexes = append(indexes, device.Index)
			}
		}
		if len(indexes) == 0 {
			for _, device := range devices {
				if device.Kind != "cpu" {
					indexes = append(indexes, device.Index)
				}
			}
		}
	default:
		for _, part := range strings.Split(choice, ",") {
			index, err := strconv.Atoi(strings.TrimSpace(part))
			if err != nil {
				fmt.Println("-gpus must be best, all or a list of GPU numbers like 0,1")
				os.Exit(1)
			}
			indexes = append(indexes, index)
		}
	}
	var opened []*gpu.Device
	for _, index := range indexes {
		device, err := gpu.Open(index)
		if err != nil {
			fmt.Printf("could not open GPU %d: %v\n", index, err)
			os.Exit(1)
		}
		opened = append(opened, device)
	}
	if len(opened) == 0 {
		fmt.Println("no GPUs to train on")
		os.Exit(1)
	}
	return opened
}

func deviceNames(devices []*gpu.Device) string {
	var names []string
	for _, device := range devices {
		names = append(names, device.Name())
	}
	return strings.Join(names, " + ")
}

func evaluationSequencesFrom(tokenIDs []int, tokenFile *datafile.TokenFile, chunkLength int, maximumNumberOfSequences int) [][]int {
	var sequences [][]int
	if tokenFile != nil {
		chunks, err := tokenFile.Chunks(chunkLength).NextBatch(maximumNumberOfSequences)
		if err != nil {
			fmt.Println("could not read evaluation tokens:", err)
			os.Exit(1)
		}
		return chunks
	}
	for start := 0; start+chunkLength <= len(tokenIDs) && len(sequences) < maximumNumberOfSequences; start += chunkLength {
		sequences = append(sequences, tokenIDs[start:start+chunkLength])
	}
	return sequences
}

func splitOffEvaluationTokens(tokenIDs []int, tokenFile *datafile.TokenFile, evaluationFraction float64) ([]int, *datafile.TokenFile, []int, *datafile.TokenFile) {
	if evaluationFraction <= 0 {
		return tokenIDs, tokenFile, nil, nil
	}
	if tokenFile != nil {
		trainingPart, evaluationPart, err := tokenFile.Split(evaluationFraction)
		if err != nil {
			fmt.Println("could not split the token file:", err)
			os.Exit(1)
		}
		return nil, trainingPart, nil, evaluationPart
	}
	firstEvaluationToken := len(tokenIDs) - int(float64(len(tokenIDs))*evaluationFraction)
	return tokenIDs[:firstEvaluationToken], nil, tokenIDs[firstEvaluationToken:], nil
}

func newGPUTrainer(devices []*gpu.Device, model *transformer.Model, options gputraining.TrainerOptions) gpuTrainer {
	if len(devices) == 1 {
		trainer, err := gputraining.NewTrainer(devices[0], model, options)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		return trainer
	}
	trainer, err := gputraining.NewDataParallelTrainer(devices, model, options)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	return trainer
}
