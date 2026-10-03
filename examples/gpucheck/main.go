package main

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/gpu"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/vectormath"
	"time"
)

func timeMultiply(backend vectormath.Backend, first vectormath.Matrix, second vectormath.Matrix) time.Duration {
	backend.MatrixTimesMatrix(first, second)
	repeats := 5
	start := time.Now()
	for range repeats {
		backend.MatrixTimesMatrix(first, second)
	}
	return time.Since(start) / time.Duration(repeats)
}

func timeGeneration(backend vectormath.Backend, weights vectormath.Matrix, repeats int) time.Duration {
	vectormath.UseBackend(backend)
	defer vectormath.UseBackend(nil)
	oneToken := vectormath.NewRandomMatrix(1, weights.Columns, -1, 1)
	vectormath.MatrixTimesTransposedWeights(oneToken, weights)
	start := time.Now()
	for range repeats {
		vectormath.MatrixTimesTransposedWeights(oneToken, weights)
	}
	return time.Since(start)
}

func timeTrainingSteps(backend vectormath.Backend, layer *perceptron.Layer, inputs vectormath.Matrix, steps int) time.Duration {
	vectormath.UseBackend(backend)
	defer vectormath.UseBackend(nil)
	sgd := optimizer.NewSGD(0.0001)
	trainOneStep := func() {
		parameter.ZeroGradients(layer.Parameters())
		outputs := layer.Forward(inputs)
		layer.Backward(outputs)
		sgd.Update(layer.Parameters())
	}
	trainOneStep()
	start := time.Now()
	for range steps {
		trainOneStep()
	}
	return time.Since(start) / time.Duration(steps)
}

func main() {
	devices, err := gpu.ListDevices()
	if err != nil {
		fmt.Println("no GPU found, everything will run on the CPU:", err)
		return
	}
	fmt.Println("GPUs found:")
	for _, device := range devices {
		fmt.Printf("  %d: %s (%s)\n", device.Index, device.Name, device.Kind)
	}

	device, err := gpu.OpenBest()
	if err != nil {
		fmt.Println("could not open a GPU:", err)
		return
	}
	defer device.Close()
	fmt.Println("using", device.Name())
	cpu := vectormath.CPUBackend{}

	size := 1024
	first := vectormath.NewRandomMatrix(size, size, -1, 1)
	second := vectormath.NewRandomMatrix(size, size, -1, 1)
	fmt.Printf("\n%dx%d times %dx%d:\n", size, size, size, size)
	fmt.Printf("  %-45s %v\n", cpu.Name(), timeMultiply(cpu, first, second))
	fmt.Printf("  %-45s %v\n", device.Name(), timeMultiply(device, first, second))

	cpuResult := cpu.MatrixTimesMatrix(first, second)
	gpuResult := device.MatrixTimesMatrix(first, second)
	largestDifference := 0.0
	for i := range cpuResult.Values {
		difference := cpuResult.Values[i] - gpuResult.Values[i]
		if difference < 0 {
			difference = -difference
		}
		if difference > largestDifference {
			largestDifference = difference
		}
	}
	fmt.Printf("  largest difference between CPU and GPU answers: %.2g\n", largestDifference)

	generationRepeats := 200
	weights := vectormath.NewRandomMatrix(size, size, -1, 1)
	fmt.Printf("\ngeneration-like: 1x%d times %dx%d weights, %d times:\n", size, size, size, generationRepeats)
	fmt.Printf("  %-45s %v\n", "CPU", timeGeneration(cpu, weights, generationRepeats))
	device.KeepWeightsOnGPU = false
	defaultMinimumWork := device.MinimumWorkForGPU
	device.MinimumWorkForGPU = 0
	fmt.Printf("  %-45s %v\n", "GPU, uploading the weights every call", timeGeneration(device, weights, generationRepeats))
	device.MinimumWorkForGPU = defaultMinimumWork
	device.KeepWeightsOnGPU = true
	fmt.Printf("  %-45s %v\n", "GPU, weights kept on the GPU", timeGeneration(device, weights, generationRepeats))

	trainingSteps := 10
	batchRows := 512
	layer := perceptron.NewLayer("layer", size, size, activationfunction.Tanh)
	inputs := vectormath.NewRandomMatrix(batchRows, size, -1, 1)
	fmt.Printf("\ntraining-like: forward, backward and an SGD step, %d rows through a %dx%d layer (per step):\n", batchRows, size, size)
	fmt.Printf("  %-45s %v\n", "CPU", timeTrainingSteps(cpu, layer, inputs, trainingSteps))
	device.KeepWeightsOnGPU = false
	fmt.Printf("  %-45s %v\n", "GPU, uploading the weights every call", timeTrainingSteps(device, layer, inputs, trainingSteps))
	device.KeepWeightsOnGPU = true
	uploadsBefore := device.WeightUploads()
	residentTime := timeTrainingSteps(device, layer, inputs, trainingSteps)
	fmt.Printf("  %-45s %v\n", "GPU, weights kept on the GPU", residentTime)
	fmt.Printf("  weight uploads during %d resident steps: %d (one per optimizer step)\n", trainingSteps+1, device.WeightUploads()-uploadsBefore)

	if err := device.LastError(); err != nil {
		fmt.Println("the GPU had a problem and the CPU answered instead:", err)
	}
}
