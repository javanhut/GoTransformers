package main

import (
	"fmt"
	"time"
	"transformer/gpu"
	"transformer/vectormath"
)

func timeMultiply(backend vectormath.Backend, first vectormath.Matrix, second vectormath.Matrix) time.Duration {
	backend.MatrixTimesMatrix(first, second)
	repeats := 5
	start := time.Now()
	for i := 0; i < repeats; i++ {
		backend.MatrixTimesMatrix(first, second)
	}
	return time.Since(start) / time.Duration(repeats)
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

	size := 1024
	first := vectormath.NewRandomMatrix(size, size, -1, 1)
	second := vectormath.NewRandomMatrix(size, size, -1, 1)

	cpu := vectormath.CPUBackend{}
	cpuTime := timeMultiply(cpu, first, second)
	gpuTime := timeMultiply(device, first, second)
	fmt.Printf("%dx%d times %dx%d:\n", size, size, size, size)
	fmt.Printf("  %-45s %v\n", cpu.Name(), cpuTime)
	fmt.Printf("  %-45s %v\n", device.Name(), gpuTime)
	if err := device.LastError(); err != nil {
		fmt.Println("the GPU had a problem and the CPU answered instead:", err)
	}

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
	fmt.Printf("largest difference between CPU and GPU answers: %.2g\n", largestDifference)
}
