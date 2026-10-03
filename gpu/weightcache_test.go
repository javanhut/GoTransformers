package gpu

import (
	"sync"
	"testing"
	"transformer/activationfunction"
	"transformer/optimizer"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/vectormath"
)

func openWithResidentWeights(t *testing.T, forceStagingBuffers bool) *Device {
	t.Helper()
	device := openForTest(t, forceStagingBuffers)
	device.KeepWeightsOnGPU = true
	device.MinimumWorkForGPUWithResidentWeights = 0
	device.MinimumWorkForGPUWithFewRowsTimesUntransposedWeights = 0
	return device
}

func bothMemoryPaths(t *testing.T, test func(t *testing.T, device *Device)) {
	for _, forceStagingBuffers := range []bool{false, true} {
		name := "host visible memory"
		if forceStagingBuffers {
			name = "staging buffers"
		}
		t.Run(name, func(t *testing.T) {
			test(t, openWithResidentWeights(t, forceStagingBuffers))
		})
	}
}

func TestResidentWeightsMatchCPU(t *testing.T) {
	bothMemoryPaths(t, func(t *testing.T, device *Device) {
		cpu := vectormath.CPUBackend{}
		for _, shape := range testShapes {
			rows, inner, columns := shape[0], shape[1], shape[2]
			first := vectormath.NewRandomMatrix(rows, inner, -1, 1)
			transposedWeights := vectormath.NewRandomMatrix(columns, inner, -1, 1)
			weights := vectormath.NewRandomMatrix(inner, columns, -1, 1)
			for repeat := 0; repeat < 2; repeat++ {
				checkClose(t, "MatrixTimesTransposedWeights", device.MatrixTimesTransposedWeights(first, transposedWeights), cpu.MatrixTimesTransposed(first, transposedWeights), inner)
				checkClose(t, "MatrixTimesWeights", device.MatrixTimesWeights(first, weights), cpu.MatrixTimesMatrix(first, weights), inner)
			}
		}
		if err := device.LastError(); err != nil {
			t.Fatalf("the GPU fell back to the CPU: %v", err)
		}
	})
}

func TestSameWeightsUploadOnce(t *testing.T) {
	bothMemoryPaths(t, func(t *testing.T, device *Device) {
		first := vectormath.NewRandomMatrix(8, 16, -1, 1)
		weights := vectormath.NewRandomMatrix(32, 16, -1, 1)
		for repeat := 0; repeat < 5; repeat++ {
			device.MatrixTimesTransposedWeights(first, weights)
		}
		if device.WeightUploads() != 1 || device.WeightCacheHits() != 4 {
			t.Errorf("5 calls with the same weights: %d uploads and %d cache hits, want 1 and 4", device.WeightUploads(), device.WeightCacheHits())
		}
	})
}

func TestChangedWeightsAreUploadedAgain(t *testing.T) {
	bothMemoryPaths(t, func(t *testing.T, device *Device) {
		cpu := vectormath.CPUBackend{}
		first := vectormath.NewRandomMatrix(8, 16, -1, 1)
		weights := vectormath.NewRandomMatrix(16, 24, -1, 1)
		device.MatrixTimesWeights(first, weights)

		for i := range weights.Values {
			weights.Values[i] = vectormath.RandomNumberBetween(-1, 1)
		}
		vectormath.MarkWeightsChanged()
		checkClose(t, "after MarkWeightsChanged", device.MatrixTimesWeights(first, weights), cpu.MatrixTimesMatrix(first, weights), 16)
		if device.WeightUploads() != 2 {
			t.Errorf("changed weights were uploaded %d times, want 2", device.WeightUploads())
		}
	})
}

func TestEditsWithoutMarkingAreNotSeen(t *testing.T) {
	device := openWithResidentWeights(t, false)
	first := vectormath.NewRandomMatrix(4, 8, -1, 1)
	weights := vectormath.NewRandomMatrix(8, 8, -1, 1)
	before := device.MatrixTimesWeights(first, weights)
	weights.Values[0] += 100
	after := device.MatrixTimesWeights(first, weights)
	for i := range before.Values {
		if before.Values[i] != after.Values[i] {
			t.Fatalf("an edit without vectormath.MarkWeightsChanged changed the GPU answer, the cache should have kept the old copy")
		}
	}
}

func TestLeastRecentlyUsedWeightsAreEvicted(t *testing.T) {
	bothMemoryPaths(t, func(t *testing.T, device *Device) {
		cpu := vectormath.CPUBackend{}
		first := vectormath.NewRandomMatrix(4, 32, -1, 1)
		oneMatrixBytes := uint64(32 * 32 * 4)
		device.WeightCacheLimitBytes = oneMatrixBytes + oneMatrixBytes/2
		var allWeights []vectormath.Matrix
		for i := 0; i < 3; i++ {
			allWeights = append(allWeights, vectormath.NewRandomMatrix(32, 32, -1, 1))
		}
		for round := 0; round < 3; round++ {
			for _, weights := range allWeights {
				checkClose(t, "evicting", device.MatrixTimesWeights(first, weights), cpu.MatrixTimesMatrix(first, weights), 32)
				if device.CachedWeightBytes() > device.WeightCacheLimitBytes {
					t.Fatalf("cache holds %d bytes, limit is %d", device.CachedWeightBytes(), device.WeightCacheLimitBytes)
				}
			}
		}
		if device.WeightUploads() != 9 {
			t.Errorf("with room for one matrix, cycling through 3 should upload every time: got %d uploads, want 9", device.WeightUploads())
		}
	})
}

func TestWeightsTooBigForTheCacheStillWork(t *testing.T) {
	device := openWithResidentWeights(t, false)
	device.WeightCacheLimitBytes = 16
	first := vectormath.NewRandomMatrix(4, 32, -1, 1)
	weights := vectormath.NewRandomMatrix(32, 32, -1, 1)
	checkClose(t, "too big for the cache", device.MatrixTimesWeights(first, weights), vectormath.CPUBackend{}.MatrixTimesMatrix(first, weights), 32)
	if device.WeightUploads() != 0 || device.CachedWeightBytes() != 0 {
		t.Errorf("weights bigger than the cache limit were cached")
	}
}

func TestCloseReleasesCachedWeights(t *testing.T) {
	device := openWithResidentWeights(t, false)
	first := vectormath.NewRandomMatrix(4, 16, -1, 1)
	weights := vectormath.NewRandomMatrix(16, 16, -1, 1)
	device.MatrixTimesWeights(first, weights)
	if device.CachedWeightBytes() == 0 {
		t.Fatal("nothing was cached")
	}
	device.Close()
	if device.CachedWeightBytes() != 0 {
		t.Errorf("Close left %d cached bytes", device.CachedWeightBytes())
	}
	checkClose(t, "after Close", device.MatrixTimesWeights(first, weights), vectormath.CPUBackend{}.MatrixTimesMatrix(first, weights), 16)
}

func TestResidentWeightsFromManyGoroutines(t *testing.T) {
	device := openWithResidentWeights(t, false)
	cpu := vectormath.CPUBackend{}
	sharedWeights := vectormath.NewRandomMatrix(40, 24, -1, 1)
	numberOfWorkers := 8
	repeats := 10
	ownWeights := make([]vectormath.Matrix, numberOfWorkers)
	inputs := make([][]vectormath.Matrix, numberOfWorkers)
	for worker := 0; worker < numberOfWorkers; worker++ {
		ownWeights[worker] = vectormath.NewRandomMatrix(24, 40, -1, 1)
		for repeat := 0; repeat < repeats; repeat++ {
			inputs[worker] = append(inputs[worker], vectormath.NewRandomMatrix(5, 24, -1, 1))
		}
	}

	var waitGroup sync.WaitGroup
	problems := make(chan string, numberOfWorkers)
	for worker := 0; worker < numberOfWorkers; worker++ {
		waitGroup.Add(1)
		go func(worker int) {
			defer waitGroup.Done()
			for _, first := range inputs[worker] {
				shared := device.MatrixTimesTransposedWeights(first, sharedWeights)
				own := device.MatrixTimesWeights(first, ownWeights[worker])
				wantShared := cpu.MatrixTimesTransposed(first, sharedWeights)
				wantOwn := cpu.MatrixTimesMatrix(first, ownWeights[worker])
				for i := range wantShared.Values {
					difference := shared.Values[i] - wantShared.Values[i]
					if difference > 1e-3 || difference < -1e-3 {
						problems <- "shared weights gave a wrong answer"
						return
					}
				}
				for i := range wantOwn.Values {
					difference := own.Values[i] - wantOwn.Values[i]
					if difference > 1e-3 || difference < -1e-3 {
						problems <- "a worker's own weights gave a wrong answer"
						return
					}
				}
			}
		}(worker)
	}
	waitGroup.Wait()
	close(problems)
	for problem := range problems {
		t.Error(problem)
	}
}

func TestLayerTrainsWithResidentWeights(t *testing.T) {
	device := openWithResidentWeights(t, false)
	vectormath.UseBackend(device)
	t.Cleanup(func() { vectormath.UseBackend(nil) })

	layer := perceptron.NewLayer("layer", 32, 16, activationfunction.Tanh)
	inputs := vectormath.NewRandomMatrix(8, 32, -1, 1)
	sgd := optimizer.NewSGD(0.1)
	for step := 0; step < 3; step++ {
		parameter.ZeroGradients(layer.Parameters())
		outputs := layer.Forward(inputs)
		layer.Backward(outputs)
		sgd.Update(layer.Parameters())

		onGPU := layer.Forward(inputs)
		vectormath.UseBackend(nil)
		onCPU := layer.Forward(inputs)
		vectormath.UseBackend(device)
		checkClose(t, "layer after an optimizer step", onGPU, onCPU, 32)
	}
	if device.WeightUploads() < 3 {
		t.Errorf("optimizer steps should have made the weights upload again, got %d uploads", device.WeightUploads())
	}
}

func TestFewRowsShader(t *testing.T) {
	bothMemoryPaths(t, func(t *testing.T, device *Device) {
		device.MinimumWorkForGPU = 0
		cpu := vectormath.CPUBackend{}
		for rows := 1; rows <= largestRowsForFewRowsShader; rows++ {
			for _, sizes := range [][2]int{{1, 1}, {13, 300}, {300, 13}, {1000, 777}} {
				inner, columns := sizes[0], sizes[1]
				first := vectormath.NewRandomMatrix(rows, inner, -1, 1)
				weights := vectormath.NewRandomMatrix(inner, columns, -1, 1)
				transposedWeights := vectormath.NewRandomMatrix(columns, inner, -1, 1)
				checkClose(t, "few rows times weights", device.MatrixTimesWeights(first, weights), cpu.MatrixTimesMatrix(first, weights), inner)
				checkClose(t, "few rows times transposed weights", device.MatrixTimesTransposedWeights(first, transposedWeights), cpu.MatrixTimesTransposed(first, transposedWeights), inner)
				checkClose(t, "few rows MatrixTimesMatrix", device.MatrixTimesMatrix(first, weights), cpu.MatrixTimesMatrix(first, weights), inner)
			}
		}
		if err := device.LastError(); err != nil {
			t.Fatalf("the GPU fell back to the CPU: %v", err)
		}
	})
}

func TestFewRowsShaderWithMoreColumnsThanOneDispatchRow(t *testing.T) {
	device := openWithResidentWeights(t, false)
	device.maxWorkGroupCount[0] = 1000
	columns := 2500
	first := vectormath.NewRandomMatrix(2, 4, -1, 1)
	transposedWeights := vectormath.NewRandomMatrix(columns, 4, -1, 1)
	checkClose(t, "wide output", device.MatrixTimesTransposedWeights(first, transposedWeights), vectormath.CPUBackend{}.MatrixTimesTransposed(first, transposedWeights), 4)
	if err := device.LastError(); err != nil {
		t.Fatalf("the GPU fell back to the CPU: %v", err)
	}
}
