package gpu

import (
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"sync"
	"testing"
)

func openForTest(t *testing.T, forceStagingBuffers bool) *Device {
	t.Helper()
	devices, err := ListDevices()
	if err != nil {
		t.Skipf("no Vulkan GPU available: %v", err)
	}
	best := 0
	for _, device := range devices {
		if kindRank(device.Kind) < kindRank(devices[best].Kind) {
			best = device.Index
		}
	}
	device, err := openDevice(best, forceStagingBuffers)
	if err != nil {
		t.Fatalf("could not open %v: %v", devices[best], err)
	}
	device.MinimumWorkForGPU = 0
	t.Cleanup(device.Close)
	return device
}

func checkClose(t *testing.T, name string, got vectormath.Matrix, want vectormath.Matrix, inner int) {
	t.Helper()
	if got.Rows != want.Rows || got.Columns != want.Columns {
		t.Fatalf("%s: got a %dx%d matrix, want %dx%d", name, got.Rows, got.Columns, want.Rows, want.Columns)
	}
	tolerance := 1e-5 * math.Sqrt(float64(inner)) * 4
	for i := range want.Values {
		difference := math.Abs(got.Values[i] - want.Values[i])
		if difference > tolerance*(1+math.Abs(want.Values[i])) {
			t.Fatalf("%s: value %d is %v, CPU says %v", name, i, got.Values[i], want.Values[i])
		}
	}
}

var testShapes = [][3]int{
	{1, 1, 1},
	{7, 13, 5},
	{64, 64, 64},
	{65, 17, 63},
	{129, 65, 257},
	{512, 512, 512},
}

func checkAllOperations(t *testing.T, device *Device) {
	t.Helper()
	cpu := vectormath.CPUBackend{}
	for _, shape := range testShapes {
		rows, inner, columns := shape[0], shape[1], shape[2]

		first := vectormath.NewRandomMatrix(rows, inner, -1, 1)
		second := vectormath.NewRandomMatrix(inner, columns, -1, 1)
		checkClose(t, "MatrixTimesMatrix", device.MatrixTimesMatrix(first, second), cpu.MatrixTimesMatrix(first, second), inner)

		secondRows := vectormath.NewRandomMatrix(columns, inner, -1, 1)
		checkClose(t, "MatrixTimesTransposed", device.MatrixTimesTransposed(first, secondRows), cpu.MatrixTimesTransposed(first, secondRows), inner)

		firstColumns := vectormath.NewRandomMatrix(inner, rows, -1, 1)
		checkClose(t, "TransposedTimesMatrix", device.TransposedTimesMatrix(firstColumns, second), cpu.TransposedTimesMatrix(firstColumns, second), inner)
	}
	if err := device.LastError(); err != nil {
		t.Fatalf("the GPU fell back to the CPU: %v", err)
	}
}

func TestMatchesCPU(t *testing.T) {
	device := openForTest(t, false)
	t.Logf("testing on %s", device.Name())
	checkAllOperations(t, device)
}

func TestMatchesCPUWithStagingBuffers(t *testing.T) {
	device := openForTest(t, true)
	checkAllOperations(t, device)
}

func TestDeviceLimitsAreReadCorrectly(t *testing.T) {
	device := openForTest(t, false)
	if device.maxStorageBufferRange < 1<<27 {
		t.Errorf("max storage buffer range %d is below the Vulkan minimum, the limits offset is probably wrong", device.maxStorageBufferRange)
	}
	for i, count := range device.maxWorkGroupCount {
		if count < 65535 {
			t.Errorf("max work group count[%d] = %d is below the Vulkan minimum, the limits offset is probably wrong", i, count)
		}
	}
}

func TestSmallWorkStaysOnCPU(t *testing.T) {
	device := openForTest(t, false)
	device.MinimumWorkForGPU = 1000
	first := vectormath.NewRandomMatrix(3, 3, -1, 1)
	second := vectormath.NewRandomMatrix(3, 3, -1, 1)
	got := device.MatrixTimesMatrix(first, second)
	want := vectormath.CPUBackend{}.MatrixTimesMatrix(first, second)
	for i := range want.Values {
		if got.Values[i] != want.Values[i] {
			t.Fatalf("small multiply should be exact on the CPU, got %v want %v", got.Values[i], want.Values[i])
		}
	}
}

func TestEmptyMatrices(t *testing.T) {
	device := openForTest(t, false)
	result := device.MatrixTimesMatrix(vectormath.NewMatrix(0, 4), vectormath.NewMatrix(4, 3))
	if result.Rows != 0 || result.Columns != 3 {
		t.Errorf("got %dx%d, want 0x3", result.Rows, result.Columns)
	}
}

func TestConcurrentUse(t *testing.T) {
	device := openForTest(t, false)
	first := vectormath.NewRandomMatrix(100, 70, -1, 1)
	second := vectormath.NewRandomMatrix(70, 90, -1, 1)
	want := vectormath.CPUBackend{}.MatrixTimesMatrix(first, second)
	var waitGroup sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			for repeat := 0; repeat < 5; repeat++ {
				checkClose(t, "concurrent", device.MatrixTimesMatrix(first, second), want, 70)
			}
		}()
	}
	waitGroup.Wait()
}

func TestWorksAsTheBackend(t *testing.T) {
	device := openForTest(t, false)
	vectormath.UseBackend(device)
	defer vectormath.UseBackend(nil)
	first := vectormath.NewRandomMatrix(33, 44, -1, 1)
	second := vectormath.NewRandomMatrix(44, 55, -1, 1)
	checkClose(t, "through vectormath", vectormath.MatrixTimesMatrix(first, second), vectormath.CPUBackend{}.MatrixTimesMatrix(first, second), 44)
}

func TestCloseTwiceAndUseAfterClose(t *testing.T) {
	device := openForTest(t, false)
	device.Close()
	device.Close()
	first := vectormath.NewRandomMatrix(10, 10, -1, 1)
	result := device.MatrixTimesMatrix(first, first)
	if result.Rows != 10 {
		t.Errorf("a closed device should still answer using the CPU")
	}
}

func TestOpenBadIndex(t *testing.T) {
	if _, err := ListDevices(); err != nil {
		t.Skipf("no Vulkan GPU available: %v", err)
	}
	if _, err := Open(999); err == nil {
		t.Error("opening GPU 999 did not fail")
	}
}
