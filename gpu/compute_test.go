package gpu

import (
	_ "embed"
	"math"
	"testing"
)

//go:embed testdata/scaledadd.spv
var scaledAddShader []byte

func openDeviceForComputeTest(t *testing.T, forceStagingBuffers bool) *Device {
	t.Helper()
	devices, err := ListDevices()
	if err != nil || len(devices) == 0 {
		t.Skip("no Vulkan GPU available")
	}
	device, err := openDevice(0, forceStagingBuffers)
	if err != nil {
		t.Skipf("could not open the GPU: %v", err)
	}
	return device
}

func TestRecorderRunsManyDispatchesInOneSubmit(t *testing.T) {
	for _, forceStaging := range []bool{false, true} {
		device := openDeviceForComputeTest(t, forceStaging)
		program, err := device.NewProgram("scaledAdd", scaledAddShader, 3, 2)
		if err != nil {
			t.Fatal(err)
		}
		count := 70000
		first, _ := device.NewBuffer(count)
		second, _ := device.NewBuffer(count)
		result, _ := device.NewBuffer(count)
		firstValues := make([]float64, count)
		secondValues := make([]float64, count)
		for i := range firstValues {
			firstValues[i] = float64(i % 97)
			secondValues[i] = float64(i%13) - 6
		}

		recorder, err := device.NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		recorder.Begin()
		if err := recorder.Upload(first, firstValues); err != nil {
			t.Fatal(err)
		}
		if err := recorder.Upload(second, secondValues); err != nil {
			t.Fatal(err)
		}
		recorder.Fill(result, 0.5)
		groupsAcross, groupsDown := GroupsFor(count, 256)
		for repeat := 0; repeat < 3; repeat++ {
			recorder.Run(program, groupsAcross, groupsDown, 1, []uint32{uint32(count), Float(2)}, first, second, result)
		}
		downloaded := make([]float64, count)
		recorder.Download(result, downloaded)
		if err := recorder.Submit(); err != nil {
			t.Fatal(err)
		}
		for i := range downloaded {
			want := 0.5 + 3*(firstValues[i]+2*secondValues[i])
			if math.Abs(downloaded[i]-want) > 1e-3 {
				t.Fatalf("staging %v value %d: got %v, want %v", forceStaging, i, downloaded[i], want)
			}
		}

		recorder.Begin()
		recorder.Run(program, groupsAcross, groupsDown, 1, []uint32{uint32(count), Float(-1)}, first, second, result)
		if err := recorder.Submit(); err != nil {
			t.Fatal(err)
		}
		if err := result.Download(downloaded); err != nil {
			t.Fatal(err)
		}
		want := 0.5 + 3*(firstValues[5]+2*secondValues[5]) + firstValues[5] - secondValues[5]
		if math.Abs(downloaded[5]-want) > 1e-3 {
			t.Errorf("staging %v: second submit gave %v, want %v", forceStaging, downloaded[5], want)
		}

		copied, _ := device.NewBuffer(count)
		recorder.Begin()
		recorder.Copy(result, copied, count)
		if err := recorder.Submit(); err != nil {
			t.Fatal(err)
		}
		copiedValues := make([]float64, count)
		if err := copied.Download(copiedValues); err != nil {
			t.Fatal(err)
		}
		if copiedValues[5] != downloaded[5] {
			t.Errorf("staging %v: copy gave %v, want %v", forceStaging, copiedValues[5], downloaded[5])
		}
		device.Close()
	}
}

func TestManyDispatchesUseMoreThanOneDescriptorPool(t *testing.T) {
	device := openDeviceForComputeTest(t, false)
	defer device.Close()
	program, err := device.NewProgram("scaledAdd", scaledAddShader, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	first, _ := device.NewBuffer(4)
	second, _ := device.NewBuffer(4)
	result, _ := device.NewBuffer(4)
	if err := first.Upload([]float64{1, 1, 1, 1}); err != nil {
		t.Fatal(err)
	}
	recorder, _ := device.NewRecorder()
	recorder.Begin()
	recorder.Fill(second, 0)
	recorder.Fill(result, 0)
	for repeat := 0; repeat < 2500; repeat++ {
		recorder.Run(program, 1, 1, 1, []uint32{4, Float(0)}, first, second, result)
	}
	if err := recorder.Submit(); err != nil {
		t.Fatal(err)
	}
	values := make([]float64, 4)
	if err := result.Download(values); err != nil {
		t.Fatal(err)
	}
	if values[3] != 2500 {
		t.Errorf("after 2500 dispatches the value is %v", values[3])
	}
}

func TestComputeMistakesPanic(t *testing.T) {
	device := openDeviceForComputeTest(t, false)
	defer device.Close()
	program, _ := device.NewProgram("scaledAdd", scaledAddShader, 3, 2)
	buffer, _ := device.NewBuffer(4)
	recorder, _ := device.NewRecorder()
	expect := func(name string, function func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s did not panic", name)
			}
		}()
		function()
	}
	expect("Run before Begin", func() { recorder.Run(program, 1, 1, 1, []uint32{4, 0}, buffer, buffer, buffer) })
	recorder.Begin()
	expect("wrong number of buffers", func() { recorder.Run(program, 1, 1, 1, []uint32{4, 0}, buffer) })
	expect("wrong push constants", func() { recorder.Run(program, 1, 1, 1, []uint32{4}, buffer, buffer, buffer) })
	expect("uploading too many values", func() { recorder.Upload(buffer, make([]float64, 5)) })
	if err := recorder.Submit(); err != nil {
		t.Fatal(err)
	}
	if _, err := device.NewBuffer(0); err == nil {
		t.Error("a buffer of 0 values should be refused")
	}
}

func TestRecorderMeasuresDispatchTimes(t *testing.T) {
	device := openDeviceForComputeTest(t, false)
	defer device.Close()
	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	defer recorder.Free()
	if err := recorder.MeasureTime(true); err != nil {
		t.Skip(err)
	}
	program, err := device.NewProgram("scaledAdd", scaledAddShader, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	values, err := device.NewBuffer(1000)
	if err != nil {
		t.Fatal(err)
	}
	recorder.Begin()
	for i := 0; i < 3; i++ {
		recorder.Run(program, 4, 1, 1, []uint32{1000, Float(1)}, values, values, values)
	}
	if err := recorder.Submit(); err != nil {
		t.Fatal(err)
	}
	times := recorder.DispatchTimes()
	if len(times) != 3 {
		t.Fatalf("expected 3 dispatch times, got %d", len(times))
	}
	for _, measured := range times {
		if measured.ProgramName != program.name || measured.Seconds < 0 || measured.Seconds > 1 {
			t.Errorf("odd measurement %+v", measured)
		}
	}
}
