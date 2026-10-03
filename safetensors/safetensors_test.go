package safetensors

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestWriteAndReadBack(t *testing.T) {
	tensors := map[string]Tensor{
		"layer.weight": {Shape: []int{2, 3}, Values: []float64{1, -2, 3.5, 0, 1e-3, -7}},
		"layer.bias":   {Shape: []int{3}, Values: []float64{0.25, 0.5, 0.75}},
		"scalar":       {Shape: []int{}, Values: []float64{42}},
	}
	for name, writeFunction := range map[string]func(string, map[string]Tensor) error{"F32": Write, "F64": WriteFloat64} {
		path := filepath.Join(t.TempDir(), "model.safetensors")
		if err := writeFunction(path, tensors); err != nil {
			t.Fatal(err)
		}
		readBack, err := Read(path)
		if err != nil {
			t.Fatal(err)
		}
		for tensorName, tensor := range tensors {
			got := readBack[tensorName]
			if len(got.Shape) != len(tensor.Shape) || (len(tensor.Shape) > 0 && !reflect.DeepEqual(got.Shape, tensor.Shape)) {
				t.Errorf("%s %s: shape %v, want %v", name, tensorName, got.Shape, tensor.Shape)
			}
			for i := range tensor.Values {
				if math.Abs(got.Values[i]-tensor.Values[i]) > 1e-6*math.Max(1, math.Abs(tensor.Values[i])) {
					t.Errorf("%s %s value %d: %v, want %v", name, tensorName, i, got.Values[i], tensor.Values[i])
				}
			}
		}
	}
}

func TestReadOnlyOneTensor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
	Write(path, map[string]Tensor{"a": {Shape: []int{2}, Values: []float64{1, 2}}, "b": {Shape: []int{1}, Values: []float64{3}}})
	opened, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if !reflect.DeepEqual(opened.Names(), []string{"a", "b"}) {
		t.Errorf("names = %v", opened.Names())
	}
	tensor, err := opened.ReadTensor("b")
	if err != nil || tensor.Values[0] != 3 {
		t.Errorf("tensor b = %v, %v", tensor, err)
	}
	if _, err := opened.ReadTensor("missing"); err == nil {
		t.Error("reading a missing tensor did not fail")
	}
}

func TestHalfPrecisionConversions(t *testing.T) {
	bfloat16Cases := map[uint16]float64{0x3f80: 1, 0xc000: -2, 0x3f00: 0.5, 0x0000: 0, 0x4049: 3.140625}
	for bits, want := range bfloat16Cases {
		if got := BFloat16ToFloat64(bits); got != want {
			t.Errorf("BF16 %#04x = %v, want %v", bits, got, want)
		}
	}
	float16Cases := map[uint16]float64{0x3c00: 1, 0xc000: -2, 0x3800: 0.5, 0x0000: 0, 0x7bff: 65504, 0x0001: math.Pow(2, -24), 0x4248: 3.140625}
	for bits, want := range float16Cases {
		if got := Float16ToFloat64(bits); got != want {
			t.Errorf("F16 %#04x = %v, want %v", bits, got, want)
		}
	}
	if !math.IsInf(Float16ToFloat64(0x7c00), 1) || !math.IsNaN(Float16ToFloat64(0x7e00)) {
		t.Error("F16 infinity or NaN converted wrong")
	}
}

func writeRawFile(t *testing.T, header string, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "raw.safetensors")
	contents := make([]byte, 8)
	binary.LittleEndian.PutUint64(contents, uint64(len(header)))
	contents = append(contents, header...)
	contents = append(contents, data...)
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadsBFloat16File(t *testing.T) {
	data := []byte{0x80, 0x3f, 0x00, 0xc0}
	path := writeRawFile(t, `{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}`, data)
	tensors, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tensors["w"].Values, []float64{1, -2}) {
		t.Errorf("BF16 values = %v", tensors["w"].Values)
	}
}

func TestBrokenFilesAreRejected(t *testing.T) {
	broken := map[string]string{
		"offsets past the end":   `{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`,
		"shape doesn't match":    `{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}`,
		"unknown data type":      `{"w":{"dtype":"I8","shape":[8],"data_offsets":[0,8]}}`,
		"not json":               `{"w":`,
		"one offset":             `{"w":{"dtype":"F32","shape":[2],"data_offsets":[0]}}`,
		"offsets in wrong order": `{"w":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}`,
	}
	for name, header := range broken {
		path := writeRawFile(t, header, make([]byte, 8))
		if _, err := Read(path); err == nil {
			t.Errorf("%s: file was accepted", name)
		}
	}
	path := filepath.Join(t.TempDir(), "short.safetensors")
	os.WriteFile(path, []byte{0xff, 0xff, 0, 0, 0, 0, 0, 0, '{', '}'}, 0o644)
	if _, err := Read(path); err == nil {
		t.Error("a header size bigger than the file was accepted")
	}
}
