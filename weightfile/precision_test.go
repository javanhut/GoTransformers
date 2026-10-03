package weightfile

import (
	"bytes"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestVersionOneFixtureStillLoads(t *testing.T) {
	fixturePath := filepath.Join("testdata", "version1.weights")
	loaded := makeBlankCopy(makeParameters())
	if err := LoadBinary(fixturePath, loaded); err != nil {
		t.Fatal(err)
	}
	expected := makeParameters()
	for i := range expected {
		if !reflect.DeepEqual(expected[i].Values, loaded[i].Values) {
			t.Errorf("%s: fixture holds %v but loaded %v", expected[i].Name, expected[i].Values, loaded[i].Values)
		}
	}

	newPath := filepath.Join(t.TempDir(), "weights")
	if err := SaveBinary(newPath, makeParameters()); err != nil {
		t.Fatal(err)
	}
	fixtureBytes, _ := os.ReadFile(fixturePath)
	newBytes, _ := os.ReadFile(newPath)
	if !bytes.Equal(fixtureBytes, newBytes) {
		t.Error("saving at Float64 with nothing compressed no longer writes the same bytes as the old format")
	}
}

func makeRandomValues(count int, seed uint64) []float64 {
	random := rand.New(rand.NewPCG(seed, seed+1))
	values := make([]float64, count)
	for i := range values {
		values[i] = (random.Float64()*2 - 1) * math.Pow(10, float64(random.IntN(8)-4))
	}
	return values
}

func TestStoragePrecisionRoundTrips(t *testing.T) {
	numberOfValues := 20000
	original := []parameter.Parameter{
		{Name: "big", Values: makeRandomValues(numberOfValues, 1)},
		{Name: "small", Values: []float64{0.1, -2.5, 1e-300, math.Pi, 0, math.Inf(-1), 1e300}},
	}
	for _, precision := range []StoragePrecision{Float64, Float32, BFloat16} {
		path := filepath.Join(t.TempDir(), "weights")
		if err := SaveBinaryWithPrecision(path, original, precision); err != nil {
			t.Fatalf("%v: %v", precision, err)
		}
		loaded := makeBlankCopy(original)
		if err := LoadBinary(path, loaded); err != nil {
			t.Fatalf("%v: %v", precision, err)
		}
		for i := range original {
			for j, value := range original[i].Values {
				expected := precision.Round(value)
				if math.Float64bits(loaded[i].Values[j]) != math.Float64bits(expected) {
					t.Fatalf("%v: %s[%d] = %v loaded as %v, expected %v", precision, original[i].Name, j, value, loaded[i].Values[j], expected)
				}
			}
		}
		fileInfo, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		bytesPerValue := float64(fileInfo.Size()) / float64(numberOfValues+len(original[1].Values))
		wantedBytesPerValue := float64(precision.BytesPerValue())
		if bytesPerValue < wantedBytesPerValue || bytesPerValue > wantedBytesPerValue+0.05 {
			t.Errorf("%v: file uses %.3f bytes per value, expected about %v", precision, bytesPerValue, wantedBytesPerValue)
		}
	}
}

func TestFloat32RoundTripMatchesFloat32Conversion(t *testing.T) {
	values := makeRandomValues(1000, 7)
	for _, value := range values {
		if Float32.Round(value) != float64(float32(value)) {
			t.Fatalf("Float32.Round(%v) = %v", value, Float32.Round(value))
		}
	}
}

func TestBFloat16KnownValues(t *testing.T) {
	cases := []struct {
		value float64
		bits  uint16
	}{
		{1, 0x3F80},
		{-2, 0xC000},
		{0, 0x0000},
		{math.Copysign(0, -1), 0x8000},
		{1 + math.Ldexp(1, -8), 0x3F80},
		{1 + 3*math.Ldexp(1, -8), 0x3F82},
		{1 + math.Ldexp(1, -8) + math.Ldexp(1, -30), 0x3F81},
		{-(1 + math.Ldexp(1, -8) + math.Ldexp(1, -30)), 0xBF81},
		{math.MaxFloat32, 0x7F80},
		{(2 - math.Ldexp(1, -7)) * math.Ldexp(1, 127), 0x7F7F},
		{1e300, 0x7F80},
		{math.Inf(-1), 0xFF80},
		{math.Ldexp(1, -133), 0x0001},
		{math.Ldexp(1, -134), 0x0000},
		{1.5 * math.Ldexp(1, -134), 0x0001},
		{math.Ldexp(1, -126), 0x0080},
		{1e-300, 0x0000},
	}
	for _, current := range cases {
		if bits := float64ToBFloat16Bits(current.value); bits != current.bits {
			t.Errorf("%v became bfloat16 bits %#04x, expected %#04x", current.value, bits, current.bits)
		}
	}
	if bits := float64ToBFloat16Bits(math.NaN()); !math.IsNaN(bFloat16BitsToFloat64(bits)) {
		t.Errorf("NaN became bfloat16 bits %#04x, which is not NaN", bits)
	}
}

func TestBFloat16RoundsToNearestEven(t *testing.T) {
	values := makeRandomValues(200000, 3)
	for _, value := range values {
		bits := float64ToBFloat16Bits(value)
		rounded := bFloat16BitsToFloat64(bits)
		below := bFloat16BitsToFloat64(bits - 1)
		above := bFloat16BitsToFloat64(bits + 1)
		distance := math.Abs(value - rounded)
		if math.Abs(value-below) < distance || math.Abs(value-above) < distance {
			t.Fatalf("%v rounded to %v, but %v or %v is closer", value, rounded, below, above)
		}
		isTie := math.Abs(value-below) == distance || math.Abs(value-above) == distance
		if isTie && bits%2 != 0 {
			t.Fatalf("%v is halfway and rounded to odd bits %#04x", value, bits)
		}
	}
}

func makeCompressedParameter(name string, precision lowprecision.Precision, numberOfRows int, width int, seed uint64) parameter.Parameter {
	matrix := vectormath.Matrix{Rows: numberOfRows, Columns: width, Values: makeRandomValues(numberOfRows*width, seed)}
	rows := lowprecision.RowsFromMatrix(matrix, precision)
	return parameter.Parameter{Name: name, CompressedValues: rows, Rows: numberOfRows, Columns: width, ReadOnly: true}
}

func TestCompressedParametersRoundTrip(t *testing.T) {
	for _, compression := range []lowprecision.Precision{lowprecision.Float64, lowprecision.Float32, lowprecision.Int8, lowprecision.FP4} {
		for _, storage := range []StoragePrecision{Float64, BFloat16} {
			saved := []parameter.Parameter{
				makeCompressedParameter("layer.weights", compression, 9, 37, 11),
				{Name: "layer.biases", Values: makeRandomValues(9, 12)},
			}
			path := filepath.Join(t.TempDir(), "weights")
			if err := SaveBinaryWithPrecision(path, saved, storage); err != nil {
				t.Fatalf("%v stored at %v: %v", compression, storage, err)
			}

			_, savedCompressedValues, err := ReadBinaryKeepingCompressed(path)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(savedCompressedValues["layer.weights"].Snapshot(), saved[0].CompressedValues.Snapshot()) {
				t.Errorf("%v stored at %v: compressed rows changed on the way through the file", compression, storage)
			}

			compressedTarget := []parameter.Parameter{
				makeCompressedParameter("layer.weights", lowprecision.Int8, 9, 37, 99),
				{Name: "layer.biases", Values: make([]float64, 9)},
			}
			if err := LoadBinary(path, compressedTarget); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(compressedTarget[0].CompressedValues.Snapshot(), saved[0].CompressedValues.Snapshot()) {
				t.Errorf("%v stored at %v: loading into a compressed parameter did not give the same rows", compression, storage)
			}

			plainTarget := []parameter.Parameter{{Name: "layer.weights", Values: make([]float64, 9*37)}}
			if err := LoadBinary(path, plainTarget); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(plainTarget[0].Values, saved[0].CompressedValues.AllValues()) {
				t.Errorf("%v stored at %v: loading into a plain parameter did not expand the rows", compression, storage)
			}
		}
	}
}

func TestCompressedFileSizes(t *testing.T) {
	numberOfRows := 64
	width := 256
	wantedBytesPerValue := map[lowprecision.Precision]float64{
		lowprecision.Float32: 4,
		lowprecision.Int8:    1 + 4.0/16,
		lowprecision.FP4:     0.5 + 4.0/16,
	}
	for compression, wanted := range wantedBytesPerValue {
		path := filepath.Join(t.TempDir(), "weights")
		if err := SaveBinary(path, []parameter.Parameter{makeCompressedParameter("weights", compression, numberOfRows, width, 5)}); err != nil {
			t.Fatal(err)
		}
		fileInfo, _ := os.Stat(path)
		bytesPerValue := float64(fileInfo.Size()) / float64(numberOfRows*width)
		if bytesPerValue < wanted || bytesPerValue > wanted+0.01 {
			t.Errorf("%v: file uses %.3f bytes per value, expected about %v", compression, bytesPerValue, wanted)
		}
	}
}

func TestCompressedLoadErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights")
	if err := SaveBinary(path, makeParameters()); err != nil {
		t.Fatal(err)
	}
	compressedTarget := []parameter.Parameter{makeCompressedParameter("layer1.biases", lowprecision.Int8, 1, 2, 1)}
	if err := LoadBinary(path, compressedTarget); err == nil {
		t.Error("loading plain numbers into a compressed parameter did not fail")
	}

	compressedPath := filepath.Join(t.TempDir(), "compressed")
	if err := SaveBinary(compressedPath, []parameter.Parameter{makeCompressedParameter("weights", lowprecision.FP4, 3, 5, 1)}); err != nil {
		t.Fatal(err)
	}
	wrongShape := []parameter.Parameter{makeCompressedParameter("weights", lowprecision.FP4, 5, 3, 1)}
	if err := LoadBinary(compressedPath, wrongShape); err == nil {
		t.Error("loading 3x5 compressed rows into a 5x3 parameter did not fail")
	}
	if err := SaveText(filepath.Join(t.TempDir(), "text"), wrongShape); err == nil {
		t.Error("saving a compressed parameter as text did not fail")
	}
	missingRows := []parameter.Parameter{{Name: "weights", Rows: 3, Columns: 5}}
	if err := SaveBinary(filepath.Join(t.TempDir(), "missing"), missingRows); err == nil {
		t.Error("saving a compressed parameter without its compressed values did not fail")
	}
	if err := SaveBinaryWithPrecision(filepath.Join(t.TempDir(), "unknown"), makeParameters(), StoragePrecision(9)); err == nil {
		t.Error("saving at an unknown storage precision did not fail")
	}

	fileBytes, _ := os.ReadFile(compressedPath)
	for length := 0; length < len(fileBytes); length++ {
		cutPath := filepath.Join(t.TempDir(), "cut")
		os.WriteFile(cutPath, fileBytes[:length], 0o644)
		if _, _, err := ReadBinaryKeepingCompressed(cutPath); err == nil {
			t.Fatalf("a file cut to %d of %d bytes did not fail", length, len(fileBytes))
		}
	}

	precisionBytePosition := len(binaryFileMarkerWithPrecisions) + 4 + 4 + len("weights") + 1
	brokenBytes := append([]byte(nil), fileBytes...)
	brokenBytes[precisionBytePosition] = 77
	brokenPath := filepath.Join(t.TempDir(), "broken")
	os.WriteFile(brokenPath, brokenBytes, 0o644)
	if _, _, err := ReadBinaryKeepingCompressed(brokenPath); err == nil {
		t.Error("a compressed parameter with an unknown precision did not fail")
	}
	brokenBytes[precisionBytePosition-1] = 42
	os.WriteFile(brokenPath, brokenBytes, 0o644)
	if _, _, err := ReadBinaryKeepingCompressed(brokenPath); err == nil {
		t.Error("a parameter stored in an unknown way did not fail")
	}
}
