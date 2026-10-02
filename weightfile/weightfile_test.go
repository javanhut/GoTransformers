package weightfile

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"transformer/parameter"
)

func makeParameters() []parameter.Parameter {
	return []parameter.Parameter{
		{Name: "layer1.weights", Values: []float64{0.1, -2.5, 1e-300, math.Pi, 0}},
		{Name: "layer1.biases", Values: []float64{7, -0.000123}},
		{Name: "empty", Values: []float64{}},
	}
}

func makeBlankCopy(parameters []parameter.Parameter) []parameter.Parameter {
	var blank []parameter.Parameter
	for _, current := range parameters {
		blank = append(blank, parameter.Parameter{Name: current.Name, Values: make([]float64, len(current.Values))})
	}
	return blank
}

func TestRoundTrips(t *testing.T) {
	formats := []struct {
		name string
		save func(string, []parameter.Parameter) error
		load func(string, []parameter.Parameter) error
	}{
		{"text", SaveText, LoadText},
		{"binary", SaveBinary, LoadBinary},
	}
	for _, format := range formats {
		saved := makeParameters()
		path := filepath.Join(t.TempDir(), "weights")
		if err := format.save(path, saved); err != nil {
			t.Fatalf("%s save: %v", format.name, err)
		}
		loaded := makeBlankCopy(saved)
		if err := format.load(path, loaded); err != nil {
			t.Fatalf("%s load: %v", format.name, err)
		}
		for i := range saved {
			if !reflect.DeepEqual(saved[i].Values, loaded[i].Values) {
				t.Errorf("%s: %s saved %v but loaded %v", format.name, saved[i].Name, saved[i].Values, loaded[i].Values)
			}
		}
	}
}

func TestLoadOnlySomeParameters(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := SaveBinary(path, makeParameters()); err != nil {
		t.Fatal(err)
	}
	onlyBiases := []parameter.Parameter{{Name: "layer1.biases", Values: make([]float64, 2)}}
	if err := LoadBinary(path, onlyBiases); err != nil {
		t.Fatal(err)
	}
	if onlyBiases[0].Values[0] != 7 {
		t.Errorf("loaded biases %v", onlyBiases[0].Values)
	}
}

func TestLoadErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.txt")
	if err := SaveText(path, makeParameters()); err != nil {
		t.Fatal(err)
	}

	wrongSize := []parameter.Parameter{{Name: "layer1.biases", Values: make([]float64, 3)}}
	if err := LoadText(path, wrongSize); err == nil {
		t.Error("loading 2 saved values into 3 slots did not fail")
	}
	missing := []parameter.Parameter{{Name: "layer2.weights", Values: make([]float64, 1)}}
	if err := LoadText(path, missing); err == nil {
		t.Error("loading a parameter that was never saved did not fail")
	}
	if err := LoadBinary(path, missing); err == nil {
		t.Error("reading a text file as binary did not fail")
	}

	broken := filepath.Join(t.TempDir(), "broken.txt")
	os.WriteFile(broken, []byte("layer1.biases 3\n1 2\n"), 0o644)
	if _, err := ReadText(broken); err == nil {
		t.Error("a file that ends early did not fail")
	}

	badName := []parameter.Parameter{{Name: "has space", Values: []float64{1}}}
	if err := SaveText(path, badName); err == nil {
		t.Error("saving a name with a space did not fail")
	}
}
