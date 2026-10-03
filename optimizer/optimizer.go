package optimizer

import (
	"fmt"
	"transformer/parameter"
)

type Optimizer interface {
	Update(parameters []parameter.Parameter)
}

func checkGradientSize(current parameter.Parameter) {
	if current.ReadOnly {
		panic(fmt.Sprintf("parameter %q is read only (its weights are compressed for running the model), decompress it before training", current.Name))
	}
	if len(current.Gradients()) != len(current.Values) {
		panic(fmt.Sprintf("parameter %q has %d values but %d gradients", current.Name, len(current.Values), len(current.Gradients())))
	}
}

func checkNamesAreUnique(parameters []parameter.Parameter) {
	seen := map[string]bool{}
	for _, current := range parameters {
		if current.Name == "" {
			panic("a parameter has no name, this optimizer remembers values by name so every parameter needs one")
		}
		if seen[current.Name] {
			panic(fmt.Sprintf("two parameters are both named %q, this optimizer remembers values by name so names must be different", current.Name))
		}
		seen[current.Name] = true
	}
}

func rememberedValuesFor(remembered map[string][]float64, current parameter.Parameter) []float64 {
	values, found := remembered[current.Name]
	if !found || len(values) != len(current.Values) {
		values = make([]float64, len(current.Values))
		remembered[current.Name] = values
	}
	return values
}
