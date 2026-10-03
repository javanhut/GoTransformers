package weightfile

import (
	"fmt"
	"strings"
	"transformer/parameter"
	"transformer/vectormath"
)

func checkNamesCanBeSaved(parameters []parameter.Parameter) error {
	seen := map[string]bool{}
	for _, current := range parameters {
		if current.Name == "" {
			return fmt.Errorf("a parameter has no name, every saved parameter needs a name")
		}
		if strings.ContainsAny(current.Name, " \t\r\n") {
			return fmt.Errorf("parameter name %q has a space or newline in it, saved names can't have those", current.Name)
		}
		if seen[current.Name] {
			return fmt.Errorf("two parameters are both named %q, saved names must be different", current.Name)
		}
		seen[current.Name] = true
	}
	return nil
}

func CopyInto(parameters []parameter.Parameter, savedValues map[string][]float64) error {
	for _, current := range parameters {
		values, found := savedValues[current.Name]
		if !found {
			return fmt.Errorf("there is no saved parameter named %q", current.Name)
		}
		if len(values) != len(current.Values) {
			return fmt.Errorf("saved parameter %q has %d values but the parameter it's loading into has %d", current.Name, len(values), len(current.Values))
		}
	}
	for _, current := range parameters {
		copy(current.Values, savedValues[current.Name])
	}
	vectormath.MarkWeightsChanged()
	return nil
}

func LoadText(path string, parameters []parameter.Parameter) error {
	savedValues, err := ReadText(path)
	if err != nil {
		return err
	}
	return CopyInto(parameters, savedValues)
}

func LoadBinary(path string, parameters []parameter.Parameter) error {
	savedValues, err := ReadBinary(path)
	if err != nil {
		return err
	}
	return CopyInto(parameters, savedValues)
}
