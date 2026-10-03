package weightfile

import (
	"fmt"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"strings"
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

func checkCompressedValuesCanBeSaved(parameters []parameter.Parameter) error {
	for _, current := range parameters {
		if !current.IsCompressed() {
			continue
		}
		if current.CompressedValues == nil {
			return fmt.Errorf("parameter %q is compressed but does not hold its compressed values, so they can't be saved", current.Name)
		}
		if current.CompressedValues.NumberOfRows() != current.Rows || current.CompressedValues.Width != current.Columns {
			return fmt.Errorf("parameter %q is %dx%d but its compressed values are %dx%d", current.Name, current.Rows, current.Columns, current.CompressedValues.NumberOfRows(), current.CompressedValues.Width)
		}
	}
	return nil
}

func checkNothingIsCompressed(parameters []parameter.Parameter) error {
	for _, current := range parameters {
		if current.IsCompressed() {
			return fmt.Errorf("parameter %q is compressed, text files only hold plain numbers, save it with SaveBinary instead", current.Name)
		}
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

func checkSavedParameterFits(current parameter.Parameter, savedValues map[string][]float64, savedCompressedValues map[string]*lowprecision.Rows) error {
	values, foundValues := savedValues[current.Name]
	rows, foundRows := savedCompressedValues[current.Name]
	switch {
	case !foundValues && !foundRows:
		return fmt.Errorf("there is no saved parameter named %q", current.Name)
	case current.IsCompressed() && current.CompressedValues == nil:
		return fmt.Errorf("parameter %q is compressed but does not hold its compressed values, so they can't be loaded into it", current.Name)
	case current.IsCompressed() && !foundRows:
		return fmt.Errorf("parameter %q is compressed but the file holds it as plain numbers, load it into an uncompressed parameter and compress it again", current.Name)
	case current.IsCompressed() && (rows.NumberOfRows() != current.Rows || rows.Width != current.Columns):
		return fmt.Errorf("saved parameter %q is %dx%d compressed values but the parameter it's loading into is %dx%d", current.Name, rows.NumberOfRows(), rows.Width, current.Rows, current.Columns)
	case foundValues && len(values) != len(current.Values):
		return fmt.Errorf("saved parameter %q has %d values but the parameter it's loading into has %d", current.Name, len(values), len(current.Values))
	case foundRows && !current.IsCompressed() && rows.NumberOfRows()*rows.Width != len(current.Values):
		return fmt.Errorf("saved parameter %q has %d compressed values but the parameter it's loading into has %d", current.Name, rows.NumberOfRows()*rows.Width, len(current.Values))
	}
	return nil
}

func CopyIntoKeepingCompressed(parameters []parameter.Parameter, savedValues map[string][]float64, savedCompressedValues map[string]*lowprecision.Rows) error {
	for _, current := range parameters {
		if err := checkSavedParameterFits(current, savedValues, savedCompressedValues); err != nil {
			return err
		}
	}
	for _, current := range parameters {
		values, foundValues := savedValues[current.Name]
		switch {
		case current.IsCompressed():
			*current.CompressedValues = *savedCompressedValues[current.Name]
		case foundValues:
			copy(current.Values, values)
		default:
			copy(current.Values, savedCompressedValues[current.Name].AllValues())
		}
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
	savedValues, savedCompressedValues, err := ReadBinaryKeepingCompressed(path)
	if err != nil {
		return err
	}
	return CopyIntoKeepingCompressed(parameters, savedValues, savedCompressedValues)
}
