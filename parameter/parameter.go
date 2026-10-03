package parameter

import "github.com/javanhut/GoTransformers/lowprecision"

type Parameter struct {
	Name             string
	Values           []float64
	CompressedValues *lowprecision.Rows
	GradientStorage  *[]float64
	Rows             int
	Columns          int
	UseAdamW         bool
	ReadOnly         bool
}

func WithGradients(name string, values []float64) Parameter {
	gradients := make([]float64, len(values))
	return Parameter{Name: name, Values: values, GradientStorage: &gradients}
}

func (current Parameter) IsMatrix() bool {
	return current.Rows > 1 && current.Columns > 1 && current.Rows*current.Columns == len(current.Values)
}

func (current Parameter) Gradients() []float64 {
	if current.GradientStorage == nil {
		return nil
	}
	if len(*current.GradientStorage) != len(current.Values) {
		*current.GradientStorage = make([]float64, len(current.Values))
	}
	return *current.GradientStorage
}

func (current Parameter) HasGradients() bool {
	return current.GradientStorage != nil && len(*current.GradientStorage) == len(current.Values)
}

func (current Parameter) ReleaseGradients() {
	if current.GradientStorage != nil {
		*current.GradientStorage = nil
	}
}

func ZeroGradients(parameters []Parameter) {
	for _, current := range parameters {
		if !current.HasGradients() {
			continue
		}
		gradients := *current.GradientStorage
		for i := range gradients {
			gradients[i] = 0
		}
	}
}

func ReleaseGradients(parameters []Parameter) {
	for _, current := range parameters {
		current.ReleaseGradients()
	}
}

func GradientBytes(parameters []Parameter) int {
	total := 0
	for _, current := range parameters {
		if current.HasGradients() {
			total += len(current.Values) * 8
		}
	}
	return total
}

func (current Parameter) Count() int {
	if current.Values == nil {
		return current.Rows * current.Columns
	}
	return len(current.Values)
}

func (current Parameter) IsCompressed() bool {
	return current.Values == nil && current.Rows*current.Columns > 0
}
