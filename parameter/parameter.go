package parameter

type Parameter struct {
	Name      string
	Values    []float64
	Gradients []float64
	Rows      int
	Columns   int
	UseAdamW  bool
}

func (current Parameter) IsMatrix() bool {
	return current.Rows > 1 && current.Columns > 1 && current.Rows*current.Columns == len(current.Values)
}

func ZeroGradients(parameters []Parameter) {
	for _, current := range parameters {
		for i := range current.Gradients {
			current.Gradients[i] = 0
		}
	}
}
