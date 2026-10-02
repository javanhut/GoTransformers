package parameter

type Parameter struct {
	Name      string
	Values    []float64
	Gradients []float64
}

func ZeroGradients(parameters []Parameter) {
	for _, current := range parameters {
		for i := range current.Gradients {
			current.Gradients[i] = 0
		}
	}
}
