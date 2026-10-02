package perceptron

import (
	"fmt"
	"transformer/activationfunction"
)

type Perceptron struct {
	Weights    []float64
	Bias       float64
	Activation activationfunction.Activation
}

func (p Perceptron) Forward(inputs []float64) float64 {
	if p.Activation.Forward == nil {
		panic("Perceptron.Forward: Activation is not set")
	}
	if len(inputs) != len(p.Weights) {
		panic(fmt.Sprintf("Perceptron.Forward: got %d inputs but the perceptron has %d weights", len(inputs), len(p.Weights)))
	}
	sum := p.Bias
	for i, input := range inputs {
		sum += input * p.Weights[i]
	}
	return p.Activation.Forward(sum)
}
