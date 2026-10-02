package main

import (
	"fmt"
	"transformer/activationfunction"
	"transformer/perceptron"
)

func main() {
	p := perceptron.Perceptron{
		Weights:    []float64{0.5, -0.2, 0.8},
		Bias:       0.1,
		Activation: activationfunction.ReLU,
	}

	output := p.Forward([]float64{1.0, 2.0, 3.0})

	fmt.Println("Output: ", output)
}
