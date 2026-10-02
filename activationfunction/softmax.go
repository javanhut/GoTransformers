package activationfunction

import (
	"fmt"
	"math"
)

func Softmax(inputs []float64) []float64 {
	if len(inputs) == 0 {
		panic("Softmax: inputs is empty")
	}
	largest := inputs[0]
	for _, input := range inputs {
		if input > largest {
			largest = input
		}
	}
	outputs := make([]float64, len(inputs))
	total := 0.0
	for i, input := range inputs {
		outputs[i] = math.Exp(input - largest)
		total += outputs[i]
	}
	for i := range outputs {
		outputs[i] = outputs[i] / total
	}
	return outputs
}

func SoftmaxBackward(outputs []float64, outputGradients []float64) []float64 {
	if len(outputs) != len(outputGradients) {
		panic(fmt.Sprintf("SoftmaxBackward: got %d outputs but %d output gradients", len(outputs), len(outputGradients)))
	}
	weightedSum := 0.0
	for i := range outputs {
		weightedSum += outputGradients[i] * outputs[i]
	}
	inputGradients := make([]float64, len(outputs))
	for i := range outputs {
		inputGradients[i] = outputs[i] * (outputGradients[i] - weightedSum)
	}
	return inputGradients
}
