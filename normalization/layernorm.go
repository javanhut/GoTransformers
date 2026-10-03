package normalization

import (
	"fmt"
	"math"
	"transformer/parameter"
	"transformer/vectormath"
)

type LayerNorm struct {
	Name            string
	Weights         vectormath.Vector
	Biases          vectormath.Vector
	WeightGradients []float64
	BiasGradients   []float64
	Epsilon         float64

	lastNormalized        vectormath.Matrix
	lastStandardDeviation []float64
}

func NewLayerNorm(name string, vectorSize int) *LayerNorm {
	weights := vectormath.NewVector(vectorSize)
	for i := range weights {
		weights[i] = 1
	}
	return &LayerNorm{
		Name:            name,
		Weights:         weights,
		Biases:          vectormath.NewVector(vectorSize),
		WeightGradients: vectormath.NewVector(vectorSize),
		BiasGradients:   vectormath.NewVector(vectorSize),
		Epsilon:         1e-5,
	}
}

func (norm *LayerNorm) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	if inputs.Columns != len(norm.Weights) {
		panic(fmt.Sprintf("LayerNorm %q: each row has %d values but the norm was made for %d", norm.Name, inputs.Columns, len(norm.Weights)))
	}
	normalized := vectormath.NewMatrix(inputs.Rows, inputs.Columns)
	outputs := vectormath.NewMatrix(inputs.Rows, inputs.Columns)
	standardDeviations := make([]float64, inputs.Rows)
	size := float64(inputs.Columns)
	for row := 0; row < inputs.Rows; row++ {
		values := inputs.Row(row)
		mean := vectormath.Sum(values) / size
		variance := 0.0
		for _, value := range values {
			variance += (value - mean) * (value - mean)
		}
		variance = variance / size
		standardDeviation := math.Sqrt(variance + norm.Epsilon)
		standardDeviations[row] = standardDeviation

		normalizedRow := normalized.Row(row)
		outputRow := outputs.Row(row)
		for i, value := range values {
			normalizedRow[i] = (value - mean) / standardDeviation
			outputRow[i] = normalizedRow[i]*norm.Weights[i] + norm.Biases[i]
		}
	}
	norm.lastNormalized = normalized
	norm.lastStandardDeviation = standardDeviations
	return outputs
}

func (norm *LayerNorm) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	if norm.lastStandardDeviation == nil {
		panic(fmt.Sprintf("LayerNorm %q: call Forward before Backward", norm.Name))
	}
	if outputGradients.Rows != norm.lastNormalized.Rows || outputGradients.Columns != norm.lastNormalized.Columns {
		panic(fmt.Sprintf("LayerNorm %q: output gradients are %dx%d but the last Forward was %dx%d", norm.Name, outputGradients.Rows, outputGradients.Columns, norm.lastNormalized.Rows, norm.lastNormalized.Columns))
	}
	inputGradients := vectormath.NewMatrix(outputGradients.Rows, outputGradients.Columns)
	size := float64(outputGradients.Columns)
	for row := 0; row < outputGradients.Rows; row++ {
		normalizedRow := norm.lastNormalized.Row(row)
		gradients := outputGradients.Row(row)

		normalizedGradients := vectormath.NewVector(len(gradients))
		averageGradient := 0.0
		averageGradientTimesNormalized := 0.0
		for i := range gradients {
			norm.WeightGradients[i] += gradients[i] * normalizedRow[i]
			norm.BiasGradients[i] += gradients[i]
			normalizedGradients[i] = gradients[i] * norm.Weights[i]
			averageGradient += normalizedGradients[i] / size
			averageGradientTimesNormalized += normalizedGradients[i] * normalizedRow[i] / size
		}

		inputGradientRow := inputGradients.Row(row)
		for i := range gradients {
			inputGradientRow[i] = (normalizedGradients[i] - averageGradient - normalizedRow[i]*averageGradientTimesNormalized) / norm.lastStandardDeviation[row]
		}
	}
	return inputGradients
}

func (norm *LayerNorm) Parameters() []parameter.Parameter {
	return []parameter.Parameter{
		{Name: norm.Name + ".weights", Values: norm.Weights, GradientStorage: &norm.WeightGradients},
		{Name: norm.Name + ".biases", Values: norm.Biases, GradientStorage: &norm.BiasGradients},
	}
}
