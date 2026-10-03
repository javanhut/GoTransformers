package normalization

import (
	"fmt"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

type RMSNorm struct {
	Name            string
	Weights         vectormath.Vector
	WeightGradients []float64
	Epsilon         float64

	lastInputs         vectormath.Matrix
	lastRootMeanSquare []float64
}

func NewRMSNorm(name string, vectorSize int) *RMSNorm {
	weights := vectormath.NewVector(vectorSize)
	for i := range weights {
		weights[i] = 1
	}
	return &RMSNorm{
		Name:            name,
		Weights:         weights,
		WeightGradients: vectormath.NewVector(vectorSize),
		Epsilon:         1e-6,
	}
}

func (norm *RMSNorm) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	if inputs.Columns != len(norm.Weights) {
		panic(fmt.Sprintf("RMSNorm %q: each row has %d values but the norm was made for %d", norm.Name, inputs.Columns, len(norm.Weights)))
	}
	outputs := vectormath.NewMatrix(inputs.Rows, inputs.Columns)
	rootMeanSquares := make([]float64, inputs.Rows)
	for row := 0; row < inputs.Rows; row++ {
		values := inputs.Row(row)
		meanOfSquares := vectormath.DotProduct(values, values) / float64(len(values))
		rootMeanSquare := math.Sqrt(meanOfSquares + norm.Epsilon)
		rootMeanSquares[row] = rootMeanSquare

		outputRow := outputs.Row(row)
		for i, value := range values {
			outputRow[i] = value / rootMeanSquare * norm.Weights[i]
		}
	}
	norm.lastInputs = inputs
	norm.lastRootMeanSquare = rootMeanSquares
	return outputs
}

func (norm *RMSNorm) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	if norm.lastRootMeanSquare == nil {
		panic(fmt.Sprintf("RMSNorm %q: call Forward before Backward", norm.Name))
	}
	if outputGradients.Rows != norm.lastInputs.Rows || outputGradients.Columns != norm.lastInputs.Columns {
		panic(fmt.Sprintf("RMSNorm %q: output gradients are %dx%d but the last Forward was %dx%d", norm.Name, outputGradients.Rows, outputGradients.Columns, norm.lastInputs.Rows, norm.lastInputs.Columns))
	}
	inputGradients := vectormath.NewMatrix(outputGradients.Rows, outputGradients.Columns)
	size := float64(outputGradients.Columns)
	for row := 0; row < outputGradients.Rows; row++ {
		values := norm.lastInputs.Row(row)
		gradients := outputGradients.Row(row)
		rootMeanSquare := norm.lastRootMeanSquare[row]

		weightedSum := 0.0
		for i := range values {
			weightedSum += gradients[i] * norm.Weights[i] * values[i]
		}

		inputGradientRow := inputGradients.Row(row)
		for i := range values {
			norm.WeightGradients[i] += gradients[i] * values[i] / rootMeanSquare
			inputGradientRow[i] = gradients[i]*norm.Weights[i]/rootMeanSquare - values[i]*weightedSum/(size*rootMeanSquare*rootMeanSquare*rootMeanSquare)
		}
	}
	return inputGradients
}

func (norm *RMSNorm) Parameters() []parameter.Parameter {
	return []parameter.Parameter{
		{Name: norm.Name + ".weights", Values: norm.Weights, GradientStorage: &norm.WeightGradients},
	}
}
