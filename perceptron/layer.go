package perceptron

import (
	"fmt"
	"math"
	"transformer/activationfunction"
	"transformer/parameter"
	"transformer/vectormath"
)

type Layer struct {
	Name            string
	Weights         vectormath.Matrix
	Biases          vectormath.Vector
	Activation      activationfunction.Activation
	WeightGradients vectormath.Matrix
	BiasGradients   vectormath.Vector

	lastInputs         vectormath.Matrix
	lastPreActivations vectormath.Matrix
}

func NewLayer(name string, numberOfInputs int, numberOfOutputs int, activation activationfunction.Activation) *Layer {
	if numberOfInputs <= 0 || numberOfOutputs <= 0 {
		panic(fmt.Sprintf("NewLayer %q: needs at least 1 input and 1 output, got %d inputs and %d outputs", name, numberOfInputs, numberOfOutputs))
	}
	limit := math.Sqrt(6 / float64(numberOfInputs+numberOfOutputs))
	return &Layer{
		Name:            name,
		Weights:         vectormath.NewRandomMatrix(numberOfOutputs, numberOfInputs, -limit, limit),
		Biases:          vectormath.NewVector(numberOfOutputs),
		Activation:      activation,
		WeightGradients: vectormath.NewMatrix(numberOfOutputs, numberOfInputs),
		BiasGradients:   vectormath.NewVector(numberOfOutputs),
	}
}

func (layer *Layer) NumberOfInputs() int {
	return layer.Weights.Columns
}

func (layer *Layer) NumberOfOutputs() int {
	return layer.Weights.Rows
}

func (layer *Layer) checkSetUp() {
	if layer.Activation.Forward == nil || layer.Activation.Derivative == nil {
		panic(fmt.Sprintf("layer %q: Activation is not set", layer.Name))
	}
	if len(layer.Biases) != layer.NumberOfOutputs() {
		panic(fmt.Sprintf("layer %q: has %d neurons in Weights but %d Biases", layer.Name, layer.NumberOfOutputs(), len(layer.Biases)))
	}
	if layer.WeightGradients.Rows != layer.Weights.Rows || layer.WeightGradients.Columns != layer.Weights.Columns || len(layer.BiasGradients) != len(layer.Biases) {
		panic(fmt.Sprintf("layer %q: gradient storage doesn't match the weights, make layers with NewLayer", layer.Name))
	}
}

func (layer *Layer) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	layer.checkSetUp()
	if inputs.Columns != layer.NumberOfInputs() {
		panic(fmt.Sprintf("layer %q: each input row has %d values but the layer takes %d inputs", layer.Name, inputs.Columns, layer.NumberOfInputs()))
	}

	preActivations := vectormath.MatrixTimesTransposedWeights(inputs, layer.Weights)
	outputs := vectormath.NewMatrix(inputs.Rows, layer.NumberOfOutputs())
	for example := 0; example < inputs.Rows; example++ {
		examplePreActivations := preActivations.Row(example)
		exampleOutputs := outputs.Row(example)
		for neuron := range examplePreActivations {
			examplePreActivations[neuron] += layer.Biases[neuron]
			exampleOutputs[neuron] = layer.Activation.Forward(examplePreActivations[neuron])
		}
	}

	layer.lastInputs = inputs
	layer.lastPreActivations = preActivations
	return outputs
}

func (layer *Layer) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	if layer.lastPreActivations.Values == nil {
		panic(fmt.Sprintf("layer %q: call Forward before Backward", layer.Name))
	}
	if outputGradients.Rows != layer.lastPreActivations.Rows || outputGradients.Columns != layer.NumberOfOutputs() {
		panic(fmt.Sprintf("layer %q: output gradients are %dx%d but the last Forward produced %dx%d", layer.Name, outputGradients.Rows, outputGradients.Columns, layer.lastPreActivations.Rows, layer.lastPreActivations.Columns))
	}

	preActivationGradients := vectormath.NewMatrix(outputGradients.Rows, outputGradients.Columns)
	for example := 0; example < outputGradients.Rows; example++ {
		for neuron := 0; neuron < layer.NumberOfOutputs(); neuron++ {
			preActivation := layer.lastPreActivations.Get(example, neuron)
			gradient := outputGradients.Get(example, neuron) * layer.Activation.Derivative(preActivation)
			preActivationGradients.Set(example, neuron, gradient)
			layer.BiasGradients[neuron] += gradient
		}
	}

	weightGradients := vectormath.TransposedTimesMatrix(preActivationGradients, layer.lastInputs)
	for i := range weightGradients.Values {
		layer.WeightGradients.Values[i] += weightGradients.Values[i]
	}

	return vectormath.MatrixTimesWeights(preActivationGradients, layer.Weights)
}

func (layer *Layer) Parameters() []parameter.Parameter {
	return []parameter.Parameter{
		{Name: layer.Name + ".weights", Values: layer.Weights.Values, Gradients: layer.WeightGradients.Values, Rows: layer.Weights.Rows, Columns: layer.Weights.Columns},
		{Name: layer.Name + ".biases", Values: layer.Biases, Gradients: layer.BiasGradients},
	}
}
