package perceptron

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

type Layer struct {
	Name              string
	Weights           vectormath.Matrix
	Biases            vectormath.Vector
	Activation        activationfunction.Activation
	WeightGradients   []float64
	BiasGradients     []float64
	CompressedWeights *lowprecision.Rows
	Adapter           *LowRankAdapter
	FreezeBase        bool

	lastInputs         vectormath.Matrix
	lastPreActivations vectormath.Matrix
}

func NewLayer(name string, numberOfInputs int, numberOfOutputs int, activation activationfunction.Activation) *Layer {
	if numberOfInputs <= 0 || numberOfOutputs <= 0 {
		panic(fmt.Sprintf("NewLayer %q: needs at least 1 input and 1 output, got %d inputs and %d outputs", name, numberOfInputs, numberOfOutputs))
	}
	limit := math.Sqrt(6 / float64(numberOfInputs+numberOfOutputs))
	return &Layer{
		Name:       name,
		Weights:    vectormath.NewRandomMatrix(numberOfOutputs, numberOfInputs, -limit, limit),
		Biases:     vectormath.NewVector(numberOfOutputs),
		Activation: activation,
	}
}

func (layer *Layer) NumberOfInputs() int {
	return layer.Weights.Columns
}

func (layer *Layer) NumberOfOutputs() int {
	return layer.Weights.Rows
}

func (layer *Layer) IsCompressed() bool {
	return layer.CompressedWeights != nil
}

func (layer *Layer) CompressWeights(precision lowprecision.Precision) {
	if layer.IsCompressed() {
		panic(fmt.Sprintf("layer %q: weights are already compressed", layer.Name))
	}
	layer.CompressedWeights = lowprecision.RowsFromMatrix(layer.Weights, precision)
	layer.Weights = vectormath.Matrix{Rows: layer.Weights.Rows, Columns: layer.Weights.Columns}
	layer.WeightGradients = nil
	layer.BiasGradients = nil
	vectormath.MarkWeightsChanged()
}

func (layer *Layer) DecompressWeights() {
	if !layer.IsCompressed() {
		return
	}
	weights := vectormath.NewMatrix(layer.Weights.Rows, layer.Weights.Columns)
	for neuron := 0; neuron < weights.Rows; neuron++ {
		weights.SetRow(neuron, layer.CompressedWeights.Row(neuron))
	}
	layer.Weights = weights
	layer.CompressedWeights = nil
	vectormath.MarkWeightsChanged()
}

func (layer *Layer) WeightBytes() int {
	if layer.IsCompressed() {
		return layer.CompressedWeights.BytesUsed() + len(layer.Biases)*8
	}
	return (len(layer.Weights.Values) + len(layer.Biases)) * 8
}

func (layer *Layer) checkSetUp() {
	if layer.Activation.Forward == nil || layer.Activation.Derivative == nil {
		panic(fmt.Sprintf("layer %q: Activation is not set", layer.Name))
	}
	if len(layer.Biases) != layer.NumberOfOutputs() {
		panic(fmt.Sprintf("layer %q: has %d neurons in Weights but %d Biases", layer.Name, layer.NumberOfOutputs(), len(layer.Biases)))
	}
	if !layer.IsCompressed() && len(layer.Weights.Values) != layer.Weights.Rows*layer.Weights.Columns {
		panic(fmt.Sprintf("layer %q: Weights are %dx%d but hold %d values", layer.Name, layer.Weights.Rows, layer.Weights.Columns, len(layer.Weights.Values)))
	}
}

func (layer *Layer) makeGradients() {
	if len(layer.WeightGradients) != layer.Weights.Rows*layer.Weights.Columns {
		layer.WeightGradients = make([]float64, layer.Weights.Rows*layer.Weights.Columns)
	}
	if len(layer.BiasGradients) != len(layer.Biases) {
		layer.BiasGradients = make([]float64, len(layer.Biases))
	}
}

func (layer *Layer) ReleaseGradients() {
	layer.WeightGradients = nil
	layer.BiasGradients = nil
}

func (layer *Layer) multiplyByCompressedWeights(inputs vectormath.Matrix) vectormath.Matrix {
	preActivations := vectormath.NewMatrix(inputs.Rows, layer.NumberOfOutputs())
	vectormath.SplitAcrossThreads(layer.NumberOfOutputs(), inputs.Rows*inputs.Columns, func(firstNeuron int, lastNeuron int) {
		for neuron := firstNeuron; neuron < lastNeuron; neuron++ {
			for example := 0; example < inputs.Rows; example++ {
				preActivations.Set(example, neuron, layer.CompressedWeights.DotRow(neuron, inputs.Row(example)))
			}
		}
	})
	return preActivations
}

func (layer *Layer) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	layer.checkSetUp()
	if inputs.Columns != layer.NumberOfInputs() {
		panic(fmt.Sprintf("layer %q: each input row has %d values but the layer takes %d inputs", layer.Name, inputs.Columns, layer.NumberOfInputs()))
	}

	var preActivations vectormath.Matrix
	if layer.IsCompressed() {
		preActivations = layer.multiplyByCompressedWeights(inputs)
	} else {
		preActivations = vectormath.MatrixTimesTransposedWeights(inputs, layer.Weights)
	}
	if layer.Adapter != nil {
		layer.lastInputs = inputs
		layer.adapterForward(inputs, preActivations)
	}
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
	if layer.IsCompressed() && !layer.FreezeBase {
		panic(fmt.Sprintf("layer %q: weights are compressed for running the model, call DecompressWeights before training, or add a low-rank adapter to train around them", layer.Name))
	}
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
		}
	}

	if !layer.FreezeBase {
		layer.makeGradients()
		for example := 0; example < preActivationGradients.Rows; example++ {
			for neuron, gradient := range preActivationGradients.Row(example) {
				layer.BiasGradients[neuron] += gradient
			}
		}
		weightGradients := vectormath.TransposedTimesMatrix(preActivationGradients, layer.lastInputs)
		for i := range weightGradients.Values {
			layer.WeightGradients[i] += weightGradients.Values[i]
		}
	}

	var inputGradients vectormath.Matrix
	if layer.IsCompressed() {
		inputGradients = layer.multiplyGradientsByCompressedWeights(preActivationGradients)
	} else {
		inputGradients = vectormath.MatrixTimesWeights(preActivationGradients, layer.Weights)
	}
	if layer.Adapter != nil {
		inputGradients = vectormath.AddMatrices(inputGradients, layer.adapterBackward(preActivationGradients))
	}
	return inputGradients
}

func (layer *Layer) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	switch {
	case layer.IsCompressed():
		parameters = []parameter.Parameter{
			{Name: layer.Name + ".weights", Rows: layer.Weights.Rows, Columns: layer.Weights.Columns, ReadOnly: true},
			{Name: layer.Name + ".biases", Values: layer.Biases, ReadOnly: true},
		}
	case layer.FreezeBase:
		parameters = []parameter.Parameter{
			{Name: layer.Name + ".weights", Values: layer.Weights.Values, Rows: layer.Weights.Rows, Columns: layer.Weights.Columns, ReadOnly: true},
			{Name: layer.Name + ".biases", Values: layer.Biases, ReadOnly: true},
		}
	default:
		parameters = []parameter.Parameter{
			{Name: layer.Name + ".weights", Values: layer.Weights.Values, GradientStorage: &layer.WeightGradients, Rows: layer.Weights.Rows, Columns: layer.Weights.Columns},
			{Name: layer.Name + ".biases", Values: layer.Biases, GradientStorage: &layer.BiasGradients},
		}
	}
	if layer.Adapter != nil {
		parameters = append(parameters, layer.adapterParameters()...)
	}
	return parameters
}

func (layer *Layer) SetWeights(values []float64) error {
	if len(values) != layer.Weights.Rows*layer.Weights.Columns {
		return fmt.Errorf("layer %q: got %d weight values but it holds %dx%d", layer.Name, len(values), layer.Weights.Rows, layer.Weights.Columns)
	}
	if layer.IsCompressed() {
		newWeights := vectormath.Matrix{Rows: layer.Weights.Rows, Columns: layer.Weights.Columns, Values: values}
		layer.CompressedWeights = lowprecision.RowsFromMatrix(newWeights, layer.CompressedWeights.Precision)
	} else {
		copy(layer.Weights.Values, values)
	}
	vectormath.MarkWeightsChanged()
	return nil
}

func (layer *Layer) SetBiases(values []float64) error {
	if len(values) != len(layer.Biases) {
		return fmt.Errorf("layer %q: got %d bias values but it holds %d", layer.Name, len(values), len(layer.Biases))
	}
	copy(layer.Biases, values)
	return nil
}
