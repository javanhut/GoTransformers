package perceptron

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/lossfunction"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
)

type MultiLayerPerceptron struct {
	Layers []*Layer
}

func NewMultiLayerPerceptron(layerSizes []int, hiddenActivation activationfunction.Activation, outputActivation activationfunction.Activation) *MultiLayerPerceptron {
	if len(layerSizes) < 2 {
		panic(fmt.Sprintf("NewMultiLayerPerceptron: needs at least an input size and an output size, got %v", layerSizes))
	}
	network := &MultiLayerPerceptron{}
	lastLayerIndex := len(layerSizes) - 2
	for i := 0; i <= lastLayerIndex; i++ {
		activation := hiddenActivation
		if i == lastLayerIndex {
			activation = outputActivation
		}
		name := fmt.Sprintf("layer%d", i+1)
		network.Layers = append(network.Layers, NewLayer(name, layerSizes[i], layerSizes[i+1], activation))
	}
	return network
}

func (network *MultiLayerPerceptron) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	if len(network.Layers) == 0 {
		panic("MultiLayerPerceptron.Forward: network has no layers")
	}
	values := inputs
	for _, layer := range network.Layers {
		values = layer.Forward(values)
	}
	return values
}

func (network *MultiLayerPerceptron) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	gradients := outputGradients
	for i := len(network.Layers) - 1; i >= 0; i-- {
		gradients = network.Layers[i].Backward(gradients)
	}
	return gradients
}

func (network *MultiLayerPerceptron) Predict(inputs vectormath.Vector) vectormath.Vector {
	oneRow := vectormath.MatrixFromRows([]vectormath.Vector{inputs})
	outputs := network.Forward(oneRow)
	return outputs.Row(0)
}

func (network *MultiLayerPerceptron) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	for _, layer := range network.Layers {
		parameters = append(parameters, layer.Parameters()...)
	}
	return parameters
}

func (network *MultiLayerPerceptron) TrainStep(inputs vectormath.Matrix, targets vectormath.Matrix, loss lossfunction.Loss, chosenOptimizer optimizer.Optimizer) float64 {
	parameters := network.Parameters()
	parameter.ZeroGradients(parameters)

	predictions := network.Forward(inputs)
	lossValue := loss.Calculate(predictions, targets)
	network.Backward(loss.Gradient(predictions, targets))

	chosenOptimizer.Update(parameters)
	return lossValue
}
