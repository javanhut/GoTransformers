package feedforward

import (
	"fmt"
	"transformer/activationfunction"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/vectormath"
)

type GatedFeedForward struct {
	Name      string
	GateLayer *perceptron.Layer
	UpLayer   *perceptron.Layer
	DownLayer *perceptron.Layer

	lastGateOutputs vectormath.Matrix
	lastUpOutputs   vectormath.Matrix
}

func NewGatedFeedForward(name string, vectorSize int, hiddenSize int, gateActivation activationfunction.Activation, upActivation activationfunction.Activation) *GatedFeedForward {
	return &GatedFeedForward{
		Name:      name,
		GateLayer: perceptron.NewLayer(name+".gate", vectorSize, hiddenSize, gateActivation),
		UpLayer:   perceptron.NewLayer(name+".up", vectorSize, hiddenSize, upActivation),
		DownLayer: perceptron.NewLayer(name+".down", hiddenSize, vectorSize, activationfunction.Linear),
	}
}

func NewSwiGLU(name string, vectorSize int, hiddenSize int) *GatedFeedForward {
	return NewGatedFeedForward(name, vectorSize, hiddenSize, activationfunction.SiLU, activationfunction.Linear)
}

func NewClampedSwiGLU(name string, vectorSize int, hiddenSize int, limit float64) *GatedFeedForward {
	if limit <= 0 {
		panic(fmt.Sprintf("NewClampedSwiGLU %q: limit must be above 0, got %v", name, limit))
	}
	return NewGatedFeedForward(name, vectorSize, hiddenSize, activationfunction.ClampedSiLU(limit), activationfunction.ClampedLinear(limit))
}

func NewGeGLU(name string, vectorSize int, hiddenSize int) *GatedFeedForward {
	return NewGatedFeedForward(name, vectorSize, hiddenSize, activationfunction.GELU, activationfunction.Linear)
}

func NewReGLU(name string, vectorSize int, hiddenSize int) *GatedFeedForward {
	return NewGatedFeedForward(name, vectorSize, hiddenSize, activationfunction.ReLU, activationfunction.Linear)
}

func (feedForward *GatedFeedForward) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	gateOutputs := feedForward.GateLayer.Forward(inputs)
	upOutputs := feedForward.UpLayer.Forward(inputs)
	hidden := vectormath.MultiplyMatricesEach(gateOutputs, upOutputs)

	feedForward.lastGateOutputs = gateOutputs
	feedForward.lastUpOutputs = upOutputs
	return feedForward.DownLayer.Forward(hidden)
}

func (feedForward *GatedFeedForward) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	hiddenGradients := feedForward.DownLayer.Backward(outputGradients)
	gateGradients := vectormath.MultiplyMatricesEach(hiddenGradients, feedForward.lastUpOutputs)
	upGradients := vectormath.MultiplyMatricesEach(hiddenGradients, feedForward.lastGateOutputs)

	inputGradients := feedForward.GateLayer.Backward(gateGradients)
	return vectormath.AddMatrices(inputGradients, feedForward.UpLayer.Backward(upGradients))
}

func (feedForward *GatedFeedForward) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	parameters = append(parameters, feedForward.GateLayer.Parameters()...)
	parameters = append(parameters, feedForward.UpLayer.Parameters()...)
	parameters = append(parameters, feedForward.DownLayer.Parameters()...)
	return parameters
}
