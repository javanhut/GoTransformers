package perceptron

import (
	"math"
	"testing"
	"transformer/activationfunction"
	"transformer/lossfunction"
	"transformer/optimizer"
	"transformer/parameter"
	"transformer/vectormath"
)

func expectPanic(t *testing.T, name string, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s did not panic", name)
		}
	}()
	function()
}

func TestPerceptronInputChecks(t *testing.T) {
	p := Perceptron{Weights: []float64{1, 2}, Activation: activationfunction.ReLU}
	expectPanic(t, "too many inputs", func() { p.Forward([]float64{1, 2, 3}) })
	expectPanic(t, "too few inputs", func() { p.Forward([]float64{1}) })
	expectPanic(t, "no activation", func() { Perceptron{Weights: []float64{1}}.Forward([]float64{1}) })
}

func TestLayerMatchesPerceptron(t *testing.T) {
	layer := NewLayer("layer", 3, 1, activationfunction.ReLU)
	layer.Weights.SetRow(0, vectormath.Vector{0.5, -0.2, 0.8})
	layer.Biases[0] = 0.1
	p := Perceptron{Weights: []float64{0.5, -0.2, 0.8}, Bias: 0.1, Activation: activationfunction.ReLU}

	inputs := vectormath.Vector{1, 2, 3}
	layerOutput := layer.Forward(vectormath.MatrixFromRows([]vectormath.Vector{inputs})).Get(0, 0)
	if layerOutput != p.Forward(inputs) {
		t.Errorf("layer gave %v, perceptron gave %v", layerOutput, p.Forward(inputs))
	}
}

func TestLayerInputChecks(t *testing.T) {
	layer := NewLayer("layer", 3, 2, activationfunction.Tanh)
	expectPanic(t, "wrong number of inputs", func() { layer.Forward(vectormath.NewMatrix(1, 4)) })
	expectPanic(t, "Backward before Forward", func() { NewLayer("fresh", 2, 2, activationfunction.Tanh).Backward(vectormath.NewMatrix(1, 2)) })
	layer.Forward(vectormath.NewMatrix(2, 3))
	expectPanic(t, "wrong gradient shape", func() { layer.Backward(vectormath.NewMatrix(1, 2)) })
}

func checkGradients(t *testing.T, network *MultiLayerPerceptron, inputs vectormath.Matrix, targets vectormath.Matrix, loss lossfunction.Loss) {
	t.Helper()
	const h = 1e-6
	parameters := network.Parameters()
	parameter.ZeroGradients(parameters)
	predictions := network.Forward(inputs)
	inputGradients := network.Backward(loss.Gradient(predictions, targets))

	lossNow := func() float64 { return loss.Calculate(network.Forward(inputs), targets) }

	for _, current := range parameters {
		for i := range current.Values {
			original := current.Values[i]
			current.Values[i] = original + h
			higher := lossNow()
			current.Values[i] = original - h
			lower := lossNow()
			current.Values[i] = original
			numerical := (higher - lower) / (2 * h)
			if math.Abs(numerical-current.Gradients[i]) > 1e-5 {
				t.Errorf("%s[%d]: backprop gradient %v, finite difference %v", current.Name, i, current.Gradients[i], numerical)
			}
		}
	}

	for i := range inputs.Values {
		original := inputs.Values[i]
		inputs.Values[i] = original + h
		higher := lossNow()
		inputs.Values[i] = original - h
		lower := lossNow()
		inputs.Values[i] = original
		numerical := (higher - lower) / (2 * h)
		if math.Abs(numerical-inputGradients.Values[i]) > 1e-5 {
			t.Errorf("input %d: backprop gradient %v, finite difference %v", i, inputGradients.Values[i], numerical)
		}
	}
}

func TestMultiLayerPerceptronGradients(t *testing.T) {
	inputs := vectormath.MatrixFromRows([]vectormath.Vector{{0.5, -1.0, 0.25}, {1.5, 0.3, -0.7}})

	network := NewMultiLayerPerceptron([]int{3, 4, 2}, activationfunction.Tanh, activationfunction.Sigmoid)
	targets := vectormath.MatrixFromRows([]vectormath.Vector{{0, 1}, {1, 0}})
	checkGradients(t, network, inputs, targets, lossfunction.MeanSquaredError)

	network = NewMultiLayerPerceptron([]int{3, 5, 4, 3}, activationfunction.GELU, activationfunction.Linear)
	targets = vectormath.MatrixFromRows([]vectormath.Vector{{0, 0, 1}, {0.2, 0.8, 0}})
	checkGradients(t, network, inputs, targets, lossfunction.SoftmaxCrossEntropy)
}

func TestMultiLayerPerceptronLearnsXOR(t *testing.T) {
	inputs := vectormath.MatrixFromRows([]vectormath.Vector{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
	targets := vectormath.MatrixFromRows([]vectormath.Vector{{0}, {1}, {1}, {0}})
	network := NewMultiLayerPerceptron([]int{2, 8, 1}, activationfunction.Tanh, activationfunction.Sigmoid)
	adam := optimizer.NewAdam(0.05)

	for step := 0; step < 2000; step++ {
		network.TrainStep(inputs, targets, lossfunction.MeanSquaredError, adam)
	}

	for row := 0; row < inputs.Rows; row++ {
		prediction := network.Predict(inputs.Row(row))[0]
		if math.Abs(prediction-targets.Get(row, 0)) > 0.1 {
			t.Errorf("XOR%v = %v, want %v", inputs.Row(row), prediction, targets.Get(row, 0))
		}
	}
}

func TestMismatchedLayersPanic(t *testing.T) {
	network := &MultiLayerPerceptron{Layers: []*Layer{
		NewLayer("first", 2, 3, activationfunction.ReLU),
		NewLayer("second", 4, 1, activationfunction.ReLU),
	}}
	expectPanic(t, "layers that don't connect", func() { network.Forward(vectormath.NewMatrix(1, 2)) })
}
