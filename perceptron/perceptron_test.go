package perceptron

import (
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/dropout"
	"github.com/javanhut/GoTransformers/gradientcheck"
	"github.com/javanhut/GoTransformers/lossfunction"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"reflect"
	"testing"
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
			if math.Abs(numerical-current.Gradients()[i]) > 1e-5 {
				t.Errorf("%s[%d]: backprop gradient %v, finite difference %v", current.Name, i, current.Gradients()[i], numerical)
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

func TestNewLayerHasNoGradientMemoryUntilTraining(t *testing.T) {
	layer := NewLayer("layer", 30, 20, activationfunction.Tanh)
	if layer.WeightGradients != nil || layer.BiasGradients != nil {
		t.Fatal("a new layer should not hold gradient memory")
	}
	layer.Forward(vectormath.NewRandomMatrix(2, 30, -1, 1))
	if layer.WeightGradients != nil {
		t.Fatal("running Forward should not create gradient memory")
	}
	layer.Backward(vectormath.NewRandomMatrix(2, 20, -1, 1))
	if len(layer.WeightGradients) != 600 || len(layer.BiasGradients) != 20 {
		t.Fatalf("Backward should create gradient memory, got %d and %d values", len(layer.WeightGradients), len(layer.BiasGradients))
	}
	layer.ReleaseGradients()
	if layer.WeightGradients != nil {
		t.Fatal("ReleaseGradients should free the gradient memory")
	}
}

func TestParametersFetchedBeforeBackwardStillSeeGradients(t *testing.T) {
	network := NewMultiLayerPerceptron([]int{2, 3, 1}, activationfunction.Tanh, activationfunction.Sigmoid)
	parameters := network.Parameters()
	network.Forward(vectormath.MatrixFromRows([]vectormath.Vector{{1, 0}}))
	network.Backward(vectormath.MatrixFromRows([]vectormath.Vector{{1}}))
	nonZero := 0
	for _, current := range parameters {
		for _, gradient := range current.Gradients() {
			if gradient != 0 {
				nonZero++
			}
		}
	}
	if nonZero == 0 {
		t.Error("parameters fetched before Backward should see the gradients Backward made")
	}
}

func TestCompressedLayerMatchesWithinPrecision(t *testing.T) {
	allowedDifference := map[lowprecision.Precision]float64{
		lowprecision.Float32: 1e-5,
		lowprecision.Int8:    0.05,
		lowprecision.FP4:     0.6,
	}
	for precision, allowed := range allowedDifference {
		layer := NewLayer("layer", 64, 32, activationfunction.Linear)
		inputs := vectormath.NewRandomMatrix(3, 64, -1, 1)
		before := layer.Forward(inputs)
		bytesBefore := layer.WeightBytes()
		layer.CompressWeights(precision)
		after := layer.Forward(inputs)
		for i := range before.Values {
			if math.Abs(before.Values[i]-after.Values[i]) > allowed {
				t.Errorf("%v value %d: full precision %v, compressed %v", precision, i, before.Values[i], after.Values[i])
				break
			}
		}
		if layer.WeightBytes() >= bytesBefore {
			t.Errorf("%v: compressed weights use %d bytes, full precision used %d", precision, layer.WeightBytes(), bytesBefore)
		}
		expectPanic(t, "Backward on a compressed layer", func() { layer.Backward(vectormath.NewMatrix(3, 32)) })
		expectPanic(t, "optimizing a compressed layer", func() { optimizer.NewSGD(0.1).Update(layer.Parameters()) })
	}
}

func trainableOnly(parameters []parameter.Parameter) []parameter.Parameter {
	var trainable []parameter.Parameter
	for _, current := range parameters {
		if !current.ReadOnly {
			trainable = append(trainable, current)
		}
	}
	return trainable
}

func layerWithTrainedAdapter(precision lowprecision.Precision) *Layer {
	layer := NewLayer("layer", 6, 5, activationfunction.Tanh)
	if precision != lowprecision.Float64 {
		layer.CompressWeights(precision)
	}
	layer.AddLowRankAdapter(2, 4)
	for i := range layer.Adapter.Up.Values {
		layer.Adapter.Up.Values[i] = vectormath.RandomNumberBetween(-0.5, 0.5)
	}
	return layer
}

func TestLowRankAdapterGradients(t *testing.T) {
	for _, precision := range []lowprecision.Precision{lowprecision.Float64, lowprecision.Int8} {
		layer := layerWithTrainedAdapter(precision)
		inputs := vectormath.NewRandomMatrix(3, 6, -1, 1)
		for _, problem := range gradientcheck.Compare(layer.Forward, layer.Backward, trainableOnly(layer.Parameters()), inputs) {
			t.Errorf("%v base weights: %s", precision, problem)
		}
	}
}

func TestLowRankAdapterLeavesBaseAlone(t *testing.T) {
	layer := NewLayer("layer", 6, 5, activationfunction.Tanh)
	inputs := vectormath.NewRandomMatrix(3, 6, -1, 1)
	before := layer.Forward(inputs)
	layer.AddLowRankAdapter(2, 4)
	after := layer.Forward(inputs)
	if !reflect.DeepEqual(before.Values, after.Values) {
		t.Error("a new adapter should not change the output, because Up starts at zero")
	}
	layer.Backward(vectormath.NewRandomMatrix(3, 5, -1, 1))
	if layer.WeightGradients != nil || layer.BiasGradients != nil {
		t.Error("frozen base weights should not get gradient memory")
	}
	if len(layer.Adapter.UpGradients) != 10 || len(layer.Adapter.DownGradients) != 12 {
		t.Errorf("adapter gradients have %d and %d values", len(layer.Adapter.UpGradients), len(layer.Adapter.DownGradients))
	}
	if trainable := trainableOnly(layer.Parameters()); len(trainable) != 2 {
		t.Errorf("only the 2 adapter matrices should be trainable, got %d parameters", len(trainable))
	}
}

func TestMergingAnAdapterKeepsTheOutput(t *testing.T) {
	layer := layerWithTrainedAdapter(lowprecision.Float64)
	inputs := vectormath.NewRandomMatrix(3, 6, -1, 1)
	withAdapter := layer.Forward(inputs)
	layer.MergeLowRankAdapter()
	merged := layer.Forward(inputs)
	for i := range withAdapter.Values {
		if math.Abs(withAdapter.Values[i]-merged.Values[i]) > 1e-12 {
			t.Fatalf("value %d: with adapter %v, merged %v", i, withAdapter.Values[i], merged.Values[i])
		}
	}
	if layer.Adapter != nil || layer.FreezeBase {
		t.Error("merging should remove the adapter and unfreeze the base")
	}
	expectPanic(t, "merging into compressed weights", func() { layerWithTrainedAdapter(lowprecision.Int8).MergeLowRankAdapter() })
}

func TestAdapterDropoutGradients(t *testing.T) {
	layer := layerWithTrainedAdapter(lowprecision.Float64)
	layer.Adapter.InputDropout = dropout.New(0.4)
	layer.Adapter.InputDropout.SetActive(true)
	inputs := vectormath.NewRandomMatrix(3, 6, -1, 1)
	layer.Forward(inputs)
	layer.Adapter.InputDropout.RepeatLastMasks = true
	for _, problem := range gradientcheck.Compare(layer.Forward, layer.Backward, trainableOnly(layer.Parameters()), inputs) {
		t.Error(problem)
	}
}
