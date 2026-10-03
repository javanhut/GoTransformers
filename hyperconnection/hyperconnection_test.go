package hyperconnection

import (
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/gradientcheck"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"testing"
)

type wrappedLayer struct {
	connection *HyperConnection
	layer      *perceptron.Layer
}

func (wrapped wrappedLayer) Forward(streams vectormath.Matrix) vectormath.Matrix {
	layerInput := wrapped.connection.LayerInput(streams)
	layerOutput := wrapped.layer.Forward(layerInput)
	return wrapped.connection.Combine(layerOutput)
}

func (wrapped wrappedLayer) Backward(newStreamGradients vectormath.Matrix) vectormath.Matrix {
	layerOutputGradients, streamGradientsFromCombine := wrapped.connection.BackwardCombine(newStreamGradients)
	layerInputGradients := wrapped.layer.Backward(layerOutputGradients)
	streamGradientsFromLayerInput := wrapped.connection.BackwardLayerInput(layerInputGradients)
	return vectormath.AddMatrices(streamGradientsFromCombine, streamGradientsFromLayerInput)
}

func (wrapped wrappedLayer) Parameters() []parameter.Parameter {
	return append(wrapped.connection.Parameters(), wrapped.layer.Parameters()...)
}

func newWrappedLayer(vectorSize int, numberOfStreams int) wrappedLayer {
	return wrappedLayer{
		connection: NewHyperConnection("connection", vectorSize, numberOfStreams),
		layer:      perceptron.NewLayer("layer", vectorSize, vectorSize, activationfunction.Tanh),
	}
}

func shakeUp(connection *HyperConnection) {
	connection.InputGate[0] = 0.7
	connection.ResidualGate[0] = -0.6
	connection.OutputGate[0] = 0.5
	for _, biases := range []vectormath.Vector{connection.InputBiases, connection.ResidualBiases, connection.OutputBiases} {
		for i := range biases {
			biases[i] += vectormath.RandomNumberBetween(-1, 1)
		}
	}
	for i := range connection.Norm.Weights {
		connection.Norm.Weights[i] = vectormath.RandomNumberBetween(0.5, 1.5)
	}
}

func TestGradients(t *testing.T) {
	for _, numberOfStreams := range []int{2, 4} {
		for _, sinkhornSteps := range []int{3, 20} {
			for _, shaken := range []bool{false, true} {
				wrapped := newWrappedLayer(3, numberOfStreams)
				wrapped.connection.SinkhornSteps = sinkhornSteps
				if shaken {
					shakeUp(wrapped.connection)
				}
				for range 3 {
					streams := vectormath.NewRandomMatrix(3, numberOfStreams*3, -1, 1)
					for _, problem := range gradientcheck.Compare(wrapped.Forward, wrapped.Backward, wrapped.Parameters(), streams) {
						t.Errorf("streams=%d steps=%d shaken=%v: %s", numberOfStreams, sinkhornSteps, shaken, problem)
					}
				}
			}
		}
	}
}

func worstSumErrors(connection *HyperConnection, numberOfTokens int) (float64, float64, bool) {
	worstRowError := 0.0
	worstColumnError := 0.0
	allNonNegative := true
	streams := connection.NumberOfStreams
	for token := range numberOfTokens {
		mixing := connection.LastResidualMixing(token)
		for row := range streams {
			rowSum := 0.0
			columnSum := 0.0
			for column := range streams {
				if mixing.Get(row, column) < 0 {
					allNonNegative = false
				}
				rowSum += mixing.Get(row, column)
				columnSum += mixing.Get(column, row)
			}
			worstRowError = math.Max(worstRowError, math.Abs(rowSum-1))
			worstColumnError = math.Max(worstColumnError, math.Abs(columnSum-1))
		}
	}
	return worstRowError, worstColumnError, allNonNegative
}

func TestResidualMixingIsDoublyStochastic(t *testing.T) {
	connection := NewHyperConnection("connection", 5, 4)
	connection.LayerInput(vectormath.NewRandomMatrix(6, 20, -2, 2))
	rowError, columnError, allNonNegative := worstSumErrors(connection, 6)
	if rowError > 1e-12 || columnError > 1e-3 || !allNonNegative {
		t.Errorf("after 20 steps: row error %v, column error %v, all non-negative %v", rowError, columnError, allNonNegative)
	}

	shaken := NewHyperConnection("connection", 5, 4)
	shakeUp(shaken)
	shaken.SinkhornSteps = 200
	shaken.LayerInput(vectormath.NewRandomMatrix(6, 20, -2, 2))
	rowError, columnError, allNonNegative = worstSumErrors(shaken, 6)
	if rowError > 1e-12 || columnError > 1e-6 || !allNonNegative {
		t.Errorf("after 200 steps: row error %v, column error %v, all non-negative %v", rowError, columnError, allNonNegative)
	}
}

func TestOneTokenMatchesManyTokens(t *testing.T) {
	wrapped := newWrappedLayer(4, 3)
	shakeUp(wrapped.connection)
	streams := vectormath.NewRandomMatrix(5, 12, -1, 1)
	allAtOnce := wrapped.Forward(streams)
	for token := range 5 {
		oneRow := vectormath.MatrixFromRows([]vectormath.Vector{streams.Row(token)})
		alone := wrapped.Forward(oneRow).Row(0)
		for i, value := range alone {
			if math.Abs(value-allAtOnce.Get(token, i)) > 1e-12 {
				t.Errorf("token %d value %d: alone %v, all at once %v", token, i, value, allAtOnce.Get(token, i))
			}
		}
	}
}

func TestStartsLikeANormalResidualConnection(t *testing.T) {
	connection := NewHyperConnection("connection", 4, 4)
	connection.InputGate[0] = 0
	connection.ResidualGate[0] = 0
	connection.OutputGate[0] = 0
	inputs := vectormath.NewRandomMatrix(3, 4, -1, 1)
	layerOutput := vectormath.NewRandomMatrix(3, 4, -1, 1)

	layerInput := connection.LayerInput(ExpandToStreams(inputs, 4))
	newStreams := connection.Combine(layerOutput)
	for token := range 3 {
		for i := range 4 {
			if math.Abs(layerInput.Get(token, i)-inputs.Get(token, i)) > 1e-12 {
				t.Errorf("layer input %v, want the original input %v", layerInput.Get(token, i), inputs.Get(token, i))
			}
			for stream := range 4 {
				want := inputs.Get(token, i) + layerOutput.Get(token, i)
				if math.Abs(newStreams.Get(token, stream*4+i)-want) > 1e-12 {
					t.Errorf("stream %d got %v, want input + layer output = %v", stream, newStreams.Get(token, stream*4+i), want)
				}
			}
		}
	}
}

func TestExpandAndCollapse(t *testing.T) {
	inputs := vectormath.NewRandomMatrix(3, 5, -1, 1)
	roundTrip := CollapseStreams(ExpandToStreams(inputs, 4), 4)
	for i := range inputs.Values {
		if math.Abs(roundTrip.Values[i]-inputs.Values[i]) > 1e-12 {
			t.Errorf("value %d: started %v, expanded and collapsed %v", i, inputs.Values[i], roundTrip.Values[i])
		}
	}

	expand := func(values vectormath.Matrix) vectormath.Matrix { return ExpandToStreams(values, 4) }
	expandBackward := func(gradients vectormath.Matrix) vectormath.Matrix { return ExpandToStreamsBackward(gradients, 4) }
	for _, problem := range gradientcheck.Compare(expand, expandBackward, nil, inputs) {
		t.Errorf("expand: %s", problem)
	}

	collapse := func(values vectormath.Matrix) vectormath.Matrix { return CollapseStreams(values, 4) }
	collapseBackward := func(gradients vectormath.Matrix) vectormath.Matrix { return CollapseStreamsBackward(gradients, 4) }
	for _, problem := range gradientcheck.Compare(collapse, collapseBackward, nil, vectormath.NewRandomMatrix(3, 20, -1, 1)) {
		t.Errorf("collapse: %s", problem)
	}
}

func TestParameterNamesAreUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, current := range NewHyperConnection("connection", 3, 2).Parameters() {
		if seen[current.Name] {
			t.Errorf("parameter name %q is used twice", current.Name)
		}
		seen[current.Name] = true
		if len(current.Gradients()) != len(current.Values) {
			t.Errorf("%s has %d values but %d gradients", current.Name, len(current.Values), len(current.Gradients()))
		}
	}
}
