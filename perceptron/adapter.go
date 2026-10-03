package perceptron

import (
	"fmt"
	"math"
	"transformer/dropout"
	"transformer/parameter"
	"transformer/vectormath"
)

type LowRankAdapter struct {
	Rank          int
	Alpha         float64
	Down          vectormath.Matrix
	Up            vectormath.Matrix
	DownGradients []float64
	UpGradients   []float64
	InputDropout  *dropout.Dropout

	lastDroppedInputs vectormath.Matrix
	lastDownOutputs   vectormath.Matrix
}

func (adapter *LowRankAdapter) Scale() float64 {
	return adapter.Alpha / float64(adapter.Rank)
}

func (layer *Layer) AddLowRankAdapter(rank int, alpha float64) {
	if layer.Adapter != nil {
		panic(fmt.Sprintf("layer %q: already has a low-rank adapter", layer.Name))
	}
	if rank < 1 || rank > layer.NumberOfInputs() || rank > layer.NumberOfOutputs() {
		panic(fmt.Sprintf("layer %q: adapter rank must be between 1 and %d, got %d", layer.Name, min(layer.NumberOfInputs(), layer.NumberOfOutputs()), rank))
	}
	limit := 1 / math.Sqrt(float64(layer.NumberOfInputs()))
	layer.Adapter = &LowRankAdapter{
		Rank:  rank,
		Alpha: alpha,
		Down:  vectormath.NewRandomMatrix(rank, layer.NumberOfInputs(), -limit, limit),
		Up:    vectormath.NewMatrix(layer.NumberOfOutputs(), rank),
	}
	layer.FreezeBase = true
	layer.ReleaseGradients()
}

func (layer *Layer) RemoveLowRankAdapter() {
	layer.Adapter = nil
	layer.FreezeBase = false
}

func (layer *Layer) MergeLowRankAdapter() {
	if layer.Adapter == nil {
		return
	}
	if layer.IsCompressed() {
		panic(fmt.Sprintf("layer %q: can't merge an adapter into compressed weights, call DecompressWeights first", layer.Name))
	}
	change := vectormath.ScaleMatrix(vectormath.MatrixTimesMatrix(layer.Adapter.Up, layer.Adapter.Down), layer.Adapter.Scale())
	for i, value := range change.Values {
		layer.Weights.Values[i] += value
	}
	layer.RemoveLowRankAdapter()
	vectormath.MarkWeightsChanged()
}

func (layer *Layer) adapterForward(inputs vectormath.Matrix, preActivations vectormath.Matrix) {
	adapter := layer.Adapter
	droppedInputs := adapter.InputDropout.Forward(inputs)
	downOutputs := vectormath.MatrixTimesTransposed(droppedInputs, adapter.Down)
	upOutputs := vectormath.MatrixTimesTransposed(downOutputs, adapter.Up)
	for i, value := range upOutputs.Values {
		preActivations.Values[i] += adapter.Scale() * value
	}
	adapter.lastDroppedInputs = droppedInputs
	adapter.lastDownOutputs = downOutputs
}

func (layer *Layer) adapterBackward(preActivationGradients vectormath.Matrix) vectormath.Matrix {
	adapter := layer.Adapter
	if len(adapter.UpGradients) != len(adapter.Up.Values) {
		adapter.UpGradients = make([]float64, len(adapter.Up.Values))
	}
	if len(adapter.DownGradients) != len(adapter.Down.Values) {
		adapter.DownGradients = make([]float64, len(adapter.Down.Values))
	}

	upGradients := vectormath.TransposedTimesMatrix(preActivationGradients, adapter.lastDownOutputs)
	for i, value := range upGradients.Values {
		adapter.UpGradients[i] += adapter.Scale() * value
	}

	downOutputGradients := vectormath.ScaleMatrix(vectormath.MatrixTimesMatrix(preActivationGradients, adapter.Up), adapter.Scale())
	downGradients := vectormath.TransposedTimesMatrix(downOutputGradients, adapter.lastDroppedInputs)
	for i, value := range downGradients.Values {
		adapter.DownGradients[i] += value
	}
	return adapter.InputDropout.Backward(vectormath.MatrixTimesMatrix(downOutputGradients, adapter.Down))
}

func (layer *Layer) adapterParameters() []parameter.Parameter {
	adapter := layer.Adapter
	return []parameter.Parameter{
		{Name: layer.Name + ".lora.down", Values: adapter.Down.Values, GradientStorage: &adapter.DownGradients, Rows: adapter.Down.Rows, Columns: adapter.Down.Columns, UseAdamW: true},
		{Name: layer.Name + ".lora.up", Values: adapter.Up.Values, GradientStorage: &adapter.UpGradients, Rows: adapter.Up.Rows, Columns: adapter.Up.Columns, UseAdamW: true},
	}
}

func (layer *Layer) multiplyGradientsByCompressedWeights(preActivationGradients vectormath.Matrix) vectormath.Matrix {
	inputGradients := vectormath.NewMatrix(preActivationGradients.Rows, layer.NumberOfInputs())
	vectormath.SplitAcrossThreads(preActivationGradients.Rows, layer.NumberOfOutputs()*layer.NumberOfInputs(), func(firstExample int, lastExample int) {
		for example := firstExample; example < lastExample; example++ {
			exampleGradients := preActivationGradients.Row(example)
			exampleInputGradients := inputGradients.Row(example)
			for neuron, gradient := range exampleGradients {
				layer.CompressedWeights.AddScaledRowTo(neuron, gradient, exampleInputGradients)
			}
		}
	})
	return inputGradients
}
