package attention

import (
	"github.com/javanhut/GoTransformers/dropout"
	"github.com/javanhut/GoTransformers/gradientcheck"
	"github.com/javanhut/GoTransformers/vectormath"
	"testing"
)

func freezeMasks(weightsDropout *dropout.Dropout, forward func(vectormath.Matrix) vectormath.Matrix, inputs vectormath.Matrix) {
	weightsDropout.SetActive(true)
	forward(inputs)
	weightsDropout.RepeatLastMasks = true
}

func TestSelfAttentionDropoutGradients(t *testing.T) {
	setUps := map[string]Options{
		"plain":       {NumberOfHeads: 2, AttentionDropout: 0.3},
		"multi query": {NumberOfHeads: 4, NumberOfKeyValueHeads: 1, AttentionDropout: 0.3, UseRotaryPositions: true},
		"with sink":   {NumberOfHeads: 2, AttentionDropout: 0.3, UseAttentionSink: true, WindowSize: 3},
	}
	for name, options := range setUps {
		layer := withSink(withOptions(options))
		inputs := vectormath.NewRandomMatrix(6, 8, -1, 1)
		freezeMasks(layer.WeightsDropout, layer.Forward, inputs)
		for _, problem := range gradientcheck.Compare(layer.Forward, layer.Backward, layer.Parameters(), inputs) {
			t.Errorf("%s: %s", name, problem)
		}
	}
}

func TestCompressedAttentionDropoutGradients(t *testing.T) {
	compressed := newCompressed(CompressedOptions{NumberOfHeads: 2, CompressionRate: 2, WindowSize: 2, TopK: 2, UseAttentionSink: true, AttentionDropout: 0.3})
	compressed.IndexerLossWeight = 0
	inputs := vectormath.NewRandomMatrix(9, 8, -1, 1)
	freezeMasks(compressed.WeightsDropout, compressed.Forward, inputs)
	for _, problem := range gradientcheck.Compare(compressed.Forward, compressed.Backward, compressed.Parameters(), inputs) {
		t.Error(problem)
	}
}

func TestAttentionDropoutIsOffUnlessActive(t *testing.T) {
	plain := withOptions(Options{NumberOfHeads: 2})
	withDropout := withOptions(Options{NumberOfHeads: 2, AttentionDropout: 0.5})
	for i, current := range withDropout.Parameters() {
		copy(current.Values, plain.Parameters()[i].Values)
	}
	inputs := vectormath.NewRandomMatrix(5, 8, -1, 1)
	plainOutputs := plain.Forward(inputs)
	inactiveOutputs := withDropout.Forward(inputs)
	for i := range plainOutputs.Values {
		if plainOutputs.Values[i] != inactiveOutputs.Values[i] {
			t.Fatal("attention dropout changed the output while not active")
		}
	}
	withDropout.WeightsDropout.SetActive(true)
	activeOutputs := withDropout.Forward(inputs)
	changed := false
	for i := range plainOutputs.Values {
		if plainOutputs.Values[i] != activeOutputs.Values[i] {
			changed = true
		}
	}
	if !changed {
		t.Error("active attention dropout at rate 0.5 changed nothing")
	}
}
