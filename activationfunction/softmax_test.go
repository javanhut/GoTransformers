package activationfunction

import (
	"math"
	"testing"
)

func TestSoftmaxAddsUpToOne(t *testing.T) {
	outputs := Softmax([]float64{1, 2, 3, -4})
	total := 0.0
	for _, output := range outputs {
		total += output
	}
	if math.Abs(total-1) > 1e-12 {
		t.Errorf("softmax outputs add up to %v, want 1", total)
	}
	if !(outputs[2] > outputs[1] && outputs[1] > outputs[0] && outputs[0] > outputs[3]) {
		t.Errorf("softmax changed the order of the inputs: %v", outputs)
	}
}

func TestSoftmaxHandlesHugeInputs(t *testing.T) {
	outputs := Softmax([]float64{1000, 1000, math.Inf(-1)})
	if outputs[0] != 0.5 || outputs[1] != 0.5 || outputs[2] != 0 {
		t.Errorf("Softmax([1000 1000 -Inf]) = %v, want [0.5 0.5 0]", outputs)
	}
}

func TestSoftmaxBackwardMatchesFiniteDifference(t *testing.T) {
	const h = 1e-6
	inputs := []float64{0.3, -1.2, 2.0, 0.7}
	outputGradients := []float64{0.5, -0.1, 0.25, 1.5}
	lossOf := func(values []float64) float64 {
		outputs := Softmax(values)
		loss := 0.0
		for i := range outputs {
			loss += outputs[i] * outputGradients[i]
		}
		return loss
	}

	inputGradients := SoftmaxBackward(Softmax(inputs), outputGradients)
	for i := range inputs {
		higher := append([]float64{}, inputs...)
		lower := append([]float64{}, inputs...)
		higher[i] += h
		lower[i] -= h
		numerical := (lossOf(higher) - lossOf(lower)) / (2 * h)
		if math.Abs(numerical-inputGradients[i]) > 1e-6 {
			t.Errorf("input %d: SoftmaxBackward = %v, finite difference = %v", i, inputGradients[i], numerical)
		}
	}
}
