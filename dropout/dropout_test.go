package dropout

import (
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"testing"
)

func TestZeroRateMeansNoDropout(t *testing.T) {
	if New(0) != nil {
		t.Fatal("a rate of 0 should give no dropout at all")
	}
	var nothing *Dropout
	inputs := vectormath.NewRandomMatrix(2, 3, -1, 1)
	if nothing.IsOn() || nothing.Forward(inputs).Values[0] != inputs.Values[0] || nothing.NextMask(4) != nil {
		t.Error("a nil dropout should pass values through unchanged")
	}
}

func TestInactiveDropoutChangesNothing(t *testing.T) {
	dropout := New(0.5)
	inputs := vectormath.NewRandomMatrix(4, 5, -1, 1)
	outputs := dropout.Forward(inputs)
	for i := range inputs.Values {
		if outputs.Values[i] != inputs.Values[i] {
			t.Fatal("dropout that isn't active should change nothing")
		}
	}
}

func TestActiveDropoutKeepsTheAverage(t *testing.T) {
	dropout := New(0.3)
	dropout.SetActive(true)
	inputs := vectormath.NewMatrix(1, 200000)
	for i := range inputs.Values {
		inputs.Values[i] = 1
	}
	outputs := dropout.Forward(inputs)
	zeros := 0
	for _, value := range outputs.Values {
		if value == 0 {
			zeros++
		} else if math.Abs(value-1/0.7) > 1e-12 {
			t.Fatalf("kept values should be scaled by 1/(1-rate), got %v", value)
		}
	}
	droppedFraction := float64(zeros) / float64(len(outputs.Values))
	average := vectormath.Sum(outputs.Values) / float64(len(outputs.Values))
	if math.Abs(droppedFraction-0.3) > 0.01 || math.Abs(average-1) > 0.01 {
		t.Errorf("dropped %.3f of the values with an average of %.3f, wanted about 0.3 and 1", droppedFraction, average)
	}
}

func TestBackwardUsesTheSameMask(t *testing.T) {
	dropout := New(0.5)
	dropout.SetActive(true)
	inputs := vectormath.NewRandomMatrix(3, 4, 1, 2)
	outputs := dropout.Forward(inputs)
	gradients := dropout.Backward(vectormath.NewRandomMatrix(3, 4, 1, 2))
	for i := range outputs.Values {
		if (outputs.Values[i] == 0) != (gradients.Values[i] == 0) {
			t.Fatalf("value %d: forward and backward used different masks", i)
		}
	}
}

func TestRepeatLastMasks(t *testing.T) {
	dropout := New(0.5)
	dropout.SetActive(true)
	dropout.StartPass()
	first := [][]float64{dropout.NextMask(8), dropout.NextMask(5)}
	dropout.RepeatLastMasks = true
	dropout.StartPass()
	second := [][]float64{dropout.NextMask(8), dropout.NextMask(5)}
	for which := range first {
		for i := range first[which] {
			if first[which][i] != second[which][i] {
				t.Fatalf("mask %d value %d changed even though RepeatLastMasks is on", which, i)
			}
		}
	}
}

func TestBadRatesPanic(t *testing.T) {
	for _, rate := range []float64{-0.1, 1, 1.5} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("rate %v did not panic", rate)
				}
			}()
			New(rate)
		}()
	}
}
