package optimizer

import (
	"math"
	"testing"
	"transformer/parameter"
)

func minimize(chosenOptimizer Optimizer, steps int) []float64 {
	target := []float64{3, -2, 0.5}
	current := parameter.Parameter{
		Name:            "point",
		Values:          []float64{0, 0, 0},
		GradientStorage: pointerTo(make([]float64, 3)),
	}
	for step := 0; step < steps; step++ {
		for i := range current.Values {
			current.Gradients()[i] = 2 * (current.Values[i] - target[i])
		}
		chosenOptimizer.Update([]parameter.Parameter{current})
	}
	return current.Values
}

func TestOptimizersFindTheMinimum(t *testing.T) {
	optimizers := map[string]Optimizer{
		"SGD":             NewSGD(0.1),
		"SGDWithMomentum": NewSGDWithMomentum(0.05, 0.9),
		"Adam":            NewAdam(0.1),
	}
	for name, chosenOptimizer := range optimizers {
		result := minimize(chosenOptimizer, 1000)
		target := []float64{3, -2, 0.5}
		for i := range target {
			if math.Abs(result[i]-target[i]) > 1e-3 {
				t.Errorf("%s ended at %v, want %v", name, result, target)
				break
			}
		}
	}
}

func TestOptimizersWorkWithoutConstructor(t *testing.T) {
	minimize(&SGDWithMomentum{LearningRate: 0.01, Momentum: 0.9}, 5)
	minimize(&Adam{LearningRate: 0.01, MomentumDecay: 0.9, SquaredGradientDecay: 0.999, Epsilon: 1e-8}, 5)
}

func TestDuplicateNamesPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("two parameters with the same name did not panic")
		}
	}()
	first := parameter.Parameter{Name: "same", Values: []float64{1}, GradientStorage: pointerTo([]float64{1})}
	second := parameter.Parameter{Name: "same", Values: []float64{1}, GradientStorage: pointerTo([]float64{1})}
	NewAdam(0.1).Update([]parameter.Parameter{first, second})
}

func pointerTo(values []float64) *[]float64 {
	return &values
}
