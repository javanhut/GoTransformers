package normalization

import (
	"math"
	"testing"
	"transformer/gradientcheck"
	"transformer/vectormath"
)

func TestRMSNormGradients(t *testing.T) {
	norm := NewRMSNorm("norm", 5)
	for i := range norm.Weights {
		norm.Weights[i] = vectormath.RandomNumberBetween(0.5, 1.5)
	}
	inputs := vectormath.NewRandomMatrix(3, 5, -2, 2)
	for _, problem := range gradientcheck.Compare(norm.Forward, norm.Backward, norm.Parameters(), inputs) {
		t.Error(problem)
	}
}

func TestLayerNormGradients(t *testing.T) {
	norm := NewLayerNorm("norm", 5)
	for i := range norm.Weights {
		norm.Weights[i] = vectormath.RandomNumberBetween(0.5, 1.5)
		norm.Biases[i] = vectormath.RandomNumberBetween(-0.5, 0.5)
	}
	inputs := vectormath.NewRandomMatrix(3, 5, -2, 2)
	for _, problem := range gradientcheck.Compare(norm.Forward, norm.Backward, norm.Parameters(), inputs) {
		t.Error(problem)
	}
}

func TestRMSNormMakesRootMeanSquareOne(t *testing.T) {
	norm := NewRMSNorm("norm", 4)
	outputs := norm.Forward(vectormath.MatrixFromRows([]vectormath.Vector{{3, -6, 9, 12}}))
	row := outputs.Row(0)
	rootMeanSquare := math.Sqrt(vectormath.DotProduct(row, row) / 4)
	if math.Abs(rootMeanSquare-1) > 1e-6 {
		t.Errorf("root mean square after RMSNorm = %v, want 1", rootMeanSquare)
	}
}
