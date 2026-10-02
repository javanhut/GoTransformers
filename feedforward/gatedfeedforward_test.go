package feedforward

import (
	"testing"
	"transformer/gradientcheck"
	"transformer/vectormath"
)

func TestGatedFeedForwardGradients(t *testing.T) {
	blocks := map[string]*GatedFeedForward{
		"SwiGLU":        NewSwiGLU("swiglu", 4, 6),
		"ClampedSwiGLU": NewClampedSwiGLU("clamped", 4, 6, 0.4),
		"GeGLU":         NewGeGLU("geglu", 4, 6),
		"ReGLU":         NewReGLU("reglu", 4, 6),
	}
	for name, block := range blocks {
		inputs := vectormath.NewRandomMatrix(3, 4, -1, 1)
		for _, problem := range gradientcheck.Compare(block.Forward, block.Backward, block.Parameters(), inputs) {
			t.Errorf("%s: %s", name, problem)
		}
	}
}
