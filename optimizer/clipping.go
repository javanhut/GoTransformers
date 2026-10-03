package optimizer

import (
	"github.com/javanhut/GoTransformers/parameter"
	"math"
)

func GradientNorm(parameters []parameter.Parameter) float64 {
	sumOfSquares := 0.0
	for _, current := range parameters {
		if !current.HasGradients() {
			continue
		}
		for _, gradient := range current.Gradients() {
			sumOfSquares += gradient * gradient
		}
	}
	return math.Sqrt(sumOfSquares)
}

func ClipGradients(parameters []parameter.Parameter, maximumNorm float64) float64 {
	norm := GradientNorm(parameters)
	if maximumNorm <= 0 || norm <= maximumNorm {
		return norm
	}
	scale := maximumNorm / (norm + 1e-6)
	for _, current := range parameters {
		if !current.HasGradients() {
			continue
		}
		gradients := current.Gradients()
		for i := range gradients {
			gradients[i] *= scale
		}
	}
	return norm
}
