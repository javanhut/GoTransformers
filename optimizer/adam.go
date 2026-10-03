package optimizer

import (
	"math"
	"transformer/parameter"
	"transformer/vectormath"
)

type Adam struct {
	LearningRate            float64
	MomentumDecay           float64
	SquaredGradientDecay    float64
	Epsilon                 float64
	stepsTaken              int
	averageGradients        map[string][]float64
	averageSquaredGradients map[string][]float64
}

func NewAdam(learningRate float64) *Adam {
	return &Adam{
		LearningRate:            learningRate,
		MomentumDecay:           0.9,
		SquaredGradientDecay:    0.999,
		Epsilon:                 1e-8,
		averageGradients:        map[string][]float64{},
		averageSquaredGradients: map[string][]float64{},
	}
}

func (adam *Adam) Update(parameters []parameter.Parameter) {
	defer vectormath.MarkWeightsChanged()
	checkNamesAreUnique(parameters)
	if adam.averageGradients == nil {
		adam.averageGradients = map[string][]float64{}
		adam.averageSquaredGradients = map[string][]float64{}
	}
	adam.stepsTaken++
	averageGradientCorrection := 1 - math.Pow(adam.MomentumDecay, float64(adam.stepsTaken))
	averageSquaredGradientCorrection := 1 - math.Pow(adam.SquaredGradientDecay, float64(adam.stepsTaken))

	for _, current := range parameters {
		checkGradientSize(current)
		averageGradient := rememberedValuesFor(adam.averageGradients, current)
		averageSquaredGradient := rememberedValuesFor(adam.averageSquaredGradients, current)
		for i := range current.Values {
			gradient := current.Gradients()[i]
			averageGradient[i] = adam.MomentumDecay*averageGradient[i] + (1-adam.MomentumDecay)*gradient
			averageSquaredGradient[i] = adam.SquaredGradientDecay*averageSquaredGradient[i] + (1-adam.SquaredGradientDecay)*gradient*gradient

			correctedAverageGradient := averageGradient[i] / averageGradientCorrection
			correctedAverageSquaredGradient := averageSquaredGradient[i] / averageSquaredGradientCorrection
			current.Values[i] -= adam.LearningRate * correctedAverageGradient / (math.Sqrt(correctedAverageSquaredGradient) + adam.Epsilon)
		}
	}
}
