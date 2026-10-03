package optimizer

import (
	"math"
	"transformer/parameter"
	"transformer/vectormath"
)

type AdamW struct {
	LearningRate            float64
	MomentumDecay           float64
	SquaredGradientDecay    float64
	Epsilon                 float64
	WeightDecay             float64
	stepsTaken              int
	averageGradients        map[string][]float64
	averageSquaredGradients map[string][]float64
}

func NewAdamW(learningRate float64, weightDecay float64) *AdamW {
	return &AdamW{
		LearningRate:            learningRate,
		MomentumDecay:           0.9,
		SquaredGradientDecay:    0.95,
		Epsilon:                 1e-8,
		WeightDecay:             weightDecay,
		averageGradients:        map[string][]float64{},
		averageSquaredGradients: map[string][]float64{},
	}
}

func (adamW *AdamW) Update(parameters []parameter.Parameter) {
	defer vectormath.MarkWeightsChanged()
	checkNamesAreUnique(parameters)
	if adamW.averageGradients == nil {
		adamW.averageGradients = map[string][]float64{}
		adamW.averageSquaredGradients = map[string][]float64{}
	}
	adamW.stepsTaken++
	averageGradientCorrection := 1 - math.Pow(adamW.MomentumDecay, float64(adamW.stepsTaken))
	averageSquaredGradientCorrection := 1 - math.Pow(adamW.SquaredGradientDecay, float64(adamW.stepsTaken))

	for _, current := range parameters {
		checkGradientSize(current)
		averageGradient := rememberedValuesFor(adamW.averageGradients, current)
		averageSquaredGradient := rememberedValuesFor(adamW.averageSquaredGradients, current)

		shrinkFactor := 1.0
		if current.IsMatrix() {
			shrinkFactor = 1 - adamW.LearningRate*adamW.WeightDecay
		}

		for i := range current.Values {
			gradient := current.Gradients()[i]
			averageGradient[i] = adamW.MomentumDecay*averageGradient[i] + (1-adamW.MomentumDecay)*gradient
			averageSquaredGradient[i] = adamW.SquaredGradientDecay*averageSquaredGradient[i] + (1-adamW.SquaredGradientDecay)*gradient*gradient

			correctedAverageGradient := averageGradient[i] / averageGradientCorrection
			correctedAverageSquaredGradient := averageSquaredGradient[i] / averageSquaredGradientCorrection
			step := adamW.LearningRate * correctedAverageGradient / (math.Sqrt(correctedAverageSquaredGradient) + adamW.Epsilon)
			current.Values[i] = current.Values[i]*shrinkFactor - step
		}
	}
}
