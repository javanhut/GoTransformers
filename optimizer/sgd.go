package optimizer

import "transformer/parameter"

type SGD struct {
	LearningRate float64
}

func NewSGD(learningRate float64) *SGD {
	return &SGD{LearningRate: learningRate}
}

func (sgd *SGD) Update(parameters []parameter.Parameter) {
	for _, current := range parameters {
		checkGradientSize(current)
		for i := range current.Values {
			current.Values[i] -= sgd.LearningRate * current.Gradients[i]
		}
	}
}

type SGDWithMomentum struct {
	LearningRate float64
	Momentum     float64
	velocities   map[string][]float64
}

func NewSGDWithMomentum(learningRate float64, momentum float64) *SGDWithMomentum {
	return &SGDWithMomentum{
		LearningRate: learningRate,
		Momentum:     momentum,
		velocities:   map[string][]float64{},
	}
}

func (sgd *SGDWithMomentum) Update(parameters []parameter.Parameter) {
	checkNamesAreUnique(parameters)
	if sgd.velocities == nil {
		sgd.velocities = map[string][]float64{}
	}
	for _, current := range parameters {
		checkGradientSize(current)
		velocity := rememberedValuesFor(sgd.velocities, current)
		for i := range current.Values {
			velocity[i] = sgd.Momentum*velocity[i] - sgd.LearningRate*current.Gradients[i]
			current.Values[i] += velocity[i]
		}
	}
}
