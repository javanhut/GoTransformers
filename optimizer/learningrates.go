package optimizer

import "fmt"

func LearningRates(chosenOptimizer Optimizer) []float64 {
	switch current := chosenOptimizer.(type) {
	case *SGD:
		return []float64{current.LearningRate}
	case *SGDWithMomentum:
		return []float64{current.LearningRate}
	case *Adam:
		return []float64{current.LearningRate}
	case *AdamW:
		return []float64{current.LearningRate}
	case *Muon:
		if current.AdamW == nil {
			return []float64{current.LearningRate}
		}
		return []float64{current.LearningRate, current.AdamW.LearningRate}
	}
	panic(fmt.Sprintf("optimizer.LearningRates: don't know where %T keeps its learning rate", chosenOptimizer))
}

func SetLearningRates(chosenOptimizer Optimizer, learningRates []float64) {
	switch current := chosenOptimizer.(type) {
	case *SGD:
		current.LearningRate = learningRates[0]
	case *SGDWithMomentum:
		current.LearningRate = learningRates[0]
	case *Adam:
		current.LearningRate = learningRates[0]
	case *AdamW:
		current.LearningRate = learningRates[0]
	case *Muon:
		current.LearningRate = learningRates[0]
		if current.AdamW != nil && len(learningRates) > 1 {
			current.AdamW.LearningRate = learningRates[1]
		}
	default:
		panic(fmt.Sprintf("optimizer.SetLearningRates: don't know where %T keeps its learning rate", chosenOptimizer))
	}
}

func ScaleLearningRates(chosenOptimizer Optimizer, peakLearningRates []float64, fraction float64) {
	scaled := make([]float64, len(peakLearningRates))
	for i, peak := range peakLearningRates {
		scaled[i] = peak * fraction
	}
	SetLearningRates(chosenOptimizer, scaled)
}
