package optimizer

import (
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

type Muon struct {
	LearningRate         float64
	Momentum             float64
	WeightDecay          float64
	UpdateRootMeanSquare float64
	AdamW                *AdamW
	momentumBuffers      map[string][]float64
}

func NewMuon(learningRate float64) *Muon {
	return &Muon{
		LearningRate:         learningRate,
		Momentum:             0.95,
		WeightDecay:          0.1,
		UpdateRootMeanSquare: 0.18,
		AdamW:                NewAdamW(learningRate, 0.1),
		momentumBuffers:      map[string][]float64{},
	}
}

func (muon *Muon) Update(parameters []parameter.Parameter) {
	defer vectormath.MarkWeightsChanged()
	checkNamesAreUnique(parameters)
	if muon.momentumBuffers == nil {
		muon.momentumBuffers = map[string][]float64{}
	}
	if muon.AdamW == nil {
		muon.AdamW = NewAdamW(muon.LearningRate, muon.WeightDecay)
	}

	var adamWParameters []parameter.Parameter
	for _, current := range parameters {
		checkGradientSize(current)
		if !current.IsMatrix() || current.UseAdamW {
			adamWParameters = append(adamWParameters, current)
			continue
		}

		momentumBuffer := rememberedValuesFor(muon.momentumBuffers, current)
		lookAhead := vectormath.NewMatrix(current.Rows, current.Columns)
		for i, gradient := range current.Gradients() {
			momentumBuffer[i] = muon.Momentum*momentumBuffer[i] + gradient
			lookAhead.Values[i] = muon.Momentum*momentumBuffer[i] + gradient
		}

		orthogonalUpdate := NewtonSchulz(lookAhead)
		largerSide := math.Max(float64(current.Rows), float64(current.Columns))
		updateScale := math.Sqrt(largerSide) * muon.UpdateRootMeanSquare
		shrinkFactor := 1 - muon.LearningRate*muon.WeightDecay
		for i := range current.Values {
			current.Values[i] = current.Values[i]*shrinkFactor - muon.LearningRate*orthogonalUpdate.Values[i]*updateScale
		}
	}

	if len(adamWParameters) > 0 {
		muon.AdamW.Update(adamWParameters)
	}
}

const numberOfFastNewtonSchulzSteps = 8

const numberOfSettlingNewtonSchulzSteps = 2

func NewtonSchulz(matrix vectormath.Matrix) vectormath.Matrix {
	needsTransposing := matrix.Rows > matrix.Columns
	current := matrix
	if needsTransposing {
		current = vectormath.Transpose(matrix)
	}

	size := vectormath.Magnitude(current.Values)
	if size == 0 {
		return vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	}
	current = vectormath.ScaleMatrix(current, 1/size)

	for step := 0; step < numberOfFastNewtonSchulzSteps+numberOfSettlingNewtonSchulzSteps; step++ {
		a, b, c := 3.4445, -4.7750, 2.0315
		if step >= numberOfFastNewtonSchulzSteps {
			a, b, c = 2, -1.5, 0.5
		}
		square := vectormath.MatrixTimesTransposed(current, current)
		squareOfSquare := vectormath.MatrixTimesMatrix(square, square)
		mixedSquares := vectormath.AddMatrices(vectormath.ScaleMatrix(square, b), vectormath.ScaleMatrix(squareOfSquare, c))
		current = vectormath.AddMatrices(vectormath.ScaleMatrix(current, a), vectormath.MatrixTimesMatrix(mixedSquares, current))
	}

	if needsTransposing {
		return vectormath.Transpose(current)
	}
	return current
}
