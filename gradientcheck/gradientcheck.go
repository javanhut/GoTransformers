package gradientcheck

import (
	"fmt"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

type ForwardFn func(inputs vectormath.Matrix) vectormath.Matrix

type BackwardFn func(outputGradients vectormath.Matrix) vectormath.Matrix

const stepSize = 1e-6

const allowedDifference = 1e-5

func Compare(forward ForwardFn, backward BackwardFn, parameters []parameter.Parameter, inputs vectormath.Matrix) []string {
	firstOutputs := forward(inputs)
	outputGradients := vectormath.NewRandomMatrix(firstOutputs.Rows, firstOutputs.Columns, -1, 1)

	lossNow := func() float64 {
		outputs := forward(inputs)
		return vectormath.DotProduct(outputs.Values, outputGradients.Values)
	}

	parameter.ZeroGradients(parameters)
	forward(inputs)
	inputGradients := backward(outputGradients)

	var problems []string
	for _, current := range parameters {
		for i := range current.Values {
			original := current.Values[i]
			current.Values[i] = original + stepSize
			higher := lossNow()
			current.Values[i] = original - stepSize
			lower := lossNow()
			current.Values[i] = original
			numerical := (higher - lower) / (2 * stepSize)
			if tooDifferent(numerical, current.Gradients()[i]) {
				problems = append(problems, fmt.Sprintf("%s[%d]: backward gave %v, finite difference gave %v", current.Name, i, current.Gradients()[i], numerical))
			}
		}
	}

	for i := range inputs.Values {
		original := inputs.Values[i]
		inputs.Values[i] = original + stepSize
		higher := lossNow()
		inputs.Values[i] = original - stepSize
		lower := lossNow()
		inputs.Values[i] = original
		numerical := (higher - lower) / (2 * stepSize)
		if tooDifferent(numerical, inputGradients.Values[i]) {
			problems = append(problems, fmt.Sprintf("input[%d]: backward gave %v, finite difference gave %v", i, inputGradients.Values[i], numerical))
		}
	}
	return problems
}

func tooDifferent(numerical float64, fromBackward float64) bool {
	return math.Abs(numerical-fromBackward) > allowedDifference*math.Max(1, math.Abs(numerical))
}
