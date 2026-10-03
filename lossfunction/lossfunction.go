package lossfunction

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

type LossFn func(predictions vectormath.Matrix, targets vectormath.Matrix) float64

type LossGradientFn func(predictions vectormath.Matrix, targets vectormath.Matrix) vectormath.Matrix

type Loss struct {
	Name      string
	Calculate LossFn
	Gradient  LossGradientFn
}

func checkShapes(functionName string, predictions vectormath.Matrix, targets vectormath.Matrix) {
	if predictions.Rows != targets.Rows || predictions.Columns != targets.Columns {
		panic(fmt.Sprintf("%s: predictions are %dx%d but targets are %dx%d", functionName, predictions.Rows, predictions.Columns, targets.Rows, targets.Columns))
	}
	if len(predictions.Values) == 0 {
		panic(fmt.Sprintf("%s: predictions are empty", functionName))
	}
}

func meanSquaredError(predictions vectormath.Matrix, targets vectormath.Matrix) float64 {
	checkShapes("MeanSquaredError", predictions, targets)
	total := 0.0
	for i := range predictions.Values {
		difference := predictions.Values[i] - targets.Values[i]
		total += difference * difference
	}
	return total / float64(len(predictions.Values))
}

func meanSquaredErrorGradient(predictions vectormath.Matrix, targets vectormath.Matrix) vectormath.Matrix {
	checkShapes("MeanSquaredError", predictions, targets)
	gradients := vectormath.NewMatrix(predictions.Rows, predictions.Columns)
	count := float64(len(predictions.Values))
	for i := range predictions.Values {
		gradients.Values[i] = 2 * (predictions.Values[i] - targets.Values[i]) / count
	}
	return gradients
}

var MeanSquaredError = Loss{
	Name:      "MeanSquaredError",
	Calculate: meanSquaredError,
	Gradient:  meanSquaredErrorGradient,
}

func softmaxCrossEntropy(predictions vectormath.Matrix, targets vectormath.Matrix) float64 {
	checkShapes("SoftmaxCrossEntropy", predictions, targets)
	total := 0.0
	for row := 0; row < predictions.Rows; row++ {
		scores := predictions.Row(row)
		targetRow := targets.Row(row)
		largest := vectormath.Max(scores)
		sumOfExponentials := 0.0
		for _, score := range scores {
			sumOfExponentials += math.Exp(score - largest)
		}
		logOfSum := largest + math.Log(sumOfExponentials)
		for column, score := range scores {
			logProbability := score - logOfSum
			total -= targetRow[column] * logProbability
		}
	}
	return total / float64(predictions.Rows)
}

func softmaxCrossEntropyGradient(predictions vectormath.Matrix, targets vectormath.Matrix) vectormath.Matrix {
	checkShapes("SoftmaxCrossEntropy", predictions, targets)
	gradients := vectormath.NewMatrix(predictions.Rows, predictions.Columns)
	for row := 0; row < predictions.Rows; row++ {
		probabilities := activationfunction.Softmax(predictions.Row(row))
		targetRow := targets.Row(row)
		gradientRow := gradients.Row(row)
		for column := range probabilities {
			gradientRow[column] = (probabilities[column] - targetRow[column]) / float64(predictions.Rows)
		}
	}
	return gradients
}

var SoftmaxCrossEntropy = Loss{
	Name:      "SoftmaxCrossEntropy",
	Calculate: softmaxCrossEntropy,
	Gradient:  softmaxCrossEntropyGradient,
}
