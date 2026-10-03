package transformer

import (
	"fmt"
	"math"
	"strings"
	"transformer/activationfunction"
	"transformer/optimizer"
	"transformer/parameter"
	"transformer/vectormath"
)

type Example struct {
	PromptIDs []int
	AnswerIDs []int
}

func (example Example) tokenIDs() []int {
	tokenIDs := append([]int(nil), example.PromptIDs...)
	return append(tokenIDs, example.AnswerIDs...)
}

func (example Example) firstCountedToken() int {
	if len(example.PromptIDs) == 0 {
		return 1
	}
	return len(example.PromptIDs)
}

func checkEnoughTokens(functionName string, tokenIDs []int) {
	if len(tokenIDs) < 2 {
		panic(fmt.Sprintf("Model.%s: needs at least 2 tokens to learn what comes next, got %d", functionName, len(tokenIDs)))
	}
}

func checkExample(functionName string, example Example) {
	if len(example.AnswerIDs) == 0 {
		panic(fmt.Sprintf("Model.%s: the answer needs at least 1 token", functionName))
	}
	checkEnoughTokens(functionName, example.tokenIDs())
}

func countedRows(numberOfRows int, firstPredictedToken int, firstCountedToken int) []bool {
	counted := make([]bool, numberOfRows)
	for row := range counted {
		counted[row] = firstPredictedToken+row >= firstCountedToken
	}
	return counted
}

func maskedCrossEntropy(scores vectormath.Matrix, targetIDs []int, counted []bool) (float64, vectormath.Matrix) {
	gradients := vectormath.NewMatrix(scores.Rows, scores.Columns)
	numberCounted := 0
	for _, isCounted := range counted {
		if isCounted {
			numberCounted++
		}
	}
	if numberCounted == 0 {
		return 0, gradients
	}
	loss := 0.0
	for row := 0; row < scores.Rows; row++ {
		if !counted[row] {
			continue
		}
		probabilities := activationfunction.Softmax(scores.Row(row))
		loss -= math.Log(math.Max(probabilities[targetIDs[row]], 1e-300))
		gradientRow := gradients.Row(row)
		for column, probability := range probabilities {
			gradientRow[column] = probability / float64(numberCounted)
		}
		gradientRow[targetIDs[row]] -= 1 / float64(numberCounted)
	}
	return loss / float64(numberCounted), gradients
}

func (model *Model) usesMultiTokenPrediction(tokenIDs []int) bool {
	return model.MultiTokenPredictor != nil && len(tokenIDs) >= 3
}

func (model *Model) lossAndGradients(tokenIDs []int, firstCountedToken int, computeGradients bool) float64 {
	model.checkTokenIDs(tokenIDs)
	if computeGradients {
		model.applyFreezing()
		model.setDropoutActive(true)
		defer model.setDropoutActive(false)
	}
	if model.usesMultiTokenPrediction(tokenIDs) {
		loss, memory := model.multiTokenForward(tokenIDs, firstCountedToken)
		if computeGradients {
			model.multiTokenBackward(memory)
		}
		return loss
	}
	inputIDs := tokenIDs[:len(tokenIDs)-1]
	scores := model.Forward(inputIDs)
	loss, gradients := maskedCrossEntropy(scores, tokenIDs[1:], countedRows(len(inputIDs), 1, firstCountedToken))
	if computeGradients {
		model.Backward(gradients)
	}
	return loss
}

func (model *Model) Loss(tokenIDs []int) float64 {
	checkEnoughTokens("Loss", tokenIDs)
	model.checkTokenIDs(tokenIDs)
	scores := model.Forward(tokenIDs[:len(tokenIDs)-1])
	loss, _ := maskedCrossEntropy(scores, tokenIDs[1:], countedRows(len(tokenIDs)-1, 1, 1))
	return loss
}

func (model *Model) TrainingLoss(tokenIDs []int) float64 {
	checkEnoughTokens("TrainingLoss", tokenIDs)
	return model.lossAndGradients(tokenIDs, 1, false)
}

func (model *Model) AnswerLoss(example Example) float64 {
	checkExample("AnswerLoss", example)
	return model.lossAndGradients(example.tokenIDs(), example.firstCountedToken(), false)
}

func (model *Model) ComputeGradients(tokenIDs []int) float64 {
	checkEnoughTokens("ComputeGradients", tokenIDs)
	return model.lossAndGradients(tokenIDs, 1, true)
}

func (model *Model) ComputeAnswerGradients(example Example) float64 {
	checkExample("ComputeAnswerGradients", example)
	return model.lossAndGradients(example.tokenIDs(), example.firstCountedToken(), true)
}

func (model *Model) Freeze(namePrefix string) {
	model.FrozenNamePrefixes = append(model.FrozenNamePrefixes, namePrefix)
}

func (model *Model) UnfreezeAll() {
	model.FrozenNamePrefixes = nil
}

func (model *Model) isFrozen(name string) bool {
	if model.Settings.AdapterRank > 0 && !strings.Contains(name, ".lora.") {
		return true
	}
	for _, prefix := range model.FrozenNamePrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func (model *Model) applyFreezing() {
	for _, layer := range model.allLayers() {
		if layer.Adapter != nil {
			continue
		}
		layer.FreezeBase = model.isFrozen(layer.Name+".weights") && model.isFrozen(layer.Name+".biases")
	}
	model.TokenEmbedding.Frozen = model.isFrozen(model.TokenEmbedding.Name + ".table")
}

func (model *Model) TrainableParameters() []parameter.Parameter {
	var trainable []parameter.Parameter
	for _, current := range model.Parameters() {
		if !current.ReadOnly && !model.isFrozen(current.Name) {
			trainable = append(trainable, current)
		}
	}
	return trainable
}

func scaleGradients(parameters []parameter.Parameter, factor float64) {
	for _, current := range parameters {
		for i := range current.Gradients() {
			current.Gradients()[i] *= factor
		}
	}
}

func (model *Model) TrainOnExamples(examples []Example, chosenOptimizer optimizer.Optimizer) float64 {
	if len(examples) == 0 {
		panic("Model.TrainOnExamples: no examples given")
	}
	if model.IsCompressed() && model.Settings.AdapterRank == 0 {
		panic("Model.TrainOnExamples: the weights are compressed for running the model, call DecompressWeights before training, or AddLowRankAdapters to train adapters around them")
	}
	parameter.ZeroGradients(model.Parameters())
	totalLoss := 0.0
	for _, example := range examples {
		totalLoss += model.ComputeAnswerGradients(example)
	}
	trainable := model.TrainableParameters()
	scaleGradients(trainable, 1/float64(len(examples)))
	chosenOptimizer.Update(trainable)
	model.UpdateExpertBalance()
	return totalLoss / float64(len(examples))
}

func (model *Model) TrainBatch(sequences [][]int, chosenOptimizer optimizer.Optimizer) float64 {
	examples := make([]Example, len(sequences))
	for i, sequence := range sequences {
		examples[i] = Example{AnswerIDs: sequence}
	}
	return model.TrainOnExamples(examples, chosenOptimizer)
}

func (model *Model) TrainStep(tokenIDs []int, chosenOptimizer optimizer.Optimizer) float64 {
	return model.TrainBatch([][]int{tokenIDs}, chosenOptimizer)
}

func (model *Model) TrainOnAnswer(example Example, chosenOptimizer optimizer.Optimizer) float64 {
	return model.TrainOnExamples([]Example{example}, chosenOptimizer)
}

func RandomChunks(tokenIDs []int, length int, numberOfChunks int) [][]int {
	chunks := make([][]int, numberOfChunks)
	for i := range chunks {
		chunks[i] = RandomChunk(tokenIDs, length)
	}
	return chunks
}
