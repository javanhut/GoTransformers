package main

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/attention"
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/lossfunction"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/vectormath"
	"strings"
)

type tinyModel struct {
	tokenEmbedding *embedding.Embedding
	selfAttention  *attention.SelfAttention
	outputLayer    *perceptron.Layer
}

func (model *tinyModel) Forward(tokenIDs []int) vectormath.Matrix {
	tokenVectors := model.tokenEmbedding.Forward(tokenIDs)
	positions := embedding.PositionalEncoding(len(tokenIDs), model.tokenEmbedding.VectorSize())
	withPositions := vectormath.AddMatrices(tokenVectors, positions)

	attended := model.selfAttention.Forward(withPositions)
	withAttention := vectormath.AddMatrices(withPositions, attended)

	return model.outputLayer.Forward(withAttention)
}

func (model *tinyModel) Backward(scoreGradients vectormath.Matrix) {
	withAttentionGradients := model.outputLayer.Backward(scoreGradients)
	attentionInputGradients := model.selfAttention.Backward(withAttentionGradients)
	withPositionsGradients := vectormath.AddMatrices(withAttentionGradients, attentionInputGradients)
	model.tokenEmbedding.Backward(withPositionsGradients)
}

func (model *tinyModel) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	parameters = append(parameters, model.tokenEmbedding.Parameters()...)
	parameters = append(parameters, model.selfAttention.Parameters()...)
	parameters = append(parameters, model.outputLayer.Parameters()...)
	return parameters
}

func oneHotRows(tokenIDs []int, vocabularySize int) vectormath.Matrix {
	matrix := vectormath.NewMatrix(len(tokenIDs), vocabularySize)
	for row, tokenID := range tokenIDs {
		matrix.Set(row, tokenID, 1)
	}
	return matrix
}

func main() {
	text := "hello world. hello transformers. "
	characters := embedding.SplitIntoCharacters(text)
	vocabulary := embedding.BuildVocabulary(characters)
	tokenIDs := vocabulary.Encode(characters)

	inputIDs := tokenIDs[:len(tokenIDs)-1]
	targetIDs := tokenIDs[1:]
	targets := oneHotRows(targetIDs, vocabulary.Size())

	vectorSize := 16
	model := &tinyModel{
		tokenEmbedding: embedding.NewEmbedding("tokens", vocabulary.Size(), vectorSize),
		selfAttention:  attention.NewSelfAttention("attention", vectorSize, 2, true),
		outputLayer:    perceptron.NewLayer("output", vectorSize, vocabulary.Size(), activationfunction.Linear),
	}
	parameters := model.Parameters()
	adam := optimizer.NewAdam(0.01)

	for step := 1; step <= 400; step++ {
		parameter.ZeroGradients(parameters)
		scores := model.Forward(inputIDs)
		loss := lossfunction.SoftmaxCrossEntropy.Calculate(scores, targets)
		model.Backward(lossfunction.SoftmaxCrossEntropy.Gradient(scores, targets))
		adam.Update(parameters)
		if step%100 == 0 {
			fmt.Printf("step %d  loss %.4f\n", step, loss)
		}
	}

	generatedIDs := vocabulary.Encode([]string{"h"})
	for len(generatedIDs) < len(inputIDs) {
		scores := model.Forward(generatedIDs)
		lastPositionScores := scores.Row(scores.Rows - 1)
		generatedIDs = append(generatedIDs, vectormath.IndexOfMax(lastPositionScores))
	}
	fmt.Printf("generated: %q\n", strings.Join(vocabulary.Decode(generatedIDs), ""))
}
