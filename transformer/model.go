package transformer

import (
	"fmt"
	"transformer/activationfunction"
	"transformer/attention"
	"transformer/embedding"
	"transformer/feedforward"
	"transformer/lossfunction"
	"transformer/normalization"
	"transformer/optimizer"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/vectormath"
)

type Model struct {
	Settings       Settings
	TokenEmbedding *embedding.Embedding
	Blocks         []*Block
	FinalNorm      *normalization.RMSNorm
	OutputLayer    *perceptron.Layer

	generatedPositions int
}

func NewModel(settings Settings) (*Model, error) {
	if err := settings.Check(); err != nil {
		return nil, err
	}
	model := &Model{
		Settings:       settings,
		TokenEmbedding: embedding.NewEmbedding("tokens", settings.VocabularySize, settings.VectorSize),
		FinalNorm:      normalization.NewRMSNorm("finalNorm", settings.VectorSize),
		OutputLayer:    perceptron.NewLayer("output", settings.VectorSize, settings.VocabularySize, activationfunction.Linear),
	}

	var groupOwner *attention.SelfAttention
	for blockIndex := 0; blockIndex < settings.NumberOfBlocks; blockIndex++ {
		name := fmt.Sprintf("block%d", blockIndex+1)

		var blockAttention *attention.SelfAttention
		if settings.blockOwnsKeysAndValues(blockIndex) {
			blockAttention = attention.NewSelfAttention(name+".attention", settings.VectorSize, settings.NumberOfHeads, true)
			blockAttention.UseRotaryPositions = settings.UseRotaryPositions
			blockAttention.CachePrecision = settings.CachePrecision
			blockAttention.TrainAtCachePrecision = settings.TrainAtCachePrecision
			groupOwner = blockAttention
		} else {
			blockAttention = attention.NewBorrowingSelfAttention(name+".attention", groupOwner, settings.GroupSharingMode)
		}
		blockAttention.TopK = settings.TopK
		blockAttention.WindowSize = 0
		if settings.blockUsesWindow(blockIndex) {
			blockAttention.WindowSize = settings.WindowSize
		}

		var blockFeedForward *feedforward.GatedFeedForward
		if settings.FeedForwardClampLimit > 0 {
			blockFeedForward = feedforward.NewClampedSwiGLU(name+".feedForward", settings.VectorSize, settings.FeedForwardSize, settings.FeedForwardClampLimit)
		} else {
			blockFeedForward = feedforward.NewSwiGLU(name+".feedForward", settings.VectorSize, settings.FeedForwardSize)
		}

		model.Blocks = append(model.Blocks, &Block{
			AttentionNorm:   normalization.NewRMSNorm(name+".attentionNorm", settings.VectorSize),
			Attention:       blockAttention,
			FeedForwardNorm: normalization.NewRMSNorm(name+".feedForwardNorm", settings.VectorSize),
			FeedForward:     blockFeedForward,
		})
	}
	return model, nil
}

func (model *Model) Forward(tokenIDs []int) vectormath.Matrix {
	if len(tokenIDs) == 0 {
		panic("Model.Forward: no tokens given")
	}
	values := model.TokenEmbedding.Forward(tokenIDs)
	if !model.Settings.UseRotaryPositions {
		values = vectormath.AddMatrices(values, embedding.PositionalEncoding(len(tokenIDs), model.Settings.VectorSize))
	}
	for _, block := range model.Blocks {
		values = block.Forward(values)
	}
	return model.OutputLayer.Forward(model.FinalNorm.Forward(values))
}

func (model *Model) Backward(scoreGradients vectormath.Matrix) {
	gradients := model.FinalNorm.Backward(model.OutputLayer.Backward(scoreGradients))
	for i := len(model.Blocks) - 1; i >= 0; i-- {
		gradients = model.Blocks[i].Backward(gradients)
	}
	model.TokenEmbedding.Backward(gradients)
}

func (model *Model) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	parameters = append(parameters, model.TokenEmbedding.Parameters()...)
	for _, block := range model.Blocks {
		parameters = append(parameters, block.Parameters()...)
	}
	parameters = append(parameters, model.FinalNorm.Parameters()...)
	parameters = append(parameters, model.OutputLayer.Parameters()...)
	return parameters
}

func (model *Model) NumberOfParameters() int {
	count := 0
	for _, current := range model.Parameters() {
		count += len(current.Values)
	}
	return count
}

func (model *Model) checkTokenIDs(tokenIDs []int) {
	for position, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= model.Settings.VocabularySize {
			panic(fmt.Sprintf("token ID %d at position %d is outside the vocabulary of %d tokens", tokenID, position, model.Settings.VocabularySize))
		}
	}
}

func OneHotTargets(targetIDs []int, vocabularySize int) vectormath.Matrix {
	targets := vectormath.NewMatrix(len(targetIDs), vocabularySize)
	for row, targetID := range targetIDs {
		targets.Set(row, targetID, 1)
	}
	return targets
}

func (model *Model) Loss(tokenIDs []int) float64 {
	if len(tokenIDs) < 2 {
		panic(fmt.Sprintf("Model.Loss: needs at least 2 tokens, got %d", len(tokenIDs)))
	}
	model.checkTokenIDs(tokenIDs)
	scores := model.Forward(tokenIDs[:len(tokenIDs)-1])
	targets := OneHotTargets(tokenIDs[1:], model.Settings.VocabularySize)
	return lossfunction.SoftmaxCrossEntropy.Calculate(scores, targets)
}

func (model *Model) TrainStep(tokenIDs []int, chosenOptimizer optimizer.Optimizer) float64 {
	if len(tokenIDs) < 2 {
		panic(fmt.Sprintf("Model.TrainStep: needs at least 2 tokens to learn what comes next, got %d", len(tokenIDs)))
	}
	model.checkTokenIDs(tokenIDs)
	inputIDs := tokenIDs[:len(tokenIDs)-1]
	targetIDs := tokenIDs[1:]

	parameters := model.Parameters()
	parameter.ZeroGradients(parameters)

	scores := model.Forward(inputIDs)
	targets := OneHotTargets(targetIDs, model.Settings.VocabularySize)
	loss := lossfunction.SoftmaxCrossEntropy.Calculate(scores, targets)
	model.Backward(lossfunction.SoftmaxCrossEntropy.Gradient(scores, targets))

	chosenOptimizer.Update(parameters)
	return loss
}

func RandomChunk(tokenIDs []int, length int) []int {
	if length >= len(tokenIDs) {
		return tokenIDs
	}
	start := int(vectormath.RandomNumberBetween(0, float64(len(tokenIDs)-length+1)))
	if start > len(tokenIDs)-length {
		start = len(tokenIDs) - length
	}
	return tokenIDs[start : start+length]
}
