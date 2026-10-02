package transformer

import (
	"fmt"
	"transformer/activationfunction"
	"transformer/attention"
	"transformer/embedding"
	"transformer/feedforward"
	"transformer/hyperconnection"
	"transformer/lossfunction"
	"transformer/mixtureofexperts"
	"transformer/normalization"
	"transformer/optimizer"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/vectormath"
)

type Model struct {
	Settings            Settings
	TokenEmbedding      *embedding.Embedding
	Blocks              []*Block
	FinalNorm           *normalization.RMSNorm
	OutputLayer         *perceptron.Layer
	MultiTokenPredictor *MultiTokenPredictor

	generatedPositions int
	lastScores         vectormath.Vector
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
	standardBlocksSoFar := 0
	for blockIndex := 0; blockIndex < settings.NumberOfBlocks; blockIndex++ {
		name := fmt.Sprintf("block%d", blockIndex+1)
		block := &Block{
			AttentionNorm:   normalization.NewRMSNorm(name+".attentionNorm", settings.VectorSize),
			FeedForwardNorm: normalization.NewRMSNorm(name+".feedForwardNorm", settings.VectorSize),
		}

		kind := settings.attentionKindFor(blockIndex)
		if kind == StandardAttention {
			block.Attention, groupOwner = makeStandardAttention(settings, name+".attention", blockIndex, standardBlocksSoFar, groupOwner)
			standardBlocksSoFar++
		} else {
			compressed := attention.NewCompressedAttention(name+".attention", settings.VectorSize, settings.compressedOptions(kind))
			compressed.AttendToAllWhileTraining = settings.AttendToAllWhileTraining
			block.Attention = compressed
		}

		block.FeedForward = makeFeedForward(settings, name+".feedForward", blockIndex)

		if settings.residualStreams() > 1 {
			block.AttentionConnection = hyperconnection.NewHyperConnection(name+".attentionConnection", settings.VectorSize, settings.residualStreams())
			block.FeedForwardConnection = hyperconnection.NewHyperConnection(name+".feedForwardConnection", settings.VectorSize, settings.residualStreams())
		}
		model.Blocks = append(model.Blocks, block)
	}

	if settings.MultiTokenPrediction {
		model.MultiTokenPredictor = newMultiTokenPredictor(settings, model.TokenEmbedding)
	}
	return model, nil
}

func makeStandardAttention(settings Settings, name string, blockIndex int, standardBlocksSoFar int, groupOwner *attention.SelfAttention) (*attention.SelfAttention, *attention.SelfAttention) {
	var standard *attention.SelfAttention
	if standardBlocksSoFar%settings.BlocksPerKeyValueGroup == 0 || groupOwner == nil {
		standard = attention.NewSelfAttentionWithOptions(name, settings.VectorSize, settings.standardOptions())
		groupOwner = standard
	} else {
		standard = attention.NewBorrowingSelfAttention(name, groupOwner, settings.GroupSharingMode)
	}
	standard.TopK = settings.TopK
	standard.WindowSize = 0
	if settings.blockUsesWindow(blockIndex) {
		standard.WindowSize = settings.WindowSize
	}
	return standard, groupOwner
}

func makeFeedForward(settings Settings, name string, blockIndex int) FeedForwardLayer {
	if settings.UseMixtureOfExperts {
		var mixture *mixtureofexperts.MixtureOfExperts
		if settings.FeedForwardClampLimit > 0 {
			mixture = mixtureofexperts.NewClampedMixtureOfExperts(name, settings.VectorSize, settings.ExpertHiddenSize, settings.NumberOfSharedExperts, settings.NumberOfRoutedExperts, settings.ExpertsPerToken, settings.FeedForwardClampLimit)
		} else {
			mixture = mixtureofexperts.NewMixtureOfExperts(name, settings.VectorSize, settings.ExpertHiddenSize, settings.NumberOfSharedExperts, settings.NumberOfRoutedExperts, settings.ExpertsPerToken)
		}
		mixture.UseHashRouting = blockIndex < settings.HashRoutedBlocks
		return mixture
	}
	if settings.FeedForwardClampLimit > 0 {
		return gatedFeedForwardLayer{inner: feedforward.NewClampedSwiGLU(name, settings.VectorSize, settings.FeedForwardSize, settings.FeedForwardClampLimit)}
	}
	return gatedFeedForwardLayer{inner: feedforward.NewSwiGLU(name, settings.VectorSize, settings.FeedForwardSize)}
}

func (model *Model) hiddenStates(tokenIDs []int) vectormath.Matrix {
	values := model.TokenEmbedding.Forward(tokenIDs)
	if !model.Settings.UseRotaryPositions {
		values = vectormath.AddMatrices(values, embedding.PositionalEncoding(len(tokenIDs), model.Settings.VectorSize))
	}
	streams := model.Settings.residualStreams()
	if streams > 1 {
		values = hyperconnection.ExpandToStreams(values, streams)
	}
	for _, block := range model.Blocks {
		values = block.Forward(values, tokenIDs)
	}
	if streams > 1 {
		values = hyperconnection.CollapseStreams(values, streams)
	}
	return values
}

func (model *Model) hiddenStatesBackward(gradients vectormath.Matrix) {
	streams := model.Settings.residualStreams()
	if streams > 1 {
		gradients = hyperconnection.CollapseStreamsBackward(gradients, streams)
	}
	for i := len(model.Blocks) - 1; i >= 0; i-- {
		gradients = model.Blocks[i].Backward(gradients)
	}
	if streams > 1 {
		gradients = hyperconnection.ExpandToStreamsBackward(gradients, streams)
	}
	model.TokenEmbedding.Backward(gradients)
}

func (model *Model) Forward(tokenIDs []int) vectormath.Matrix {
	if len(tokenIDs) == 0 {
		panic("Model.Forward: no tokens given")
	}
	model.checkTokenIDs(tokenIDs)
	return model.OutputLayer.Forward(model.FinalNorm.Forward(model.hiddenStates(tokenIDs)))
}

func (model *Model) Backward(scoreGradients vectormath.Matrix) {
	model.hiddenStatesBackward(model.FinalNorm.Backward(model.OutputLayer.Backward(scoreGradients)))
}

func (model *Model) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	parameters = append(parameters, model.TokenEmbedding.Parameters()...)
	for _, block := range model.Blocks {
		parameters = append(parameters, block.Parameters()...)
	}
	if model.MultiTokenPredictor != nil {
		parameters = append(parameters, model.MultiTokenPredictor.Parameters()...)
	}
	parameters = append(parameters, model.FinalNorm.Parameters()...)
	for _, outputParameter := range model.OutputLayer.Parameters() {
		outputParameter.UseAdamW = true
		parameters = append(parameters, outputParameter)
	}
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

func checkEnoughTokens(functionName string, tokenIDs []int) {
	if len(tokenIDs) < 2 {
		panic(fmt.Sprintf("Model.%s: needs at least 2 tokens to learn what comes next, got %d", functionName, len(tokenIDs)))
	}
}

func (model *Model) Loss(tokenIDs []int) float64 {
	checkEnoughTokens("Loss", tokenIDs)
	scores := model.Forward(tokenIDs[:len(tokenIDs)-1])
	targets := OneHotTargets(tokenIDs[1:], model.Settings.VocabularySize)
	return lossfunction.SoftmaxCrossEntropy.Calculate(scores, targets)
}

func (model *Model) usesMultiTokenPrediction(tokenIDs []int) bool {
	return model.MultiTokenPredictor != nil && len(tokenIDs) >= 3
}

func (model *Model) TrainingLoss(tokenIDs []int) float64 {
	checkEnoughTokens("TrainingLoss", tokenIDs)
	model.checkTokenIDs(tokenIDs)
	if !model.usesMultiTokenPrediction(tokenIDs) {
		return model.Loss(tokenIDs)
	}
	loss, _ := model.multiTokenForward(tokenIDs)
	return loss
}

func (model *Model) ComputeGradients(tokenIDs []int) float64 {
	checkEnoughTokens("ComputeGradients", tokenIDs)
	model.checkTokenIDs(tokenIDs)
	if model.usesMultiTokenPrediction(tokenIDs) {
		loss, memory := model.multiTokenForward(tokenIDs)
		model.multiTokenBackward(memory)
		return loss
	}
	scores := model.Forward(tokenIDs[:len(tokenIDs)-1])
	targets := OneHotTargets(tokenIDs[1:], model.Settings.VocabularySize)
	loss := lossfunction.SoftmaxCrossEntropy.Calculate(scores, targets)
	model.Backward(lossfunction.SoftmaxCrossEntropy.Gradient(scores, targets))
	return loss
}

func (model *Model) TrainStep(tokenIDs []int, chosenOptimizer optimizer.Optimizer) float64 {
	parameters := model.Parameters()
	parameter.ZeroGradients(parameters)
	loss := model.ComputeGradients(tokenIDs)
	chosenOptimizer.Update(parameters)
	model.UpdateExpertBalance()
	return loss
}

func (model *Model) UpdateExpertBalance() {
	for _, block := range model.Blocks {
		if mixture := block.mixtureOfExperts(); mixture != nil {
			mixture.UpdateBalance(model.Settings.BalanceUpdateRate)
		}
	}
	if model.MultiTokenPredictor != nil {
		if mixture := model.MultiTokenPredictor.Block.mixtureOfExperts(); mixture != nil {
			mixture.UpdateBalance(model.Settings.BalanceUpdateRate)
		}
	}
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
