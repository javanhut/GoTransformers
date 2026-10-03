package transformer

import (
	"transformer/activationfunction"
	"transformer/attention"
	"transformer/embedding"
	"transformer/feedforward"
	"transformer/normalization"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/vectormath"
)

type MultiTokenPredictor struct {
	HiddenNorm    *normalization.RMSNorm
	EmbeddingNorm *normalization.RMSNorm
	CombineLayer  *perceptron.Layer
	Block         *Block
	FinalNorm     *normalization.RMSNorm

	tokenEmbedding   *embedding.Embedding
	lastNextTokenIDs []int
}

func newMultiTokenPredictor(settings Settings, tokenEmbedding *embedding.Embedding) *MultiTokenPredictor {
	name := "multiToken"
	block := &Block{
		AttentionNorm:   normalization.NewRMSNorm(name+".block.attentionNorm", settings.VectorSize),
		Attention:       attention.NewSelfAttentionWithOptions(name+".block.attention", settings.VectorSize, settings.standardOptions()),
		FeedForwardNorm: normalization.NewRMSNorm(name+".block.feedForwardNorm", settings.VectorSize),
		FeedForward:     gatedFeedForwardLayer{inner: feedforward.NewSwiGLU(name+".block.feedForward", settings.VectorSize, settings.FeedForwardSize)},
	}
	return &MultiTokenPredictor{
		HiddenNorm:     normalization.NewRMSNorm(name+".hiddenNorm", settings.VectorSize),
		EmbeddingNorm:  normalization.NewRMSNorm(name+".embeddingNorm", settings.VectorSize),
		CombineLayer:   perceptron.NewLayer(name+".combine", 2*settings.VectorSize, settings.VectorSize, activationfunction.Linear),
		Block:          block,
		FinalNorm:      normalization.NewRMSNorm(name+".finalNorm", settings.VectorSize),
		tokenEmbedding: tokenEmbedding,
	}
}

func (predictor *MultiTokenPredictor) Forward(hidden vectormath.Matrix, nextTokenIDs []int) vectormath.Matrix {
	embedded := predictor.tokenEmbedding.TableRows(nextTokenIDs)
	joined := vectormath.JoinColumns(predictor.HiddenNorm.Forward(hidden), predictor.EmbeddingNorm.Forward(embedded))
	combined := predictor.CombineLayer.Forward(joined)
	predictor.lastNextTokenIDs = nextTokenIDs
	return predictor.FinalNorm.Forward(predictor.Block.Forward(combined, nextTokenIDs))
}

func (predictor *MultiTokenPredictor) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	combinedGradients := predictor.Block.Backward(predictor.FinalNorm.Backward(outputGradients))
	joinedGradients := predictor.CombineLayer.Backward(combinedGradients)
	hiddenGradients, embeddedGradients := vectormath.SplitColumns(joinedGradients, len(predictor.HiddenNorm.Weights))
	predictor.tokenEmbedding.AddGradients(predictor.lastNextTokenIDs, predictor.EmbeddingNorm.Backward(embeddedGradients))
	return predictor.HiddenNorm.Backward(hiddenGradients)
}

func (predictor *MultiTokenPredictor) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	parameters = append(parameters, predictor.HiddenNorm.Parameters()...)
	parameters = append(parameters, predictor.EmbeddingNorm.Parameters()...)
	parameters = append(parameters, predictor.CombineLayer.Parameters()...)
	parameters = append(parameters, predictor.Block.Parameters()...)
	parameters = append(parameters, predictor.FinalNorm.Parameters()...)
	return parameters
}

type multiTokenMemory struct {
	numberOfInputs  int
	mainGradients   vectormath.Matrix
	secondGradients vectormath.Matrix
}

func (model *Model) multiTokenForward(tokenIDs []int, firstCountedToken int) (float64, multiTokenMemory) {
	inputIDs := tokenIDs[:len(tokenIDs)-1]
	numberOfInputs := len(inputIDs)

	hidden := model.hiddenStates(inputIDs)
	mainNormalized := model.FinalNorm.Forward(hidden)
	hiddenForSecondToken, _ := vectormath.SplitRows(hidden, numberOfInputs-1)
	secondTokenHidden := model.MultiTokenPredictor.Forward(hiddenForSecondToken, inputIDs[1:])

	allScores := model.OutputLayer.Forward(vectormath.StackRows(mainNormalized, secondTokenHidden))
	mainScores, secondScores := vectormath.SplitRows(allScores, numberOfInputs)
	mainLoss, mainGradients := maskedCrossEntropy(mainScores, tokenIDs[1:], countedRows(numberOfInputs, 1, firstCountedToken))
	secondLoss, secondGradients := maskedCrossEntropy(secondScores, tokenIDs[2:], countedRows(numberOfInputs-1, 2, firstCountedToken))

	weight := model.Settings.MultiTokenLossWeight
	memory := multiTokenMemory{
		numberOfInputs:  numberOfInputs,
		mainGradients:   mainGradients,
		secondGradients: vectormath.ScaleMatrix(secondGradients, weight),
	}
	return mainLoss + weight*secondLoss, memory
}

func (model *Model) multiTokenBackward(memory multiTokenMemory) {
	allNormalizedGradients := model.OutputLayer.Backward(vectormath.StackRows(memory.mainGradients, memory.secondGradients))
	mainNormalizedGradients, secondTokenHiddenGradients := vectormath.SplitRows(allNormalizedGradients, memory.numberOfInputs)

	hiddenGradients := model.FinalNorm.Backward(mainNormalizedGradients)
	hiddenGradientsFromSecondToken := model.MultiTokenPredictor.Backward(secondTokenHiddenGradients)
	for row := 0; row < hiddenGradientsFromSecondToken.Rows; row++ {
		target := hiddenGradients.Row(row)
		for i, gradient := range hiddenGradientsFromSecondToken.Row(row) {
			target[i] += gradient
		}
	}
	model.hiddenStatesBackward(hiddenGradients)
}
