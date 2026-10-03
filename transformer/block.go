package transformer

import (
	"transformer/feedforward"
	"transformer/hyperconnection"
	"transformer/mixtureofexperts"
	"transformer/normalization"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/vectormath"
)

type AttentionLayer interface {
	Forward(inputs vectormath.Matrix) vectormath.Matrix
	Backward(outputGradients vectormath.Matrix) vectormath.Matrix
	StartGenerating()
	ForwardOneToken(input vectormath.Vector) vectormath.Vector
	Parameters() []parameter.Parameter
	CacheBytesUsed() int
	Layers() []*perceptron.Layer
}

type FeedForwardLayer interface {
	Forward(inputs vectormath.Matrix, tokenIDs []int) vectormath.Matrix
	Backward(outputGradients vectormath.Matrix) vectormath.Matrix
	Parameters() []parameter.Parameter
	Layers() []*perceptron.Layer
}

type gatedFeedForwardLayer struct {
	inner *feedforward.GatedFeedForward
}

func (layer gatedFeedForwardLayer) Forward(inputs vectormath.Matrix, tokenIDs []int) vectormath.Matrix {
	return layer.inner.Forward(inputs)
}

func (layer gatedFeedForwardLayer) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	return layer.inner.Backward(outputGradients)
}

func (layer gatedFeedForwardLayer) Parameters() []parameter.Parameter {
	return layer.inner.Parameters()
}

func (layer gatedFeedForwardLayer) Layers() []*perceptron.Layer {
	return layer.inner.Layers()
}

func (block *Block) Layers() []*perceptron.Layer {
	layers := block.Attention.Layers()
	return append(layers, block.FeedForward.Layers()...)
}

type Block struct {
	AttentionNorm         *normalization.RMSNorm
	Attention             AttentionLayer
	FeedForwardNorm       *normalization.RMSNorm
	FeedForward           FeedForwardLayer
	AttentionConnection   *hyperconnection.HyperConnection
	FeedForwardConnection *hyperconnection.HyperConnection
}

func (block *Block) usesHyperConnections() bool {
	return block.AttentionConnection != nil
}

func (block *Block) Forward(inputs vectormath.Matrix, tokenIDs []int) vectormath.Matrix {
	if !block.usesHyperConnections() {
		attended := block.Attention.Forward(block.AttentionNorm.Forward(inputs))
		afterAttention := vectormath.AddMatrices(inputs, attended)

		fedForward := block.FeedForward.Forward(block.FeedForwardNorm.Forward(afterAttention), tokenIDs)
		return vectormath.AddMatrices(afterAttention, fedForward)
	}

	attentionInput := block.AttentionConnection.LayerInput(inputs)
	attended := block.Attention.Forward(block.AttentionNorm.Forward(attentionInput))
	streams := block.AttentionConnection.Combine(attended)

	feedForwardInput := block.FeedForwardConnection.LayerInput(streams)
	fedForward := block.FeedForward.Forward(block.FeedForwardNorm.Forward(feedForwardInput), tokenIDs)
	return block.FeedForwardConnection.Combine(fedForward)
}

func (block *Block) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	if !block.usesHyperConnections() {
		feedForwardInputGradients := block.FeedForwardNorm.Backward(block.FeedForward.Backward(outputGradients))
		afterAttentionGradients := vectormath.AddMatrices(outputGradients, feedForwardInputGradients)

		attentionInputGradients := block.AttentionNorm.Backward(block.Attention.Backward(afterAttentionGradients))
		return vectormath.AddMatrices(afterAttentionGradients, attentionInputGradients)
	}

	fedForwardGradients, streamGradientsFromCombine := block.FeedForwardConnection.BackwardCombine(outputGradients)
	feedForwardInputGradients := block.FeedForwardNorm.Backward(block.FeedForward.Backward(fedForwardGradients))
	streamGradients := vectormath.AddMatrices(streamGradientsFromCombine, block.FeedForwardConnection.BackwardLayerInput(feedForwardInputGradients))

	attendedGradients, streamGradientsFromAttentionCombine := block.AttentionConnection.BackwardCombine(streamGradients)
	attentionInputGradients := block.AttentionNorm.Backward(block.Attention.Backward(attendedGradients))
	return vectormath.AddMatrices(streamGradientsFromAttentionCombine, block.AttentionConnection.BackwardLayerInput(attentionInputGradients))
}

func (block *Block) ForwardOneToken(input vectormath.Vector, tokenID int) vectormath.Vector {
	tokenIDs := []int{tokenID}
	if !block.usesHyperConnections() {
		normalized := block.AttentionNorm.Forward(oneRow(input)).Row(0)
		afterAttention := vectormath.Add(input, block.Attention.ForwardOneToken(normalized))

		fedForward := block.FeedForward.Forward(block.FeedForwardNorm.Forward(oneRow(afterAttention)), tokenIDs).Row(0)
		return vectormath.Add(afterAttention, fedForward)
	}

	attentionInput := block.AttentionConnection.LayerInput(oneRow(input))
	attended := block.Attention.ForwardOneToken(block.AttentionNorm.Forward(attentionInput).Row(0))
	streams := block.AttentionConnection.Combine(oneRow(attended))

	feedForwardInput := block.FeedForwardConnection.LayerInput(streams)
	fedForward := block.FeedForward.Forward(block.FeedForwardNorm.Forward(feedForwardInput), tokenIDs)
	return block.FeedForwardConnection.Combine(fedForward).Row(0)
}

func (block *Block) mixtureOfExperts() *mixtureofexperts.MixtureOfExperts {
	mixture, isMixture := block.FeedForward.(*mixtureofexperts.MixtureOfExperts)
	if !isMixture {
		return nil
	}
	return mixture
}

func (block *Block) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	if block.usesHyperConnections() {
		parameters = append(parameters, block.AttentionConnection.Parameters()...)
	}
	parameters = append(parameters, block.AttentionNorm.Parameters()...)
	parameters = append(parameters, block.Attention.Parameters()...)
	if block.usesHyperConnections() {
		parameters = append(parameters, block.FeedForwardConnection.Parameters()...)
	}
	parameters = append(parameters, block.FeedForwardNorm.Parameters()...)
	parameters = append(parameters, block.FeedForward.Parameters()...)
	return parameters
}

func oneRow(vector vectormath.Vector) vectormath.Matrix {
	return vectormath.MatrixFromRows([]vectormath.Vector{vector})
}
