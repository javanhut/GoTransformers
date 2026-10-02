package transformer

import (
	"transformer/attention"
	"transformer/feedforward"
	"transformer/normalization"
	"transformer/parameter"
	"transformer/vectormath"
)

type Block struct {
	AttentionNorm   *normalization.RMSNorm
	Attention       *attention.SelfAttention
	FeedForwardNorm *normalization.RMSNorm
	FeedForward     *feedforward.GatedFeedForward
}

func (block *Block) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	attended := block.Attention.Forward(block.AttentionNorm.Forward(inputs))
	afterAttention := vectormath.AddMatrices(inputs, attended)

	fedForward := block.FeedForward.Forward(block.FeedForwardNorm.Forward(afterAttention))
	return vectormath.AddMatrices(afterAttention, fedForward)
}

func (block *Block) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	feedForwardInputGradients := block.FeedForwardNorm.Backward(block.FeedForward.Backward(outputGradients))
	afterAttentionGradients := vectormath.AddMatrices(outputGradients, feedForwardInputGradients)

	attentionInputGradients := block.AttentionNorm.Backward(block.Attention.Backward(afterAttentionGradients))
	return vectormath.AddMatrices(afterAttentionGradients, attentionInputGradients)
}

func (block *Block) ForwardOneToken(input vectormath.Vector) vectormath.Vector {
	normalized := block.AttentionNorm.Forward(oneRow(input)).Row(0)
	afterAttention := vectormath.Add(input, block.Attention.ForwardOneToken(normalized))

	fedForward := block.FeedForward.Forward(block.FeedForwardNorm.Forward(oneRow(afterAttention))).Row(0)
	return vectormath.Add(afterAttention, fedForward)
}

func (block *Block) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	parameters = append(parameters, block.AttentionNorm.Parameters()...)
	parameters = append(parameters, block.Attention.Parameters()...)
	parameters = append(parameters, block.FeedForwardNorm.Parameters()...)
	parameters = append(parameters, block.FeedForward.Parameters()...)
	return parameters
}

func oneRow(vector vectormath.Vector) vectormath.Matrix {
	return vectormath.MatrixFromRows([]vectormath.Vector{vector})
}
