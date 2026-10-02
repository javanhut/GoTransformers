package embedding

import (
	"fmt"
	"math"
	"transformer/parameter"
	"transformer/vectormath"
)

type Embedding struct {
	Name           string
	Table          vectormath.Matrix
	TableGradients vectormath.Matrix

	lastTokenIDs []int
}

func NewEmbedding(name string, vocabularySize int, vectorSize int) *Embedding {
	if vocabularySize <= 0 || vectorSize <= 0 {
		panic(fmt.Sprintf("NewEmbedding %q: vocabulary size and vector size must be at least 1, got %d and %d", name, vocabularySize, vectorSize))
	}
	limit := 1 / math.Sqrt(float64(vectorSize))
	return &Embedding{
		Name:           name,
		Table:          vectormath.NewRandomMatrix(vocabularySize, vectorSize, -limit, limit),
		TableGradients: vectormath.NewMatrix(vocabularySize, vectorSize),
	}
}

func (embedding *Embedding) VocabularySize() int {
	return embedding.Table.Rows
}

func (embedding *Embedding) VectorSize() int {
	return embedding.Table.Columns
}

func (embedding *Embedding) Forward(tokenIDs []int) vectormath.Matrix {
	outputs := vectormath.NewMatrix(len(tokenIDs), embedding.VectorSize())
	for position, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= embedding.VocabularySize() {
			panic(fmt.Sprintf("embedding %q: token ID %d at position %d is outside the vocabulary of %d tokens", embedding.Name, tokenID, position, embedding.VocabularySize()))
		}
		outputs.SetRow(position, embedding.Table.Row(tokenID))
	}

	embedding.lastTokenIDs = make([]int, len(tokenIDs))
	copy(embedding.lastTokenIDs, tokenIDs)
	return outputs
}

func (embedding *Embedding) Backward(outputGradients vectormath.Matrix) {
	if outputGradients.Rows != len(embedding.lastTokenIDs) || outputGradients.Columns != embedding.VectorSize() {
		panic(fmt.Sprintf("embedding %q: output gradients are %dx%d but the last Forward produced %dx%d", embedding.Name, outputGradients.Rows, outputGradients.Columns, len(embedding.lastTokenIDs), embedding.VectorSize()))
	}
	for position, tokenID := range embedding.lastTokenIDs {
		tokenGradients := embedding.TableGradients.Row(tokenID)
		positionGradients := outputGradients.Row(position)
		for i := range tokenGradients {
			tokenGradients[i] += positionGradients[i]
		}
	}
}

func (embedding *Embedding) Parameters() []parameter.Parameter {
	return []parameter.Parameter{
		{Name: embedding.Name + ".table", Values: embedding.Table.Values, Gradients: embedding.TableGradients.Values},
	}
}

func PositionalEncoding(sequenceLength int, vectorSize int) vectormath.Matrix {
	encoding := vectormath.NewMatrix(sequenceLength, vectorSize)
	for position := 0; position < sequenceLength; position++ {
		encoding.SetRow(position, PositionalEncodingAt(position, vectorSize))
	}
	return encoding
}

func PositionalEncodingAt(position int, vectorSize int) vectormath.Vector {
	encoding := vectormath.NewVector(vectorSize)
	for i := 0; i < vectorSize; i++ {
		pairNumber := i / 2
		frequency := 1 / math.Pow(10000, float64(2*pairNumber)/float64(vectorSize))
		angle := float64(position) * frequency
		if i%2 == 0 {
			encoding[i] = math.Sin(angle)
		} else {
			encoding[i] = math.Cos(angle)
		}
	}
	return encoding
}
