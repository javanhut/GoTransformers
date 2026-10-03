package embedding

import (
	"fmt"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

type Embedding struct {
	Name            string
	Table           vectormath.Matrix
	TableGradients  []float64
	CompressedTable *lowprecision.Rows
	Frozen          bool

	lastTokenIDs []int
}

func NewEmbedding(name string, vocabularySize int, vectorSize int) *Embedding {
	if vocabularySize <= 0 || vectorSize <= 0 {
		panic(fmt.Sprintf("NewEmbedding %q: vocabulary size and vector size must be at least 1, got %d and %d", name, vocabularySize, vectorSize))
	}
	limit := 1 / math.Sqrt(float64(vectorSize))
	return &Embedding{
		Name:  name,
		Table: vectormath.NewRandomMatrix(vocabularySize, vectorSize, -limit, limit),
	}
}

func (embedding *Embedding) VocabularySize() int {
	return embedding.Table.Rows
}

func (embedding *Embedding) VectorSize() int {
	return embedding.Table.Columns
}

func (embedding *Embedding) IsCompressed() bool {
	return embedding.CompressedTable != nil
}

func (embedding *Embedding) CompressTable(precision lowprecision.Precision) {
	if embedding.IsCompressed() {
		panic(fmt.Sprintf("embedding %q: table is already compressed", embedding.Name))
	}
	embedding.CompressedTable = lowprecision.RowsFromMatrix(embedding.Table, precision)
	embedding.Table = vectormath.Matrix{Rows: embedding.Table.Rows, Columns: embedding.Table.Columns}
	embedding.TableGradients = nil
}

func (embedding *Embedding) TableBytes() int {
	if embedding.IsCompressed() {
		return embedding.CompressedTable.BytesUsed()
	}
	return len(embedding.Table.Values) * 8
}

func (embedding *Embedding) VectorFor(tokenID int) vectormath.Vector {
	if tokenID < 0 || tokenID >= embedding.VocabularySize() {
		panic(fmt.Sprintf("embedding %q: token ID %d is outside the vocabulary of %d tokens", embedding.Name, tokenID, embedding.VocabularySize()))
	}
	if embedding.IsCompressed() {
		return embedding.CompressedTable.Row(tokenID)
	}
	return vectormath.CopyVector(embedding.Table.Row(tokenID))
}

func (embedding *Embedding) TableRows(tokenIDs []int) vectormath.Matrix {
	rows := vectormath.NewMatrix(len(tokenIDs), embedding.VectorSize())
	for position, tokenID := range tokenIDs {
		rows.SetRow(position, embedding.VectorFor(tokenID))
	}
	return rows
}

func (embedding *Embedding) Forward(tokenIDs []int) vectormath.Matrix {
	outputs := embedding.TableRows(tokenIDs)
	embedding.lastTokenIDs = make([]int, len(tokenIDs))
	copy(embedding.lastTokenIDs, tokenIDs)
	return outputs
}

func (embedding *Embedding) makeGradients() {
	if embedding.IsCompressed() {
		panic(fmt.Sprintf("embedding %q: table is compressed for running the model and can't be trained", embedding.Name))
	}
	if len(embedding.TableGradients) != len(embedding.Table.Values) {
		embedding.TableGradients = make([]float64, len(embedding.Table.Values))
	}
}

func (embedding *Embedding) ReleaseGradients() {
	embedding.TableGradients = nil
}

func (embedding *Embedding) AddGradients(tokenIDs []int, gradients vectormath.Matrix) {
	if gradients.Rows != len(tokenIDs) || gradients.Columns != embedding.VectorSize() {
		panic(fmt.Sprintf("embedding %q: got %dx%d gradients for %d tokens of size %d", embedding.Name, gradients.Rows, gradients.Columns, len(tokenIDs), embedding.VectorSize()))
	}
	if embedding.Frozen {
		return
	}
	embedding.makeGradients()
	vectorSize := embedding.VectorSize()
	for position, tokenID := range tokenIDs {
		tokenGradients := embedding.TableGradients[tokenID*vectorSize : (tokenID+1)*vectorSize]
		positionGradients := gradients.Row(position)
		for i := range tokenGradients {
			tokenGradients[i] += positionGradients[i]
		}
	}
}

func (embedding *Embedding) Backward(outputGradients vectormath.Matrix) {
	if outputGradients.Rows != len(embedding.lastTokenIDs) || outputGradients.Columns != embedding.VectorSize() {
		panic(fmt.Sprintf("embedding %q: output gradients are %dx%d but the last Forward produced %dx%d", embedding.Name, outputGradients.Rows, outputGradients.Columns, len(embedding.lastTokenIDs), embedding.VectorSize()))
	}
	embedding.AddGradients(embedding.lastTokenIDs, outputGradients)
}

func (embedding *Embedding) Parameters() []parameter.Parameter {
	if embedding.IsCompressed() {
		return []parameter.Parameter{
			{Name: embedding.Name + ".table", Rows: embedding.Table.Rows, Columns: embedding.Table.Columns, UseAdamW: true, ReadOnly: true},
		}
	}
	return []parameter.Parameter{
		{Name: embedding.Name + ".table", Values: embedding.Table.Values, GradientStorage: &embedding.TableGradients, Rows: embedding.Table.Rows, Columns: embedding.Table.Columns, UseAdamW: true},
	}
}

func PositionalEncoding(sequenceLength int, vectorSize int) vectormath.Matrix {
	encoding := vectormath.NewMatrix(sequenceLength, vectorSize)
	for position := range sequenceLength {
		encoding.SetRow(position, PositionalEncodingAt(position, vectorSize))
	}
	return encoding
}

func PositionalEncodingAt(position int, vectorSize int) vectormath.Vector {
	encoding := vectormath.NewVector(vectorSize)
	for i := range vectorSize {
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

func (embedding *Embedding) SetTable(values []float64) error {
	if len(values) != embedding.Table.Rows*embedding.Table.Columns {
		return fmt.Errorf("embedding %q: got %d values but the table is %dx%d", embedding.Name, len(values), embedding.Table.Rows, embedding.Table.Columns)
	}
	if embedding.IsCompressed() {
		newTable := vectormath.Matrix{Rows: embedding.Table.Rows, Columns: embedding.Table.Columns, Values: values}
		embedding.CompressedTable = lowprecision.RowsFromMatrix(newTable, embedding.CompressedTable.Precision)
	} else {
		copy(embedding.Table.Values, values)
	}
	return nil
}
