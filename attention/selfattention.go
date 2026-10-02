package attention

import (
	"fmt"
	"math"
	"transformer/activationfunction"
	"transformer/lowprecision"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/vectormath"
)

type SharingMode int

const (
	OwnKeysAndValues SharingMode = iota
	BorrowKeysAndValues
	BorrowKeysValuesAndChoices
)

func (mode SharingMode) String() string {
	switch mode {
	case OwnKeysAndValues:
		return "OwnKeysAndValues"
	case BorrowKeysAndValues:
		return "BorrowKeysAndValues"
	case BorrowKeysValuesAndChoices:
		return "BorrowKeysValuesAndChoices"
	}
	return fmt.Sprintf("SharingMode(%d)", int(mode))
}

func (mode SharingMode) MarshalText() ([]byte, error) {
	return []byte(mode.String()), nil
}

func (mode *SharingMode) UnmarshalText(text []byte) error {
	for _, possible := range []SharingMode{OwnKeysAndValues, BorrowKeysAndValues, BorrowKeysValuesAndChoices} {
		if possible.String() == string(text) {
			*mode = possible
			return nil
		}
	}
	return fmt.Errorf("unknown sharing mode %q", string(text))
}

type lookedAt struct {
	positions []int
	weights   []float64
}

type SelfAttention struct {
	Name                  string
	NumberOfHeads         int
	HideFutureTokens      bool
	UseRotaryPositions    bool
	WindowSize            int
	TopK                  int
	SharingMode           SharingMode
	SharedFrom            *SelfAttention
	CachePrecision        lowprecision.Precision
	TrainAtCachePrecision bool

	QueryLayer  *perceptron.Layer
	KeyLayer    *perceptron.Layer
	ValueLayer  *perceptron.Layer
	OutputLayer *perceptron.Layer

	borrowers []*SelfAttention

	lastQueries            vectormath.Matrix
	lastKeys               vectormath.Matrix
	lastValues             vectormath.Matrix
	lastLookedAt           [][]lookedAt
	borrowedKeyGradients   vectormath.Matrix
	borrowedValueGradients vectormath.Matrix
	backwardFinished       bool

	generation *generationState
}

func NewSelfAttention(name string, vectorSize int, numberOfHeads int, hideFutureTokens bool) *SelfAttention {
	if numberOfHeads <= 0 || vectorSize%numberOfHeads != 0 {
		panic(fmt.Sprintf("NewSelfAttention %q: vector size %d must split evenly into %d heads", name, vectorSize, numberOfHeads))
	}
	return &SelfAttention{
		Name:             name,
		NumberOfHeads:    numberOfHeads,
		HideFutureTokens: hideFutureTokens,
		SharingMode:      OwnKeysAndValues,
		QueryLayer:       perceptron.NewLayer(name+".query", vectorSize, vectorSize, activationfunction.Linear),
		KeyLayer:         perceptron.NewLayer(name+".key", vectorSize, vectorSize, activationfunction.Linear),
		ValueLayer:       perceptron.NewLayer(name+".value", vectorSize, vectorSize, activationfunction.Linear),
		OutputLayer:      perceptron.NewLayer(name+".output", vectorSize, vectorSize, activationfunction.Linear),
	}
}

func NewBorrowingSelfAttention(name string, sharedFrom *SelfAttention, mode SharingMode) *SelfAttention {
	if sharedFrom == nil {
		panic(fmt.Sprintf("NewBorrowingSelfAttention %q: sharedFrom is nil", name))
	}
	if mode != BorrowKeysAndValues && mode != BorrowKeysValuesAndChoices {
		panic(fmt.Sprintf("NewBorrowingSelfAttention %q: mode must be BorrowKeysAndValues or BorrowKeysValuesAndChoices, got %v", name, mode))
	}
	vectorSize := sharedFrom.VectorSize()
	attention := &SelfAttention{
		Name:               name,
		NumberOfHeads:      sharedFrom.NumberOfHeads,
		HideFutureTokens:   sharedFrom.HideFutureTokens,
		UseRotaryPositions: sharedFrom.UseRotaryPositions,
		WindowSize:         sharedFrom.WindowSize,
		TopK:               sharedFrom.TopK,
		SharingMode:        mode,
		SharedFrom:         sharedFrom,
		QueryLayer:         perceptron.NewLayer(name+".query", vectorSize, vectorSize, activationfunction.Linear),
		OutputLayer:        perceptron.NewLayer(name+".output", vectorSize, vectorSize, activationfunction.Linear),
	}
	owner := attention.keyValueOwner()
	owner.borrowers = append(owner.borrowers, attention)
	return attention
}

func (attention *SelfAttention) VectorSize() int {
	return attention.QueryLayer.NumberOfInputs()
}

func (attention *SelfAttention) HeadSize() int {
	return attention.VectorSize() / attention.NumberOfHeads
}

func (attention *SelfAttention) keyValueOwner() *SelfAttention {
	owner := attention
	for owner.SharingMode != OwnKeysAndValues {
		owner = owner.SharedFrom
	}
	return owner
}

func (attention *SelfAttention) choicesSource() *SelfAttention {
	source := attention
	for source.SharingMode == BorrowKeysValuesAndChoices {
		source = source.SharedFrom
	}
	return source
}

func (attention *SelfAttention) checkSetUp() {
	if attention.NumberOfHeads <= 0 || attention.VectorSize()%attention.NumberOfHeads != 0 {
		panic(fmt.Sprintf("attention %q: vector size %d must split evenly into %d heads", attention.Name, attention.VectorSize(), attention.NumberOfHeads))
	}
	if attention.WindowSize < 0 || attention.TopK < 0 {
		panic(fmt.Sprintf("attention %q: WindowSize and TopK can't be negative, got %d and %d", attention.Name, attention.WindowSize, attention.TopK))
	}
	if attention.UseRotaryPositions && attention.HeadSize()%2 != 0 {
		panic(fmt.Sprintf("attention %q: rotary positions need an even head size, got %d", attention.Name, attention.HeadSize()))
	}
	if attention.SharingMode == OwnKeysAndValues {
		if attention.KeyLayer == nil || attention.ValueLayer == nil {
			panic(fmt.Sprintf("attention %q: owns its keys and values but has no KeyLayer or ValueLayer", attention.Name))
		}
		return
	}
	if attention.SharedFrom == nil {
		panic(fmt.Sprintf("attention %q: is %v but SharedFrom is nil", attention.Name, attention.SharingMode))
	}
	owner := attention.keyValueOwner()
	if owner.VectorSize() != attention.VectorSize() || owner.NumberOfHeads != attention.NumberOfHeads || owner.UseRotaryPositions != attention.UseRotaryPositions {
		panic(fmt.Sprintf("attention %q: borrows from %q, so both need the same vector size, number of heads and UseRotaryPositions", attention.Name, owner.Name))
	}
}

func headSlice(vector vectormath.Vector, head int, headSize int) vectormath.Vector {
	start := head * headSize
	end := start + headSize
	return vector[start:end:end]
}

func (attention *SelfAttention) headPart(matrix vectormath.Matrix, position int, head int) vectormath.Vector {
	return headSlice(matrix.Row(position), head, attention.HeadSize())
}

func (attention *SelfAttention) scale() float64 {
	return 1 / math.Sqrt(float64(attention.HeadSize()))
}

func (attention *SelfAttention) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	attention.checkSetUp()
	if inputs.Columns != attention.VectorSize() {
		panic(fmt.Sprintf("attention %q: each token vector has %d values but attention expects %d", attention.Name, inputs.Columns, attention.VectorSize()))
	}
	sequenceLength := inputs.Rows
	queries := attention.rotateRows(attention.QueryLayer.Forward(inputs), rotateForward)

	if attention.SharingMode == OwnKeysAndValues {
		keys := attention.rotateRows(attention.KeyLayer.Forward(inputs), rotateForward)
		values := attention.ValueLayer.Forward(inputs)
		if attention.TrainAtCachePrecision {
			keys = lowprecision.RoundTripMatrix(keys, attention.CachePrecision)
			values = lowprecision.RoundTripMatrix(values, attention.CachePrecision)
		}
		attention.lastKeys = keys
		attention.lastValues = values
		attention.borrowedKeyGradients = vectormath.NewMatrix(sequenceLength, attention.VectorSize())
		attention.borrowedValueGradients = vectormath.NewMatrix(sequenceLength, attention.VectorSize())
		attention.backwardFinished = false
	} else {
		owner := attention.keyValueOwner()
		if owner.lastKeys.Rows != sequenceLength {
			panic(fmt.Sprintf("attention %q: borrows keys and values from %q, run %q Forward on the same tokens first", attention.Name, owner.Name, owner.Name))
		}
		attention.lastKeys = owner.lastKeys
		attention.lastValues = owner.lastValues
	}

	var choicesSource *SelfAttention
	if attention.SharingMode == BorrowKeysValuesAndChoices {
		choicesSource = attention.choicesSource()
		if len(choicesSource.lastLookedAt) != attention.NumberOfHeads || len(choicesSource.lastLookedAt[0]) != sequenceLength {
			panic(fmt.Sprintf("attention %q: borrows choices from %q, run %q Forward on the same tokens first", attention.Name, choicesSource.Name, choicesSource.Name))
		}
	}

	combined := vectormath.NewMatrix(sequenceLength, attention.VectorSize())
	lookedAtForHead := make([][]lookedAt, attention.NumberOfHeads)
	for head := 0; head < attention.NumberOfHeads; head++ {
		lookedAtForHead[head] = make([]lookedAt, sequenceLength)
		keyAt := func(position int) vectormath.Vector { return attention.headPart(attention.lastKeys, position, head) }
		valueAt := func(position int) vectormath.Vector { return attention.headPart(attention.lastValues, position, head) }

		for position := 0; position < sequenceLength; position++ {
			query := attention.headPart(queries, position, head)

			var positions []int
			if choicesSource != nil {
				positions = choicesSource.lastLookedAt[head][position].positions
			} else {
				first, last := attention.visiblePositions(position, sequenceLength)
				positions = attention.choosePositions(query, keyAt, first, last)
			}

			weights, output := attendTo(query, positions, keyAt, valueAt, attention.scale())
			copy(attention.headPart(combined, position, head), output)
			lookedAtForHead[head][position] = lookedAt{positions: positions, weights: weights}
		}
	}

	attention.lastQueries = queries
	attention.lastLookedAt = lookedAtForHead
	return attention.OutputLayer.Forward(combined)
}

func attendTo(query vectormath.Vector, positions []int, keyAt func(int) vectormath.Vector, valueAt func(int) vectormath.Vector, scale float64) ([]float64, vectormath.Vector) {
	scores := vectormath.NewVector(len(positions))
	for i, position := range positions {
		scores[i] = vectormath.DotProduct(query, keyAt(position)) * scale
	}
	weights := activationfunction.Softmax(scores)

	output := vectormath.NewVector(len(query))
	for i, position := range positions {
		value := valueAt(position)
		for j := range output {
			output[j] += weights[i] * value[j]
		}
	}
	return weights, output
}

func (attention *SelfAttention) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	if attention.lastLookedAt == nil {
		panic(fmt.Sprintf("attention %q: call Forward before Backward", attention.Name))
	}
	sequenceLength := attention.lastQueries.Rows
	scale := attention.scale()

	combinedGradients := attention.OutputLayer.Backward(outputGradients)
	queryGradients := vectormath.NewMatrix(sequenceLength, attention.VectorSize())
	keyGradients := vectormath.NewMatrix(sequenceLength, attention.VectorSize())
	valueGradients := vectormath.NewMatrix(sequenceLength, attention.VectorSize())

	for head := 0; head < attention.NumberOfHeads; head++ {
		for position := 0; position < sequenceLength; position++ {
			looked := attention.lastLookedAt[head][position]
			outputGradient := attention.headPart(combinedGradients, position, head)

			weightGradients := vectormath.NewVector(len(looked.positions))
			for i, otherPosition := range looked.positions {
				otherValue := attention.headPart(attention.lastValues, otherPosition, head)
				weightGradients[i] = vectormath.DotProduct(outputGradient, otherValue)

				otherValueGradient := attention.headPart(valueGradients, otherPosition, head)
				for j := range otherValueGradient {
					otherValueGradient[j] += looked.weights[i] * outputGradient[j]
				}
			}

			scoreGradients := activationfunction.SoftmaxBackward(looked.weights, weightGradients)

			query := attention.headPart(attention.lastQueries, position, head)
			queryGradient := attention.headPart(queryGradients, position, head)
			for i, otherPosition := range looked.positions {
				key := attention.headPart(attention.lastKeys, otherPosition, head)
				keyGradient := attention.headPart(keyGradients, otherPosition, head)
				for j := range query {
					queryGradient[j] += scoreGradients[i] * scale * key[j]
					keyGradient[j] += scoreGradients[i] * scale * query[j]
				}
			}
		}
	}

	inputGradients := attention.QueryLayer.Backward(attention.rotateRows(queryGradients, rotateBackward))

	if attention.SharingMode == OwnKeysAndValues {
		keyGradients = vectormath.AddMatrices(keyGradients, attention.borrowedKeyGradients)
		keyGradients = attention.rotateRows(keyGradients, rotateBackward)
		valueGradients = vectormath.AddMatrices(valueGradients, attention.borrowedValueGradients)
		inputGradients = vectormath.AddMatrices(inputGradients, attention.KeyLayer.Backward(keyGradients))
		inputGradients = vectormath.AddMatrices(inputGradients, attention.ValueLayer.Backward(valueGradients))
		attention.backwardFinished = true
		return inputGradients
	}

	owner := attention.keyValueOwner()
	if owner.backwardFinished {
		panic(fmt.Sprintf("attention %q: borrows from %q, but %q already ran Backward, run Backward in the reverse order of Forward", attention.Name, owner.Name, owner.Name))
	}
	owner.borrowedKeyGradients = vectormath.AddMatrices(owner.borrowedKeyGradients, keyGradients)
	owner.borrowedValueGradients = vectormath.AddMatrices(owner.borrowedValueGradients, valueGradients)
	return inputGradients
}

func (attention *SelfAttention) LookedAt(head int, position int) ([]int, []float64) {
	looked := attention.lastLookedAt[head][position]
	return looked.positions, looked.weights
}

func (attention *SelfAttention) LastAttentionWeights(head int) vectormath.Matrix {
	sequenceLength := len(attention.lastLookedAt[head])
	weights := vectormath.NewMatrix(sequenceLength, sequenceLength)
	for position, looked := range attention.lastLookedAt[head] {
		for i, otherPosition := range looked.positions {
			weights.Set(position, otherPosition, looked.weights[i])
		}
	}
	return weights
}

func (attention *SelfAttention) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	parameters = append(parameters, attention.QueryLayer.Parameters()...)
	if attention.SharingMode == OwnKeysAndValues {
		parameters = append(parameters, attention.KeyLayer.Parameters()...)
		parameters = append(parameters, attention.ValueLayer.Parameters()...)
	}
	parameters = append(parameters, attention.OutputLayer.Parameters()...)
	return parameters
}
