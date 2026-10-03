package attention

import (
	"fmt"
	"github.com/javanhut/GoTransformers/activationfunction"
	"github.com/javanhut/GoTransformers/dropout"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/normalization"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/perceptron"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
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

type Options struct {
	NumberOfHeads           int
	NumberOfKeyValueHeads   int
	HideFutureTokens        bool
	ShareKeyAsValue         bool
	QueryRank               int
	NormalizeQueriesAndKeys bool
	UseAttentionSink        bool
	UseRotaryPositions      bool
	RotaryDimensions        int
	RotaryBase              float64
	RotateHalves            bool
	WindowSize              int
	TopK                    int
	CachePrecision          lowprecision.Precision
	TrainAtCachePrecision   bool
	AttentionDropout        float64
}

type lookedAt struct {
	positions  []int
	weights    []float64
	sinkWeight float64
	dropMask   []float64
}

func keptFraction(dropMask []float64, index int) float64 {
	if dropMask == nil {
		return 1
	}
	return dropMask[index]
}

type SelfAttention struct {
	Name                  string
	NumberOfHeads         int
	NumberOfKeyValueHeads int
	HideFutureTokens      bool
	ShareKeyAsValue       bool
	UseRotaryPositions    bool
	RotaryDimensions      int
	RotaryBase            float64
	RotateHalves          bool
	WindowSize            int
	TopK                  int
	SharingMode           SharingMode
	SharedFrom            *SelfAttention
	CachePrecision        lowprecision.Precision
	TrainAtCachePrecision bool

	QueryDownLayer     *perceptron.Layer
	QueryLayer         *perceptron.Layer
	QueryNorm          *normalization.RMSNorm
	KeyLayer           *perceptron.Layer
	KeyNorm            *normalization.RMSNorm
	ValueLayer         *perceptron.Layer
	OutputLayer        *perceptron.Layer
	SinkLogits         vectormath.Vector
	SinkLogitGradients []float64
	WeightsDropout     *dropout.Dropout

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
	return NewSelfAttentionWithOptions(name, vectorSize, Options{NumberOfHeads: numberOfHeads, HideFutureTokens: hideFutureTokens})
}

func checkOptions(name string, vectorSize int, options Options) {
	if options.NumberOfHeads <= 0 || vectorSize%options.NumberOfHeads != 0 {
		panic(fmt.Sprintf("attention %q: vector size %d must split evenly into %d heads", name, vectorSize, options.NumberOfHeads))
	}
	keyValueHeads := options.NumberOfKeyValueHeads
	if keyValueHeads == 0 {
		keyValueHeads = options.NumberOfHeads
	}
	if keyValueHeads < 0 || options.NumberOfHeads%keyValueHeads != 0 {
		panic(fmt.Sprintf("attention %q: %d query heads must split evenly into %d key/value heads", name, options.NumberOfHeads, keyValueHeads))
	}
	if options.QueryRank < 0 {
		panic(fmt.Sprintf("attention %q: QueryRank can't be negative, got %d", name, options.QueryRank))
	}
}

func NewSelfAttentionWithOptions(name string, vectorSize int, options Options) *SelfAttention {
	checkOptions(name, vectorSize, options)
	keyValueHeads := options.NumberOfKeyValueHeads
	if keyValueHeads == 0 {
		keyValueHeads = options.NumberOfHeads
	}
	headSize := vectorSize / options.NumberOfHeads

	attention := &SelfAttention{
		Name:                  name,
		NumberOfHeads:         options.NumberOfHeads,
		NumberOfKeyValueHeads: keyValueHeads,
		HideFutureTokens:      options.HideFutureTokens,
		ShareKeyAsValue:       options.ShareKeyAsValue,
		UseRotaryPositions:    options.UseRotaryPositions,
		RotaryDimensions:      options.RotaryDimensions,
		RotaryBase:            options.RotaryBase,
		RotateHalves:          options.RotateHalves,
		WindowSize:            options.WindowSize,
		TopK:                  options.TopK,
		SharingMode:           OwnKeysAndValues,
		CachePrecision:        options.CachePrecision,
		TrainAtCachePrecision: options.TrainAtCachePrecision,
	}
	attention.makeQueryAndOutputParts(vectorSize, options)

	attention.KeyLayer = perceptron.NewLayer(name+".key", vectorSize, keyValueHeads*headSize, activationfunction.Linear)
	if !options.ShareKeyAsValue {
		attention.ValueLayer = perceptron.NewLayer(name+".value", vectorSize, keyValueHeads*headSize, activationfunction.Linear)
	}
	if options.NormalizeQueriesAndKeys {
		attention.KeyNorm = normalization.NewRMSNorm(name+".keyNorm", headSize)
	}
	return attention
}

func (attention *SelfAttention) makeQueryAndOutputParts(vectorSize int, options Options) {
	name := attention.Name
	headSize := vectorSize / options.NumberOfHeads
	queryInputSize := vectorSize
	if options.QueryRank > 0 {
		attention.QueryDownLayer = perceptron.NewLayer(name+".queryDown", vectorSize, options.QueryRank, activationfunction.Linear)
		queryInputSize = options.QueryRank
	}
	attention.QueryLayer = perceptron.NewLayer(name+".query", queryInputSize, vectorSize, activationfunction.Linear)
	if options.NormalizeQueriesAndKeys {
		attention.QueryNorm = normalization.NewRMSNorm(name+".queryNorm", headSize)
	}
	if options.UseAttentionSink {
		attention.SinkLogits = vectormath.NewVector(options.NumberOfHeads)
		attention.SinkLogitGradients = vectormath.NewVector(options.NumberOfHeads)
	}
	attention.OutputLayer = perceptron.NewLayer(name+".output", vectorSize, vectorSize, activationfunction.Linear)
	attention.WeightsDropout = dropout.New(options.AttentionDropout)
}

func (attention *SelfAttention) Options() Options {
	queryRank := 0
	if attention.QueryDownLayer != nil {
		queryRank = attention.QueryDownLayer.NumberOfOutputs()
	}
	attentionDropout := 0.0
	if attention.WeightsDropout != nil {
		attentionDropout = attention.WeightsDropout.Rate
	}
	return Options{
		NumberOfHeads:           attention.NumberOfHeads,
		NumberOfKeyValueHeads:   attention.NumberOfKeyValueHeads,
		HideFutureTokens:        attention.HideFutureTokens,
		ShareKeyAsValue:         attention.ShareKeyAsValue,
		QueryRank:               queryRank,
		NormalizeQueriesAndKeys: attention.QueryNorm != nil,
		UseAttentionSink:        attention.SinkLogits != nil,
		UseRotaryPositions:      attention.UseRotaryPositions,
		RotaryDimensions:        attention.RotaryDimensions,
		RotaryBase:              attention.RotaryBase,
		RotateHalves:            attention.RotateHalves,
		WindowSize:              attention.WindowSize,
		TopK:                    attention.TopK,
		CachePrecision:          attention.CachePrecision,
		TrainAtCachePrecision:   attention.TrainAtCachePrecision,
		AttentionDropout:        attentionDropout,
	}
}

func NewBorrowingSelfAttention(name string, sharedFrom *SelfAttention, mode SharingMode) *SelfAttention {
	if sharedFrom == nil {
		panic(fmt.Sprintf("NewBorrowingSelfAttention %q: sharedFrom is nil", name))
	}
	if mode != BorrowKeysAndValues && mode != BorrowKeysValuesAndChoices {
		panic(fmt.Sprintf("NewBorrowingSelfAttention %q: mode must be BorrowKeysAndValues or BorrowKeysValuesAndChoices, got %v", name, mode))
	}
	options := sharedFrom.Options()
	attention := &SelfAttention{
		Name:                  name,
		NumberOfHeads:         options.NumberOfHeads,
		NumberOfKeyValueHeads: options.NumberOfKeyValueHeads,
		HideFutureTokens:      options.HideFutureTokens,
		ShareKeyAsValue:       options.ShareKeyAsValue,
		UseRotaryPositions:    options.UseRotaryPositions,
		RotaryDimensions:      options.RotaryDimensions,
		RotaryBase:            options.RotaryBase,
		RotateHalves:          options.RotateHalves,
		WindowSize:            options.WindowSize,
		TopK:                  options.TopK,
		SharingMode:           mode,
		SharedFrom:            sharedFrom,
	}
	attention.makeQueryAndOutputParts(sharedFrom.VectorSize(), options)
	owner := attention.keyValueOwner()
	owner.borrowers = append(owner.borrowers, attention)
	return attention
}

func (attention *SelfAttention) VectorSize() int {
	return attention.OutputLayer.NumberOfOutputs()
}

func (attention *SelfAttention) HeadSize() int {
	return attention.VectorSize() / attention.NumberOfHeads
}

func (attention *SelfAttention) keyValueSize() int {
	return attention.NumberOfKeyValueHeads * attention.HeadSize()
}

func (attention *SelfAttention) keyValueHeadFor(queryHead int) int {
	queryHeadsPerKeyValueHead := attention.NumberOfHeads / attention.NumberOfKeyValueHeads
	return queryHead / queryHeadsPerKeyValueHead
}

func (attention *SelfAttention) rotaryDimensions() int {
	if attention.RotaryDimensions == 0 {
		return attention.HeadSize()
	}
	return attention.RotaryDimensions
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
	if attention.NumberOfKeyValueHeads <= 0 || attention.NumberOfHeads%attention.NumberOfKeyValueHeads != 0 {
		panic(fmt.Sprintf("attention %q: %d query heads must split evenly into %d key/value heads", attention.Name, attention.NumberOfHeads, attention.NumberOfKeyValueHeads))
	}
	if attention.WindowSize < 0 || attention.TopK < 0 {
		panic(fmt.Sprintf("attention %q: WindowSize and TopK can't be negative, got %d and %d", attention.Name, attention.WindowSize, attention.TopK))
	}
	if attention.UseRotaryPositions {
		rotary := attention.rotaryDimensions()
		if rotary%2 != 0 || rotary > attention.HeadSize() || rotary < 0 {
			panic(fmt.Sprintf("attention %q: rotary dimensions must be even and at most the head size %d, got %d", attention.Name, attention.HeadSize(), rotary))
		}
	}
	if attention.SharingMode == OwnKeysAndValues {
		if attention.KeyLayer == nil || (attention.ValueLayer == nil && !attention.ShareKeyAsValue) {
			panic(fmt.Sprintf("attention %q: owns its keys and values but has no KeyLayer or ValueLayer", attention.Name))
		}
		if attention.KeyLayer.NumberOfOutputs() != attention.keyValueSize() {
			panic(fmt.Sprintf("attention %q: KeyLayer makes %d values but %d key/value heads need %d", attention.Name, attention.KeyLayer.NumberOfOutputs(), attention.NumberOfKeyValueHeads, attention.keyValueSize()))
		}
		return
	}
	if attention.SharedFrom == nil {
		panic(fmt.Sprintf("attention %q: is %v but SharedFrom is nil", attention.Name, attention.SharingMode))
	}
	owner := attention.keyValueOwner()
	sameShape := owner.VectorSize() == attention.VectorSize() && owner.NumberOfHeads == attention.NumberOfHeads && owner.NumberOfKeyValueHeads == attention.NumberOfKeyValueHeads
	samePositions := owner.UseRotaryPositions == attention.UseRotaryPositions && owner.rotary() == attention.rotary() && owner.ShareKeyAsValue == attention.ShareKeyAsValue
	if !sameShape || !samePositions {
		panic(fmt.Sprintf("attention %q: borrows from %q, so both need the same vector size, heads, key/value heads, rotary settings and ShareKeyAsValue", attention.Name, owner.Name))
	}
}

func headSlice(vector vectormath.Vector, head int, headSize int) vectormath.Vector {
	start := head * headSize
	end := start + headSize
	return vector[start:end:end]
}

func (attention *SelfAttention) queryPart(matrix vectormath.Matrix, position int, head int) vectormath.Vector {
	return headSlice(matrix.Row(position), head, attention.HeadSize())
}

func (attention *SelfAttention) keyValuePart(matrix vectormath.Matrix, position int, keyValueHead int) vectormath.Vector {
	return headSlice(matrix.Row(position), keyValueHead, attention.HeadSize())
}

func (attention *SelfAttention) scale() float64 {
	return 1 / math.Sqrt(float64(attention.HeadSize()))
}

func normalizeEachHead(norm *normalization.RMSNorm, matrix vectormath.Matrix, numberOfHeads int) vectormath.Matrix {
	if norm == nil {
		return matrix
	}
	headSize := matrix.Columns / numberOfHeads
	oneHeadPerRow := vectormath.Matrix{Rows: matrix.Rows * numberOfHeads, Columns: headSize, Values: matrix.Values}
	normalized := norm.Forward(oneHeadPerRow)
	return vectormath.Matrix{Rows: matrix.Rows, Columns: matrix.Columns, Values: normalized.Values}
}

func normalizeEachHeadBackward(norm *normalization.RMSNorm, gradients vectormath.Matrix, numberOfHeads int) vectormath.Matrix {
	if norm == nil {
		return gradients
	}
	headSize := gradients.Columns / numberOfHeads
	oneHeadPerRow := vectormath.Matrix{Rows: gradients.Rows * numberOfHeads, Columns: headSize, Values: gradients.Values}
	inputGradients := norm.Backward(oneHeadPerRow)
	return vectormath.Matrix{Rows: gradients.Rows, Columns: gradients.Columns, Values: inputGradients.Values}
}

func (attention *SelfAttention) makeQueries(inputs vectormath.Matrix, firstPosition int) vectormath.Matrix {
	queries := inputs
	if attention.QueryDownLayer != nil {
		queries = attention.QueryDownLayer.Forward(queries)
	}
	queries = attention.QueryLayer.Forward(queries)
	queries = normalizeEachHead(attention.QueryNorm, queries, attention.NumberOfHeads)
	return attention.rotateRows(queries, attention.NumberOfHeads, firstPosition, rotateForward)
}

func (attention *SelfAttention) makeQueriesBackward(queryGradients vectormath.Matrix) vectormath.Matrix {
	gradients := attention.rotateRows(queryGradients, attention.NumberOfHeads, 0, rotateBackward)
	gradients = normalizeEachHeadBackward(attention.QueryNorm, gradients, attention.NumberOfHeads)
	gradients = attention.QueryLayer.Backward(gradients)
	if attention.QueryDownLayer != nil {
		gradients = attention.QueryDownLayer.Backward(gradients)
	}
	return gradients
}

func (attention *SelfAttention) makeKeysAndValues(inputs vectormath.Matrix, firstPosition int) (vectormath.Matrix, vectormath.Matrix) {
	keys := attention.KeyLayer.Forward(inputs)
	keys = normalizeEachHead(attention.KeyNorm, keys, attention.NumberOfKeyValueHeads)
	keys = attention.rotateRows(keys, attention.NumberOfKeyValueHeads, firstPosition, rotateForward)
	if attention.ShareKeyAsValue {
		return keys, keys
	}
	return keys, attention.ValueLayer.Forward(inputs)
}

func (attention *SelfAttention) makeKeysBackward(keyGradients vectormath.Matrix) vectormath.Matrix {
	gradients := attention.rotateRows(keyGradients, attention.NumberOfKeyValueHeads, 0, rotateBackward)
	gradients = normalizeEachHeadBackward(attention.KeyNorm, gradients, attention.NumberOfKeyValueHeads)
	return attention.KeyLayer.Backward(gradients)
}

func (attention *SelfAttention) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	attention.checkSetUp()
	if inputs.Columns != attention.VectorSize() {
		panic(fmt.Sprintf("attention %q: each token vector has %d values but attention expects %d", attention.Name, inputs.Columns, attention.VectorSize()))
	}
	sequenceLength := inputs.Rows
	attention.WeightsDropout.StartPass()
	queries := attention.makeQueries(inputs, 0)

	if attention.SharingMode == OwnKeysAndValues {
		keys, values := attention.makeKeysAndValues(inputs, 0)
		if attention.TrainAtCachePrecision {
			keys = lowprecision.RoundTripMatrix(keys, attention.CachePrecision)
			if attention.ShareKeyAsValue {
				values = keys
			} else {
				values = lowprecision.RoundTripMatrix(values, attention.CachePrecision)
			}
		}
		attention.lastKeys = keys
		attention.lastValues = values
		attention.borrowedKeyGradients = vectormath.NewMatrix(sequenceLength, attention.keyValueSize())
		attention.borrowedValueGradients = vectormath.NewMatrix(sequenceLength, attention.keyValueSize())
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
		keyValueHead := attention.keyValueHeadFor(head)
		keyAt := func(position int) vectormath.Vector {
			return attention.keyValuePart(attention.lastKeys, position, keyValueHead)
		}
		valueAt := func(position int) vectormath.Vector {
			return attention.keyValuePart(attention.lastValues, position, keyValueHead)
		}

		for position := 0; position < sequenceLength; position++ {
			query := attention.queryPart(queries, position, head)

			var positions []int
			if choicesSource != nil {
				positions = choicesSource.lastLookedAt[head][position].positions
			} else {
				first, last := attention.visiblePositions(position, sequenceLength)
				positions = attention.choosePositions(query, keyAt, first, last)
			}

			looked, output := attention.attendTo(query, positions, keyAt, valueAt, head, attention.WeightsDropout.NextMask(len(positions)))
			copy(attention.queryPart(combined, position, head), attention.rotateOutput(output, position, rotateBackward))
			lookedAtForHead[head][position] = looked
		}
	}

	attention.lastQueries = queries
	attention.lastLookedAt = lookedAtForHead
	return attention.OutputLayer.Forward(combined)
}

func (attention *SelfAttention) attendTo(query vectormath.Vector, positions []int, keyAt func(int) vectormath.Vector, valueAt func(int) vectormath.Vector, head int, dropMask []float64) (lookedAt, vectormath.Vector) {
	numberOfScores := len(positions)
	if attention.SinkLogits != nil {
		numberOfScores++
	}
	scores := vectormath.NewVector(numberOfScores)
	for i, position := range positions {
		scores[i] = vectormath.DotProduct(query, keyAt(position)) * attention.scale()
	}
	if attention.SinkLogits != nil {
		scores[len(positions)] = attention.SinkLogits[head]
	}
	allWeights := activationfunction.Softmax(scores)
	looked := lookedAt{positions: positions, weights: allWeights[:len(positions)], dropMask: dropMask}
	if attention.SinkLogits != nil {
		looked.sinkWeight = allWeights[len(positions)]
	}

	output := vectormath.NewVector(len(query))
	for i, position := range positions {
		value := valueAt(position)
		for j := range output {
			output[j] += looked.weights[i] * keptFraction(dropMask, i) * value[j]
		}
	}
	return looked, output
}

func (attention *SelfAttention) scoreGradients(looked lookedAt, weightGradients []float64, head int) []float64 {
	if attention.SinkLogits == nil {
		return activationfunction.SoftmaxBackward(looked.weights, weightGradients)
	}
	allWeights := append(vectormath.CopyVector(looked.weights), looked.sinkWeight)
	allWeightGradients := append(vectormath.CopyVector(weightGradients), 0)
	allScoreGradients := activationfunction.SoftmaxBackward(allWeights, allWeightGradients)
	attention.SinkLogitGradients[head] += allScoreGradients[len(looked.weights)]
	return allScoreGradients[:len(looked.weights)]
}

func (attention *SelfAttention) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	if attention.lastLookedAt == nil {
		panic(fmt.Sprintf("attention %q: call Forward before Backward", attention.Name))
	}
	sequenceLength := attention.lastQueries.Rows
	scale := attention.scale()

	combinedGradients := attention.OutputLayer.Backward(outputGradients)
	queryGradients := vectormath.NewMatrix(sequenceLength, attention.VectorSize())
	keyGradients := vectormath.NewMatrix(sequenceLength, attention.keyValueSize())
	valueGradients := vectormath.NewMatrix(sequenceLength, attention.keyValueSize())

	for head := 0; head < attention.NumberOfHeads; head++ {
		keyValueHead := attention.keyValueHeadFor(head)
		for position := 0; position < sequenceLength; position++ {
			looked := attention.lastLookedAt[head][position]
			outputGradient := attention.rotateOutput(attention.queryPart(combinedGradients, position, head), position, rotateForward)

			weightGradients := vectormath.NewVector(len(looked.positions))
			for i, otherPosition := range looked.positions {
				otherValue := attention.keyValuePart(attention.lastValues, otherPosition, keyValueHead)
				weightGradients[i] = vectormath.DotProduct(outputGradient, otherValue) * keptFraction(looked.dropMask, i)

				otherValueGradient := attention.keyValuePart(valueGradients, otherPosition, keyValueHead)
				for j := range otherValueGradient {
					otherValueGradient[j] += looked.weights[i] * keptFraction(looked.dropMask, i) * outputGradient[j]
				}
			}

			scoreGradients := attention.scoreGradients(looked, weightGradients, head)

			query := attention.queryPart(attention.lastQueries, position, head)
			queryGradient := attention.queryPart(queryGradients, position, head)
			for i, otherPosition := range looked.positions {
				key := attention.keyValuePart(attention.lastKeys, otherPosition, keyValueHead)
				keyGradient := attention.keyValuePart(keyGradients, otherPosition, keyValueHead)
				for j := range query {
					queryGradient[j] += scoreGradients[i] * scale * key[j]
					keyGradient[j] += scoreGradients[i] * scale * query[j]
				}
			}
		}
	}

	inputGradients := attention.makeQueriesBackward(queryGradients)

	if attention.SharingMode == OwnKeysAndValues {
		keyGradients = vectormath.AddMatrices(keyGradients, attention.borrowedKeyGradients)
		valueGradients = vectormath.AddMatrices(valueGradients, attention.borrowedValueGradients)
		if attention.ShareKeyAsValue {
			keyGradients = vectormath.AddMatrices(keyGradients, valueGradients)
		} else {
			inputGradients = vectormath.AddMatrices(inputGradients, attention.ValueLayer.Backward(valueGradients))
		}
		inputGradients = vectormath.AddMatrices(inputGradients, attention.makeKeysBackward(keyGradients))
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
	if attention.QueryDownLayer != nil {
		parameters = append(parameters, attention.QueryDownLayer.Parameters()...)
	}
	parameters = append(parameters, attention.QueryLayer.Parameters()...)
	if attention.QueryNorm != nil {
		parameters = append(parameters, attention.QueryNorm.Parameters()...)
	}
	if attention.SharingMode == OwnKeysAndValues {
		parameters = append(parameters, attention.KeyLayer.Parameters()...)
		if attention.KeyNorm != nil {
			parameters = append(parameters, attention.KeyNorm.Parameters()...)
		}
		if !attention.ShareKeyAsValue {
			parameters = append(parameters, attention.ValueLayer.Parameters()...)
		}
	}
	parameters = append(parameters, attention.OutputLayer.Parameters()...)
	if attention.SinkLogits != nil {
		parameters = append(parameters, parameter.Parameter{Name: attention.Name + ".sinkLogits", Values: attention.SinkLogits, GradientStorage: &attention.SinkLogitGradients, UseAdamW: true})
	}
	return parameters
}

func (attention *SelfAttention) Layers() []*perceptron.Layer {
	var layers []*perceptron.Layer
	if attention.QueryDownLayer != nil {
		layers = append(layers, attention.QueryDownLayer)
	}
	layers = append(layers, attention.QueryLayer)
	if attention.SharingMode == OwnKeysAndValues {
		layers = append(layers, attention.KeyLayer)
		if !attention.ShareKeyAsValue {
			layers = append(layers, attention.ValueLayer)
		}
	}
	return append(layers, attention.OutputLayer)
}
