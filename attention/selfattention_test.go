package attention

import (
	"math"
	"testing"
	"transformer/gradientcheck"
	"transformer/lowprecision"
	"transformer/parameter"
	"transformer/vectormath"
)

type stack struct {
	layers []*SelfAttention
}

func (layers stack) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	values := inputs
	for _, layer := range layers.layers {
		values = vectormath.AddMatrices(values, layer.Forward(values))
	}
	return values
}

func (layers stack) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	gradients := outputGradients
	for i := len(layers.layers) - 1; i >= 0; i-- {
		gradients = vectormath.AddMatrices(gradients, layers.layers[i].Backward(gradients))
	}
	return gradients
}

func (layers stack) ForwardOneToken(input vectormath.Vector) vectormath.Vector {
	values := input
	for _, layer := range layers.layers {
		values = vectormath.Add(values, layer.ForwardOneToken(values))
	}
	return values
}

func (layers stack) StartGenerating() {
	for _, layer := range layers.layers {
		layer.StartGenerating()
	}
}

func (layers stack) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	for _, layer := range layers.layers {
		parameters = append(parameters, layer.Parameters()...)
	}
	return parameters
}

func single(layer *SelfAttention) stack {
	return stack{layers: []*SelfAttention{layer}}
}

func withSettings(windowSize int, topK int) *SelfAttention {
	layer := NewSelfAttention("attention", 8, 2, true)
	layer.WindowSize = windowSize
	layer.TopK = topK
	return layer
}

func sharingChain(windowSize int, topK int) stack {
	owner := withSettings(windowSize, topK)
	owner.Name = "owner"
	borrowsKeysAndValues := NewBorrowingSelfAttention("borrowsKeysAndValues", owner, BorrowKeysAndValues)
	borrowsChoices := NewBorrowingSelfAttention("borrowsChoices", borrowsKeysAndValues, BorrowKeysValuesAndChoices)
	return stack{layers: []*SelfAttention{owner, borrowsKeysAndValues, borrowsChoices}}
}

func withOptions(options Options) *SelfAttention {
	options.HideFutureTokens = true
	return NewSelfAttentionWithOptions("attention", 8, options)
}

func withSink(layer *SelfAttention) *SelfAttention {
	for head := range layer.SinkLogits {
		layer.SinkLogits[head] = vectormath.RandomNumberBetween(-1, 1)
	}
	return layer
}

func everythingChain() stack {
	owner := withSink(withOptions(Options{
		NumberOfHeads:           4,
		NumberOfKeyValueHeads:   2,
		ShareKeyAsValue:         true,
		QueryRank:               5,
		NormalizeQueriesAndKeys: true,
		UseAttentionSink:        true,
		UseRotaryPositions:      true,
		RotaryDimensions:        2,
		WindowSize:              4,
		TopK:                    2,
	}))
	owner.Name = "owner"
	borrowsKeysAndValues := withSink(NewBorrowingSelfAttention("borrowsKeysAndValues", owner, BorrowKeysAndValues))
	borrowsChoices := withSink(NewBorrowingSelfAttention("borrowsChoices", borrowsKeysAndValues, BorrowKeysValuesAndChoices))
	return stack{layers: []*SelfAttention{owner, borrowsKeysAndValues, borrowsChoices}}
}

func rotary(layer *SelfAttention) *SelfAttention {
	layer.UseRotaryPositions = true
	return layer
}

func rotarySharingChain() stack {
	owner := rotary(withSettings(0, 2))
	owner.Name = "owner"
	borrowsKeysAndValues := NewBorrowingSelfAttention("borrowsKeysAndValues", owner, BorrowKeysAndValues)
	borrowsChoices := NewBorrowingSelfAttention("borrowsChoices", owner, BorrowKeysValuesAndChoices)
	return stack{layers: []*SelfAttention{owner, borrowsKeysAndValues, borrowsChoices}}
}

func TestRotaryUndoesItself(t *testing.T) {
	vector := vectormath.Vector{1, 2, 3, 4, 5, 6, 7, 8}
	rotated := rotateEachHead(vector, 7, 2, 4, 4, rotateForward)
	back := rotateEachHead(rotated, 7, 2, 4, 4, rotateBackward)
	for i := range vector {
		if math.Abs(back[i]-vector[i]) > 1e-12 {
			t.Errorf("value %d: started %v, rotated and back %v", i, vector[i], back[i])
		}
	}
	if math.Abs(vectormath.Magnitude(rotated)-vectormath.Magnitude(vector)) > 1e-12 {
		t.Errorf("rotation changed the length of the vector")
	}
}

func TestRotaryOnlyCaresAboutDistance(t *testing.T) {
	query := vectormath.Vector{0.3, -1, 0.5, 2}
	key := vectormath.Vector{1, 0.2, -0.7, 0.4}
	near := vectormath.DotProduct(rotateEachHead(query, 5, 1, 4, 4, rotateForward), rotateEachHead(key, 3, 1, 4, 4, rotateForward))
	far := vectormath.DotProduct(rotateEachHead(query, 105, 1, 4, 4, rotateForward), rotateEachHead(key, 103, 1, 4, 4, rotateForward))
	if math.Abs(near-far) > 1e-9 {
		t.Errorf("positions 5 and 3 scored %v but positions 105 and 103 scored %v", near, far)
	}
}

func lowPrecisionLayer(precision lowprecision.Precision) *SelfAttention {
	layer := withSettings(0, 0)
	layer.CachePrecision = precision
	layer.TrainAtCachePrecision = true
	return layer
}

func testStacks() map[string]stack {
	return map[string]stack{
		"plain":                           single(withSettings(0, 0)),
		"window 3":                        single(withSettings(3, 0)),
		"top 2":                           single(withSettings(0, 2)),
		"window 4 top 2":                  single(withSettings(4, 2)),
		"sharing":                         sharingChain(0, 0),
		"sharing window 3 top 2":          sharingChain(3, 2),
		"two heads both ways":             single(NewSelfAttention("attention", 8, 2, false)),
		"rotary":                          single(rotary(withSettings(0, 0))),
		"rotary sharing top 2":            rotarySharingChain(),
		"grouped query":                   single(withOptions(Options{NumberOfHeads: 4, NumberOfKeyValueHeads: 2})),
		"multi query":                     single(withOptions(Options{NumberOfHeads: 4, NumberOfKeyValueHeads: 1})),
		"shared key value":                single(withOptions(Options{NumberOfHeads: 2, ShareKeyAsValue: true})),
		"shared key value rotary partial": single(withOptions(Options{NumberOfHeads: 2, ShareKeyAsValue: true, UseRotaryPositions: true, RotaryDimensions: 2})),
		"query and key norm":              single(withOptions(Options{NumberOfHeads: 2, NormalizeQueriesAndKeys: true, UseRotaryPositions: true})),
		"attention sink":                  single(withSink(withOptions(Options{NumberOfHeads: 2, UseAttentionSink: true, WindowSize: 3}))),
		"low rank queries":                single(withOptions(Options{NumberOfHeads: 2, QueryRank: 3})),
		"everything":                      everythingChain(),
	}
}

func TestGradients(t *testing.T) {
	for name, layers := range testStacks() {
		inputs := vectormath.NewRandomMatrix(6, 8, -1, 1)
		for _, problem := range gradientcheck.Compare(layers.Forward, layers.Backward, layers.Parameters(), inputs) {
			t.Errorf("%s: %s", name, problem)
		}
	}
}

func TestGeneratingOneTokenAtATimeMatchesTraining(t *testing.T) {
	stacks := testStacks()
	delete(stacks, "two heads both ways")
	stacks["Float32 cache"] = single(lowPrecisionLayer(lowprecision.Float32))
	stacks["Int8 cache"] = single(lowPrecisionLayer(lowprecision.Int8))
	stacks["FP4 cache"] = single(lowPrecisionLayer(lowprecision.FP4))
	fp4Shared := withOptions(Options{NumberOfHeads: 4, NumberOfKeyValueHeads: 1, ShareKeyAsValue: true, UseRotaryPositions: true, CachePrecision: lowprecision.FP4, TrainAtCachePrecision: true})
	stacks["FP4 multi query shared key value"] = single(fp4Shared)

	for name, layers := range stacks {
		inputs := vectormath.NewRandomMatrix(10, 8, -1, 1)
		allAtOnce := layers.Forward(inputs)

		layers.StartGenerating()
		for position := 0; position < inputs.Rows; position++ {
			oneAtATime := layers.ForwardOneToken(inputs.Row(position))
			for i, value := range oneAtATime {
				if math.Abs(value-allAtOnce.Get(position, i)) > 1e-9 {
					t.Errorf("%s position %d value %d: one at a time %v, all at once %v", name, position, i, value, allAtOnce.Get(position, i))
				}
			}
		}
	}
}

func TestWindowKeepsTheCacheSmall(t *testing.T) {
	layer := withSettings(3, 0)
	layer.StartGenerating()
	for position := 0; position < 20; position++ {
		layer.ForwardOneToken(vectormath.NewRandomMatrix(1, 8, -1, 1).Row(0))
	}
	if layer.generation.keys.NumberOfRows() != 3 {
		t.Errorf("window of 3 kept %d rows in the cache", layer.generation.keys.NumberOfRows())
	}
}

func TestLowerPrecisionCacheIsSmaller(t *testing.T) {
	bytesUsed := map[lowprecision.Precision]int{}
	for _, precision := range []lowprecision.Precision{lowprecision.Float64, lowprecision.FP4} {
		layer := withSettings(0, 0)
		layer.CachePrecision = precision
		layer.StartGenerating()
		for position := 0; position < 16; position++ {
			layer.ForwardOneToken(vectormath.NewRandomMatrix(1, 8, -1, 1).Row(0))
		}
		bytesUsed[precision] = layer.CacheBytesUsed()
	}
	if bytesUsed[lowprecision.FP4]*5 > bytesUsed[lowprecision.Float64] {
		t.Errorf("FP4 cache used %d bytes, Float64 used %d", bytesUsed[lowprecision.FP4], bytesUsed[lowprecision.Float64])
	}
}

func TestSmallerCacheOptions(t *testing.T) {
	cacheBytes := func(options Options) int {
		layer := withOptions(options)
		layer.StartGenerating()
		for position := 0; position < 10; position++ {
			layer.ForwardOneToken(vectormath.NewRandomMatrix(1, 8, -1, 1).Row(0))
		}
		return layer.CacheBytesUsed()
	}
	full := cacheBytes(Options{NumberOfHeads: 4})
	multiQuery := cacheBytes(Options{NumberOfHeads: 4, NumberOfKeyValueHeads: 1})
	multiQueryShared := cacheBytes(Options{NumberOfHeads: 4, NumberOfKeyValueHeads: 1, ShareKeyAsValue: true})
	if multiQuery*4 != full || multiQueryShared*2 != multiQuery {
		t.Errorf("cache bytes: full %d, multi query %d, multi query sharing key as value %d", full, multiQuery, multiQueryShared)
	}
}

func TestAttentionSinkCanIgnoreEverything(t *testing.T) {
	layer := withOptions(Options{NumberOfHeads: 2, UseAttentionSink: true})
	for head := range layer.SinkLogits {
		layer.SinkLogits[head] = 50
	}
	layer.Forward(vectormath.NewRandomMatrix(4, 8, -1, 1))
	_, weights := layer.LookedAt(0, 3)
	if vectormath.Sum(weights) > 1e-6 {
		t.Errorf("with a huge sink logit the weights should add up to almost 0, got %v", vectormath.Sum(weights))
	}
}

func TestTopKOnlyLooksAtK(t *testing.T) {
	layer := withSettings(0, 2)
	layer.Forward(vectormath.NewRandomMatrix(6, 8, -1, 1))
	for position := 0; position < 6; position++ {
		positions, _ := layer.LookedAt(0, position)
		if len(positions) > 2 {
			t.Errorf("position %d looked at %v", position, positions)
		}
	}
}

func TestHideFutureTokens(t *testing.T) {
	attention := NewSelfAttention("attention", 4, 2, true)
	attention.Forward(vectormath.NewRandomMatrix(5, 4, -1, 1))
	for head := 0; head < 2; head++ {
		weights := attention.LastAttentionWeights(head)
		for position := 0; position < 5; position++ {
			for otherPosition := position + 1; otherPosition < 5; otherPosition++ {
				if weights.Get(position, otherPosition) != 0 {
					t.Errorf("head %d: position %d looked at future position %d", head, position, otherPosition)
				}
			}
		}
	}
}

func expectPanic(t *testing.T, name string, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s did not panic", name)
		}
	}()
	function()
}

func TestMistakesPanic(t *testing.T) {
	expectPanic(t, "6 values into 4 heads", func() { NewSelfAttention("attention", 6, 4, false) })

	owner := NewSelfAttention("owner", 4, 2, true)
	borrower := NewBorrowingSelfAttention("borrower", owner, BorrowKeysAndValues)
	expectPanic(t, "borrower before owner", func() { borrower.Forward(vectormath.NewMatrix(3, 4)) })

	owner.Forward(vectormath.NewMatrix(3, 4))
	borrower.Forward(vectormath.NewMatrix(3, 4))
	owner.Backward(vectormath.NewMatrix(3, 4))
	expectPanic(t, "owner backward before borrower backward", func() { borrower.Backward(vectormath.NewMatrix(3, 4)) })

	expectPanic(t, "generating without hiding future tokens", func() { NewSelfAttention("both ways", 4, 2, false).StartGenerating() })
}
