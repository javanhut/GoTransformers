package attention

import (
	"github.com/javanhut/GoTransformers/gradientcheck"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/optimizer"
	"github.com/javanhut/GoTransformers/parameter"
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"testing"
)

func randomize(matrix vectormath.Matrix) {
	for i := range matrix.Values {
		matrix.Values[i] = vectormath.RandomNumberBetween(-0.5, 0.5)
	}
}

func newCompressed(options CompressedOptions) *CompressedAttention {
	compressed := NewCompressedAttention("compressed", 8, options)
	randomize(compressed.PositionBiases)
	if compressed.Overlap {
		randomize(compressed.OverlapPositionBiases)
	}
	for head := range compressed.SinkLogits {
		compressed.SinkLogits[head] = vectormath.RandomNumberBetween(-1, 1)
	}
	return compressed
}

func compressedSetUps() map[string]CompressedOptions {
	return map[string]CompressedOptions{
		"heavily compressed":            {NumberOfHeads: 2, CompressionRate: 4, WindowSize: 2},
		"heavily compressed sink":       {NumberOfHeads: 2, CompressionRate: 3, WindowSize: 3, UseAttentionSink: true},
		"heavily compressed rotary":     {NumberOfHeads: 2, CompressionRate: 2, WindowSize: 2, UseRotaryPositions: true, RotaryDimensions: 2},
		"compressed sparse":             {NumberOfHeads: 2, CompressionRate: 2, WindowSize: 2, TopK: 2},
		"compressed sparse overlap":     {NumberOfHeads: 2, CompressionRate: 2, WindowSize: 3, TopK: 2, Overlap: true, UseAttentionSink: true, UseRotaryPositions: true},
		"compressed sparse wide heads":  {NumberOfHeads: 3, HeadSize: 6, CompressionRate: 2, WindowSize: 2, TopK: 1, NumberOfIndexerHeads: 3, IndexerHeadSize: 4},
		"compression rate 1 everything": {NumberOfHeads: 2, CompressionRate: 1, WindowSize: 2, TopK: 3, Overlap: true, UseAttentionSink: true, UseRotaryPositions: true},
	}
}

func TestCompressedGradients(t *testing.T) {
	for name, options := range compressedSetUps() {
		compressed := newCompressed(options)
		compressed.IndexerLossWeight = 0
		inputs := vectormath.NewRandomMatrix(11, 8, -1, 1)
		for _, problem := range gradientcheck.Compare(compressed.Forward, compressed.Backward, compressed.Parameters(), inputs) {
			t.Errorf("%s: %s", name, problem)
		}
	}
}

func TestCompressedGeneratingMatchesTraining(t *testing.T) {
	setUps := compressedSetUps()
	setUps["FP4 cache"] = CompressedOptions{NumberOfHeads: 2, CompressionRate: 2, WindowSize: 3, TopK: 2, UseRotaryPositions: true, CachePrecision: lowprecision.FP4, TrainAtCachePrecision: true}
	for name, options := range setUps {
		compressed := newCompressed(options)
		inputs := vectormath.NewRandomMatrix(17, 8, -1, 1)
		allAtOnce := compressed.Forward(inputs)

		compressed.StartGenerating()
		for position := 0; position < inputs.Rows; position++ {
			oneAtATime := compressed.ForwardOneToken(inputs.Row(position))
			for i, value := range oneAtATime {
				if math.Abs(value-allAtOnce.Get(position, i)) > 1e-9 {
					t.Errorf("%s position %d value %d: one at a time %v, all at once %v", name, position, i, value, allAtOnce.Get(position, i))
					break
				}
			}
		}
	}
}

func TestCompressionWeightsAddUpToOnePerChannel(t *testing.T) {
	var sources []compressionSource
	for slot := 0; slot < 5; slot++ {
		sources = append(sources, compressionSource{
			entry:  vectormath.NewRandomMatrix(1, 3, -1, 1).Row(0),
			weight: vectormath.NewRandomMatrix(1, 3, -2, 2).Row(0),
			bias:   vectormath.NewRandomMatrix(1, 3, -1, 1).Row(0),
		})
	}
	_, softmaxWeights := compress(sources, 3)
	for channel := 0; channel < 3; channel++ {
		total := 0.0
		for slot := range sources {
			total += softmaxWeights[slot][channel]
		}
		if math.Abs(total-1) > 1e-12 {
			t.Errorf("channel %d weights add up to %v", channel, total)
		}
	}
}

func TestIndexerLearnsWhereAttentionLooks(t *testing.T) {
	compressed := newCompressed(CompressedOptions{NumberOfHeads: 2, CompressionRate: 2, WindowSize: 1, TopK: 3})
	compressed.AttendToAllWhileTraining = true
	inputs := vectormath.NewRandomMatrix(24, 8, -1, 1)

	var indexerParameters []parameter.Parameter
	for _, current := range compressed.Parameters() {
		if len(current.Name) > len("compressed.indexer") && current.Name[:len("compressed.indexer")] == "compressed.indexer" {
			indexerParameters = append(indexerParameters, current)
		}
	}
	adam := optimizer.NewAdam(0.01)
	compressed.Forward(inputs)
	firstLoss := compressed.LastIndexerLoss()
	for step := 0; step < 300; step++ {
		parameter.ZeroGradients(compressed.Parameters())
		outputs := compressed.Forward(inputs)
		compressed.Backward(vectormath.NewMatrix(outputs.Rows, outputs.Columns))
		adam.Update(indexerParameters)
	}
	compressed.Forward(inputs)
	lastLoss := compressed.LastIndexerLoss()
	if !(lastLoss < firstLoss/2) {
		t.Errorf("indexer loss only went from %v to %v", firstLoss, lastLoss)
	}
}

func TestCompressedCacheIsSmall(t *testing.T) {
	full := withOptions(Options{NumberOfHeads: 2})
	compressed := newCompressed(CompressedOptions{NumberOfHeads: 2, CompressionRate: 8, WindowSize: 4})
	full.StartGenerating()
	compressed.StartGenerating()
	for position := 0; position < 64; position++ {
		input := vectormath.NewRandomMatrix(1, 8, -1, 1).Row(0)
		full.ForwardOneToken(input)
		compressed.ForwardOneToken(input)
	}
	if compressed.CacheBytesUsed()*5 > full.CacheBytesUsed() {
		t.Errorf("compressed cache %d bytes, full cache %d bytes", compressed.CacheBytesUsed(), full.CacheBytesUsed())
	}
}

func TestCompressedMistakesPanic(t *testing.T) {
	expectPanic(t, "window size 0", func() {
		NewCompressedAttention("compressed", 8, CompressedOptions{NumberOfHeads: 2, CompressionRate: 2})
	})
	expectPanic(t, "compression rate 0", func() { NewCompressedAttention("compressed", 8, CompressedOptions{NumberOfHeads: 2, WindowSize: 2}) })
	expectPanic(t, "Backward before Forward", func() {
		NewCompressedAttention("compressed", 8, CompressedOptions{NumberOfHeads: 2, CompressionRate: 2, WindowSize: 2}).Backward(vectormath.NewMatrix(3, 8))
	})
}
