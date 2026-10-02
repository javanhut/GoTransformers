package attention

import (
	"fmt"
	"math"
	"sort"
	"transformer/activationfunction"
	"transformer/lowprecision"
	"transformer/normalization"
	"transformer/parameter"
	"transformer/perceptron"
	"transformer/vectormath"
)

type CompressedOptions struct {
	NumberOfHeads         int
	HeadSize              int
	CompressionRate       int
	Overlap               bool
	TopK                  int
	NumberOfIndexerHeads  int
	IndexerHeadSize       int
	WindowSize            int
	UseAttentionSink      bool
	UseRotaryPositions    bool
	RotaryDimensions      int
	CachePrecision        lowprecision.Precision
	TrainAtCachePrecision bool
}

type entryReference struct {
	fromBlock bool
	index     int
}

type compressedLookedAt struct {
	weights    []float64
	sinkWeight float64
}

type compressedMemory struct {
	queries            vectormath.Matrix
	rawEntries         vectormath.Matrix
	rawWeights         vectormath.Matrix
	overlapEntries     vectormath.Matrix
	overlapWeights     vectormath.Matrix
	blockSources       [][]compressionSource
	blockSoftmax       [][][]float64
	compressedEntries  vectormath.Matrix
	windowEntries      vectormath.Matrix
	entriesForPosition [][]entryReference
	chosenBlocks       [][]int
	looked             [][]compressedLookedAt
	indexerQueries     vectormath.Matrix
	indexerHeadWeights vectormath.Matrix
	indexerKeys        vectormath.Matrix
	indexerGradients   [][]float64
	usedIndexer        bool
	numberOfBlocks     int
	sequenceLength     int
	forwardHasFinished bool
}

type CompressedAttention struct {
	Name                     string
	NumberOfHeads            int
	CompressionRate          int
	Overlap                  bool
	TopK                     int
	WindowSize               int
	UseRotaryPositions       bool
	RotaryDimensions         int
	IndexerLossWeight        float64
	AttendToAllWhileTraining bool
	CachePrecision           lowprecision.Precision
	TrainAtCachePrecision    bool

	QueryLayer                   *perceptron.Layer
	QueryNorm                    *normalization.RMSNorm
	EntryLayer                   *perceptron.Layer
	WeightLayer                  *perceptron.Layer
	PositionBiases               vectormath.Matrix
	PositionBiasGradients        vectormath.Matrix
	OverlapEntryLayer            *perceptron.Layer
	OverlapWeightLayer           *perceptron.Layer
	OverlapPositionBiases        vectormath.Matrix
	OverlapPositionBiasGradients vectormath.Matrix
	EntryNorm                    *normalization.RMSNorm
	SinkLogits                   vectormath.Vector
	SinkLogitGradients           vectormath.Vector
	IndexerQueryLayer            *perceptron.Layer
	IndexerHeadWeightLayer       *perceptron.Layer
	IndexerKeyLayer              *perceptron.Layer
	OutputLayer                  *perceptron.Layer

	last            compressedMemory
	lastIndexerLoss float64
	generation      *compressedGenerationState
}

func NewCompressedAttention(name string, vectorSize int, options CompressedOptions) *CompressedAttention {
	if options.NumberOfHeads <= 0 {
		panic(fmt.Sprintf("compressed attention %q: needs at least 1 head, got %d", name, options.NumberOfHeads))
	}
	headSize := options.HeadSize
	if headSize == 0 {
		if vectorSize%options.NumberOfHeads != 0 {
			panic(fmt.Sprintf("compressed attention %q: vector size %d must split evenly into %d heads, or set HeadSize", name, vectorSize, options.NumberOfHeads))
		}
		headSize = vectorSize / options.NumberOfHeads
	}
	if options.CompressionRate < 1 {
		panic(fmt.Sprintf("compressed attention %q: CompressionRate must be at least 1, got %d", name, options.CompressionRate))
	}
	if options.WindowSize < 1 {
		panic(fmt.Sprintf("compressed attention %q: WindowSize must be at least 1 so every token can see something, got %d", name, options.WindowSize))
	}
	allHeadsSize := options.NumberOfHeads * headSize
	compressed := &CompressedAttention{
		Name:                  name,
		NumberOfHeads:         options.NumberOfHeads,
		CompressionRate:       options.CompressionRate,
		Overlap:               options.Overlap,
		TopK:                  options.TopK,
		WindowSize:            options.WindowSize,
		UseRotaryPositions:    options.UseRotaryPositions,
		RotaryDimensions:      options.RotaryDimensions,
		IndexerLossWeight:     1,
		CachePrecision:        options.CachePrecision,
		TrainAtCachePrecision: options.TrainAtCachePrecision,
		QueryLayer:            perceptron.NewLayer(name+".query", vectorSize, allHeadsSize, activationfunction.Linear),
		QueryNorm:             normalization.NewRMSNorm(name+".queryNorm", headSize),
		EntryLayer:            perceptron.NewLayer(name+".entry", vectorSize, headSize, activationfunction.Linear),
		WeightLayer:           perceptron.NewLayer(name+".compressionWeight", vectorSize, headSize, activationfunction.Linear),
		PositionBiases:        vectormath.NewMatrix(options.CompressionRate, headSize),
		PositionBiasGradients: vectormath.NewMatrix(options.CompressionRate, headSize),
		EntryNorm:             normalization.NewRMSNorm(name+".entryNorm", headSize),
		OutputLayer:           perceptron.NewLayer(name+".output", allHeadsSize, vectorSize, activationfunction.Linear),
	}
	if options.Overlap {
		compressed.OverlapEntryLayer = perceptron.NewLayer(name+".overlapEntry", vectorSize, headSize, activationfunction.Linear)
		compressed.OverlapWeightLayer = perceptron.NewLayer(name+".overlapCompressionWeight", vectorSize, headSize, activationfunction.Linear)
		compressed.OverlapPositionBiases = vectormath.NewMatrix(options.CompressionRate, headSize)
		compressed.OverlapPositionBiasGradients = vectormath.NewMatrix(options.CompressionRate, headSize)
	}
	if options.UseAttentionSink {
		compressed.SinkLogits = vectormath.NewVector(options.NumberOfHeads)
		compressed.SinkLogitGradients = vectormath.NewVector(options.NumberOfHeads)
	}
	if options.TopK > 0 {
		indexerHeads := options.NumberOfIndexerHeads
		if indexerHeads == 0 {
			indexerHeads = 2
		}
		indexerHeadSize := options.IndexerHeadSize
		if indexerHeadSize == 0 {
			indexerHeadSize = headSize
		}
		compressed.IndexerQueryLayer = perceptron.NewLayer(name+".indexerQuery", vectorSize, indexerHeads*indexerHeadSize, activationfunction.Linear)
		compressed.IndexerHeadWeightLayer = perceptron.NewLayer(name+".indexerHeadWeight", vectorSize, indexerHeads, activationfunction.Linear)
		compressed.IndexerKeyLayer = perceptron.NewLayer(name+".indexerKey", headSize, indexerHeadSize, activationfunction.Linear)
	}
	return compressed
}

func (compressed *CompressedAttention) VectorSize() int {
	return compressed.OutputLayer.NumberOfOutputs()
}

func (compressed *CompressedAttention) HeadSize() int {
	return compressed.EntryLayer.NumberOfOutputs()
}

func (compressed *CompressedAttention) numberOfIndexerHeads() int {
	return compressed.IndexerHeadWeightLayer.NumberOfOutputs()
}

func (compressed *CompressedAttention) indexerHeadSize() int {
	return compressed.IndexerKeyLayer.NumberOfOutputs()
}

func (compressed *CompressedAttention) rotaryDimensions() int {
	if compressed.RotaryDimensions == 0 {
		return compressed.HeadSize()
	}
	return compressed.RotaryDimensions
}

func (compressed *CompressedAttention) scale() float64 {
	return 1 / math.Sqrt(float64(compressed.HeadSize()))
}

func (compressed *CompressedAttention) checkSetUp() {
	if compressed.CompressionRate < 1 || compressed.WindowSize < 1 || compressed.TopK < 0 {
		panic(fmt.Sprintf("compressed attention %q: needs CompressionRate >= 1, WindowSize >= 1 and TopK >= 0, got %d, %d and %d", compressed.Name, compressed.CompressionRate, compressed.WindowSize, compressed.TopK))
	}
	if compressed.PositionBiases.Rows != compressed.CompressionRate {
		panic(fmt.Sprintf("compressed attention %q: CompressionRate %d doesn't match its %d position biases, make it with NewCompressedAttention", compressed.Name, compressed.CompressionRate, compressed.PositionBiases.Rows))
	}
	if compressed.Overlap && compressed.OverlapEntryLayer == nil {
		panic(fmt.Sprintf("compressed attention %q: Overlap is on but it was made without overlap layers", compressed.Name))
	}
	if compressed.TopK > 0 && compressed.IndexerKeyLayer == nil {
		panic(fmt.Sprintf("compressed attention %q: TopK is on but it was made without an indexer", compressed.Name))
	}
	if compressed.UseRotaryPositions {
		rotary := compressed.rotaryDimensions()
		if rotary%2 != 0 || rotary > compressed.HeadSize() || rotary < 0 {
			panic(fmt.Sprintf("compressed attention %q: rotary dimensions must be even and at most the head size %d, got %d", compressed.Name, compressed.HeadSize(), rotary))
		}
	}
}

func (compressed *CompressedAttention) blockPosition(block int) int {
	return (block+1)*compressed.CompressionRate - 1
}

func (compressed *CompressedAttention) rotateRowsAt(matrix vectormath.Matrix, numberOfHeads int, positionOfRow func(int) int, direction float64) vectormath.Matrix {
	if !compressed.UseRotaryPositions {
		return matrix
	}
	rotated := vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	headSize := matrix.Columns / numberOfHeads
	for row := 0; row < matrix.Rows; row++ {
		rotated.SetRow(row, rotateEachHead(matrix.Row(row), positionOfRow(row), numberOfHeads, headSize, compressed.rotaryDimensions(), direction))
	}
	return rotated
}

func (compressed *CompressedAttention) rotateVectorAt(vector vectormath.Vector, numberOfHeads int, position int, direction float64) vectormath.Vector {
	if !compressed.UseRotaryPositions {
		return vector
	}
	return rotateEachHead(vector, position, numberOfHeads, len(vector)/numberOfHeads, compressed.rotaryDimensions(), direction)
}

func samePosition(row int) int {
	return row
}

func (compressed *CompressedAttention) makeQueries(inputs vectormath.Matrix, firstPosition int) vectormath.Matrix {
	queries := compressed.QueryLayer.Forward(inputs)
	queries = normalizeEachHead(compressed.QueryNorm, queries, compressed.NumberOfHeads)
	return compressed.rotateRowsAt(queries, compressed.NumberOfHeads, func(row int) int { return firstPosition + row }, rotateForward)
}

func (compressed *CompressedAttention) blockSources(block int, entries vectormath.Matrix, weights vectormath.Matrix, overlapEntries vectormath.Matrix, overlapWeights vectormath.Matrix) []compressionSource {
	var sources []compressionSource
	for slot := 0; slot < compressed.CompressionRate; slot++ {
		row := block*compressed.CompressionRate + slot
		sources = append(sources, compressionSource{entry: entries.Row(row), weight: weights.Row(row), bias: compressed.PositionBiases.Row(slot), entryRow: row, biasRow: slot})
	}
	if compressed.Overlap && block > 0 {
		for slot := 0; slot < compressed.CompressionRate; slot++ {
			row := (block-1)*compressed.CompressionRate + slot
			sources = append(sources, compressionSource{entry: overlapEntries.Row(row), weight: overlapWeights.Row(row), bias: compressed.OverlapPositionBiases.Row(slot), entryRow: row, biasRow: slot, fromOverlap: true})
		}
	}
	return sources
}

func (compressed *CompressedAttention) chooseBlocks(visibleBlocks int, indexerScore func(int) float64, attendToAll bool) []int {
	var candidates []int
	for block := 0; block < visibleBlocks; block++ {
		candidates = append(candidates, block)
	}
	if compressed.TopK == 0 || attendToAll || len(candidates) <= compressed.TopK {
		return candidates
	}
	scores := make(map[int]float64, len(candidates))
	for _, block := range candidates {
		scores[block] = indexerScore(block)
	}
	sort.SliceStable(candidates, func(i int, j int) bool {
		return scores[candidates[i]] > scores[candidates[j]]
	})
	chosen := candidates[:compressed.TopK]
	sort.Ints(chosen)
	return chosen
}

func indexerScoreOf(indexerQuery vectormath.Vector, headWeights vectormath.Vector, indexerKey vectormath.Vector) float64 {
	score := 0.0
	headSize := len(indexerKey)
	for head, headWeight := range headWeights {
		dot := vectormath.DotProduct(headSlice(indexerQuery, head, headSize), indexerKey)
		if dot > 0 {
			score += headWeight * dot
		}
	}
	return score
}

func attendToEntries(query vectormath.Vector, entries []vectormath.Vector, scale float64, sinkLogits vectormath.Vector, head int) (compressedLookedAt, vectormath.Vector) {
	numberOfScores := len(entries)
	if sinkLogits != nil {
		numberOfScores++
	}
	scores := vectormath.NewVector(numberOfScores)
	for i, entry := range entries {
		scores[i] = vectormath.DotProduct(query, entry) * scale
	}
	if sinkLogits != nil {
		scores[len(entries)] = sinkLogits[head]
	}
	allWeights := activationfunction.Softmax(scores)
	looked := compressedLookedAt{weights: allWeights[:len(entries)]}
	if sinkLogits != nil {
		looked.sinkWeight = allWeights[len(entries)]
	}
	output := vectormath.NewVector(len(query))
	for i, entry := range entries {
		for j := range output {
			output[j] += looked.weights[i] * entry[j]
		}
	}
	return looked, output
}

func (compressed *CompressedAttention) entriesFor(references []entryReference, compressedEntries vectormath.Matrix, windowEntries vectormath.Matrix) []vectormath.Vector {
	entries := make([]vectormath.Vector, len(references))
	for i, reference := range references {
		if reference.fromBlock {
			entries[i] = compressedEntries.Row(reference.index)
		} else {
			entries[i] = windowEntries.Row(reference.index)
		}
	}
	return entries
}

func (compressed *CompressedAttention) Forward(inputs vectormath.Matrix) vectormath.Matrix {
	compressed.checkSetUp()
	if inputs.Columns != compressed.VectorSize() {
		panic(fmt.Sprintf("compressed attention %q: each token vector has %d values but it expects %d", compressed.Name, inputs.Columns, compressed.VectorSize()))
	}
	sequenceLength := inputs.Rows
	numberOfBlocks := sequenceLength / compressed.CompressionRate
	memory := compressedMemory{sequenceLength: sequenceLength, numberOfBlocks: numberOfBlocks}

	memory.queries = compressed.makeQueries(inputs, 0)
	memory.rawEntries = compressed.EntryLayer.Forward(inputs)
	memory.rawWeights = compressed.WeightLayer.Forward(inputs)
	if compressed.Overlap {
		memory.overlapEntries = compressed.OverlapEntryLayer.Forward(inputs)
		memory.overlapWeights = compressed.OverlapWeightLayer.Forward(inputs)
	}

	compressedRaw := vectormath.NewMatrix(numberOfBlocks, compressed.HeadSize())
	for block := 0; block < numberOfBlocks; block++ {
		sources := compressed.blockSources(block, memory.rawEntries, memory.rawWeights, memory.overlapEntries, memory.overlapWeights)
		entry, softmaxWeights := compress(sources, compressed.HeadSize())
		compressedRaw.SetRow(block, entry)
		memory.blockSources = append(memory.blockSources, sources)
		memory.blockSoftmax = append(memory.blockSoftmax, softmaxWeights)
	}

	normalized := compressed.EntryNorm.Forward(vectormath.StackRows(compressedRaw, memory.rawEntries))
	normalizedCompressed, normalizedWindow := vectormath.SplitRows(normalized, numberOfBlocks)
	memory.compressedEntries = compressed.rotateRowsAt(normalizedCompressed, 1, compressed.blockPosition, rotateForward)
	memory.windowEntries = compressed.rotateRowsAt(normalizedWindow, 1, samePosition, rotateForward)
	if compressed.TrainAtCachePrecision {
		memory.compressedEntries = lowprecision.RoundTripMatrix(memory.compressedEntries, compressed.CachePrecision)
		memory.windowEntries = lowprecision.RoundTripMatrix(memory.windowEntries, compressed.CachePrecision)
	}

	memory.usedIndexer = compressed.TopK > 0 && numberOfBlocks > 0
	if memory.usedIndexer {
		memory.indexerQueries = compressed.IndexerQueryLayer.Forward(inputs)
		memory.indexerHeadWeights = compressed.IndexerHeadWeightLayer.Forward(inputs)
		memory.indexerKeys = compressed.IndexerKeyLayer.Forward(normalizedCompressed)
	}

	combined := vectormath.NewMatrix(sequenceLength, compressed.NumberOfHeads*compressed.HeadSize())
	memory.entriesForPosition = make([][]entryReference, sequenceLength)
	memory.chosenBlocks = make([][]int, sequenceLength)
	memory.looked = make([][]compressedLookedAt, sequenceLength)
	for position := 0; position < sequenceLength; position++ {
		visibleBlocks := position / compressed.CompressionRate
		indexerScore := func(block int) float64 {
			return indexerScoreOf(memory.indexerQueries.Row(position), memory.indexerHeadWeights.Row(position), memory.indexerKeys.Row(block))
		}
		chosen := compressed.chooseBlocks(visibleBlocks, indexerScore, compressed.AttendToAllWhileTraining)
		memory.chosenBlocks[position] = chosen

		var references []entryReference
		for _, block := range chosen {
			references = append(references, entryReference{fromBlock: true, index: block})
		}
		windowStart := position - compressed.WindowSize + 1
		if windowStart < 0 {
			windowStart = 0
		}
		for windowPosition := windowStart; windowPosition <= position; windowPosition++ {
			references = append(references, entryReference{fromBlock: false, index: windowPosition})
		}
		memory.entriesForPosition[position] = references
		entries := compressed.entriesFor(references, memory.compressedEntries, memory.windowEntries)

		memory.looked[position] = make([]compressedLookedAt, compressed.NumberOfHeads)
		for head := 0; head < compressed.NumberOfHeads; head++ {
			query := headSlice(memory.queries.Row(position), head, compressed.HeadSize())
			looked, output := attendToEntries(query, entries, compressed.scale(), compressed.SinkLogits, head)
			output = compressed.rotateVectorAt(output, 1, position, rotateBackward)
			copy(headSlice(combined.Row(position), head, compressed.HeadSize()), output)
			memory.looked[position][head] = looked
		}
	}

	if memory.usedIndexer {
		compressed.computeIndexerGradients(&memory)
	}
	memory.forwardHasFinished = true
	compressed.last = memory
	return compressed.OutputLayer.Forward(combined)
}

func (compressed *CompressedAttention) computeIndexerGradients(memory *compressedMemory) {
	memory.indexerGradients = make([][]float64, memory.sequenceLength)
	totalLoss := 0.0
	positionsWithBlocks := 0
	for position := 0; position < memory.sequenceLength; position++ {
		if len(memory.chosenBlocks[position]) > 0 {
			positionsWithBlocks++
		}
	}
	if positionsWithBlocks == 0 {
		compressed.lastIndexerLoss = 0
		return
	}
	for position := 0; position < memory.sequenceLength; position++ {
		chosen := memory.chosenBlocks[position]
		if len(chosen) == 0 {
			continue
		}
		attentionOnBlock := vectormath.NewVector(len(chosen))
		for head := 0; head < compressed.NumberOfHeads; head++ {
			for i := range chosen {
				attentionOnBlock[i] += memory.looked[position][head].weights[i]
			}
		}
		totalAttention := vectormath.Sum(attentionOnBlock)
		if totalAttention <= 0 {
			continue
		}
		indexerScores := vectormath.NewVector(len(chosen))
		for i, block := range chosen {
			indexerScores[i] = indexerScoreOf(memory.indexerQueries.Row(position), memory.indexerHeadWeights.Row(position), memory.indexerKeys.Row(block))
		}
		indexerDistribution := activationfunction.Softmax(indexerScores)

		gradients := make([]float64, len(chosen))
		for i := range chosen {
			target := attentionOnBlock[i] / totalAttention
			if target > 0 {
				totalLoss += target * math.Log(target/math.Max(indexerDistribution[i], 1e-300))
			}
			gradients[i] = compressed.IndexerLossWeight * (indexerDistribution[i] - target) / float64(positionsWithBlocks)
		}
		memory.indexerGradients[position] = gradients
	}
	compressed.lastIndexerLoss = totalLoss / float64(positionsWithBlocks)
}

func (compressed *CompressedAttention) LastIndexerLoss() float64 {
	return compressed.lastIndexerLoss
}

func scoreGradientsWithSink(looked compressedLookedAt, weightGradients []float64, hasSink bool) ([]float64, float64) {
	if !hasSink {
		return activationfunction.SoftmaxBackward(looked.weights, weightGradients), 0
	}
	allWeights := append(vectormath.CopyVector(looked.weights), looked.sinkWeight)
	allWeightGradients := append(vectormath.CopyVector(weightGradients), 0)
	allScoreGradients := activationfunction.SoftmaxBackward(allWeights, allWeightGradients)
	return allScoreGradients[:len(looked.weights)], allScoreGradients[len(looked.weights)]
}

func (compressed *CompressedAttention) Backward(outputGradients vectormath.Matrix) vectormath.Matrix {
	memory := compressed.last
	if !memory.forwardHasFinished {
		panic(fmt.Sprintf("compressed attention %q: call Forward before Backward", compressed.Name))
	}
	headSize := compressed.HeadSize()
	scale := compressed.scale()

	combinedGradients := compressed.OutputLayer.Backward(outputGradients)
	queryGradients := vectormath.NewMatrix(memory.sequenceLength, compressed.NumberOfHeads*headSize)
	compressedEntryGradients := vectormath.NewMatrix(memory.numberOfBlocks, headSize)
	windowEntryGradients := vectormath.NewMatrix(memory.sequenceLength, headSize)

	for position := 0; position < memory.sequenceLength; position++ {
		references := memory.entriesForPosition[position]
		entries := compressed.entriesFor(references, memory.compressedEntries, memory.windowEntries)
		entryGradients := compressed.entriesFor(references, compressedEntryGradients, windowEntryGradients)

		for head := 0; head < compressed.NumberOfHeads; head++ {
			looked := memory.looked[position][head]
			outputGradient := compressed.rotateVectorAt(headSlice(combinedGradients.Row(position), head, headSize), 1, position, rotateForward)

			weightGradients := vectormath.NewVector(len(entries))
			for i, entry := range entries {
				weightGradients[i] = vectormath.DotProduct(outputGradient, entry)
				for j := range entryGradients[i] {
					entryGradients[i][j] += looked.weights[i] * outputGradient[j]
				}
			}

			scoreGradients, sinkGradient := scoreGradientsWithSink(looked, weightGradients, compressed.SinkLogits != nil)
			if compressed.SinkLogits != nil {
				compressed.SinkLogitGradients[head] += sinkGradient
			}

			query := headSlice(memory.queries.Row(position), head, headSize)
			queryGradient := headSlice(queryGradients.Row(position), head, headSize)
			for i, entry := range entries {
				for j := range query {
					queryGradient[j] += scoreGradients[i] * scale * entry[j]
					entryGradients[i][j] += scoreGradients[i] * scale * query[j]
				}
			}
		}
	}

	compressedEntryGradients = compressed.rotateRowsAt(compressedEntryGradients, 1, compressed.blockPosition, rotateBackward)
	windowEntryGradients = compressed.rotateRowsAt(windowEntryGradients, 1, samePosition, rotateBackward)
	normalizedGradients := compressed.EntryNorm.Backward(vectormath.StackRows(compressedEntryGradients, windowEntryGradients))
	compressedRawGradients, rawEntryGradients := vectormath.SplitRows(normalizedGradients, memory.numberOfBlocks)

	rawWeightGradients := vectormath.NewMatrix(memory.sequenceLength, headSize)
	overlapEntryGradients := vectormath.NewMatrix(memory.sequenceLength, headSize)
	overlapWeightGradients := vectormath.NewMatrix(memory.sequenceLength, headSize)
	for block := 0; block < memory.numberOfBlocks; block++ {
		sources := memory.blockSources[block]
		gradients := compressBackward(sources, memory.blockSoftmax[block], compressedRawGradients.Row(block))
		for i, source := range sources {
			entryTarget := rawEntryGradients.Row(source.entryRow)
			weightTarget := rawWeightGradients.Row(source.entryRow)
			biasTarget := compressed.PositionBiasGradients.Row(source.biasRow)
			if source.fromOverlap {
				entryTarget = overlapEntryGradients.Row(source.entryRow)
				weightTarget = overlapWeightGradients.Row(source.entryRow)
				biasTarget = compressed.OverlapPositionBiasGradients.Row(source.biasRow)
			}
			for j := range entryTarget {
				entryTarget[j] += gradients.entries[i][j]
				weightTarget[j] += gradients.logits[i][j]
				biasTarget[j] += gradients.logits[i][j]
			}
		}
	}

	queryGradients = compressed.rotateRowsAt(queryGradients, compressed.NumberOfHeads, samePosition, rotateBackward)
	queryGradients = normalizeEachHeadBackward(compressed.QueryNorm, queryGradients, compressed.NumberOfHeads)
	inputGradients := compressed.QueryLayer.Backward(queryGradients)
	inputGradients = vectormath.AddMatrices(inputGradients, compressed.EntryLayer.Backward(rawEntryGradients))
	inputGradients = vectormath.AddMatrices(inputGradients, compressed.WeightLayer.Backward(rawWeightGradients))
	if compressed.Overlap {
		inputGradients = vectormath.AddMatrices(inputGradients, compressed.OverlapEntryLayer.Backward(overlapEntryGradients))
		inputGradients = vectormath.AddMatrices(inputGradients, compressed.OverlapWeightLayer.Backward(overlapWeightGradients))
	}

	if memory.usedIndexer {
		compressed.indexerBackward(memory)
	}
	return inputGradients
}

func (compressed *CompressedAttention) indexerBackward(memory compressedMemory) {
	indexerHeadSize := compressed.indexerHeadSize()
	queryGradients := vectormath.NewMatrix(memory.sequenceLength, compressed.numberOfIndexerHeads()*indexerHeadSize)
	headWeightGradients := vectormath.NewMatrix(memory.sequenceLength, compressed.numberOfIndexerHeads())
	keyGradients := vectormath.NewMatrix(memory.numberOfBlocks, indexerHeadSize)

	for position, gradients := range memory.indexerGradients {
		for i, block := range memory.chosenBlocks[position] {
			if gradients == nil || gradients[i] == 0 {
				continue
			}
			key := memory.indexerKeys.Row(block)
			for head := 0; head < compressed.numberOfIndexerHeads(); head++ {
				query := headSlice(memory.indexerQueries.Row(position), head, indexerHeadSize)
				dot := vectormath.DotProduct(query, key)
				if dot <= 0 {
					continue
				}
				headWeight := memory.indexerHeadWeights.Get(position, head)
				headWeightGradients.AddTo(position, head, gradients[i]*dot)
				queryGradient := headSlice(queryGradients.Row(position), head, indexerHeadSize)
				keyGradient := keyGradients.Row(block)
				for j := range query {
					queryGradient[j] += gradients[i] * headWeight * key[j]
					keyGradient[j] += gradients[i] * headWeight * query[j]
				}
			}
		}
	}
	compressed.IndexerQueryLayer.Backward(queryGradients)
	compressed.IndexerHeadWeightLayer.Backward(headWeightGradients)
	compressed.IndexerKeyLayer.Backward(keyGradients)
}

func (compressed *CompressedAttention) Parameters() []parameter.Parameter {
	var parameters []parameter.Parameter
	parameters = append(parameters, compressed.QueryLayer.Parameters()...)
	parameters = append(parameters, compressed.QueryNorm.Parameters()...)
	parameters = append(parameters, compressed.EntryLayer.Parameters()...)
	parameters = append(parameters, compressed.WeightLayer.Parameters()...)
	parameters = append(parameters, parameter.Parameter{Name: compressed.Name + ".positionBiases", Values: compressed.PositionBiases.Values, Gradients: compressed.PositionBiasGradients.Values, UseAdamW: true})
	if compressed.Overlap {
		parameters = append(parameters, compressed.OverlapEntryLayer.Parameters()...)
		parameters = append(parameters, compressed.OverlapWeightLayer.Parameters()...)
		parameters = append(parameters, parameter.Parameter{Name: compressed.Name + ".overlapPositionBiases", Values: compressed.OverlapPositionBiases.Values, Gradients: compressed.OverlapPositionBiasGradients.Values, UseAdamW: true})
	}
	parameters = append(parameters, compressed.EntryNorm.Parameters()...)
	if compressed.SinkLogits != nil {
		parameters = append(parameters, parameter.Parameter{Name: compressed.Name + ".sinkLogits", Values: compressed.SinkLogits, Gradients: compressed.SinkLogitGradients, UseAdamW: true})
	}
	if compressed.TopK > 0 {
		parameters = append(parameters, compressed.IndexerQueryLayer.Parameters()...)
		parameters = append(parameters, compressed.IndexerHeadWeightLayer.Parameters()...)
		parameters = append(parameters, compressed.IndexerKeyLayer.Parameters()...)
	}
	parameters = append(parameters, compressed.OutputLayer.Parameters()...)
	return parameters
}
