package attention

import (
	"fmt"
	"transformer/lowprecision"
	"transformer/vectormath"
)

type compressedGenerationState struct {
	compressedEntries      *lowprecision.Rows
	indexerKeys            []vectormath.Vector
	windowEntries          *lowprecision.Rows
	windowFirstPosition    int
	tailEntries            []vectormath.Vector
	tailWeights            []vectormath.Vector
	tailOverlapEntries     []vectormath.Vector
	tailOverlapWeights     []vectormath.Vector
	previousOverlapEntries []vectormath.Vector
	previousOverlapWeights []vectormath.Vector
	tokensSeen             int
}

func (compressed *CompressedAttention) StartGenerating() {
	compressed.checkSetUp()
	compressed.generation = &compressedGenerationState{
		compressedEntries: lowprecision.NewRows(compressed.CachePrecision, compressed.HeadSize()),
		windowEntries:     lowprecision.NewRows(compressed.CachePrecision, compressed.HeadSize()),
	}
}

func oneRowOf(vector vectormath.Vector) vectormath.Matrix {
	return vectormath.MatrixFromRows([]vectormath.Vector{vector})
}

func (compressed *CompressedAttention) finishBlock(state *compressedGenerationState) {
	var sources []compressionSource
	for slot := range state.tailEntries {
		sources = append(sources, compressionSource{entry: state.tailEntries[slot], weight: state.tailWeights[slot], bias: compressed.PositionBiases.Row(slot)})
	}
	if compressed.Overlap && state.previousOverlapEntries != nil {
		for slot := range state.previousOverlapEntries {
			sources = append(sources, compressionSource{entry: state.previousOverlapEntries[slot], weight: state.previousOverlapWeights[slot], bias: compressed.OverlapPositionBiases.Row(slot), fromOverlap: true})
		}
	}
	entry, _ := compress(sources, compressed.HeadSize())
	normalized := compressed.EntryNorm.Forward(oneRowOf(entry)).Row(0)
	if compressed.TopK > 0 {
		state.indexerKeys = append(state.indexerKeys, compressed.IndexerKeyLayer.Forward(oneRowOf(normalized)).Row(0))
	}
	block := state.compressedEntries.NumberOfRows()
	state.compressedEntries.Append(compressed.rotateVectorAt(normalized, 1, compressed.blockPosition(block), rotateForward))

	state.previousOverlapEntries = state.tailOverlapEntries
	state.previousOverlapWeights = state.tailOverlapWeights
	state.tailEntries = nil
	state.tailWeights = nil
	state.tailOverlapEntries = nil
	state.tailOverlapWeights = nil
}

func (compressed *CompressedAttention) ForwardOneToken(input vectormath.Vector) vectormath.Vector {
	state := compressed.generation
	if state == nil {
		panic(fmt.Sprintf("compressed attention %q: call StartGenerating before ForwardOneToken", compressed.Name))
	}
	position := state.tokensSeen
	inputRow := oneRowOf(input)
	query := compressed.makeQueries(inputRow, position).Row(0)

	rawEntry := compressed.EntryLayer.Forward(inputRow).Row(0)
	rawWeight := compressed.WeightLayer.Forward(inputRow).Row(0)
	windowEntry := compressed.EntryNorm.Forward(oneRowOf(rawEntry)).Row(0)
	state.windowEntries.Append(compressed.rotateVectorAt(windowEntry, 1, position, rotateForward))
	if state.windowEntries.NumberOfRows() > compressed.WindowSize {
		state.windowEntries.DropOldestRows(1)
		state.windowFirstPosition++
	}

	state.tailEntries = append(state.tailEntries, rawEntry)
	state.tailWeights = append(state.tailWeights, rawWeight)
	if compressed.Overlap {
		state.tailOverlapEntries = append(state.tailOverlapEntries, compressed.OverlapEntryLayer.Forward(inputRow).Row(0))
		state.tailOverlapWeights = append(state.tailOverlapWeights, compressed.OverlapWeightLayer.Forward(inputRow).Row(0))
	}
	if len(state.tailEntries) == compressed.CompressionRate {
		compressed.finishBlock(state)
	}
	state.tokensSeen++

	visibleBlocks := position / compressed.CompressionRate
	var indexerQuery, indexerHeadWeights vectormath.Vector
	if compressed.TopK > 0 && visibleBlocks > compressed.TopK {
		indexerQuery = compressed.IndexerQueryLayer.Forward(inputRow).Row(0)
		indexerHeadWeights = compressed.IndexerHeadWeightLayer.Forward(inputRow).Row(0)
	}
	indexerScore := func(block int) float64 {
		return indexerScoreOf(indexerQuery, indexerHeadWeights, state.indexerKeys[block])
	}
	chosen := compressed.chooseBlocks(visibleBlocks, indexerScore, false)

	var entries []vectormath.Vector
	for _, block := range chosen {
		entries = append(entries, state.compressedEntries.Row(block))
	}
	windowStart := position - compressed.WindowSize + 1
	if windowStart < 0 {
		windowStart = 0
	}
	for windowPosition := windowStart; windowPosition <= position; windowPosition++ {
		entries = append(entries, state.windowEntries.Row(windowPosition-state.windowFirstPosition))
	}

	combined := vectormath.NewVector(compressed.NumberOfHeads * compressed.HeadSize())
	for head := 0; head < compressed.NumberOfHeads; head++ {
		headQuery := headSlice(query, head, compressed.HeadSize())
		_, output := attendToEntries(headQuery, entries, compressed.scale(), compressed.SinkLogits, head, nil)
		copy(headSlice(combined, head, compressed.HeadSize()), compressed.rotateVectorAt(output, 1, position, rotateBackward))
	}
	return compressed.OutputLayer.Forward(oneRowOf(combined)).Row(0)
}

func (compressed *CompressedAttention) CacheBytesUsed() int {
	state := compressed.generation
	if state == nil {
		return 0
	}
	total := state.compressedEntries.BytesUsed() + state.windowEntries.BytesUsed()
	total += len(state.indexerKeys) * compressed.indexerHeadSizeOrZero() * 8
	tailRows := len(state.tailEntries) + len(state.tailWeights) + len(state.tailOverlapEntries) + len(state.tailOverlapWeights) + len(state.previousOverlapEntries) + len(state.previousOverlapWeights)
	total += tailRows * compressed.HeadSize() * 8
	return total
}

func (compressed *CompressedAttention) indexerHeadSizeOrZero() int {
	if compressed.IndexerKeyLayer == nil {
		return 0
	}
	return compressed.indexerHeadSize()
}
