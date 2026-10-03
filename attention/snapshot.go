package attention

import (
	"fmt"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/vectormath"
)

type GenerationSnapshot struct {
	Keys          *lowprecision.RowsSnapshot
	Values        *lowprecision.RowsSnapshot
	FirstPosition int
	TokensSeen    int
}

func (attention *SelfAttention) GenerationSnapshot() GenerationSnapshot {
	state := attention.generation
	if state == nil {
		panic(fmt.Sprintf("attention %q: nothing to save, call StartGenerating first", attention.Name))
	}
	snapshot := GenerationSnapshot{FirstPosition: state.firstPosition, TokensSeen: state.tokensSeen}
	if state.keys != nil {
		keys := state.keys.Snapshot()
		snapshot.Keys = &keys
	}
	if state.values != nil {
		values := state.values.Snapshot()
		snapshot.Values = &values
	}
	return snapshot
}

func (attention *SelfAttention) RestoreGeneration(snapshot GenerationSnapshot) error {
	attention.StartGenerating()
	state := attention.generation
	ownsCache := attention.SharingMode == OwnKeysAndValues
	if ownsCache != (snapshot.Keys != nil) || (ownsCache && !attention.ShareKeyAsValue) != (snapshot.Values != nil) {
		return fmt.Errorf("attention %q: saved cache doesn't match how this attention stores keys and values", attention.Name)
	}
	if snapshot.Keys != nil {
		keys, err := lowprecision.RowsFromSnapshot(*snapshot.Keys)
		if err != nil {
			return fmt.Errorf("attention %q keys: %w", attention.Name, err)
		}
		if keys.Width != attention.keyValueSize() || keys.Precision != attention.CachePrecision {
			return fmt.Errorf("attention %q: saved keys are %d wide at %v but this attention uses %d at %v", attention.Name, keys.Width, keys.Precision, attention.keyValueSize(), attention.CachePrecision)
		}
		state.keys = keys
	}
	if snapshot.Values != nil {
		values, err := lowprecision.RowsFromSnapshot(*snapshot.Values)
		if err != nil {
			return fmt.Errorf("attention %q values: %w", attention.Name, err)
		}
		state.values = values
	}
	state.firstPosition = snapshot.FirstPosition
	state.tokensSeen = snapshot.TokensSeen
	return nil
}

type CompressedGenerationSnapshot struct {
	CompressedEntries      lowprecision.RowsSnapshot
	WindowEntries          lowprecision.RowsSnapshot
	IndexerKeys            [][]float64
	WindowFirstPosition    int
	TailEntries            [][]float64
	TailWeights            [][]float64
	TailOverlapEntries     [][]float64
	TailOverlapWeights     [][]float64
	PreviousOverlapEntries [][]float64
	PreviousOverlapWeights [][]float64
	TokensSeen             int
}

func vectorsToLists(vectors []vectormath.Vector) [][]float64 {
	lists := make([][]float64, len(vectors))
	for i, vector := range vectors {
		lists[i] = vectormath.CopyVector(vector)
	}
	return lists
}

func listsToVectors(lists [][]float64) []vectormath.Vector {
	if lists == nil {
		return nil
	}
	vectors := make([]vectormath.Vector, len(lists))
	for i, list := range lists {
		vectors[i] = vectormath.CopyVector(list)
	}
	return vectors
}

func (compressed *CompressedAttention) GenerationSnapshot() CompressedGenerationSnapshot {
	state := compressed.generation
	if state == nil {
		panic(fmt.Sprintf("compressed attention %q: nothing to save, call StartGenerating first", compressed.Name))
	}
	return CompressedGenerationSnapshot{
		CompressedEntries:      state.compressedEntries.Snapshot(),
		WindowEntries:          state.windowEntries.Snapshot(),
		IndexerKeys:            vectorsToLists(state.indexerKeys),
		WindowFirstPosition:    state.windowFirstPosition,
		TailEntries:            vectorsToLists(state.tailEntries),
		TailWeights:            vectorsToLists(state.tailWeights),
		TailOverlapEntries:     vectorsToLists(state.tailOverlapEntries),
		TailOverlapWeights:     vectorsToLists(state.tailOverlapWeights),
		PreviousOverlapEntries: vectorsToLists(state.previousOverlapEntries),
		PreviousOverlapWeights: vectorsToLists(state.previousOverlapWeights),
		TokensSeen:             state.tokensSeen,
	}
}

func (compressed *CompressedAttention) RestoreGeneration(snapshot CompressedGenerationSnapshot) error {
	compressed.StartGenerating()
	state := compressed.generation
	compressedEntries, err := lowprecision.RowsFromSnapshot(snapshot.CompressedEntries)
	if err != nil {
		return fmt.Errorf("compressed attention %q entries: %w", compressed.Name, err)
	}
	windowEntries, err := lowprecision.RowsFromSnapshot(snapshot.WindowEntries)
	if err != nil {
		return fmt.Errorf("compressed attention %q window: %w", compressed.Name, err)
	}
	if compressedEntries.Width != compressed.HeadSize() || windowEntries.Width != compressed.HeadSize() {
		return fmt.Errorf("compressed attention %q: saved entries don't match its head size %d", compressed.Name, compressed.HeadSize())
	}
	state.compressedEntries = compressedEntries
	state.windowEntries = windowEntries
	state.indexerKeys = listsToVectors(snapshot.IndexerKeys)
	state.windowFirstPosition = snapshot.WindowFirstPosition
	state.tailEntries = listsToVectors(snapshot.TailEntries)
	state.tailWeights = listsToVectors(snapshot.TailWeights)
	state.tailOverlapEntries = listsToVectors(snapshot.TailOverlapEntries)
	state.tailOverlapWeights = listsToVectors(snapshot.TailOverlapWeights)
	state.previousOverlapEntries = listsToVectors(snapshot.PreviousOverlapEntries)
	state.previousOverlapWeights = listsToVectors(snapshot.PreviousOverlapWeights)
	state.tokensSeen = snapshot.TokensSeen
	return nil
}
