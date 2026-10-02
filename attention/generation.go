package attention

import (
	"fmt"
	"transformer/lowprecision"
	"transformer/vectormath"
)

type generationState struct {
	keys          *lowprecision.Rows
	values        *lowprecision.Rows
	firstPosition int
	tokensSeen    int
	lookedAt      []lookedAt
}

func (attention *SelfAttention) StartGenerating() {
	attention.checkSetUp()
	if !attention.HideFutureTokens {
		panic(fmt.Sprintf("attention %q: generating one token at a time needs HideFutureTokens to be true", attention.Name))
	}
	state := &generationState{lookedAt: make([]lookedAt, attention.NumberOfHeads)}
	if attention.SharingMode == OwnKeysAndValues {
		state.keys = lowprecision.NewRows(attention.CachePrecision, attention.VectorSize())
		state.values = lowprecision.NewRows(attention.CachePrecision, attention.VectorSize())
	}
	attention.generation = state
}

func (attention *SelfAttention) rowsToKeepInCache() int {
	largestWindow := attention.WindowSize
	if largestWindow == 0 {
		return 0
	}
	for _, borrower := range attention.borrowers {
		if borrower.WindowSize == 0 {
			return 0
		}
		if borrower.WindowSize > largestWindow {
			largestWindow = borrower.WindowSize
		}
	}
	return largestWindow
}

func (attention *SelfAttention) ForwardOneToken(input vectormath.Vector) vectormath.Vector {
	state := attention.generation
	if state == nil {
		panic(fmt.Sprintf("attention %q: call StartGenerating before ForwardOneToken", attention.Name))
	}
	position := state.tokensSeen
	inputRow := vectormath.MatrixFromRows([]vectormath.Vector{input})
	query := attention.rotateOne(attention.QueryLayer.Forward(inputRow).Row(0), position)

	if attention.SharingMode == OwnKeysAndValues {
		state.keys.Append(attention.rotateOne(attention.KeyLayer.Forward(inputRow).Row(0), position))
		state.values.Append(attention.ValueLayer.Forward(inputRow).Row(0))
		state.tokensSeen++
		rowsToKeep := attention.rowsToKeepInCache()
		if rowsToKeep > 0 && state.keys.NumberOfRows() > rowsToKeep {
			extraRows := state.keys.NumberOfRows() - rowsToKeep
			state.keys.DropOldestRows(extraRows)
			state.values.DropOldestRows(extraRows)
			state.firstPosition += extraRows
		}
	}

	owner := attention.keyValueOwner()
	ownerState := owner.generation
	if ownerState == nil || ownerState.tokensSeen != position+1 {
		panic(fmt.Sprintf("attention %q: borrows keys and values from %q, call StartGenerating on both and run %q ForwardOneToken first for each token", attention.Name, owner.Name, owner.Name))
	}

	var choicesSource *SelfAttention
	if attention.SharingMode == BorrowKeysValuesAndChoices {
		choicesSource = attention.choicesSource()
		if choicesSource.generation == nil || choicesSource.generation.tokensSeen != position+1 {
			panic(fmt.Sprintf("attention %q: borrows choices from %q, run %q ForwardOneToken first for each token", attention.Name, choicesSource.Name, choicesSource.Name))
		}
	}

	cachedKeys := make([]vectormath.Vector, ownerState.keys.NumberOfRows())
	cachedValues := make([]vectormath.Vector, ownerState.values.NumberOfRows())
	for row := range cachedKeys {
		cachedKeys[row] = ownerState.keys.Row(row)
		cachedValues[row] = ownerState.values.Row(row)
	}

	combined := vectormath.NewVector(attention.VectorSize())
	for head := 0; head < attention.NumberOfHeads; head++ {
		keyAt := func(otherPosition int) vectormath.Vector {
			return headSlice(cachedKeys[otherPosition-ownerState.firstPosition], head, attention.HeadSize())
		}
		valueAt := func(otherPosition int) vectormath.Vector {
			return headSlice(cachedValues[otherPosition-ownerState.firstPosition], head, attention.HeadSize())
		}
		headQuery := headSlice(query, head, attention.HeadSize())

		var positions []int
		if choicesSource != nil {
			positions = choicesSource.generation.lookedAt[head].positions
		} else {
			first, last := attention.visiblePositions(position, position+1)
			if first < ownerState.firstPosition {
				panic(fmt.Sprintf("attention %q: needs position %d but %q only kept positions from %d in its cache", attention.Name, first, owner.Name, ownerState.firstPosition))
			}
			positions = attention.choosePositions(headQuery, keyAt, first, last)
		}

		weights, output := attendTo(headQuery, positions, keyAt, valueAt, attention.scale())
		copy(headSlice(combined, head, attention.HeadSize()), output)
		state.lookedAt[head] = lookedAt{positions: positions, weights: weights}
	}

	if attention.SharingMode != OwnKeysAndValues {
		state.tokensSeen++
	}
	return attention.OutputLayer.Forward(vectormath.MatrixFromRows([]vectormath.Vector{combined})).Row(0)
}

func (attention *SelfAttention) CacheBytesUsed() int {
	if attention.generation == nil || attention.SharingMode != OwnKeysAndValues {
		return 0
	}
	return attention.generation.keys.BytesUsed() + attention.generation.values.BytesUsed()
}
