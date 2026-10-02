package attention

import (
	"sort"
	"transformer/vectormath"
)

func (attention *SelfAttention) visiblePositions(position int, numberOfPositions int) (int, int) {
	first := 0
	last := numberOfPositions - 1
	if attention.HideFutureTokens {
		last = position
	}
	if attention.WindowSize > 0 {
		if position-attention.WindowSize+1 > first {
			first = position - attention.WindowSize + 1
		}
		if !attention.HideFutureTokens && position+attention.WindowSize-1 < last {
			last = position + attention.WindowSize - 1
		}
	}
	return first, last
}

func (attention *SelfAttention) choosePositions(query vectormath.Vector, keyAt func(int) vectormath.Vector, first int, last int) []int {
	var candidates []int
	for position := first; position <= last; position++ {
		candidates = append(candidates, position)
	}
	if attention.TopK == 0 || len(candidates) <= attention.TopK {
		return candidates
	}

	scores := make(map[int]float64, len(candidates))
	for _, position := range candidates {
		scores[position] = vectormath.DotProduct(query, keyAt(position))
	}
	sort.SliceStable(candidates, func(i int, j int) bool {
		return scores[candidates[i]] > scores[candidates[j]]
	})
	chosen := candidates[:attention.TopK]
	sort.Ints(chosen)
	return chosen
}
