package tokenizer

import (
	"container/heap"
	"sort"
)

type symbolPair struct {
	left  int
	right int
}

type trainingWord struct {
	symbols []int
	count   int
}

type pairCount struct {
	pair  symbolPair
	count int
}

type pairHeap []pairCount

func (pairs pairHeap) Len() int {
	return len(pairs)
}

func (pairs pairHeap) Less(i int, j int) bool {
	if pairs[i].count != pairs[j].count {
		return pairs[i].count > pairs[j].count
	}
	if pairs[i].pair.left != pairs[j].pair.left {
		return pairs[i].pair.left < pairs[j].pair.left
	}
	return pairs[i].pair.right < pairs[j].pair.right
}

func (pairs pairHeap) Swap(i int, j int) {
	pairs[i], pairs[j] = pairs[j], pairs[i]
}

func (pairs *pairHeap) Push(item any) {
	*pairs = append(*pairs, item.(pairCount))
}

func (pairs *pairHeap) Pop() any {
	old := *pairs
	last := old[len(old)-1]
	*pairs = old[:len(old)-1]
	return last
}

func Train(text string, vocabularySize int) *Tokenizer {
	tokenizer := newEmptyTokenizer([]preTokenizerStep{{splitStyle: GPT2Split}})
	for value := range 256 {
		tokenizer.setToken(value, string(byteToCharacter[value]))
	}

	wordCounts := map[string]int{}
	for _, piece := range preTokenize(text, tokenizer.steps) {
		if piece != "" {
			wordCounts[bytesToByteLevelText(piece)]++
		}
	}
	uniqueWords := make([]string, 0, len(wordCounts))
	for word := range wordCounts {
		uniqueWords = append(uniqueWords, word)
	}
	sort.Strings(uniqueWords)

	words := make([]trainingWord, len(uniqueWords))
	for index, word := range uniqueWords {
		var symbols []int
		for _, character := range word {
			symbols = append(symbols, tokenizer.tokenToID[string(character)])
		}
		words[index] = trainingWord{symbols: symbols, count: wordCounts[word]}
	}

	pairCounts := map[symbolPair]int{}
	wordsWithPair := map[symbolPair]map[int]bool{}
	for index, word := range words {
		for i := 0; i+1 < len(word.symbols); i++ {
			pair := symbolPair{left: word.symbols[i], right: word.symbols[i+1]}
			pairCounts[pair] += word.count
			if wordsWithPair[pair] == nil {
				wordsWithPair[pair] = map[int]bool{}
			}
			wordsWithPair[pair][index] = true
		}
	}
	waiting := &pairHeap{}
	for pair, count := range pairCounts {
		heap.Push(waiting, pairCount{pair: pair, count: count})
	}

	for tokenizer.VocabularySize() < vocabularySize && waiting.Len() > 0 {
		best := heap.Pop(waiting).(pairCount)
		if best.count <= 0 || pairCounts[best.pair] != best.count {
			continue
		}
		leftToken := tokenizer.idToToken[best.pair.left]
		rightToken := tokenizer.idToToken[best.pair.right]
		newID := tokenizer.VocabularySize()
		tokenizer.setToken(newID, leftToken+rightToken)
		tokenizer.addMerge(leftToken, rightToken)

		changedPairs := map[symbolPair]bool{}
		affectedWords := make([]int, 0, len(wordsWithPair[best.pair]))
		for index := range wordsWithPair[best.pair] {
			affectedWords = append(affectedWords, index)
		}
		sort.Ints(affectedWords)
		for _, index := range affectedWords {
			word := &words[index]
			for i := 0; i+1 < len(word.symbols); i++ {
				pair := symbolPair{left: word.symbols[i], right: word.symbols[i+1]}
				pairCounts[pair] -= word.count
				changedPairs[pair] = true
			}
			var merged []int
			for i := 0; i < len(word.symbols); i++ {
				if i+1 < len(word.symbols) && word.symbols[i] == best.pair.left && word.symbols[i+1] == best.pair.right {
					merged = append(merged, newID)
					i++
					continue
				}
				merged = append(merged, word.symbols[i])
			}
			word.symbols = merged
			for i := 0; i+1 < len(word.symbols); i++ {
				pair := symbolPair{left: word.symbols[i], right: word.symbols[i+1]}
				pairCounts[pair] += word.count
				changedPairs[pair] = true
				if wordsWithPair[pair] == nil {
					wordsWithPair[pair] = map[int]bool{}
				}
				wordsWithPair[pair][index] = true
			}
		}
		delete(wordsWithPair, best.pair)
		for pair := range changedPairs {
			if pairCounts[pair] > 0 {
				heap.Push(waiting, pairCount{pair: pair, count: pairCounts[pair]})
			} else {
				delete(pairCounts, pair)
			}
		}
	}
	return tokenizer
}
