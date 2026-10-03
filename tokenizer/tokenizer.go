package tokenizer

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

type mergePair struct {
	left  string
	right string
}

type Tokenizer struct {
	tokenToID                   map[string]int
	idToToken                   []string
	mergeRanks                  map[mergePair]int
	merges                      []mergePair
	specialTokenToID            map[string]int
	specialTokensByLen          []string
	steps                       []preTokenizerStep
	addPrefixSpace              bool
	useWholeWordsFromVocabulary bool
	wordCache                   map[string][]int
}

func newEmptyTokenizer(steps []preTokenizerStep) *Tokenizer {
	return &Tokenizer{
		tokenToID:        map[string]int{},
		mergeRanks:       map[mergePair]int{},
		specialTokenToID: map[string]int{},
		steps:            steps,
		wordCache:        map[string][]int{},
	}
}

func (tokenizer *Tokenizer) setToken(id int, token string) {
	for len(tokenizer.idToToken) <= id {
		tokenizer.idToToken = append(tokenizer.idToToken, "")
	}
	tokenizer.idToToken[id] = token
	tokenizer.tokenToID[token] = id
}

func (tokenizer *Tokenizer) addMerge(left string, right string) {
	pair := mergePair{left: left, right: right}
	if _, alreadyThere := tokenizer.mergeRanks[pair]; alreadyThere {
		return
	}
	tokenizer.mergeRanks[pair] = len(tokenizer.merges)
	tokenizer.merges = append(tokenizer.merges, pair)
}

func (tokenizer *Tokenizer) addSpecialTokenWithID(id int, content string) {
	tokenizer.setToken(id, content)
	if _, found := tokenizer.specialTokenToID[content]; !found {
		tokenizer.specialTokensByLen = append(tokenizer.specialTokensByLen, content)
	}
	tokenizer.specialTokenToID[content] = id
	sort.SliceStable(tokenizer.specialTokensByLen, func(i int, j int) bool {
		return len(tokenizer.specialTokensByLen[i]) > len(tokenizer.specialTokensByLen[j])
	})
	tokenizer.wordCache = map[string][]int{}
}

func (tokenizer *Tokenizer) AddSpecialToken(content string) int {
	if id, found := tokenizer.specialTokenToID[content]; found {
		return id
	}
	id, found := tokenizer.tokenToID[content]
	if !found {
		id = len(tokenizer.idToToken)
		tokenizer.setToken(id, content)
	}
	tokenizer.specialTokenToID[content] = id
	tokenizer.specialTokensByLen = append(tokenizer.specialTokensByLen, content)
	sort.SliceStable(tokenizer.specialTokensByLen, func(i int, j int) bool {
		return len(tokenizer.specialTokensByLen[i]) > len(tokenizer.specialTokensByLen[j])
	})
	tokenizer.wordCache = map[string][]int{}
	return id
}

func (tokenizer *Tokenizer) VocabularySize() int {
	return len(tokenizer.idToToken)
}

func (tokenizer *Tokenizer) SpecialTokenID(content string) (int, bool) {
	id, found := tokenizer.specialTokenToID[content]
	return id, found
}

func (tokenizer *Tokenizer) TokenID(byteLevelToken string) (int, bool) {
	id, found := tokenizer.tokenToID[byteLevelToken]
	return id, found
}

func (tokenizer *Tokenizer) applyMerges(byteLevelWord string) []string {
	var symbols []string
	for _, character := range byteLevelWord {
		symbols = append(symbols, string(character))
	}
	for len(symbols) > 1 {
		bestRank := -1
		var bestPair mergePair
		for i := 0; i+1 < len(symbols); i++ {
			pair := mergePair{left: symbols[i], right: symbols[i+1]}
			rank, found := tokenizer.mergeRanks[pair]
			if found && (bestRank == -1 || rank < bestRank) {
				bestRank = rank
				bestPair = pair
			}
		}
		if bestRank == -1 {
			break
		}
		var merged []string
		for i := 0; i < len(symbols); i++ {
			if i+1 < len(symbols) && symbols[i] == bestPair.left && symbols[i+1] == bestPair.right {
				merged = append(merged, bestPair.left+bestPair.right)
				i++
				continue
			}
			merged = append(merged, symbols[i])
		}
		symbols = merged
	}
	return symbols
}

func (tokenizer *Tokenizer) encodeWord(byteLevelWord string) []int {
	if ids, found := tokenizer.wordCache[byteLevelWord]; found {
		return ids
	}
	var ids []int
	if tokenizer.useWholeWordsFromVocabulary {
		if id, found := tokenizer.tokenToID[byteLevelWord]; found {
			tokenizer.wordCache[byteLevelWord] = []int{id}
			return []int{id}
		}
	}
	for _, symbol := range tokenizer.applyMerges(byteLevelWord) {
		if id, found := tokenizer.tokenToID[symbol]; found {
			ids = append(ids, id)
			continue
		}
		for _, character := range symbol {
			if id, found := tokenizer.tokenToID[string(character)]; found {
				ids = append(ids, id)
			}
		}
	}
	if len(tokenizer.wordCache) > 100000 {
		tokenizer.wordCache = map[string][]int{}
	}
	tokenizer.wordCache[byteLevelWord] = ids
	return ids
}

func (tokenizer *Tokenizer) encodePlainText(text string) []int {
	if text == "" {
		return nil
	}
	var ids []int
	for _, piece := range preTokenize(text, tokenizer.steps) {
		if piece == "" {
			continue
		}
		ids = append(ids, tokenizer.encodeWord(bytesToByteLevelText(piece))...)
	}
	return ids
}

func (tokenizer *Tokenizer) nextSpecialToken(text string) (int, string) {
	bestStart := -1
	bestToken := ""
	for _, special := range tokenizer.specialTokensByLen {
		start := strings.Index(text, special)
		if start >= 0 && (bestStart == -1 || start < bestStart) {
			bestStart = start
			bestToken = special
		}
	}
	return bestStart, bestToken
}

func (tokenizer *Tokenizer) hasEveryByte() bool {
	for value := range 256 {
		if _, found := tokenizer.tokenToID[string(byteToCharacter[value])]; !found {
			return false
		}
	}
	return true
}

func (tokenizer *Tokenizer) Encode(text string) []int {
	if !utf8.ValidString(text) && !tokenizer.hasEveryByte() {
		text = strings.ToValidUTF8(text, "\uFFFD")
	}
	if tokenizer.addPrefixSpace && text != "" && !strings.HasPrefix(text, " ") {
		text = " " + text
	}
	var ids []int
	for text != "" {
		start, special := tokenizer.nextSpecialToken(text)
		if start == -1 {
			ids = append(ids, tokenizer.encodePlainText(text)...)
			break
		}
		ids = append(ids, tokenizer.encodePlainText(text[:start])...)
		ids = append(ids, tokenizer.specialTokenToID[special])
		text = text[start+len(special):]
	}
	return ids
}

func (tokenizer *Tokenizer) isSpecial(id int) bool {
	if id < 0 || id >= len(tokenizer.idToToken) {
		return false
	}
	_, found := tokenizer.specialTokenToID[tokenizer.idToToken[id]]
	return found
}

func (tokenizer *Tokenizer) Decode(ids []int) string {
	var result []byte
	for _, id := range ids {
		if id < 0 || id >= len(tokenizer.idToToken) {
			panic(fmt.Sprintf("Decode: token ID %d is outside the vocabulary of %d tokens", id, len(tokenizer.idToToken)))
		}
		token := tokenizer.idToToken[id]
		if tokenizer.isSpecial(id) {
			result = append(result, token...)
			continue
		}
		result = append(result, byteLevelTextToBytes(token)...)
	}
	return string(result)
}

func (tokenizer *Tokenizer) TokenText(id int) string {
	return tokenizer.Decode([]int{id})
}
