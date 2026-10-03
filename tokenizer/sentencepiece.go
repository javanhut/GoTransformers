package tokenizer

import (
	"container/heap"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

type Kind string

const (
	ByteLevelBPE         Kind = "byte-level-bpe"
	SentencePieceBPE     Kind = "sentencepiece-bpe"
	SentencePieceUnigram Kind = "sentencepiece-unigram"
)

type pieceType int

const (
	normalPiece      pieceType = 1
	unknownPiece     pieceType = 2
	controlPiece     pieceType = 3
	userDefinedPiece pieceType = 4
	unusedPiece      pieceType = 5
	bytePiece        pieceType = 6
)

type mergeOrder string

const (
	mergeHighestScoreFirst   mergeOrder = "score"
	mergeEarliestListedFirst mergeOrder = "list"
)

const spaceSymbol = "▁"

const unigramUnknownPenalty = 10.0

type sentencePieceModel struct {
	mergeOrder                             mergeOrder
	addDummyPrefix                         bool
	dummyPrefixOnlyAtStartOfText           bool
	skipDummyPrefixWhenTextStartsWithSpace bool
	replaceSpacesWithSpaceSymbol           bool
	splitBeforeEachSpaceSymbol             bool
	removeExtraWhitespace                  bool
	byteFallback                           bool
	unknownID                              int
	scores                                 []float32
	types                                  []pieceType
	byteTokenIDs                           [256]int
	userDefinedPiecesLongestFirst          []string
	longestPieceBytes                      int
	lowestNormalScore                      float32
	highestNormalScore                     float32
}

type sentencePieceEntry struct {
	piece string
	score float32
	kind  pieceType
}

func newSentencePieceTokenizer(kind Kind, entries []sentencePieceEntry, settings sentencePieceModel) *Tokenizer {
	tokenizer := newEmptyTokenizer(nil)
	tokenizer.kind = kind
	tokenizer.sentencePiece = settings
	tokenizer.sentencePiece.scores = make([]float32, len(entries))
	tokenizer.sentencePiece.types = make([]pieceType, len(entries))
	tokenizer.idToToken = make([]string, len(entries))
	for id, entry := range entries {
		tokenizer.sentencePiece.scores[id] = entry.score
		tokenizer.sentencePiece.types[id] = entry.kind
		if entry.piece != "" {
			tokenizer.setToken(id, entry.piece)
		}
	}
	for id, entry := range entries {
		isControlOrUnknown := entry.kind == controlPiece || entry.kind == unknownPiece
		if isControlOrUnknown && entry.piece != "" {
			tokenizer.addSpecialTokenWithID(id, entry.piece)
		}
	}
	tokenizer.prepareSentencePiece()
	return tokenizer
}

func byteValueOfPieceText(text string) (byte, bool) {
	hexDigits, found := strings.CutPrefix(text, "<0x")
	if !found || len(text) != 6 {
		return 0, false
	}
	hexDigits, found = strings.CutSuffix(hexDigits, ">")
	if !found {
		return 0, false
	}
	value, err := strconv.ParseUint(hexDigits, 16, 8)
	if err != nil {
		return 0, false
	}
	return byte(value), true
}

func (tokenizer *Tokenizer) prepareSentencePiece() {
	model := &tokenizer.sentencePiece
	for value := range 256 {
		model.byteTokenIDs[value] = -1
	}
	model.unknownID = -1
	model.userDefinedPiecesLongestFirst = nil
	model.longestPieceBytes = 0
	sawNormalPiece := false
	for id, kind := range model.types {
		piece := tokenizer.idToToken[id]
		switch kind {
		case bytePiece:
			if value, isByte := byteValueOfPieceText(piece); isByte {
				model.byteTokenIDs[value] = id
			}
		case unknownPiece:
			if model.unknownID == -1 {
				model.unknownID = id
			}
		case userDefinedPiece:
			model.userDefinedPiecesLongestFirst = append(model.userDefinedPiecesLongestFirst, piece)
		case normalPiece:
			score := model.scores[id]
			if !sawNormalPiece || score < model.lowestNormalScore {
				model.lowestNormalScore = score
			}
			if !sawNormalPiece || score > model.highestNormalScore {
				model.highestNormalScore = score
			}
			sawNormalPiece = true
		}
		if len(piece) > model.longestPieceBytes {
			model.longestPieceBytes = len(piece)
		}
	}
	sort.SliceStable(model.userDefinedPiecesLongestFirst, func(i int, j int) bool {
		return len(model.userDefinedPiecesLongestFirst[i]) > len(model.userDefinedPiecesLongestFirst[j])
	})
}

func (tokenizer *Tokenizer) pieceTypeOf(id int) pieceType {
	if id < 0 || id >= len(tokenizer.sentencePiece.types) {
		return controlPiece
	}
	return tokenizer.sentencePiece.types[id]
}

func (tokenizer *Tokenizer) pieceIsPlainText(id int) bool {
	kind := tokenizer.pieceTypeOf(id)
	return kind == normalPiece || kind == userDefinedPiece || kind == unusedPiece
}

func removeExtraSpaces(text string) string {
	var result strings.Builder
	previousWasSpace := true
	for _, character := range text {
		if character == ' ' {
			if !previousWasSpace {
				result.WriteRune(' ')
			}
			previousWasSpace = true
			continue
		}
		result.WriteRune(character)
		previousWasSpace = false
	}
	return strings.TrimSuffix(result.String(), " ")
}

func (tokenizer *Tokenizer) normalizeForSentencePiece(text string, isStartOfText bool) string {
	model := &tokenizer.sentencePiece
	if model.removeExtraWhitespace {
		text = removeExtraSpaces(text)
	}
	if text == "" {
		return ""
	}
	space := " "
	if model.replaceSpacesWithSpaceSymbol {
		text = strings.ReplaceAll(text, " ", spaceSymbol)
		space = spaceSymbol
	}
	wantsPrefix := model.addDummyPrefix && (isStartOfText || !model.dummyPrefixOnlyAtStartOfText)
	if wantsPrefix && model.skipDummyPrefixWhenTextStartsWithSpace && strings.HasPrefix(text, space) {
		wantsPrefix = false
	}
	if wantsPrefix {
		text = space + text
	}
	return text
}

func splitBeforeSpaceSymbols(text string) []string {
	var words []string
	wordStart := 0
	for offset := 0; offset < len(text); {
		if offset > wordStart && strings.HasPrefix(text[offset:], spaceSymbol) {
			words = append(words, text[wordStart:offset])
			wordStart = offset
		}
		_, size := utf8.DecodeRuneInString(text[offset:])
		offset += size
	}
	if wordStart < len(text) {
		words = append(words, text[wordStart:])
	}
	return words
}

func (tokenizer *Tokenizer) encodeSentencePiece(text string) []int {
	text = strings.ToValidUTF8(text, "�")
	var ids []int
	isStartOfText := true
	for text != "" {
		start, special := tokenizer.nextSpecialToken(text)
		plainText := text
		if start != -1 {
			plainText = text[:start]
		}
		if plainText != "" {
			ids = append(ids, tokenizer.encodeSentencePieceChunk(plainText, isStartOfText)...)
		}
		if start == -1 {
			break
		}
		ids = append(ids, tokenizer.specialTokenToID[special])
		text = text[start+len(special):]
		isStartOfText = false
	}
	return ids
}

func (tokenizer *Tokenizer) encodeSentencePieceChunk(text string, isStartOfText bool) []int {
	normalized := tokenizer.normalizeForSentencePiece(text, isStartOfText)
	if normalized == "" {
		return nil
	}
	words := []string{normalized}
	if tokenizer.sentencePiece.splitBeforeEachSpaceSymbol {
		words = splitBeforeSpaceSymbols(normalized)
	}
	var ids []int
	for _, word := range words {
		if tokenizer.kind == SentencePieceUnigram {
			ids = append(ids, tokenizer.encodeUnigramWord(word)...)
			continue
		}
		for _, piece := range tokenizer.applySentencePieceMerges(word) {
			ids = append(ids, tokenizer.idsForPiece(piece)...)
		}
	}
	return ids
}

func (tokenizer *Tokenizer) idsForPiece(piece string) []int {
	if id, found := tokenizer.tokenToID[piece]; found && tokenizer.pieceIsPlainText(id) {
		return []int{id}
	}
	return tokenizer.idsForUnknownText(piece)
}

func (tokenizer *Tokenizer) idsForUnknownText(text string) []int {
	model := &tokenizer.sentencePiece
	if !model.byteFallback {
		if model.unknownID < 0 {
			return nil
		}
		return []int{model.unknownID}
	}
	ids := make([]int, 0, len(text))
	for i := 0; i < len(text); i++ {
		id := model.byteTokenIDs[text[i]]
		if id < 0 {
			id = model.unknownID
		}
		if id >= 0 {
			ids = append(ids, id)
		}
	}
	return ids
}

type bpeSymbol struct {
	text     string
	previous int
	next     int
	frozen   bool
}

type bpeCandidate struct {
	left         int
	right        int
	priority     float64
	mergedLength int
}

type bpeCandidateQueue []bpeCandidate

func (queue bpeCandidateQueue) Len() int {
	return len(queue)
}

func (queue bpeCandidateQueue) Less(i int, j int) bool {
	if queue[i].priority != queue[j].priority {
		return queue[i].priority > queue[j].priority
	}
	return queue[i].left < queue[j].left
}

func (queue bpeCandidateQueue) Swap(i int, j int) {
	queue[i], queue[j] = queue[j], queue[i]
}

func (queue *bpeCandidateQueue) Push(item any) {
	*queue = append(*queue, item.(bpeCandidate))
}

func (queue *bpeCandidateQueue) Pop() any {
	old := *queue
	last := old[len(old)-1]
	*queue = old[:len(old)-1]
	return last
}

func (tokenizer *Tokenizer) initialSymbolLength(rest string) (int, bool) {
	for _, userDefined := range tokenizer.sentencePiece.userDefinedPiecesLongestFirst {
		if strings.HasPrefix(rest, userDefined) {
			return len(userDefined), true
		}
	}
	_, size := utf8.DecodeRuneInString(rest)
	return size, false
}

func (tokenizer *Tokenizer) splitIntoInitialSymbols(word string) []bpeSymbol {
	var symbols []bpeSymbol
	for offset := 0; offset < len(word); {
		length, frozen := tokenizer.initialSymbolLength(word[offset:])
		symbols = append(symbols, bpeSymbol{
			text:     word[offset : offset+length],
			previous: len(symbols) - 1,
			next:     len(symbols) + 1,
			frozen:   frozen,
		})
		offset += length
	}
	if len(symbols) > 0 {
		symbols[len(symbols)-1].next = -1
	}
	return symbols
}

func (tokenizer *Tokenizer) mergePriority(left string, right string) (float64, bool) {
	if tokenizer.sentencePiece.mergeOrder == mergeEarliestListedFirst {
		rank, found := tokenizer.mergeRanks[mergePair{left: left, right: right}]
		return -float64(rank), found
	}
	id, found := tokenizer.tokenToID[left+right]
	if !found || !tokenizer.pieceIsPlainText(id) {
		return 0, false
	}
	return float64(tokenizer.sentencePiece.scores[id]), true
}

func (tokenizer *Tokenizer) addMergeCandidate(symbols []bpeSymbol, queue *bpeCandidateQueue, left int, right int) {
	if left == -1 || right == -1 || symbols[left].frozen || symbols[right].frozen {
		return
	}
	priority, found := tokenizer.mergePriority(symbols[left].text, symbols[right].text)
	if !found {
		return
	}
	heap.Push(queue, bpeCandidate{
		left:         left,
		right:        right,
		priority:     priority,
		mergedLength: len(symbols[left].text) + len(symbols[right].text),
	})
}

func (tokenizer *Tokenizer) applySentencePieceMerges(word string) []string {
	symbols := tokenizer.splitIntoInitialSymbols(word)
	if len(symbols) == 0 {
		return nil
	}
	queue := &bpeCandidateQueue{}
	for i := 1; i < len(symbols); i++ {
		tokenizer.addMergeCandidate(symbols, queue, i-1, i)
	}
	partsOfUnusedPieces := map[string]mergePair{}
	for queue.Len() > 0 {
		candidate := heap.Pop(queue).(bpeCandidate)
		left := &symbols[candidate.left]
		right := &symbols[candidate.right]
		if left.text == "" || right.text == "" || len(left.text)+len(right.text) != candidate.mergedLength {
			continue
		}
		merged := left.text + right.text
		if id, found := tokenizer.tokenToID[merged]; found && tokenizer.pieceTypeOf(id) == unusedPiece {
			partsOfUnusedPieces[merged] = mergePair{left: left.text, right: right.text}
		}
		left.text = merged
		left.next = right.next
		if right.next >= 0 {
			symbols[right.next].previous = candidate.left
		}
		right.text = ""
		tokenizer.addMergeCandidate(symbols, queue, left.previous, candidate.left)
		tokenizer.addMergeCandidate(symbols, queue, candidate.left, left.next)
	}
	var pieces []string
	for index := 0; index != -1; index = symbols[index].next {
		pieces = append(pieces, tokenizer.splitUnusedPiece(symbols[index].text, partsOfUnusedPieces)...)
	}
	return pieces
}

func (tokenizer *Tokenizer) splitUnusedPiece(piece string, partsOfUnusedPieces map[string]mergePair) []string {
	id, found := tokenizer.tokenToID[piece]
	parts, wasMerged := partsOfUnusedPieces[piece]
	if !found || tokenizer.pieceTypeOf(id) != unusedPiece || !wasMerged {
		return []string{piece}
	}
	pieces := tokenizer.splitUnusedPiece(parts.left, partsOfUnusedPieces)
	return append(pieces, tokenizer.splitUnusedPiece(parts.right, partsOfUnusedPieces)...)
}

type unigramPathStep struct {
	reached   bool
	tokenID   int
	start     int
	bestScore float64
}

func (tokenizer *Tokenizer) unigramPieceScore(id int, lengthInBytes int) float64 {
	if tokenizer.pieceTypeOf(id) == userDefinedPiece {
		return float64(lengthInBytes)*float64(tokenizer.sentencePiece.highestNormalScore) - 0.1
	}
	return float64(tokenizer.sentencePiece.scores[id])
}

func offerUnigramStep(steps []unigramPathStep, end int, tokenID int, start int, score float64) {
	if !steps[end].reached || score > steps[end].bestScore {
		steps[end] = unigramPathStep{reached: true, tokenID: tokenID, start: start, bestScore: score}
	}
}

func (tokenizer *Tokenizer) bestUnigramPath(word string) []unigramPathStep {
	model := &tokenizer.sentencePiece
	unknownScore := float64(model.lowestNormalScore) - unigramUnknownPenalty
	steps := make([]unigramPathStep, len(word)+1)
	steps[0].reached = true
	for start := 0; start < len(word); {
		_, characterLength := utf8.DecodeRuneInString(word[start:])
		scoreSoFar := steps[start].bestScore
		foundSingleCharacterPiece := false
		for end := start + 1; end <= len(word) && end-start <= model.longestPieceBytes; end++ {
			id, found := tokenizer.tokenToID[word[start:end]]
			kind := tokenizer.pieceTypeOf(id)
			if !found || (kind != normalPiece && kind != userDefinedPiece) {
				continue
			}
			offerUnigramStep(steps, end, id, start, scoreSoFar+tokenizer.unigramPieceScore(id, end-start))
			if end-start == characterLength {
				foundSingleCharacterPiece = true
			}
		}
		if !foundSingleCharacterPiece {
			offerUnigramStep(steps, start+characterLength, -1, start, scoreSoFar+unknownScore)
		}
		start += characterLength
	}
	return steps
}

type unigramSegment struct {
	start   int
	end     int
	tokenID int
}

func (tokenizer *Tokenizer) encodeUnigramWord(word string) []int {
	steps := tokenizer.bestUnigramPath(word)
	var segmentsBackwards []unigramSegment
	for end := len(word); end > 0; end = steps[end].start {
		step := steps[end]
		last := len(segmentsBackwards) - 1
		if step.tokenID == -1 && last >= 0 && segmentsBackwards[last].tokenID == -1 {
			segmentsBackwards[last].start = step.start
			continue
		}
		segmentsBackwards = append(segmentsBackwards, unigramSegment{start: step.start, end: end, tokenID: step.tokenID})
	}
	var ids []int
	for index := len(segmentsBackwards) - 1; index >= 0; index-- {
		segment := segmentsBackwards[index]
		if segment.tokenID == -1 {
			ids = append(ids, tokenizer.idsForUnknownText(word[segment.start:segment.end])...)
			continue
		}
		ids = append(ids, segment.tokenID)
	}
	return ids
}

func (tokenizer *Tokenizer) sentencePieceTokenBytes(id int) []byte {
	piece := tokenizer.idToToken[id]
	switch tokenizer.pieceTypeOf(id) {
	case controlPiece, unknownPiece:
		return nil
	case bytePiece:
		value, isByte := byteValueOfPieceText(piece)
		if !isByte {
			return nil
		}
		return []byte{value}
	}
	return []byte(strings.ReplaceAll(piece, spaceSymbol, " "))
}

func (tokenizer *Tokenizer) decodeSentencePiece(ids []int) string {
	model := &tokenizer.sentencePiece
	var result []byte
	dropNextLeadingSpace := model.addDummyPrefix
	for _, id := range ids {
		tokenizer.checkTokenID(id)
		if tokenizer.isSpecial(id) {
			result = append(result, tokenizer.idToToken[id]...)
			dropNextLeadingSpace = model.addDummyPrefix && !model.dummyPrefixOnlyAtStartOfText
			continue
		}
		pieceBytes := tokenizer.sentencePieceTokenBytes(id)
		startsWithSpace := len(pieceBytes) > 0 && pieceBytes[0] == ' '
		if dropNextLeadingSpace && startsWithSpace && tokenizer.pieceTypeOf(id) != bytePiece {
			pieceBytes = pieceBytes[1:]
		}
		dropNextLeadingSpace = false
		result = append(result, pieceBytes...)
	}
	return string(result)
}
