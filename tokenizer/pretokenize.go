package tokenizer

import "unicode"

type SplitStyle string

const (
	NoSplit       SplitStyle = "none"
	GPT2Split     SplitStyle = "gpt2"
	Llama3Split   SplitStyle = "llama3"
	Qwen2Split    SplitStyle = "qwen2"
	gpt2Pattern              = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
	llama3Pattern            = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	qwen2Pattern             = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

type DigitSplitting string

const (
	KeepDigitsTogether  DigitSplitting = "none"
	IsolateEachDigit    DigitSplitting = "each"
	IsolateRunsOfDigits DigitSplitting = "runs"
)

func isLetter(character rune) bool {
	return unicode.IsLetter(character)
}

func isNumber(character rune) bool {
	return unicode.IsNumber(character)
}

func isSpace(character rune) bool {
	return unicode.IsSpace(character)
}

func isNewline(character rune) bool {
	return character == '\r' || character == '\n'
}

func isOther(character rune) bool {
	return !isSpace(character) && !isLetter(character) && !isNumber(character)
}

func countWhile(characters []rune, start int, keepGoing func(rune) bool) int {
	count := 0
	for start+count < len(characters) && keepGoing(characters[start+count]) {
		count++
	}
	return count
}

func contractionLength(characters []rune, start int, ignoreCase bool) int {
	if characters[start] != '\'' || start+1 >= len(characters) {
		return 0
	}
	lower := func(index int) rune {
		if index >= len(characters) {
			return 0
		}
		if ignoreCase {
			return unicode.ToLower(characters[index])
		}
		return characters[index]
	}
	second := lower(start + 1)
	third := lower(start + 2)
	if (second == 'r' && third == 'e') || (second == 'v' && third == 'e') || (second == 'l' && third == 'l') {
		return 3
	}
	if second == 's' || second == 't' || second == 'm' || second == 'd' {
		return 2
	}
	return 0
}

func whitespaceMatchLength(characters []rune, start int) int {
	run := countWhile(characters, start, isSpace)
	if run == 0 {
		return 0
	}
	end := start + run
	if end == len(characters) || run == 1 {
		return run
	}
	return run - 1
}

func gpt2PieceLength(characters []rune, start int) int {
	if length := contractionLength(characters, start, false); length > 0 {
		return length
	}
	afterSpace := start
	if characters[start] == ' ' && start+1 < len(characters) {
		afterSpace = start + 1
	}
	for _, kind := range []func(rune) bool{isLetter, isNumber, isOther} {
		if kind(characters[afterSpace]) {
			return afterSpace - start + countWhile(characters, afterSpace, kind)
		}
	}
	return whitespaceMatchLength(characters, start)
}

func llamaPieceLength(characters []rune, start int, longestNumber int) int {
	if length := contractionLength(characters, start, true); length > 0 {
		return length
	}
	first := characters[start]
	if isLetter(first) {
		return countWhile(characters, start, isLetter)
	}
	if !isNewline(first) && !isNumber(first) && start+1 < len(characters) && isLetter(characters[start+1]) {
		return 1 + countWhile(characters, start+1, isLetter)
	}
	if isNumber(first) {
		length := countWhile(characters, start, isNumber)
		if length > longestNumber {
			length = longestNumber
		}
		return length
	}
	afterSpace := start
	if first == ' ' && start+1 < len(characters) && isOther(characters[start+1]) {
		afterSpace = start + 1
	}
	if isOther(characters[afterSpace]) {
		end := afterSpace + countWhile(characters, afterSpace, isOther)
		end += countWhile(characters, end, isNewline)
		return end - start
	}
	run := countWhile(characters, start, isSpace)
	lastNewline := -1
	for i := start; i < start+run; i++ {
		if isNewline(characters[i]) {
			lastNewline = i
		}
	}
	if lastNewline >= 0 {
		return lastNewline + 1 - start
	}
	return whitespaceMatchLength(characters, start)
}

func charactersWithByteOffsets(text string) ([]rune, []int) {
	var characters []rune
	var offsets []int
	for offset, character := range text {
		characters = append(characters, character)
		offsets = append(offsets, offset)
	}
	offsets = append(offsets, len(text))
	return characters, offsets
}

func splitWithStyle(text string, style SplitStyle) []string {
	if style == NoSplit || text == "" {
		return []string{text}
	}
	characters, offsets := charactersWithByteOffsets(text)
	var pieces []string
	for start := 0; start < len(characters); {
		var length int
		switch style {
		case GPT2Split:
			length = gpt2PieceLength(characters, start)
		case Llama3Split:
			length = llamaPieceLength(characters, start, 3)
		case Qwen2Split:
			length = llamaPieceLength(characters, start, 1)
		}
		if length <= 0 {
			length = 1
		}
		pieces = append(pieces, text[offsets[start]:offsets[start+length]])
		start += length
	}
	return pieces
}

func splitDigits(text string, digits DigitSplitting) []string {
	if digits == KeepDigitsTogether {
		return []string{text}
	}
	var pieces []string
	pieceStart := 0
	currentIsDigits := false
	for offset, character := range text {
		characterIsDigit := unicode.IsNumber(character)
		startNewPiece := offset > pieceStart && (characterIsDigit != currentIsDigits || (characterIsDigit && digits == IsolateEachDigit))
		if startNewPiece {
			pieces = append(pieces, text[pieceStart:offset])
			pieceStart = offset
		}
		currentIsDigits = characterIsDigit
	}
	if pieceStart < len(text) {
		pieces = append(pieces, text[pieceStart:])
	}
	return pieces
}

type preTokenizerStep struct {
	splitStyle SplitStyle
	digits     DigitSplitting
}

func preTokenize(text string, steps []preTokenizerStep) []string {
	pieces := []string{text}
	for _, step := range steps {
		var nextPieces []string
		for _, piece := range pieces {
			if step.digits != "" {
				nextPieces = append(nextPieces, splitDigits(piece, step.digits)...)
			} else {
				nextPieces = append(nextPieces, splitWithStyle(piece, step.splitStyle)...)
			}
		}
		pieces = nextPieces
	}
	return pieces
}
