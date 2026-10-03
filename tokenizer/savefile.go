package tokenizer

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

const saveFileFormat = "gotransformers-tokenizer-2"

const firstSaveFileFormat = "gotransformers-tokenizer-1"

func (tokenizer *Tokenizer) SaveToFile(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	fmt.Fprintln(writer, saveFileFormat)
	fmt.Fprintf(writer, "kind %s\n", tokenizer.kind)
	fmt.Fprintf(writer, "addPrefixSpace %v\n", tokenizer.addPrefixSpace)
	fmt.Fprintf(writer, "useWholeWordsFromVocabulary %v\n", tokenizer.useWholeWordsFromVocabulary)
	fmt.Fprintf(writer, "steps %d\n", len(tokenizer.steps))
	for _, step := range tokenizer.steps {
		fmt.Fprintf(writer, "%s %s\n", strconv.Quote(string(step.splitStyle)), strconv.Quote(string(step.digits)))
	}
	fmt.Fprintf(writer, "vocabulary %d\n", len(tokenizer.idToToken))
	for _, token := range tokenizer.idToToken {
		fmt.Fprintln(writer, strconv.Quote(token))
	}
	fmt.Fprintf(writer, "merges %d\n", len(tokenizer.merges))
	for _, merge := range tokenizer.merges {
		fmt.Fprintf(writer, "%s %s\n", strconv.Quote(merge.left), strconv.Quote(merge.right))
	}
	fmt.Fprintf(writer, "special %d\n", len(tokenizer.specialTokensByLen))
	for _, content := range tokenizer.specialTokensByLen {
		fmt.Fprintf(writer, "%d %s\n", tokenizer.specialTokenToID[content], strconv.Quote(content))
	}
	if tokenizer.kind != ByteLevelBPE {
		tokenizer.writeSentencePieceSection(writer)
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

type lineReader struct {
	scanner    *bufio.Scanner
	path       string
	lineNumber int
}

func (reader *lineReader) next() (string, error) {
	if !reader.scanner.Scan() {
		if err := reader.scanner.Err(); err != nil {
			return "", err
		}
		return "", fmt.Errorf("%s: file ended early after line %d", reader.path, reader.lineNumber)
	}
	reader.lineNumber++
	return reader.scanner.Text(), nil
}

func (reader *lineReader) countAfter(label string) (int, error) {
	line, err := reader.next()
	if err != nil {
		return 0, err
	}
	countText, found := strings.CutPrefix(line, label+" ")
	count, err := strconv.Atoi(countText)
	if !found || err != nil || count < 0 {
		return 0, fmt.Errorf("%s line %d: expected %q followed by a count, got %q", reader.path, reader.lineNumber, label, line)
	}
	return count, nil
}

func (reader *lineReader) textAfter(label string) (string, error) {
	line, err := reader.next()
	if err != nil {
		return "", err
	}
	value, found := strings.CutPrefix(line, label+" ")
	if !found {
		return "", fmt.Errorf("%s line %d: expected %q followed by a value, got %q", reader.path, reader.lineNumber, label, line)
	}
	return value, nil
}

func (reader *lineReader) flagAfter(label string) (bool, error) {
	line, err := reader.next()
	if err != nil {
		return false, err
	}
	value, found := strings.CutPrefix(line, label+" ")
	if !found || (value != "true" && value != "false") {
		return false, fmt.Errorf("%s line %d: expected %q followed by true or false, got %q", reader.path, reader.lineNumber, label, line)
	}
	return value == "true", nil
}

func (reader *lineReader) quotedPair() (string, string, error) {
	line, err := reader.next()
	if err != nil {
		return "", "", err
	}
	first, rest, err := readQuoted(line)
	if err != nil || !strings.HasPrefix(rest, " ") {
		return "", "", fmt.Errorf("%s line %d: expected two quoted strings, got %q", reader.path, reader.lineNumber, line)
	}
	second, rest, err := readQuoted(rest[1:])
	if err != nil || rest != "" {
		return "", "", fmt.Errorf("%s line %d: expected two quoted strings, got %q", reader.path, reader.lineNumber, line)
	}
	return first, second, nil
}

func readQuoted(text string) (string, string, error) {
	quoted, err := strconv.QuotedPrefix(text)
	if err != nil {
		return "", "", err
	}
	value, err := strconv.Unquote(quoted)
	return value, text[len(quoted):], err
}

func LoadFromFile(path string) (*Tokenizer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	reader := &lineReader{scanner: scanner, path: path}

	firstLine, err := reader.next()
	if err != nil {
		return nil, err
	}
	if firstLine != saveFileFormat && firstLine != firstSaveFileFormat {
		return nil, fmt.Errorf("%s: not a saved tokenizer (first line is %q)", path, firstLine)
	}
	kind := ByteLevelBPE
	if firstLine == saveFileFormat {
		kindText, err := reader.textAfter("kind")
		if err != nil {
			return nil, err
		}
		kind = Kind(kindText)
		if kind != ByteLevelBPE && kind != SentencePieceBPE && kind != SentencePieceUnigram {
			return nil, fmt.Errorf("%s line %d: tokenizer kind %q is unknown", path, reader.lineNumber, kindText)
		}
	}
	addPrefixSpace, err := reader.flagAfter("addPrefixSpace")
	if err != nil {
		return nil, err
	}
	useWholeWords, err := reader.flagAfter("useWholeWordsFromVocabulary")
	if err != nil {
		return nil, err
	}

	numberOfSteps, err := reader.countAfter("steps")
	if err != nil {
		return nil, err
	}
	var steps []preTokenizerStep
	for range numberOfSteps {
		splitStyle, digits, err := reader.quotedPair()
		if err != nil {
			return nil, err
		}
		steps = append(steps, preTokenizerStep{splitStyle: SplitStyle(splitStyle), digits: DigitSplitting(digits)})
	}
	tokenizer := newEmptyTokenizer(steps)
	tokenizer.addPrefixSpace = addPrefixSpace
	tokenizer.useWholeWordsFromVocabulary = useWholeWords

	vocabularySize, err := reader.countAfter("vocabulary")
	if err != nil {
		return nil, err
	}
	for id := range vocabularySize {
		line, err := reader.next()
		if err != nil {
			return nil, err
		}
		token, rest, err := readQuoted(line)
		if err != nil || rest != "" {
			return nil, fmt.Errorf("%s line %d: expected a quoted token, got %q", path, reader.lineNumber, line)
		}
		tokenizer.setToken(id, token)
	}

	numberOfMerges, err := reader.countAfter("merges")
	if err != nil {
		return nil, err
	}
	for range numberOfMerges {
		left, right, err := reader.quotedPair()
		if err != nil {
			return nil, err
		}
		tokenizer.addMerge(left, right)
	}

	numberOfSpecial, err := reader.countAfter("special")
	if err != nil {
		return nil, err
	}
	for range numberOfSpecial {
		line, err := reader.next()
		if err != nil {
			return nil, err
		}
		idText, quoted, found := strings.Cut(line, " ")
		id, idErr := strconv.Atoi(idText)
		content, rest, quoteErr := readQuoted(quoted)
		if !found || idErr != nil || quoteErr != nil || rest != "" || id < 0 {
			return nil, fmt.Errorf("%s line %d: expected an ID and a quoted special token, got %q", path, reader.lineNumber, line)
		}
		tokenizer.addSpecialTokenWithID(id, content)
	}
	tokenizer.kind = kind
	if kind != ByteLevelBPE {
		if err := tokenizer.readSentencePieceSection(reader); err != nil {
			return nil, err
		}
	}
	return tokenizer, nil
}

func (tokenizer *Tokenizer) sentencePieceFlags() []savedFlag {
	model := &tokenizer.sentencePiece
	return []savedFlag{
		{label: "addDummyPrefix", value: &model.addDummyPrefix},
		{label: "dummyPrefixOnlyAtStartOfText", value: &model.dummyPrefixOnlyAtStartOfText},
		{label: "skipDummyPrefixWhenTextStartsWithSpace", value: &model.skipDummyPrefixWhenTextStartsWithSpace},
		{label: "replaceSpacesWithSpaceSymbol", value: &model.replaceSpacesWithSpaceSymbol},
		{label: "splitBeforeEachSpaceSymbol", value: &model.splitBeforeEachSpaceSymbol},
		{label: "removeExtraWhitespace", value: &model.removeExtraWhitespace},
		{label: "byteFallback", value: &model.byteFallback},
	}
}

type savedFlag struct {
	label string
	value *bool
}

func (tokenizer *Tokenizer) writeSentencePieceSection(writer *bufio.Writer) {
	model := &tokenizer.sentencePiece
	fmt.Fprintf(writer, "mergeOrder %s\n", model.mergeOrder)
	for _, flag := range tokenizer.sentencePieceFlags() {
		fmt.Fprintf(writer, "%s %v\n", flag.label, *flag.value)
	}
	fmt.Fprintf(writer, "pieces %d\n", len(model.types))
	for id := range model.types {
		fmt.Fprintf(writer, "%s %d\n", strconv.FormatFloat(float64(model.scores[id]), 'g', -1, 32), model.types[id])
	}
}

func (tokenizer *Tokenizer) readSentencePieceSection(reader *lineReader) error {
	model := &tokenizer.sentencePiece
	mergeOrderText, err := reader.textAfter("mergeOrder")
	if err != nil {
		return err
	}
	model.mergeOrder = mergeOrder(mergeOrderText)
	if model.mergeOrder != mergeHighestScoreFirst && model.mergeOrder != mergeEarliestListedFirst {
		return fmt.Errorf("%s line %d: merge order %q is unknown", reader.path, reader.lineNumber, mergeOrderText)
	}
	for _, flag := range tokenizer.sentencePieceFlags() {
		value, err := reader.flagAfter(flag.label)
		if err != nil {
			return err
		}
		*flag.value = value
	}
	numberOfPieces, err := reader.countAfter("pieces")
	if err != nil {
		return err
	}
	if numberOfPieces > len(tokenizer.idToToken) {
		return fmt.Errorf("%s line %d: %d pieces but only %d tokens in the vocabulary", reader.path, reader.lineNumber, numberOfPieces, len(tokenizer.idToToken))
	}
	model.scores = make([]float32, numberOfPieces)
	model.types = make([]pieceType, numberOfPieces)
	for id := range numberOfPieces {
		line, err := reader.next()
		if err != nil {
			return err
		}
		scoreText, typeText, found := strings.Cut(line, " ")
		score, scoreErr := strconv.ParseFloat(scoreText, 32)
		typeNumber, typeErr := strconv.Atoi(typeText)
		if !found || scoreErr != nil || typeErr != nil || typeNumber < int(normalPiece) || typeNumber > int(bytePiece) {
			return fmt.Errorf("%s line %d: expected a score and a piece type, got %q", reader.path, reader.lineNumber, line)
		}
		model.scores[id] = float32(score)
		model.types[id] = pieceType(typeNumber)
	}
	tokenizer.prepareSentencePiece()
	return nil
}
