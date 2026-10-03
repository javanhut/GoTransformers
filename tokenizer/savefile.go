package tokenizer

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

const saveFileFormat = "gotransformers-tokenizer-1"

func (tokenizer *Tokenizer) SaveToFile(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	fmt.Fprintln(writer, saveFileFormat)
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
	if firstLine != saveFileFormat {
		return nil, fmt.Errorf("%s: not a saved tokenizer (first line is %q)", path, firstLine)
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
	for i := 0; i < numberOfSteps; i++ {
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
	for id := 0; id < vocabularySize; id++ {
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
	for i := 0; i < numberOfMerges; i++ {
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
	for i := 0; i < numberOfSpecial; i++ {
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
	return tokenizer, nil
}
