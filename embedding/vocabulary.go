package embedding

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

const UnknownToken = "<unknown>"

type Vocabulary struct {
	TokenToID map[string]int
	IDToToken []string
}

func emptyVocabulary() *Vocabulary {
	return &Vocabulary{TokenToID: map[string]int{}}
}

func NewVocabulary() *Vocabulary {
	vocabulary := emptyVocabulary()
	vocabulary.Add(UnknownToken)
	return vocabulary
}

func BuildVocabulary(tokens []string) *Vocabulary {
	vocabulary := NewVocabulary()
	for _, token := range tokens {
		vocabulary.Add(token)
	}
	return vocabulary
}

func (vocabulary *Vocabulary) Add(token string) int {
	if id, found := vocabulary.TokenToID[token]; found {
		return id
	}
	id := len(vocabulary.IDToToken)
	vocabulary.TokenToID[token] = id
	vocabulary.IDToToken = append(vocabulary.IDToToken, token)
	return id
}

func (vocabulary *Vocabulary) Size() int {
	return len(vocabulary.IDToToken)
}

func (vocabulary *Vocabulary) Encode(tokens []string) []int {
	unknownID := vocabulary.TokenToID[UnknownToken]
	ids := make([]int, len(tokens))
	for i, token := range tokens {
		id, found := vocabulary.TokenToID[token]
		if !found {
			id = unknownID
		}
		ids[i] = id
	}
	return ids
}

func (vocabulary *Vocabulary) Decode(ids []int) []string {
	tokens := make([]string, len(ids))
	for i, id := range ids {
		if id < 0 || id >= vocabulary.Size() {
			panic(fmt.Sprintf("Decode: token ID %d is outside the vocabulary of %d tokens", id, vocabulary.Size()))
		}
		tokens[i] = vocabulary.IDToToken[id]
	}
	return tokens
}

func SplitIntoWords(text string) []string {
	return strings.Fields(text)
}

func SplitIntoCharacters(text string) []string {
	var characters []string
	for _, character := range text {
		characters = append(characters, string(character))
	}
	return characters
}

func (vocabulary *Vocabulary) SaveToFile(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	for _, token := range vocabulary.IDToToken {
		writer.WriteString(strconv.Quote(token))
		writer.WriteString("\n")
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func LoadVocabularyFromFile(path string) (*Vocabulary, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	vocabulary := emptyVocabulary()
	scanner := bufio.NewScanner(file)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		token, err := strconv.Unquote(scanner.Text())
		if err != nil {
			return nil, fmt.Errorf("%s line %d: expected a quoted token, got %q", path, lineNumber, scanner.Text())
		}
		vocabulary.Add(token)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	vocabulary.Add(UnknownToken)
	return vocabulary, nil
}
