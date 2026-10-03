package datafile

import (
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const testVocabularySize = 1000

func characterCodes(text string) []int {
	var tokenIDs []int
	for _, character := range text {
		tokenIDs = append(tokenIDs, int(character)%testVocabularySize)
	}
	return tokenIDs
}

func slowOnSomeTexts(text string) []int {
	if len(text)%3 == 0 {
		time.Sleep(50 * time.Microsecond)
	}
	return characterCodes(text)
}

func makeDocuments(numberOfDocuments int) []string {
	documents := make([]string, numberOfDocuments)
	for index := 0; index < numberOfDocuments; index++ {
		documents[index] = fmt.Sprintf("document %d %s", index, strings.Repeat("x", index%17))
	}
	return documents
}

func readWholeTokenFile(t *testing.T, path string) *TokenFile {
	t.Helper()
	tokenFile, err := OpenTokenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { tokenFile.Close() })
	return tokenFile
}

func TestStreamingTokenizeKeepsDocumentOrder(t *testing.T) {
	documents := makeDocuments(2000)
	endOfDocumentTokenID := testVocabularySize - 1
	var expected []int
	var expectedEnds []int
	for _, document := range documents {
		expected = append(expected, characterCodes(document)...)
		expected = append(expected, endOfDocumentTokenID)
		expectedEnds = append(expectedEnds, len(expected))
	}

	numberOfWorkersToTry := []int{1, 3, 8}
	for _, numberOfWorkers := range numberOfWorkersToTry {
		path := filepath.Join(t.TempDir(), "tokens.bin")
		var lastProgress TokenizeProgress
		numberOfReports := 0
		options := StreamingTokenizeOptions{
			NumberOfWorkers:          numberOfWorkers,
			DocumentsPerBatch:        7,
			AppendEndOfDocumentToken: true,
			EndOfDocumentTokenID:     endOfDocumentTokenID,
			MarkDocumentEnds:         true,
			ReportProgress: func(progress TokenizeProgress) {
				if progress.DocumentsDone < lastProgress.DocumentsDone || progress.TokensWritten < lastProgress.TokensWritten {
					t.Errorf("progress went backwards: %+v after %+v", progress, lastProgress)
				}
				lastProgress = progress
				numberOfReports++
			},
		}
		progress, err := TokenizeIntoTokenFile(NewSliceTextSource(documents), SameTokenizeFunctionForEveryWorker(slowOnSomeTexts), path, testVocabularySize, options)
		if err != nil {
			t.Fatal(err)
		}
		if progress.DocumentsDone != len(documents) || progress.TokensWritten != int64(len(expected)) || lastProgress != progress || numberOfReports == 0 {
			t.Errorf("%d workers: progress %+v, last report %+v", numberOfWorkers, progress, lastProgress)
		}

		tokenFile := readWholeTokenFile(t, path)
		everyToken, err := tokenFile.ReadTokens(0, tokenFile.NumberOfTokens())
		if err != nil || !reflect.DeepEqual(everyToken, expected) {
			t.Fatalf("%d workers: the tokens are not in document order, %v", numberOfWorkers, err)
		}
		documentEnds, err := tokenFile.ReadDocumentEnds()
		if err != nil || !reflect.DeepEqual(documentEnds, expectedEnds) {
			t.Errorf("%d workers: document ends differ, %v", numberOfWorkers, err)
		}
	}
}

func TestStreamingTokenizeOfPiecesMatchesTokenizingInOneGo(t *testing.T) {
	var contents strings.Builder
	for lineNumber := 0; lineNumber < 5000; lineNumber++ {
		fmt.Fprintf(&contents, "line %d: the quick brown fox ü\n", lineNumber)
	}
	textPath := writeFile(t, "book.txt", contents.String())
	source, err := OpenPlainTextSource(textPath, PlainTextSourceOptions{PieceSizeInBytes: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()

	createdTokenizers := 0
	newTokenizeFunction := func() TokenizeFunction {
		createdTokenizers++
		return characterCodes
	}
	tokenPath := filepath.Join(t.TempDir(), "tokens.bin")
	if _, err := TokenizeIntoTokenFile(source, newTokenizeFunction, tokenPath, testVocabularySize, StreamingTokenizeOptions{NumberOfWorkers: 4}); err != nil {
		t.Fatal(err)
	}
	if createdTokenizers != 4 {
		t.Errorf("made %d tokenize functions for 4 workers", createdTokenizers)
	}
	tokenFile := readWholeTokenFile(t, tokenPath)
	everyToken, _ := tokenFile.ReadTokens(0, tokenFile.NumberOfTokens())
	if !reflect.DeepEqual(everyToken, characterCodes(contents.String())) {
		t.Error("tokenizing in pieces gave different tokens than tokenizing the whole text")
	}
	if tokenFile.NumberOfDocumentsInWholeFile() != 0 {
		t.Error("document ends were stored without being asked for")
	}
}

type failingTextSource struct {
	textsBeforeFailing int
}

func (source *failingTextSource) NextText() (string, error) {
	if source.textsBeforeFailing == 0 {
		return "", errors.New("the disk went away")
	}
	source.textsBeforeFailing--
	return "some text", nil
}

func TestStreamingTokenizeErrorsLeaveNoFile(t *testing.T) {
	folder := t.TempDir()
	path := filepath.Join(folder, "tokens.bin")
	_, err := TokenizeIntoTokenFile(&failingTextSource{textsBeforeFailing: 500}, SameTokenizeFunctionForEveryWorker(characterCodes), path, testVocabularySize, StreamingTokenizeOptions{NumberOfWorkers: 3, DocumentsPerBatch: 4})
	if err == nil || !strings.Contains(err.Error(), "disk went away") {
		t.Errorf("expected the read error, got %v", err)
	}

	tooSmallVocabulary := 50
	_, err = TokenizeIntoTokenFile(NewSliceTextSource(makeDocuments(3000)), SameTokenizeFunctionForEveryWorker(characterCodes), path, tooSmallVocabulary, StreamingTokenizeOptions{NumberOfWorkers: 3, DocumentsPerBatch: 2})
	if err == nil || !strings.Contains(err.Error(), "outside the vocabulary") {
		t.Errorf("expected a vocabulary error, got %v", err)
	}

	leftovers, _ := filepath.Glob(filepath.Join(folder, "*"))
	if len(leftovers) != 0 {
		t.Errorf("failed tokenizing left files behind: %v", leftovers)
	}
}
