package datafile

import (
	"errors"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

func randomTokenIDs(numberOfTokens int, vocabularySize int, seed uint64) []int {
	randomNumbers := rand.New(rand.NewPCG(seed, seed+1))
	tokenIDs := make([]int, numberOfTokens)
	for index := 0; index < numberOfTokens; index++ {
		tokenIDs[index] = randomNumbers.IntN(vocabularySize)
	}
	tokenIDs[0] = vocabularySize - 1
	return tokenIDs
}

func openWrittenTokenFile(t *testing.T, tokenIDs []int, vocabularySize int) *TokenFile {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokens.bin")
	if err := WriteTokenFile(path, tokenIDs, vocabularySize); err != nil {
		t.Fatal(err)
	}
	tokenFile, err := OpenTokenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { tokenFile.Close() })
	return tokenFile
}

func findChunkStart(reference []int, chunk []int) int {
	for start := 0; start+len(chunk) <= len(reference); start++ {
		if reflect.DeepEqual(reference[start:start+len(chunk)], chunk) {
			return start
		}
	}
	return -1
}

func TestTokenFileRoundTripForBothWidths(t *testing.T) {
	vocabularySizes := []int{300, 65536, 65537, 200000}
	expectedBytesPerToken := []int{2, 2, 4, 4}
	for sizeIndex := 0; sizeIndex < len(vocabularySizes); sizeIndex++ {
		vocabularySize := vocabularySizes[sizeIndex]
		reference := randomTokenIDs(5000, vocabularySize, uint64(sizeIndex))
		tokenFile := openWrittenTokenFile(t, reference, vocabularySize)
		if tokenFile.BytesPerToken() != expectedBytesPerToken[sizeIndex] {
			t.Errorf("vocabulary %d: %d bytes per token, expected %d", vocabularySize, tokenFile.BytesPerToken(), expectedBytesPerToken[sizeIndex])
		}
		if tokenFile.NumberOfTokens() != len(reference) || tokenFile.VocabularySize() != vocabularySize {
			t.Fatalf("vocabulary %d: file has %d tokens and vocabulary %d", vocabularySize, tokenFile.NumberOfTokens(), tokenFile.VocabularySize())
		}

		everyToken, err := tokenFile.ReadTokens(0, len(reference))
		if err != nil || !reflect.DeepEqual(everyToken, reference) {
			t.Fatalf("vocabulary %d: reading every token back gave a different list, %v", vocabularySize, err)
		}

		lastChunk, err := tokenFile.ReadTokens(len(reference)-129, 129)
		if err != nil || !reflect.DeepEqual(lastChunk, reference[len(reference)-129:]) {
			t.Errorf("vocabulary %d: the chunk at the very end is wrong, %v", vocabularySize, err)
		}

		tokenFile.SetRandomSeed(7)
		chunks, err := tokenFile.RandomChunks(129, 50)
		if err != nil {
			t.Fatal(err)
		}
		for _, chunk := range chunks {
			if len(chunk) != 129 || findChunkStart(reference, chunk) < 0 {
				t.Fatalf("vocabulary %d: a random chunk isn't a piece of the tokens", vocabularySize)
			}
		}

		iterator := tokenFile.Chunks(129)
		numberOfChunks := 0
		for {
			chunk, err := iterator.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			start := numberOfChunks * 129
			if !reflect.DeepEqual(chunk, reference[start:start+129]) {
				t.Fatalf("vocabulary %d: sequential chunk %d is wrong", vocabularySize, numberOfChunks)
			}
			numberOfChunks++
		}
		if numberOfChunks != len(reference)/129 || numberOfChunks != iterator.NumberOfChunks() {
			t.Errorf("vocabulary %d: walked %d chunks, expected %d", vocabularySize, numberOfChunks, len(reference)/129)
		}
	}
}

func TestTokenFileBoundaryCases(t *testing.T) {
	reference := []int{5, 6, 7, 8, 9}
	tokenFile := openWrittenTokenFile(t, reference, 10)

	wholeChunk, err := tokenFile.RandomChunk(5)
	if err != nil || !reflect.DeepEqual(wholeChunk, reference) {
		t.Errorf("a chunk as long as the file should be the whole file, got %v, %v", wholeChunk, err)
	}
	if _, err := tokenFile.RandomChunk(6); err == nil {
		t.Error("a chunk longer than the file should be an error")
	}
	if _, err := tokenFile.RandomChunk(0); err == nil {
		t.Error("a chunk of length 0 should be an error")
	}
	if _, err := tokenFile.ReadTokens(3, 3); err == nil {
		t.Error("reading past the end should be an error")
	}
	if _, err := tokenFile.ReadTokens(-1, 2); err == nil {
		t.Error("reading before the start should be an error")
	}
	nothing, err := tokenFile.ReadTokens(5, 0)
	if err != nil || len(nothing) != 0 {
		t.Errorf("reading zero tokens at the end should give an empty list, got %v, %v", nothing, err)
	}

	steppedChunks, err := tokenFile.ChunksWithStep(3, 2).NextBatch(10)
	if err != nil || !reflect.DeepEqual(steppedChunks, [][]int{{5, 6, 7}, {7, 8, 9}}) {
		t.Errorf("chunks of 3 every 2 tokens = %v, %v", steppedChunks, err)
	}
	tooLong := tokenFile.Chunks(6)
	if _, err := tooLong.Next(); !errors.Is(err, io.EOF) || tooLong.NumberOfChunks() != 0 {
		t.Errorf("chunks longer than the file should end at once, got %v", err)
	}

	emptyFile := openWrittenTokenFile(t, nil, 10)
	if emptyFile.NumberOfTokens() != 0 {
		t.Errorf("empty file has %d tokens", emptyFile.NumberOfTokens())
	}
	if _, err := emptyFile.RandomChunk(1); err == nil {
		t.Error("a chunk from an empty file should be an error")
	}
	if _, err := emptyFile.Chunks(1).Next(); !errors.Is(err, io.EOF) {
		t.Errorf("walking an empty file should end at once, got %v", err)
	}
}

func TestTokenFileWriterRejectsBadTokensAndLeavesNoFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokens.bin")
	if err := WriteTokenFile(path, []int{1, 2, 10}, 10); err == nil {
		t.Error("a token outside the vocabulary should be an error")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("a failed write should not leave a token file behind")
	}
	if _, err := os.Stat(path + ".partial"); !os.IsNotExist(err) {
		t.Error("a failed write should not leave a partial file behind")
	}
}

func TestOpenTokenFileRejectsOtherFiles(t *testing.T) {
	notTokens := writeFile(t, "notes.txt", "this is plain text that is long enough to fill a whole header of sixty four bytes.....")
	if _, err := OpenTokenFile(notTokens); err == nil {
		t.Error("a text file should not open as a token file")
	}

	path := filepath.Join(t.TempDir(), "tokens.bin")
	if err := WriteTokenFile(path, randomTokenIDs(100, 50, 3), 50); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, 100); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenTokenFile(path); err == nil {
		t.Error("a cut short token file should not open")
	}
}

func TestDocumentEnds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokens.bin")
	writer, err := CreateTokenFile(path, 100)
	if err != nil {
		t.Fatal(err)
	}
	documents := [][]int{{1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {10}}
	for _, document := range documents {
		if err := writer.WriteTokens(document); err != nil {
			t.Fatal(err)
		}
		writer.EndDocument()
	}
	writer.EndDocument()
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	tokenFile, err := OpenTokenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer tokenFile.Close()
	if tokenFile.NumberOfDocumentsInWholeFile() != 4 {
		t.Errorf("%d documents, expected 4", tokenFile.NumberOfDocumentsInWholeFile())
	}
	documentEnds, err := tokenFile.ReadDocumentEnds()
	if err != nil || !reflect.DeepEqual(documentEnds, []int{3, 5, 9, 10}) {
		t.Errorf("document ends = %v, %v", documentEnds, err)
	}
	everyToken, _ := tokenFile.ReadTokens(0, 10)
	if !reflect.DeepEqual(everyToken, []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
		t.Errorf("tokens = %v", everyToken)
	}

	trainingPart, evaluationPart, err := tokenFile.SplitLast(4)
	if err != nil {
		t.Fatal(err)
	}
	trainingEnds, _ := trainingPart.ReadDocumentEnds()
	evaluationEnds, _ := evaluationPart.ReadDocumentEnds()
	if !reflect.DeepEqual(trainingEnds, []int{3, 5}) || !reflect.DeepEqual(evaluationEnds, []int{3, 4}) {
		t.Errorf("document ends inside the parts = %v and %v", trainingEnds, evaluationEnds)
	}
}

func TestSplitNeverOverlaps(t *testing.T) {
	numberOfTokens := 10000
	reference := make([]int, numberOfTokens)
	for index := 0; index < numberOfTokens; index++ {
		reference[index] = index
	}
	tokenFile := openWrittenTokenFile(t, reference, numberOfTokens)

	trainingPart, evaluationPart, err := tokenFile.Split(0.1)
	if err != nil {
		t.Fatal(err)
	}
	if trainingPart.NumberOfTokens() != 9000 || evaluationPart.NumberOfTokens() != 1000 {
		t.Fatalf("parts have %d and %d tokens", trainingPart.NumberOfTokens(), evaluationPart.NumberOfTokens())
	}

	chunkLength := 129
	trainingChunks, err := trainingPart.RandomChunks(chunkLength, 2000)
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range trainingChunks {
		if chunk[0] < 0 || chunk[chunkLength-1] >= 9000 || chunk[chunkLength-1]-chunk[0] != chunkLength-1 {
			t.Fatalf("training chunk from %d to %d leaves the training part", chunk[0], chunk[chunkLength-1])
		}
	}

	evaluationChunks, err := evaluationPart.RandomChunks(chunkLength, 2000)
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range evaluationChunks {
		if chunk[0] < 9000 || chunk[chunkLength-1] >= numberOfTokens {
			t.Fatalf("evaluation chunk from %d to %d leaves the evaluation part", chunk[0], chunk[chunkLength-1])
		}
	}

	firstEvaluationChunk, err := evaluationPart.Chunks(chunkLength).Next()
	if err != nil || firstEvaluationChunk[0] != 9000 {
		t.Errorf("walking the evaluation part should start at token 9000, got %v", err)
	}

	if _, _, err := tokenFile.Split(1.5); err == nil {
		t.Error("a fraction above 1 should be an error")
	}
	allTraining, noEvaluation, err := tokenFile.Split(0)
	if err != nil || allTraining.NumberOfTokens() != numberOfTokens || noEvaluation.NumberOfTokens() != 0 {
		t.Errorf("split(0) should keep everything for training, %v", err)
	}
}

func TestSamplingDoesNotLoadTheFile(t *testing.T) {
	numberOfTokens := 5_000_000
	path := filepath.Join(t.TempDir(), "big.bin")
	writer, err := CreateTokenFile(path, 50000)
	if err != nil {
		t.Fatal(err)
	}
	piece := make([]int, 10000)
	for written := 0; written < numberOfTokens; written += len(piece) {
		for index := 0; index < len(piece); index++ {
			piece[index] = (written + index) % 50000
		}
		if err := writer.WriteTokens(piece); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	tokenFile, err := OpenTokenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer tokenFile.Close()
	trainingPart, _, err := tokenFile.Split(0.01)
	if err != nil {
		t.Fatal(err)
	}

	var before runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	chunkLength := 129
	for step := 0; step < 2000; step++ {
		chunks, err := trainingPart.RandomChunks(chunkLength, 8)
		if err != nil {
			t.Fatal(err)
		}
		for _, chunk := range chunks {
			if chunk[chunkLength-1] != (chunk[0]+chunkLength-1)%50000 {
				t.Fatalf("chunk starting with %d isn't consecutive tokens", chunk[0])
			}
		}
	}
	var after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&after)

	fileBytes := int64(numberOfTokens * 2)
	heapGrowth := int64(after.HeapAlloc) - int64(before.HeapAlloc)
	if heapGrowth > fileBytes/10 {
		t.Errorf("the heap grew by %d bytes while sampling a %d byte file", heapGrowth, fileBytes)
	}
}

func BenchmarkRandomChunkOf129Tokens(b *testing.B) {
	path := filepath.Join(b.TempDir(), "tokens.bin")
	if err := WriteTokenFile(path, randomTokenIDs(1_000_000, 50000, 9), 50000); err != nil {
		b.Fatal(err)
	}
	tokenFile, err := OpenTokenFile(path)
	if err != nil {
		b.Fatal(err)
	}
	defer tokenFile.Close()
	for b.Loop() {
		if _, err := tokenFile.RandomChunk(129); err != nil {
			b.Fatal(err)
		}
	}
}
