package main

import (
	"fmt"
	"github.com/javanhut/GoTransformers/datafile"
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/tokenizer"
	"github.com/javanhut/GoTransformers/transformer"
	"os"
)

const bytesOfTextToLearnBPEFrom = 64 << 20

func openTrainingTextSource(textPath string, textField string) datafile.TextSource {
	if textPath == "" {
		return datafile.NewSliceTextSource([]string{builtInText})
	}
	source, err := datafile.OpenTrainingTextSource([]string{textPath}, textField)
	if err != nil {
		fmt.Println("could not open the text:", err)
		os.Exit(1)
	}
	return source
}

func learnCharacterVocabularyFromSource(source datafile.TextSource) (*embedding.Vocabulary, error) {
	vocabulary := embedding.NewVocabulary()
	err := datafile.ForEachText(source, func(text string) error {
		for _, character := range embedding.SplitIntoCharacters(text) {
			vocabulary.Add(character)
		}
		return nil
	})
	return vocabulary, err
}

func learnCoderFromSource(textPath string, textField string, tokenKind string, bpeVocabularySize int) textCoder {
	source := openTrainingTextSource(textPath, textField)
	defer datafile.CloseTextSource(source)
	if tokenKind == "characters" {
		vocabulary, err := learnCharacterVocabularyFromSource(source)
		if err != nil {
			fmt.Println("could not read the text:", err)
			os.Exit(1)
		}
		return characterCoderFromVocabulary(vocabulary)
	}
	sample, err := datafile.ReadTextSample(source, bytesOfTextToLearnBPEFrom)
	if err != nil {
		fmt.Println("could not read the text:", err)
		os.Exit(1)
	}
	return learnCoder(sample, tokenKind, bpeVocabularySize)
}

func loadCoder(tokenKind string, tokenFilePath string) (textCoder, error) {
	if tokenKind != "characters" && tokenKind != "bpe" {
		return textCoder{}, fmt.Errorf("-tokens must be characters or bpe, got %q", tokenKind)
	}
	if tokenKind == "bpe" {
		textTokenizer, err := tokenizer.LoadFromFile(tokenFilePath + ".tokenizer")
		if err != nil {
			return textCoder{}, err
		}
		return bpeCoder(textTokenizer), nil
	}
	vocabulary, err := embedding.LoadVocabularyFromFile(tokenFilePath + ".vocabulary.txt")
	if err != nil {
		return textCoder{}, err
	}
	return characterCoderFromVocabulary(vocabulary), nil
}

func tokenizeFunctionForEachWorker(tokenKind string, coder textCoder, tokenizerPath string) func() datafile.TokenizeFunction {
	if tokenKind != "bpe" {
		return datafile.SameTokenizeFunctionForEveryWorker(coder.encode)
	}
	return func() datafile.TokenizeFunction {
		workerTokenizer, err := tokenizer.LoadFromFile(tokenizerPath)
		if err != nil {
			fmt.Println("could not load the tokenizer for a worker:", err)
			os.Exit(1)
		}
		return workerTokenizer.Encode
	}
}

func buildTokenFile(tokenFilePath string, textPath string, textField string, tokenKind string, bpeVocabularySize int) {
	coder := learnCoderFromSource(textPath, textField, tokenKind, bpeVocabularySize)
	coderPath, err := coder.save(tokenFilePath)
	if err != nil {
		fmt.Println("could not save the tokens next to the token file:", err)
		os.Exit(1)
	}
	source := openTrainingTextSource(textPath, textField)
	defer datafile.CloseTextSource(source)
	options := datafile.StreamingTokenizeOptions{
		ReportProgress: func(progress datafile.TokenizeProgress) {
			fmt.Printf("\rtokenized %d documents, %d tokens", progress.DocumentsDone, progress.TokensWritten)
		},
	}
	fmt.Println("building", tokenFilePath, "from", textPath)
	_, err = datafile.TokenizeIntoTokenFile(source, tokenizeFunctionForEachWorker(tokenKind, coder, coderPath), tokenFilePath, coder.vocabularySize, options)
	fmt.Println()
	if err != nil {
		fmt.Println("could not build the token file:", err)
		os.Exit(1)
	}
}

func openOrBuildTokenFile(tokenFilePath string, textPath string, textField string, tokenKind string, bpeVocabularySize int) (textCoder, *datafile.TokenFile) {
	if _, err := os.Stat(tokenFilePath); os.IsNotExist(err) {
		buildTokenFile(tokenFilePath, textPath, textField, tokenKind, bpeVocabularySize)
	}
	coder, err := loadCoder(tokenKind, tokenFilePath)
	if err != nil {
		fmt.Println("could not load the tokens that go with the token file (was it built with another -tokens?):", err)
		os.Exit(1)
	}
	tokenFile, err := datafile.OpenTokenFile(tokenFilePath)
	if err != nil {
		fmt.Println("could not open the token file:", err)
		os.Exit(1)
	}
	if tokenFile.VocabularySize() != coder.vocabularySize {
		fmt.Printf("the token file has a vocabulary of %d but its tokenizer has %d\n", tokenFile.VocabularySize(), coder.vocabularySize)
		os.Exit(1)
	}
	return coder, tokenFile
}

func countTrainingTokens(tokenIDs []int, tokenFile *datafile.TokenFile) int {
	if tokenFile != nil {
		return tokenFile.NumberOfTokens()
	}
	return len(tokenIDs)
}

func randomTrainingChunks(tokenIDs []int, tokenFile *datafile.TokenFile, chunkLength int, numberOfChunks int) [][]int {
	if tokenFile == nil {
		return transformer.RandomChunks(tokenIDs, chunkLength, numberOfChunks)
	}
	chunks, err := tokenFile.RandomChunks(chunkLength, numberOfChunks)
	if err != nil {
		fmt.Println("could not read from the token file:", err)
		os.Exit(1)
	}
	return chunks
}

func firstTrainingTokens(tokenIDs []int, tokenFile *datafile.TokenFile, count int) []int {
	if tokenFile == nil {
		return tokenIDs[:count]
	}
	firstTokens, err := tokenFile.ReadTokens(0, count)
	if err != nil {
		fmt.Println("could not read from the token file:", err)
		os.Exit(1)
	}
	return firstTokens
}
