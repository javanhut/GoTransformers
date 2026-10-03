package main

import (
	"fmt"
	"github.com/javanhut/GoTransformers/datafile"
	"github.com/javanhut/GoTransformers/embedding"
	"github.com/javanhut/GoTransformers/transformer"
	"os"
)

func openTrainingTextSource(textPath string, textField string) datafile.TextSource {
	if textPath == "" {
		return datafile.NewSliceTextSource([]string{builtInText})
	}
	source, err := datafile.OpenTrainingTextSource([]string{textPath}, textField)
	if err != nil {
		panic(err)
	}
	return source
}

func learnCharacterVocabularyFromText(textPath string, textField string) *embedding.Vocabulary {
	source := openTrainingTextSource(textPath, textField)
	defer datafile.CloseTextSource(source)
	vocabulary := embedding.NewVocabulary()
	err := datafile.ForEachText(source, func(text string) error {
		for _, character := range embedding.SplitIntoCharacters(text) {
			vocabulary.Add(character)
		}
		return nil
	})
	if err != nil {
		panic(err)
	}
	return vocabulary
}

func buildCharacterTokenFile(tokenFilePath string, vocabularyPath string, textPath string, textField string) {
	vocabulary := learnCharacterVocabularyFromText(textPath, textField)
	if err := vocabulary.SaveToFile(vocabularyPath); err != nil {
		panic(err)
	}
	encode := func(text string) []int {
		return vocabulary.Encode(embedding.SplitIntoCharacters(text))
	}
	source := openTrainingTextSource(textPath, textField)
	defer datafile.CloseTextSource(source)
	options := datafile.StreamingTokenizeOptions{
		ReportProgress: func(progress datafile.TokenizeProgress) {
			fmt.Printf("\rtokenized %d documents, %d characters", progress.DocumentsDone, progress.TokensWritten)
		},
	}
	fmt.Println("building", tokenFilePath, "from", textPath)
	_, err := datafile.TokenizeIntoTokenFile(source, datafile.SameTokenizeFunctionForEveryWorker(encode), tokenFilePath, vocabulary.Size(), options)
	fmt.Println()
	if err != nil {
		panic(err)
	}
}

func openOrBuildCharacterTokenFile(tokenFilePath string, textPath string, textField string) (*embedding.Vocabulary, *datafile.TokenFile) {
	vocabularyPath := tokenFilePath + ".vocabulary.txt"
	if _, err := os.Stat(tokenFilePath); os.IsNotExist(err) {
		buildCharacterTokenFile(tokenFilePath, vocabularyPath, textPath, textField)
	}
	vocabulary, err := embedding.LoadVocabularyFromFile(vocabularyPath)
	if err != nil {
		panic(err)
	}
	tokenFile, err := datafile.OpenTokenFile(tokenFilePath)
	if err != nil {
		panic(err)
	}
	if tokenFile.VocabularySize() != vocabulary.Size() {
		panic(fmt.Sprintf("the token file has a vocabulary of %d but %s has %d", tokenFile.VocabularySize(), vocabularyPath, vocabulary.Size()))
	}
	return vocabulary, tokenFile
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
		panic(err)
	}
	return chunks
}

func holdOutEvaluationChunks(tokenIDs []int, tokenFile *datafile.TokenFile, evaluationFraction float64, chunkLength int, maximumNumberOfChunks int) ([]int, *datafile.TokenFile, [][]int) {
	if evaluationFraction <= 0 {
		return tokenIDs, tokenFile, nil
	}
	if tokenFile != nil {
		trainingPart, evaluationPart, err := tokenFile.Split(evaluationFraction)
		if err != nil {
			fmt.Println("could not split the token file:", err)
			os.Exit(1)
		}
		chunks, err := evaluationPart.Chunks(chunkLength).NextBatch(maximumNumberOfChunks)
		if err != nil {
			fmt.Println("could not read evaluation characters:", err)
			os.Exit(1)
		}
		return nil, trainingPart, chunks
	}
	firstEvaluationToken := len(tokenIDs) - int(float64(len(tokenIDs))*evaluationFraction)
	if len(tokenIDs)-firstEvaluationToken < chunkLength || firstEvaluationToken < chunkLength {
		return tokenIDs, nil, nil
	}
	var chunks [][]int
	for start := firstEvaluationToken; start+chunkLength <= len(tokenIDs) && len(chunks) < maximumNumberOfChunks; start += chunkLength {
		chunks = append(chunks, tokenIDs[start:start+chunkLength])
	}
	return tokenIDs[:firstEvaluationToken], nil, chunks
}
