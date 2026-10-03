package datafile

import (
	"errors"
	"io"
	"runtime"
	"sync"
)

type TokenizeFunction func(text string) []int

func SameTokenizeFunctionForEveryWorker(tokenize TokenizeFunction) func() TokenizeFunction {
	return func() TokenizeFunction {
		return tokenize
	}
}

type TokenizeProgress struct {
	DocumentsDone int
	TokensWritten int64
}

type StreamingTokenizeOptions struct {
	NumberOfWorkers          int
	DocumentsPerBatch        int
	AppendEndOfDocumentToken bool
	EndOfDocumentTokenID     int
	MarkDocumentEnds         bool
	ReportProgress           func(progress TokenizeProgress)
}

type textBatch struct {
	batchNumber int
	texts       []string
}

type tokenizedBatch struct {
	batchNumber        int
	tokenIDsOfEachText [][]int
}

func readBatchOfTexts(source TextSource, documentsPerBatch int) ([]string, error) {
	texts := make([]string, 0, documentsPerBatch)
	for len(texts) < documentsPerBatch {
		text, err := source.NextText()
		if err != nil {
			return texts, err
		}
		texts = append(texts, text)
	}
	return texts, nil
}

func tokenizeBatches(tokenize TokenizeFunction, batches <-chan textBatch, results chan<- tokenizedBatch) {
	for batch := range batches {
		tokenIDsOfEachText := make([][]int, len(batch.texts))
		for index := 0; index < len(batch.texts); index++ {
			tokenIDsOfEachText[index] = tokenize(batch.texts[index])
		}
		results <- tokenizedBatch{batchNumber: batch.batchNumber, tokenIDsOfEachText: tokenIDsOfEachText}
	}
}

func writeTokenizedBatch(writer *TokenFileWriter, batch tokenizedBatch, options StreamingTokenizeOptions, progress *TokenizeProgress) error {
	for index := 0; index < len(batch.tokenIDsOfEachText); index++ {
		if err := writer.WriteTokens(batch.tokenIDsOfEachText[index]); err != nil {
			return err
		}
		if options.AppendEndOfDocumentToken {
			if err := writer.WriteTokens([]int{options.EndOfDocumentTokenID}); err != nil {
				return err
			}
		}
		if options.MarkDocumentEnds {
			writer.EndDocument()
		}
		progress.DocumentsDone++
	}
	progress.TokensWritten = writer.NumberOfTokens()
	if options.ReportProgress != nil {
		options.ReportProgress(*progress)
	}
	return nil
}

func fillInDefaultTokenizeOptions(options StreamingTokenizeOptions) StreamingTokenizeOptions {
	if options.NumberOfWorkers <= 0 {
		options.NumberOfWorkers = runtime.NumCPU()
	}
	if options.DocumentsPerBatch <= 0 {
		options.DocumentsPerBatch = 64
	}
	return options
}

func TokenizeIntoWriter(source TextSource, newTokenizeFunction func() TokenizeFunction, writer *TokenFileWriter, options StreamingTokenizeOptions) (TokenizeProgress, error) {
	options = fillInDefaultTokenizeOptions(options)
	numberOfBatchesInFlight := 2 * options.NumberOfWorkers
	batchesInFlight := make(chan struct{}, numberOfBatchesInFlight)
	batchesToTokenize := make(chan textBatch, numberOfBatchesInFlight)
	tokenizedBatches := make(chan tokenizedBatch, numberOfBatchesInFlight)
	stopReading := make(chan struct{})

	var readError error
	go func() {
		defer close(batchesToTokenize)
		for batchNumber := 0; ; batchNumber++ {
			select {
			case <-stopReading:
				return
			default:
			}
			texts, err := readBatchOfTexts(source, options.DocumentsPerBatch)
			if len(texts) > 0 {
				select {
				case batchesInFlight <- struct{}{}:
				case <-stopReading:
					return
				}
				batchesToTokenize <- textBatch{batchNumber: batchNumber, texts: texts}
			}
			if errors.Is(err, io.EOF) {
				return
			}
			if err != nil {
				readError = err
				return
			}
		}
	}()

	var workersRunning sync.WaitGroup
	for workerNumber := 0; workerNumber < options.NumberOfWorkers; workerNumber++ {
		tokenize := newTokenizeFunction()
		workersRunning.Add(1)
		go func() {
			defer workersRunning.Done()
			tokenizeBatches(tokenize, batchesToTokenize, tokenizedBatches)
		}()
	}
	go func() {
		workersRunning.Wait()
		close(tokenizedBatches)
	}()

	var progress TokenizeProgress
	var writeError error
	waitingBatches := map[int]tokenizedBatch{}
	nextBatchNumberToWrite := 0
	for batch := range tokenizedBatches {
		waitingBatches[batch.batchNumber] = batch
		for {
			nextBatch, isReady := waitingBatches[nextBatchNumberToWrite]
			if !isReady {
				break
			}
			delete(waitingBatches, nextBatchNumberToWrite)
			nextBatchNumberToWrite++
			if writeError == nil {
				writeError = writeTokenizedBatch(writer, nextBatch, options, &progress)
				if writeError != nil {
					close(stopReading)
				}
			}
			<-batchesInFlight
		}
	}
	if writeError != nil {
		return progress, writeError
	}
	return progress, readError
}

func TokenizeIntoTokenFile(source TextSource, newTokenizeFunction func() TokenizeFunction, path string, vocabularySize int, options StreamingTokenizeOptions) (TokenizeProgress, error) {
	writer, err := CreateTokenFile(path, vocabularySize)
	if err != nil {
		return TokenizeProgress{}, err
	}
	progress, err := TokenizeIntoWriter(source, newTokenizeFunction, writer, options)
	if err != nil {
		writer.Discard()
		return progress, err
	}
	return progress, writer.Close()
}
