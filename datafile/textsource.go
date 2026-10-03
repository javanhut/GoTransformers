package datafile

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

type TextSource interface {
	NextText() (string, error)
}

func CloseTextSource(source TextSource) error {
	closer, canClose := source.(io.Closer)
	if !canClose {
		return nil
	}
	return closer.Close()
}

type SliceTextSource struct {
	texts         []string
	nextTextIndex int
}

func NewSliceTextSource(texts []string) *SliceTextSource {
	return &SliceTextSource{texts: texts}
}

func (source *SliceTextSource) NextText() (string, error) {
	if source.nextTextIndex >= len(source.texts) {
		return "", io.EOF
	}
	text := source.texts[source.nextTextIndex]
	source.nextTextIndex++
	return text, nil
}

type PlainTextSplitting int

const (
	SplitIntoPieces PlainTextSplitting = iota
	SplitOnBlankLines
)

const defaultPieceSizeInBytes = 1 << 20
const longestSearchForLineEnd = 1 << 16

type PlainTextSourceOptions struct {
	Splitting        PlainTextSplitting
	PieceSizeInBytes int
}

type PlainTextSource struct {
	path             string
	file             *os.File
	reader           *bufio.Reader
	splitting        PlainTextSplitting
	pieceSizeInBytes int
}

func OpenPlainTextSource(path string, options PlainTextSourceOptions) (*PlainTextSource, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	pieceSizeInBytes := options.PieceSizeInBytes
	if pieceSizeInBytes <= 0 {
		pieceSizeInBytes = defaultPieceSizeInBytes
	}
	return &PlainTextSource{
		path:             path,
		file:             file,
		reader:           bufio.NewReaderSize(file, 1<<20),
		splitting:        options.Splitting,
		pieceSizeInBytes: pieceSizeInBytes,
	}, nil
}

func (source *PlainTextSource) Close() error {
	return source.file.Close()
}

func (source *PlainTextSource) NextText() (string, error) {
	if source.splitting == SplitOnBlankLines {
		return source.nextParagraph()
	}
	return source.nextPiece()
}

func (source *PlainTextSource) nextPiece() (string, error) {
	piece := make([]byte, source.pieceSizeInBytes)
	bytesRead, err := io.ReadFull(source.reader, piece)
	piece = piece[:bytesRead]
	if errors.Is(err, io.EOF) {
		return "", io.EOF
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return string(piece), nil
	}
	if err != nil {
		return "", fmt.Errorf("%s: %w", source.path, err)
	}
	piece, err = source.extendToLineEnd(piece)
	if err != nil {
		return "", fmt.Errorf("%s: %w", source.path, err)
	}
	return string(piece), nil
}

func (source *PlainTextSource) extendToLineEnd(piece []byte) ([]byte, error) {
	for extraBytes := 0; extraBytes < longestSearchForLineEnd; extraBytes++ {
		if piece[len(piece)-1] == '\n' {
			return piece, nil
		}
		nextByte, err := source.reader.ReadByte()
		if errors.Is(err, io.EOF) {
			return piece, nil
		}
		if err != nil {
			return nil, err
		}
		piece = append(piece, nextByte)
	}
	return source.extendToCharacterEnd(piece)
}

func (source *PlainTextSource) extendToCharacterEnd(piece []byte) ([]byte, error) {
	for {
		nextBytes, err := source.reader.Peek(1)
		if errors.Is(err, io.EOF) {
			return piece, nil
		}
		if err != nil {
			return nil, err
		}
		if utf8.RuneStart(nextBytes[0]) {
			return piece, nil
		}
		nextByte, _ := source.reader.ReadByte()
		piece = append(piece, nextByte)
	}
}

func (source *PlainTextSource) nextParagraph() (string, error) {
	var paragraph strings.Builder
	for {
		line, err := source.reader.ReadString('\n')
		lineIsBlank := strings.TrimSpace(line) == ""
		if !lineIsBlank {
			paragraph.WriteString(line)
		}
		if lineIsBlank && paragraph.Len() > 0 && err == nil {
			return strings.TrimRight(paragraph.String(), "\r\n"), nil
		}
		if errors.Is(err, io.EOF) {
			if paragraph.Len() > 0 {
				return strings.TrimRight(paragraph.String(), "\r\n"), nil
			}
			return "", io.EOF
		}
		if err != nil {
			return "", fmt.Errorf("%s: %w", source.path, err)
		}
	}
}

type JSONLTextSource struct {
	path         string
	file         *os.File
	reader       *bufio.Reader
	lineNumber   int
	recordToText func(record JSONRecord) (string, error)
}

func OpenJSONLTextSource(path string, field string) (*JSONLTextSource, error) {
	return OpenJSONLRecordSource(path, func(record JSONRecord) (string, error) {
		return record.Text(field)
	})
}

func OpenJSONLRecordSource(path string, recordToText func(record JSONRecord) (string, error)) (*JSONLTextSource, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &JSONLTextSource{
		path:         path,
		file:         file,
		reader:       bufio.NewReaderSize(file, 1<<20),
		recordToText: recordToText,
	}, nil
}

func (source *JSONLTextSource) Close() error {
	return source.file.Close()
}

func (source *JSONLTextSource) NextText() (string, error) {
	for {
		line, readErr := source.reader.ReadBytes('\n')
		if len(line) > 0 {
			source.lineNumber++
			line = bytes.TrimSpace(line)
		}
		if len(line) > 0 {
			return source.textFromLine(line)
		}
		if errors.Is(readErr, io.EOF) {
			return "", io.EOF
		}
		if readErr != nil {
			return "", fmt.Errorf("%s: %w", source.path, readErr)
		}
	}
}

func (source *JSONLTextSource) textFromLine(line []byte) (string, error) {
	var record JSONRecord
	if err := json.Unmarshal(line, &record); err != nil {
		return "", fmt.Errorf("%s line %d: not a JSON object: %w", source.path, source.lineNumber, err)
	}
	text, err := source.recordToText(record)
	if err != nil {
		return "", fmt.Errorf("%s line %d: %w", source.path, source.lineNumber, err)
	}
	return text, nil
}

type SuffixTextSource struct {
	source TextSource
	suffix string
}

func AppendToEachText(source TextSource, suffix string) *SuffixTextSource {
	return &SuffixTextSource{source: source, suffix: suffix}
}

func (suffixSource *SuffixTextSource) NextText() (string, error) {
	text, err := suffixSource.source.NextText()
	if err != nil {
		return "", err
	}
	return text + suffixSource.suffix, nil
}

func (suffixSource *SuffixTextSource) Close() error {
	return CloseTextSource(suffixSource.source)
}

type MultipleFileTextSource struct {
	paths         []string
	nextPathIndex int
	openOneFile   func(path string) (TextSource, error)
	currentSource TextSource
}

func OpenMultipleFileTextSource(paths []string, openOneFile func(path string) (TextSource, error)) *MultipleFileTextSource {
	return &MultipleFileTextSource{paths: paths, openOneFile: openOneFile}
}

func (multipleSource *MultipleFileTextSource) NextText() (string, error) {
	for {
		if multipleSource.currentSource == nil {
			if multipleSource.nextPathIndex >= len(multipleSource.paths) {
				return "", io.EOF
			}
			source, err := multipleSource.openOneFile(multipleSource.paths[multipleSource.nextPathIndex])
			if err != nil {
				return "", err
			}
			multipleSource.currentSource = source
			multipleSource.nextPathIndex++
		}
		text, err := multipleSource.currentSource.NextText()
		if err == nil {
			return text, nil
		}
		if !errors.Is(err, io.EOF) {
			return "", err
		}
		if closeErr := multipleSource.Close(); closeErr != nil {
			return "", closeErr
		}
	}
}

func (multipleSource *MultipleFileTextSource) Close() error {
	if multipleSource.currentSource == nil {
		return nil
	}
	err := CloseTextSource(multipleSource.currentSource)
	multipleSource.currentSource = nil
	return err
}

func ExpandFilePatterns(patterns []string) ([]string, error) {
	var paths []string
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return nil, fmt.Errorf("file pattern %q: %w", pattern, err)
		}
		if len(matches) == 0 {
			return nil, fmt.Errorf("no file matches %q", pattern)
		}
		sort.Strings(matches)
		paths = append(paths, matches...)
	}
	return paths, nil
}

func OpenTrainingTextFile(path string, textField string) (TextSource, error) {
	if isParquet(path) {
		parquetSource, err := OpenParquetTextSource(path, textField)
		if err != nil {
			return nil, err
		}
		return AppendToEachText(parquetSource, "\n\n"), nil
	}
	if isJSONL(path) {
		jsonlSource, err := OpenJSONLTextSource(path, textField)
		if err != nil {
			return nil, err
		}
		return AppendToEachText(jsonlSource, "\n\n"), nil
	}
	return OpenPlainTextSource(path, PlainTextSourceOptions{Splitting: SplitIntoPieces})
}

func OpenTrainingTextSource(pathPatterns []string, textField string) (TextSource, error) {
	paths, err := ExpandFilePatterns(pathPatterns)
	if err != nil {
		return nil, err
	}
	openOneFile := func(path string) (TextSource, error) {
		return OpenTrainingTextFile(path, textField)
	}
	return OpenMultipleFileTextSource(paths, openOneFile), nil
}

func ReadTextSample(source TextSource, maximumNumberOfBytes int) (string, error) {
	var sample strings.Builder
	for sample.Len() < maximumNumberOfBytes {
		text, err := source.NextText()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return "", err
		}
		sample.WriteString(text)
	}
	return sample.String(), nil
}

func ForEachText(source TextSource, handle func(text string) error) error {
	for {
		text, err := source.NextText()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := handle(text); err != nil {
			return err
		}
	}
}
