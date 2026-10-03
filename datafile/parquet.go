package datafile

import (
	"errors"
	"io"
	"path/filepath"
	"strings"

	"github.com/javanhut/GoTransformers/parquet"
)

func isParquet(path string) bool {
	return strings.ToLower(filepath.Ext(path)) == ".parquet"
}

func ReadParquetTexts(path string, column string) ([]string, error) {
	file, err := parquet.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return file.ReadAllStrings(column)
}

type ParquetTextSource struct {
	file              *parquet.File
	column            string
	nextRowGroupIndex int
	textsOfRowGroup   []string
	nextTextIndex     int
}

func OpenParquetTextSource(path string, column string) (*ParquetTextSource, error) {
	file, err := parquet.Open(path)
	if err != nil {
		return nil, err
	}
	source := &ParquetTextSource{file: file, column: column}
	if err := source.readNextRowGroup(); err != nil && !errors.Is(err, io.EOF) {
		file.Close()
		return nil, err
	}
	return source, nil
}

func (source *ParquetTextSource) readNextRowGroup() error {
	if source.nextRowGroupIndex >= source.file.NumberOfRowGroups() {
		return io.EOF
	}
	texts, err := source.file.ReadStringColumn(source.nextRowGroupIndex, source.column)
	if err != nil {
		return err
	}
	source.textsOfRowGroup = texts
	source.nextTextIndex = 0
	source.nextRowGroupIndex++
	return nil
}

func (source *ParquetTextSource) NextText() (string, error) {
	for source.nextTextIndex >= len(source.textsOfRowGroup) {
		if err := source.readNextRowGroup(); err != nil {
			return "", err
		}
	}
	text := source.textsOfRowGroup[source.nextTextIndex]
	source.nextTextIndex++
	return text, nil
}

func (source *ParquetTextSource) Close() error {
	return source.file.Close()
}
