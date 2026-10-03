package datafile

import (
	"errors"
	"io"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

var realParquetFile = filepath.Join("..", "parquet", "testdata", "diamonds.parquet")

var expectedDiamondCuts = []string{"Ideal", "Premium", "Good", "Premium", "Good", "Very Good", "Very Good", "Very Good", "Fair", "Very Good"}

func TestReadParquetTexts(t *testing.T) {
	cuts, err := ReadParquetTexts(realParquetFile, "cut")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cuts, expectedDiamondCuts) {
		t.Errorf("cuts = %q", cuts)
	}
	if _, err := ReadParquetTexts(realParquetFile, "price"); err == nil {
		t.Error("reading a number column as text did not fail")
	}
	if _, err := ReadParquetTexts(realParquetFile, "missing"); err == nil || !strings.Contains(err.Error(), "clarity") {
		t.Errorf("missing column error should list the columns, got %v", err)
	}
}

func TestReadTrainingTextReadsParquet(t *testing.T) {
	joined, err := ReadTrainingText(realParquetFile, "clarity")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(joined, "SI2\n\nSI1\n\nVS1") || !strings.HasSuffix(joined, "VS2\n\nVS1") {
		t.Errorf("joined text = %q", joined)
	}
}

func TestParquetTextSourceStreamsEveryText(t *testing.T) {
	source, err := OpenParquetTextSource(realParquetFile, "cut")
	if err != nil {
		t.Fatal(err)
	}
	defer CloseTextSource(source)
	var texts []string
	for {
		text, err := source.NextText()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		texts = append(texts, text)
	}
	if !reflect.DeepEqual(texts, expectedDiamondCuts) {
		t.Errorf("streamed texts = %q", texts)
	}
	if _, err := OpenParquetTextSource(realParquetFile, "missing"); err == nil {
		t.Error("opening a missing column did not fail")
	}
}

func TestTrainingTextSourceStreamsParquet(t *testing.T) {
	source, err := OpenTrainingTextFile("../parquet/testdata/diamonds.parquet", "cut")
	if err != nil {
		t.Fatal(err)
	}
	defer CloseTextSource(source)
	var texts []string
	if err := ForEachText(source, func(text string) error {
		texts = append(texts, text)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(texts) != 10 || texts[0] != "Ideal\n\n" {
		t.Fatalf("got %d texts starting with %q", len(texts), texts[0])
	}
}
