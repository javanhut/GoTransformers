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
)

type JSONRecord map[string]any

func ForEachJSONLRecord(path string, handle func(lineNumber int, record JSONRecord) error) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	reader := bufio.NewReaderSize(file, 1<<20)
	lineNumber := 0
	for {
		line, readErr := reader.ReadBytes('\n')
		if len(line) > 0 {
			lineNumber++
			line = bytes.TrimSpace(line)
			if len(line) > 0 {
				var record JSONRecord
				if err := json.Unmarshal(line, &record); err != nil {
					return fmt.Errorf("%s line %d: not a JSON object: %w", path, lineNumber, err)
				}
				if err := handle(lineNumber, record); err != nil {
					return fmt.Errorf("%s line %d: %w", path, lineNumber, err)
				}
			}
		}
		if errors.Is(readErr, io.EOF) {
			return nil
		}
		if readErr != nil {
			return fmt.Errorf("%s: %w", path, readErr)
		}
	}
}

func ReadJSONL(path string) ([]JSONRecord, error) {
	var records []JSONRecord
	err := ForEachJSONLRecord(path, func(lineNumber int, record JSONRecord) error {
		records = append(records, record)
		return nil
	})
	return records, err
}

func (record JSONRecord) Text(field string) (string, error) {
	value, found := record[field]
	if !found {
		return "", fmt.Errorf("has no %q field (it has %v)", field, record.FieldNames())
	}
	text, isText := value.(string)
	if !isText {
		return "", fmt.Errorf("field %q is %T, not text", field, value)
	}
	return text, nil
}

func (record JSONRecord) FieldNames() []string {
	var names []string
	for name := range record {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func ReadJSONLTexts(path string, field string) ([]string, error) {
	var texts []string
	err := ForEachJSONLRecord(path, func(lineNumber int, record JSONRecord) error {
		text, err := record.Text(field)
		if err != nil {
			return err
		}
		texts = append(texts, text)
		return nil
	})
	return texts, err
}

func (record JSONRecord) HasTextFields(fields ...string) bool {
	for _, field := range fields {
		if _, err := record.Text(field); err != nil {
			return false
		}
	}
	return true
}

var knownPairFields = [][2]string{
	{"prompt", "completion"},
	{"question", "answer"},
	{"input", "target"},
	{"input", "output"},
}

func (record JSONRecord) Pair(inputField string, targetField string) (Pair, error) {
	if record.HasTextFields(inputField, targetField) {
		input, _ := record.Text(inputField)
		target, _ := record.Text(targetField)
		return Pair{Input: input, Target: target}, nil
	}
	if record.HasTextFields("instruction", "output") {
		instruction, _ := record.Text("instruction")
		output, _ := record.Text("output")
		if extraInput, err := record.Text("input"); err == nil && strings.TrimSpace(extraInput) != "" {
			instruction += "\n\n" + extraInput
		}
		return Pair{Input: instruction, Target: output}, nil
	}
	for _, fields := range knownPairFields {
		if record.HasTextFields(fields[0], fields[1]) {
			input, _ := record.Text(fields[0])
			target, _ := record.Text(fields[1])
			return Pair{Input: input, Target: target}, nil
		}
	}
	return Pair{}, fmt.Errorf("has no %q and %q text fields, and isn't prompt/completion, question/answer, instruction/output or input/output either (it has %v)",
		inputField, targetField, record.FieldNames())
}

func ReadJSONLPairs(path string, inputField string, targetField string) ([]Pair, error) {
	var pairs []Pair
	err := ForEachJSONLRecord(path, func(lineNumber int, record JSONRecord) error {
		pair, err := record.Pair(inputField, targetField)
		if err != nil {
			return err
		}
		pairs = append(pairs, pair)
		return nil
	})
	return pairs, err
}

func isJSONL(path string) bool {
	extension := strings.ToLower(filepath.Ext(path))
	return extension == ".jsonl" || extension == ".ndjson"
}

func ReadTrainingText(path string, textField string) (string, error) {
	if !isJSONL(path) {
		return ReadText(path)
	}
	texts, err := ReadJSONLTexts(path, textField)
	if err != nil {
		return "", err
	}
	return strings.Join(texts, "\n\n"), nil
}
