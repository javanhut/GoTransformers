package datafile

import (
	"path/filepath"
	"strings"
)

type Pair struct {
	Input  string
	Target string
}

func (table Table) Pairs(inputColumn string, targetColumn string) ([]Pair, error) {
	inputs, err := table.Column(inputColumn)
	if err != nil {
		return nil, err
	}
	targets, err := table.Column(targetColumn)
	if err != nil {
		return nil, err
	}
	pairs := make([]Pair, len(inputs))
	for row := range inputs {
		pairs[row] = Pair{Input: inputs[row], Target: targets[row]}
	}
	return pairs, nil
}

func ReadPairs(path string, inputColumn string, targetColumn string) ([]Pair, error) {
	var table Table
	var err error
	if strings.ToLower(filepath.Ext(path)) == ".tsv" {
		table, err = ReadTSV(path, true)
	} else {
		table, err = ReadCSV(path, true)
	}
	if err != nil {
		return nil, err
	}
	return table.Pairs(inputColumn, targetColumn)
}
