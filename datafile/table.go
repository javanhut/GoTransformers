package datafile

import (
	"encoding/csv"
	"fmt"
	"github.com/javanhut/GoTransformers/vectormath"
	"os"
	"sort"
	"strconv"
	"strings"
)

type Table struct {
	ColumnNames []string
	Rows        [][]string
}

func ReadCSV(path string, hasHeader bool) (Table, error) {
	return ReadSeparated(path, ',', hasHeader)
}

func ReadTSV(path string, hasHeader bool) (Table, error) {
	return ReadSeparated(path, '\t', hasHeader)
}

func ReadSeparated(path string, separator rune, hasHeader bool) (Table, error) {
	file, err := os.Open(path)
	if err != nil {
		return Table{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = separator
	reader.TrimLeadingSpace = true
	records, err := reader.ReadAll()
	if err != nil {
		return Table{}, fmt.Errorf("%s: %w", path, err)
	}
	if len(records) == 0 {
		return Table{}, fmt.Errorf("%s: file has no rows", path)
	}

	table := Table{}
	if hasHeader {
		table.ColumnNames = records[0]
		table.Rows = records[1:]
	} else {
		for column := range records[0] {
			table.ColumnNames = append(table.ColumnNames, fmt.Sprintf("column%d", column))
		}
		table.Rows = records
	}
	return table, nil
}

func (table Table) NumberOfRows() int {
	return len(table.Rows)
}

func (table Table) ColumnIndex(name string) (int, error) {
	for index, columnName := range table.ColumnNames {
		if columnName == name {
			return index, nil
		}
	}
	return 0, fmt.Errorf("there is no column named %q, the columns are %v", name, table.ColumnNames)
}

func (table Table) Column(name string) ([]string, error) {
	index, err := table.ColumnIndex(name)
	if err != nil {
		return nil, err
	}
	values := make([]string, len(table.Rows))
	for row := range table.Rows {
		values[row] = table.Rows[row][index]
	}
	return values, nil
}

func (table Table) ColumnAsNumbers(name string) ([]float64, error) {
	texts, err := table.Column(name)
	if err != nil {
		return nil, err
	}
	numbers := make([]float64, len(texts))
	for row, text := range texts {
		number, err := strconv.ParseFloat(strings.TrimSpace(text), 64)
		if err != nil {
			return nil, fmt.Errorf("column %q row %d is %q, which is not a number", name, row, text)
		}
		numbers[row] = number
	}
	return numbers, nil
}

func (table Table) ColumnsAsMatrix(names []string) (vectormath.Matrix, error) {
	matrix := vectormath.NewMatrix(len(table.Rows), len(names))
	for column, name := range names {
		numbers, err := table.ColumnAsNumbers(name)
		if err != nil {
			return vectormath.Matrix{}, err
		}
		for row, number := range numbers {
			matrix.Set(row, column, number)
		}
	}
	return matrix, nil
}

func (table Table) ColumnsExcept(excludedNames []string) []string {
	var remaining []string
	for _, name := range table.ColumnNames {
		excluded := false
		for _, excludedName := range excludedNames {
			if name == excludedName {
				excluded = true
			}
		}
		if !excluded {
			remaining = append(remaining, name)
		}
	}
	return remaining
}

func OneHot(labels []string) (vectormath.Matrix, []string) {
	var classNames []string
	seen := map[string]bool{}
	for _, label := range labels {
		if !seen[label] {
			seen[label] = true
			classNames = append(classNames, label)
		}
	}
	sort.Strings(classNames)

	classColumn := map[string]int{}
	for column, className := range classNames {
		classColumn[className] = column
	}

	matrix := vectormath.NewMatrix(len(labels), len(classNames))
	for row, label := range labels {
		matrix.Set(row, classColumn[label], 1)
	}
	return matrix, classNames
}
