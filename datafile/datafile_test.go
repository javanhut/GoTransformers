package datafile

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeFile(t *testing.T, name string, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadCSV(t *testing.T) {
	path := writeFile(t, "flowers.csv", "length,width,kind\n1.5, 0.2,small\n4.0,1.3,big\n1.4,0.25,small\n")
	table, err := ReadCSV(path, true)
	if err != nil {
		t.Fatal(err)
	}
	if table.NumberOfRows() != 3 || !reflect.DeepEqual(table.ColumnNames, []string{"length", "width", "kind"}) {
		t.Fatalf("read %v", table)
	}

	inputColumns := table.ColumnsExcept([]string{"kind"})
	inputs, err := table.ColumnsAsMatrix(inputColumns)
	if err != nil {
		t.Fatal(err)
	}
	if inputs.Rows != 3 || inputs.Columns != 2 || inputs.Get(1, 1) != 1.3 {
		t.Errorf("inputs = %v", inputs)
	}

	kinds, _ := table.Column("kind")
	targets, classNames := OneHot(kinds)
	if !reflect.DeepEqual(classNames, []string{"big", "small"}) {
		t.Errorf("class names = %v", classNames)
	}
	if targets.Get(0, 1) != 1 || targets.Get(1, 0) != 1 || targets.Get(0, 0) != 0 {
		t.Errorf("one hot targets = %v", targets.Values)
	}

	if _, err := table.ColumnsAsMatrix([]string{"kind"}); err == nil {
		t.Error("turning a text column into numbers did not fail")
	}
	if _, err := table.Column("missing"); err == nil {
		t.Error("asking for a missing column did not fail")
	}
}

func TestReadTSVWithoutHeader(t *testing.T) {
	path := writeFile(t, "numbers.tsv", "1\t2\n3\t4\n")
	table, err := ReadTSV(path, false)
	if err != nil {
		t.Fatal(err)
	}
	numbers, err := table.ColumnAsNumbers("column1")
	if err != nil || !reflect.DeepEqual(numbers, []float64{2, 4}) {
		t.Errorf("column1 = %v, %v", numbers, err)
	}
}

func TestReadCSVWithUnevenRows(t *testing.T) {
	path := writeFile(t, "uneven.csv", "a,b\n1,2\n3\n")
	if _, err := ReadCSV(path, true); err == nil {
		t.Error("a row with a missing value did not fail")
	}
}

func TestReadLines(t *testing.T) {
	path := writeFile(t, "lines.txt", "first\r\nsecond\nthird\n")
	lines, err := ReadLines(path)
	if err != nil || !reflect.DeepEqual(lines, []string{"first", "second", "third"}) {
		t.Errorf("ReadLines = %q, %v", lines, err)
	}
}
