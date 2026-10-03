package vectormath

import (
	"math"
	"reflect"
	"testing"
)

func TestVectorMath(t *testing.T) {
	first := Vector{1, 2, 3}
	second := Vector{4, 5, 6}
	if got := Add(first, second); !reflect.DeepEqual(got, Vector{5, 7, 9}) {
		t.Errorf("Add = %v", got)
	}
	if got := Subtract(first, second); !reflect.DeepEqual(got, Vector{-3, -3, -3}) {
		t.Errorf("Subtract = %v", got)
	}
	if got := MultiplyEach(first, second); !reflect.DeepEqual(got, Vector{4, 10, 18}) {
		t.Errorf("MultiplyEach = %v", got)
	}
	if got := DotProduct(first, second); got != 32 {
		t.Errorf("DotProduct = %v", got)
	}
	if got := IndexOfMax(Vector{3, 9, 2}); got != 1 {
		t.Errorf("IndexOfMax = %v", got)
	}
}

func TestMatrixMath(t *testing.T) {
	first := MatrixFromRows([]Vector{{1, 2, 3}, {4, 5, 6}})
	second := MatrixFromRows([]Vector{{7, 8}, {9, 10}, {11, 12}})

	product := MatrixTimesMatrix(first, second)
	want := MatrixFromRows([]Vector{{58, 64}, {139, 154}})
	if !reflect.DeepEqual(product, want) {
		t.Errorf("MatrixTimesMatrix = %v, want %v", product, want)
	}

	if got := MatrixTimesVector(first, Vector{1, 0, -1}); !reflect.DeepEqual(got, Vector{-2, -2}) {
		t.Errorf("MatrixTimesVector = %v", got)
	}

	transposed := Transpose(first)
	if transposed.Rows != 3 || transposed.Columns != 2 || transposed.Get(2, 1) != 6 {
		t.Errorf("Transpose = %v", transposed)
	}
}

func TestRowChangesTheMatrix(t *testing.T) {
	matrix := NewMatrix(2, 2)
	row := matrix.Row(1)
	row[0] = 5
	if matrix.Get(1, 0) != 5 {
		t.Errorf("writing into Row(1) did not change the matrix")
	}
}

func expectPanic(t *testing.T, name string, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s did not panic", name)
		}
	}()
	function()
}

func TestSizeChecks(t *testing.T) {
	expectPanic(t, "Add with different sizes", func() { Add(Vector{1}, Vector{1, 2}) })
	expectPanic(t, "DotProduct with different sizes", func() { DotProduct(Vector{1}, Vector{1, 2}) })
	expectPanic(t, "MatrixTimesMatrix with wrong shapes", func() { MatrixTimesMatrix(NewMatrix(2, 3), NewMatrix(2, 3)) })
	expectPanic(t, "Get outside the matrix", func() { NewMatrix(2, 2).Get(0, 2) })
	expectPanic(t, "IndexOfMax of empty vector", func() { IndexOfMax(Vector{}) })
}

func slowMatrixTimesMatrix(first Matrix, second Matrix) Matrix {
	result := NewMatrix(first.Rows, second.Columns)
	for row := 0; row < first.Rows; row++ {
		for column := 0; column < second.Columns; column++ {
			sum := 0.0
			for step := 0; step < first.Columns; step++ {
				sum += first.Get(row, step) * second.Get(step, column)
			}
			result.Set(row, column, sum)
		}
	}
	return result
}

func TestThreadedCPUMatchesSlowVersion(t *testing.T) {
	first := NewRandomMatrix(130, 70, -1, 1)
	second := NewRandomMatrix(70, 90, -1, 1)
	want := slowMatrixTimesMatrix(first, second)
	results := map[string]Matrix{
		"MatrixTimesMatrix":     MatrixTimesMatrix(first, second),
		"MatrixTimesTransposed": MatrixTimesTransposed(first, Transpose(second)),
		"TransposedTimesMatrix": TransposedTimesMatrix(Transpose(first), second),
	}
	for name, got := range results {
		for i := range want.Values {
			if math.Abs(got.Values[i]-want.Values[i]) > 1e-9 {
				t.Errorf("%s value %d: got %v, want %v", name, i, got.Values[i], want.Values[i])
				break
			}
		}
	}
}

func TestOneRowSplitsAcrossColumns(t *testing.T) {
	first := NewRandomMatrix(1, 300, -1, 1)
	second := NewRandomMatrix(300, 900, -1, 1)
	want := slowMatrixTimesMatrix(first, second)
	results := map[string]Matrix{
		"MatrixTimesMatrix":     MatrixTimesMatrix(first, second),
		"MatrixTimesTransposed": MatrixTimesTransposed(first, Transpose(second)),
	}
	for name, got := range results {
		for i := range want.Values {
			if math.Abs(got.Values[i]-want.Values[i]) > 1e-9 {
				t.Errorf("%s value %d: got %v, want %v", name, i, got.Values[i], want.Values[i])
				break
			}
		}
	}
}
