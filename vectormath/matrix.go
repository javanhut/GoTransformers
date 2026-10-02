package vectormath

import "fmt"

type Matrix struct {
	Rows    int
	Columns int
	Values  []float64
}

func NewMatrix(rows int, columns int) Matrix {
	if rows < 0 || columns < 0 {
		panic(fmt.Sprintf("NewMatrix: rows and columns can't be negative, got %dx%d", rows, columns))
	}
	return Matrix{
		Rows:    rows,
		Columns: columns,
		Values:  make([]float64, rows*columns),
	}
}

func MatrixFromRows(rows []Vector) Matrix {
	if len(rows) == 0 {
		return NewMatrix(0, 0)
	}
	matrix := NewMatrix(len(rows), len(rows[0]))
	for row, values := range rows {
		matrix.SetRow(row, values)
	}
	return matrix
}

func (matrix Matrix) checkInside(row int, column int) {
	if row < 0 || row >= matrix.Rows || column < 0 || column >= matrix.Columns {
		panic(fmt.Sprintf("position (%d, %d) is outside a %dx%d matrix", row, column, matrix.Rows, matrix.Columns))
	}
}

func (matrix Matrix) Get(row int, column int) float64 {
	matrix.checkInside(row, column)
	return matrix.Values[row*matrix.Columns+column]
}

func (matrix Matrix) Set(row int, column int, value float64) {
	matrix.checkInside(row, column)
	matrix.Values[row*matrix.Columns+column] = value
}

func (matrix Matrix) AddTo(row int, column int, value float64) {
	matrix.checkInside(row, column)
	matrix.Values[row*matrix.Columns+column] += value
}

func (matrix Matrix) Row(row int) Vector {
	if row < 0 || row >= matrix.Rows {
		panic(fmt.Sprintf("row %d is outside a matrix with %d rows", row, matrix.Rows))
	}
	start := row * matrix.Columns
	end := start + matrix.Columns
	return matrix.Values[start:end:end]
}

func (matrix Matrix) SetRow(row int, values Vector) {
	if len(values) != matrix.Columns {
		panic(fmt.Sprintf("SetRow: got %d values but the matrix has %d columns", len(values), matrix.Columns))
	}
	copy(matrix.Row(row), values)
}

func (matrix Matrix) Copy() Matrix {
	result := NewMatrix(matrix.Rows, matrix.Columns)
	copy(result.Values, matrix.Values)
	return result
}

func checkSameShape(functionName string, first Matrix, second Matrix) {
	if first.Rows != second.Rows || first.Columns != second.Columns {
		panic(fmt.Sprintf("%s: first matrix is %dx%d but second matrix is %dx%d", functionName, first.Rows, first.Columns, second.Rows, second.Columns))
	}
}

func Transpose(matrix Matrix) Matrix {
	result := NewMatrix(matrix.Columns, matrix.Rows)
	for row := 0; row < matrix.Rows; row++ {
		for column := 0; column < matrix.Columns; column++ {
			result.Set(column, row, matrix.Get(row, column))
		}
	}
	return result
}

func MatrixTimesVector(matrix Matrix, vector Vector) Vector {
	if len(vector) != matrix.Columns {
		panic(fmt.Sprintf("MatrixTimesVector: matrix has %d columns but vector has %d values", matrix.Columns, len(vector)))
	}
	result := NewVector(matrix.Rows)
	for row := 0; row < matrix.Rows; row++ {
		result[row] = DotProduct(matrix.Row(row), vector)
	}
	return result
}

func AddMatrices(first Matrix, second Matrix) Matrix {
	checkSameShape("AddMatrices", first, second)
	result := NewMatrix(first.Rows, first.Columns)
	for i := range first.Values {
		result.Values[i] = first.Values[i] + second.Values[i]
	}
	return result
}

func SubtractMatrices(first Matrix, second Matrix) Matrix {
	checkSameShape("SubtractMatrices", first, second)
	result := NewMatrix(first.Rows, first.Columns)
	for i := range first.Values {
		result.Values[i] = first.Values[i] - second.Values[i]
	}
	return result
}

func MultiplyMatricesEach(first Matrix, second Matrix) Matrix {
	checkSameShape("MultiplyMatricesEach", first, second)
	result := NewMatrix(first.Rows, first.Columns)
	for i := range first.Values {
		result.Values[i] = first.Values[i] * second.Values[i]
	}
	return result
}

func ScaleMatrix(matrix Matrix, amount float64) Matrix {
	result := NewMatrix(matrix.Rows, matrix.Columns)
	for i := range matrix.Values {
		result.Values[i] = matrix.Values[i] * amount
	}
	return result
}

func OuterProduct(first Vector, second Vector) Matrix {
	result := NewMatrix(len(first), len(second))
	for row := range first {
		for column := range second {
			result.Set(row, column, first[row]*second[column])
		}
	}
	return result
}

func StackRows(top Matrix, bottom Matrix) Matrix {
	if top.Rows > 0 && bottom.Rows > 0 && top.Columns != bottom.Columns {
		panic(fmt.Sprintf("StackRows: top matrix has %d columns but bottom matrix has %d", top.Columns, bottom.Columns))
	}
	columns := top.Columns
	if top.Rows == 0 {
		columns = bottom.Columns
	}
	result := NewMatrix(top.Rows+bottom.Rows, columns)
	copy(result.Values, top.Values)
	copy(result.Values[len(top.Values):], bottom.Values)
	return result
}

func SplitRows(matrix Matrix, topRows int) (Matrix, Matrix) {
	if topRows < 0 || topRows > matrix.Rows {
		panic(fmt.Sprintf("SplitRows: can't take %d top rows from a matrix with %d rows", topRows, matrix.Rows))
	}
	top := NewMatrix(topRows, matrix.Columns)
	bottom := NewMatrix(matrix.Rows-topRows, matrix.Columns)
	copy(top.Values, matrix.Values[:topRows*matrix.Columns])
	copy(bottom.Values, matrix.Values[topRows*matrix.Columns:])
	return top, bottom
}

func JoinColumns(left Matrix, right Matrix) Matrix {
	if left.Rows != right.Rows {
		panic(fmt.Sprintf("JoinColumns: left matrix has %d rows but right matrix has %d", left.Rows, right.Rows))
	}
	result := NewMatrix(left.Rows, left.Columns+right.Columns)
	for row := 0; row < left.Rows; row++ {
		resultRow := result.Row(row)
		copy(resultRow[:left.Columns], left.Row(row))
		copy(resultRow[left.Columns:], right.Row(row))
	}
	return result
}

func SplitColumns(matrix Matrix, leftColumns int) (Matrix, Matrix) {
	if leftColumns < 0 || leftColumns > matrix.Columns {
		panic(fmt.Sprintf("SplitColumns: can't take %d left columns from a matrix with %d columns", leftColumns, matrix.Columns))
	}
	left := NewMatrix(matrix.Rows, leftColumns)
	right := NewMatrix(matrix.Rows, matrix.Columns-leftColumns)
	for row := 0; row < matrix.Rows; row++ {
		values := matrix.Row(row)
		copy(left.Row(row), values[:leftColumns])
		copy(right.Row(row), values[leftColumns:])
	}
	return left, right
}
