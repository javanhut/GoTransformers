package vectormath

import (
	"fmt"
	"runtime"
	"sync"
)

type Backend interface {
	Name() string
	MatrixTimesMatrix(first Matrix, second Matrix) Matrix
	MatrixTimesTransposed(first Matrix, second Matrix) Matrix
	TransposedTimesMatrix(first Matrix, second Matrix) Matrix
}

var currentBackend Backend = CPUBackend{}

func UseBackend(backend Backend) {
	if backend == nil {
		backend = CPUBackend{}
	}
	currentBackend = backend
}

func CurrentBackend() Backend {
	return currentBackend
}

func MatrixTimesMatrix(first Matrix, second Matrix) Matrix {
	if first.Columns != second.Rows {
		panic(fmt.Sprintf("MatrixTimesMatrix: first matrix is %dx%d and second is %dx%d, first columns must equal second rows", first.Rows, first.Columns, second.Rows, second.Columns))
	}
	return currentBackend.MatrixTimesMatrix(first, second)
}

func MatrixTimesTransposed(first Matrix, second Matrix) Matrix {
	if first.Columns != second.Columns {
		panic(fmt.Sprintf("MatrixTimesTransposed: first matrix is %dx%d and second is %dx%d, both need the same number of columns", first.Rows, first.Columns, second.Rows, second.Columns))
	}
	return currentBackend.MatrixTimesTransposed(first, second)
}

func TransposedTimesMatrix(first Matrix, second Matrix) Matrix {
	if first.Rows != second.Rows {
		panic(fmt.Sprintf("TransposedTimesMatrix: first matrix is %dx%d and second is %dx%d, both need the same number of rows", first.Rows, first.Columns, second.Rows, second.Columns))
	}
	return currentBackend.TransposedTimesMatrix(first, second)
}

type CPUBackend struct{}

func (CPUBackend) Name() string {
	return fmt.Sprintf("CPU (%d threads)", runtime.GOMAXPROCS(0))
}

const workBeforeUsingMoreThreads = 64 * 64 * 64

func splitRowsAcrossThreads(numberOfRows int, workPerRow int, doRows func(firstRow int, lastRow int)) {
	numberOfThreads := runtime.GOMAXPROCS(0)
	if numberOfRows*workPerRow < workBeforeUsingMoreThreads || numberOfThreads == 1 || numberOfRows < 2 {
		doRows(0, numberOfRows)
		return
	}
	if numberOfThreads > numberOfRows {
		numberOfThreads = numberOfRows
	}
	rowsPerThread := (numberOfRows + numberOfThreads - 1) / numberOfThreads
	var waitGroup sync.WaitGroup
	for firstRow := 0; firstRow < numberOfRows; firstRow += rowsPerThread {
		lastRow := firstRow + rowsPerThread
		if lastRow > numberOfRows {
			lastRow = numberOfRows
		}
		waitGroup.Add(1)
		go func(firstRow int, lastRow int) {
			defer waitGroup.Done()
			doRows(firstRow, lastRow)
		}(firstRow, lastRow)
	}
	waitGroup.Wait()
}

func (CPUBackend) MatrixTimesMatrix(first Matrix, second Matrix) Matrix {
	result := NewMatrix(first.Rows, second.Columns)
	splitRowsAcrossThreads(first.Rows, first.Columns*second.Columns, func(firstRow int, lastRow int) {
		for row := firstRow; row < lastRow; row++ {
			resultRow := result.Row(row)
			firstRowValues := first.Row(row)
			for step, firstValue := range firstRowValues {
				if firstValue == 0 {
					continue
				}
				secondRowValues := second.Row(step)
				for column := range resultRow {
					resultRow[column] += firstValue * secondRowValues[column]
				}
			}
		}
	})
	return result
}

func (CPUBackend) MatrixTimesTransposed(first Matrix, second Matrix) Matrix {
	result := NewMatrix(first.Rows, second.Rows)
	splitRowsAcrossThreads(first.Rows, first.Columns*second.Rows, func(firstRow int, lastRow int) {
		for row := firstRow; row < lastRow; row++ {
			firstRowValues := first.Row(row)
			resultRow := result.Row(row)
			for column := range resultRow {
				resultRow[column] = DotProduct(firstRowValues, second.Row(column))
			}
		}
	})
	return result
}

func (CPUBackend) TransposedTimesMatrix(first Matrix, second Matrix) Matrix {
	result := NewMatrix(first.Columns, second.Columns)
	splitRowsAcrossThreads(first.Columns, first.Rows*second.Columns, func(firstRow int, lastRow int) {
		for step := 0; step < first.Rows; step++ {
			firstRowValues := first.Row(step)
			secondRowValues := second.Row(step)
			for row := firstRow; row < lastRow; row++ {
				firstValue := firstRowValues[row]
				if firstValue == 0 {
					continue
				}
				resultRow := result.Row(row)
				for column := range resultRow {
					resultRow[column] += firstValue * secondRowValues[column]
				}
			}
		}
	})
	return result
}
