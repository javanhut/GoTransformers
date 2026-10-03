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

func SplitAcrossThreads(numberOfItems int, workPerItem int, doItems func(firstItem int, lastItem int)) {
	numberOfThreads := runtime.GOMAXPROCS(0)
	if numberOfItems*workPerItem < workBeforeUsingMoreThreads || numberOfThreads == 1 || numberOfItems < 2 {
		doItems(0, numberOfItems)
		return
	}
	if numberOfThreads > numberOfItems {
		numberOfThreads = numberOfItems
	}
	itemsPerThread := (numberOfItems + numberOfThreads - 1) / numberOfThreads
	var waitGroup sync.WaitGroup
	for firstItem := 0; firstItem < numberOfItems; firstItem += itemsPerThread {
		lastItem := firstItem + itemsPerThread
		if lastItem > numberOfItems {
			lastItem = numberOfItems
		}
		waitGroup.Add(1)
		go func(firstItem int, lastItem int) {
			defer waitGroup.Done()
			doItems(firstItem, lastItem)
		}(firstItem, lastItem)
	}
	waitGroup.Wait()
}

func tooFewRowsForThreads(rows int) bool {
	return rows < runtime.GOMAXPROCS(0)
}

func (CPUBackend) MatrixTimesMatrix(first Matrix, second Matrix) Matrix {
	result := NewMatrix(first.Rows, second.Columns)
	multiplyColumns := func(row int, firstColumn int, lastColumn int) {
		resultRow := result.Row(row)[firstColumn:lastColumn]
		for step, firstValue := range first.Row(row) {
			if firstValue == 0 {
				continue
			}
			secondRowValues := second.Row(step)[firstColumn:lastColumn]
			for column := range resultRow {
				resultRow[column] += firstValue * secondRowValues[column]
			}
		}
	}
	if tooFewRowsForThreads(first.Rows) {
		SplitAcrossThreads(second.Columns, first.Rows*first.Columns, func(firstColumn int, lastColumn int) {
			for row := 0; row < first.Rows; row++ {
				multiplyColumns(row, firstColumn, lastColumn)
			}
		})
		return result
	}
	SplitAcrossThreads(first.Rows, first.Columns*second.Columns, func(firstRow int, lastRow int) {
		for row := firstRow; row < lastRow; row++ {
			multiplyColumns(row, 0, second.Columns)
		}
	})
	return result
}

func (CPUBackend) MatrixTimesTransposed(first Matrix, second Matrix) Matrix {
	result := NewMatrix(first.Rows, second.Rows)
	multiplyColumns := func(row int, firstColumn int, lastColumn int) {
		firstRowValues := first.Row(row)
		resultRow := result.Row(row)
		for column := firstColumn; column < lastColumn; column++ {
			resultRow[column] = DotProduct(firstRowValues, second.Row(column))
		}
	}
	if tooFewRowsForThreads(first.Rows) {
		SplitAcrossThreads(second.Rows, first.Rows*first.Columns, func(firstColumn int, lastColumn int) {
			for row := 0; row < first.Rows; row++ {
				multiplyColumns(row, firstColumn, lastColumn)
			}
		})
		return result
	}
	SplitAcrossThreads(first.Rows, first.Columns*second.Rows, func(firstRow int, lastRow int) {
		for row := firstRow; row < lastRow; row++ {
			multiplyColumns(row, 0, second.Rows)
		}
	})
	return result
}

func (CPUBackend) TransposedTimesMatrix(first Matrix, second Matrix) Matrix {
	result := NewMatrix(first.Columns, second.Columns)
	SplitAcrossThreads(first.Columns, first.Rows*second.Columns, func(firstRow int, lastRow int) {
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
