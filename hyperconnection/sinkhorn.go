package hyperconnection

import (
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
)

type sinkhornRecord struct {
	start        vectormath.Matrix
	afterColumns []vectormath.Matrix
	afterRows    []vectormath.Matrix
}

func sinkhornKnopp(raw vectormath.Matrix, steps int) (vectormath.Matrix, sinkhornRecord) {
	largest := vectormath.Max(raw.Values)
	current := vectormath.NewMatrix(raw.Rows, raw.Columns)
	for i, value := range raw.Values {
		current.Values[i] = math.Exp(value - largest)
	}
	record := sinkhornRecord{start: current}
	for step := 0; step < steps; step++ {
		current = normalizeColumns(current)
		record.afterColumns = append(record.afterColumns, current)
		current = normalizeRows(current)
		record.afterRows = append(record.afterRows, current)
	}
	return current, record
}

func columnSums(matrix vectormath.Matrix) vectormath.Vector {
	sums := vectormath.NewVector(matrix.Columns)
	for row := 0; row < matrix.Rows; row++ {
		for column := 0; column < matrix.Columns; column++ {
			sums[column] += matrix.Get(row, column)
		}
	}
	return sums
}

func rowSums(matrix vectormath.Matrix) vectormath.Vector {
	sums := vectormath.NewVector(matrix.Rows)
	for row := 0; row < matrix.Rows; row++ {
		sums[row] = vectormath.Sum(matrix.Row(row))
	}
	return sums
}

func normalizeColumns(matrix vectormath.Matrix) vectormath.Matrix {
	sums := columnSums(matrix)
	result := vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	for row := 0; row < matrix.Rows; row++ {
		for column := 0; column < matrix.Columns; column++ {
			result.Set(row, column, matrix.Get(row, column)/sums[column])
		}
	}
	return result
}

func normalizeRows(matrix vectormath.Matrix) vectormath.Matrix {
	sums := rowSums(matrix)
	result := vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	for row := 0; row < matrix.Rows; row++ {
		for column := 0; column < matrix.Columns; column++ {
			result.Set(row, column, matrix.Get(row, column)/sums[row])
		}
	}
	return result
}

func normalizeColumnsBackward(input vectormath.Matrix, output vectormath.Matrix, outputGradients vectormath.Matrix) vectormath.Matrix {
	sums := columnSums(input)
	inputGradients := vectormath.NewMatrix(input.Rows, input.Columns)
	for column := 0; column < input.Columns; column++ {
		weightedSum := 0.0
		for row := 0; row < input.Rows; row++ {
			weightedSum += outputGradients.Get(row, column) * output.Get(row, column)
		}
		for row := 0; row < input.Rows; row++ {
			inputGradients.Set(row, column, (outputGradients.Get(row, column)-weightedSum)/sums[column])
		}
	}
	return inputGradients
}

func normalizeRowsBackward(input vectormath.Matrix, output vectormath.Matrix, outputGradients vectormath.Matrix) vectormath.Matrix {
	sums := rowSums(input)
	inputGradients := vectormath.NewMatrix(input.Rows, input.Columns)
	for row := 0; row < input.Rows; row++ {
		weightedSum := vectormath.DotProduct(outputGradients.Row(row), output.Row(row))
		for column := 0; column < input.Columns; column++ {
			inputGradients.Set(row, column, (outputGradients.Get(row, column)-weightedSum)/sums[row])
		}
	}
	return inputGradients
}

func sinkhornKnoppBackward(record sinkhornRecord, resultGradients vectormath.Matrix) vectormath.Matrix {
	gradients := resultGradients
	for step := len(record.afterRows) - 1; step >= 0; step-- {
		gradients = normalizeRowsBackward(record.afterColumns[step], record.afterRows[step], gradients)
		columnInput := record.start
		if step > 0 {
			columnInput = record.afterRows[step-1]
		}
		gradients = normalizeColumnsBackward(columnInput, record.afterColumns[step], gradients)
	}
	rawGradients := vectormath.NewMatrix(gradients.Rows, gradients.Columns)
	for i := range gradients.Values {
		rawGradients.Values[i] = gradients.Values[i] * record.start.Values[i]
	}
	return rawGradients
}
