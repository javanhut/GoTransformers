package attention

import (
	"math"
	"transformer/vectormath"
)

const rotaryBase = 10000

const rotateForward = 1.0

const rotateBackward = -1.0

func rotateOneHead(headValues vectormath.Vector, position int, rotaryDimensions int, direction float64) {
	headSize := len(headValues)
	firstRotated := headSize - rotaryDimensions
	for pair := 0; pair < rotaryDimensions/2; pair++ {
		frequency := math.Pow(rotaryBase, -float64(2*pair)/float64(rotaryDimensions))
		angle := float64(position) * frequency * direction
		firstIndex := firstRotated + 2*pair
		first := headValues[firstIndex]
		second := headValues[firstIndex+1]
		headValues[firstIndex] = first*math.Cos(angle) - second*math.Sin(angle)
		headValues[firstIndex+1] = first*math.Sin(angle) + second*math.Cos(angle)
	}
}

func rotateEachHead(vector vectormath.Vector, position int, numberOfHeads int, headSize int, rotaryDimensions int, direction float64) vectormath.Vector {
	rotated := vectormath.CopyVector(vector)
	for head := 0; head < numberOfHeads; head++ {
		rotateOneHead(headSlice(rotated, head, headSize), position, rotaryDimensions, direction)
	}
	return rotated
}

func (attention *SelfAttention) rotateRows(matrix vectormath.Matrix, numberOfHeads int, firstPosition int, direction float64) vectormath.Matrix {
	if !attention.UseRotaryPositions {
		return matrix
	}
	rotated := vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	for row := 0; row < matrix.Rows; row++ {
		position := firstPosition + row
		rotated.SetRow(row, rotateEachHead(matrix.Row(row), position, numberOfHeads, attention.HeadSize(), attention.rotaryDimensions(), direction))
	}
	return rotated
}

func (attention *SelfAttention) rotateOutput(headOutput vectormath.Vector, position int, direction float64) vectormath.Vector {
	if !attention.UseRotaryPositions || !attention.ShareKeyAsValue {
		return headOutput
	}
	rotated := vectormath.CopyVector(headOutput)
	rotateOneHead(rotated, position, attention.rotaryDimensions(), direction)
	return rotated
}
