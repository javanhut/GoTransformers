package attention

import (
	"math"
	"transformer/vectormath"
)

const rotaryBase = 10000

const rotateForward = 1.0

const rotateBackward = -1.0

func rotateEachHead(vector vectormath.Vector, position int, numberOfHeads int, headSize int, direction float64) vectormath.Vector {
	rotated := vectormath.CopyVector(vector)
	for head := 0; head < numberOfHeads; head++ {
		headValues := headSlice(rotated, head, headSize)
		for pair := 0; pair < headSize/2; pair++ {
			frequency := math.Pow(rotaryBase, -float64(2*pair)/float64(headSize))
			angle := float64(position) * frequency * direction
			first := headValues[2*pair]
			second := headValues[2*pair+1]
			headValues[2*pair] = first*math.Cos(angle) - second*math.Sin(angle)
			headValues[2*pair+1] = first*math.Sin(angle) + second*math.Cos(angle)
		}
	}
	return rotated
}

func (attention *SelfAttention) rotateRows(matrix vectormath.Matrix, direction float64) vectormath.Matrix {
	if !attention.UseRotaryPositions {
		return matrix
	}
	rotated := vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	for position := 0; position < matrix.Rows; position++ {
		rotated.SetRow(position, rotateEachHead(matrix.Row(position), position, attention.NumberOfHeads, attention.HeadSize(), direction))
	}
	return rotated
}

func (attention *SelfAttention) rotateOne(vector vectormath.Vector, position int) vectormath.Vector {
	if !attention.UseRotaryPositions {
		return vector
	}
	return rotateEachHead(vector, position, attention.NumberOfHeads, attention.HeadSize(), rotateForward)
}
