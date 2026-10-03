package attention

import (
	"math"
	"transformer/vectormath"
)

const rotaryBase = 10000

const rotateForward = 1.0

const rotateBackward = -1.0

type rotarySettings struct {
	dimensions   int
	base         float64
	rotateHalves bool
}

func rotateOneHead(headValues vectormath.Vector, position int, rotary rotarySettings, direction float64) {
	headSize := len(headValues)
	firstRotated := headSize - rotary.dimensions
	half := rotary.dimensions / 2
	for pair := 0; pair < half; pair++ {
		frequency := math.Pow(rotary.base, -float64(2*pair)/float64(rotary.dimensions))
		angle := float64(position) * frequency * direction
		firstIndex := firstRotated + 2*pair
		secondIndex := firstIndex + 1
		if rotary.rotateHalves {
			firstIndex = firstRotated + pair
			secondIndex = firstIndex + half
		}
		first := headValues[firstIndex]
		second := headValues[secondIndex]
		headValues[firstIndex] = first*math.Cos(angle) - second*math.Sin(angle)
		headValues[secondIndex] = first*math.Sin(angle) + second*math.Cos(angle)
	}
}

func rotateEachHead(vector vectormath.Vector, position int, numberOfHeads int, headSize int, rotary rotarySettings, direction float64) vectormath.Vector {
	rotated := vectormath.CopyVector(vector)
	for head := 0; head < numberOfHeads; head++ {
		rotateOneHead(headSlice(rotated, head, headSize), position, rotary, direction)
	}
	return rotated
}

func (attention *SelfAttention) rotary() rotarySettings {
	base := attention.RotaryBase
	if base == 0 {
		base = rotaryBase
	}
	return rotarySettings{dimensions: attention.rotaryDimensions(), base: base, rotateHalves: attention.RotateHalves}
}

func (attention *SelfAttention) rotateRows(matrix vectormath.Matrix, numberOfHeads int, firstPosition int, direction float64) vectormath.Matrix {
	if !attention.UseRotaryPositions {
		return matrix
	}
	rotated := vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	for row := 0; row < matrix.Rows; row++ {
		position := firstPosition + row
		rotated.SetRow(row, rotateEachHead(matrix.Row(row), position, numberOfHeads, attention.HeadSize(), attention.rotary(), direction))
	}
	return rotated
}

func (attention *SelfAttention) rotateOutput(headOutput vectormath.Vector, position int, direction float64) vectormath.Vector {
	if !attention.UseRotaryPositions || !attention.ShareKeyAsValue {
		return headOutput
	}
	rotated := vectormath.CopyVector(headOutput)
	rotateOneHead(rotated, position, attention.rotary(), direction)
	return rotated
}
