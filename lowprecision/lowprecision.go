package lowprecision

import (
	"fmt"
	"math"
	"transformer/vectormath"
)

type Precision int

const (
	Float64 Precision = iota
	Float32
	Int8
	FP4
)

func (precision Precision) String() string {
	switch precision {
	case Float64:
		return "Float64"
	case Float32:
		return "Float32"
	case Int8:
		return "Int8"
	case FP4:
		return "FP4"
	}
	return fmt.Sprintf("Precision(%d)", int(precision))
}

func (precision Precision) MarshalText() ([]byte, error) {
	return []byte(precision.String()), nil
}

func (precision *Precision) UnmarshalText(text []byte) error {
	for _, possible := range []Precision{Float64, Float32, Int8, FP4} {
		if possible.String() == string(text) {
			*precision = possible
			return nil
		}
	}
	return fmt.Errorf("unknown precision %q", string(text))
}

const ValuesPerScale = 16

var fp4Magnitudes = [8]float64{0, 0.5, 1, 1.5, 2, 3, 4, 6}

const largestFP4Magnitude = 6

func numberOfScaleBlocks(width int) int {
	return (width + ValuesPerScale - 1) / ValuesPerScale
}

func largestMagnitude(values []float64) float64 {
	largest := 0.0
	for _, value := range values {
		if math.Abs(value) > largest {
			largest = math.Abs(value)
		}
	}
	return largest
}

func encodeInt8(value float64, scale float64) int8 {
	if scale == 0 {
		return 0
	}
	rounded := math.Round(value / scale)
	if rounded > 127 {
		rounded = 127
	}
	if rounded < -127 {
		rounded = -127
	}
	return int8(rounded)
}

func encodeFP4(value float64, scale float64) byte {
	if scale == 0 {
		return 0
	}
	var signBit byte
	if value < 0 {
		signBit = 8
	}
	scaled := math.Abs(value) / scale
	closestIndex := 0
	for index, magnitude := range fp4Magnitudes {
		if math.Abs(scaled-magnitude) < math.Abs(scaled-fp4Magnitudes[closestIndex]) {
			closestIndex = index
		}
	}
	return signBit | byte(closestIndex)
}

func decodeFP4(code byte, scale float64) float64 {
	value := fp4Magnitudes[code&7] * scale
	if code&8 != 0 {
		return -value
	}
	return value
}

func RoundTrip(values vectormath.Vector, precision Precision) vectormath.Vector {
	rows := NewRows(precision, len(values))
	rows.Append(values)
	return rows.Row(0)
}

func RoundTripMatrix(matrix vectormath.Matrix, precision Precision) vectormath.Matrix {
	if precision == Float64 {
		return matrix
	}
	result := vectormath.NewMatrix(matrix.Rows, matrix.Columns)
	for row := 0; row < matrix.Rows; row++ {
		result.SetRow(row, RoundTrip(matrix.Row(row), precision))
	}
	return result
}
