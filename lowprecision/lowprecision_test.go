package lowprecision

import (
	"github.com/javanhut/GoTransformers/vectormath"
	"math"
	"testing"
)

func TestRowsKeepValuesCloseEnough(t *testing.T) {
	allowedError := map[Precision]float64{
		Float64: 0,
		Float32: 1e-6,
		Int8:    0.01,
		FP4:     0.25,
	}
	for precision, allowed := range allowedError {
		rows := NewRows(precision, 37)
		var saved []vectormath.Vector
		for range 5 {
			values := vectormath.NewVector(37)
			for i := range values {
				values[i] = vectormath.RandomNumberBetween(-3, 3)
			}
			saved = append(saved, values)
			rows.Append(values)
		}
		for row, values := range saved {
			readBack := rows.Row(row)
			for i := range values {
				difference := math.Abs(readBack[i] - values[i])
				if difference > allowed*3 {
					t.Errorf("%v row %d value %d: saved %v read back %v", precision, row, i, values[i], readBack[i])
				}
			}
		}
	}
}

func TestFP4KeepsExactValues(t *testing.T) {
	values := vectormath.Vector{0, 0.5, -1, 1.5, 2, -3, 4, 6, -6, 3}
	readBack := RoundTrip(values, FP4)
	for i := range values {
		if readBack[i] != values[i] {
			t.Errorf("value %d: %v came back as %v", i, values[i], readBack[i])
		}
	}
}

func TestLowerPrecisionUsesLessMemory(t *testing.T) {
	bytesUsed := map[Precision]int{}
	for _, precision := range []Precision{Float64, Float32, Int8, FP4} {
		rows := NewRows(precision, 64)
		for range 10 {
			rows.Append(vectormath.NewVector(64))
		}
		bytesUsed[precision] = rows.BytesUsed()
	}
	if !(bytesUsed[Float64] > bytesUsed[Float32] && bytesUsed[Float32] > bytesUsed[Int8] && bytesUsed[Int8] > bytesUsed[FP4]) {
		t.Errorf("memory used: %v", bytesUsed)
	}
	if bytesUsed[Float64] != 10*64*8 || bytesUsed[FP4] != 10*(32+4*4) {
		t.Errorf("memory used: %v", bytesUsed)
	}
}

func TestDropOldestRows(t *testing.T) {
	for _, precision := range []Precision{Float64, Float32, Int8, FP4} {
		rows := NewRows(precision, 3)
		for row := range 5 {
			rows.Append(vectormath.Vector{float64(row), 0, 0})
		}
		rows.DropOldestRows(2)
		firstValueIsClose := func(row int, want float64) bool {
			return math.Abs(rows.Row(row)[0]-want) < 1e-6
		}
		if rows.NumberOfRows() != 3 || !firstValueIsClose(0, 2) || !firstValueIsClose(1, 3) || !firstValueIsClose(2, 4) {
			t.Errorf("%v: after dropping 2 rows, first value of each row is %v %v %v", precision, rows.Row(0)[0], rows.Row(1)[0], rows.Row(2)[0])
		}
	}
}

func TestDotRowMatchesReadingTheRow(t *testing.T) {
	for _, precision := range []Precision{Float64, Float32, Int8, FP4} {
		rows := NewRows(precision, 37)
		for range 4 {
			values := vectormath.NewVector(37)
			for i := range values {
				values[i] = vectormath.RandomNumberBetween(-3, 3)
			}
			rows.Append(values)
		}
		vector := vectormath.NewVector(37)
		for i := range vector {
			vector[i] = vectormath.RandomNumberBetween(-1, 1)
		}
		for row := range 4 {
			want := vectormath.DotProduct(rows.Row(row), vector)
			if got := rows.DotRow(row, vector); math.Abs(got-want) > 1e-9 {
				t.Errorf("%v row %d: DotRow gave %v, reading the row and multiplying gave %v", precision, row, got, want)
			}
		}
	}
}

func TestAddScaledRowToMatchesReadingTheRow(t *testing.T) {
	for _, precision := range []Precision{Float64, Float32, Int8, FP4} {
		rows := NewRows(precision, 21)
		values := vectormath.NewVector(21)
		for i := range values {
			values[i] = vectormath.RandomNumberBetween(-2, 2)
		}
		rows.Append(values)
		target := vectormath.NewVector(21)
		target[3] = 1
		rows.AddScaledRowTo(0, 0.5, target)
		readBack := rows.Row(0)
		for i := range target {
			want := 0.5 * readBack[i]
			if i == 3 {
				want += 1
			}
			if math.Abs(target[i]-want) > 1e-12 {
				t.Errorf("%v value %d: got %v, want %v", precision, i, target[i], want)
			}
		}
	}
}
