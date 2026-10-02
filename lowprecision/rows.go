package lowprecision

import (
	"fmt"
	"transformer/vectormath"
)

type Rows struct {
	Precision Precision
	Width     int

	numberOfRows  int
	float64Values []float64
	float32Values []float32
	int8Values    []int8
	fp4Values     []byte
	scales        []float32
}

func NewRows(precision Precision, width int) *Rows {
	if width <= 0 {
		panic(fmt.Sprintf("NewRows: width must be at least 1, got %d", width))
	}
	if precision < Float64 || precision > FP4 {
		panic(fmt.Sprintf("NewRows: unknown precision %v", precision))
	}
	return &Rows{Precision: precision, Width: width}
}

func (rows *Rows) NumberOfRows() int {
	return rows.numberOfRows
}

func (rows *Rows) bytesPerRowForFP4() int {
	return (rows.Width + 1) / 2
}

func (rows *Rows) Append(values vectormath.Vector) {
	if len(values) != rows.Width {
		panic(fmt.Sprintf("Rows.Append: got %d values but each row holds %d", len(values), rows.Width))
	}
	switch rows.Precision {
	case Float64:
		rows.float64Values = append(rows.float64Values, values...)
	case Float32:
		for _, value := range values {
			rows.float32Values = append(rows.float32Values, float32(value))
		}
	case Int8:
		for block := 0; block < numberOfScaleBlocks(rows.Width); block++ {
			blockValues := valuesInBlock(values, block)
			scale := largestMagnitude(blockValues) / 127
			rows.scales = append(rows.scales, float32(scale))
			for _, value := range blockValues {
				rows.int8Values = append(rows.int8Values, encodeInt8(value, float64(float32(scale))))
			}
		}
	case FP4:
		codes := make([]byte, rows.Width)
		for block := 0; block < numberOfScaleBlocks(rows.Width); block++ {
			blockValues := valuesInBlock(values, block)
			scale := largestMagnitude(blockValues) / largestFP4Magnitude
			rows.scales = append(rows.scales, float32(scale))
			for i, value := range blockValues {
				codes[block*ValuesPerScale+i] = encodeFP4(value, float64(float32(scale)))
			}
		}
		packed := make([]byte, rows.bytesPerRowForFP4())
		for i, code := range codes {
			if i%2 == 0 {
				packed[i/2] = code
			} else {
				packed[i/2] |= code << 4
			}
		}
		rows.fp4Values = append(rows.fp4Values, packed...)
	}
	rows.numberOfRows++
}

func valuesInBlock(values vectormath.Vector, block int) vectormath.Vector {
	start := block * ValuesPerScale
	end := start + ValuesPerScale
	if end > len(values) {
		end = len(values)
	}
	return values[start:end]
}

func (rows *Rows) Row(index int) vectormath.Vector {
	if index < 0 || index >= rows.numberOfRows {
		panic(fmt.Sprintf("Rows.Row: row %d is outside %d rows", index, rows.numberOfRows))
	}
	result := vectormath.NewVector(rows.Width)
	switch rows.Precision {
	case Float64:
		copy(result, rows.float64Values[index*rows.Width:(index+1)*rows.Width])
	case Float32:
		for i := range result {
			result[i] = float64(rows.float32Values[index*rows.Width+i])
		}
	case Int8:
		blocksPerRow := numberOfScaleBlocks(rows.Width)
		for i := range result {
			scale := float64(rows.scales[index*blocksPerRow+i/ValuesPerScale])
			result[i] = float64(rows.int8Values[index*rows.Width+i]) * scale
		}
	case FP4:
		blocksPerRow := numberOfScaleBlocks(rows.Width)
		rowStart := index * rows.bytesPerRowForFP4()
		for i := range result {
			packed := rows.fp4Values[rowStart+i/2]
			code := packed & 15
			if i%2 == 1 {
				code = packed >> 4
			}
			scale := float64(rows.scales[index*blocksPerRow+i/ValuesPerScale])
			result[i] = decodeFP4(code, scale)
		}
	}
	return result
}

func (rows *Rows) DropOldestRows(count int) {
	if count <= 0 {
		return
	}
	if count > rows.numberOfRows {
		count = rows.numberOfRows
	}
	blocksPerRow := numberOfScaleBlocks(rows.Width)
	switch rows.Precision {
	case Float64:
		rows.float64Values = append(rows.float64Values[:0], rows.float64Values[count*rows.Width:]...)
	case Float32:
		rows.float32Values = append(rows.float32Values[:0], rows.float32Values[count*rows.Width:]...)
	case Int8:
		rows.int8Values = append(rows.int8Values[:0], rows.int8Values[count*rows.Width:]...)
		rows.scales = append(rows.scales[:0], rows.scales[count*blocksPerRow:]...)
	case FP4:
		rows.fp4Values = append(rows.fp4Values[:0], rows.fp4Values[count*rows.bytesPerRowForFP4():]...)
		rows.scales = append(rows.scales[:0], rows.scales[count*blocksPerRow:]...)
	}
	rows.numberOfRows -= count
}

func (rows *Rows) BytesUsed() int {
	return len(rows.float64Values)*8 + len(rows.float32Values)*4 + len(rows.int8Values) + len(rows.fp4Values) + len(rows.scales)*4
}

type RowsSnapshot struct {
	Precision     Precision
	Width         int
	NumberOfRows  int
	Float64Values []float64
	Float32Values []float32
	Int8Values    []int8
	FP4Values     []byte
	Scales        []float32
}

func (rows *Rows) Snapshot() RowsSnapshot {
	return RowsSnapshot{
		Precision:     rows.Precision,
		Width:         rows.Width,
		NumberOfRows:  rows.numberOfRows,
		Float64Values: append([]float64(nil), rows.float64Values...),
		Float32Values: append([]float32(nil), rows.float32Values...),
		Int8Values:    append([]int8(nil), rows.int8Values...),
		FP4Values:     append([]byte(nil), rows.fp4Values...),
		Scales:        append([]float32(nil), rows.scales...),
	}
}

func RowsFromSnapshot(snapshot RowsSnapshot) (*Rows, error) {
	if snapshot.Width <= 0 || snapshot.NumberOfRows < 0 {
		return nil, fmt.Errorf("rows snapshot has width %d and %d rows", snapshot.Width, snapshot.NumberOfRows)
	}
	rows := NewRows(snapshot.Precision, snapshot.Width)
	rows.numberOfRows = snapshot.NumberOfRows
	rows.float64Values = snapshot.Float64Values
	rows.float32Values = snapshot.Float32Values
	rows.int8Values = snapshot.Int8Values
	rows.fp4Values = snapshot.FP4Values
	rows.scales = snapshot.Scales
	expectedBytes := map[Precision]int{
		Float64: len(rows.float64Values) * 8,
		Float32: len(rows.float32Values) * 4,
		Int8:    len(rows.int8Values) + len(rows.scales)*4,
		FP4:     len(rows.fp4Values) + len(rows.scales)*4,
	}
	blocksPerRow := numberOfScaleBlocks(rows.Width)
	wantedBytes := map[Precision]int{
		Float64: rows.numberOfRows * rows.Width * 8,
		Float32: rows.numberOfRows * rows.Width * 4,
		Int8:    rows.numberOfRows * (rows.Width + blocksPerRow*4),
		FP4:     rows.numberOfRows * (rows.bytesPerRowForFP4() + blocksPerRow*4),
	}
	if expectedBytes[rows.Precision] != wantedBytes[rows.Precision] {
		return nil, fmt.Errorf("rows snapshot for %d rows of width %d at %v has the wrong amount of data", rows.numberOfRows, rows.Width, rows.Precision)
	}
	return rows, nil
}
