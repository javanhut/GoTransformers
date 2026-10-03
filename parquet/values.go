package parquet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const (
	textKind    = "text"
	integerKind = "integer"
	floatKind   = "float"
	booleanKind = "boolean"
)

func valueKindOf(physicalType int) (string, error) {
	switch physicalType {
	case byteArrayType, fixedLengthByteArrayType:
		return textKind, nil
	case int32Type, int64Type:
		return integerKind, nil
	case floatType, doubleType:
		return floatKind, nil
	case booleanType:
		return booleanKind, nil
	}
	return "", fmt.Errorf("physical type %s isn't supported", nameFromList(physicalTypeNames, physicalType))
}

type valueList struct {
	texts    []string
	integers []int64
	floats   []float64
	booleans []bool
}

func (list *valueList) length() int {
	return len(list.texts) + len(list.integers) + len(list.floats) + len(list.booleans)
}

func (list *valueList) appendFromDictionary(dictionary *valueList, index int) {
	switch {
	case dictionary.texts != nil:
		list.texts = append(list.texts, dictionary.texts[index])
	case dictionary.integers != nil:
		list.integers = append(list.integers, dictionary.integers[index])
	case dictionary.floats != nil:
		list.floats = append(list.floats, dictionary.floats[index])
	case dictionary.booleans != nil:
		list.booleans = append(list.booleans, dictionary.booleans[index])
	}
}

func decodePlainValues(data []byte, physicalType int, typeLength int, count int, list *valueList) error {
	switch physicalType {
	case byteArrayType:
		return decodePlainByteArrays(data, count, list)
	case fixedLengthByteArrayType:
		return decodePlainFixedLengthByteArrays(data, typeLength, count, list)
	case int32Type:
		return decodePlainFixedWidthNumbers(data, 4, count, func(littleEndianBytes []byte) {
			list.integers = append(list.integers, int64(int32(binary.LittleEndian.Uint32(littleEndianBytes))))
		})
	case int64Type:
		return decodePlainFixedWidthNumbers(data, 8, count, func(littleEndianBytes []byte) {
			list.integers = append(list.integers, int64(binary.LittleEndian.Uint64(littleEndianBytes)))
		})
	case floatType:
		return decodePlainFixedWidthNumbers(data, 4, count, func(littleEndianBytes []byte) {
			list.floats = append(list.floats, float64(math.Float32frombits(binary.LittleEndian.Uint32(littleEndianBytes))))
		})
	case doubleType:
		return decodePlainFixedWidthNumbers(data, 8, count, func(littleEndianBytes []byte) {
			list.floats = append(list.floats, math.Float64frombits(binary.LittleEndian.Uint64(littleEndianBytes)))
		})
	case booleanType:
		return decodePlainBooleans(data, count, list)
	}
	return fmt.Errorf("physical type %s isn't supported", nameFromList(physicalTypeNames, physicalType))
}

func decodePlainByteArrays(data []byte, count int, list *valueList) error {
	position := 0
	for index := 0; index < count; index++ {
		if position+4 > len(data) {
			return fmt.Errorf("byte array %d of %d starts past the end of the page", index, count)
		}
		length := int(binary.LittleEndian.Uint32(data[position:]))
		position += 4
		if length < 0 || length > len(data)-position {
			return fmt.Errorf("byte array %d of %d says it is %d bytes, which runs past the end of the page", index, count, length)
		}
		list.texts = append(list.texts, string(data[position:position+length]))
		position += length
	}
	return nil
}

func decodePlainFixedLengthByteArrays(data []byte, typeLength int, count int, list *valueList) error {
	if typeLength <= 0 {
		return fmt.Errorf("fixed length byte array column has length %d", typeLength)
	}
	if count*typeLength > len(data) {
		return fmt.Errorf("page holds %d bytes, too few for %d values of %d bytes", len(data), count, typeLength)
	}
	for index := 0; index < count; index++ {
		start := index * typeLength
		list.texts = append(list.texts, string(data[start:start+typeLength]))
	}
	return nil
}

func decodePlainFixedWidthNumbers(data []byte, bytesPerValue int, count int, appendValue func(littleEndianBytes []byte)) error {
	if count*bytesPerValue > len(data) {
		return fmt.Errorf("page holds %d bytes, too few for %d values of %d bytes", len(data), count, bytesPerValue)
	}
	for index := 0; index < count; index++ {
		start := index * bytesPerValue
		appendValue(data[start : start+bytesPerValue])
	}
	return nil
}

func decodePlainBooleans(data []byte, count int, list *valueList) error {
	if (count+7)/8 > len(data) {
		return fmt.Errorf("page holds %d bytes, too few for %d booleans", len(data), count)
	}
	for index := 0; index < count; index++ {
		bit := (data[index/8] >> (index % 8)) & 1
		list.booleans = append(list.booleans, bit == 1)
	}
	return nil
}

func bitWidthFor(largestValue int) int {
	bitWidth := 0
	for largestValue > 0 {
		bitWidth++
		largestValue >>= 1
	}
	return bitWidth
}

func readBitPackedValue(data []byte, bitOffset int, bitWidth int) int {
	firstByte := bitOffset / 8
	window := uint64(0)
	for index := 0; index < 5 && firstByte+index < len(data); index++ {
		window |= uint64(data[firstByte+index]) << (8 * index)
	}
	return int((window >> (bitOffset % 8)) & (uint64(1)<<bitWidth - 1))
}

func decodeRLEBitPackedHybrid(data []byte, bitWidth int, count int) ([]int, error) {
	if bitWidth > 32 {
		return nil, fmt.Errorf("RLE bit width %d is above 32", bitWidth)
	}
	values := make([]int, 0, min(count, 1<<20))
	bytesPerRunValue := (bitWidth + 7) / 8
	position := 0
	for len(values) < count {
		if position >= len(data) {
			return nil, fmt.Errorf("RLE data ends after %d of %d values", len(values), count)
		}
		header, headerLength := binary.Uvarint(data[position:])
		if headerLength <= 0 {
			return nil, errors.New("RLE data has a broken run header")
		}
		position += headerLength
		if header&1 == 0 {
			runLength := int(header >> 1)
			if position+bytesPerRunValue > len(data) {
				return nil, errors.New("RLE run value runs past the end of the data")
			}
			runValue := 0
			for index := 0; index < bytesPerRunValue; index++ {
				runValue |= int(data[position+index]) << (8 * index)
			}
			position += bytesPerRunValue
			for index := 0; index < runLength && len(values) < count; index++ {
				values = append(values, runValue)
			}
		} else {
			numberOfGroups := int(header >> 1)
			numberOfBytes := numberOfGroups * bitWidth
			if numberOfBytes > len(data)-position {
				numberOfBytes = len(data) - position
			}
			packedBytes := data[position : position+numberOfBytes]
			numberOfPackedValues := numberOfGroups * 8
			for index := 0; index < numberOfPackedValues && len(values) < count; index++ {
				values = append(values, readBitPackedValue(packedBytes, index*bitWidth, bitWidth))
			}
			position += numberOfBytes
		}
	}
	return values, nil
}
