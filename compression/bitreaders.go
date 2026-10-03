package compression

import (
	"errors"
	"math/bits"
)

type forwardBitReader struct {
	data        []byte
	bitPosition int
}

func (reader *forwardBitReader) peekBits(count int) int {
	value := 0
	for index := 0; index < count; index++ {
		bitIndex := reader.bitPosition + index
		byteIndex := bitIndex / 8
		if byteIndex >= len(reader.data) {
			break
		}
		bit := int(reader.data[byteIndex]>>(bitIndex%8)) & 1
		value |= bit << index
	}
	return value
}

func (reader *forwardBitReader) skipBits(count int) {
	reader.bitPosition += count
}

func (reader *forwardBitReader) readBits(count int) int {
	value := reader.peekBits(count)
	reader.skipBits(count)
	return value
}

func (reader *forwardBitReader) ranPastEnd() bool {
	return reader.bitPosition > len(reader.data)*8
}

func (reader *forwardBitReader) bytesUsed() int {
	return (reader.bitPosition + 7) / 8
}

type backwardBitReader struct {
	data          []byte
	bitsRemaining int
}

func newBackwardBitReader(data []byte) (*backwardBitReader, error) {
	if len(data) == 0 {
		return nil, errors.New("zstd: a backward bitstream is empty")
	}
	lastByte := data[len(data)-1]
	if lastByte == 0 {
		return nil, errors.New("zstd: a backward bitstream has no end marker bit in its last byte")
	}
	highestSetBit := bits.Len8(lastByte) - 1
	return &backwardBitReader{data: data, bitsRemaining: (len(data)-1)*8 + highestSetBit}, nil
}

func (reader *backwardBitReader) peekBits(count int) int {
	if count == 0 {
		return 0
	}
	end := reader.bitsRemaining
	start := end - count
	missingBits := 0
	if start < 0 {
		missingBits = -start
		start = 0
	}
	if end <= start {
		return 0
	}
	firstByte := start / 8
	lastByte := (end - 1) / 8
	window := uint64(0)
	for byteIndex := lastByte; byteIndex >= firstByte; byteIndex-- {
		window = window<<8 | uint64(reader.data[byteIndex])
	}
	numberOfBits := end - start
	value := (window >> (start % 8)) & (uint64(1)<<numberOfBits - 1)
	return int(value << missingBits)
}

func (reader *backwardBitReader) skipBits(count int) {
	reader.bitsRemaining -= count
}

func (reader *backwardBitReader) readBits(count int) int {
	value := reader.peekBits(count)
	reader.skipBits(count)
	return value
}

func (reader *backwardBitReader) ranPastStart() bool {
	return reader.bitsRemaining < 0
}

func (reader *backwardBitReader) finished() bool {
	return reader.bitsRemaining == 0
}
