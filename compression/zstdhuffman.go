package compression

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math/bits"
)

const largestHuffmanCodeLength = 11

type huffmanTableEntry struct {
	symbol       byte
	numberOfBits int
}

type huffmanTable struct {
	maximumNumberOfBits int
	entries             []huffmanTableEntry
}

func readHuffmanTable(data []byte) (*huffmanTable, int, error) {
	if len(data) == 0 {
		return nil, 0, errors.New("zstd: Huffman table description is missing")
	}
	headerByte := int(data[0])
	var weights []int
	var bytesUsed int
	var err error
	if headerByte < 128 {
		compressedSize := headerByte
		if 1+compressedSize > len(data) {
			return nil, 0, errors.New("zstd: compressed Huffman weights run past the end of the block")
		}
		weights, err = decodeCompressedHuffmanWeights(data[1 : 1+compressedSize])
		bytesUsed = 1 + compressedSize
	} else {
		numberOfWeights := headerByte - 127
		weights, bytesUsed, err = readDirectHuffmanWeights(data, numberOfWeights)
	}
	if err != nil {
		return nil, 0, err
	}
	table, err := buildHuffmanTable(weights)
	if err != nil {
		return nil, 0, err
	}
	return table, bytesUsed, nil
}

func readDirectHuffmanWeights(data []byte, numberOfWeights int) ([]int, int, error) {
	numberOfBytes := (numberOfWeights + 1) / 2
	if 1+numberOfBytes > len(data) {
		return nil, 0, errors.New("zstd: Huffman weights run past the end of the block")
	}
	weights := make([]int, numberOfWeights)
	for index := 0; index < numberOfWeights; index++ {
		packedByte := data[1+index/2]
		if index%2 == 0 {
			weights[index] = int(packedByte >> 4)
		} else {
			weights[index] = int(packedByte & 15)
		}
	}
	return weights, 1 + numberOfBytes, nil
}

func decodeCompressedHuffmanWeights(data []byte) ([]int, error) {
	counts, accuracyLog, headerSize, err := readFSENormalizedCounts(data, 255, 6)
	if err != nil {
		return nil, err
	}
	table, err := buildFSETable(counts, accuracyLog)
	if err != nil {
		return nil, err
	}
	reader, err := newBackwardBitReader(data[headerSize:])
	if err != nil {
		return nil, err
	}
	firstState := reader.readBits(accuracyLog)
	secondState := reader.readBits(accuracyLog)
	var weights []int
	for {
		if len(weights) > 254 {
			return nil, errors.New("zstd: Huffman table has more than 255 weights")
		}
		weights = append(weights, table.symbolAt(firstState))
		firstState = table.nextState(firstState, reader)
		if reader.ranPastStart() {
			weights = append(weights, table.symbolAt(secondState))
			break
		}
		weights = append(weights, table.symbolAt(secondState))
		secondState = table.nextState(secondState, reader)
		if reader.ranPastStart() {
			weights = append(weights, table.symbolAt(firstState))
			break
		}
	}
	return weights, nil
}

func buildHuffmanTable(weights []int) (*huffmanTable, error) {
	if len(weights) > 255 {
		return nil, fmt.Errorf("zstd: Huffman table has %d weights, more than 255", len(weights))
	}
	weightTotal := 0
	for symbol := 0; symbol < len(weights); symbol++ {
		weight := weights[symbol]
		if weight > largestHuffmanCodeLength {
			return nil, fmt.Errorf("zstd: Huffman weight %d is above the limit of %d", weight, largestHuffmanCodeLength)
		}
		if weight > 0 {
			weightTotal += 1 << (weight - 1)
		}
	}
	if weightTotal == 0 {
		return nil, errors.New("zstd: Huffman weights are all zero")
	}
	maximumNumberOfBits := bits.Len(uint(weightTotal))
	if maximumNumberOfBits > largestHuffmanCodeLength {
		return nil, fmt.Errorf("zstd: Huffman codes would be %d bits long, above the limit of %d", maximumNumberOfBits, largestHuffmanCodeLength)
	}
	leftOver := (1 << maximumNumberOfBits) - weightTotal
	if leftOver&(leftOver-1) != 0 {
		return nil, errors.New("zstd: Huffman weights don't leave a power of two for the last symbol")
	}
	lastWeight := bits.Len(uint(leftOver))
	allWeights := append(append([]int{}, weights...), lastWeight)

	numberOfSymbolsWithWeight := make([]int, maximumNumberOfBits+1)
	for symbol := 0; symbol < len(allWeights); symbol++ {
		numberOfSymbolsWithWeight[allWeights[symbol]]++
	}
	nextPositionForWeight := make([]int, maximumNumberOfBits+1)
	position := 0
	for weight := 1; weight <= maximumNumberOfBits; weight++ {
		nextPositionForWeight[weight] = position
		position += numberOfSymbolsWithWeight[weight] << (weight - 1)
	}

	entries := make([]huffmanTableEntry, 1<<maximumNumberOfBits)
	for symbol := 0; symbol < len(allWeights); symbol++ {
		weight := allWeights[symbol]
		if weight == 0 {
			continue
		}
		numberOfEntries := 1 << (weight - 1)
		start := nextPositionForWeight[weight]
		for offset := 0; offset < numberOfEntries; offset++ {
			entries[start+offset] = huffmanTableEntry{symbol: byte(symbol), numberOfBits: maximumNumberOfBits + 1 - weight}
		}
		nextPositionForWeight[weight] += numberOfEntries
	}
	return &huffmanTable{maximumNumberOfBits: maximumNumberOfBits, entries: entries}, nil
}

func decodeHuffmanStream(table *huffmanTable, streamBytes []byte, numberOfSymbols int, output []byte) ([]byte, error) {
	reader, err := newBackwardBitReader(streamBytes)
	if err != nil {
		return nil, err
	}
	for index := 0; index < numberOfSymbols; index++ {
		entry := table.entries[reader.peekBits(table.maximumNumberOfBits)]
		output = append(output, entry.symbol)
		reader.skipBits(entry.numberOfBits)
	}
	if !reader.finished() {
		return nil, fmt.Errorf("zstd: Huffman stream has %d bits left over after its symbols", reader.bitsRemaining)
	}
	return output, nil
}

func decodeHuffmanLiterals(table *huffmanTable, streamBytes []byte, regeneratedSize int, numberOfStreams int) ([]byte, error) {
	output := make([]byte, 0, regeneratedSize)
	if numberOfStreams == 1 {
		return decodeHuffmanStream(table, streamBytes, regeneratedSize, output)
	}
	if len(streamBytes) < 6 {
		return nil, errors.New("zstd: Huffman literals are too short for their jump table")
	}
	streamSizes := []int{
		int(binary.LittleEndian.Uint16(streamBytes[0:2])),
		int(binary.LittleEndian.Uint16(streamBytes[2:4])),
		int(binary.LittleEndian.Uint16(streamBytes[4:6])),
	}
	streamSizes = append(streamSizes, len(streamBytes)-6-streamSizes[0]-streamSizes[1]-streamSizes[2])
	if streamSizes[3] < 0 {
		return nil, errors.New("zstd: Huffman jump table points past the end of the literals")
	}
	symbolsPerStream := (regeneratedSize + 3) / 4
	symbolsInLastStream := regeneratedSize - 3*symbolsPerStream
	if symbolsInLastStream < 0 {
		return nil, fmt.Errorf("zstd: %d literals can't be split into four Huffman streams", regeneratedSize)
	}
	position := 6
	var err error
	for streamIndex := 0; streamIndex < 4; streamIndex++ {
		numberOfSymbols := symbolsPerStream
		if streamIndex == 3 {
			numberOfSymbols = symbolsInLastStream
		}
		streamEnd := position + streamSizes[streamIndex]
		output, err = decodeHuffmanStream(table, streamBytes[position:streamEnd], numberOfSymbols, output)
		if err != nil {
			return nil, err
		}
		position = streamEnd
	}
	return output, nil
}
