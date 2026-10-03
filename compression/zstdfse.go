package compression

import (
	"errors"
	"fmt"
	"math/bits"
)

type fseTableEntry struct {
	symbol       int
	numberOfBits int
	baseline     int
}

type fseTable struct {
	accuracyLog int
	entries     []fseTableEntry
}

func (table *fseTable) symbolAt(state int) int {
	return table.entries[state].symbol
}

func (table *fseTable) nextState(state int, reader *backwardBitReader) int {
	entry := table.entries[state]
	return entry.baseline + reader.readBits(entry.numberOfBits)
}

func readFSENormalizedCounts(data []byte, largestSymbol int, largestAccuracyLog int) ([]int, int, int, error) {
	reader := &forwardBitReader{data: data}
	accuracyLog := reader.readBits(4) + 5
	if accuracyLog > largestAccuracyLog {
		return nil, 0, 0, fmt.Errorf("zstd: FSE table accuracy log %d is above the limit of %d", accuracyLog, largestAccuracyLog)
	}
	remaining := (1 << accuracyLog) + 1
	threshold := 1 << accuracyLog
	numberOfBits := accuracyLog + 1
	var counts []int
	previousCountWasZero := false
	for remaining > 1 && len(counts) <= largestSymbol {
		if previousCountWasZero {
			numberOfZeros := 0
			repeatFlag := reader.readBits(2)
			for repeatFlag == 3 {
				numberOfZeros += 3
				repeatFlag = reader.readBits(2)
			}
			numberOfZeros += repeatFlag
			for zeroIndex := 0; zeroIndex < numberOfZeros; zeroIndex++ {
				counts = append(counts, 0)
			}
			if len(counts) > largestSymbol {
				break
			}
		}
		largestShortValue := (2*threshold - 1) - remaining
		value := reader.peekBits(numberOfBits - 1)
		if value < largestShortValue {
			reader.skipBits(numberOfBits - 1)
		} else {
			value = reader.peekBits(numberOfBits)
			if value >= threshold {
				value -= largestShortValue
			}
			reader.skipBits(numberOfBits)
		}
		count := value - 1
		if count < 0 {
			remaining += count
		} else {
			remaining -= count
		}
		if remaining < 1 {
			return nil, 0, 0, errors.New("zstd: FSE table counts add up to more than the table size")
		}
		counts = append(counts, count)
		previousCountWasZero = count == 0
		for remaining < threshold {
			numberOfBits--
			threshold >>= 1
		}
	}
	if remaining != 1 {
		return nil, 0, 0, errors.New("zstd: FSE table counts don't add up to the table size")
	}
	if reader.ranPastEnd() {
		return nil, 0, 0, errors.New("zstd: FSE table description runs past the end of its data")
	}
	return counts, accuracyLog, reader.bytesUsed(), nil
}

func buildFSETable(counts []int, accuracyLog int) (*fseTable, error) {
	tableSize := 1 << accuracyLog
	entries := make([]fseTableEntry, tableSize)
	nextStateForSymbol := make([]int, len(counts))
	highestFreePosition := tableSize - 1
	for symbol := 0; symbol < len(counts); symbol++ {
		if counts[symbol] == -1 {
			if highestFreePosition < 0 {
				return nil, errors.New("zstd: FSE table has too many low probability symbols")
			}
			entries[highestFreePosition].symbol = symbol
			highestFreePosition--
			nextStateForSymbol[symbol] = 1
		} else {
			nextStateForSymbol[symbol] = counts[symbol]
		}
	}

	step := (tableSize >> 1) + (tableSize >> 3) + 3
	mask := tableSize - 1
	position := 0
	for symbol := 0; symbol < len(counts); symbol++ {
		for occurrence := 0; occurrence < counts[symbol]; occurrence++ {
			entries[position].symbol = symbol
			position = (position + step) & mask
			for position > highestFreePosition {
				position = (position + step) & mask
			}
		}
	}
	if position != 0 {
		return nil, errors.New("zstd: FSE table counts don't fill the table evenly")
	}

	for state := 0; state < tableSize; state++ {
		symbol := entries[state].symbol
		nextState := nextStateForSymbol[symbol]
		nextStateForSymbol[symbol]++
		numberOfBits := accuracyLog - (bits.Len(uint(nextState)) - 1)
		entries[state].numberOfBits = numberOfBits
		entries[state].baseline = (nextState << numberOfBits) - tableSize
	}
	return &fseTable{accuracyLog: accuracyLog, entries: entries}, nil
}

func buildSingleSymbolFSETable(symbol int) *fseTable {
	return &fseTable{accuracyLog: 0, entries: []fseTableEntry{{symbol: symbol}}}
}

func mustBuildFSETable(counts []int, accuracyLog int) *fseTable {
	table, err := buildFSETable(counts, accuracyLog)
	if err != nil {
		panic(err)
	}
	return table
}

var predefinedLiteralLengthTable = mustBuildFSETable([]int{
	4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
	-1, -1, -1, -1,
}, 6)

var predefinedMatchLengthTable = mustBuildFSETable([]int{
	1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
	-1, -1, -1, -1, -1,
}, 6)

var predefinedOffsetTable = mustBuildFSETable([]int{
	1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
}, 5)
