package compression

import (
	"errors"
	"fmt"
)

type zstdSequence struct {
	literalLength int
	matchLength   int
	offset        int
}

const (
	predefinedTableMode = 0
	singleSymbolMode    = 1
	fseCompressedMode   = 2
	repeatTableMode     = 3
)

const (
	largestLiteralLengthCode = 35
	largestMatchLengthCode   = 52
	largestOffsetCode        = 31
)

var literalLengthBaselines = []int{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096,
	8192, 16384, 32768, 65536,
}

var literalLengthExtraBits = []int{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12,
	13, 14, 15, 16,
}

var matchLengthBaselines = []int{
	3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
	19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
	35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051,
	4099, 8195, 16387, 32771, 65539,
}

var matchLengthExtraBits = []int{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11,
	12, 13, 14, 15, 16,
}

func readNumberOfSequences(data []byte) (int, int, error) {
	if len(data) == 0 {
		return 0, 0, errors.New("zstd: sequences section is missing")
	}
	firstByte := int(data[0])
	if firstByte < 128 {
		return firstByte, 1, nil
	}
	if firstByte < 255 {
		if len(data) < 2 {
			return 0, 0, errors.New("zstd: number of sequences is cut off")
		}
		return (firstByte-128)<<8 + int(data[1]), 2, nil
	}
	if len(data) < 3 {
		return 0, 0, errors.New("zstd: number of sequences is cut off")
	}
	return int(data[1]) + int(data[2])<<8 + 0x7F00, 3, nil
}

func (decoder *zstdFrameDecoder) decodeSequencesSection(data []byte) ([]zstdSequence, error) {
	numberOfSequences, position, err := readNumberOfSequences(data)
	if err != nil {
		return nil, err
	}
	if numberOfSequences == 0 {
		return nil, nil
	}
	if position >= len(data) {
		return nil, errors.New("zstd: sequence table modes are missing")
	}
	modes := int(data[position])
	position++
	if modes&3 != 0 {
		return nil, errors.New("zstd: sequence table modes have reserved bits set")
	}
	decoder.literalLengthTable, position, err = selectSequenceTable(modes>>6, data, position, decoder.literalLengthTable, predefinedLiteralLengthTable, largestLiteralLengthCode, 9)
	if err != nil {
		return nil, fmt.Errorf("%w (literal lengths)", err)
	}
	decoder.offsetTable, position, err = selectSequenceTable((modes>>4)&3, data, position, decoder.offsetTable, predefinedOffsetTable, largestOffsetCode, 8)
	if err != nil {
		return nil, fmt.Errorf("%w (offsets)", err)
	}
	decoder.matchLengthTable, position, err = selectSequenceTable((modes>>2)&3, data, position, decoder.matchLengthTable, predefinedMatchLengthTable, largestMatchLengthCode, 9)
	if err != nil {
		return nil, fmt.Errorf("%w (match lengths)", err)
	}
	reader, err := newBackwardBitReader(data[position:])
	if err != nil {
		return nil, err
	}
	return decoder.decodeSequences(reader, numberOfSequences)
}

func selectSequenceTable(mode int, data []byte, position int, previousTable *fseTable, predefinedTable *fseTable, largestSymbol int, largestAccuracyLog int) (*fseTable, int, error) {
	switch mode {
	case predefinedTableMode:
		return predefinedTable, position, nil
	case singleSymbolMode:
		if position >= len(data) {
			return nil, 0, errors.New("zstd: single symbol table is missing its symbol")
		}
		symbol := int(data[position])
		if symbol > largestSymbol {
			return nil, 0, fmt.Errorf("zstd: single symbol table uses code %d, above the limit of %d", symbol, largestSymbol)
		}
		return buildSingleSymbolFSETable(symbol), position + 1, nil
	case fseCompressedMode:
		counts, accuracyLog, bytesUsed, err := readFSENormalizedCounts(data[position:], largestSymbol, largestAccuracyLog)
		if err != nil {
			return nil, 0, err
		}
		table, err := buildFSETable(counts, accuracyLog)
		if err != nil {
			return nil, 0, err
		}
		return table, position + bytesUsed, nil
	default:
		if previousTable == nil {
			return nil, 0, errors.New("zstd: sequences reuse a table, but there is no earlier one")
		}
		return previousTable, position, nil
	}
}

func (decoder *zstdFrameDecoder) decodeSequences(reader *backwardBitReader, numberOfSequences int) ([]zstdSequence, error) {
	literalLengthState := reader.readBits(decoder.literalLengthTable.accuracyLog)
	offsetState := reader.readBits(decoder.offsetTable.accuracyLog)
	matchLengthState := reader.readBits(decoder.matchLengthTable.accuracyLog)
	sequences := make([]zstdSequence, numberOfSequences)
	for sequenceIndex := 0; sequenceIndex < numberOfSequences; sequenceIndex++ {
		literalLengthCode := decoder.literalLengthTable.symbolAt(literalLengthState)
		offsetCode := decoder.offsetTable.symbolAt(offsetState)
		matchLengthCode := decoder.matchLengthTable.symbolAt(matchLengthState)

		offsetValue := 1<<offsetCode + reader.readBits(offsetCode)
		matchLength := matchLengthBaselines[matchLengthCode] + reader.readBits(matchLengthExtraBits[matchLengthCode])
		literalLength := literalLengthBaselines[literalLengthCode] + reader.readBits(literalLengthExtraBits[literalLengthCode])
		offset := decoder.resolveOffset(offsetValue, literalLength)
		sequences[sequenceIndex] = zstdSequence{literalLength: literalLength, matchLength: matchLength, offset: offset}

		if sequenceIndex < numberOfSequences-1 {
			literalLengthState = decoder.literalLengthTable.nextState(literalLengthState, reader)
			matchLengthState = decoder.matchLengthTable.nextState(matchLengthState, reader)
			offsetState = decoder.offsetTable.nextState(offsetState, reader)
		}
		if reader.ranPastStart() {
			return nil, errors.New("zstd: sequence bitstream ran out early")
		}
	}
	if !reader.finished() {
		return nil, fmt.Errorf("zstd: sequence bitstream has %d bits left over", reader.bitsRemaining)
	}
	return sequences, nil
}

func (decoder *zstdFrameDecoder) resolveOffset(offsetValue int, literalLength int) int {
	if offsetValue > 3 {
		offset := offsetValue - 3
		decoder.repeatedOffsets[2] = decoder.repeatedOffsets[1]
		decoder.repeatedOffsets[1] = decoder.repeatedOffsets[0]
		decoder.repeatedOffsets[0] = offset
		return offset
	}
	repeatIndex := offsetValue - 1
	if literalLength == 0 {
		repeatIndex++
	}
	if repeatIndex == 0 {
		return decoder.repeatedOffsets[0]
	}
	var offset int
	if repeatIndex == 3 {
		offset = decoder.repeatedOffsets[0] - 1
	} else {
		offset = decoder.repeatedOffsets[repeatIndex]
	}
	if repeatIndex != 1 {
		decoder.repeatedOffsets[2] = decoder.repeatedOffsets[1]
	}
	decoder.repeatedOffsets[1] = decoder.repeatedOffsets[0]
	decoder.repeatedOffsets[0] = offset
	return offset
}

func (decoder *zstdFrameDecoder) executeSequences(literals []byte, sequences []zstdSequence) error {
	literalPosition := 0
	for sequenceIndex := 0; sequenceIndex < len(sequences); sequenceIndex++ {
		sequence := sequences[sequenceIndex]
		if literalPosition+sequence.literalLength > len(literals) {
			return errors.New("zstd: sequence asks for more literals than the block has")
		}
		decoder.output = append(decoder.output, literals[literalPosition:literalPosition+sequence.literalLength]...)
		literalPosition += sequence.literalLength
		if err := decoder.copyMatch(sequence.offset, sequence.matchLength); err != nil {
			return err
		}
	}
	decoder.output = append(decoder.output, literals[literalPosition:]...)
	return nil
}

func (decoder *zstdFrameDecoder) copyMatch(offset int, matchLength int) error {
	bytesInFrame := len(decoder.output) - decoder.frameStart
	if offset <= 0 || offset > bytesInFrame {
		return fmt.Errorf("zstd: match reaches back %d bytes but the frame has only %d so far", offset, bytesInFrame)
	}
	start := len(decoder.output) - offset
	if offset >= matchLength {
		decoder.output = append(decoder.output, decoder.output[start:start+matchLength]...)
		return nil
	}
	for index := 0; index < matchLength; index++ {
		decoder.output = append(decoder.output, decoder.output[start+index])
	}
	return nil
}
