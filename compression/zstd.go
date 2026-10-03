package compression

import (
	"encoding/binary"
	"errors"
	"fmt"
)

const (
	zstdMagicNumber           = 0xFD2FB528
	skippableFrameMagicNumber = 0x184D2A50
	skippableFrameMagicMask   = 0xFFFFFFF0
	largestZstdBlockSize      = 128 * 1024
)

const (
	rawBlock          = 0
	repeatedByteBlock = 1
	compressedBlock   = 2
)

type zstdFrameHeader struct {
	hasContentSize bool
	contentSize    uint64
	hasChecksum    bool
}

type zstdFrameDecoder struct {
	input              []byte
	position           int
	output             []byte
	frameStart         int
	repeatedOffsets    [3]int
	huffmanTable       *huffmanTable
	literalLengthTable *fseTable
	offsetTable        *fseTable
	matchLengthTable   *fseTable
}

func DecompressZstd(compressedBytes []byte) ([]byte, error) {
	if len(compressedBytes) == 0 {
		return nil, errors.New("zstd: input is empty, it has no frame")
	}
	output := []byte{}
	position := 0
	for position < len(compressedBytes) {
		if len(compressedBytes)-position < 4 {
			return nil, fmt.Errorf("zstd: %d stray bytes after the last frame", len(compressedBytes)-position)
		}
		magicNumber := binary.LittleEndian.Uint32(compressedBytes[position:])
		if magicNumber&skippableFrameMagicMask == skippableFrameMagicNumber {
			nextPosition, err := skipSkippableFrame(compressedBytes, position)
			if err != nil {
				return nil, err
			}
			position = nextPosition
			continue
		}
		if magicNumber != zstdMagicNumber {
			return nil, fmt.Errorf("zstd: byte %d doesn't start a zstd frame (magic number %#x)", position, magicNumber)
		}
		decoder := &zstdFrameDecoder{
			input:           compressedBytes,
			position:        position + 4,
			output:          output,
			frameStart:      len(output),
			repeatedOffsets: [3]int{1, 4, 8},
		}
		if err := decoder.decodeFrame(); err != nil {
			return nil, err
		}
		output = decoder.output
		position = decoder.position
	}
	return output, nil
}

func skipSkippableFrame(compressedBytes []byte, position int) (int, error) {
	if len(compressedBytes)-position < 8 {
		return 0, errors.New("zstd: skippable frame header is cut off")
	}
	frameSize := int(binary.LittleEndian.Uint32(compressedBytes[position+4:]))
	nextPosition := position + 8 + frameSize
	if frameSize < 0 || nextPosition > len(compressedBytes) {
		return 0, errors.New("zstd: skippable frame runs past the end of the input")
	}
	return nextPosition, nil
}

func (decoder *zstdFrameDecoder) decodeFrame() error {
	header, err := decoder.readFrameHeader()
	if err != nil {
		return err
	}
	if header.hasContentSize && header.contentSize < largestDecompressedSize {
		decoder.growOutput(int(header.contentSize))
	}
	for {
		isLastBlock, err := decoder.decodeBlock()
		if err != nil {
			return err
		}
		if len(decoder.output)-decoder.frameStart > largestDecompressedSize {
			return fmt.Errorf("zstd: frame decompresses to more than the %d byte limit", largestDecompressedSize)
		}
		if isLastBlock {
			break
		}
	}
	frameContent := decoder.output[decoder.frameStart:]
	if header.hasContentSize && uint64(len(frameContent)) != header.contentSize {
		return fmt.Errorf("zstd: frame says it holds %d bytes but decoded to %d", header.contentSize, len(frameContent))
	}
	if header.hasChecksum {
		storedChecksum, err := decoder.readLittleEndian(4)
		if err != nil {
			return err
		}
		computedChecksum := xxhash64(frameContent) & 0xFFFFFFFF
		if uint64(storedChecksum) != computedChecksum {
			return fmt.Errorf("zstd: checksum %#08x doesn't match the decoded content's %#08x", storedChecksum, computedChecksum)
		}
	}
	return nil
}

func (decoder *zstdFrameDecoder) growOutput(extraBytes int) {
	if cap(decoder.output)-len(decoder.output) >= extraBytes {
		return
	}
	grown := make([]byte, len(decoder.output), len(decoder.output)+extraBytes)
	copy(grown, decoder.output)
	decoder.output = grown
}

func (decoder *zstdFrameDecoder) readLittleEndian(numberOfBytes int) (uint64, error) {
	if decoder.position+numberOfBytes > len(decoder.input) {
		return 0, fmt.Errorf("zstd: input ends early at byte %d", decoder.position)
	}
	value := uint64(0)
	for index := 0; index < numberOfBytes; index++ {
		value |= uint64(decoder.input[decoder.position+index]) << (8 * index)
	}
	decoder.position += numberOfBytes
	return value, nil
}

func (decoder *zstdFrameDecoder) readFrameHeader() (zstdFrameHeader, error) {
	descriptor, err := decoder.readLittleEndian(1)
	if err != nil {
		return zstdFrameHeader{}, err
	}
	contentSizeFlag := int(descriptor >> 6)
	isSingleSegment := (descriptor>>5)&1 == 1
	hasChecksum := (descriptor>>2)&1 == 1
	dictionaryIDFlag := int(descriptor & 3)
	if (descriptor>>3)&1 == 1 {
		return zstdFrameHeader{}, errors.New("zstd: frame header has its reserved bit set")
	}
	if !isSingleSegment {
		if _, err := decoder.readLittleEndian(1); err != nil {
			return zstdFrameHeader{}, err
		}
	}
	dictionaryIDSizes := []int{0, 1, 2, 4}
	dictionaryID, err := decoder.readLittleEndian(dictionaryIDSizes[dictionaryIDFlag])
	if err != nil {
		return zstdFrameHeader{}, err
	}
	if dictionaryID != 0 {
		return zstdFrameHeader{}, fmt.Errorf("zstd: frame needs dictionary %d, but dictionaries aren't supported", dictionaryID)
	}
	contentSizeSizes := []int{0, 2, 4, 8}
	contentSizeSize := contentSizeSizes[contentSizeFlag]
	if contentSizeFlag == 0 && isSingleSegment {
		contentSizeSize = 1
	}
	contentSize, err := decoder.readLittleEndian(contentSizeSize)
	if err != nil {
		return zstdFrameHeader{}, err
	}
	if contentSizeSize == 2 {
		contentSize += 256
	}
	return zstdFrameHeader{hasContentSize: contentSizeSize > 0, contentSize: contentSize, hasChecksum: hasChecksum}, nil
}

func (decoder *zstdFrameDecoder) decodeBlock() (bool, error) {
	blockHeader, err := decoder.readLittleEndian(3)
	if err != nil {
		return false, err
	}
	isLastBlock := blockHeader&1 == 1
	blockType := int((blockHeader >> 1) & 3)
	blockSize := int(blockHeader >> 3)
	if blockSize > largestZstdBlockSize {
		return false, fmt.Errorf("zstd: block of %d bytes is larger than the %d byte limit", blockSize, largestZstdBlockSize)
	}
	switch blockType {
	case rawBlock:
		if decoder.position+blockSize > len(decoder.input) {
			return false, errors.New("zstd: raw block runs past the end of the input")
		}
		decoder.output = append(decoder.output, decoder.input[decoder.position:decoder.position+blockSize]...)
		decoder.position += blockSize
	case repeatedByteBlock:
		repeatedByte, err := decoder.readLittleEndian(1)
		if err != nil {
			return false, err
		}
		for index := 0; index < blockSize; index++ {
			decoder.output = append(decoder.output, byte(repeatedByte))
		}
	case compressedBlock:
		if decoder.position+blockSize > len(decoder.input) {
			return false, errors.New("zstd: compressed block runs past the end of the input")
		}
		blockBytes := decoder.input[decoder.position : decoder.position+blockSize]
		decoder.position += blockSize
		if err := decoder.decodeCompressedBlock(blockBytes); err != nil {
			return false, err
		}
	default:
		return false, errors.New("zstd: block uses the reserved block type")
	}
	return isLastBlock, nil
}

func (decoder *zstdFrameDecoder) decodeCompressedBlock(blockBytes []byte) error {
	literals, literalsSectionSize, err := decoder.decodeLiteralsSection(blockBytes)
	if err != nil {
		return err
	}
	sequences, err := decoder.decodeSequencesSection(blockBytes[literalsSectionSize:])
	if err != nil {
		return err
	}
	return decoder.executeSequences(literals, sequences)
}

const (
	rawLiterals          = 0
	repeatedByteLiterals = 1
	compressedLiterals   = 2
	treelessLiterals     = 3
)

func (decoder *zstdFrameDecoder) decodeLiteralsSection(blockBytes []byte) ([]byte, int, error) {
	if len(blockBytes) == 0 {
		return nil, 0, errors.New("zstd: compressed block is empty")
	}
	literalsType := int(blockBytes[0] & 3)
	sizeFormat := int((blockBytes[0] >> 2) & 3)
	if literalsType == rawLiterals || literalsType == repeatedByteLiterals {
		return decodeRawOrRepeatedLiterals(blockBytes, literalsType, sizeFormat)
	}
	return decoder.decodeHuffmanCodedLiterals(blockBytes, literalsType, sizeFormat)
}

func decodeRawOrRepeatedLiterals(blockBytes []byte, literalsType int, sizeFormat int) ([]byte, int, error) {
	headerSize := 1
	regeneratedSize := int(blockBytes[0] >> 3)
	if sizeFormat == 1 || sizeFormat == 3 {
		headerSize = 2 + sizeFormat/2
		if headerSize > len(blockBytes) {
			return nil, 0, errors.New("zstd: literals header is cut off")
		}
		regeneratedSize = int(blockBytes[0]>>4) | int(blockBytes[1])<<4
		if sizeFormat == 3 {
			regeneratedSize |= int(blockBytes[2]) << 12
		}
	}
	if literalsType == rawLiterals {
		if headerSize+regeneratedSize > len(blockBytes) {
			return nil, 0, errors.New("zstd: raw literals run past the end of the block")
		}
		return blockBytes[headerSize : headerSize+regeneratedSize], headerSize + regeneratedSize, nil
	}
	if headerSize+1 > len(blockBytes) {
		return nil, 0, errors.New("zstd: repeated byte literals are missing their byte")
	}
	literals := make([]byte, regeneratedSize)
	for index := 0; index < regeneratedSize; index++ {
		literals[index] = blockBytes[headerSize]
	}
	return literals, headerSize + 1, nil
}

func (decoder *zstdFrameDecoder) decodeHuffmanCodedLiterals(blockBytes []byte, literalsType int, sizeFormat int) ([]byte, int, error) {
	headerSizes := []int{3, 3, 4, 5}
	sizeBitCounts := []int{10, 10, 14, 18}
	headerSize := headerSizes[sizeFormat]
	sizeBits := sizeBitCounts[sizeFormat]
	numberOfStreams := 4
	if sizeFormat == 0 {
		numberOfStreams = 1
	}
	if headerSize > len(blockBytes) {
		return nil, 0, errors.New("zstd: literals header is cut off")
	}
	headerValue := uint64(0)
	for index := 0; index < headerSize; index++ {
		headerValue |= uint64(blockBytes[index]) << (8 * index)
	}
	sizeMask := uint64(1)<<sizeBits - 1
	regeneratedSize := int((headerValue >> 4) & sizeMask)
	compressedSize := int((headerValue >> (4 + sizeBits)) & sizeMask)
	if headerSize+compressedSize > len(blockBytes) {
		return nil, 0, errors.New("zstd: compressed literals run past the end of the block")
	}
	compressedBytes := blockBytes[headerSize : headerSize+compressedSize]
	if literalsType == compressedLiterals {
		table, tableSize, err := readHuffmanTable(compressedBytes)
		if err != nil {
			return nil, 0, err
		}
		decoder.huffmanTable = table
		compressedBytes = compressedBytes[tableSize:]
	} else if decoder.huffmanTable == nil {
		return nil, 0, errors.New("zstd: literals reuse a Huffman table, but there is no earlier one")
	}
	literals, err := decodeHuffmanLiterals(decoder.huffmanTable, compressedBytes, regeneratedSize, numberOfStreams)
	if err != nil {
		return nil, 0, err
	}
	return literals, headerSize + compressedSize, nil
}
