package compression

import (
	"encoding/binary"
	"errors"
	"fmt"
)

const largestDecompressedSize = 1 << 31

const (
	snappyLiteral           = 0
	snappyCopyWithOneByte   = 1
	snappyCopyWithTwoBytes  = 2
	snappyCopyWithFourBytes = 3
)

type snappyDecoder struct {
	input    []byte
	position int
	output   []byte
}

func DecompressSnappy(compressedBytes []byte) ([]byte, error) {
	uncompressedLength, headerLength := binary.Uvarint(compressedBytes)
	if headerLength <= 0 {
		return nil, errors.New("snappy: can't read the uncompressed length at the start")
	}
	if uncompressedLength > largestDecompressedSize {
		return nil, fmt.Errorf("snappy: says it holds %d bytes, which is more than the %d byte limit", uncompressedLength, largestDecompressedSize)
	}
	decoder := &snappyDecoder{
		input:    compressedBytes,
		position: headerLength,
		output:   make([]byte, 0, uncompressedLength),
	}
	for decoder.position < len(decoder.input) {
		if err := decoder.decodeElement(); err != nil {
			return nil, err
		}
		if len(decoder.output) > int(uncompressedLength) {
			return nil, fmt.Errorf("snappy: produced more than the %d bytes it promised", uncompressedLength)
		}
	}
	if len(decoder.output) != int(uncompressedLength) {
		return nil, fmt.Errorf("snappy: produced %d bytes but promised %d", len(decoder.output), uncompressedLength)
	}
	return decoder.output, nil
}

func (decoder *snappyDecoder) decodeElement() error {
	tag := decoder.input[decoder.position]
	decoder.position++
	elementType := tag & 3
	switch elementType {
	case snappyLiteral:
		return decoder.decodeLiteral(tag)
	case snappyCopyWithOneByte:
		lowOffsetByte, err := decoder.readLittleEndian(1)
		if err != nil {
			return err
		}
		length := 4 + int((tag>>2)&7)
		offset := int(tag>>5)<<8 | lowOffsetByte
		return decoder.copyFromOutput(offset, length)
	case snappyCopyWithTwoBytes:
		offset, err := decoder.readLittleEndian(2)
		if err != nil {
			return err
		}
		return decoder.copyFromOutput(offset, 1+int(tag>>2))
	default:
		offset, err := decoder.readLittleEndian(4)
		if err != nil {
			return err
		}
		return decoder.copyFromOutput(offset, 1+int(tag>>2))
	}
}

func (decoder *snappyDecoder) decodeLiteral(tag byte) error {
	lengthMinusOne := int(tag >> 2)
	if lengthMinusOne >= 60 {
		numberOfLengthBytes := lengthMinusOne - 59
		value, err := decoder.readLittleEndian(numberOfLengthBytes)
		if err != nil {
			return err
		}
		lengthMinusOne = value
	}
	length := lengthMinusOne + 1
	if length <= 0 || length > len(decoder.input)-decoder.position {
		return fmt.Errorf("snappy: literal of %d bytes at byte %d runs past the end of the input", length, decoder.position)
	}
	decoder.output = append(decoder.output, decoder.input[decoder.position:decoder.position+length]...)
	decoder.position += length
	return nil
}

func (decoder *snappyDecoder) readLittleEndian(numberOfBytes int) (int, error) {
	if decoder.position+numberOfBytes > len(decoder.input) {
		return 0, fmt.Errorf("snappy: input ends in the middle of an element at byte %d", decoder.position)
	}
	value := 0
	for index := 0; index < numberOfBytes; index++ {
		value |= int(decoder.input[decoder.position+index]) << (8 * index)
	}
	decoder.position += numberOfBytes
	return value, nil
}

func (decoder *snappyDecoder) copyFromOutput(offset int, length int) error {
	if offset <= 0 || offset > len(decoder.output) {
		return fmt.Errorf("snappy: copy reaches back %d bytes but only %d bytes are decoded so far", offset, len(decoder.output))
	}
	start := len(decoder.output) - offset
	for index := 0; index < length; index++ {
		decoder.output = append(decoder.output, decoder.output[start+index])
	}
	return nil
}
