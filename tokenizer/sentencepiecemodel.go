package tokenizer

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const (
	wireTypeVarint          = 0
	wireTypeFixed64         = 1
	wireTypeLengthDelimited = 2
	wireTypeFixed32         = 5
)

const (
	sentencePieceUnigramModelType = 1
	sentencePieceBPEModelType     = 2
	sentencePieceWordModelType    = 3
	sentencePieceCharModelType    = 4
)

type protobufReader struct {
	data     []byte
	position int
}

func (reader *protobufReader) finished() bool {
	return reader.position >= len(reader.data)
}

func (reader *protobufReader) readVarint() (uint64, error) {
	var value uint64
	for shift := uint(0); shift < 64; shift += 7 {
		if reader.position >= len(reader.data) {
			return 0, fmt.Errorf("protobuf data ends in the middle of a number at byte %d", reader.position)
		}
		nextByte := reader.data[reader.position]
		reader.position++
		value |= uint64(nextByte&0x7F) << shift
		if nextByte < 0x80 {
			return value, nil
		}
	}
	return 0, fmt.Errorf("protobuf number at byte %d is longer than 10 bytes", reader.position)
}

func (reader *protobufReader) readFieldHeader() (int, int, error) {
	header, err := reader.readVarint()
	if err != nil {
		return 0, 0, err
	}
	return int(header >> 3), int(header & 7), nil
}

func (reader *protobufReader) readBytes() ([]byte, error) {
	length, err := reader.readVarint()
	if err != nil {
		return nil, err
	}
	if length > uint64(len(reader.data)-reader.position) {
		return nil, fmt.Errorf("protobuf field at byte %d says it is %d bytes long but only %d bytes are left", reader.position, length, len(reader.data)-reader.position)
	}
	start := reader.position
	reader.position += int(length)
	return reader.data[start:reader.position], nil
}

func (reader *protobufReader) readFixed32() (uint32, error) {
	if len(reader.data)-reader.position < 4 {
		return 0, fmt.Errorf("protobuf data ends in the middle of a 4-byte value at byte %d", reader.position)
	}
	value := binary.LittleEndian.Uint32(reader.data[reader.position:])
	reader.position += 4
	return value, nil
}

func (reader *protobufReader) skipField(wireType int) error {
	switch wireType {
	case wireTypeVarint:
		_, err := reader.readVarint()
		return err
	case wireTypeFixed64:
		if len(reader.data)-reader.position < 8 {
			return fmt.Errorf("protobuf data ends in the middle of an 8-byte value at byte %d", reader.position)
		}
		reader.position += 8
		return nil
	case wireTypeLengthDelimited:
		_, err := reader.readBytes()
		return err
	case wireTypeFixed32:
		_, err := reader.readFixed32()
		return err
	}
	return fmt.Errorf("protobuf wire type %d at byte %d is not supported", wireType, reader.position)
}

func expectWireType(fieldName string, gotWireType int, wantWireType int) error {
	if gotWireType != wantWireType {
		return fmt.Errorf("field %s has protobuf wire type %d, expected %d", fieldName, gotWireType, wantWireType)
	}
	return nil
}

type sentencePieceTrainerSpec struct {
	modelType               int
	byteFallback            bool
	treatWhitespaceAsSuffix bool
}

type sentencePieceNormalizerSpec struct {
	name                   string
	precompiledCharsmap    []byte
	addDummyPrefix         bool
	removeExtraWhitespaces bool
	escapeWhitespaces      bool
}

type sentencePieceModelProto struct {
	pieces     []sentencePieceEntry
	trainer    sentencePieceTrainerSpec
	normalizer sentencePieceNormalizerSpec
}

func readVarintField(reader *protobufReader, fieldName string, wireType int) (uint64, error) {
	if err := expectWireType(fieldName, wireType, wireTypeVarint); err != nil {
		return 0, err
	}
	return reader.readVarint()
}

func readBytesField(reader *protobufReader, fieldName string, wireType int) ([]byte, error) {
	if err := expectWireType(fieldName, wireType, wireTypeLengthDelimited); err != nil {
		return nil, err
	}
	return reader.readBytes()
}

func decodeSentencePieceEntry(data []byte) (sentencePieceEntry, error) {
	entry := sentencePieceEntry{kind: normalPiece}
	reader := &protobufReader{data: data}
	for !reader.finished() {
		fieldNumber, wireType, err := reader.readFieldHeader()
		if err != nil {
			return entry, err
		}
		switch fieldNumber {
		case 1:
			piece, err := readBytesField(reader, "piece", wireType)
			if err != nil {
				return entry, err
			}
			entry.piece = string(piece)
		case 2:
			if err := expectWireType("score", wireType, wireTypeFixed32); err != nil {
				return entry, err
			}
			bits, err := reader.readFixed32()
			if err != nil {
				return entry, err
			}
			entry.score = math.Float32frombits(bits)
		case 3:
			kind, err := readVarintField(reader, "type", wireType)
			if err != nil {
				return entry, err
			}
			entry.kind = pieceType(kind)
		default:
			if err := reader.skipField(wireType); err != nil {
				return entry, err
			}
		}
	}
	return entry, nil
}

func decodeTrainerSpec(data []byte) (sentencePieceTrainerSpec, error) {
	spec := sentencePieceTrainerSpec{modelType: sentencePieceUnigramModelType}
	reader := &protobufReader{data: data}
	for !reader.finished() {
		fieldNumber, wireType, err := reader.readFieldHeader()
		if err != nil {
			return spec, err
		}
		switch fieldNumber {
		case 3:
			modelType, err := readVarintField(reader, "model_type", wireType)
			if err != nil {
				return spec, err
			}
			spec.modelType = int(modelType)
		case 24:
			value, err := readVarintField(reader, "treat_whitespace_as_suffix", wireType)
			if err != nil {
				return spec, err
			}
			spec.treatWhitespaceAsSuffix = value != 0
		case 35:
			value, err := readVarintField(reader, "byte_fallback", wireType)
			if err != nil {
				return spec, err
			}
			spec.byteFallback = value != 0
		default:
			if err := reader.skipField(wireType); err != nil {
				return spec, err
			}
		}
	}
	return spec, nil
}

func defaultNormalizerSpec() sentencePieceNormalizerSpec {
	return sentencePieceNormalizerSpec{addDummyPrefix: true, removeExtraWhitespaces: true, escapeWhitespaces: true}
}

func decodeNormalizerSpec(data []byte) (sentencePieceNormalizerSpec, error) {
	spec := defaultNormalizerSpec()
	reader := &protobufReader{data: data}
	for !reader.finished() {
		fieldNumber, wireType, err := reader.readFieldHeader()
		if err != nil {
			return spec, err
		}
		switch fieldNumber {
		case 1:
			name, err := readBytesField(reader, "name", wireType)
			if err != nil {
				return spec, err
			}
			spec.name = string(name)
		case 2:
			charsmap, err := readBytesField(reader, "precompiled_charsmap", wireType)
			if err != nil {
				return spec, err
			}
			spec.precompiledCharsmap = charsmap
		case 3:
			value, err := readVarintField(reader, "add_dummy_prefix", wireType)
			if err != nil {
				return spec, err
			}
			spec.addDummyPrefix = value != 0
		case 4:
			value, err := readVarintField(reader, "remove_extra_whitespaces", wireType)
			if err != nil {
				return spec, err
			}
			spec.removeExtraWhitespaces = value != 0
		case 5:
			value, err := readVarintField(reader, "escape_whitespaces", wireType)
			if err != nil {
				return spec, err
			}
			spec.escapeWhitespaces = value != 0
		default:
			if err := reader.skipField(wireType); err != nil {
				return spec, err
			}
		}
	}
	return spec, nil
}

func decodeSentencePieceModelProto(data []byte) (sentencePieceModelProto, error) {
	model := sentencePieceModelProto{
		trainer:    sentencePieceTrainerSpec{modelType: sentencePieceUnigramModelType},
		normalizer: defaultNormalizerSpec(),
	}
	reader := &protobufReader{data: data}
	for !reader.finished() {
		fieldNumber, wireType, err := reader.readFieldHeader()
		if err != nil {
			return model, err
		}
		switch fieldNumber {
		case 1:
			message, err := readBytesField(reader, "pieces", wireType)
			if err != nil {
				return model, err
			}
			entry, err := decodeSentencePieceEntry(message)
			if err != nil {
				return model, fmt.Errorf("piece %d: %w", len(model.pieces), err)
			}
			model.pieces = append(model.pieces, entry)
		case 2:
			message, err := readBytesField(reader, "trainer_spec", wireType)
			if err != nil {
				return model, err
			}
			model.trainer, err = decodeTrainerSpec(message)
			if err != nil {
				return model, fmt.Errorf("trainer_spec: %w", err)
			}
		case 3:
			message, err := readBytesField(reader, "normalizer_spec", wireType)
			if err != nil {
				return model, err
			}
			model.normalizer, err = decodeNormalizerSpec(message)
			if err != nil {
				return model, fmt.Errorf("normalizer_spec: %w", err)
			}
		default:
			if err := reader.skipField(wireType); err != nil {
				return model, err
			}
		}
	}
	return model, nil
}

func checkSentencePieceModelIsSupported(model sentencePieceModelProto) error {
	switch model.trainer.modelType {
	case sentencePieceBPEModelType, sentencePieceUnigramModelType:
	case sentencePieceWordModelType:
		return fmt.Errorf("model type WORD is not supported, only BPE and Unigram are")
	case sentencePieceCharModelType:
		return fmt.Errorf("model type CHAR is not supported, only BPE and Unigram are")
	default:
		return fmt.Errorf("model type %d is unknown", model.trainer.modelType)
	}
	if model.trainer.treatWhitespaceAsSuffix {
		return fmt.Errorf("treat_whitespace_as_suffix is not supported")
	}
	if len(model.normalizer.precompiledCharsmap) > 0 {
		return fmt.Errorf("normalization rule %q (a precompiled character map, as in T5) is not supported, only the identity rule used by Llama 2 and Mistral is", model.normalizer.name)
	}
	if len(model.pieces) == 0 {
		return fmt.Errorf("the model has no pieces")
	}
	for id, entry := range model.pieces {
		if entry.kind < normalPiece || entry.kind > bytePiece {
			return fmt.Errorf("piece %d %q has unknown type %d", id, entry.piece, entry.kind)
		}
	}
	return nil
}

func sentencePieceFromModelProto(data []byte) (*Tokenizer, error) {
	model, err := decodeSentencePieceModelProto(data)
	if err != nil {
		return nil, err
	}
	if err := checkSentencePieceModelIsSupported(model); err != nil {
		return nil, err
	}
	kind := SentencePieceBPE
	if model.trainer.modelType == sentencePieceUnigramModelType {
		kind = SentencePieceUnigram
	}
	settings := sentencePieceModel{
		mergeOrder:                   mergeHighestScoreFirst,
		addDummyPrefix:               model.normalizer.addDummyPrefix,
		replaceSpacesWithSpaceSymbol: model.normalizer.escapeWhitespaces,
		removeExtraWhitespace:        model.normalizer.removeExtraWhitespaces,
		byteFallback:                 model.trainer.byteFallback,
	}
	return newSentencePieceTokenizer(kind, model.pieces, settings), nil
}

func LoadSentencePiece(path string) (*Tokenizer, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	tokenizer, err := sentencePieceFromModelProto(contents)
	if err != nil {
		return nil, fmt.Errorf("%s: not a supported SentencePiece tokenizer.model: %w", path, err)
	}
	return tokenizer, nil
}
