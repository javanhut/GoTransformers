package parquet

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/javanhut/GoTransformers/compression"
)

type leafColumn struct {
	name                   string
	physicalType           int
	typeLength             int
	maximumDefinitionLevel int
	maximumRepetitionLevel int
}

type decodedColumn struct {
	values    valueList
	isPresent []bool
}

type columnChunkDecoder struct {
	column     leafColumn
	codec      int
	dictionary *valueList
	decoded    decodedColumn
	levelsRead int
}

func decodeColumnChunk(chunkBytes []byte, column leafColumn, chunk columnChunk) (decodedColumn, error) {
	decoder := &columnChunkDecoder{column: column, codec: chunk.codec}
	position := 0
	for int64(decoder.levelsRead) < chunk.numberOfValues {
		if position >= len(chunkBytes) {
			return decodedColumn{}, fmt.Errorf("column chunk ends after %d of its %d values", decoder.levelsRead, chunk.numberOfValues)
		}
		header, headerSize, err := readPageHeader(chunkBytes[position:])
		if err != nil {
			return decodedColumn{}, err
		}
		position += headerSize
		if header.compressedSize > len(chunkBytes)-position {
			return decodedColumn{}, fmt.Errorf("page of %d bytes runs past the end of the column chunk", header.compressedSize)
		}
		pageBytes := chunkBytes[position : position+header.compressedSize]
		position += header.compressedSize
		if err := decoder.decodePage(header, pageBytes); err != nil {
			return decodedColumn{}, err
		}
	}
	return decoder.decoded, nil
}

func (decoder *columnChunkDecoder) decodePage(header pageHeader, pageBytes []byte) error {
	switch header.pageType {
	case dictionaryPage:
		return decoder.decodeDictionaryPage(header, pageBytes)
	case dataPage:
		return decoder.decodeDataPage(header, pageBytes)
	case dataPageV2:
		return decoder.decodeDataPageV2(header, pageBytes)
	}
	return nil
}

func decompressPage(codec int, compressedBytes []byte, uncompressedSize int) ([]byte, error) {
	var decompressed []byte
	var err error
	switch codec {
	case uncompressedCodec:
		decompressed = compressedBytes
	case snappyCodec:
		decompressed, err = compression.DecompressSnappy(compressedBytes)
	case gzipCodec:
		decompressed, err = decompressGzip(compressedBytes)
	case zstdCodec:
		decompressed, err = compression.DecompressZstd(compressedBytes)
	default:
		return nil, fmt.Errorf("compression %s isn't supported, only UNCOMPRESSED, SNAPPY, GZIP and ZSTD are", nameFromList(codecNames, codec))
	}
	if err != nil {
		return nil, err
	}
	if len(decompressed) != uncompressedSize {
		return nil, fmt.Errorf("page decompressed to %d bytes but its header says %d", len(decompressed), uncompressedSize)
	}
	return decompressed, nil
}

func decompressGzip(compressedBytes []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(compressedBytes))
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	return io.ReadAll(reader)
}

func (decoder *columnChunkDecoder) decodeDictionaryPage(header pageHeader, pageBytes []byte) error {
	if header.encoding != plainEncoding && header.encoding != plainDictionaryEncoding {
		return fmt.Errorf("dictionary page uses encoding %s, only PLAIN is supported", nameFromList(encodingNames, header.encoding))
	}
	if header.numberOfValues < 0 {
		return fmt.Errorf("dictionary page says it has %d values", header.numberOfValues)
	}
	pageData, err := decompressPage(decoder.codec, pageBytes, header.uncompressedSize)
	if err != nil {
		return err
	}
	dictionary := &valueList{}
	if err := decodePlainValues(pageData, decoder.column.physicalType, decoder.column.typeLength, header.numberOfValues, dictionary); err != nil {
		return fmt.Errorf("dictionary page: %w", err)
	}
	decoder.dictionary = dictionary
	return nil
}

func (decoder *columnChunkDecoder) decodeDataPage(header pageHeader, pageBytes []byte) error {
	if header.numberOfValues < 0 {
		return fmt.Errorf("data page says it has %d values", header.numberOfValues)
	}
	pageData, err := decompressPage(decoder.codec, pageBytes, header.uncompressedSize)
	if err != nil {
		return err
	}
	var definitionLevels []int
	position := 0
	if decoder.column.maximumDefinitionLevel > 0 {
		if header.definitionLevelEncoding != rleEncoding {
			return fmt.Errorf("definition levels use encoding %s, only RLE is supported", nameFromList(encodingNames, header.definitionLevelEncoding))
		}
		if len(pageData) < 4 {
			return fmt.Errorf("data page is too short for its definition levels")
		}
		levelsLength := int(binary.LittleEndian.Uint32(pageData))
		if levelsLength < 0 || levelsLength > len(pageData)-4 {
			return fmt.Errorf("definition levels say they are %d bytes, which runs past the end of the page", levelsLength)
		}
		bitWidth := bitWidthFor(decoder.column.maximumDefinitionLevel)
		definitionLevels, err = decodeRLEBitPackedHybrid(pageData[4:4+levelsLength], bitWidth, header.numberOfValues)
		if err != nil {
			return fmt.Errorf("definition levels: %w", err)
		}
		position = 4 + levelsLength
	}
	return decoder.decodeValuesAndLevels(header, pageData[position:], definitionLevels)
}

func (decoder *columnChunkDecoder) decodeDataPageV2(header pageHeader, pageBytes []byte) error {
	levelsLength := header.repetitionLevelsByteLength + header.definitionLevelsByteLength
	if levelsLength > len(pageBytes) {
		return fmt.Errorf("data page v2 levels say they are %d bytes, but the page is %d", levelsLength, len(pageBytes))
	}
	valueBytes := pageBytes[levelsLength:]
	if header.valuesAreCompressed {
		var err error
		valueBytes, err = decompressPage(decoder.codec, valueBytes, header.uncompressedSize-levelsLength)
		if err != nil {
			return err
		}
	}
	var definitionLevels []int
	if decoder.column.maximumDefinitionLevel > 0 {
		levelBytes := pageBytes[header.repetitionLevelsByteLength:levelsLength]
		bitWidth := bitWidthFor(decoder.column.maximumDefinitionLevel)
		var err error
		definitionLevels, err = decodeRLEBitPackedHybrid(levelBytes, bitWidth, header.numberOfValues)
		if err != nil {
			return fmt.Errorf("definition levels: %w", err)
		}
	}
	return decoder.decodeValuesAndLevels(header, valueBytes, definitionLevels)
}

func (decoder *columnChunkDecoder) decodeValuesAndLevels(header pageHeader, valueBytes []byte, definitionLevels []int) error {
	numberOfPresentValues := header.numberOfValues
	if definitionLevels != nil {
		numberOfPresentValues = 0
		for index := 0; index < len(definitionLevels); index++ {
			isPresent := definitionLevels[index] == decoder.column.maximumDefinitionLevel
			if isPresent {
				numberOfPresentValues++
			}
			decoder.decoded.isPresent = append(decoder.decoded.isPresent, isPresent)
		}
	}
	valuesBefore := decoder.decoded.values.length()
	if err := decoder.decodeValues(header.encoding, valueBytes, numberOfPresentValues); err != nil {
		return err
	}
	if decoder.decoded.values.length()-valuesBefore != numberOfPresentValues {
		return fmt.Errorf("data page decoded %d values but needed %d", decoder.decoded.values.length()-valuesBefore, numberOfPresentValues)
	}
	decoder.levelsRead += header.numberOfValues
	return nil
}

func (decoder *columnChunkDecoder) decodeValues(encoding int, valueBytes []byte, count int) error {
	switch encoding {
	case plainEncoding:
		return decodePlainValues(valueBytes, decoder.column.physicalType, decoder.column.typeLength, count, &decoder.decoded.values)
	case plainDictionaryEncoding, rleDictionaryEncoding:
		return decoder.decodeDictionaryIndices(valueBytes, count)
	case rleEncoding:
		if decoder.column.physicalType == booleanType {
			return decoder.decodeRLEBooleans(valueBytes, count)
		}
	}
	return fmt.Errorf("data page uses encoding %s, only PLAIN, PLAIN_DICTIONARY and RLE_DICTIONARY are supported", nameFromList(encodingNames, encoding))
}

func (decoder *columnChunkDecoder) decodeDictionaryIndices(valueBytes []byte, count int) error {
	if count == 0 {
		return nil
	}
	if decoder.dictionary == nil {
		return fmt.Errorf("data page uses a dictionary, but the column chunk has no dictionary page before it")
	}
	if len(valueBytes) == 0 {
		return fmt.Errorf("dictionary encoded data page is empty")
	}
	bitWidth := int(valueBytes[0])
	indices, err := decodeRLEBitPackedHybrid(valueBytes[1:], bitWidth, count)
	if err != nil {
		return fmt.Errorf("dictionary indices: %w", err)
	}
	dictionarySize := decoder.dictionary.length()
	for index := 0; index < len(indices); index++ {
		dictionaryIndex := indices[index]
		if dictionaryIndex >= dictionarySize {
			return fmt.Errorf("dictionary index %d is past the end of the %d entry dictionary", dictionaryIndex, dictionarySize)
		}
		decoder.decoded.values.appendFromDictionary(decoder.dictionary, dictionaryIndex)
	}
	return nil
}

func (decoder *columnChunkDecoder) decodeRLEBooleans(valueBytes []byte, count int) error {
	if len(valueBytes) < 4 {
		return fmt.Errorf("RLE boolean data is too short for its length")
	}
	length := int(binary.LittleEndian.Uint32(valueBytes))
	if length < 0 || length > len(valueBytes)-4 {
		return fmt.Errorf("RLE boolean data says it is %d bytes, which runs past the end of the page", length)
	}
	bits, err := decodeRLEBitPackedHybrid(valueBytes[4:4+length], 1, count)
	if err != nil {
		return err
	}
	for index := 0; index < len(bits); index++ {
		decoder.decoded.values.booleans = append(decoder.decoded.values.booleans, bits[index] == 1)
	}
	return nil
}
