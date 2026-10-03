package parquet

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

type thriftWriter struct {
	output       []byte
	lastFieldIDs []int16
}

func (writer *thriftWriter) varint(value uint64) {
	writer.output = binary.AppendUvarint(writer.output, value)
}

func (writer *thriftWriter) zigzag(value int64) {
	writer.varint(uint64((value << 1) ^ (value >> 63)))
}

func (writer *thriftWriter) beginStruct() {
	writer.lastFieldIDs = append(writer.lastFieldIDs, 0)
}

func (writer *thriftWriter) endStruct() {
	writer.output = append(writer.output, thriftStop)
	writer.lastFieldIDs = writer.lastFieldIDs[:len(writer.lastFieldIDs)-1]
}

func (writer *thriftWriter) fieldHeader(fieldID int16, fieldType byte) {
	lastFieldID := writer.lastFieldIDs[len(writer.lastFieldIDs)-1]
	delta := fieldID - lastFieldID
	if delta > 0 && delta <= 15 {
		writer.output = append(writer.output, byte(delta)<<4|fieldType)
	} else {
		writer.output = append(writer.output, fieldType)
		writer.zigzag(int64(fieldID))
	}
	writer.lastFieldIDs[len(writer.lastFieldIDs)-1] = fieldID
}

func (writer *thriftWriter) int32Field(fieldID int16, value int64) {
	writer.fieldHeader(fieldID, thriftInt32)
	writer.zigzag(value)
}

func (writer *thriftWriter) int64Field(fieldID int16, value int64) {
	writer.fieldHeader(fieldID, thriftInt64)
	writer.zigzag(value)
}

func (writer *thriftWriter) binaryField(fieldID int16, value []byte) {
	writer.fieldHeader(fieldID, thriftBinary)
	writer.varint(uint64(len(value)))
	writer.output = append(writer.output, value...)
}

func (writer *thriftWriter) booleanField(fieldID int16, value bool) {
	if value {
		writer.fieldHeader(fieldID, thriftBooleanTrue)
	} else {
		writer.fieldHeader(fieldID, thriftBooleanFalse)
	}
}

func (writer *thriftWriter) structField(fieldID int16, writeFields func()) {
	writer.fieldHeader(fieldID, thriftStruct)
	writer.beginStruct()
	writeFields()
	writer.endStruct()
}

func (writer *thriftWriter) listHeader(elementType byte, size int) {
	if size < 15 {
		writer.output = append(writer.output, byte(size)<<4|elementType)
		return
	}
	writer.output = append(writer.output, 0xF0|elementType)
	writer.varint(uint64(size))
}

func (writer *thriftWriter) listField(fieldID int16, elementType byte, size int) {
	writer.fieldHeader(fieldID, thriftList)
	writer.listHeader(elementType, size)
}

func (writer *thriftWriter) structElement(writeFields func()) {
	writer.beginStruct()
	writeFields()
	writer.endStruct()
}

type testColumn struct {
	name         string
	physicalType int
	typeLength   int
	optional     bool
	repeated     bool
	texts        []string
	integers     []int64
	floats       []float64
	booleans     []bool
	isNull       []bool
}

type testFileOptions struct {
	codec               int
	useDictionary       bool
	plainAfterFirstPage bool
	useDataPageV2       bool
	rowsPerPage         int
	rowsPerRowGroup     int
}

type writtenColumnChunk struct {
	column                *testColumn
	codec                 int
	numberOfValues        int
	totalUncompressedSize int
	totalCompressedSize   int
	dataPageOffset        int
	dictionaryPageOffset  int
	encodings             []int
}

type writtenRowGroup struct {
	numberOfRows int
	chunks       []writtenColumnChunk
}

type testFileWriter struct {
	t         *testing.T
	options   testFileOptions
	fileBytes []byte
}

func writeTestParquetFile(t *testing.T, columns []testColumn, numberOfRows int, options testFileOptions) string {
	t.Helper()
	writer := &testFileWriter{t: t, options: options, fileBytes: []byte(parquetMagic)}
	var rowGroups []writtenRowGroup
	for rowGroupStart := 0; rowGroupStart < numberOfRows; rowGroupStart += options.rowsPerRowGroup {
		rowGroupEnd := min(rowGroupStart+options.rowsPerRowGroup, numberOfRows)
		group := writtenRowGroup{numberOfRows: rowGroupEnd - rowGroupStart}
		for columnIndex := 0; columnIndex < len(columns); columnIndex++ {
			group.chunks = append(group.chunks, writer.writeColumnChunk(&columns[columnIndex], rowGroupStart, rowGroupEnd))
		}
		rowGroups = append(rowGroups, group)
	}
	footer := encodeTestFileMetadata(columns, rowGroups, numberOfRows)
	writer.fileBytes = append(writer.fileBytes, footer...)
	writer.fileBytes = binary.LittleEndian.AppendUint32(writer.fileBytes, uint32(len(footer)))
	writer.fileBytes = append(writer.fileBytes, parquetMagic...)
	path := filepath.Join(t.TempDir(), "written.parquet")
	if err := os.WriteFile(path, writer.fileBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func (column *testColumn) isNullAt(row int) bool {
	return column.isNull != nil && column.isNull[row]
}

func (column *testColumn) plainEncodeValue(row int) []byte {
	switch column.physicalType {
	case byteArrayType:
		encoded := binary.LittleEndian.AppendUint32(nil, uint32(len(column.texts[row])))
		return append(encoded, column.texts[row]...)
	case fixedLengthByteArrayType:
		return []byte(column.texts[row])
	case int32Type:
		return binary.LittleEndian.AppendUint32(nil, uint32(int32(column.integers[row])))
	case int64Type:
		return binary.LittleEndian.AppendUint64(nil, uint64(column.integers[row]))
	case floatType:
		return binary.LittleEndian.AppendUint32(nil, math.Float32bits(float32(column.floats[row])))
	case doubleType:
		return binary.LittleEndian.AppendUint64(nil, math.Float64bits(column.floats[row]))
	}
	panic("plainEncodeValue can't encode this type")
}

func (column *testColumn) plainEncodeRows(rows []int) []byte {
	if column.physicalType == booleanType {
		packed := make([]byte, (len(rows)+7)/8)
		for index := 0; index < len(rows); index++ {
			if column.booleans[rows[index]] {
				packed[index/8] |= 1 << (index % 8)
			}
		}
		return packed
	}
	var encoded []byte
	for index := 0; index < len(rows); index++ {
		encoded = append(encoded, column.plainEncodeValue(rows[index])...)
	}
	return encoded
}

func appendRLERun(output []byte, runLength int, value int, bitWidth int) []byte {
	output = binary.AppendUvarint(output, uint64(runLength)<<1)
	for index := 0; index < (bitWidth+7)/8; index++ {
		output = append(output, byte(value>>(8*index)))
	}
	return output
}

func appendBitPackedGroup(output []byte, groupOfEight []int, bitWidth int) []byte {
	output = binary.AppendUvarint(output, 1<<1|1)
	packed := make([]byte, bitWidth)
	for index := 0; index < 8; index++ {
		for bit := 0; bit < bitWidth; bit++ {
			if groupOfEight[index]>>bit&1 == 1 {
				bitOffset := index*bitWidth + bit
				packed[bitOffset/8] |= 1 << (bitOffset % 8)
			}
		}
	}
	return append(output, packed...)
}

func encodeRLEBitPackedHybrid(values []int, bitWidth int) []byte {
	var output []byte
	position := 0
	for position < len(values) {
		runLength := 1
		for position+runLength < len(values) && values[position+runLength] == values[position] {
			runLength++
		}
		if runLength >= 8 {
			output = appendRLERun(output, runLength, values[position], bitWidth)
			position += runLength
			continue
		}
		if position+8 <= len(values) {
			output = appendBitPackedGroup(output, values[position:position+8], bitWidth)
			position += 8
			continue
		}
		output = appendRLERun(output, 1, values[position], bitWidth)
		position++
	}
	return output
}

func (writer *testFileWriter) compress(data []byte) []byte {
	switch writer.options.codec {
	case uncompressedCodec:
		return data
	case snappyCodec:
		return compressSnappyWithLiteralsOnly(data)
	case gzipCodec:
		var buffer bytes.Buffer
		gzipWriter := gzip.NewWriter(&buffer)
		gzipWriter.Write(data)
		gzipWriter.Close()
		return buffer.Bytes()
	case zstdCodec:
		return compressWithZstdCommand(writer.t, data)
	}
	writer.t.Fatalf("test writer can't compress with codec %d", writer.options.codec)
	return nil
}

func compressSnappyWithLiteralsOnly(data []byte) []byte {
	output := binary.AppendUvarint(nil, uint64(len(data)))
	for start := 0; start < len(data); start += 60 {
		end := min(start+60, len(data))
		output = append(output, byte((end-start-1)<<2))
		output = append(output, data[start:end]...)
	}
	return output
}

func zstdCommandPath() string {
	if path, err := exec.LookPath("zstd"); err == nil {
		return path
	}
	if _, err := os.Stat("/usr/sbin/zstd"); err == nil {
		return "/usr/sbin/zstd"
	}
	return ""
}

func compressWithZstdCommand(t *testing.T, data []byte) []byte {
	commandPath := zstdCommandPath()
	if commandPath == "" {
		t.Skip("the zstd command isn't installed")
	}
	command := exec.Command(commandPath, "-q", "-c", "-3")
	command.Stdin = bytes.NewReader(data)
	compressed, err := command.Output()
	if err != nil {
		t.Fatalf("zstd command failed: %v", err)
	}
	return compressed
}

type testDictionary struct {
	indexOfEncodedValue map[string]int
	encodedValues       [][]byte
}

func buildTestDictionary(column *testColumn, rowStart int, rowEnd int) *testDictionary {
	dictionary := &testDictionary{indexOfEncodedValue: map[string]int{}}
	for row := rowStart; row < rowEnd; row++ {
		if column.isNullAt(row) {
			continue
		}
		encoded := column.plainEncodeValue(row)
		if _, found := dictionary.indexOfEncodedValue[string(encoded)]; !found {
			dictionary.indexOfEncodedValue[string(encoded)] = len(dictionary.encodedValues)
			dictionary.encodedValues = append(dictionary.encodedValues, encoded)
		}
	}
	return dictionary
}

func (writer *testFileWriter) writeColumnChunk(column *testColumn, rowStart int, rowEnd int) writtenColumnChunk {
	chunk := writtenColumnChunk{column: column, codec: writer.options.codec, numberOfValues: rowEnd - rowStart, dictionaryPageOffset: -1}
	chunkStart := len(writer.fileBytes)
	var dictionary *testDictionary
	if writer.options.useDictionary && column.physicalType != booleanType {
		dictionary = buildTestDictionary(column, rowStart, rowEnd)
		chunk.dictionaryPageOffset = len(writer.fileBytes)
		writer.writeDictionaryPage(dictionary, &chunk)
	}
	for pageStart := rowStart; pageStart < rowEnd; pageStart += writer.options.rowsPerPage {
		pageEnd := min(pageStart+writer.options.rowsPerPage, rowEnd)
		if pageStart == rowStart {
			chunk.dataPageOffset = len(writer.fileBytes)
		}
		pageDictionary := dictionary
		if writer.options.plainAfterFirstPage && pageStart != rowStart {
			pageDictionary = nil
		}
		writer.writeDataPage(column, pageStart, pageEnd, pageDictionary, &chunk)
	}
	chunk.totalCompressedSize = len(writer.fileBytes) - chunkStart
	return chunk
}

func (writer *testFileWriter) writeDictionaryPage(dictionary *testDictionary, chunk *writtenColumnChunk) {
	var pageData []byte
	for index := 0; index < len(dictionary.encodedValues); index++ {
		pageData = append(pageData, dictionary.encodedValues[index]...)
	}
	compressed := writer.compress(pageData)
	dictionaryEncoding := int64(plainDictionaryEncoding)
	if writer.options.useDataPageV2 {
		dictionaryEncoding = plainEncoding
	}
	header := &thriftWriter{}
	header.beginStruct()
	header.int32Field(pageHeaderTypeField, dictionaryPage)
	header.int32Field(pageHeaderUncompressedSizeField, int64(len(pageData)))
	header.int32Field(pageHeaderCompressedSizeField, int64(len(compressed)))
	header.structField(pageHeaderDictionaryPageField, func() {
		header.int32Field(dictionaryPageNumberOfValuesField, int64(len(dictionary.encodedValues)))
		header.int32Field(dictionaryPageEncodingField, dictionaryEncoding)
		header.booleanField(3, false)
	})
	header.endStruct()
	writer.fileBytes = append(writer.fileBytes, header.output...)
	writer.fileBytes = append(writer.fileBytes, compressed...)
	chunk.totalUncompressedSize += len(header.output) + len(pageData)
	chunk.encodings = append(chunk.encodings, int(dictionaryEncoding))
}

func (writer *testFileWriter) encodePageValues(column *testColumn, presentRows []int, dictionary *testDictionary) ([]byte, int) {
	if dictionary == nil {
		if column.physicalType == booleanType && writer.options.useDataPageV2 {
			return encodeBooleansAsRLE(column, presentRows), rleEncoding
		}
		return column.plainEncodeRows(presentRows), plainEncoding
	}
	indices := make([]int, len(presentRows))
	for index := 0; index < len(presentRows); index++ {
		indices[index] = dictionary.indexOfEncodedValue[string(column.plainEncodeValue(presentRows[index]))]
	}
	bitWidth := bitWidthFor(len(dictionary.encodedValues) - 1)
	encoded := append([]byte{byte(bitWidth)}, encodeRLEBitPackedHybrid(indices, bitWidth)...)
	if writer.options.useDataPageV2 {
		return encoded, rleDictionaryEncoding
	}
	return encoded, plainDictionaryEncoding
}

func (writer *testFileWriter) writeDataPage(column *testColumn, pageStart int, pageEnd int, dictionary *testDictionary, chunk *writtenColumnChunk) {
	var presentRows []int
	var definitionLevels []int
	for row := pageStart; row < pageEnd; row++ {
		if column.isNullAt(row) {
			definitionLevels = append(definitionLevels, 0)
		} else {
			definitionLevels = append(definitionLevels, 1)
			presentRows = append(presentRows, row)
		}
	}
	valueBytes, encoding := writer.encodePageValues(column, presentRows, dictionary)
	var levelBytes []byte
	if column.optional || column.repeated {
		levelBytes = encodeRLEBitPackedHybrid(definitionLevels, 1)
	}
	numberOfValues := pageEnd - pageStart
	numberOfNulls := numberOfValues - len(presentRows)
	header := &thriftWriter{}
	header.beginStruct()
	var pageBody []byte
	if writer.options.useDataPageV2 {
		compressedValues := writer.compress(valueBytes)
		valuesAreCompressed := writer.options.codec != uncompressedCodec
		pageBody = append(append([]byte{}, levelBytes...), compressedValues...)
		header.int32Field(pageHeaderTypeField, dataPageV2)
		header.int32Field(pageHeaderUncompressedSizeField, int64(len(levelBytes)+len(valueBytes)))
		header.int32Field(pageHeaderCompressedSizeField, int64(len(pageBody)))
		header.structField(pageHeaderDataPageV2Field, func() {
			header.int32Field(dataPageV2NumberOfValuesField, int64(numberOfValues))
			header.int32Field(dataPageV2NumberOfNullsField, int64(numberOfNulls))
			header.int32Field(3, int64(numberOfValues))
			header.int32Field(dataPageV2EncodingField, int64(encoding))
			header.int32Field(dataPageV2DefinitionLevelsLengthField, int64(len(levelBytes)))
			header.int32Field(dataPageV2RepetitionLevelsLengthField, 0)
			header.booleanField(dataPageV2IsCompressedField, valuesAreCompressed)
		})
		chunk.totalUncompressedSize += len(levelBytes) + len(valueBytes)
	} else {
		var uncompressedPage []byte
		if levelBytes != nil {
			uncompressedPage = binary.LittleEndian.AppendUint32(uncompressedPage, uint32(len(levelBytes)))
			uncompressedPage = append(uncompressedPage, levelBytes...)
		}
		uncompressedPage = append(uncompressedPage, valueBytes...)
		pageBody = writer.compress(uncompressedPage)
		header.int32Field(pageHeaderTypeField, dataPage)
		header.int32Field(pageHeaderUncompressedSizeField, int64(len(uncompressedPage)))
		header.int32Field(pageHeaderCompressedSizeField, int64(len(pageBody)))
		header.int32Field(4, 12345)
		header.structField(pageHeaderDataPageField, func() {
			header.int32Field(dataPageNumberOfValuesField, int64(numberOfValues))
			header.int32Field(dataPageEncodingField, int64(encoding))
			header.int32Field(dataPageDefinitionLevelEncodingField, rleEncoding)
			header.int32Field(4, rleEncoding)
			header.structField(5, func() {
				header.binaryField(1, []byte("largest"))
				header.binaryField(2, []byte("smallest"))
				header.int64Field(3, int64(numberOfNulls))
			})
		})
		chunk.totalUncompressedSize += len(uncompressedPage)
	}
	header.endStruct()
	writer.fileBytes = append(writer.fileBytes, header.output...)
	writer.fileBytes = append(writer.fileBytes, pageBody...)
	chunk.totalUncompressedSize += len(header.output)
	chunk.encodings = append(chunk.encodings, encoding)
}

func encodeTestFileMetadata(columns []testColumn, rowGroups []writtenRowGroup, numberOfRows int) []byte {
	writer := &thriftWriter{}
	writer.beginStruct()
	writer.int32Field(1, 1)
	writer.listField(fileMetadataSchemaField, thriftStruct, len(columns)+1)
	writer.structElement(func() {
		writer.binaryField(schemaElementNameField, []byte("schema"))
		writer.int32Field(schemaElementNumberOfChildrenField, int64(len(columns)))
	})
	for index := 0; index < len(columns); index++ {
		column := columns[index]
		writer.structElement(func() {
			writer.int32Field(schemaElementTypeField, int64(column.physicalType))
			if column.physicalType == fixedLengthByteArrayType {
				writer.int32Field(schemaElementTypeLengthField, int64(column.typeLength))
			}
			repetition := int64(requiredRepetition)
			if column.optional {
				repetition = optionalRepetition
			}
			if column.repeated {
				repetition = repeatedRepetition
			}
			writer.int32Field(schemaElementRepetitionField, repetition)
			writer.binaryField(schemaElementNameField, []byte(column.name))
			if column.physicalType == byteArrayType {
				writer.int32Field(6, 0)
			}
		})
	}
	writer.int64Field(fileMetadataNumberOfRowsField, int64(numberOfRows))
	writer.listField(fileMetadataRowGroupsField, thriftStruct, len(rowGroups))
	for groupIndex := 0; groupIndex < len(rowGroups); groupIndex++ {
		group := rowGroups[groupIndex]
		writer.structElement(func() {
			writer.listField(rowGroupColumnsField, thriftStruct, len(group.chunks))
			for chunkIndex := 0; chunkIndex < len(group.chunks); chunkIndex++ {
				encodeTestColumnChunk(writer, group.chunks[chunkIndex])
			}
			writer.int64Field(2, 0)
			writer.int64Field(rowGroupNumberOfRowsField, int64(group.numberOfRows))
		})
	}
	writer.listField(5, thriftStruct, 1)
	writer.structElement(func() {
		writer.binaryField(1, []byte("writer"))
		writer.binaryField(2, []byte("parquet test writer"))
	})
	writer.binaryField(fileMetadataCreatedByField, []byte("GoTransformers test writer"))
	writer.endStruct()
	return writer.output
}

func encodeTestColumnChunk(writer *thriftWriter, chunk writtenColumnChunk) {
	writer.structElement(func() {
		writer.int64Field(2, int64(chunk.dataPageOffset))
		writer.structField(columnChunkMetadataField, func() {
			writer.int32Field(columnMetadataTypeField, int64(chunk.column.physicalType))
			writer.listField(2, thriftInt32, len(chunk.encodings))
			for index := 0; index < len(chunk.encodings); index++ {
				writer.zigzag(int64(chunk.encodings[index]))
			}
			writer.listField(columnMetadataPathField, thriftBinary, 1)
			writer.varint(uint64(len(chunk.column.name)))
			writer.output = append(writer.output, chunk.column.name...)
			writer.int32Field(columnMetadataCodecField, int64(chunk.codec))
			writer.int64Field(columnMetadataNumberOfValuesField, int64(chunk.numberOfValues))
			writer.int64Field(6, int64(chunk.totalUncompressedSize))
			writer.int64Field(columnMetadataTotalCompressedSizeField, int64(chunk.totalCompressedSize))
			writer.int64Field(columnMetadataDataPageOffsetField, int64(chunk.dataPageOffset))
			if chunk.dictionaryPageOffset >= 0 {
				writer.int64Field(columnMetadataDictionaryPageOffsetField, int64(chunk.dictionaryPageOffset))
			}
		})
	})
}

func encodeBooleansAsRLE(column *testColumn, rows []int) []byte {
	bits := make([]int, len(rows))
	for index := 0; index < len(rows); index++ {
		if column.booleans[rows[index]] {
			bits[index] = 1
		}
	}
	encoded := encodeRLEBitPackedHybrid(bits, 1)
	return append(binary.LittleEndian.AppendUint32(nil, uint32(len(encoded))), encoded...)
}
