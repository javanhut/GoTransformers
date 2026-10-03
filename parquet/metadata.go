package parquet

import (
	"fmt"
	"strings"
)

const (
	booleanType              = 0
	int32Type                = 1
	int64Type                = 2
	int96Type                = 3
	floatType                = 4
	doubleType               = 5
	byteArrayType            = 6
	fixedLengthByteArrayType = 7
)

var physicalTypeNames = []string{"BOOLEAN", "INT32", "INT64", "INT96", "FLOAT", "DOUBLE", "BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY"}

const (
	requiredRepetition = 0
	optionalRepetition = 1
	repeatedRepetition = 2
)

const (
	plainEncoding           = 0
	plainDictionaryEncoding = 2
	rleEncoding             = 3
	bitPackedEncoding       = 4
	rleDictionaryEncoding   = 8
)

var encodingNames = []string{"PLAIN", "GROUP_VAR_INT", "PLAIN_DICTIONARY", "RLE", "BIT_PACKED", "DELTA_BINARY_PACKED", "DELTA_LENGTH_BYTE_ARRAY", "DELTA_BYTE_ARRAY", "RLE_DICTIONARY", "BYTE_STREAM_SPLIT"}

const (
	uncompressedCodec = 0
	snappyCodec       = 1
	gzipCodec         = 2
	zstdCodec         = 6
)

var codecNames = []string{"UNCOMPRESSED", "SNAPPY", "GZIP", "LZO", "BROTLI", "LZ4", "ZSTD", "LZ4_RAW"}

const (
	dataPage       = 0
	indexPage      = 1
	dictionaryPage = 2
	dataPageV2     = 3
)

func nameFromList(names []string, index int) string {
	if index >= 0 && index < len(names) {
		return names[index]
	}
	return fmt.Sprintf("unknown (%d)", index)
}

const (
	fileMetadataSchemaField       = 2
	fileMetadataNumberOfRowsField = 3
	fileMetadataRowGroupsField    = 4
	fileMetadataCreatedByField    = 6
)

const (
	schemaElementTypeField             = 1
	schemaElementTypeLengthField       = 2
	schemaElementRepetitionField       = 3
	schemaElementNameField             = 4
	schemaElementNumberOfChildrenField = 5
)

const (
	rowGroupColumnsField      = 1
	rowGroupNumberOfRowsField = 3
)

const (
	columnChunkFilePathField = 1
	columnChunkMetadataField = 3
)

const (
	columnMetadataTypeField                 = 1
	columnMetadataPathField                 = 3
	columnMetadataCodecField                = 4
	columnMetadataNumberOfValuesField       = 5
	columnMetadataTotalCompressedSizeField  = 7
	columnMetadataDataPageOffsetField       = 9
	columnMetadataDictionaryPageOffsetField = 11
)

type fileMetadata struct {
	numberOfRows int64
	schema       []schemaElement
	rowGroups    []rowGroup
	createdBy    string
}

type schemaElement struct {
	name             string
	hasPhysicalType  bool
	physicalType     int
	typeLength       int
	repetition       int
	numberOfChildren int
}

type rowGroup struct {
	numberOfRows int64
	columns      []columnChunk
}

type columnChunk struct {
	filePath             string
	physicalType         int
	path                 []string
	codec                int
	numberOfValues       int64
	totalCompressedSize  int64
	dataPageOffset       int64
	dictionaryPageOffset int64
}

func readFileMetadata(fields thriftFields) (fileMetadata, error) {
	numberOfRows, err := fields.requiredInteger(fileMetadataNumberOfRowsField, "the number of rows")
	if err != nil {
		return fileMetadata{}, err
	}
	metadata := fileMetadata{numberOfRows: numberOfRows, createdBy: fields.text(fileMetadataCreatedByField)}
	schemaList := fields.list(fileMetadataSchemaField)
	if len(schemaList) == 0 {
		return fileMetadata{}, fmt.Errorf("parquet metadata has no schema")
	}
	for index := 0; index < len(schemaList); index++ {
		elementFields, isStruct := schemaList[index].(thriftFields)
		if !isStruct {
			return fileMetadata{}, fmt.Errorf("parquet schema element %d is not a struct", index)
		}
		metadata.schema = append(metadata.schema, readSchemaElement(elementFields))
	}
	rowGroupList := fields.list(fileMetadataRowGroupsField)
	for index := 0; index < len(rowGroupList); index++ {
		rowGroupFields, isStruct := rowGroupList[index].(thriftFields)
		if !isStruct {
			return fileMetadata{}, fmt.Errorf("parquet row group %d is not a struct", index)
		}
		group, err := readRowGroup(rowGroupFields)
		if err != nil {
			return fileMetadata{}, fmt.Errorf("row group %d: %w", index, err)
		}
		metadata.rowGroups = append(metadata.rowGroups, group)
	}
	return metadata, nil
}

func readSchemaElement(fields thriftFields) schemaElement {
	physicalType, hasPhysicalType := fields.integer(schemaElementTypeField)
	return schemaElement{
		name:             fields.text(schemaElementNameField),
		hasPhysicalType:  hasPhysicalType,
		physicalType:     int(physicalType),
		typeLength:       int(fields.integerOrZero(schemaElementTypeLengthField)),
		repetition:       int(fields.integerOrZero(schemaElementRepetitionField)),
		numberOfChildren: int(fields.integerOrZero(schemaElementNumberOfChildrenField)),
	}
}

func readRowGroup(fields thriftFields) (rowGroup, error) {
	numberOfRows, err := fields.requiredInteger(rowGroupNumberOfRowsField, "the row group's number of rows")
	if err != nil {
		return rowGroup{}, err
	}
	group := rowGroup{numberOfRows: numberOfRows}
	columnList := fields.list(rowGroupColumnsField)
	for index := 0; index < len(columnList); index++ {
		chunkFields, isStruct := columnList[index].(thriftFields)
		if !isStruct {
			return rowGroup{}, fmt.Errorf("column chunk %d is not a struct", index)
		}
		chunk, err := readColumnChunk(chunkFields)
		if err != nil {
			return rowGroup{}, fmt.Errorf("column chunk %d: %w", index, err)
		}
		group.columns = append(group.columns, chunk)
	}
	return group, nil
}

func readColumnChunk(fields thriftFields) (columnChunk, error) {
	metadataFields, found := fields.structure(columnChunkMetadataField)
	if !found {
		return columnChunk{}, fmt.Errorf("parquet column chunk has no metadata")
	}
	chunk := columnChunk{filePath: fields.text(columnChunkFilePathField)}
	physicalType, err := metadataFields.requiredInteger(columnMetadataTypeField, "the column's type")
	if err != nil {
		return columnChunk{}, err
	}
	chunk.physicalType = int(physicalType)
	codec, err := metadataFields.requiredInteger(columnMetadataCodecField, "the column's compression codec")
	if err != nil {
		return columnChunk{}, err
	}
	chunk.codec = int(codec)
	chunk.numberOfValues, err = metadataFields.requiredInteger(columnMetadataNumberOfValuesField, "the column's number of values")
	if err != nil {
		return columnChunk{}, err
	}
	chunk.totalCompressedSize, err = metadataFields.requiredInteger(columnMetadataTotalCompressedSizeField, "the column's compressed size")
	if err != nil {
		return columnChunk{}, err
	}
	chunk.dataPageOffset, err = metadataFields.requiredInteger(columnMetadataDataPageOffsetField, "the column's data page offset")
	if err != nil {
		return columnChunk{}, err
	}
	chunk.dictionaryPageOffset = metadataFields.integerOrZero(columnMetadataDictionaryPageOffsetField)
	pathList := metadataFields.list(columnMetadataPathField)
	for index := 0; index < len(pathList); index++ {
		part, _ := pathList[index].([]byte)
		chunk.path = append(chunk.path, string(part))
	}
	return chunk, nil
}

func (chunk columnChunk) dottedPath() string {
	return strings.Join(chunk.path, ".")
}

func (chunk columnChunk) firstPageOffset() int64 {
	if chunk.dictionaryPageOffset > 0 && chunk.dictionaryPageOffset < chunk.dataPageOffset {
		return chunk.dictionaryPageOffset
	}
	return chunk.dataPageOffset
}

const (
	pageHeaderTypeField             = 1
	pageHeaderUncompressedSizeField = 2
	pageHeaderCompressedSizeField   = 3
	pageHeaderDataPageField         = 5
	pageHeaderDictionaryPageField   = 7
	pageHeaderDataPageV2Field       = 8
)

const (
	dataPageNumberOfValuesField          = 1
	dataPageEncodingField                = 2
	dataPageDefinitionLevelEncodingField = 3
)

const (
	dictionaryPageNumberOfValuesField = 1
	dictionaryPageEncodingField       = 2
)

const (
	dataPageV2NumberOfValuesField         = 1
	dataPageV2NumberOfNullsField          = 2
	dataPageV2EncodingField               = 4
	dataPageV2DefinitionLevelsLengthField = 5
	dataPageV2RepetitionLevelsLengthField = 6
	dataPageV2IsCompressedField           = 7
)

type pageHeader struct {
	pageType                   int
	uncompressedSize           int
	compressedSize             int
	numberOfValues             int
	numberOfNulls              int
	encoding                   int
	definitionLevelEncoding    int
	definitionLevelsByteLength int
	repetitionLevelsByteLength int
	valuesAreCompressed        bool
}

func readPageHeader(data []byte) (pageHeader, int, error) {
	fields, headerSize, err := readThriftStruct(data)
	if err != nil {
		return pageHeader{}, 0, fmt.Errorf("can't read a page header: %w", err)
	}
	pageType, err := fields.requiredInteger(pageHeaderTypeField, "the page type")
	if err != nil {
		return pageHeader{}, 0, err
	}
	uncompressedSize, err := fields.requiredInteger(pageHeaderUncompressedSizeField, "the page's uncompressed size")
	if err != nil {
		return pageHeader{}, 0, err
	}
	compressedSize, err := fields.requiredInteger(pageHeaderCompressedSizeField, "the page's compressed size")
	if err != nil {
		return pageHeader{}, 0, err
	}
	if uncompressedSize < 0 || compressedSize < 0 {
		return pageHeader{}, 0, fmt.Errorf("page header has negative sizes %d and %d", uncompressedSize, compressedSize)
	}
	header := pageHeader{pageType: int(pageType), uncompressedSize: int(uncompressedSize), compressedSize: int(compressedSize), valuesAreCompressed: true}
	switch header.pageType {
	case dataPage:
		err = header.readDataPageFields(fields)
	case dictionaryPage:
		err = header.readDictionaryPageFields(fields)
	case dataPageV2:
		err = header.readDataPageV2Fields(fields)
	}
	if err != nil {
		return pageHeader{}, 0, err
	}
	return header, headerSize, nil
}

func (header *pageHeader) readDataPageFields(fields thriftFields) error {
	dataPageFields, found := fields.structure(pageHeaderDataPageField)
	if !found {
		return fmt.Errorf("data page header is missing")
	}
	numberOfValues, err := dataPageFields.requiredInteger(dataPageNumberOfValuesField, "the data page's number of values")
	if err != nil {
		return err
	}
	header.numberOfValues = int(numberOfValues)
	header.encoding = int(dataPageFields.integerOrZero(dataPageEncodingField))
	header.definitionLevelEncoding = int(dataPageFields.integerOrZero(dataPageDefinitionLevelEncodingField))
	return nil
}

func (header *pageHeader) readDictionaryPageFields(fields thriftFields) error {
	dictionaryFields, found := fields.structure(pageHeaderDictionaryPageField)
	if !found {
		return fmt.Errorf("dictionary page header is missing")
	}
	numberOfValues, err := dictionaryFields.requiredInteger(dictionaryPageNumberOfValuesField, "the dictionary's number of values")
	if err != nil {
		return err
	}
	header.numberOfValues = int(numberOfValues)
	header.encoding = int(dictionaryFields.integerOrZero(dictionaryPageEncodingField))
	return nil
}

func (header *pageHeader) readDataPageV2Fields(fields thriftFields) error {
	v2Fields, found := fields.structure(pageHeaderDataPageV2Field)
	if !found {
		return fmt.Errorf("data page v2 header is missing")
	}
	numberOfValues, err := v2Fields.requiredInteger(dataPageV2NumberOfValuesField, "the data page's number of values")
	if err != nil {
		return err
	}
	header.numberOfValues = int(numberOfValues)
	header.numberOfNulls = int(v2Fields.integerOrZero(dataPageV2NumberOfNullsField))
	header.encoding = int(v2Fields.integerOrZero(dataPageV2EncodingField))
	header.definitionLevelsByteLength = int(v2Fields.integerOrZero(dataPageV2DefinitionLevelsLengthField))
	header.repetitionLevelsByteLength = int(v2Fields.integerOrZero(dataPageV2RepetitionLevelsLengthField))
	header.valuesAreCompressed = v2Fields.boolean(dataPageV2IsCompressedField, true)
	if header.numberOfValues < 0 || header.definitionLevelsByteLength < 0 || header.repetitionLevelsByteLength < 0 {
		return fmt.Errorf("data page v2 header has negative sizes")
	}
	return nil
}
