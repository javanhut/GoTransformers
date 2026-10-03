package parquet

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

const parquetMagic = "PAR1"

const encryptedFooterMagic = "PARE"

const largestFooterSize = 256 * 1024 * 1024

type Column struct {
	Name         string
	PhysicalType string
	Optional     bool
	Repeated     bool
}

type File struct {
	Path        string
	CreatedBy   string
	file        *os.File
	fileSize    int64
	metadata    fileMetadata
	leafColumns []leafColumn
}

func Open(path string) (*File, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	opened, err := readFooter(path, file)
	if err != nil {
		file.Close()
		return nil, err
	}
	return opened, nil
}

func (opened *File) Close() error {
	return opened.file.Close()
}

func readFooter(path string, file *os.File) (*File, error) {
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}
	fileSize := fileInfo.Size()
	if fileSize < 12 {
		return nil, fmt.Errorf("%s: is %d bytes, too small to be a parquet file", path, fileSize)
	}
	startMagic := make([]byte, 4)
	if _, err := file.ReadAt(startMagic, 0); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	tail := make([]byte, 8)
	if _, err := file.ReadAt(tail, fileSize-8); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	endMagic := string(tail[4:])
	if endMagic == encryptedFooterMagic {
		return nil, fmt.Errorf("%s: is an encrypted parquet file, which isn't supported", path)
	}
	if string(startMagic) != parquetMagic || endMagic != parquetMagic {
		return nil, fmt.Errorf("%s: is not a parquet file (it doesn't start and end with %q)", path, parquetMagic)
	}
	footerSize := int64(binary.LittleEndian.Uint32(tail))
	if footerSize > largestFooterSize || footerSize > fileSize-12 {
		return nil, fmt.Errorf("%s: footer says it is %d bytes, which doesn't fit in a %d byte file", path, footerSize, fileSize)
	}
	footerBytes := make([]byte, footerSize)
	if _, err := file.ReadAt(footerBytes, fileSize-8-footerSize); err != nil {
		return nil, fmt.Errorf("%s: can't read the footer: %w", path, err)
	}
	footerFields, _, err := readThriftStruct(footerBytes)
	if err != nil {
		return nil, fmt.Errorf("%s: footer is not valid parquet metadata: %w", path, err)
	}
	metadata, err := readFileMetadata(footerFields)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	leafColumns, err := collectLeafColumns(metadata.schema)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	for groupIndex := 0; groupIndex < len(metadata.rowGroups); groupIndex++ {
		if len(metadata.rowGroups[groupIndex].columns) != len(leafColumns) {
			return nil, fmt.Errorf("%s: row group %d has %d columns but the schema has %d", path, groupIndex, len(metadata.rowGroups[groupIndex].columns), len(leafColumns))
		}
	}
	return &File{
		Path:        path,
		CreatedBy:   metadata.createdBy,
		file:        file,
		fileSize:    fileSize,
		metadata:    metadata,
		leafColumns: leafColumns,
	}, nil
}

type schemaWalker struct {
	schema      []schemaElement
	position    int
	leafColumns []leafColumn
}

func collectLeafColumns(schema []schemaElement) ([]leafColumn, error) {
	walker := &schemaWalker{schema: schema, position: 1}
	if err := walker.walkChildren(schema[0].numberOfChildren, nil, 0, 0); err != nil {
		return nil, err
	}
	return walker.leafColumns, nil
}

func (walker *schemaWalker) walkChildren(numberOfChildren int, parentPath []string, definitionLevel int, repetitionLevel int) error {
	for childIndex := 0; childIndex < numberOfChildren; childIndex++ {
		if walker.position >= len(walker.schema) {
			return fmt.Errorf("schema says it has more elements than the %d it lists", len(walker.schema))
		}
		element := walker.schema[walker.position]
		walker.position++
		path := append(append([]string{}, parentPath...), element.name)
		childDefinitionLevel := definitionLevel
		childRepetitionLevel := repetitionLevel
		if element.repetition == optionalRepetition {
			childDefinitionLevel++
		}
		if element.repetition == repeatedRepetition {
			childDefinitionLevel++
			childRepetitionLevel++
		}
		if element.numberOfChildren > 0 {
			if err := walker.walkChildren(element.numberOfChildren, path, childDefinitionLevel, childRepetitionLevel); err != nil {
				return err
			}
			continue
		}
		if !element.hasPhysicalType {
			continue
		}
		walker.leafColumns = append(walker.leafColumns, leafColumn{
			name:                   strings.Join(path, "."),
			physicalType:           element.physicalType,
			typeLength:             element.typeLength,
			maximumDefinitionLevel: childDefinitionLevel,
			maximumRepetitionLevel: childRepetitionLevel,
		})
	}
	return nil
}

func (opened *File) ColumnNames() []string {
	names := make([]string, len(opened.leafColumns))
	for index := 0; index < len(opened.leafColumns); index++ {
		names[index] = opened.leafColumns[index].name
	}
	return names
}

func (opened *File) Columns() []Column {
	columns := make([]Column, len(opened.leafColumns))
	for index := 0; index < len(opened.leafColumns); index++ {
		leaf := opened.leafColumns[index]
		columns[index] = Column{
			Name:         leaf.name,
			PhysicalType: nameFromList(physicalTypeNames, leaf.physicalType),
			Optional:     leaf.maximumDefinitionLevel > 0,
			Repeated:     leaf.maximumRepetitionLevel > 0,
		}
	}
	return columns
}

func (opened *File) NumberOfRows() int {
	return int(opened.metadata.numberOfRows)
}

func (opened *File) NumberOfRowGroups() int {
	return len(opened.metadata.rowGroups)
}

func (opened *File) NumberOfRowsInRowGroup(rowGroupIndex int) int {
	if rowGroupIndex < 0 || rowGroupIndex >= len(opened.metadata.rowGroups) {
		return 0
	}
	return int(opened.metadata.rowGroups[rowGroupIndex].numberOfRows)
}

func (opened *File) findColumn(columnName string) (int, error) {
	for index := 0; index < len(opened.leafColumns); index++ {
		if opened.leafColumns[index].name == columnName {
			return index, nil
		}
	}
	return 0, fmt.Errorf("%s: has no column %q (it has %v)", opened.Path, columnName, opened.ColumnNames())
}

func (opened *File) readColumn(rowGroupIndex int, columnName string, wantedKinds ...string) (decodedColumn, error) {
	if rowGroupIndex < 0 || rowGroupIndex >= len(opened.metadata.rowGroups) {
		return decodedColumn{}, fmt.Errorf("%s: has %d row groups, there is no row group %d", opened.Path, len(opened.metadata.rowGroups), rowGroupIndex)
	}
	columnIndex, err := opened.findColumn(columnName)
	if err != nil {
		return decodedColumn{}, err
	}
	column := opened.leafColumns[columnIndex]
	if column.maximumRepetitionLevel > 0 {
		return decodedColumn{}, fmt.Errorf("%s: column %q is a repeated (list) column, and only flat columns are supported", opened.Path, columnName)
	}
	kind, err := valueKindOf(column.physicalType)
	if err != nil {
		return decodedColumn{}, fmt.Errorf("%s: column %q: %w", opened.Path, columnName, err)
	}
	if !containsText(wantedKinds, kind) {
		return decodedColumn{}, fmt.Errorf("%s: column %q holds %s values (%s), not %s", opened.Path, columnName, kind, nameFromList(physicalTypeNames, column.physicalType), strings.Join(wantedKinds, " or "))
	}
	chunk := opened.metadata.rowGroups[rowGroupIndex].columns[columnIndex]
	if chunk.dottedPath() != column.name {
		return decodedColumn{}, fmt.Errorf("%s: row group %d stores column %q where the schema expects %q", opened.Path, rowGroupIndex, chunk.dottedPath(), column.name)
	}
	chunkBytes, err := opened.readColumnChunkBytes(chunk)
	if err != nil {
		return decodedColumn{}, fmt.Errorf("%s: row group %d column %q: %w", opened.Path, rowGroupIndex, columnName, err)
	}
	decoded, err := decodeColumnChunk(chunkBytes, column, chunk)
	if err != nil {
		return decodedColumn{}, fmt.Errorf("%s: row group %d column %q: %w", opened.Path, rowGroupIndex, columnName, err)
	}
	return decoded, nil
}

func containsText(texts []string, wanted string) bool {
	for index := 0; index < len(texts); index++ {
		if texts[index] == wanted {
			return true
		}
	}
	return false
}

func (opened *File) readColumnChunkBytes(chunk columnChunk) ([]byte, error) {
	if chunk.filePath != "" {
		return nil, fmt.Errorf("column data lives in another file (%s), which isn't supported", chunk.filePath)
	}
	start := chunk.firstPageOffset()
	size := chunk.totalCompressedSize
	if start < 4 || size < 0 || start+size > opened.fileSize {
		return nil, fmt.Errorf("column data at bytes %d to %d doesn't fit in the %d byte file", start, start+size, opened.fileSize)
	}
	chunkBytes := make([]byte, size)
	if _, err := opened.file.ReadAt(chunkBytes, start); err != nil && err != io.EOF {
		return nil, err
	}
	return chunkBytes, nil
}
