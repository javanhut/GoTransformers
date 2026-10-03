package weightfile

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"github.com/javanhut/GoTransformers/lowprecision"
	"github.com/javanhut/GoTransformers/parameter"
	"io"
	"os"
)

const binaryFileMarker = "GOTRANSFORMERS-WEIGHTS-1"
const binaryFileMarkerWithPrecisions = "GOTRANSFORMERS-WEIGHTS-2"

const (
	storedAsFloat64        byte = 1
	storedAsFloat32        byte = 2
	storedAsBFloat16       byte = 3
	storedAsCompressedRows byte = 4
)

func SaveBinary(path string, parameters []parameter.Parameter) error {
	return SaveBinaryWithPrecision(path, parameters, Float64)
}

func SaveBinaryWithPrecision(path string, parameters []parameter.Parameter, precision StoragePrecision) error {
	if err := precision.check(); err != nil {
		return err
	}
	if err := checkNamesCanBeSaved(parameters); err != nil {
		return err
	}
	if err := checkCompressedValuesCanBeSaved(parameters); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	if precision == Float64 && !anyAreCompressed(parameters) {
		writeVersionOne(writer, parameters)
	} else {
		writeVersionTwo(writer, parameters, precision)
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func anyAreCompressed(parameters []parameter.Parameter) bool {
	for _, current := range parameters {
		if current.IsCompressed() {
			return true
		}
	}
	return false
}

func writeName(writer *bufio.Writer, name string) {
	binary.Write(writer, binary.LittleEndian, uint32(len(name)))
	writer.WriteString(name)
}

func writeCountAndValues(writer *bufio.Writer, count int, values any) {
	binary.Write(writer, binary.LittleEndian, uint64(count))
	if count > 0 {
		binary.Write(writer, binary.LittleEndian, values)
	}
}

func writeVersionOne(writer *bufio.Writer, parameters []parameter.Parameter) {
	writer.WriteString(binaryFileMarker)
	binary.Write(writer, binary.LittleEndian, uint32(len(parameters)))
	for _, current := range parameters {
		writeName(writer, current.Name)
		writeCountAndValues(writer, len(current.Values), current.Values)
	}
}

func writeVersionTwo(writer *bufio.Writer, parameters []parameter.Parameter, precision StoragePrecision) {
	writer.WriteString(binaryFileMarkerWithPrecisions)
	binary.Write(writer, binary.LittleEndian, uint32(len(parameters)))
	for _, current := range parameters {
		writeName(writer, current.Name)
		if current.IsCompressed() {
			writeCompressedRows(writer, current.CompressedValues)
		} else {
			writeValues(writer, current.Values, precision)
		}
	}
}

func writeValues(writer *bufio.Writer, values []float64, precision StoragePrecision) {
	switch precision {
	case Float64:
		writer.WriteByte(storedAsFloat64)
		writeCountAndValues(writer, len(values), values)
	case Float32:
		writer.WriteByte(storedAsFloat32)
		writeCountAndValues(writer, len(values), toFloat32(values))
	case BFloat16:
		writer.WriteByte(storedAsBFloat16)
		writeCountAndValues(writer, len(values), toBFloat16Bits(values))
	}
}

func toFloat32(values []float64) []float32 {
	result := make([]float32, len(values))
	for i, value := range values {
		result[i] = float32(value)
	}
	return result
}

func toBFloat16Bits(values []float64) []uint16 {
	result := make([]uint16, len(values))
	for i, value := range values {
		result[i] = float64ToBFloat16Bits(value)
	}
	return result
}

func writeCompressedRows(writer *bufio.Writer, rows *lowprecision.Rows) {
	snapshot := rows.Snapshot()
	writer.WriteByte(storedAsCompressedRows)
	writer.WriteByte(byte(snapshot.Precision))
	binary.Write(writer, binary.LittleEndian, uint64(snapshot.NumberOfRows))
	binary.Write(writer, binary.LittleEndian, uint64(snapshot.Width))
	writeCountAndValues(writer, len(snapshot.Float64Values), snapshot.Float64Values)
	writeCountAndValues(writer, len(snapshot.Float32Values), snapshot.Float32Values)
	writeCountAndValues(writer, len(snapshot.Int8Values), snapshot.Int8Values)
	writeCountAndValues(writer, len(snapshot.FP4Values), snapshot.FP4Values)
	writeCountAndValues(writer, len(snapshot.Scales), snapshot.Scales)
}

func ReadBinary(path string) (map[string][]float64, error) {
	savedValues, savedCompressedValues, err := ReadBinaryKeepingCompressed(path)
	if err != nil {
		return nil, err
	}
	for name, rows := range savedCompressedValues {
		savedValues[name] = rows.AllValues()
	}
	return savedValues, nil
}

type openedBinaryFile struct {
	path   string
	size   uint64
	reader *bufio.Reader
}

func ReadBinaryKeepingCompressed(path string) (map[string][]float64, map[string]*lowprecision.Rows, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, nil, err
	}
	opened := openedBinaryFile{path: path, size: uint64(fileInfo.Size()), reader: bufio.NewReader(file)}

	marker := make([]byte, len(binaryFileMarker))
	if _, err := io.ReadFull(opened.reader, marker); err != nil {
		return nil, nil, fmt.Errorf("%s: this is not a binary weights file", path)
	}
	isVersionOne := string(marker) == binaryFileMarker
	if !isVersionOne && string(marker) != binaryFileMarkerWithPrecisions {
		return nil, nil, fmt.Errorf("%s: this is not a binary weights file", path)
	}

	var numberOfParameters uint32
	if err := binary.Read(opened.reader, binary.LittleEndian, &numberOfParameters); err != nil {
		return nil, nil, fmt.Errorf("%s: could not read how many parameters are saved: %w", path, err)
	}

	savedValues := map[string][]float64{}
	savedCompressedValues := map[string]*lowprecision.Rows{}
	for i := uint32(0); i < numberOfParameters; i++ {
		name, err := opened.readName(i)
		if err != nil {
			return nil, nil, err
		}
		_, alreadySeenValues := savedValues[name]
		_, alreadySeenCompressedValues := savedCompressedValues[name]
		if alreadySeenValues || alreadySeenCompressedValues {
			return nil, nil, fmt.Errorf("%s: parameter %q is in the file twice", path, name)
		}

		if isVersionOne {
			values, err := opened.readFloat64List(name, "values")
			if err != nil {
				return nil, nil, err
			}
			savedValues[name] = values
			continue
		}

		storedAs, err := opened.reader.ReadByte()
		if err != nil {
			return nil, nil, fmt.Errorf("%s: could not read how parameter %q is stored: %w", path, name, err)
		}
		switch storedAs {
		case storedAsFloat64:
			savedValues[name], err = opened.readFloat64List(name, "values")
		case storedAsFloat32:
			savedValues[name], err = opened.readFloat32Values(name)
		case storedAsBFloat16:
			savedValues[name], err = opened.readBFloat16Values(name)
		case storedAsCompressedRows:
			savedCompressedValues[name], err = opened.readCompressedRows(name)
		default:
			return nil, nil, fmt.Errorf("%s: parameter %q is stored in an unknown way (%d)", path, name, storedAs)
		}
		if err != nil {
			return nil, nil, err
		}
	}
	return savedValues, savedCompressedValues, nil
}

func (opened openedBinaryFile) readName(index uint32) (string, error) {
	var nameLength uint32
	if err := binary.Read(opened.reader, binary.LittleEndian, &nameLength); err != nil {
		return "", fmt.Errorf("%s: could not read the name length of parameter %d: %w", opened.path, index, err)
	}
	if uint64(nameLength) > opened.size {
		return "", fmt.Errorf("%s: parameter %d says its name is %d bytes long, which is bigger than the file", opened.path, index, nameLength)
	}
	nameBytes := make([]byte, nameLength)
	if _, err := io.ReadFull(opened.reader, nameBytes); err != nil {
		return "", fmt.Errorf("%s: could not read the name of parameter %d: %w", opened.path, index, err)
	}
	return string(nameBytes), nil
}

func (opened openedBinaryFile) readCount(name string, what string, bytesPerValue uint64) (uint64, error) {
	var count uint64
	if err := binary.Read(opened.reader, binary.LittleEndian, &count); err != nil {
		return 0, fmt.Errorf("%s: could not read the %s count of parameter %q: %w", opened.path, what, name, err)
	}
	if count > opened.size/bytesPerValue {
		return 0, fmt.Errorf("%s: parameter %q says it has %d %s, which is more than the file can hold", opened.path, name, count, what)
	}
	return count, nil
}

func (opened openedBinaryFile) readList(name string, what string, count uint64, list any) error {
	if count == 0 {
		return nil
	}
	if err := binary.Read(opened.reader, binary.LittleEndian, list); err != nil {
		return fmt.Errorf("%s: could not read the %s of parameter %q: %w", opened.path, what, name, err)
	}
	return nil
}

func (opened openedBinaryFile) readFloat64List(name string, what string) ([]float64, error) {
	count, err := opened.readCount(name, what, 8)
	if err != nil {
		return nil, err
	}
	list := make([]float64, count)
	return list, opened.readList(name, what, count, list)
}

func (opened openedBinaryFile) readFloat32List(name string, what string) ([]float32, error) {
	count, err := opened.readCount(name, what, 4)
	if err != nil {
		return nil, err
	}
	list := make([]float32, count)
	return list, opened.readList(name, what, count, list)
}

func (opened openedBinaryFile) readUint16List(name string, what string) ([]uint16, error) {
	count, err := opened.readCount(name, what, 2)
	if err != nil {
		return nil, err
	}
	list := make([]uint16, count)
	return list, opened.readList(name, what, count, list)
}

func (opened openedBinaryFile) readInt8List(name string, what string) ([]int8, error) {
	count, err := opened.readCount(name, what, 1)
	if err != nil {
		return nil, err
	}
	list := make([]int8, count)
	return list, opened.readList(name, what, count, list)
}

func (opened openedBinaryFile) readByteList(name string, what string) ([]byte, error) {
	count, err := opened.readCount(name, what, 1)
	if err != nil {
		return nil, err
	}
	list := make([]byte, count)
	return list, opened.readList(name, what, count, list)
}

func (opened openedBinaryFile) readFloat32Values(name string) ([]float64, error) {
	stored, err := opened.readFloat32List(name, "float32 values")
	if err != nil {
		return nil, err
	}
	values := make([]float64, len(stored))
	for i, value := range stored {
		values[i] = float64(value)
	}
	return values, nil
}

func (opened openedBinaryFile) readBFloat16Values(name string) ([]float64, error) {
	stored, err := opened.readUint16List(name, "bfloat16 values")
	if err != nil {
		return nil, err
	}
	values := make([]float64, len(stored))
	for i, bits := range stored {
		values[i] = bFloat16BitsToFloat64(bits)
	}
	return values, nil
}

func (opened openedBinaryFile) readCompressedRows(name string) (*lowprecision.Rows, error) {
	precision, err := opened.reader.ReadByte()
	if err != nil {
		return nil, fmt.Errorf("%s: could not read the precision of compressed parameter %q: %w", opened.path, name, err)
	}
	var numberOfRows uint64
	var width uint64
	if err := binary.Read(opened.reader, binary.LittleEndian, &numberOfRows); err != nil {
		return nil, fmt.Errorf("%s: could not read the row count of compressed parameter %q: %w", opened.path, name, err)
	}
	if err := binary.Read(opened.reader, binary.LittleEndian, &width); err != nil {
		return nil, fmt.Errorf("%s: could not read the row width of compressed parameter %q: %w", opened.path, name, err)
	}
	mostValuesTheFileCanHold := 2 * opened.size
	if width == 0 || width > mostValuesTheFileCanHold || numberOfRows > mostValuesTheFileCanHold/width {
		return nil, fmt.Errorf("%s: compressed parameter %q says it is %d rows of %d values, which does not fit in the file", opened.path, name, numberOfRows, width)
	}

	snapshot := lowprecision.RowsSnapshot{Precision: lowprecision.Precision(precision), NumberOfRows: int(numberOfRows), Width: int(width)}
	if snapshot.Float64Values, err = opened.readFloat64List(name, "float64 values"); err != nil {
		return nil, err
	}
	if snapshot.Float32Values, err = opened.readFloat32List(name, "float32 values"); err != nil {
		return nil, err
	}
	if snapshot.Int8Values, err = opened.readInt8List(name, "int8 values"); err != nil {
		return nil, err
	}
	if snapshot.FP4Values, err = opened.readByteList(name, "fp4 values"); err != nil {
		return nil, err
	}
	if snapshot.Scales, err = opened.readFloat32List(name, "scales"); err != nil {
		return nil, err
	}
	rows, err := lowprecision.RowsFromSnapshot(snapshot)
	if err != nil {
		return nil, fmt.Errorf("%s: compressed parameter %q: %w", opened.path, name, err)
	}
	return rows, nil
}
