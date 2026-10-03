package parquet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const (
	thriftStop         = 0
	thriftBooleanTrue  = 1
	thriftBooleanFalse = 2
	thriftByte         = 3
	thriftInt16        = 4
	thriftInt32        = 5
	thriftInt64        = 6
	thriftDouble       = 7
	thriftBinary       = 8
	thriftList         = 9
	thriftSet          = 10
	thriftMap          = 11
	thriftStruct       = 12
)

const deepestThriftNesting = 64

type thriftFields map[int16]any

type thriftReader struct {
	data     []byte
	position int
	depth    int
}

func readThriftStruct(data []byte) (thriftFields, int, error) {
	reader := &thriftReader{data: data}
	fields, err := reader.readStruct()
	if err != nil {
		return nil, 0, err
	}
	return fields, reader.position, nil
}

func (reader *thriftReader) readByte() (byte, error) {
	if reader.position >= len(reader.data) {
		return 0, errors.New("thrift data ends early")
	}
	value := reader.data[reader.position]
	reader.position++
	return value, nil
}

func (reader *thriftReader) readVarint() (uint64, error) {
	value, numberOfBytes := binary.Uvarint(reader.data[reader.position:])
	if numberOfBytes <= 0 {
		return 0, fmt.Errorf("thrift data has a broken number at byte %d", reader.position)
	}
	reader.position += numberOfBytes
	return value, nil
}

func (reader *thriftReader) readZigzagInteger() (int64, error) {
	value, err := reader.readVarint()
	if err != nil {
		return 0, err
	}
	return int64(value>>1) ^ -int64(value&1), nil
}

func (reader *thriftReader) readLength() (int, error) {
	length, err := reader.readVarint()
	if err != nil {
		return 0, err
	}
	if length > uint64(len(reader.data)-reader.position) {
		return 0, fmt.Errorf("thrift data says %d items or bytes follow, but only %d bytes are left", length, len(reader.data)-reader.position)
	}
	return int(length), nil
}

func (reader *thriftReader) readStruct() (thriftFields, error) {
	reader.depth++
	defer func() { reader.depth-- }()
	if reader.depth > deepestThriftNesting {
		return nil, errors.New("thrift data is nested too deeply")
	}
	fields := thriftFields{}
	lastFieldID := int16(0)
	for {
		header, err := reader.readByte()
		if err != nil {
			return nil, err
		}
		if header == thriftStop {
			return fields, nil
		}
		fieldType := header & 0x0F
		fieldIDDelta := int16(header >> 4)
		fieldID := lastFieldID + fieldIDDelta
		if fieldIDDelta == 0 {
			explicitFieldID, err := reader.readZigzagInteger()
			if err != nil {
				return nil, err
			}
			fieldID = int16(explicitFieldID)
		}
		lastFieldID = fieldID
		value, err := reader.readValue(fieldType)
		if err != nil {
			return nil, err
		}
		fields[fieldID] = value
	}
}

func (reader *thriftReader) readValue(valueType byte) (any, error) {
	switch valueType {
	case thriftBooleanTrue:
		return true, nil
	case thriftBooleanFalse:
		return false, nil
	case thriftByte:
		value, err := reader.readByte()
		return int64(int8(value)), err
	case thriftInt16, thriftInt32, thriftInt64:
		return reader.readZigzagInteger()
	case thriftDouble:
		if reader.position+8 > len(reader.data) {
			return nil, errors.New("thrift data ends inside a double")
		}
		bitPattern := binary.LittleEndian.Uint64(reader.data[reader.position:])
		reader.position += 8
		return math.Float64frombits(bitPattern), nil
	case thriftBinary:
		length, err := reader.readLength()
		if err != nil {
			return nil, err
		}
		value := reader.data[reader.position : reader.position+length]
		reader.position += length
		return value, nil
	case thriftList, thriftSet:
		return reader.readList()
	case thriftMap:
		return reader.readMap()
	case thriftStruct:
		return reader.readStruct()
	}
	return nil, fmt.Errorf("thrift data has unknown type %d at byte %d", valueType, reader.position)
}

func (reader *thriftReader) readElement(elementType byte) (any, error) {
	if elementType == thriftBooleanTrue || elementType == thriftBooleanFalse {
		value, err := reader.readByte()
		return value == thriftBooleanTrue, err
	}
	return reader.readValue(elementType)
}

func (reader *thriftReader) readList() ([]any, error) {
	header, err := reader.readByte()
	if err != nil {
		return nil, err
	}
	elementType := header & 0x0F
	size := int(header >> 4)
	if size == 15 {
		size, err = reader.readLength()
		if err != nil {
			return nil, err
		}
	}
	elements := make([]any, size)
	for index := 0; index < size; index++ {
		elements[index], err = reader.readElement(elementType)
		if err != nil {
			return nil, err
		}
	}
	return elements, nil
}

func (reader *thriftReader) readMap() ([]any, error) {
	size, err := reader.readLength()
	if err != nil {
		return nil, err
	}
	if size == 0 {
		return nil, nil
	}
	types, err := reader.readByte()
	if err != nil {
		return nil, err
	}
	keyType := types >> 4
	valueType := types & 0x0F
	keysAndValues := make([]any, 0, 2*size)
	for index := 0; index < size; index++ {
		key, err := reader.readElement(keyType)
		if err != nil {
			return nil, err
		}
		value, err := reader.readElement(valueType)
		if err != nil {
			return nil, err
		}
		keysAndValues = append(keysAndValues, key, value)
	}
	return keysAndValues, nil
}

func (fields thriftFields) integer(fieldID int16) (int64, bool) {
	value, isInteger := fields[fieldID].(int64)
	return value, isInteger
}

func (fields thriftFields) integerOrZero(fieldID int16) int64 {
	value, _ := fields.integer(fieldID)
	return value
}

func (fields thriftFields) requiredInteger(fieldID int16, description string) (int64, error) {
	value, found := fields.integer(fieldID)
	if !found {
		return 0, fmt.Errorf("parquet metadata is missing %s", description)
	}
	return value, nil
}

func (fields thriftFields) text(fieldID int16) string {
	value, _ := fields[fieldID].([]byte)
	return string(value)
}

func (fields thriftFields) boolean(fieldID int16, valueWhenMissing bool) bool {
	value, found := fields[fieldID].(bool)
	if !found {
		return valueWhenMissing
	}
	return value
}

func (fields thriftFields) structure(fieldID int16) (thriftFields, bool) {
	value, found := fields[fieldID].(thriftFields)
	return value, found
}

func (fields thriftFields) list(fieldID int16) []any {
	value, _ := fields[fieldID].([]any)
	return value
}
