package parquet

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

func TestThriftReaderDecodesEveryType(t *testing.T) {
	writer := &thriftWriter{}
	writer.beginStruct()
	writer.int32Field(1, -5)
	writer.int64Field(2, math.MaxInt64)
	writer.booleanField(3, true)
	writer.booleanField(4, false)
	writer.binaryField(5, []byte("hello"))
	writer.fieldHeader(6, thriftDouble)
	writer.output = binary.LittleEndian.AppendUint64(writer.output, math.Float64bits(2.5))
	writer.fieldHeader(7, thriftByte)
	writer.output = append(writer.output, 0xff)
	writer.int32Field(100, 42)
	writer.int32Field(-3, 9)
	writer.listField(101, thriftBooleanTrue, 3)
	writer.output = append(writer.output, 1, 0, 1)
	writer.fieldHeader(102, thriftSet)
	writer.listHeader(thriftInt32, 20)
	for index := 0; index < 20; index++ {
		writer.zigzag(int64(index))
	}
	writer.fieldHeader(103, thriftMap)
	writer.varint(2)
	writer.output = append(writer.output, thriftBinary<<4|thriftInt64)
	writer.varint(1)
	writer.output = append(writer.output, 'a')
	writer.zigzag(-1)
	writer.varint(1)
	writer.output = append(writer.output, 'b')
	writer.zigzag(1)
	writer.fieldHeader(104, thriftMap)
	writer.varint(0)
	writer.structField(105, func() {
		writer.int32Field(1, 7)
		writer.structField(2, func() {
			writer.binaryField(1, []byte("deep"))
		})
	})
	writer.endStruct()
	writer.output = append(writer.output, 0xAA)

	fields, bytesUsed, err := readThriftStruct(writer.output)
	if err != nil {
		t.Fatal(err)
	}
	if bytesUsed != len(writer.output)-1 {
		t.Errorf("used %d bytes, want %d", bytesUsed, len(writer.output)-1)
	}
	if value, _ := fields.integer(1); value != -5 {
		t.Errorf("field 1 = %d", value)
	}
	if value, _ := fields.integer(2); value != math.MaxInt64 {
		t.Errorf("field 2 = %d", value)
	}
	if !fields.boolean(3, false) || fields.boolean(4, true) || !fields.boolean(99, true) {
		t.Errorf("booleans read wrong")
	}
	if fields.text(5) != "hello" {
		t.Errorf("field 5 = %q", fields.text(5))
	}
	if fields[6] != 2.5 || fields[7] != int64(-1) {
		t.Errorf("double %v byte %v", fields[6], fields[7])
	}
	if fields.integerOrZero(100) != 42 || fields.integerOrZero(-3) != 9 {
		t.Errorf("long form field ids read wrong: %v", fields)
	}
	if !reflect.DeepEqual(fields.list(101), []any{true, false, true}) {
		t.Errorf("boolean list = %v", fields.list(101))
	}
	if len(fields.list(102)) != 20 || fields.list(102)[19] != int64(19) {
		t.Errorf("set = %v", fields.list(102))
	}
	if !reflect.DeepEqual(fields[103], []any{[]byte("a"), int64(-1), []byte("b"), int64(1)}) {
		t.Errorf("map = %v", fields[103])
	}
	nested, found := fields.structure(105)
	if !found || nested.integerOrZero(1) != 7 {
		t.Fatalf("nested struct = %v", fields[105])
	}
	deeper, _ := nested.structure(2)
	if deeper.text(1) != "deep" {
		t.Errorf("deeper struct = %v", deeper)
	}
}

func TestThriftReaderRejectsBrokenData(t *testing.T) {
	cases := map[string][]byte{
		"empty":                   {},
		"no stop":                 {0x15, 0x02},
		"binary longer than data": {0x18, 0x10, 'a'},
		"list longer than data":   {0x19, 0xF5, 0x7F},
		"unknown type":            {0x1D},
		"double cut off":          {0x17, 1, 2, 3},
		"broken varint":           {0x15, 0xff, 0xff},
	}
	for name, data := range cases {
		if _, _, err := readThriftStruct(data); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	deeplyNested := []byte{}
	for depth := 0; depth < 100; depth++ {
		deeplyNested = append(deeplyNested, 0x1C)
	}
	if _, _, err := readThriftStruct(deeplyNested); err == nil {
		t.Error("deeply nested structs: no error")
	}
}
