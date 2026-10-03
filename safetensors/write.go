package safetensors

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

func Write(path string, tensors map[string]Tensor) error {
	return write(path, tensors, "F32")
}

func WriteFloat64(path string, tensors map[string]Tensor) error {
	return write(path, tensors, "F64")
}

func write(path string, tensors map[string]Tensor, dataType string) error {
	valueSize, err := bytesPerValue(dataType)
	if err != nil {
		return err
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		if name == "__metadata__" {
			return fmt.Errorf("a tensor can't be named __metadata__")
		}
		names = append(names, name)
	}
	sort.Strings(names)

	header := map[string]headerEntry{}
	var offset int64
	for _, name := range names {
		tensor := tensors[name]
		if numberOfValues(tensor.Shape) != len(tensor.Values) {
			return fmt.Errorf("tensor %q has shape %v, which needs %d values, but has %d", name, tensor.Shape, numberOfValues(tensor.Shape), len(tensor.Values))
		}
		size := int64(len(tensor.Values) * valueSize)
		shape := tensor.Shape
		if shape == nil {
			shape = []int{}
		}
		header[name] = headerEntry{DataType: dataType, Shape: shape, DataOffsets: []int64{offset, offset + size}}
		offset += size
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(headerBytes)%8 != 0 {
		headerBytes = append(headerBytes, ' ')
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	binary.Write(writer, binary.LittleEndian, uint64(len(headerBytes)))
	writer.Write(headerBytes)
	valueBytes := make([]byte, valueSize)
	for _, name := range names {
		for _, value := range tensors[name].Values {
			if dataType == "F64" {
				binary.LittleEndian.PutUint64(valueBytes, math.Float64bits(value))
			} else {
				binary.LittleEndian.PutUint32(valueBytes, math.Float32bits(float32(value)))
			}
			writer.Write(valueBytes)
		}
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}
