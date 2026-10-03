package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

type Tensor struct {
	Shape  []int
	Values []float64
}

type TensorInfo struct {
	Name     string
	DataType string
	Shape    []int
	Start    int64
	End      int64
}

type File struct {
	Path      string
	Tensors   map[string]TensorInfo
	Metadata  map[string]string
	file      *os.File
	dataStart int64
	fileSize  int64
}

const largestHeaderSize = 100 * 1024 * 1024

func bytesPerValue(dataType string) (int, error) {
	switch dataType {
	case "F64":
		return 8, nil
	case "F32":
		return 4, nil
	case "F16", "BF16":
		return 2, nil
	}
	return 0, fmt.Errorf("data type %q is not supported, only F64, F32, F16 and BF16 are", dataType)
}

func numberOfValues(shape []int) int {
	count := 1
	for _, size := range shape {
		count *= size
	}
	return count
}

type headerEntry struct {
	DataType    string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

func Open(path string) (*File, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	opened, err := readHeader(path, file)
	if err != nil {
		file.Close()
		return nil, err
	}
	return opened, nil
}

func readHeader(path string, file *os.File) (*File, error) {
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}
	fileSize := fileInfo.Size()

	var headerSize uint64
	if err := binary.Read(file, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("%s: can't read the header size: %w", path, err)
	}
	if headerSize > largestHeaderSize || int64(headerSize) > fileSize-8 {
		return nil, fmt.Errorf("%s: header says it is %d bytes, which doesn't fit in a %d byte file", path, headerSize, fileSize)
	}
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return nil, fmt.Errorf("%s: can't read the header: %w", path, err)
	}

	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("%s: header is not valid JSON: %w", path, err)
	}

	opened := &File{
		Path:      path,
		Tensors:   map[string]TensorInfo{},
		Metadata:  map[string]string{},
		file:      file,
		dataStart: 8 + int64(headerSize),
		fileSize:  fileSize,
	}
	dataSize := fileSize - opened.dataStart
	for name, raw := range rawHeader {
		if name == "__metadata__" {
			if err := json.Unmarshal(raw, &opened.Metadata); err != nil {
				return nil, fmt.Errorf("%s: metadata is not a map of strings: %w", path, err)
			}
			continue
		}
		var entry headerEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			return nil, fmt.Errorf("%s: tensor %q has a bad header entry: %w", path, name, err)
		}
		valueSize, err := bytesPerValue(entry.DataType)
		if err != nil {
			return nil, fmt.Errorf("%s: tensor %q: %w", path, name, err)
		}
		if len(entry.DataOffsets) != 2 {
			return nil, fmt.Errorf("%s: tensor %q needs 2 data offsets, got %d", path, name, len(entry.DataOffsets))
		}
		start := entry.DataOffsets[0]
		end := entry.DataOffsets[1]
		if start < 0 || end < start || end > dataSize {
			return nil, fmt.Errorf("%s: tensor %q says its data is at bytes %d to %d, but the data part of the file is %d bytes", path, name, start, end, dataSize)
		}
		for _, size := range entry.Shape {
			if size < 0 {
				return nil, fmt.Errorf("%s: tensor %q has a negative size in its shape %v", path, name, entry.Shape)
			}
		}
		if int64(numberOfValues(entry.Shape))*int64(valueSize) != end-start {
			return nil, fmt.Errorf("%s: tensor %q has shape %v of %s, which needs %d bytes, but its data is %d bytes", path, name, entry.Shape, entry.DataType, numberOfValues(entry.Shape)*valueSize, end-start)
		}
		opened.Tensors[name] = TensorInfo{Name: name, DataType: entry.DataType, Shape: entry.Shape, Start: start, End: end}
	}
	return opened, nil
}

func (opened *File) Names() []string {
	names := make([]string, 0, len(opened.Tensors))
	for name := range opened.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (opened *File) ReadTensor(name string) (Tensor, error) {
	info, found := opened.Tensors[name]
	if !found {
		return Tensor{}, fmt.Errorf("%s: there is no tensor named %q", opened.Path, name)
	}
	rawBytes := make([]byte, info.End-info.Start)
	if _, err := opened.file.ReadAt(rawBytes, opened.dataStart+info.Start); err != nil {
		return Tensor{}, fmt.Errorf("%s: can't read tensor %q: %w", opened.Path, name, err)
	}
	values := make([]float64, numberOfValues(info.Shape))
	switch info.DataType {
	case "F64":
		for i := range values {
			values[i] = math.Float64frombits(binary.LittleEndian.Uint64(rawBytes[i*8:]))
		}
	case "F32":
		for i := range values {
			values[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(rawBytes[i*4:])))
		}
	case "BF16":
		for i := range values {
			values[i] = BFloat16ToFloat64(binary.LittleEndian.Uint16(rawBytes[i*2:]))
		}
	case "F16":
		for i := range values {
			values[i] = Float16ToFloat64(binary.LittleEndian.Uint16(rawBytes[i*2:]))
		}
	}
	shape := append([]int(nil), info.Shape...)
	return Tensor{Shape: shape, Values: values}, nil
}

func (opened *File) Close() error {
	return opened.file.Close()
}

func Read(path string) (map[string]Tensor, error) {
	opened, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer opened.Close()
	tensors := map[string]Tensor{}
	for _, name := range opened.Names() {
		tensor, err := opened.ReadTensor(name)
		if err != nil {
			return nil, err
		}
		tensors[name] = tensor
	}
	return tensors, nil
}

func BFloat16ToFloat64(bits uint16) float64 {
	return float64(math.Float32frombits(uint32(bits) << 16))
}

func Float16ToFloat64(bits uint16) float64 {
	sign := 1.0
	if bits&0x8000 != 0 {
		sign = -1.0
	}
	exponent := int((bits >> 10) & 0x1f)
	fraction := float64(bits & 0x3ff)
	switch exponent {
	case 0:
		return sign * fraction / 1024 * math.Pow(2, -14)
	case 31:
		if fraction == 0 {
			return sign * math.Inf(1)
		}
		return math.NaN()
	}
	return sign * (1 + fraction/1024) * math.Pow(2, float64(exponent-15))
}
