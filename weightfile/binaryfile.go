package weightfile

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"transformer/parameter"
)

const binaryFileMarker = "GOTRANSFORMERS-WEIGHTS-1"

func SaveBinary(path string, parameters []parameter.Parameter) error {
	if err := checkNamesCanBeSaved(parameters); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	writer.WriteString(binaryFileMarker)
	binary.Write(writer, binary.LittleEndian, uint32(len(parameters)))
	for _, current := range parameters {
		binary.Write(writer, binary.LittleEndian, uint32(len(current.Name)))
		writer.WriteString(current.Name)
		binary.Write(writer, binary.LittleEndian, uint64(len(current.Values)))
		binary.Write(writer, binary.LittleEndian, current.Values)
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func ReadBinary(path string) (map[string][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}
	fileSize := uint64(fileInfo.Size())
	reader := bufio.NewReader(file)

	marker := make([]byte, len(binaryFileMarker))
	if _, err := io.ReadFull(reader, marker); err != nil || string(marker) != binaryFileMarker {
		return nil, fmt.Errorf("%s: this is not a binary weights file", path)
	}

	var numberOfParameters uint32
	if err := binary.Read(reader, binary.LittleEndian, &numberOfParameters); err != nil {
		return nil, fmt.Errorf("%s: could not read how many parameters are saved: %w", path, err)
	}

	savedValues := map[string][]float64{}
	for i := uint32(0); i < numberOfParameters; i++ {
		var nameLength uint32
		if err := binary.Read(reader, binary.LittleEndian, &nameLength); err != nil {
			return nil, fmt.Errorf("%s: could not read the name length of parameter %d: %w", path, i, err)
		}
		if uint64(nameLength) > fileSize {
			return nil, fmt.Errorf("%s: parameter %d says its name is %d bytes long, which is bigger than the file", path, i, nameLength)
		}
		nameBytes := make([]byte, nameLength)
		if _, err := io.ReadFull(reader, nameBytes); err != nil {
			return nil, fmt.Errorf("%s: could not read the name of parameter %d: %w", path, i, err)
		}
		name := string(nameBytes)
		if _, alreadySeen := savedValues[name]; alreadySeen {
			return nil, fmt.Errorf("%s: parameter %q is in the file twice", path, name)
		}

		var count uint64
		if err := binary.Read(reader, binary.LittleEndian, &count); err != nil {
			return nil, fmt.Errorf("%s: could not read the value count of parameter %q: %w", path, name, err)
		}
		if count > fileSize/8 {
			return nil, fmt.Errorf("%s: parameter %q says it has %d values, which is more than the file can hold", path, name, count)
		}
		values := make([]float64, count)
		if err := binary.Read(reader, binary.LittleEndian, values); err != nil {
			return nil, fmt.Errorf("%s: could not read the values of parameter %q: %w", path, name, err)
		}
		savedValues[name] = values
	}
	return savedValues, nil
}
