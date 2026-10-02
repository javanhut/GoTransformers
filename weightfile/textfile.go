package weightfile

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"transformer/parameter"
)

func SaveText(path string, parameters []parameter.Parameter) error {
	if err := checkNamesCanBeSaved(parameters); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	for _, current := range parameters {
		fmt.Fprintf(writer, "%s %d\n", current.Name, len(current.Values))
		for i, value := range current.Values {
			if i > 0 {
				writer.WriteString(" ")
			}
			writer.WriteString(strconv.FormatFloat(value, 'g', -1, 64))
		}
		writer.WriteString("\n")
	}
	if err := writer.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func ReadText(path string) (map[string][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanWords)
	savedValues := map[string][]float64{}
	for scanner.Scan() {
		name := scanner.Text()
		if _, alreadySeen := savedValues[name]; alreadySeen {
			return nil, fmt.Errorf("%s: parameter %q is in the file twice", path, name)
		}

		if !scanner.Scan() {
			return nil, fmt.Errorf("%s: parameter %q is missing its value count", path, name)
		}
		count, err := strconv.Atoi(scanner.Text())
		if err != nil || count < 0 || int64(count) > fileInfo.Size() {
			return nil, fmt.Errorf("%s: parameter %q has a bad value count %q", path, name, scanner.Text())
		}

		values := make([]float64, count)
		for i := 0; i < count; i++ {
			if !scanner.Scan() {
				return nil, fmt.Errorf("%s: parameter %q ended after %d of its %d values", path, name, i, count)
			}
			value, err := strconv.ParseFloat(scanner.Text(), 64)
			if err != nil {
				return nil, fmt.Errorf("%s: parameter %q value %d is %q, which is not a number", path, name, i, scanner.Text())
			}
			values[i] = value
		}
		savedValues[name] = values
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return savedValues, nil
}
